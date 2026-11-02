"""Gradio demo: three grammar correctors side by side (spec section 12).

::

    python app/app.py [--config app/config.yaml] [--port 7860] [--share]

Shows the same sentence corrected by the **Synthetic-only**, **Real-only** and
**Mixed** models, with each edit highlighted and explained in plain English.

The demo must be useful before the experiments finish, so a missing checkpoint
is handled rather than fatal: that column renders "Not trained yet" plus the
command that would produce it, and the rest of the app works normally. With
zero checkpoints the app still starts (spec section 10, Phase 10).
"""

from __future__ import annotations

import argparse
import html
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make `gec` importable without an editable install, so `python app/app.py`
# works from a fresh clone.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from gec import explain as explain_mod  # noqa: E402
from gec import paths, repro, tok  # noqa: E402

COLUMNS = ("Synthetic-only", "Real-only", "Mixed")

_INS_STYLE = (
    "background:#d6f5d6;color:#14532d;border-radius:3px;padding:0 2px;"
    "font-weight:600"
)
_DEL_STYLE = (
    "background:#ffd6d6;color:#7f1d1d;border-radius:3px;padding:0 2px;"
    "text-decoration:line-through"
)


@dataclass
class EditView:
    before: str
    after: str
    errant_type: str
    label: str
    reason: str
    operation: str


@dataclass
class Result:
    corrected_tok: str = ""
    corrected_display: str = ""
    edits: list[EditView] = field(default_factory=list)
    html: str = ""
    ms: float = 0.0
    error: str | None = None


_ANNOTATOR = None


def annotator():
    """ERRANT annotator as a module singleton (loading costs several seconds)."""
    global _ANNOTATOR
    if _ANNOTATOR is None:
        import errant

        print("[app] loading ERRANT...")
        _ANNOTATOR = errant.load("en")
    return _ANNOTATOR


class Corrector:
    """One model column. Lazily loaded, and tolerant of a missing checkpoint."""

    def __init__(self, name: str, path: Path | None, cfg: dict) -> None:
        self.name = name
        self.path = Path(path) if path else None
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.load_error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.path and (self.path / "config.json").exists())

    @property
    def missing_message(self) -> str:
        run = self.path.parent.name if self.path else "?"
        return (
            f"**Not trained yet.**\n\n`{paths.rel(self.path) if self.path else 'unset'}` "
            f"does not exist.\n\nProduce it with:\n\n"
            f"```\npython -m gec.run_experiments --plan e3 --only {run} --seeds 13 --yes\n```"
        )

    def load(self) -> None:
        if self.model is not None or not self.available:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        want = self.cfg.get("device", "auto")
        self.device = repro.best_device(want)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), legacy=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.path))
            self.model.to(self.device)
            self.model.eval()
            print(f"[app] loaded {self.name} on {self.device}")
        except (OSError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            # Three models in fp32 are ~250 MB each, comfortable in 4 GB, but
            # fall back per model rather than taking the whole app down.
            if self.device == "cuda":
                print(f"[app] {self.name}: GPU load failed ({exc}); retrying on CPU")
                self.device = "cpu"
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.path))
                    self.model.to("cpu")
                    self.model.eval()
                    return
                except Exception as exc2:  # noqa: BLE001
                    self.load_error = str(exc2)
            else:
                self.load_error = str(exc)

    def correct(self, text: str) -> Result:
        if not self.available:
            return Result(error=self.missing_message)
        self.load()
        if self.load_error:
            return Result(error=f"Failed to load this model: {self.load_error}")

        import torch

        t0 = time.perf_counter()
        src_tok = tok.tokenise(tok.normalise(text))
        prompt = self.cfg.get("prefix", "grammar: ") + src_tok
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=192)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                num_beams=int(self.cfg.get("num_beams", 4)),
                max_new_tokens=int(self.cfg.get("max_new_tokens", 128)),
                early_stopping=int(self.cfg.get("num_beams", 4)) > 1,
            )
        raw = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        ms = (time.perf_counter() - t0) * 1000

        # Same convention fix as predict.py: T5 drifts toward detokenised text,
        # and ERRANT would read that as a punctuation edit.
        cor_tok = tok.tokenise(raw.strip()) or src_tok
        edits = annotate_pair(src_tok, cor_tok)
        return Result(
            corrected_tok=cor_tok,
            corrected_display=tok.detokenise(cor_tok),
            edits=edits,
            html=render_html(src_tok, cor_tok, edits),
            ms=ms,
        )


def annotate_pair(src_tok: str, cor_tok: str) -> list[EditView]:
    """ERRANT-annotate a tokenised (source, correction) pair."""
    ann = annotator()
    orig = ann.parse(src_tok, tokenise=False)
    cor = ann.parse(cor_tok, tokenise=False)
    views: list[EditView] = []
    for e in ann.annotate(orig, cor):
        if e.type in ("noop", "UNK"):
            continue
        ex = explain_mod.explain(e.type)
        views.append(
            EditView(
                before=e.o_str,
                after=e.c_str,
                errant_type=e.type,
                label=ex.label,
                reason=ex.reason,
                operation=ex.operation,
            )
        )
    return views


def render_html(src_tok: str, cor_tok: str, edits: list[EditView]) -> str:
    """Corrected sentence with deletions struck through and insertions in green.

    Everything is escaped with :func:`html.escape` **before** any span is added,
    so a user typing ``<script>`` cannot inject into the page.
    """
    if not edits:
        safe = html.escape(tok.detokenise(cor_tok))
        return (
            f'<div style="font-size:1.05em;line-height:1.7">{safe}</div>'
            '<div style="color:#666;margin-top:6px">No changes suggested.</div>'
        )

    parts: list[str] = []
    for e in edits:
        if e.before and e.after:
            parts.append(
                f'<span style="{_DEL_STYLE}">{html.escape(e.before)}</span> '
                f'<span style="{_INS_STYLE}">{html.escape(e.after)}</span>'
            )
        elif e.after:
            parts.append(f'<span style="{_INS_STYLE}">+{html.escape(e.after)}</span>')
        else:
            parts.append(f'<span style="{_DEL_STYLE}">{html.escape(e.before)}</span>')

    corrected = html.escape(tok.detokenise(cor_tok))
    return (
        f'<div style="font-size:1.05em;line-height:1.7;margin-bottom:8px">{corrected}</div>'
        f'<div style="line-height:2">{" ".join(parts)}</div>'
    )


def render_column(name: str, r: Result) -> tuple[str, str]:
    """Return ``(html_block, markdown_edit_list)`` for one column."""
    if r.error:
        return f'<div style="color:#92400e">{r.error}</div>', ""
    lines = [f"**{len(r.edits)} edit(s)** - {r.ms:.0f} ms"]
    for e in r.edits:
        before = e.before or "(nothing)"
        after = e.after or "(nothing)"
        lines.append(
            f"- `{before}` -> `{after}` - **{e.label}** "
            f"({e.errant_type}, {e.operation})  \n  {e.reason}"
        )
    if not r.edits:
        lines.append("- no changes suggested")
    return r.html, "\n".join(lines)


def load_correctors(cfg: dict) -> dict[str, Corrector]:
    out: dict[str, Corrector] = {}
    models = cfg.get("models", {}) or {}
    for name in COLUMNS:
        raw = models.get(name)
        out[name] = Corrector(name, (_ROOT / raw) if raw else None, cfg)
    return out


def build_ui(correctors: dict[str, Corrector], cfg: dict):
    import gradio as gr

    n_ready = sum(1 for c in correctors.values() if c.available)
    status = (
        f"{n_ready} of {len(correctors)} models available."
        if n_ready
        else "**No checkpoints yet** - the demo runs, but every column will tell you "
        "which command to run. Try `python -m gec.run_experiments --plan e3 --seeds 13 --yes`."
    )

    with gr.Blocks(title="Fake Errors, Real Grammar") as demo:
        gr.Markdown(
            "# Fake Errors, Real Grammar\n"
            "### Can synthetic errors teach a small model to correct real grammar?\n"
            "Three `t5-small` correctors trained on **synthetic**, **real** and "
            "**mixed** data. Type a sentence and compare them.\n\n"
            f"{status}"
        )
        box = gr.Textbox(
            label="Your sentence",
            placeholder="He go to school yesterday and buy a apple.",
            lines=2,
        )
        btn = gr.Button("Correct", variant="primary")
        gr.Examples(examples=[[e] for e in cfg.get("examples", [])], inputs=box)

        html_out: dict[str, object] = {}
        md_out: dict[str, object] = {}
        with gr.Row():
            for name in COLUMNS:
                with gr.Column():
                    tag = "" if correctors[name].available else " _(not trained)_"
                    gr.Markdown(f"#### {name}{tag}")
                    html_out[name] = gr.HTML()
                    md_out[name] = gr.Markdown()

        def run(text: str):
            text = (text or "").strip()
            if not text:
                empty = '<div style="color:#666">Type a sentence above.</div>'
                return [empty, ""] * len(COLUMNS)
            outputs: list[str] = []
            for name in COLUMNS:
                h, m = render_column(name, correctors[name].correct(text))
                outputs.extend([h, m])
            return outputs

        targets: list[object] = []
        for name in COLUMNS:
            targets.extend([html_out[name], md_out[name]])
        btn.click(run, inputs=box, outputs=targets)
        box.submit(run, inputs=box, outputs=targets)

        gr.Markdown(
            "---\n"
            "Edits are extracted and classified with **ERRANT**; each "
            "explanation comes from `src/gec/explain.py`. Insertions are shown "
            "in green, deletions struck through in red. Times are per-model "
            "inference on this machine."
        )
    return demo


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(prog="python app/app.py", description="Gradio GEC demo.")
    ap.add_argument("--config", default=str(_ROOT / "app" / "config.yaml"))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-preload", action="store_true",
                    help="load each model on first use instead of at startup")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                    help="override config.yaml. Use --device cpu to open the "
                         "demo while a training run owns the GPU: three T5-small "
                         "models at ~250 MB each would otherwise compete with "
                         "the trainer for a 4 GB card")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.device:
        cfg["device"] = args.device
    correctors = load_correctors(cfg)

    ready = [n for n, c in correctors.items() if c.available]
    print(f"[app] checkpoints found: {ready or 'none'}")
    if ready and not args.no_preload:
        annotator()
        for c in correctors.values():
            c.load()

    build_ui(correctors, cfg).launch(
        server_port=args.port, share=args.share, show_error=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
