"""Batch inference (spec section 8.2).

::

    python -m gec.predict --run results/runs/e3_mixed/seed42 --split test

Four decisions:

* **Length-bucketed generation with order restoration.** Padding dominates
  beam-search cost, and the test set has sources from 3 to 90 tokens, so
  sorting by length is roughly a 2x saving. Restoring the original order
  afterwards is non-negotiable: the whole ERRANT pipeline is positional.

* **:func:`postprocess` is the empty-output firewall**, and it is mandatory
  rather than defensive. ERRANT's ``parallel_to_m2`` does
  ``if not orig: continue``, so a blank line produces no M2 block, which
  desynchronises hypothesis and reference and surfaces as a bare
  ``AssertionError`` inside ``compare_m2`` much later. An empty prediction
  falls back to the source and the repair is counted.

* **``predict_max_source_len`` is 192, longer than training's 128.** Truncating
  a *test* source invents end-of-sentence deletions and depresses F0.5 for a
  reason unrelated to the research question. T5's relative position embeddings
  extrapolate fine.

* **Generation runs in fp32** even when training used bf16: decoding is
  memory- and launch-bound rather than FLOP-bound, so autocast buys little and
  only adds a source of nondeterminism.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Sequence

from gec import config as cfgmod
from gec import m2, paths, repro, tok

_PREFIX_LEAK = re.compile(r"^\s*grammar\s*:\s*", re.IGNORECASE)


def postprocess(pred: str, src: str) -> tuple[str, str | None]:
    """Clean one prediction; returns ``(text, repair_reason)``.

    Never returns an empty string - see the module docstring.
    """
    out = _PREFIX_LEAK.sub("", pred or "")
    out = re.sub(r"\s+", " ", out).strip()
    if not out:
        return src, "empty"
    return out, None


def retokenise(preds: list[str]) -> tuple[list[str], int]:
    """Force predictions into the project's tokenisation convention.

    Spec section 5.3 requires model inputs, targets, **predictions** and M2
    files to all be spaCy-tokenised and space-separated. T5's pretraining pulls
    its output towards ordinary detokenised English, so it readily emits
    ``information.`` where the source had ``information .``.

    That difference is a surface convention, not a grammatical edit - but
    ERRANT cannot tell, so left alone it manufactures a spurious ``PUNCT`` or
    ``ORTH`` edit on nearly every sentence, inflating false positives and
    depressing F0.5 for every condition equally. Re-tokenising removes the
    artefact while preserving any genuine edit (an inserted comma survives
    tokenisation; a deleted space before a period does not).

    Returns the re-tokenised predictions and how many actually changed.
    """
    out = list(tok.tokenise_batch(preds))
    n_changed = sum(1 for a, b in zip(preds, out) if a != b)
    return out, n_changed


def load_run_model(run: Path | str, device: str | None = None):
    """Load a run's saved model, tokenizer and resolved config."""
    import yaml
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    run = Path(run)
    cfg_path = run / "config.yaml"
    cfg = yaml.safe_load(paths.read_text(cfg_path)) if cfg_path.exists() else {}
    model_dir = run / "model"
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"no model at {paths.rel(model_dir)}. Train it first:\n"
            f"    python -m gec.train --config configs/experiments/"
            f"{cfg.get('name', run.parent.name)}.yaml --seed {run.name.replace('seed', '')}"
        )
    device = device or repro.best_device(cfg.get("device", "auto"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return model, tokenizer, cfg


def generate_batched(
    model,
    tokenizer,
    srcs: Sequence[str],
    cfg: dict,
    device: str,
    *,
    batch_size: int | None = None,
    num_beams: int | None = None,
    sort_by_length: bool = True,
) -> tuple[list[str], dict]:
    """Generate corrections for *srcs*, returning them in the ORIGINAL order."""
    import torch

    prefix = cfg.get("prefix", "grammar: ")
    gen_cfg = cfg.get("generation", {}) or {}
    beams = num_beams if num_beams is not None else int(gen_cfg.get("num_beams", 4))
    bs = batch_size if batch_size is not None else int(cfg.get("eval_batch_size", 16))
    max_src = int(cfg.get("predict_max_source_len", 192))
    max_new = int(gen_cfg.get("max_new_tokens", 128))

    order = list(range(len(srcs)))
    if sort_by_length:
        order.sort(key=lambda i: len(srcs[i].split()))

    out: list[str | None] = [None] * len(srcs)
    n_truncated = 0
    i = 0
    while i < len(order):
        idx = order[i : i + bs]
        batch = [prefix + srcs[j] for j in idx]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_src
        )
        n_truncated += sum(
            1 for j in idx
            if len(tokenizer(prefix + srcs[j], truncation=False)["input_ids"]) > max_src
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        src_len = int(enc["input_ids"].shape[1])
        try:
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    num_beams=beams,
                    # Bound runaway repetition without ever clipping a
                    # legitimate correction (which is at most a few tokens
                    # longer than its source).
                    max_new_tokens=min(max_new, src_len + 24),
                    early_stopping=beams > 1,
                )
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            print(f"[predict] CUDA OOM; halving batch size to {bs} and retrying")
            continue
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j, text in zip(idx, decoded):
            out[j] = text
        i += len(idx)

    return [x if x is not None else "" for x in out], {
        "num_beams": beams,
        "batch_size": bs,
        "n_truncated": n_truncated,
        "max_source_len": max_src,
    }


def _write_run_local_reference(run: Path, split: str, n: int) -> None:
    """For a limited split, write a matching n-block reference M2 into the run.

    Uses the same ``M2Block.raw`` mechanism as the val split, so the sliced
    reference still carries the original gold annotations.
    """
    ref = paths.split_files(split)["m2"]
    if not ref.exists():
        return
    blocks = []
    for k, b in enumerate(m2.read_m2(ref)):
        if k >= n:
            break
        blocks.append(b)
    m2.write_m2(blocks, run / "predictions" / f"{split}.ref.m2")


def predict_split(
    run: Path | str,
    split: str,
    cfg: dict | None = None,
    limit: int | None = None,
    force: bool = False,
    **gen_kwargs,
) -> Path:
    """Predict one split and write the three prediction artefacts."""
    run = Path(run)
    pred_dir = run / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    src_out = pred_dir / f"{split}.src.txt"
    pred_out = pred_dir / f"{split}.pred.txt"

    if cfg is None:
        import yaml

        cfg = yaml.safe_load(paths.read_text(run / "config.yaml"))

    files = paths.split_files(split)
    if not files["src"].exists():
        raise FileNotFoundError(
            f"{paths.rel(files['src'])} is missing. Run: "
            f"python -m gec.prepare_data --all"
        )
    srcs = paths.read_lines(files["src"])

    if limit is None:
        limit = (cfg.get("eval", {}) or {}).get("limit")
    if limit:
        srcs = srcs[: int(limit)]
        _write_run_local_reference(run, split, len(srcs))

    if pred_out.exists() and not force and paths.count_lines(pred_out) == len(srcs):
        print(f"[predict] reusing {paths.rel(pred_out)} ({len(srcs)} sentences)")
        return pred_out

    t0 = time.time()
    condition = cfg.get("condition")

    if condition == "copy":
        # E0: the prediction IS the source. No model, no GPU, ~1 second.
        print(f"[predict] copy baseline on {split}: {len(srcs)} sentences")
        preds = list(srcs)
        meta = {"num_beams": 0, "batch_size": 0, "n_truncated": 0, "device": "none",
                "precision": "none"}
        n_repaired = 0
    else:
        device = repro.best_device(cfg.get("device", "auto"))
        model, tokenizer, _ = load_run_model(run, device)
        print(f"[predict] {split}: {len(srcs)} sentences on {device}")
        repro.reset_peak_vram()
        raw, meta = generate_batched(model, tokenizer, srcs, cfg, device, **gen_kwargs)
        cleaned: list[str] = []
        n_repaired = 0
        for pred, src in zip(raw, srcs):
            text, reason = postprocess(pred, src)
            n_repaired += reason is not None
            cleaned.append(text)

        # Bring predictions back into the project's tokenisation convention
        # before anything scores them (see retokenise()).
        preds, n_retokenised = retokenise(cleaned)
        # A prediction can become blank only if it was pure punctuation that
        # tokenised away; fall back to the source, same as postprocess.
        n_blank = 0
        for i, (p, s) in enumerate(zip(preds, srcs)):
            if not p.strip():
                preds[i] = s
                n_blank += 1
        n_repaired += n_blank
        print(f"[predict] re-tokenised {n_retokenised}/{len(preds)} predictions "
              "to the spaCy convention")
        meta.update({"device": device, "precision": "fp32",
                     "peak_vram_mb": repro.peak_vram_mb(),
                     "n_retokenised": n_retokenised})
        del model

    wall = time.time() - t0

    # Both files are written together, from the same list, in the same order.
    paths.write_lines(src_out, srcs)
    paths.write_lines(pred_out, preds)

    n_same = sum(1 for s, p in zip(srcs, preds) if s.split() == p.split())
    paths.write_json(
        run / f"predict_meta.{split}.json",
        {
            "split": split,
            "n": len(srcs),
            "wall_clock_s": round(wall, 2),
            "ms_per_sentence": round(1000 * wall / len(srcs), 2) if srcs else None,
            "sentences_per_s": round(len(srcs) / wall, 2) if wall else None,
            "n_empty_repaired": n_repaired,
            "copy_rate": round(n_same / len(srcs), 4) if srcs else None,
            **meta,
        },
    )
    print(f"[predict] wrote {paths.rel(pred_out)} in {wall:.1f}s "
          f"({1000 * wall / max(len(srcs), 1):.1f} ms/sentence); "
          f"copy_rate={n_same / max(len(srcs), 1):.3f}, repaired={n_repaired}")
    return pred_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gec.predict", description="Batch inference for one run."
    )
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--all-splits", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--beams", type=int, default=None)
    ap.add_argument("--greedy", action="store_true", help="shorthand for --beams 1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    run = Path(args.run)
    if not run.exists():
        ap.error(f"run directory does not exist: {run}")

    import yaml

    cfg = yaml.safe_load(paths.read_text(run / "config.yaml"))
    kwargs: dict = {}
    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
    beams = 1 if args.greedy else args.beams
    if beams:
        kwargs["num_beams"] = beams

    splits = (cfg.get("eval", {}) or {}).get("splits", ["val", "test"]) \
        if args.all_splits else [args.split]
    for split in splits:
        if split == "indian" and (paths.PROCESSED / "indian.skip").exists():
            print("[predict] skipping indian: fewer than 20 real rows (indian.skip present)")
            continue
        predict_split(run, split, cfg, limit=args.limit, force=args.force, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
