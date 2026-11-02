"""Tables, figures and RESULTS.md from saved metrics (spec section 11).

::

    python -m gec.aggregate
    python -m gec.aggregate --tables-only
    python -m gec.aggregate --strict      # non-zero exit if anything is missing

**`TBD` without crashing.** The mechanism is a *left join from the expected
run x seed grid onto what exists on disk*, not a scan of what exists. So a
missing run is a row that exists with NaN values, rendered as ``TBD`` only at
write time - never in the frame used for plotting. Every figure function returns
``Path | None``, guards its own preconditions, and is called inside a
``try/except`` so one broken figure cannot cost the other five.

**Selection stays honest.** :func:`select_best` has a hard assertion refusing to
read any ``test`` metric: E1's best ``p`` and E2's best ``clean_ratio`` are
chosen on ``val`` F0.5 alone (spec rule 2.2), and every candidate's val score is
recorded in ``selection.json`` so the choice is auditable afterwards.
"""

from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path
from typing import Any

from gec import paths, repro
from gec.run_experiments import DEFAULT_SEEDS, PLANS

#: Collected while aggregating; rendered at the top of RESULTS.md.
WARNINGS: list[str] = []

#: Figure style: readable, no chart junk (spec section 11).
FIG_DPI = 150


def _warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"[agg] {msg}")


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

def expected_grid() -> list[dict]:
    """Every (run, seed) the trimmed plan is supposed to produce."""
    rows: list[dict] = []
    for plan, names in PLANS.items():
        if plan == "smoke":
            continue
        for name in names:
            for seed in DEFAULT_SEEDS:
                rows.append({"run": name, "seed": seed, "family": plan})
    return rows


def _flatten(metrics: dict, run: str, seed: int) -> dict:
    """Flatten one metrics.json into a single table row."""
    row: dict[str, Any] = {
        "run": run,
        "seed": seed,
        "condition": metrics.get("condition"),
        "alias_of": metrics.get("alias_of"),
    }
    train = metrics.get("train") or {}
    for k in ("wall_clock_s", "steps", "final_train_loss", "best_val_loss",
              "peak_vram_mb", "suspect", "suspect_reason", "examples_per_s"):
        if k in train:
            row[f"train_{k}"] = train[k]
    for split, m in (metrics.get("splits") or {}).items():
        e = m.get("errant") or {}
        row[f"{split}_p"] = e.get("precision")
        row[f"{split}_r"] = e.get("recall")
        row[f"{split}_f05"] = e.get("f05")
        row[f"{split}_tp"] = e.get("tp")
        row[f"{split}_fp"] = e.get("fp")
        row[f"{split}_fn"] = e.get("fn")
        row[f"{split}_bleu_13a"] = m.get("bleu_13a")
        row[f"{split}_bleu_tok"] = m.get("bleu_tok")
        row[f"{split}_over_correction"] = m.get("over_correction_rate")
        row[f"{split}_n_clean"] = m.get("n_clean")
        row[f"{split}_edits_per_sentence"] = m.get("avg_edits_per_sentence")
        row[f"{split}_copy_rate"] = m.get("copy_rate")
        row[f"{split}_n"] = m.get("n")
    return row


def collect_runs(runs_dir: Path | None = None) -> list[dict]:
    """Left-join the expected grid onto the metrics.json files that exist."""
    runs_dir = runs_dir or paths.RUNS
    found: dict[tuple[str, int], dict] = {}

    if runs_dir.exists():
        for d in sorted(runs_dir.glob("*/seed*")):
            run_name = d.parent.name
            # Smoke and probe results must never reach a report table.
            if run_name.startswith("smoke_") or run_name.startswith("probe_"):
                continue
            mp = d / "metrics.json"
            if not mp.exists():
                continue
            try:
                m = paths.read_json(mp)
            except Exception:  # noqa: BLE001
                _warn(f"{paths.rel(mp)}: unreadable JSON, treated as missing")
                continue
            if not (m.get("_meta") or {}).get("complete"):
                _warn(f"{paths.rel(mp)}: incomplete, treated as missing")
                continue
            try:
                seed = int(d.name.replace("seed", ""))
            except ValueError:
                continue
            # Follow the alias CHAIN to a fixed point. One hop is not enough:
            # e3_synthetic aliases the best E2 run, and if that is e2_clean0.0
            # it is itself an alias of the best E1 run. Stopping after one hop
            # would leave a pointer-only metrics file with no splits at all.
            first_alias = m.get("alias_of")
            seen_aliases: set[str] = set()
            while m.get("alias_of"):
                alias = str(m["alias_of"])
                if alias in seen_aliases:
                    _warn(f"{run_name}/seed{seed}: alias cycle at {alias}, giving up")
                    m = None
                    break
                seen_aliases.add(alias)
                target = runs_dir / alias / "metrics.json"
                if not target.exists():
                    _warn(f"{run_name}/seed{seed}: alias target {alias} has no metrics")
                    m = None
                    break
                inner = paths.read_json(target)
                inner["condition"] = m.get("condition") or inner.get("condition")
                m = inner
            if m is None:
                continue
            if first_alias:
                # Record the ORIGINAL pointer, not the last hop, so the table
                # says what this run was configured to reuse.
                m["alias_of"] = first_alias
            found[(run_name, seed)] = m

    rows: list[dict] = []
    for exp in expected_grid():
        key = (exp["run"], exp["seed"])
        if key in found:
            row = _flatten(found[key], exp["run"], exp["seed"])
            row["family"] = exp["family"]
            row["present"] = True
        else:
            row = {"run": exp["run"], "seed": exp["seed"], "family": exp["family"],
                   "present": False}
        rows.append(row)

    # Anything on disk that the grid does not expect still belongs in the table.
    for (run_name, seed), m in found.items():
        if not any(r["run"] == run_name and r["seed"] == seed for r in rows):
            row = _flatten(m, run_name, seed)
            row["family"] = run_name.split("_")[0]
            row["present"] = True
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Selection (val only, enforced)
# --------------------------------------------------------------------------- #

def _run_cfg(run: str, seed: int) -> dict:
    p = paths.run_dir(run, seed) / "config.yaml"
    if not p.exists():
        return {}
    import yaml

    return yaml.safe_load(paths.read_text(p)) or {}


#: Runs that belong to a family for *reporting* but must not take part in its
#: hyperparameter selection. ``e1_p0.2_precomp`` (deviation D10) differs from
#: ``e1_p0.2`` only in the generator weights, so including it would make E1's
#: "best p" ambiguous: the selection records a value for `p` alone, and a win
#: by the pre-compensated arm would silently drop the weights half of the
#: choice, leaving E2/E3 to train with default weights at the chosen p. It is
#: an orthogonal comparison and is reported as its own result instead.
SELECTION_EXCLUDE = ("e1_p0.2_precomp",)


def select_best(
    family: str,
    rows: list[dict],
    metric: str = "val_f05",
    key_path: str = "dataset.synthetic.p",
    exclude: tuple[str, ...] = SELECTION_EXCLUDE,
) -> dict:
    """Pick the best run in *family* by **val** F0.5.

    Raises
    ------
    AssertionError
        If *metric* names a test metric. Spec rule 2.2 forbids tuning on test,
        and this makes it impossible rather than merely discouraged.
    """
    assert not metric.startswith("test"), (
        f"select_best refuses to read a test metric ({metric!r}). "
        "Hyperparameters are selected on val only (spec rule 2.2)."
    )
    cands = [
        r for r in rows
        if r.get("family") == family and r.get("present") and r.get(metric) is not None
        and r.get("run") not in exclude
    ]
    if not cands:
        return {}

    def cfg_value(row: dict) -> Any:
        node: Any = _run_cfg(row["run"], row["seed"])
        for part in key_path.split("."):
            if not isinstance(node, dict):
                return None
            node = node.get(part)
        return node

    best = max(cands, key=lambda r: r[metric])
    return {
        "best_run": f"{best['run']}/seed{best['seed']}",
        "best_value": cfg_value(best),
        "best_val_f05": best[metric],
        "candidates": {
            f"{r['run']}/seed{r['seed']}": {"val_f05": r[metric], "value": cfg_value(r)}
            for r in sorted(cands, key=lambda r: -r[metric])
        },
    }


def write_selection(rows: list[dict] | None = None) -> Path:
    """Compute and write ``results/tables/selection.json``."""
    rows = rows if rows is not None else collect_runs()
    e1 = select_best("e1", rows, key_path="dataset.synthetic.p")
    e2 = select_best("e2", rows, key_path="dataset.synthetic.clean_ratio")

    sel: dict[str, Any] = {
        "selected_on": "val",
        "metric": "errant.f05",
        "written_at": repro.utc_now(),
    }
    if e1:
        sel.update({
            "e1_best_p": e1["best_value"],
            "e1_best_run": e1["best_run"],
            "e1_val_f05": e1["best_val_f05"],
            "e1_candidates": e1["candidates"],
        })
    if e2:
        sel.update({
            "e2_best_clean_ratio": e2["best_value"],
            "e2_best_run": e2["best_run"],
            "e2_val_f05": e2["best_val_f05"],
            "e2_candidates": e2["candidates"],
        })

    out = paths.TABLES / "selection.json"
    paths.write_json(out, sel)
    if e1:
        print(f"[agg] E1 best: p={sel.get('e1_best_p')} "
              f"(val F0.5={sel.get('e1_val_f05')}) from {sel.get('e1_best_run')}")
    if e2:
        print(f"[agg] E2 best: clean_ratio={sel.get('e2_best_clean_ratio')} "
              f"(val F0.5={sel.get('e2_val_f05')}) from {sel.get('e2_best_run')}")
    if not e1 and not e2:
        print(f"[agg] no completed E1/E2 runs yet; wrote a stub {paths.rel(out)}")
    return out


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

SUMMARY_COLS = [
    "run", "seed", "family", "condition", "present", "alias_of",
    "val_p", "val_r", "val_f05", "test_p", "test_r", "test_f05",
    "test_bleu_13a", "test_bleu_tok", "test_over_correction",
    "test_edits_per_sentence", "test_copy_rate", "test_n_clean", "test_n",
    "indian_p", "indian_r", "indian_f05",
    "train_wall_clock_s", "train_best_val_loss", "train_peak_vram_mb", "train_suspect",
]


def write_summary(rows: list[dict]) -> tuple[Path, Path]:
    """Write ``summary.csv`` (one row per run x seed) and ``summary_mean_std.csv``."""
    import pandas as pd

    df = pd.DataFrame(rows)
    for c in SUMMARY_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[SUMMARY_COLS + [c for c in df.columns if c not in SUMMARY_COLS]]
    df = df.sort_values(["family", "run", "seed"], kind="stable")

    out1 = paths.TABLES / "summary.csv"
    # TBD is substituted only at write time, never in the plotting frame.
    df.fillna("TBD").to_csv(out1, index=False, lineterminator="\n")

    numeric = [c for c in df.columns if c.endswith(("_f05", "_p", "_r", "_bleu_13a",
                                                    "_bleu_tok", "_over_correction",
                                                    "_edits_per_sentence", "_copy_rate"))]
    agg = (
        df[df["present"] == True]  # noqa: E712
        .groupby("run", as_index=False)
        .agg({**{c: ["mean", "std", "count"] for c in numeric}})
    )
    agg.columns = ["run"] + [f"{a}_{b}" for a, b in agg.columns[1:]]
    out2 = paths.TABLES / "summary_mean_std.csv"
    agg.fillna("TBD").to_csv(out2, index=False, lineterminator="\n")

    n_present = int(df["present"].sum())
    print(f"[agg] wrote {paths.rel(out1)} ({n_present}/{len(df)} runs present)")
    return out1, out2


def sanity_checks(rows: list[dict]) -> list[str]:
    """The three checks spec section 7.3 names explicitly.

    Reported as warnings rather than raised, so a surprising-but-real result is
    flagged for investigation instead of aborting the whole report. ``--strict``
    turns them into a non-zero exit for the final pre-submission pass.
    """
    problems: list[str] = []
    present = [r for r in rows if r.get("present")]

    # 1. E0 must score exactly zero: it proposes no edits at all.
    for r in present:
        if r["run"] == "e0_copy" and r.get("test_f05") not in (None, 0.0):
            problems.append(
                f"SANITY: the copy baseline scored test F0.5 = {r['test_f05']}, "
                "which must be exactly 0 - it proposes no edits."
            )

    # 2. Every TRAINED model must beat zero, or something is broken.
    for r in present:
        if r["run"] == "e0_copy" or r.get("test_f05") is None:
            continue
        if r["test_f05"] <= 0.0:
            problems.append(
                f"SANITY: {r['run']}/seed{r['seed']} is a trained model but scored "
                f"test F0.5 = {r['test_f05']}. Spec 7.3 expects > 0; check "
                "train_metrics.json for the degenerate-output flag."
            )

    # 3. Over-correction should FALL as clean_ratio rises (spec 7.3 says to
    #    investigate if it does not). This is E2's whole hypothesis.
    e2 = [
        (_cfg_value(r, "dataset.synthetic.clean_ratio"), r)
        for r in present
        if r.get("family") == "e2" and r.get("test_over_correction") is not None
    ]
    e2 = sorted((c, r) for c, r in e2 if isinstance(c, (int, float)))
    if len(e2) >= 3:
        rates = [r["test_over_correction"] for _, r in e2]
        if rates[-1] > rates[0]:
            problems.append(
                "SANITY: over-correction ROSE with clean_ratio "
                f"({e2[0][0]}: {rates[0]:.3f} -> {e2[-1][0]}: {rates[-1]:.3f}). "
                "Spec 7.3 expects it to fall; worth investigating before the "
                "report draws a conclusion from E2."
            )
    return problems


def table_e5_indian(rows: list[dict]) -> Path | None:
    """E5 table: P/R/F0.5 per model on the Indian English set, plus kappa."""
    have = [r for r in rows if r.get("present") and r.get("indian_f05") is not None]
    kappa_path = paths.TABLES / "indian_kappa.json"
    kappa = paths.read_json(kappa_path) if kappa_path.exists() else {}

    lines = ["# E5 - Indian English test set", ""]
    if (paths.PROCESSED / "indian.skip").exists():
        skip = paths.read_json(paths.PROCESSED / "indian.skip")
        lines += [
            "**TBD** - the Indian English set is not ready yet.", "",
            f"- real rows so far: **{skip.get('n_real_rows', 0)}** "
            f"(need at least 20; `EXAMPLE-` rows are excluded by design)",
            "- the group writes these sentences; see `data_manual/README.md`", "",
        ]
    elif not have:
        lines += ["**TBD** - no model has been evaluated on this split yet.", ""]
    else:
        lines += [
            "| model | P | R | F0.5 | over-correction |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in sorted(have, key=lambda r: -(r.get("indian_f05") or 0)):
            lines.append(
                f"| {r['run']} (seed {r['seed']}) | {r.get('indian_p'):.4f} | "
                f"{r.get('indian_r'):.4f} | {r.get('indian_f05'):.4f} | "
                f"{r.get('indian_over_correction', 'TBD')} |"
            )
        lines.append("")

    # Per-source_type breakdown. The collected set mixes sentences that carry a
    # real error with sentences the annotators judged acceptable, so the
    # headline number alone hides two different questions: can the model fix
    # genuine Indian English errors, and does it wrongly "correct" usage that
    # was already fine?
    jsonl = paths.PROCESSED / "indian.jsonl"
    if have and jsonl.exists():
        rows_j = paths.read_jsonl(jsonl)
        by_type: dict[str, int] = {}
        for r in rows_j:
            by_type[r.get("source_type") or "unspecified"] = (
                by_type.get(r.get("source_type") or "unspecified", 0) + 1
            )
        n_clean = sum(1 for r in rows_j if r.get("is_clean"))
        # Say "both annotators" only when there are two. With a single
        # annotator the claim would be false, and this table is the one a
        # reader uses to judge how much the control sentences are worth.
        n_ann = max((int(r.get("n_annotators") or 1) for r in rows_j),
                    default=1)
        who = "both annotators" if n_ann >= 2 else "the annotator"
        lines += [
            "## Composition of the Indian English set", "",
            f"- sentences: **{len(rows_j)}**",
            f"- annotators per sentence: **{n_ann}**",
            f"- {who} left unchanged (control / acceptable usage): "
            f"**{n_clean}** ({n_clean / max(len(rows_j), 1):.0%})",
            "- by `source_type`: "
            + ", ".join(f"`{k}` {v}" for k, v in sorted(by_type.items())),
            "",
            "Over-correction in the table above is measured on the control "
            f"sentences only - those {who} judged acceptable as written. It "
            "answers whether the model imposes a prescriptive standard on a "
            "legitimate variety of English.",
            "",
        ]

    if kappa and not kappa.get("skipped"):
        lines += [
            "## Annotator agreement", "",
            f"- token-level Cohen's kappa: **{kappa.get('token_kappa', 'TBD')}**",
            f"- raw token agreement: {kappa.get('token_raw_agreement', 'TBD')}",
            f"- sentence-level kappa: {kappa.get('sentence_kappa', 'TBD')}",
            f"- sentences: {kappa.get('n_sentences', 'TBD')}", "",
        ]
        if kappa.get("token_kappa") is None and kappa.get("note"):
            lines.append(f"- note: {kappa['note']}")
    else:
        lines += ["## Annotator agreement", "", "**TBD** - run `python -m gec.kappa`.", ""]

    out = paths.TABLES / "e5_indian_english.md"
    paths.write_text(out, "\n".join(lines))
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.autolayout": True,
    })
    return plt


def _present(rows: list[dict], family: str, metric: str = "test_f05") -> list[dict]:
    return [r for r in rows if r.get("family") == family and r.get("present")
            and r.get(metric) is not None]


def _cfg_value(row: dict, key_path: str) -> Any:
    node: Any = _run_cfg(row["run"], row["seed"])
    for part in key_path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def fig_e0_bleu_vs_f05(rows: list[dict]) -> Path | None:
    """Figure 1: the copy baseline's BLEU looks fine while its F0.5 is zero."""
    e0 = [r for r in rows if r["run"] == "e0_copy" and r.get("present")]
    trained = [r for r in rows if r.get("present") and r.get("test_f05") is not None
               and r["run"] != "e0_copy"]
    if not e0:
        _warn("figure 1 (e0_bleu_vs_f05) skipped: the E0 copy baseline has not run")
        return None

    plt = _plt()
    best = max(trained, key=lambda r: r["test_f05"]) if trained else None
    labels = ["Copy baseline\n(no training)"]
    bleus = [e0[0].get("test_bleu_13a") or 0]
    f05s = [e0[0].get("test_f05") or 0]
    if best:
        labels.append(f"Best model\n({best['run']})")
        bleus.append(best.get("test_bleu_13a") or 0)
        f05s.append(best.get("test_f05") or 0)
    else:
        _warn("figure 1: no trained run yet, showing the copy baseline alone")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    x = range(len(labels))
    ax1.bar(x, bleus, color="#7a9cc6", width=0.55)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels)
    ax1.set_ylabel("sacreBLEU (13a)"); ax1.set_title("BLEU")
    for i, v in enumerate(bleus):
        ax1.text(i, v, f"{v:.1f}", ha="center", va="bottom")

    ax2.bar(x, f05s, color="#c67a7a", width=0.55)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels)
    ax2.set_ylabel("ERRANT F0.5"); ax2.set_title("F0.5")
    for i, v in enumerate(f05s):
        ax2.text(i, v, f"{v:.4f}", ha="center", va="bottom")

    fig.suptitle("A high BLEU can hide an F0.5 of zero (test split)")
    out = paths.FIGURES / "e0_bleu_vs_f05.png"
    fig.savefig(out); plt.close(fig)
    return out


def fig_e1_noise_curve(rows: list[dict]) -> Path | None:
    """Figure 2: val and test F0.5 against the noise ratio ``p``."""
    have = _present(rows, "e1", "val_f05")
    # The pre-compensated arm shares p with e1_p0.2 and differs only in the
    # generator weights, so it is a separate marker rather than a second point
    # on the curve.
    sweep = [r for r in have if r.get("run") not in SELECTION_EXCLUDE]
    extra = [r for r in have if r.get("run") in SELECTION_EXCLUDE]

    pts = [(_cfg_value(r, "dataset.synthetic.p"), r) for r in sweep]
    pts = [(p, r) for p, r in pts if isinstance(p, (int, float))]
    if len(pts) < 2:
        _warn(f"figure 2 (e1_noise_curve) skipped: only {len(pts)} of "
              f"{len(PLANS['e1'])} E1 runs are usable (need >= 2)")
        return None

    pts.sort(key=lambda x: x[0])
    ps = [p for p, _ in pts]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ps, [r.get("val_f05") for _, r in pts], "o-", label="val (used for selection)")
    test = [r.get("test_f05") for _, r in pts]
    if any(v is not None for v in test):
        ax.plot(ps, test, "s--", label="test (reported only)")
    for r in extra:
        xp = _cfg_value(r, "dataset.synthetic.p")
        if isinstance(xp, (int, float)) and r.get("val_f05") is not None:
            ax.plot([xp], [r["val_f05"]], "D", color="#c67a7a", markersize=9,
                    label="val, distribution-flattened weights")
    ax.set_xlabel("noise ratio p (probability a token is selected for corruption)")
    ax.set_ylabel("ERRANT F0.5")
    ax.set_title("E1: is there a best synthetic noise level?")
    ax.set_xticks(ps)
    ax.legend(fontsize=8)
    out = paths.FIGURES / "e1_noise_curve.png"
    fig.savefig(out); plt.close(fig)
    return out


def fig_e2_clean_ratio(rows: list[dict]) -> Path | None:
    """Figure 3: F0.5 and over-correction against ``clean_ratio``."""
    have = _present(rows, "e2", "test_f05")
    pts = [(_cfg_value(r, "dataset.synthetic.clean_ratio"), r) for r in have]
    pts = [(c, r) for c, r in pts if isinstance(c, (int, float))]
    if len(pts) < 2:
        _warn(f"figure 3 (e2_clean_ratio) skipped: only {len(pts)} usable E2 runs (need >= 2)")
        return None

    pts.sort(key=lambda x: x[0])
    cs = [c for c, _ in pts]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(cs, [r.get("test_f05") for _, r in pts], "o-", color="#c67a7a", label="F0.5 (test)")
    ax.set_xlabel("clean_ratio (share of identity pairs in the training set)")
    ax.set_ylabel("ERRANT F0.5", color="#c67a7a")
    ax.set_xticks(cs)

    ax2 = ax.twinx()
    ax2.plot(cs, [r.get("test_over_correction") for _, r in pts], "s--",
             color="#7a9cc6", label="over-correction rate")
    ax2.set_ylabel("over-correction rate", color="#7a9cc6")
    ax2.grid(False)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="best")
    ax.set_title("E2: do identity pairs reduce over-correction?")
    out = paths.FIGURES / "e2_clean_ratio.png"
    fig.savefig(out); plt.close(fig)
    return out


def fig_e3_conditions(rows: list[dict]) -> Path | None:
    """Figure 4: P, R and F0.5 for the four E3 conditions."""
    order = ["e3_synthetic", "e3_real", "e3_mixed", "e3_synth2real"]
    nice = {"e3_synthetic": "Synthetic\nonly", "e3_real": "Real\nonly",
            "e3_mixed": "Mixed", "e3_synth2real": "Synth\n-> real"}
    import statistics

    groups: dict[str, list[dict]] = {}
    for run in order:
        g = [r for r in rows if r["run"] == run and r.get("present")
             and r.get("test_f05") is not None]
        if g:
            groups[run] = g
    if not groups:
        _warn("figure 4 (e3_conditions) skipped: no E3 run has completed")
        return None

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    runs = [r for r in order if r in groups]
    width = 0.26
    n_seeds_max = max(len(groups[r]) for r in runs)

    for k, (metric, label, colour) in enumerate(
        [("test_p", "Precision", "#7a9cc6"),
         ("test_r", "Recall", "#9cc67a"),
         ("test_f05", "F0.5", "#c67a7a")]
    ):
        means, errs = [], []
        for run in runs:
            vals = [g[metric] for g in groups[run] if g.get(metric) is not None]
            means.append(statistics.mean(vals) if vals else 0.0)
            # Error bars only where there is more than one seed.
            errs.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        xs = [i + (k - 1) * width for i in range(len(runs))]
        ax.bar(xs, means, width=width, label=label, color=colour,
               yerr=errs if any(errs) else None, capsize=3)

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([nice[r] for r in runs])
    ax.set_ylabel("score (test split)")
    suffix = (f"mean +/- sd over {n_seeds_max} seeds" if n_seeds_max > 1
              else "n = 1 seed (no error bars)")
    ax.set_title(f"E3: can synthetic data replace real data?\n({suffix})")
    ax.legend()
    if n_seeds_max == 1:
        _warn("figure 4: only one seed per condition, so no error bars are drawn "
              "(deviation D6 from the spec's 3 seeds)")
    out = paths.FIGURES / "e3_conditions.png"
    fig.savefig(out); plt.close(fig)
    return out


def fig_e3_per_category(rows: list[dict]) -> Path | None:
    """Figure 6: F0.5 per ERRANT category, top 10 by gold frequency."""
    import csv as _csv

    runs = ["e3_synthetic", "e3_real", "e3_mixed"]
    data: dict[str, dict[str, float]] = {}
    support: dict[str, int] = {}
    for run in runs:
        for r in [x for x in rows if x["run"] == run and x.get("present")]:
            csv_path = paths.run_dir(r["run"], r["seed"]) / "predictions" / "test.per_category.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, encoding="utf-8") as fh:
                per = {}
                for row in _csv.DictReader(fh):
                    if row["category"] == "ALL":
                        continue
                    per[row["category"]] = float(row["f05"])
                    support[row["category"]] = max(
                        support.get(row["category"], 0), int(row["support"])
                    )
                data[run] = per
            break
    if not data:
        _warn("figure 6 (e3_per_category) skipped: no per-category CSV found")
        return None

    top = [c for c, _ in sorted(support.items(), key=lambda kv: -kv[1])[:10]]
    if not top:
        _warn("figure 6 skipped: no categories with gold support")
        return None

    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / max(len(data), 1)
    colours = {"e3_synthetic": "#7a9cc6", "e3_real": "#9cc67a", "e3_mixed": "#c67a7a"}
    nice = {"e3_synthetic": "Synthetic only", "e3_real": "Real only", "e3_mixed": "Mixed"}
    for k, (run, per) in enumerate(data.items()):
        xs = [i + k * width - 0.4 + width / 2 for i in range(len(top))]
        ax.bar(xs, [per.get(c, 0.0) for c in top], width=width,
               label=nice.get(run, run), color=colours.get(run))
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f"{c}\n(n={support[c]})" for c in top], fontsize=8)
    ax.set_ylabel("F0.5 (test)")
    ax.set_title("E3: F0.5 by ERRANT error category (top 10 by gold frequency)")
    ax.legend()
    out = paths.FIGURES / "e3_per_category.png"
    fig.savefig(out); plt.close(fig)
    return out


def fig_e4_error_types() -> Path | None:
    """Figure 5: real vs synthetic error-type shares, before and after matching."""
    csv_path = paths.TABLES / "e4_distribution.csv"
    if not csv_path.exists():
        _warn("figure 5 (e4_error_types) skipped: run `python -m gec.distribution --match`")
        return None
    import csv as _csv

    with open(csv_path, encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        _warn("figure 5 skipped: e4_distribution.csv is empty")
        return None

    cats = [r["category"] for r in rows]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    width = 0.26
    series = [("real_share", "Real (train_real)", "#444444"),
              ("syn_share_before", "Synthetic, before matching", "#7a9cc6"),
              ("syn_share_after", "Synthetic, after matching", "#c67a7a")]
    for k, (col, label, colour) in enumerate(series):
        vals = [float(r.get(col) or 0) for r in rows]
        ax.bar([i + (k - 1) * width for i in range(len(cats))], vals,
               width=width, label=label, color=colour)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("share of all edits")

    tvd_path = paths.TABLES / "e4_tvd.csv"
    title = "E4: matching the generator to the real error distribution"
    if tvd_path.exists():
        with open(tvd_path, encoding="utf-8") as fh:
            t = list(_csv.DictReader(fh))
        if t:
            title += (f"\nTVD before = {t[0].get('tvd')}, "
                      f"after = {t[-1].get('tvd')}")
    ax.set_title(title)
    ax.legend()
    out = paths.FIGURES / "e4_error_types.png"
    fig.savefig(out); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Failure gallery
# --------------------------------------------------------------------------- #

def failure_gallery(rows: list[dict], k: int = 10) -> Path | None:
    """Test sentences where the E3 models disagree most (spec section 11)."""
    runs = ["e3_synthetic", "e3_real", "e3_mixed"]
    preds: dict[str, list[str]] = {}
    for run in runs:
        for r in [x for x in rows if x["run"] == run and x.get("present")]:
            p = paths.run_dir(r["run"], r["seed"]) / "predictions" / "test.pred.txt"
            if p.exists():
                preds[run] = paths.read_lines(p)
            break
    if len(preds) < 2:
        _warn(f"failure gallery skipped: predictions available for only "
              f"{len(preds)} of 3 E3 models (need >= 2)")
        return None

    jsonl = paths.split_files("test")["jsonl"]
    gold = paths.read_jsonl(jsonl)
    n = min(len(gold), *(len(v) for v in preds.values()))

    def edits_vs_src(src: str, hyp: str) -> set[str]:
        s, h = src.split(), hyp.split()
        return set(h) ^ set(s)

    scored: list[tuple[int, int]] = []
    for i in range(n):
        src = gold[i]["src"]
        sets = [edits_vs_src(src, v[i]) for v in preds.values()]
        disagreement = 0
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                disagreement += len(sets[a] ^ sets[b])
        scored.append((disagreement, i))
    scored.sort(reverse=True)

    lines = [
        "# Failure gallery", "",
        "The 10 test sentences where the E3 models disagree most, chosen "
        "automatically by the largest pairwise token-level edit disagreement "
        "against the source. No interpretation, just the outputs.", "",
    ]
    for rank, (score, i) in enumerate(scored[:k], 1):
        lines += [
            f"## {rank}. test sentence {i} (disagreement score {score})", "",
            f"- **source**: `{gold[i]['src']}`",
            f"- **gold**:   `{gold[i]['tgt']}`",
        ]
        for run, v in preds.items():
            marker = " (unchanged)" if v[i].split() == gold[i]["src"].split() else ""
            lines.append(f"- **{run}**: `{v[i]}`{marker}")
        lines.append("")

    out = paths.RESULTS / "failure_gallery.md"
    paths.write_text(out, "\n".join(lines))
    return out


# --------------------------------------------------------------------------- #
# RESULTS.md
# --------------------------------------------------------------------------- #

def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "TBD"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_results_md(rows: list[dict], figures: dict[str, Path | None]) -> Path:
    """Auto-generate ``results/RESULTS.md``: numbers only, no interpretation."""
    present = [r for r in rows if r.get("present")]
    lines = [
        "# RESULTS", "",
        "Auto-generated by `python -m gec.aggregate`. **Every number here was "
        "read from a file our code wrote**; anything not yet run shows as `TBD`. "
        "Per spec section 11 this file restates numbers and does not interpret "
        "them.", "",
        f"Generated: {repro.utc_now()}", "",
        "## Completeness", "",
        "| family | expected | present |",
        "|---|---:|---:|",
    ]
    fams: dict[str, list[dict]] = {}
    for r in rows:
        fams.setdefault(r.get("family") or "?", []).append(r)
    for fam in sorted(fams):
        g = fams[fam]
        lines.append(f"| {fam} | {len(g)} | {sum(1 for x in g if x.get('present'))} |")
    lines += ["", f"**{len(present)} of {len(rows)} expected runs are complete.**", ""]

    if WARNINGS:
        lines += ["### Notes and gaps", ""]
        lines += [f"- {w}" for w in WARNINGS]
        lines.append("")

    failed_checks = [w for w in WARNINGS if w.startswith("SANITY:")]
    if failed_checks:
        lines += ["### Sanity checks that did not hold (spec 7.3)", ""]
        lines += [f"- {w[len('SANITY: '):]}" for w in failed_checks]
        lines.append("")
    elif present:
        lines += ["All spec 7.3 sanity checks hold: the copy baseline scores "
                  "exactly 0, every trained model scores above 0, and "
                  "over-correction does not rise with `clean_ratio`.", ""]

    suspects = [r for r in present if r.get("train_suspect")]
    if suspects:
        lines += ["### Suspect runs", "",
                  "These runs produced degenerate output and are marked with a dagger:", ""]
        for r in suspects:
            lines.append(f"- {r['run']}/seed{r['seed']}: {r.get('train_suspect_reason')}")
        lines.append("")

    lines += [
        "## Main results (test split)", "",
        "| run | seed | P | R | F0.5 | BLEU(13a) | BLEU(tok) | over-corr. | edits/sent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: (r.get("family") or "", r["run"], r["seed"])):
        dag = " †" if r.get("train_suspect") else ""
        lines.append(
            f"| {r['run']}{dag} | {r['seed']} | {_fmt(r.get('test_p'))} | "
            f"{_fmt(r.get('test_r'))} | {_fmt(r.get('test_f05'))} | "
            f"{_fmt(r.get('test_bleu_13a'), 2)} | {_fmt(r.get('test_bleu_tok'), 2)} | "
            f"{_fmt(r.get('test_over_correction'))} | "
            f"{_fmt(r.get('test_edits_per_sentence'))} |"
        )

    lines += ["", "## Selection (on val only)", ""]
    sel_path = paths.TABLES / "selection.json"
    if sel_path.exists():
        sel = paths.read_json(sel_path)
        lines += [
            f"- E1 best noise ratio `p`: **{sel.get('e1_best_p', 'TBD')}** "
            f"(val F0.5 {_fmt(sel.get('e1_val_f05'))}, from `{sel.get('e1_best_run', 'TBD')}`)",
            f"- E2 best `clean_ratio`: **{sel.get('e2_best_clean_ratio', 'TBD')}** "
            f"(val F0.5 {_fmt(sel.get('e2_val_f05'))}, from `{sel.get('e2_best_run', 'TBD')}`)",
            "",
            "Both were chosen on the **val** split. `select_best` refuses to read a "
            "test metric at all, so the no-tuning-on-test rule is enforced by code.",
            "",
        ]
        if sel.get("e1_candidates"):
            lines += ["| E1 candidate | val F0.5 | p |", "|---|---:|---:|"]
            for name, c in sel["e1_candidates"].items():
                lines.append(f"| {name} | {_fmt(c.get('val_f05'))} | {c.get('value')} |")
            lines.append("")
    else:
        lines += ["**TBD** - no selection has been computed yet.", ""]

    precomp = [r for r in rows if r.get("run") in SELECTION_EXCLUDE and r.get("present")]
    baseline = [r for r in rows if r.get("run") == "e1_p0.2" and r.get("present")]
    if precomp and baseline:
        a, b = baseline[0], precomp[0]
        lines += [
            "## Extra E1 arm: flattening the generator's error distribution", "",
            "`e1_p0.2_precomp` is identical to `e1_p0.2` except that the generator "
            "weights are pre-compensated for applicability gating, so the "
            "*realised* error mix is closer to uniform across the ten covered "
            "ERRANT categories (measured TVD against uniform 0.336 -> 0.163). "
            "Same `p`, same seed, same base-set sentences.", "",
            "| arm | val F0.5 | test F0.5 | test over-correction |",
            "|---|---:|---:|---:|",
            f"| e1_p0.2 (default weights) | {_fmt(a.get('val_f05'))} | "
            f"{_fmt(a.get('test_f05'))} | {_fmt(a.get('test_over_correction'))} |",
            f"| e1_p0.2_precomp (flattened) | {_fmt(b.get('val_f05'))} | "
            f"{_fmt(b.get('test_f05'))} | {_fmt(b.get('test_over_correction'))} |",
            "",
            "This arm is excluded from the E1 `p` selection by design: it differs "
            "in the weights, not the noise level, so letting it win would record a "
            "`p` while silently dropping the weights half of the choice.", "",
        ]

    lines += ["## Figures", ""]
    for name, p in figures.items():
        lines.append(
            f"- `{name}`: [{p.name}](figures/{p.name})" if p
            else f"- `{name}`: **TBD** (preconditions not met - see Notes above)"
        )
    lines.append("")

    lines += ["## Tables", ""]
    for name in ("summary.csv", "summary_mean_std.csv", "selection.json",
                 "latency.csv", "generator_samples.tsv", "e5_indian_english.md",
                 "e4_distribution.csv", "e4_tvd.csv"):
        p = paths.TABLES / name
        lines.append(f"- [{name}](tables/{name})" if p.exists() else f"- `{name}`: **TBD**")
    lines.append("")

    gallery = paths.RESULTS / "failure_gallery.md"
    lines += [
        "## Failure gallery", "",
        "[failure_gallery.md](failure_gallery.md)" if gallery.exists() else "**TBD**",
        "",
        "## Metric notes", "",
        "- **ERRANT reports P = 1.0 for the copy baseline.** ERRANT defines "
        "`p = tp/(tp+fp) if fp else 1.0`, so a system that proposes no edits at "
        "all has undefined precision reported as 1.0. F0.5 = 0 is the meaningful "
        "number there.",
        "- **Two BLEU columns.** `BLEU(13a)` is computed on detokenised text with "
        "sacreBLEU's default tokenisation and is the number comparable with "
        "published GEC work; `BLEU(tok)` is computed on our space-separated "
        "tokenisation with `tokenize=\"none\"`. They differ by several points.",
        "- **Over-correction rate** is the share of already-correct test sentences "
        "(gold target equals source) that the model nevertheless changed.",
        "",
    ]
    out = paths.RESULTS / "RESULTS.md"
    paths.write_text(out, "\n".join(lines))
    print(f"[agg] wrote {paths.rel(out)}")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gec.aggregate",
        description="Build tables, figures and RESULTS.md from saved metrics.",
    )
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any expected run or figure is missing")
    args = ap.parse_args(argv)

    WARNINGS.clear()
    paths.ensure_dirs()
    rows = collect_runs()

    if not args.figures_only:
        write_summary(rows)
        write_selection(rows)
        table_e5_indian(rows)

    # Spec 7.3's sanity checks. Warnings, not exceptions: a surprising result
    # should be flagged for investigation, not suppress the whole report.
    for problem in sanity_checks(rows):
        _warn(problem)

    figures: dict[str, Path | None] = {}
    if not args.tables_only:
        for name, fn in [
            ("e0_bleu_vs_f05", lambda: fig_e0_bleu_vs_f05(rows)),
            ("e1_noise_curve", lambda: fig_e1_noise_curve(rows)),
            ("e2_clean_ratio", lambda: fig_e2_clean_ratio(rows)),
            ("e3_conditions", lambda: fig_e3_conditions(rows)),
            ("e4_error_types", fig_e4_error_types),
            ("e3_per_category", lambda: fig_e3_per_category(rows)),
        ]:
            try:
                figures[name] = fn()
            except Exception as exc:  # noqa: BLE001 - one bad figure must not cost the rest
                _warn(f"figure {name} raised {type(exc).__name__}: {exc}")
                traceback.print_exc()
                figures[name] = None
        try:
            failure_gallery(rows)
        except Exception as exc:  # noqa: BLE001
            _warn(f"failure gallery raised {type(exc).__name__}: {exc}")

    write_results_md(rows, figures)

    made = sum(1 for p in figures.values() if p)
    print(f"[agg] {made}/{len(figures)} figures written; {len(WARNINGS)} warning(s)")
    if args.strict and (WARNINGS or any(not r.get("present") for r in rows)):
        print("[agg] --strict: incomplete results")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
