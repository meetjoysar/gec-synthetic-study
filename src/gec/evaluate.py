"""Evaluation: ERRANT P/R/F0.5, sacreBLEU, over-correction, latency (spec 7).

::

    python -m gec.evaluate --run results/runs/e3_mixed/seed42 --split test
    python -m gec.evaluate --run results/runs/e3_mixed/seed42 --all-splits
    python -m gec.evaluate --latency --run results/runs/e3_real/seed13

**ERRANT is the scorer; we never reimplement it.** ``errant_compare -cat 2`` is
the official tool including its best-reference search, and reimplementing that
would be the single worst correctness risk in this project. What we *do* add is
:func:`score_m2_pure_python`, an independent reimplementation used **only as a
cross-check**: for a single-annotator reference it is mathematically identical
to ERRANT, so :func:`evaluate_run` asserts the two agree within 0.005. That
makes the stdout parsers self-validating - if a future ERRANT changes its output
format, the assertion fires loudly instead of silently producing wrong numbers.

File flow for one run/seed/split::

    predictions/<split>.src.txt   <- predict.py (non-blank, original order)
    predictions/<split>.pred.txt  <- predict.py (one line per source)
        |  errant_parallel -orig src -cor pred
        v
    predictions/<split>.hyp.m2
        |  errant_compare -hyp <split>.hyp.m2 -ref data/processed/<split>.m2 -cat 2
        v
    logs/errant_compare.<split>.txt      (raw stdout, saved BEFORE parsing)
    predictions/<split>.per_category.csv (parsed category table + an ALL row)
    metrics.json                         (overall + BLEU + over-correction)
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from gec import errant_cli, m2, paths, repro, tok


class ErrantParseError(RuntimeError):
    """``errant_compare`` stdout did not look the way we expect."""


# --------------------------------------------------------------------------- #
# Parsing errant_compare stdout
# --------------------------------------------------------------------------- #

#: The category table header, printed as
#: ``print("Category".ljust(14), "TP".ljust(8), ..., "F"+str(beta))``.
CAT_HEADER = re.compile(r"^Category\s+TP\s+FP\s+FN\s+P\s+R\s+F[\d.]+\s*$")

#: The overall table is TAB-separated: ``"\t".join(["TP","FP","FN","Prec","Rec","F0.5"])``.
OVERALL_HEADER = ("TP", "FP", "FN", "Prec", "Rec")


def parse_errant_categories(out: str) -> list[dict]:
    """Parse the per-category table from ``errant_compare -cat 2`` stdout.

    The block is printed with ``str.ljust`` padding and joined by single spaces,
    and ``-cat 2`` strips the ``M:``/``U:``/``R:`` prefix (``cat[2:]``) while
    passing ``UNK`` through verbatim. Category names therefore never contain a
    space, so ``split()`` yields exactly 7 fields and the parse is unambiguous.
    """
    rows: list[dict] = []
    inside = False
    for line in out.splitlines():
        s = line.strip()
        if CAT_HEADER.match(s):
            inside = True
            continue
        if not inside:
            continue
        if not s or s.startswith("=") or s.startswith("-"):
            break
        parts = s.split()
        if len(parts) != 7:
            continue
        cat, tp, fp, fn, p, r, f = parts
        try:
            rows.append(
                {
                    "category": cat,
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "p": float(p),
                    "r": float(r),
                    "f05": float(f),
                    "support": int(tp) + int(fn),
                }
            )
        except ValueError:
            continue
    if not rows:
        raise ErrantParseError(
            "no category rows found in errant_compare output.\n"
            "--- last 3000 chars ---\n" + out[-3000:]
        )
    return rows


def parse_errant_overall(out: str) -> dict:
    """Parse the overall span-level P/R/F table (tab-separated)."""
    lines = out.splitlines()
    for i, line in enumerate(lines):
        parts = line.strip().split("\t")
        if len(parts) == 6 and tuple(parts[:5]) == OVERALL_HEADER:
            if i + 1 >= len(lines):
                break
            vals = lines[i + 1].strip().split("\t")
            if len(vals) != 6:
                break
            tp, fp, fn, p, r, f = vals
            return {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(p),
                "recall": float(r),
                "f05": float(f),
            }
    raise ErrantParseError(
        "overall score block not found in errant_compare output.\n"
        "--- last 3000 chars ---\n" + out[-3000:]
    )


# --------------------------------------------------------------------------- #
# Independent scorer, for cross-checking the parsers
# --------------------------------------------------------------------------- #

def _m2_edit_dicts(path: Path | str) -> list[dict[int, dict[tuple, list[str]]]]:
    """Per sentence, per coder, ERRANT's ``process_edits`` dictionary.

    Mirrors ``errant.commands.compare_m2.process_edits`` in its **default**
    mode - span-based correction *without* error-type classification. Three
    details matter and are easy to get wrong:

    * The key is ``(start, end, cor)`` and **excludes the category**. Including
      it would be ``-cse`` mode, which scores strictly lower.
    * The value is a *list* of categories, because two edits can share one
      span, and ERRANT counts them with multiplicity. A ``set`` would silently
      undercount.
    * A sentence with no edits gets an explicit ``noop`` entry. That is what
      keeps coder 0 present in the dict, so the hyp/ref combination loop still
      runs for that sentence. ``noop`` entries can never be TP, FP or FN.
    * ``UNK`` is dropped in correction mode (it is only useful for detection).
    """
    out: list[dict[int, dict[tuple, list[str]]]] = []
    for block in m2.read_m2(path):
        per: dict[int, dict[tuple, list[str]]] = defaultdict(dict)
        edits = list(block.edits) or [m2.Edit(-1, -1, "noop", "-NONE-", 0)]
        for e in edits:
            per.setdefault(e.annotator, {})
            if e.type == "UNK":
                continue
            key = (e.start, e.end, e.cor)
            per[e.annotator].setdefault(key, []).append(e.type)
        out.append(dict(per))
    return out


def _compare_edits(
    hyp: dict[tuple, list[str]], ref: dict[tuple, list[str]]
) -> tuple[int, int, int]:
    """ERRANT's ``compareEdits``: TP/FP/FN for one hyp coder vs one ref coder."""
    tp = fp = fn = 0
    for key, h_cats in hyp.items():
        if h_cats and h_cats[0] == "noop":
            continue
        if key in ref:
            tp += len(ref[key])      # ERRANT counts using the REF categories
        else:
            fp += len(h_cats)
    for key, r_cats in ref.items():
        if r_cats and r_cats[0] == "noop":
            continue
        if key not in hyp:
            fn += len(r_cats)
    return tp, fp, fn


def f_beta(tp: int, fp: int, fn: int, beta: float = 0.5) -> tuple[float, float, float]:
    """P, R and F-beta, byte-for-byte as ERRANT's ``computeFScore`` does it.

    Note the guards: ``p = tp/(tp+fp) if fp else 1.0``. A system that proposes
    **no** edits at all therefore scores P = 1.0, R = 0.0, F0.5 = 0.0, which is
    why the E0 copy baseline reports perfect precision. That is ERRANT's
    convention for an empty hypothesis set, not a bug; ``RESULTS.md`` annotates
    it rather than "fixing" it.

    The rounding to 4dp is also deliberate: ERRANT rounds here, and this
    function is called inside the best-combination search, where rounding
    changes which combination wins a tie.
    """
    p = float(tp) / (tp + fp) if fp else 1.0
    r = float(tp) / (tp + fn) if fn else 1.0
    f = (1 + beta**2) * p * r / ((beta**2) * p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def score_m2_pure_python(
    hyp_m2: Path | str, ref_m2: Path | str, beta: float = 0.5
) -> dict:
    """Independent P/R/F-beta, used only to cross-check the stdout parsers.

    A faithful reimplementation of ``errant_compare``'s default span-correction
    mode, including its greedy best-combination search: for each sentence it
    picks the hyp/ref coder pair that maximises the F-score of the **running
    corpus totals**, not of that sentence alone, with ERRANT's exact tie-break
    order (higher F, then higher TP, then lower FP, then lower FN).

    Because it agrees with ERRANT by construction, :func:`errant_scores`
    asserts the two match within 0.005 on single-annotator references. That is
    what makes the stdout parsers self-validating: a future ERRANT release that
    changed its output format would break the assertion loudly rather than
    silently yield wrong numbers.
    """
    hyp = _m2_edit_dicts(hyp_m2)
    ref = _m2_edit_dicts(ref_m2)
    if len(hyp) != len(ref):
        raise AssertionError(
            f"hyp/ref block mismatch: {hyp_m2} has {len(hyp)}, {ref_m2} has {len(ref)}"
        )

    tp = fp = fn = 0
    for h_coders, r_coders in zip(hyp, ref):
        best = (0, 0, 0)
        best_f = -1.0
        for h_id in h_coders:
            for r_id in r_coders:
                t, f_pos, f_neg = _compare_edits(h_coders[h_id], r_coders[r_id])
                # ERRANT maximises the GLOBAL (running corpus) F-score.
                _, _, f = f_beta(tp + t, fp + f_pos, fn + f_neg, beta)
                if (
                    f > best_f
                    or (f == best_f and t > best[0])
                    or (f == best_f and t == best[0] and f_pos < best[1])
                    or (
                        f == best_f and t == best[0]
                        and f_pos == best[1] and f_neg < best[2]
                    )
                ):
                    best, best_f = (t, f_pos, f_neg), f
        tp += best[0]
        fp += best[1]
        fn += best[2]

    p, r, f = f_beta(tp, fp, fn, beta)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4),
            "recall": round(r, 4), "f05": round(f, 4)}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def over_correction(
    srcs: Sequence[str], hyps: Sequence[str], is_clean: Sequence[bool]
) -> dict:
    """Of the sentences whose gold target equals the source, how many changed?

    Spec section 7's over-correction rate. Comparing on ``.split()`` rather
    than raw strings makes it insensitive to incidental whitespace. E2 predicts
    this falls as ``clean_ratio`` rises; spec section 7.3 says to investigate if
    it does not.
    """
    idx = [i for i, c in enumerate(is_clean) if c]
    if not idx:
        return {"over_correction_rate": None, "n_clean": 0, "n_clean_changed": 0}
    changed = sum(1 for i in idx if srcs[i].split() != hyps[i].split())
    return {
        "over_correction_rate": round(changed / len(idx), 4),
        "n_clean": len(idx),
        "n_clean_changed": changed,
    }


def bleu_scores(hyps: Sequence[str], refs_per_sent: Sequence[Sequence[str]]) -> dict:
    """sacreBLEU, reported two ways because E0 is a critique of BLEU.

    ``bleu_13a``
        On **detokenised** text with sacreBLEU's default ``13a`` tokenisation.
        This is the headline number and the one comparable with published GEC
        BLEU, including Koo et al.
    ``bleu_tok``
        On our space-separated tokenised strings with ``tokenize="none"``, i.e.
        BLEU over our own tokenisation.

    They differ by several points, so an unlabelled BLEU would be exactly the
    sloppiness E0 exists to criticise. Sentences with no usable reference (a
    handful of W&I annotations delete the whole sentence) are excluded, and the
    count is reported.
    """
    import sacrebleu

    keep = [i for i, refs in enumerate(refs_per_sent) if any(r.strip() for r in refs)]
    n_excluded = len(hyps) - len(keep)
    if not keep:
        return {"bleu_13a": None, "bleu_tok": None, "n_refs": 0,
                "n_bleu_excluded": n_excluded}

    h = [hyps[i] for i in keep]
    r_lists = [[x for x in refs_per_sent[i] if x.strip()] for i in keep]
    max_refs = max(len(x) for x in r_lists)
    # sacreBLEU wants a list per reference slot; pad short rows by repeating
    # the first reference, which is how multi-reference corpora are normally
    # squared off and never invents a new correction.
    streams = [[(x[j] if j < len(x) else x[0]) for x in r_lists] for j in range(max_refs)]

    detok_h = [tok.detokenise(x) for x in h]
    detok_streams = [[tok.detokenise(x) for x in s] for s in streams]

    return {
        "bleu_13a": round(sacrebleu.corpus_bleu(detok_h, detok_streams).score, 4),
        # force=True suppresses sacreBLEU's "looks like you forgot to detokenize"
        # warning, which is CORRECT here and not a mistake: bleu_tok is
        # deliberately BLEU over our own space-separated tokenisation, so the
        # input really is tokenised and `tokenize="none"` really is intended.
        "bleu_tok": round(
            sacrebleu.corpus_bleu(list(h), streams, tokenize="none", force=True).score, 4
        ),
        "n_refs": max_refs,
        "n_bleu_excluded": n_excluded,
    }


def avg_edits_per_sentence(hyp_m2: Path | str) -> float:
    """Mean number of scoreable edits the model proposed per sentence."""
    total = 0
    n = 0
    for block in m2.read_m2(hyp_m2):
        n += 1
        total += len(block.edits_for(0))
    return round(total / n, 4) if n else 0.0


def copy_rate(srcs: Sequence[str], hyps: Sequence[str]) -> float:
    """Fraction of sentences the model left exactly unchanged."""
    if not srcs:
        return 0.0
    same = sum(1 for s, h in zip(srcs, hyps) if s.split() == h.split())
    return round(same / len(srcs), 4)


# --------------------------------------------------------------------------- #
# Reference resolution
# --------------------------------------------------------------------------- #

def reference_m2(split: str, run: Path | None = None) -> Path:
    """The gold M2 to score *split* against.

    ``test``   byte copy of ``ABCN.dev.gold.bea19.m2``, original order.
    ``val``    a verbatim 2,000-block slice of the training M2, carrying the
               ORIGINAL gold annotations (see :mod:`gec.prepare_data`).
    ``indian`` built by ``errant_parallel`` from the two annotator columns.

    A run directory may carry its own sliced reference (the smoke plan writes
    one for its 50-sentence subset), so that is preferred when present - which
    is why this takes the reference path from the run rather than hardcoding
    ``data/processed/<split>.m2``.
    """
    if run is not None:
        local = Path(run) / "predictions" / f"{split}.ref.m2"
        if local.exists():
            return local
    ref = paths.split_files(split)["m2"]
    if not ref.exists():
        raise FileNotFoundError(
            f"{paths.rel(ref)} is missing. Run: python -m gec.prepare_data --all"
        )
    return ref


def build_hyp_m2(run: Path, split: str, force: bool = False) -> Path:
    """Turn predictions into a hypothesis M2 via ``errant_parallel``.

    Checks alignment *before* invoking ERRANT, with both paths and both counts
    in any failure message. ``compare_m2`` asserts equal lengths without a
    message, so doing this here is the difference between a clear error and an
    inscrutable one twenty minutes into an evaluation.
    """
    run = Path(run)
    pred_dir = run / "predictions"
    src_f = pred_dir / f"{split}.src.txt"
    pred_f = pred_dir / f"{split}.pred.txt"
    hyp_m2 = pred_dir / f"{split}.hyp.m2"

    for f in (src_f, pred_f):
        if not f.exists():
            raise FileNotFoundError(
                f"{paths.rel(f)} is missing. Run: "
                f"python -m gec.predict --run {paths.rel(run)} --split {split}"
            )

    n_src = paths.count_lines(src_f)
    n_pred = paths.count_lines(pred_f)
    if n_src != n_pred:
        raise AssertionError(
            f"prediction/source length mismatch:\n  {src_f}: {n_src}\n  {pred_f}: {n_pred}"
        )

    # Re-running evaluate after a parser fix should not redo the slow step.
    if hyp_m2.exists() and not force and hyp_m2.stat().st_mtime >= pred_f.stat().st_mtime:
        print(f"[eval] reusing {paths.rel(hyp_m2)}")
        return hyp_m2

    errant_cli.parallel_to_m2(src_f, [pred_f], hyp_m2)
    n_hyp = m2.block_count(hyp_m2)
    if n_hyp != n_src:
        raise AssertionError(
            f"errant_parallel produced {n_hyp} blocks from {n_src} sources "
            f"({paths.rel(hyp_m2)}). The usual cause is a blank line in "
            f"{paths.rel(src_f)}, which parallel_to_m2 skips silently."
        )
    return hyp_m2


def errant_scores(
    hyp_m2: Path, ref_m2: Path, run: Path, split: str, beta: float = 0.5
) -> tuple[dict, list[dict], float | None]:
    """Score with ERRANT and cross-check the parse. Returns (overall, cats, delta)."""
    # Fail with a useful message rather than compare_m2's bare assert.
    n_hyp, n_ref = m2.block_count(hyp_m2), m2.block_count(ref_m2)
    if n_hyp != n_ref:
        raise AssertionError(
            f"hypothesis/reference block mismatch:\n"
            f"  {paths.rel(hyp_m2)}: {n_hyp} blocks\n"
            f"  {paths.rel(ref_m2)}: {n_ref} blocks\n"
            "compare_m2 asserts these are equal with no message."
        )

    out = errant_cli.compare_m2(hyp_m2, ref_m2, cat=2, beta=beta)

    # Always persist raw stdout BEFORE parsing, so a format change never costs
    # us the numbers - they can be recovered without re-running the scorer.
    log = Path(run) / "logs" / f"errant_compare.{split}.txt"
    paths.write_text(log, out)

    overall = parse_errant_overall(out)
    cats = parse_errant_categories(out)

    # Self-validating parser: for a single-annotator reference our independent
    # scorer must agree, so a silent format change becomes a loud failure.
    n_annotators = max(len(b.annotators()) for b in m2.read_m2(ref_m2))
    delta: float | None = None
    try:
        pure = score_m2_pure_python(hyp_m2, ref_m2, beta)
        delta = round(abs(pure["f05"] - overall["f05"]), 6)
        if n_annotators == 1:
            if delta > 0.005:
                raise ErrantParseError(
                    "cross-check FAILED: our independent scorer and the parsed "
                    f"errant_compare output disagree.\n"
                    f"  errant_compare: P={overall['precision']} R={overall['recall']} "
                    f"F0.5={overall['f05']} (tp={overall['tp']} fp={overall['fp']} "
                    f"fn={overall['fn']})\n"
                    f"  pure python   : P={pure['precision']} R={pure['recall']} "
                    f"F0.5={pure['f05']} (tp={pure['tp']} fp={pure['fp']} "
                    f"fn={pure['fn']})\n"
                    f"  |delta F0.5| = {delta} > 0.005\n"
                    f"Raw stdout kept at {paths.rel(log)}. Either the stdout "
                    "format changed or the M2 files are misaligned."
                )
        elif delta > 0.02:
            print(f"[eval] note: cross-check delta {delta} on a {n_annotators}-annotator "
                  "reference; ERRANT's corpus-level best-combination search can "
                  "legitimately beat our per-sentence greedy approximation.")
    except AssertionError as exc:
        print(f"[eval] cross-check skipped: {exc}")

    return overall, cats, delta


def write_per_category_csv(run: Path, split: str, cats: list[dict], overall: dict) -> Path:
    """Write the per-category CSV plus an ``ALL`` row carrying the overall numbers."""
    out = Path(run) / "predictions" / f"{split}.per_category.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["category", "tp", "fp", "fn", "p", "r", "f05", "support"]
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in sorted(cats, key=lambda r: -r["support"]):
            w.writerow({k: row[k] for k in fields})
        w.writerow(
            {
                "category": "ALL",
                "tp": overall["tp"],
                "fp": overall["fp"],
                "fn": overall["fn"],
                "p": overall["precision"],
                "r": overall["recall"],
                "f05": overall["f05"],
                "support": overall["tp"] + overall["fn"],
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Driving one split
# --------------------------------------------------------------------------- #

def evaluate_split(run: Path, split: str, force: bool = False) -> dict:
    """Compute every metric for one split of one run."""
    run = Path(run)
    t0 = time.time()
    print(f"\n[eval] === {paths.rel(run)} :: {split} ===")

    pred_dir = run / "predictions"
    srcs = paths.read_lines(pred_dir / f"{split}.src.txt")
    hyps = paths.read_lines(pred_dir / f"{split}.pred.txt")

    hyp_m2 = build_hyp_m2(run, split, force=force)
    ref_m2 = reference_m2(split, run)
    overall, cats, delta = errant_scores(hyp_m2, ref_m2, run, split)
    csv_path = write_per_category_csv(run, split, cats, overall)

    # refs / is_clean come precomputed from prepare_data, so we never re-parse
    # the gold M2 here.
    rows = paths.read_jsonl(paths.split_files(split)["jsonl"])[: len(srcs)]
    refs_per_sent = [r.get("refs") or [r["tgt"]] for r in rows]
    is_clean = [bool(r["is_clean"]) for r in rows]

    bleu = bleu_scores(hyps, refs_per_sent)
    oc = over_correction(srcs, hyps, is_clean)

    out: dict = {
        "n": len(srcs),
        "errant": overall,
        "cross_check_delta": delta,
        **bleu,
        **oc,
        "avg_edits_per_sentence": avg_edits_per_sentence(hyp_m2),
        "copy_rate": copy_rate(srcs, hyps),
        "per_category_csv": paths.rel(csv_path),
        "errant_raw_log": paths.rel(run / "logs" / f"errant_compare.{split}.txt"),
        "reference_m2": paths.rel(ref_m2),
        "eval_seconds": round(time.time() - t0, 1),
    }

    # Sanity check from the plan: W&I is roughly 1/3 already-correct sentences,
    # so a clean share near 0 or near n means the M2 parsing is wrong.
    if split in ("val", "test") and oc["n_clean"]:
        share = oc["n_clean"] / len(srcs)
        if not 0.1 < share < 0.6:
            out["warning"] = (
                f"clean share {share:.1%} is outside the expected 10-60% for {split}; "
                "check the M2 parsing"
            )
            print(f"[eval] WARNING: {out['warning']}")

    print(
        f"[eval] P={overall['precision']:.4f} R={overall['recall']:.4f} "
        f"F0.5={overall['f05']:.4f}  (tp={overall['tp']} fp={overall['fp']} fn={overall['fn']})"
    )
    print(
        f"[eval] BLEU(13a)={bleu['bleu_13a']}  BLEU(tok)={bleu['bleu_tok']}  "
        f"over-correction={oc['over_correction_rate']} "
        f"({oc['n_clean_changed']}/{oc['n_clean']} clean sentences changed)"
    )
    print(f"[eval] edits/sentence={out['avg_edits_per_sentence']}  "
          f"copy_rate={out['copy_rate']}  cross_check_delta={delta}")
    return out


def evaluate_run(
    run: Path, splits: Iterable[str] = ("val", "test"), force: bool = False
) -> dict:
    """Evaluate every requested split and update ``metrics.json`` atomically."""
    run = Path(run)
    metrics_path = run / "metrics.json"
    metrics: dict = paths.read_json(metrics_path) if metrics_path.exists() else {}

    cfg_path = run / "config.yaml"
    if cfg_path.exists():
        import yaml

        cfg = yaml.safe_load(paths.read_text(cfg_path)) or {}
        metrics.setdefault("run", cfg.get("name", run.parent.name))
        metrics.setdefault("condition", cfg.get("condition"))
    metrics.setdefault("run", run.parent.name)
    metrics.setdefault("seed", int(run.name.replace("seed", "")) if run.name.startswith("seed") else None)
    metrics.setdefault("splits", {})

    for split in splits:
        pred = run / "predictions" / f"{split}.pred.txt"
        if not pred.exists():
            print(f"[eval] skipping {split}: no predictions at {paths.rel(pred)}")
            continue
        metrics["splits"][split] = evaluate_split(run, split, force=force)

    sel = metrics["splits"].get("val", {}).get("errant", {})
    if sel:
        metrics["selection_metric"] = {
            "split": "val",
            "name": "f05",
            "value": sel.get("f05"),
        }

    metrics["_meta"] = {
        **metrics.get("_meta", {}),
        "complete": True,
        "errant_version": errant_cli.errant_version(),
        "written_at": repro.utc_now(),
        "schema": 1,
    }
    paths.write_json_atomic(metrics_path, metrics)
    print(f"\n[eval] wrote {paths.rel(metrics_path)}")
    return metrics


# --------------------------------------------------------------------------- #
# Latency (spec section 7.2)
# --------------------------------------------------------------------------- #

def latency_table(run: Path, n: int = 200, warmup: int = 10) -> Path:
    """Measure ms/sentence across device x batch size x decoding strategy.

    Spec section 7.2: one model suffices because every condition shares the
    ``t5-small`` architecture. Writes ``results/tables/latency.csv``.
    """
    import torch

    from gec import predict

    run = Path(run)
    srcs = paths.read_lines(paths.split_files("test")["src"])[: n + warmup]
    if len(srcs) < n + warmup:
        print(f"[latency] only {len(srcs)} test sentences available")

    rows: list[dict] = []
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    out = paths.TABLES / "latency.csv"
    fields = ["run", "device", "batch_size", "decode", "n", "total_s",
              "ms_per_sentence", "sentences_per_s", "peak_vram_mb"]

    for device in devices:
        model, tokenizer, cfg = predict.load_run_model(run, device)
        for batch_size in (1, 32):
            for decode, beams in (("greedy", 1), ("beam4", 4)):
                repro.reset_peak_vram()
                # Warm up first; the first CUDA kernel launch and the first
                # autograd-free graph are not representative.
                predict.generate_batched(
                    model, tokenizer, srcs[:warmup], cfg, device,
                    batch_size=batch_size, num_beams=beams,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                predict.generate_batched(
                    model, tokenizer, srcs[warmup:warmup + n], cfg, device,
                    batch_size=batch_size, num_beams=beams,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                total = time.perf_counter() - t0
                m = min(n, max(0, len(srcs) - warmup))
                rows.append(
                    {
                        "run": run.parent.name,
                        "device": device,
                        "batch_size": batch_size,
                        "decode": decode,
                        "n": m,
                        "total_s": round(total, 3),
                        "ms_per_sentence": round(1000 * total / m, 2) if m else None,
                        "sentences_per_s": round(m / total, 2) if total else None,
                        "peak_vram_mb": repro.peak_vram_mb() if device == "cuda" else None,
                    }
                )
                print(f"[latency] {device:4s} batch={batch_size:2d} {decode:6s} "
                      f"-> {rows[-1]['ms_per_sentence']} ms/sentence")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[latency] wrote {paths.rel(out)}")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gec.evaluate",
        description="ERRANT P/R/F0.5, sacreBLEU, over-correction and latency.",
    )
    ap.add_argument("--run", required=True, help="results/runs/<name>/seed<k>")
    ap.add_argument("--split", default=None, help="val | test | indian")
    ap.add_argument("--all-splits", action="store_true")
    ap.add_argument("--latency", action="store_true", help="write results/tables/latency.csv")
    ap.add_argument("--n", type=int, default=200, help="sentences for --latency")
    ap.add_argument("--force", action="store_true", help="rebuild the hypothesis M2")
    args = ap.parse_args(argv)

    run = Path(args.run)
    if not run.exists():
        ap.error(f"run directory does not exist: {run}")

    if args.latency:
        latency_table(run, n=args.n)
        return 0

    if args.all_splits:
        splits = [s for s in ("val", "test", "indian")
                  if (run / "predictions" / f"{s}.pred.txt").exists()]
    elif args.split:
        splits = [args.split]
    else:
        ap.error("pass --split, --all-splits or --latency")

    evaluate_run(run, splits=splits, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
