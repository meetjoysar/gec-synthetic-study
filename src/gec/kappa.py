"""Indian English TSV validation and annotator agreement (spec section 5.5).

::

    python -m gec.kappa --tsv data_manual/indian_english_test.tsv
    python -m gec.kappa --strict          # non-zero exit on any validation problem

**Claude Code does not write the test sentences.** Spec rule 2.3 reserves that
for the group, so this module ships only the tooling; the TSV template carries
at most five rows whose ids start with ``EXAMPLE-``, and those are excluded from
every computation here and from evaluation.

Cohen's kappa is computed at the token level, as spec section 5.5 specifies: for
each annotator, label a source token 1 if any of their edits touches it, else 0,
then compare the two label vectors over all tokens. An insertion marks the token
*after* the insertion point.

The degenerate case is handled explicitly rather than left to produce ``nan``:
if one annotator made no edits at all, their label vector is constant,
``cohen_kappa_score`` is undefined, and a bare ``nan`` in a report would be
worse than an explanation. With only ~150 sentences this is a live possibility,
so :func:`token_kappa` always reports raw agreement alongside kappa.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

from gec import paths, tok

#: The six columns spec section 5.5 defines.
REQUIRED_COLUMNS = ["id", "source", "annotator1", "annotator2", "category", "notes"]

#: Provenance columns added for the manually collected sentences. Optional, so
#: a hand-made TSV with only the six required columns still validates.
OPTIONAL_COLUMNS = ["source_type", "source_note"]

COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

#: Spec section 5.5's suggested categories. Unknown values warn rather than fail,
#: because the group may legitimately discover a new error pattern.
VALID_CATEGORIES = {
    "tense-perfect", "stative-progressive", "redundancy", "article",
    "preposition", "question-form", "uncountable-plural", "dialect-usage",
    "agreement", "word-order", "spelling", "other",
}

#: Below this many real (non-EXAMPLE) rows, E5 is skipped gracefully.
MIN_REAL_ROWS = 20

EXAMPLE_PREFIX = "EXAMPLE-"


@functools.lru_cache(maxsize=1)
def annotator():
    """A cached ERRANT annotator (loading it costs several seconds)."""
    import errant

    return errant.load("en")


def is_annotated(df):
    """Boolean mask: rows where BOTH annotators have filled their column.

    The group's workflow is "one person writes every ``source``, then the two
    annotators fill their own column independently" (see
    ``data_manual/README.md``), so the file legitimately spends time with
    sources present and annotations missing. Those rows are **pending**, not
    broken: they must not be validation errors, must not count toward the
    :data:`MIN_REAL_ROWS` threshold, and must not reach
    ``errant_parallel`` - an empty annotation line would make it drop the
    block and desynchronise the M2 files.
    """
    return (
        (df["annotator1"].astype(str).str.strip() != "")
        & (df["annotator2"].astype(str).str.strip() != "")
    )


def load_indian_tsv(path: Path | str):
    """Read the TSV, returning ``(all_rows, ready_rows)`` as DataFrames.

    ``ready_rows`` excludes both the ``EXAMPLE-`` demonstration rows and any
    row still awaiting annotation, so every consumer sees only rows that are
    actually usable.

    ``keep_default_na=False`` matters: without it a sentence consisting of the
    literal word ``NA``, ``None`` or ``null`` becomes ``NaN`` and silently
    disappears.
    """
    import pandas as pd

    df = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8"
    )
    df.columns = [c.strip() for c in df.columns]
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    real = df[~df["id"].astype(str).str.startswith(EXAMPLE_PREFIX)].reset_index(drop=True)
    ready = real[is_annotated(real)].reset_index(drop=True)
    return df, ready


def pending_count(path: Path | str) -> tuple[int, int]:
    """``(n_ready, n_pending)`` among the real (non-EXAMPLE) rows."""
    all_rows, ready = load_indian_tsv(path)
    real = all_rows[~all_rows["id"].astype(str).str.startswith(EXAMPLE_PREFIX)]
    return len(ready), len(real) - len(ready)


def validate(df) -> list[str]:
    """Return a list of human-readable problems (empty means the file is fine)."""
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(
            f"missing required column(s): {missing}; expected {REQUIRED_COLUMNS} "
            f"(plus the optional {OPTIONAL_COLUMNS})"
        )
        return problems
    extra = [c for c in df.columns if c not in COLUMNS]
    if extra:
        problems.append(f"unexpected column(s): {extra}")

    ids = df["id"].astype(str).str.strip()
    dupes = ids[ids.duplicated()].unique().tolist()
    if dupes:
        problems.append(f"duplicate id(s): {dupes}")
    if (ids == "").any():
        problems.append(f"{int((ids == '').sum())} row(s) have an empty id")

    # An empty source is always an error: there is nothing to annotate.
    blank_src = df[df["source"].astype(str).str.strip() == ""]
    if len(blank_src):
        problems.append(
            f"{len(blank_src)} row(s) have an empty 'source' "
            f"(ids: {blank_src['id'].tolist()[:8]})"
        )

    # Missing ANNOTATIONS are a normal intermediate state, not an error: the
    # workflow is to write every source first, then have the two annotators
    # fill their own column independently. Report them as pending so the group
    # can see progress without the validator crying wolf.
    real = df[~df["id"].astype(str).str.startswith(EXAMPLE_PREFIX)]
    pending = real[~is_annotated(real)]
    if len(pending):
        problems.append(
            f"PENDING (not an error): {len(pending)} real row(s) await annotation "
            f"(ids: {pending['id'].tolist()[:8]}). They do not count toward the "
            f"{MIN_REAL_ROWS}-row threshold until BOTH annotator columns are filled."
        )

    # An EXAMPLE- row ships complete, so a gap there IS a problem.
    ex = df[df["id"].astype(str).str.startswith(EXAMPLE_PREFIX)]
    for col in ("annotator1", "annotator2"):
        bad = ex[ex[col].astype(str).str.strip() == ""]
        if len(bad):
            problems.append(f"{len(bad)} EXAMPLE- row(s) have an empty '{col}'")

    # The classic hand-edited-TSV bug: a literal tab or newline inside a field
    # silently shifts every later column.
    for col in COLUMNS:
        if col not in df.columns:
            continue
        bad = df[df[col].astype(str).str.contains("\t|\n|\r", regex=True, na=False)]
        if len(bad):
            problems.append(
                f"{len(bad)} row(s) contain a tab or newline inside '{col}' "
                f"(ids: {bad['id'].tolist()[:5]})"
            )

    unknown = sorted(
        {c.strip() for c in df["category"].astype(str) if c.strip()} - VALID_CATEGORIES
    )
    if unknown:
        problems.append(
            f"WARNING (not fatal): unrecognised category value(s) {unknown}; "
            f"known: {sorted(VALID_CATEGORIES)}"
        )
    return problems


def edit_token_labels(src_tok: str, cor_tok: str) -> list[int]:
    """Per source token, 1 if any edit touches it (spec section 5.5).

    Both strings must already be spaCy-tokenised and space-separated, so
    ``parse(..., tokenise=False)`` keeps our token boundaries.

    An insertion has ``o_start == o_end``, so it marks the token *after* the
    insertion point, clamped to the last index when the insertion is at the very
    end of the sentence.
    """
    ann = annotator()
    orig = ann.parse(src_tok, tokenise=False)
    cor = ann.parse(cor_tok, tokenise=False)
    n = len(orig)
    labels = [0] * n
    if n == 0:
        return labels
    for edit in ann.annotate(orig, cor):
        if edit.type in ("noop", "UNK"):
            continue
        if edit.o_start == edit.o_end:               # insertion
            labels[min(edit.o_start, n - 1)] = 1
        else:
            for i in range(max(0, edit.o_start), min(edit.o_end, n)):
                labels[i] = 1
    return labels


def _kappa(a: list[int], b: list[int], what: str) -> dict:
    """Cohen's kappa with the no-variance case handled explicitly."""
    from sklearn.metrics import cohen_kappa_score

    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    out = {
        f"{what}_n": n,
        f"{what}_raw_agreement": round(agree / n, 4) if n else None,
        f"{what}_both_1": sum(1 for x, y in zip(a, b) if x == y == 1),
        f"{what}_both_0": sum(1 for x, y in zip(a, b) if x == y == 0),
        f"{what}_disagree": n - agree,
    }
    if n == 0 or len(set(a)) < 2 or len(set(b)) < 2:
        out[f"{what}_kappa"] = None
        out[f"{what}_note"] = (
            "kappa is undefined here: one annotator's labels have no variance "
            "(they edited everything, or nothing). Raw agreement is still "
            "meaningful and is reported above."
        )
        return out
    out[f"{what}_kappa"] = round(float(cohen_kappa_score(a, b)), 4)
    return out


def report(tsv: Path | str, out: Path | str | None = None, strict: bool = False) -> dict:
    """Validate, compute both kappas, and write ``results/tables/indian_kappa.json``."""
    tsv = Path(tsv)
    out = Path(out) if out else paths.TABLES / "indian_kappa.json"

    all_rows, real = load_indian_tsv(tsv)
    problems = validate(all_rows)
    for p in problems:
        print(f"[kappa] {p}")

    n_real = len(real)            # `real` is now READY rows (both annotations)
    n_ready, n_pending = pending_count(tsv)
    n_example = len(all_rows) - (n_ready + n_pending)
    print(f"[kappa] {paths.rel(tsv)}: {n_ready} ready, {n_pending} pending "
          f"annotation, {n_example} EXAMPLE- row(s)")

    if n_real < MIN_REAL_ROWS:
        payload = {
            "skipped": True,
            "n_real_rows": n_ready,
            "n_pending_annotation": n_pending,
            "n_example_rows": n_example,
            "reason": (
                f"only {n_ready} fully annotated rows "
                f"({n_pending} more await annotation); at least {MIN_REAL_ROWS} are needed for a "
                "meaningful kappa. The group writes these sentences (spec rule 2.3); "
                "see data_manual/README.md. E5 renders as TBD until then."
            ),
            "validation_problems": problems,
        }
        paths.write_json(out, payload)
        print(f"[kappa] WARNING: {payload['reason']}")
        print(f"[kappa] wrote {paths.rel(out)}")
        return payload

    srcs = [tok.tokenise(tok.normalise(s)) for s in real["source"]]
    a1 = [tok.tokenise(tok.normalise(s)) for s in real["annotator1"]]
    a2 = [tok.tokenise(tok.normalise(s)) for s in real["annotator2"]]

    tok_a: list[int] = []
    tok_b: list[int] = []
    sent_a: list[int] = []
    sent_b: list[int] = []
    for src, c1, c2 in zip(srcs, a1, a2):
        la = edit_token_labels(src, c1)
        lb = edit_token_labels(src, c2)
        # Both label vectors index the same source, so they are always equal length.
        tok_a.extend(la)
        tok_b.extend(lb)
        sent_a.append(int(c1.split() != src.split()))
        sent_b.append(int(c2.split() != src.split()))

    payload: dict = {
        "skipped": False,
        "n_sentences": len(srcs),
        "n_real_rows": n_real,
        "n_example_rows": n_example,
        "validation_problems": problems,
        "n_changed_by_annotator1": sum(sent_a),
        "n_changed_by_annotator2": sum(sent_b),
        "categories": dict(real["category"].value_counts()),
        "source_types": (
            dict(real["source_type"].replace("", "unspecified").value_counts())
            if "source_type" in real.columns else {}
        ),
    }
    payload.update(_kappa(tok_a, tok_b, "token"))
    payload.update(_kappa(sent_a, sent_b, "sentence"))

    paths.write_json(out, payload)
    print(f"[kappa] token-level kappa   = {payload.get('token_kappa')} "
          f"(raw agreement {payload.get('token_raw_agreement')}, "
          f"{payload.get('token_n')} tokens)")
    print(f"[kappa] sentence-level kappa = {payload.get('sentence_kappa')} "
          f"(raw agreement {payload.get('sentence_raw_agreement')})")
    print(f"[kappa] wrote {paths.rel(out)}")

    if strict and any(
        not p.startswith(("WARNING", "PENDING")) for p in problems
    ):
        raise SystemExit(1)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gec.kappa",
        description="Validate the Indian English TSV and compute annotator agreement.",
    )
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if the TSV has any (non-warning) problem")
    ap.add_argument("--import-manual", metavar="FILE", default=None,
                    help="append manually collected sentences (see indian_import.py)")
    ap.add_argument("--make-annotation-files", action="store_true",
                    help="write data_manual/annotation/annotator{1,2}.tsv")
    ap.add_argument("--merge-annotations", action="store_true",
                    help="copy the annotators' corrections back into the TSV")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting non-empty annotation cells")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    tsv = Path(args.tsv) if args.tsv else paths.DATA_MANUAL / "indian_english_test.tsv"

    from gec import indian_import

    if args.import_manual:
        indian_import.import_manual(args.import_manual, tsv, dry_run=args.dry_run)
    if args.make_annotation_files:
        indian_import.make_annotation_files(tsv, force=args.force)
    if args.merge_annotations:
        indian_import.merge_annotations(tsv, force=args.force)

    if not tsv.exists():
        print(f"[kappa] {paths.rel(tsv)} not found")
        return 1
    report(tsv, args.out, strict=args.strict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
