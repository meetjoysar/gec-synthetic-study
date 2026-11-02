"""Import manually collected Indian English sentences and merge annotations.

::

    python -m gec.kappa --import-manual data_manual/manual_sentences.txt
    python -m gec.kappa --make-annotation-files
    python -m gec.kappa --merge-annotations [--force]

**This module never writes or alters a sentence's wording, and never writes an
annotation.** Spec rule 2.3 reserves both for the group. The only text change
applied is curly quotes/apostrophes to straight ones plus outer whitespace
trimming, because T5's SentencePiece vocabulary has no curly quotes and the
rest of the pipeline is ASCII-folded anyway. Typos, spacing before punctuation,
case, exclamation marks, slashes and brackets are all preserved exactly.

The input format is a flat text file where ``# post: <label>`` lines mark a
writer and every other non-empty line is one sentence belonging to the most
recent label. Labels containing ``[sensitive]`` mark posts whose content should
not be quoted in the report.
"""

from __future__ import annotations

import csv
import re
import statistics
from collections import Counter, OrderedDict
from pathlib import Path

from gec import paths

LABEL_PREFIX = "# post:"
SENTINELS = {"BEGIN-SENTENCES", "END-SENTENCES"}
SENSITIVE_MARK = "[sensitive]"
SENSITIVE_NOTE = "sensitive: do not quote in report"
SOURCE_TYPE = "reddit-manual"
ID_TEMPLATE = "IN-{:04d}"

#: Curly -> straight only. Everything else is left exactly as collected.
_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"', "´": "'",
}

ANNOTATION_DIR = "annotation"
ANNOTATION_COLUMNS = ["id", "source", "correction"]


def normalise_quotes(s: str) -> str:
    """Straighten curly quotes and trim the outer whitespace. Nothing else."""
    for a, b in _QUOTE_MAP.items():
        s = s.replace(a, b)
    return s.strip()


def parse_manual(path: Path | str) -> "OrderedDict[str, list[str]]":
    """Parse the collected file into ``{label: [sentence, ...]}`` in file order."""
    out: OrderedDict[str, list[str]] = OrderedDict()
    current: str | None = None
    for lineno, line in enumerate(paths.read_lines(path), 1):
        s = line.strip()
        if not s or s in SENTINELS:
            continue
        if s.startswith(LABEL_PREFIX):
            current = s[len(LABEL_PREFIX):].strip()
            out.setdefault(current, [])
            continue
        if current is None:
            raise ValueError(
                f"{path}: line {lineno} is a sentence before any "
                f"'{LABEL_PREFIX} <label>' line: {s[:60]!r}"
            )
        out[current].append(normalise_quotes(s))
    if not out:
        raise ValueError(f"{path}: no '{LABEL_PREFIX}' labels found")
    return out


def writer_ids(labels: "OrderedDict[str, list[str]]") -> dict[str, str]:
    """``{label: 'W01'}`` in first-appearance order."""
    return {lab: f"W{i:02d}" for i, lab in enumerate(labels, 1)}


# --------------------------------------------------------------------------- #
# Reading and writing the TSV without altering any cell
# --------------------------------------------------------------------------- #

def read_tsv_rows(path: Path | str) -> tuple[list[str], list[dict[str, str]]]:
    """Read the TSV as ``(header, rows)`` with every cell kept verbatim."""
    path = Path(path)
    if not path.exists():
        return [], []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = list(reader.fieldnames or [])
        rows = [{k: (v if v is not None else "") for k, v in r.items()}
                for r in reader]
    return header, rows


def write_tsv_rows(path: Path | str, header: list[str], rows: list[dict[str, str]]) -> None:
    """Write the TSV with standard CSV quoting, so any cell round-trips.

    ``QUOTE_MINIMAL`` matters: several sentences contain ``"`` characters (for
    example ``She had "saved" or "marked" 3 places``), and an unquoted writer
    would produce a file that pandas then mis-parses. The quoting is invisible
    once read back, and a test asserts the round-trip is exact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter="\t",
                           quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #

def import_manual(
    manual: Path | str,
    tsv: Path | str | None = None,
    dry_run: bool = False,
) -> dict:
    """Append the collected sentences to the Indian English TSV.

    Existing rows - including the ``EXAMPLE-`` demonstration rows - are kept
    untouched. Annotation columns are left **empty** for the group to fill in
    independently. Exact duplicates (case-insensitive, against both the file
    and rows already present) are skipped and reported.
    """
    from gec.kappa import COLUMNS, EXAMPLE_PREFIX

    manual = Path(manual)
    tsv = Path(tsv) if tsv else paths.DATA_MANUAL / "indian_english_test.tsv"

    labels = parse_manual(manual)
    wids = writer_ids(labels)

    header, existing = read_tsv_rows(tsv)
    # Widen the header to carry source_type / source_note, padding old rows.
    header = list(COLUMNS) if not header else header + [
        c for c in COLUMNS if c not in header
    ]
    for r in existing:
        for c in header:
            r.setdefault(c, "")

    seen = {str(r.get("source", "")).strip().lower() for r in existing}
    used_ids = {str(r.get("id", "")).strip() for r in existing}

    # Continue the IN- numbering rather than restarting it.
    next_n = 1
    for rid in used_ids:
        m = re.fullmatch(r"IN-(\d+)", rid)
        if m:
            next_n = max(next_n, int(m.group(1)) + 1)

    new_rows: list[dict[str, str]] = []
    duplicates: list[tuple[str, str]] = []
    per_writer: Counter[str] = Counter()

    for label, sentences in labels.items():
        wid = wids[label]
        sensitive = SENSITIVE_MARK in label
        for sent in sentences:
            key = sent.strip().lower()
            if key in seen:
                duplicates.append((wid, sent))
                continue
            seen.add(key)
            rid = ID_TEMPLATE.format(next_n)
            next_n += 1
            new_rows.append({
                "id": rid,
                "source": sent,
                "annotator1": "",
                "annotator2": "",
                "category": "",
                "notes": SENSITIVE_NOTE if sensitive else "",
                "source_type": SOURCE_TYPE,
                "source_note": f"{wid} {label}",
            })
            per_writer[f"{wid} {label}"] += 1

    if not dry_run:
        write_tsv_rows(tsv, header, existing + new_rows)

    # ---- summary (spec step 8) ---------------------------------------------
    from gec import tok

    lens = [len(tok.tokenise(r["source"]).split()) for r in new_rows] or [0]
    stats = {
        "manual_file": paths.rel(manual),
        "tsv": paths.rel(tsv),
        "n_writers": len(labels),
        "n_imported": len(new_rows),
        "n_duplicates_skipped": len(duplicates),
        "n_existing_rows_kept": len(existing),
        "n_sensitive": sum(1 for r in new_rows if r["notes"] == SENSITIVE_NOTE),
        "per_writer": dict(per_writer),
        "token_len_min": min(lens),
        "token_len_median": statistics.median(lens),
        "token_len_mean": round(statistics.mean(lens), 1),
        "token_len_max": max(lens),
        "id_range": (f"{new_rows[0]['id']}..{new_rows[-1]['id']}" if new_rows else None),
    }

    print(f"\n[import] {stats['n_imported']} sentence(s) from "
          f"{stats['n_writers']} writer(s) in {paths.rel(manual)}")
    print(f"[import] ids {stats['id_range']}; kept "
          f"{stats['n_existing_rows_kept']} existing row(s)")
    print(f"[import] {stats['n_sensitive']} row(s) marked "
          f"'{SENSITIVE_NOTE}'")
    print(f"[import] token length: min {stats['token_len_min']}, median "
          f"{stats['token_len_median']}, mean {stats['token_len_mean']}, "
          f"max {stats['token_len_max']}")
    print(f"\n[import] sentences per writer:")
    for k, v in per_writer.items():
        print(f"           {k:<34} {v:>3}")
    if duplicates:
        print(f"\n[import] skipped {len(duplicates)} exact duplicate(s):")
        for wid, s in duplicates:
            print(f"           ({wid}) {s[:70]}")
    else:
        print("\n[import] no duplicates found")
    if dry_run:
        print("\n[import] --dry-run: nothing written")
    else:
        print(f"\n[import] wrote {paths.rel(tsv)}")
    paths.write_json(paths.DATA_MANUAL / "manual_import.meta.json", stats)
    return stats


# --------------------------------------------------------------------------- #
# Independent annotation files
# --------------------------------------------------------------------------- #

def annotation_path(which: int) -> Path:
    return paths.DATA_MANUAL / ANNOTATION_DIR / f"annotator{which}.tsv"


def make_annotation_files(tsv: Path | str | None = None, force: bool = False) -> list[Path]:
    """Write one simple edit-me file per annotator.

    ``correction`` is pre-filled with the source so the annotator edits in
    place - and so leaving a sentence **unchanged** (because they judge it
    acceptable) is the natural default rather than extra work. That matters:
    those rows become the control sentences the over-correction metric needs.
    """
    from gec.kappa import EXAMPLE_PREFIX

    tsv = Path(tsv) if tsv else paths.DATA_MANUAL / "indian_english_test.tsv"
    _, rows = read_tsv_rows(tsv)
    todo = [r for r in rows if not str(r["id"]).startswith(EXAMPLE_PREFIX)]

    out: list[Path] = []
    for which in (1, 2):
        dest = annotation_path(which)
        if dest.exists() and not force:
            print(f"[annotate] {paths.rel(dest)} already exists; "
                  "pass --force to regenerate (this would DISCARD edits)")
            out.append(dest)
            continue
        existing_col = f"annotator{which}"
        write_tsv_rows(
            dest,
            ANNOTATION_COLUMNS,
            [
                {
                    "id": r["id"],
                    "source": r["source"],
                    # If this annotator already has a correction, carry it over
                    # so regenerating never loses work.
                    "correction": r.get(existing_col) or r["source"],
                }
                for r in todo
            ],
        )
        print(f"[annotate] wrote {paths.rel(dest)} ({len(todo)} rows)")
        out.append(dest)
    print("[annotate] Annotator 1 and 2 must NOT look at each other's file "
          "until both are finished.")
    return out


def merge_annotations(
    tsv: Path | str | None = None,
    force: bool = False,
) -> dict:
    """Copy each annotator's ``correction`` column into the main TSV.

    Matching is by ``id``. A non-empty cell is **never** overwritten without
    ``--force``, so re-running cannot silently destroy annotation work.
    """
    from gec.kappa import EXAMPLE_PREFIX

    tsv = Path(tsv) if tsv else paths.DATA_MANUAL / "indian_english_test.tsv"
    header, rows = read_tsv_rows(tsv)
    by_id = {str(r["id"]): r for r in rows}

    summary = {"filled": 0, "unchanged_left_as_is": 0, "refused_overwrite": 0,
               "unknown_ids": [], "missing_files": []}

    for which in (1, 2):
        src = annotation_path(which)
        col = f"annotator{which}"
        if not src.exists():
            summary["missing_files"].append(paths.rel(src))
            print(f"[merge] {paths.rel(src)} not found; skipping annotator {which}")
            continue
        _, ann_rows = read_tsv_rows(src)
        for a in ann_rows:
            rid = str(a.get("id", "")).strip()
            correction = normalise_quotes(str(a.get("correction", "")))
            if rid not in by_id:
                summary["unknown_ids"].append(rid)
                continue
            if not correction:
                continue
            current = str(by_id[rid].get(col, "")).strip()
            if current and current != correction and not force:
                summary["refused_overwrite"] += 1
                continue
            if current == correction:
                summary["unchanged_left_as_is"] += 1
                continue
            by_id[rid][col] = correction
            summary["filled"] += 1
        print(f"[merge] annotator {which}: read {len(ann_rows)} row(s) from "
              f"{paths.rel(src)}")

    if summary["filled"] or force:
        write_tsv_rows(tsv, header, rows)
        print(f"[merge] wrote {paths.rel(tsv)}")
    print(f"[merge] filled {summary['filled']} cell(s); "
          f"{summary['refused_overwrite']} refused (would overwrite - use --force); "
          f"{summary['unchanged_left_as_is']} already identical")
    if summary["unknown_ids"]:
        print(f"[merge] WARNING: {len(summary['unknown_ids'])} unknown id(s), "
              f"e.g. {summary['unknown_ids'][:5]}")
    return summary
