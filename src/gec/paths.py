"""All filesystem paths and text I/O in one place.

This module has ZERO project imports; everything else imports it.

Two conventions enforced here, both of which prevent whole classes of bug:

1. **No module ever calls bare ``open()``.** Python 3.10 on Windows defaults to
   cp1252, and the W&I essays contain curly quotes, en-dashes and accented
   names. Every read/write below pins ``encoding="utf-8"`` and ``newline="\\n"``.

2. **``write_lines`` refuses blank lines.** ERRANT's ``parallel_to_m2`` does
   ``if not orig: continue``, so a blank source line produces no M2 block at
   all. That desynchronises the hypothesis and reference M2 files and surfaces
   much later as a bare ``AssertionError`` inside ``compare_m2``.

Three environment variables redirect the big directories, which is what the
Colab notebook uses to put data/models/results on Google Drive so they survive
a disconnect (spec section 9)::

    GEC_DATA_DIR   GEC_MODELS_DIR   GEC_RESULTS_DIR
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

# src/gec/paths.py -> src/gec -> src -> repo root
ROOT = Path(__file__).resolve().parents[2]

CONFIGS = ROOT / "configs"
EXPERIMENTS = CONFIGS / "experiments"
PLANS = CONFIGS / "plans"
DATA_MANUAL = ROOT / "data_manual"
TESTS = ROOT / "tests"
APP = ROOT / "app"


def _env_dir(var: str, default: Path) -> Path:
    """Read a directory from an environment variable, falling back to *default*."""
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


DATA = _env_dir("GEC_DATA_DIR", ROOT / "data")
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

MODELS = _env_dir("GEC_MODELS_DIR", ROOT / "models")

RESULTS = _env_dir("GEC_RESULTS_DIR", ROOT / "results")
RUNS = RESULTS / "runs"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

#: Splits that have the full five-file layout described by :func:`split_files`.
SPLITS = ("train_real", "val", "test", "indian")


def ensure_dirs() -> None:
    """Create every project directory that code writes into."""
    for d in (RAW, PROCESSED, MODELS, RUNS, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)


def run_dir(run_name: str, seed: int) -> Path:
    """``results/runs/<run_name>/seed<k>`` - one experiment run at one seed."""
    return RUNS / run_name / f"seed{seed}"


def split_files(split: str) -> dict[str, Path]:
    """The five files that make up one data split.

    This is the single definition of the data-file naming contract; it is what
    keeps ``prepare_data``, ``predict`` and ``evaluate`` from drifting apart.
    All five are in **identical sentence order**:

    ``m2``
        Blocks with every annotator's ``A`` lines, verbatim. The ERRANT
        reference used for scoring.
    ``src``
        One source sentence per line, spaCy-tokenised and space-separated.
        ERRANT's ``-orig`` and the model's input. Never blank.
    ``tgt``
        Annotator-0 target per line. The sacreBLEU reference.
    ``jsonl``
        One row per sentence with precomputed ``refs`` and ``is_clean``.
    ``stats``
        Counts written by ``prepare_data``.
    """
    return {
        "m2": PROCESSED / f"{split}.m2",
        "src": PROCESSED / f"{split}.src.txt",
        "tgt": PROCESSED / f"{split}.tgt.txt",
        "jsonl": PROCESSED / f"{split}.jsonl",
        "stats": PROCESSED / f"{split}.stats.json",
    }


def hf_cache() -> Path | None:
    """The HuggingFace cache directory, honouring ``HF_HOME`` if set."""
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        raw = os.environ.get(var, "").strip()
        if raw:
            return Path(raw).expanduser()
    default = Path.home() / ".cache" / "huggingface"
    return default if default.exists() else None


# --------------------------------------------------------------------------- #
# Text I/O - always UTF-8, always LF
# --------------------------------------------------------------------------- #

def read_text(p: Path | str) -> str:
    """Read a whole file as UTF-8."""
    return Path(p).read_text(encoding="utf-8")


def read_lines(p: Path | str) -> list[str]:
    """Read a file as a list of lines with trailing newlines removed.

    Uses ``splitlines``, so meaningful leading or trailing spaces inside a line
    survive.
    """
    return read_text(p).splitlines()


def write_text(p: Path | str, s: str) -> None:
    """Write UTF-8 text with LF newlines, creating parent directories."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)


def write_lines(p: Path | str, lines: Iterable[str], allow_blank: bool = False) -> int:
    """Write one line per item, UTF-8 and LF, and return the count.

    Parameters
    ----------
    allow_blank
        Permit empty lines. **Only** for files that never reach ERRANT as an
        ``-orig`` or ``-cor`` argument - in practice just ``<split>.tgt.txt``,
        where a blank line is real data: a handful of W&I annotations delete an
        entire sentence (junk input like ``fsdjgdofg``, or a fragment merged
        into its neighbour), so the annotator-0 target is genuinely empty.
        ``tgt.txt`` must stay line-parallel with ``src.txt``, so those lines are
        kept blank rather than dropped or invented.

    Raises
    ------
    ValueError
        If a line is empty (and ``allow_blank`` is False) or contains an
        embedded newline. ERRANT's ``parallel_to_m2`` silently skips blank
        source lines (``if not orig: continue``), which desynchronises the M2
        files and later trips ``compare_m2``'s messageless length assertion.
        Failing here, at the write, is far cheaper to diagnose.
    """
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = list(lines)
    for i, line in enumerate(out):
        if not allow_blank and (not line or not line.strip()):
            raise ValueError(
                f"{p}: line {i + 1} is empty or whitespace-only. ERRANT's "
                "parallel_to_m2 skips blank lines, which would desynchronise "
                "the hypothesis and reference M2 files. If this file never "
                "reaches ERRANT (e.g. a .tgt.txt), pass allow_blank=True."
            )
        if "\n" in line or "\r" in line:
            raise ValueError(f"{p}: line {i + 1} has an embedded newline: {line!r}")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for line in out:
            fh.write(line + "\n")
    return len(out)


def count_lines(p: Path | str) -> int:
    """Number of lines in a text file."""
    return len(read_lines(p))


def read_json(p: Path | str) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(read_text(p))


def write_json(p: Path | str, obj: Any) -> None:
    """Write pretty, stable JSON (indent 2, sorted keys, non-ASCII preserved)."""
    write_text(p, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_json_atomic(p: Path | str, obj: Any) -> None:
    """Write JSON via a temp file and ``os.replace``.

    Used for ``metrics.json``, whose existence with ``_meta.complete == true``
    is the single resumability sentinel (spec section 8.2). The atomic replace
    is what stops a crash mid-write from leaving a half-written file that fools
    the resume check into skipping a run.
    """
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    write_json(tmp, obj)
    os.replace(tmp, p)


def read_jsonl(p: Path | str) -> list[dict]:
    """Read a JSON-lines file, skipping blank lines."""
    return [json.loads(ln) for ln in read_lines(p) if ln.strip()]


def write_jsonl(p: Path | str, rows: Iterable[dict]) -> int:
    """Write one compact JSON object per line; returns the row count."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256(p: Path | str, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    """Hex SHA-256 of a string, encoded UTF-8."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def rel(p: Path | str) -> str:
    """Path relative to the repo root, for logs and JSON (forward slashes)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()
