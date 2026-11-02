"""The RESULTS.md auditor must be able to FAIL.

``scripts/audit_results.py`` is the independent check behind this project's
central promise - *every reported number came from a file our code wrote*. An
auditor that always passes is worse than none at all, because it converts an
unchecked claim into a checked-looking one. So the tests here doctor a single
cell and require a non-zero exit.

The script is exercised as a subprocess because that is how it is used, and
because it reads ``paths.RESULTS`` at import time via ``GEC_RESULTS_DIR``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from gec import paths

SCRIPT = paths.ROOT / "scripts" / "audit_results.py"

HEADER = (
    "| run | seed | P | R | F0.5 | BLEU(13a) | BLEU(tok) | over-corr. | "
    "edits/sent |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
)
HONEST = ("| e3_real | 13 | 0.5270 | 0.2970 | 0.4563 | 83.36 | 83.22 | "
          "0.1462 | 0.9592 |\n")

METRICS = {
    "run": "e3_real", "seed": 13, "_meta": {"complete": True},
    "splits": {"test": {
        "errant": {"precision": 0.5270, "recall": 0.2970, "f05": 0.4563},
        "bleu_13a": 83.36, "bleu_tok": 83.22,
        "over_correction_rate": 0.1462, "avg_edits_per_sentence": 0.9592,
    }},
}


def _audit(tmp_path, results_md: str, metrics: dict | None = METRICS):
    if metrics is not None:
        d = tmp_path / "runs" / "e3_real" / "seed13"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (tmp_path / "RESULTS.md").write_text(results_md, encoding="utf-8")
    env = dict(os.environ, GEC_RESULTS_DIR=str(tmp_path),
               PYTHONNOUSERSITE="1", PYTHONUTF8="1")
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=paths.ROOT,
                          env=env, capture_output=True, text=True)


def test_audit_passes_on_numbers_that_match(tmp_path):
    r = _audit(tmp_path, HEADER + HONEST)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "checked 7 reported number" in r.stdout


@pytest.mark.parametrize("column,bad", [
    (4, "0.9999"),   # F0.5
    (2, "0.9000"),   # precision
    (5, "99.99"),    # BLEU(13a)
    (7, "0.0001"),   # over-correction
])
def test_audit_catches_a_single_doctored_cell(tmp_path, column, bad):
    cells = HONEST.strip().strip("|").split("|")
    cells[column] = f" {bad} "
    doctored = "|" + "|".join(cells) + "|\n"
    r = _audit(tmp_path, HEADER + doctored)
    assert r.returncode == 1, f"auditor missed a doctored cell: {doctored}"
    assert "PROBLEM" in r.stdout


def test_audit_catches_a_run_with_no_metrics_on_disk(tmp_path):
    r = _audit(tmp_path, HEADER + HONEST, metrics=None)
    assert r.returncode == 1
    assert "no metrics.json on disk" in r.stdout


def test_audit_catches_a_dead_link(tmp_path):
    md = HEADER + HONEST + "\n[summary.csv](tables/summary.csv)\n"
    r = _audit(tmp_path, md)
    assert r.returncode == 1
    assert "does not exist" in r.stdout


def test_audit_accepts_tbd_cells(tmp_path):
    """An unfinished run renders as TBD; that is honest, not a mismatch."""
    row = ("| e3_real | 13 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |\n")
    r = _audit(tmp_path, HEADER + row)
    assert r.returncode == 0, r.stdout
