"""Tests for aggregation, selection and the TBD path (spec section 11).

Two properties matter most:

* **Missing runs must render as ``TBD`` without crashing.** The whole report
  is built from a left join of the expected grid onto what exists on disk, so
  this is exercised with *everything* missing - the hardest case.
* **Selection must be impossible to do on test data.** ``select_best`` asserts
  it, so the rule is enforced by code rather than by discipline.
"""

from __future__ import annotations

import json

import pytest

from gec import aggregate as agg
from gec import paths


def _write_run(tmp_runs, run, seed, *, val_f05=None, test_f05=None,
               complete=True, alias_of=None, cfg=None, suspect=False):
    """Create a minimal run directory on disk."""
    d = tmp_runs / run / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    metrics = {"run": run, "seed": seed, "condition": "synthetic",
               "_meta": {"complete": complete}}
    if alias_of:
        metrics["alias_of"] = alias_of
    else:
        splits = {}
        if val_f05 is not None:
            splits["val"] = {"errant": {"precision": 0.3, "recall": 0.1,
                                        "f05": val_f05, "tp": 1, "fp": 2, "fn": 3},
                             "n": 100}
        if test_f05 is not None:
            splits["test"] = {"errant": {"precision": 0.3, "recall": 0.1,
                                         "f05": test_f05, "tp": 1, "fp": 2, "fn": 3},
                              "bleu_13a": 60.0, "bleu_tok": 59.0,
                              "over_correction_rate": 0.2, "n_clean": 30,
                              "avg_edits_per_sentence": 0.5, "copy_rate": 0.4,
                              "n": 100}
        metrics["splits"] = splits
        if suspect:
            metrics["train"] = {"suspect": True, "suspect_reason": "all empty"}
    paths.write_json(d / "metrics.json", metrics)
    if cfg is not None:
        import yaml

        paths.write_text(d / "config.yaml", yaml.safe_dump(cfg))
    return d


@pytest.fixture
def tmp_runs(tmp_path, monkeypatch):
    """Point paths.RUNS/TABLES/FIGURES/RESULTS at a temp directory."""
    runs = tmp_path / "runs"
    runs.mkdir()
    for name, value in [("RUNS", runs), ("TABLES", tmp_path / "tables"),
                        ("FIGURES", tmp_path / "figures"), ("RESULTS", tmp_path)]:
        monkeypatch.setattr(paths, name, value)
        monkeypatch.setattr(agg.paths, name, value)
    (tmp_path / "tables").mkdir(exist_ok=True)
    (tmp_path / "figures").mkdir(exist_ok=True)
    agg.WARNINGS.clear()
    return runs


# --------------------------------------------------------------------------- #
# The expected grid and the TBD path
# --------------------------------------------------------------------------- #

def test_expected_grid_covers_every_plan_but_smoke():
    rows = agg.expected_grid()
    runs = {r["run"] for r in rows}
    assert "e0_copy" in runs and "e3_mixed" in runs and "e4_mixed_matched" in runs
    assert not any(r.startswith("smoke") for r in runs)
    assert "e1_p0.2_precomp" in runs, "the extra E1 arm must be in the grid"


def test_collect_runs_with_nothing_on_disk_still_returns_the_grid(tmp_runs):
    rows = agg.collect_runs(tmp_runs)
    assert len(rows) == len(agg.expected_grid())
    assert all(r["present"] is False for r in rows)


def test_collect_runs_marks_what_exists(tmp_runs):
    _write_run(tmp_runs, "e1_p0.2", 13, val_f05=0.19, test_f05=0.17)
    rows = agg.collect_runs(tmp_runs)
    hit = [r for r in rows if r["run"] == "e1_p0.2" and r["seed"] == 13][0]
    assert hit["present"] is True
    assert hit["val_f05"] == 0.19
    assert hit["test_f05"] == 0.17


def test_incomplete_runs_are_treated_as_missing(tmp_runs):
    _write_run(tmp_runs, "e1_p0.2", 13, val_f05=0.19, complete=False)
    rows = agg.collect_runs(tmp_runs)
    hit = [r for r in rows if r["run"] == "e1_p0.2"][0]
    assert hit["present"] is False
    assert any("incomplete" in w for w in agg.WARNINGS)


def test_unreadable_metrics_are_treated_as_missing(tmp_runs):
    d = tmp_runs / "e1_p0.2" / "seed13"
    d.mkdir(parents=True)
    paths.write_text(d / "metrics.json", "{not json")
    rows = agg.collect_runs(tmp_runs)
    assert [r for r in rows if r["run"] == "e1_p0.2"][0]["present"] is False
    assert any("unreadable" in w for w in agg.WARNINGS)


def test_smoke_and_probe_runs_are_skipped(tmp_runs):
    """A smoke result must never be able to reach a report table."""
    _write_run(tmp_runs, "smoke_e3_mixed", 13, val_f05=0.9, test_f05=0.9)
    _write_run(tmp_runs, "probe_e1_p0.1", 13, val_f05=0.9, test_f05=0.9)
    rows = agg.collect_runs(tmp_runs)
    assert not any(str(r["run"]).startswith(("smoke_", "probe_")) for r in rows)


def test_alias_runs_follow_the_pointer(tmp_runs):
    _write_run(tmp_runs, "e1_p0.2", 13, val_f05=0.19, test_f05=0.17)
    _write_run(tmp_runs, "e2_clean0.0", 13, alias_of="e1_p0.2/seed13")
    rows = agg.collect_runs(tmp_runs)
    alias = [r for r in rows if r["run"] == "e2_clean0.0"][0]
    assert alias["present"] is True
    assert alias["test_f05"] == 0.17, "metrics must come from the alias target"


def test_alias_with_a_missing_target_warns_and_is_missing(tmp_runs):
    _write_run(tmp_runs, "e2_clean0.0", 13, alias_of="e1_p0.9/seed13")
    rows = agg.collect_runs(tmp_runs)
    assert [r for r in rows if r["run"] == "e2_clean0.0"][0]["present"] is False
    assert any("alias target" in w for w in agg.WARNINGS)


# --------------------------------------------------------------------------- #
# Selection: val only, enforced
# --------------------------------------------------------------------------- #

def test_select_best_refuses_a_test_metric(tmp_runs):
    """Spec rule 2.2 made a property of the code."""
    rows = agg.collect_runs(tmp_runs)
    with pytest.raises(AssertionError, match="refuses to read a test metric"):
        agg.select_best("e1", rows, metric="test_f05")


def test_select_best_picks_the_argmax_on_val(tmp_runs):
    for run, p, val in [("e1_p0.1", 0.1, 0.15), ("e1_p0.2", 0.2, 0.19),
                        ("e1_p0.4", 0.4, 0.12)]:
        _write_run(tmp_runs, run, 13, val_f05=val, test_f05=0.99 if p == 0.4 else 0.1,
                   cfg={"dataset": {"synthetic": {"p": p}}})
    rows = agg.collect_runs(tmp_runs)
    best = agg.select_best("e1", rows, key_path="dataset.synthetic.p")
    assert best["best_run"] == "e1_p0.2/seed13"
    assert best["best_value"] == 0.2
    # e1_p0.4 has the best TEST score and must NOT be chosen.
    assert "e1_p0.4" not in best["best_run"]
    assert len(best["candidates"]) == 3, "every candidate must be auditable"


def test_select_best_with_nothing_present(tmp_runs):
    assert agg.select_best("e1", agg.collect_runs(tmp_runs)) == {}


def test_write_selection_records_candidates(tmp_runs):
    for run, p, val in [("e1_p0.1", 0.1, 0.15), ("e1_p0.2", 0.2, 0.19)]:
        _write_run(tmp_runs, run, 13, val_f05=val, test_f05=0.1,
                   cfg={"dataset": {"synthetic": {"p": p}}})
    out = agg.write_selection(agg.collect_runs(tmp_runs))
    sel = json.loads(paths.read_text(out))
    assert sel["e1_best_p"] == 0.2
    assert sel["selected_on"] == "val"
    assert "e1_candidates" in sel


def test_write_selection_with_no_runs_writes_a_stub(tmp_runs):
    out = agg.write_selection(agg.collect_runs(tmp_runs))
    sel = json.loads(paths.read_text(out))
    assert "e1_best_p" not in sel
    assert sel["selected_on"] == "val"


# --------------------------------------------------------------------------- #
# Figures guard their own preconditions
# --------------------------------------------------------------------------- #

def test_every_figure_returns_none_when_nothing_has_run(tmp_runs):
    rows = agg.collect_runs(tmp_runs)
    for fn in (agg.fig_e0_bleu_vs_f05, agg.fig_e1_noise_curve,
               agg.fig_e2_clean_ratio, agg.fig_e3_conditions,
               agg.fig_e3_per_category):
        assert fn(rows) is None
    assert agg.fig_e4_error_types() is None
    assert agg.WARNINGS, "each skip must be explained"


def test_noise_curve_needs_two_points(tmp_runs):
    _write_run(tmp_runs, "e1_p0.2", 13, val_f05=0.19, test_f05=0.17,
               cfg={"dataset": {"synthetic": {"p": 0.2}}})
    rows = agg.collect_runs(tmp_runs)
    assert agg.fig_e1_noise_curve(rows) is None
    assert any("need >= 2" in w for w in agg.WARNINGS)


def test_noise_curve_is_drawn_with_two_points(tmp_runs):
    for run, p in [("e1_p0.1", 0.1), ("e1_p0.2", 0.2)]:
        _write_run(tmp_runs, run, 13, val_f05=0.1 + p, test_f05=0.1,
                   cfg={"dataset": {"synthetic": {"p": p}}})
    out = agg.fig_e1_noise_curve(agg.collect_runs(tmp_runs))
    assert out is not None and out.exists() and out.stat().st_size > 1000


# --------------------------------------------------------------------------- #
# Tables and RESULTS.md
# --------------------------------------------------------------------------- #

def test_summary_csv_contains_the_full_grid_with_tbd(tmp_runs):
    _write_run(tmp_runs, "e1_p0.2", 13, val_f05=0.19, test_f05=0.17)
    out1, _ = agg.write_summary(agg.collect_runs(tmp_runs))
    text = paths.read_text(out1)
    assert "TBD" in text, "missing runs must render as TBD"
    assert "e1_p0.2" in text
    assert "e4_mixed_matched" in text, "unrun rows must still appear"


def test_results_md_is_written_with_nothing_present(tmp_runs):
    rows = agg.collect_runs(tmp_runs)
    out = agg.write_results_md(rows, {"e1_noise_curve": None})
    text = paths.read_text(out)
    assert "TBD" in text
    assert "Completeness" in text
    assert "0 of" in text or "expected runs are complete" in text


def test_results_md_flags_suspect_runs(tmp_runs):
    _write_run(tmp_runs, "e1_p0.2", 13, val_f05=0.0, test_f05=0.0, suspect=True)
    rows = agg.collect_runs(tmp_runs)
    text = paths.read_text(agg.write_results_md(rows, {}))
    assert "Suspect runs" in text
    assert "†" in text


def test_results_md_explains_the_precision_convention(tmp_runs):
    """A reader must not mistake E0's P = 1.0 for a bug."""
    text = paths.read_text(agg.write_results_md(agg.collect_runs(tmp_runs), {}))
    assert "P = 1.0" in text and "tp+fp" in text


def test_e5_table_reports_tbd_when_skipped(tmp_runs, monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    paths.write_json(processed / "indian.skip", {"skipped": True, "n_real_rows": 0})
    monkeypatch.setattr(agg.paths, "PROCESSED", processed)
    out = agg.table_e5_indian(agg.collect_runs(tmp_runs))
    text = paths.read_text(out)
    assert "TBD" in text and "real rows" in text


def test_failure_gallery_needs_two_models(tmp_runs):
    _write_run(tmp_runs, "e3_real", 13, val_f05=0.1, test_f05=0.1)
    assert agg.failure_gallery(agg.collect_runs(tmp_runs)) is None
    assert any("need >= 2" in w for w in agg.WARNINGS)
