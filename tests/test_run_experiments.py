"""Tests for the plan runner: job expansion, resumability and the budget gate.

No training happens here. The point is that the *scheduling* logic is correct,
because a mistake in it silently wastes hours of GPU time.
"""

from __future__ import annotations

import pytest

from gec import config as cfgmod
from gec import paths
from gec import run_experiments as rx


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #

def test_every_plan_names_configs_that_exist():
    for plan, names in rx.PLANS.items():
        for name in names:
            assert (paths.EXPERIMENTS / f"{name}.yaml").exists(), f"{plan}: {name}"


def test_e1_covers_the_specs_noise_points_plus_the_extra_arm():
    assert rx.PLANS["e1"] == [
        "e1_p0.1", "e1_p0.2", "e1_p0.4", "e1_p0.6", "e1_p0.2_precomp",
    ]


def test_e3_covers_the_four_conditions():
    assert set(rx.PLANS["e3"]) == {
        "e3_synthetic", "e3_real", "e3_mixed", "e3_synth2real",
    }


def test_e5_is_evaluation_only():
    """e5 has no configs; it re-scores existing checkpoints."""
    assert rx.PLANS["e5"] == []


def test_unknown_plan_raises():
    with pytest.raises(KeyError, match="unknown plan"):
        rx.iter_jobs("nonsense", [13])


# --------------------------------------------------------------------------- #
# Job expansion
# --------------------------------------------------------------------------- #

def test_iter_jobs_expands_configs_times_seeds():
    jobs = rx.iter_jobs("e1", [13, 42], resolve=False)
    assert len(jobs) == len(rx.PLANS["e1"]) * 2
    assert {j.seed for j in jobs} == {13, 42}


def test_copy_condition_needs_no_training():
    job = rx.iter_jobs("e0", [13])[0]
    assert job.needs_train is False
    assert job.cfg["condition"] == "copy"


def test_synthetic_conditions_need_training():
    for job in rx.iter_jobs("e1", [13], resolve=False):
        assert job.needs_train is True


def test_only_filters_the_plan():
    jobs = rx.iter_jobs("e3", [13], only=["e3_real"], resolve=False)
    assert len(jobs) == 1 and jobs[0].config == "e3_real"


def test_only_that_matches_nothing_raises():
    with pytest.raises(ValueError, match="matched nothing"):
        rx.iter_jobs("e3", [13], only=["not_a_run"], resolve=False)


def test_overrides_reach_the_job_config():
    """`--set` must not be silently ignored, as it once was."""
    jobs = rx.iter_jobs("e1", [13], resolve=False, overrides=["epochs=7"])
    assert all(j.cfg["epochs"] == 7 for j in jobs)


def test_smoke_plan_applies_the_smoke_layer():
    jobs = rx.iter_jobs("smoke", [13])
    assert all(j.run.startswith("smoke_") for j in jobs)
    assert all(j.cfg["device"] == "cpu" for j in jobs)
    assert all(j.cfg["max_steps"] == 20 for j in jobs)


def test_e3_configs_declare_the_indian_split():
    """Spec 7 E5 evaluates every E3/E4 model on the Indian English set."""
    for job in rx.iter_jobs("e3", [13], resolve=False):
        assert "indian" in job.splits, job.config
    for job in rx.iter_jobs("e4", [13], resolve=False):
        assert "indian" in job.splits, job.config


def test_e1_and_e2_do_not_waste_time_on_the_indian_split():
    for plan in ("e1", "e2"):
        for job in rx.iter_jobs(plan, [13], resolve=False):
            assert "indian" not in job.splits, job.config


def test_stage2_config_declares_its_warm_start():
    job = [j for j in rx.iter_jobs("e3", [13], resolve=False)
           if j.config == "e3_synth2real"][0]
    assert job.cfg["stage2"]["enabled"] is True
    assert job.cfg["stage2"]["init_from"] == "e3_synthetic/seed13"


def test_alias_configs_are_recognised():
    """E2's clean_ratio=0.0 and E3's synthetic arm reuse earlier runs."""
    jobs = {j.config: j for j in rx.iter_jobs("e2", [13], resolve=False)}
    assert jobs["e2_clean0.0"].alias_of is not None
    assert jobs["e2_clean0.2"].alias_of is None


# --------------------------------------------------------------------------- #
# Resumability
# --------------------------------------------------------------------------- #

@pytest.fixture
def tmp_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(paths, "RUNS", runs)
    monkeypatch.setattr(rx.paths, "RUNS", runs)
    return runs


def test_status_todo_when_nothing_exists(tmp_runs):
    job = rx.iter_jobs("e1", [13], resolve=False)[0]
    assert rx.job_status(job) == "todo"


def test_status_done_only_when_metrics_are_complete(tmp_runs):
    job = rx.iter_jobs("e1", [13], resolve=False)[0]
    job.dir.mkdir(parents=True)
    paths.write_json(job.dir / "metrics.json", {"_meta": {"complete": False}})
    assert rx.job_status(job) == "todo"
    paths.write_json(job.dir / "metrics.json", {"_meta": {"complete": True}})
    assert rx.job_status(job) == "done"


def test_status_partial_with_a_model_but_no_metrics(tmp_runs):
    """A crash during evaluation must not look like a fresh job."""
    job = rx.iter_jobs("e1", [13], resolve=False)[0]
    (job.dir / "model").mkdir(parents=True)
    paths.write_json(job.dir / "model" / "config.json", {})
    assert rx.job_status(job) == "partial"


def test_status_partial_with_predictions_only(tmp_runs):
    job = rx.iter_jobs("e1", [13], resolve=False)[0]
    (job.dir / "predictions").mkdir(parents=True)
    paths.write_lines(job.dir / "predictions" / "val.pred.txt", ["x"])
    assert rx.job_status(job) == "partial"


def test_unreadable_metrics_do_not_count_as_done(tmp_runs):
    job = rx.iter_jobs("e1", [13], resolve=False)[0]
    job.dir.mkdir(parents=True)
    paths.write_text(job.dir / "metrics.json", "{broken")
    assert rx.job_status(job) != "done"


def test_write_alias_refuses_a_missing_target(tmp_runs):
    job = rx.iter_jobs("e2", [13], resolve=False)[0]
    job.alias_of = "e1_p0.9/seed13"
    with pytest.raises(FileNotFoundError, match="alias of"):
        rx.write_alias(job)


def test_write_alias_records_the_pointer(tmp_runs):
    target = tmp_runs / "e1_p0.2" / "seed13"
    target.mkdir(parents=True)
    paths.write_json(target / "metrics.json", {"_meta": {"complete": True}})

    job = [j for j in rx.iter_jobs("e2", [13], resolve=False)
           if j.config == "e2_clean0.0"][0]
    job.alias_of = "e1_p0.2/seed13"
    rx.write_alias(job)

    m = paths.read_json(job.dir / "metrics.json")
    assert m["alias_of"] == "e1_p0.2/seed13"
    assert m["_meta"]["complete"] is True
    assert not (job.dir / "model").exists(), "an alias must not copy weights"


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #

def test_alias_jobs_are_free_to_estimate():
    job = [j for j in rx.iter_jobs("e2", [13], resolve=False)
           if j.config == "e2_clean0.0"][0]
    minutes, source = rx.estimate_minutes(job)
    assert minutes == 0.0 and "alias" in source


def test_copy_job_is_cheap_to_estimate():
    minutes, _ = rx.estimate_minutes(rx.iter_jobs("e0", [13])[0])
    assert 0 < minutes < 10


def test_blocked_job_estimates_as_blocked(monkeypatch):
    """A missing e4_matched_weights.yaml must not crash the dry run."""
    job = rx.iter_jobs("e4", [13], resolve=False)[0]
    job.cfg["dataset"]["synthetic"]["weights_file"] = "configs/experiments/absent.yaml"
    job.cfg["dataset"]["synthetic"]["weights"] = None
    minutes, source = rx.estimate_minutes(job)
    assert minutes == 0.0 and source == "blocked"


def test_mixed_is_estimated_higher_than_synthetic_only():
    """Mixed trains on real + synthetic, so it must cost more."""
    jobs = {j.config: j for j in rx.iter_jobs("e3", [13], resolve=False)}
    # resolve=False leaves ${select.*} strings, so give them concrete values.
    for j in jobs.values():
        syn = j.cfg["dataset"]["synthetic"]
        if not isinstance(syn.get("p"), (int, float)):
            syn["p"] = 0.2
        if not isinstance(syn.get("clean_ratio"), (int, float)):
            syn["clean_ratio"] = 0.0
        j.alias_of = None
    mixed, _ = rx.estimate_minutes(jobs["e3_mixed"])
    real, _ = rx.estimate_minutes(jobs["e3_real"])
    assert mixed > real


def test_needs_selection_names_the_dependent_plans():
    assert rx.NEEDS_SELECTION == {"e2", "e3", "e4"}
    # e1 must NOT need selection, or the grid could never start.
    assert "e1" not in rx.NEEDS_SELECTION
