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


def test_alias_configs_declare_their_reuse():
    """E2's clean_ratio=0.0 and E3's synthetic arm are configured to reuse.

    Asserted on the CONFIG, not on the resolved Job: whether an alias actually
    survives also depends on disk state (seed match, and whether the chain
    still has weights), which _alias_for_seed decides and its own tests cover.
    """
    jobs = {j.config: j for j in rx.iter_jobs("e2", [13], resolve=False)}
    assert jobs["e2_clean0.0"].cfg.get("alias_of")
    assert not jobs["e2_clean0.2"].cfg.get("alias_of")


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
    # The target here has no model/, so there is nothing to copy. Weight
    # copying is covered by
    # test_alias_copies_the_targets_weights_when_a_checkpoint_is_wanted.
    assert not (job.dir / "model").exists()


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #

def test_alias_jobs_are_free_to_estimate():
    job = [j for j in rx.iter_jobs("e2", [13], resolve=False)
           if j.config == "e2_clean0.0"][0]
    job.alias_of = "e1_p0.2/seed13"          # set explicitly; see the note above
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


def test_alias_copies_the_targets_weights_when_a_checkpoint_is_wanted(tmp_runs):
    """e3_synth2real warm-starts from e3_synthetic/model, and so does the demo.

    An alias that stored only a pointer would leave both with nothing to load.
    """
    target = tmp_runs / "e2_clean0.33" / "seed13"
    (target / "model").mkdir(parents=True)
    paths.write_json(target / "metrics.json", {"_meta": {"complete": True}})
    paths.write_json(target / "model" / "config.json", {"model_type": "t5"})
    paths.write_text(target / "model" / "weights.bin", "pretend weights")

    job = [j for j in rx.iter_jobs("e3", [13], resolve=False)
           if j.config == "e3_synthetic"][0]
    job.alias_of = "e2_clean0.33/seed13"
    assert job.cfg["keep_checkpoint"] is True
    rx.write_alias(job)

    assert (job.dir / "model" / "config.json").exists(), \
        "stage-2 training and the demo both need real weights at this path"
    assert paths.read_text(job.dir / "model" / "weights.bin") == "pretend weights"


def test_alias_without_a_checkpoint_warns(tmp_runs, capsys):
    target = tmp_runs / "e2_clean0.33" / "seed13"
    target.mkdir(parents=True)
    paths.write_json(target / "metrics.json", {"_meta": {"complete": True}})

    job = [j for j in rx.iter_jobs("e3", [13], resolve=False)
           if j.config == "e3_synthetic"][0]
    job.alias_of = "e2_clean0.33/seed13"
    rx.write_alias(job)
    assert "no model/" in capsys.readouterr().out


def test_e2_runs_keep_their_checkpoints():
    """Because e3_synthetic aliases the best one and needs real weights."""
    for job in rx.iter_jobs("e2", [13], resolve=False):
        assert job.cfg["keep_checkpoint"] is True, job.config


def test_e1_runs_do_not_keep_checkpoints():
    """Nothing downstream needs them, and they are ~240 MB each."""
    for job in rx.iter_jobs("e1", [13], resolve=False):
        assert job.cfg["keep_checkpoint"] is False, job.config


# --------------------------------------------------------------------------- #
# Campaign driver: the concurrency guard
# --------------------------------------------------------------------------- #

def test_campaign_detects_a_running_trainer(monkeypatch):
    """The guard against two trainers sharing one 4 GB GPU.

    A chained launcher once used `kill -0 <windows-pid>` from Git Bash, which
    cannot see native Windows PIDs, so it started a second trainer alongside
    the first: throughput fell from 181 to 7.2 examples/s and both wrote into
    the same run directory.
    """
    import importlib.util
    import subprocess

    spec = importlib.util.spec_from_file_location(
        "run_campaign", paths.ROOT / "scripts" / "run_campaign.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fake = (
        '[{"ProcessId":4242,"CommandLine":"python.exe -m gec.run_experiments '
        '--plan e1 --yes"},'
        '{"ProcessId":4243,"CommandLine":"python.exe -c print(1)"}]'
    )

    class R:
        stdout = fake

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    hits = mod.already_running()
    assert [p for p, _ in hits] == [4242], "only project trainers count"


def test_campaign_ignores_unrelated_python_processes(monkeypatch):
    import importlib.util
    import subprocess

    spec = importlib.util.spec_from_file_location(
        "run_campaign2", paths.ROOT / "scripts" / "run_campaign.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class R:
        stdout = '[{"ProcessId":999,"CommandLine":"python.exe -m pytest"}]'

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert mod.already_running() == []


def test_campaign_stage_order_is_dependency_correct():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_campaign3", paths.ROOT / "scripts" / "run_campaign.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    names = [n for n, _, _ in mod.STAGES]
    # Selection must sit between the plans that produce and consume it.
    assert names.index("e1") < names.index("select1") < names.index("e2")
    assert names.index("e2") < names.index("select2") < names.index("e3")
    # E4 needs its matched weights first.
    assert names.index("distribution") < names.index("e4")
    # A complete report must exist before the optional extra seeds.
    assert names.index("aggregate") < names.index("e3_seeds")


# --------------------------------------------------------------------------- #
# Seed-aware aliasing
# --------------------------------------------------------------------------- #

def test_alias_is_dropped_when_the_target_belongs_to_another_seed():
    """selection.json records a concrete winner like e2_clean0.33/seed13.

    Applied naively, e3_synthetic/seed42 would alias that seed-13 run, so the
    extra seeds added for error bars would be three identical copies and
    figure 4 would report std = 0 for the synthetic condition.
    """
    assert rx._alias_for_seed("e2_clean0.33/seed13", 13, "e3_synthetic") == \
        "e2_clean0.33/seed13"
    assert rx._alias_for_seed("e2_clean0.33/seed13", 42, "e3_synthetic") is None
    assert rx._alias_for_seed("e2_clean0.33/seed13", 2026, "e3_synthetic") is None


def test_alias_without_a_seed_suffix_is_kept():
    assert rx._alias_for_seed("some_run", 42, "x") == "some_run"
    assert rx._alias_for_seed(None, 42, "x") is None


def test_extra_e3_seeds_train_the_synthetic_arm(monkeypatch, tmp_path):
    """End to end through iter_jobs, with a realistic selection.json."""
    from gec import config as cfg_mod

    sel = {"e1_best_p": 0.2, "e1_best_run": "e1_p0.2/seed13",
           "e2_best_clean_ratio": 0.33, "e2_best_run": "e2_clean0.33/seed13"}
    monkeypatch.setattr(cfg_mod, "load_selection", lambda: sel)

    seed42 = [j for j in rx.iter_jobs("e3", [42]) if j.config == "e3_synthetic"][0]
    assert seed42.alias_of is None, "seed 42 must train its own model"
    assert seed42.needs_train is True

    # The seed rule in isolation: seed 13 passes it, other seeds do not.
    # (Whether the alias ALSO survives the weights check is covered separately.)
    assert rx._alias_for_seed("e2_clean0.33/seed13", 13, "e3_synthetic") ==         "e2_clean0.33/seed13"
    assert rx._alias_for_seed("e2_clean0.33/seed13", 42, "e3_synthetic") is None


def test_alias_is_dropped_when_the_chain_has_no_weights(tmp_runs):
    """E1 prunes its checkpoints, so an alias chain ending there has no model.

    e2_clean0.0 aliases the best E1 run, whose weights are gone by then. If
    e2_clean0.0 wins E2, e3_synthetic aliases IT -- and e3_synth2real's warm
    start plus the demo's Synthetic-only column would have nothing to load.
    """
    # e1_p0.2: complete, but its model/ was pruned (keep_checkpoint: false).
    e1 = tmp_runs / "e1_p0.2" / "seed13"
    e1.mkdir(parents=True)
    paths.write_json(e1 / "metrics.json", {"_meta": {"complete": True}})

    # e2_clean0.0: an alias of that E1 run, so also weightless.
    e2 = tmp_runs / "e2_clean0.0" / "seed13"
    e2.mkdir(parents=True)
    paths.write_json(e2 / "metrics.json",
                     {"alias_of": "e1_p0.2/seed13", "_meta": {"complete": True}})

    assert rx._resolve_alias_chain("e2_clean0.0/seed13") == "e1_p0.2/seed13"

    # A run that needs a checkpoint must refuse the alias and train.
    assert rx._alias_for_seed("e2_clean0.0/seed13", 13, "e3_synthetic",
                              needs_checkpoint=True) is None
    # A run that does not need one may still reuse it.
    assert rx._alias_for_seed("e2_clean0.0/seed13", 13, "e2_clean0.0",
                              needs_checkpoint=False) == "e2_clean0.0/seed13"


def test_alias_is_kept_when_the_chain_does_have_weights(tmp_runs):
    tgt = tmp_runs / "e2_clean0.33" / "seed13"
    (tgt / "model").mkdir(parents=True)
    paths.write_json(tgt / "metrics.json", {"_meta": {"complete": True}})
    paths.write_json(tgt / "model" / "config.json", {"model_type": "t5"})
    assert rx._alias_for_seed("e2_clean0.33/seed13", 13, "e3_synthetic",
                              needs_checkpoint=True) == "e2_clean0.33/seed13"


def test_resolve_alias_chain_survives_a_cycle(tmp_runs):
    for a, b in (("x", "y"), ("y", "x")):
        d = tmp_runs / a / "seed13"
        d.mkdir(parents=True)
        paths.write_json(d / "metrics.json",
                         {"alias_of": f"{b}/seed13", "_meta": {"complete": True}})
    assert rx._resolve_alias_chain("x/seed13") in ("x/seed13", "y/seed13")
