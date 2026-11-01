"""Tests for config layering, validation and ``${select.*}`` resolution.

The unknown-key check and the test-metric assertion are the two that matter
most: the first stops a typo from silently costing a GPU run, the second makes
the no-tuning-on-test rule a property of the code rather than of discipline.
"""

from __future__ import annotations

import pytest

from gec import config as cfgmod
from gec import paths


# --------------------------------------------------------------------------- #
# deep_merge
# --------------------------------------------------------------------------- #

def test_deep_merge_recurses_into_dicts():
    base = {"generation": {"num_beams": 4, "max_new_tokens": 128}, "epochs": 2}
    over = {"generation": {"num_beams": 1}}
    out = cfgmod.deep_merge(base, over)
    assert out["generation"] == {"num_beams": 1, "max_new_tokens": 128}
    assert out["epochs"] == 2


def test_deep_merge_replaces_lists_wholesale():
    """Lists replace rather than merge - `eval.splits` must not accumulate."""
    base = {"eval": {"splits": ["val", "test", "indian"]}}
    over = {"eval": {"splits": ["val"]}}
    assert cfgmod.deep_merge(base, over)["eval"]["splits"] == ["val"]


def test_deep_merge_does_not_mutate_its_inputs():
    base = {"a": {"b": 1}}
    over = {"a": {"c": 2}}
    cfgmod.deep_merge(base, over)
    assert base == {"a": {"b": 1}}


# --------------------------------------------------------------------------- #
# apply_set
# --------------------------------------------------------------------------- #

def test_apply_set_parses_scalars_as_yaml():
    cfg = cfgmod.apply_set({}, ["epochs=3", "precision=bf16", "keep_checkpoint=false"])
    assert cfg["epochs"] == 3 and isinstance(cfg["epochs"], int)
    assert cfg["precision"] == "bf16"
    assert cfg["keep_checkpoint"] is False


def test_apply_set_handles_nested_paths():
    cfg = cfgmod.apply_set({"dataset": {"synthetic": {"p": 0.2}}},
                           ["dataset.synthetic.p=0.4"])
    assert cfg["dataset"]["synthetic"]["p"] == 0.4


def test_apply_set_rejects_a_missing_equals():
    with pytest.raises(cfgmod.ConfigError, match="key.path=value"):
        cfgmod.apply_set({}, ["epochs"])


# --------------------------------------------------------------------------- #
# validate_config
# --------------------------------------------------------------------------- #

def test_validate_rejects_unknown_keys():
    """The typo guard: `clean_ration` must fail loudly, not silently do nothing."""
    base = cfgmod.load_yaml(paths.CONFIGS / "base.yaml")
    bad = cfgmod.deep_merge(base, {"dataset": {"synthetic": {"clean_ration": 0.33}}})
    with pytest.raises(cfgmod.ConfigError, match="unknown config key"):
        cfgmod.validate_config(bad, base)


def test_validate_allows_freeform_weight_names():
    """generator.weights keys are data, not config structure."""
    base = cfgmod.load_yaml(paths.CONFIGS / "base.yaml")
    ok = cfgmod.deep_merge(base, {"generator": {"weights": {"spell": 0.5, "sva": 0.5}}})
    cfgmod.validate_config(ok, base)


def test_fp16_is_forbidden():
    """CLAUDE.md: never fp16 for T5. Make it impossible, not merely discouraged."""
    base = cfgmod.load_yaml(paths.CONFIGS / "base.yaml")
    for value in ("fp16", "float16", "half"):
        with pytest.raises(cfgmod.ConfigError, match="fp16 is forbidden"):
            cfgmod.validate_config({**base, "precision": value}, base)


def test_precision_must_be_a_known_value():
    base = cfgmod.load_yaml(paths.CONFIGS / "base.yaml")
    with pytest.raises(cfgmod.ConfigError, match="precision must be"):
        cfgmod.validate_config({**base, "precision": "int8"}, base)


def test_condition_must_be_known():
    base = cfgmod.load_yaml(paths.CONFIGS / "base.yaml")
    with pytest.raises(cfgmod.ConfigError, match="condition must be"):
        cfgmod.validate_config({**base, "condition": "magic"}, base)


@pytest.mark.parametrize("key,value", [("p", 1.5), ("p", -0.1),
                                       ("clean_ratio", 2.0), ("clean_ratio", -1)])
def test_out_of_range_ratios_are_rejected(key, value):
    base = cfgmod.load_yaml(paths.CONFIGS / "base.yaml")
    bad = cfgmod.deep_merge(base, {"dataset": {"synthetic": {key: value}}})
    with pytest.raises(cfgmod.ConfigError, match="must be in"):
        cfgmod.validate_config(bad, base)


def test_stage2_without_init_from_is_rejected():
    base = cfgmod.load_yaml(paths.CONFIGS / "base.yaml")
    bad = cfgmod.deep_merge(base, {"stage2": {"enabled": True, "init_from": None}})
    with pytest.raises(cfgmod.ConfigError, match="init_from"):
        cfgmod.validate_config(bad, base)


# --------------------------------------------------------------------------- #
# ${select.*} resolution
# --------------------------------------------------------------------------- #

def test_resolve_refs_substitutes_from_selection():
    cfg = {"dataset": {"synthetic": {"p": "${select.e1_best_p}"}}}
    out = cfgmod.resolve_refs(cfg, selection={"e1_best_p": 0.4})
    assert out["dataset"]["synthetic"]["p"] == 0.4


def test_resolve_refs_fails_with_an_actionable_message():
    """An unresolvable ref must not silently default to something plausible."""
    with pytest.raises(cfgmod.ConfigError) as exc:
        cfgmod.resolve_refs({"p": "${select.e1_best_p}"}, selection={})
    assert "run_experiments --plan e1" in str(exc.value)


def test_resolve_refs_leaves_other_strings_alone():
    cfg = {"name": "e3_mixed", "prefix": "grammar: "}
    assert cfgmod.resolve_refs(cfg, selection={}) == cfg


def test_resolve_refs_walks_lists():
    out = cfgmod.resolve_refs({"xs": ["${select.a}", "plain"]}, selection={"a": 7})
    assert out["xs"] == [7, "plain"]


# --------------------------------------------------------------------------- #
# load_config end to end
# --------------------------------------------------------------------------- #

def test_load_config_merges_base_and_experiment():
    cfg = cfgmod.load_config(paths.EXPERIMENTS / "e1_p0.2.yaml", seed=13)
    assert cfg["name"] == "e1_p0.2"
    assert cfg["model_name"] == "t5-small"          # from base
    assert cfg["dataset"]["synthetic"]["p"] == 0.2   # from the experiment
    assert cfg["dataset"]["real"]["enabled"] is False
    assert cfg["seed"] == 13


def test_smoke_plan_overrides_the_experiment():
    cfg = cfgmod.load_config(paths.EXPERIMENTS / "e3_mixed.yaml", plan="smoke", seed=13)
    assert cfg["device"] == "cpu"
    assert cfg["max_steps"] == 20
    assert cfg["data"]["synthetic_n"] == 300
    assert cfg["run_prefix"] == "smoke_"
    # The smoke layer must also pin the generator settings, since e3_mixed
    # normally references ${select.*} which cannot resolve on a clean checkout.
    assert cfg["dataset"]["synthetic"]["p"] == 0.2
    assert cfg["alias_of"] is None


def test_run_name_applies_the_prefix():
    cfg = cfgmod.load_config(paths.EXPERIMENTS / "e3_mixed.yaml", plan="smoke", seed=13)
    assert cfgmod.run_name(cfg) == "smoke_e3_mixed"
    plain = cfgmod.load_config(paths.EXPERIMENTS / "e3_real.yaml", seed=13, resolve=False)
    assert cfgmod.run_name(plain) == "e3_real"


def test_stage2_init_from_gets_the_seed_substituted():
    cfg = cfgmod.load_config(paths.EXPERIMENTS / "e3_synth2real.yaml", seed=42,
                             resolve=False)
    assert cfg["stage2"]["init_from"] == "e3_synthetic/seed42"


def test_resolve_false_leaves_refs_as_strings():
    """--dry-run must be able to list jobs before selection.json exists."""
    cfg = cfgmod.load_config(paths.EXPERIMENTS / "e3_mixed.yaml", seed=13, resolve=False)
    assert cfg["dataset"]["synthetic"]["p"] == "${select.e1_best_p}"


def test_every_shipped_experiment_config_is_valid():
    """Catches a malformed or stale YAML before it costs a GPU slot."""
    for p in sorted(paths.EXPERIMENTS.glob("*.yaml")):
        if p.name.endswith("_weights.yaml"):
            continue  # weight files are data, not run configs
        cfg = cfgmod.load_config(p, seed=13, resolve=False)
        assert cfg["name"] == p.stem, f"{p.name}: name field does not match the filename"
        assert cfg.get("family"), f"{p.name}: missing family"
        assert cfg.get("condition"), f"{p.name}: missing condition"


def test_save_config_round_trips(tmp_path):
    import yaml

    cfg = cfgmod.load_config(paths.EXPERIMENTS / "e1_p0.2.yaml", seed=13)
    out = cfgmod.save_config(cfg, tmp_path)
    assert yaml.safe_load(paths.read_text(out))["dataset"]["synthetic"]["p"] == 0.2
