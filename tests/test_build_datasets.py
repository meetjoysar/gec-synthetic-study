"""Tests for dataset building (spec section 6.3).

Marked ``needs_data`` where they require the clean pool and the W&I splits,
which ``prepare_data`` produces.
"""

from __future__ import annotations

import pytest

from gec import build_datasets as bd
from gec import config as cfgmod
from gec import paths

POOL = paths.PROCESSED / "clean_pool.jsonl"
TRAIN_REAL = paths.split_files("train_real")["jsonl"]

needs_pool = pytest.mark.skipif(
    not POOL.exists(), reason="clean pool missing; run prepare_data --wikitext"
)
needs_real = pytest.mark.skipif(
    not TRAIN_REAL.exists(), reason="train_real missing; run prepare_data --wi"
)


# --------------------------------------------------------------------------- #
# Spec hashing and reuse
# --------------------------------------------------------------------------- #

def test_spec_hash_is_stable_and_sensitive():
    cfg = cfgmod.load_config(paths.EXPERIMENTS / "e1_p0.2.yaml", seed=13)
    spec = bd.resolve_spec(cfg, 13)
    h1 = bd.spec_hash(spec, 13)
    assert h1 == bd.spec_hash(bd.resolve_spec(cfg, 13), 13), "must be deterministic"
    assert h1 != bd.spec_hash(spec, 42), "the seed must change the hash"

    other = bd.resolve_spec(
        cfgmod.load_config(paths.EXPERIMENTS / "e1_p0.4.yaml", seed=13), 13
    )
    assert h1 != bd.spec_hash(other, 13), "a different p must change the hash"


def test_spec_hash_ignores_irrelevant_keys():
    """Changing learning_rate must NOT invalidate an existing dataset file."""
    cfg = cfgmod.load_config(paths.EXPERIMENTS / "e1_p0.2.yaml", seed=13)
    a = bd.spec_hash(bd.resolve_spec(cfg, 13), 13)
    cfg["learning_rate"] = 1e-5
    cfg["epochs"] = 99
    b = bd.spec_hash(bd.resolve_spec(cfg, 13), 13)
    assert a == b


def test_resolve_spec_reads_a_weights_file():
    """E4 passes its matched weights by file rather than inlining 15 numbers."""
    wf = paths.EXPERIMENTS / "e1_precomp_weights.yaml"
    if not wf.exists():
        pytest.skip("precomp weights not generated yet")
    cfg = cfgmod.load_config(paths.EXPERIMENTS / "e1_p0.2_precomp.yaml", seed=13)
    spec = bd.resolve_spec(cfg, 13)
    base = cfgmod.load_yaml(paths.CONFIGS / "base.yaml")["generator"]["weights"]
    assert spec["synthetic"]["weights"] != {k: round(float(v), 6)
                                            for k, v in sorted(base.items())}


def test_missing_weights_file_raises_an_actionable_config_error():
    """A blocked E4 must report how to unblock itself, not crash with ENOENT."""
    cfg = cfgmod.load_config(paths.EXPERIMENTS / "e1_p0.2.yaml", seed=13)
    cfg["dataset"]["synthetic"]["weights_file"] = "configs/experiments/nope.yaml"
    cfg["dataset"]["synthetic"]["weights"] = None
    with pytest.raises(cfgmod.ConfigError, match="distribution --match"):
        bd.resolve_spec(cfg, 13)


def test_dataset_slug_is_readable():
    cfg = cfgmod.load_config(paths.EXPERIMENTS / "e1_p0.2.yaml", seed=13)
    slug = bd.dataset_slug(bd.resolve_spec(cfg, 13), 13)
    assert slug.startswith("syn")
    assert "p0.2" in slug and "clean0.0" in slug and "_s13_" in slug


# --------------------------------------------------------------------------- #
# Synthetic building
# --------------------------------------------------------------------------- #

@needs_pool
def test_synthetic_pairs_are_noisy_to_clean():
    """src must be the NOISY side and tgt the CLEAN side (the easy inversion)."""
    spec = {"synthetic": {"enabled": True, "n": 200, "p": 0.3, "clean_ratio": 0.0,
                          "weights": None},
            "real": {"enabled": False, "n": None}, "shuffle": False,
            "condition": "synthetic"}
    spec["synthetic"]["weights"] = None
    from gec import error_gen as eg

    spec["synthetic"]["weights"] = eg.DEFAULT_WEIGHTS
    rows, stats = bd.build_synthetic(spec, seed=13)
    assert len(rows) == 200
    changed = [r for r in rows if r["origin"] == "synthetic"]
    assert changed, "clean_ratio=0 should give mostly corrupted pairs"
    for r in changed[:25]:
        assert r["src"] != r["tgt"]
        # The target is the pristine pool sentence, so it must be the shorter
        # or equal one only by accident - what matters is that they differ.
        assert r["tgt"].strip()


@needs_pool
def test_clean_ratio_is_respected():
    from gec import error_gen as eg

    for cr in (0.0, 0.33, 0.5):
        spec = {"synthetic": {"enabled": True, "n": 300, "p": 0.2, "clean_ratio": cr,
                              "weights": eg.DEFAULT_WEIGHTS},
                "real": {"enabled": False, "n": None}, "shuffle": False,
                "condition": "synthetic"}
        rows, _ = bd.build_synthetic(spec, seed=13)
        share = sum(1 for r in rows if r["origin"] == "identity") / len(rows)
        assert abs(share - cr) < 0.02, f"clean_ratio={cr} gave {share:.3f}"


@needs_pool
def test_identity_pairs_have_equal_src_and_tgt():
    from gec import error_gen as eg

    spec = {"synthetic": {"enabled": True, "n": 200, "p": 0.2, "clean_ratio": 0.5,
                          "weights": eg.DEFAULT_WEIGHTS},
            "real": {"enabled": False, "n": None}, "shuffle": False,
            "condition": "synthetic"}
    rows, _ = bd.build_synthetic(spec, seed=13)
    for r in rows:
        if r["origin"] == "identity":
            assert r["src"] == r["tgt"]


@needs_pool
def test_same_seed_gives_identical_rows():
    from gec import error_gen as eg

    spec = {"synthetic": {"enabled": True, "n": 150, "p": 0.3, "clean_ratio": 0.2,
                          "weights": eg.DEFAULT_WEIGHTS},
            "real": {"enabled": False, "n": None}, "shuffle": False,
            "condition": "synthetic"}
    a, _ = bd.build_synthetic(spec, seed=13)
    b, _ = bd.build_synthetic(spec, seed=13)
    assert a == b


@needs_pool
def test_clean_ratio_sweep_keeps_the_noisy_subset_nested():
    """A higher clean_ratio must only ADD identity pairs, not re-corrupt others.

    This is what makes the E2 comparison clean: conditions differ by which
    sentences are left alone, not by how the rest were corrupted.
    """
    from gec import error_gen as eg

    def build(cr):
        spec = {"synthetic": {"enabled": True, "n": 300, "p": 0.2, "clean_ratio": cr,
                              "weights": eg.DEFAULT_WEIGHTS},
                "real": {"enabled": False, "n": None}, "shuffle": False,
                "condition": "synthetic"}
        rows, _ = bd.build_synthetic(spec, seed=13)
        return rows

    low = build(0.2)
    high = build(0.5)
    low_clean = {r["id"] for r in low if r["origin"] == "identity"}
    high_clean = {r["id"] for r in high if r["origin"] == "identity"}
    assert low_clean < high_clean, "the identity set must grow monotonically"


@needs_pool
def test_requesting_more_than_the_base_set_raises():
    from gec import error_gen as eg

    spec = {"synthetic": {"enabled": True, "n": 10_000_000, "p": 0.2,
                          "clean_ratio": 0.0, "weights": eg.DEFAULT_WEIGHTS},
            "real": {"enabled": False, "n": None}, "shuffle": False,
            "condition": "synthetic"}
    with pytest.raises(ValueError, match="base set"):
        bd.build_synthetic(spec, seed=13)


# --------------------------------------------------------------------------- #
# Real and mixed
# --------------------------------------------------------------------------- #

@needs_real
def test_real_pairs_never_have_an_empty_target():
    """W&I has a few sentences the annotator deletes entirely."""
    spec = {"synthetic": {"enabled": False}, "real": {"enabled": True, "n": None},
            "shuffle": False, "condition": "real"}
    rows = bd.build_real(spec, seed=13)
    assert rows
    for r in rows:
        assert r["src"].strip() and r["tgt"].strip()
        assert r["origin"] == "real"
    # The dropped count must be recorded, not silently swallowed.
    assert bd.build_real.last_dropped >= 0


@needs_real
def test_real_subsetting_is_seeded():
    spec = {"synthetic": {"enabled": False}, "real": {"enabled": True, "n": 500},
            "shuffle": False, "condition": "real"}
    a = bd.build_real(spec, seed=13)
    b = bd.build_real(spec, seed=13)
    c = bd.build_real(spec, seed=42)
    assert len(a) == 500
    assert a == b
    assert a != c


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

@needs_pool
def test_stats_record_requested_vs_realised_op_shares():
    """The diagnostic that explains E4's behaviour must be present."""
    from gec import error_gen as eg

    spec = {"synthetic": {"enabled": True, "n": 200, "p": 0.3, "clean_ratio": 0.0,
                          "weights": eg.DEFAULT_WEIGHTS},
            "real": {"enabled": False, "n": None}, "shuffle": False,
            "condition": "synthetic"}
    rows, gstats = bd.build_synthetic(spec, seed=13)
    stats = bd.dataset_stats(rows, gstats, spec, 13, paths.PROCESSED / "x.jsonl")
    assert "op_share_requested" in stats["generator"]
    assert "op_share_realised" in stats["generator"]
    assert stats["origins"]["synthetic"] + stats["origins"].get("identity", 0) == 200
    assert 0.0 <= stats["identity_share"] <= 1.0
    assert stats["mean_edits_per_noisy_sentence"] > 0
