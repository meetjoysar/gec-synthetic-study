"""Tests for error-type distributions and generator re-weighting (spec 7.1).

The direction test is the important one here. If the probe were annotated with
``-orig clean -cor noisy`` instead of the other way round, every ``M:``/``U:``
pairing would invert (a *missing* determiner would be counted as an
*unnecessary* one) and all three E4 iterations would be silently wrong while
still producing plausible-looking numbers.
"""

from __future__ import annotations

import pytest

from gec import distribution as dist
from gec import error_gen as eg
from gec import paths

POOL = paths.PROCESSED / "clean_pool.jsonl"
needs_pool = pytest.mark.skipif(
    not POOL.exists(), reason="clean pool missing; run prepare_data --wikitext"
)
needs_real = pytest.mark.skipif(
    not paths.split_files("train_real")["m2"].exists(),
    reason="train_real missing; run prepare_data --wi",
)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def test_shares_normalises_and_buckets_uncovered():
    from collections import Counter

    counts = Counter({"DET": 50, "PREP": 30, "WO": 20})
    out = dist.shares(counts, covered={"DET", "PREP"})
    assert out["DET"] == pytest.approx(0.5)
    assert out["PREP"] == pytest.approx(0.3)
    assert out[dist.UNCOVERED] == pytest.approx(0.2)
    assert sum(out.values()) == pytest.approx(1.0)


def test_shares_includes_covered_types_with_zero_count():
    from collections import Counter

    out = dist.shares(Counter({"DET": 1}), covered={"DET", "SPELL"})
    assert out["SPELL"] == 0.0


def test_shares_of_nothing_is_empty():
    from collections import Counter

    assert dist.shares(Counter()) == {}


def test_tvd_properties():
    p = {"a": 0.5, "b": 0.5}
    assert dist.tvd(p, p) == 0.0
    assert dist.tvd({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}) == 1.0
    # Symmetric
    q = {"a": 0.25, "b": 0.75}
    assert dist.tvd(p, q) == dist.tvd(q, p)


def test_covered_types_matches_the_generator():
    assert dist.covered_types() == eg.covered_errant_types()
    assert len(dist.covered_types()) == 10


# --------------------------------------------------------------------------- #
# reweight
# --------------------------------------------------------------------------- #

def test_reweight_moves_toward_the_target():
    weights = {"spell": 0.5, "sva": 0.5}
    real = {"SPELL": 0.1, "VERB:SVA": 0.9}
    syn = {"SPELL": 0.9, "VERB:SVA": 0.1}
    out = dist.reweight(weights, real, syn)
    assert out["sva"] > out["spell"], "the under-produced op must gain weight"
    assert sum(out.values()) == pytest.approx(1.0)


def test_reweight_clips_each_step():
    """Spec 7.1 step 3: clip the ratio to [0.2x, 5x] per iteration."""
    weights = {"spell": 0.5, "sva": 0.5}
    # A 1000x gap must still only move by 5x before renormalising.
    out = dist.reweight(weights, {"SPELL": 0.001, "VERB:SVA": 0.999},
                        {"SPELL": 0.999, "VERB:SVA": 0.001})
    ratio = out["sva"] / out["spell"]
    assert ratio == pytest.approx(25.0, rel=0.01), "5x up and 5x down, nothing more"


def test_reweight_handles_a_zero_synthetic_share():
    """An op that never fires must not divide by zero."""
    out = dist.reweight({"spell": 0.5, "sva": 0.5},
                        {"SPELL": 0.5, "VERB:SVA": 0.5},
                        {"SPELL": 1.0, "VERB:SVA": 0.0})
    assert all(v > 0 for v in out.values())
    assert sum(out.values()) == pytest.approx(1.0)


def test_reweight_gives_ops_of_one_type_the_same_ratio():
    """DET's three ops share a category, so they scale together."""
    weights = {"det_delete": 0.2, "det_insert": 0.2, "det_swap": 0.2, "spell": 0.4}
    out = dist.reweight(weights, {"DET": 0.9, "SPELL": 0.1}, {"DET": 0.1, "SPELL": 0.9})
    # Within-group proportions are preserved (all three were equal).
    assert out["det_delete"] == pytest.approx(out["det_insert"])
    assert out["det_insert"] == pytest.approx(out["det_swap"])
    assert out["det_delete"] > out["spell"]


# --------------------------------------------------------------------------- #
# Probe generation and direction
# --------------------------------------------------------------------------- #

@needs_pool
def test_make_probe_returns_a_named_tuple(tmp_path):
    """Named fields, so a call site cannot swap noisy and clean by position."""
    probe = dist.make_probe(eg.DEFAULT_WEIGHTS, p=0.3, n=20, seed=1, out_dir=tmp_path)
    assert isinstance(probe, dist.Probe)
    assert probe.noisy.name.endswith("noisy.txt")
    assert probe.clean.name.endswith("clean.txt")
    assert probe.n == paths.count_lines(probe.noisy) == paths.count_lines(probe.clean)


@needs_pool
def test_probe_draws_from_the_held_back_pool_tail(tmp_path):
    """A probe must never share sentences with any training set."""
    from gec import prepare_data

    probe = dist.make_probe(eg.DEFAULT_WEIGHTS, p=0.2, n=50, seed=1, out_dir=tmp_path)
    clean = set(paths.read_lines(probe.clean))
    base = {t.text for t in prepare_data.load_pool(limit=prepare_data.POOL_BASE_SET)}
    assert not (clean & base), "probe sentences leaked into the training base set"


@needs_pool
def test_probe_noisy_and_clean_differ(tmp_path):
    probe = dist.make_probe(eg.DEFAULT_WEIGHTS, p=0.3, n=30, seed=1, out_dir=tmp_path)
    noisy = paths.read_lines(probe.noisy)
    clean = paths.read_lines(probe.clean)
    assert sum(1 for a, b in zip(noisy, clean) if a != b) >= 25, \
        "force_one should corrupt nearly every probe sentence"


@needs_pool
@pytest.mark.needs_errant
@pytest.mark.slow
def test_probe_direction_yields_missing_not_unnecessary(tmp_path):
    """The guard against inverting the generation direction.

    A det_delete-only generator removes determiners from the clean text, so the
    noisy text is MISSING them. Annotated correctly (-orig noisy -cor clean)
    ERRANT must therefore report predominantly M:DET. If the direction were
    reversed it would report U:DET instead, and the whole E4 loop would be
    optimising against an inverted target.
    """
    gen_weights = {"det_delete": 1.0}
    probe = dist.make_probe(gen_weights, p=1.0, n=60, seed=5, out_dir=tmp_path)
    counts = dist.measure_probe(probe, tmp_path / "probe.m2")
    assert counts, "the probe produced no edits at all"

    from gec import m2

    level3 = m2.count_types(tmp_path / "probe.m2", level=3)
    missing = sum(v for k, v in level3.items() if k.startswith("M:DET"))
    unnecessary = sum(v for k, v in level3.items() if k.startswith("U:DET"))
    assert missing > unnecessary, (
        f"direction looks INVERTED: M:DET={missing}, U:DET={unnecessary}. "
        "measure_probe must pass -orig noisy -cor clean."
    )


# --------------------------------------------------------------------------- #
# Real distribution
# --------------------------------------------------------------------------- #

@needs_real
def test_real_distribution_sums_to_one_and_has_uncovered_mass():
    counts, sh = dist.real_distribution()
    assert sum(counts.values()) > 10_000
    assert sum(sh.values()) == pytest.approx(1.0)
    # W&I contains WO/MORPH/NOUN:POSS etc. that the generator cannot produce.
    assert sh[dist.UNCOVERED] > 0.1


@needs_real
def test_real_distribution_has_the_expected_big_categories():
    _, sh = dist.real_distribution()
    for cat in ("DET", "PREP", "PUNCT", "VERB:TENSE"):
        assert sh[cat] > 0.01, f"{cat} share looks implausibly low"
