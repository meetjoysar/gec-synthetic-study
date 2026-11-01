"""Tests for the synthetic error generator (spec section 6.2).

The spec requires: determinism, ``p=0`` gives no change, ``force_one``
guarantees at least one change, each op produces the expected kind of change on
a hand-made sentence, and no crashes on punctuation-only or very short input.
All of those are here, plus the span-disjointness invariant that enforces
"never corrupt the same token twice".
"""

from __future__ import annotations

import pytest

from gec import error_gen as eg
from gec import tok


@pytest.fixture(scope="module")
def ctx_simple() -> eg.Ctx:
    """`The student writes a long essay about the books .`"""
    return eg.Ctx.from_tagged(tok.tag_one("The student writes a long essay about the books ."))


def ctx_of(text: str) -> eg.Ctx:
    return eg.Ctx.from_tagged(tok.tag_one(text))


def find_index(ctx: eg.Ctx, word: str) -> int:
    return ctx.toks.index(word)


def apply_op(name: str, ctx: eg.Ctx, i: int, seed: int = 0) -> eg.GenEdit | None:
    import random

    return eg.OPS[name].apply(ctx, i, random.Random(seed))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_all_spec_ops_registered():
    """Spec 6.2 names exactly these 15 operations."""
    expected = {
        "det_delete", "det_insert", "det_swap",
        "prep_swap", "prep_delete",
        "sva", "verb_tense", "verb_form",
        "noun_num", "spell",
        "punct_delete", "punct_insert",
        "word_delete", "word_dup",
        "case",
    }
    assert set(eg.OPS) == expected


def test_errant_types_match_spec_table():
    """Each op maps to the ERRANT category spec 6.2's table assigns it."""
    expected = {
        "det_delete": "DET", "det_insert": "DET", "det_swap": "DET",
        "prep_swap": "PREP", "prep_delete": "PREP",
        "sva": "VERB:SVA", "verb_tense": "VERB:TENSE", "verb_form": "VERB:FORM",
        "noun_num": "NOUN:NUM", "spell": "SPELL",
        "punct_delete": "PUNCT", "punct_insert": "PUNCT",
        "word_delete": "OTHER", "word_dup": "OTHER",
        "case": "ORTH",
    }
    assert {n: o.errant_type for n, o in eg.OPS.items()} == expected


def test_default_weights_cover_every_op():
    assert set(eg.DEFAULT_WEIGHTS) == set(eg.OPS)
    assert all(w > 0 for w in eg.DEFAULT_WEIGHTS.values())


def test_covered_errant_types():
    assert eg.covered_errant_types() == {
        "DET", "PREP", "VERB:SVA", "VERB:TENSE", "VERB:FORM",
        "NOUN:NUM", "SPELL", "PUNCT", "OTHER", "ORTH",
    }


# --------------------------------------------------------------------------- #
# One targeted test per operation
# --------------------------------------------------------------------------- #

def test_det_delete(ctx_simple):
    e = apply_op("det_delete", ctx_simple, find_index(ctx_simple, "The"))
    assert e is not None and e.errant_type == "DET"
    assert e.before == ("The",) and e.after == ()


def test_det_delete_declines_on_non_determiner(ctx_simple):
    assert apply_op("det_delete", ctx_simple, find_index(ctx_simple, "student")) is None


def test_det_insert_before_noun(ctx_simple):
    e = apply_op("det_insert", ctx_simple, find_index(ctx_simple, "essay"))
    assert e is not None and e.errant_type == "DET"
    assert e.before == () and e.after[0].lower() in eg.DETERMINERS
    assert e.start == e.end, "an insertion has a zero-width span"


def test_det_insert_declines_when_determiner_already_present(ctx_simple):
    # "the books" - books is preceded by a determiner already.
    assert apply_op("det_insert", ctx_simple, find_index(ctx_simple, "books")) is None


def test_det_swap(ctx_simple):
    e = apply_op("det_swap", ctx_simple, find_index(ctx_simple, "a"))
    assert e is not None and e.errant_type == "DET"
    assert e.after[0].lower() in eg.DETERMINERS
    assert e.after[0].lower() != "a"


def test_prep_swap(ctx_simple):
    e = apply_op("prep_swap", ctx_simple, find_index(ctx_simple, "about"))
    assert e is not None and e.errant_type == "PREP"
    assert e.after[0].lower() in eg.PREPOSITIONS and e.after[0].lower() != "about"


def test_prep_delete(ctx_simple):
    e = apply_op("prep_delete", ctx_simple, find_index(ctx_simple, "about"))
    assert e is not None and e.after == ()


def test_sva_swaps_copula():
    ctx = ctx_of("The results are clear .")
    e = apply_op("sva", ctx, find_index(ctx, "are"))
    assert e is not None and e.errant_type == "VERB:SVA"
    assert e.after == ("is",)


def test_sva_toggles_third_person_s(ctx_simple):
    e = apply_op("sva", ctx_simple, find_index(ctx_simple, "writes"))
    assert e is not None and e.after[0] == "write", "VBZ -> VBP drops the -s"


def test_sva_handles_have_has():
    ctx = ctx_of("They have finished the work .")
    e = apply_op("sva", ctx, find_index(ctx, "have"))
    assert e is not None and e.after == ("has",)


def test_verb_tense_present_to_past(ctx_simple):
    e = apply_op("verb_tense", ctx_simple, find_index(ctx_simple, "writes"))
    assert e is not None and e.errant_type == "VERB:TENSE"
    assert e.after == ("wrote",)


def test_verb_tense_past_to_present():
    ctx = ctx_of("He went to the shop .")
    e = apply_op("verb_tense", ctx, find_index(ctx, "went"))
    assert e is not None
    assert e.after[0] in ("go", "goes"), f"VBD -> VBP/VBZ, got {e.after}"


def test_verb_form_after_to():
    ctx = ctx_of("She wants to go home .")
    e = apply_op("verb_form", ctx, find_index(ctx, "go"))
    assert e is not None and e.errant_type == "VERB:FORM"
    assert e.after[0] in ("going", "goes")


def test_verb_form_after_modal():
    ctx = ctx_of("He can swim very well .")
    e = apply_op("verb_form", ctx, find_index(ctx, "swim"))
    assert e is not None and e.after[0] in ("swimming", "swims")


def test_noun_num_singular_to_plural(ctx_simple):
    e = apply_op("noun_num", ctx_simple, find_index(ctx_simple, "essay"))
    assert e is not None and e.errant_type == "NOUN:NUM"
    assert e.after == ("essays",)


def test_noun_num_plural_to_singular(ctx_simple):
    e = apply_op("noun_num", ctx_simple, find_index(ctx_simple, "books"))
    assert e is not None and e.after == ("book",)


@pytest.mark.parametrize("seed", range(8))
def test_spell_changes_one_word_only(ctx_simple, seed):
    i = find_index(ctx_simple, "student")
    e = apply_op("spell", ctx_simple, i, seed=seed)
    assert e is not None and e.errant_type == "SPELL"
    assert len(e.after) == 1 and e.after[0] != "student"
    # One edit operation means the length changes by at most one character.
    assert abs(len(e.after[0]) - len("student")) <= 1
    assert e.after[0][0] == "s", "the first character is left alone"


def test_spell_declines_on_short_words(ctx_simple):
    assert apply_op("spell", ctx_simple, find_index(ctx_simple, "a")) is None
    assert apply_op("spell", ctx_simple, find_index(ctx_simple, ".")) is None


def test_punct_delete_final_period(ctx_simple):
    e = apply_op("punct_delete", ctx_simple, len(ctx_simple) - 1)
    assert e is not None and e.errant_type == "PUNCT"
    assert e.before == (".",) and e.after == ()


def test_punct_delete_declines_on_non_final_period():
    ctx = ctx_of("He left . She stayed .")
    assert apply_op("punct_delete", ctx, 2) is None, "only the FINAL period may go"


def test_punct_insert_adds_comma(ctx_simple):
    e = apply_op("punct_insert", ctx_simple, 3)
    assert e is not None and e.errant_type == "PUNCT"
    assert e.after == (",",)
    assert e.start == e.end, "an insertion has a zero-width span"


def test_word_delete_function_word():
    ctx = ctx_of("I think that he is right .")
    e = apply_op("word_delete", ctx, find_index(ctx, "that"))
    assert e is not None and e.errant_type == "OTHER"
    assert e.after == ()


def test_word_dup(ctx_simple):
    e = apply_op("word_dup", ctx_simple, find_index(ctx_simple, "student"))
    assert e is not None and e.errant_type == "OTHER"
    assert e.after == ("student",) and e.start == e.end


def test_case_lowercases_sentence_initial(ctx_simple):
    e = apply_op("case", ctx_simple, 0)
    assert e is not None and e.errant_type == "ORTH"
    assert e.after == ("the",)


def test_case_capitalises_mid_sentence_common_word(ctx_simple):
    e = apply_op("case", ctx_simple, find_index(ctx_simple, "student"))
    assert e is not None and e.after == ("Student",)


# --------------------------------------------------------------------------- #
# Determinism and the p / force_one contract
# --------------------------------------------------------------------------- #

def test_determinism_same_seed_same_output():
    text = "The committee decided to postpone the meeting until Friday ."
    t = tok.tag_one(text)
    a = eg.ErrorGenerator(seed=13).corrupt_tagged(t, p=0.4, force_one=True)[0]
    b = eg.ErrorGenerator(seed=13).corrupt_tagged(t, p=0.4, force_one=True)[0]
    assert a == b


def test_different_seeds_usually_differ():
    t = tok.tag_one("The committee decided to postpone the meeting until Friday .")
    outs = {
        " ".join(eg.ErrorGenerator(seed=s).corrupt_tagged(t, p=0.4, force_one=True)[0])
        for s in (13, 42, 2026, 7, 99)
    }
    assert len(outs) > 1


def test_determinism_across_a_sequence():
    """A whole corpus must be reproducible, not just one sentence."""
    texts = [
        "The student writes a long essay about the books .",
        "She has been working at the university since 2019 .",
        "They have finished the project and submitted the report .",
    ]
    tagged = list(tok.tag_batch(texts))

    def run():
        gen = eg.ErrorGenerator(seed=13)
        return [" ".join(gen.corrupt_tagged(t, p=0.3, force_one=True)[0]) for t in tagged]

    assert run() == run()


def test_p_zero_gives_no_change():
    t = tok.tag_one("The student writes a long essay about the books .")
    gen = eg.ErrorGenerator(seed=13)
    noisy, edits = gen.corrupt_tagged(t, p=0.0, force_one=False)
    assert noisy == t.toks
    assert edits == []


def test_force_one_guarantees_a_change_on_normal_sentences():
    texts = [
        "The student writes a long essay about the books .",
        "She has been working at the university since 2019 .",
        "He goes to the library every morning before class .",
        "A small group of scientists is studying the problem .",
        "This book describes the history of the city .",
    ]
    gen = eg.ErrorGenerator(seed=13)
    for t in tok.tag_batch(texts):
        noisy, edits = gen.corrupt_tagged(t, p=0.0, force_one=True)
        assert edits, f"force_one produced no edit for: {t.text}"
        assert noisy != t.toks


def test_higher_p_produces_more_edits():
    texts = ["The student writes a long essay about the books ."] * 40
    tagged = list(tok.tag_batch(texts))

    def total(p: float) -> int:
        gen = eg.ErrorGenerator(seed=13)
        return sum(len(gen.corrupt_tagged(t, p=p)[1]) for t in tagged)

    assert total(0.1) < total(0.5)


# --------------------------------------------------------------------------- #
# "Never corrupt the same token twice"
# --------------------------------------------------------------------------- #

def test_edit_spans_are_disjoint_at_high_p():
    """The `used` mask must prevent any two edits touching the same token."""
    texts = [
        "The student writes a long essay about the books .",
        "She has been working at the university since 2019 .",
        "The committee decided to postpone the meeting until Friday .",
    ] * 60
    gen = eg.ErrorGenerator(seed=13)
    checked = 0
    for t in tok.tag_batch(texts):
        _, edits = gen.corrupt_tagged(t, p=0.9, force_one=True)
        claimed: set[int] = set()
        for e in edits:
            span = set(range(e.start, max(e.end, e.start + 1)))
            assert not (span & claimed), f"overlapping edits on {t.text}: {edits}"
            claimed |= span
        checked += 1
    assert checked == len(texts)


def test_edits_reference_original_indices():
    """Collect-then-apply: every edit's span must index the ORIGINAL tokens."""
    t = tok.tag_one("The student writes a long essay about the books .")
    gen = eg.ErrorGenerator(seed=5)
    _, edits = gen.corrupt_tagged(t, p=0.9, force_one=True)
    for e in edits:
        assert 0 <= e.start <= len(t.toks)
        assert e.end <= len(t.toks)
        if e.before:
            assert tuple(t.toks[e.start:e.end]) == e.before


# --------------------------------------------------------------------------- #
# Robustness (spec 6.2: no crashes on degenerate input)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "toks",
    [
        [],
        ["."],
        [",", ".", "!"],
        ["Hi", "."],
        ["a"],
        ["..."],
        ["123", "456"],
    ],
)
def test_no_crash_on_degenerate_input(toks):
    gen = eg.ErrorGenerator(seed=13)
    noisy, edits = gen.corrupt(toks, p=0.9, force_one=True)
    assert isinstance(noisy, list)
    assert all(isinstance(x, str) for x in noisy)


def test_uncorruptible_input_returns_unchanged_and_counts_failure():
    """When nothing applies, force_one must give up and TALLY the failure.

    `["!", "?"]` has no applicable op: punct_delete only takes "," or ".",
    punct_insert needs an interior position, and every lexical op declines on
    punctuation. Such a pair would silently become an identity pair, so the
    count is surfaced in dataset stats rather than passing unnoticed.
    """
    eg.FORCE_ONE_FAILURES.clear()
    gen = eg.ErrorGenerator(seed=13)
    noisy, edits = gen.corrupt(["!", "?"], p=0.9, force_one=True)
    assert noisy == ["!", "?"]
    assert edits == []
    assert eg.FORCE_ONE_FAILURES["total"] == 1, "a silent identity pair must be counted"


def test_stray_comma_is_deletable():
    """Documents that punct_delete DOES fire on a comma-only input."""
    gen = eg.ErrorGenerator(seed=13)
    noisy, edits = gen.corrupt([".", ","], p=0.9, force_one=True)
    assert edits, "a stray comma is a legitimate thing to delete"
    assert noisy == ["."]


def test_corrupt_without_tags_falls_back_to_spacy():
    """The spec 6.1 signature allows calling with tokens only."""
    gen = eg.ErrorGenerator(seed=13)
    noisy, edits = gen.corrupt("The student writes a long essay .".split(), p=0.9, force_one=True)
    assert edits


# --------------------------------------------------------------------------- #
# Weights and configuration
# --------------------------------------------------------------------------- #

def test_unknown_op_in_weights_raises():
    with pytest.raises(ValueError, match="unknown generator ops"):
        eg.ErrorGenerator(weights={"not_a_real_op": 1.0}, seed=13)


def test_zero_total_weight_raises():
    with pytest.raises(ValueError, match="positive"):
        eg.ErrorGenerator(weights={"spell": 0.0}, seed=13)


def test_weights_are_normalised():
    gen = eg.ErrorGenerator(weights={"spell": 2.0, "case": 2.0}, seed=13)
    assert pytest.approx(sum(gen.weights.values()), abs=1e-9) == 1.0


def test_enabled_restricts_ops():
    """E4 and the distribution tests need single-op generators."""
    gen = eg.ErrorGenerator(seed=13, enabled=["det_delete"])
    assert set(gen.weights) == {"det_delete"}
    texts = ["The student reads the book about the city ."] * 20
    for t in tok.tag_batch(texts):
        _, edits = gen.corrupt_tagged(t, p=0.9, force_one=True)
        assert all(e.op == "det_delete" for e in edits)


def test_single_op_generator_only_deletes_determiners():
    """A det_delete-only generator must only ever remove a/an/the.

    distribution.py relies on this: it asserts that a det_delete-only probe
    yields predominantly M:DET (a MISSING determiner in the noisy text), which
    is how we catch the generation direction being inverted.
    """
    gen = eg.ErrorGenerator(seed=13, enabled=["det_delete"])
    t = tok.tag_one("The student reads the book about the city .")
    noisy, edits = gen.corrupt_tagged(t, p=1.0, force_one=True)
    assert edits
    removed = [w for w in t.toks if w not in noisy or t.toks.count(w) > noisy.count(w)]
    assert all(w.lower() in eg.DETERMINERS for w in removed), removed
    assert len(noisy) < len(t.toks)


def test_stats_are_recorded():
    gen = eg.ErrorGenerator(seed=13)
    for t in tok.tag_batch(["The student writes a long essay about the books ."] * 10):
        gen.corrupt_tagged(t, p=0.3, force_one=True)
    s = gen.stats
    assert s["n_sentences"] == 10
    assert s["n_changed"] == 10, "force_one means every sentence changes"
    assert s["total_edits"] >= 10
    assert sum(gen.op_counts.values()) == s["total_edits"]


def test_match_case():
    assert eg.match_case("The", "a") == "A"
    assert eg.match_case("the", "A") == "a", "a lowercase original must stay lowercase"
    assert eg.match_case("BOOK", "books") == "BOOKS"
    assert eg.match_case("book", "books") == "books"
    assert eg.match_case("Book", "books") == "Books"
