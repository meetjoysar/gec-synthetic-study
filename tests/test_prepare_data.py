"""Tests for data preparation (spec section 5).

The alignment-contract tests run against the real prepared splits when they
exist, because that is the invariant the whole ERRANT pipeline rests on:
``compare_m2`` asserts hypothesis and reference have equal sentence counts with
no message, so any drift must be caught here instead.
"""

from __future__ import annotations

import pytest

from gec import m2, paths, prepare_data, tok

SPLITS = ("train_real", "val", "test")
needs_wi = pytest.mark.skipif(
    not paths.split_files("val")["m2"].exists(),
    reason="W&I splits missing; run prepare_data --wi",
)
needs_pool = pytest.mark.skipif(
    not (paths.PROCESSED / "clean_pool.jsonl").exists(),
    reason="clean pool missing; run prepare_data --wikitext",
)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def test_clean_wikitext_line_drops_headings_and_blanks():
    assert prepare_data._clean_wikitext_line(" = Heading = ") is None
    assert prepare_data._clean_wikitext_line("") is None
    assert prepare_data._clean_wikitext_line("   ") is None


def test_clean_wikitext_line_applies_the_spec_substitutions():
    """Spec 5.4: ` @-@ ` -> `-`, ` @,@ ` -> `,`, ` @.@ ` -> `.`"""
    out = prepare_data._clean_wikitext_line("a @-@ b and 1 @,@ 000 and 2 @.@ 5")
    assert out == "a-b and 1,000 and 2.5"


def test_pool_filter_rejects_by_length():
    short = tok.tag_one("Too short .")
    assert prepare_data._pool_sentence_ok(short) == "length"


def test_pool_filter_requires_a_verb_or_aux():
    """spaCy tags `is` as AUX, so an AUX-only sentence must be ACCEPTED."""
    copula = tok.tag_one("The result of the experiment is remarkably clear today .")
    assert prepare_data._pool_sentence_ok(copula) is None

    nominal = tok.Tagged(
        toks=["A", "list", "of", "nouns", "and", "more", "nouns", "here", "."],
        upos=["DET", "NOUN", "ADP", "NOUN", "CCONJ", "ADJ", "NOUN", "ADV", "PUNCT"],
        tag=["DT", "NN", "IN", "NNS", "CC", "JJ", "NNS", "RB", "."],
        lemma=["a", "list", "of", "noun", "and", "more", "noun", "here", "."],
    )
    assert prepare_data._pool_sentence_ok(nominal) == "no_verb"


def test_pool_filter_rejects_text_that_folding_would_destroy():
    """The charset test is "does folding lose information", not a ratio.

    A long English sentence with one embedded Greek word clears any 95%
    character threshold while still leaving a "??????" token behind.
    """
    greek = tok.tag_one(
        "Lippmann conjectured a hypothetical Greek word "
        "ανθεμονιον which is a floret ."
    )
    assert prepare_data._pool_sentence_ok(greek) == "charset"


def test_pool_filter_keeps_accented_latin():
    """`cafe` is a clean fold, so the sentence is useful and must be kept."""
    accented = tok.tag_one(
        "She visited the café in Montréal and wrote a short résumé there ."
    )
    assert prepare_data._pool_sentence_ok(accented) is None


def test_fold_tagged_preserves_token_count_and_tags():
    t = tok.tag_one("She visited the café in Montréal yesterday evening .")
    folded = prepare_data._fold_tagged(t)
    assert len(folded.toks) == len(t.toks)
    assert folded.upos == list(t.upos) and folded.tag == list(t.tag)
    assert "café" not in folded.text and "cafe" in folded.text


# --------------------------------------------------------------------------- #
# M2 normalisation must not damage the format
# --------------------------------------------------------------------------- #

def test_normalise_a_line_preserves_the_pipe_separators():
    """`|` is not in T5's vocabulary, so folding a whole A line destroys it.

    This bit once turned every `|||` into `???`.
    """
    line = "A 1 2|||R:VERB:SVA|||café|||REQUIRED|||-NONE-|||0"
    out = prepare_data._normalise_a_line(line)
    assert out.count("|||") == line.count("|||")
    assert out.split("|||")[2] == "cafe"
    assert out.split("|||")[1] == "R:VERB:SVA"
    assert out.split("|||")[-1] == "0"


def test_normalise_a_line_leaves_a_deletion_empty():
    """Rewriting "" as -NONE- would stop the edit tuple matching the hypothesis."""
    line = "A 2 3|||U:ADV||||||REQUIRED|||-NONE-|||0"
    assert prepare_data._normalise_a_line(line) == line


def test_normalise_a_line_is_a_noop_for_ascii():
    line = "A 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0"
    assert prepare_data._normalise_a_line(line) == line


def test_normalise_block_folds_source_and_corrections(tmp_path):
    raw = ('S He said “hello” to the café owner .\n'
           'A 3 4|||R:NOUN|||cafés|||REQUIRED|||-NONE-|||0')
    block = m2.parse_block(raw)
    out = prepare_data._normalise_block(block)
    assert out.src == 'He said "hello" to the cafe owner .'
    assert len(out.src.split()) == len(block.src.split())
    assert out.edits[0].cor == "cafes"
    # The raw text must still parse as valid M2.
    reparsed = m2.parse_block(out.raw)
    assert reparsed.edits[0].type == "R:NOUN"


# --------------------------------------------------------------------------- #
# The alignment contract, on the real data
# --------------------------------------------------------------------------- #

@needs_wi
@pytest.mark.parametrize("split", SPLITS)
def test_split_files_are_aligned(split):
    files = paths.split_files(split)
    n = m2.assert_aligned(files["m2"], files["src"])
    m2.assert_same_length(files["src"], files["tgt"])
    assert len(paths.read_jsonl(files["jsonl"])) == n


@needs_wi
@pytest.mark.parametrize("split", SPLITS)
def test_no_source_line_is_blank(split):
    """ERRANT's parallel_to_m2 silently skips blank lines."""
    for i, line in enumerate(paths.read_lines(paths.split_files(split)["src"])):
        assert line.strip(), f"{split}: blank source at line {i + 1}"


@needs_wi
def test_split_sizes_match_the_spec():
    """Spec 5.1: ~34k train M2, 2,000 held out for val, ~4.4k dev as test."""
    counts = {s: paths.read_json(paths.split_files(s)["stats"])["n"] for s in SPLITS}
    assert counts["val"] == 2000
    assert counts["train_real"] + counts["val"] == 34308
    assert counts["test"] == 4384


@needs_wi
def test_val_and_train_real_are_disjoint():
    idx = paths.read_json(paths.PROCESSED / "split_index.json")
    assert idx["seed"] == 42
    assert idx["disjoint"] is True
    assert not (set(idx["val_idx"]) & set(idx["train_idx"]))
    assert len(idx["val_idx"]) == 2000


@needs_wi
def test_val_m2_is_a_verbatim_subset_of_the_training_m2():
    """The mechanism that makes val scores comparable with test scores.

    val.m2 must carry the ORIGINAL gold annotation lines, not edits re-derived
    by running ERRANT over (src, tgt).
    """
    idx = paths.read_json(paths.PROCESSED / "split_index.json")
    source = paths.RAW / "wi+locness" / "m2" / "ABC.train.gold.bea19.m2"
    if not source.exists():
        pytest.skip("raw W&I archive not present")

    original = [prepare_data._normalise_block(b) for b in m2.read_m2(source)]
    val = list(m2.read_m2(paths.split_files("val")["m2"]))
    assert len(val) == 2000
    for pos in (0, 1, 500, 1999):
        assert val[pos].raw == original[idx["val_idx"][pos]].raw


@needs_wi
def test_clean_share_is_plausible():
    """Spec 7 expects roughly a third of W&I sentences to need no edit.

    A share near 0 or near 1 means the M2 parsing is wrong, and the
    over-correction metric would be meaningless.
    """
    for split in SPLITS:
        share = paths.read_json(paths.split_files(split)["stats"])["clean_share"]
        assert 0.1 < share < 0.6, f"{split}: clean share {share}"


@needs_wi
def test_test_split_keeps_the_gold_file_order():
    """test must never be shuffled - scores stay comparable with published work."""
    source = paths.RAW / "wi+locness" / "m2" / "ABCN.dev.gold.bea19.m2"
    if not source.exists():
        pytest.skip("raw W&I archive not present")
    original = [b.src for b in m2.read_m2(source)][:50]
    ours = paths.read_lines(paths.split_files("test")["src"])[:50]
    assert [tok.normalise(s) for s in original] == ours


@needs_wi
def test_empty_targets_are_recorded_not_hidden():
    """A few W&I annotations delete the whole sentence."""
    for split in SPLITS:
        stats = paths.read_json(paths.split_files(split)["stats"])
        assert "n_empty_targets" in stats
        assert stats["n_empty_targets"] < 20, "should be a handful, not systemic"


# --------------------------------------------------------------------------- #
# The clean pool
# --------------------------------------------------------------------------- #

@needs_pool
def test_pool_stats_and_size():
    stats = paths.read_json(paths.PROCESSED / "clean_pool.stats.json")
    assert stats["n"] == paths.count_lines(paths.PROCESSED / "clean_pool.txt")
    assert stats["n"] >= prepare_data.POOL_BASE_SET
    assert stats["seed"] == 42
    # Version provenance, so a cache built by another spaCy cannot be mixed in.
    assert stats["spacy_version"] and stats["en_core_web_sm_version"]
    assert stats["rejected"].get("charset", 0) > 0, \
        "the charset filter must actually fire, or it is a no-op"


@needs_pool
def test_pool_rows_are_parallel_and_within_length_bounds():
    pool = prepare_data.load_pool(limit=500)
    assert len(pool) == 500
    for t in pool:
        assert len(t.toks) == len(t.upos) == len(t.tag) == len(t.lemma)
        assert prepare_data.POOL_MIN_TOKENS <= len(t.toks) <= prepare_data.POOL_MAX_TOKENS
        assert any(p in ("VERB", "AUX") for p in t.upos)


@needs_pool
def test_pool_text_matches_the_jsonl_cache():
    """clean_pool.txt and clean_pool.jsonl must be the same sentences in order."""
    lines = paths.read_lines(paths.PROCESSED / "clean_pool.txt")[:200]
    pool = prepare_data.load_pool(limit=200)
    assert [t.text for t in pool] == lines


@needs_pool
def test_pool_is_deduplicated():
    lines = paths.read_lines(paths.PROCESSED / "clean_pool.txt")
    lowered = [ln.lower() for ln in lines]
    assert len(set(lowered)) == len(lowered)


@needs_pool
def test_pool_has_almost_no_folding_damage():
    """The charset filter should keep "?" rare: real questions, not lost script."""
    lines = paths.read_lines(paths.PROCESSED / "clean_pool.txt")
    many = [ln for ln in lines if ln.count("?") >= 2]
    assert len(many) <= 5, f"{len(many)} sentences have >=2 '?' - charset filter leaking"
