"""Tests for M2 parsing and edit application (spec section 5.2).

The spec names six cases explicitly - insertion, deletion, replacement,
multiple edits, noop and UNK - and each gets a test here. The ``raw``
round-trip test is the important one: it is what guarantees the sliced ``val``
reference M2 carries the original gold annotations.
"""

from __future__ import annotations

import pytest

from gec import m2, paths


# --------------------------------------------------------------------------- #
# Edit parsing
# --------------------------------------------------------------------------- #

def test_parse_edit_replacement():
    e = m2.parse_edit("A 1 2|||R:VERB:SVA|||is|||REQUIRED|||-NONE-|||0")
    assert (e.start, e.end) == (1, 2)
    assert e.type == "R:VERB:SVA"
    assert e.cor_tokens == ["is"]
    assert e.cat2 == "VERB:SVA"
    assert e.prefix == "R"
    assert not e.is_noop and not e.is_skipped


def test_parse_edit_insertion_and_deletion():
    ins = m2.parse_edit("A 4 4|||M:DET|||the|||REQUIRED|||-NONE-|||0")
    assert ins.is_insertion and not ins.is_deletion
    assert ins.cat2 == "DET" and ins.prefix == "M"

    # ERRANT renders a deletion with an empty correction field, which is six
    # consecutive pipes: "|||".join(["A 2 3", "U:ADV", "", "REQUIRED", ...]).
    dele = m2.parse_edit("A 2 3|||U:ADV||||||REQUIRED|||-NONE-|||0")
    assert dele.type == "U:ADV" and dele.cat2 == "ADV"
    assert dele.cor_tokens == []
    assert dele.annotator == 0
    assert dele.is_deletion and not dele.is_insertion


def test_parse_edit_noop_and_unk_are_skipped():
    noop = m2.parse_edit("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
    assert noop.is_noop and noop.is_skipped
    assert noop.cat2 == "noop"

    unk = m2.parse_edit("A 1 3|||UNK|||-NONE-|||REQUIRED|||-NONE-|||0")
    assert not unk.is_noop
    assert unk.is_skipped, "spec 5.2: treat UNK as no change"
    assert unk.cat2 == "UNK", "errant -cat 2 passes UNK through verbatim"


def test_parse_edit_annotator_id():
    assert m2.parse_edit("A 0 1|||R:ORTH|||The|||REQUIRED|||-NONE-|||2").annotator == 2
    # Short lines (no annotator field) default to annotator 0.
    assert m2.parse_edit("A 0 1|||R:ORTH|||The").annotator == 0


def test_parse_edit_malformed_raises():
    with pytest.raises(ValueError):
        m2.parse_edit("A 1 2|||R:VERB:SVA")
    with pytest.raises(ValueError):
        m2.parse_edit("A one two|||R:ORTH|||x|||REQUIRED|||-NONE-|||0")


def test_cat2_matches_errant_level_2():
    """``cat2`` must mirror ERRANT's ``processCategories(cat, 2)`` = ``cat[2:]``."""
    cases = {
        "R:VERB:SVA": "VERB:SVA",
        "M:DET": "DET",
        "U:PREP": "PREP",
        "R:NOUN:NUM": "NOUN:NUM",
        "R:ORTH": "ORTH",
        "UNK": "UNK",
    }
    for full, expected in cases.items():
        assert m2.parse_edit(f"A 0 1|||{full}|||x|||REQUIRED|||-NONE-|||0").cat2 == expected


# --------------------------------------------------------------------------- #
# Applying edits
# --------------------------------------------------------------------------- #

def test_apply_replacement():
    toks = "This are a sentence .".split()
    edits = [m2.Edit(1, 2, "R:VERB:SVA", "is")]
    assert m2.apply_edits(toks, edits) == "This is a sentence .".split()


def test_apply_insertion():
    toks = "I go school .".split()
    edits = [m2.Edit(2, 2, "M:DET", "to the")]
    assert m2.apply_edits(toks, edits) == "I go to the school .".split()


def test_apply_insertion_at_start_and_end():
    toks = "cat sat".split()
    assert m2.apply_edits(toks, [m2.Edit(0, 0, "M:DET", "The")]) == ["The", "cat", "sat"]
    assert m2.apply_edits(toks, [m2.Edit(2, 2, "M:PUNCT", ".")]) == ["cat", "sat", "."]


def test_apply_deletion():
    toks = "She is very very happy .".split()
    edits = [m2.Edit(2, 3, "U:ADV", "")]
    assert m2.apply_edits(toks, edits) == "She is very happy .".split()


def test_apply_multiple_edits_right_to_left():
    """Multiple edits must not disturb each other's indices."""
    toks = "I go to school yesterday".split()
    edits = [
        m2.Edit(1, 2, "R:VERB:TENSE", "went"),
        m2.Edit(3, 4, "R:NOUN", "college"),
        m2.Edit(5, 5, "M:PUNCT", "."),
    ]
    assert m2.apply_edits(toks, edits) == "I went to college yesterday .".split()


def test_apply_edits_skips_noop_and_unk():
    toks = "This sentence is fine .".split()
    edits = [
        m2.Edit(-1, -1, "noop", "-NONE-"),
        m2.Edit(1, 2, "UNK", "-NONE-"),
    ]
    assert m2.apply_edits(toks, edits) == toks


def test_apply_edits_empty_returns_copy():
    toks = ["a", "b"]
    out = m2.apply_edits(toks, [])
    assert out == toks and out is not toks


def test_overlapping_edits_keep_leftmost_and_warn():
    m2.OVERLAP_WARNINGS.clear()
    toks = "I have been going there".split()
    edits = [
        m2.Edit(1, 3, "R:VERB:TENSE", "went"),
        m2.Edit(2, 4, "R:VERB:FORM", "was going"),  # overlaps the first
    ]
    out = m2.apply_edits(toks, edits)
    assert out == "I went going there".split()
    assert sum(m2.OVERLAP_WARNINGS.values()) == 1


# --------------------------------------------------------------------------- #
# Blocks and files
# --------------------------------------------------------------------------- #

def test_read_m2_block_count_and_targets(m2_fixture_path):
    blocks = list(m2.read_m2(m2_fixture_path))
    assert len(blocks) == 5

    assert blocks[0].target() == "This is a sentence ."
    assert blocks[1].target() == "I went to school the yesterday ."
    assert blocks[2].target() == "She is very happy ."
    assert blocks[3].target() == "This sentence is already correct ."
    assert blocks[4].target() == "Something odd happens here .", "UNK = no change"


def test_is_clean_identifies_noop_and_unk_blocks(m2_fixture_path):
    blocks = list(m2.read_m2(m2_fixture_path))
    assert [b.is_clean() for b in blocks] == [False, False, False, True, True]


def test_n_edits_excludes_skipped(m2_fixture_path):
    blocks = list(m2.read_m2(m2_fixture_path))
    assert [b.n_edits() for b in blocks] == [1, 2, 1, 0, 0]


def test_raw_round_trip_is_byte_identical(m2_fixture_path, tmp_path):
    """A sliced/reordered subset must preserve the ORIGINAL annotation lines.

    This is the mechanism the ``val`` reference M2 depends on: re-deriving edits
    by running ERRANT over (src, tgt) would differ in merge and classification
    decisions, making ``val`` scores incomparable with ``test`` scores.
    """
    blocks = list(m2.read_m2(m2_fixture_path))
    out = tmp_path / "subset.m2"

    # Reorder and slice, exactly as prepare_data does for val/train_real.
    subset = [blocks[3], blocks[0], blocks[4]]
    assert m2.write_m2(subset, out) == 3

    reread = list(m2.read_m2(out))
    assert [b.raw for b in reread] == [b.raw for b in subset]
    assert [b.src for b in reread] == [b.src for b in subset]
    assert [e.to_m2() for e in reread[1].edits] == [e.to_m2() for e in subset[1].edits]


def test_write_m2_synthesises_noop_when_no_raw(tmp_path):
    out = tmp_path / "synth.m2"
    m2.write_m2([m2.M2Block(src="A clean sentence .", edits=[])], out)
    text = paths.read_text(out)
    assert "S A clean sentence ." in text
    assert "noop" in text, "a block with no edits needs an explicit noop line"


def test_multi_annotator_targets(multi_annotator_m2):
    blocks = list(m2.read_m2(multi_annotator_m2))
    assert blocks[0].annotators() == [0, 1]
    targets = blocks[0].targets()
    assert targets[0] == "We discussed the plan ."
    assert targets[1] == "We talked about the plan ."
    # Annotator 1 left the second sentence alone.
    assert blocks[1].is_clean(annotator=1)
    assert not blocks[1].is_clean(annotator=0)


def test_count_types_level_2_and_3(m2_fixture_path):
    lvl2 = m2.count_types(m2_fixture_path, level=2)
    assert dict(lvl2) == {"VERB:SVA": 1, "VERB:TENSE": 1, "DET": 1, "ADV": 1}
    assert "UNK" not in lvl2, "skipped edits are not counted"
    assert "noop" not in lvl2

    lvl3 = m2.count_types(m2_fixture_path, level=3)
    assert lvl3["R:VERB:SVA"] == 1
    assert lvl3["M:DET"] == 1


# --------------------------------------------------------------------------- #
# Alignment invariants
# --------------------------------------------------------------------------- #

def test_assert_aligned_passes_on_matching_files(m2_fixture_path, tmp_path):
    srcs = [b.src for b in m2.read_m2(m2_fixture_path)]
    src_file = tmp_path / "src.txt"
    paths.write_lines(src_file, srcs)
    assert m2.assert_aligned(m2_fixture_path, src_file) == 5


def test_assert_aligned_reports_count_mismatch(m2_fixture_path, tmp_path):
    srcs = [b.src for b in m2.read_m2(m2_fixture_path)][:3]
    src_file = tmp_path / "short.txt"
    paths.write_lines(src_file, srcs)
    with pytest.raises(AssertionError, match="length mismatch"):
        m2.assert_aligned(m2_fixture_path, src_file)


def test_assert_aligned_reports_text_mismatch(m2_fixture_path, tmp_path):
    srcs = [b.src for b in m2.read_m2(m2_fixture_path)]
    srcs[2] = "A completely different sentence ."
    src_file = tmp_path / "wrong.txt"
    paths.write_lines(src_file, srcs)
    with pytest.raises(AssertionError, match="text mismatch at sentence 2"):
        m2.assert_aligned(m2_fixture_path, src_file)


# --------------------------------------------------------------------------- #
# paths.write_lines blank-line guard
# --------------------------------------------------------------------------- #

def test_write_lines_rejects_blank_lines(tmp_path):
    """ERRANT's parallel_to_m2 does `if not orig: continue` - blanks vanish."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        paths.write_lines(tmp_path / "bad.txt", ["ok", "", "also ok"])
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        paths.write_lines(tmp_path / "bad2.txt", ["ok", "   "])


def test_write_lines_rejects_embedded_newline(tmp_path):
    with pytest.raises(ValueError, match="embedded newline"):
        paths.write_lines(tmp_path / "bad3.txt", ["ok", "two\nlines"])


def test_write_lines_round_trip(tmp_path):
    p = tmp_path / "good.txt"
    lines = ["first line", "second line", "third ."]
    assert paths.write_lines(p, lines) == 3
    assert paths.read_lines(p) == lines
    # LF only, even on Windows.
    assert b"\r\n" not in p.read_bytes()
