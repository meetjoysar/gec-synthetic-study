"""Tests for tokenisation, detokenisation and ASCII normalisation."""

from __future__ import annotations

import pytest

from gec import tok


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

def test_tokenise_splits_contractions_and_punctuation():
    """Spec 5.3: output must match the M2 convention (`do n't`, `school .`)."""
    assert tok.tokenise("He doesn't go.") == "He does n't go ."
    assert tok.tokenise("It's a test.") == "It 's a test ."
    assert tok.tokenise("I go to school.") == "I go to school ."


def test_tokenise_never_produces_blank_or_double_space():
    for text in ["  spaced   out  ", "\ttabbed\t", "a", "."]:
        out = tok.tokenise(text)
        assert out == out.strip()
        assert "  " not in out
        if text.strip():
            assert out, f"non-blank input {text!r} must not tokenise to blank"


def test_blank_tokenizer_agrees_with_full_model():
    """`tokenise` uses spacy.blank('en') for speed; it must match the real model.

    If this ever fails, the clean pool's cached POS tags would no longer line up
    with the tokens the rest of the pipeline sees.
    """
    sentences = [
        "He doesn't want to go, does he?",
        "The cat's toys aren't here.",
        "I've been waiting since 2019 (roughly).",
        "She said: \"That's it!\"",
        "U.S.A. and Dr. Smith arrived at 3 p.m.",
        "well-known but ill-advised",
        "It costs $4.50 or 30%.",
    ]
    blank = [tok.tokenise(s) for s in sentences]
    full = [t.text for t in tok.tag_batch(sentences)]
    assert blank == full


def test_tag_batch_returns_equal_length_parallel_lists():
    tagged = list(tok.tag_batch(["The cat is sitting on the mat ."]))
    assert len(tagged) == 1
    t = tagged[0]
    assert len(t.toks) == len(t.upos) == len(t.tag) == len(t.lemma)
    assert t.toks == "The cat is sitting on the mat .".split()


def test_tag_batch_marks_copula_as_aux():
    """The clean-pool filter must accept AUX, not only VERB.

    spaCy tags `is`/`has`/`can` as AUX. A VERB-only verb check would discard
    most copula sentences - exactly the sentences the `sva` operation needs.
    """
    t = tok.tag_one("The result is clear .")
    assert "AUX" in t.upos
    assert "VERB" not in t.upos, "this sentence has no lexical verb at all"


def test_tag_batch_ptb_tags_present():
    """lemminflect needs PTB tags (VBZ/VBD/NN/NNS), not UPOS."""
    t = tok.tag_one("He walks to the stores .")
    assert "VBZ" in t.tag
    assert "NNS" in t.tag


# --------------------------------------------------------------------------- #
# Normalisation - the T5 <unk> guard
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw",
    [
        "He said “hello” to me .",          # curly double quotes
        "It ’s John ’s book .",              # curly apostrophes
        "A dash – and an em dash — here .",  # en/em dashes
        "Café résumé naïve .",     # accents
        "Non breaking space .",                   # NBSP
        "Ellipsis … done .",                      # horizontal ellipsis
    ],
)
def test_normalise_preserves_token_count(raw):
    """One character in, one character out - token boundaries must not move.

    This is what makes it safe to normalise source AND target: ERRANT's
    `simplify_edits` reads only the A lines, so S-line text is irrelevant to
    scoring, but the token COUNT must stay fixed or every edit span shifts.
    """
    out = tok.normalise(raw)
    assert len(out.split()) == len(raw.split())


def test_normalise_output_is_t5_representable():
    raw = "He said “hello” – café …"
    out = tok.normalise(raw)
    assert out.isascii(), f"normalise left non-ASCII: {out!r}"
    for banned in "{}<>^~\\`":
        assert banned not in out


def test_normalise_specific_mappings():
    assert tok.normalise("“quoted”") == '"quoted"'
    assert tok.normalise("don’t") == "don't"
    assert tok.normalise("a–b") == "a-b"
    assert tok.normalise("café") == "cafe"


def test_normalise_is_idempotent():
    raw = "He said “hello” – café résumé …"
    once = tok.normalise(raw)
    assert tok.normalise(once) == once


def test_normalise_leaves_plain_ascii_untouched():
    plain = "He goes to school every day , does n't he ?"
    assert tok.normalise(plain) == plain


def test_normalise_records_substitutions():
    tok.NORM_STATS.clear()
    tok.normalise("“a”")
    report = tok.norm_report()
    assert sum(report.values()) == 2
    assert any("->" in k for k in report)


# --------------------------------------------------------------------------- #
# Detokenisation (display only)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "tokenised,expected",
    [
        ("He does n't go .", "He doesn't go."),
        ("It 's a test .", "It's a test."),
        ("I go to school .", "I go to school."),
        ("Hello , world !", "Hello, world!"),
        ("She said ( quietly ) yes .", "She said (quietly) yes."),
        ("We 've been here .", "We've been here."),
        ("Is it 30 % ?", "Is it 30%?"),
        ("They 're happy ; we are not .", "They're happy; we are not."),
    ],
)
def test_detokenise_cases(tokenised, expected):
    assert tok.detokenise(tokenised) == expected


def test_detokenise_round_trips_plain_sentences():
    originals = [
        "He goes to school.",
        "I have two brothers.",
        "We discussed the plan.",
        "She has been working here since 2019.",
        "The cat sat on the mat.",
    ]
    for original in originals:
        assert tok.detokenise(tok.tokenise(original)) == original


def test_detokenise_accepts_list_and_handles_empty():
    assert tok.detokenise(["He", "goes", "."]) == "He goes."
    assert tok.detokenise("") == ""
    assert tok.detokenise([]) == ""
