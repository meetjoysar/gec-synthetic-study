"""Tests for the ERRANT-category explanations (spec section 12)."""

from __future__ import annotations

import pytest

from gec import explain


#: The ten categories spec section 12 names explicitly.
SPEC_CATEGORIES = [
    "DET", "PREP", "VERB:SVA", "VERB:TENSE", "VERB:FORM",
    "NOUN:NUM", "SPELL", "PUNCT", "ORTH", "WO",
]


@pytest.mark.parametrize("cat", SPEC_CATEGORIES)
def test_spec_categories_never_fall_back(cat):
    """Each category spec 12 names must have its own reason, not the fallback."""
    ex = explain.explain(cat)
    assert ex.reason != explain.FALLBACK[1], f"{cat} fell back to the generic reason"
    assert ex.label and ex.reason
    assert ex.category == cat


@pytest.mark.parametrize("prefix,expected", [
    ("M", "missing word"),
    ("U", "unnecessary word"),
    ("R", "replaced word"),
])
def test_prefix_becomes_an_operation(prefix, expected):
    assert explain.explain(f"{prefix}:DET").operation == expected


def test_prefix_is_stripped_from_the_category():
    assert explain.explain("R:VERB:SVA").category == "VERB:SVA"
    assert explain.explain("M:DET").category == "DET"
    assert explain.explain("U:PREP").category == "PREP"


def test_prefixed_and_bare_types_give_the_same_reason():
    assert explain.explain("R:VERB:SVA").reason == explain.explain("VERB:SVA").reason


def test_split_type_passes_unprefixed_types_through():
    assert explain.split_type("UNK") == (None, "UNK")
    assert explain.split_type("VERB:SVA") == (None, "VERB:SVA")
    assert explain.split_type("R:DET") == ("R", "DET")
    # A two-character type cannot carry a prefix.
    assert explain.split_type("WO") == (None, "WO")


def test_unknown_category_falls_back_to_its_head():
    """`NOUN:WEIRD` should still be explained as a noun problem."""
    ex = explain.explain("R:NOUN:WEIRD")
    assert ex.label == explain.REASONS["NOUN"][0]


def test_completely_unknown_category_uses_the_fallback():
    ex = explain.explain("R:TOTALLY:MADE:UP")
    assert ex.reason == explain.FALLBACK[1]
    assert ex.operation == "replaced word"


def test_empty_and_none_do_not_crash():
    for bad in ("", "   "):
        assert explain.explain(bad).reason == explain.FALLBACK[1]


def test_one_line_renders_both_directions():
    ex = explain.explain("R:VERB:TENSE")
    line = ex.one_line("go", "went")
    assert "go -> went" in line and "Verb tense" in line
    # Insertions and deletions have an empty side.
    assert "(nothing)" in explain.explain("M:DET").one_line("", "the")


def test_no_reason_is_an_empty_string():
    for cat in explain.all_types():
        label, reason = explain.REASONS[cat]
        assert label.strip() and reason.strip(), cat


def test_module_has_no_heavy_imports():
    """explain.py must stay importable without torch/spacy/errant."""
    import sys

    mod = sys.modules["gec.explain"]
    src = open(mod.__file__, encoding="utf-8").read()
    for heavy in ("import torch", "import spacy", "import errant", "import transformers"):
        assert heavy not in src, f"explain.py should not {heavy}"
