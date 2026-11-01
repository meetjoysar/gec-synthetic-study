"""Tests for the Gradio demo's logic (spec sections 10 and 12).

The demo must be useful *before* the experiments finish, so the important
property here is that a missing checkpoint degrades gracefully instead of
crashing the app.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import app as demo  # noqa: E402  (app/app.py)


@pytest.fixture
def cfg() -> dict:
    return {
        "models": {
            "Synthetic-only": "results/runs/e3_synthetic/seed13/model",
            "Real-only": "results/runs/e3_real/seed13/model",
            "Mixed": "results/runs/e3_mixed/seed13/model",
        },
        "device": "cpu",
        "num_beams": 1,
        "examples": ["He go to school ."],
    }


def test_load_correctors_builds_all_three_columns(cfg):
    cs = demo.load_correctors(cfg)
    assert set(cs) == set(demo.COLUMNS)


def test_missing_checkpoint_is_not_fatal(cfg):
    """Spec Phase 10: the demo must fall back gracefully."""
    cfg = {**cfg, "models": {"Synthetic-only": "results/runs/does_not_exist/seed13/model"}}
    c = demo.load_correctors(cfg)["Synthetic-only"]
    assert not c.available
    r = c.correct("He go to school .")
    assert r.error is not None
    assert "Not trained yet" in r.error
    # The message must tell the user how to produce it.
    assert "run_experiments" in r.error


def test_unconfigured_column_is_not_fatal():
    c = demo.load_correctors({"models": {}})["Mixed"]
    assert not c.available
    assert c.correct("He go .").error is not None


def test_render_column_handles_an_error_result():
    r = demo.Result(error="**Not trained yet.**")
    html_block, md = demo.render_column("Mixed", r)
    assert "Not trained yet" in html_block
    assert md == ""


def test_render_column_lists_edits():
    r = demo.Result(
        corrected_tok="He goes to school .",
        corrected_display="He goes to school.",
        edits=[
            demo.EditView("go", "goes", "R:VERB:SVA", "Subject-verb agreement",
                          "Subject-verb agreement: the verb must match its subject.",
                          "replaced word")
        ],
        html="<div>x</div>",
        ms=12.3,
    )
    _, md = demo.render_column("Mixed", r)
    assert "1 edit(s)" in md
    assert "`go` -> `goes`" in md
    assert "Subject-verb agreement" in md
    assert "R:VERB:SVA" in md


def test_render_html_escapes_user_input():
    """A user typing <script> must not be able to inject into the page."""
    edits = [demo.EditView("<script>alert(1)</script>", "safe", "R:OTHER",
                           "Other", "Other change.", "replaced word")]
    out = demo.render_html("<script>alert(1)</script> x", "safe x", edits)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_render_html_with_no_edits_says_so():
    out = demo.render_html("He goes to school .", "He goes to school .", [])
    assert "No changes suggested" in out


def test_render_html_marks_insertions_and_deletions():
    ins = demo.render_html("He goes school .", "He goes to school .",
                           [demo.EditView("", "to", "M:PREP", "Preposition",
                                          "Preposition.", "missing word")])
    assert "+to" in ins
    dele = demo.render_html("He goes to to school .", "He goes to school .",
                            [demo.EditView("to", "", "U:PREP", "Preposition",
                                           "Preposition.", "unnecessary word")])
    assert "line-through" in dele


@pytest.mark.needs_errant
def test_annotate_pair_finds_a_real_edit():
    """End-to-end through the real ERRANT annotator."""
    views = demo.annotate_pair("He go to school .", "He goes to school .")
    assert len(views) == 1
    v = views[0]
    assert (v.before, v.after) == ("go", "goes")
    assert v.errant_type == "R:VERB:SVA"
    assert v.label == "Subject-verb agreement"
    assert v.operation == "replaced word"


@pytest.mark.needs_errant
def test_annotate_pair_returns_nothing_for_an_unchanged_sentence():
    assert demo.annotate_pair("He goes to school .", "He goes to school .") == []


def test_config_yaml_is_loadable_and_complete():
    """The shipped app/config.yaml must name all three columns and 8 examples."""
    import yaml

    cfg = yaml.safe_load((ROOT / "app" / "config.yaml").read_text(encoding="utf-8"))
    assert set(cfg["models"]) == set(demo.COLUMNS)
    assert len(cfg["examples"]) >= 6, "spec 12 asks for 6-8 example buttons"


def test_examples_do_not_leak_the_real_test_set():
    """Demo examples may only come from the EXAMPLE- rows, never real items.

    Spec rule 2.3 reserves the Indian English sentences for the group; using a
    real row as a demo button would put evaluation data in the UI.
    """
    import csv

    import yaml

    cfg = yaml.safe_load((ROOT / "app" / "config.yaml").read_text(encoding="utf-8"))
    tsv = ROOT / "data_manual" / "indian_english_test.tsv"
    with open(tsv, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    real_sources = {
        r["source"].strip() for r in rows
        if not str(r["id"]).startswith("EXAMPLE-")
    }
    for ex in cfg["examples"]:
        assert ex.strip() not in real_sources, f"demo example leaks a real test item: {ex}"
