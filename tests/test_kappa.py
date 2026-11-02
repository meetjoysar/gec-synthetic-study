"""Tests for the Indian English validator and annotator agreement (spec 5.5).

The pending-annotation behaviour is the important part. The group's workflow is
to write every ``source`` first and then have two annotators fill their columns
*independently*, so the file legitimately spends time half-finished. That state
must read as progress, not as corruption - and it must not let an empty
annotation reach ``errant_parallel``, which would drop the block and
desynchronise the M2 files.
"""

from __future__ import annotations

import pytest

from gec import kappa, paths

HEADER = "id\tsource\tannotator1\tannotator2\tcategory\tnotes"


def write_tsv(tmp_path, rows: list[str]):
    p = tmp_path / "indian.tsv"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(HEADER + "\n")
        for r in rows:
            fh.write(r + "\n")
    return p


def row(rid, src, a1="", a2="", cat="other", note=""):
    return "\t".join([rid, src, a1, a2, cat, note])


# --------------------------------------------------------------------------- #
# Pending annotations are progress, not errors
# --------------------------------------------------------------------------- #

def test_pending_rows_are_not_errors(tmp_path):
    p = write_tsv(tmp_path, [
        row("IN-0001", "He is working here since 2019 ."),
        row("IN-0002", "We discussed about the plan .",
            "We discussed the plan .", "We discussed the plan ."),
    ])
    all_rows, ready = kappa.load_indian_tsv(p)
    assert len(ready) == 1, "only fully annotated rows are usable"

    problems = kappa.validate(all_rows)
    pending = [x for x in problems if x.startswith("PENDING")]
    assert len(pending) == 1
    assert "IN-0001" in pending[0]
    # Nothing else was flagged as a genuine problem.
    assert not [x for x in problems if not x.startswith(("PENDING", "WARNING"))]


def test_pending_rows_do_not_count_toward_the_threshold(tmp_path):
    """30 sources with no annotations must NOT unblock E5."""
    rows = [row(f"IN-{i:04d}", f"This is candidate sentence number {i} here .")
            for i in range(1, 31)]
    p = write_tsv(tmp_path, rows)
    n_ready, n_pending = kappa.pending_count(p)
    assert (n_ready, n_pending) == (0, 30)
    assert n_ready < kappa.MIN_REAL_ROWS


def test_half_annotated_row_is_still_pending(tmp_path):
    """One annotator done is not enough - kappa needs both."""
    p = write_tsv(tmp_path, [
        row("IN-0001", "I am having two brothers .", "I have two brothers .", ""),
    ])
    _, ready = kappa.load_indian_tsv(p)
    assert len(ready) == 0
    assert kappa.pending_count(p) == (0, 1)


def test_fully_annotated_rows_unblock_the_threshold(tmp_path):
    rows = [row(f"IN-{i:04d}", f"Sentence number {i} is written here .",
                f"Sentence number {i} is written here .",
                f"Sentence number {i} is written here .")
            for i in range(1, 21)]
    p = write_tsv(tmp_path, rows)
    n_ready, n_pending = kappa.pending_count(p)
    assert (n_ready, n_pending) == (20, 0)
    assert n_ready >= kappa.MIN_REAL_ROWS


def test_strict_mode_tolerates_pending(tmp_path):
    p = write_tsv(tmp_path, [row("IN-0001", "A sentence awaiting work .")])
    all_rows, _ = kappa.load_indian_tsv(p)
    assert all(x.startswith(("PENDING", "WARNING")) for x in kappa.validate(all_rows))


def test_empty_source_is_still_an_error(tmp_path):
    p = write_tsv(tmp_path, [row("IN-0001", "   ", "x", "y")])
    all_rows, _ = kappa.load_indian_tsv(p)
    assert any("empty 'source'" in x for x in kappa.validate(all_rows))


def test_incomplete_example_row_is_an_error(tmp_path):
    """EXAMPLE- rows ship complete, so a gap there IS a defect."""
    p = write_tsv(tmp_path, [row("EXAMPLE-001", "He go to school .", "He goes .", "")])
    all_rows, _ = kappa.load_indian_tsv(p)
    assert any("EXAMPLE- row(s) have an empty" in x for x in kappa.validate(all_rows))


# --------------------------------------------------------------------------- #
# Other validation
# --------------------------------------------------------------------------- #

def test_duplicate_ids_are_caught(tmp_path):
    p = write_tsv(tmp_path, [
        row("IN-0001", "First sentence .", "a", "b"),
        row("IN-0001", "Second sentence .", "c", "d"),
    ])
    all_rows, _ = kappa.load_indian_tsv(p)
    assert any("duplicate id" in x for x in kappa.validate(all_rows))


def test_unknown_category_warns_but_does_not_fail(tmp_path):
    p = write_tsv(tmp_path, [row("IN-0001", "A sentence .", "a", "b", cat="nonsense")])
    all_rows, _ = kappa.load_indian_tsv(p)
    problems = kappa.validate(all_rows)
    assert any(x.startswith("WARNING") and "nonsense" in x for x in problems)


def test_literal_na_is_not_turned_into_nan(tmp_path):
    """keep_default_na=False: a sentence that IS the word "NA" must survive."""
    p = write_tsv(tmp_path, [row("IN-0001", "NA", "NA", "NA")])
    _, ready = kappa.load_indian_tsv(p)
    assert len(ready) == 1 and ready.iloc[0]["source"] == "NA"


# --------------------------------------------------------------------------- #
# The shipped template
# --------------------------------------------------------------------------- #

def test_shipped_tsv_is_valid():
    """The committed TSV must validate, whatever stage of annotation it is at."""
    tsv = paths.DATA_MANUAL / "indian_english_test.tsv"
    all_rows, ready = kappa.load_indian_tsv(tsv)

    examples = all_rows[all_rows["id"].str.startswith(kappa.EXAMPLE_PREFIX)]
    assert len(examples) <= 5, "spec 2.3 allows at most 5 EXAMPLE- rows"
    assert len(examples) > 0, "the format demonstration rows must survive"
    # Every EXAMPLE row ships complete.
    assert (examples["annotator1"] != "").all()
    assert (examples["annotator2"] != "").all()

    # Real rows are either fully annotated or pending; never half-done in a way
    # that would reach errant_parallel.
    n_ready, n_pending = kappa.pending_count(tsv)
    assert len(ready) == n_ready

    problems = [x for x in kappa.validate(all_rows)
                if not x.startswith(("WARNING", "PENDING"))]
    assert problems == [], problems


def test_report_tracks_gold_availability(tmp_path):
    """E5 needs >= 20 rows with a GOLD reference (annotator1 filled).

    Kappa additionally needs annotator2; the report says which of the two
    situations it is in rather than conflating them.
    """
    out = tmp_path / "kappa.json"
    tsv = paths.DATA_MANUAL / "indian_english_test.tsv"
    n_gold, n_no_gold = kappa.pending_count(tsv, require_both=False)
    payload = kappa.report(tsv, out)

    assert payload["n_real_rows"] == n_gold
    assert payload["n_pending_annotation"] == n_no_gold

    if n_gold < kappa.MIN_REAL_ROWS:
        assert payload["skipped"] is True
    else:
        assert payload["skipped"] is False
        cols = kappa.annotator_columns(kappa.load_indian_tsv(tsv)[0])
        if len(cols) < 2:
            # Single reference: scoring is valid, kappa is undefined.
            assert payload["single_annotator"] is True
            assert payload["token_kappa"] is None
            assert "INDEPENDENT" in payload["note"]
        else:
            assert payload.get("token_kappa") is not None or                 payload.get("token_note")


def test_report_counts_pending_rows_in_its_payload(tmp_path):
    p = write_tsv(tmp_path, [
        row("EXAMPLE-001", "He go .", "He goes .", "He goes ."),
        row("IN-0001", "Awaiting annotation here ."),
        row("IN-0002", "Also awaiting annotation ."),
    ])
    payload = kappa.report(p, tmp_path / "k.json")
    assert payload["skipped"] is True
    assert payload["n_real_rows"] == 0
    assert payload["n_pending_annotation"] == 2
    assert payload["n_example_rows"] == 1


# --------------------------------------------------------------------------- #
# Kappa itself
# --------------------------------------------------------------------------- #

def test_kappa_is_none_when_one_annotator_made_no_edits():
    """Guarded explicitly: a bare nan in a report is worse than an explanation."""
    out = kappa._kappa([1, 1, 0, 0], [0, 0, 0, 0], "token")
    assert out["token_kappa"] is None
    assert "no variance" in out["token_note"]
    assert out["token_raw_agreement"] is not None, "still meaningful"


def test_kappa_is_one_for_perfect_agreement():
    out = kappa._kappa([1, 0, 1, 0], [1, 0, 1, 0], "token")
    assert out["token_kappa"] == 1.0
    assert out["token_disagree"] == 0


def test_kappa_counts_cells():
    out = kappa._kappa([1, 1, 0, 0], [1, 0, 1, 0], "token")
    assert out["token_both_1"] == 1
    assert out["token_both_0"] == 1
    assert out["token_disagree"] == 2
    assert out["token_raw_agreement"] == 0.5


@pytest.mark.needs_errant
def test_edit_token_labels_marks_the_changed_token():
    assert kappa.edit_token_labels("He go to school .", "He goes to school .") == \
        [0, 1, 0, 0, 0]


@pytest.mark.needs_errant
def test_edit_token_labels_marks_a_token_for_an_insertion():
    """Spec 5.5: an insertion marks the token after the insertion point."""
    labels = kappa.edit_token_labels("He goes school .", "He goes to school .")
    assert len(labels) == 4
    assert sum(labels) >= 1


@pytest.mark.needs_errant
def test_edit_token_labels_all_zero_for_an_unchanged_sentence():
    assert kappa.edit_token_labels("He goes to school .", "He goes to school .") == \
        [0, 0, 0, 0, 0]
