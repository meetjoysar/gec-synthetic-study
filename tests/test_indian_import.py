"""Tests for importing the manually collected sentences and merging annotations.

Two properties matter most, because breaking either would quietly damage the
group's work:

* **A sentence's wording is never altered** beyond curly-quote straightening.
  Spec rule 2.3 reserves the text and the annotations for the group.
* **Merging never overwrites an existing annotation** without ``--force``, so
  re-running the command cannot destroy hours of human judgement.
"""

from __future__ import annotations

import pytest

from gec import indian_import as ii
from gec import kappa, paths

MANUAL = """# post: alpha
First sentence here is fine .
Second sentence has a error .
# post: beta [sensitive]
Third sentence under a sensitive label .
"""


@pytest.fixture
def manual(tmp_path):
    p = tmp_path / "manual_sentences.txt"
    paths.write_text(p, MANUAL)
    return p


@pytest.fixture
def tsv(tmp_path):
    """A TSV with one EXAMPLE row and only the six required columns."""
    p = tmp_path / "indian.tsv"
    paths.write_text(
        p,
        "id\tsource\tannotator1\tannotator2\tcategory\tnotes\n"
        "EXAMPLE-001\tHe go to school .\tHe goes to school .\t"
        "He goes to school .\tother\tdemo row\n",
    )
    return p


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def test_parse_groups_sentences_under_their_label(manual):
    labels = ii.parse_manual(manual)
    assert list(labels) == ["alpha", "beta [sensitive]"]
    assert len(labels["alpha"]) == 2
    assert len(labels["beta [sensitive]"]) == 1


def test_parse_ignores_the_begin_end_sentinels(tmp_path):
    p = tmp_path / "m.txt"
    paths.write_text(p, "BEGIN-SENTENCES\n# post: a\nOne sentence .\nEND-SENTENCES\n")
    labels = ii.parse_manual(p)
    assert labels == {"a": ["One sentence ."]}


def test_parse_rejects_a_sentence_before_any_label(tmp_path):
    p = tmp_path / "m.txt"
    paths.write_text(p, "Orphan sentence .\n# post: a\nOne .\n")
    with pytest.raises(ValueError, match="before any"):
        ii.parse_manual(p)


def test_parse_rejects_an_empty_file(tmp_path):
    """No labels and no sentences at all."""
    p = tmp_path / "m.txt"
    paths.write_text(p, "BEGIN-SENTENCES\n\nEND-SENTENCES\n")
    with pytest.raises(ValueError, match="no '# post:' labels"):
        ii.parse_manual(p)


def test_a_bare_line_reports_the_more_specific_error(tmp_path):
    """A stray sentence hits the orphan check first, which is more useful."""
    p = tmp_path / "m.txt"
    paths.write_text(p, "just a line\n")
    with pytest.raises(ValueError, match="before any"):
        ii.parse_manual(p)


def test_writer_ids_follow_first_appearance_order(manual):
    labels = ii.parse_manual(manual)
    assert ii.writer_ids(labels) == {"alpha": "W01", "beta [sensitive]": "W02"}


# --------------------------------------------------------------------------- #
# Normalisation is quote-straightening ONLY
# --------------------------------------------------------------------------- #

def test_normalise_straightens_quotes():
    assert ii.normalise_quotes("it’s “fine”") == "it's \"fine\""


def test_normalise_trims_only_the_outer_whitespace():
    assert ii.normalise_quotes("  He  go   to school .  ") == "He  go   to school ."


@pytest.mark.parametrize("text", [
    "He's a an arrogant red flag for world .",      # duplicated article
    "nowdays even if they can't",                    # typo
    "BUT, how to be safe in such an environment!??", # case + punctuation
    "I hv bn getting all bad news at once.",         # text-speak
    "Like it was our shared responsibility..",       # double period
    "word / slash and (bracket) stay",               # spacing
    "aunty did say sorry from her side",             # no final punctuation
])
def test_normalise_changes_nothing_else(text):
    """Typos, spacing, case, punctuation and brackets must all survive."""
    assert ii.normalise_quotes(text) == text


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #

def test_import_appends_rows_and_keeps_existing(manual, tsv):
    stats = ii.import_manual(manual, tsv)
    assert stats["n_imported"] == 3
    assert stats["n_existing_rows_kept"] == 1

    all_rows, _ = kappa.load_indian_tsv(tsv)
    assert len(all_rows) == 4
    assert list(all_rows["id"])[0] == "EXAMPLE-001"
    assert list(all_rows["id"])[1:] == ["IN-0001", "IN-0002", "IN-0003"]


def test_import_leaves_annotation_columns_empty(manual, tsv):
    ii.import_manual(manual, tsv)
    _, _ = kappa.load_indian_tsv(tsv)
    all_rows, ready = kappa.load_indian_tsv(tsv)
    new = all_rows[all_rows["id"].str.startswith("IN-")]
    assert (new["annotator1"] == "").all()
    assert (new["annotator2"] == "").all()
    assert (new["category"] == "").all()
    assert len(ready) == 0, "nothing is annotated yet"


def test_import_does_not_touch_the_example_row(manual, tsv):
    ii.import_manual(manual, tsv)
    all_rows, _ = kappa.load_indian_tsv(tsv)
    ex = all_rows[all_rows["id"] == "EXAMPLE-001"].iloc[0]
    assert ex["annotator1"] == "He goes to school ."
    assert ex["notes"] == "demo row"


def test_import_sets_provenance_columns(manual, tsv):
    ii.import_manual(manual, tsv)
    all_rows, _ = kappa.load_indian_tsv(tsv)
    new = all_rows[all_rows["id"].str.startswith("IN-")]
    assert (new["source_type"] == "reddit-manual").all()
    assert list(new["source_note"]) == [
        "W01 alpha", "W01 alpha", "W02 beta [sensitive]",
    ]


def test_import_marks_sensitive_rows(manual, tsv):
    ii.import_manual(manual, tsv)
    all_rows, _ = kappa.load_indian_tsv(tsv)
    sens = all_rows[all_rows["notes"] == ii.SENSITIVE_NOTE]
    assert len(sens) == 1
    assert sens.iloc[0]["source_note"].startswith("W02")
    # Non-sensitive rows keep an empty note.
    others = all_rows[all_rows["id"].isin(["IN-0001", "IN-0002"])]
    assert (others["notes"] == "").all()


def test_import_skips_exact_duplicates(tmp_path, tsv):
    p = tmp_path / "m.txt"
    paths.write_text(p, "# post: a\nSame sentence .\nSame sentence .\n"
                        "# post: b\nSAME SENTENCE .\n")
    stats = ii.import_manual(p, tsv)
    assert stats["n_imported"] == 1
    assert stats["n_duplicates_skipped"] == 2, "case-insensitive"


def test_import_is_idempotent(manual, tsv):
    """Re-running must not duplicate rows."""
    ii.import_manual(manual, tsv)
    second = ii.import_manual(manual, tsv)
    assert second["n_imported"] == 0
    assert second["n_duplicates_skipped"] == 3
    all_rows, _ = kappa.load_indian_tsv(tsv)
    assert len(all_rows) == 4


def test_import_continues_the_id_numbering(manual, tsv):
    ii.import_manual(manual, tsv)
    extra = tsv.parent / "more.txt"
    paths.write_text(extra, "# post: gamma\nA brand new sentence .\n")
    ii.import_manual(extra, tsv)
    all_rows, _ = kappa.load_indian_tsv(tsv)
    assert "IN-0004" in list(all_rows["id"])


def test_import_dry_run_writes_nothing(manual, tsv):
    before = paths.read_text(tsv)
    ii.import_manual(manual, tsv, dry_run=True)
    assert paths.read_text(tsv) == before


def test_import_round_trips_sources_exactly(tmp_path, tsv):
    """Quotes and odd spacing must survive the TSV write/read cycle."""
    tricky = [
        'She had "saved" or "marked" 3 places till now.',
        "He's a an arrogant red flag for world .",
        "word , with  odd   spacing !??",
        "a tab-free line with a (bracket) and a / slash",
    ]
    p = tmp_path / "m.txt"
    paths.write_text(p, "# post: a\n" + "\n".join(tricky) + "\n")
    ii.import_manual(p, tsv)
    all_rows, _ = kappa.load_indian_tsv(tsv)
    got = list(all_rows[all_rows["id"].str.startswith("IN-")]["source"])
    assert got == tricky


def test_validator_accepts_the_new_optional_columns(manual, tsv):
    ii.import_manual(manual, tsv)
    all_rows, _ = kappa.load_indian_tsv(tsv)
    problems = kappa.validate(all_rows)
    assert not [x for x in problems if "unexpected column" in x]
    assert not [x for x in problems if "missing required column" in x]
    # The pending rows are reported, as pending.
    assert any(x.startswith("PENDING") for x in problems)


def test_validator_still_requires_the_six_core_columns(tmp_path):
    p = tmp_path / "bad.tsv"
    paths.write_text(p, "id\tsource\n" + "IN-0001\tjust two columns\n")
    import pandas as pd

    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    problems = kappa.validate(df)
    assert any("missing required column" in x for x in problems)


# --------------------------------------------------------------------------- #
# Annotation files and merging
# --------------------------------------------------------------------------- #

def test_annotation_files_prefill_correction_with_the_source(manual, tsv, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA_MANUAL", tmp_path)
    monkeypatch.setattr(ii.paths, "DATA_MANUAL", tmp_path)
    ii.import_manual(manual, tsv)
    ii.make_annotation_files(tsv)

    for which in (1, 2):
        _, rows = ii.read_tsv_rows(ii.annotation_path(which))
        assert len(rows) == 3, "EXAMPLE rows are excluded"
        for r in rows:
            assert r["correction"] == r["source"], \
                "pre-filled so leaving a sentence unchanged is the default"


def test_merge_copies_corrections_into_the_main_tsv(manual, tsv, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA_MANUAL", tmp_path)
    monkeypatch.setattr(ii.paths, "DATA_MANUAL", tmp_path)
    ii.import_manual(manual, tsv)
    ii.make_annotation_files(tsv)

    # Annotator 1 fixes one sentence; annotator 2 leaves everything alone.
    path1 = ii.annotation_path(1)
    header, rows = ii.read_tsv_rows(path1)
    rows[1]["correction"] = "Second sentence has an error ."
    ii.write_tsv_rows(path1, header, rows)

    ii.merge_annotations(tsv)
    all_rows, ready = kappa.load_indian_tsv(tsv)
    r = all_rows[all_rows["id"] == "IN-0002"].iloc[0]
    assert r["annotator1"] == "Second sentence has an error ."
    assert r["annotator2"] == "Second sentence has a error ."
    assert len(ready) == 3, "all three rows now have both annotations"


def test_merge_never_overwrites_without_force(manual, tsv, monkeypatch, tmp_path):
    """The guard that stops a re-run destroying human judgement."""
    monkeypatch.setattr(paths, "DATA_MANUAL", tmp_path)
    monkeypatch.setattr(ii.paths, "DATA_MANUAL", tmp_path)
    ii.import_manual(manual, tsv)
    ii.make_annotation_files(tsv)
    ii.merge_annotations(tsv)

    # Someone edits the annotation file afterwards and re-merges.
    path1 = ii.annotation_path(1)
    header, rows = ii.read_tsv_rows(path1)
    rows[0]["correction"] = "A DIFFERENT correction ."
    ii.write_tsv_rows(path1, header, rows)

    summary = ii.merge_annotations(tsv)
    assert summary["refused_overwrite"] >= 1
    all_rows, _ = kappa.load_indian_tsv(tsv)
    kept = all_rows[all_rows["id"] == "IN-0001"].iloc[0]["annotator1"]
    assert kept == "First sentence here is fine .", "the original must survive"

    # --force lets it through.
    ii.merge_annotations(tsv, force=True)
    all_rows, _ = kappa.load_indian_tsv(tsv)
    assert all_rows[all_rows["id"] == "IN-0001"].iloc[0]["annotator1"] == \
        "A DIFFERENT correction ."


def test_merge_tolerates_missing_annotation_files(manual, tsv, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA_MANUAL", tmp_path)
    monkeypatch.setattr(ii.paths, "DATA_MANUAL", tmp_path)
    ii.import_manual(manual, tsv)
    summary = ii.merge_annotations(tsv)
    assert len(summary["missing_files"]) == 2
    assert summary["filled"] == 0


def test_merge_reports_unknown_ids(manual, tsv, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA_MANUAL", tmp_path)
    monkeypatch.setattr(ii.paths, "DATA_MANUAL", tmp_path)
    ii.import_manual(manual, tsv)
    ii.write_tsv_rows(ii.annotation_path(1), ii.ANNOTATION_COLUMNS,
                      [{"id": "IN-9999", "source": "x", "correction": "y"}])
    summary = ii.merge_annotations(tsv)
    assert "IN-9999" in summary["unknown_ids"]


def test_make_annotation_files_does_not_clobber_existing_work(manual, tsv, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA_MANUAL", tmp_path)
    monkeypatch.setattr(ii.paths, "DATA_MANUAL", tmp_path)
    ii.import_manual(manual, tsv)
    ii.make_annotation_files(tsv)

    path1 = ii.annotation_path(1)
    header, rows = ii.read_tsv_rows(path1)
    rows[0]["correction"] = "hours of careful work ."
    ii.write_tsv_rows(path1, header, rows)

    ii.make_annotation_files(tsv)          # no --force
    _, rows2 = ii.read_tsv_rows(path1)
    assert rows2[0]["correction"] == "hours of careful work ."


# --------------------------------------------------------------------------- #
# The real imported file
# --------------------------------------------------------------------------- #

def test_real_tsv_has_the_imported_sentences():
    tsv = paths.DATA_MANUAL / "indian_english_test.tsv"
    all_rows, ready = kappa.load_indian_tsv(tsv)
    imported = all_rows[all_rows["id"].str.startswith("IN-")]
    if len(imported) == 0:
        pytest.skip("sentences not imported in this checkout")
    assert len(imported) == 149
    assert (imported["source_type"] == "reddit-manual").all()
    # Every source must be ASCII, or T5 would map characters to <unk>.
    for s in imported["source"]:
        assert s.isascii(), f"non-ASCII survived import: {s[:60]!r}"
    # 20 writers, ids contiguous.
    writers = {n.split()[0] for n in imported["source_note"]}
    assert len(writers) == 20
    assert list(imported["id"]) == [f"IN-{i:04d}" for i in range(1, 150)]


def test_real_tsv_sensitive_rows_are_marked():
    tsv = paths.DATA_MANUAL / "indian_english_test.tsv"
    all_rows, _ = kappa.load_indian_tsv(tsv)
    imported = all_rows[all_rows["id"].str.startswith("IN-")]
    if len(imported) == 0:
        pytest.skip("sentences not imported in this checkout")
    # notes may also carry an appended annotator comment, so match on prefix.
    sens = imported[imported["notes"].str.startswith(ii.SENSITIVE_NOTE)]
    assert len(sens) == 28
    # Every sensitive row's label must actually carry the marker.
    assert all("[sensitive]" in n for n in sens["source_note"])
    # ...and no non-sensitive row is marked.
    others = imported[~imported["notes"].str.startswith(ii.SENSITIVE_NOTE)]
    assert not any("[sensitive]" in n for n in others["source_note"])
    # The sensitive warning must never be lost when a comment is appended.
    for n in sens["notes"]:
        assert n.startswith(ii.SENSITIVE_NOTE)


def test_demo_examples_cannot_leak_an_imported_sentence():
    """The Gradio demo must only ever show EXAMPLE- rows."""
    import yaml

    cfg = yaml.safe_load(paths.read_text(paths.APP / "config.yaml"))
    all_rows, _ = kappa.load_indian_tsv(
        paths.DATA_MANUAL / "indian_english_test.tsv"
    )
    real_sources = {
        s.strip() for s, i in zip(all_rows["source"], all_rows["id"])
        if not str(i).startswith("EXAMPLE-")
    }
    for ex in cfg["examples"]:
        assert ex.strip() not in real_sources, f"demo leaks an E5 item: {ex}"
