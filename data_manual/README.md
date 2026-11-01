# Indian English test set — instructions for the group

This is the one part of the project **we** write by hand, and that is
deliberate: it keeps E5 a genuinely human-made evaluation set. Claude Code built
the template, the validator and the agreement tooling, and wrote the five
`EXAMPLE-` rows that show the format — nothing else. The 150 real sentences are
ours to write.

**Target: ~150 sentences.** Below 20 real rows, E5 is skipped automatically and
shows as `TBD` in the report, so nothing breaks while the file is being filled
in.

---

## The file

`indian_english_test.tsv` — **tab**-separated, UTF-8, one row per line:

| column | what goes in it |
|---|---|
| `id` | A unique id, e.g. `IE-001`. **Do not** start a real row with `EXAMPLE-` — that prefix means "excluded from evaluation". |
| `source` | A sentence containing an error common among Indian English speakers. Written by a group member. |
| `annotator1` | Annotator 1's corrected version, written **independently**. |
| `annotator2` | Annotator 2's corrected version, written **independently**. |
| `category` | One of the categories below. |
| `notes` | Optional: why it is an error, or why you disagreed. |

### Categories

`tense-perfect`, `stative-progressive`, `redundancy`, `article`, `preposition`,
`question-form`, `uncountable-plural`, `dialect-usage`, `agreement`,
`word-order`, `spelling`, `other`.

An unrecognised value produces a warning, not an error — if you find a pattern
that genuinely does not fit, use it and tell the others.

---

## How to write a good item

1. **One clear error per sentence**, so the error type is unambiguous. Two
   errors in one sentence makes the per-category F0.5 in the report harder to
   read.
2. **Write sentences you have actually heard or read**, not inventions. The
   point of E5 is to test models trained on standard learner data (W&I is
   mostly European learners) against a different error distribution.
3. **Keep them 8–30 words.** Long sentences get truncated and add noise.
4. **Spread across categories.** Aim for roughly 12–20 per category rather than
   100 tense errors.
5. **Plain text only.** No tabs, no newlines inside a field — those silently
   shift every later column, which is the single most common way this kind of
   file gets corrupted. The validator checks for it.

## How to annotate

**The two annotators must not look at each other's corrections.** That is the
whole basis of the agreement number: if you copy each other, Cohen's kappa
measures nothing. Practical approach: one person writes all the `source`
sentences, then annotators 1 and 2 each fill in their own column in separate
copies of the file, and only then are the columns merged.

**Leaving a sentence unchanged is a legitimate annotation.** If you think the
sentence is acceptable English as it stands, copy the source into your column
verbatim. This matters for `dialect-usage` items: words like *prepone* and
constructions like *"Please revert back"* are standard in Indian English and
arguably need no correction. Disagreement there is a finding worth discussing in
the report, not a mistake to be smoothed over. `EXAMPLE-004` shows two
annotators legitimately disagreeing.

---

## Running the tooling

Use the project interpreter (`python` on PATH is MSYS2 and has no pip):

```powershell
$PY = "C:\Users\91942\miniconda3\envs\gec\python.exe"
$env:PYTHONNOUSERSITE = "1"
```

**1. Validate and measure agreement** — run this after every editing session:

```powershell
& $PY -m gec.kappa --tsv data_manual/indian_english_test.tsv
```

Checks for duplicate ids, empty fields, embedded tabs/newlines and unknown
categories, then reports:

- **token-level Cohen's kappa** — per source token, did both annotators touch
  it? (This is the headline agreement number.)
- **sentence-level kappa** — did both annotators change the sentence at all?
- **raw agreement** for both, which stays meaningful even when kappa is
  undefined (that happens when one annotator edited everything, or nothing).

Results go to `results/tables/indian_kappa.json`.

**2. Build the evaluation files** once there are ≥ 20 real rows:

```powershell
& $PY -m gec.prepare_data --indian
```

This tokenises all three text columns and builds a two-annotator reference M2
(`errant_parallel -orig src -cor ann1 ann2`), so a model is scored as correct if
it matches **either** annotator.

**3. Evaluate the trained models on it:**

```powershell
& $PY -m gec.run_experiments --plan e5
& $PY -m gec.aggregate
```

The E5 table lands in `results/tables/e5_indian_english.md`.

---

## A note for the report

The models are trained on W&I+LOCNESS, whose learners are predominantly
European. Indian English has a different error distribution *and* genuine
dialectal norms that are not errors at all. A low score on this set is therefore
not purely a model failure — part of it is a mismatch between what the training
data calls an error and what this variety of English actually does. Say that
explicitly rather than reporting the number alone.
