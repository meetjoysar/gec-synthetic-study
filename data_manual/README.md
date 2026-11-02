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
| `id` | A unique id. The imported sentences use `IN-0001`, `IN-0002`, ... **Do not** start a real row with `EXAMPLE-` — that prefix means "excluded from evaluation". |
| `source` | A sentence containing an error common among Indian English speakers. Written by a group member. |
| `annotator1` | Annotator 1's corrected version, written **independently**. |
| `annotator2` | Annotator 2's corrected version, written **independently**. |
| `category` | One of the categories below. |
| `notes` | Optional: why it is an error, or why you disagreed. Pre-filled with a warning on rows from sensitive posts. |
| `source_type` | Provenance. `reddit-manual` for the collected sentences; empty for hand-written ones. |
| `source_note` | Writer id and post label, e.g. `W11 flat-hunting`. No usernames or links are ever stored. |

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

### Writing sources first is fine

The tooling expects the workflow above: all the `source` sentences written
first, annotator columns filled in later. A row with a `source` but empty
annotator columns is reported as **pending annotation**, not as an error, and it
does **not** count toward the 20-row threshold until *both* annotator columns
are filled. So you can commit 150 sources straight away and annotate over
several sessions; `python -m gec.kappa` will tell you how many are ready and how
many are still pending each time you run it.

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

## Control sentences come for free

You do not need a separate column or flag for these. If you include sentences
you consider **acceptable** Indian English and both annotators copy the source
into their column unchanged, the pipeline already:

- marks the row `is_clean` (both annotations identical to the source), and
- reports **over-correction** on exactly those rows in the E5 table.

That measures something nothing else in the project does: whether the model
imposes a prescriptive standard on a legitimate variety of English by
"correcting" sentences that were fine. Around 20-25 such sentences is plenty.
`EXAMPLE-004` ("Please revert back to me soon.") shows the annotator-disagreement
case; a control sentence is the case where both annotators agree to change
nothing.

---

## Annotation rules (agreed)

Both annotators follow these. The point is that disagreement should reflect a
genuine judgement call, not a difference of convention - a conventional
disagreement lowers Cohen's kappa without telling us anything.

**Scope**

1. **Correct grammar errors only, with minimal edits.** Do not rewrite for
   style, do not improve the phrasing, do not shorten. If a sentence is
   grammatical but clumsy, leave it.
2. **Correct only what is wrong *within* the sentence. Never supply missing
   context.** Many sentences came from the middle of a post, so a pronoun or
   subject may refer to something not present (*"Said that such kind of a
   complaint had never come before..."*). Fix the grammar you can see; do not
   invent the missing subject.
3. **Leave comma splices as they are.** A few sentences are long runs of
   clauses joined by commas. Do not split them, do not insert semicolons.

**Punctuation and case**

4. **Never add missing sentence-final punctuation.** 15 sentences end without a
   full stop. Leave them. (Agreed so that the agreement score reflects
   grammatical judgement rather than a formatting convention.)
5. **Leave exclamation marks and `...` exactly as they are**, however many
   there are.
6. **Fix the capitalisation of `i` and of sentence starts.**

**Words to leave alone**

7. **Indian English loanwords stay unchanged**: *jugaad, raja, sanskari,
   chalta hai, IPS, PG, EMIs, aunty, hubby*.
8. **Informal words stay unchanged**: *tbh, idk, ig, haha, y'all, bs, gf, esp,
   def*.
9. **In-group abbreviations stay unchanged**: *fl* and *ml* are Wattpad for
   "female lead" and "male lead"; *PS* is a games console; *TRP* is a
   television rating. They are terms, not typos.

**Words to change**

10. **Text-speak expands to the standard word**: *hv -> have, bn -> been,
    nd -> and, livin -> live-in*.
11. **Clear spelling typos get fixed**: *nowdays, posessive, seprates, thier,
    ssad, socail, thode*.

**Judgement calls**

12. **Dialect usages that are standard in Indian English** - *"like this only",
    "do the needful", "till now", "from her side", "revert back"* - are
    **each annotator's own decision**. Correct them or leave them, but write
    your choice in the `notes` column so the report can discuss the
    disagreement. This is a finding, not a mistake.

**Process**

13. **Never look at the other annotator's file until both are finished.** This
    is the entire basis of the agreement statistic.

### A sentence you judge acceptable

Leave the `correction` cell exactly equal to the `source`. That is a real
annotation, not a skipped row: if both annotators do it, the row becomes a
**control** and the pipeline reports over-correction on it automatically. No
extra flag needed.

---

## The annotation workflow, step by step

**1. Each annotator edits their own file**, in place:

- Annotator 1: `data_manual/annotation/annotator1.tsv`
- Annotator 2: `data_manual/annotation/annotator2.tsv`

Three columns: `id`, `source`, `correction`. The `correction` column arrives
pre-filled with the source, so you **edit it where there is an error and leave
it alone where there is not**. Do not change `id` or `source`. Both files are
gitignored, so neither annotator can see the other's work through git.

**2. When BOTH files are finished**, merge them:

```
python -m gec.kappa --merge-annotations
```

This copies each `correction` into the `annotator1` / `annotator2` columns of
`indian_english_test.tsv`, matching on `id`. It will **never overwrite a cell
that already has content** unless you pass `--force`, so re-running it is safe.

**3. Check agreement:**

```
python -m gec.kappa
```

**4. Build the evaluation files and score the models:**

```
python -m gec.prepare_data --indian
python -m gec.run_experiments --plan e5
python -m gec.aggregate
```

## A note for the report

The models are trained on W&I+LOCNESS, whose learners are predominantly
European. Indian English has a different error distribution *and* genuine
dialectal norms that are not errors at all. A low score on this set is therefore
not purely a model failure — part of it is a mismatch between what the training
data calls an error and what this variety of English actually does. Say that
explicitly rather than reporting the number alone.
