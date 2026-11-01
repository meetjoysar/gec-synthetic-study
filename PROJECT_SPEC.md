# PROJECT SPEC — "Fake Errors, Real Grammar"

**Can synthetic errors teach a small model to correct real (and Indian English) grammar errors?**

NLP IA Activity 2 — Mini-Project Implementation and Experimental Evaluation
Course: 316P07E123 Natural Language Processing, FY MTech Computer Engineering, K. J. Somaiya School of Engineering
Group: Suryoday Kumar Saroj (16030726001), Aryan Gole (16030726006), Meet P Joysar (16030726016)
Faculty: Dr. Grishma Sharma

> **For Claude Code:** this file is the single source of truth. Work through the phases in §10 **in order**. After each phase, run its acceptance checks, update `PROGRESS.md`, commit, and summarise what you did before starting the next phase. Follow the rules in §2 at all times.

---

## 1. What we are building and why

### 1.1 Research question
Our Activity 1 report studied three papers:
- **Pan & Cao (IEEE ICWS 2021):** efficient GEC by detecting errors first, then correcting only those; uses `difflib.SequenceMatcher` to extract edits.
- **Koo et al. (IEEE Access 2023):** claims data-centric tricks (noise injection, balanced clean/noisy data) behave differently when models are trained **only on synthetic data**, but never runs the real-data comparison, and uses BLEU.
- **Wang et al. (Findings of ACL 2024):** synthetic data helps when it **matches the real error distribution** and is mixed with real data.

**Gap:** no paper compares synthetic-only, real-only and mixed training on the same task with the same metric (F0.5). That is our project.

### 1.2 Deliverables
1. A rule-based **synthetic error generator** with a tunable noise ratio.
2. A **training pipeline** that fine-tunes `t5-small` as a grammar corrector.
3. An **evaluation pipeline**: ERRANT P/R/F0.5 (main), per-error-type F0.5, sacreBLEU, over-correction rate, latency.
4. **Experiments E0–E5** (§7), with all numbers produced by code and saved to `results/`.
5. An **Indian English test set** template plus agreement (Cohen's kappa) tooling. The group writes the sentences; Claude Code builds the tooling.
6. A **Gradio demo**: a user types a sentence and sees corrections from 3 models side by side, with each edit highlighted and explained.
7. Plots and tables ready for the Activity 2 report, plus `README.md` and `CREDITS.md`.

### 1.3 What the user sees in the demo
Input: *"He go to school yesterday and buy a apple."*
Output (per model): the corrected sentence with edits highlighted, plus a list of edits, e.g. `go → went · Verb tense · past action needs past tense`. Three columns: **Synthetic-only**, **Real-only**, **Mixed**.

---

## 2. Rules for Claude Code (non-negotiable)

1. **Never invent results.** Every number in a table, plot, README or `RESULTS.md` must come from a file written by our code. If a run hasn't happened, write `TBD`.
2. **Never train or tune on test data.** The test set is `ABCN.dev.gold.bea19.m2` (BEA-2019 dev). Select hyperparameters (best noise ratio, best mix) on **our validation split only**. Never generate synthetic data from test sentences.
3. **Do not write the Indian English test sentences.** Create the template, validation tooling and at most 5 clearly marked `EXAMPLE` rows that show the format. The group writes the real items, so it stays a human-made evaluation set.
4. **Ask the user before:** downloading anything over 500 MB, starting any run expected to take over 15 minutes on the current machine, or deleting data or checkpoints.
5. **Reproducibility:** every run takes a config file and a seed; save the resolved config, seed, git commit hash and library versions next to its results.
6. **Keep it simple.** This is an undergrad-scale project. Prefer clear, short modules over abstractions. No experiment-tracking services; plain JSON/CSV files.
7. **Cross-platform:** the user may be on Windows. Use Python entry points (`python -m ...`) and `pathlib`; no bash-only scripts. ERRANT CLI tools may be called via `subprocess`.
8. **Commit after each phase** with a clear message. Keep `data/`, `models/` and large outputs out of git.
9. **Tests:** every non-trivial module gets at least one `pytest` test. Keep the full test suite under ~2 minutes on CPU.
10. Keep `PROGRESS.md` updated: phase status, what was done, open issues, exact commands to reproduce.

---

## 3. Environment

- Python 3.10 or 3.11.
- Packages (pin in `requirements.txt`; check versions are mutually compatible when installing):
  - `torch` (CUDA build if a GPU is present)
  - `transformers`, `datasets`, `sentencepiece`, `accelerate`
  - `spacy` + model `en_core_web_sm`
  - `errant` (latest 3.x; it installs spaCy and `en_core_web_sm`. If its spaCy requirement conflicts with ours, follow errant's.)
  - `lemminflect` (word inflection for verb/noun errors)
  - `sacrebleu`, `scikit-learn` (Cohen's kappa), `pandas`, `matplotlib`, `pyyaml`, `tqdm`
  - `gradio` (demo), `pytest`
- **Compute:** development and smoke tests run on CPU. Full experiments need a GPU: a local NVIDIA GPU if available, otherwise Google Colab (free T4). Code must auto-detect the device. Provide `notebooks/colab_run.ipynb` (§9).
- **Precision:** T5 is unstable in fp16. Default to fp32; use bf16 only if `torch.cuda.is_bf16_supported()`.

---

## 4. Repository layout

```
gec-synthetic-study/
├── CLAUDE.md                  # short instructions for Claude Code
├── PROJECT_SPEC.md            # this file
├── PROGRESS.md                # phase-by-phase log, kept current by Claude Code
├── README.md                  # how to install, run experiments, launch demo
├── CREDITS.md                 # datasets, libraries, external code, AI-tool use
├── requirements.txt
├── .gitignore                 # data/, models/, results/runs/*/checkpoint*, __pycache__, .venv
├── configs/
│   ├── base.yaml
│   └── experiments/           # one YAML per run family (see §7)
├── data/                      # gitignored
│   ├── raw/                   # downloaded archives
│   └── processed/             # jsonl + text + m2 files we create
├── data_manual/
│   ├── indian_english_test.tsv        # template; group fills it
│   └── README.md                      # instructions for writing/annotating
├── src/gec/
│   ├── __init__.py
│   ├── paths.py               # all paths in one place
│   ├── m2.py                  # parse M2, apply edits, write M2
│   ├── tok.py                 # spaCy tokenisation + detokenisation
│   ├── prepare_data.py        # download/parse W&I+LOCNESS, build clean WikiText pool
│   ├── error_gen.py           # rule-based synthetic error generator
│   ├── build_datasets.py      # synthetic / real / mixed jsonl per config
│   ├── train.py               # fine-tune t5-small
│   ├── predict.py             # batch inference
│   ├── evaluate.py            # ERRANT, sacreBLEU, over-correction, latency
│   ├── distribution.py        # error-type distributions + generator re-weighting (E4)
│   ├── kappa.py               # Indian English annotator agreement
│   ├── explain.py             # ERRANT type → plain-English reason
│   ├── run_experiments.py     # runs a list of configs × seeds, resumable
│   └── aggregate.py           # tables + plots + RESULTS.md
├── app/
│   └── app.py                 # Gradio demo
├── notebooks/
│   └── colab_run.ipynb
├── tests/
└── results/
    ├── runs/<run_name>/seed<k>/   # config.yaml, metrics.json, predictions, logs
    ├── figures/
    ├── tables/
    └── RESULTS.md               # auto-generated
```

---

## 5. Data

### 5.1 Real data: W&I+LOCNESS v2.1 (BEA-2019)
- Download: `https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz` (small, tens of MB). Extract to `data/raw/`. Show the user the licence text included in the archive (non-commercial research and education use) and record it in `CREDITS.md`.
- Inside the archive, use the **M2** files (folder `wi+locness/m2/`). Expected names include `ABC.train.gold.bea19.m2` and `ABCN.dev.gold.bea19.m2`. Inspect the folder after extraction and adapt if names differ.
- Splits:
  - **train + val:** `ABC.train.gold.bea19.m2` (~34k sentences). Shuffle with seed 42 and hold out **2,000 sentences as `val`**; the rest is `train_real`.
  - **test:** `ABCN.dev.gold.bea19.m2` (~4.4k sentences). Never used for training or selection. Keep the original gold M2 as the evaluation reference.
- The official BEA-2019 test set has hidden answers; do not use it.

### 5.2 M2 format (implement in `m2.py`)
```
S This are a sentence .
A 1 2|||R:VERB:SVA|||is|||REQUIRED|||-NONE-|||0

```
- `S` line: the tokenised source sentence (tokens separated by spaces).
- `A start end|||type|||correction|||REQUIRED|||-NONE-|||annotator_id` — replace tokens `[start, end)` with `correction`. An empty correction means deletion; `start == end` means insertion.
- `A -1 -1|||noop|||-NONE-|||...` means no edit; skip it. Treat type `UNK` as no change.
- To build a target sentence: take edits for annotator 0, sort by `start`, apply right-to-left (or track offsets).
- Unit tests: insertion, deletion, replacement, multiple edits, noop, UNK.

### 5.3 Tokenisation convention
- Everything — model inputs, targets, predictions, M2 files — is in **spaCy-tokenised, space-separated** form, matching the M2 files and ERRANT's expectations (e.g. `do n't`, `it 's`, `school .`).
- `tok.tokenise(text) -> str` uses `en_core_web_sm` with only the tokenizer (disable other pipes for speed).
- `tok.detokenise(tokens) -> str` is for **display only** (demo): remove spaces before punctuation, rejoin `n't`, `'s`, `'re`, etc. Simple rules are fine.

### 5.4 Clean text for synthetic data: WikiText-103
- Load `Salesforce/wikitext`, config `wikitext-103-raw-v1`, train split, via `datasets` (a few hundred MB; tell the user before downloading).
- Clean: drop heading lines (starting with `=`) and empty lines; replace ` @-@ ` → `-`, ` @,@ ` → `,`, ` @.@ ` → `.`; split into sentences (spaCy `sentencizer` is enough); re-tokenise with `tok.tokenise`.
- Keep sentences with 8–40 tokens, at least one verb (via POS tags), no unusual characters (mostly ASCII letters, digits, common punctuation), and deduplicate.
- Save a pool of **60,000** sentences to `data/processed/clean_pool.txt` (seed 42). The first 50,000 are the **base set** reused by every synthetic condition, so conditions differ only in how sentences are corrupted.
- Note for the report: Wikipedia text differs in style from learner essays; this domain mismatch is a known limitation.

### 5.5 Indian English test set (manual)
- File: `data_manual/indian_english_test.tsv`, columns: `id, source, annotator1, annotator2, category, notes`.
- `source`: a sentence with errors common among Indian English speakers, written by a group member. `annotator1` and `annotator2`: each annotator's corrected version, written **independently**. `category`: e.g. `tense-perfect`, `stative-progressive`, `redundancy`, `article`, `preposition`, `question-form`, `uncountable-plural`, `dialect-usage`.
- Target: ~150 sentences. Claude Code adds ≤ 5 rows with `id` starting `EXAMPLE-`, excluded automatically from evaluation, e.g.:
  - `He is working here since 2019.` → `He has been working here since 2019.`
  - `We discussed about the plan.` → `We discussed the plan.`
  - `I am having two brothers.` → `I have two brothers.`
- `data_manual/README.md`: explain how to write items, that annotators must not look at each other's corrections, that a sentence may be left unchanged if the annotator thinks it is acceptable (dialect usage like "prepone" is a valid discussion point), and how to run the tooling.
- Tooling (`kappa.py` + `prepare_data.py`):
  - Validate the TSV (no empty sources, both annotations present, unique ids).
  - Tokenise all columns; build a two-annotator reference M2 with `errant_parallel -orig src.txt -cor ann1.txt ann2.txt -out indian_ref.m2`.
  - **Cohen's kappa:** for each annotator, use ERRANT to extract edits against the source; label each source token 1 if any edit touches it (an insertion marks the token after it), else 0; compute `sklearn.metrics.cohen_kappa_score` over all tokens. Also report sentence-level agreement (did both annotators change the sentence?).
  - If the TSV has fewer than 20 real rows, print a warning and skip E5 gracefully.

---

## 6. The synthetic error generator (`error_gen.py`)

### 6.1 Interface
```python
gen = ErrorGenerator(weights: dict[str, float], seed: int)
noisy_tokens, edits = gen.corrupt(tokens: list[str], pos_tags: list[str], p: float, force_one: bool)
```
- `p` = noise ratio: probability that each token is **selected** for corruption (as in Koo et al.'s word-level ratio). A selected token gets one operation sampled by `weights` among the operations applicable to it. If none applies, skip it.
- `force_one=True`: if no edit happened, retry up to 10 times; then force one applicable edit on a random token.
- Returns the corrupted tokens and an edit log (`op`, position, before, after) for debugging.
- Deterministic given seed. Never corrupt the same token twice in one pass. Precompute POS tags once per sentence with spaCy (`tagger` only) and cache them with the pool.

### 6.2 Operations (each maps to an ERRANT category)
| Op name | ERRANT type | What it does |
|---|---|---|
| `det_delete` / `det_insert` / `det_swap` | DET | delete `a/an/the`; insert one before a noun; swap among `a/an/the` |
| `prep_swap` / `prep_delete` | PREP | swap within `{in, on, at, for, to, of, with, from, by, about, into}`; delete one |
| `sva` | VERB:SVA | `is↔are`, `was↔were`, `has↔have`, `does↔do`; add/remove 3rd-person `-s` on present verbs (use `lemminflect`) |
| `verb_tense` | VERB:TENSE | past ↔ present, e.g. `went → go`, `walks → walked` (lemminflect with VBD/VBZ/VBP) |
| `verb_form` | VERB:FORM | e.g. `to go → to going`, `can go → can goes`, `is going → is go` |
| `noun_num` | NOUN:NUM | singular ↔ plural (lemminflect NN/NNS) |
| `spell` | SPELL | one character insert / delete / substitute / adjacent swap, words with ≥ 4 letters |
| `punct_delete` / `punct_insert` | PUNCT | drop a comma or final period; insert a stray comma |
| `word_delete` | OTHER | delete a short function word (e.g. `is`, `to`, `that`) |
| `word_dup` | OTHER | duplicate a token |
| `case` | ORTH | lowercase the sentence-initial word or capitalise a mid-sentence common word |

- Default weights in `configs/base.yaml` (start roughly uniform across the ERRANT types; E4 re-tunes them).
- Add a helper `python -m gec.error_gen --demo 20` that prints 20 random clean→noisy examples for human inspection, and save a sample of 200 to `results/tables/generator_samples.tsv` for the report.
- Tests: determinism, `p=0` gives no change, `force_one` guarantees ≥ 1 change, each op produces the expected kind of change on a hand-made sentence, no crashes on punctuation-only or very short input.

### 6.3 Dataset building (`build_datasets.py`)
Each training set is a JSONL file of `{"id", "src", "tgt", "origin"}`, where `origin ∈ {real, synthetic, identity}`.
- **Synthetic set** parameters: `n` (default 50,000, drawn from the base set), `p` (noise ratio), `clean_ratio` (fraction of pairs left as identity pairs `src == tgt`; the rest use `force_one=True`), `weights`, `seed`.
- **Real set:** `train_real` from W&I (keeps W&I's own correct sentences, ~1/3 of the data).
- **Mixed set:** real + synthetic, shuffled with the run seed.
- Log for every dataset: size, share of identity pairs, average edits per noisy sentence, and the op counts. Save to `data/processed/<name>.stats.json`.

---

## 7. Experiments

All runs use `t5-small` and the base hyperparameters (§8) unless stated. Seeds: `[13, 42, 2026]`. **Budget mode** (default): E1, E2 and E4 use seed 13 only; E3 uses all three seeds. Selection uses `val`; reporting uses `test`.

| # | Name | Runs | Question |
|---|---|---|---|
| **E0** | Copy baseline | no training: prediction = source | BLEU looks high but F0.5 = 0. Is BLEU a fair GEC metric? (critique of Koo et al.) |
| **E1** | Noise-ratio sweep | synthetic-only, `clean_ratio=0`, `p ∈ {0.1, 0.2, 0.4, 0.6}` | Is there a best noise level? (Koo et al., noise injection) |
| **E2** | Clean:noisy mix | synthetic-only at best `p` from E1 (by val F0.5), `clean_ratio ∈ {0.0, 0.2, 0.33, 0.5}` (0.0 = the E1 run; reuse it) | Do identity pairs reduce over-correction? (Koo et al., balanced data). 0.33 ≈ W&I's own share of correct sentences. |
| **E3** | Main comparison | (a) **synthetic-only** (best E2 config), (b) **real-only**, (c) **mixed** = real + synthetic, (d) **synth→real** = train on synthetic, then fine-tune on real | Can synthetic replace real data, or only supplement it? |
| **E4** | Distribution matching | re-weight the generator to match real error types (§7.1); rerun (a) and (c) of E3 with the matched generator | Does matching real error patterns help? (Wang et al.) |
| **E5** | Indian English | evaluate every E3/E4 model on the Indian English set | How do models trained on standard learner data handle Indian English errors? |

Run naming: `e1_p0.2`, `e2_clean0.33`, `e3_synthetic`, `e3_real`, `e3_mixed`, `e3_synth2real`, `e4_synthetic_matched`, `e4_mixed_matched`.

For **every** trained run, compute on `test` (and on `val` for selection):
- ERRANT P / R / F0.5 overall and per category (`errant_compare -cat 2`, saved as CSV).
- sacreBLEU of predictions against the reference corrected sentences (reconstructed from M2, annotator 0).
- **Over-correction rate:** among test sentences whose gold target equals the source, the % the model changed.
- Average edits per sentence in predictions.

### 7.1 E4 procedure (`distribution.py`)
1. **Real distribution:** count ERRANT edit types (category level, without the M/R/U prefix, e.g. `DET`, `PREP`, `VERB:SVA`) in `train_real` M2 annotations (annotator 0). Normalise to proportions.
2. **Synthetic distribution:** take 5,000 pairs from the E2-best synthetic set, run `errant_parallel` on (src, tgt), count types the same way.
3. **Re-weight:** for each generator op, `w_new = w_old × (target_share / synthetic_share)` for its ERRANT type; renormalise; clip to [0.2×, 5×] per iteration. Regenerate 5,000 pairs, re-measure; do **3 iterations** maximum.
4. Report: before/after bar chart against real (only categories the generator can produce, plus an "uncovered" bucket for the real share it cannot produce, e.g. `OTHER`, `WO`, `MORPH`), and total variation distance before/after.
5. Save the final weights to `configs/experiments/e4_matched_weights.yaml`.

### 7.2 Latency
For `e3_real` seed 13 (models share the same architecture, so one model suffices): ms per sentence on 200 test sentences at batch size 1 and 32, greedy vs beam 4, on GPU (if present) and on CPU. Save to `results/tables/latency.csv`.

### 7.3 Expected scale (for planning, not a target)
`t5-small` will score far below published systems (which report 50–75 F0.5 using much larger models and data). That is fine: the result is the **comparison between conditions**. Sanity checks: trained models should have F0.5 > 0 on test, E0 must give F0.5 = 0, and over-correction rates should fall as `clean_ratio` rises (if not, investigate).

---

## 8. Training and inference

### 8.1 `configs/base.yaml` (starting values)
```yaml
model_name: t5-small
prefix: "grammar: "
max_source_len: 128
max_target_len: 128
learning_rate: 3.0e-4
weight_decay: 0.01
warmup_ratio: 0.05
train_batch_size: 32        # use gradient accumulation if memory is short
eval_batch_size: 64
epochs: 3
stage2_epochs: 2            # synth→real second stage
stage2_learning_rate: 1.0e-4
precision: auto             # fp32, or bf16 if supported; never fp16 for T5
generation:
  num_beams: 4
  max_new_tokens: 128
keep_checkpoint: false      # true only for models needed by the demo / E5
seeds: [13, 42, 2026]
data:
  synthetic_n: 50000
  val_size: 2000
```
Truncate inputs longer than `max_source_len` tokens (log how many). Use HuggingFace `Seq2SeqTrainer`; evaluate loss on `val` each epoch; keep the final model.

### 8.2 Commands
```bash
python -m gec.prepare_data --all
python -m gec.build_datasets --config configs/experiments/e1_p0.2.yaml --seed 13
python -m gec.train --config configs/experiments/e3_mixed.yaml --seed 42
python -m gec.predict --run results/runs/e3_mixed/seed42 --split test
python -m gec.evaluate --run results/runs/e3_mixed/seed42 --split test
python -m gec.run_experiments --plan e3 --seeds 13 42 2026   # resumable
python -m gec.aggregate
python app/app.py
```
`run_experiments.py` skips runs whose `metrics.json` already exists, so a Colab disconnect only loses the current run. It must also print a time estimate after the first run.

### 8.3 Smoke test
`python -m gec.run_experiments --plan smoke` runs the full pipeline on CPU with 300 training pairs, 50 test sentences and `max_steps=20`, finishing in a few minutes. It must pass before any GPU run.

### 8.4 Storage
A `t5-small` checkpoint is ~250 MB. Keep checkpoints only for: `e3_synthetic`, `e3_real`, `e3_mixed` (seed 13) and the E4 models. For other runs, keep predictions and metrics, then delete the weights.

---

## 9. Colab notebook (`notebooks/colab_run.ipynb`)
Cells: (1) check GPU; (2) mount Google Drive and set `RESULTS_DIR` / `MODELS_DIR` / `DATA_DIR` to a Drive folder so results survive disconnects; (3) clone or upload the repo; (4) `pip install -r requirements.txt`; (5) run the smoke test; (6) one cell per experiment plan (`--plan e1`, `e2`, `e3`, `e4`, `e5`); (7) run `aggregate` and show the figures. Paths come from environment variables read in `paths.py`.

---

## 10. Phases and acceptance checks

| Phase | Work | Done when |
|---|---|---|
| **0. Setup** | Repo layout, `requirements.txt`, `.gitignore`, `paths.py`, `PROGRESS.md`, git init | `pip install -r requirements.txt` works; `python -c "import errant, spacy, transformers"` works; `pytest` runs |
| **1. Data** | `m2.py`, `tok.py`, `prepare_data.py` (W&I download/parse/split, WikiText pool) | Tests pass; `data/processed/` has train_real/val/test files and `clean_pool.txt`; printed counts match §5 |
| **2. Generator** | `error_gen.py` + tests + demo printout | All ops tested; `--demo 20` output looks plausible; show the user 10 examples and ask them to confirm it looks right |
| **3. Evaluation + E0** | `evaluate.py` (ERRANT wrapper, sacreBLEU, over-correction), E0 copy baseline | E0 on test: F0.5 = 0, BLEU reported; per-category CSV written |
| **4. Training** | `build_datasets.py`, `train.py`, `predict.py`, `run_experiments.py`, smoke plan | Smoke test passes end-to-end on CPU |
| **5. Colab** | `colab_run.ipynb`, environment-variable paths | Notebook runs the smoke test (user confirms on Colab) |
| **6. E1–E3 runs** | Plans for E1, E2, E3 | Prepared and handed to the user to run on GPU; results land in `results/runs/` |
| **7. E4** | `distribution.py`, matched weights, E4 plan | Distribution chart + TVD before/after; E4 runs prepared |
| **8. Indian English** | TSV template, `data_manual/README.md`, validation, kappa, E5 plan | Tooling works on the EXAMPLE rows; E5 skips gracefully until ≥ 20 real rows exist |
| **9. Aggregation** | `aggregate.py`, figures, tables, latency, `RESULTS.md` | All figures in §11 generated from saved metrics; missing runs shown as TBD |
| **10. Demo** | `app/app.py`, `explain.py` | Demo runs locally with any available checkpoints (falls back gracefully if some are missing) |
| **11. Docs** | `README.md`, `CREDITS.md`, final `PROGRESS.md` | A new user can reproduce everything from the README |

Phases 6–7 need a GPU. If none is available locally, prepare everything, then tell the user exactly which notebook cells to run on Colab and stop until results are back.

---

## 11. Outputs for the report (`aggregate.py`)
- `results/tables/summary.csv`: one row per run × seed with all metrics; `summary_mean_std.csv` aggregated over seeds.
- Figures (`results/figures/`, PNG, 150 dpi, readable axis labels, no chart junk):
  1. `e0_bleu_vs_f05.png`: copy baseline vs best model, BLEU and F0.5 side by side.
  2. `e1_noise_curve.png`: val and test F0.5 vs noise ratio `p`.
  3. `e2_clean_ratio.png`: F0.5 and over-correction rate vs `clean_ratio`.
  4. `e3_conditions.png`: P, R, F0.5 for the four conditions, mean ± std over 3 seeds.
  5. `e4_error_types.png`: real vs synthetic error-type shares, before and after matching.
  6. `e3_per_category.png`: F0.5 per ERRANT category for the E3 conditions (top 10 categories by frequency).
- Tables: `e5_indian_english.md` (P/R/F0.5 per model plus kappa), `latency.csv`, `generator_samples.tsv`.
- `results/failure_gallery.md`: 10 test sentences where models disagree most or are clearly wrong (source, gold, each model's output). Chosen automatically by a simple rule, e.g. largest edit disagreement between models.
- `results/RESULTS.md`: auto-generated summary of all tables with links to figures. Numbers only; no interpretation beyond restating them.

---

## 12. Demo (`app/app.py`)
- Gradio Blocks UI: title, a textbox, a "Correct" button, 6–8 example buttons (mix of standard learner errors and Indian English examples), and three result columns: **Synthetic-only**, **Real-only**, **Mixed** (from `e3_*` seed 13 checkpoints; checkpoint paths set in `app/config.yaml`).
- Pipeline per model: tokenise input → generate → ERRANT annotate (`annotator.parse(text, tokenise=False)` on the tokenised strings, `annotator.annotate(orig, cor)`) → render.
- Rendering: detokenised corrected sentence with deletions in red strikethrough and insertions in green (HTML); below it, a list of edits: `before → after · Error type · reason`.
- `explain.py` maps ERRANT categories to short reasons, e.g.:
  - `DET`: "Article use: a/an/the is missing, extra or wrong."
  - `PREP`: "Preposition: a different preposition fits here."
  - `VERB:SVA`: "Subject–verb agreement: the verb must match its subject."
  - `VERB:TENSE`: "Verb tense: the time of the action needs a different tense."
  - `VERB:FORM`: "Verb form: after this word the verb takes a different form."
  - `NOUN:NUM`: "Noun number: singular/plural doesn't fit."
  - `SPELL`: "Spelling."
  - `PUNCT`: "Punctuation."
  - `ORTH`: "Capitalisation or spacing."
  - `WO`: "Word order."
  - fallback: "Other change." Prefix meanings: `M:` missing word, `U:` unnecessary word, `R:` replaced word.
- Show per-model inference time under each column. Load models once at startup; run on GPU if available.

---

## 13. Report-relevant notes to keep in `PROGRESS.md`
The Activity 2 report needs: Title & group details, Problem Definition, Data Pre-processing, Algorithms/Techniques, Implementation, Evaluation & Results, References, and a declaration of AI tools and external code. As you work, record in `PROGRESS.md`: dataset sizes after each preprocessing step, generator settings, hyperparameters actually used, run times, and any deviation from this spec and why.

## 14. CREDITS.md must list
- Datasets: W&I+LOCNESS v2.1 (BEA-2019; licence from the archive), WikiText-103 (Merity et al., 2016; CC BY-SA).
- Tools: ERRANT (Bryant et al., 2017), spaCy, HuggingFace Transformers/Datasets, T5 (Raffel et al., 2020), lemminflect, sacreBLEU, Gradio, scikit-learn.
- Any external code snippets adapted, with source links.
- AI tools: "Code written with the assistance of Claude Code (Anthropic); all code reviewed and run by the group."
