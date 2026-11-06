# PROGRESS

Phase-by-phase log for **"Fake Errors, Real Grammar"** (NLP IA Activity 2).
Read this at the start of every session to know where we are.

Rule: **no invented numbers.** Every figure below either came from a file our
code wrote, or says `TBD`.

---

## Status at a glance

| Phase | Work | Status |
|---|---|---|
| 0 | Setup: env, requirements, repo, `paths.py`, `repro.py`, git | **DONE** |
| 1 | Data: `m2.py`, `tok.py`, `prepare_data.py` | **DONE** — counts match §5 exactly |
| 2 | Generator: `error_gen.py` | **DONE** — 15 ops / 10 categories, examples reviewed |
| 3 | Evaluation + E0 | **DONE** — E0 test F0.5 = 0.0000, BLEU 80.03 |
| 4 | Training: `build_datasets.py`, `train.py`, `predict.py`, `run_experiments.py` | **DONE** — smoke passes on CPU in 1.4 min |
| 5 | Colab notebook | **DONE** — `notebooks/colab_run.ipynb`, env-var paths, `requirements-colab.txt` (not needed in the end: the local GPU carried the campaign) |
| 6 | E1–E3 runs (GPU) | **DONE** — E0 + 5 E1 arms + 4 E2 + 4 E3; E3 at **3 seeds** (13, 42, 2026), so deviation D6 is withdrawn. 24 runs total |
| 7 | E4 distribution matching | **DONE** — matching loop (TVD 0.2502 → 0.1146) + both matched runs. Matching costs F0.5 in both arms |
| 8 | Indian English: template, kappa, E5 | **149 sentences imported + annotated by annotator 1**; all runs already score the `indian` split; E5 table awaits the `e5` stage. Cohen's kappa needs a second annotator |
| 9 | Aggregation: figures, tables, `RESULTS.md` | **DONE** — 6/6 figures, 12 tables, `RESULTS.md` written from 16/16 runs |
| 10 | Gradio demo | **DONE** — `app/app.py` + `explain.py`, serves with zero checkpoints and now has real E3 models |
| 11 | Docs: `README.md`, `CREDITS.md` | **DONE** — final pass after the last `aggregate` |

---

## Environment (measured, not assumed)

| | |
|---|---|
| Interpreter | `C:\Users\91942\miniconda3\envs\gec\python.exe` (conda env `gec`, **Python 3.10.22**) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, **4,096 MiB VRAM**, compute capability **(8, 6)** |
| torch | `2.6.0+cu124`, `cuda_available=True` |
| Precision | **bf16 is real here** (capability >= 8.0), not emulated. Never fp16 for T5. |
| errant | `3.0.2`; spaCy `3.8.7`; `en_core_web_sm` `3.8.0` |
| Disk free | ~151 GiB |

### Measured GPU throughput (`--timing-probe`, not estimated)

| train batch × accum | examples/s | peak VRAM | sentences/s (beam 4) | |
|---|---:|---:|---:|---|
| 8 × 4 | 96.9 | 1600 MB | 35.7 | the original default |
| **16 × 2** | **161.5** | **2117 MB** | **50.8** | **chosen — 1.67×** |
| 32 × 1 | 187.7 | 3383 MB | 49.9 | a further 16%, but only ~700 MB headroom |

All three keep the effective batch at the spec's 32. `32 × 1` was rejected
because this GPU also drives the display, and ~700 MB of headroom is not a
sensible margin across a multi-hour campaign. bf16 is confirmed **real** here
(compute capability 8.6), not emulated.

My pre-measurement analytical estimate was 55 examples/s — **2.9× too
pessimistic**, which is why the original budget trim (D4/D5) turned out to be
unnecessary and was withdrawn. Recorded as a caution: estimate, then measure,
then re-decide.

### Two machine-specific gotchas — every command needs these

```powershell
$env:PYTHONNOUSERSITE = "1"   # a broken pip 23.1.2 in %APPDATA%\Python shadows the env's pip
$env:PIP_CONFIG_FILE  = "nul" # the global pip.ini adds the dead pypi.ngc.nvidia.com index
$PY = "C:\Users\91942\miniconda3\envs\gec\python.exe"
```

1. **`python` and `pip` on PATH are MSYS2 and have no pip at all.** Always use `$PY`.
   Never a bare `python`.
2. Without `PYTHONNOUSERSITE=1`, `python -m pip` resolves to a **broken** pip 23.1.2 in
   `%APPDATA%\Roaming\Python\Python310` which fails with
   `ValueError: Unable to find resource t64.exe`.
3. Without `PIP_CONFIG_FILE=nul`, every install wastes 5 DNS retries on
   `pypi.ngc.nvidia.com` (a dead host in the global `pip.ini`). `--isolated` does **not**
   suppress it, because `--isolated` ignores *user* config but still reads the global file.

---

## Acceptance checks passed so far

| Phase | Check (spec §10) | Result |
|---|---|---|
| 0 | `pip install` works; `import errant, spacy, transformers` works; `pytest` runs | PASS — errant annotates `He go`→`He goes` as `R:VERB:SVA` |
| 1 | `data/processed/` has the splits + `clean_pool.txt`; printed counts match §5 | PASS — 32,308 / 2,000 / 4,384 / 60,000; all four alignment invariants hold |
| 2 | All ops tested; `--demo 20` plausible; user confirms 10 examples | PASS — 115 tests; examples reviewed at p=0.1/0.2/0.4 |
| 3 | E0 on test: **F0.5 = 0**, BLEU reported, per-category CSV written | PASS — F0.5 = **0.0000**, BLEU(13a) = **80.03** |
| 4 | Smoke test passes end-to-end on CPU | PASS — 1.4 min |
| 9 | All figures generated from saved metrics; missing runs shown as TBD | PASS — verified with 13 of 14 runs missing, no crash |

`cross_check_delta = 0.0` on every split evaluated so far: our independent
reimplementation of `errant_compare` agrees exactly with the parsed CLI output,
which makes the stdout parsers self-validating.

## Phase 0 — Setup (DONE)

### What was done
- Created conda env `gec` (Python 3.10.22); installed `torch==2.6.0+cu124` from the PyTorch
  index, then `requirements.txt`, then `en_core_web_sm` 3.8.0.
- Repo root is **`IA/gec-synthetic-study/`**; `git init -b main`, `core.autocrlf false`.
- Wrote `requirements.txt`, `requirements-colab.txt`, `.gitignore`, `.gitattributes`,
  `pyproject.toml`, `src/gec/paths.py`, `src/gec/repro.py`, `src/gec/m2.py`, `src/gec/tok.py`,
  `tests/conftest.py`, `tests/test_m2.py`, `tests/test_tok.py`.

### Acceptance check — PASSED
```
errant 3.0.2
[('go', '->', 'goes', 'R:VERB:SVA')]
```
One line proving spaCy, the model, errant and its alignment classifier all cooperate.
`pytest -q`: **54 passed**.

### Reproduce
```powershell
$env:PYTHONNOUSERSITE="1"; $env:PIP_CONFIG_FILE="nul"
& "C:\Users\91942\miniconda3\Scripts\conda.exe" create -y -n gec python=3.10
$PY = "C:\Users\91942\miniconda3\envs\gec\python.exe"
& $PY -m pip install --no-warn-script-location torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
& $PY -m pip install --no-warn-script-location -r requirements.txt
& $PY -m spacy download en_core_web_sm
& $PY -m pytest -q
```

---

## Deviations from PROJECT_SPEC.md (and why)

Recorded as they happen, per spec section 13.

| # | Spec says | We do | Why |
|---|---|---|---|
| D1 | Repo named `gec-synthetic-study/` (§4) | Repo root **is** `IA/gec-synthetic-study/`; `CLAUDE.md`, `PROJECT_SPEC.md`, `IA1_REPORT_GEC.md` are **copies**, the originals stay in `IA/` | User chose the subfolder layout. The repo must be self-contained to be reproducible. `CLAUDE.md`, `IA1_REPORT_GEC.md` and `HOW_TO_RUN.md` are now **gitignored** but kept on disk: the first is assistant instructions whose rules all live in `PROJECT_SPEC.md` anyway (the six code and test citations now point there), the second is the Activity 1 submission, the third hardcodes this machine's interpreter path. |
| D2 | `errant` "installs spaCy and `en_core_web_sm`" (§3) | `python -m spacy download en_core_web_sm` run separately | **The spec is wrong for errant 3.x.** `errant==3.0.2` depends only on `spacy>=3.2,<4` and `rapidfuzz`; it does not pull the model. Also means the feared spaCy version conflict does not exist. |
| D3 | — | Added `src/gec/errant_cli.py` and `src/gec/repro.py` beyond the §4 module list | Each has >=3 consumers. Keeping ERRANT behind a subprocess wrapper means `train.py` never imports spaCy. |
| ~~D4~~ | `data.synthetic_n: 50000`, `epochs: 3` (§8.1) | **WITHDRAWN** — spec values restored | Was trimmed to 30000/2 on an *analytical* estimate of 55 examples/s. The timing probe measured **161.5/s** (2.9× better) after batch tuning, so the full setting is affordable. No longer a deviation. |
| ~~D5~~ | E1 sweeps `p ∈ {0.1, 0.2, 0.4, 0.6}` (§7) | **WITHDRAWN** — all four points restored | Same reason as D4. E1 now runs 5 arms: the spec's four noise points plus the pre-compensated arm (D10). |
| D6 | E3 uses seeds `[13, 42, 2026]` (§7) | **WITHDRAWN — all three seeds ran** (173 min) | Was the last budget cut standing. Running it was the right call: the seeds reversed one of the two small conclusions seed 13 had produced (mixed vs real on Indian English) and gave a measured noise floor of ~0.02 F0.5. Figure 4 now carries real error bars. E1/E2/E4/E5 remain seed 13, as spec §7's budget mode prescribes. |
| D7 | WikiText-103 full train split (§5.4) | `streaming=True` with early stop at 60k sentences (~60–90 MB instead of ~250 MB) | User-confirmed. We need 60k of ~4M sentences; streaming avoids the §2.4 download threshold entirely. |
| D8 | `datasets` implied for the training path (§3) | Plain `torch.utils.data.Dataset` + `DataCollatorForSeq2Seq`; `datasets` only used for WikiText | Decouples training from Arrow/`huggingface-hub` version churn. `datasets>=4.3` requires `huggingface-hub>=1.31` while `transformers 4.53.1` requires `<1.0`. |
| D9 | Phase order 0–11 (§10) | Indian-English template moved to **after Phase 2**; `aggregate.py` skeleton moved to **before the first GPU run** | The 150 hand-written sentences are the longest-lead *human* task and must start early. Building the TBD path first verifies it in the hardest case (everything missing). |
| D10 | E1 sweeps `p` only, with one fixed set of generator weights (§6.2, §7) | **Extra E1 arm `e1_p0.2_precomp`**: same `p`, seed and base-set sentences as `e1_p0.2`, but with generator weights pre-compensated for applicability gating | **User-requested.** §6.2's weights are uniform across ERRANT types *as requested*, but each op's `applies` check gates whether it can fire, so the **realised** mix is badly skewed (measured below). The extra arm separates two questions the default weights conflate: "is there a best noise level?" and "does the generator's skewed realised distribution hold synthetic-only back?". Costs ~30 min of GPU time. |
| D11 | — | Predictions are re-tokenised with spaCy before scoring (`predict.retokenise`) | T5's pretraining pulls its output toward ordinary detokenised English, so it emits `information.` where the source had `information .` — **48 of 50** smoke predictions needed it. ERRANT cannot distinguish that surface convention from a real `PUNCT` edit, so left alone it manufactures a spurious edit on nearly every sentence. Spec §5.3 already requires predictions to be in tokenised form. |
| D12 | §5.4 keeps sentences with "no unusual characters (mostly ASCII…)" | Clean-pool charset filter rejects a sentence if ASCII folding **destroys** information (introduces a `?`), rather than applying a 95% character ratio | A ratio test is the wrong instrument: a long English sentence with one embedded Greek or CJK word clears 95% easily and still leaves a `??????????` token in the training data. Accented Latin folds cleanly (`café`→`cafe`) and is kept. Junk with ≥2 `?` fell from 326 to **1** of 60,000. |
| D14 | §5.5: each `source` is "written by a group member" | 149 sources were **manually collected from public Indian subreddits** by the group and human-selected, not composed from scratch | **User-requested.** Authentic text carries errors people actually make, rather than the errors a writer *believes* are common - and it makes Cohen's kappa meaningful, since with invented sentences both annotators already know the intended error and agreement inflates toward 1. Two consequences the report must state: (a) §5.5 assumes every source *contains* an Indian English error, which a collected set cannot guarantee - some read as fluent and become control sentences; (b) 5 of 20 writers supply 40% of the set, so per-category results partly reflect a few idiolects. |
| D15 | — | Annotation rules fix two conventions the spec leaves open: **never add missing sentence-final punctuation** (15 sentences lack it) and **never supply missing context** (21 start with a connective) | A conventional disagreement lowers kappa without telling us anything. Both decisions are recorded in `data_manual/README.md` so the two annotators apply them identically. |
| D13 | §8.4 delete checkpoints except the demo/E5 models | Checkpoint pruning happens in `run_experiments` **after** prediction and evaluation, never inside `train()` | Deleting the weights at the end of training leaves `predict.py` with nothing to load. Caught by the smoke test. Retention is then a disk question, not a correctness one, and spec §7's "evaluate every E3/E4 model on the Indian set" stays possible. |

---

## Dataset sizes after each preprocessing step

All measured, from the stats files our code wrote.

| Step | Count | Source file |
|---|---|---|
| W&I `ABC.train.gold.bea19.m2` blocks | **34,308** | — |
| → `train_real` | **32,308** (33.8% already correct, 1.81 edits/sentence) | `data/processed/train_real.stats.json` |
| → `val` (held out, seed 42) | **2,000** (33.1% correct) | `data/processed/val.stats.json` |
| W&I `ABCN.dev.gold.bea19.m2` → `test` | **4,384** (35.7% correct, 1.70 edits/sentence) | `data/processed/test.stats.json` |
| WikiText-103 clean pool | **60,000** (mean 23.3 tokens; 50,000 base set + 10,000 held back for the E4 probe) | `data/processed/clean_pool.stats.json` |
| → scanned to produce it | 118,922 sentences from 27,392 lines; rejected 19,154 length / 2,088 no-verb / 1,492 charset / 188 duplicate | same |
| Indian English (real rows, excl. `EXAMPLE-`) | **0** — awaiting the group (5 `EXAMPLE-` rows ship as the format demo) | `data/processed/indian.skip` |

**Note for the report:** the 33.8% already-correct share of `train_real`
empirically confirms spec §7's claim that `clean_ratio = 0.33` matches W&I's own
proportion of correct sentences. That was an assumption in the spec; it is now a
measurement.

## Generator error distribution (measured, 30,000 pairs at p=0.2)

The realised mix is **not** the requested mix, because each operation's
`applies` check gates whether it can fire at all. `word_dup` applies to almost
any token; `sva` needs a copula or a VBZ/VBP.

| ERRANT type | real `train_real` | synthetic (default weights) | note |
|---|---:|---:|---|
| PUNCT | 17.6% | 24.9% | |
| OTHER | 13.1% | 20.9% | |
| DET | 11.6% | 10.0% | |
| PREP | 10.0% | 7.4% | |
| VERB:TENSE | 6.2% | 2.3% | under-produced |
| ORTH | 4.9% | 8.8% | |
| NOUN:NUM | 4.2% | 5.8% | |
| SPELL | 3.8% | 18.0% | **4.7× over-produced** |
| VERB:FORM | 3.7% | 0.8% | **4.6× under-produced** |
| VERB:SVA | 2.3% | 1.1% | under-produced |
| *uncovered* (WO, MORPH, NOUN:POSS, …) | 22.6% | 0.0% | the generator structurally cannot produce these |

Verb errors overall: **15.8% real vs 4.2% synthetic.** Total variation distance
over the covered categories: **0.220**. This is the gap E4 exists to close, and
the reason for the extra `e1_p0.2_precomp` arm (deviation D10).

Pre-compensation result (`python -m gec.distribution --precompensate`): TVD
against uniform falls **0.336 → 0.163** over 3 clipped iterations; SPELL lands
on target (21.5% → 9.9%) and verb categories roughly quadruple. They still fall
short of 10% because of a **structural ceiling**: no weighting can create a verb
error on a sentence with no eligible verb.

## Hyperparameters actually used

Per-run values are written to `results/runs/<run>/seed<k>/config.yaml` and
`env.json`. Base values in `configs/base.yaml`; the trimmed-budget ones are
`epochs: 2`, `data.synthetic_n: 30000`, `train_batch_size: 8` with
`gradient_accumulation_steps: 4` (effective 32, per spec §8.1).

## Run times

| What | Measured |
|---|---|
| `pytest -q` | ~55 s (115 tests) |
| `--plan smoke` (CPU, end to end) | **1.4 min** |
| E0 copy baseline (CPU, val + test) | **0.9 min** |
| WikiText pool build (CPU) | ~11 min |
| W&I download + split (CPU) | ~1 min |
| GPU training run (50k × 3 epochs) | **~19.4 min** each (measured), + ~6 min evaluation |
| E1 plan (5 arms) | **~97 min** estimated from measurement |
| Full remaining grid (E1–E4, single seed) | ~5.3 h ideal / ~6.5 h realistic |
| Per-run detail | `results/runs/_timing.jsonl` |

## Observations for the report (from E1 arm 1, e1_p0.1)

**1. BLEU is almost blind to the improvement F0.5 measures.** On `test`:

| | F0.5 | BLEU(13a) |
|---|---:|---:|
| E0 copy baseline | 0.0000 | 80.03 |
| e1_p0.1 (synthetic-only) | **0.2986** | 80.53 |

A system that goes from correcting *nothing* to a usable 0.30 F0.5 moves BLEU by
**+0.5 points**. This is a sharper version of the E0 argument than the copy
baseline alone and the strongest single piece of evidence for the Koo et al.
critique. Figure 1 should show both bars side by side.

**2. Val loss rises after epoch 1 — the models overfit.**
`val_loss_per_epoch = [0.4569, 0.4750, 0.4932]` for `e1_p0.1`.

Spec §8.1 says to evaluate val loss each epoch and **keep the final model**, so
this is spec-compliant and we are not changing it mid-campaign: every condition
is affected identically, so the between-condition comparison — which *is* the
research question — stays valid. But it should be stated in the report, and it
means our absolute F0.5 figures are probably a little below what early stopping
would give. Note also that rising cross-entropy does not strictly imply falling
F0.5 for a generation task, so this is a caution, not a correction.

**3. Per-category F0.5 tracks what the generator produces.** From
`e1_p0.1`'s test per-category CSV:

| category | gold support | F0.5 | generator share vs real |
|---|---:|---:|---|
| SPELL | 387 | **0.5621** | 18.0% vs 3.8% real — heavily over-produced |
| ORTH | 352 | 0.3747 | 8.8% vs 4.9% — over-produced |
| DET | 796 | 0.3693 | 10.0% vs 11.6% — well matched |
| PUNCT | 1478 | 0.2878 | 24.9% vs 17.6% — over-produced |
| PREP | 740 | 0.2733 | 7.4% vs 10.0% — slightly under |
| VERB:TENSE | 473 | 0.2349 | 2.3% vs 6.2% — under-produced |
| OTHER | 980 | 0.1020 | 20.9% vs 13.1% |
| VERB (generic) | 402 | **0.0664** | the generator cannot produce this at all |

The model is best at exactly the categories the generator over-produces and
worst at those it cannot produce. That is direct, per-category evidence for
Wang et al.'s distribution-matching claim, and it makes E4 the most interesting
experiment in the set rather than a formality.

## E5 Indian English set (imported)

| | |
|---|---|
| Sentences | **149**, ids `IN-0001`..`IN-0149` |
| Writers | **20** (`W01`..`W20`), labelled by post |
| Duplicates found | **0** |
| Token length | min 6, median **18**, mean 20.8, max 55 |
| Within the 8-30 guidance | 117 (79%); 22 at 31-40; 6 over 40; 4 under 8 |
| Marked `[sensitive]` | **28** (W02, W08, W10) - not to be quoted in the report |
| Fully annotated | **0** so far; all 149 pending |
| Provenance | public Indian subreddits, copied by hand; no usernames or links stored |

Structural notes recorded at import, for the report's limitations paragraph:
15 sentences have no sentence-final punctuation, 21 begin with a connective and
depend on context outside the sentence, and 6 are long comma-spliced runs. The
agreed annotation rules (D15) cover all three so the annotators handle them
identically rather than disagreeing by convention.

Raw collected text is **gitignored** (`data_manual/manual_sentences.txt`), as
are the two in-progress annotation files. Only the curated TSV is tracked.

## E1 results - the noise-ratio sweep (measured)

Selection is on **val** only. Test is reported, never selected on.

| arm | p | val P | val R | **val F0.5** | test F0.5 | over-corr | edits/sent | copy rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `e1_p0.1` | 0.1 | 0.4523 | 0.1272 | 0.2993 | 0.2986 | 0.112 | 0.48 | 0.661 |
| `e1_p0.2` | 0.2 | 0.4530 | 0.1701 | **0.3399** | 0.3413 | 0.148 | 0.67 | 0.577 |
| `e1_p0.4` | 0.4 | 0.3949 | 0.2168 | 0.3392 | 0.3332 | 0.261 | 1.00 | 0.451 |
| `e1_p0.6` | 0.6 | 0.3383 | 0.2527 | 0.3169 | 0.3044 | 0.372 | 1.38 | 0.351 |
| `e1_p0.2_precomp` | 0.2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**The headline finding.** The noise ratio controls the precision/recall
trade-off almost monotonically: recall rises 0.127 -> 0.170 -> 0.217 -> 0.253
while precision falls 0.452 -> 0.453 -> 0.395 -> 0.338. Because F0.5 weights
precision twice as heavily as recall, the optimum sits at **low-to-moderate
noise** - an inverted U peaking near p = 0.2. Over-correction (0.112 -> 0.372)
and edits per sentence (0.48 -> 1.38) rise monotonically with p, and copy rate
falls (0.661 -> 0.351), all consistent with a model trained on progressively
noisier data becoming progressively more willing to edit.

This is a direct, quantitative answer to Koo et al.'s noise-injection question,
measured under F0.5 rather than BLEU.

**Honest caveat on the selection.** p = 0.2 beats p = 0.4 by **0.0007** val
F0.5. On a single seed that is noise, not a difference. The report must say
"F0.5 is flat between p = 0.2 and p = 0.4, and both clearly beat p = 0.1 and
p = 0.6" rather than "p = 0.2 is optimal". The two are not equivalent in
*character* though: p = 0.2 is the higher-precision, more conservative model
(over-correction 0.148) and p = 0.4 the higher-recall one (0.261). Since both
F0.5 and the over-correction metric favour precision, p = 0.2 was kept as the
selected value and flows into E2/E3/E4.

Val and test agree within 0.009 at every point, so nothing is overfitted to the
selection split.

## Incident: two trainers on one GPU (resolved)

Recorded because it cost real time and the fix is now part of the tooling.

To start the remaining campaign the moment E1 finished, the launcher waited on
the in-flight process with `kill -0 <windows-pid>` **from Git Bash**. MSYS2
cannot see native Windows PIDs, so the check passed immediately and a second
`run_experiments --plan e1` started alongside the first.

| | throughput | training time |
|---|---:|---:|
| `e1_p0.1`, alone | **181 examples/s** | 13.8 min |
| `e1_p0.2`, two trainers | **7.18 examples/s** | **5.8 hours** |

25x slower: two processes at ~2.6 GB peak each on a 4 GB card evict each
other's memory continuously. Both also wrote into
`results/runs/e1_p0.2/seed13`, so that directory was **discarded** and the arm
re-run cleanly in 16.9 min. `e1_p0.1` was unaffected (it finished hours before
the duplicate appeared; `cross_check_delta` 0.0).

Fix: `scripts/run_campaign.py` now calls `already_running()` before any stage
and refuses to start when another `gec.run_experiments` / `gec.train` /
`run_campaign.py` process exists, overridable only with `--allow-concurrent`.
Covered by tests. **Never wait on a native Windows PID from Git Bash.**

## E2 results - the clean:noisy balance (measured)

| run | clean_ratio | val P | val R | **val F0.5** | test F0.5 | over-corr | edits/sent | copy rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `e2_clean0.0` | 0.0 | 0.4530 | 0.1701 | **0.3399** | 0.3413 | 0.1475 | 0.67 | 0.577 |
| `e2_clean0.2` | 0.2 | 0.4546 | 0.1397 | 0.3134 | 0.3046 | 0.1181 | 0.53 | 0.645 |
| `e2_clean0.33` | 0.33 | 0.4508 | 0.1053 | 0.2722 | 0.2747 | 0.1009 | 0.42 | 0.712 |
| `e2_clean0.5` | 0.5 | 0.4582 | 0.0719 | 0.2209 | 0.2252 | 0.0690 | 0.29 | 0.790 |

Selected: **clean_ratio = 0.0** (val F0.5 0.3399).

**The finding, and it is sharp.** Read the three trends together:

- **Precision is flat**: 0.4530, 0.4546, 0.4508, 0.4582. Identity pairs buy
  essentially *no* precision.
- **Recall collapses**: 0.1701 -> 0.0719, less than half.
- **Over-correction falls monotonically**: 0.1475 -> 0.0690, halved.
- **F0.5 falls monotonically**: 0.3399 -> 0.2209.

So Koo et al.'s "balanced clean/noisy data" claim is **half right, and the
wrong half matters**. Balancing does reduce over-correction exactly as they
report - it halves it - but it does so purely by making the model *refuse to
edit* (copy rate 0.577 -> 0.790, edits per sentence 0.67 -> 0.29), not by
making its edits better. Precision does not move. Under F0.5 that is a straight
loss.

**Report-worthy consequence:** over-correction and F0.5 trade off directly
here, so a study that reports only over-correction - or only BLEU, which barely
moves either (see E1) - would present a net-harmful intervention as a
beneficial one. This is E0's argument arriving from a second direction.

Spec section 7.3 predicted over-correction would fall as `clean_ratio` rose,
and it does, monotonically: the prediction **held**. But the spec's framing
treats that as desirable, and measured under F0.5 it is not.

## E3 results - the main comparison (measured, 3 seeds: 13, 42, 2026)

This is the project's research question: **synthetic-only vs real-only vs
mixed, same task, same metric.** All four conditions share the selected
generator settings (p = 0.2, clean_ratio = 0.0) and differ only in training
data. At seed 13 `e3_synthetic` is an alias of the E2 winner (no retraining);
at seeds 42 and 2026 it trains, because aliasing a seed-13 checkpoint under
another seed's label would have produced three identical "seeds".

Mean +/- sample sd over the three seeds, from
`results/tables/summary_mean_std.csv`:

| condition | train data | **test F0.5** | test P | test R | indian F0.5 |
|---|---|---:|---:|---:|---:|
| `e3_synthetic` | 50k synthetic | **0.3366** +/- 0.0093 | 0.4421 | 0.1722 | 0.2149 +/- 0.0067 |
| `e3_real` | 32.3k W&I | **0.4579** +/- 0.0014 | 0.5285 | 0.2984 | 0.3544 +/- 0.0005 |
| `e3_mixed` | 82.3k both | **0.4613** +/- 0.0033 | 0.5223 | 0.3144 | 0.3439 +/- 0.0215 |
| `e3_synth2real` | synth -> 32.3k W&I | **0.4452** +/- 0.0024 | 0.5211 | 0.2814 | 0.3170 +/- 0.0057 |

Per-seed test F0.5, because the spread is itself a finding:

| condition | seed 13 | seed 42 | seed 2026 | spread |
|---|---:|---:|---:|---:|
| `e3_synthetic` | 0.3413 | 0.3259 | 0.3425 | **0.0166** |
| `e3_real` | 0.4563 | 0.4588 | 0.4585 | 0.0025 |
| `e3_mixed` | 0.4580 | 0.4645 | 0.4613 | 0.0065 |
| `e3_synth2real` | 0.4440 | 0.4437 | 0.4480 | 0.0043 |

**Paired** differences against `e3_real` - same seed, so the same init and the
same data order, which removes the variance common to both conditions:

| comparison | seed 13 | seed 42 | seed 2026 | mean | same sign |
|---|---:|---:|---:|---:|---|
| synthetic - real (test) | -0.1150 | -0.1329 | -0.1160 | **-0.1213** | yes |
| mixed - real (test) | +0.0017 | +0.0057 | +0.0028 | **+0.0034** | yes |
| synth->real - real (test) | -0.0123 | -0.0151 | -0.0105 | **-0.0126** | yes |
| mixed - real (**indian**) | +0.0131 | -0.0160 | -0.0286 | **-0.0105** | **no** |

**Answer to the research question, stated carefully.**

1. **Synthetic data cannot replace real data.** Synthetic-only loses a quarter
   of its F0.5 (**-0.1213 +/- 0.0101**, -26.5%), consistent across all three
   seeds and **7.3x** the largest per-condition seed spread. The loss is almost
   entirely **recall**: 0.172 vs 0.298. Precision holds up far better (0.442 vs
   0.529). The synthetic-only model edits half as often (0.66 vs 0.96
   edits/sentence, copy rate 0.577 vs 0.495) and so simply misses most real
   learner errors. This is the project's firmest result.
2. **Mixing adds a consistent but very small in-domain gain.** Paired, mixed
   beats real-only on **all three seeds**: +0.0017, +0.0057, +0.0028, mean
   **+0.0034 +/- 0.0021** (+0.7%). Three of three agreeing in direction is a
   one-sided sign test p = 0.125 - too few seeds to call significant, but not
   nothing either. The defensible statement is "consistent in direction, and
   small enough that 2.5x the data and 2.3x the training time buy under one
   point of F0.5". Note the *unpaired* comparison would have hidden this
   entirely: mixed's own seed spread (0.0065) is twice the effect.
3. **Pre-training on synthetic then fine-tuning on real is reliably worse**
   than just training on real (**-0.0126 +/- 0.0023**, -2.8%, all three seeds).
   This is the one result that contradicts common practice directly, and the
   most likely explanation is scale: at 60.5M parameters and 3 epochs there is
   no capacity shortage for 32.3k real pairs, so stage 1 only installs a bias
   the fine-tune has to undo.
4. **Out of domain, mixing is worse on average and far less stable.** The
   seed-13 result - mixed beating real-only on Indian English by +0.0131 - did
   **not** replicate. It reversed at both other seeds (-0.0160, -0.0286) for a
   mean of **-0.0105**. Worse, `e3_mixed` has an Indian English spread of
   **0.0417** against real-only's **0.0009**: adding synthetic data made
   out-of-domain behaviour **43x more variable**. Every condition scores far
   lower on Indian English than on W&I (0.354 vs 0.461 at best), which is a
   domain-shift result that holds for the real-only model too.

**The methodological finding, which may be the most useful one here.** Had this
project stopped at seed 13 - as the original trimmed budget planned (deviation
D6) - it would have reported "synthetic data helps out-of-domain
generalisation" with a +3.7% number attached. Three seeds show the opposite
sign and a 43x variance penalty. The two claims the single seed got wrong were
precisely the two small ones; the large effects (-0.1213, -0.0126) were right
at seed 13 and stayed right. **Effect sizes below roughly 0.02 F0.5 are not
measurable at this scale**, and that threshold is now measured rather than
asserted.

**Why mixed gains recall but not much F0.5.** Mixed has the highest recall of
all four (0.3144 vs 0.2984) and the highest over-correction (0.1579 vs 0.1417).
It edits more, catching more real errors and also making more spurious ones.
Under F0.5 those nearly cancel. A recall-weighted metric would have reported
mixed as a clear winner - the third time in this project that the choice of
metric decides the conclusion.

## E4 results - distribution matching (measured, seed 13)

Wang et al.'s claim is that synthetic data helps when its error-type
distribution matches the real one. E4 tests it directly: measure the real
distribution from `train_real.m2`, measure the generator's, reweight, repeat.

### The matching loop worked

| iter | TVD vs real | chosen |
|---|---:|---|
| 0 (default weights) | 0.2502 | |
| 1 | 0.1280 | |
| 2 | **0.1146** | **yes** |

Per-category shares, renormalised over the 10 covered categories
(`results/tables/e4_distribution.csv`):

| category | real | synthetic before | synthetic after |
|---|---:|---:|---:|
| PUNCT | 0.227 | 0.272 | 0.234 |
| OTHER | 0.170 | 0.051 | 0.106 |
| DET | 0.150 | 0.145 | 0.210 |
| PREP | 0.129 | 0.101 | 0.177 |
| VERB:TENSE | 0.080 | 0.035 | 0.074 |
| ORTH | 0.063 | 0.098 | 0.053 |
| NOUN:NUM | 0.054 | 0.061 | 0.053 |
| SPELL | 0.049 | 0.214 | 0.039 |
| VERB:FORM | 0.047 | 0.012 | 0.032 |
| VERB:SVA | 0.030 | 0.011 | 0.022 |

The loop fixed the two largest discrepancies: SPELL was **4.3x
over-represented** (0.214 vs 0.049) and is now right, and the verb categories
roughly tripled toward their real shares. It over-shot DET and PREP in doing
so. **TVD more than halved (-54%).**

A ceiling the loop cannot touch: TVD is computed over the covered categories
only, because **22.6% of real W&I edits are in categories the generator cannot
produce at all** (WO, MORPH, NOUN:POSS, CONTR, VERB:INFL). Matching can change
the mix inside its 10 categories; it cannot add the missing five.

### Matching the distribution made the model worse

| run | weights | val F0.5 | test F0.5 | test P | test R | indian F0.5 |
|---|---|---:|---:|---:|---:|---:|
| `e3_synthetic` | default | 0.3399 | **0.3413** | 0.4478 | 0.1749 | 0.2078 |
| `e4_synthetic_matched` | matched (TVD 0.1146) | 0.3109 | **0.3194** | 0.4623 | 0.1429 | **0.2194** |
| `e3_mixed` | default | 0.4705 | **0.4580** | 0.5195 | 0.3110 | **0.3678** |
| `e4_mixed_matched` | matched (TVD 0.1146) | 0.4744 | **0.4472** | 0.5123 | 0.2966 | 0.3556 |

Matching costs test F0.5 in **both** arms:

| arm | default | matched | delta | relative |
|---|---:|---:|---:|---:|
| synthetic-only | 0.3413 | 0.3194 | **-0.0219** | -6.4% |
| mixed | 0.4580 | 0.4472 | **-0.0108** | -2.4% |

In both arms the mechanism is **recall**: 0.175 -> 0.143 synthetic-only and
0.311 -> 0.297 mixed, while precision barely moves (it even rises, 0.448 ->
0.462, in the synthetic-only arm). Halving the distance to the real error
distribution made both models **more conservative and less useful**. The
damage is roughly halved when real data is present, which is what you would
expect if the real half of the mixture is carrying the model regardless.

One honest wrinkle: `e4_mixed_matched` is **better on val** (0.4744 vs 0.4705)
and worse on test (0.4472 vs 0.4580). Val and test agreed to within 0.009 at
every E1 and E2 point, so a 0.015 reversal here is a reminder that these
single-seed gaps sit close to the noise floor. It does not change the sign of
the synthetic-only result, which is twice as large and consistent across both
splits.

**This is the project's clearest negative result**, and it is the third
independent one pointing the same way:

- E1: flattening the distribution toward uniform noise also hurt (-0.027 at
  p = 0.2, `e1_p0.2_precomp`).
- E3: synthetic data neither replaced real data nor supplemented it.
- E4: matching the real distribution did not rescue it, in either arm.

Taken together, **the generator's error-type mix is not the binding
constraint.** Three separate interventions on that mix - flatten it, match it,
tune the noise ratio - all moved F0.5 by less than they moved precision/recall
balance, and none closed the gap to real data. What remains as the likely
limit is the *text*: corrupted Wikipedia prose has the wrong sentence
structure, vocabulary and error co-occurrence pattern for learner writing,
regardless of which error types are injected into it. Wang et al.'s claim does
not reproduce at this scale, and the per-category table shows why that is not
a matching failure: the match succeeded and the model still got worse.

**The Indian English picture, and why it must not be reported as a finding.**
Matching *improved* Indian English in the synthetic-only arm (0.2078 ->
0.2194, +5.6%) while hurting W&I, and **did not** reproduce in the mixed arm
(0.3678 -> 0.3556, -3.3%, worse on both splits).

Both E4 arms are **seed 13 only**, and E3's three seeds have since measured
what that is worth out of domain: `e3_mixed` moves **0.0417** across seeds on
the Indian English split, and E3's own seed-13 Indian result (mixed beating
real by +0.0131) **reversed at both other seeds**. The +0.0116 that matching
appears to gain for synthetic-only is a quarter of that spread.

So the honest statement is that **E4 says nothing measurable about Indian
English in either direction** - not "matching helps out of domain when there is
no real data". Confirming that would need E4 at three seeds, which is ~1.5 h of
GPU time and is listed as the first follow-on below. What E4 *does* support,
because both arms agree and the effect is larger, is the in-domain penalty:
-0.0219 and -0.0108, against an in-domain seed spread of 0.0025-0.0166.

## E5 results - Indian English (measured, seed 13)

149 sentences hand-collected by the group from public Indian Reddit posts and
annotated by **one** annotator (65 of them, 44%, judged acceptable as written
and left unchanged). Scored with the same ERRANT pipeline as W&I.

| model | P | R | F0.5 | over-correction on the 65 control sentences |
|---|---:|---:|---:|---:|
| `e3_mixed` | 0.3613 | 0.3966 | **0.3678** | 0.3538 |
| `e4_mixed_matched` | 0.3490 | 0.3851 | **0.3556** | 0.4000 |
| `e3_real` | 0.3543 | 0.3563 | **0.3547** | 0.4154 |
| `e3_synth2real` | 0.3200 | 0.3218 | **0.3204** | 0.3385 |
| `e4_synthetic_matched` | 0.2414 | 0.1609 | **0.2194** | 0.2769 |
| `e3_synthetic` | 0.2168 | 0.1782 | **0.2078** | 0.3538 |

Three things to report, in order of how much they matter.

**1. Every model is much worse out of domain.** The best Indian English score
(0.3678) is **20% below** the same model's W&I test score (0.4580), and the
ordering of conditions is otherwise preserved. Whatever these models learned,
it transfers only partly to a variety of English absent from their training
data. This holds for the real-only model too, so it is a domain-shift result,
not a synthetic-data result.

**2. The real-only model is the most prescriptive one.** On the 65 sentences
the annotator judged acceptable as written, `e3_real` changed **41.5%** of
them - the highest of any model, and 12 points above `e3_synthetic` (35.4%)
and 14 above `e4_synthetic_matched` (27.7%). Trained exclusively on
British/American learner writing corrected toward a standard, it imposes that
standard on usage a speaker of the variety accepts. The synthetic-trained
models are less prescriptive mostly because they edit less overall, so this is
a caution rather than a virtue - but the direction is worth stating, because
"train on real learner data" is normally assumed to be the safe choice and on
this axis it is the least safe one.

**3. Mixed leads, but only just, and matching does not help here.**
`e3_mixed` beats `e3_real` by **+0.0131** (+3.7%), the one place in the whole
study where adding synthetic data shows a visible gain. But
`e4_mixed_matched` (0.3556) falls *below* `e3_real`, so the gain does not
survive reweighting the generator toward the real distribution. Single seed,
small margins: an observation, not a finding, until the extra E3 seeds land.

**Caveats that belong in the report.** One annotator, so Cohen's kappa is
undefined (spec section 5.5 wants two; `kappa.py` reports `None` with an
explanation rather than a number). 149 sentences is a small set, and all 149
come from one source type (`reddit-manual`), so this measures written online
Indian English, not the variety as a whole.

## Latency (spec section 7.2, measured on `e3_real`)

200 test sentences, `results/tables/latency.csv`:

| device | batch | decode | ms/sentence | sentences/s | peak VRAM |
|---|---:|---|---:|---:|---:|
| cuda | 32 | greedy | **18.0** | 55.6 | 368 MB |
| cuda | 32 | beam 4 | **27.8** | 36.0 | 781 MB |
| cuda | 1 | greedy | 271.5 | 3.7 | 247 MB |
| cuda | 1 | beam 4 | 323.0 | 3.1 | 274 MB |
| cpu | 32 | greedy | 87.7 | 11.4 | - |
| cpu | 32 | beam 4 | 191.8 | 5.2 | - |
| cpu | 1 | greedy | 421.6 | 2.4 | - |
| cpu | 1 | beam 4 | 733.5 | 1.4 | - |

Batching is worth far more than the device: **batch 32 on CPU (87.7 ms) beats
batch 1 on the GPU (271.5 ms)** by 3x. Beam 4 costs 54% over greedy on GPU and
119% on CPU. At batch 1 - the interactive case the demo actually runs - a
60.5M-parameter T5 needs ~0.3 s per sentence on this GPU, which is acceptable
for a web demo and is the number to quote for deployment, not the batched one.

## Campaign complete

`scripts/run_campaign.py` finished all eleven stages in **316 min (5.3 h)**,
exit code 0, no failed stage.

| stage | minutes |
|---|---:|
| e1 (5 arms) | 84.8 |
| select1 | 0.0 |
| e2 (3 new + 1 alias) | 17.6 |
| select2 | 0.0 |
| e3 seed 13 (4 conditions) | 68.2 |
| distribution (E4 matching, CPU) | 3.8 |
| e4 (2 arms) | 45.3 |
| e5 | 0.6 |
| latency | 7.5 |
| aggregate | 0.0 |
| e3_seeds (42, 2026 - 8 runs) | 173.1 |
| aggregate2 | 0.0 |

**24 of 24 expected runs complete. 6 of 6 figures. 0 warnings.**
`aggregate --strict` exits 0, `scripts/audit_results.py` verifies **200
numbers** against the per-seed `metrics.json` files, and `pytest` passes 403
tests.

Measured throughput held at 170-186 examples/s for every run except the one
CPU-starved by my own concurrent test runs (see the incident below), so the
8 h estimate for the full spec-faithful grid was accurate to within ~30 min
once that 38-minute loss is subtracted.

## Audit: every RESULTS.md number re-checked independently

`scripts/audit_results.py` re-reads the rendered `results/RESULTS.md`, loads
each run's `metrics.json` (following `alias_of` pointers), and compares every
cell - **without going through `aggregate.py`**, so a bug in the report writer
cannot hide behind the report writer. It also checks that every linked figure
and table exists and that the completeness claim matches the directories on
disk.

```
$ python scripts/audit_results.py
[audit] checked 126 reported number(s) against metrics.json
[audit] OK: every number in RESULTS.md matches the run that produced it
```

An auditor that cannot fail would be worse than none at all, so
`tests/test_audit_results.py` doctors a single cell (F0.5, precision, BLEU,
over-correction in turn), deletes a `metrics.json`, and breaks a link, and
requires a non-zero exit each time. `TBD` cells are accepted: an unfinished run
rendering as `TBD` is honest, not a mismatch.

## Demo verified against the real checkpoints

Phase 10 was accepted earlier with **zero** checkpoints (the graceful-fallback
requirement). Re-checked now that `e3_synthetic`, `e3_real` and `e3_mixed` have
weights, all three columns load and answer:

```
SRC: He go to school yesterday and buy a apple.
  Synthetic-only   He went to school yesterday and bought an apple.   [3 edits, 1602 ms]
  Real-only        He went to school yesterday and bought an apple.   [3 edits,  808 ms]
  Mixed            He went to school yesterday and bought an apple.   [3 edits,  788 ms]

SRC: What is your good name?
  Synthetic-only   What is your good name?   [0 edits, 231 ms]
  Real-only        What is your good name?   [0 edits, 238 ms]
  Mixed            What is your good name?   [0 edits, 233 ms]
```

Two things worth showing in the viva. All three models fix the spec section
1.3 example completely (SVA, tense, determiner), **including synthetic-only** -
its 25% F0.5 deficit is not visible on an easy sentence, which is why the
project measures F0.5 over 4,384 sentences rather than demoing. And all three
leave "What is your good name?" untouched, an Indian English usage a
prescriptive corrector might well have rewritten.

Added `app/app.py --device {auto,cpu,cuda}`. The config says `device: auto`, so
opening the demo during a run would put three more 250 MB models on the 4 GB
card - the contention that measured 25x slower once already. The first
numbers above were produced with `--device cpu` while seed 42 was training.

## Reproducibility note on the extra E3 seeds

`run_experiments` printed, correctly, at the start of the seed 42 / 2026 stage:

```
[run] WARNING: the working tree has 50 uncommitted change(s);
      these runs will not be exactly reproducible.
```

Checked rather than waved away. `results/runs/e3_synthetic/seed42/env.json`
records commit `8847032` with `dirty: true` and names all 50 files: **49 are
results artefacts the campaign had just written** (per-run `metrics.json`,
ERRANT logs, per-category CSVs, the E4 tables) plus `PROGRESS.md` and the
generated `configs/experiments/e4_matched_weights.yaml`. **No file under
`src/`, `configs/base.yaml` or `configs/experiments/e3_*.yaml` was modified**,
so the training, prediction and scoring code at those runs was exactly the
committed code.

Verified independently rather than by inspection of the file list: the seed 42
run's own `config.yaml` is **byte-identical** to seed 13's once the seed and
run name are removed, including the `${select.*}`-resolved `p = 0.2` and
`clean_ratio = 0.0`. The extra seeds are a clean replicate of the same
condition, which is the only property the error bars depend on.

Also logged by the tool and expected: `e3_synthetic` cannot alias
`e2_clean0.0/seed13` at seeds 42 and 2026 (the target is a seed-13 run), so it
trains instead. That is the alias guard working - reusing a seed-13 checkpoint
under a seed-42 label would have silently produced three identical "seeds".

## Incident: CPU starvation of a GPU run (resolved, no data affected)

`e3_synthetic/seed2026` trained at **66.1 examples/s** against the ~175 ex/s
every comparable run achieved - 57 minutes of wall clock for a 19-minute job.

| run | examples/s | train time |
|---|---:|---:|
| `e3_synthetic` seed 42 | 174.5 | 14.3 min |
| `e3_synthetic` seed 2026 | **66.1** | **37.8 min** |
| `e3_real` seed 42 (next run) | 185.7 | 8.7 min |

Cause: `dataloader_num_workers: 0` (set for Windows), so the trainer tokenises
and collates every batch in its own main process. During that window the
machine was also running the full pytest suite repeatedly, re-running
`aggregate`, and - worst of it - loading three `t5-small` models on **CPU** for
a demo smoke test with beam 4. The GPU was idle waiting for batches.

Throughput recovered to 185 ex/s on the very next run with no config change,
which is what identifies the cause. **No numbers are affected**: seed 2026
scored test F0.5 0.3425, in line with seeds 13 (0.3413) and 42 (0.3259). The
cost was ~38 minutes of GPU time.

Same lesson as the two-trainers incident above, one level down the stack: on a
single-worker dataloader, *CPU* contention throttles a *GPU* run, and the
symptom (slow training) points nowhere near the cause. Rule adopted: no test
suites, no `aggregate`, and no CPU model loading while a run is in flight -
`app/app.py --device cpu` is for when the GPU is busy and the CPU is **not**.

## Open issues

Current as of **2 November 2026**, after the full campaign finished.
Nothing below blocks the submission.

### Still open

- [ ] **A second independent annotator for the Indian English set.** Annotator 1 has
      corrected all 149 rows and is merged in, so E5 scores normally against that single
      reference. Only **Cohen's kappa** needs the second column (spec section 5.5), and
      `kappa.py` reports it as undefined with an explanation rather than printing a
      meaningless number. No rework when it arrives: annotator 2 fills
      `data_manual/annotation/annotator2.tsv`, then
      `--merge-annotations` -> `python -m gec.kappa` -> `python -m gec.aggregate`.
- [ ] **E4 at three seeds (~1.5 h GPU) is the highest-value follow-on.** Both E4 arms are
      seed 13 only, and E3's three seeds measured the noise floor at ~0.02 F0.5. E4's
      in-domain penalties (-0.0219 synthetic-only, -0.0108 mixed) sit either side of it,
      and its Indian English claim is a quarter of `e3_mixed`'s 0.0417 out-of-domain
      spread - which is why that claim is withdrawn rather than reported.
      `--plan e4 --seeds 42 2026 --yes` adds them without redoing anything.
- [ ] **E1, E2 and E5 remain single-seed**, as spec section 7's budget mode prescribes.
      E1's p = 0.2 vs p = 0.4 (0.0007 apart) was already called noise in the E1 section.
      E1 and E2's headline trends are monotonic across four points each and much larger
      than the spread, so extra seeds would not change them.
- [ ] **Verb errors are structurally hard for the generator to produce**: 15.8% of real
      W&I edits vs 4.2% synthetic, and only ~4.5% even after pre-compensation. Worth one
      paragraph in the report - part of the synthetic-only shortfall is a generator
      coverage limit, not evidence about synthetic data in principle.
- [ ] **22.6% of real W&I edits are in categories the generator cannot produce at all**
      (WO, MORPH, NOUN:POSS, CONTR, VERB:INFL). That ceiling bounds what synthetic-only
      training can reach here, and it is also why E4's TVD is computed over the ten
      covered categories only.

### Closed

- [x] **Stray `CLAUDE.md` / `PROJECT_SPEC.md` copies in `Downloads/`** - deleted manually
      by the group. The copy at the `Downloads` root had been applying this project's
      instructions to the entire Downloads tree.
- [x] **The Indian English set exists**: 149 sentences, hand-collected by the group from
      public Indian subreddits, no usernames or links stored. A harvesting collector was
      considered and **dropped** - `SocialGrep/one-year-of-r-india` ships only a loading
      script whose data host (`exports.socialgrep.com`) no longer resolves, and
      `datasets 4.0.0` cannot execute remote loading scripts anyway. The set stays
      group-written, exactly as spec section 5.5 intends.
- [x] **GPU campaign approved** after the `--timing-probe` numbers were restated:
      161.5 examples/s at batch 16 x accum 2, so the budget trims were withdrawn and the
      campaign ran spec-faithful (50k pairs, 3 epochs).
- [x] **E3 at three seeds** (13, 42, 2026). Both differences previously flagged as
      fragile were inside the spread, and one *reversed*: mixed vs real on Indian English
      went +0.0131, -0.0160, -0.0286. Measured threshold: effects below ~0.02 F0.5 are
      not resolvable at this scale. See the E3 section. Deviation D6 withdrawn.
- [x] **Every number in `RESULTS.md` is independently re-checked** -
      `scripts/audit_results.py` verifies 200 of them against the per-seed
      `metrics.json` files without going through `aggregate.py`.
