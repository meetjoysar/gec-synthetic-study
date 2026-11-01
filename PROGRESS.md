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
| 5 | Colab notebook | in progress |
| 6 | E1–E3 runs (GPU) | blocked on the timing probe + time approval |
| 7 | E4 distribution matching | `distribution.py` **DONE**; runs await GPU |
| 8 | Indian English: template, kappa, E5 | template + `kappa.py` **DONE**; E5 awaits the group's sentences |
| 9 | Aggregation: figures, tables, `RESULTS.md` | **DONE** (skeleton verified all-TBD); figures fill in as runs land |
| 10 | Gradio demo | TODO |
| 11 | Docs: `README.md`, `CREDITS.md` | TODO |

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
| D1 | Repo named `gec-synthetic-study/` (§4) | Repo root **is** `IA/gec-synthetic-study/`; `CLAUDE.md`, `PROJECT_SPEC.md`, `IA1_REPORT_GEC.md` are **copies**, the originals stay in `IA/` | User chose the subfolder layout. The repo must be self-contained to be reproducible. |
| D2 | `errant` "installs spaCy and `en_core_web_sm`" (§3) | `python -m spacy download en_core_web_sm` run separately | **The spec is wrong for errant 3.x.** `errant==3.0.2` depends only on `spacy>=3.2,<4` and `rapidfuzz`; it does not pull the model. Also means the feared spaCy version conflict does not exist. |
| D3 | — | Added `src/gec/errant_cli.py` and `src/gec/repro.py` beyond the §4 module list | Each has >=3 consumers. Keeping ERRANT behind a subprocess wrapper means `train.py` never imports spaCy. |
| ~~D4~~ | `data.synthetic_n: 50000`, `epochs: 3` (§8.1) | **WITHDRAWN** — spec values restored | Was trimmed to 30000/2 on an *analytical* estimate of 55 examples/s. The timing probe measured **161.5/s** (2.9× better) after batch tuning, so the full setting is affordable. No longer a deviation. |
| ~~D5~~ | E1 sweeps `p ∈ {0.1, 0.2, 0.4, 0.6}` (§7) | **WITHDRAWN** — all four points restored | Same reason as D4. E1 now runs 5 arms: the spec's four noise points plus the pre-compensated arm (D10). |
| D6 | E3 uses seeds `[13, 42, 2026]` (§7) | E3 uses **seed 13** for now | The only budget cut still standing. Figure 4 carries `n=1` and no error bars, stated plainly in the report and annotated by `aggregate.py`. The extra seeds are a **resumable follow-on** (~2 h): `--plan e3 --seeds 42 2026 --yes` adds them without redoing anything. |
| D7 | WikiText-103 full train split (§5.4) | `streaming=True` with early stop at 60k sentences (~60–90 MB instead of ~250 MB) | User-confirmed. We need 60k of ~4M sentences; streaming avoids the §2.4 download threshold entirely. |
| D8 | `datasets` implied for the training path (§3) | Plain `torch.utils.data.Dataset` + `DataCollatorForSeq2Seq`; `datasets` only used for WikiText | Decouples training from Arrow/`huggingface-hub` version churn. `datasets>=4.3` requires `huggingface-hub>=1.31` while `transformers 4.53.1` requires `<1.0`. |
| D9 | Phase order 0–11 (§10) | Indian-English template moved to **after Phase 2**; `aggregate.py` skeleton moved to **before the first GPU run** | The 150 hand-written sentences are the longest-lead *human* task and must start early. Building the TBD path first verifies it in the hardest case (everything missing). |
| D10 | E1 sweeps `p` only, with one fixed set of generator weights (§6.2, §7) | **Extra E1 arm `e1_p0.2_precomp`**: same `p`, seed and base-set sentences as `e1_p0.2`, but with generator weights pre-compensated for applicability gating | **User-requested.** §6.2's weights are uniform across ERRANT types *as requested*, but each op's `applies` check gates whether it can fire, so the **realised** mix is badly skewed (measured below). The extra arm separates two questions the default weights conflate: "is there a best noise level?" and "does the generator's skewed realised distribution hold synthetic-only back?". Costs ~30 min of GPU time. |
| D11 | — | Predictions are re-tokenised with spaCy before scoring (`predict.retokenise`) | T5's pretraining pulls its output toward ordinary detokenised English, so it emits `information.` where the source had `information .` — **48 of 50** smoke predictions needed it. ERRANT cannot distinguish that surface convention from a real `PUNCT` edit, so left alone it manufactures a spurious edit on nearly every sentence. Spec §5.3 already requires predictions to be in tokenised form. |
| D12 | §5.4 keeps sentences with "no unusual characters (mostly ASCII…)" | Clean-pool charset filter rejects a sentence if ASCII folding **destroys** information (introduces a `?`), rather than applying a 95% character ratio | A ratio test is the wrong instrument: a long English sentence with one embedded Greek or CJK word clears 95% easily and still leaves a `??????????` token in the training data. Accented Latin folds cleanly (`café`→`cafe`) and is kept. Junk with ≥2 `?` fell from 326 to **1** of 60,000. |
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

## Open issues

- [ ] `Downloads/CLAUDE.md` and `Downloads/PROJECT_SPEC.md` are stray byte-identical copies.
      The one at the `Downloads` root applies this project's instructions to the **entire**
      Downloads tree. Worth deleting — **ask the user first** (spec §2.4).
- [ ] The group must write ~150 Indian English sentences with two independent annotators.
      Template ships in Phase 8a; E5 stays `TBD` until >= 20 real rows exist.
- [ ] GPU campaign not yet approved with measured numbers. The `--timing-probe` run comes
      first, then the hours get restated before anything long starts.
- [ ] Verb errors are structurally hard for the generator to produce (15.8% real vs 4.2%
      synthetic, and only ~4.5% even after pre-compensation). Worth one paragraph in the
      report: part of any synthetic-only shortfall is a generator-coverage limit, not
      evidence about synthetic data in principle.
- [ ] The generator covers 10 ERRANT categories; **22.6% of real W&I edits are in
      categories it cannot produce at all** (WO, MORPH, NOUN:POSS, CONTR, VERB:INFL).
      That ceiling bounds what synthetic-only training can achieve here.
