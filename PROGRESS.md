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
| 1 | Data: `m2.py`, `tok.py`, `prepare_data.py` | in progress |
| 2 | Generator: `error_gen.py` | TODO |
| 3 | Evaluation + E0 | TODO |
| 4 | Training: `build_datasets.py`, `train.py`, `predict.py`, `run_experiments.py` | TODO |
| 5 | Colab notebook | TODO |
| 6 | E1–E3 runs (GPU) | TODO |
| 7 | E4 distribution matching (GPU) | TODO |
| 8 | Indian English: template, kappa, E5 | TODO |
| 9 | Aggregation: figures, tables, `RESULTS.md` | TODO |
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
| D4 | `data.synthetic_n: 50000`, `epochs: 3` (§8.1) | `synthetic_n: 30000`, `epochs: 2` | **User chose the trimmed ~8 h GPU budget.** The full §7 grid is 24–30 h on this laptop, ~100x the §2.4 "ask before 15 minutes" rule. |
| D5 | E1 sweeps `p ∈ {0.1, 0.2, 0.4, 0.6}` (§7) | `p ∈ {0.1, 0.2, 0.4}` | Same budget decision. The noise curve survives with 3 points. |
| D6 | E3 uses seeds `[13, 42, 2026]` (§7) | E3 uses **seed 13 only** | Same budget decision. Figure 4 will carry `n=1` and no error bars, stated plainly in the report. |
| D7 | WikiText-103 full train split (§5.4) | `streaming=True` with early stop at 60k sentences (~60–90 MB instead of ~250 MB) | User-confirmed. We need 60k of ~4M sentences; streaming avoids the §2.4 download threshold entirely. |
| D8 | `datasets` implied for the training path (§3) | Plain `torch.utils.data.Dataset` + `DataCollatorForSeq2Seq`; `datasets` only used for WikiText | Decouples training from Arrow/`huggingface-hub` version churn. `datasets>=4.3` requires `huggingface-hub>=1.31` while `transformers 4.53.1` requires `<1.0`. |
| D9 | Phase order 0–11 (§10) | Indian-English template moved to **after Phase 2**; `aggregate.py` skeleton moved to **before the first GPU run** | The 150 hand-written sentences are the longest-lead *human* task and must start early. Building the TBD path first verifies it in the hardest case (everything missing). |

---

## Dataset sizes after each preprocessing step

| Step | Count | Source file |
|---|---|---|
| W&I `ABC.train.gold.bea19.m2` blocks | TBD | — |
| → `train_real` | TBD | `data/processed/train_real.stats.json` |
| → `val` (held out, seed 42) | TBD | `data/processed/val.stats.json` |
| W&I `ABCN.dev.gold.bea19.m2` → `test` | TBD | `data/processed/test.stats.json` |
| WikiText-103 clean pool | TBD | `data/processed/clean_pool.stats.json` |
| Indian English (real rows, excl. `EXAMPLE-`) | TBD | `data/processed/indian.stats.json` |

## Hyperparameters actually used

TBD — written per run to `results/runs/<run>/seed<k>/config.yaml` and `env.json`.

## Run times

TBD — measured by `--timing-probe`, then appended to `results/runs/_timing.jsonl`.

## Open issues

- [ ] `Downloads/CLAUDE.md` and `Downloads/PROJECT_SPEC.md` are stray byte-identical copies.
      The one at the `Downloads` root applies this project's instructions to the **entire**
      Downloads tree. Worth deleting — **ask the user first** (spec §2.4).
- [ ] The group must write ~150 Indian English sentences with two independent annotators.
      Template ships in Phase 8a; E5 stays `TBD` until >= 20 real rows exist.
- [ ] GPU campaign not yet approved with measured numbers. The `--timing-probe` run comes
      first, then the hours get restated before anything long starts.
