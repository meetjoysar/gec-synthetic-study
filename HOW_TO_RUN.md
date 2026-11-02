# HOW TO RUN — the demo, and everything else

A plain walkthrough for **"Fake Errors, Real Grammar"** on this machine
(Windows 10, conda env `gec`, RTX 3050 Laptop 4 GB).

Nothing here needs you to understand the research first. Start at the top,
copy the commands as written, and stop when the demo opens.

> **Copy the commands exactly, including `C:\Users\91942\miniconda3\envs\gec\python.exe`.**
> Typing plain `python` runs the *wrong* Python on this machine and nothing will
> work. [Why](#why-the-long-python-path) is explained below — it is the single
> most common way to get stuck.

---

## Start here: three commands

Open **PowerShell**, then:

```powershell
cd C:\Users\91942\Downloads\kjsse\NLP\IA\gec-synthetic-study
$PY = "C:\Users\91942\miniconda3\envs\gec\python.exe"
& $PY app\app.py
```

Wait 30-60 seconds (it loads ERRANT and three models), then open
**<http://127.0.0.1:7860>** in your browser. That's the demo.

To stop it: click in the PowerShell window and press **Ctrl + C**.

If that worked, you can skip to [Using the demo](#using-the-demo). If it didn't,
go through the steps below in order.

---

## What this project is, in five sentences

The question: **does synthetic (fake) error data help a small grammar
corrector, compared with real learner data?** We corrupt clean Wikipedia
sentences with 15 rule-based error operations to make fake "learner" mistakes,
then train the same small model (`t5-small`, 60.5M parameters) three ways —
on fake errors only, on real errors only, and on both mixed. We score all three
with ERRANT's F0.5, the standard grammar-correction metric. The answer is that
**fake data cannot replace real data** (26.5% worse) and **mixing it in buys
almost nothing** (+0.7%). The demo lets you type a sentence and watch those
three models disagree, which is the whole study in one screen.

---

## Step 1 — Open the right terminal

Use **PowerShell** (press `Win`, type "PowerShell", Enter). Not Git Bash, not
cmd.

Then move into the project:

```powershell
cd C:\Users\91942\Downloads\kjsse\NLP\IA\gec-synthetic-study
```

Check you're in the right place — this must list `app`, `src`, `results`:

```powershell
ls
```

---

## Step 2 — Point at the right Python

```powershell
$PY = "C:\Users\91942\miniconda3\envs\gec\python.exe"
& $PY -c "import sys, torch; print(sys.executable); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

**You should see:**

```
C:\Users\91942\miniconda3\envs\gec\python.exe
torch 2.6.0+cu124 cuda True
```

`$PY` only lasts as long as the window is open. Open a new PowerShell → set it
again.

### Why the long Python path

You might expect `conda activate gec` followed by `python`. On **this** machine
that silently fails:

```powershell
conda activate gec        # reports success
python -c "import sys; print(sys.executable)"
# C:\msys64\ucrt64\bin\python.exe      <-- WRONG. This is MSYS2's Python.
```

MSYS2 sits earlier on the PATH and wins. That Python has no torch, no
transformers, and a broken pip — so you'd get `ModuleNotFoundError` and have no
idea why. **Using the full path avoids the problem entirely**, and works whether
or not you activated anything.

### If the env is missing or broken

Only needed on a fresh machine, or if Step 2 failed:

```powershell
& "C:\Users\91942\miniconda3\Scripts\conda.exe" create -y -n gec python=3.10
$PY = "C:\Users\91942\miniconda3\envs\gec\python.exe"
$env:PYTHONNOUSERSITE = "1"
$env:PIP_CONFIG_FILE = "nul"
& $PY -m pip install --upgrade pip setuptools wheel
& $PY -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
& $PY -m pip install -r requirements.txt
& $PY -m spacy download en_core_web_sm
```

The two `$env:` lines are not optional on this machine: a broken pip in the
Windows user site-packages shadows the env's pip, and the global `pip.ini`
points at a dead NVIDIA package index. They each cost hours to diagnose once.

---

## Step 3 — Check the three models are there

The demo needs three trained checkpoints. They exist already:

```powershell
ls results\runs\e3_synthetic\seed13\model\model.safetensors
ls results\runs\e3_real\seed13\model\model.safetensors
ls results\runs\e3_mixed\seed13\model\model.safetensors
```

Each should be about **231 MB** (230.8 MB measured). If any is missing, the demo still starts and
that column says "Not trained yet" with the command to produce it — nothing
crashes.

---

## Step 4 — Start the demo

```powershell
& $PY app\app.py
```

**What you'll see** (30-60 s; loading ERRANT and three models is the slow part, and the GPU path is slower to start than `--device cpu`):

```
[app] checkpoints found: ['Synthetic-only', 'Real-only', 'Mixed']
[app] loading ERRANT...
[app] loaded Synthetic-only on cuda
[app] loaded Real-only on cuda
[app] loaded Mixed on cuda
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```

Open **<http://127.0.0.1:7860>**. Windows may ask to allow Python through the
firewall — you can **Cancel**; the demo is local-only and does not need it.

### Useful variations

| Command | When to use it |
|---|---|
| `& $PY app\app.py` | normal |
| `& $PY app\app.py --device cpu` | **if any training is running.** Keeps the demo off the GPU so it can't fight the trainer for the 4 GB card |
| `& $PY app\app.py --port 7862` | if port 7860 is already taken |
| `& $PY app\app.py --no-preload` | starts in ~5 s; each model loads on first use instead |
| `& $PY app\app.py --share` | creates a temporary public URL (sends your input to Gradio's relay — avoid for the sensitive test sentences) |
| `& $PY app\app.py --help` | lists all of the above |

---

## Using the demo

**Type a sentence with a mistake, press "Correct".** Or click one of the eight
example buttons underneath.

You get **three columns side by side**:

| Column | Trained on | What it tells you |
|---|---|---|
| **Synthetic-only** | 50,000 corrupted Wikipedia sentences | what fake errors alone can teach |
| **Real-only** | 32,305 real learner sentences (W&I+LOCNESS) | the baseline to beat |
| **Mixed** | both, 82,305 total | whether adding fake data helps |

In each column:

- **Green highlight** = something the model *inserted*
- **Red strike-through** = something it *deleted*
- Below the sentence, each change is named in plain English ("subject-verb
  agreement", "wrong verb tense", "missing determiner") rather than as an
  ERRANT code.

### Try these two

```
He go to school yesterday and buy a apple.
```
All three fix it completely — three edits each. Worth knowing: the
synthetic-only model is 26.5% worse *on average over 4,384 sentences*, and this
sentence shows none of that. One easy example proves nothing, which is exactly
why the project measures instead of demos.

```
What is your good name?
```
All three leave it alone. That's a normal Indian English phrasing, and a
prescriptive corrector might well have "fixed" it. Our measurements say the
**real-only** model is the most prescriptive of the three — it rewrites 41.5% of
the Indian English sentences our annotator judged perfectly acceptable.

### Speed

First correction after startup takes ~1.6 s (warm-up). After that, roughly
**0.3 s per sentence** on the GPU, **0.2–0.8 s** on CPU. If it feels slow, that
is beam search with 4 beams, not a problem.

---

## Step 5 — Stop the demo

**Ctrl + C** in the PowerShell window. If the window is gone and the port is
still held:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like "*app\app.py*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

---

## Troubleshooting

Every symptom below actually happened on this machine.

| What you see | What it means | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` (or `gradio`, `transformers`) | you used plain `python`, so MSYS2's Python ran | use `& $PY ...` with the full path (Step 2) |
| `OSError: Cannot find empty port in range: 7860-7860` | the demo is already running in another window | use that window, or `--port 7862`, or kill it (Step 5) |
| Browser says "can't reach this page" | the app hasn't finished loading | wait for the `Running on local URL` line, then refresh |
| A column says "Not trained yet" | that checkpoint is missing | the column tells you the command; or just use the other two |
| `CUDA out of memory` | something else is using the 4 GB GPU | `--device cpu`, or close the other program |
| Demo works but is very slow, and training is running | both are sharing the CPU/GPU | `--device cpu`, and don't run the test suite at the same time |
| `conda activate gec` works but nothing imports | see [Why the long Python path](#why-the-long-python-path) | always use `$PY` |
| `UnicodeDecodeError` from an ERRANT command | the UTF-8 flag didn't reach the subprocess | `$env:PYTHONUTF8 = "1"` then retry |

---

## Other things you can run

None of these are needed for the demo. All assume Step 1 and Step 2 are done.

### Check the project is healthy

```powershell
& $PY -m pytest -q
```
403 tests, about 60 seconds. Everything should pass.

```powershell
& $PY scripts\audit_results.py
```
Re-reads every number in `results\RESULTS.md` straight from the per-run metrics
files and reports mismatches. Should say **200 numbers OK**. This is the script
that backs the claim that no number in the report was invented.

### Rebuild the tables and figures

```powershell
$env:PYTHONPATH = "src"
& $PY -m gec.aggregate
```
Rewrites `results\RESULTS.md`, 6 figures and 12 tables from the saved metrics.
Safe to run any time; it never retrains anything. Add `--strict` to make it exit
non-zero if anything is missing (the pre-submission check).

### Annotator agreement for the Indian English set

```powershell
$env:PYTHONPATH = "src"
& $PY -m gec.kappa
```
Right now it reports that one annotator has filled in all 149 rows and Cohen's
kappa is undefined until a second one does. After a second annotator fills
`data_manual\annotation\annotator2.tsv`:

```powershell
& $PY -m gec.indian_import --merge-annotations
& $PY -m gec.kappa
& $PY -m gec.aggregate
```

### See the error generator at work

```powershell
$env:PYTHONPATH = "src"
& $PY -m gec.error_gen --demo 20
```
Prints 20 clean → corrupted pairs. This is the "fake errors" half of the project
title, and the quickest way to get a feel for what the synthetic data looks like.

### Re-run an experiment (GPU, slow)

**Always dry-run first** — it prints what would run and how long it would take,
without starting anything:

```powershell
$env:PYTHONPATH = "src"
& $PY -m gec.run_experiments --plan e3 --dry-run
```

Everything is already done (24 of 24 runs), so a real run only happens if you
delete results or add seeds. The full campaign took **5.3 hours**. A run longer
than 15 minutes refuses to start without `--yes`, on purpose.

---

## Where everything lives

```
gec-synthetic-study/
├── HOW_TO_RUN.md          <- you are here
├── README.md              the project overview and design decisions
├── PROGRESS.md            the full log: every result, deviation and incident
├── PROJECT_SPEC.md        the specification all of this was built from
├── CREDITS.md             datasets, licences, papers, AI-tool declaration
│
├── app/app.py             the demo
├── app/config.yaml        which checkpoints the demo loads, and its examples
│
├── src/gec/               all the project code (16 modules)
├── tests/                 403 tests
├── scripts/               campaign runner, auditor, config/notebook generators
├── configs/               experiment settings (base + per-experiment layers)
│
├── results/
│   ├── RESULTS.md         every measured number, no interpretation
│   ├── failure_gallery.md 10 sentences where the three models disagree most
│   ├── figures/           6 PNGs for the report
│   ├── tables/            summary.csv, latency.csv, E4/E5 tables, selection.json
│   └── runs/              24 runs: metrics.json, config.yaml, env.json, checkpoints
│
├── data/                  downloaded + preprocessed data (not in git, 466 MB)
└── data_manual/           the 149 hand-collected Indian English sentences
```

**If you only read two files:** `results/RESULTS.md` for the numbers, and the
E3 section of `PROGRESS.md` for what they mean.

---

## Three things not to do

1. **Don't run training and the test suite at the same time.** The dataloader
   uses no worker processes on Windows, so a busy CPU starves the GPU. This
   happened once and cost 38 minutes: one run dropped from 175 to 66
   examples/s.
2. **Don't start a second training run while one is going.** Two processes on a
   4 GB card measured **25× slower** and both wrote to the same directory, so
   the result had to be thrown away. `scripts/run_campaign.py` now refuses to
   start if it detects another trainer.
3. **Don't tune anything on the test set.** Settings are chosen on `val`;
   `aggregate.select_best` has an assertion that refuses to read a test metric
   at all, so this is enforced by code rather than memory.
