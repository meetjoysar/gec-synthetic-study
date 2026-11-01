"""Generate notebooks/colab_run.ipynb (spec section 9).

Written as a generator script rather than hand-edited JSON so the notebook
stays valid and diffable.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("notebooks/colab_run.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md("""# Fake Errors, Real Grammar - Colab runner

NLP IA Activity 2. Runs the experiment grid on a Colab GPU, with results saved
to Google Drive so a disconnect costs at most one run.

**Set expectations before you start.** A free Colab **T4 is compute capability
7.5: no bf16 and no TF32**, so `precision: auto` correctly resolves to fp32
there and a T4 run is roughly as fast as (or slower than) the project's local
RTX 3050 running bf16. Colab's value here is a **second machine** you can run
in parallel, not a speedup. Its 16 GB does let you raise the batch size, which
recovers some throughput through better utilisation.

Run the cells in order. Cell 5 (the smoke test) must pass before any GPU run.
"""),

    md("## 1. Check the GPU"),
    code("""!nvidia-smi

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print("device    ", torch.cuda.get_device_name(0))
    print("capability", cap)
    print("VRAM      ", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")
    # The project gates bf16 on capability >= 8.0, NOT on
    # torch.cuda.is_bf16_supported(): since torch 2.6 that defaults to
    # including_emulation=True and returns True on a T4, where emulated bf16 is
    # far SLOWER than fp32.
    print("real bf16 ", cap[0] >= 8, "| reported", torch.cuda.is_bf16_supported())
else:
    print("No GPU! Runtime > Change runtime type > T4 GPU")"""),

    md("""## 2. Mount Drive and point the project at it

`paths.py` reads `GEC_DATA_DIR`, `GEC_MODELS_DIR` and `GEC_RESULTS_DIR`, so
data, checkpoints and results all live on Drive and survive a disconnect."""),
    code("""from google.colab import drive
drive.mount('/content/drive')

import os, pathlib
BASE = pathlib.Path('/content/drive/MyDrive/gec-synthetic-study')
for sub in ('data', 'models', 'results'):
    (BASE / sub).mkdir(parents=True, exist_ok=True)

os.environ['GEC_DATA_DIR']    = str(BASE / 'data')
os.environ['GEC_MODELS_DIR']  = str(BASE / 'models')
os.environ['GEC_RESULTS_DIR'] = str(BASE / 'results')
# ERRANT opens every file with a bare open(); force UTF-8 for its subprocesses.
os.environ['PYTHONUTF8'] = '1'
# Helps the 4 GB-class allocator; harmless on a 16 GB T4.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

for k in ('GEC_DATA_DIR', 'GEC_MODELS_DIR', 'GEC_RESULTS_DIR'):
    print(k, '=', os.environ[k])"""),

    md("""## 3. Get the code

Either clone the repo, or upload it as a zip. Edit `REPO_URL` if you have
pushed this to GitHub."""),
    code("""REPO_URL = ''   # e.g. 'https://github.com/<user>/gec-synthetic-study.git'

import os, pathlib, sys
WORK = pathlib.Path('/content/gec-synthetic-study')

if REPO_URL:
    if not WORK.exists():
        !git clone $REPO_URL {WORK}
    else:
        !cd {WORK} && git pull --ff-only
else:
    # No remote: upload a zip of the repo (Files pane) and unzip it here.
    if not WORK.exists():
        print('Upload gec-synthetic-study.zip via the Files pane, then run:')
        print('  !unzip -q /content/gec-synthetic-study.zip -d /content')
    else:
        print('using the existing', WORK)

os.chdir(WORK)
sys.path.insert(0, str(WORK / 'src'))
print('cwd =', os.getcwd())"""),

    md("""## 4. Install dependencies

Colab already ships a CUDA torch, so `requirements-colab.txt` deliberately
leaves torch alone. The spaCy model is a separate install: **errant 3.x does
not pull `en_core_web_sm`**, despite what the spec says."""),
    code("""!pip install -q -r requirements-colab.txt
!python -m spacy download en_core_web_sm

# One line proving spaCy + the model + errant + its classifier all cooperate.
!python -c "import errant; a=errant.load('en'); print([(e.o_str,'->',e.c_str,e.type) for e in a.annotate(a.parse('He go to school .'), a.parse('He goes to school .'))])"
# Expected: [('go', '->', 'goes', 'R:VERB:SVA')]"""),

    md("""## 5. Prepare the data

W&I+LOCNESS is ~6 MB. The WikiText pool streams and stops at 60k sentences
(~60-90 MB) instead of downloading the full ~250 MB split. Takes ~12 min; it is
cached on Drive, so later sessions skip it."""),
    code("""!python -m gec.prepare_data --wi
!python -m gec.prepare_data --wikitext
!python -m gec.prepare_data --inspect"""),

    md("""## 6. Tests and the smoke plan

**The smoke test must pass before any GPU run** (spec 8.3). It runs the whole
pipeline on CPU in a few minutes with 300 training pairs and `max_steps=20`."""),
    code("""!python -m pytest -q
!python -m gec.run_experiments --plan smoke --yes"""),

    md("""## 7. Timing probe

Measures throughput so the budget stops being guesswork: ~10 minutes, which is
under the project's 15-minute consent rule. Everything after this is estimated
from measured numbers."""),
    code("""!python -m gec.run_experiments --plan e1 --timing-probe

import json, pathlib
p = pathlib.Path(os.environ['GEC_RESULTS_DIR']) / 'tables' / 'timing_probe.json'
print(json.dumps(json.loads(p.read_text()), indent=2))"""),

    md("""## 8. Inspect the plan before committing time

`--dry-run` prints every job with its status, resolved dataset spec and
estimated minutes. Make this your habitual first command: it is how a
multi-hour commitment gets inspected before it starts."""),
    code("""!python -m gec.run_experiments --plan all --dry-run"""),

    md("""## 9. Raise the batch size for a 16 GB T4 (optional)

The defaults target a 4 GB laptop GPU. On a T4 a bigger batch means fewer
optimiser steps and better utilisation. fp32 is correct on a T4 - do not force
bf16, which would be emulated and slower."""),
    code("""BIG_BATCH = ['--set', 'train_batch_size=32', 'gradient_accumulation_steps=1',
             'eval_batch_size=64']
print('append these to any run command:', ' '.join(BIG_BATCH))"""),

    md("""## 10. The experiment plans

One cell per plan, in dependency order. Each is **resumable**: a job whose
`metrics.json` is complete is skipped, so re-running a cell after a disconnect
only redoes what is missing.

`--yes` is required because every real plan exceeds the 15-minute budget gate.
Selection between plans is automatic: E2 reads E1's best `p`, and E3/E4 read
E2's best `clean_ratio`, both from **val** F0.5 only."""),
    code("""# E0 - copy baseline, no training, no GPU. BLEU looks high, F0.5 is 0.
!python -m gec.run_experiments --plan e0 --yes"""),
    code("""# E1 - noise-ratio sweep (4 arms incl. the pre-compensated one)
!python -m gec.run_experiments --plan e1 --yes --set train_batch_size=32 gradient_accumulation_steps=1 eval_batch_size=64"""),
    code("""# E2 - clean:noisy balance at the best p from E1
!python -m gec.run_experiments --plan e2 --yes --set train_batch_size=32 gradient_accumulation_steps=1 eval_batch_size=64"""),
    code("""# E3 - the main comparison: synthetic vs real vs mixed vs synth->real
!python -m gec.run_experiments --plan e3 --yes --set train_batch_size=32 gradient_accumulation_steps=1 eval_batch_size=64"""),
    code("""# E4 - distribution matching (the CPU matching loop, then 2 runs)
!python -m gec.distribution --match
!python -m gec.run_experiments --plan e4 --yes --set train_batch_size=32 gradient_accumulation_steps=1 eval_batch_size=64"""),
    code("""# E5 - Indian English, once the group has written >= 20 real rows.
# Skips gracefully until then.
!python -m gec.kappa
!python -m gec.prepare_data --indian
!python -m gec.run_experiments --plan e5 --yes"""),

    md("""## 11. Latency table (spec 7.2)

device x batch size x decoding strategy, over 200 test sentences."""),
    code("""!python -m gec.evaluate --latency --run $GEC_RESULTS_DIR/runs/e3_real/seed13"""),

    md("""## 12. Aggregate and show the figures"""),
    code("""!python -m gec.aggregate

import pathlib
from IPython.display import Image, Markdown, display

RES = pathlib.Path(os.environ['GEC_RESULTS_DIR'])
display(Markdown((RES / 'RESULTS.md').read_text(encoding='utf-8')))
for png in sorted((RES / 'figures').glob('*.png')):
    print(png.name)
    display(Image(filename=str(png)))"""),

    md("""## 13. Copy results back

Everything already lives on Drive via the environment variables in cell 2.
This zips the small text artefacts for easy download."""),
    code("""!cd $GEC_RESULTS_DIR/.. && zip -qr /content/gec_results.zip results \\
    -x 'results/runs/*/*/model/*' -x 'results/runs/*/*/predictions/*.txt' \\
    -x 'results/runs/*/*/predictions/*.m2'
print('wrote /content/gec_results.zip')
from google.colab import files
files.download('/content/gec_results.zip')"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print(f"wrote {OUT} ({len(cells)} cells)")
