# CLAUDE.md

This repo is our NLP IA Activity 2 mini-project: **"Fake Errors, Real Grammar"** — does synthetic error data help a small T5 grammar corrector, compared with real and mixed data?

**The full specification is in `PROJECT_SPEC.md`. Read it fully before doing anything.** Work through its phases (§10) in order, one at a time.

## Always
- Track progress in `PROGRESS.md` (phase status, what changed, commands to reproduce, open issues). Read it at the start of every session to know where we are.
- After each phase: run its acceptance checks, run `pytest`, commit, and give a short summary before moving on.
- Never invent results; every reported number must come from a file our code wrote. Use `TBD` for runs not yet done.
- Never train, tune or generate synthetic data from the test set (`ABCN.dev.gold.bea19.m2`). Select settings on our `val` split.
- Do not write the Indian English test sentences — only the template, tooling and ≤ 5 `EXAMPLE-` rows.
- Ask before: downloads over 500 MB, runs over ~15 minutes on this machine, deleting data or checkpoints.
- Keep code simple and cross-platform (`python -m gec.<module>`, `pathlib`); the user may be on Windows.
- T5: never use fp16.

## Common commands
```bash
pip install -r requirements.txt
pytest -q
python -m gec.run_experiments --plan smoke      # must pass before any GPU run
python -m gec.run_experiments --plan e3 --seeds 13 42 2026
python -m gec.aggregate
python app/app.py
```
