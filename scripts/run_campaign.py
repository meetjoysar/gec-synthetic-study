"""Run the remaining experiment campaign unattended, in dependency order.

::

    python scripts/run_campaign.py            # everything that is not done yet
    python scripts/run_campaign.py --dry-run  # just print the stage list
    python scripts/run_campaign.py --stop-after e4

Every stage is **idempotent**: a run whose ``metrics.json`` is complete is
skipped, so re-running after a crash or a reboot resumes rather than restarts.

Stage order matters and is encoded here rather than left to a human:

1. ``e1``                 - the noise sweep (no dependencies)
2. ``select``             - writes selection.json from **val** scores only
3. ``e2``                 - needs E1's best p
4. ``select``             - adds E2's best clean_ratio
5. ``e3``                 - needs both; e3_synthetic aliases the E2 winner
6. ``distribution``       - the E4 matching loop (CPU, ~20 min)
7. ``e4``                 - needs e4_matched_weights.yaml from stage 6
8. ``latency``            - spec 7.2, needs any trained checkpoint
9. ``aggregate``          - a COMPLETE report exists from here on
10. ``e3_seeds``          - optional extra seeds for error bars on figure 4
11. ``aggregate``         - re-run so figure 4 gains its error bars

Stage 9 deliberately comes before stage 10: if anything goes wrong in the
optional extra seeds, a full single-seed report is already on disk.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PY = sys.executable

#: (name, argv, description)
STAGES: list[tuple[str, list[str], str]] = [
    ("e1", ["-m", "gec.run_experiments", "--plan", "e1", "--yes"],
     "noise-ratio sweep, 5 arms"),
    ("select1", ["-m", "gec.aggregate", "--tables-only"],
     "write selection.json (E1 best p, from val only)"),
    ("e2", ["-m", "gec.run_experiments", "--plan", "e2", "--yes"],
     "clean:noisy balance, 3 new runs + 1 alias"),
    ("select2", ["-m", "gec.aggregate", "--tables-only"],
     "add E2 best clean_ratio to selection.json"),
    ("e3", ["-m", "gec.run_experiments", "--plan", "e3", "--yes"],
     "the main comparison: synthetic / real / mixed / synth->real"),
    ("distribution", ["-m", "gec.distribution", "--report", "--match"],
     "E4 distribution matching loop (CPU)"),
    ("e4", ["-m", "gec.run_experiments", "--plan", "e4", "--yes"],
     "distribution-matched synthetic and mixed"),
    ("e5", ["-m", "gec.run_experiments", "--plan", "e5"],
     "Indian English (skips cleanly until the group writes >= 20 rows)"),
    ("latency", ["-m", "gec.evaluate", "--latency",
                 "--run", "results/runs/e3_real/seed13"],
     "latency table, spec 7.2"),
    ("aggregate", ["-m", "gec.aggregate"],
     "figures, tables, RESULTS.md - a complete report from here"),
    ("e3_seeds", ["-m", "gec.run_experiments", "--plan", "e3",
                  "--seeds", "42", "2026", "--yes"],
     "OPTIONAL: extra E3 seeds, for error bars on figure 4"),
    ("aggregate2", ["-m", "gec.aggregate"],
     "re-aggregate so figure 4 gains its error bars"),
]


def env() -> dict[str, str]:
    e = os.environ.copy()
    e["PYTHONNOUSERSITE"] = "1"       # a broken pip/user site shadows the env
    e["PYTHONUTF8"] = "1"             # ERRANT opens files with a bare open()
    e["PYTHONPATH"] = str(ROOT / "src")
    e["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    e["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return e


def run_stage(name: str, argv: list[str], desc: str) -> int:
    print(f"\n{'=' * 78}\n[campaign] STAGE {name}: {desc}\n{'=' * 78}", flush=True)
    t0 = time.time()
    p = subprocess.run([PY, *argv], cwd=ROOT, env=env())
    mins = (time.time() - t0) / 60
    status = "OK" if p.returncode == 0 else f"FAILED (rc={p.returncode})"
    print(f"\n[campaign] stage {name}: {status} in {mins:.1f} min", flush=True)
    return p.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-after", default=None, help="last stage to run")
    ap.add_argument("--skip", nargs="*", default=[], help="stage names to skip")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going past a failed stage (aggregate still runs)")
    args = ap.parse_args(argv)

    stages = [s for s in STAGES if s[0] not in args.skip]
    if args.stop_after:
        names = [s[0] for s in stages]
        if args.stop_after not in names:
            ap.error(f"--stop-after {args.stop_after!r} is not a stage; "
                     f"choose from {names}")
        stages = stages[: names.index(args.stop_after) + 1]

    print("[campaign] stages:")
    for n, _, d in stages:
        print(f"  {n:<14} {d}")
    if args.dry_run:
        return 0

    t0 = time.time()
    failures: list[str] = []
    for name, cmd, desc in stages:
        rc = run_stage(name, cmd, desc)
        if rc != 0:
            failures.append(name)
            # The optional extra seeds must never cost us the report that
            # already exists on disk.
            if name in ("e3_seeds", "e5", "latency"):
                print(f"[campaign] {name} is optional; continuing.", flush=True)
                continue
            if not args.continue_on_error:
                print(f"[campaign] stopping after {name}. Fix it and re-run this "
                      "script; completed runs are skipped.", flush=True)
                break

    total = (time.time() - t0) / 60
    print(f"\n[campaign] finished in {total:.1f} min ({total / 60:.1f} h)")
    if failures:
        print(f"[campaign] failed stage(s): {', '.join(failures)}")
        return 1
    print("[campaign] all stages OK. See results/RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
