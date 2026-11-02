"""Run a plan of configs x seeds, resumably (spec section 8.2).

::

    python -m gec.run_experiments --plan smoke
    python -m gec.run_experiments --plan e1 --dry-run
    python -m gec.run_experiments --plan e1 --timing-probe
    python -m gec.run_experiments --plan e3 --seeds 13 --yes
    python -m gec.run_experiments --plan all --yes

**Resumability.** A job is ``done`` when its ``metrics.json`` has
``_meta.complete == true``. Within a job each stage checks its own artefact
(``model/config.json`` for training, ``<split>.pred.txt`` for prediction,
``<split>.hyp.m2`` for the ERRANT step), so a crash during evaluation does not
re-run training or generation. A Colab disconnect costs at most one run.

**The budget gate encodes CLAUDE.md's rule in code.** Nothing longer than
``--budget-minutes`` (default 15) starts without ``--yes``, and the estimate is
printed first. Every real plan here trips it, which is the point: it forces an
explicit, informed decision each time rather than one blanket approval.
"""

from __future__ import annotations

import argparse
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from gec import config as cfgmod
from gec import paths, repro

#: Plans: ordered lists of (config name, seeds or None for the plan default).
#: E1 sweeps the spec's four noise points plus the pre-compensated arm (D10).
#: E3 is single-seed for now (D6); the extra seeds are a resumable follow-on.
PLANS: dict[str, list[str]] = {
    "e0": ["e0_copy"],
    # e1_p0.2_precomp is an extra arm (deviation D10): same p and sentences
    # as e1_p0.2, only the generator weights differ.
    "e1": ["e1_p0.1", "e1_p0.2", "e1_p0.4", "e1_p0.6", "e1_p0.2_precomp"],
    "e2": ["e2_clean0.0", "e2_clean0.2", "e2_clean0.33", "e2_clean0.5"],
    "e3": ["e3_synthetic", "e3_real", "e3_mixed", "e3_synth2real"],
    "e4": ["e4_synthetic_matched", "e4_mixed_matched"],
    "smoke": ["e0_copy", "e3_mixed"],
    # e5 is evaluation-only: it has no configs of its own, it re-scores
    # existing E3/E4 checkpoints on the Indian English split. See run_e5().
    "e5": [],
}
#: Plans whose configs reference ${select.*} and so need an earlier plan's
#: selection.json to exist first.
NEEDS_SELECTION = {"e2", "e3", "e4"}

DEFAULT_SEEDS = [13]

#: Analytical fallbacks used only until a timing probe measures the real values.
#: ~55 training examples/s in bf16 on an RTX 3050; see PROGRESS.md.
FALLBACK_EXAMPLES_PER_S = 55.0
FALLBACK_EVAL_MINUTES = 11.0


@dataclass
class Job:
    run: str
    config: str
    seed: int
    family: str
    cfg: dict = field(default_factory=dict)
    alias_of: str | None = None
    needs_train: bool = True
    splits: list[str] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return paths.run_dir(self.run, self.seed)


def iter_jobs(
    plan: str,
    seeds: list[int],
    only: list[str] | None = None,
    resolve: bool = True,
    overrides: list[str] | None = None,
) -> list[Job]:
    """Expand a plan into concrete jobs."""
    if plan not in PLANS:
        raise KeyError(f"unknown plan {plan!r}; known: {sorted(PLANS)}")
    names = PLANS[plan]
    if only:
        names = [n for n in names if n in only]
        if not names:
            raise ValueError(f"--only {only} matched nothing in plan {plan}")

    jobs: list[Job] = []
    for name in names:
        for seed in seeds:
            cfg = cfgmod.load_config(
                paths.EXPERIMENTS / f"{name}.yaml",
                plan="smoke" if plan == "smoke" else None,
                overrides=overrides,
                seed=seed,
                resolve=resolve,
            )
            jobs.append(
                Job(
                    run=cfgmod.run_name(cfg),
                    config=name,
                    seed=seed,
                    family=cfg.get("family") or plan,
                    cfg=cfg,
                    alias_of=_alias_for_seed(cfg.get("alias_of"), seed, name),
                    needs_train=cfg.get("condition") != "copy",
                    splits=list((cfg.get("eval", {}) or {}).get("splits", ["val", "test"])),
                )
            )
    return jobs


def _alias_for_seed(alias: str | None, seed: int, config_name: str) -> str | None:
    """Drop an alias whose target belongs to a different seed.

    ``selection.json`` records a *concrete* winner such as
    ``e2_clean0.33/seed13``, and ``e3_synthetic`` aliases
    ``${select.e2_best_run}``. Applied naively that makes
    ``e3_synthetic/seed42`` point at the **seed-13** run, so adding seeds 42 and
    2026 for error bars would silently produce three identical copies of one
    run - and figure 4 would report a standard deviation of zero for the
    synthetic condition, which looks like a result rather than a bug.

    An alias is only sound when the target's seed matches the job's. Otherwise
    the job has to train for real.
    """
    if not alias:
        return None
    m = re.search(r"/seed(\d+)$", str(alias))
    if m and int(m.group(1)) != seed:
        print(f"[run] {config_name}/seed{seed}: not aliasing {alias} "
              f"(target is seed {m.group(1)}); will train instead")
        return None
    return alias


def job_status(job: Job) -> str:
    """``done`` | ``partial`` | ``todo``."""
    metrics = job.dir / "metrics.json"
    if metrics.exists():
        try:
            if paths.read_json(metrics).get("_meta", {}).get("complete"):
                return "done"
        except Exception:  # noqa: BLE001 - unreadable JSON counts as not done
            pass
    has_model = (job.dir / "model" / "config.json").exists()
    has_preds = any(
        (job.dir / "predictions" / f"{s}.pred.txt").exists() for s in job.splits
    )
    return "partial" if (has_model or has_preds) else "todo"


def _timing() -> dict:
    p = paths.TABLES / "timing_probe.json"
    return paths.read_json(p) if p.exists() else {}


def estimate_minutes(job: Job) -> tuple[float, str]:
    """Estimate one job's wall-clock, preferring measured over analytical."""
    if job.alias_of:
        return 0.0, "alias (no training)"
    if not job.needs_train:
        return 2.0, "no training"

    from gec import build_datasets

    try:
        spec = build_datasets.resolve_spec(job.cfg, job.seed)
    except cfgmod.ConfigError:
        return 0.0, "blocked"
    n = 0
    if spec["synthetic"]["enabled"]:
        n += int(spec["synthetic"]["n"])
    if spec["real"]["enabled"]:
        n += int(spec["real"]["n"] or 32308)

    stage2 = (job.cfg.get("stage2", {}) or {}).get("enabled")
    epochs = (
        (job.cfg["stage2"].get("epochs") or job.cfg["stage2_epochs"])
        if stage2 else job.cfg["epochs"]
    )

    t = _timing()
    eps = t.get("examples_per_s") or FALLBACK_EXAMPLES_PER_S
    eval_min = t.get("eval_minutes") or FALLBACK_EVAL_MINUTES
    source = "measured" if t.get("examples_per_s") else "analytical, UNVERIFIED"
    train_min = (n * float(epochs)) / float(eps) / 60.0
    return train_min + eval_min * max(1, len(job.splits)) / 2, source


def write_alias(job: Job) -> None:
    """Point a run at an equivalent, already-trained run instead of repeating it.

    Spec section 7 says E2's ``clean_ratio=0.0`` *is* the best E1 run and should
    be reused; the same applies to E3(a), which is the best E2 run.
    ``aggregate.collect_runs`` follows the pointer, so the table still has a row
    for every condition while no model is trained twice.
    """
    target = str(job.alias_of)
    target_metrics = paths.RUNS / target / "metrics.json"
    if not target_metrics.exists():
        raise FileNotFoundError(
            f"{job.run}/seed{job.seed} is an alias of {target}, whose metrics.json "
            f"does not exist yet ({paths.rel(target_metrics)})."
        )
    job.dir.mkdir(parents=True, exist_ok=True)
    cfgmod.save_config({**job.cfg, "seed": job.seed}, job.dir)

    # An alias normally stores only a pointer. But a run can be aliased AND
    # still be needed as a real checkpoint: `e3_synthetic` aliases the best E2
    # run, while `e3_synth2real` warm-starts from `e3_synthetic/seed{seed}/model`
    # and the Gradio demo loads the same path. Copying the weights (~240 MB) is
    # far cheaper than retraining an identical configuration, and it keeps both
    # consumers working.
    if job.cfg.get("keep_checkpoint", True):
        src_model = paths.RUNS / target / "model"
        dst_model = job.dir / "model"
        if src_model.exists() and not (dst_model / "config.json").exists():
            import shutil

            shutil.copytree(src_model, dst_model, dirs_exist_ok=True)
            print(f"[run] copied the alias target's weights into "
                  f"{paths.rel(dst_model)} (needed by stage 2 and the demo)")
        elif not src_model.exists():
            print(
                f"[run] WARNING: {job.run} is an alias of {target}, which has no "
                "model/ directory, so no weights could be copied. Anything that "
                "needs this checkpoint (stage-2 training, the demo) will fail. "
                f"Re-run {target} with keep_checkpoint=true."
            )
    paths.write_json_atomic(
        job.dir / "metrics.json",
        {
            "run": job.run,
            "seed": job.seed,
            "condition": job.cfg.get("condition"),
            "alias_of": target,
            "_meta": {
                "complete": True,
                "written_at": repro.utc_now(),
                "schema": 1,
                "note": "alias - metrics come from the target run; no model retrained",
            },
        },
    )
    print(f"[run] {job.run}/seed{job.seed} -> alias of {target} (no training)")


def run_job(job: Job, force: bool = False) -> dict:
    """Train (if needed), predict every split, then evaluate."""
    from gec import evaluate, predict, train

    if job.alias_of:
        write_alias(job)
        return {"alias_of": job.alias_of}

    job.dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if job.needs_train and (force or not (job.dir / "model" / "config.json").exists()):
        train.train(job.cfg, job.seed, run=job.dir)
    elif job.needs_train:
        print(f"[run] reusing trained model at {paths.rel(job.dir / 'model')}")
    else:
        cfgmod.save_config({**job.cfg, "seed": job.seed}, job.dir)
        repro.write_env_json(job.dir, job.cfg, job.seed)

    splits = list(job.splits)
    if "indian" in splits and (paths.PROCESSED / "indian.skip").exists():
        print("[run] skipping the indian split: fewer than 20 real rows yet")
        splits.remove("indian")

    for split in splits:
        predict.predict_split(job.dir, split, job.cfg, force=force)

    metrics = evaluate.evaluate_run(job.dir, splits=splits, force=force)

    # Prune the checkpoint only now that every split has been predicted and
    # scored. Spec section 8.4 keeps weights only for the demo/E5 models, but
    # deleting them before prediction would make the run unevaluable - which is
    # exactly the bug the smoke test caught.
    if job.needs_train and not job.cfg.get("keep_checkpoint", True):
        import shutil

        model_dir = job.dir / "model"
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
            print(f"[run] keep_checkpoint=false: removed {paths.rel(model_dir)}")

    wall = time.time() - t0
    paths.PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.RUNS / "_timing.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        import json

        tm = paths.read_json(job.dir / "train_metrics.json") if (
            job.dir / "train_metrics.json"
        ).exists() else {}
        fh.write(json.dumps({
            "run": job.run, "seed": job.seed, "wall_clock_s": round(wall, 1),
            "n_train": tm.get("n_train"), "train_s": tm.get("wall_clock_s"),
            "examples_per_s": tm.get("examples_per_s"),
            "peak_vram_mb": tm.get("peak_vram_mb"),
            "finished_at": repro.utc_now(),
        }) + "\n")
    return metrics


def run_e5(seeds: list[int], force: bool = False) -> int:
    """E5: evaluate every existing E3/E4 model on the Indian English split.

    Evaluation only - no training. Normally unnecessary, because E3/E4 configs
    list ``indian`` in ``eval.splits`` and so predict it in the same invocation
    as ``test``. This exists for the case the spec's schedule makes likely: the
    models were trained before the group finished writing the sentences.
    """
    from gec import evaluate, predict

    if (paths.PROCESSED / "indian.skip").exists():
        skip = paths.read_json(paths.PROCESSED / "indian.skip")
        print(f"[run] E5 skipped: {skip.get('reason')}")
        return 0
    if not paths.split_files("indian")["src"].exists():
        print("[run] E5 skipped: data/processed/indian.src.txt is missing. Run:\n"
              "    python -m gec.prepare_data --indian")
        return 0

    targets: list[Path] = []
    for pattern in ("e3_*", "e4_*"):
        for d in sorted(paths.RUNS.glob(f"{pattern}/seed*")):
            if (d / "model" / "config.json").exists():
                targets.append(d)
    if not targets:
        print("[run] E5: no E3/E4 checkpoint found yet; nothing to evaluate.")
        return 0

    import yaml

    failures = 0
    for run in targets:
        print(f"\n[run] E5 on {paths.rel(run)}")
        try:
            cfg = yaml.safe_load(paths.read_text(run / "config.yaml"))
            predict.predict_split(run, "indian", cfg, force=force)
            evaluate.evaluate_run(run, splits=["indian"], force=force)
        except Exception as exc:  # noqa: BLE001
            print(f"[run] E5 failed for {paths.rel(run)}: {type(exc).__name__}: {exc}")
            failures += 1
    return failures


def timing_probe(
    plan: str,
    seed: int = 13,
    steps: int = 150,
    n_pred: int = 300,
    overrides: list[str] | None = None,
    tag: str = "probe_",
) -> dict:
    """Measure throughput cheaply, so the budget stops being guesswork.

    Trains the plan's first trainable config for *steps* steps and generates on
    *n_pred* sentences. Takes about ten minutes, i.e. under the 15-minute
    consent rule, and replaces the analytical constants in
    :func:`estimate_minutes` with measured ones.
    """
    from gec import predict, train

    jobs = [j for j in iter_jobs(plan, [seed], resolve=False)
            if j.needs_train and not j.alias_of]
    if not jobs:
        raise ValueError(f"plan {plan} has no trainable config to probe")
    job = jobs[0]

    cfg = cfgmod.load_config(
        paths.EXPERIMENTS / f"{job.config}.yaml",
        overrides=[f"max_steps={steps}", f"run_prefix={tag}", "keep_checkpoint=true",
                   "eval.splits=[]", *(overrides or [])],
        seed=seed,
        resolve=False,
    )
    # The probe must not depend on a selection.json that does not exist yet.
    syn = cfg.setdefault("dataset", {}).setdefault("synthetic", {})
    if not isinstance(syn.get("p"), (int, float)):
        syn["p"] = 0.2
    if not isinstance(syn.get("clean_ratio"), (int, float)):
        syn["clean_ratio"] = 0.0
    cfg["alias_of"] = None

    run = paths.run_dir(cfgmod.run_name(cfg), seed)
    print(f"\n[probe] training {job.config} for {steps} steps to measure throughput")
    tm = train.train(cfg, seed, run=run)

    srcs = paths.read_lines(paths.split_files("test")["src"])[:n_pred]
    device = repro.best_device(cfg.get("device", "auto"))
    model, tokenizer, _ = predict.load_run_model(run, device)
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    predict.generate_batched(model, tokenizer, srcs, cfg, device)
    if device == "cuda":
        torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0

    sent_per_s = len(srcs) / gen_s
    # Each run evaluates val (2,000) + test (4,384); ERRANT adds ~4 min of CPU.
    eval_minutes = (6384 / sent_per_s) / 60.0 + 4.0

    out = {
        "probed_config": job.config,
        "seed": seed,
        "device": tm["device"],
        "precision": tm["precision"],
        "steps": tm["steps"],
        "s_per_step": tm["s_per_step"],
        "examples_per_s": tm["examples_per_s"],
        "effective_batch_size": tm["effective_batch_size"],
        "train_batch_size": cfg["train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "peak_vram_mb": tm["peak_vram_mb"],
        "sentences_per_s_beam4": round(sent_per_s, 2),
        "eval_minutes": round(eval_minutes, 1),
        "measured_at": repro.utc_now(),
    }
    paths.write_json(paths.TABLES / "timing_probe.json", out)

    print("\n" + "=" * 70)
    print("TIMING PROBE - measured, not estimated")
    print("=" * 70)
    for k, v in out.items():
        print(f"  {k:26s} {v}")
    print("=" * 70)
    return out


def print_plan(jobs: list[Job], budget: float) -> float:
    """Print every job with status and estimate; return the total minutes."""
    print(f"\n{'run':28s} {'seed':>5s}  {'status':8s} {'est.min':>8s}  dataset")
    print("-" * 100)
    total = 0.0
    source = ""
    for job in jobs:
        status = job_status(job)
        est, source = estimate_minutes(job)
        if status != "done":
            total += est
        desc = "alias of " + str(job.alias_of) if job.alias_of else ""
        if not desc:
            from gec import build_datasets

            try:
                spec = build_datasets.resolve_spec(job.cfg, job.seed)
            except cfgmod.ConfigError as exc:
                status = "blocked"
                desc = str(exc).splitlines()[0]
                print(f"{job.run:28s} {job.seed:>5d}  {status:8s} {'-':>8s}  {desc}")
                continue
            bits = []
            if spec["synthetic"]["enabled"]:
                bits.append(f"syn n={spec['synthetic']['n']} p={spec['synthetic']['p']} "
                            f"clean={spec['synthetic']['clean_ratio']}")
            if spec["real"]["enabled"]:
                bits.append("real=all")
            desc = " + ".join(bits) or "(no training data)"
        print(f"{job.run:28s} {job.seed:>5d}  {status:8s} {est:>8.1f}  {desc}")
    print("-" * 100)
    todo = [j for j in jobs if job_status(j) != "done"]
    print(f"{len(todo)} of {len(jobs)} job(s) to run; "
          f"estimated {total:.0f} min ({total / 60:.1f} h) [{source}]")
    if total > budget:
        print(f"\n  *** This exceeds --budget-minutes {budget:.0f}. "
              "Pass --yes to proceed. ***")
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gec.run_experiments",
        description="Run a plan of experiment configs x seeds, resumably.",
    )
    ap.add_argument("--plan", required=True,
                    help=f"one of {sorted(PLANS)} or 'all'")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these config names")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the job list with statuses and estimates, then exit")
    ap.add_argument("--timing-probe", action="store_true",
                    help="measure throughput (~10 min) and write timing_probe.json")
    ap.add_argument("--budget-minutes", type=float, default=15.0,
                    help="refuse to start a longer plan without --yes (CLAUDE.md rule)")
    ap.add_argument("--yes", action="store_true", help="approve the estimated runtime")
    ap.add_argument("--force", action="store_true", help="re-run jobs that are already done")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--set", dest="overrides", nargs="*", default=None,
                    help="config overrides, e.g. --set train_batch_size=16")
    args = ap.parse_args(argv)

    paths.ensure_dirs()
    seeds = args.seeds or DEFAULT_SEEDS

    if args.timing_probe:
        timing_probe(args.plan if args.plan != "all" else "e1", seed=seeds[0],
                     overrides=args.overrides)
        print("\nNext: re-run with --dry-run to see estimates based on these measurements.")
        return 0

    if args.plan == "e5":
        return 1 if run_e5(seeds, force=args.force) else 0

    plans = ["e0", "e1", "e2", "e3", "e4"] if args.plan == "all" else [args.plan]

    git = repro.git_state()
    if git.get("dirty"):
        print(f"[run] WARNING: the working tree has {len(git['dirty_files'])} "
              "uncommitted change(s); these runs will not be exactly reproducible.")

    overall_total = 0.0
    all_jobs: list[tuple[str, list[Job]]] = []
    for plan in plans:
        # A plan that references ${select.*} cannot be resolved until the
        # earlier plan has run, so list it unresolved.
        resolve = plan not in NEEDS_SELECTION or cfgmod.selection_path().exists()
        try:
            jobs = iter_jobs(plan, seeds, only=args.only, resolve=resolve,
                             overrides=args.overrides)
        except cfgmod.ConfigError as exc:
            print(f"\n[run] plan {plan} is blocked:\n{exc}")
            if len(plans) == 1:
                return 1
            continue
        all_jobs.append((plan, jobs))
        print(f"\n=== plan {plan} ===")
        overall_total += print_plan(jobs, args.budget_minutes)

    if args.dry_run:
        return 0

    if overall_total > args.budget_minutes and not args.yes:
        print(
            f"\nRefusing to start: estimated {overall_total:.0f} min "
            f"({overall_total / 60:.1f} h) exceeds --budget-minutes "
            f"{args.budget_minutes:.0f}.\n"
            "CLAUDE.md requires asking before runs over ~15 minutes. "
            "Re-run with --yes once you have approved the time."
        )
        return 2

    failures: list[str] = []
    done = 0
    t_start = time.time()
    for plan, jobs in all_jobs:
        # Selection must be recomputed between plans: E2 needs E1's best p,
        # E3/E4 need E2's best clean_ratio.
        if plan in NEEDS_SELECTION:
            from gec import aggregate

            aggregate.write_selection()
            jobs = iter_jobs(plan, seeds, only=args.only, resolve=True,
                             overrides=args.overrides)

        for job in jobs:
            status = job_status(job)
            if status == "done" and not args.force:
                print(f"\n[run] skipping {job.run}/seed{job.seed} (already complete)")
                continue
            print(f"\n{'=' * 70}\n[run] {job.run} seed={job.seed} (status: {status})\n{'=' * 70}")
            try:
                run_job(job, force=args.force)
                done += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{job.run}/seed{job.seed}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                if not args.continue_on_error:
                    print(f"\n[run] stopping after failure in {job.run}/seed{job.seed}")
                    break
            elapsed = (time.time() - t_start) / 60
            print(f"[run] {done} job(s) done, {elapsed:.1f} min elapsed")
        if failures and not args.continue_on_error:
            break

    print(f"\n[run] finished {done} job(s) in {(time.time() - t_start) / 60:.1f} min")
    if failures:
        print(f"[run] {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
