"""Fine-tune ``t5-small`` as a grammar corrector (spec section 8).

::

    python -m gec.train --config configs/experiments/e3_mixed.yaml --seed 42

Decisions worth knowing:

* **No ``datasets.Dataset`` in the training path.** A 40-line
  :class:`torch.utils.data.Dataset` plus ``DataCollatorForSeq2Seq`` does the
  job and decouples training from Arrow / ``huggingface-hub`` version churn.
  (``datasets>=4.3`` requires ``huggingface-hub>=1.31`` while
  ``transformers 4.53.1`` requires ``<1.0``.) Logged as deviation D8.

* **Eager tokenisation is required, not an optimisation.** ``group_by_length``
  makes HuggingFace's ``LengthGroupedSampler`` read every example's length up
  front, so lazy tokenisation would tokenise the corpus twice. Eager also makes
  truncation counts exact and free (spec section 8.1 asks us to log them).

* **Precision gates on compute capability, not ``is_bf16_supported()``.** In
  torch >= 2.6 that function defaults to ``including_emulation=True`` and
  returns True on a Colab T4 (SM 7.5), where emulated bf16 is far slower than
  fp32. See :func:`resolve_precision`. fp16 raises unconditionally.

* **Two guards against silent garbage**: a NaN watchdog that aborts early with
  the fp32 rerun command, and a degenerate-output check that flags a run whose
  predictions are all-empty or pure-copy, so a near-zero F0.5 is never quietly
  plotted as a finding.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from gec import config as cfgmod
from gec import build_datasets, paths, repro


def resolve_precision(cfg: dict, device: str) -> str:
    """Decide fp32 vs bf16. Never fp16 - T5 produces NaN losses in fp16."""
    want = str(cfg.get("precision", "auto")).lower()
    if want in ("fp16", "float16", "half"):
        raise ValueError(
            "precision: fp16 is forbidden for T5 (NaN losses). Use fp32 or bf16."
        )
    if device == "cpu":
        return "fp32"

    import torch

    cap = torch.cuda.get_device_capability()
    # Gate on capability, NOT torch.cuda.is_bf16_supported(): since torch 2.6
    # that defaults to including_emulation=True and returns True on a T4
    # (capability 7.5), where emulated bf16 is far slower than plain fp32.
    real_bf16 = cap[0] >= 8
    if want in ("bf16", "bfloat16"):
        if not real_bf16:
            print(f"[train] WARNING: bf16 requested but compute capability is {cap}; "
                  "using fp32 (emulated bf16 would be slower than fp32)")
            return "fp32"
        return "bf16"
    if want == "auto":
        return "bf16" if real_bf16 else "fp32"
    return "fp32"


class GecDataset:
    """Eagerly tokenised ``(src, tgt)`` pairs; ``__getitem__`` is O(1)."""

    def __init__(self, rows: list[dict], tokenizer, cfg: dict, name: str = "") -> None:
        import torch  # noqa: F401  (ensures a clear error if torch is missing)

        prefix = cfg.get("prefix", "grammar: ")
        max_src = int(cfg.get("max_source_len", 128))
        max_tgt = int(cfg.get("max_target_len", 128))

        srcs = [prefix + r["src"] for r in rows]
        tgts = [r["tgt"] for r in rows]

        enc = tokenizer(srcs, truncation=True, max_length=max_src)
        lab = tokenizer(text_target=tgts, truncation=True, max_length=max_tgt)

        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = lab["input_ids"]
        self.name = name

        # Exact truncation counts, free because tokenisation is eager.
        full_src = tokenizer(srcs, truncation=False)["input_ids"]
        full_tgt = tokenizer(text_target=tgts, truncation=False)["input_ids"]
        self.n_truncated_src = sum(1 for x in full_src if len(x) > max_src)
        self.n_truncated_tgt = sum(1 for x in full_tgt if len(x) > max_tgt)
        if self.n_truncated_src or self.n_truncated_tgt:
            print(f"[train] {name}: truncated {self.n_truncated_src} sources and "
                  f"{self.n_truncated_tgt} targets at {max_src}/{max_tgt} tokens "
                  f"({len(rows)} examples)")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


def _nan_watchdog():
    """Abort training the moment the loss becomes NaN or inf.

    The known T5 mixed-precision failure mode. Cheap insurance: without it a
    bf16 run can spend 20 minutes producing a model that outputs nothing.
    """
    from transformers import TrainerCallback

    class NaNWatchdog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is None:
                return
            if math.isnan(loss) or math.isinf(loss):
                raise RuntimeError(
                    f"training loss became {loss} at step {state.global_step}. "
                    "This is the T5 mixed-precision instability. Rerun with "
                    "`--set precision=fp32`."
                )

    return NaNWatchdog()


def build_model(cfg: dict, init_from: Path | str | None = None):
    """Load ``t5-small`` (or a stage-1 checkpoint) and its tokenizer."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    source = str(init_from) if init_from else cfg["model_name"]
    print(f"[train] loading model from {source}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(source)
    return model, tokenizer


def training_args(cfg: dict, run: Path, seed: int, device: str, precision: str, stage: int = 1):
    """Build ``Seq2SeqTrainingArguments`` for transformers 4.53."""
    from transformers import Seq2SeqTrainingArguments

    epochs = cfg["epochs"] if stage == 1 else (
        (cfg.get("stage2", {}) or {}).get("epochs") or cfg["stage2_epochs"]
    )
    lr = cfg["learning_rate"] if stage == 1 else (
        (cfg.get("stage2", {}) or {}).get("learning_rate") or cfg["stage2_learning_rate"]
    )
    max_steps = cfg.get("max_steps")

    kwargs: dict[str, Any] = dict(
        output_dir=str(run / "trainer"),
        overwrite_output_dir=True,
        seed=seed,
        data_seed=seed,
        num_train_epochs=float(epochs),
        learning_rate=float(lr),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        per_device_train_batch_size=int(cfg["train_batch_size"]),
        per_device_eval_batch_size=int(cfg["eval_batch_size"]),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_steps=25,
        # Renamed in transformers 4.x: `evaluation_strategy` was REMOVED.
        eval_strategy="epoch",
        save_strategy="no",
        # Loss-only epoch eval (spec 8.1). Generating on 2,000 val sentences
        # every epoch would add ~15 min/run for zero selection value, because
        # selection uses ERRANT F0.5 computed later by evaluate.py.
        predict_with_generate=False,
        report_to=[],              # tensorboard is installed and would auto-attach
        group_by_length=bool(cfg.get("group_by_length", True)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=(device == "cuda"),
        bf16=(precision == "bf16"),
        fp16=False,                # never, for T5
        use_cpu=(device == "cpu"),  # renamed from `no_cuda`
        disable_tqdm=False,
        label_smoothing_factor=0.0,
    )
    if max_steps:
        kwargs["max_steps"] = int(max_steps)
    return Seq2SeqTrainingArguments(**kwargs)


def _degenerate_check(model, tokenizer, cfg: dict, device: str, n: int = 24) -> dict:
    """Generate on a few val sentences and look for all-empty / pure-copy output.

    A model that only ever copies its input scores F0.5 near zero, which looks
    exactly like a real finding. Flagging it here means ``aggregate`` can render
    a footnote instead of silently plotting it.
    """
    from gec import predict

    rows = paths.read_jsonl(paths.split_files("val")["jsonl"])[:n]
    if not rows:
        return {}
    srcs = [r["src"] for r in rows]
    preds, _ = predict.generate_batched(
        model, tokenizer, srcs, cfg, device, batch_size=8, num_beams=1
    )
    n_empty = sum(1 for p in preds if not p.strip())
    n_copy = sum(1 for s, p in zip(srcs, preds) if s.split() == p.split())
    out = {
        "empty_pred_rate": round(n_empty / len(preds), 4),
        "copy_rate": round(n_copy / len(preds), 4),
        "n_probe": len(preds),
    }
    if out["empty_pred_rate"] > 0.2:
        out["suspect"] = True
        out["suspect_reason"] = f"{n_empty}/{len(preds)} predictions were empty"
    elif out["copy_rate"] > 0.98:
        out["suspect"] = True
        out["suspect_reason"] = (
            f"{n_copy}/{len(preds)} predictions were identical to the source"
        )
    else:
        out["suspect"] = False

    print("\n[train] sample predictions (src / pred / tgt):")
    for i in range(min(5, len(preds))):
        print(f"  src : {srcs[i]}")
        print(f"  pred: {preds[i]}")
        print(f"  tgt : {rows[i]['tgt']}")
    if out.get("suspect"):
        print(f"[train] WARNING: run flagged SUSPECT - {out['suspect_reason']}")
    return out


def train(cfg: dict, seed: int, run: Path | None = None) -> dict:
    """Train one run and write ``train_metrics.json``. Returns those metrics."""
    import torch
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

    name = cfgmod.run_name(cfg)
    run = Path(run) if run else paths.run_dir(name, seed)
    run.mkdir(parents=True, exist_ok=True)

    repro.set_seed(seed)
    device = repro.best_device(cfg.get("device", "auto"))
    precision = resolve_precision(cfg, device)

    if device == "cuda":
        # TF32 is off by default for matmul in torch 2.x; free ~1.2-1.5x on Ampere.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[train] run={name} seed={seed} device={device} precision={precision}")
    cfgmod.save_config({**cfg, "seed": seed}, run)

    stage2 = cfg.get("stage2", {}) or {}
    init_from = None
    if stage2.get("enabled"):
        ref = str(stage2["init_from"])
        init_from = paths.RUNS / ref / "model"
        if not (init_from / "config.json").exists():
            raise FileNotFoundError(
                f"stage2 needs the stage-1 checkpoint at {paths.rel(init_from)}, "
                "which does not exist. Produce it first:\n"
                f"    python -m gec.run_experiments --plan e3 --only e3_synthetic "
                f"--seeds {seed} --yes"
            )
        print(f"[train] stage 2: warm-starting from {paths.rel(init_from)}")

    train_jsonl = build_datasets.build_for_config(cfg, seed)
    rows = paths.read_jsonl(train_jsonl)
    val_rows = paths.read_jsonl(paths.split_files("val")["jsonl"])
    val_limit = (cfg.get("eval", {}) or {}).get("limit")
    if val_limit:
        val_rows = val_rows[: int(val_limit)]
    # An empty target would teach the model to emit nothing.
    val_rows = [r for r in val_rows if r["tgt"].strip()]

    repro.write_env_json(
        run,
        cfg,
        seed,
        extra={
            "precision_resolved": precision,
            "data": {
                "train_jsonl": paths.rel(train_jsonl),
                "train_sha256": paths.sha256(train_jsonl),
                "n_train": len(rows),
                "n_val": len(val_rows),
            },
        },
    )

    model, tokenizer = build_model(cfg, init_from)
    train_ds = GecDataset(rows, tokenizer, cfg, name="train")
    val_ds = GecDataset(val_rows, tokenizer, cfg, name="val")

    args = training_args(cfg, run, seed, device, precision, stage=2 if init_from else 1)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,   # `tokenizer=` is deprecated in 4.53
        callbacks=[_nan_watchdog()],
    )

    repro.reset_peak_vram()
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0

    model_dir = run / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"[train] saved {paths.rel(model_dir)}  ({wall / 60:.1f} min)")

    history = [h for h in trainer.state.log_history if "eval_loss" in h]
    metrics = {
        "run": name,
        "seed": seed,
        "stage": 2 if init_from else 1,
        "device": device,
        "precision": precision,
        "wall_clock_s": round(wall, 1),
        "steps": int(trainer.state.global_step),
        "epochs": float(args.num_train_epochs),
        "n_train": len(rows),
        "n_val": len(val_ds),
        "effective_batch_size": (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        ),
        "final_train_loss": round(float(result.training_loss), 6),
        "best_val_loss": (
            round(min(h["eval_loss"] for h in history), 6) if history else None
        ),
        "val_loss_per_epoch": [round(float(h["eval_loss"]), 6) for h in history],
        "n_truncated_src": train_ds.n_truncated_src,
        "n_truncated_tgt": train_ds.n_truncated_tgt,
        "peak_vram_mb": repro.peak_vram_mb(),
        # Throughput must come from steps ACTUALLY taken, not from the dataset
        # size times epochs: with max_steps set (the timing probe) training
        # stops early, and the naive form over-reports by the ratio of the full
        # epochs to the steps run - it read 1281 examples/s instead of ~103.
        # This number drives every budget estimate, so it has to be right.
        "examples_per_s": (
            round(
                trainer.state.global_step
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                / wall,
                2,
            )
            if wall and trainer.state.global_step
            else None
        ),
        "s_per_step": round(wall / max(trainer.state.global_step, 1), 4),
        "train_jsonl": paths.rel(train_jsonl),
    }
    metrics.update(_degenerate_check(model, tokenizer, cfg, device))

    paths.write_json(run / "train_metrics.json", metrics)
    paths.write_jsonl(run / "logs" / "train.jsonl", trainer.state.log_history)
    repro.finish_env_json(run, train_wall_clock_s=metrics["wall_clock_s"])

    # NOTE: `keep_checkpoint: false` is honoured by run_experiments AFTER
    # prediction and evaluation, not here. Deleting the weights at the end of
    # training would leave predict.py with nothing to load.
    # The Trainer's own output_dir holds nothing we need (save_strategy="no").
    import shutil

    shutil.rmtree(run / "trainer", ignore_errors=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gec.train", description="Fine-tune t5-small for GEC."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=None)
    args = ap.parse_args(argv)

    overrides = list(args.overrides or [])
    if args.device:
        overrides.append(f"device={args.device}")
    if args.max_steps:
        overrides.append(f"max_steps={args.max_steps}")

    cfg = cfgmod.load_config(args.config, plan=args.plan, overrides=overrides, seed=args.seed)
    metrics = train(cfg, args.seed)
    print(json.dumps({k: v for k, v in metrics.items() if k != "val_loss_per_epoch"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
