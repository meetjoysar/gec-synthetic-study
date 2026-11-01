"""Reproducibility: seeding, git state, library versions, device info.

Spec rule 2.5 requires every run to save its resolved config, seed, git commit
hash and library versions next to its results. :func:`write_env_json` is the one
function that does it, and ``train``/``predict``/``evaluate``/``distribution``
all call it.

Deliberate choice: we do **not** enable ``torch.use_deterministic_algorithms``.
Full determinism would cost roughly 20% throughput for no scientific gain here,
because seeds 13/42/2026 exist to *measure* run-to-run variance, not eliminate
it. Seeding is still done properly so a single run is repeatable to within
cuDNN's nondeterminism.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from gec import paths

#: Libraries recorded in every ``env.json``. ``en_core_web_sm`` is listed
#: because a spaCy *model* version change silently alters tokenisation and POS
#: tags, which would invalidate a cached clean pool.
TRACKED_LIBS = (
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "accelerate",
    "datasets",
    "sentencepiece",
    "spacy",
    "en_core_web_sm",
    "errant",
    "lemminflect",
    "sacrebleu",
    "scikit-learn",
    "pandas",
    "numpy",
    "matplotlib",
    "pyyaml",
    "gradio",
)


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string with a trailing ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU and CUDA).

    ``PYTHONHASHSEED`` is set for completeness, though it only affects freshly
    spawned interpreters, not the current one.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def lib_versions() -> dict[str, str]:
    """Installed version of every library in :data:`TRACKED_LIBS`."""
    out: dict[str, str] = {}
    for name in TRACKED_LIBS:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def _git(*args: str) -> str | None:
    """Run a git command in the repo root, returning stripped stdout or None."""
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=paths.ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def git_state() -> dict[str, Any]:
    """Commit, branch and dirty-tree state.

    ``dirty`` matters: a run made from an uncommitted working tree is not
    reproducible, and recording that honestly is better than pretending
    otherwise. ``run_experiments`` warns (but does not block) when it is true.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None, "dirty_files": [],
                "note": "not a git repository, or git unavailable"}
    status = _git("status", "--porcelain") or ""
    dirty_files = [ln[3:] for ln in status.splitlines() if ln.strip()]
    return {
        "commit": commit,
        "short": commit[:8],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--always", "--dirty"),
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:50],
    }


def device_info() -> dict[str, Any]:
    """What we are computing on, including the bf16 decision inputs.

    ``capability`` is recorded because ``torch.cuda.is_bf16_supported()`` is not
    trustworthy: in torch >= 2.6 it defaults to ``including_emulation=True`` and
    returns ``True`` on a Colab T4 (SM 7.5), where emulated bf16 is far slower
    than fp32. Every precision decision in this project gates on
    ``capability[0] >= 8`` instead. See :func:`gec.train.resolve_precision`.
    """
    info: dict[str, Any] = {"type": "cpu", "name": platform.processor() or "unknown"}
    try:
        import torch
    except ImportError:
        info["note"] = "torch not installed"
        return info

    info["torch"] = torch.__version__
    if not torch.cuda.is_available():
        info["cuda_available"] = False
        return info

    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    info.update(
        {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "cuda_available": True,
            "capability": list(cap),
            "total_vram_mb": round(props.total_memory / (1 << 20)),
            "cuda": torch.version.cuda,
            "bf16_real": cap[0] >= 8,
            "bf16_reported": bool(torch.cuda.is_bf16_supported()),
        }
    )
    return info


def best_device(requested: str = "auto") -> str:
    """Resolve ``auto`` / ``cuda`` / ``cpu`` to an actual device string."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device: cuda requested but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def peak_vram_mb() -> float | None:
    """Peak CUDA allocation since the last reset, in MiB."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1 << 20), 1)
    except ImportError:
        return None


def reset_peak_vram() -> None:
    """Reset the CUDA peak-memory counter, so a run measures only itself."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def env_payload(
    cfg: dict | None = None,
    seed: int | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    """Build the full reproducibility record (spec rule 2.5)."""
    payload: dict[str, Any] = {
        "seed": seed,
        "git": git_state(),
        "python": f"{platform.python_version()} ({platform.python_implementation()}, {sys.platform})",
        "interpreter": sys.executable,
        "platform": platform.platform(),
        "libs": lib_versions(),
        "device": device_info(),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else None,
        "started_at": utc_now(),
        "finished_at": None,
    }
    if cfg is not None:
        import json

        payload["config_sha256"] = paths.sha256_str(
            json.dumps(cfg, sort_keys=True, default=str)
        )
    if extra:
        payload.update(extra)
    return payload


def write_env_json(
    run: Path,
    cfg: dict | None = None,
    seed: int | None = None,
    extra: dict | None = None,
) -> Path:
    """Write ``<run>/env.json`` and return its path."""
    out = Path(run) / "env.json"
    paths.write_json(out, env_payload(cfg, seed, extra))
    return out


def finish_env_json(run: Path, **updates: Any) -> None:
    """Stamp ``finished_at`` (and any extra fields) into an existing env.json."""
    p = Path(run) / "env.json"
    if not p.exists():
        return
    payload = paths.read_json(p)
    payload["finished_at"] = utc_now()
    payload.update(updates)
    paths.write_json(p, payload)
