"""Config loading, merging, validation and ``${select.*}`` resolution.

Four layers, merged in order so later wins::

    1. configs/base.yaml                   every key, with defaults
    2. configs/experiments/<name>.yaml     the run family
    3. configs/plans/<plan>.yaml           only when --plan is given (e.g. smoke)
    4. --set key.path=value                CLI overrides, YAML-parsed

**Selection stays honest by code, not discipline.** Spec section 7 requires E2
to use the best ``p`` from E1 by *val* F0.5 and E3/E4 to use the best E2
configuration. Hardcoding those numbers into YAML after eyeballing a table is
exactly how test-set leakage creeps in, so experiment configs instead write
``${select.e1_best_p}`` and :func:`resolve_refs` substitutes values from
``results/tables/selection.json``, which :func:`gec.aggregate.write_selection`
computes from **val** metrics only (and refuses to read a ``test`` metric at
all). The resolved value lands in each run's own ``config.yaml``, so every run
records what it was given.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from gec import paths

#: ``${select.foo}`` references, resolved from results/tables/selection.json.
REF = re.compile(r"\$\{select\.([a-zA-Z0-9_]+)\}")


class ConfigError(ValueError):
    """A config is malformed, has unknown keys, or has an unresolvable reference."""


def load_yaml(p: Path | str) -> dict:
    """Load a YAML file as a dict (an empty file gives ``{}``)."""
    p = Path(p)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    data = yaml.safe_load(paths.read_text(p))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge *over* into *base*, returning a new dict.

    Dicts recurse; **lists and scalars replace**. That is the right rule here:
    ``generation`` and ``data`` are dicts you want to patch key by key, while
    ``seeds`` and ``eval.splits`` are lists you want to replace wholesale.
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _key_tree(d: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for k, v in d.items():
        path = f"{prefix}{k}"
        keys.add(path)
        if isinstance(v, dict):
            keys |= _key_tree(v, f"{path}.")
    return keys


def apply_set(cfg: dict, items: Iterable[str] | None) -> dict:
    """Apply ``--set key.path=value`` overrides, parsing values as YAML scalars."""
    if not items:
        return cfg
    out = dict(cfg)
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set expects key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
            node[p] = dict(nxt)
            node = node[p]
        node[parts[-1]] = value
    return out


def selection_path() -> Path:
    return paths.TABLES / "selection.json"


def load_selection() -> dict:
    p = selection_path()
    return paths.read_json(p) if p.exists() else {}


def resolve_refs(cfg: Any, selection: dict | None = None, _where: str = "") -> Any:
    """Replace every ``${select.key}`` with its value from ``selection.json``.

    Fails with the command needed to produce the value rather than quietly
    defaulting, because a silent default would mean a run trained on a different
    configuration than its name claims.
    """
    sel = load_selection() if selection is None else selection

    def one(value: Any, where: str) -> Any:
        if isinstance(value, dict):
            return {k: one(v, f"{where}.{k}" if where else k) for k, v in value.items()}
        if isinstance(value, list):
            return [one(v, f"{where}[{i}]") for i, v in enumerate(value)]
        if not isinstance(value, str):
            return value
        m = REF.fullmatch(value.strip())
        if m:
            key = m.group(1)
            if key not in sel:
                raise ConfigError(
                    f"cannot resolve ${{select.{key}}} (needed by {where or 'config'}).\n"
                    f"{paths.rel(selection_path())} "
                    f"{'lacks that key' if sel else 'does not exist yet'}.\n"
                    "Run the earlier plan first, e.g.:\n"
                    "    python -m gec.run_experiments --plan e1 --yes\n"
                    "    python -m gec.aggregate --tables-only"
                )
            return sel[key]
        # Also handle references embedded in a larger string.
        def sub(mm: re.Match[str]) -> str:
            key = mm.group(1)
            if key not in sel:
                raise ConfigError(
                    f"cannot resolve ${{select.{key}}} (needed by {where or 'config'})"
                )
            return str(sel[key])

        return REF.sub(sub, value)

    return one(cfg, _where)


def validate_config(cfg: dict, base: dict | None = None) -> None:
    """Reject unknown keys and illegal values.

    The unknown-key check is what stops a typo like ``clean_ration: 0.33`` from
    silently doing nothing, producing a duplicate run, and costing a GPU slot
    for a result that is indistinguishable from its neighbour afterwards.
    """
    base = base if base is not None else load_yaml(paths.CONFIGS / "base.yaml")
    allowed = _key_tree(base)
    # Sub-trees whose keys are user-defined data, not config structure.
    freeform = {"generator.weights", "dataset.synthetic.weights"}

    def walk(d: dict, prefix: str = "") -> list[str]:
        bad: list[str] = []
        for k, v in d.items():
            path = f"{prefix}{k}"
            if any(path.startswith(f) for f in freeform):
                continue
            if path not in allowed:
                bad.append(path)
                continue
            if isinstance(v, dict) and path not in freeform:
                bad.extend(walk(v, f"{path}."))
        return bad

    unknown = walk(cfg)
    if unknown:
        raise ConfigError(
            "unknown config key(s): "
            + ", ".join(sorted(unknown))
            + f"\nEvery key must exist in {paths.rel(paths.CONFIGS / 'base.yaml')}. "
            "This check exists so a typo fails loudly instead of silently "
            "producing a duplicate run."
        )

    # T5 is unstable in fp16 (spec section 3 / CLAUDE.md). Make it impossible.
    prec = str(cfg.get("precision", "auto")).lower()
    if prec in ("fp16", "float16", "half", "16"):
        raise ConfigError(
            "precision: fp16 is forbidden - T5 produces NaN losses in fp16. "
            "Use fp32, or bf16 on a GPU with compute capability >= 8.0."
        )
    if prec not in ("auto", "fp32", "bf16", "float32", "bfloat16"):
        raise ConfigError(f"precision must be auto|fp32|bf16, got {cfg['precision']!r}")

    cond = cfg.get("condition")
    valid = {"synthetic", "real", "mixed", "synth2real", "copy", None}
    if cond not in valid:
        raise ConfigError(f"condition must be one of {sorted(v for v in valid if v)}, got {cond!r}")

    syn = (cfg.get("dataset", {}) or {}).get("synthetic", {}) or {}
    p = syn.get("p")
    if isinstance(p, (int, float)) and not 0.0 <= float(p) <= 1.0:
        raise ConfigError(f"dataset.synthetic.p must be in [0, 1], got {p}")
    cr = syn.get("clean_ratio")
    if isinstance(cr, (int, float)) and not 0.0 <= float(cr) <= 1.0:
        raise ConfigError(f"dataset.synthetic.clean_ratio must be in [0, 1], got {cr}")

    if (cfg.get("stage2", {}) or {}).get("enabled") and not cfg["stage2"].get("init_from"):
        raise ConfigError("stage2.enabled is true but stage2.init_from is not set")


def load_config(
    experiment: Path | str,
    plan: str | None = None,
    overrides: Iterable[str] | None = None,
    seed: int | None = None,
    resolve: bool = True,
) -> dict:
    """Load and merge all four config layers, then validate.

    Parameters
    ----------
    resolve
        When False, ``${select.*}`` references are left as literal strings.
        ``run_experiments --dry-run`` uses that to list jobs before the earlier
        plan has produced a ``selection.json``.
    """
    base = load_yaml(paths.CONFIGS / "base.yaml")
    exp_path = Path(experiment)
    if not exp_path.is_absolute() and not exp_path.exists():
        exp_path = paths.EXPERIMENTS / exp_path.name
    cfg = deep_merge(base, load_yaml(exp_path))

    if plan:
        plan_path = paths.PLANS / f"{plan}.yaml"
        if plan_path.exists():
            cfg = deep_merge(cfg, load_yaml(plan_path))

    cfg = apply_set(cfg, overrides)

    if resolve:
        cfg = resolve_refs(cfg)
    if seed is not None:
        cfg["seed"] = seed
        # stage2.init_from uses a {seed} placeholder so one config serves all seeds.
        init = (cfg.get("stage2", {}) or {}).get("init_from")
        if isinstance(init, str) and "{seed}" in init:
            cfg["stage2"]["init_from"] = init.format(seed=seed)

    validate_config({k: v for k, v in cfg.items() if k != "seed"}, base)
    cfg.setdefault("name", exp_path.stem)
    return cfg


def run_name(cfg: dict) -> str:
    """Directory name for a run, including any plan prefix (e.g. ``smoke_``)."""
    return f"{cfg.get('run_prefix', '') or ''}{cfg['name']}"


def save_config(cfg: dict, run: Path) -> Path:
    """Write the fully resolved config next to a run's results (spec rule 2.5)."""
    out = Path(run) / "config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False, allow_unicode=True)
    paths.write_text(out, text)
    return out
