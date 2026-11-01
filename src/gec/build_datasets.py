"""Build synthetic / real / mixed training sets (spec section 6.3).

::

    python -m gec.build_datasets --config configs/experiments/e1_p0.2.yaml --seed 13

Each training set is a JSONL of ``{"id", "src", "tgt", "origin"}`` with
``origin`` in ``{real, synthetic, identity}``.

**Direction, restated because it is easy to invert:** the generator corrupts a
clean sentence, so the training pair is ``src = noisy`` and ``tgt = clean``.

Two facts the report must state, both recorded in every ``.stats.json``:

* **"Mixed" is not 50/50.** With ``synthetic_n = 30000`` and 32,308 real
  sentences the realised split is about 52% real, and the stats record the true
  proportion.
* **Hyperparameters are selected on the REAL ``val`` split for every
  condition**, because that is what spec section 7 mandates. So "synthetic-only"
  means *no real data in training*, not *no real data at all*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from gec import error_gen, paths, prepare_data, tok


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def spec_hash(spec: dict, seed: int) -> str:
    """Short stable hash of a dataset spec, so reuse is safe."""
    return hashlib.sha1(_canonical({"spec": spec, "seed": seed}).encode()).hexdigest()[:8]


def resolve_spec(cfg: dict, seed: int) -> dict:
    """Flatten the parts of a config that actually determine the dataset.

    Only these keys feed the hash, so changing an unrelated key (say
    ``learning_rate``) correctly reuses an existing dataset file, while changing
    ``p`` or the generator weights correctly produces a new one.
    """
    ds = cfg.get("dataset", {}) or {}
    syn = dict(ds.get("synthetic", {}) or {})
    real = dict(ds.get("real", {}) or {})

    weights = syn.get("weights")
    wf = syn.get("weights_file")
    if weights is None and wf:
        import yaml

        loaded = yaml.safe_load(paths.read_text(paths.ROOT / wf)) or {}
        weights = loaded.get("weights", loaded)
    if weights is None:
        weights = (cfg.get("generator", {}) or {}).get("weights") or error_gen.DEFAULT_WEIGHTS

    return {
        "synthetic": {
            "enabled": bool(syn.get("enabled")),
            "n": syn.get("n") or (cfg.get("data", {}) or {}).get("synthetic_n", 30000),
            "p": syn.get("p", 0.2),
            "clean_ratio": syn.get("clean_ratio", 0.0),
            "weights": {k: round(float(v), 6) for k, v in sorted(weights.items())},
        },
        "real": {
            "enabled": bool(real.get("enabled")),
            "n": real.get("n"),
        },
        "shuffle": bool(ds.get("shuffle", True)),
        "condition": cfg.get("condition"),
    }


def dataset_slug(spec: dict, seed: int) -> str:
    """Readable filename prefix plus the spec hash.

    The hash makes reuse safe; the prefix keeps ``data/processed/`` legible.
    """
    syn, real = spec["synthetic"], spec["real"]
    bits: list[str] = []
    if syn["enabled"]:
        bits.append(f"syn{syn['n']}_p{syn['p']}_clean{syn['clean_ratio']}")
    if real["enabled"]:
        bits.append("real" + (str(real["n"]) if real["n"] else "all"))
    return f"{'_'.join(bits) or 'empty'}_s{seed}_{spec_hash(spec, seed)}"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def build_synthetic(spec: dict, seed: int) -> tuple[list[dict], dict]:
    """Corrupt the base set into ``(src=noisy, tgt=clean)`` pairs."""
    syn = spec["synthetic"]
    n = int(syn["n"])
    p = float(syn["p"])
    clean_ratio = float(syn["clean_ratio"])

    pool = prepare_data.load_pool()
    base_n = min(prepare_data.POOL_BASE_SET, len(pool))
    if n > base_n:
        raise ValueError(
            f"requested n={n} synthetic pairs but the base set has only {base_n} "
            f"sentences (pool={len(pool)}). Rebuild with "
            f"`python -m gec.prepare_data --wikitext --pool-size {n + 10000}`."
        )

    # Always draw from the first POOL_BASE_SET sentences so every synthetic
    # condition corrupts the SAME sentences and conditions differ only in how
    # they are corrupted (spec section 5.4).
    idx = list(range(base_n))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]

    # Which rows stay clean is drawn from the same seeded stream, so a
    # clean_ratio sweep keeps the noisy subset NESTED - a cleaner E2 comparison
    # than redrawing the corrupted sentences for every ratio.
    rng = random.Random(seed + 1)
    n_clean = int(round(clean_ratio * n))
    clean_positions = set(rng.sample(range(n), n_clean)) if n_clean else set()

    gen = error_gen.ErrorGenerator(weights=syn["weights"], seed=seed)
    rows: list[dict] = []
    n_identity_unintended = 0

    for k, i in enumerate(idx):
        t = pool[i]
        clean_text = t.text
        if k in clean_positions:
            rows.append(
                {"id": f"syn-{k:06d}", "src": clean_text, "tgt": clean_text,
                 "origin": "identity", "pool_i": i}
            )
            continue
        noisy, edits = gen.corrupt_tagged(t, p=p, force_one=True)
        noisy_text = " ".join(noisy)
        if noisy_text == clean_text:
            # force_one could not find any applicable op (punctuation-only and
            # similar). Counted, not hidden: see error_gen.FORCE_ONE_FAILURES.
            n_identity_unintended += 1
            rows.append(
                {"id": f"syn-{k:06d}", "src": clean_text, "tgt": clean_text,
                 "origin": "identity", "pool_i": i}
            )
            continue
        rows.append(
            {"id": f"syn-{k:06d}", "src": noisy_text, "tgt": clean_text,
             "origin": "synthetic", "pool_i": i, "n_edits": len(edits)}
        )

    gstats = gen.stats
    gstats["n_identity_requested"] = n_clean
    gstats["n_identity_unintended"] = n_identity_unintended
    # Realised vs requested op shares: the diagnostic that explains E4's
    # behaviour. `applies` gates which ops can fire on a given token, so the
    # measured mix never equals the requested weights.
    total_edits = sum(gen.op_counts.values())
    gstats["op_share_realised"] = (
        {k: round(v / total_edits, 4) for k, v in gen.op_counts.most_common()}
        if total_edits else {}
    )
    gstats["op_share_requested"] = {k: round(v, 4) for k, v in sorted(gen.weights.items())}
    return rows, gstats


def build_real(spec: dict, seed: int) -> list[dict]:
    """Load ``train_real`` as ``(src=learner, tgt=corrected)`` pairs.

    Drops the handful of W&I sentences whose annotator-0 correction deletes the
    whole sentence: an empty target would teach the model to emit nothing. The
    count is recorded in the stats file.
    """
    jsonl = paths.split_files("train_real")["jsonl"]
    if not jsonl.exists():
        raise FileNotFoundError(
            f"{paths.rel(jsonl)} is missing. Run: python -m gec.prepare_data --wi"
        )
    rows: list[dict] = []
    dropped = 0
    for r in paths.read_jsonl(jsonl):
        if not r["tgt"].strip() or not r["src"].strip():
            dropped += 1
            continue
        rows.append({"id": r["id"], "src": r["src"], "tgt": r["tgt"], "origin": "real"})
    if dropped:
        print(f"[data] dropped {dropped} real pairs with an empty source or target")
    n = spec["real"]["n"]
    if n:
        random.Random(seed).shuffle(rows)
        rows = rows[: int(n)]
    build_real.last_dropped = dropped  # type: ignore[attr-defined]
    return rows


build_real.last_dropped = 0  # type: ignore[attr-defined]


def build_for_config(cfg: dict, seed: int, force: bool = False) -> Path:
    """Build (or reuse) the training JSONL for one config and seed."""
    spec = resolve_spec(cfg, seed)
    slug = dataset_slug(spec, seed)
    out = paths.PROCESSED / f"{slug}.jsonl"
    stats_path = paths.PROCESSED / f"{slug}.stats.json"

    if out.exists() and stats_path.exists() and not force:
        prev = paths.read_json(stats_path)
        if prev.get("spec_hash") == spec_hash(spec, seed):
            print(f"[data] reusing {paths.rel(out)} ({prev.get('n')} pairs)")
            return out

    _warn_on_pool_version_drift()

    rows: list[dict] = []
    gstats: dict = {}
    if spec["synthetic"]["enabled"]:
        syn_rows, gstats = build_synthetic(spec, seed)
        rows.extend(syn_rows)
    if spec["real"]["enabled"]:
        rows.extend(build_real(spec, seed))
    if not rows:
        raise ValueError(
            f"config {cfg.get('name')} enables neither synthetic nor real data"
        )

    if spec["shuffle"]:
        random.Random(seed).shuffle(rows)

    paths.write_jsonl(out, rows)
    paths.write_json(stats_path, dataset_stats(rows, gstats, spec, seed, out))
    return out


def dataset_stats(
    rows: list[dict], gstats: dict, spec: dict, seed: int, out: Path
) -> dict:
    """Everything the report and E4 need to know about a built dataset."""
    origins = Counter(r["origin"] for r in rows)
    n = len(rows)
    n_identity = origins["identity"]
    edit_counts = [r["n_edits"] for r in rows if "n_edits" in r]
    len_hist = Counter(len(r["src"].split()) for r in rows)

    stats = {
        "n": n,
        "slug": out.stem,
        "spec": spec,
        "spec_hash": spec_hash(spec, seed),
        "seed": seed,
        "origins": dict(origins),
        "identity_share": round(n_identity / n, 4) if n else None,
        "real_share": round(origins["real"] / n, 4) if n else None,
        "clean_ratio_requested": spec["synthetic"]["clean_ratio"],
        "mean_edits_per_noisy_sentence": (
            round(sum(edit_counts) / len(edit_counts), 4) if edit_counts else None
        ),
        "median_edits_per_noisy_sentence": (
            sorted(edit_counts)[len(edit_counts) // 2] if edit_counts else None
        ),
        "mean_src_len": round(sum(k * v for k, v in len_hist.items()) / n, 2) if n else None,
        "max_src_len": max(len_hist) if len_hist else 0,
        "n_real_dropped_empty_target": getattr(build_real, "last_dropped", 0),
        "generator": gstats,
        "pool_sha256": _pool_sha(),
    }
    print(
        f"[data] {out.name}\n"
        f"[data]   n={n}  origins={dict(origins)}  identity_share={stats['identity_share']}"
        + (f"  real_share={stats['real_share']}" if origins["real"] else "")
    )
    if edit_counts:
        print(f"[data]   mean edits/noisy sentence={stats['mean_edits_per_noisy_sentence']}")
    if gstats.get("op_counts"):
        top = list(gstats["op_counts"].items())[:6]
        print(f"[data]   top ops={dict(top)}")
    if gstats.get("n_identity_unintended"):
        print(f"[data]   WARNING: {gstats['n_identity_unintended']} pairs became "
              "identity because no operation applied")
    return stats


def _pool_sha() -> str | None:
    p = paths.PROCESSED / "clean_pool.stats.json"
    return paths.read_json(p).get("pool_sha256") if p.exists() else None


def _warn_on_pool_version_drift() -> None:
    """Loudly warn if the cached pool came from a different spaCy build.

    A spaCy or model version change silently alters tokenisation and POS tags,
    so a cache built by one version and used by another would mix two taggers'
    output without any visible symptom.
    """
    p = paths.PROCESSED / "clean_pool.stats.json"
    if not p.exists():
        return
    st = paths.read_json(p)
    import spacy

    cur_spacy = spacy.__version__
    cur_model = tok.tagger().meta.get("version", "unknown")
    if st.get("spacy_version") != cur_spacy or st.get("en_core_web_sm_version") != cur_model:
        print(
            "[data] WARNING: the clean pool was built with "
            f"spacy {st.get('spacy_version')} / en_core_web_sm "
            f"{st.get('en_core_web_sm_version')}, but this environment has "
            f"spacy {cur_spacy} / en_core_web_sm {cur_model}. Cached POS tags may "
            "not match current tokenisation. Rebuild with "
            "`python -m gec.prepare_data --wikitext --force`."
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    from gec.config import load_config

    ap = argparse.ArgumentParser(
        prog="python -m gec.build_datasets",
        description="Build the training JSONL for one experiment config.",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--plan", default=None, help="apply configs/plans/<plan>.yaml")
    ap.add_argument("--set", dest="overrides", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config), plan=args.plan, overrides=args.overrides,
                      seed=args.seed)
    out = build_for_config(cfg, args.seed, force=args.force)
    print(f"[data] -> {paths.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
