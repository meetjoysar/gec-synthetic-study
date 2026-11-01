"""Error-type distributions and generator re-weighting (spec section 7.1, E4).

::

    python -m gec.distribution --report
    python -m gec.distribution --match [--iters 3] [--probe-n 5000]
    python -m gec.distribution --precompensate

Two different re-weightings live here, and they aim at different targets:

:func:`match_distribution` (**E4**, spec 7.1)
    Iteratively re-weight the generator so its *measured* error-type
    distribution approaches **real W&I**. This is the Wang et al. (2024)
    hypothesis: synthetic data helps when it matches the real error
    distribution.

:func:`precompensate_weights` (an extra E1 arm)
    Re-weight so the measured distribution approaches **uniform across the ten
    covered categories**, i.e. what spec 6.2's "roughly uniform" weights were
    *intended* to produce before applicability gating distorted them.

Why either is needed at all: an operation's ``applies`` check gates whether it
can fire on a given token. ``word_dup`` applies to almost any token while
``sva`` needs a copula or a VBZ/VBP, so with uniform weights the realised mix is
badly skewed - measured on 30,000 pairs, SPELL is 18.0% of synthetic edits
against 3.8% in real W&I, while verb errors are 4.2% against 15.8%.

**Generation direction is critical and easy to invert.** The probe must be
annotated as ``-orig noisy -cor clean``, matching a learner corpus (original =
erroneous, corrected = fixed), so the counted types are directly comparable
with ``train_real.m2``. Reversing it swaps every ``M:``/``U:`` pairing
(a missing determiner becomes an unnecessary one) and silently corrupts all
three iterations. :func:`make_probe` returns a named tuple rather than
positional paths so a call site cannot get them the wrong way round.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from gec import errant_cli, error_gen as eg
from gec import m2, paths, prepare_data

#: Bucket for real error types the generator cannot produce at all
#: (WO, MORPH, NOUN:POSS, CONTR, VERB:INFL, ...). Reported, never matched.
UNCOVERED = "UNCOVERED"

E4_DIR = "e4_matching"


class Probe(NamedTuple):
    """Paths to a generated probe set. Named so the direction cannot be mixed up."""

    noisy: Path   # the ERRANT -orig side (the erroneous text)
    clean: Path   # the ERRANT -cor side (the corrected text)
    n: int


def type_counts(m2_path: Path | str, annotator: int = 0, level: int = 2) -> Counter[str]:
    """ERRANT edit-type counts in an M2 file (category level, no M/U/R prefix)."""
    return m2.count_types(m2_path, annotator=annotator, level=level)


def shares(counts: Counter[str], covered: set[str] | None = None) -> dict[str, float]:
    """Normalise counts to proportions, folding unknown types into UNCOVERED."""
    total = sum(counts.values())
    if not total:
        return {}
    if covered is None:
        return {k: v / total for k, v in counts.items()}
    out: dict[str, float] = {c: 0.0 for c in covered}
    out[UNCOVERED] = 0.0
    for k, v in counts.items():
        out[k if k in covered else UNCOVERED] += v / total
    return out


def tvd(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance between two distributions over the same keys."""
    keys = set(p) | set(q)
    return round(0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys), 4)


def covered_types(weights: dict[str, float] | None = None) -> set[str]:
    """ERRANT categories the generator's enabled ops can produce."""
    return eg.covered_errant_types(list(weights) if weights else None)


def make_probe(
    weights: dict[str, float],
    p: float,
    n: int,
    seed: int,
    out_dir: Path,
    clean_ratio: float = 0.0,
) -> Probe:
    """Generate *n* noisy/clean pairs from the held-back pool tail.

    Sentences are drawn from pool indices ``[POOL_BASE_SET, ...)`` - the 10,000
    held back by :mod:`gec.prepare_data` - so a probe never shares sentences
    with any training set.
    """
    pool = prepare_data.load_pool()
    start = min(prepare_data.POOL_BASE_SET, max(0, len(pool) - n))
    tail = pool[start:]
    if len(tail) < n:
        raise ValueError(
            f"only {len(tail)} held-back pool sentences available, need {n}. "
            "Rebuild a bigger pool: python -m gec.prepare_data --wikitext "
            f"--pool-size {prepare_data.POOL_BASE_SET + n} --force"
        )
    idx = list(range(len(tail)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]

    gen = eg.ErrorGenerator(weights=weights, seed=seed)
    noisy_lines: list[str] = []
    clean_lines: list[str] = []
    for i in idx:
        t = tail[i]
        noisy, _ = gen.corrupt_tagged(t, p=p, force_one=True)
        noisy_text = " ".join(noisy)
        if not noisy_text.strip() or not t.text.strip():
            continue
        noisy_lines.append(noisy_text)
        clean_lines.append(t.text)

    out_dir.mkdir(parents=True, exist_ok=True)
    noisy_p = out_dir / "probe.noisy.txt"
    clean_p = out_dir / "probe.clean.txt"
    paths.write_lines(noisy_p, noisy_lines)
    paths.write_lines(clean_p, clean_lines)
    return Probe(noisy=noisy_p, clean=clean_p, n=len(noisy_lines))


def measure_probe(probe: Probe, out_m2: Path) -> Counter[str]:
    """Annotate a probe with ERRANT and count its error types.

    *** DIRECTION: -orig is the NOISY side, -cor is the CLEAN side. ***
    That matches a learner M2 (original = erroneous, corrected = fixed), which
    is what makes these counts comparable with train_real.m2. Swapping them
    would turn every M:DET into a U:DET and invert the whole experiment.
    """
    errant_cli.parallel_to_m2(probe.noisy, [probe.clean], out_m2)
    return type_counts(out_m2)


def reweight(
    weights: dict[str, float],
    real: dict[str, float],
    syn: dict[str, float],
    clip: tuple[float, float] = (0.2, 5.0),
) -> dict[str, float]:
    """Multiplicative update toward *real* (spec 7.1 step 3).

    ``w_new = w_old * clip(target_share / synthetic_share)``, renormalised.

    Operations sharing one ERRANT type (DET has three, PREP/PUNCT/OTHER have
    two each) all receive the same ratio, which scales the group's total share
    toward the target while preserving the proportions within the group.
    Splitting a group by its M/U/R sub-type would need level-3 counts and would
    not converge in three iterations; that limitation belongs in the report.
    """
    eps = 1e-4
    out: dict[str, float] = {}
    for op_name, w in weights.items():
        t = eg.OPS[op_name].errant_type
        ratio = real.get(t, 0.0) / max(syn.get(t, 0.0), eps)
        ratio = min(max(ratio, clip[0]), clip[1])
        out[op_name] = max(w * ratio, eps)
    z = sum(out.values())
    return {k: round(v / z, 6) for k, v in out.items()}


def real_distribution() -> tuple[Counter[str], dict[str, float]]:
    """Error-type counts and covered-type shares of ``train_real``."""
    m2_path = paths.split_files("train_real")["m2"]
    if not m2_path.exists():
        raise FileNotFoundError(
            f"{paths.rel(m2_path)} is missing. Run: python -m gec.prepare_data --wi"
        )
    counts = type_counts(m2_path)
    return counts, shares(counts, covered_types())


def report() -> Path:
    """Write the real error-type distribution to a CSV and print it."""
    counts, sh = real_distribution()
    total = sum(counts.values())
    out = paths.TABLES / "error_dist_real.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "count", "share", "covered_by_generator"])
        cov = covered_types()
        for cat, c in counts.most_common():
            w.writerow([cat, c, round(c / total, 6), cat in cov])
    print(f"[dist] real train_real: {total} edits across {len(counts)} categories")
    for cat, share in sorted(sh.items(), key=lambda kv: -kv[1]):
        print(f"[dist]   {cat:<12} {share:>7.2%}")
    print(f"[dist] wrote {paths.rel(out)}")
    return out


# --------------------------------------------------------------------------- #
# E4: match the real distribution
# --------------------------------------------------------------------------- #

def match_distribution(
    p: float = 0.2,
    clean_ratio: float = 0.0,
    iters: int = 3,
    probe_n: int = 5000,
    probe_seed: int = 777,
    base_weights: dict[str, float] | None = None,
    out_yaml: Path | None = None,
) -> dict:
    """Iteratively re-weight the generator toward the real distribution.

    Keeps the **argmin-TVD** iteration rather than blindly the last one: the
    multiplicative update is not guaranteed to be monotonic, and there is no
    reason to ship a worse set of weights than one we already measured.
    """
    work = paths.RESULTS / E4_DIR
    work.mkdir(parents=True, exist_ok=True)

    cov = covered_types()
    real_counts, real_sh = real_distribution()
    # Compare only over categories the generator can produce, renormalised, so
    # the 22.6% of real edits we structurally cannot make (WO, MORPH, ...) does
    # not dominate the distance and make the target unreachable.
    real_cov = {k: v for k, v in real_sh.items() if k != UNCOVERED}
    z = sum(real_cov.values()) or 1.0
    real_cov = {k: v / z for k, v in real_cov.items()}

    weights = dict(base_weights or eg.DEFAULT_WEIGHTS)
    history: list[dict] = []

    for it in range(iters):
        it_dir = work / f"iter{it}"
        probe = make_probe(weights, p, probe_n, probe_seed, it_dir, clean_ratio)
        counts = measure_probe(probe, it_dir / "probe.m2")
        syn_sh = shares(counts, cov)
        syn_cov = {k: v for k, v in syn_sh.items() if k != UNCOVERED}
        zz = sum(syn_cov.values()) or 1.0
        syn_cov = {k: v / zz for k, v in syn_cov.items()}

        d = tvd(real_cov, syn_cov)
        entry = {
            "iter": it,
            "tvd": d,
            "weights": dict(weights),
            "syn_shares": {k: round(v, 6) for k, v in syn_cov.items()},
            "n_edits": sum(counts.values()),
            "edits_per_sentence": round(sum(counts.values()) / max(probe.n, 1), 3),
        }
        history.append(entry)
        paths.write_json(it_dir / "iter.json", entry)
        print(f"[dist] iter {it}: TVD={d}  edits/sentence={entry['edits_per_sentence']}")

        if it < iters - 1:
            weights = reweight(weights, real_cov, syn_cov)

    best = min(history, key=lambda h: h["tvd"])
    print(f"[dist] best iteration: {best['iter']} (TVD={best['tvd']}; "
          f"iteration 0 was {history[0]['tvd']})")

    out_yaml = out_yaml or (paths.EXPERIMENTS / "e4_matched_weights.yaml")
    _write_weights_yaml(
        out_yaml,
        best["weights"],
        header=(
            "# E4 matched generator weights (spec 7.1).\n"
            "# Generated by: python -m gec.distribution --match\n"
            f"# Chosen iteration {best['iter']} of {iters}, "
            f"TVD {history[0]['tvd']} -> {best['tvd']} vs real train_real.\n"
            "# Compared over the 10 categories the generator can produce,\n"
            "# renormalised; the ~22.6% of real edits it structurally cannot\n"
            "# make (WO, MORPH, NOUN:POSS, ...) are excluded from the target.\n"
        ),
    )

    # Figure 5 and the TVD table.
    dist_csv = paths.TABLES / "e4_distribution.csv"
    with open(dist_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "real_share", "syn_share_before", "syn_share_after"])
        cats = sorted(set(real_cov) | set(history[0]["syn_shares"]))
        for c in cats:
            w.writerow([
                c,
                round(real_cov.get(c, 0.0), 6),
                round(history[0]["syn_shares"].get(c, 0.0), 6),
                round(best["syn_shares"].get(c, 0.0), 6),
            ])
        w.writerow([UNCOVERED, round(real_sh.get(UNCOVERED, 0.0), 6), 0.0, 0.0])

    tvd_csv = paths.TABLES / "e4_tvd.csv"
    with open(tvd_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iter", "tvd", "n_edits", "edits_per_sentence", "chosen"])
        for h in history:
            w.writerow([h["iter"], h["tvd"], h["n_edits"], h["edits_per_sentence"],
                        h["iter"] == best["iter"]])

    print(f"[dist] wrote {paths.rel(out_yaml)}, {paths.rel(dist_csv)}, {paths.rel(tvd_csv)}")
    return {"history": history, "best": best, "weights_file": str(out_yaml)}


# --------------------------------------------------------------------------- #
# Extra E1 arm: pre-compensate for applicability gating
# --------------------------------------------------------------------------- #

def _measure_shares(
    weights: dict[str, float], p: float, n: int, seed: int, work: Path, cov: set[str]
) -> dict[str, float]:
    """Generate a probe, annotate it, and return renormalised covered shares."""
    probe = make_probe(weights, p, n, seed, work)
    counts = measure_probe(probe, work / "probe.m2")
    sh = {k: v for k, v in shares(counts, cov).items() if k != UNCOVERED}
    z = sum(sh.values()) or 1.0
    return {k: v / z for k, v in sh.items()}


def precompensate_weights(
    p: float = 0.2,
    probe_n: int = 5000,
    probe_seed: int = 555,
    iters: int = 3,
    base_weights: dict[str, float] | None = None,
    out_yaml: Path | None = None,
) -> dict:
    """Re-weight so the *realised* mix is ~uniform across the 10 categories.

    Spec 6.2's weights are uniform across ERRANT types *as requested*, but
    applicability gating means the realised mix is not. This divides each
    operation's weight by its measured firing rate, so the distribution the
    model actually trains on is close to what the spec intended.

    Iterated, for the same reason E4 is: ``reweight`` clips each step to
    ``[0.2x, 5x]``, and the gap for the verb operations is larger than 5x
    (``sva`` fires at about 0.11x its requested share), so one step cannot
    close it.

    Added as an extra E1 arm at the user's request, to separate two questions
    that the default weights otherwise conflate: "is there a best noise level?"
    and "does the generator's skewed realised distribution hold synthetic-only
    back?". Logged as a deviation in PROGRESS.md.
    """
    cov = covered_types()
    uniform = {c: 1.0 / len(cov) for c in cov}
    weights = dict(base_weights or eg.DEFAULT_WEIGHTS)
    work = paths.RESULTS / "e1_precompensate"

    history: list[dict] = []
    best_w, best_tvd, best_shares = dict(weights), None, {}
    for it in range(iters + 1):
        sh = _measure_shares(weights, p, probe_n, probe_seed, work / f"iter{it}", cov)
        d = tvd(uniform, sh)
        history.append({"iter": it, "tvd": d, "shares": {k: round(v, 6) for k, v in sh.items()},
                        "weights": dict(weights)})
        print(f"[dist] precompensate iter {it}: TVD vs uniform = {d}")
        if best_tvd is None or d < best_tvd:
            best_w, best_tvd, best_shares = dict(weights), d, sh
        if it < iters:
            weights = reweight(weights, uniform, sh)

    before, after = history[0]["tvd"], best_tvd
    print(f"[dist] pre-compensation: TVD vs uniform {before} -> {after} "
          f"(kept iteration {min(history, key=lambda h: h['tvd'])['iter']})")
    for c in sorted(cov):
        print(f"[dist]   {c:<12} before={history[0]['shares'].get(c, 0):>6.2%} "
              f"after={best_shares.get(c, 0):>6.2%}  (target {uniform[c]:.2%})")

    out_yaml = out_yaml or (paths.EXPERIMENTS / "e1_precomp_weights.yaml")
    _write_weights_yaml(
        out_yaml,
        best_w,
        header=(
            "# Pre-compensated generator weights - extra E1 arm.\n"
            "# Generated by: python -m gec.distribution --precompensate\n"
            "# Each weight is divided by the operation's MEASURED firing rate, so\n"
            "# the realised error mix is ~uniform across the 10 covered ERRANT\n"
            "# categories instead of merely the requested weights being uniform.\n"
            f"# TVD vs uniform: {before} -> {after} over {iters} clipped iterations\n"
            f"# (measured on {probe_n} probe pairs at p={p}).\n"
            "# Deviation from spec 6.2, logged in PROGRESS.md.\n"
        ),
    )
    out = {"weights": best_w, "tvd_before": before, "tvd_after": after,
           "iters": iters, "history": [{k: h[k] for k in ("iter", "tvd", "shares")}
                                       for h in history],
           "weights_file": str(out_yaml)}
    paths.write_json(paths.TABLES / "e1_precompensation.json", out)
    print(f"[dist] wrote {paths.rel(out_yaml)}")
    return out


def _write_weights_yaml(path: Path, weights: dict[str, float], header: str = "") -> None:
    import yaml

    body = yaml.safe_dump({"weights": {k: float(v) for k, v in sorted(weights.items())}},
                          sort_keys=True, default_flow_style=False)
    paths.write_text(path, header + body)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gec.distribution",
        description="Error-type distributions and generator re-weighting.",
    )
    ap.add_argument("--report", action="store_true", help="real error-type distribution")
    ap.add_argument("--match", action="store_true", help="E4: match the real distribution")
    ap.add_argument("--precompensate", action="store_true",
                    help="extra E1 arm: flatten the realised distribution")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--probe-n", type=int, default=5000)
    ap.add_argument("--probe-seed", type=int, default=777)
    ap.add_argument("--p", type=float, default=None,
                    help="noise ratio for the probe (default: the E1-selected p)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if not any([args.report, args.match, args.precompensate]):
        ap.error("pick one of --report / --match / --precompensate")

    paths.ensure_dirs()

    p = args.p
    clean_ratio = 0.0
    if p is None:
        from gec.config import load_selection

        sel = load_selection()
        p = sel.get("e1_best_p") or 0.2
        clean_ratio = sel.get("e2_best_clean_ratio") or 0.0
        print(f"[dist] using p={p} clean_ratio={clean_ratio} "
              f"{'from selection.json' if sel else '(defaults; E1/E2 have not run)'}")

    if args.report:
        report()
    if args.precompensate:
        precompensate_weights(p=p, probe_n=args.probe_n, probe_seed=args.probe_seed,
                              out_yaml=Path(args.out) if args.out else None)
    if args.match:
        match_distribution(p=p, clean_ratio=clean_ratio, iters=args.iters,
                           probe_n=args.probe_n, probe_seed=args.probe_seed,
                           out_yaml=Path(args.out) if args.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
