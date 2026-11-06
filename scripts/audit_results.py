"""Check that every number in ``results/RESULTS.md`` came from a run's metrics.

::

    python scripts/audit_results.py          # exits non-zero on any mismatch

Spec section 2, rule 1: *never invent results; every reported number must come
from a file our code wrote.* ``aggregate.py`` is written to honour that, but the
claim is only worth as much as an independent check of it, so this script
re-reads the main table out of the rendered Markdown and compares every cell
against the ``metrics.json`` it is supposed to have come from - without going
through ``aggregate`` at all.

It also verifies that every file the report links to exists, and that the
report's completeness count matches the runs actually on disk.
"""

from __future__ import annotations

import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gec import paths  # noqa: E402

#: RESULTS.md main-table column -> where the value lives in metrics.json.
COLUMNS = [
    ("P", ("test", "errant", "precision"), 4),
    ("R", ("test", "errant", "recall"), 4),
    ("F0.5", ("test", "errant", "f05"), 4),
    ("BLEU(13a)", ("test", "bleu_13a"), 2),
    ("BLEU(tok)", ("test", "bleu_tok"), 2),
    ("over-corr.", ("test", "over_correction_rate"), 4),
    ("edits/sent", ("test", "avg_edits_per_sentence"), 4),
]


def dig(metrics: dict, path: tuple[str, ...]):
    node = metrics.get("splits", {})
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def resolve(run: str, seed: int) -> dict | None:
    """Load a run's metrics, following ``alias_of`` to the run holding them."""
    p = paths.RUNS / run / f"seed{seed}" / "metrics.json"
    if not p.exists():
        return None
    m = paths.read_json(p)
    for _ in range(5):
        if not m.get("alias_of"):
            return m
        target = paths.RUNS / str(m["alias_of"])
        if target.name.startswith("seed"):
            p = target / "metrics.json"
        else:
            p = target / f"seed{seed}" / "metrics.json"
        if not p.exists():
            return None
        m = paths.read_json(p)
    return None


def main() -> int:
    md = paths.RESULTS / "RESULTS.md"
    if not md.exists():
        print(f"[audit] {md} does not exist - run `python -m gec.aggregate`")
        return 1
    text = paths.read_text(md)
    problems: list[str] = []
    checked = 0

    # ---- the main table ---------------------------------------------------
    rows = re.findall(r"^\| (e\d[\w.]*) \| (\d+) \|(.+)\|$", text, re.M)
    if not rows:
        problems.append("no main-table rows found in RESULTS.md")
    for run, seed, rest in rows:
        cells = [c.strip() for c in rest.split("|") if c.strip() != ""]
        if len(cells) != len(COLUMNS):
            continue  # a different table (E3/E4 deltas) - handled below
        m = resolve(run, int(seed))
        if m is None:
            problems.append(f"{run}/seed{seed}: reported but no metrics.json on disk")
            continue
        for (label, path, nd), cell in zip(COLUMNS, cells):
            if cell == "TBD":
                continue
            want = dig(m, path)
            if want is None:
                problems.append(f"{run}/seed{seed} {label}: {cell} is not in metrics.json")
                continue
            if f"{float(want):.{nd}f}" != cell:
                problems.append(
                    f"{run}/seed{seed} {label}: RESULTS.md says {cell}, "
                    f"metrics.json says {float(want):.{nd}f}"
                )
            checked += 1

    # ---- the E3 aggregate table ------------------------------------------
    # Format: | run | mean +/- sd (n=k) | delta +/- sd | same sign | indian |
    # Everything here is recomputed from the per-seed metrics.json files, so a
    # mistake in aggregate's own averaging or pairing shows up as a mismatch.
    SEEDS = (13, 42, 2026)

    def series(run: str, path: tuple[str, ...]) -> dict[int, float]:
        out = {}
        for s in SEEDS:
            m = resolve(run, s)
            if m is None:
                continue
            v = dig(m, path)
            if v is not None:
                out[s] = float(v)
        return out

    def agg_text(v: dict[int, float]) -> str:
        x = list(v.values())
        if not x:
            return "TBD"
        if len(x) == 1:
            return f"{x[0]:.4f} (n=1)"
        return f"{st.mean(x):.4f} +/- {st.stdev(x):.4f} (n={len(x)})"

    F05 = ("test", "errant", "f05")
    IND = ("indian", "errant", "f05")
    ref_t, ref_i = series("e3_real", F05), series("e3_real", IND)

    # The second cell must be an UNSIGNED mean: the per-seed Indian delta
    # table below has the same five-column shape but signed values, and a
    # looser pattern matched both tables against each other's numbers.
    for run, cell_t, cell_d, cell_sign, cell_i in re.findall(
        r"^\| (e3_\w+) \| (\d\.\d{4}[^|]*?) \| ([^|]+?) \| ([^|]+?) \| "
        r"([^|]+?) \|$",
        text, re.M,
    ):
        vt, vi = series(run, F05), series(run, IND)
        if not vt:
            problems.append(f"{run}: in the E3 table but has no metrics.json")
            continue
        for label, want, cell in (("test F0.5", agg_text(vt), cell_t),
                                  ("indian F0.5", agg_text(vi), cell_i)):
            if want != cell.strip():
                problems.append(f"{run} E3 {label}: RESULTS.md says "
                                f"{cell.strip()!r}, metrics give {want!r}")
            checked += 1
        if run != "e3_real" and ref_t:
            d = [vt[s] - ref_t[s] for s in sorted(vt) if s in ref_t]
            want_d = (f"{st.mean(d):+.4f}" if len(d) == 1
                      else f"{st.mean(d):+.4f} +/- {st.stdev(d):.4f}")
            if want_d != cell_d.strip():
                problems.append(f"{run} E3 paired delta: RESULTS.md says "
                                f"{cell_d.strip()!r}, computed {want_d!r}")
            want_sign = ("yes" if all(x > 0 for x in d) or all(x < 0 for x in d)
                         else "**no**")
            if want_sign != cell_sign.strip():
                problems.append(f"{run} E3 same-sign: RESULTS.md says "
                                f"{cell_sign.strip()!r}, computed {want_sign!r}")
            checked += 2

    # ---- the per-seed Indian paired table --------------------------------
    for run, cells in re.findall(
        r"^\| (e3_\w+) \| ([-+]\d\.\d{4} \|(?: [-+]\d\.\d{4} \|)+)$",
        text, re.M,
    ):
        got = [c.strip() for c in cells.split("|") if c.strip()]
        vi = series(run, IND)
        d = [vi[s] - ref_i[s] for s in sorted(ref_i) if s in vi]
        if len(d) != len(ref_i):
            continue
        want = [f"{x:+.4f}" for x in d] + [f"{st.mean(d):+.4f}"]
        if want != got:
            problems.append(f"{run} indian per-seed deltas: RESULTS.md says "
                            f"{got}, computed {want}")
        checked += len(want)

    # ---- the E4 arm table ------------------------------------------------
    for a, b, fa, fb, delta in re.findall(
        r"^\| (e3_\w+) -> (e4_\w+) \| (\d\.\d{4}) \| (\d\.\d{4}) \| "
        r"([-+]\d\.\d{4}) \|", text, re.M
    ):
        ma, mb = resolve(a, 13), resolve(b, 13)
        if ma is None or mb is None:
            problems.append(f"{a}->{b}: in the E4 table but a metrics.json is missing")
            continue
        va, vb = float(dig(ma, F05)), float(dig(mb, F05))
        for label, want, cell in ((a, va, fa), (b, vb, fb)):
            if f"{want:.4f}" != cell:
                problems.append(f"{label} E4 table F0.5: {cell} vs metrics {want:.4f}")
        if f"{vb - va:+.4f}" != delta:
            problems.append(f"{a}->{b} E4 delta: {delta} vs computed {vb - va:+.4f}")
        checked += 3

    # ---- every linked file exists ----------------------------------------
    for target in re.findall(r"\]\((figures/[^)]+|tables/[^)]+|[\w.]+\.md)\)", text):
        if not (paths.RESULTS / target).exists():
            problems.append(f"RESULTS.md links {target}, which does not exist")

    # ---- the completeness count ------------------------------------------
    claim = re.search(r"\*\*(\d+) of (\d+) expected runs are complete\.\*\*", text)
    if claim:
        on_disk = sum(
            1 for d in sorted(paths.RUNS.glob("e*/seed*"))
            if (d / "metrics.json").exists()
            and paths.read_json(d / "metrics.json").get("_meta", {}).get("complete")
        )
        # Count every seed, not just seed 13: the expected grid is runs x
        # seeds, and E3 carries three seeds. An earlier version compared the
        # claim against seed-13 directories alone and reported a false
        # mismatch the moment the extra seeds landed.
        if int(claim.group(1)) > on_disk:
            problems.append(
                f"RESULTS.md claims {claim.group(1)} complete runs but only "
                f"{on_disk} run directories carry a complete metrics.json"
            )

    print(f"[audit] checked {checked} reported number(s) against metrics.json")
    if problems:
        print(f"[audit] {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("[audit] OK: every number in RESULTS.md matches the run that produced it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
