"""Check that every number in ``results/RESULTS.md`` came from a run's metrics.

::

    python scripts/audit_results.py          # exits non-zero on any mismatch

``CLAUDE.md`` rule: *never invent results; every reported number must come from
a file our code wrote.* ``aggregate.py`` is written to honour that, but the
claim is only worth as much as an independent check of it, so this script
re-reads the main table out of the rendered Markdown and compares every cell
against the ``metrics.json`` it is supposed to have come from - without going
through ``aggregate`` at all.

It also verifies that every file the report links to exists, and that the
report's completeness count matches the runs actually on disk.
"""

from __future__ import annotations

import re
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

    # ---- E3 / E4 delta tables --------------------------------------------
    ref = resolve("e3_real", 13)
    for run, f05, delta in re.findall(
        r"^\| (e3_\w+) \| (\d\.\d{4}) \| ([-+]\d\.\d{4}|-) \|", text, re.M
    ):
        m = resolve(run, 13)
        if m is None:
            problems.append(f"{run}: in the E3 table but has no metrics.json")
            continue
        got = dig(m, ("test", "errant", "f05"))
        if f"{float(got):.4f}" != f05:
            problems.append(f"{run} E3 table F0.5: {f05} vs metrics {got}")
        if delta != "-" and ref is not None:
            want = float(got) - float(dig(ref, ("test", "errant", "f05")))
            if f"{want:+.4f}" != delta:
                problems.append(f"{run} E3 delta: {delta} vs computed {want:+.4f}")
        checked += 2

    for a, b, fa, fb, delta in re.findall(
        r"^\| (e3_\w+) -> (e4_\w+) \| (\d\.\d{4}) \| (\d\.\d{4}) \| "
        r"([-+]\d\.\d{4}) \|", text, re.M
    ):
        ma, mb = resolve(a, 13), resolve(b, 13)
        if ma is None or mb is None:
            problems.append(f"{a}->{b}: in the E4 table but a metrics.json is missing")
            continue
        va = float(dig(ma, ("test", "errant", "f05")))
        vb = float(dig(mb, ("test", "errant", "f05")))
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
        seed13 = sum(
            1 for d in sorted(paths.RUNS.glob("e*/seed13"))
            if (d / "metrics.json").exists()
        )
        if int(claim.group(1)) > seed13:
            problems.append(
                f"RESULTS.md claims {claim.group(1)} complete runs but only "
                f"{seed13} seed-13 run directories exist ({on_disk} total)"
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
