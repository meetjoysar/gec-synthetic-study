"""Capture a REAL errant_compare output as a test fixture.

The previous fixture was hand-written and its F-score column was internally
inconsistent, which made three tests assert the wrong numbers. Generating it
from an actual ERRANT run makes the parser tests a genuine regression guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
from gec import errant_cli, paths  # noqa: E402

OUT = Path("tests/fixtures/errant_compare_cat2.txt")

# A small set of sentences chosen so the output exercises several categories
# and a mix of TP / FP / FN, including an unchanged sentence (noop).
src = [
    "He go to school yesterday .",
    "She have been study English since three years .",
    "The informations in this report is not correct .",
    "I have two brothers .",
    "We discussed about the plan .",
    "A apple is on the table .",
]
# "Hypothesis": a plausible imperfect system output.
hyp = [
    "He goes to school yesterday .",          # right span, wrong correction vs gold
    "She has been study English since three years .",
    "The informations in this report are not correct .",
    "I have two brothers .",                   # correctly left alone
    "We discussed about the plan .",           # missed edit -> FN
    "An apple is on the table .",              # correct
]
# "Reference": the gold corrections.
ref = [
    "He went to school yesterday .",
    "She has been studying English for three years .",
    "The information in this report is not correct .",
    "I have two brothers .",
    "We discussed the plan .",
    "An apple is on the table .",
]

work = Path("tests/fixtures/_tmp")
work.mkdir(parents=True, exist_ok=True)
s, h, r = work / "src.txt", work / "hyp.txt", work / "ref.txt"
paths.write_lines(s, src)
paths.write_lines(h, hyp)
paths.write_lines(r, ref)

hyp_m2 = work / "hyp.m2"
ref_m2 = work / "ref.m2"
errant_cli.parallel_to_m2(s, [h], hyp_m2)
errant_cli.parallel_to_m2(s, [r], ref_m2)

out = errant_cli.compare_m2(hyp_m2, ref_m2, cat=2, beta=0.5)
OUT.parent.mkdir(parents=True, exist_ok=True)
paths.write_text(OUT, out)
print(out)
print(f"wrote {OUT}")

# Keep the M2 pair too: it lets the cross-check test run without ERRANT.
paths.write_text(Path("tests/fixtures/golden_hyp.m2"), paths.read_text(hyp_m2))
paths.write_text(Path("tests/fixtures/golden_ref.m2"), paths.read_text(ref_m2))
print("wrote tests/fixtures/golden_{hyp,ref}.m2")

import shutil

shutil.rmtree(work, ignore_errors=True)
