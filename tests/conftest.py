"""Shared pytest fixtures.

``pythonpath = ["src"]`` in pyproject.toml makes ``import gec`` work, but the
explicit insert below keeps the suite runnable with a bare ``pytest`` from any
directory and inside Colab, where the rootdir detection sometimes differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


#: A hand-built M2 fixture exercising every edit kind the spec names
#: (spec section 5.2): replacement, insertion, deletion, multiple edits, noop
#: and UNK. Written as a literal so the tests do not depend on a download.
M2_FIXTURE = """S This are a sentence .
A 1 2|||R:VERB:SVA|||is|||REQUIRED|||-NONE-|||0

S I go to school yesterday .
A 1 2|||R:VERB:TENSE|||went|||REQUIRED|||-NONE-|||0
A 4 4|||M:DET|||the|||REQUIRED|||-NONE-|||0

S She is very very happy .
A 2 3|||U:ADV||||||REQUIRED|||-NONE-|||0

S This sentence is already correct .
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0

S Something odd happens here .
A 1 3|||UNK|||-NONE-|||REQUIRED|||-NONE-|||0
"""


@pytest.fixture
def m2_fixture_text() -> str:
    """The raw text of :data:`M2_FIXTURE`."""
    return M2_FIXTURE


@pytest.fixture
def m2_fixture_path(tmp_path: Path) -> Path:
    """:data:`M2_FIXTURE` written to a temp file with LF newlines."""
    p = tmp_path / "fixture.m2"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(M2_FIXTURE)
    return p


@pytest.fixture
def multi_annotator_m2(tmp_path: Path) -> Path:
    """Two annotators disagreeing on one sentence - for multi-reference tests."""
    # Two valid but different corrections: annotator 0 deletes the intrusive
    # preposition, annotator 1 changes the verb instead.
    text = """S We discussed about the plan .
A 2 3|||U:PREP||||||REQUIRED|||-NONE-|||0
A 1 2|||R:VERB|||talked|||REQUIRED|||-NONE-|||1

S He is working here since 2019 .
A 1 3|||R:VERB:TENSE|||has been working|||REQUIRED|||-NONE-|||0
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||1
"""
    p = tmp_path / "multi.m2"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return p
