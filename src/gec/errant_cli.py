"""Subprocess wrapper around the ERRANT command-line tools.

Not in the spec's module list (section 4), but added because three modules need
it - :mod:`gec.evaluate`, :mod:`gec.kappa` and :mod:`gec.distribution` - and
because keeping ERRANT behind ``subprocess`` means :mod:`gec.train` never
imports spaCy at all. ``errant.load("en")`` costs several seconds and a few
hundred MB of RAM, which has no business being inside a training process.

Two things here are non-obvious and were verified against the installed
``errant==3.0.2`` source rather than assumed:

1. **``parallel_to_m2.py`` has no ``if __name__ == "__main__"`` guard** (only
   ``compare_m2.py`` does). So ``python -m errant.commands.parallel_to_m2``
   runs the module, defines ``main``, and exits having done *nothing* - a
   silent no-op that would waste hours. :func:`_cmd` therefore uses
   ``python -c "from <module> import main; main()"``, which works for every
   entrypoint and does not depend on the console scripts in ``Scripts/`` being
   on PATH.

2. **ERRANT opens every file with a bare ``open()``** - no ``encoding=``. On
   Windows under Python 3.10 that means cp1252, and the W&I essays contain
   curly quotes, en-dashes and accented names, so a ``UnicodeDecodeError`` is
   guaranteed rather than merely possible. :func:`_env` sets ``PYTHONUTF8=1``
   in the child environment, which fixes it globally for the subprocess.

Also worth knowing: ``compare_m2.main`` does ``assert len(hyp_m2) ==
len(ref_m2)`` with **no message**, so any sentence-count drift produces an
undiagnosable crash. Callers must run :func:`gec.m2.assert_aligned` first, and
:func:`run` echoes the full stderr on failure so what is left is at least
readable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from gec import paths

#: ERRANT entrypoint modules, each exposing a ``main()`` that parses ``sys.argv``.
ENTRYPOINTS = {
    "parallel": "errant.commands.parallel_to_m2",
    "compare": "errant.commands.compare_m2",
    "m2": "errant.commands.m2_to_m2",
}


class ErrantError(RuntimeError):
    """An ERRANT subprocess exited non-zero."""


def _cmd(entry: str, args: list[str]) -> list[str]:
    """Build the argv for an ERRANT entrypoint.

    Uses the ``-c "from ... import main; main()"`` form because
    ``parallel_to_m2`` lacks a ``__main__`` guard, and ``sys.executable`` so the
    current interpreter (and therefore the right environment) is always used
    regardless of PATH.
    """
    if entry not in ENTRYPOINTS:
        raise KeyError(f"unknown ERRANT entrypoint {entry!r}; known: {sorted(ENTRYPOINTS)}")
    mod = ENTRYPOINTS[entry]
    return [sys.executable, "-c", f"from {mod} import main; main()", *(str(a) for a in args)]


def _env() -> dict[str, str]:
    """Child environment with UTF-8 forced on."""
    e = os.environ.copy()
    e["PYTHONUTF8"] = "1"        # ERRANT uses bare open(); Windows defaults to cp1252
    e["PYTHONIOENCODING"] = "utf-8"
    e.setdefault("PYTHONNOUSERSITE", "1")
    return e


def run(entry: str, args: list[str], *, timeout: int = 3600, quiet: bool = False) -> str:
    """Run an ERRANT entrypoint and return its stdout.

    Never uses ``shell=True``: argv is passed as a list, so spaces in paths
    (``C:\\Users\\91942\\Downloads\\...``) need no quoting and cannot be
    misparsed.
    """
    cmd = _cmd(entry, args)
    if not quiet:
        print(f"[errant] {entry} {' '.join(str(a) for a in args)}")
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_env(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ErrantError(f"{entry} timed out after {timeout}s\ncmd={cmd}") from exc

    if p.returncode != 0:
        raise ErrantError(
            f"ERRANT {entry} failed (returncode={p.returncode})\n"
            f"cmd={cmd}\n"
            f"--- stdout (last 4000 chars) ---\n{p.stdout[-4000:]}\n"
            f"--- stderr (last 4000 chars) ---\n{p.stderr[-4000:]}\n"
            "Note: compare_m2 asserts hypothesis and reference have equal "
            "sentence counts without a message - run m2.assert_aligned first."
        )
    return p.stdout


def parallel_to_m2(
    orig: Path | str,
    cors: list[Path | str],
    out: Path | str,
    *,
    timeout: int = 3600,
) -> Path:
    """Build an M2 file from parallel original/corrected text files.

    Passing several *cors* writes one M2 with annotator ids 0, 1, ... in a
    single pass. Only the CLI does that, which is why the Indian English
    two-annotator reference is built here rather than through the Python API.

    We deliberately do **not** pass ERRANT's ``-tok`` flag. It defaults to False
    and asks spaCy to re-tokenise the input; our files are already
    spaCy-tokenised and space-separated (spec section 5.3), so re-tokenising
    could shift token counts and invalidate every edit span.
    """
    orig, out = Path(orig), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = ["-orig", str(orig), "-cor", *[str(c) for c in cors], "-out", str(out)]
    n_orig = paths.count_lines(orig)
    print(f"[errant] annotating {n_orig} sentence pairs "
          f"({len(cors)} annotator{'s' if len(cors) > 1 else ''}); this takes a few minutes")
    run("parallel", args, timeout=timeout, quiet=True)
    if not out.exists():
        raise ErrantError(
            f"parallel_to_m2 reported success but {out} does not exist. "
            "This is the signature of the missing __main__ guard - check _cmd()."
        )
    return out


def compare_m2(
    hyp: Path | str,
    ref: Path | str,
    *,
    cat: int | None = 2,
    beta: float = 0.5,
    timeout: int = 3600,
) -> str:
    """Score a hypothesis M2 against a reference M2 and return raw stdout.

    With ``cat=2`` a single call prints **both** the per-category table and the
    overall span-level table (they come from the same ``best`` dict inside
    ERRANT), so one subprocess is enough for everything
    :func:`gec.evaluate.errant_scores` needs.
    """
    args: list[str] = ["-hyp", str(hyp), "-ref", str(ref), "-b", str(beta)]
    if cat is not None:
        args += ["-cat", str(cat)]
    return run("compare", args, timeout=timeout, quiet=True)


def errant_version() -> str:
    """Installed ERRANT version, or ``"unknown"``."""
    try:
        import errant

        return getattr(errant, "__version__", "unknown")
    except ImportError:
        return "not-installed"


def self_check() -> bool:
    """Round-trip a non-ASCII sentence through ERRANT to prove the env is sane.

    Exercises exactly the two failure modes this module exists to prevent: the
    silent no-op from the missing ``__main__`` guard, and the cp1252
    ``UnicodeDecodeError`` on non-ASCII input.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        orig, cor, out = d / "orig.txt", d / "cor.txt", d / "out.m2"
        # Deliberately includes characters that break a cp1252 read.
        paths.write_lines(orig, ["He go to school .", 'She said " caf\u00e9 \u2013 oui " .'])
        paths.write_lines(cor, ["He goes to school .", 'She said " caf\u00e9 \u2013 oui " .'])
        parallel_to_m2(orig, [cor], out)
        text = paths.read_text(out)
        ok = "R:VERB:SVA" in text
        print(f"[errant] self-check {'PASSED' if ok else 'FAILED'} (errant {errant_version()})")
        if not ok:
            print(text)
        return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_check() else 1)
