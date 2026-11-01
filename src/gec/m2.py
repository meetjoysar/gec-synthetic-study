"""Parse, apply and write M2 files (spec section 5.2).

The M2 format, as used by BEA-2019 / W&I+LOCNESS::

    S This are a sentence .
    A 1 2|||R:VERB:SVA|||is|||REQUIRED|||-NONE-|||0
    <blank line>

An ``A`` line replaces source tokens ``[start, end)`` with ``correction``.
An empty correction is a deletion; ``start == end`` is an insertion.
``A -1 -1|||noop|||...`` means the annotator made no edit, and type ``UNK``
means "something is wrong here but we cannot classify it" - both are skipped
when building a target sentence.

Two design points carry most of the weight in this module:

* **:attr:`M2Block.raw` holds the original block text verbatim**, and
  :func:`write_m2` emits it. That makes a sliced or reordered subset of a gold
  M2 file round-trip byte-for-byte, which is what lets ``prepare_data`` carve a
  2,000-sentence ``val`` reference out of the W&I training M2 while keeping the
  *original* gold annotations. Re-deriving those edits by running ERRANT over
  (src, tgt) would differ in merge and classification decisions and make ``val``
  scores incomparable with ``test`` scores - and that comparability is exactly
  what E1/E2 hyperparameter selection depends on.

* **:func:`assert_aligned` exists because ERRANT's own check is unusable.**
  ``errant.commands.compare_m2.main`` does ``assert len(hyp_m2) == len(ref_m2)``
  with no message, so any sentence-count drift produces an undiagnosable crash
  minutes into an evaluation. We check first, with both paths and both counts in
  the message.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from gec import paths

#: Corrections that mean "no tokens" (a deletion, or a noop placeholder).
_EMPTY_COR = {"", "-NONE-"}

#: Incremented when two edits by the same annotator overlap. ``prepare_data
#: --inspect`` reports it rather than letting it pass silently.
OVERLAP_WARNINGS: Counter[str] = Counter()


@dataclass(frozen=True)
class Edit:
    """One ``A`` line: replace source tokens ``[start, end)`` with ``cor``."""

    start: int
    end: int
    type: str
    cor: str
    annotator: int = 0

    @property
    def is_noop(self) -> bool:
        """True for ``A -1 -1|||noop|||...`` - the annotator changed nothing."""
        return self.type == "noop" or self.start < 0

    @property
    def is_skipped(self) -> bool:
        """True if this edit must not be applied when building a target.

        Covers ``noop`` and ``UNK``; spec section 5.2 says to treat ``UNK`` as
        no change, because the annotator flagged an error without supplying a
        usable correction.
        """
        return self.is_noop or self.type == "UNK"

    @property
    def cor_tokens(self) -> list[str]:
        """The correction as a token list; empty for a deletion."""
        return [] if self.cor in _EMPTY_COR else self.cor.split()

    @property
    def cat2(self) -> str:
        """Category without the ``M:``/``U:``/``R:`` prefix, e.g. ``VERB:SVA``.

        Mirrors ERRANT's ``processCategories(cat, 2)`` exactly - it does
        ``cat[2:]`` and passes ``UNK`` through verbatim - so our counts and
        ``errant_compare -cat 2`` agree by construction rather than by luck.
        """
        if self.type == "UNK" or self.is_noop:
            return self.type
        return self.type[2:] if len(self.type) > 2 and self.type[1] == ":" else self.type

    @property
    def prefix(self) -> str | None:
        """``M``, ``U`` or ``R`` if the type carries one, else None."""
        if len(self.type) > 2 and self.type[1] == ":" and self.type[0] in "MUR":
            return self.type[0]
        return None

    @property
    def is_insertion(self) -> bool:
        return self.start == self.end and not self.is_noop

    @property
    def is_deletion(self) -> bool:
        return bool(self.cor in _EMPTY_COR and self.end > self.start)

    def to_m2(self) -> str:
        """Render back to an ``A`` line."""
        cor = self.cor if self.cor else "-NONE-"
        return f"A {self.start} {self.end}|||{self.type}|||{cor}|||REQUIRED|||-NONE-|||{self.annotator}"


@dataclass
class M2Block:
    """One sentence: its ``S`` line, its edits, and the verbatim original text."""

    src: str
    edits: list[Edit] = field(default_factory=list)
    raw: str = ""

    @property
    def tokens(self) -> list[str]:
        return self.src.split()

    def annotators(self) -> list[int]:
        """Sorted annotator ids appearing in this block (``[0]`` if none)."""
        ids = sorted({e.annotator for e in self.edits})
        return ids or [0]

    def edits_for(self, annotator: int = 0, applicable_only: bool = True) -> list[Edit]:
        """This annotator's edits, by default excluding ``noop``/``UNK``."""
        out = [e for e in self.edits if e.annotator == annotator]
        return [e for e in out if not e.is_skipped] if applicable_only else out

    def target_tokens(self, annotator: int = 0) -> list[str]:
        """Source tokens with this annotator's edits applied."""
        return apply_edits(self.tokens, self.edits_for(annotator))

    def target(self, annotator: int = 0) -> str:
        """The corrected sentence for this annotator, space-separated."""
        return " ".join(self.target_tokens(annotator))

    def targets(self) -> dict[int, str]:
        """Every annotator's corrected sentence - used for multi-reference BLEU."""
        return {a: self.target(a) for a in self.annotators()}

    def n_edits(self, annotator: int = 0) -> int:
        return len(self.edits_for(annotator))

    def is_clean(self, annotator: int = 0) -> bool:
        """True if this annotator considered the sentence already correct.

        Two legs, both needed for the over-correction metric (spec section 7):

        * no applicable edits at all (the ``noop`` case the spec names), **or**
        * the reconstructed target is token-identical to the source, which
          catches the rarer case of a non-noop edit that happens to rebuild the
          source exactly (e.g. an ``ORTH`` edit that normalises to the same
          tokens).
        """
        edits = self.edits_for(annotator)
        return not edits or self.target_tokens(annotator) == self.tokens

    def to_m2(self) -> str:
        """Render the block. Prefers :attr:`raw` so gold blocks round-trip."""
        if self.raw:
            return self.raw.rstrip("\n")
        lines = [f"S {self.src}"]
        lines.extend(e.to_m2() for e in self.edits)
        if not self.edits:
            lines.append(Edit(-1, -1, "noop", "-NONE-", 0).to_m2())
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_edit(line: str, _strict: bool = True) -> Edit | None:
    """Parse one ``A`` line. Returns None for a malformed line when lenient."""
    body = line[2:] if line.startswith("A ") else line
    parts = body.split("|||")
    if len(parts) < 3:
        if _strict:
            raise ValueError(f"malformed A line (need >=3 '|||' fields): {line!r}")
        return None
    span = parts[0].split()
    if len(span) != 2:
        if _strict:
            raise ValueError(f"malformed A-line span: {line!r}")
        return None
    try:
        start, end = int(span[0]), int(span[1])
    except ValueError:
        if _strict:
            raise ValueError(f"non-integer A-line span: {line!r}") from None
        return None
    etype, cor = parts[1], parts[2]
    annotator = 0
    if len(parts) >= 6:
        try:
            annotator = int(parts[5])
        except ValueError:
            annotator = 0
    return Edit(start=start, end=end, type=etype, cor=cor, annotator=annotator)


def parse_block(text: str) -> M2Block:
    """Parse one M2 block (an ``S`` line plus zero or more ``A`` lines)."""
    lines = [ln for ln in text.strip("\n").split("\n") if ln.strip()]
    if not lines or not lines[0].startswith("S "):
        raise ValueError(f"block does not start with an 'S ' line: {text[:120]!r}")
    src = lines[0][2:]
    edits: list[Edit] = []
    for ln in lines[1:]:
        if not ln.startswith("A "):
            continue
        e = parse_edit(ln)
        if e is not None:
            edits.append(e)
    return M2Block(src=src, edits=edits, raw=text.strip("\n"))


def read_m2(path: Path | str) -> Iterator[M2Block]:
    """Stream :class:`M2Block` objects from an M2 file.

    Blocks are separated by blank lines, matching how ERRANT reads them
    (``.read().strip().split("\\n\\n")``).
    """
    buf: list[str] = []
    for line in paths.read_lines(path):
        if line.strip():
            buf.append(line)
        elif buf:
            yield parse_block("\n".join(buf))
            buf = []
    if buf:
        yield parse_block("\n".join(buf))


def write_m2(blocks: Iterable[M2Block], path: Path | str) -> int:
    """Write blocks separated by blank lines; returns the block count.

    Emits :attr:`M2Block.raw` when present, so a sliced subset of a gold file is
    byte-identical to the original (modulo line endings, which are always LF).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for b in blocks:
            fh.write(b.to_m2() + "\n\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Applying edits
# --------------------------------------------------------------------------- #

def apply_edits(tokens: list[str], edits: Iterable[Edit], _warn_key: str = "") -> list[str]:
    """Apply edits to *tokens* and return the corrected token list.

    Edits are applied **right-to-left** (sorted by ``(start, end)`` descending)
    so earlier indices stay valid as the list is spliced. Insertions
    (``start == end``) and deletions (empty correction) need no special-casing
    under this scheme.

    Overlapping edits by the same annotator are not applicable in sequence; the
    first one in left-to-right order wins and :data:`OVERLAP_WARNINGS` is
    incremented so ``prepare_data --inspect`` can report it.
    """
    usable = [e for e in edits if not e.is_skipped]
    if not usable:
        return list(tokens)

    # Drop overlaps, keeping the leftmost edit of any overlapping pair.
    ordered = sorted(usable, key=lambda e: (e.start, e.end))
    kept: list[Edit] = []
    last_end = -1
    for e in ordered:
        if e.start < last_end:
            OVERLAP_WARNINGS[_warn_key or "total"] += 1
            continue
        kept.append(e)
        last_end = max(e.end, e.start)

    out = list(tokens)
    for e in sorted(kept, key=lambda e: (e.start, e.end), reverse=True):
        start = max(0, min(e.start, len(out)))
        end = max(start, min(e.end, len(out)))
        out[start:end] = e.cor_tokens
    return out


# --------------------------------------------------------------------------- #
# Counting and invariants
# --------------------------------------------------------------------------- #

def count_types(
    path: Path | str,
    annotator: int = 0,
    level: int = 2,
) -> Counter[str]:
    """Count ERRANT edit types in an M2 file.

    ``level=2`` strips the ``M:``/``U:``/``R:`` prefix (matching
    ``errant_compare -cat 2``); ``level=3`` keeps the full type. Skipped edits
    (``noop``, ``UNK``) are not counted.

    Used by E4 (spec section 7.1) instead of re-running ERRANT: the M2 files
    already exist, so re-deriving types ERRANT itself wrote would be wasted
    minutes.
    """
    counts: Counter[str] = Counter()
    for block in read_m2(path):
        for e in block.edits_for(annotator):
            counts[e.cat2 if level == 2 else e.type] += 1
    return counts


def block_count(path: Path | str) -> int:
    """Number of blocks in an M2 file."""
    return sum(1 for _ in read_m2(path))


def assert_aligned(m2_path: Path | str, src_path: Path | str) -> int:
    """Check an M2 file and a source text file line up; return the count.

    This guards the project's most likely runtime failure. ERRANT's
    ``compare_m2`` asserts ``len(hyp) == len(ref)`` with no message, so a drift
    of even one sentence yields a bare ``AssertionError`` after minutes of
    scoring. Checking here names both files, both counts, and the first
    mismatching sentence.
    """
    blocks = list(read_m2(m2_path))
    srcs = paths.read_lines(src_path)
    if len(blocks) != len(srcs):
        raise AssertionError(
            f"M2/source length mismatch:\n"
            f"  {m2_path}: {len(blocks)} blocks\n"
            f"  {src_path}: {len(srcs)} lines\n"
            "ERRANT's compare_m2 asserts these are equal without a message, so "
            "this must be fixed before scoring."
        )
    for i, (b, s) in enumerate(zip(blocks, srcs)):
        if b.src != s:
            raise AssertionError(
                f"M2/source text mismatch at sentence {i} (0-based):\n"
                f"  {m2_path}: {b.src!r}\n"
                f"  {src_path}: {s!r}"
            )
    return len(blocks)


def assert_same_length(a: Path | str, b: Path | str) -> int:
    """Check two text files have the same number of lines; return the count."""
    na, nb = paths.count_lines(a), paths.count_lines(b)
    if na != nb:
        raise AssertionError(
            f"line-count mismatch:\n  {a}: {na} lines\n  {b}: {nb} lines"
        )
    return na
