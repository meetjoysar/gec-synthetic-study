"""Rule-based synthetic error generator (spec section 6).

::

    python -m gec.error_gen --demo 20 [--p 0.2] [--seed 13]
    python -m gec.error_gen --samples 200 [--out results/tables/generator_samples.tsv]
    python -m gec.error_gen --list-ops

**Direction** - the easiest thing in this project to get backwards:
:meth:`ErrorGenerator.corrupt` takes a *clean* sentence and returns a *noisy*
one. The training pair is therefore ``src = noisy``, ``tgt = clean``, and the
:class:`GenEdit` log describes clean -> noisy, i.e. the inverse of the
correction a model must learn. :mod:`gec.distribution` depends on getting this
right: when it measures the generator's error distribution it must run ERRANT
with ``-orig noisy -cor clean`` so the counted types match a learner corpus.

Five design decisions carry the correctness of this module:

1. **Collect-then-apply, never mutate incrementally.** Every applicability
   check and every :class:`GenEdit` is computed against the *original* token and
   tag arrays; only once selection is finished does :func:`_splice` apply them
   right-to-left. Without this, a single deletion invalidates every later index
   and every cached POS tag - the likeliest source of silent corruption here.

2. **"Never corrupt the same token twice" is enforced by a ``used`` mask** over
   original indices, so it is deterministic and directly testable.

3. **``applies`` is separate from ``apply``, cheap, and side-effect-free.** For
   each selected token we collect the *applicable* ops, then sample among those
   with probabilities renormalised over that subset. Restricting-then-
   renormalising (rather than sampling globally and discarding) is what keeps
   the realised op mix close to ``weights`` - which is precisely what E4's
   re-weighting assumes.

4. **``force_one``** retries the whole pass up to 10 times, then forces one
   applicable edit, exactly as spec section 6.1 requires.

5. **All inflection goes through :meth:`Ctx.inflect`**, so lemminflect's quirks
   are handled in one place. ``match_case`` there preserves sentence-initial
   capitalisation, which stops ``verb_tense`` from *also* producing an ``ORTH``
   error and polluting the error-type distribution that E4 measures.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from gec import paths, tok

#: Incremented when ``force_one=True`` could not produce any edit at all (for
#: example on punctuation-only input). Surfaced in dataset stats so a silently
#: identity-heavy training set cannot pass unnoticed.
FORCE_ONE_FAILURES: Counter[str] = Counter()

DETERMINERS = ("a", "an", "the")
PREPOSITIONS = (
    "in", "on", "at", "for", "to", "of", "with", "from", "by", "about", "into",
)
SVA_SWAPS = {
    "is": "are", "are": "is",
    "was": "were", "were": "was",
    "has": "have", "have": "has",
    "does": "do", "do": "does",
}
#: Short function words `word_delete` may drop (spec section 6.2, OTHER).
FUNCTION_WORDS = ("is", "are", "was", "were", "to", "that", "of", "a", "the", "be", "it")
VOWEL_SOUNDS = "aeiou"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GenEdit:
    """One corruption, expressed against the ORIGINAL token indices."""

    op: str
    errant_type: str
    start: int
    end: int
    before: tuple[str, ...]
    after: tuple[str, ...]

    def __str__(self) -> str:
        b = " ".join(self.before) or "0"
        a = " ".join(self.after) or "0"
        return f"{self.op}[{self.start}:{self.end}] {b!r} -> {a!r}"


@dataclass(frozen=True)
class Ctx:
    """Read-only view of one clean sentence, with POS and lemma information.

    All four lists are parallel and are never mutated during corruption.
    """

    toks: tuple[str, ...]
    upos: tuple[str, ...]
    tag: tuple[str, ...]
    lemma: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.toks)

    @classmethod
    def from_tagged(cls, t: tok.Tagged) -> "Ctx":
        return cls(tuple(t.toks), tuple(t.upos), tuple(t.tag), tuple(t.lemma))

    @classmethod
    def build(
        cls,
        toks: Sequence[str],
        upos: Sequence[str] | None = None,
        tag: Sequence[str] | None = None,
        lemma: Sequence[str] | None = None,
    ) -> "Ctx":
        """Build a context, tagging with spaCy only if tags were not supplied."""
        if upos is None or tag is None or lemma is None:
            tagged = tok.tag_one(" ".join(toks))
            if len(tagged.toks) == len(toks):
                return cls.from_tagged(tagged)
            # Re-tokenisation disagreed; fall back to empty annotations so the
            # applicability checks simply decline rather than crash.
            n = len(toks)
            return cls(tuple(toks), ("X",) * n, ("XX",) * n, tuple(toks))
        return cls(tuple(toks), tuple(upos), tuple(tag), tuple(lemma))

    # -- small predicates used by the ops ---------------------------------- #

    def is_noun(self, i: int) -> bool:
        return self.upos[i] in ("NOUN", "PROPN")

    def is_verbish(self, i: int) -> bool:
        return self.upos[i] in ("VERB", "AUX")

    def is_punct(self, i: int) -> bool:
        return self.upos[i] == "PUNCT" or not any(c.isalnum() for c in self.toks[i])

    def is_det(self, i: int) -> bool:
        return self.toks[i].lower() in DETERMINERS

    def is_prep(self, i: int) -> bool:
        return self.toks[i].lower() in PREPOSITIONS and self.upos[i] in ("ADP", "PART")

    def inflect(self, i: int, target_tag: str) -> str | None:
        """Inflect token *i* to *target_tag* via lemminflect.

        Returns None when lemminflect has no form, or the form is unchanged.
        Case is matched to the original surface form, which keeps a
        sentence-initial word capitalised and so avoids emitting an incidental
        ``ORTH`` error on top of the intended one.
        """
        import lemminflect

        lemma = self.lemma[i] or self.toks[i]
        try:
            forms = lemminflect.getInflection(lemma.lower(), tag=target_tag)
        except (KeyError, ValueError):
            return None
        if not forms:
            return None
        out = match_case(self.toks[i], forms[0])
        return None if out == self.toks[i] else out


@dataclass
class Op:
    """One corruption operation and its ERRANT category."""

    name: str
    errant_type: str
    applies: Callable[[Ctx, int], bool]
    apply: Callable[[Ctx, int, random.Random], GenEdit | None]
    doc: str = ""


OPS: dict[str, Op] = {}


def match_case(original: str, new: str) -> str:
    """Give *new* the capitalisation pattern of *original*.

    The lowercase branch is explicit rather than a pass-through: an inflected
    form arriving capitalised where the original was lowercase would add an
    incidental ``ORTH`` error on top of the intended one, skewing the
    error-type distribution that E4 measures.
    """
    if not original or not new:
        return new
    if original.isupper() and len(original) > 1:
        return new.upper()
    if original[0].isupper():
        return new[0].upper() + new[1:]
    return new[0].lower() + new[1:]


def _register(name: str, errant_type: str, doc: str = ""):
    """Decorator registering a function as both ``applies`` and ``apply``.

    The decorated function returns a :class:`GenEdit` or None. Applicability is
    then "would this return a GenEdit?", evaluated through a cheap probe. Ops
    that need an expensive check define ``applies`` separately via
    :func:`_register_pair`.
    """

    def deco(fn: Callable[[Ctx, int, random.Random], GenEdit | None]):
        def applies(ctx: Ctx, i: int, _fn=fn) -> bool:
            return _fn(ctx, i, random.Random(0)) is not None

        OPS[name] = Op(name=name, errant_type=errant_type, applies=applies,
                       apply=fn, doc=doc or (fn.__doc__ or "").strip())
        return fn

    return deco


# --------------------------------------------------------------------------- #
# Operations (spec section 6.2)
# --------------------------------------------------------------------------- #

@_register("det_delete", "DET", "Delete a/an/the.")
def _det_delete(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    if not ctx.is_det(i):
        return None
    return GenEdit("det_delete", "DET", i, i + 1, (ctx.toks[i],), ())


@_register("det_insert", "DET", "Insert a/an/the before a noun.")
def _det_insert(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    if not ctx.is_noun(i):
        return None
    # Do not stack a second determiner on one that is already there.
    if i > 0 and (ctx.is_det(i - 1) or ctx.upos[i - 1] in ("DET", "PRON")):
        return None
    word = ctx.toks[i]
    det = "an" if word[:1].lower() in VOWEL_SOUNDS else rng.choice(("a", "the"))
    det = match_case(word, det) if i == 0 else det
    return GenEdit("det_insert", "DET", i, i, (), (det,))


@_register("det_swap", "DET", "Swap among a/an/the.")
def _det_swap(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    if not ctx.is_det(i):
        return None
    cur = ctx.toks[i].lower()
    choices = [d for d in DETERMINERS if d != cur]
    new = match_case(ctx.toks[i], rng.choice(choices))
    if new == ctx.toks[i]:
        return None
    return GenEdit("det_swap", "DET", i, i + 1, (ctx.toks[i],), (new,))


@_register("prep_swap", "PREP", "Swap one preposition for another.")
def _prep_swap(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    if not ctx.is_prep(i):
        return None
    cur = ctx.toks[i].lower()
    choices = [p for p in PREPOSITIONS if p != cur]
    new = match_case(ctx.toks[i], rng.choice(choices))
    if new == ctx.toks[i]:
        return None
    return GenEdit("prep_swap", "PREP", i, i + 1, (ctx.toks[i],), (new,))


@_register("prep_delete", "PREP", "Delete a preposition.")
def _prep_delete(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    if not ctx.is_prep(i):
        return None
    return GenEdit("prep_delete", "PREP", i, i + 1, (ctx.toks[i],), ())


@_register("sva", "VERB:SVA", "Break subject-verb agreement.")
def _sva(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    word = ctx.toks[i]
    low = word.lower()
    if low in SVA_SWAPS:
        new = match_case(word, SVA_SWAPS[low])
        if new != word:
            return GenEdit("sva", "VERB:SVA", i, i + 1, (word,), (new,))
        return None
    if not ctx.is_verbish(i):
        return None
    # walks -> walk, walk -> walks
    target = {"VBZ": "VBP", "VBP": "VBZ"}.get(ctx.tag[i])
    if target is None:
        return None
    new = ctx.inflect(i, target)
    return GenEdit("sva", "VERB:SVA", i, i + 1, (word,), (new,)) if new else None


@_register("verb_tense", "VERB:TENSE", "Swap past and present tense.")
def _verb_tense(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    if not ctx.is_verbish(i):
        return None
    cur = ctx.tag[i]
    if cur == "VBD":            # went -> go / goes
        target = rng.choice(("VBP", "VBZ"))
    elif cur in ("VBP", "VBZ"):  # walks -> walked
        target = "VBD"
    else:
        return None
    new = ctx.inflect(i, target)
    return GenEdit("verb_tense", "VERB:TENSE", i, i + 1, (ctx.toks[i],), (new,)) if new else None


@_register("verb_form", "VERB:FORM", "Wrong verb form after to/modal/auxiliary.")
def _verb_form(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    if not ctx.is_verbish(i) or i == 0:
        return None
    prev = ctx.toks[i - 1].lower()
    prev_tag = ctx.tag[i - 1]
    if prev == "to" or prev_tag == "MD":
        # "to go" -> "to going"; "can go" -> "can goes"
        target = rng.choice(("VBG", "VBZ"))
    elif prev_tag in ("VBZ", "VBP", "VBD") and prev in ("is", "are", "was", "were"):
        # "is going" -> "is go"
        target = "VB"
    else:
        return None
    if ctx.tag[i] == target:
        return None
    new = ctx.inflect(i, target)
    return GenEdit("verb_form", "VERB:FORM", i, i + 1, (ctx.toks[i],), (new,)) if new else None


@_register("noun_num", "NOUN:NUM", "Swap singular and plural nouns.")
def _noun_num(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    if ctx.upos[i] != "NOUN":
        return None
    target = {"NN": "NNS", "NNS": "NN"}.get(ctx.tag[i])
    if target is None:
        return None
    new = ctx.inflect(i, target)
    return GenEdit("noun_num", "NOUN:NUM", i, i + 1, (ctx.toks[i],), (new,)) if new else None


@_register("spell", "SPELL", "Insert/delete/substitute/transpose one character.")
def _spell(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    word = ctx.toks[i]
    if len(word) < 4 or not word.isalpha():
        return None
    kind = rng.choice(("sub", "del", "ins", "swap"))
    j = rng.randrange(1, len(word))  # leave the first character alone
    if kind == "sub":
        letter = rng.choice("abcdefghijklmnopqrstuvwxyz")
        new = word[:j] + letter + word[j + 1:]
    elif kind == "del":
        new = word[:j] + word[j + 1:]
    elif kind == "ins":
        new = word[:j] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[j:]
    else:
        if j + 1 >= len(word):
            j = len(word) - 2
        if j < 1:
            return None
        new = word[:j] + word[j + 1] + word[j] + word[j + 2:]
    if new == word or not new:
        return None
    return GenEdit("spell", "SPELL", i, i + 1, (word,), (new,))


@_register("punct_delete", "PUNCT", "Drop a comma or the final period.")
def _punct_delete(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    if ctx.toks[i] not in (",", "."):
        return None
    if ctx.toks[i] == "." and i != len(ctx) - 1:
        return None
    return GenEdit("punct_delete", "PUNCT", i, i + 1, (ctx.toks[i],), ())


@_register("punct_insert", "PUNCT", "Insert a stray comma.")
def _punct_insert(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    # Not at the very start, not adjacent to existing punctuation, and not
    # before the final token (which would read as a legitimate comma).
    if i <= 0 or i >= len(ctx) - 1:
        return None
    if ctx.is_punct(i) or ctx.is_punct(i - 1):
        return None
    return GenEdit("punct_insert", "PUNCT", i, i, (), (",",))


@_register("word_delete", "OTHER", "Delete a short function word.")
def _word_delete(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    if ctx.toks[i].lower() not in FUNCTION_WORDS:
        return None
    if ctx.is_det(i) or ctx.is_prep(i):
        return None  # leave those to det_delete / prep_delete
    return GenEdit("word_delete", "OTHER", i, i + 1, (ctx.toks[i],), ())


@_register("word_dup", "OTHER", "Duplicate a token.")
def _word_dup(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    if ctx.is_punct(i):
        return None
    if i + 1 < len(ctx) and ctx.toks[i + 1] == ctx.toks[i]:
        return None  # already repeated
    return GenEdit("word_dup", "OTHER", i, i, (), (ctx.toks[i],))


@_register("case", "ORTH", "Lowercase the first word, or capitalise a mid-sentence word.")
def _case(ctx: Ctx, i: int, rng: random.Random) -> GenEdit | None:
    word = ctx.toks[i]
    if not word[:1].isalpha():
        return None
    if i == 0:
        if not word[0].isupper():
            return None
        new = word[0].lower() + word[1:]
    else:
        # Only common words; capitalising a proper noun changes nothing useful.
        if ctx.upos[i] == "PROPN" or word[0].isupper() or len(word) < 3:
            return None
        new = word[0].upper() + word[1:]
    if new == word:
        return None
    return GenEdit("case", "ORTH", i, i + 1, (word,), (new,))


#: Default weights live in configs/base.yaml; this mirrors them so the module is
#: usable standalone (``--demo``) without loading a config.
DEFAULT_WEIGHTS: dict[str, float] = {
    "det_delete": 0.06, "det_insert": 0.05, "det_swap": 0.05,
    "prep_swap": 0.09, "prep_delete": 0.07,
    "sva": 0.10, "verb_tense": 0.10, "verb_form": 0.08,
    "noun_num": 0.10, "spell": 0.08,
    "punct_delete": 0.06, "punct_insert": 0.04,
    "word_delete": 0.05, "word_dup": 0.04,
    "case": 0.03,
}


def covered_errant_types(weights: Iterable[str] | None = None) -> set[str]:
    """ERRANT categories the generator can produce (E4 needs this)."""
    names = list(weights) if weights is not None else list(OPS)
    return {OPS[n].errant_type for n in names if n in OPS}


# --------------------------------------------------------------------------- #
# Applying edits
# --------------------------------------------------------------------------- #

def _splice(toks: Sequence[str], edits: Sequence[GenEdit]) -> list[str]:
    """Apply edits right-to-left so earlier indices stay valid."""
    out = list(toks)
    for e in sorted(edits, key=lambda x: (x.start, x.end), reverse=True):
        out[e.start:e.end] = list(e.after)
    return out


class ErrorGenerator:
    """Corrupt clean sentences into plausible learner errors (spec section 6.1).

    Deterministic given ``seed``: one :class:`random.Random` is created in
    ``__init__`` and consumed sequentially, so a given
    ``(seed, p, sentence order)`` always produces the same corpus.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        seed: int = 13,
        enabled: list[str] | None = None,
    ) -> None:
        w = dict(weights or DEFAULT_WEIGHTS)
        unknown = set(w) - set(OPS)
        if unknown:
            raise ValueError(f"unknown generator ops in weights: {sorted(unknown)}. "
                             f"Known ops: {sorted(OPS)}")
        if enabled is not None:
            w = {k: v for k, v in w.items() if k in enabled}
            if not w:
                raise ValueError(f"no enabled ops left; enabled={enabled}")
        total = sum(w.values())
        if total <= 0:
            raise ValueError("generator weights must sum to a positive number")
        self.weights = {k: v / total for k, v in w.items()}
        self.seed = seed
        self.rng = random.Random(seed)
        self._op_counts: Counter[str] = Counter()
        self._n_sentences = 0
        self._n_changed = 0

    @property
    def op_counts(self) -> Counter[str]:
        return self._op_counts.copy()

    @property
    def stats(self) -> dict:
        return {
            "n_sentences": self._n_sentences,
            "n_changed": self._n_changed,
            "op_counts": dict(self._op_counts.most_common()),
            "total_edits": sum(self._op_counts.values()),
            "edits_per_changed_sentence": (
                round(sum(self._op_counts.values()) / self._n_changed, 3)
                if self._n_changed else 0.0
            ),
            "force_one_failures": FORCE_ONE_FAILURES.get("total", 0),
        }

    # -- the selection loop ------------------------------------------------ #

    def _one_pass(self, ctx: Ctx, p: float, rng: random.Random) -> list[GenEdit]:
        """One corruption pass: select tokens, pick an applicable op for each."""
        used = [False] * len(ctx)
        edits: list[GenEdit] = []
        names = list(self.weights)

        for i in range(len(ctx)):
            if rng.random() >= p:
                continue
            if used[i]:
                continue
            # Applicable ops only, with weights renormalised over that subset.
            cands = [n for n in names if OPS[n].applies(ctx, i)]
            if not cands:
                continue
            ws = [self.weights[n] for n in cands]
            choice = rng.choices(cands, weights=ws, k=1)[0]
            edit = OPS[choice].apply(ctx, i, rng)
            if edit is None:
                continue
            # Claim the span. An insertion claims its single slot.
            lo, hi = edit.start, max(edit.end, edit.start + 1)
            if any(used[j] for j in range(lo, min(hi, len(ctx)))):
                continue
            for j in range(lo, min(hi, len(ctx))):
                used[j] = True
            edits.append(edit)
        return edits

    def _force_one(self, ctx: Ctx, rng: random.Random) -> list[GenEdit]:
        """Guarantee one edit: shuffle all applicable (token, op) pairs and take one."""
        pairs = [
            (i, n)
            for i in range(len(ctx))
            for n in self.weights
            if OPS[n].applies(ctx, i)
        ]
        rng.shuffle(pairs)
        for i, n in pairs:
            edit = OPS[n].apply(ctx, i, rng)
            if edit is not None:
                return [edit]
        FORCE_ONE_FAILURES["total"] += 1
        return []

    def corrupt(
        self,
        tokens: Sequence[str],
        pos_tags: Sequence[str] | None = None,
        p: float = 0.2,
        force_one: bool = False,
        *,
        tag: Sequence[str] | None = None,
        lemma: Sequence[str] | None = None,
    ) -> tuple[list[str], list[GenEdit]]:
        """Corrupt *tokens* and return ``(noisy_tokens, edits)``.

        Signature matches spec section 6.1. ``pos_tags`` is UPOS; ``tag``
        (PTB) and ``lemma`` are optional and come free from the cached clean
        pool, which avoids a spaCy call per sentence.
        """
        ctx = Ctx.build(tokens, pos_tags, tag, lemma)
        return self.corrupt_ctx(ctx, p=p, force_one=force_one)

    def corrupt_tagged(
        self, t: tok.Tagged, p: float = 0.2, force_one: bool = False
    ) -> tuple[list[str], list[GenEdit]]:
        """Convenience path used by :mod:`gec.build_datasets` (tags already cached)."""
        return self.corrupt_ctx(Ctx.from_tagged(t), p=p, force_one=force_one)

    def corrupt_ctx(
        self, ctx: Ctx, p: float = 0.2, force_one: bool = False
    ) -> tuple[list[str], list[GenEdit]]:
        """Corrupt a prepared :class:`Ctx`."""
        self._n_sentences += 1
        if len(ctx) == 0:
            return [], []

        edits: list[GenEdit] = []
        if p > 0:
            edits = self._one_pass(ctx, p, self.rng)
            if force_one and not edits:
                for _ in range(10):  # spec 6.1: retry up to 10 times
                    edits = self._one_pass(ctx, p, self.rng)
                    if edits:
                        break
        if force_one and not edits:
            edits = self._force_one(ctx, self.rng)

        for e in edits:
            self._op_counts[e.op] += 1
        if edits:
            self._n_changed += 1
        return _splice(ctx.toks, edits), edits


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_pool_or_fallback(n: int) -> list[tok.Tagged]:
    """Load the clean pool, or tag a few built-in sentences if it is missing."""
    try:
        from gec.prepare_data import load_pool

        return load_pool(limit=max(n * 4, 200))
    except FileNotFoundError:
        print("[gen] clean pool not built yet; using built-in demo sentences.\n"
              "      Run `python -m gec.prepare_data --wikitext` for the real pool.")
        demo = [
            "The researchers published a paper about the new method .",
            "She has been working at the university since 2019 .",
            "He goes to the library every morning before class .",
            "The students were discussing the results of the experiment .",
            "This book describes the history of the city .",
            "A small group of scientists is studying the problem .",
            "They have finished the project and submitted the report .",
            "The committee decided to postpone the meeting until Friday .",
        ]
        return list(tok.tag_batch(demo))


def demo(n: int = 20, p: float = 0.2, seed: int = 13) -> None:
    """Print *n* clean -> noisy examples for human inspection (spec 6.2)."""
    pool = _load_pool_or_fallback(n)
    gen = ErrorGenerator(seed=seed)
    rng = random.Random(seed)
    picks = rng.sample(range(len(pool)), min(n, len(pool)))

    print(f"\n{'=' * 78}")
    print(f"Generator demo: {len(picks)} examples, p={p}, seed={seed}, force_one=True")
    print("Reading direction: CLEAN is the training TARGET, NOISY is the training INPUT.")
    print("=" * 78)
    for k, i in enumerate(picks, 1):
        t = pool[i]
        noisy, edits = gen.corrupt_tagged(t, p=p, force_one=True)
        print(f"\n[{k:02d}] clean: {t.text}")
        print(f"     noisy: {' '.join(noisy)}")
        for e in edits:
            print(f"       - {e}  ({e.errant_type})")
    print(f"\n{'=' * 78}")
    print("op counts:", dict(gen.op_counts.most_common()))
    print("=" * 78)


def samples(n: int = 200, p: float = 0.2, seed: int = 13, out: str | None = None) -> paths.Path:
    """Write *n* clean/noisy pairs to a TSV for the report (spec 6.2)."""
    pool = _load_pool_or_fallback(n)
    gen = ErrorGenerator(seed=seed)
    rng = random.Random(seed)
    picks = rng.sample(range(len(pool)), min(n, len(pool)))

    dest = paths.Path(out) if out else paths.TABLES / "generator_samples.tsv"
    rows = ["clean\tnoisy\tn_edits\tops\terrant_types"]
    for i in picks:
        t = pool[i]
        noisy, edits = gen.corrupt_tagged(t, p=p, force_one=True)
        rows.append(
            "\t".join(
                [
                    t.text,
                    " ".join(noisy),
                    str(len(edits)),
                    ",".join(e.op for e in edits),
                    ",".join(e.errant_type for e in edits),
                ]
            )
        )
    paths.write_lines(dest, rows)
    print(f"[gen] wrote {len(rows) - 1} samples to {paths.rel(dest)}")
    print(f"[gen] op counts: {dict(gen.op_counts.most_common())}")
    return dest


def list_ops() -> None:
    """Print the op table (spec section 6.2)."""
    print(f"\n{'op':<14} {'ERRANT type':<12} {'default w':>9}  description")
    print("-" * 78)
    for name in sorted(OPS, key=lambda n: (OPS[n].errant_type, n)):
        op = OPS[name]
        w = DEFAULT_WEIGHTS.get(name, 0.0)
        print(f"{name:<14} {op.errant_type:<12} {w:>9.3f}  {op.doc}")
    types = sorted({o.errant_type for o in OPS.values()})
    print("-" * 78)
    print(f"{len(OPS)} ops covering {len(types)} ERRANT categories: {', '.join(types)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gec.error_gen",
        description="Rule-based synthetic error generator.",
    )
    ap.add_argument("--demo", type=int, metavar="N", help="print N clean->noisy examples")
    ap.add_argument("--samples", type=int, metavar="N", help="write N samples to a TSV")
    ap.add_argument("--out", type=str, default=None, help="output TSV path for --samples")
    ap.add_argument("--list-ops", action="store_true", help="print the operation table")
    ap.add_argument("--p", type=float, default=0.2, help="noise ratio (default 0.2)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    if args.list_ops:
        list_ops()
    if args.demo:
        demo(args.demo, p=args.p, seed=args.seed)
    if args.samples:
        samples(args.samples, p=args.p, seed=args.seed, out=args.out)
    if not any([args.list_ops, args.demo, args.samples]):
        ap.error("pick one of --demo N / --samples N / --list-ops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
