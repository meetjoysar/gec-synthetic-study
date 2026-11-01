"""spaCy tokenisation, POS tagging, detokenisation and ASCII normalisation.

Spec section 5.3: everything in this project - model inputs, targets,
predictions and M2 files - is spaCy-tokenised and space-separated (``do n't``,
``it 's``, ``school .``), matching the M2 files and what ERRANT expects.

Three decisions worth knowing about:

* :func:`tokenise` uses ``spacy.blank("en")``, not ``en_core_web_sm``. They
  share the identical tokenizer (English defaults plus exceptions), so the
  output matches ERRANT's, but ``blank`` loads in milliseconds rather than
  seconds. ``tests/test_tok.py`` asserts the two agree on real W&I sentences.

* :func:`tagger` loads the real model only when POS tags are needed (building
  the clean pool), and excludes ``ner`` and ``parser``: the tagger,
  attribute_ruler and lemmatizer are all we need, and dropping the parser
  roughly doubles throughput over 60k sentences.

* :func:`normalise` is the T5 ``<unk>`` guard - see its docstring. It is the
  reason we can trust that a prediction differs from its source only where the
  model actually changed something.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

#: Substitutions made by :func:`normalise`, for the PROGRESS.md record.
NORM_STATS: Counter[str] = Counter()

_SPACY_ERR = (
    "spaCy model 'en_core_web_sm' is not installed. Run:\n"
    "    python -m spacy download en_core_web_sm\n"
    "errant 3.x does NOT install it (despite what PROJECT_SPEC.md section 3 says)."
)


@dataclass
class Tagged:
    """Parallel per-token annotations for one sentence.

    Both ``upos`` and ``tag`` are kept because they serve different jobs:
    ``upos`` drives the generator's applicability checks (is this a noun? a
    verb?), while ``tag`` is the PTB tag that ``lemminflect.getInflection``
    requires. ``lemma`` is cached too, which means :mod:`gec.error_gen` needs no
    spaCy at generation time at all - only lemminflect dictionary lookups. That
    is what makes regenerating a 30k-pair synthetic set take seconds instead of
    minutes, and it matters because E1/E2/E4 regenerate it eight or more times.
    """

    toks: list[str]
    upos: list[str]
    tag: list[str]
    lemma: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.toks)

    def __len__(self) -> int:
        return len(self.toks)


@functools.lru_cache(maxsize=1)
def tokeniser():
    """A blank English pipeline - tokenizer only, loads in milliseconds."""
    import spacy

    return spacy.blank("en")


@functools.lru_cache(maxsize=1)
def tagger():
    """``en_core_web_sm`` with ``ner`` and ``parser`` excluded, plus a sentencizer."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])
    except OSError as exc:  # pragma: no cover - environment problem
        raise OSError(_SPACY_ERR) from exc
    if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

#: One character in, one character out. Never changes token boundaries.
_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"', "´": "'", "`": "'",
    "–": "-", "—": "-", "‒": "-", "―": "-",
    "−": "-", "­": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", "　": " ",
    "​": "", "﻿": "",
    "•": "*", "·": "*", "∙": "*",
    "«": '"', "»": '"', "‹": "'", "›": "'",
}

#: Characters T5's SentencePiece vocabulary genuinely has.
_ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " !\"#$%&'()*+,-./:;=?@[]_"
)


def normalise(text: str) -> str:
    """Fold *text* to characters T5 can represent, preserving token count.

    Why this is mandatory rather than cosmetic: T5's SentencePiece vocabulary
    lacks ``{ } < > ^ ~ \\`` and every non-ASCII character, mapping them to
    ``<unk>``. Because ``<unk>`` is a special token, decoding with
    ``skip_special_tokens=True`` **deletes** it - so an unnormalised source
    silently loses characters, and the resulting prediction differs from its
    source in places the model never touched, producing spurious ERRANT edits.

    The mapping is strictly one character to one character (or to nothing for
    zero-width characters), so ``len(text.split())`` is unchanged and every
    token boundary survives. Applying it to source *and* target is provably
    safe for scoring, because ERRANT's ``simplify_edits`` reads only the ``A``
    lines of an M2 block and ignores the ``S`` line text entirely.

    Unmapped characters are NFKD-folded (``é`` -> ``e``); anything still outside
    :data:`_ALLOWED` becomes ``?``. Every substitution is tallied in
    :data:`NORM_STATS` for the PROGRESS.md record.
    """
    out: list[str] = []
    for ch in text:
        if ch in _ALLOWED or ch == "\n":
            out.append(ch)
            continue
        if ch in _CHAR_MAP:
            repl = _CHAR_MAP[ch]
            NORM_STATS[f"{ch}->{repl}"] += 1
            out.append(repl)
            continue
        # NFKD fold: strip combining marks, keep the base letter if usable.
        folded = "".join(
            c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c)
        )
        if folded and all(c in _ALLOWED for c in folded):
            # Keep length stable: a 1:many fold would shift nothing (no spaces
            # introduced), but prefer the first character when it expands.
            repl = folded if len(folded) == 1 else folded[0]
            NORM_STATS[f"{ch}->{repl}"] += 1
            out.append(repl)
        else:
            NORM_STATS[f"{ch}->?"] += 1
            out.append("?")
    return "".join(out)


def norm_report() -> dict[str, int]:
    """Substitution tallies so far, most frequent first."""
    return dict(NORM_STATS.most_common())


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

def tokenise(text: str) -> str:
    """Tokenise *text* into a space-separated string.

    Whitespace-only tokens are dropped, so the result never contains a double
    space and never ends up blank for non-blank input.
    """
    doc = tokeniser()(text)
    return " ".join(t.text for t in doc if t.text.strip())


def tokenise_batch(texts: Iterable[str], batch_size: int = 256) -> Iterator[str]:
    """Tokenise many strings, using spaCy's ``pipe`` for throughput."""
    for doc in tokeniser().pipe(texts, batch_size=batch_size):
        yield " ".join(t.text for t in doc if t.text.strip())


def tag_batch(texts: Iterable[str], batch_size: int = 256) -> Iterator[Tagged]:
    """Tokenise and POS-tag many strings, yielding :class:`Tagged` per input."""
    for doc in tagger().pipe(texts, batch_size=batch_size):
        toks, upos, tag, lemma = [], [], [], []
        for t in doc:
            if not t.text.strip():
                continue
            toks.append(t.text)
            upos.append(t.pos_)
            tag.append(t.tag_)
            lemma.append(t.lemma_)
        yield Tagged(toks=toks, upos=upos, tag=tag, lemma=lemma)


def tag_one(text: str) -> Tagged:
    """Tokenise and tag a single string."""
    return next(tag_batch([text]))


def sentencize(paragraph: str) -> list[str]:
    """Split a paragraph into sentences (spaCy's rule-based sentencizer)."""
    doc = tagger()(paragraph)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


# --------------------------------------------------------------------------- #
# Detokenisation (display only)
# --------------------------------------------------------------------------- #

_NO_SPACE_BEFORE = {
    ".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "'s", "n't", "'re",
    "'ve", "'ll", "'d", "'m", "na", "'", "...",
}
_NO_SPACE_AFTER = {"(", "[", "{", "$", "#", "``", "`"}
_CONTRACTIONS = {"n't", "'s", "'re", "'ve", "'ll", "'d", "'m", "na", "'S", "'t"}


def detokenise(tokens: str | list[str]) -> str:
    """Rejoin tokens into readable text. **Display only** (spec section 5.3).

    Never used for scoring, model input or M2 files - only for the Gradio demo
    and for sacreBLEU's ``13a`` variant. Simple rules are sufficient: no space
    before closing punctuation, no space after opening brackets, contractions
    rejoined, and paired quotes alternated.
    """
    toks = tokens.split() if isinstance(tokens, str) else list(tokens)
    if not toks:
        return ""

    out = ""
    open_double = True
    for i, t in enumerate(toks):
        if i == 0:
            out = t
            if t == '"':
                open_double = False
            continue

        prev = toks[i - 1]
        if t == '"':
            # Alternate: opening quote hugs what follows, closing hugs what precedes.
            if open_double:
                out += " " + t
                open_double = False
            else:
                out += t
                open_double = True
            continue

        if t in _CONTRACTIONS or t in _NO_SPACE_BEFORE or prev in _NO_SPACE_AFTER:
            out += t
        else:
            out += " " + t

    # A closing quote directly after an opening one would have hugged wrongly.
    out = re.sub(r'\s+([.,!?;:])', r"\1", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()
