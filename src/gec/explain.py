"""Map ERRANT categories to short plain-English reasons (spec section 12).

Pure data plus a lookup: no I/O, no project imports, so the demo can import it
without pulling in the modelling stack.

Lookup order: the exact category (``VERB:SVA``), then the head before the colon
(``VERB``), then a generic fallback. Spec section 12 names ten categories
verbatim; the rest are added so the demo never shows a user a bare type code
like ``R:NOUN:POSS``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: ERRANT's operation prefixes (spec section 12).
PREFIX = {
    "M": "missing word",
    "U": "unnecessary word",
    "R": "replaced word",
}

#: category -> (short label, full reason). The first ten are spec section 12
#: verbatim; the remainder keep the demo from ever showing a raw type code.
REASONS: dict[str, tuple[str, str]] = {
    # --- spec section 12, verbatim ---
    "DET": ("Article", "Article use: a/an/the is missing, extra or wrong."),
    "PREP": ("Preposition", "Preposition: a different preposition fits here."),
    "VERB:SVA": ("Subject-verb agreement",
                 "Subject-verb agreement: the verb must match its subject."),
    "VERB:TENSE": ("Verb tense",
                   "Verb tense: the time of the action needs a different tense."),
    "VERB:FORM": ("Verb form",
                  "Verb form: after this word the verb takes a different form."),
    "NOUN:NUM": ("Noun number", "Noun number: singular/plural doesn't fit."),
    "SPELL": ("Spelling", "Spelling."),
    "PUNCT": ("Punctuation", "Punctuation."),
    "ORTH": ("Capitalisation", "Capitalisation or spacing."),
    "WO": ("Word order", "Word order."),
    # --- additions, so no raw code ever reaches the user ---
    "CONTR": ("Contraction", "Contraction: the shortened form is wrong here."),
    "MORPH": ("Word form",
              "Word form: the right word, but the wrong part of speech."),
    "NOUN": ("Noun choice", "Noun choice: a different noun fits here."),
    "NOUN:POSS": ("Possessive", "Possessive: the apostrophe form is wrong."),
    "NOUN:INFL": ("Noun inflection",
                  "Noun inflection: this noun is formed incorrectly."),
    "VERB": ("Verb choice", "Verb choice: a different verb fits here."),
    "VERB:INFL": ("Verb inflection",
                  "Verb inflection: this verb is formed incorrectly."),
    "ADJ": ("Adjective", "Adjective choice: a different adjective fits here."),
    "ADJ:FORM": ("Adjective form",
                 "Adjective form: comparative or superlative is wrong."),
    "ADV": ("Adverb", "Adverb choice: a different adverb fits here."),
    "PRON": ("Pronoun", "Pronoun: this pronoun does not fit its reference."),
    "CONJ": ("Conjunction", "Conjunction: a different linking word fits here."),
    "PART": ("Particle", "Particle: the small word after this verb is wrong."),
    "OTHER": ("Other", "Other change."),
    "UNK": ("Unclear", "Something is wrong here, but the type is unclear."),
    "noop": ("No change", "No change needed."),
}

FALLBACK = ("Other", "Other change.")


@dataclass(frozen=True)
class Explanation:
    """A human-readable account of one ERRANT edit."""

    category: str      # "VERB:SVA"
    label: str         # "Subject-verb agreement"
    reason: str        # "Subject-verb agreement: the verb must match its subject."
    operation: str     # "replaced word" | "missing word" | "unnecessary word" | "change"

    def one_line(self, before: str = "", after: str = "") -> str:
        """``go -> went - Verb tense - the time of the action needs ...``"""
        head = f"{before or '(nothing)'} -> {after or '(nothing)'}"
        return f"{head} - {self.label} - {self.reason}"


def split_type(errant_type: str) -> tuple[str | None, str]:
    """Split ``R:VERB:SVA`` into ``("R", "VERB:SVA")``.

    Mirrors ERRANT's own ``cat[2:]`` convention, so an unprefixed type such as
    ``UNK`` passes through untouched.
    """
    t = (errant_type or "").strip()
    if len(t) > 2 and t[1] == ":" and t[0] in PREFIX:
        return t[0], t[2:]
    return None, t


def explain(errant_type: str) -> Explanation:
    """Explain an ERRANT type, with or without its ``M:``/``U:``/``R:`` prefix."""
    prefix, cat = split_type(errant_type)
    if cat in REASONS:
        label, reason = REASONS[cat]
    else:
        head = cat.split(":")[0]
        label, reason = REASONS.get(head, FALLBACK)
    return Explanation(
        category=cat,
        label=label,
        reason=reason,
        operation=PREFIX.get(prefix or "", "change"),
    )


def all_types() -> list[str]:
    """Every category this module explains without falling back."""
    return sorted(REASONS)
