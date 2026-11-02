"""Download and prepare all data (spec section 5).

::

    python -m gec.prepare_data --all
    python -m gec.prepare_data --wi
    python -m gec.prepare_data --wikitext [--pool-size 60000] [--no-streaming]
    python -m gec.prepare_data --indian
    python -m gec.prepare_data --inspect

Three things this module is responsible for getting right:

**The alignment contract.** Every split writes five files in *identical*
sentence order (see :func:`gec.paths.split_files`). Four invariants hold and are
asserted: equal counts across all five files; ``src.txt`` line *i* equals M2
block *i*'s ``S`` line; no blank lines anywhere; identical ordering. ERRANT's
``compare_m2`` asserts hypothesis and reference lengths match *without a
message*, so a drift of one sentence would otherwise surface as a bare
``AssertionError`` minutes into scoring.

**The ``val`` reference M2.** Spec section 5.1 says to hold out 2,000 sentences
from the training M2, but not how to produce a *reference* for them. We slice
:class:`gec.m2.M2Block` objects and write them back via their preserved ``raw``
text, so ``val.m2`` is a verbatim subset of ``ABC.train.gold.bea19.m2`` carrying
the **original** gold annotations. Re-deriving edits by running ERRANT over
(src, tgt) would differ in merge and classification decisions, making ``val``
scores incomparable with ``test`` scores - and E1/E2 selection depends on
exactly that comparability.

**Test-set hygiene.** ``ABCN.dev.gold.bea19.m2`` becomes ``test`` as a byte copy
in its original order and is never shuffled, never trained on, and never used to
generate synthetic data (spec rule 2.2).
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from gec import m2, paths, tok

WI_URL = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"
WI_ARCHIVE = "wi+locness_v2.1.bea19.tar.gz"

#: Spec section 5.1. The archive is ~6.1 MB, far under the 500 MB consent
#: threshold in spec rule 2.4.
WI_EXPECTED_MB = 6.1

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"

#: Spec section 5.4 clean-pool filters.
POOL_MIN_TOKENS = 8
POOL_MAX_TOKENS = 40
POOL_SIZE = 60_000
#: The first 50,000 pool sentences are the base set every synthetic condition
#: corrupts, so conditions differ only in *how* sentences are corrupted. The
#: remaining 10,000 are held back for the E4 probe, which therefore never shares
#: sentences with any training set.
POOL_BASE_SET = 50_000

_ALLOWED_POOL_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " !\"#$%&'()*+,-./:;=?@[]_"
)


# --------------------------------------------------------------------------- #
# W&I + LOCNESS
# --------------------------------------------------------------------------- #

def download_wi(force: bool = False) -> Path:
    """Download the W&I+LOCNESS v2.1 archive to ``data/raw/``."""
    paths.ensure_dirs()
    dest = paths.RAW / WI_ARCHIVE
    if dest.exists() and not force:
        print(f"[wi] already downloaded: {paths.rel(dest)} "
              f"({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"[wi] downloading {WI_URL}")
    print(f"[wi] expected size ~{WI_EXPECTED_MB} MB (well under the 500 MB consent rule)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(WI_URL, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)
    print(f"[wi] saved {paths.rel(dest)} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def extract_wi(archive: Path, force: bool = False) -> Path:
    """Extract the archive and return the ``wi+locness`` root directory."""
    out = paths.RAW / "wi+locness"
    if out.exists() and not force and any(out.rglob("*.m2")):
        print(f"[wi] already extracted: {paths.rel(out)}")
        return out
    print(f"[wi] extracting {paths.rel(archive)}")
    with tarfile.open(archive, "r:gz") as tf:
        # filter="data" (Python 3.12+) is safer, but 3.10 lacks it; the archive
        # is a known, fixed artefact from a university host.
        tf.extractall(paths.RAW)
    if not out.exists():
        candidates = [d for d in paths.RAW.iterdir() if d.is_dir() and any(d.rglob("*.m2"))]
        if len(candidates) == 1:
            out = candidates[0]
        else:
            raise FileNotFoundError(
                f"could not find the extracted W&I root under {paths.RAW}. "
                f"Directories present: {[d.name for d in paths.RAW.iterdir() if d.is_dir()]}"
            )
    print(f"[wi] extracted to {paths.rel(out)}")
    return out


def show_licence(root: Path) -> str:
    """Print and return the licence text shipped in the archive (spec 5.1)."""
    names = ["LICENSE.txt", "LICENCE.txt", "LICENSE", "LICENCE", "readme.txt", "README.txt"]
    found: list[Path] = []
    for n in names:
        found.extend(root.rglob(n))
    if not found:
        print("[wi] WARNING: no licence file found in the archive.")
        return ""
    chunks = []
    for p in sorted(set(found)):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        header = f"----- {paths.rel(p)} -----"
        print(f"\n{header}\n{text.strip()}\n")
        chunks.append(f"{header}\n{text.strip()}")
    return "\n\n".join(chunks)


def find_m2_files(root: Path) -> dict[str, Path]:
    """Locate the train and dev gold M2 files.

    Spec section 5.1 says to inspect the folder and adapt if names differ, so we
    match by pattern rather than hardcoding. On 0 or >1 candidates for either
    role we print every M2 file we found with its size and sentence count and
    fail clearly, rather than guessing and silently training on the wrong file.
    """
    all_m2 = sorted(root.rglob("*.m2"))
    if not all_m2:
        raise FileNotFoundError(f"no .m2 files found under {root}")

    def pick(role: str, exact: str, pattern: re.Pattern[str]) -> Path:
        # Prefer the exact filename the spec names. The archive also ships
        # per-CEFR-level splits (A./B./C./N.) whose names match any loose
        # "train.*gold" pattern, so pattern matching alone is ambiguous: we want
        # the COMBINED file, not one proficiency level.
        for p in all_m2:
            if p.name == exact:
                return p
        hits = [p for p in all_m2 if pattern.search(p.name.lower())]
        if len(hits) == 1:
            return hits[0]
        listing = "\n".join(
            f"    {paths.rel(p)}  ({p.stat().st_size / 1e6:.2f} MB, "
            f"{m2.block_count(p)} sentences)"
            for p in all_m2
        )
        raise FileNotFoundError(
            f"expected '{exact}' or exactly 1 '{role}' match, found {len(hits)}: "
            f"{[p.name for p in hits]}\nAll .m2 files under {paths.rel(root)}:\n{listing}\n"
            "Adapt find_m2_files() in prepare_data.py if the archive layout changed."
        )

    train = pick("train", "ABC.train.gold.bea19.m2", re.compile(r"^abc\.train.*gold"))
    test = pick("dev/test", "ABCN.dev.gold.bea19.m2", re.compile(r"^abcn\.dev.*gold"))
    print(f"[wi] train M2: {paths.rel(train)}")
    print(f"[wi] test  M2: {paths.rel(test)}  (NEVER trained or tuned on)")
    return {"train": train, "test": test}


#: How many ``S`` lines and correction fields :func:`_normalise_block` changed.
NORMALISED_BLOCKS: Counter[str] = Counter()


def _normalise_a_line(line: str) -> str:
    """ASCII-fold only the correction field of an ``A`` line.

    **Never pass a whole ``A`` line to :func:`gec.tok.normalise`.** The ``|``
    character is not in T5's vocabulary and therefore not in the allowed set, so
    folding the whole line turns the ``|||`` field separators into ``???`` and
    destroys the file. Only field 2 (the correction) is text; every other field
    is structure and is copied byte-for-byte.

    An empty correction (a deletion) stays empty: ERRANT writes deletions with
    an empty correction field, and rewriting it as ``-NONE-`` would make the
    reference edit tuple stop matching the hypothesis's, silently inflating
    false negatives.
    """
    parts = line.split("|||")
    if len(parts) < 3 or not parts[2] or parts[2] == "-NONE-":
        return line
    folded = tok.normalise(parts[2])
    if folded == parts[2]:
        return line
    NORMALISED_BLOCKS["corrections"] += 1
    parts[2] = folded
    return "|||".join(parts)


def _normalise_block(block: m2.M2Block) -> m2.M2Block:
    """ASCII-fold a block's ``S`` line and its corrections, keeping structure.

    Necessary because T5's SentencePiece vocabulary maps curly quotes, dashes
    and accents to ``<unk>``, and ``skip_special_tokens=True`` *deletes*
    ``<unk>`` on decode - so an unnormalised source silently loses characters
    and the prediction then differs from its source in places the model never
    touched, producing spurious ERRANT edits.

    :func:`gec.tok.normalise` maps one character to one character, so token
    counts and every edit span are unchanged. Corrections are folded as well as
    sources: the model's input is normalised, so it can only ever emit ASCII,
    and leaving a curly quote in a *reference* correction would create an
    artificial false negative where the model actually got the edit right.
    Logged as a deviation in PROGRESS.md.
    """
    src = tok.normalise(block.src)
    raw_lines = block.raw.split("\n") if block.raw else []

    if src != block.src:
        NORMALISED_BLOCKS["sources"] += 1
        if raw_lines and raw_lines[0].startswith("S "):
            raw_lines[0] = f"S {src}"

    fixed_edits = []
    for e in block.edits:
        cor = tok.normalise(e.cor) if e.cor and e.cor != "-NONE-" else e.cor
        fixed_edits.append(
            e if cor == e.cor else m2.Edit(e.start, e.end, e.type, cor, e.annotator)
        )
    raw_lines = [_normalise_a_line(ln) if ln.startswith("A ") else ln for ln in raw_lines]

    return m2.M2Block(src=src, edits=fixed_edits, raw="\n".join(raw_lines))


def write_split(blocks: list[m2.M2Block], name: str, source_file: str = "") -> dict:
    """Write the five files for one split and return its stats.

    All five are written from the *same* block list in the *same* order, which
    is what makes the alignment contract hold by construction rather than by
    care.
    """
    files = paths.split_files(name)
    n = m2.write_m2(blocks, files["m2"])

    srcs = [b.src for b in blocks]
    tgts = [b.target(0) for b in blocks]
    paths.write_lines(files["src"], srcs)
    # A handful of W&I annotations delete the whole sentence (4 of 34,308 in
    # train, 2 of 4,384 in dev - junk inputs like "fsdjgdofg", or a fragment
    # merged into its neighbour). The M2, src and jsonl stay complete so test
    # integrity and ERRANT alignment are untouched; tgt.txt keeps the blank so
    # it remains line-parallel. build_datasets drops these from TRAINING data
    # and evaluate excludes them from BLEU, both with a recorded count.
    n_empty_tgt = sum(1 for t in tgts if not t.strip())
    paths.write_lines(files["tgt"], tgts, allow_blank=True)

    rows = []
    len_hist: Counter[int] = Counter()
    n_clean = 0
    total_edits = 0
    for i, b in enumerate(blocks):
        refs = [t for t in dict.fromkeys(b.targets().values()) if t.strip()]
        clean = b.is_clean(0)
        n_clean += clean
        total_edits += b.n_edits(0)
        len_hist[len(b.tokens)] += 1
        rows.append(
            {
                "id": f"{name}-{i:05d}",
                "src": b.src,
                "tgt": b.target(0),
                "refs": refs,
                "is_clean": clean,
                "n_gold_edits": b.n_edits(0),
                "n_annotators": len(b.annotators()),
                "src_len": len(b.tokens),
            }
        )
    paths.write_jsonl(files["jsonl"], rows)

    # Invariants: fail here, not inside ERRANT's messageless assert.
    m2.assert_aligned(files["m2"], files["src"])
    m2.assert_same_length(files["src"], files["tgt"])
    assert len(rows) == n, f"{name}: {len(rows)} jsonl rows vs {n} M2 blocks"

    stats = {
        "split": name,
        "n": n,
        "source_file": source_file,
        "n_empty_targets": n_empty_tgt,
        "n_clean": n_clean,
        "clean_share": round(n_clean / n, 4) if n else None,
        "n_gold_edits_total": total_edits,
        "edits_per_sentence": round(total_edits / n, 4) if n else None,
        "mean_src_len": round(sum(k * v for k, v in len_hist.items()) / n, 2) if n else None,
        "max_src_len": max(len_hist) if len_hist else 0,
        "multi_annotator_sentences": sum(1 for r in rows if r["n_annotators"] > 1),
        "m2_sha256": paths.sha256(files["m2"]),
        "files": {k: paths.rel(v) for k, v in files.items()},
    }
    paths.write_json(files["stats"], stats)
    print(
        f"[split] {name:11s} n={n:6d}  clean={n_clean:5d} ({stats['clean_share']:.1%})  "
        f"edits/sent={stats['edits_per_sentence']:.2f}  mean_len={stats['mean_src_len']:.1f}"
    )
    return stats


def split_wi(
    train_m2: Path,
    test_m2: Path,
    val_size: int = 2000,
    seed: int = 42,
) -> dict[str, dict]:
    """Build ``train_real``/``val`` from the train M2 and ``test`` from the dev M2.

    Spec section 5.1: shuffle with seed 42 and hold out 2,000 sentences as
    ``val``; the rest is ``train_real``. ``test`` keeps the gold file's original
    order so scores stay comparable with published numbers.
    """
    print(f"\n[split] reading {paths.rel(train_m2)}")
    blocks = [_normalise_block(b) for b in m2.read_m2(train_m2)]
    print(f"[split] {len(blocks)} blocks in the training M2")
    if len(blocks) <= val_size:
        raise ValueError(f"only {len(blocks)} blocks; cannot hold out {val_size} for val")

    idx = list(range(len(blocks)))
    random.Random(seed).shuffle(idx)
    val_idx, train_idx = idx[:val_size], idx[val_size:]

    stats = {
        "val": write_split([blocks[i] for i in val_idx], "val", paths.rel(train_m2)),
        "train_real": write_split(
            [blocks[i] for i in train_idx], "train_real", paths.rel(train_m2)
        ),
    }

    # Auditable proof that val and train_real are disjoint.
    paths.write_json(
        paths.PROCESSED / "split_index.json",
        {
            "seed": seed,
            "val_size": val_size,
            "source_file": paths.rel(train_m2),
            "source_sha256": paths.sha256(train_m2),
            "n_blocks": len(blocks),
            "val_idx": val_idx,
            "train_idx": train_idx,
            "disjoint": not (set(val_idx) & set(train_idx)),
        },
    )

    print(f"\n[split] reading {paths.rel(test_m2)} (test - never trained or tuned on)")
    test_blocks = [_normalise_block(b) for b in m2.read_m2(test_m2)]
    stats["test"] = write_split(test_blocks, "test", paths.rel(test_m2))
    return stats


def prepare_wi(force: bool = False, val_size: int = 2000, seed: int = 42) -> dict:
    """Full W&I pipeline: download, extract, show the licence, split."""
    archive = download_wi(force=force)
    root = extract_wi(archive, force=force)
    print("\n" + "=" * 70)
    print("W&I+LOCNESS v2.1 LICENCE (spec 5.1 requires showing this)")
    print("=" * 70)
    licence = show_licence(root)
    paths.write_text(paths.RAW / "wi_licence_shown.txt", licence)
    files = find_m2_files(root)
    return split_wi(files["train"], files["test"], val_size=val_size, seed=seed)


# --------------------------------------------------------------------------- #
# WikiText-103 clean pool
# --------------------------------------------------------------------------- #

def _iter_wikitext_lines(streaming: bool = True) -> Iterator[str]:
    """Yield raw WikiText-103 training lines.

    Two backends behind one interface so the filter code is shared:

    ``streaming=True`` (default, user-confirmed)
        ``load_dataset(..., streaming=True)``; the caller stops as soon as the
        pool is full, so only the first parquet shard or two is fetched
        (~60-90 MB rather than the ~250 MB full split). We need 60k of roughly
        4M sentences, so this costs nothing scientifically and keeps us well
        under the spec rule 2.4 download threshold.

    ``streaming=False``
        Spec section 5.4 read literally: the whole split is downloaded.
    """
    from datasets import load_dataset

    if streaming:
        print(f"[pool] streaming {WIKITEXT_REPO}/{WIKITEXT_CONFIG} (stops early)")
        try:
            ds = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split="train", streaming=True)
            for row in ds:
                yield row["text"]
            return
        except Exception as exc:  # noqa: BLE001 - fall back on any streaming failure
            print(f"[pool] streaming failed ({type(exc).__name__}: {exc});"
                  " falling back to a single parquet shard")
            yield from _iter_wikitext_parquet()
            return

    print(f"[pool] downloading the FULL {WIKITEXT_CONFIG} train split (~250 MB)")
    ds = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split="train")
    for row in ds:
        yield row["text"]


def _iter_wikitext_parquet() -> Iterator[str]:
    """Fallback: fetch one parquet shard directly and read it with pandas."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    for shard in ("train-00000-of-00002.parquet", "train-00000-of-00001.parquet"):
        try:
            p = hf_hub_download(
                repo_id=WIKITEXT_REPO,
                filename=f"{WIKITEXT_CONFIG}/{shard}",
                repo_type="dataset",
            )
        except Exception:  # noqa: BLE001
            continue
        print(f"[pool] reading shard {shard}")
        for text in pd.read_parquet(p)["text"]:
            yield text
        return
    raise RuntimeError(
        "could not fetch a WikiText parquet shard; check the network or pass --no-streaming"
    )


def _clean_wikitext_line(line: str) -> str | None:
    """Apply spec section 5.4's text fixes; return None to drop the line."""
    s = line.strip()
    if not s or s.startswith("="):  # headings and blanks
        return None
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    return s or None


def _pool_sentence_ok(t: tok.Tagged) -> str | None:
    """Return a reject reason, or None if the sentence belongs in the pool.

    **The charset check must run on the ORIGINAL text, before ASCII folding.**
    :func:`gec.tok.normalise` maps anything unrepresentable to ``?``, which is
    itself an allowed character - so folding first makes this filter a no-op and
    lets sentences that were mostly Greek, Cyrillic or CJK through as
    ``? ? ? ? ?`` garbage. (Measured: 247 of 60,000 on the first build.) Those
    would be worthless training targets, so the caller tags and filters the raw
    text and only then folds token by token.
    """
    n = len(t.toks)
    if n < POOL_MIN_TOKENS or n > POOL_MAX_TOKENS:
        return "length"
    # Spec 5.4 says "at least one verb". spaCy tags is/has/can as AUX, not VERB,
    # so a VERB-only check would throw away most copula sentences - exactly the
    # ones the `sva` operation needs.
    if not any(p in ("VERB", "AUX") for p in t.upos):
        return "no_verb"
    text = " ".join(t.toks)
    if not text:
        return "charset"
    # Reject if ASCII folding would DESTROY information rather than preserve it.
    # `normalise` maps an unrepresentable character to "?", so a new "?" means
    # T5 genuinely cannot encode that character. Accented Latin folds cleanly
    # (cafe, Beyonce) and is kept - those are perfectly good English sentences.
    # A character-ratio threshold is the wrong test: a long English sentence
    # with one embedded Greek or CJK word clears 95% easily and still leaves a
    # "??????????" token in the training data.
    if tok.normalise(text).count("?") > text.count("?"):
        return "charset"
    return None


def _fold_tagged(t: tok.Tagged) -> tok.Tagged:
    """ASCII-fold a tagged sentence token by token, keeping tags aligned.

    Safe because :func:`gec.tok.normalise` is one character in, one character
    out and never touches whitespace, so token boundaries cannot move and the
    POS/lemma lists stay parallel.
    """
    return tok.Tagged(
        toks=[tok.normalise(w) for w in t.toks],
        upos=list(t.upos),
        tag=list(t.tag),
        lemma=[tok.normalise(w) for w in t.lemma],
    )


def build_clean_pool(
    n: int = POOL_SIZE,
    seed: int = 42,
    streaming: bool = True,
    force: bool = False,
) -> Path:
    """Build the clean sentence pool for synthetic data (spec section 5.4).

    Writes ``clean_pool.txt`` (one tokenised sentence per line, as the spec
    requires) plus ``clean_pool.jsonl``, which caches ``toks``/``upos``/``tag``/
    ``lemma`` per sentence. That cache is why :mod:`gec.error_gen` needs no spaCy
    at generation time, which turns regenerating a 30k-pair set from minutes
    into seconds - and E1/E2/E4 regenerate it eight or more times.
    """
    paths.ensure_dirs()
    out_txt = paths.PROCESSED / "clean_pool.txt"
    out_jsonl = paths.PROCESSED / "clean_pool.jsonl"
    out_stats = paths.PROCESSED / "clean_pool.stats.json"
    if out_txt.exists() and out_jsonl.exists() and not force:
        existing = paths.count_lines(out_txt)
        if existing >= n:
            print(f"[pool] already built: {paths.rel(out_txt)} ({existing} sentences)")
            return out_txt
        print(f"[pool] rebuilding: have {existing}, need {n}")

    # Oversample before shuffling so the pool is not just "the first N
    # sentences of Wikipedia", which would skew to whatever the first shard is.
    target_raw = int(n * 1.6)
    rejects: Counter[str] = Counter()
    seen: set[str] = set()
    kept: list[tok.Tagged] = []
    n_lines = 0
    n_sentences = 0

    print(f"[pool] target {n} sentences ({POOL_MIN_TOKENS}-{POOL_MAX_TOKENS} tokens, "
          ">=1 verb/aux, mostly-ASCII, deduplicated)")

    def sentence_stream() -> Iterator[str]:
        """Cleaned but NOT yet ASCII-folded lines - see _pool_sentence_ok."""
        nonlocal n_lines
        for raw in _iter_wikitext_lines(streaming=streaming):
            cleaned = _clean_wikitext_line(raw)
            if cleaned is None:
                continue
            n_lines += 1
            yield cleaned

    # Stream through the tagger pipeline directly, so sentence boundaries and
    # POS tags both come from a single spaCy pass over the text.
    nlp = tok.tagger()
    for doc in nlp.pipe(sentence_stream(), batch_size=64):
        for sent in doc.sents:
            n_sentences += 1
            keep = [x for x in sent if x.text.strip()]
            if not keep:
                continue
            raw_tagged = tok.Tagged(
                toks=[x.text for x in keep],
                upos=[x.pos_ for x in keep],
                tag=[x.tag_ for x in keep],
                lemma=[x.lemma_ for x in keep],
            )
            # Filter on the RAW text, then fold. Folding first would turn
            # non-Latin script into "?" and silently pass the charset filter.
            reason = _pool_sentence_ok(raw_tagged)
            if reason:
                rejects[reason] += 1
                continue
            t = _fold_tagged(raw_tagged)
            key = t.text.lower()
            if key in seen:
                rejects["duplicate"] += 1
                continue
            seen.add(key)
            kept.append(t)
            if len(kept) >= target_raw:
                break
        if len(kept) >= target_raw:
            break
        if n_sentences and n_sentences % 20000 == 0:
            print(f"[pool]   scanned {n_sentences} sentences, kept {len(kept)}")

    if len(kept) < n:
        print(f"[pool] WARNING: only {len(kept)} sentences passed the filters "
              f"(wanted {n}); the pool will be smaller than the spec's {POOL_SIZE}")
    random.Random(seed).shuffle(kept)
    kept = kept[:n]

    paths.write_lines(out_txt, [t.text for t in kept])
    paths.write_jsonl(
        out_jsonl,
        [
            {"i": i, "toks": t.toks, "upos": t.upos, "tag": t.tag, "lemma": t.lemma}
            for i, t in enumerate(kept)
        ],
    )

    import spacy

    stats = {
        "n": len(kept),
        "base_set_size": min(POOL_BASE_SET, len(kept)),
        "held_back_for_e4_probe": max(0, len(kept) - POOL_BASE_SET),
        "seed": seed,
        "streaming": streaming,
        "source": f"{WIKITEXT_REPO}/{WIKITEXT_CONFIG}:train",
        "raw_lines_scanned": n_lines,
        "sentences_scanned": n_sentences,
        "rejected": dict(rejects.most_common()),
        "filters": {
            "min_tokens": POOL_MIN_TOKENS,
            "max_tokens": POOL_MAX_TOKENS,
            "verb_pos": ["VERB", "AUX"],
            "min_allowed_char_share": 0.95,
        },
        "mean_tokens": round(sum(len(t.toks) for t in kept) / len(kept), 2) if kept else None,
        # A spaCy or model version change silently alters tokenisation and POS
        # tags; build_datasets warns loudly if these drift from the cache.
        "spacy_version": spacy.__version__,
        "en_core_web_sm_version": tok.tagger().meta.get("version", "unknown"),
        "pool_sha256": paths.sha256(out_txt),
        "normalise_substitutions": tok.norm_report(),
    }
    paths.write_json(out_stats, stats)
    print(
        f"[pool] wrote {len(kept)} sentences to {paths.rel(out_txt)}\n"
        f"[pool]   scanned {n_sentences} sentences from {n_lines} lines; "
        f"rejected {dict(rejects.most_common(4))}\n"
        f"[pool]   base set = first {stats['base_set_size']}, "
        f"{stats['held_back_for_e4_probe']} held back for the E4 probe"
    )
    return out_txt


def load_pool(limit: int | None = None) -> list[tok.Tagged]:
    """Load the cached clean pool as :class:`gec.tok.Tagged` objects."""
    p = paths.PROCESSED / "clean_pool.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"{paths.rel(p)} is missing. Run: python -m gec.prepare_data --wikitext"
        )
    out: list[tok.Tagged] = []
    for row in paths.read_jsonl(p):
        out.append(
            tok.Tagged(toks=row["toks"], upos=row["upos"], tag=row["tag"], lemma=row["lemma"])
        )
        if limit is not None and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# Indian English
# --------------------------------------------------------------------------- #

def prepare_indian(tsv: Path | None = None) -> dict:
    """Tokenise the Indian English TSV and build its two-annotator reference M2.

    Delegates validation to :mod:`gec.kappa`. Below 20 real rows (excluding the
    ``EXAMPLE-`` demonstration rows that Claude Code wrote), writes an
    ``indian.skip`` marker and exits cleanly, so E5 renders as ``TBD`` rather
    than crashing the pipeline (spec section 5.5).
    """
    from gec import errant_cli, kappa

    tsv = Path(tsv) if tsv else paths.DATA_MANUAL / "indian_english_test.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"{paths.rel(tsv)} is missing (the template ships with the repo)")

    # `real` here is READY rows only: both annotator columns filled. Rows still
    # awaiting annotation are excluded, because an empty annotation line would
    # make errant_parallel drop the block and desynchronise the M2 files.
    all_rows, real = kappa.load_indian_tsv(tsv)
    n_ready, n_pending = kappa.pending_count(tsv)
    problems = kappa.validate(all_rows)
    for p in problems:
        print(f"[indian] VALIDATION: {p}")

    skip_marker = paths.PROCESSED / "indian.skip"
    if len(real) < kappa.MIN_REAL_ROWS:
        msg = (
            f"only {n_ready} fully annotated rows (need >= {kappa.MIN_REAL_ROWS}); "
            f"{n_pending} more await annotation. E5 stays skipped until BOTH "
            "annotator columns are filled. EXAMPLE- rows are excluded by design "
            "(spec 2.3)."
        )
        print(f"[indian] WARNING: {msg}")
        paths.ensure_dirs()
        paths.write_json(skip_marker, {"skipped": True, "n_real_rows": n_ready,
                                       "n_pending_annotation": n_pending,
                                       "reason": msg})
        return {"skipped": True, "n_real_rows": n_ready,
                "n_pending_annotation": n_pending, "reason": msg}

    if skip_marker.exists():
        skip_marker.unlink()

    srcs = [tok.tokenise(tok.normalise(s)) for s in real["source"]]
    ann1 = [tok.tokenise(tok.normalise(s)) for s in real["annotator1"]]
    ann2 = [tok.tokenise(tok.normalise(s)) for s in real["annotator2"]]

    files = paths.split_files("indian")
    src_f = paths.PROCESSED / "indian.src.txt"
    a1_f = paths.PROCESSED / "indian.ann1.txt"
    a2_f = paths.PROCESSED / "indian.ann2.txt"
    paths.write_lines(src_f, srcs)
    paths.write_lines(a1_f, ann1)
    paths.write_lines(a2_f, ann2)

    # One errant_parallel call with two -cor files writes annotator ids 0 and 1
    # into a single M2 - only the CLI does that, which is why we shell out.
    errant_cli.parallel_to_m2(src_f, [a1_f, a2_f], files["m2"])

    rows = []
    for i, (s, c1, c2) in enumerate(zip(srcs, ann1, ann2)):
        refs = list(dict.fromkeys([c1, c2]))
        rows.append(
            {
                "id": str(real.iloc[i]["id"]),
                "src": s,
                "tgt": c1,
                "refs": refs,
                # For the Indian set, "clean" means BOTH annotators left it alone.
                "is_clean": (c1 == s and c2 == s),
                "n_gold_edits": None,
                "n_annotators": 2,
                "src_len": len(s.split()),
                "category": str(real.iloc[i].get("category", "")),
                # Provenance, so RESULTS.md can break E5 down by source_type.
                "source_type": str(real.iloc[i].get("source_type", "")),
            }
        )
    paths.write_jsonl(files["jsonl"], rows)
    paths.write_lines(files["src"], srcs)
    paths.write_lines(files["tgt"], ann1)

    m2.assert_aligned(files["m2"], files["src"])

    stats = {
        "split": "indian",
        "n": len(rows),
        "n_real_rows": n_ready,
        "n_pending_annotation": n_pending,
        "n_example_rows": len(all_rows) - (n_ready + n_pending),
        "n_clean_both": sum(r["is_clean"] for r in rows),
        "categories": dict(Counter(r["category"] for r in rows).most_common()),
        "source_file": paths.rel(tsv),
        "validation_problems": problems,
    }
    paths.write_json(files["stats"], stats)
    print(f"[indian] prepared {len(rows)} sentences from {paths.rel(tsv)}")
    return stats


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #

def inspect() -> dict:
    """Verify the alignment contract on every prepared split and print a report."""
    report: dict = {"splits": {}, "problems": []}
    for split in paths.SPLITS:
        files = paths.split_files(split)
        if not files["m2"].exists():
            report["splits"][split] = {"present": False}
            continue
        entry: dict = {"present": True}
        try:
            n = m2.assert_aligned(files["m2"], files["src"])
            m2.assert_same_length(files["src"], files["tgt"])
            rows = paths.read_jsonl(files["jsonl"])
            if len(rows) != n:
                raise AssertionError(f"{len(rows)} jsonl rows vs {n} M2 blocks")
            srcs = paths.read_lines(files["src"])
            blanks = [i for i, s in enumerate(srcs) if not s.strip()]
            if blanks:
                raise AssertionError(f"blank source lines at {blanks[:5]}")
            entry.update(
                {
                    "n": n,
                    "aligned": True,
                    "n_clean": sum(r["is_clean"] for r in rows),
                    "clean_share": round(sum(r["is_clean"] for r in rows) / n, 4),
                    "mean_src_len": round(sum(r["src_len"] for r in rows) / n, 2),
                }
            )
            # Over-correction needs a sane clean share; near 0 or near 1 means
            # the M2 parsing is wrong (spec 7 expects roughly 1/3 for W&I).
            if split in ("train_real", "val", "test") and not 0.1 < entry["clean_share"] < 0.6:
                report["problems"].append(
                    f"{split}: clean share {entry['clean_share']:.1%} is outside 10-60%; "
                    "check the M2 parsing"
                )
        except AssertionError as exc:
            entry.update({"aligned": False, "error": str(exc)})
            report["problems"].append(f"{split}: {exc}")
        report["splits"][split] = entry

    pool = paths.PROCESSED / "clean_pool.txt"
    report["clean_pool"] = (
        {"present": True, "n": paths.count_lines(pool)} if pool.exists() else {"present": False}
    )
    if m2.OVERLAP_WARNINGS:
        report["overlapping_edits"] = dict(m2.OVERLAP_WARNINGS)

    print("\n=== prepare_data --inspect ===")
    for split, entry in report["splits"].items():
        if not entry.get("present"):
            print(f"  {split:11s} MISSING")
        elif entry.get("aligned"):
            print(
                f"  {split:11s} n={entry['n']:6d}  aligned=OK  "
                f"clean={entry['clean_share']:.1%}  mean_len={entry['mean_src_len']:.1f}"
            )
        else:
            print(f"  {split:11s} MISALIGNED: {entry['error']}")
    print(f"  clean_pool  {report['clean_pool']}")
    if report["problems"]:
        print("\n  PROBLEMS:")
        for p in report["problems"]:
            print(f"    - {p}")
    else:
        print("\n  All alignment invariants hold.")
    paths.write_json(paths.PROCESSED / "inspect_report.json", report)
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gec.prepare_data",
        description="Download and prepare W&I+LOCNESS, the WikiText clean pool, "
        "and the Indian English set.",
    )
    ap.add_argument("--all", action="store_true", help="run --wi, --wikitext and --indian")
    ap.add_argument("--wi", action="store_true", help="download, split W&I+LOCNESS")
    ap.add_argument("--wikitext", action="store_true", help="build the clean sentence pool")
    ap.add_argument("--indian", action="store_true", help="prepare the Indian English set")
    ap.add_argument("--inspect", action="store_true", help="verify the alignment contract")
    ap.add_argument("--pool-size", type=int, default=POOL_SIZE)
    ap.add_argument("--val-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-streaming",
        action="store_true",
        help="download the FULL WikiText-103 split (~250 MB) instead of streaming",
    )
    ap.add_argument("--force", action="store_true", help="re-download and rebuild")
    args = ap.parse_args(argv)

    if not any([args.all, args.wi, args.wikitext, args.indian, args.inspect]):
        ap.error("pick at least one of --all/--wi/--wikitext/--indian/--inspect")

    paths.ensure_dirs()
    if args.all or args.wi:
        prepare_wi(force=args.force, val_size=args.val_size, seed=args.seed)
    if args.all or args.wikitext:
        build_clean_pool(
            n=args.pool_size,
            seed=args.seed,
            streaming=not args.no_streaming,
            force=args.force,
        )
    if args.all or args.indian:
        prepare_indian()
    if args.all or args.inspect:
        report = inspect()
        if report["problems"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
