# CREDITS

Datasets, tools, external code and AI-tool use for **"Fake Errors, Real
Grammar"** (NLP IA Activity 2).

---

## Datasets

### W&I+LOCNESS v2.1 (BEA-2019 shared task)

The real learner data: Cambridge English Write & Improve plus an annotated
subset of LOCNESS, released for the BEA-2019 shared task on Grammatical Error
Correction.

- Downloaded from
  `https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz`
  (~6.1 MB) by `python -m gec.prepare_data --wi`.
- **Licence:** the archive's own `readme.txt` governs use, which is
  **non-commercial research and education only**. `prepare_data.py` prints the
  licence text on every download and saves a copy to
  `data/raw/wi_licence_shown.txt`. The corpus is **not** redistributed in this
  repository — `data/` is gitignored, and the download is reproducible from the
  URL above.
- Files used: `m2/ABC.train.gold.bea19.m2` (34,308 sentences → our
  `train_real` 32,308 + `val` 2,000) and `m2/ABCN.dev.gold.bea19.m2`
  (4,384 sentences → our `test`).
- The official BEA-2019 **test** set has withheld answers and is not used.
- Papers: Bryant, Felice, Andersen & Briscoe (2019), *The BEA-2019 Shared Task
  on Grammatical Error Correction*, BEA Workshop; Yannakoudakis, Andersen,
  Geranpayeh, Briscoe & Nicholls (2018), *Developing an automated writing
  placement system for ESL learners*, Applied Measurement in Education;
  Granger (1998), *The computer learner corpus* (LOCNESS).

### WikiText-103 (raw)

The clean sentence pool that the error generator corrupts.

- `Salesforce/wikitext`, config `wikitext-103-raw-v1`, train split, loaded in
  **streaming** mode and stopped once 60,000 sentences pass the filters, so
  only the first shard or two is fetched.
- **Licence:** CC BY-SA 3.0.
- Paper: Merity, Xiong, Bradbury & Socher (2016), *Pointer Sentinel Mixture
  Models*, arXiv:1609.07843.
- Known limitation, recorded in the report: Wikipedia prose differs in style
  from learner essays, so corrupted Wikipedia text is not a perfect stand-in
  for genuine learner error.

### Indian English test set (ours)

`data_manual/indian_english_test.tsv`. 149 sentences, **manually collected from
public Indian subreddits** by the group and copied by hand; **usernames and
links are not stored**, and the raw collected file
(`data_manual/manual_sentences.txt`) is gitignored. Provenance is recorded per
row only as a writer id and post label (e.g. `W11 flat-hunting`) in the
`source_note` column.

Selection was done by humans: the group chose which sentences to include, and
both independent annotations are the group's own work. Claude Code supplied the
template, the importer, the validator and the agreement tooling, plus five rows
whose ids begin `EXAMPLE-` demonstrating the format; those are excluded from
every computation.

28 rows come from posts the group marked `[sensitive]` and carry
`notes = "sensitive: do not quote in report"`. Their text is used as evaluation
data but must not be quoted in the report or surfaced in the demo.

The source text originates from Reddit users writing in public. It is used here
as research data under academic fair use, with no attribution to individuals and
no redistribution of identifiers.

---

## Tools and libraries

| Tool | Used for | Reference |
|---|---|---|
| **ERRANT** 3.0.2 | Edit extraction, classification and all P/R/F0.5 scoring | Bryant, Felice & Briscoe (2017), *Automatic annotation and evaluation of error types for grammatical error correction*, ACL; Felice, Bryant & Briscoe (2016), *Automatic extraction of learner errors in ESL sentences using linguistically enhanced alignments*, COLING |
| **T5** (`t5-small`) | The corrector model | Raffel et al. (2020), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 21(140) |
| **HuggingFace Transformers** 4.53.1 | `Seq2SeqTrainer`, tokenisers, generation | Wolf et al. (2020), *Transformers: State-of-the-Art Natural Language Processing*, EMNLP demos |
| **HuggingFace Datasets** 4.0.0 | Streaming WikiText-103 only (not the training path) | Lhoest et al. (2021), EMNLP demos |
| **spaCy** 3.8.7 + `en_core_web_sm` 3.8.0 | Tokenisation, POS tags, lemmas, sentence splitting | Honnibal, Montani, Van Landeghem & Boyd (2020), spaCy |
| **LemmInflect** 0.2.3 | Verb and noun inflection in the generator | https://github.com/bjascob/LemmInflect |
| **sacreBLEU** 2.5.1 | BLEU, reported two ways | Post (2018), *A Call for Clarity in Reporting BLEU Scores*, WMT |
| **scikit-learn** 1.7.1 | Cohen's kappa for annotator agreement | Pedregosa et al. (2011), JMLR 12 |
| **PyTorch** 2.6.0+cu124 | Training and inference | Paszke et al. (2019), NeurIPS |
| **Gradio** 5.35.0 | The side-by-side demo | https://gradio.app |
| pandas, matplotlib, PyYAML, tqdm, rapidfuzz, pytest | Analysis, plotting, config, tests | — |

---

## Papers this project responds to

The three studied in Activity 1, whose combined gap defines this project:

1. **Pan & Cao (2021)**, *Towards Efficient Grammatical Error Correction with
   Error Detection*, IEEE ICWS. Detect errors first, then correct only those;
   uses `difflib.SequenceMatcher` to extract edits.
2. **Koo, Kim, Lee et al. (2023)**, *Uncovering the Risks and Drawbacks
   Associated With the Use of Synthetic Data for Grammatical Error
   Correction*, IEEE Access 11. Studies noise injection and balanced
   clean/noisy data, but trains **only** on synthetic data and evaluates with
   BLEU. Our **E0** is a direct response to that metric choice; our **E1/E2**
   reproduce its noise-ratio and clean-ratio questions under F0.5.
3. **Wang, Zhao et al. (2024)**, *Improving Grammatical Error Correction via
   Contextual Data Augmentation*, Findings of ACL. Synthetic data helps when it
   matches the real error distribution and is mixed with real data. Our **E4**
   tests the distribution-matching claim directly.

---

## External code

No third-party source files were copied into this repository. Two places are
worth naming explicitly because the implementation depends on details of
ERRANT's *behaviour*, which were read from the installed package source
(`errant/commands/compare_m2.py`, `errant/commands/parallel_to_m2.py`,
errant 3.0.2, Apache-2.0) rather than from its documentation:

- `gec.evaluate.score_m2_pure_python` reimplements the scoring algorithm of
  `errant_compare` — the `(start, end, cor)` edit key, the category-list
  multiplicity, the `noop` coder marker and the greedy best-combination search
  with its exact tie-break order — purely so it can **cross-check** the numbers
  parsed out of the CLI's stdout. ERRANT itself remains the scorer of record.
- `gec.errant_cli` works around two behaviours of errant 3.0.2:
  `parallel_to_m2.py` has no `if __name__ == "__main__"` guard (so `python -m`
  is a silent no-op, and we invoke `main()` via `-c` instead), and every
  command opens files with a bare `open()` (so `PYTHONUTF8=1` is set for the
  subprocess, Windows defaulting to cp1252).

---

## Declaration of AI-tool use

**Code written with the assistance of Claude Code (Anthropic); all code
reviewed and run by the group.**

Specifically: Claude Code wrote the modules under `src/gec/`, `app/`,
`tests/`, `scripts/` and `configs/`, the Colab notebook, and the project
documentation (`README.md`, `PROGRESS.md`, this file), working from the
group's own `PROJECT_SPEC.md`. Every experiment was executed by the group on
the group's hardware, and every number reported in `results/RESULTS.md` and in
the Activity 2 report was produced by running this code — none was supplied by
the model.

Per the project's own rules, Claude Code did **not** write, choose, correct or
annotate any Indian English test sentence. It produced the TSV template, the
importer, the validator, the agreement tooling, the per-annotator edit files,
and five rows marked `EXAMPLE-` that demonstrate the format. The 149 real items
were collected and selected by the group from public posts, and both sets of
independent annotations are the group's work - which is what keeps E5 a
human-made evaluation set.

The importer applies exactly one text transformation, curly quotes and
apostrophes to straight ones, because T5's SentencePiece vocabulary cannot
represent the curly forms. Typos, spacing, capitalisation, exclamation marks,
slashes and brackets are all preserved verbatim; a test asserts the imported
sources round-trip identically to the collected file.

The group's Activity 1 report (`IA1_REPORT_GEC.md`), the research question and
the experimental design in `PROJECT_SPEC.md` are the group's own.
