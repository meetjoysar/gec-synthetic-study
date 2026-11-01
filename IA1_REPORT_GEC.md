# Research Paper Study and Project Proposal
## Grammatical Error Correction

**Internal Assessment — Activity 1**

| | |
|---|---|
| **Course** | 316P07E123 — Natural Language Processing |
| **Programme** | FY M.Tech, Computer Engineering, Semester I |
| **Academic year** | 2026–27 |
| **Faculty in charge** | Dr. Grishma Sharma |
| **Common NLP problem** | Grammatical Error Correction (GEC) |
| **Date of submission** | _____________ |

### Group members

| # | Name | Roll number | Paper analysed |
|---|---|---|---|
| 1 | _____________ | _____________ | Pan & Cao, IEEE ICWS 2021 |
| 2 | _____________ | _____________ | Koo et al., IEEE Access 2023 |
| 3 | _____________ | _____________ | Wang et al., Findings of ACL 2024 |

---

> **Note to the group before you submit this.** This is a complete working draft, not a
> finished submission. Three things still need your hands on them:
> fill in the title page; rewrite each paper section in your own voice, because Activity 1
> marks each of you individually on *your* understanding and a shared phrasing style is easy
> to spot in a viva; and verify the numbers in Tables 1.6, 2.6 and 3.6 against the PDFs
> yourself, since you will be asked where each one came from. The critique lists are
> deliberately longer than you need, so keep the points you can defend and cut the rest.

---

## Table of contents

1. [Introduction](#1-introduction)
2. [Paper 1 — Pan & Cao (IEEE ICWS 2021)](#2-paper-1--pan--cao-ieee-icws-2021)
3. [Paper 2 — Koo et al. (IEEE Access 2023)](#3-paper-2--koo-et-al-ieee-access-2023)
4. [Paper 3 — Wang et al. (Findings of ACL 2024)](#4-paper-3--wang-et-al-findings-of-acl-2024)
5. [Comparative analysis](#5-comparative-analysis)
6. [Proposed mini-project](#6-proposed-mini-project)
7. [Individual contribution of each student](#7-individual-contribution-of-each-student)
8. [Declaration of AI tools used](#8-declaration-of-ai-tools-used)
9. [References](#9-references)

---

## 1. Introduction

Grammatical Error Correction (GEC) is the task of taking a sentence that contains mistakes
and returning a corrected version of it. A GEC system given *"He go to school yesterday and
buy a apple"* should return *"He went to school yesterday and bought an apple"*, fixing two
verb tenses and one article, while leaving everything else alone. The task shows up in
writing assistants such as Grammarly, in search engines that need to interpret malformed
queries, and as a post-processing step after speech recognition, where the transcript is
often fluent-sounding but wrong.

Two properties make GEC harder than it looks. First, **most text is already correct**, so a
system that edits too eagerly does more damage than one that does nothing. This is why the
field evaluates with F0.5 rather than plain F1: F0.5 weights precision twice as heavily as
recall, because a wrong "correction" irritates a user more than a missed error does. Second,
**labelled data is scarce**. Annotating learner text requires trained annotators, so the
largest high-quality corpus in common use, W&I+LOCNESS, contains only about 34,000
sentences. For a sequence-to-sequence task, that is very little.

Those two pressures explain the shape of modern GEC research, and they are what connect the
three papers we selected. All three address GEC, but from different directions:

- **Pan & Cao (2021)** ask how to make correction *fast* enough for live services, and
  restructure the model into a detect-then-correct pipeline to do it.
- **Koo et al. (2023)** ask what happens when a model is trained on *synthetic data alone*,
  and find that the standard data-quality techniques backfire badly.
- **Wang et al. (2024)** ask how to build synthetic data that *does* help, and fix the two
  specific defects (mismatched error distribution and noisy labels) that the other
  two
  papers run into.

Read together they form an argument rather than three isolated results. Pan & Cao show the
architecture, Koo et al. show the failure mode of cheap data, and Wang et al. show the
repair. The gap between Koo et al. and Wang et al. is a genuine disagreement about whether
synthetic data helps GEC, and testing that disagreement at a small scale is what our
mini-project does.

---

## 2. Paper 1 — Pan & Cao (IEEE ICWS 2021)

**Fayu Pan, Bin Cao.** *Efficient Grammatical Error Correction with Hierarchical Error
Detections and Correction.* 2021 IEEE International Conference on Web Services (ICWS),
pp. 525–530. DOI: 10.1109/ICWS53863.2021.00073
Code: `github.com/AnticPan/Hierarchical-GEC`

**Analysed by:** Student 1

### 2.1 Problem

Semantic services such as chatbots and search are trained on clean text but receive noisy
text at run time, which hurts their accuracy. GEC can clean the input first, but the usual
neural machine translation approach to GEC takes several hundred milliseconds per sentence
because its decoder is autoregressive: each token waits for the previous one. That is too
slow for a service that has to answer in real time. The paper asks whether GEC can be
restructured so that correction is fast enough to sit in front of a live service.

### 2.2 Methodology

The authors' observation is that people do not proofread by rewriting a sentence from
scratch. They skim it, decide whether anything is wrong, locate the problems, and then fix
those spots in place. The architecture imitates that sequence, using one shared encoder and
three task-specific heads trained together as a multi-task model.

```
                    Input sentence X
                           │
                  ┌────────▼────────┐
                  │  SpanBERT       │   shared encoder
                  │  encoder        │   (sentence + token features)
                  └────────┬────────┘
                           │
          ┌────────────────▼─────────────────┐
          │  DISCRIMINATOR  (task: SED)      │
          │  binary correct / incorrect      │
          │  from pooled [CLS]               │
          └────────┬────────────────┬────────┘
                   │                │
            correct │                │ incorrect
                   ▼                ▼
            ┌─────────────┐  ┌──────────────────────────────┐
            │ OUTPUT X    │  │  DETECTOR  (task: GED)       │
            │ unchanged   │  │  BIO tagging + CRF layer     │
            │ (early exit)│  │  error types: R / M / WS      │
            └─────────────┘  └──────────────┬───────────────┘
                                            │ error spans (s, e)
                                            ▼
                             ┌──────────────────────────────┐
                             │  CORRECTOR  (task: GEC)      │
                             │  non-autoregressive decoder  │
                             │  emits patch (s, e, Y')      │
                             │  max length M = 4, [EOP] end │
                             └──────────────┬───────────────┘
                                            │
                                            ▼
                                   Corrected sentence Y
```

**Encoder.** SpanBERT is used instead of BERT. The authors argue its pre-training objective,
masking and predicting a whole span of words, is closer to error correction than BERT's
next-sentence prediction, and their own experiments support this.

**Discriminator (Sentence Error Discrimination).** The pooled `[CLS]` state is passed
through a `tanh` layer and a sigmoid classifier to give the probability that the sentence is
correct. A threshold, chosen by grid search on the development set, decides the cut-off.
Sentences judged correct skip the remaining two stages entirely. This is the stage that buys
the speed: roughly 10% of the CoNLL-2014 test set exits here, and the authors point out that
the proportion would be much higher on ordinary text, which contains far fewer errors than a
GEC benchmark.

**Detector (Grammatical Error Detection).** Each token is tagged with a BIO label over three
error types taken from the ERRANT toolkit's categories: Redundancy (**R**) for tokens that
should be deleted, Missing (**M**) where something should be inserted, and Word Selection
(**WS**) where a token should be replaced. BIO tagging is needed because one error can span
several tokens. A CRF layer sits on top to keep the label sequence coherent.

**Corrector.** Rather than regenerating the sentence, the model produces a *patch* for each
detected error: a triple `(s, e, Y')` meaning "replace everything strictly between position
`s` and position `e` with `Y'`". Because the detector has already supplied `s` and `e`, only
`Y'` has to be generated. The decoder is **non-autoregressive**: every token of `Y'` is
predicted in parallel from the error's context representation plus a position embedding, so
there is no sequential dependency. A constant `M = 4` caps the generated length at inference
time, and a special `[EOP]` token appended during training lets the output be clipped at the
right point.

### 2.3 Dataset

Following the restricted track of the BEA-2019 shared task:

| Corpus | Role |
|---|---|
| FCE | Training |
| NUCLE | Training |
| Lang-8 | Training |
| W&I+LOCNESS | Training |
| CoNLL-2014 test | Evaluation (2 annotators) |
| BEA-2019 test | Evaluation (CodaLab) |

English only. No pseudo-data pre-training was used, which the authors state explicitly so
that their comparison against other systems is like-for-like.

### 2.4 Preprocessing

This is the most reusable part of the paper, and it is given as explicit pseudocode
(Algorithm 1).

1. If an annotator supplied more than one acceptable correction, keep the target with the
   **minimum Levenshtein distance** from the source, that is, the correction requiring the
   fewest edits.
2. Prepend `[CLS]` and append `[SEP]`.
3. Derive the sentence-level label by testing whether the source equals the target.
4. Run Python's `difflib.SequenceMatcher` on the (source, target) pair to obtain edit
   operations, then convert each operation into labels and a patch:

| Edit operation | BIO labels assigned | Patch emitted |
|---|---|---|
| `delete` | `B-R`, then `I-R` | `(s−1, e, ('[NONE]'))` |
| `insert` | `B-M` | `(s−1, e, Y[sY:eY])` |
| `replace` | `B-WS`, then `I-WS` | `(s−1, e, Y[sY:eY])` |

Worked example from the paper's Table I:

```
X : [CLS] He is the first of all in the list . [SEP]
Y : [CLS] He is the first        in the list . [SEP]
operation : ('delete', 5, 7, 5, 5)
labels    : [ O O O O O B-R I-R O O O O O ]
patch     : [ (4, 7, ('[NONE]')) ]
```

`M = 4` was chosen from the average observed patch length in the training data.

### 2.5 Evaluation metrics

| Metric | What it measures | Why chosen |
|---|---|---|
| **M² (MaxMatch)** | Finds the edit sequence between source and hypothesis that best matches the gold edits, then scores P/R/F0.5 over edits | The official CoNLL-2014 metric, so results are comparable with prior work |
| **ERRANT** | Extracts and classifies edits automatically, then scores them | The official BEA-2019 metric |
| **F0.5** | Precision-weighted harmonic mean | A false correction is worse for a user than a missed error |
| **Inference time** | ms per batch and total seconds, at batch sizes 1 / 8 / 16 / 32 | The paper's whole claim is about speed, so it has to be measured |

All timings on one NVIDIA V100, CUDA 10.2, three repeats per setting.

### 2.6 Results

**Main comparison** (best non-ensemble results, no pseudo-data pre-training):

| Method | Type | CoNLL-14 P / R / F0.5 | BEA-19 P / R / F0.5 |
|---|---|---|---|
| Zhao et al. (2019) | NMT | 65.5 / 33.2 / 54.9 | — |
| LaserTagger | SL | 50.9 / 26.9 / 43.2 | 53.4 / 38.5 / 49.6 |
| GECToR | SL | 66.8 / 33.7 / **55.8** | 64.2 / 47.0 / **59.8** |
| Chen et al. (2020) | SL + NMT | 66.0 / 24.7 / 49.5 | 62.7 / 38.6 / 55.7 |
| **This paper (BERT)** | SL | 55.1 / 32.3 / 48.3 | 50.6 / 40.5 / 48.2 |
| **This paper (SpanBERT)** | SL | 57.2 / 34.1 / 50.4 | 54.5 / 44.3 / 52.1 |

**Inference time and accuracy together:**

| System | Params | ms/sentence at batch 1 | CoNLL-14 F0.5 |
|---|---|---|---|
| Zhao et al., beam 1 (greedy) | 97M | 245 | 53.9 |
| Zhao et al., beam 4 | 97M | 431 | 54.3 |
| Chen et al., beam 5 | 125M + 209M | 104 | 49.5 |
| GECToR, 4 iterations | 112M | 20.65 | 55.8 |
| **This paper** | 133M | **14.89** | 50.4 |

**Module cost** (seconds over the test set, batch 1): Encoder 13.58 · Detector 3.52 ·
Corrector 1.29 · Discriminator 0.26 · Total 19.53.

**Ablation (CoNLL-14):**

| Variant | F0.5 | Time (s, batch 16) |
|---|---|---|
| Full approach | 50.4 | 3.09 |
| − CRF layer | 48.0 | 2.30 |
| − Discriminator | 45.6 | 2.24 |
| − patch correction for redundancy (just delete) | 42.6 | 2.24 |

**Sentence-level detection accuracy:** 78.2% on CoNLL-14 A1, 88.6% on A2, 93.8% when a
prediction counts as correct if it matches either annotator.

**What the numbers show.** The speed claim holds: 14.89 ms/sentence is about 30% faster than
GECToR and over ten times faster than the NMT systems. The accuracy claim is weaker. At
50.4 F0.5 the model sits 5.4 points below GECToR, which the abstract describes as
"comparable", which is generous, but the gap is reported honestly in the table. The ablation is the
most informative part: removing the discriminator costs 4.8 F0.5 while saving only 0.06
seconds, meaning the early-exit gate is improving *accuracy*, not just speed, presumably by
teaching the shared encoder to distinguish correct from incorrect text.

### 2.7 Strengths

1. **The efficiency claim is actually measured**, module by module and across four batch
   sizes, rather than asserted. Many architecture papers skip this.
2. **The ablation isolates each design decision** and reports the latency each one costs,
   so a reader can decide which parts to keep.
3. **The early-exit insight is sound and under-used.** Real text is mostly correct, so
   spending full model capacity on every sentence is wasteful. No prior sequence-labelling
   work modelled sentence-level correctness at all.
4. **Reproducible.** Standard datasets, standard metrics, released code, and preprocessing
   given as explicit pseudocode.

### 2.8 Limitations

1. **The accuracy gap is real and under-stated.** 50.4 against GECToR's 55.8 is not
   "comparable performance"; it is a deliberate trade that the abstract soft-pedals.
2. **The optimisation targets the wrong bottleneck.** The encoder accounts for 13.58 of
   19.53 seconds, about 70% of inference. All the architectural cleverness is spent on the
   heads, which together cost about 5 seconds. Compressing the encoder would have paid far
   better.
3. **Errors propagate with no recovery path.** A sentence the discriminator wrongly clears
   is never examined again, and its accuracy is only 78.2% against the first annotator. The
   authors' own ablation also shows the detector misses redundancy errors.
4. **Hard structural caps.** Only three coarse error types against ERRANT's roughly 25, and
   `M = 4` makes any correction longer than three tokens impossible to express.
5. **Non-autoregressive generation predicts tokens independently**, so a multi-token patch
   can come out internally incoherent. The paper does not examine this.
6. **133M parameters against GECToR's 112M**, which is awkward for a paper whose selling
   point is efficiency. Single GPU, single seed, no variance reported.

### 2.9 Possible improvements

1. **Shrink or distil the encoder** (DistilBERT, ALBERT or MobileBERT), since that is where
   the inference time actually goes. This is the change with the largest expected payoff.
2. **Make the discriminator recoverable**, for example by tuning its threshold for high
   recall on "incorrect" so that borderline sentences fall through to the detector rather
   than exiting silently.
3. **Add pseudo-data pre-training**, which the authors list as future work and which the
   other two papers in this report address directly. *Our mini-project tests exactly this.*

---

## 3. Paper 2 — Koo et al. (IEEE Access 2023)

**Seonmin Koo, Chanjun Park, Seolhwa Lee, Jaehyung Seo, Sugyeong Eo, Hyeonseok Moon,
Heuiseok Lim.** *Uncovering the Risks and Drawbacks Associated With the Use of Synthetic
Data for Grammatical Error Correction.* IEEE Access, vol. 11, 2023, pp. 95747–95756.
DOI: 10.1109/ACCESS.2023.3310257

**Analysed by:** Student 2

### 3.1 Problem

Data-centric AI improves a model by improving its data rather than its architecture, and
techniques such as noise injection and deliberately balancing clean against noisy examples
are well established on real datasets. Two trends are now pushing teams toward datasets that
are *entirely* synthetic: large language models have made good synthetic text cheap, and
privacy law makes real data containing personal information legally awkward. The paper asks
the obvious follow-up question, which nobody had tested:

> *Do data quality control techniques known to help data-centric AI still help a model
> trained exclusively on synthetic data?*

### 3.2 Methodology

The task is Korean speech-recognition post-processing, treated as a GEC problem: take the
error-filled output of a speech recogniser and correct it.

Synthetic training data comes from **BackTranScription (BTS)**, which exploits the fact that
a round trip through speech introduces realistic errors for free:

```
  clean monolingual text
          │
          ▼  TTS  (WaveNet)
     synthesised speech
          │
          ▼  STT  (Naver CLOVA CSR)
    transcribed text  ──── noisy
          │
          ▼
  parallel pair = (transcribed, original)
```

No human annotation is needed, which is the whole appeal. On top of this base the authors
run three experiments.

**Experiment 1 — noise injection.** Six Korean-specific noise types are applied to the
source side: separation of a character into its consonants and vowels, vowel transformation,
pronunciation-based alteration, punctuation errors, loanword transformation, and neologism
errors. Each is applied at the word level with probability 0.1, 0.2, 0.4, 0.6, 0.8 or 1.0,
giving six models to compare against an un-injected baseline.

**Experiment 2 — balanced data.** Clean BTS pairs and noise-injected pairs are mixed in
fixed proportions (5:5, 4:6, 3:7, 2:8 and 1:9), crossed with the noise ratios above.

**Experiment 3 — data-driven training strategies (DATS).** Parallel pairs are split and
re-paired so that both sides share the same characteristics: source with source (both
error-filled) giving `DATS_src`, and target with target (both clean) giving `DATS_tgt`. Each
is then optionally fine-tuned on the real source–target synthetic data, producing
`fine-tuned DATS_src` and `fine-tuned DATS_tgt`.

The model is deliberately a **plain vanilla Transformer** with the original Vaswani et al.
hyperparameters, implemented in Fairseq with SentencePiece subwords and a 50,000 vocabulary.
Using no pretrained language model is intentional: it keeps the results attributable to the
data rather than the architecture.

### 3.3 Dataset

| Corpus | Sentences | Role |
|---|---|---|
| AI-HUB (Korean) | 92,000 | Raw monolingual input to BTS |
| TED Korean transcripts | 119,883 | Raw monolingual input to BTS |

Both are openly available and well curated. Everything used for training is synthetic, which is the
point of the paper. Korean only, which the authors justify by noting the original BTS work
was also Korean.

### 3.4 Preprocessing

1. Crawl the pre-built monolingual corpus.
2. Convert text to speech with WaveNet TTS.
3. Convert speech back to text with the CLOVA speech recognition API. Reported conversion
   times run to 120 and 72 hours for TTS and 36 and 24 hours for STT across the two corpora.
4. Pair the round-tripped output with the original text to make a pseudo-parallel corpus.
5. Apply the experiment-specific transformation: noise injection, ratio mixing, or
   re-pairing for DATS.
6. SentencePiece subword tokenisation, 50,000 vocabulary.

### 3.5 Evaluation metrics

| Metric | What it measures | Why chosen |
|---|---|---|
| **BLEU** | n-gram overlap between output and reference | Standard in machine translation; carried over from the earlier BTS work |
| **GLEU** | BLEU adapted for GEC — rewards n-grams that match the reference and penalises those wrongly kept from the source | A partial correction for BLEU's weaknesses on this task |

The choice of metric is the paper's most consequential methodological decision, and we
return to it in the limitations.

### 3.6 Results

**Experiment 1 — noise injection.** Baselines: 65.69 (AI-HUB), 56.14 (TED).

| Outcome | AI-HUB | TED |
|---|---|---|
| Best observed | 65.37 at ratio 0.4 | 56.55 at ratio 0.2 |
| Worst observed | 63.82 at ratio 0.1 | 54.2 at ratio 0.8 |
| Beats baseline? | No | Only at ratio 0.2, by 0.41 |

**Experiment 2 — balanced data.** Performance falls once the noise ratio passes 0.6. AI-HUB
drops 19.07% to 53.16 at a 1:9 clean-to-noisy ratio with noise 1.0. TED shows no improvement
at all and sits 32.04% below baseline at 38.15. Mild gains appear only in the 0.1–0.4 noise
band.

**Experiment 3 — training strategies** (AI-HUB BLEU):

| Model | BLEU | Change vs baseline |
|---|---|---|
| Baseline (plain BTS data) | 65.69 | — |
| DATS_src | 45.82 | −19.87 |
| DATS_tgt | 30.98 | −34.71 |
| fine-tuned DATS_src | 44.90 | −20.79 |
| fine-tuned DATS_tgt | 21.66 | **−44.03** |

**What the numbers show.** Every technique made things worse, and the plain baseline won
every comparison. The largest drop, 44.03 BLEU, is severe enough that it is not a tuning
artefact. Additional fine-tuning on synthetic data did not rescue the degraded models, which
suggests the damage is done early in training. The honest conclusion is the one the authors
draw: a method that reliably helps on real data cannot be assumed to help when the data is
entirely synthetic.

### 3.7 Strengths

1. **It is a negative result, published.** Most of the field reports gains; establishing
   that a popular technique *fails* under a specific condition is rarer and arguably more
   useful, because it stops other people wasting effort.
2. **Three independent experimental designs**, each swept across a range of settings rather
   than spot-checked at one value.
3. **The deliberately plain model is the right choice** for the question. A pretrained
   encoder-decoder would have confounded data effects with architecture effects.
4. **The cases that did work are reported**, not buried. The mild gains at low noise ratios
   are stated even though they complicate the headline.

### 3.8 Limitations

1. **There is no real-data control.** The paper's claim is that synthetic data behaves
   *differently* from real data, but it never runs the same three experiments on real data
   for comparison. The baseline is synthetic data without treatment, not real data. Without
   that control the central claim is not actually tested. This is the paper's most serious
   gap, and the one our mini-project fills.
2. **BLEU is the wrong metric for GEC.** BLEU rewards n-gram overlap with the reference, and
   in GEC the source and reference are over 80% identical. A model that simply copies its
   input therefore scores a high BLEU while correcting nothing. GEC convention is F0.5 over
   edits precisely to avoid this. Some of the reported degradation may be the metric
   reacting to increased editing rather than to genuinely worse output.
3. **The injected noise does not match the test errors.** The six noise types are
   typing-style and orthographic, while the test data contains speech-recognition errors,
   which are structurally different: homophones and word-boundary mistakes. Injecting the
   wrong kind of noise is a plausible alternative explanation for the whole result, and it is
   never considered.
4. **The LLM framing is not supported by any experiment.** The introduction leans heavily on
   GPT-3, ChatGPT and LaMDA, and Table 1 displays ChatGPT output, but every experiment uses
   BTS data. No LLM-generated data is tested anywhere. The motivation and the evidence are
   about different things.
5. **Single runs, no variance.** No seeds, no confidence intervals, no significance tests.
   Several of the differences discussed in the text are under 1.5 BLEU, which is within the
   range seeds alone can produce.
6. **A likely transcription error in the results.** Table 4 gives `DATS_src` and `DATS_tgt`
   the identical TED score and identical delta (39.27, −16.87), although their AI-HUB scores
   differ sharply (45.82 against 30.98). One of the two is probably a copy error, and it is
   not addressed.
7. **No mechanism is offered.** The paper establishes *that* the techniques fail but never
   investigates *why*. The most plausible explanation, that BTS data is already noisy so adding
   more noise pushes it past a usable signal-to-noise ratio, is never tested, although it
   would have been straightforward to do so.

### 3.9 Possible improvements

1. **Add the missing real-data control.** Run all three experiments on a real learner corpus
   under identical settings. Without it, the difference between synthetic and real data is
   asserted rather than demonstrated. *Our mini-project does this.*
2. **Re-score everything with ERRANT F0.5**, and report a copy-the-input baseline alongside
   it to show what BLEU scores for doing nothing. *Our mini-project does this too, as
   experiment E0.*
3. **Match the injected noise to the observed error distribution** instead of choosing noise
   types a priori, which is, in effect, what Wang et al. (2024) go on to do.

---

## 4. Paper 3 — Wang et al. (Findings of ACL 2024)

**Yixuan Wang, Baoxin Wang, Yijun Liu, Qingfu Zhu, Dayong Wu, Wanxiang Che.** *Improving
Grammatical Error Correction via Contextual Data Augmentation.* Findings of the Association
for Computational Linguistics: ACL 2024, pp. 10898–10910.
Harbin Institute of Technology and iFLYTEK Research.
Code: `github.com/wyxstriker/CDA4GEC`

**Analysed by:** Student 3

### 4.1 Problem

Synthetic data is standard practice in GEC, but almost always restricted to the
**pre-training** phase. Nobody uses it in the **fine-tuning** phase, where data is scarcest
and where it would help most. The paper identifies two concrete reasons for this:

1. **Inconsistent error distribution.** Synthetic data is generated too randomly to match
   the error distribution of a particular high-quality corpus. Training on both together
   therefore *degrades* performance. Rule-based substitution is limited by the vocabulary of
   whatever unlabelled corpus it draws on; model-based generation produces varied errors but
   cannot be steered reliably toward specific ones when only a small amount of data is being
   generated.
2. **Noisy labels.** Synthetic data is not human-checked, so it contains bad annotations.
   Their example is *"I think you are right"* being "corrected" to *"I think that you are
   right"*, which is not an error at all. GEC is scored with token-level metrics, so it is
   unusually sensitive to this kind of mislabelling.

Figure 1 of the paper makes the first problem visible: plotted against the 100 most frequent
error patterns in W&I+LOCNESS, both rule-based and model-based synthetic corpora have
distributions that diverge sharply from the real one, while the authors' method tracks it
closely.

### 4.2 Methodology

The method is called **Contextual Data Augmentation (CDA)**. The idea is to keep the real
errors and synthesise everything around them. This is the inverse of normal practice, which keeps
real sentences and synthesises the errors.

```
STEP 1  Error pattern extraction
        Run ERRANT over a real parallel corpus.
        Each edit becomes a pattern (wrong form, correct form).
        Build one error pattern pool per dataset.
        Extract patterns at 1-gram, 3-gram and 5-gram lengths.
                            │
                            ▼
STEP 2  Pattern sampling
        Sample 1-2 patterns per example, AT THE TRUE FREQUENCY
        of the original dataset.   ← this is what fixes the distribution
                            │
                            ▼
STEP 3  Contextual corpus generation   (hard-constrained generation)
        Input :  [...] correct_pattern_1 [...] correct_pattern_2 [...]
        Output:  a fluent sentence containing those correct patterns
        Generators tried:
          - GPT-2 base, supervised fine-tuning
          - LLaMA2-7b-chat, 5-shot in-context learning
                            │
                            ▼
STEP 4  Pattern substitution
        Replace correct patterns with their wrong counterparts by rule.
        Only 50% of examples get substituted, to keep the error rate
        consistent with the real annotated data.
        → parallel corpus (wrong sentence, correct sentence)
                            │
                            ▼
STEP 5  Denoising by relabeling
        Treat the synthetic data as unlabelled error-filled text.
        Relabel it with a strong baseline GEC model.
        Token-level, unlike sentence-level perplexity filtering.
                            │
                            ▼
STEP 6  Joint training with real data in the fine-tuning stages
```

Note the asymmetry in step 3: because pretrained language models have only ever seen
grammatical text, the generator is asked to produce context around the **correct** version of
the pattern. The error is introduced afterwards by rule. This is what makes the generation
reliable.

For the GPT-2 generator, the training data is built from the correct sentences of C-Lang8:
random consecutive segments are replaced with a `[...]` placeholder, and the model learns to
regenerate the missing context from the remaining fragments, exactly the shape of the
inference-time task.

**Three-stage joint training**, following Bout et al. (2023):

| Stage | Real data | Synthetic augmentation |
|---|---|---|
| I — pre-training | C4_200M (180M pairs) | — |
| II — fine-tuning | Lang-8 + NUCLE + FCE + W&I+L | StageII-Syn, 2M pairs |
| III — fine-tuning | W&I+L alone | StageIII-Syn, 200,000 pairs |

### 4.3 Dataset

| Dataset | Errorful % | Sentences | Stage |
|---|---|---|---|
| C4_200M (synthetic) | 99.4 | 180M | I |
| Lang-8 | 48.0 | 1,037,561 | II |
| NUCLE | 38.0 | 56,958 | II |
| FCE | 62.5 | 28,350 | II |
| W&I+LOCNESS | 67.3 | 34,304 | II & III |
| StageII-Syn (theirs) | 50.0 | 2,000,000 | II+ |
| StageIII-Syn (theirs) | 50.0 | 200,000 | III+ |
| BEA19-Dev | 64.3 | 4,384 | Validation |
| CoNLL14-Test | 71.9 | 1,312 | Test |
| BEA19-Test | — | 4,477 | Test |

Size of the extracted error pattern pools (Appendix A): Lang-8 677,475 patterns · W&I+L
62,952 · NUCLE 49,347 · FCE 43,854. English only.

### 4.4 Preprocessing

1. **Pattern extraction** with ERRANT over each annotated corpus, following Choe et al.
   (2019); patterns kept at three different n-gram lengths.
2. **Frequency-matched sampling** from the pool, so the synthetic corpus inherits the real
   corpus's error distribution instead of a uniform or arbitrary one.
3. **Context generation** as a hard-constrained task, with sampling-based decoding for
   diversity. Where the generator fails to reproduce a pattern exactly, that pattern is
   discarded rather than forced.
4. **Rule-based substitution** of correct patterns with wrong ones, on 50% of examples.
5. **Relabeling** of the whole synthetic set by the baseline GEC model, to clean token-level
   label noise.

### 4.5 Evaluation metrics

| Metric | What it measures | Why chosen |
|---|---|---|
| **M² scorer** | P/R/F0.5 over edits on CoNLL14-Test | Official CoNLL-2014 metric |
| **ERRANT** | P/R/F0.5 over automatically classified edits, scored on the CodaLab platform | Official BEA-2019 metric |
| **F0.5 on BEA19-Dev** | Validation during development | Lets them report stage-by-stage and ablation results without touching the test sets |
| **Per-category F0.5** | ERRANT error-type breakdown | Shows *which* errors improve, not just the aggregate |

### 4.6 Results

**Main comparison:**

| Model | Params | CoNLL14 P / R / F0.5 | BEA19-Test P / R / F0.5 |
|---|---|---|---|
| GECToR | 350M | 77.5 / 40.1 / 65.3 | 79.2 / 53.9 / 72.4 |
| T5-large | 770M | — / — / 66.1 | — / — / 72.1 |
| T5-XL | 3B | — / — / 67.8 | — / — / 73.9 |
| T5-XXL | 11B | — / — / 68.9 | — / — / 75.9 |
| SynGEC | 400M | 74.7 / 49.0 / 67.6 | 75.1 / 65.5 / 72.9 |
| TemplateGEC | 770M | 74.8 / 50.0 / 68.1 | 76.8 / 64.8 / 74.1 |
| MixEdit | 400M | 75.6 / 46.8 / 67.3 | 76.4 / 62.7 / 73.2 |
| MultiTaskBART | 400M | 75.4 / 51.2 / — | 78.2 / 65.5 / 75.3 |
| BART baseline | 400M | 73.8 / 53.5 / 68.6 | 74.5 / 68.9 / 73.5 |
| **+ CDA without denoising** | 400M | 76.7 / 44.8 / 67.1 | 76.1 / 67.3 / 74.1 |
| **+ CDA with denoising** | 400M | 76.2 / 52.2 / **69.8** | 77.7 / 67.5 / **75.4** |

**Ablation — generator and pattern length** (BEA19-Dev F0.5):

| Setting | P | R | F0.5 |
|---|---|---|---|
| Stage 2 model only | 61.28 | 28.84 | 50.02 |
| + Stage 3, GPT-2 generator | 64.16 | 51.13 | 61.05 |
| + Stage 3, LLaMA2 generator | 64.23 | 51.07 | 61.08 |
| + Stage 3, 1-gram patterns | 63.73 | 52.50 | 61.12 |
| + Stage 3, 3-gram patterns | 64.39 | — | **61.20** |
| + Stage 3, 5-gram patterns | 64.16 | 51.13 | 61.05 |

**Comparison against other synthetic data, all denoised** (BEA19-Dev):

| Synthetic data in joint training | P | R | F0.5 |
|---|---|---|---|
| None | 63.61 | 52.59 | 61.06 |
| PIE (rule-based) | 63.89 | 51.68 | 61.01 |
| C4_200M (model-based) | 63.96 | 51.42 | 60.98 |
| **CDA (theirs)** | 64.39 | 52.23 | **61.53** |

**Per-error-type F0.5**, five most and least frequent categories:

| Error type | Baseline | + CDA | Change |
|---|---|---|---|
| PUNCT | 77.26 | 76.96 | −0.30 |
| DET | 79.04 | 80.42 | +1.38 |
| PREP | 74.92 | 76.84 | +1.92 |
| OTHER | 84.23 | 82.32 | −1.91 |
| SPELL | 89.61 | 91.51 | +1.90 |
| CONTR | 91.67 | 93.02 | +1.35 |
| ADJ | 58.44 | 62.15 | +3.71 |
| ADV | 59.43 | 62.72 | +3.29 |
| CONJ | 48.39 | 53.33 | +4.94 |
| NOUN | 45.39 | 48.57 | +3.18 |

**Stage-wise gains** (Figure 4, BEA19-Dev F0.5): Stage 1 scores 39.57. Stage 2 goes from
50.02 to 53.05 with CDA. Stage 3 goes from 61.06 to 61.53.

**Mixing ratio** (Figure 5): as the synthetic volume rises from 50,000 to 200,000 pairs,
precision rises steadily while recall falls, an explicit trade-off rather than a free gain.

**Training setup:** BART-large in Fairseq on 4× Tesla V100S-PCIE-32GB, 10 epochs, 4096 max
tokens, update frequency 8, Adam at learning rate 3e-5, dropout 0.2, label smoothing 0.1.
GPT-2 generator: 20 epochs, learning rate 1e-5, 2000 warmup steps, patience 5.

**What the numbers show.** Three findings matter. First, a 400M BART with CDA reaches 69.8
F0.5 on CoNLL-14, above the 11-billion-parameter T5-XXL at 68.9, roughly 27 times smaller
for better accuracy. Second, and easy to miss, **CDA without denoising actively hurts**:
67.1 against the baseline's 68.6. The augmentation only pays off once the relabeling step
cleans it, which is strong direct evidence for the noisy-label diagnosis. Third, Table 6
shows PIE and C4_200M synthetic data perform *no better than using no synthetic data at all*
in joint fine-tuning (61.01 and 60.98 against 61.06), while CDA reaches 61.53. The
distribution-matching is doing real work. The per-category table explains where: rare error
types gain 3–5 points, because they are the ones starved of real examples, while the already
data-rich categories move slightly backwards.

### 4.7 Strengths

1. **It diagnoses two specific failure modes and addresses each separately**, then shows
   with the no-denoising row that both fixes were necessary. This is unusually disciplined
   experimental design.
2. **The parameter-efficiency result is striking and well supported**: 400M beating 11B is
   a stronger argument for better data than for bigger models.
3. **The per-error-type analysis explains the mechanism** rather than just reporting an
   aggregate gain, and it predicts correctly that rare categories benefit most.
4. **Table 6 is the comparison that matters** and most papers would omit it: holding
   denoising and training constant, it isolates the contribution of the data construction
   method itself.
5. Synthetic data and code are released, and the limitations section is candid.

### 4.8 Limitations

1. **It cannot introduce new error types.** The authors state this plainly: CDA enriches the
   context around errors that already exist in the annotated corpus, so an error type absent
   from the real data stays absent. The method widens the data it has; it cannot broaden the
   error inventory.
2. **The denoising step is circular.** Relabeling needs a strong baseline GEC model, which
   needs a C4_200M-pretrained 400M BART, which needs 180M synthetic pairs. The method cannot
   bootstrap from nothing; it improves a good model into a slightly better one.
3. **"Only a few synthetic data" is relative and overstated.** 2M plus 200,000 pairs is a
   lot in absolute terms; it is only small compared with the 180M–300M other systems use.
   The abstract's phrasing invites a misreading.
4. **The margins are thin and unmeasured.** +1.2 F0.5 on CoNLL-14 and +1.9 on BEA19-Test
   over their own baseline; CDA's 75.4 against MultiTaskBART's 75.3 on BEA19-Test is a
   0.1-point difference presented as state of the art. As with Koo et al., there are no
   seeds, no variance and no significance testing, and Table 6's decisive margin is 0.47.
5. **Figure 1 is argued qualitatively.** The claim that CDA's distribution best matches the
   real one is the paper's conceptual foundation, yet no divergence measure (KL, JS, or anything else)
   is reported. The reader is asked to compare curves by eye.
6. **A missing ablation.** Relabeling is applied to CDA data, and Table 6 denoises every
   method. But the effect of relabeling applied to real data alone is never shown, so we
   cannot tell how much of the gain is the clever data construction and how much is simply
   distillation.
7. **Expensive.** Every example needs two inference passes, one for generation and one for
   denoising, on top of training. English only, with no multilingual check.

### 4.9 Possible improvements

1. **Quantify the distribution match.** Report KL or JS divergence between synthetic and
   real error distributions and correlate it with downstream F0.5. That would turn Figure 1
   from an illustration into evidence, and would give practitioners a cheap way to screen
   synthetic data before spending GPU time on it. *A small-scale version of this is our
   experiment E4.*
2. **Use an LLM to introduce genuinely new error types**, addressing their own stated
   limitation. For instance, prompting for error categories that are rare or absent in the
   annotated corpus, then validating them against ERRANT.
3. **Run the missing ablation and report seeds**, so that a 0.1-point SOTA claim can be
   distinguished from noise.

---

## 5. Comparative analysis

| | **Pan & Cao** (ICWS 2021) | **Koo et al.** (IEEE Access 2023) | **Wang et al.** (Findings ACL 2024) |
|---|---|---|---|
| **Focus** | Architecture, inference speed | Synthetic-only training (a risk study) | Better synthetic data (a method) |
| **Research question** | Can GEC be fast enough for live services? | Do data-quality techniques work when all data is synthetic? | How do you build synthetic data usable during fine-tuning? |
| **Language** | English | Korean | English |
| **Data** | BEA-2019 real corpora | BTS synthetic only (TTS→STT) | Real corpora + CDA-augmented synthetic |
| **Error source** | Real learner errors | Speech-recognition round-trip errors | Real error patterns, synthetic context |
| **Model** | SpanBERT + 3 heads, 133M | Vanilla Transformer, no pretraining | BART-large 400M + GPT-2 generator |
| **Metrics** | F0.5 (M², ERRANT), latency | **BLEU, GLEU** | F0.5 (M², ERRANT), per-category F0.5 |
| **Headline result** | 50.4 F0.5, 14.89 ms/sentence | Data-centric techniques cost up to −44.03 BLEU | 69.8 F0.5; 400M model beats 11B T5-XXL |
| **Verdict on synthetic data** | Not used; listed as future work | **Harmful** | **Helpful, if distribution-matched and denoised** |
| **Main strength** | Efficiency genuinely measured; clean ablations | A published negative result, systematically swept | Diagnoses two failure modes and fixes both |
| **Main weakness** | 5.4 F0.5 below GECToR; optimises the wrong bottleneck | No real-data control; BLEU rewards copying | Cannot introduce new error types; circular denoising; thin margins |

### How the three papers fit together

The three papers form a chain rather than a set of parallel results.

**Pan & Cao** supply the architecture and the efficiency frame. Their contribution is the
detect-then-correct decomposition, and their honest reporting of a 5.4-point accuracy deficit
raises the obvious follow-up: where would the missing accuracy come from? They answer it
themselves, in their future work section: more training data, via pseudo-data pre-training
that they did not do.

**Koo et al.** take up exactly that suggestion and find it dangerous. If you train only on
cheap synthetic data, the standard tricks for improving data quality make the model
dramatically worse. Their paper reads as a warning against the obvious fix. But their study
has a structural hole: there is no real-data control, so what they actually demonstrate is
that *these techniques* fail on *this* synthetic corpus, not that synthetic and real data
differ in kind. And their choice of BLEU, a metric that rewards copying the input in a task
where source and target are already more than 80% identical, leaves open whether they
measured worse output or merely more editing.

**Wang et al.** resolve the tension by identifying what was wrong with the synthetic data
rather than with synthetic data as such. Their two diagnoses, mismatched error distribution
and noisy labels, map directly onto Koo et al.'s two failures: injecting arbitrary noise
distorts the distribution, and unverified synthetic pairs carry label noise. Fix both, by sampling
error patterns at the real corpus's own frequencies and relabelling with a trained model, and
the same joint training that broke Koo et al.'s models produces state-of-the-art results. The
no-denoising row makes the point sharply: without the fix, their own augmentation also hurts,
67.1 against a baseline of 68.6. Koo et al. were not wrong about the symptom; they
generalised it too far.

That leaves a question none of the three answers, because each is positioned differently.
Wang et al. show that *carefully constructed* synthetic data helps a *large, well-pretrained*
model during fine-tuning. Koo et al. show that *crudely constructed* synthetic data hurts a
*small, un-pretrained* model trained on nothing else. Nobody has asked what happens in
between: whether a small model, trained with modest resources on synthetic data generated by
a simple rule-based script, can learn to correct real errors, and how its behaviour changes
as the synthetic-to-real ratio shifts. That gap is the subject of our mini-project.

---

## 6. Proposed mini-project

### Working title

> **Fake Errors, Real Grammar: Can Synthetic Data Teach a Small Model to Correct Indian
> English?**

### The question

**Can cheap synthetic errors teach a small model to fix real grammatical errors, and at what
mixing ratio?**

This sits precisely in the gap identified above. Koo et al. say synthetic-only training
fails; Wang et al. say distribution-matched synthetic data succeeds. We test both claims at a
scale that runs on free Colab, using GEC-standard metrics, and we add the real-data control
Koo et al. omitted.

### Design

Three models, identical in every respect except their training data, evaluated on the same
held-out sentences:

| Model | Training data |
|---|---|
| **A — synthetic only** | Corrupted clean sentences from our generator |
| **B — real only** | W&I+LOCNESS learner data |
| **C — mixed** | Both, at the best ratio found in E3 |

Holding the model, the hyperparameters and the test set constant is what makes the comparison
a controlled study rather than three unrelated runs.

### Components

| Part | Choice | Notes |
|---|---|---|
| Model | `t5-small` (60M), prefix `grammar:` | HuggingFace `Seq2SeqTrainer`; roughly 15–30 min per run on a Colab T4 |
| Real data | W&I+LOCNESS (BEA-2019), optionally FCE | Free for research; registration required |
| Extra test sets | JFLEG, plus our Indian English set | JFLEG scored with GLEU |
| Clean text to corrupt | WikiText-103 sentences, 8–40 tokens | Never corrupt anything used for dev or test |
| Error generator | Our own Python script, about 150 lines | Articles, prepositions, subject–verb agreement, verb tense (`lemminflect`), noun number, spelling, punctuation, word deletion and duplication; each token corrupted with probability *p* |
| Metrics | ERRANT F0.5 (primary), GLEU, BLEU (to expose its flaw), latency in ms/sentence | `pip install errant` plus spaCy `en_core_web_sm` |
| Demo | Gradio web app | Edits highlighted with `difflib.SequenceMatcher`, following Pan & Cao's preprocessing |

### Experiments

| # | Experiment | Question it answers | Paper it engages |
|---|---|---|---|
| **E0** | Copy-input baseline scored with BLEU, GLEU and F0.5 | Is BLEU a fair GEC metric? | Koo et al. — metric critique |
| **E1** | Noise ratio *p* = 0.1, 0.2, 0.4, 0.6; synthetic only, ~50k pairs | Is there a sweet spot of noise? | Koo et al. Exp. 1 |
| **E2** | Clean-to-noisy mix at the best *p*: 0:10, 2:8, 5:5 | Do unchanged correct pairs reduce over-correction? | Koo et al. Exp. 2 |
| **E3** | Synthetic only vs real only vs mixed vs synthetic-then-real | Is synthetic data harmful alone but useful mixed? | Koo's missing control; Wang's mixing ratio |
| **E4** | Realism check: compare ERRANT error-type histograms of synthetic vs real data, retune the generator to match, retrain | Does matching the real error distribution help? | Wang et al. — their Figure 1, quantified |
| **E5** | All three models on our Indian English test set | Do models trained on standard learner data handle Indian English? | Novel contribution |
| **Stretch** | "Do no harm" test on already-correct sentences; or an LLM baseline | How often does each model damage correct text? | Pan & Cao — the SED motivation |

Run the main E3 settings with 2–3 random seeds and report mean ± standard deviation. Both Koo
et al. and Wang et al. omit this, and both draw conclusions from margins small enough that it
matters.

### What makes this different from a typical online GEC project

Most published tutorials fine-tune T5 on a grammar dataset and ship a demo. Four things
separate this from that:

1. **The real-versus-synthetic control that Koo et al. never ran** (E3).
2. **A copy-input baseline** showing that doing nothing earns a high BLEU and an F0.5 of
   zero, a direct empirical critique of Koo et al.'s metric choice (E0).
3. **An Indian English test set** of roughly 150 sentences that we write and correct
   ourselves, with inter-annotator agreement reported as Cohen's kappa (E5). No standard GEC
   benchmark covers this variety.
4. **A realism check** that measures the error-type distribution gap between our synthetic
   data and real learner data, then closes it, a small-scale, *quantified* version of Wang
   et al.'s Figure 1 (E4).

### Expected output

**For a user:** a Gradio page. Type a sentence with mistakes, get it back corrected, with
every change highlighted and labelled with a one-line reason.

| Change | Reason shown |
|---|---|
| go → went | Verb tense: past action ("yesterday") |
| buy → bought | Verb tense: past action |
| a → an | Article: "an" before a vowel sound |

All three models' outputs appear side by side, so the audience can see training data change
behaviour directly.

**For the evaluators**, which is the real result:

- A chart of F0.5 for synthetic, real and mixed training (E3)
- The noise-ratio curve (E1) and the BLEU-versus-F0.5 comparison (E0)
- Error-type histograms, synthetic against real (E4)
- Results on the Indian English test set (E5)
- A failure gallery of sentences all three models get wrong

**Deliverables:** `error_generator.py`, `train.ipynb`, `evaluate.ipynb`,
`indian_english_test.tsv` with both annotators' corrections, `app.py`, and the Activity 2
report in the prescribed sections.

### Honest expectation

F0.5 will land well below the published numbers in all three papers. Those systems use 400M
parameter models pretrained on 180M synthetic pairs across four GPUs; we are fine-tuning a
60M model on free Colab. **The result is the comparison between conditions, not a leaderboard
position.** Stating this up front is better than being asked about it in the viva.

---

## 7. Individual contribution of each student

Each student analysed one paper for Activity 1 and owns the matching part of the
implementation in Activity 2, so that everyone can explain their paper through work they did
themselves.

| Student | Roll no. | Activity 1 — paper analysed | Activity 2 — implementation ownership |
|---|---|---|---|
| 1. _____________ | ______ | Pan & Cao, IEEE ICWS 2021 | Training pipeline; edit highlighting with `SequenceMatcher`; error-type explanation lookup; Gradio demo |
| 2. _____________ | ______ | Koo et al., IEEE Access 2023 | Error generator; noise and mixing experiments (E1, E2); realism check (E4) |
| 3. _____________ | ______ | Wang et al., Findings of ACL 2024 | Real-data setup; evaluation harness and BLEU critique (E0); synthetic-versus-real study (E3); Indian English test set (E5) |

**Shared work:** problem selection and paper screening; the comparative analysis in
Section 5; independent annotation of the Indian English test set by two students with Cohen's
kappa computed between them; report writing and the demo rehearsal.

> If the group is two students rather than three: drop the Wang et al. section, and reassign
> E0 and E3 to Student 1 and E5 to Student 2.

---

## 8. Declaration of AI tools used

| Tool | Version / model | Used for | Not used for |
|---|---|---|---|
| **Claude Code** (Anthropic) | Opus 5 | Extracting text from the paper PDFs; cross-checking reported numbers against the papers' tables; identifying the quality and rubric-compliance problems in a candidate paper we subsequently rejected (Geetha et al., ICSCNA 2023); suggesting the replacement paper; structuring this report and drafting its prose; scoping the mini-project experiments and compute budget | Generating experimental results; writing the final submitted code; fabricating any number in this report |

**Specific notes for honesty:**

- Our originally selected third paper, *Grammar Error Correction using NLP* (Geetha et al.,
  ICSCNA 2023), was **rejected after review**. It names no dataset, reports no GEC metric,
  promises a four-algorithm comparison in its abstract but reports two, measures binary
  acceptability while claiming correction, presents rising validation loss as a result, and
  expands BERT as "Bidirectional Encipher Representations from Instruments", a sign of
  paraphrase-tool rewriting. It is also a comparison study rather than a research paper,
  which the IA rules exclude. We replaced it with Wang et al. (Findings of ACL 2024).
- All numerical values in Sections 2, 3 and 4 were read from the papers themselves and
  verified against their tables by the student who owns each section.
- Every paper section was rewritten by its owning student after the drafting stage, so that
  the analysis and the phrasing are our own.
- No external code has been incorporated yet. Any library or repository used in Activity 2
  will be declared in that report, including `errant`, spaCy, HuggingFace Transformers,
  `lemminflect` and Gradio.

---

## 9. References

1. F. Pan and B. Cao, "Efficient Grammatical Error Correction with Hierarchical Error
   Detections and Correction," in *2021 IEEE International Conference on Web Services
   (ICWS)*, 2021, pp. 525–530. doi: 10.1109/ICWS53863.2021.00073
2. S. Koo, C. Park, S. Lee, J. Seo, S. Eo, H. Moon, and H. Lim, "Uncovering the Risks and
   Drawbacks Associated With the Use of Synthetic Data for Grammatical Error Correction,"
   *IEEE Access*, vol. 11, pp. 95747–95756, 2023. doi: 10.1109/ACCESS.2023.3310257
3. Y. Wang, B. Wang, Y. Liu, Q. Zhu, D. Wu, and W. Che, "Improving Grammatical Error
   Correction via Contextual Data Augmentation," in *Findings of the Association for
   Computational Linguistics: ACL 2024*, 2024, pp. 10898–10910.
4. K. Omelianchuk, V. Atrasevych, A. Chernodub, and O. Skurzhanskyi, "GECToR — Grammatical
   Error Correction: Tag, Not Rewrite," in *Proc. 15th Workshop on Innovative Use of NLP for
   Building Educational Applications (BEA)*, 2020, pp. 163–170.
5. S. Kiyono, J. Suzuki, M. Mita, T. Mizumoto, and K. Inui, "An Empirical Study of
   Incorporating Pseudo Data into Grammatical Error Correction," in *Proc. EMNLP-IJCNLP*,
   2019.
6. A. Awasthi, S. Sarawagi, R. Goyal, S. Ghosh, and V. Piratla, "Parallel Iterative Edit
   Models for Local Sequence Transduction," in *Proc. EMNLP-IJCNLP*, 2019, pp. 4260–4270.
7. C. Bryant, M. Felice, and T. Briscoe, "Automatic Annotation and Evaluation of Error Types
   for Grammatical Error Correction," in *Proc. ACL*, 2017. — the ERRANT scorer
8. D. Dahlmeier and H. T. Ng, "Better Evaluation for Grammatical Error Correction," in
   *Proc. NAACL-HLT*, 2012, pp. 568–572. — the M² scorer
9. H. T. Ng, S. M. Wu, T. Briscoe, C. Hadiwinoto, R. H. Susanto, and C. Bryant, "The
   CoNLL-2014 Shared Task on Grammatical Error Correction," in *Proc. CoNLL: Shared Task*,
   2014, pp. 1–14.
10. C. Bryant, M. Felice, Ø. E. Andersen, and T. Briscoe, "The BEA-2019 Shared Task on
    Grammatical Error Correction," in *Proc. 14th Workshop on Innovative Use of NLP for
    Building Educational Applications*, 2019, pp. 52–75.
11. C. Napoles, K. Sakaguchi, M. Post, and J. Tetreault, "Ground Truth for Grammatical Error
    Correction Metrics," in *Proc. ACL-IJCNLP*, 2015, pp. 588–593. — the GLEU metric
12. M. Joshi, D. Chen, Y. Liu, D. S. Weld, L. Zettlemoyer, and O. Levy, "SpanBERT: Improving
    Pre-training by Representing and Predicting Spans," *TACL*, vol. 8, pp. 64–77, 2020.
13. C. Park, J. Seo, S. Lee, C. Lee, H. Moon, S. Eo, and H. Lim, "BTS: Back Transcription for
    Speech-to-Text Post-Processor using Text-to-Speech-to-Text," in *Proc. 8th Workshop on
    Asian Translation (WAT)*, 2021, pp. 106–116.
14. F. Stahlberg and S. Kumar, "Synthetic Data Generation for Grammatical Error Correction
    with Tagged Corruption Models," in *Proc. 16th Workshop on Innovative Use of NLP for
    Building Educational Applications*, 2021, pp. 37–47. — the C4_200M dataset
15. Somaiya Vidyavihar University, "Scheme for Internal Assessment — 316P07E123 Natural
    Language Processing," 9 September 2026.
