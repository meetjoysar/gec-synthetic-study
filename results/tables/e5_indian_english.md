# E5 - Indian English test set

| model | P | R | F0.5 | over-correction |
|---|---:|---:|---:|---:|
| e3_mixed (seed 13) | 0.3613 | 0.3966 | 0.3678 | 0.3538 |
| e4_mixed_matched (seed 13) | 0.3490 | 0.3851 | 0.3556 | 0.4 |
| e3_real (seed 13) | 0.3543 | 0.3563 | 0.3547 | 0.4154 |
| e3_synth2real (seed 13) | 0.3200 | 0.3218 | 0.3204 | 0.3385 |
| e4_synthetic_matched (seed 13) | 0.2414 | 0.1609 | 0.2194 | 0.2769 |
| e3_synthetic (seed 13) | 0.2168 | 0.1782 | 0.2078 | 0.3538 |

## Composition of the Indian English set

- sentences: **149**
- annotators per sentence: **1**
- the annotator left unchanged (control / acceptable usage): **65** (44%)
- by `source_type`: `reddit-manual` 149

Over-correction in the table above is measured on the control sentences only - those the annotator judged acceptable as written. It answers whether the model imposes a prescriptive standard on a legitimate variety of English.

## Annotator agreement

- token-level Cohen's kappa: **None**
- raw token agreement: TBD
- sentence-level kappa: None
- sentences: 149

- note: Only one annotator has filled in corrections, so Cohen's kappa is undefined: it measures agreement between two INDEPENDENT judgements and there is no second one. E5 still reports valid P/R/F0.5 against this single-reference gold standard. Add a second annotator's column and re-run to obtain kappa.