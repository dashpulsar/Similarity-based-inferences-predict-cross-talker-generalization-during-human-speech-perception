# Why are the acoustic baselines unusually competitive in AN19?

## Question and current answer

The question is why MFCC and STRF rival or exceed HuBERT CNN/Transformer representations in AN19, while HuBERT has a clearer advantage in X21. Establishing that acoustic predictors contain information does not answer this relative-performance question.

The historical ranking difference is verified. Its cause is not yet established. The current fixed-parameter analysis also shows unusually competitive AN19 acoustic baselines, but HuBERT-base Tr-24 now narrowly outperforms both. Consequently, the explanation needs to separate dataset-dependent information from changes in predictor transformation, fitting, and evaluation.

This review uses existing outputs and descriptive calculations. It does not refit GLMMs, change results, resume parameter optimization, or resume HVE work.

## 1. The ranking that prompted the question

The following are means of three historical **test-fold-refit Wald z** values. The best layer is selected descriptively from the saved layer means and has no independent winner evaluation.

| Dataset | MFCC39 | STRF24 | HuBERT-base Tr-24 | Highest HuBERT-base layer mean | HuBERT-FT Tr-24 | Highest HuBERT-FT layer mean |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| AN19 | 11.72 | 13.94 | 10.83 | 13.69, Tr-12 | 9.03 | 10.46, Tr-20 |
| X21 | 5.74 | 5.72 | 7.76 | 8.07, Tr-22 | 7.68 | 7.97, Tr-22 |

Thus, STRF exceeds every saved AN19 base-layer mean, although its lead over the highest one is only 0.25 z units. MFCC exceeds AN19 Tr-24 and all saved FT-layer means, but not every base-layer mean. These numerical rankings do not establish statistically significant differences between representations.

The corresponding AN19 baseline values are 52.89% and 62.92% of mean ceiling z; X21 values are 26.39% and 26.33%. Within each dataset, division by the same positive ceiling cannot change the ranking of representations. Ceiling normalization therefore cannot explain this within-dataset reversal.

Sources: [historical fold values](../model_comparison/reference_checks/z_value_review/tables/historical_sbi_fold_z.csv), [original baseline audit](../reference/tables/figure00_acoustic_baseline_audit.csv). The later [plotting script](../../scripts/build_september9_statistical_panels.py) retains these historical values. The exact original notebook execution that produced each baseline still needs an unambiguous source match; the saved plotting tables alone do not establish it.

## 2. What happens under the current analysis?

These are **training-fold predictor-only z** means and pooled held-out mean log losses from the fixed-parameter analysis. They are not substitutes for the historical z values above. The predictor is negative training-standardized distance, with tau fixed at 2; the stored exponential-similarity column is not the fitted predictor.

| Representation | AN19 mean training z | AN19 test loss | X21 mean training z | X21 test loss |
| --- | ---: | ---: | ---: | ---: |
| MFCC39 | 2.40 | 0.662179 | 3.76 | 0.488427 |
| STRF24 | 3.82 | 0.640384 | 3.48 | 0.488914 |
| HuBERT-base Tr-24, 3-D t-SNE | 4.01 | 0.638476 | 4.11 | 0.483585 |
| HuBERT-FT Tr-24, 3-D t-SNE | 2.37 | 0.660524 | 4.47 | 0.483256 |

Higher z and lower loss favor the representation within each respective column. AN19 STRF remains close to base Tr-24 and outperforms FT Tr-24. X21 HuBERT is ahead of both baselines. The exact claim that STRF exceeds HuBERT-base does not persist for current Tr-24.

The four representations use exactly matching available response rows and fold assignments within each dataset: 5,685 responses for AN19 and 16,477 for X21. AN19 contains 120 participants and 24 test talkers; X21 contains 320 participants and four test talkers. Registered model structures differ across datasets: X21 includes fixed test-talker effects, whereas AN19 uses talker random intercepts with a documented fallback. Cross-dataset z magnitudes should therefore not be read as a controlled comparison.

Sources: `coefficients.csv`, `cv_metrics.csv`, and `diagnostics.csv` in the six [model directories](../../artifacts/models) named `{AN19|X21}-{dataset}_{acoustic|hubert_base_tsne|hubert_ft_tsne}-confirmatory-{full|tr24}-20260906`; corresponding `*-model-input.csv` files in [derived inputs](../../artifacts/derived). Model specification: [fit_confirmatory.R](../../R/fit_confirmatory.R); historical differences: [optimizer audit](../sbi/OPTIMIZER_AUDIT.md).

## 3. A useful dataset-dependent clue

I checked whether AN19 associations mainly track differences between test items or changes across exposure conditions for the same test item. Here an item is `analysis_item_id`, which includes the test talker and word identity, not just the lexical word.

For each representation, I averaged accuracy and distance within each condition × test-item cell, producing 574 AN19 cells. I correlated negative distance with accuracy across cells, then repeated the calculation after separately subtracting each item's mean distance and mean accuracy. Each cell receives equal weight. These are descriptive Pearson correlations, not GLMM tests or cross-validated scores.

| Representation | Correlation across AN19 cells | Correlation after removing test-item means |
| --- | ---: | ---: |
| MFCC39 | 0.376 | 0.006 |
| STRF24 | 0.456 | 0.068 |
| HuBERT-base Tr-24 | 0.440 | 0.068 |
| HuBERT-FT Tr-24 | 0.364 | -0.055 |

Approximately 84% of response-weighted distance variation for MFCC, STRF, and base Tr-24 lies between AN19 test items. This fraction is `1 - sum((d - mean(d within item))^2) / sum((d - grand_mean(d))^2)`, calculated over available response rows. It describes **predictor variation**, not explained behavioral variance.

This points to a substantial association with stable differences in how intelligible particular test recordings are. It also explains why fitting within each experimental condition does not by itself isolate exposure-dependent generalization: different test items remain within every condition. All AN19 test-fold items and talkers also occur in the corresponding training folds, as intended for participant-held-out CV. This is not a held-out-item or held-out-talker evaluation.

The pattern is shared by HuBERT in AN19, so it does **not** by itself explain why STRF rivals HuBERT. It identifies the relevant next contrast: which representation captures between-item intelligibility versus exposure-dependent changes for the same item.

As a separate descriptive comparison, subtracting condition means from cell-level distance and accuracy gives the following correlations:

| Dataset | MFCC39 | STRF24 | HuBERT-base Tr-24 | HuBERT-FT Tr-24 |
| --- | ---: | ---: | ---: | ---: |
| AN19 | 0.378 | 0.448 | 0.436 | 0.344 |
| X21 | 0.073 | 0.058 | 0.248 | 0.256 |

This preserves the qualitative dataset-dependent ranking outside the GLMM z calculation. It is preliminary evidence that the difference is not exclusively a z-denominator phenomenon. Unequal designs, cell sizes, and acoustic contexts remain; X21 includes Talker-specific, and this table is not a replacement for the separate Talker-specific-excluded analysis.

AN19's untrained Control responses are excluded from this exposure-based predictor analysis (`no_exposure`); X21's available Control rows are included. The term acoustic **baseline** must not be confused with the human experimental **Control condition**.

## 4. What the completed checks establish

### Additional check for the two-slide explanation

The [two-slide presentation](../../../outputs/an19_baseline_explanation_2026-09-18/presentation/AN19_acoustic_baselines.pptx) adds a more specific lexical-word check. Here `response_expected` identifies the lexical word across talkers and conditions, whereas the earlier recording-centered check uses `analysis_item_id`. These are different groupings.

I repeated the cell-level correlation after subtracting each lexical word's mean from both negative distance and accuracy. The 574 AN19 cells cover 48 lexical words. This changes the representation ranking:

| Representation | Across cells | After subtracting word means | After removing word and condition effects together |
| --- | ---: | ---: | ---: |
| MFCC39 | 0.376 | 0.249 | 0.245 |
| STRF24 | 0.456 | 0.198 | 0.154 |
| HuBERT-base Tr-24 | 0.440 | 0.339 | 0.323 |
| HuBERT-FT Tr-24 | 0.364 | 0.362 | 0.319 |

The last column correlates residuals after projecting each variable onto an intercept plus word and condition indicators using ordinary least squares. It is a descriptive calculation, not a new GLMM fit. The main chart gives every cell equal weight. A response-count-weighted check of the word-centered correlations gives 0.249, 0.198, 0.335, and 0.362 in the same representation order.

This is evidence that lexical-word differences contribute to the *descriptive* AN19 baseline advantage. It does not identify which word properties matter or establish that they caused the historical z ranking. Word duration, phonetic composition, and recording characteristics remain untested possibilities. Shared words/talkers and noisy cell accuracy also limit inference from these correlations. A change in correlation after centering is not a percentage of explained variance.

For X21, removing Talker-specific leaves 1,236 cells. Raw correlations are 0.094 (MFCC), 0.058 (STRF), 0.283 (base Tr-24), and 0.284 (FT Tr-24). Removing Control as well leaves 824 cells and correlations of 0.097, 0.076, 0.294, and 0.290. Thus the descriptive HuBERT advantage also occurs in these subsets. No additional GLMM, t-SNE, feature extraction, or parameter search was run.

Reproduction: [analysis script](../../../outputs/an19_baseline_explanation_2026-09-18/.build/analyze.py) reads the saved inputs, asserts matching response rows/folds across representations, and writes the exact values and source hashes to `evidence.json`. Run it from the repository root. The slides contain editable charts and explanatory speaker notes. Slide 1 uses historical z; slide 2 explicitly uses current negative-distance predictors and descriptive correlations.

- Full MFCC/STRF distances reproduced to within 1.78e-15 after matching coordinate scaling. The earlier component-scaling mistake does not explain the high full-baseline values.
- The current pair audit contains 5,459 pairs, no duplicate pair IDs, and no identical test/source physical unit or speaker. The AN19 advantage is not the same self-comparison issue encountered in X21 Talker-specific.
- Fixing participant/item random intercepts across acoustic representations preserved their component ranking. That sensitivity did not include HuBERT, so it cannot establish that the acoustic-versus-HuBERT ranking is insensitive to model structure.
- Matched component analyses favor MFCC delta/delta-delta over static MFCC and favor the STRF scale-0.25 subset over the scale-1.0 subset. This localizes useful baseline information. It does not establish why HuBERT captures it less well, or separate pronunciation information from recording/duration effects.

Sources: [preparation checks](components/preparation_checks.json), [pair audit](../model_comparison/reference_checks/tables/an19_acoustic_pair_integrity_audit.csv), [common-structure results](common_structure/comparison_summary.csv).

## 5. Next checks directed at the actual question

1. Resolve the exact historical baseline and HuBERT fitting sources, then compare their transformation, parameter selection, rows and random-effects formulas. Do not attribute the historical/current ranking change to one factor when several changed together.
2. Compare acoustic and HuBERT predictors on the same AN19/X21 item/context units, separating between-item associations from within-item exposure changes. The descriptive results above motivate this check; a formal within/between model would be a new analysis.
3. Check whether duration, recording characteristics or the different extraction/reduction choices preferentially preserve AN19 baseline information. MFCC/STRF retain 39/24 dimensions while the displayed HuBERT results use 3-D t-SNE. This asymmetry exists in both datasets, so dimensionality alone is not an explanation; its dataset-dependent effect would need testing.

The defensible interim conclusion is that AN19 acoustic baselines capture strong item-level behavioral associations and can rival the selected HuBERT representations. The relative advantage is smaller under the current analysis and differs from X21. A definitive explanation of the dataset-by-representation difference remains open.
