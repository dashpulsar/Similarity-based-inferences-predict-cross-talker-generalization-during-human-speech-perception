# SBI training/test diagnostic review

## Conclusion

The six existing SBI figures implement the test/train mean log-loss ratio accepted in Florian's September 15 email. The saved values, three-fold intervals, and scoring conventions check out. They can be retained as the current fixed-predictor SI diagnostic without rerunning the models.

Mean ratios are close to one. X21 has much larger differences between individual folds than AN19 or B23, so the average alone understates the variation. The X21 pattern also appears in models without SBI and is consistent with differing response difficulty across participant folds. This is evidence against attributing the entire gap to SBI overfitting, not proof that overfitting is absent.

No CV split, model, predictor, or historical z/ceiling figure changed during this review. HVE and parameter optimization remain paused.

## Six figures

| Dataset | HuBERT base | HuBERT ASR-FT |
| --- | --- | --- |
| AN19 | [Figure](../../analysis_update_2026-09-17/si_diagnostics/figures/AN19_SBI_base_similarity_all_participants_test_train_ratio.png) | [Figure](../../analysis_update_2026-09-17/si_diagnostics/figures/AN19_SBI_ft_similarity_all_participants_test_train_ratio.png) |
| X21 | [Figure](../../analysis_update_2026-09-17/si_diagnostics/figures/X21_SBI_base_similarity_all_participants_test_train_ratio.png) | [Figure](../../analysis_update_2026-09-17/si_diagnostics/figures/X21_SBI_ft_similarity_all_participants_test_train_ratio.png) |
| B23 | [Figure](../../analysis_update_2026-09-17/si_diagnostics/figures/B23_SBI_base_similarity_all_participants_test_train_ratio.png) | [Figure](../../analysis_update_2026-09-17/si_diagnostics/figures/B23_SBI_ft_similarity_all_participants_test_train_ratio.png) |

Matching PDF and SVG files are in the same directory. All six PNGs were visually reviewed. They show MFCC39 and STRF24 followed by 18 registered HuBERT layers, three fold points, mean ratios, 95% fold-bootstrap intervals and the ratio=1 reference. The acoustic points repeat in base and FT panels; they are the same acoustic fits, not additional specifications. Vertical scales differ between datasets, so compare axis values rather than apparent error-bar heights.

## Calculation and checks

For each participant-held-out fold, the model fits two folds and scores both training and test observations without refitting. Training-derived predictor scaling stays fixed. Both predictions set random effects to zero (`re.form=NA`). X21/B23 fixed talker effects remain part of the fixed predictor where specified.

For probabilities p and word-response counts c (correct) and e (incorrect):

```text
mean log loss = -sum(c*log(p) + e*log(1-p)) / sum(c+e)
ratio = mean test log loss / mean training log loss
```

Grouped B23 rows therefore count once per word response, rather than once per sentence row. This is a response-level prediction score, not the GLMM's integrated fitted `logLik()`. It follows the proposal that Florian accepted. It is also numerically different from a ratio of gains relative to a baseline.

Each figure uses `M_predictor`. This batch fixes tau=2 and uses negative training-standardized distance. It does not diagnose newly optimized tau/k models or reproduce the old exponential-predictor test-refit z procedure.

The independent saved-score check covered nine SBI sources, 114 unique feature specifications, 1,368 train/test pairs across four models and 456 three-fold summaries. The plotted model contributes 342 unique fold ratios and 114 summaries. Source-score and model-input hashes match the recorded inventory. All ratios reproduce from the saved losses, all losses reproduce from totals divided by word-response counts, and train/test metadata agree on formulas and scaling. No saved selected-fit warnings or invalid pairs were found. This verifies saved outputs and source code; it does not regenerate probabilities or refit models.

For each feature, the interval is the 2.5th–97.5th percentile of mean ratios over all 27 ordered bootstrap samples of the three folds. All 114 intervals contain one. With only three folds and overlapping training sets, these intervals are descriptive and do not constitute a significance test.

## Results

The ranges below cover the 18 neural layers in each of the two variants. Acoustic points are excluded from these ranges.

| Dataset | Range of three-fold mean ratios | Relative test-loss difference implied by those means | Range across individual fold ratios |
| --- | ---: | ---: | ---: |
| AN19 | 0.999615–1.000893 | -0.039% to +0.089% | 0.988690–1.011749 |
| X21 | 1.005850–1.006788 | +0.585% to +0.679% | 0.868877–1.094280 |
| B23 | 1.000519–1.000690 | +0.052% to +0.069% | 0.987107–1.022017 |

Percentages are `100*(ratio-1)` and describe relative loss differences, not accuracy changes or the percentage of human performance achieved. These are arithmetic means of fold ratios, not ratios of pooled losses. The curves show little layer variation in this diagnostic, particularly for X21 and B23. That is a descriptive observation, not a tested claim of equivalent overfitting across layers.

### Why X21 needs a fold-level explanation

For X21 base Tr-24, the individual ratios are:

| Held-out fold, stored index | M_null | M_condition | M_predictor | M_joint | Observed test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.874154 | 0.877108 | 0.869632 | 0.874795 | 85.42% |
| 1 | 1.092370 | 1.096869 | 1.093561 | 1.098009 | 82.57% |
| 2 | 1.051696 | 1.051139 | 1.055301 | 1.052236 | 83.06% |

For `M_predictor`, fold 0 has test loss about 13.0% below training loss, while folds 1 and 2 have test loss about 9.4% and 5.5% above training loss. Their mean is 1.006165. The same direction and broadly similar sizes occur without SBI in `M_null` and `M_condition`. X21's `M_null` includes its specified fixed talker effects and random intercepts in fitting; it excludes the SBI predictor.

This shared pattern, together with the different observed accuracies, is consistent with fold composition contributing substantially to the diagnostic. Accuracy alone does not determine log loss, and these observations do not quantify how much of the gap comes from composition versus model estimation. The three folds contain 107, 107 and 106 participants. There is no basis here for changing the split to make the ratios look better.

## Suggested SI wording

We assessed training-to-test differences using the ratio of held-out to training mean log loss, normalized by the number of word responses. Each fold used the same training-fitted model and predictor scaling for both scores, with random effects set to zero for prediction. Across layers, mean ratios were close to one. Individual X21 folds showed larger, opposing differences that also occurred in reference models without SBI. We therefore report the individual folds alongside their means and descriptive 95% fold-bootstrap intervals. These diagnostics do not establish absence of overfitting or independently validate configurations selected using the same cross-validation scores.

## Sources and reproduction

- [Approved email exchange](../../analysis_update_2026-09-17/CORRESPONDENCE_2026-09-15.md).
- [Saved paired scores](../../analysis_update_2026-09-17/si_diagnostics/tables/paired_fold_ratios.csv) and [three-fold summaries](../../analysis_update_2026-09-17/si_diagnostics/tables/three_fold_ratio_summary.csv).
- [Scoring code](../../R/fit_confirmatory.R), especially `score_split` and the fold loop, and [plotting code](../../scripts/build_train_test_ratio_figures.py).
- [Verification record](verification.json) and [read-only verification script](verify_saved_scores.py).

From the repository root:

```text
python cross_talker_generalization/analysis_update_2026-09-18/sbi_diagnostic_review/verify_saved_scores.py
```

This writes only the verification record. It reads local paths recorded in the existing source inventory and does not change the source scores, figures or models.
