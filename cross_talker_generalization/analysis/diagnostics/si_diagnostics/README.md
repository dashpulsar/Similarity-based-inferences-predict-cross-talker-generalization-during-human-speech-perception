# Training/test diagnostics for the SI

Completed September 17: **18 layer profiles (6 SBI + 12 HVE)**, 396 specifications and 4,752 paired-fold ratios. Combined coverage/arithmetic validation passed, with no invalid ratios or CV fit warnings. The separate historical-repeatability check retains one small X21-FT difference; see the [progress report](../PROGRESS_REPORT.md).

The y-axis uses the ratio approved in the [September 15 email exchange](../CORRESPONDENCE_2026-09-15.md):

```text
ratio = mean test log loss / mean training log loss
```

Each loss is divided by its own number of word responses. The same model, fitted to two participant folds, scores those training observations and the remaining test fold. Predictor scaling is estimated on training data and frozen. Random effects are set to zero for both predictions (`re.form=NA`). The diagnostic uses response-level Bernoulli loss, including correct/incorrect counts for grouped B23 observations, without a binomial coefficient.

## Reading the figures

The x-axis follows the registered feature-space order. SBI panels place MFCC39 and STRF24 first, followed by the 18 HuBERT layers. HVE profiles display each definition separately. Base and ASR-fine-tuned representations have separate panels.

Gray points are the three paired-fold ratios. Black points show their arithmetic mean; intervals are the 2.5th and 97.5th percentiles of all 27 ordered bootstrap resamples of those three ratios. The dashed line at 1 indicates equal loss; values above it indicate higher test loss. These intervals summarize variation across the three folds, whose training samples overlap. They are not a formal test of overfitting. The legend sits outside the data area.

Plots show the theoretical-predictor model, `M_predictor`. Tables retain `M_null`, `M_condition`, `M_predictor`, and `M_joint`. Orange markers identify retained fit warnings, if any. Invalid or incomplete score pairs remain in the tables with reasons and receive no complete three-fold summary. A near-one ratio does not establish unbiased evaluation of a layer or method selected using those same CV scores.

## Files and scope

- [Figures](figures): PNG, PDF and SVG versions.
- [Figure inventory](tables/figure_inventory.csv): exact plotted versus table-only coverage.
- [Paired-fold scores and ratios](tables/paired_fold_ratios.csv) and [three-fold summaries](tables/three_fold_ratio_summary.csv).
- [Source inventory](tables/source_inventory.csv) and [build record](tables/build_record.json): input identities, score hashes, conventions and warning counts.
- [Daily work log](../WORK_LOG_2026-09-17.md): actual batch completion and remaining work.

The complete SBI batch covers all three datasets, both HuBERT variants and both acoustic baselines. The declared HVE batch covers all available Tr-24 definitions plus the two overall definitions at every registered layer. B23 global order-sensitive HVE uses 97 participants; its other available definitions use 168. These strata stay separate. Other HVE definitions at Tr-24 remain table-only; the package does not imply complete all-method/all-layer coverage.

These diagnostics retain fixed existing predictors and participant folds. They do not retune k/tau, add nested CV, replace the SBI z/ceiling figures, or substitute for the pending z-versus-likelihood optimization comparison. The [scientific specification](../../../docs/SCIENTIFIC_SPEC.md) records those separate analyses.
