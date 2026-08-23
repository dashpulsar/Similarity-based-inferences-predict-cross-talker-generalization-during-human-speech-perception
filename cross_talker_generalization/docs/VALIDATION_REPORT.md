# Validation report

This report describes checks that were actually executed; it does not label unrun matrices as complete.

## Input contract

- 84 audit checks passed and none failed. The only warnings were harmless `__staging__` roots in the two B23 full-dimensional HDF5 files.
- Behavior: AN19 24,960 rows / 160 participants; X21 16,477 / 320; B23 11,700 / 195.
- HDF5 inventory: AN19 6,261 units; X21 660; B23 480. HuBERT stores contain all 18 registered layers.
- Deterministic physical pairs: AN19 5,459; X21 4,532; B23 240.

## Numerical regression against existing derived results

- All 4,532 X21 HuBERT-base 3-D t-SNE `tr_24` raw distances matched the existing table exactly: maximum absolute difference 0.
- All 3,296 X21 condition-item aggregates matched within floating-point summation order: maximum absolute difference `1.07e-14`.
- All six defined AN19 `tr_24` HVE measures matched existing output pool by pool.
- All 16 X21 `tr_24` presentation-weighted HVE measures matched to `1e-9` rounding.
- All defined HVE values for the four currently implemented B23 `tr_24` single-talker pools matched to `1e-9`. This check predates ingestion of the public B23 multi-talker stimulus lists and does not validate multi-talker HVE.

## Statistical smoke tests

These tests validate the currently implemented condition-only versus joint analysis. They do not validate the planned predictor-only GLMM likelihood as a feature-selection criterion, because the current model outputs do not yet record or use that criterion.

For X21 HuBERT-base 3-D t-SNE `tr_24`, the confirmatory joint model produced coefficient `0.0962`, SE `0.0473`, Wald z `2.032`, p `.0421`, and 95% CI `[0.0034, 0.1890]`. The condition-only versus joint LRT was chi-square(1) `3.859`, p `.0495`. OOF mean binomial log loss was `0.487218` for condition-only and `0.482370` for joint, a gain of `0.004848`. Fits converged and were non-singular; the four X21 test talkers were fixed blocking factors.

The X21 `tr_24` overall-HVE smoke test produced joint coefficient `-0.0599`, z `-0.898`, p `.369`, LRT p `.374`, and joint OOF log loss `0.487761`; it did not improve on condition-only. This is a pipeline check, not a final scientific conclusion.

The compatibility ceiling smoke test yielded three finite X21 held-out-refit z values: `20.90`, `23.72`, and `21.53`. All three fits converged and were non-singular. AN19 produced `22.91`, `22.00`, and `21.49`; fold 0 retained the historical singular random-slope fit and recorded that diagnostic. B23 produced `4.73`, `6.08`, and `5.97`, all converged and non-singular.

## Figure and matrix QA

- S-curves include PNG, SVG, and full source CSV. Constant Talker-specific predictors are labeled as self-comparisons instead of receiving a fabricated curve.
- GLMM profiles show coefficient/CI, Wald z, LRT, and frozen-model OOF log-loss gain.
- The X21 `tr_22` / `tr_24` raw-distance correlation smoke test used all 4,532 physical pairs and produced long, matrix, complete-case CSV, PNG, and SVG outputs.

## Production release check

On 2026-08-21, `python -m unittest discover -s cross_talker_generalization/tests -v` passed 15/15 tests. The unified final-report builder completed and its report-level matrix invariants and SHA-256 audit passed.
