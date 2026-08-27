# To-do

This file tracks work that can still change the production results. Scientific definitions are kept in [`cross_talker_generalization/docs/SCIENTIFIC_SPEC.md`](cross_talker_generalization/docs/SCIENTIFIC_SPEC.md).

## 1. Rerun predictor selection and model comparisons

- [x] Rank candidate SBI and HVE specifications by three-fold held-out total log loss from `M_predictor` (predictor plus random effects, without condition).
- [x] Require candidates in the same ranking to use identical held-out observations and trial counts.
- [x] Record full-data log likelihood, deviance, AIC, observation count, and convergence status for auditability.
- [x] Produce both comparisons after selection: `M_condition` versus `M_joint`, and `M_predictor` versus `M_joint`.
- [x] Correct SBI selection using the existing true OOF `M_predictor` scores and replace the old best-layer labels in the dated analysis update.
- [x] Rerun the complete revised HVE candidate sets for AN19, X21, and B23 and publish corrected best-method labels in the dated analysis update.
- [ ] Replace or retire the remaining old best-method labels in the broader August 21 report package.

## 2. Complete the revised HVE analysis

- [x] Add `overall_order_sensitive`, defined by concatenating complete exposure tokens in presentation order and including cross-token frame transitions.
- [x] Recover participant-level order for AN19 and X21.
- [x] Integrate the B23 public stimulus lists and training table, using the actual stimulus filename to resolve the documented speaker-label discrepancy.
- [x] Keep unordered B23 HVE available when trial indices are incomplete, while marking only `overall_order_sensitive` unavailable for those participants.
- [x] Run `overall_order_sensitive` across all 18 t-SNE layers for base and fine-tuned HuBERT in all three datasets.
- [x] Fit all 14 modelable B23 order-independent HVE measures across 18 layers and both HuBERT variants using predictor-only selection.
- [x] Fit both downstream comparisons for the two selected B23 order-independent predictors and add participant-cluster bootstrap intervals.
- [x] Recompute the revised AN19/X21 HVE candidates and regenerate the complete cross-method HVE figures.

## 3. Multivariable model analyses

- [ ] Prespecify the backward-selection removal rule and stopping criterion.
- [ ] At HuBERT layer 24, fit the combined SBI + HVE + MFCC + STRF model and perform the prespecified backward selection.
- [ ] Separately fit all-layer SBI and all-layer HVE models to test whether layers retain nonredundant information.

## 4. Uncertainty and sensitivity analyses

- [x] Add participant-cluster bootstrap intervals for both held-out comparisons in the global order-sensitive HVE analysis.
- [x] Extend the paired participant-cluster bootstrap to every selected HVE analysis.
- [x] Add the same paired participant-cluster bootstrap intervals to the selected SBI analyses.
- [ ] Treat data-driven layer selection as exploratory or evaluate it with nested participant-level cross-validation before making confirmatory claims.
- [ ] Run the planned full-dimensional HuBERT and DTW path-length-normalization sensitivity analyses for the key conclusions.

## 5. Revised report and release review

- [x] Build the dated methodological update `analysis_update_2026-08-27` without overwriting `final_report_2026-08-21`.
- [x] Review the update's numerical inventories, candidate diagnostics, selected-model diagnostics, source tables, labels, and figures.
- [x] Update the root documentation so the August 21 broad report is not presented as the authority for corrected selection results.
- [ ] Decide whether to promote the dated update or rebuild the broad report after the remaining multivariable and sensitivity-analysis decisions are resolved.
