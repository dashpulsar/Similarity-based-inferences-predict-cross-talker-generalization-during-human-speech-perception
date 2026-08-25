# To-do

This file tracks remediable gaps in the current production analysis. Intrinsic interpretation boundaries are documented separately in [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md) and [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).

## Priority 1: align model selection with the intended theoretical test

- Record the maximized log-likelihood (and, for auditability, deviance, AIC, observation count, and convergence status) for every full-data predictor-only GLMM, `M_predictor`.
- Define and document the exact comparison rule for SBI and HVE representations and parameterizations using `M_predictor`, without the original condition predictor. Confirm whether the intended score is full-data log-likelihood, summed held-out log-likelihood, or a nested-cross-validation analogue before rerunning selection.
- Rerun feature-space and parameter comparisons using that rule. Do not use condition-only versus joint OOF log-loss gain as the optimization target.
- Retain condition-only versus joint likelihood-ratio tests and participant-held-out log-loss differences as a separate analysis of information beyond experimental condition.
- Revise “best layer” labels, figures, tables, and report prose after the intended selection analysis is complete.

## Priority 2: integrate actual B23 multi-talker exposure

- Add the public B23 exposure sources from OSF project `T83XK` (DOI `10.17605/OSF.IO/T83XK`), especially `BBP-2023-StimLists.xlsx` and `BBP-2023-TrainingData.xlsx`, to the documented data-ingestion workflow.
- Build B23 exposure pools from the stimulus filename actually presented, not only from the nominal `speaker` column. The stimulus lists contain a Spanish recording of “THE TABLE HAS THREE LEGS” in conditions where the nominal row labels it as Turkish; this must be reconciled with the paper's presentation-error note and verified before analysis.
- Validate that every training condition contains 60 presentations and that every sentence maps to one recording within a condition. Record actual talker counts after resolving the filename/label discrepancy.
- Recompute all mathematically defined B23 HVE measures for both single- and multi-talker exposure conditions, rebuild the B23 GLMM inputs, and regenerate affected figures and source tables.
- Replace the current `blocked` states only after the imported mapping and new outputs pass project-contract tests.

## Priority 3: develop model comparisons

- We already have comparisons of GLMMs with the only the theoretical predictors (plus random effects) against joint GLMMs that also contain the condition predictors. These comparisons assess whether the theoretical predictors can explain variance in human behavior that goes beyond that explained by the experiment's conditions. We also need comparisons of the joint GLMMs against GLMMs with only the condition predictors. These comparisons assess whether the theoretical predictors completely subsume / explain all of the variance of the experimental conditions.
- Similarly, we need a GLMM with theoretical SBI and HVE predictors from layer 24 + the two control predictors (MFCC, STRF), and we need to conduct backward-model selection on this model. The resulting model will tell us which predictors explain variance not explained by the other predictors.
- We can apply the same backward-model selection procedure to GLMMs that contain SBI OR HVE predictors from all layers. The resulting model will tell us whether different layers capture different information about human behavior.

## Priority 4: statistical uncertainty and selection safeguards

- Add participant-cluster bootstrap intervals for held-out log-loss differences and other key predictive summaries.
- Prespecify the layer-selection strategy or implement nested participant-level cross-validation when a selected layer is used for confirmatory reporting.
- Complete the planned full-dimensional HuBERT and DTW-normalization analyses for the key conclusions and report them alongside the established 3-D t-SNE results.

## Priority 5: documentation and release maintenance

- After priorities 1–2 are rerun, rebuild the final report into a new dated directory and update the authoritative-report pointer only after numerical and visual review.
- Remove transitional compatibility files from `results/` when the report builder no longer depends on them.
- Keep scientific rationale in `PROJECT_DESCRIPTION.md`, implementation details in `TECHNICAL_DOCUMENTATION.md`, commands in the runbook, and file locations in `FILE_GUIDE.md`; avoid restating the same material across documents.
- Before consolidating remaining README/project-description duplication, propose a small section-movement diff for collaborator review; do not broadly rewrite reviewed prose.

