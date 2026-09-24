# September 24 checkpoint before repository cleanup

This checkpoint preserves the accumulated September code, figures, and compact result tables before a separate cleanup. No experiment is deleted, no statistical definition is changed, and no full analysis is rerun for this publication step. Older dated results retain their original meaning and limitations.

## Start here

| Content | Files | Scope |
| --- | --- | --- |
| X21 parameter maps | [Presentation folder and two offline interactive maps](../../outputs/sbi_parameter_maps_2026-09-18/presentation/README.md) | One HuBERT layer and one training split, not a completed cross-fold optimization comparison. |
| Full-condition X21 analysis | [Methods and results](../analysis_update_2026-09-18/x21_parameter_landscape/README.md) | Training z and likelihood surfaces. |
| X21 excluding Talker-specific | [Methods and results](../analysis_update_2026-09-18/x21_parameter_landscape_no_talker_specific/README.md) | Separate response sample; do not directly compare raw score magnitudes between samples. |
| Training/test diagnostics | [Six SBI plots and scoring details](../analysis_update_2026-09-18/sbi_diagnostic_review/README.md) | Per-response test/train mean log loss; existing fixed-distance analyses. |
| AN19 acoustic-baseline investigation | [Evidence and open questions](../analysis_update_2026-09-18/AN19_BASELINE_EXPLANATION.md) | Descriptive checks provide clues, not a definitive causal explanation. |
| Cross-fitted model comparisons | [September 1 results](../analysis_update_2026-09-01/README.md) | Pooled GLMM comparisons with the limitations recorded there. |
| Figure and phoneme work | [September 9 report index](../analysis_update_2026-09-09/README.md) | Automatic annotation and manuscript-panel drafts, with coverage caveats. |
| Existing progress record | [September 17](../analysis_update_2026-09-17/PROGRESS_REPORT.md), [September 18](../analysis_update_2026-09-18/README.md) | Dated records; not evidence that all open tasks are complete. |

## Publication boundary

Included: analysis source, configurations, checks, technical reports, figures, presentations, and compact aggregate results. September updates are retained together at this checkpoint rather than selecting experiments for deletion now.

Kept locally and excluded from new Git additions: HDF5 features, feature archives, recordings, model/runtime dependencies, duplicated worker outputs, response-level model inputs/predictions, intermediate grid preparation, original meeting transcripts, email drafts, and correspondence records. Some dated documents still refer to these local resources; those references do not mean that the resources are included in a clone. Existing tracked public data are unchanged. Recomputing analyses still requires the separately distributed inputs.

Private correspondence is not reproduced here. Its methodological proposals are summarized below as future work, not implemented features.

## Follow-up from the recent methodological discussion

1. Compare a distance model that varies tau with an exponential-similarity model that fixes tau at 2 and varies k.
2. Compare likelihood and signed-z selection, always obtaining either objective from a fitted predictor-only GLMM.
3. Specify and document any regularization and training/validation/test protocol before implementing it. Parameter tuning, regularization, and held-out evaluation have distinct roles.
4. Distinguish frozen-model prediction, test-set GLMM refitting with theoretical parameters fixed, and the proposed fixed-effect offset with test-set random effects estimated. These answer different evaluation questions and must be labeled separately.

This checkpoint does not implement those new alternatives or resolve outstanding methodological choices. HVE follow-up remains paused. The next cleanup should identify the retained analyses and dependencies before removing obsolete experiments.

## Validation

Checks run on September 24:

- Python test suite: **59 passed**. Three environment/deprecation warnings remain (numexpr, bottleneck, and Jupyter paths).
- R landscape scaling checks: **12 cases passed**, including small-exponent cancellation and underflow cases.
- R training/test scoring checks: **24 split scores passed**, including fixed training scaling, frozen-model scoring, and grouped-response weighting.
- R random-structure checks: default and explicit registered policies agree; the declared participant/item policy and variance exports pass. Expected synthetic boundary-fit warnings, locale warnings, and a Matrix build-version warning were emitted.

No full feature extraction, t-SNE, parameter search, or corpus-level GLMM rerun was performed. Tests check implementation behavior; they do not establish scientific validity or reproduce all previous experimental results.

The pre-update remote `main` was `a2d9272f0f72cfcf6326714c8acd3bf297ede60b`, verified against GitHub on September 24. The checkpoint is assembled from the existing `main` worktree, not the older detached worktree.

The staged inventory contains 1,439 added/modified files (approximately 207 MiB uncompressed); the largest individual file is 7.33 MiB. No tracked file is deleted. The publication checks found no newly included HDF5/archive/audio/video files, dependency directories, email files, or CSV tables with the checked participant-ID column names. A targeted credential-pattern scan found no matches; this is not an exhaustive privacy audit. All links in this checkpoint index resolve locally to included files. Older dated reports can still contain machine-specific paths or links to deliberately excluded inputs; consolidating those reports belongs to the subsequent cleanup.
