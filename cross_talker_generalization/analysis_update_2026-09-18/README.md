# September 18 analysis update

## Latest: Talker-specific exclusion sensitivity

The [separate X21 three-condition version](x21_parameter_landscape_no_talker_specific/README.md) is complete, with rotatable maps and actual selected-predictor tables. All 1,276 fits converged on 8,290 training responses / 161 participants in 6.31 minutes. Both objectives favor tau=1, k=0.017914 within the metric domain, with signed training z=4.092719. Source distances were reused unchanged, and candidate scaling and GLMM fits were recomputed. The full-condition results below remain intact; score magnitudes across the two response samples are not a direct performance comparison.

## Latest: X21 dense parameter landscape completed

The [X21 tau/k landscape](x21_parameter_landscape/README.md) is complete: 1,247 grid points plus 29 linear-limit fits, all converged, 15.83 minutes with eight R workers. Three-dimensional z/likelihood surfaces, heatmaps and profiles are available. Both objectives select tau=0.5, k=0.017914 in the expanded range, or tau=1, k=0.053183 when restricted to the metric domain. Both maxima are on the lower tau boundary. These are Tr-24 training-only results; the complete multi-fold/layer optimization comparison remains open.

## Earlier: joint SBI parameter-search feasibility

The [tau/k pilot](sbi_optimizer_probe/README.md) completed in 101 seconds on AN19 base Tr-24, using one training split. Grid, TPE and differential evolution each received twelve proposals per objective; 39 distinct GLMM fits converged. The grid's best evaluated pair lay at both lower bounds; no optimizer superiority is established. The larger multi-seed/layer benchmark was deferred on cost grounds. HVE work is paused at Zhengyang's request. The pilot uses training fitted likelihood and signed z; it does not replace existing results or complete the three-fold optimization analysis.

## Completed: conditional behavioral reference

The direct behavioral reference now keeps experimental condition separate for AN19 and X21. X21 also distinguishes the sentence context of repeated keywords. B23 already used conditional cells and retains its previous definition. All three datasets were recomputed with the existing participant folds and training-only Jeffreys-smoothed probabilities. No held-out responses enter their own prediction.

| Dataset | Scored word responses | Previous mean test log loss | Revised mean test log loss |
| --- | ---: | ---: | ---: |
| AN19 | 7,680 | 0.351929 | 0.383318 |
| X21 | 16,477 | 0.351961 | 0.379428 |
| B23 | 62,790 | 0.502830 | 0.502830 |

Every test row has a training-derived reference prediction. B23 probabilities reproduce within `1.11e-16`. X21 has 112 keyword/condition/talker cells containing two sentence contexts, out of 1,536 such cells; those contexts are now separate. The X21 comparison reflects both condition and sentence-context corrections and does not isolate their individual contributions.

AN19/X21 losses are higher under the revised definition. More detailed cells use fewer training observations, so better held-out performance is not guaranteed. The measured difference does not by itself identify the cause of the higher loss or a problem with the theoretical models. This human-response reference is not a mathematical maximum achievable score.

The [comparison figure](conditional_ceiling/conditional_ceiling_comparison.png) connects the old and revised loss within each fold. It is a calculation diagnostic, with separate y-axis ranges by dataset, not a replacement for the SBI z-value display. PNG, PDF and SVG versions are saved alongside the [numerical comparison](conditional_ceiling/comparison.csv), [coverage table](conditional_ceiling/coverage_by_condition_fold.csv), and per-dataset predictions and source records.

## Decisions retained

- Zhengyang confirmed **signed z** for the optimization comparison on September 18. No absolute-z substitution is made.
- Participant-level three-fold train–test CV remains unchanged; no inner tuning split was added.
- Previous result directories and z/ceiling figures remain intact.

## Still to complete

1. Check whether the X21 ridge and preferred parameter region persist across training splits. The dense single-split boundary check is complete; broader search stability and a multi-seed optimizer comparison remain open. Cross-theory SBI/HVE matching is deferred while HVE is paused.
2. Fit compatible current-definition z-ceilings and redraw the requested normalized z profiles. The direct likelihood reference above does not supply a Wald z.
3. Complete the signed-z versus likelihood parameter-optimization comparison across folds/layers; the feasibility run and X21 dense grid each cover one layer and training split.
4. Complete the remaining manuscript-panel tasks in the [project checklist](../../TODO.md).

The corrected reference currently uses all behavioral test-phase responses within the participant splits. It must not be applied automatically to models that use a smaller response sample. No historical normalized figure has been silently recalculated.

## Reproduce

From the repository root:

```text
python cross_talker_generalization/scripts/audit_condition_ceiling.py
```

This reads the September 6 reference predictions for comparison and writes the September 18 outputs. It does not fit GLMMs or access feature arrays. The [work log](WORK_LOG_2026-09-18.md) records the code changes and checks.

Validation: all 59 checks in the Python test directory passed, including six new ceiling checks. Existing dependency/deprecation warnings remain documented in the work log.
