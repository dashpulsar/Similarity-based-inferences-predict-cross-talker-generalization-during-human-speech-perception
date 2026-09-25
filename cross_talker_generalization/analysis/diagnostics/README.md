# Training/test diagnostics

**Completed diagnostic batch:** the [progress report](PROGRESS_REPORT.md) now includes the post-run validation: 396 SBI/HVE specifications, 4,752 paired-fold ratios and 18 layer figures (6 SBI + 12 HVE). HVE fitting and figure generation are complete. Other scientific TODO items remain open; the earlier 15:32 snapshot is retained in the daily work log.

It preserves previous results and the existing CV design. The SI diagnostic uses the ratio of test to training mean log loss. The principal SBI z/ceiling figures remain unchanged. Today's changes, commands, checks and remaining tasks are recorded in the [daily work log](WORK_LOG_2026-09-17.md) for the later report.

## Completed in this update

1. **AN19 acoustic scaling correction.** All 14 components were rerun with exact subsets of the full baseline coordinate means and standard deviations. The old components used `none` because the configuration omitted `acoustic_subset`; that mapping now explicitly uses `global_z` in the main and sensitivity profiles. Full MFCC39/STRF24 distances agree with retained values within `1.78e-15`, and their z and prediction scores are unchanged. All 256 selected component/full-baseline fits completed without fit, singularity or convergence flags.
2. **Matched training/test scoring.** The first nine runs covered 46 feature specifications and produced 1,104 split-score rows and 736 selected fits, with no fit, singularity or convergence flags. Both splits use the same training fit, training scaling and fixed-effect prediction convention. All newly exported test scores match the existing held-out metric columns. The fitted-model `logLik()` remains a separate output. The continuation adds all remaining SBI layers and HVE diagnostics; see the coverage section below.
3. **Optimizer identification.** The [optimizer audit](../sbi/OPTIMIZER_AUDIT.md) maps historical Optuna and SciPy searches to their concrete source code. Several historical searches used penalized z objectives and refitted the target-sample GLMM. The current frozen-model prediction runner fixes tau and does not search k. The audit completes source identification while leaving the matched objective comparison and search-stability experiments open.
4. **Within-condition AN19 check.** MFCC39/STRF24 were fitted separately in all six exposure/test conditions, retaining the original participant folds. Both features improve pooled held-out prediction in the four conditions with unflagged fits. Two conditions have singular fits and need model review. The full acoustic signal therefore includes within-condition information under the current protocol; its complete explanation remains open.
5. **Common-random-structure sensitivity.** All 16 acoustic features and the six condition-specific baseline analyses were rerun with fixed participant/item random intercepts: 352 selected fits, no failures. All-condition feature rankings are unchanged for both mean training z and pooled held-out loss. Twenty boundary-singular fits remain, with near-zero participant variance and retained item variance. No additional convergence messages occurred. Primary fits and their hashes are preserved.

## Results to inspect

- [Corrected acoustic-component figure](../acoustic_baselines/components/matched_acoustic_components.pdf): the full baselines followed by 14 components, with training-fold z and frozen-model test loss. Gray dots show folds, blue points/intervals show their mean and 95% fold-bootstrap interval, and orange crosses mark the earlier unscaled component means. The dotted z=1.96 line is an unadjusted Wald reference. These training z values use the current distance predictor and have a different scope from historical test-refit z/ceiling figures.
- [AN19 base training/test profile](train_test/figures/AN19-base_train_test_loss.pdf) and [ASR-FT profile](train_test/figures/AN19-ft_train_test_loss.pdf): direct matched loss plots, with MFCC/STRF first. Training squares/dashed lines and test circles/solid lines are offset slightly horizontally for readability. They share a vertical scale.
- [Within-condition summary](../acoustic_baselines/components/within_condition/summary.csv) and [flagged fits](../acoustic_baselines/components/within_condition/flagged_fits.csv).
- [Combined split scores](train_test/tables/matched_train_test_scores.csv), [validation checks](train_test/tables/validation.csv), and [model diagnostics](train_test/tables/fit_diagnostics.csv).
- [Approved SI ratio figures](si_diagnostics/figures), [paired-fold ratios](si_diagnostics/tables/paired_fold_ratios.csv), and [figure coverage](si_diagnostics/tables/figure_inventory.csv). These are likelihood diagnostics, separate from the SBI z/ceiling displays.
- [Common-structure comparisons](../acoustic_baselines/common_structure/comparison_summary.csv), [variance components for flagged fits](../acoustic_baselines/common_structure/flagged_variance_components.csv), and [run checks](../acoustic_baselines/common_structure/result_metadata.json).
- [Independent input validation](source_validation/input_identity_validation.json): 814 checks passed, including exact Tr-24 input agreement and response identity checks across features and HVE strata.

### What the within-condition check shows

Each cell contains 20 participants. Within a cell and fold, the null and predictor models use identical observations and random-effect structures. Positive values below mean that the predictor reduces held-out mean log loss relative to the null model.

| Exposure → test accent | MFCC39 loss reduction | STRF24 loss reduction |
| --- | ---: | ---: |
| Korean → Korean | 0.060235 | 0.089250 |
| Mixed → Korean | 0.041573 | 0.042113 |
| Mixed → Spanish | 0.045855 | 0.057043 |
| Spanish → Korean | 0.014165 | 0.026902 |

These four cells have no selected-fit flags. Korean → Spanish has a singular fitting scope in one fold; Spanish → Spanish has singular fits in the full-data fit and all folds. Across the separate condition analyses, 20 of 96 selected fits carry boundary-singularity messages. They are retained and flagged, not treated as clean positive findings. In Spanish → Spanish, MFCC39 increases pooled test loss by 0.017787. The summaries use response-weighted held-out losses, while training z describes the estimated association in two training folds.

The coordinate correction is complete. The original registered fits retain their feature-specific random-effect fallback. The separate common-structure sensitivity removes this difference across acoustic components and leaves their rankings unchanged (Spearman rank correlation 1 for both metrics). Each component still recomputes its DTW path, so these are feature-space comparisons rather than additive contributions to one fixed alignment. Small cell sizes and boundary variances remain relevant to interpretation. This sensitivity does not yet harmonize the AN19 SBI-versus-HVE comparison or all neural-layer structures.

## Matched training/test scores

For each feature and fold, all four registered models are fitted using two participant folds. That same fitted model predicts the training observations and the third fold with `re.form=NA`, so random effects are set to zero in both predictions. The training predictor mean and SD are reused unchanged. The diagnostic score is

```text
mean log loss = -sum(correct * log(p) + incorrect * log(1-p))
                / sum(correct + incorrect)
```

Probabilities are clipped to the existing numerical bounds of `1e-12` and `1-1e-12`. Lower loss is better. For B23, each data row summarizes 4–7 word responses; dividing by row count would give a different scale. Both splits now use word-response counts. The fitting formulas, fallback rules and held-out `cv_metrics.csv` schema are unchanged.

The initial batch covered both HuBERT versions at all 18 registered AN19 layers, Tr-24 for X21/B23, and both full acoustic baselines. The continuation has completed X21/B23 base/FT at all 18 layers. All-layer SBI input tables are read from the retained sibling worktree; their Tr-24 rows match the current September 6 input tables exactly. Each fit retains its input CSV and source hashes. The consolidated diagnostic manifest uses each feature once, omitting the superseded Tr-24-only duplicate runs.

The new diagnostic divides mean test loss by mean training loss within each fold, then plots the three ratios, their arithmetic mean and a 95% percentile interval over all 27 ordered bootstrap resamples. The horizontal line at 1 indicates equal loss; values above 1 indicate worse test loss. These intervals describe fold variability with overlapping training sets. They do not establish an independent population-level test of overfitting. The direct-loss figures remain available separately. Definitions represented only at Tr-24 remain in tables without redundant one-point layer plots.

AN19's registered random-effect fallback remains logged for every primary fit. Harmonizing its structure for the matched optimization/theory comparison remains open. A small train/test ratio does not validate reuse of these same CV scores for selecting a winning configuration. This batch evaluates fixed predictors; it does not select a layer, change k/tau, or add nested CV. Tr-24 is the previously specified summary layer. Data-selected best-layer/method summaries elsewhere retain their selection caveat.

## Diagnostic coverage

- SBI: three datasets × two HuBERT variants × 18 layers, plus two acoustic baselines per dataset: 114 unique specifications.
- HVE batch complete: all available Tr-24 definitions (78 specifications), plus `overall` and `overall_order_sensitive` across the other 17 layers (204). This declared subset supplies two full layer profiles per dataset/variant and Tr-24 tables for the other methods. It does not cover every HVE definition at every layer.
- AN19 HVE uses 120 participants; X21 uses 320. B23 global order-sensitive HVE uses 97 participants with known exposure order, and its other available definitions use 168. Their figures and summaries remain separate.

The figures show `M_predictor`; tables retain all four models. No acoustic SBI baseline is inserted into an HVE profile. Use the [source inventory](si_diagnostics/tables/source_inventory.csv) and [daily log](WORK_LOG_2026-09-17.md) for completed coverage and any warnings.

## Reproduce the scoring and figures

Use the project's Python/R environment and run from the repository root. To rerun one input table into a new output directory:

```powershell
$env:PYTHONPATH = "$PWD/cross_talker_generalization/src"
python -m ctg.cli fit-glmm-parallel `
  --input PATH_TO_MODEL_INPUT.csv --output NEW_MODEL_DIRECTORY `
  --jobs 8 --model-set all
```

The batch inputs and their checksums are listed in `train_test/run_inputs.json`. For each entry, run the command above with its `path` and an output directory named by its `run_label`. Keep that manifest alongside those directories, then build the summary:

```powershell
python cross_talker_generalization/scripts/build_train_test_diagnostics.py `
  --runs cross_talker_generalization/analysis/diagnostics/train_test
```

The summary builder checks paired splits, fold coverage, frozen scaling, response normalization and agreement with the existing held-out score columns. It saves combined tables, fit diagnostics and direct train/test plots. It rejects missing/failed split scores and flags singular or nonconverged predictor fits on the affected plots.

The acoustic preparation, fits and figure can be reproduced with `scripts/run_an19_matched_acoustic_audit.py` using its `prepare`, `fit`, and `report` stages and a new `--output` directory. The separate `scripts/run_an19_within_condition_audit.py` takes that preparation's `model_input.csv` via `--input` and a new `--output` directory. The first script defaults to eight workers; the condition runner uses four concurrent R workers. Existing model/pair outputs are read without changing the original feature files.

Numerical checks passed for coordinate subsets, shared standardizers, frozen-fit scoring, grouped-response weighting, and the three-fold interval calculation. The source code checks and the scientific fit diagnostics are separate: the within-condition boundary warnings remain explicit in the results despite the calculation checks passing.

To regenerate the approved ratio figures from the consolidated source list:

```powershell
python cross_talker_generalization/scripts/build_train_test_ratio_figures.py `
  --manifest cross_talker_generalization/analysis/diagnostics/diagnostic_inputs.json `
  --output cross_talker_generalization/analysis/diagnostics/si_diagnostics
```

`scripts/run_hve_train_test_diagnostics.py` prepares and scores the revised HVE inputs. `--global-profiles --exclude-tr24` selects the two overall definitions at the other 17 layers. Keep the prepared manifest and its participant strata. The optional `scripts/run_an19_common_structure.py --jobs 4` reproduces the acoustic sensitivity while preserving the original registered runs.

## Still open

- Obtain a decision on nested CV for selection analyses; the loss-ratio definition is approved.
- Full all-method/all-layer HVE coverage remains beyond the completed diagnostic batch. Preserve the small X21-FT historical-repeatability difference described in the progress report.
- Compare z-based and likelihood-based parameter searches on the same observations and model structures; examine optimization stability.
- Interpret the two boundary-singular condition analyses cautiously; harmonize AN19 neural-layer/theory comparisons separately. The acoustic common-structure sensitivity is complete. Run finer-component analyses only when warranted by these checks.
- Complete the other outstanding figure and analysis items in the shared checklist.

These runs did not replace manuscript figures.
