# Analysis progress report

September 18 handoff: further parameter-search research is paused at Zhengyang's request. The two maps are collected in a [two-slide PPTX with linked offline HTML](../../../outputs/sbi_parameter_maps_2026-09-18/presentation/README.md). See remaining Florian requests for the next priorities. HVE remains paused; no new analysis was launched for the deck.

Latest September 18 sensitivity: [X21 with Talker-specific excluded](../sbi/parameter_maps/x21_without_talker_specific/README.md) is complete. Re-standardized and refitted 1,276 candidates on 8,290 responses / 161 participants; all converged, 6.31 minutes with 16 workers. Static and interactive maps plus actual selected-predictor tables are available. Metric-domain best: tau=1, k=0.017914, signed training z=4.092719; both objectives select this pair. The full-condition run is preserved. This remains one training split, not a new held-out evaluation.

September 18 follow-up: the expanded **X21** parameter landscape is complete: 1,247 training-only tau/k grid fits plus 29 small-k limit fits, all converged in 15.83 minutes. See the [figures and interpretation](../sbi/parameter_maps/x21_all_conditions/README.md). Both objectives select the same grid point; the tau maximum remains at the lower search boundary. This covers one layer/split, with no held-out scoring or optimizer-superiority claim. The interrupted AN19 grid is retained and excluded from analysis.

Latest September 18: [joint SBI tau/k feasibility test](../sbi/optimizer_probe/README.md) completed in 101 seconds, with 39 successful distinct fits. HVE is paused at Zhengyang's request. The larger optimizer benchmark is deferred; current evidence is one layer/split/seed with boundary-limited results.

September 18 continuation: the direct behavioral ceiling's condition/context correction has now been computed and checked for all three datasets. See the [new update](../sbi/README.md). Signed z is confirmed; matched-response z-ceilings and the optimization comparison remain open. The September 17 batch record below is retained.

Updated September 17, 2026, after the 16:06 validation (Asia/Shanghai). Both HVE batches have finished. Their saved results were assembled, checked and plotted without launching additional model fits. The earlier 15:32 handoff remains recorded in the daily work log.

## Summary

The approved training/test diagnostic is complete for this batch: **six SBI and twelve HVE layer profiles**, covering **396 unique specifications** in 25 runs. All **9,504 split scores** and **4,752 paired-fold ratios** passed the combined coverage/arithmetic checks; no invalid ratios or CV fit warnings were recorded. The AN19 acoustic-component correction and common-random-structure sensitivity are also complete. The matched z-versus-likelihood optimization experiment, optimization stability, compatible current-definition ceilings and several manuscript panels remain unfinished. The overall meeting TODO is therefore still substantially open.

The [daily work log](WORK_LOG_2026-09-17.md) records today's changes, runs, checks and caveats.

## 1. Completed and available

### Approved SI diagnostic

The diagnostic uses `mean test log loss / mean training log loss`. Both losses are normalized per word response. Each fold uses the same training-fitted model and frozen predictor scaling for both splits, with random effects set to zero in both predictions.

- Three datasets, base and ASR-FT, all 18 registered HuBERT layers, plus MFCC39/STRF24: **114 unique specifications**.
- **Six layer-wise figures**, in PNG/PDF/SVG, and **1,368 paired-fold ratios** across all four models. No invalid ratios or fit warnings in these SBI ratio tables.
- Each figure shows `M_predictor`, the three folds, their mean, a descriptive 95% fold-bootstrap interval and the ratio=1 reference line. Primary SBI z/ceiling figures were preserved.
- Repeated X21/B23 Tr-24 runs reproduce the earlier scores and formulas exactly.

Read the [SI figure guide](si_diagnostics/README.md), [figure inventory](si_diagnostics/tables/figure_inventory.csv) and [paired-fold table](si_diagnostics/tables/paired_fold_ratios.csv).

| Dataset | Range of mean ratios over base/FT neural layers |
| --- | ---: |
| AN19 | 0.999615–1.000893 |
| X21 | 1.005850–1.006788 |
| B23 | 1.000519–1.000690 |

These means are close to 1, with larger variation between individual folds. They do not establish that selecting the best layer using the same CV scores is unbiased. Nested selection has not been implemented in these runs.

### AN19 acoustic baselines

The acoustic-component configuration omitted `acoustic_subset`, causing components to use unstandardized coordinates. I added the explicit mapping and reran all 14 components using subsets of the full baseline's coordinate means and standard deviations. Full MFCC39/STRF24 distances reproduced within `1.78e-15`; their scores were unchanged. The correction therefore does not explain away the strong full-baseline results.

The [component figure](../acoustic_baselines/components/matched_acoustic_components.pdf) shows current training-fold z and held-out loss. Its z values are not normalized using a historical test-refit ceiling.

A separate common-structure sensitivity fixed participant/item random intercepts across 16 acoustic features and six condition-specific baseline analyses. **352 selected fits completed with no failures**; all-condition feature rankings were unchanged for both mean training z and pooled test loss. Twenty singular fits remained in two conditions, caused by near-zero participant variance; item variance remained, with no additional convergence messages. Four unflagged conditions retained prediction improvements for both baselines. See the [comparison summary](../acoustic_baselines/common_structure/comparison_summary.csv) and [flagged variance components](../acoustic_baselines/common_structure/flagged_variance_components.csv).

### Source and calculation checks

- Historical parameter searches, penalties, seeds and target refits are identified in the [optimizer audit](../sbi/OPTIMIZER_AUDIT.md). The current fixed-parameter prediction run does not complete the requested optimization comparison.
- **814 input-identity checks passed**. **56 Python checks** passed across the main and real-data suites, plus two R integration checks. Existing dependency/locale warnings were retained; no packages were installed.
- All eight HVE Tr-24 runs have now been compared with the September 6 results; seven reproduced within the strict tolerance. X21-FT `within_type_phoneme` retained identical responses, folds, formulas and fit flags, but differed by up to `3.60e-7` in mean loss; predictor serialization differed by up to `5.68e-14`. The strict historical-repeatability check remains explicitly flagged, separately from the passing new-batch arithmetic checks. No rerun was used to force agreement. See the [comparison record](source_validation/hve_repeated_tr24_score_validation.json).
- The [figure-input audit](source_validation/FIGURE_INPUT_AUDIT.md) confirms **20 ms Tr-24 frame stride**, a **25 ms convolutional frontend receptive field**, and **138 shared words / 42 talkers / 861 pairs** for the current AN19 matrix. Correcting the schematic's approximate frame positions and caption remains pending.

## 2. HVE results completed

All **282 unique HVE specifications** are complete: 78 available Tr-24 definitions across dataset/variant combinations, plus the two overall definitions at the other 17 layers (204). This produces **12 full layer profiles**, with the other 66 dataset/variant/method profiles retained as Tr-24 tables. It does not cover every HVE definition at every layer.

| Batch | Completed specifications | Fully merged runs |
| --- | ---: | --- |
| AN19 Tr-24 | 14/14 | Base and FT |
| X21 Tr-24 | 34/34 | Base and FT |
| B23 Tr-24, 97-participant order sample | 2/2 | Base and FT |
| B23 Tr-24, 168-participant exposure sample | 28/28 | Base and FT |
| AN19 overall definitions, other 17 layers | 68/68 | Base and FT |
| X21 overall definitions, other 17 layers | 68/68 | Base and FT |
| B23 overall definitions, other 17 layers | 68/68 | All four sample/variant runs |

All 16 HVE run directories have merged scores, metrics and diagnostics. The [combined validation](source_validation/final_validation.json) confirms complete four-model/three-fold/two-split coverage, input/output hashes, response-normalized scores and agreement with fold-level and pooled CV metrics. The combined package contains **18 figures**, each exported as PNG/PDF/SVG. Representative HVE plots from all three datasets were visually inspected for labels, intervals and clipping.

The 97- and 168-participant B23 samples remain separate throughout. The ranges below describe mean ratios across both variants and 18 layers for each stated definition; they are not comparisons on matched B23 samples.

| Dataset/sample | HVE definition | Range of mean test/train loss ratios |
| --- | --- | ---: |
| AN19, 120 participants | Overall | 0.999881–1.000827 |
| AN19, 120 participants | Overall order-sensitive | 0.999951–1.001027 |
| X21, 320 participants | Overall | 1.006479–1.010118 |
| X21, 320 participants | Overall order-sensitive | 1.006052–1.009013 |
| B23, 168 participants | Overall | 1.000557–1.003507 |
| B23, 97 participants | Overall order-sensitive | 1.001859–1.006771 |

The means are close to 1, while individual folds vary more. This diagnostic does not replace predictor-effect analyses or independently validate selected winners. The 20 boundary warnings in the separate AN19 within-condition sensitivity remain in that analysis; they are not part of these unflagged CV ratio tables.

## 3. Major work still open

1. Compare z-based and likelihood-based predictor-parameter optimization on matched observations, folds and model structures; check search stability and representative parameter surfaces.
2. Decide on nested CV for data-selected layers/HVE methods. Florian's ratio approval did not answer that question.
3. Extend HVE diagnostics to other methods across all layers if needed; the declared 282-specification batch and combined figures are complete. Keep the small X21-FT historical-repeatability difference documented.
4. Match current-definition z and behavioral ceilings, including condition in the reference keys; harmonize AN19 SBI/HVE response sets and random structures.
5. Finish the remaining manuscript work: schematic timing/layout, phonological-reference source checks, listener-response phone errors/Figure 2d, and agreed layer/matrix comparisons. Some items still require collaborator choices or missing source materials.

## 4. Reproduce the completed summary

The following commands were run successfully on the completed outputs. They assemble and check saved results and generate figures; they do not fit new models. Run from the repository root:

```text
python cross_talker_generalization/analysis/diagnostics/source_validation/assemble_diagnostic_inputs.py --assemble
python cross_talker_generalization/scripts/build_train_test_ratio_figures.py --manifest cross_talker_generalization/analysis/diagnostics/diagnostic_inputs.json --output cross_talker_generalization/analysis/diagnostics/si_diagnostics
python cross_talker_generalization/analysis/diagnostics/source_validation/assemble_diagnostic_inputs.py --validate
python cross_talker_generalization/analysis/diagnostics/source_validation/verify_hve_repeated_tr24_scores.py
```

Assembly rejects missing merged runs and verifies input hashes, full scoring grids and agreement with CV metrics. The verified final scope is 25 runs, 396 SBI/HVE specifications, 4,752 paired-fold ratios and 18 layer profiles. Per-profile ranges are saved in [profile_summary.csv](source_validation/profile_summary.csv). The separate historical-repeatability checker records the X21-FT difference described above.

All work is local. Existing results and feature files were preserved; no Git commit/push, collaborator message, feature extraction or t-SNE recomputation was performed.
