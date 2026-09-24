# X21 joint tau/k parameter landscape

Completed September 18, 2026. All **1,247 parameter pairs and 29 linear-limit reference fits** converged without singular estimates. The main run took **949.89 seconds (15.83 minutes)** using eight R workers. Figures and validation were generated after the run.

## Figures

- [Interactive 3-D version](figures/parameter_landscape_interactive.html): open in a browser, drag to rotate and use the mouse wheel to zoom. Both views rotate together. Hover over a plotted grid point for its exact tau, k, signed z and log likelihood; the red point marks the expanded-domain maximum. The toolbar also supports panning, restoring the view and saving a PNG. Plotly and the aggregate data are embedded, so no server or online dependency is required.

- [Three-dimensional surfaces](figures/parameter_landscape_3d.png): tau and log10(k) are the horizontal coordinates; height is signed training Wald z (left) or fitted training GLMM log likelihood (right). Higher values indicate a better score under the respective objective.
- Separate [z surface](figures/landscape_3d_z.png) and [likelihood surface](figures/landscape_3d_log_likelihood.png).
- [Heatmaps and contours](figures/parameter_landscape_2d.png): red stars mark the best evaluated points; the vertical dashed line marks tau=1. Values to its left are a non-metric sensitivity analysis.
- [Parameter profiles](figures/parameter_profiles.png): selected tau slices and comparison against standardized negative distance, the small-k limit.
- [Preferred parameter pairs](figures/parameter_ridge.png) and [objective agreement](figures/objective_agreement.png).

Every figure also has an SVG with the same filename stem. Meshes and contours connect evaluated points; they do not establish the scores between those points.

## Results

Signed z and fitted likelihood select the **same evaluated pair** in each of the two search domains below.

| Search domain | Best tau | Best k | Signed training z | Fitted training log likelihood |
| --- | ---: | ---: | ---: | ---: |
| Expanded, 0.5 <= tau <= 4 | 0.5 | 0.0179141 | 4.634733 | -3891.877108 |
| Metric domain, 1 <= tau <= 4 | 1 | 0.0531830 | 4.597854 | -3892.030890 |
| Fixed tau=2, best evaluated k | 2 | 0.1098561 | 4.571308 | -3892.114814 |

Both domain-wide winners lie on the lower tau boundary. The expanded search therefore leaves the optimum outside its lower boundary unresolved. For a Minkowski metric, tau=1 is the admissible lower boundary; tau<1 need not satisfy the triangle inequality.

The preferred k increases as tau increases. Changing tau alters both relative distances and their numerical scale, and k partly compensates for that scale change. The surface contains an extended high-scoring ridge. With k optimized, tau=2 is only 0.02655 z units and 0.08392 fitted log-likelihood units below the best metric-domain grid point. These differences are descriptive, with no significance claim.

Small k approaches a plateau because standardized exp(-k*d) approaches standardized negative distance. At tau=2, the optimized exponential increases z from 4.333616 to 4.571308 and fitted log likelihood from -3893.213677 to -3892.114814 relative to that limit. Large k eventually reduces both scores in this grid.

The metric-domain region within 0.1 fitted log-likelihood units of the best point contains 19 evaluated pairs, spanning tau=1 to 2.25 and k=0.03700 to 0.10986. This is a descriptive score band, not a confidence interval or a rectangular region in which every pair is equally good.

## What was fitted

X21, HuBERT non-ASR-fine-tuned Tr-24, existing 3-D t-SNE features. Training participant folds 1 and 2 contain 10,969 word responses from 213 participants. Fold 0 was excluded from scoring and selection.

For each tau, keyword intervals and the current physical train/test pairing are used to recompute hard DTW with a rooted Minkowski frame cost and mean-sequence-length normalization. Distances are averaged within each source talker, then equally over source talkers; the predictor is exp(-k*d) of that aggregate distance. Each candidate is standardized over the same training responses.

The fixed structure is the registered X21 predictor model:

```r
cbind(response_correct, response_incorrect) ~
    similarity_z + test_talker_id +
    (1 | participant_id) + (1 | analysis_item_id)
```

Test talker is a fixed blocking factor. Experimental condition is not included. Each tau/k pair receives a fresh GLMM fit (lme4, binomial-logit, bobyqa, maxfun=20000, nAGQ=1). GLMM coefficients and variance parameters are thus fitted conditional on each pair. This maps the two SBI parameters jointly while retaining an inner GLMM fit.

Tau has 29 values from 0.5 to 4 at increments of 0.125. K has 41 log-spaced values from 1e-6 to 2 plus exact 0.001 and 0.05. The 29 additional reference fits use standardized negative distance; literal exp(0) would be constant and is not used. Numerical evaluation uses a shifted exponential/expm1 form that gives the same standardized predictor while avoiding underflow and cancellation.

## Checks and interpretation boundaries

- Complete grid, identical response identities, exclusion of fold 0, source hashes, predictor scaling and z=coefficient/SE passed [validation](validation.json).
- Stored tau=2 distances reproduced within 7.11e-15; twelve numerical transformation checks passed.
- Independent Nelder-Mead refits of the two domain winners converged without singular estimates. Fitted likelihood changed by less than 4.5e-6 and z by less than 0.00072; see [solver check](peak_solver_check.csv). This checks numerical stability of the inner fit.
- All 2,679 talker-specific training rows have zero distance under the inherited pairing. At very large k, their similarity remains one while positive-distance rows approach zero. These predictor-only scores therefore cannot establish an SBI contribution beyond condition; that requires the separate model comparison.
- This is one layer and one training split. It does not replace the three-fold SBI/ceiling figures, provide a held-out performance estimate, or establish general equivalence of the two optimization objectives.
- No outer-optimizer winner is established. The grid provides a reference for later equal-budget, multi-seed searches. A useful next check is whether the ridge and preferred region persist in the other training splits before expanding to every layer.

HVE remains paused. Previous outputs are unchanged. No feature extraction or t-SNE was rerun.

## Files and reproduction

`landscape.csv` contains all 1,276 fits; `best_grid_points.csv`, `tau_profiles.csv`, `near_best_regions.csv`, and `tau2_distance_by_condition.csv` support the summaries. `manifest.json` contains the actual run settings and source records; `RUN_RECORD.md` describes computation safeguards. `preparation/` and the per-tau fit tables retain the intermediate data.

From the repository root, regenerate the plots without fitting models:

```text
python cross_talker_generalization/scripts/plot_sbi_landscape.py --input cross_talker_generalization/analysis/sbi/parameter_maps/x21_all_conditions
python cross_talker_generalization/scripts/export_sbi_landscape_interactive.py --input cross_talker_generalization/analysis/sbi/parameter_maps/x21_all_conditions
```

To repeat model fitting, use `scripts/run_sbi_landscape.py --dataset X21 --output <new-directory> --rscript <Rscript-executable> --jobs 8 --budget-seconds 1800`. Preserve this completed directory.
