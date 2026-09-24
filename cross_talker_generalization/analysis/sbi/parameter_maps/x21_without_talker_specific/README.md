# X21 parameter landscape excluding Talker-specific

This separate sensitivity analysis removes `X21.Talker_specific` from the training data. The original four-condition results remain unchanged in `../x21_parameter_landscape/`.

Completed September 18: all **1,276 fits converged without singular estimates**, in **378.35 seconds (6.31 minutes)**. Validation passed for grid coverage, response identities, excluded-condition and held-out exclusion, source hashes, scaling and z arithmetic.

## Open the results

- [Rotatable 3-D map](figures/parameter_landscape_interactive.html): open in a browser; drag to rotate both panels together, scroll to zoom, and hover on grid points for exact fitted values. The HTML includes Plotly and all aggregate data for offline viewing.
- [Static 3-D map](figures/parameter_landscape_3d.png), [heatmaps](figures/parameter_landscape_2d.png) and [parameter profiles](figures/parameter_profiles.png); SVG copies are also available.
- Actual training-row distance, raw similarity and standardized predictor: [expanded-domain winner](selected_predictors/candidate_00.csv), [metric-domain winner](selected_predictors/candidate_01.csv). These row-level tables contain participant identifiers; share the aggregate figure rather than these tables when row-level data are unnecessary.
- [Selected parameter comparison](selected_parameters_comparison.csv), [predictor distributions](selected_predictor_distribution.csv), [condition summaries](selected_predictor_by_condition.csv) and [validation](validation.json).

## Findings

Both objectives select the same pair within each search domain:

| Domain | Best tau | Best k | Signed training z | Fitted training log likelihood |
| --- | ---: | ---: | ---: | ---: |
| Expanded, tau >= 0.5 | 0.5 | 0.00603418 | 4.147746 | -3161.879799 |
| Metric, tau >= 1 | 1 | 0.01791411 | 4.092719 | -3162.096992 |
| Fixed tau=2, best evaluated k | 2 | 0.05 | 4.009015 | -3162.410104 |

The preferred tau remains at the lower boundary in both domains. The best metric-domain k shifts from approximately 0.05318 in the full sample to 0.01791 in this subset. A positive fitted association remains after removing all self-comparison rows, although this selected training z does not provide an independent confirmatory significance test.

At the metric-domain winner, raw distance spans 0.51985–75.17857. Similarity spans 0.26008–0.99073, with mean 0.78924 and SD 0.10245. Its central 90% lies between 0.59742 and 0.92201. No row has zero distance or similarity exactly one; just one response has similarity above 0.99, and none is below 0.01. Similarity therefore retains substantial numerical variation.

The small-k plateau remains: after standardization, an almost linear exponential is close to standardized negative distance. At tau=2, optimizing k improves fitted log likelihood by 0.27787 relative to that limit. Large k reduces the scores sharply. These figures alone establish neither a preferred optimization algorithm nor an SBI effect beyond condition.

Independent Nelder-Mead refits at the tau=0.5 and tau=1 likelihood winners agreed with the main bobyqa fits within 0.000011 in z and 0.000008 in fitted log likelihood, with no convergence or singular-fit warning. See [solver check](peak_solver_check.csv). R emitted existing locale and Matrix build-version startup warnings.

## Scope

- Retain Control, Single-talker and Multi-talker: 8,290 training word responses, 161 participants; exclude 2,679 responses.
- Preserve the original participant split; use training folds 1+2 and do not score fold 0.
- Reuse the identical cached keyword DTW distances for the retained rows, across all 29 tau values. Exclusion does not change individual pair distances or their aggregation.
- Refit every GLMM, recalculating each candidate's standardization on the retained responses. Keep test-talker fixed blocking and participant/item random intercepts. No condition term.
- Retain the original grid: 29 tau values from 0.5 to 4, 43 positive k values from 1e-6 to 2, plus one standardized-negative-distance reference per tau; 1,276 fits.
- Sixteen R workers, one numerical thread each; 30-minute total budget. Existing feature files and the full-condition outputs are not modified.

The stale historical `similarity_exp_k` column is omitted from the cached-distance input tables. Similarity is calculated from raw distance and each candidate k inside R. Selected-predictor tables will explicitly identify their tau/k values and contain the corresponding raw similarity and training-standardized predictor.

## Interpretation

This analysis asks whether similarity predicts responses within the three conditions that have nonzero distances under the inherited pairing. It removes the Talker-specific self-comparison point mass at similarity=1. A predictor-only association still does not establish an effect beyond condition.

The two samples have different response and participant counts. Their raw z-values and total fitted log likelihoods cannot be used as a direct ranking of predictive performance. Compare the shape of the parameter landscape and the preferred parameter region descriptively. Held-out evaluation and across-fold stability remain separate tasks. Tau below 1 remains a non-metric sensitivity.

## Execution

`manifest.json` records completion and saved-fit counts. Its inherited physical-pair/feature metadata describe the source distance cache; the retained-response counts describe this subset fit. The source grid did the DTW calculations; this run filtered its saved tables and refitted the GLMMs.

```text
python cross_talker_generalization/scripts/run_sbi_landscape_subset.py --source cross_talker_generalization/analysis/sbi/parameter_maps/x21_all_conditions --output <new-directory> --rscript <Rscript-executable> --jobs 16 --budget-seconds 1800
```

After fitting, run `plot_sbi_landscape.py`, `export_sbi_landscape_interactive.py` and `summarize_sbi_subset_predictors.py`, each with `--input <new-directory>`. The first performs validation before drawing the figures. The other two require that validation to pass. Total runtime differs from the original full-condition run because both sample size and concurrency changed; it is not a controlled CPU speedup benchmark.
