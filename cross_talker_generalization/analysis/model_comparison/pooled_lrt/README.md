# Cross-fitted predictor nested-model update

> **Analysis scope:** this package is a supplementary nested-association analysis. It is distinct from both the principal SBI layerwise z display and frozen-model held-out likelihood. Tr-24 serves as a common manuscript representation alongside layerwise displays. See the [z figure specification](../../../docs/Z_VALUE_FIGURE_SPEC.md).

This supplementary analysis implements the combined-test-fold procedure requested for nested GLMM comparison. For each selected SBI or HVE specification, the predictor is standardized for each held-out participant fold using the mean and standard deviation from the other two folds. The three held-out partitions are then concatenated, and one set of GLMMs is fit to the combined data.

This operationalizes option B rather than parameter averaging (option C). In the current analysis profile, tau, the distance-to-predictor transformation, and the aggregation rule are fixed; the selected layer or HVE definition is categorical and cannot be averaged across folds. Consequently, there is no fold-specific continuous theoretical parameter such as tau or k to average. Only the nuisance standardization moments are fold-specific, and those are estimated without the held-out fold.

All models include the outer-fold label as a fixed blocking factor because the predictor scale was estimated separately for each fold. The participant, item, and talker structure otherwise follows the confirmatory analysis. B23 retains count-binomial responses.

Two likelihood-ratio tests are reported:

- `M_condition` versus `M_joint`: whether the theoretical predictor adds information beyond experimental condition.
- `M_predictor` versus `M_joint`: whether condition adds information beyond the theoretical predictor.

The run produced 28 successful LRTs out of 28 planned comparisons. All 42 GLMM fits converged and were non-singular.

## Important inferential boundary

The predictor values are cross-fitted with respect to fold-specific scaling, but the HuBERT layer or HVE definition was selected using the same three-fold study. These p-values are therefore selection-conditional supplementary tests, not selection-adjusted confirmatory p-values. The participant-held-out OOF log-loss and cluster-bootstrap results remain the primary predictive evaluation. A fully selection-independent claim would require prespecifying the predictor or nesting candidate selection inside a new outer participant split.

A significant combined-data LRT can coexist with weak or negative frozen-model OOF gain. The LRT tests an in-sample association after the cross-fitted predictor values have been assembled, whereas OOF log loss tests transport to unseen participants using model coefficients frozen on training folds. Such a disagreement is scientifically meaningful and is not a software inconsistency.

## Outputs

- `figures/figure_01_crossfitted_nested_lrt`: visual summary of both LRTs.
- `tables/crossfitted_lrt_results.csv`: chi-square, degrees of freedom, p-value, and log-likelihood change.
- `tables/crossfitted_model_diagnostics.csv`: formula and convergence audit.
- `tables/crossfitted_coefficients.csv`: combined-data fixed-effect estimates.
- `tables/fold_scaling.csv`: fold-specific training moments used to construct each held-out predictor.
- `models/`: row-level cross-fitted predictors and per-analysis R outputs.

## Reproduce

Run from the repository root with R/lme4 and the Python project dependencies available:

```powershell
python .\cross_talker_generalization\scripts\run_crossfitted_lrt.py --jobs 5
```

## Results with p < .05 (uncorrected, selection-conditional)

| Dataset | Family | Variant | Feature | Comparison | LRT chi-square | df | p |
|---|---|---|---|---|---:|---:|---:|
| AN19 | HVE | base | `cnn_3::overall_order_sensitive` | condition_beyond_predictor | 18.887 | 5 | 0.002017 |
| AN19 | HVE | ft | `cnn_3::overall_order_sensitive` | condition_beyond_predictor | 18.887 | 5 | 0.002017 |
| AN19 | SBI | base | `tr_10` | condition_beyond_predictor | 18.560 | 5 | 0.002321 |
| AN19 | SBI | base | `tr_10` | predictor_beyond_condition | 17.190 | 1 | 3.383e-05 |
| AN19 | SBI | ft | `tr_10` | condition_beyond_predictor | 16.627 | 5 | 0.005265 |
| AN19 | SBI | ft | `tr_10` | predictor_beyond_condition | 9.376 | 1 | 0.002199 |
| B23 | HVE | base | `tr_6::overall_order_sensitive` | predictor_beyond_condition | 6.133 | 1 | 0.01327 |
| B23 | HVE | base | `tr_12::between_type_sentence` | predictor_beyond_condition | 7.060 | 1 | 0.007884 |
| B23 | HVE | ft | `tr_20::overall_order_sensitive` | predictor_beyond_condition | 4.102 | 1 | 0.04284 |
| B23 | HVE | ft | `tr_4::between_type_sentence` | predictor_beyond_condition | 6.102 | 1 | 0.0135 |
| X21 | HVE | base | `cnn_6::order_word` | condition_beyond_predictor | 11.008 | 3 | 0.01168 |
| X21 | HVE | ft | `cnn_6::order_word` | condition_beyond_predictor | 11.008 | 3 | 0.01168 |
| X21 | SBI | base | `tr_14` | condition_beyond_predictor | 21.003 | 3 | 0.0001051 |
| X21 | SBI | base | `tr_14` | predictor_beyond_condition | 6.518 | 1 | 0.01068 |
| X21 | SBI | ft | `tr_22` | condition_beyond_predictor | 27.740 | 3 | 4.119e-06 |
| X21 | SBI | ft | `tr_22` | predictor_beyond_condition | 4.805 | 1 | 0.02837 |
