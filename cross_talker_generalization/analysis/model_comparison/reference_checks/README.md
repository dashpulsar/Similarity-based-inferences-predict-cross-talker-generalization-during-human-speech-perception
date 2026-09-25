# September 6 analysis update

**Validation panels:** The retained figures include Tr-24 Figure 1a/b, checked language coverage (1c), the 36-L2-talker matrix (2a), and matched-word control curves (2c). Figure 1d now has a computed, explicitly narrower PHOIBLE inventory proposal. Figure 2b/d still lack verified AN19 segment inputs. The older `figure_drafts/` placeholders below have not been overwritten.

> **Review correction:** these are likelihood-based companion analyses, not the principal SBI z display or a completed manuscript figure set. Start with the [z-value review](z_value_review/README.md) and numbered requirements report. B23 HVE objective selection in Figure 4 mixes 97- and 168-participant samples and must be stratified or rerun on identical rows. Acoustic component distances used `coordinate_scaling=none`, whereas full baselines used `global_z`; their relative ranking is not a controlled ablation. Figure 1d and Figure 2b/d remain placeholders. These issues are not resolved by successful GLMM convergence.

This package implements the first analysis batch requested after the September meeting. All predictive gains use models fitted on two participant folds and frozen before scoring the third. `M_null` is now included, so predictor-only gain is oriented upward. Confidence intervals resample participants within folds.

## Main findings

The fixed Tr-24 HuBERT predictor has a positive predictor-only held-out gain in all six dataset-by-variant cells. It also adds positive held-out gain beyond condition in all six cells. By contrast, the confidence interval for condition beyond the HuBERT predictor includes zero in all six cells.

No positive fixed-Tr-24 HVE predictor-only interval excludes zero. Four gain intervals are entirely negative, meaning worse held-out prediction than the reference model; negative gain alone does not establish a negative predictor coefficient. Signed z is a separate statistic. Figure 4's B23 selection additionally needs the comparable-sample correction described above, so its winner labels should not be interpreted as validated selections.

| Dataset | Predictor | OOF gain (% ceiling) | 95% participant-cluster CI |
|---|---|---:|---:|
| AN19 | MFCC39 | 8.72 | [7.83, 9.61] |
| AN19 | STRF24 | 15.03 | [13.72, 16.38] |
| AN19 | HuBERT base | 15.58 | [13.62, 17.43] |
| AN19 | HuBERT ASR-FT | 9.20 | [7.82, 10.62] |
| X21 | MFCC39 | 3.91 | [1.46, 6.21] |
| X21 | STRF24 | 3.57 | [1.42, 5.54] |
| X21 | HuBERT base | 7.32 | [5.78, 8.85] |
| X21 | HuBERT ASR-FT | 7.55 | [6.04, 8.97] |
| B23 | MFCC39 | -0.03 | [-0.22, 0.16] |
| B23 | STRF24 | 0.29 | [0.15, 0.44] |
| B23 | HuBERT base | 0.77 | [0.56, 0.97] |
| B23 | HuBERT ASR-FT | 0.40 | [0.22, 0.61] |

## Figures

- `figure_01_fixed_tr24_sbi_predictor_gain`: fixed Tr-24 SBI and the two acoustic baselines, normalized to the directly cross-validated behavioral ceiling on matched rows.
- `figure_02_fixed_tr24_sbi_downstream_comparisons`: predictor beyond condition and condition beyond predictor.
- `figure_03_fixed_tr24_hve_methods`: all modelable fixed-Tr-24 HVE definitions.
- `figure_03b_fixed_tr24_hve_methods_zoomed`: the same HVE estimates on a readable local scale; the full-scale companion retains the ceiling.
- `figure_04_hve_objective_reporting_matrix`: likelihood-versus-z selection crossed with gain-versus-z reporting.
- `figure_04b_hve_objective_predictive_gain_zoomed`: a presentation-scale view of the predictive half of Figure 4.
- `figure_04c_selected_hve_downstream_comparisons`: predictor-beyond-condition and condition-beyond-predictor gains for both HVE selection objectives.
- `figure_05a_an19_acoustic_component_audit`: MFCC and STRF group diagnostics.
- `figure_05a2_an19_acoustic_component_gain_zoomed`: presentation-scale acoustic component gains.
- `figure_05b_an19_acoustic_gain_by_condition`: where the AN19 acoustic predictive gain occurs across conditions.
- `figure_05c_an19_acoustic_hubert_distance_correlations`: Spearman correlations on the same 5,459 physical train-test pairs.
- `figure_drafts/`: review drafts of manuscript Figures 1 and 2. Figure 2a/2c and the language inventory use current sources; panels requiring PHOIBLE metric decisions or segment-level response alignment remain visibly marked as missing.

## Leading AN19 acoustic groups

| Feature group | OOF gain (% ceiling) | 95% participant-cluster CI |
|---|---:|---:|
| `mfcc_delta_delta13` | 15.3 | [13.6, 17.0] |
| `mfcc_delta13` | 13.7 | [12.4, 15.0] |
| `strf_rate_16` | 10.5 | [9.7, 11.4] |
| `strf_rate_4` | 8.8 | [7.8, 9.9] |
| `strf_direction_positive` | 8.0 | [6.9, 9.1] |

These group analyses are diagnostic. They identify where to run individual-dimension and leave-one-group-out checks; they do not yet establish that the full AN19 baseline is free of condition encoding or recording leakage.
The pair audit found no duplicate pair identifiers, no self-recording comparisons, and no same-speaker train-test comparisons. Control rows have no exposure predictor by design, and 75 exposed-test rows per feature have incomplete source mappings; the fitted models therefore use the same 5,685 non-control rows for every representation.
MFCC39 and STRF24 correlate moderately with HuBERT Tr-24 distances on identical pairs (Spearman 0.38–0.46). The pooled-model gain from MFCC delta, MFCC delta-delta, and STRF rate-16 is positive within each of the six modeled AN19 exposure conditions, so simple between-condition separation is not a sufficient explanation. Separate within-condition refits and individual-dimension/leave-one-group-out tests remain necessary.

## Inferential boundaries

The fixed Tr-24 panels are common-layer likelihood summaries, not substitutes for the SBI layerwise z display. HVE method selection in Figure 4 is exploratory because the same study supplies candidate comparisons and summaries, and B23 additionally requires comparable-sample correction. The z interval is a bootstrap over three training-fold z values; the predictive intervals resample participants. These uncertainty procedures must be labeled separately.

## Run validation

All 1,664 requested GLMM fits completed; 1,664 reported normal convergence, with 0 singular fits, 0 warnings, and 0 errors. See `tables/run_diagnostics_summary.csv`.
