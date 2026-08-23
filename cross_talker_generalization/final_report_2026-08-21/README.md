# Final experimental report and figures (2026-08-21)

This directory is the current reviewed report package. Superseded reports remain available through Git history.

> **Status note (2026-08-23):** feature spaces in this release were ranked by condition-only versus joint OOF log-loss gain. That is an incremental prediction comparison, not the planned predictor-only GLMM likelihood optimization. B23 HVE in this release also predates integration of the public multi-talker stimulus lists. See the repository-root [TODO.md](../../TODO.md) before treating “best” labels or B23 HVE coverage as final.

## Four principal corrections

1. `figure_00_notebook_ceiling_normalized_profiles` reproduces the historical three-fold Wald-z/ceiling display while explicitly labeling it as a held-out-refit association statistic, not OOF predictive accuracy or variance explained.
2. `figure_01_sbi_layer_profiles` asks only whether HuBERT similarity improves prediction for unseen participants. MFCC, STRF, and a z=1.96 reference line are not mixed into this figure; acoustic OOF comparisons appear in Figure 02.
3. `figure_05b_experiment_talker_distance_matrices` contains genuine all-talker matrices: AN19 is 42×42 with 138 shared words per off-diagonal cell, X21 is 11×11 with all 32 matched experimental sentences, and B23 is 4×4 with 120 shared sentences. DTW uses `tau=2` and historical mean-sequence-length normalization.
4. The overall-only compatibility variability figures are retained, and every computable variability method plus true OOF results is included in the Figure 03 series.

## Why MFCC/STRF are high in Figure 00

The plotting code does not inflate these values. They come directly from stored GLMM `z_test` values, and the historical AN19 notebooks show the same high acoustic-baseline z values. Wald z is coefficient divided by standard error, so large samples, small standard errors, and acoustic–behavior association can all make z large. Dividing by behavioral-ceiling z only rescales z; it does not state how much human behavior is predicted. Acoustic predictive value should be assessed with participant-held-out OOF log-loss gain in Figure 02.

Audited values:

- AN19 MFCC (39-D): mean z=11.72, z/ceiling=52.9%, true OOF log-loss gain=0.029294.

- AN19 STRF (24-D): mean z=13.94, z/ceiling=62.9%, true OOF log-loss gain=0.048058.

- X21 MFCC (39-D): mean z=5.74, z/ceiling=26.4%, true OOF log-loss gain=-0.00018913.

- X21 STRF (24-D): mean z=5.72, z/ceiling=26.3%, true OOF log-loss gain=3.1794e-06.

- B23 MFCC (39-D): mean z=0.51, z/ceiling=11.0%, true OOF log-loss gain=1.4439e-05.

- B23 STRF (24-D): mean z=0.28, z/ceiling=6.1%, true OOF log-loss gain=0.0002278.

## Figure 05b source data

- Prefer the dataset-specific `figure_05b1_an19_*`, `figure_05b2_x21_*`, and `figure_05b3_b23_*` panels for presentation. `figure_05b_experiment_*` is the six-matrix overview.
- `tables/talker_distance_tr24.csv` is the long table for all six dataset × model matrices, including zero diagonals.
- `tables/talker_distance_item_level_x21_b23.csv` contains sentence-level DTW distances for every undirected X21/B23 talker pair. AN19 word-level details remain in `results/derived/AN19-talker-validation-*/`; the report uses the validated 138-word summary rather than recomputing it.

## Variability interpretation boundary

The earlier `figure_03a_variability_ceiling_normalized_profiles` displays only `overall` and takes `abs(z_test)`. It is retained strictly as a compatibility output. Complete sign-preserving Figure 03 profiles and OOF incremental results are described below. B23 covers only the currently implemented single-talker exposure pools and must be rerun after the public multi-talker stimulus lists are integrated.

## Complete exposure-variability results

This report contains every computed variability measure rather than only `overall`:

- `figure_03a_variability_true_oof_core_profiles`: participant-held-out condition-incremental OOF result;
- `figure_03b_variability_ceiling_normalized_core_profiles`: signed three-fold Wald-z/ceiling display;
- Figures 03c–03e: every computable method for AN19, X21, and B23;
- Figures 03f–03g: true OOF results for every method in AN19 and X21.

The overall-only `abs(z_test)` figures are retained strictly for historical compatibility.
Three-fold held-out-refit Wald z measures association stability, not frozen-model OOF
prediction. The public B23 multi-talker exposure assignment has not yet been integrated,
so this release does not report a multi-talker B23 HVE model.

AN19 `between_type_word` is not the same as the historical pooled-token `BetweenWord`.
The label `Between word types` prevents the two estimands from being conflated.
