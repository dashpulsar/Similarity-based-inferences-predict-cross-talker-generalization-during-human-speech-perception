# Complete Figure 03 variability package

This component adds complete variability figures without overwriting earlier compatibility outputs.

## Recommended presentation order

1. `figure_03a_variability_true_oof_core_profiles`: participant-held-out condition-incremental OOF log-loss gain. This is separate from the planned predictor-only likelihood optimization. B23 is absent because the public multi-talker stimulus mapping was not integrated for this release.
2. `figure_03b_variability_ceiling_normalized_core_profiles`: notebook-compatible three-fold Wald-z/behavioral-ceiling display with the sign of z preserved.
3. Figures 03c–03e: every currently computable variability method for AN19, X21, and B23.
4. Figures 03f–03g: true OOF results for all methods in AN19 and X21.

## Differences from the earlier Figure 03

- The package is not restricted to `overall`.
- It shows 6 AN19, 16 X21, and 14 B23 computed methods. For the current B23 single-talker pools, `within_type_sentence` and `mean_dissimilarity_sentence` are undefined because there is only one recording per sentence type; this count is separate from the missing multi-talker condition coverage.
- It does not use `abs(z)`: positive values associate greater variability with greater accuracy; negative values indicate the opposite direction.
- Three-fold association figures retain fold points, fold means, exact fold-bootstrap 95% intervals, and behavioral-ceiling 95% bands.
- Held-out-refit Wald z is explicitly separated from frozen-model OOF prediction.

## Historical measure that is not claimed as reproduced

Current AN19 `between_type_word` first combines token centroids within word type. The historical notebook's `BetweenWord` pooled all talker × word token centroids. These are different estimands. The new figure uses `Between word types` and does not claim reproduction of historical `BetweenWord`. Exact reproduction would require adding that measure and rerunning the AN19 variability GLMM.

All three-fold association results use `tau=2`. Percentages are Wald z rescaled by mean behavioral-ceiling z, not the percentage of human behavior explained.
