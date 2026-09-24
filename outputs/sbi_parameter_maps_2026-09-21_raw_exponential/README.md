# X21 parameter landscape without exponential normalization

This reruns the original X21 all-condition, HuBERT-base Tr-24 experiment using the full positive-k grid (29 tau values by 43 k values; 1,247 pairs). The predictor is directly `exp(-k * raw_distance)`: there is no minimum-distance shift, `expm1`, centering, or division by the predictor standard deviation. DTW mean-sequence-length normalization is unchanged. All cached input distances, training folds and the GLMM specification are retained.

## Files

- `x21_raw_exponential.html`: offline interactive z and fitted-log-likelihood surfaces. Check the status banner: an HTML generated during the run is a partial snapshot. The runner regenerates it when the run ends.
- `raw_landscape_preview.png`: static preview of the same scores on linear axes.
- `progress.json`: checkpointed completion counts, updated during the run.
- `manifest.json`: settings, scope, input hashes and final run status.
- `landscape.csv`: completed point records, including failures and timeouts.
- `comparison_with_stable.csv`: point-matched comparison against the original standardized grid; refreshed whenever the HTML is generated.
- `points/`: individual R results and process logs.

All finite scores are displayed, including flagged fits. Red markers identify recorded warnings, nonconvergence or singular fits. Large values from such fits must not be interpreted as better scientific performance. Timeout/failed points have no imputed z or likelihood. Surface facets only connect evaluated points. An optional signed-log z-axis button changes the display scale only; the default view is linear, and hover labels always report the original scores.

## Run settings

- Participant training folds 1 and 2 only: 10,969 responses, 213 participants. Fold 0 is unused.
- GLMM: binomial logit; raw similarity and test-talker fixed effects, participant and analysis-item random intercepts.
- lme4 / bobyqa, maxfun 20,000, nAGQ=1, derivatives enabled, no fallback structure.
- Sixteen concurrent R fits, with one BLAS/OpenMP thread each.
- A 90-second wall-time limit per point and a 30-minute overall budget. A fixed shuffled execution order spreads attempted points across the domain if the budget is reached. Unattempted points are labeled `not_started_budget`.
- The original 29 standardized negative-distance limit fits are not repeated: the original HTML displays only positive k; literal k=0 gives a constant raw predictor.

No original model results, maps, email draft or attachments are overwritten. No audio extraction, t-SNE, DTW, or HVE run is involved.

`run.py` starts a fresh run and refuses to overwrite an existing manifest. `plot.py` safely refreshes the derived HTML/PNG/comparison using the current checkpoint. `check_html.cjs` checks offline rendering, display-scale switching and linked rotation; it does not change model results.
