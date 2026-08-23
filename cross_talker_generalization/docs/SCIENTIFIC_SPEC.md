# Scientific specification

## Questions and estimands

The project compares two mechanisms:

- SBI: does behavior vary with representational similarity between exposure talker(s) and the test talker?
- HVE: does behavior vary with the variability of the speech actually heard during exposure?

The stable SBI predictor available in all three datasets is `same_content_talker_proxy`: the test token is compared with an exposure-talker recording of the same linguistic content. That recording is not necessarily the token heard by the participant and must not be called `actual_exposure_exemplar_similarity`.

Match units are the same word for AN19, the same keyword in the same sentence for X21, and the same sentence for B23. Multi-talker aggregation defaults to arithmetic mean raw distance. Maximum historical similarity is represented by the `min_distance` sensitivity profile.

## Representations

- HuBERT large base and ASR fine-tuned each contribute 18 layers: CNN 2–6 and Transformer 0, 2, ..., 24.
- Each 3-D t-SNE representation was fit separately for dataset × model × layer using the complete registered corpus. The primary analysis does not z-score its three axes again.
- Full-dimensional HuBERT, MFCC39, and STRF24 are standardized per dimension using corpus-global moments before distance computation.
- Absolute raw distances are not compared across dataset-specific t-SNE spaces.

## DTW and similarity

The default local cost is Minkowski with `tau=2`. DTW returns minimum accumulated cost and applies an explicit normalization:

- `mean_sequence_length`: divide by `(n + m) / 2`; this matches the historical notebooks and current derived tables.
- `path_length`: divide by the optimal warping-path length; this matches the earlier paper formula and is a required sensitivity analysis.
- `none`: diagnostic use only.

All outputs retain raw distance. Descriptive bounded similarity is `exp(-k d)`. Confirmatory GLMMs use `-(d - mean_train) / sd_train`, with moments estimated on the training fold only.

## Cross-validation

Folds are participant-disjoint with fixed seed 230519 and dataset-specific design-cell stratification. Held-out participants cannot influence `tau`, `k`, random-effects structure, or predictor scaling.

For the incremental comparison, each training-fold model is frozen before population-level binomial log loss is computed for held-out participants. A Wald z obtained by refitting a GLMM on the held-out fold measures association stability, not prediction.

## Behavioral models

Each predictor compares:

```text
M_condition: original condition + random effects
M_predictor: predictor_z + random effects
M_joint:     original condition + predictor_z + random effects
```

AN19 and X21 use binary rows. B23 retains 4–7 keyword counts per sentence as `cbind(correct, incorrect)` count-binomial observations.

The intended optimization criterion for theoretical predictors is the maximized likelihood of `M_predictor`, without the condition term. The current pipeline fits `M_predictor` but does not persist its full-data log likelihood or use it for report selection. The reported `M_condition` versus `M_joint` LRT and held-out log-loss difference answer the separate question of whether a predictor adds information beyond condition. `z_predictor / z_ceiling` appears only in compatibility figures and is not described as variance explained.

## HVE status

- AN19: the exposure phase supports reconstruction of the actual 144-token list and isolated-word measures.
- X21: the public design supports reconstruction of 80 presentations; each of 16 Single-talker/Talker-specific tokens repeats five times. Unique-token HVE is secondary.
- B23: current production outputs cover four single-talker pools. Fourteen definitions are mathematically available there; the two same-sentence, multiple-token measures are undefined. The public OSF stimulus lists provide multi-talker sentence-to-recording assignments, but production ingestion and validation are pending, so those cells remain marked `blocked` for the current release.
