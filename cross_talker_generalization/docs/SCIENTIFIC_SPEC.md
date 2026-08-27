# Scientific analysis specification

This file defines analysis choices that affect the scientific estimands. Implementation commands belong in `RUNBOOK.md`; project navigation belongs in the root `FILE_GUIDE.md`.

## 1. Theoretical predictors

The project evaluates two predictor families:

- **SBI (similarity-based inference):** whether responses vary with representational similarity between exposure-talker and test-talker speech.
- **HVE (heard variability during exposure):** whether responses vary with variability in the speech presentations that a listener heard during exposure.

The cross-dataset SBI predictor is a same-content talker proxy. It compares the test recording with an exposure-talker recording of the same word (AN19), keyword in the same sentence (X21), or sentence (B23). The counterfactual recording is not necessarily the exact token heard by the participant.

## 2. Representations and DTW

The primary analysis uses the established 3-D t-SNE representations for HuBERT base and ASR fine-tuned layers. Full-dimensional HuBERT is a sensitivity analysis. MFCC39 and STRF24 are acoustic baselines.

The default local DTW cost is Minkowski distance with `tau=2`. The primary normalization divides accumulated path cost by mean sequence length, matching the historical analyses. Path-length normalization is retained as a sensitivity analysis. Bounded descriptive similarity is `exp(-k d)`; GLMM predictors use fold-specific standardized negative distance instead.

## 3. HVE definitions

Generalized frame dispersion is

```text
mean_t sum_j |x[t,j] - mean_t(x[t,j])|^tau
```

The quantity intentionally does not take the `1/tau` root. At `tau=2` it is a generalized variance (mean squared Euclidean deviation), not a Euclidean distance.

The registry contains 17 definitions:

- `overall`: dispersion across all exposure frames, independent of presentation order;
- `overall_order_sensitive`: concatenate complete exposure tokens in actual presentation order and average powered distances between every pair of consecutive frames, including token-boundary transitions;
- `within_token_*`, `within_type_*`, `between_type_*`, `order_*`, and `mean_dissimilarity_*` at sentence, word, and phoneme levels.

The three `order_*` measures describe adjacent frames inside each token and then average across tokens. They do not connect token boundaries and do not change when whole tokens are reordered. This is why `overall_order_sensitive` is a separate measure.

## 4. Exposure reconstruction

- **AN19:** `trial.within_phase` recovers each trained participant's 144 presentations in order.
- **X21:** the public participant-level training data recover all 80 presentations and their order for every participant in the production sample.
- **B23:** the public stimulus lists identify the actual recording for every training sentence, including the documented Spanish/Turkish filename-label discrepancy. The public training table covers all 168 trained participants. Ninety-seven participants have a complete unique sequence of trial indices. Seventy-one have duplicate or missing trial indices; their unordered exposure sets remain usable, but `overall_order_sensitive` is marked unavailable rather than imputed.

The two same-sentence repeated-token measures, `within_type_sentence` and `mean_dissimilarity_sentence`, are not usable as B23 cross-participant predictors. The public table gives 167 trained participants one token per sentence type; one participant has a duplicated sentence/segment, yielding a trivial value for that participant alone and insufficient coverage for a behavioral model.

## 5. Cross-validation and predictor selection

Participants are assigned to three fixed, participant-disjoint folds. For each candidate theoretical predictor:

1. fit `M_predictor` on two training folds;
2. freeze the fitted model and training-fold predictor scaling;
3. compute population-level binomial log likelihood on the held-out participants;
4. sum held-out log loss across all three folds;
5. select the candidate with the smallest held-out `M_predictor` total log loss.

Candidate predictors may be ranked together only when they were scored on identical held-out observations and trial counts. The report code enforces this condition. Full-data log likelihood, deviance, AIC, observation count, and convergence status are retained for auditing, not used as the selection score.

## 6. Behavioral model comparisons

```text
M_condition = condition + registered random effects
M_predictor = theoretical predictor + registered random effects
M_joint     = condition + theoretical predictor + registered random effects
```

After a predictor specification has been selected without condition:

- `M_condition` versus `M_joint` tests whether the theoretical predictor adds information beyond condition;
- `M_predictor` versus `M_joint` tests whether condition adds information beyond the theoretical predictor.

Both full-data likelihood-ratio comparisons and paired held-out log-loss differences are retained. Neither comparison is the predictor-selection objective. A held-out-fold GLMM refit z-value is an association-stability summary, not a cross-validated prediction score.

## 7. Reporting status

The checked-in `final_report_2026-08-21` predates the changes above. It remains a historical reviewed report until the revised exposure tables, HVE values, GLMMs, selection tables, and figures have been regenerated and checked in a new dated report directory.
