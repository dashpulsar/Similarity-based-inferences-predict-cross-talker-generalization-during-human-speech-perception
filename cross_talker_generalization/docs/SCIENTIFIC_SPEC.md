# Scientific analysis specification

This file defines analysis choices that affect the scientific estimands. Implementation commands belong in `RUNBOOK.md`; project navigation belongs in the root `FILE_GUIDE.md`.

## 1. Theoretical predictors

The project evaluates two predictor families:

- **SBI (similarity-based inference):** whether responses vary with representational similarity between exposure-talker and test-talker speech.
- **HVE (heard variability during exposure):** whether responses vary with variability in the speech presentations that a listener heard during exposure.

The cross-dataset SBI predictor is a same-content talker proxy. It compares the test recording with an exposure-talker recording of the same word (AN19), keyword in the same sentence (X21), or sentence (B23). The counterfactual recording is not necessarily the exact token heard by the participant.

## 2. Representations and DTW

The primary analysis uses the established 3-D t-SNE representations for HuBERT base and ASR fine-tuned layers. Full-dimensional HuBERT is a sensitivity analysis. MFCC39 and STRF24 are acoustic baselines.

The primary paper-facing HuBERT result is fixed at Transformer layer 24 (`tr_24`) for every dataset and both HuBERT variants. Layer sweeps and best-layer searches are secondary robustness analyses. A data-selected layer must not be described as confirmatory unless candidate selection is nested inside an independent participant-level outer split.

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

## 5. Optimization objective and reporting metric

Model optimization and model reporting are separate choices. `M_predictor`, without experimental condition, is the model used to compare theoretical-predictor specifications.

The requested robustness analysis crosses two optimization summaries with two reporting summaries:

- optimize candidate specifications by predictor-only held-out likelihood/log loss;
- optimize candidate specifications by the signed predictor Wald z from `M_predictor`;
- report predictive fit as held-out gain relative to `M_null` and, where valid on the same observations, relative to the behavioral ceiling;
- report fixed-effect evidence as the predictor Wald z with its coefficient and confidence interval.

Zhengyang explicitly confirmed the signed-z convention on September 18. Negative associations are not converted to absolute z during candidate selection.

Likelihood and z are expected to correlate but are not interchangeable. Likelihood reflects the fit of the complete model, including its random-effects structure; z is the estimated theoretical-predictor coefficient divided by its standard error. Any z-based candidate search used for OOF prediction must select the candidate from training data only. Existing likelihood-ranked tables evaluate fixed candidate predictors. A matched comparison of z-based and likelihood-based parameter searches is still pending; the [optimizer audit](../analysis/sbi/OPTIMIZER_AUDIT.md) identifies the historical searches and their different evaluation procedures.

## 6. Cross-validation and predictor selection

Participants are assigned to three fixed, participant-disjoint folds. For each candidate theoretical predictor:

1. fit `M_predictor` on two training folds;
2. freeze the fitted model and training-fold predictor scaling;
3. compute population-level binomial log likelihood on the held-out participants;
4. sum held-out log loss across all three folds;
5. select the candidate with the smallest held-out `M_predictor` total log loss.

Candidate predictors may be ranked together only when they were scored on identical held-out observations and trial counts. The report code enforces this condition. Full-data log likelihood, deviance, AIC, observation count, and convergence status are retained for auditing, not used as the selection score.

This is currently a train-test design without an independent inner tuning split. Reusing these scores to choose a layer or HVE definition does not give the selected configuration an additional independent final test. An independent selection-evaluation split remains to be specified.

### Matched training/test diagnostic scores

From September 17, the GLMM runner additionally exports `train_test_scores.csv`. Each fold's fitted training model scores both its training and test observations using the same training-derived predictor mean/SD and `predict(..., re.form=NA)`. All fitted random effects are set to zero for both splits; this does not integrate predictions over a random-effects distribution. The score is the Bernoulli negative log likelihood summed over word responses, divided by `sum(response_correct + response_incorrect)`. Grouped rows therefore contribute their number of word responses, and no binomial combinatorial constant is added.

The fitted-model `logLik()` remains a separate model-fit diagnostic. It is not substituted for the new training prediction score. The per-response loss ratio is:

```text
ratio_fold = mean_test_log_loss / mean_training_log_loss
```

The ratio pairs training and test scores from the same fitted model within each fold. With a positive denominator, a value above 1 indicates worse held-out loss, and 1 indicates equal loss. The plot shows the equal-weight mean of the three fold ratios, their individual values, and a 95% percentile bootstrap interval obtained by enumerating all 27 ordered resamples of three folds. This is a descriptive summary of fold variability: the training folds overlap, and the interval does not establish an independent population-level test of overfitting. Nonpositive or nonfinite denominators and incomplete score pairs remain flagged; a three-fold summary is produced only when all three ratios are valid. Fit warnings are retained alongside valid scores.

These SI diagnostics use the existing fixed predictor specifications and participant folds. They do not implement the pending z-versus-likelihood optimization comparison or add an inner tuning split. The separate question about nested CV remains unanswered. A ratio of raw losses and a ratio of gains relative to a baseline are different quantities; neither their numerical equivalence nor universal positivity of gains is assumed. Existing z/ceiling figures retain their own statistical meaning. The [ratio figure script](../scripts/build_train_test_ratio_figures.py) implements this specification; run coverage and checks are recorded in the [September 17 work log](../analysis/diagnostics/WORK_LOG_2026-09-17.md).

### Direct behavioral reference (September 18 correction)

`ceiling_cv.py` estimates item-response probabilities using only participants in the two training folds, with Jeffreys smoothing: `(correct + 0.5) / (correct + incorrect + 1)`. The reference cells are condition × word × test talker for AN19, condition × sentence × keyword × test talker for X21, and the existing original-condition × sentence × test-talker cells for B23. X21 sentence context prevents the same keyword in different sentences from being pooled. The condition/context definition is recorded as `condition_and_token_context_v2`.

The default rejects cells with no training observations. A diagnostic option retains these as unavailable, reports coverage, and computes loss only over scored word responses; it never falls back to pooling conditions. All responses in the September 18 all-behavioral-row runs have predictions. The [comparison record](../analysis/sbi/README.md) preserves previous outputs and documents the changes.

This finite-sample human-response reference is called a ceiling by project convention; it is not a guaranteed upper bound on predictive performance. It produces held-out probabilities and likelihood scores, not a Wald-z ceiling. Before normalizing theoretical models, both the training reference pool and evaluated responses must be matched to the relevant analysis sample. The all-behavioral-row correction alone does not establish that match. Historical test-refit z-ceilings remain unchanged and must not normalize training-fit z-values.

## 7. Behavioral model comparisons

```text
M_null      = registered blocking terms + registered random effects
M_condition = condition + registered random effects
M_predictor = theoretical predictor + registered random effects
M_joint     = condition + theoretical predictor + registered random effects
```

`M_null` contains the same talker blocking term and random-effects structure as `M_predictor` but neither condition nor the theoretical predictor. Predictor-only OOF gain is `loss(M_null) - loss(M_predictor)`, so positive values indicate improvement. The matching nested full-data comparison is `M_null` versus `M_predictor`.

After a predictor specification has been selected without condition:

- `M_condition` versus `M_joint` tests whether the theoretical predictor adds information beyond condition;
- `M_predictor` versus `M_joint` tests whether condition adds information beyond the theoretical predictor.

Both full-data likelihood-ratio comparisons and paired held-out log-loss differences are retained. Neither comparison is the predictor-selection objective. A held-out-fold GLMM refit z-value is an association-stability summary, not a cross-validated prediction score.

### Declared common-structure sensitivity for AN19

The default `random_policy=registered` preserves the existing talker blocking/random-intercept rules and shared-within-scope fallback. The optional `random_policy=participant_item` fixes `(1 | participant_id) + (1 | analysis_item_id)` for every feature, fold and nested model in a run. This is the existing AN19 fallback used as an explicitly declared sensitivity analysis; it omits the talker term and has no automatic fallback. The policy is not selected using test performance and does not replace primary registered results. Both the policy and source hashes are included in run provenance and cache validation.

The runner also exports `variance_components.csv` for selected fitted models, using `VarCorr()` to retain estimated variances and standard deviations. The descriptive `variance_below_1e_8` flag is reported separately from `isSingular(tol=1e-4)`, convergence messages and failed fits. A near-zero random-intercept variance is a boundary estimate and does not automatically mean optimizer failure; inference from singular fits still requires caution ([lme4 documentation](https://lme4.github.io/lme4/reference/isSingular.html), [VarCorr](https://lme4.github.io/lme4/reference/VarCorr.html)). No further variance term is removed to suppress a warning.

The September 17 AN19 sensitivity covers 352 selected fits across the 16 acoustic feature spaces and six within-condition baseline analyses. Twenty fits retain a near-zero participant variance. The [common-structure runner](../scripts/run_an19_common_structure.py), [result metadata](../analysis/acoustic_baselines/common_structure/result_metadata.json) and [work log](../analysis/diagnostics/WORK_LOG_2026-09-17.md) document the comparisons and remaining flags.

## 8. Reporting status

The checked-in `analysis/reference` predates the changes above. It remains the historical broad August 21 package; corrected selection and revised HVE results are in `analysis/model_comparison/selection`.

The remaining analysis runs, common figure grammar, and Figure 1/2 panel requirements are tracked in `NEXT_ANALYSIS_PLAN.md` and `MAIN_FIGURE_SPEC.md`.
