# Technical documentation

## 1. Research questions and analysis units

The two core estimands are:

- **SBI, similarity-based inference**: whether representational distance between exposure talkers and the test talker, for matched linguistic content, explains or predicts behavior.
- **HVE, heard/exposure variability**: whether dispersion within the speech set heard during exposure explains or predicts behavior.

## 2. Inputs and data contract

`configs/project.json` registers behavioral CSV files, stimulus manifests, and 15 HDF5 feature stores. Loading checks expected row counts, participant counts, speech-unit counts, layer names, dataset/variant attributes, and finite values. Feature files are read-only inputs.

The registered HDF5 inventories are 6,261 units for AN19, 660 for X21, and 480 for B23. Each dataset includes:

- HuBERT large base, full-dimensional;
- HuBERT large base, 3-D t-SNE;
- HuBERT large ASR fine-tuned, full-dimensional;
- HuBERT large ASR fine-tuned, 3-D t-SNE;
- MFCC39 and STRF24 acoustic baselines.

The pipeline does not re-extract HuBERT features or refit t-SNE. It consumes the supplied frame-level sequences after validating their structure.

## 3. Representations

The 18 registered HuBERT layers are `cnn_2` through `cnn_6`, `tr_0`, and `tr_2, tr_4, ..., tr_24`. Base and ASR fine-tuned variants are analyzed separately.

Existing 3-D t-SNE sequences are the primary method-reproduction representation. Each dataset × model variant × layer has its own coordinate space. The primary analysis does not z-score the three t-SNE coordinates again, and absolute raw distances are not compared across datasets.

Full-dimensional HuBERT, MFCC39, and STRF24 are standardized per dimension using the complete representation corpus before DTW. These corpus-global statistics are not refit by participant or condition and do not use behavioral outcomes. Full-dimensional HuBERT is retained as a key sensitivity analysis.

## 4. DTW distance

Let `X=(x_1,...,x_n)` and `Y=(y_1,...,y_m)` be frame-level sequences. The default local cost is a Minkowski distance:

```text
c(i,j) = (sum_r |x_ir - y_jr|^tau)^(1/tau),  tau = 2
```

Dynamic programming finds the warping path with minimum cumulative cost. The primary reproduction result is:

```text
d_mean = raw_minimum_cost / ((n + m) / 2)
```

This mean-sequence-length normalization matches the historical notebook implementation. The earlier paper described:

```text
d_path = raw_minimum_cost / optimal_path_length
```

The two denominators can differ systematically when sequence lengths or warping patterns differ. `configs/sensitivities/path_length.json` therefore implements path-length normalization as a required sensitivity analysis. `tau=1`, `tau=3`, cosine local cost, and no normalization are also supported for declared analyses or diagnostics. Every distance output retains raw cost, path length, both frame counts, and normalized distance.

## 5. From distance to the SBI predictor

Physical speech pairs are computed once in a non-behavior-replicated pair table and then aggregated to experiment condition × test item. Participant count therefore cannot change the weighting of a physical speech comparison.

Single-talker conditions use the corresponding exposure–test distance. The primary multi-talker definition averages raw distance across exposure talkers:

```text
d_multi = mean_t d(exposure_talker_t, test_talker)
```

The `min_distance` sensitivity uses only the closest exposure talker. Control cells have no exposure talker and are explicitly labeled `no_exposure`; no artificial zero similarity is assigned.

Descriptive figures may use bounded similarity:

```text
s = exp(-k d),  k > 0
```

The compatibility profile uses a declared `k` grid. Confirmatory GLMMs do not tune `k` to maximize z. Instead, each training fold defines:

```text
similarity_z = -(d - mean_train) / sd_train
```

The sign makes greater similarity correspond to a larger predictor. Training-fold moments are applied unchanged to held-out participants.

## 6. Exposure variability: 17 registered measures

An exposure pool contains tokens represented as frame × dimension sequences, each with a token ID and a type ID. Generalized dispersion is computed without taking the `1/tau` root:

```text
V(A) = mean_i sum_r |a_ir - mean(A)_r|^tau
```

At `tau=2`, this is mean squared Euclidean deviation, not ordinary Euclidean standard deviation. There are 17 registered measures: two overall measures plus five families at sentence, word, and phoneme levels.

| Measure | Definition |
|---|---|
| `overall` | Pool all available exposure frames and compute generalized dispersion |
| `overall_order_sensitive` | Concatenate complete tokens in actual exposure order and average powered adjacent-frame distance, including cross-token boundaries |
| `within_token_{unit}` | Compute frame dispersion within each token, then average across tokens |
| `within_type_{unit}` | Compute each token center, dispersion among centers within a linguistic type, then average across types |
| `between_type_{unit}` | Compute one center per linguistic type, then dispersion among type centers |
| `order_{unit}` | Mean powered distance between adjacent acoustic frames within each token; order is frame order, not trial order |
| `mean_dissimilarity_{unit}` | Mean pairwise DTW distance within each type, then mean across types |

The set contains `overall`, `overall_order_sensitive`, and `within_token_*`, `within_type_*`, `between_type_*`, `order_*`, and `mean_dissimilarity_*` for three units: `2 + 5 × 3 = 17`.

Availability differs by dataset:

- AN19 contains isolated-word exposure and supports both overall measures plus the five word-level measures. Sentence and phoneme variants are explicitly unsupported.
- X21 reconstructs 80 exposure presentations. In Single-talker and Talker-specific conditions, each of 16 tokens is repeated five times. The primary HVE estimand preserves presentation weighting; a unique-token version remains available.
- B23 now uses the public stimulus lists and training table for all single- and multi-talker conditions. The actual filename, not the nominal speaker label alone, determines the recording. Fifteen definitions are modelable. The two same-sentence measures are unavailable as cross-participant predictors: 167 participants have one token per sentence type, while one source record contains a duplicated sentence/segment and yields coverage for only that participant.
- B23 participant-level trial order is complete for 97 trained participants. For the other 71, duplicate or missing public trial indices make only `overall_order_sensitive` unavailable; order-independent measures remain available.

## 7. Participant-disjoint three-fold analysis

Folds are defined by participant, not trial. `seed=230519`, `n_folds=3`, with stratification by dataset-specific design or condition cells. Each participant belongs to exactly one fold.

Each predictor is evaluated with:

```text
M_condition = original experimental condition + registered random effects
M_predictor = predictor_z + registered random effects
M_joint     = original condition + predictor_z + registered random effects
```

AN19 and X21 use binary responses. B23 retains correct/incorrect keyword counts per sentence and uses a count-binomial `cbind(correct, incorrect)` outcome; a sentence is not treated as one Bernoulli trial.

R/lme4 fits the GLMMs. Full-data models report coefficient, standard error, 95% CI, and Wald z. The pipeline records both likelihood-ratio comparisons: `M_condition` versus `M_joint` for predictor information beyond condition, and `M_predictor` versus `M_joint` for condition information beyond the predictor. Random-effects structure, convergence, and singularity are recorded in diagnostics rather than silently altered.

There are two distinct model-comparison questions:

1. **Theoretical-predictor optimization:** compare representations or parameterizations using summed three-fold held-out log loss from `M_predictor`, which excludes the original experimental condition predictor. Lower loss is better, and candidates must use identical held-out observations.
2. **Incremental prediction beyond condition:** compare `M_condition` with `M_joint`, including on held-out participants.

The R code can fit only `M_predictor` during candidate selection, then fits all three models for selected candidates. It stores full-data fit statistics for auditing and scores frozen models on held-out participants. Report selection uses predictor-only held-out loss. The August 21 report predates this implementation; revised results are in `analysis_update_2026-08-27`.

## 8. True OOF prediction versus compatibility z

These quantities have different meanings.

**True cross-validated prediction:** a GLMM is fit on two training folds and frozen. It predicts the third, entirely unseen participant fold at the population level without held-out participant random effects. The three held-out partitions form the OOF predictions. Incremental performance beyond condition is:

```text
OOF gain = logloss(M_condition) - logloss(M_joint)
```

Positive values mean that adding the predictor improves prediction for unseen participants; negative values mean worse prediction. This comparison is additional to, and must not be substituted for, predictor selection based on held-out `M_predictor` log loss.

**Held-out-refit Wald z:** after selecting one fold, the GLMM is refit directly on that fold's behavioral responses and its z is recorded. Because those responses were used for fitting, the statistic measures association stability across participant subsets, not out-of-sample prediction. The historical method is preserved for compatibility figures but is not rerun by default.

## 9. Behavioral ceiling

The ceiling also respects participant folds. For each fold, item-level accuracy or log odds are estimated from the other two folds and attached to held-out responses. The project provides:

- Direct OOF item-probability ceiling predictions evaluated by log loss;
- Three compatibility z values from refitting a ceiling predictor on each held-out fold.

Figure 00 and its variability analogue define 100% as the mean of the three compatibility ceiling z values. A gray 95% band around 100% represents fold uncertainty. `z / mean(z_ceiling) × 100` is only a rescaling of association statistics; it is not variance explained, predictive accuracy, or the percentage of human behavior captured.

## 10. Figures and interpretation

- **Figure 00:** MFCC39 and STRF24 appear first, followed by 18 HuBERT layers. Gray dots are fold-specific held-out-refit z values; black points/lines show the fold mean and fold-bootstrap 95% interval; the gray band represents ceiling uncertainty. This is a compatibility association figure.
- **Figures 01–02 in the August 21 package:** participant-held-out frozen-model OOF log-loss gain asks whether a representation improves prediction for unseen participants beyond condition. The zero line means no incremental prediction, not a significance threshold. Its old best-layer labels are superseded by the predictor-only selection and paired participant-bootstrap results in `analysis_update_2026-08-27`.
- **Figure 03 series:** the compatibility panel matches Figure 00 semantics, while the core and dataset-specific profiles report OOF incremental results across every currently computed HVE method.
- **S-curves:** panels follow available test talkers and conditions. Points are trial-count-weighted accuracy in predictor quantile bins with Wilson 95% binomial intervals. Curves are descriptive binomial logistic fits, not hierarchical GLMM conditional-effect plots.
- **Talker distance matrices:** for talkers A and B, DTW is computed for each shared linguistic item and averaged. X21 uses all 32 matched experimental sentences per cell, B23 uses 120 common sentences, and AN19 uses the shared word set. Matrices are symmetric with a zero diagonal.
- **Correlation matrices:** raw distances from exactly the same physical pairs/cells are compared across layers or representations. Inputs containing participant/fold/response replication are rejected.

Every formal figure has a CSV source table. PNG is intended for presentation; SVG is retained for publication-quality editing.

## 11. Sensitivity analyses

The main profile fixes `tau=2`, mean-sequence-length normalization, multi-talker mean distance, and training-fold-standardized negative distance. Implemented variants include:

- DTW path-length normalization;
- `tau=1` and `tau=3`;
- Multi-talker minimum distance;
- `exp(-d)` predictor transformation;
- Full-dimensional HuBERT;
- HuBERT base versus ASR fine-tuned;
- MFCC39 and STRF24 acoustic baselines;
- Presentation-weighted versus unique-token HVE where identifiable.

Each profile must write to a distinct output directory and preserve its parameters in provenance.

## 12. Parallel execution and provenance

DTW and HVE tasks are parallelized by feature layer. Each worker owns its HDF5 handle, and BLAS threads within a worker are restricted to one to avoid `jobs × BLAS threads` oversubscription. The default is eight jobs. Independent matched-content comparisons for all talker pairs may use up to 32 threads, bounded by CPU count and task count. Full-dimensional runs should reduce jobs if memory pressure is high.

Provenance JSON records random seed, input paths and hashes, runtime versions, parameters, output hashes, and status counts. Stable IDs and explicit join keys connect tables. Unavailable, unsupported, or blocked cells are recorded as states rather than silently removed or guessed.

### Result authority and retention

The reviewed package at `cross_talker_generalization/analysis_update_2026-08-21/` is the
broad presentation inventory. Corrected SBI selection/downstream comparisons and the complete
revised t-SNE HVE candidate analysis are in
`cross_talker_generalization/analysis_update_2026-08-27/`.
`cross_talker_generalization/artifacts/` supplies the
refactored true OOF SBI and HVE products. The top-level `results/` directory is retained
only where the report builder still requires historical notebook-compatible summaries,
S-curve source files, validated AN19 talker-distance inputs, or direct audit inputs for
those summaries.

Consequently, a retained file's timestamp or presence in `results/` does not make it a
current result. The statistical semantics determine its status. Held-out-refit Wald-z
outputs remain compatibility association measures; participant-disjoint frozen-model
log-loss gains remain incremental predictive comparisons. Superseded versions are
available from Git history and are not runtime inputs.

## 13. Current limitations

- SBI is a same-content counterfactual proxy, not the verified token heard by each participant.
- 3-D t-SNE changes high-dimensional geometry. Full-dimensional sensitivity results remain important even though the reporting emphasis follows the established 3-D method.
- Compatibility three-fold z figures cannot replace OOF prediction.
- Absolute raw distances from dataset-specific t-SNE spaces are not directly comparable across datasets.

Remediable gaps, including the predictor-only likelihood criterion and B23 multi-talker exposure integration, are listed in [TODO.md](TODO.md) rather than treated as intrinsic limitations.

## 14. Implementation entry points

- Data and store registry: `cross_talker_generalization/configs/project.json`
- DTW and variability definitions: `cross_talker_generalization/src/ctg/metrics.py`
- Exposure pools: `cross_talker_generalization/src/ctg/exposure.py`
- Participant folds: `cross_talker_generalization/src/ctg/folds.py`
- GLMM fitting: `cross_talker_generalization/R/fit_confirmatory.R`
- Ceilings: `cross_talker_generalization/src/ctg/ceiling.py` and `ceiling_cv.py`
- Figures and reports: `cross_talker_generalization/src/ctg/report_*.py` and `final_report.py`
- Execution commands: `cross_talker_generalization/docs/RUNBOOK.md`
