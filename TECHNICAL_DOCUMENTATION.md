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

## 6. Exposure variability: 16 registered measures

An exposure pool contains tokens represented as frame × dimension sequences, each with a token ID and a type ID. Generalized dispersion is computed without taking the `1/tau` root:

```text
V(A) = mean_i sum_r |a_ir - mean(A)_r|^tau
```

At `tau=2`, this is mean squared Euclidean deviation, not ordinary Euclidean standard deviation. There are 16 registered measures: one overall measure plus five families at sentence, word, and phoneme levels.

| Measure | Definition |
|---|---|
| `overall` | Pool all available exposure frames and compute generalized dispersion |
| `within_token_{unit}` | Compute frame dispersion within each token, then average across tokens |
| `within_type_{unit}` | Compute each token center, dispersion among centers within a linguistic type, then average across types |
| `between_type_{unit}` | Compute one center per linguistic type, then dispersion among type centers |
| `order_{unit}` | Mean powered distance between adjacent acoustic frames within each token; order is frame order, not trial order |
| `mean_dissimilarity_{unit}` | Mean pairwise DTW distance within each type, then mean across types |

The set is `overall` plus `within_token_*`, `within_type_*`, `between_type_*`, `order_*`, and `mean_dissimilarity_*` for three units: `1 + 5×3 = 16`.

Availability differs by dataset:

- AN19 contains isolated-word exposure and supports `overall` plus the five word-level measures. Sentence and phoneme variants are explicitly unsupported.
- X21 reconstructs 80 exposure presentations. In Single-talker and Talker-specific conditions, each of 16 tokens is repeated five times. The primary HVE estimand preserves presentation weighting; a unique-token version remains available.
- B23 supports four identifiable single-talker pools. The repository does not contain the full 20/20/20 sentence-to-talker assignment for multi-talker exposure, including the noSPA presentation error. Those actual-exposure cells are marked `blocked`; the union of available recordings is not substituted as if it were the heard set.

## 7. Participant-disjoint three-fold analysis

Folds are defined by participant, not trial. `seed=230519`, `n_folds=3`, with stratification by dataset-specific design or condition cells. Each participant belongs to exactly one fold.

Each predictor is evaluated with:

```text
M_condition = original experimental condition + registered random effects
M_predictor = predictor_z + registered random effects
M_joint     = original condition + predictor_z + registered random effects
```

AN19 and X21 use binary responses. B23 retains correct/incorrect keyword counts per sentence and uses a count-binomial `cbind(correct, incorrect)` outcome; a sentence is not treated as one Bernoulli trial.

R/lme4 fits the GLMMs. Full-data models report coefficient, standard error, 95% CI, and Wald z. The likelihood-ratio comparison of `M_condition` and `M_joint` tests whether the predictor contributes association beyond condition. Random-effects structure, convergence, and singularity are recorded in diagnostics rather than silently altered.

## 8. True OOF prediction versus compatibility z

These quantities have different meanings.

**True cross-validated prediction:** a GLMM is fit on two training folds and frozen. It predicts the third, entirely unseen participant fold at the population level without held-out participant random effects. The three held-out partitions form the OOF predictions. Primary incremental performance is:

```text
OOF gain = logloss(M_condition) - logloss(M_joint)
```

Positive values mean that adding the predictor improves prediction for unseen participants; negative values mean worse prediction.

**Held-out-refit Wald z:** after selecting one fold, the GLMM is refit directly on that fold's behavioral responses and its z is recorded. Because those responses were used for fitting, the statistic measures association stability across participant subsets, not out-of-sample prediction. The historical method is preserved for compatibility figures but is not rerun by default.

## 9. Behavioral ceiling

The ceiling also respects participant folds. For each fold, item-level accuracy or log odds are estimated from the other two folds and attached to held-out responses. The project provides:

- Direct OOF item-probability ceiling predictions evaluated by log loss;
- Three compatibility z values from refitting a ceiling predictor on each held-out fold.

Figure 00 and its variability analogue define 100% as the mean of the three compatibility ceiling z values. A gray 95% band around 100% represents fold uncertainty. `z / mean(z_ceiling) × 100` is only a rescaling of association statistics; it is not variance explained, predictive accuracy, or the percentage of human behavior captured.

## 10. Figures and interpretation

- **Figure 00:** MFCC39 and STRF24 appear first, followed by 18 HuBERT layers. Gray dots are fold-specific held-out-refit z values; black points/lines show the fold mean and fold-bootstrap 95% interval; the gray band represents ceiling uncertainty. This is a compatibility association figure.
- **Figures 01–02:** participant-held-out frozen-model OOF log-loss gain asks whether a representation improves prediction for unseen participants. The zero line means no incremental prediction, not a significance threshold.
- **Figure 03 series:** the compatibility panel matches Figure 00 semantics, while the core and dataset-specific profiles report true OOF results across every available HVE method.
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

DTW and HVE tasks are parallelized by feature layer. Each worker owns its HDF5 handle, and BLAS threads within a worker are restricted to one to avoid `jobs × BLAS threads` oversubscription. The default is eight jobs. Independent talker-matrix pairs may use up to 32 threads, bounded by CPU count and task count. Full-dimensional runs should reduce jobs if memory pressure is high.

Provenance JSON records random seed, input paths and hashes, runtime versions, parameters, output hashes, and status counts. Stable IDs and explicit join keys connect tables. Unavailable, unsupported, or blocked cells are recorded as states rather than silently removed or guessed.

### Result authority and retention

The reviewed package at `cross_talker_generalization/final_report_2026-08-21/` is the
authoritative presentation layer. `cross_talker_generalization/artifacts/` supplies the
refactored true OOF SBI and HVE products. The top-level `results/` directory is retained
only where the report builder still requires historical notebook-compatible summaries,
S-curve source files, validated AN19 talker-distance inputs, or direct audit inputs for
those summaries.

Consequently, a retained file's timestamp or presence in `results/` does not make it a
primary result. The statistical semantics determine its status. Held-out-refit Wald-z
outputs remain compatibility association measures; participant-disjoint frozen-model
log-loss gains remain the predictive results. Superseded duplicates are recoverably
archived under `recycle_bin/results_old_20260821/` and are never runtime inputs.

## 13. Current limitations

- SBI is a same-content counterfactual proxy, not the verified token heard by each participant.
- 3-D t-SNE changes high-dimensional geometry. Full-dimensional sensitivity results remain important even though the reporting emphasis follows the established 3-D method.
- B23 multi-talker actual-exposure HVE is not recoverable without the missing assignment table.
- Compatibility three-fold z figures cannot replace OOF prediction.
- Absolute raw distances from dataset-specific t-SNE spaces are not directly comparable across datasets.

## 14. Implementation entry points

- Data and store registry: `cross_talker_generalization/configs/project.json`
- DTW and variability definitions: `cross_talker_generalization/src/ctg/metrics.py`
- Exposure pools: `cross_talker_generalization/src/ctg/exposure.py`
- Participant folds: `cross_talker_generalization/src/ctg/folds.py`
- GLMM fitting: `cross_talker_generalization/R/fit_confirmatory.R`
- Ceilings: `cross_talker_generalization/src/ctg/ceiling.py` and `ceiling_cv.py`
- Figures and reports: `cross_talker_generalization/src/ctg/report_*.py` and `final_report.py`
- Execution commands: `cross_talker_generalization/docs/RUNBOOK.md`
