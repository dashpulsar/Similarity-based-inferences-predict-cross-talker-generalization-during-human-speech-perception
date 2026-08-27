# Project description

## Overview

Human listeners routinely understand speech from talkers they have never heard before. This ability is remarkable because speech varies substantially across individuals: vocal-tract anatomy, accent, speaking rate, phonetic realization, and recording context all alter the acoustic signal. A central problem for speech-perception theory is therefore not only how listeners recognize a familiar token, but how experience with one set of speech exemplars supports generalization to another talker.

This project tests a computational account of that generalization. The central proposal is that exposure creates a basis for inference: when a test talker's speech is similar to the speech associated with prior exposure, recognition should be easier; when it is dissimilar, generalization should be weaker. The project operationalizes this proposal using acoustic features and learned speech representations, then evaluates the resulting predictors against human behavior in three experiments.

The goal is not to identify one universally best speech model or one layer with the largest statistic. The broader objective is to determine which representational relationships are behaviorally informative, whether those relationships replicate across experimental designs, and where they fail.

## Scientific framework

The project separates two mechanisms that are related but not interchangeable.

### Similarity-based inference

Similarity-based inference (SBI) concerns the relationship between exposure and test speech. For a test item, the analysis identifies recordings in which exposure talker(s) produce the same linguistic content. Their frame-level representations are aligned with dynamic time warping, producing an exposure–test distance. Smaller distance corresponds to greater similarity.

This predictor is intentionally described as a **same-content talker proxy**. In the available datasets, the comparison recording is not always the exact acoustic token presented to a particular participant. It therefore measures how similarly the relevant talkers realize matched content, rather than reconstructing a literal episodic memory trace.

### High exposure variability

Heard/exposure variability (HVE) concerns the internal structure of the speech set encountered during exposure. A set may contain tightly clustered or highly dispersed frame-level representations, repeated instances of the same linguistic type, or substantial differences among types and talkers. The project registers multiple measures that separate within-token, within-type, between-type, sequential-frame, and pairwise-DTW variability.

SBI asks whether exposure speech resembles test speech. HVE asks how heterogeneous the exposure set itself is. Keeping these estimands separate makes it possible to test whether generalization depends primarily on a match to the test talker, on exposure diversity, or on both.

## Why use computational speech representations?

The project treats an English-trained speech model as a computational observer: a defined transformation from waveform to a sequence of feature vectors. Such a model offers a reproducible way to compare speech at multiple representational stages.

HuBERT large was selected because its self-supervised pretraining provides a rich hierarchy of speech representations. Two variants are analyzed:

- **HuBERT base:** the English-pretrained model before ASR fine-tuning;
- **HuBERT ASR fine-tuned:** the same model family after optimization for speech recognition.

The comparison tests whether behavioral relevance is associated with generic learned speech structure, recognition-oriented fine-tuning, particular model depths, or lower-level acoustics. MFCC39 and STRF24 serve as acoustic baselines under the same DTW and statistical analysis contract.

The established method uses precomputed 3-D t-SNE trajectories for every registered layer. Because t-SNE can alter high-dimensional geometry, the project also retains full-dimensional HuBERT representations for sensitivity analysis. Results are interpreted within each dataset-specific representation space rather than treating t-SNE distances as globally calibrated units.

## The three datasets

The three experiments provide complementary tests rather than interchangeable replications. The three experiments are analyzed separately under a shared computational and statistical contract. Dimensionality reduction is conducted separately for each experiment and neural layer (or acoustic/perceptual baseline).

| Dataset | Participants | Task | Behavioral observations | Observational unit | Outcome | Tested generalization |
|---|---:|---:|---|---|---|---|
| AN19 | 160 | Word transcription | 24,960 total; 7,680 test rows | Word | Bernoulli | Across-talker within and across accents | Binary response |
| X21 | 320 | Sentence transcription | 16,477 | Word (within sentence) | Bernoulli | Within- and across talker within-accent |
| B23 | 195 | Sentence transcription | 11,700  | Sentence | Binomial | Within- and across talker within- or across accent |


### AN19

AN19 contains 160 participants and 24,960 behavioral observations, including 7,680 test-phase rows. The key comparison is word-level: exposure and test talkers are compared while producing the same word. Its relatively rich talker structure supports both behavioral prediction and a large all-talker validation matrix. In the current results, AN19 supplies the strongest evidence that computational similarity improves prediction beyond experimental condition.

### X21

X21 contains 320 participants and 16,477 binary-response observations. It includes control, multi-talker, single-talker, and talker-specific conditions. Similarity is defined for the same keyword within the same sentence. The talker-specific condition creates an informative self-comparison boundary, while repeated exposure tokens allow presentation-weighted and unique-token variability estimands to be distinguished. Current incremental predictive effects are present but substantially smaller than in AN19.

### B23

B23 contains 195 participants and 11,700 sentence-level observations. Each observation retains the number of correctly and incorrectly recognized keywords, so the appropriate outcome is count-binomial rather than one binary value per sentence. Same-content SBI is defined at the sentence level. The current SBI results are close to zero, making B23 important as a possible boundary condition.

The production exposure builder now integrates the public B23 stimulus lists and participant-level training table for all four single-talker and all four multi-talker conditions. It maps the stimulus filename actually presented, which resolves the speaker-label irregularity described in the paper. Revised B23 HVE candidate selection and downstream comparisons are reported in `analysis_update_2026-08-27`.

## Computational construction

Every recording is represented as a variable-length sequence of frame vectors. For HuBERT, the registry includes CNN layers 2–6, Transformer layer 0, and even-numbered Transformer layers 2–24, for 18 feature spaces per model variant.

Dynamic time warping aligns two sequences without assuming equal duration. The main method-reproduction profile uses Minkowski `tau=2` frame cost and mean-sequence-length normalization. This normalization matches the historical notebook computation. Optimal-path-length normalization, which appeared in the earlier written description, is implemented separately as a sensitivity analysis rather than silently substituted.

For multi-talker exposure, the primary predictor averages raw distance across exposure talkers. A minimum-distance profile tests the alternative hypothesis that the closest available exemplar dominates generalization. Descriptive plots may convert distance to bounded similarity with `exp(-k d)`, while confirmatory models use negative distance standardized from training-fold data only.

Exposure variability is constructed from identifiable exposure presentations. The registry contains 17 definitions: one order-independent overall measure, one global trial-order-sensitive measure, and five measure families at sentence, word, and phoneme levels. Unsupported or unavailable measures are recorded explicitly rather than dropped silently.

## Behavioral modeling

The statistical design prioritizes generalization to new participants. Participants are assigned to three deterministic, stratified folds. For each predictor and feature space, three GLMMs are compared:

```text
M_condition = experimental condition + registered random effects
M_predictor = computational predictor + registered random effects
M_joint     = experimental condition + computational predictor + registered random effects
```

Implementations of SBI or HVE are compared using the summed participant-held-out log loss of `M_predictor`, without the original condition predictor. Full-data likelihood and related fit statistics are retained for auditing. The selected predictor is then carried unchanged into comparisons with `M_condition` and `M_joint`.

A separate incremental analysis compares `M_condition` with `M_joint`. For this comparison, the model is fit on training participants, frozen, and scored on the held-out participant fold using population-level predictions. Improvement is measured as the reduction in binomial log loss from `M_condition` to `M_joint`.

This distinction corrects an ambiguity in the historical analysis. Refitting a GLMM on held-out responses and reporting its z value evaluates association in another subset, not prediction of unseen responses. The project retains these values only for compatibility with earlier figures and labels them accordingly.

## Major outputs

The project produces several complementary views of the evidence:

- layer-by-layer HuBERT association and OOF-prediction profiles;
- visualizations of layer-specific & hypothesis-specific predictions against actual human behavior
- MFCC39 and STRF24 comparisons under the same folds and GLMM contract;
- behavioral-ceiling-normalized compatibility figures with fold uncertainty;
- complete HVE profiles across all identifiable measures;
- matched-content all-talker distance matrices;
- raw-distance correlations among feature layers;
- sensitivity analyses for DTW normalization, `tau`, aggregation, dimensionality, and predictor transformation;
- figure source tables and SHA-256 provenance for the final report.

## Scope and limitations

- HuBERT is treated as an English-trained computational observer, not assumed to be a literal native-talker representation.
- The SBI predictor is counterfactual `same_content_talker_proxy`. It compares a test token with another recording of the same content by an exposure talker. That recording is not necessarily the exact token heard by a participant. The predictor therefore represents talker-level acoustic/representational correspondence, not a verified token-level memory trace.
- 3-D t-SNE can distort high-dimensional distance

## Interpretational constraints: things to keep in mind when interpreting the results
- Compatibility z/ceiling percentages are descriptive rescalings of Wald z, not variance explained or predictive accuracy.
- t-SNE distance is dataset- and layer-specific and cannot be compared in absolute units across datasets.
- Selecting the best HuBERT layer from the same held-out results is exploratory unless layer choice is prespecified or nested within cross-validation.


## Reproducible project structure

The production implementation is located in `cross_talker_generalization/`. Configuration files define inputs and estimands; Python performs auditing, pairing, DTW, aggregation, variability construction, and plotting; R/lme4 performs GLMM estimation; tests enforce numerical and project-level contracts; and the final report builder assembles figures, tables, interpretation notes, and provenance without overwriting existing releases.

For detailed navigation, see [FILE_GUIDE.md](FILE_GUIDE.md). Exact equations, variability definitions, statistical semantics, and known limitations are documented in [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md). Execution commands are provided in the [runbook](cross_talker_generalization/docs/RUNBOOK.md). Review-driven changes to this document are itemized in [DOCUMENTATION_CHANGES.md](DOCUMENTATION_CHANGES.md).
