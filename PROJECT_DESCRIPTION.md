# Project description

## Overview

Human listeners routinely understand speech from talkers they have never heard before. Speech varies across individuals because of vocal-tract anatomy, accent, speaking rate, phonetic realization, and recording context. This project studies how experience with one set of speech exemplars supports recognition of another talker.

The analyses operationalize relationships among speech exemplars using acoustic features and learned speech representations, then evaluate the resulting predictors against human transcription behavior from three experiments. The experiments are analyzed separately under a shared computational and statistical framework.

## Theoretical predictors

The project distinguishes two related hypotheses.

### Similarity-based inference

Similarity-based inference (SBI) concerns the relationship between exposure and test speech. For a test item, recordings of matched linguistic content from the relevant exposure and test talkers are aligned using dynamic time warping (DTW). Smaller distance corresponds to greater similarity.

The currently recoverable SBI predictor is a **same-content talker proxy**. The comparison recording is not always the exact acoustic token presented to a participant. It therefore measures how similarly the relevant talkers produce matched content rather than reconstructing a verified token-level memory trace.

### High exposure variability

High exposure variability (HVE) concerns the internal structure of the speech set presented during exposure. The analysis distinguishes variability within a token, among tokens of one linguistic type, among linguistic types, across adjacent frames, and in pairwise DTW distances.

SBI asks how similar exposure and test speech are. HVE asks how heterogeneous the exposure set is. They are constructed and evaluated as separate predictors.

## Speech representations

An English-trained speech model is used as a computational observer: a reproducible transformation from a waveform to a sequence of feature vectors. The project analyzes HuBERT large before and after ASR fine-tuning. Each variant contributes CNN layers 2–6, Transformer layer 0, and even-numbered Transformer layers 2–24. MFCC39 and STRF24 provide acoustic and spectrotemporal baselines.

The established analysis uses supplied frame-level 3-D t-SNE trajectories. Dimensionality reduction was performed separately for each dataset, model variant, and layer. Full-dimensional HuBERT representations are retained for sensitivity analysis because t-SNE can alter high-dimensional geometry.

Variable-length sequences are compared with DTW. The method-reproduction setting uses Minkowski `tau=2` and mean-sequence-length normalization. Alternative normalization, distance parameters, and multi-talker aggregation rules are available as prespecified sensitivity analyses. Exact formulas are documented in [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).

## Behavioral datasets

| Dataset | Participants | Task | Behavioral observations | Analysis unit | Outcome | Generalization contrast |
|---|---:|---|---:|---|---|---|
| AN19 | 160 | Word transcription | 24,960 total; 7,680 test rows | Word | Bernoulli | Across talkers, within and across accents |
| X21 | 320 | Sentence transcription | 16,477 | Keyword within sentence | Bernoulli | Within and across talkers within an accent |
| B23 | 195 | Sentence transcription | 11,700 | Sentence | Count-binomial keyword accuracy | Within and across talkers and accents |

### AN19

AN19 uses isolated words. The SBI comparison matches the same word across exposure and test talkers. Its exposure phase supports word-level HVE measures, and its talker inventory supports matched-word distances for all relevant talker pairs.

### X21

X21 includes control, multi-talker, single-talker, and talker-specific conditions. SBI matches the same keyword within the same sentence. Repeated exposure presentations allow presentation-weighted and unique-token HVE definitions to be distinguished.

### B23

B23 retains the number of correctly and incorrectly recognized keywords for each sentence, so it is modeled as a count-binomial outcome. SBI matches sentences across talkers. Single-talker and multi-talker exposure conditions each use 60 sentences.

The public data archive includes the stimulus lists and training data needed to recover the multi-talker sentence-to-recording assignments, including the presentation irregularity discussed in the paper. These sources have been located but are not yet integrated into the production exposure-pool builder. The integration and rerun are tracked in [TODO.md](TODO.md).

## Statistical questions

For each theoretical predictor, the pipeline fits three GLMMs:

```text
M_condition = experimental condition + registered random effects
M_predictor = theoretical predictor + registered random effects
M_joint     = experimental condition + theoretical predictor + registered random effects
```

The intended optimization criterion for comparing representations or parameterizations of a theoretical predictor is the likelihood of `M_predictor`, which deliberately excludes the original condition predictor. A separate analysis compares `M_condition` with `M_joint` to ask whether the theoretical predictor adds information beyond the experimental design factors. Participant-held-out predictions are used for that incremental comparison.

The current report builder implements the second comparison but has not yet implemented predictor-only likelihood as its selection rule. This discrepancy is documented in the technical reference and tracked as the highest-priority open task.

## Interpretation boundaries

- The SBI predictor is a same-content talker comparison, not necessarily the exact token heard by a participant.
- Compatibility figures that divide Wald z by a ceiling z rescale association statistics; they do not report variance explained or predictive accuracy.
- Distances in independently fitted t-SNE spaces are not directly comparable in absolute units across datasets or layers.
- Choosing a layer using the same data on which its performance is reported is exploratory unless the choice is prespecified or nested within cross-validation.

Repository navigation is in [FILE_GUIDE.md](FILE_GUIDE.md), implementation details are in [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md), executable commands are in the [runbook](cross_talker_generalization/docs/RUNBOOK.md), and remediable gaps are listed in [TODO.md](TODO.md).
