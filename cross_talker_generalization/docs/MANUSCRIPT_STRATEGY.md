# Cross-dataset reporting strategy

## Recommended central claim

A defensible claim is that self-supervised and ASR representations trained on English speech act as a computational observer, and that observer estimates exposure–test talker acoustic-phonetic similarity that may explain and predict cross-talker generalization across behavioral datasets.

“English-trained representation” is a property of model training history. “Native listener or talker representation” is a construct interpretation that requires behavioral evidence; the two should remain distinct.

## Analysis hierarchy

1. **Cross-dataset replication:** estimate standardized similarity effects, condition-incremental LRTs, and participant-held-out log loss separately for each dataset.
2. **Representation comparison:** compare HuBERT base and ASR fine-tuned effect profiles and OOF gain, not only the layer with maximum z.
3. **Dimensionality robustness:** retain 3-D t-SNE for established-method reproduction and full-dimensional HuBERT as key robustness evidence. t-SNE can distort global distance.
4. **Acoustic baselines:** run MFCC39 and STRF24 with the same pair table, DTW normalization, folds, and GLMM contract, then compare held-out improvement.
5. **Exposure variability:** keep SBI and HVE as different estimands. SBI uses a same-content counterfactual talker proxy; HVE uses identifiable heard-exposure pools.
6. **Mechanistic specificity:** use layer-distance correlations to quantify redundancy; do not treat correlated layers as independent replications.

## Primary statistical reporting

For every registered predictor, report:

- Standardized coefficient, 95% CI, and Wald z;
- One-degree-of-freedom `M_condition` versus `M_joint` LRT;
- Participant-disjoint frozen-model OOF log loss and gain over condition-only;
- Fit diagnostics and participant/item/talker handling;
- Benjamini–Hochberg q across an 18-layer scan, or a small set of prespecified layers.

Do not call Wald z variance explained, and do not call a z from a GLMM refit on held-out responses a prediction metric. Compatibility figures should state their association estimand explicitly.

## Design-specific caveats

- AN19 control/no-exposure cells do not have exposure-talker similarity and must not be assigned zero.
- X21 Talker-specific is a self-comparison and can have a constant predictor. Descriptive plots must label this boundary.
- B23 is a sentence-level count-binomial outcome, not one Bernoulli trial per sentence.
- B23 actual multi-talker HVE is unidentified without the missing assignment; an available-recording union is at most an `available_pool_proxy`.

## Suggested figure logic

- Figure 1: three experimental designs and separate SBI/HVE computational paths.
- Figure 2: HuBERT base/FT layer profiles emphasizing coefficient/CI and OOF gain; z is secondary.
- Figure 3: condition-specific percentile-bin S-curves for a prespecified layer, with Wilson intervals and constant-predictor labels.
- Figure 4: HuBERT versus MFCC/STRF held-out performance and layer raw-distance correlations.
- Figure 5 or supplement: all HVE measures, DTW normalization/`tau`/aggregation/full-vs-t-SNE sensitivities, and diagnostics.

## Decisions to freeze before new production runs

- Whether full-dimensional or 3-D t-SNE is primary;
- Primary `tau`, DTW normalization, and multi-talker aggregation;
- Prespecified layers or an ordered 18-layer family with multiplicity control;
- SBI/HVE joint-model formula;
- Participant-level pooled modeling versus effect-level cross-dataset synthesis;
- Restriction of B23 HVE to identifiable single-talker conditions if the assignment remains unavailable.

These choices should be recorded before inspecting new full-layer outcomes to avoid selecting methods by peak z.
