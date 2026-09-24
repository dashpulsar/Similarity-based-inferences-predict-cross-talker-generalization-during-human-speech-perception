# Next analysis plan

This plan records the analysis decisions from the September project meeting and the written follow-up. It separates results that already exist from work that still has to be run. The scientific definitions in `SCIENTIFIC_SPEC.md` remain authoritative.

The [numbered requirements review](FLORIAN_REQUIREMENTS_REVIEW.md) now distinguishes direct meeting/email requests, the latest SBI display clarification, and unfinished work. The [z-value specification](Z_VALUE_FIGURE_SPEC.md) governs the principal **SBI result display**; it does not designate manuscript main figures. Existing fixed-Tr-24 likelihood plots are companion analyses, not replacements for layerwise SBI z.

## What the current results support

- SBI is the more consistent theoretical predictor across AN19, X21, and B23. HVE is weaker and the best-looking HVE definition or layer is not stable enough to support a strong winner claim.
- Exact HuBERT layer, base versus ASR fine-tuned weights, 3-D versus full-dimensional representations, and local distance choices have not shown a simple, systematic ranking. These are robustness dimensions, not the main scientific conclusion.
- The primary paper-facing HuBERT analysis will therefore use Transformer layer 24 (`tr_24`) consistently. Layer searches remain secondary and explicitly exploratory.
- The unexpectedly strong AN19 MFCC39 and STRF24 results are unresolved. They must be diagnosed before the acoustic baselines can be interpreted as substantive evidence.

## Analysis and reporting contract

The model used to choose a theoretical predictor is `M_predictor`, which contains the theoretical predictor and the registered random-effects structure but not experimental condition. The downstream models are:

```text
M_null      = blocking terms + registered random effects
M_condition = condition + blocking terms + registered random effects
M_predictor = theoretical predictor + blocking terms + registered random effects
M_joint     = condition + theoretical predictor + blocking terms + registered random effects
```

The following quantities answer different questions and must not share an ambiguous label such as “performance”:

| Quantity | Question | Direction |
|---|---|---|
| Held-out log likelihood or log loss of `M_predictor` | How well does the theoretical predictor transport to unseen participants? | Higher likelihood or lower loss is better |
| Predictor-only gain, `loss(M_null) - loss(M_predictor)` | How much predictive information does the theoretical predictor add over the null model? | Up is better |
| Predictor Wald z in `M_predictor` | How strong and precise is the fixed theoretical-predictor association? | Larger positive z is stronger evidence in the predicted direction |
| `M_condition` versus `M_joint` | Does the theoretical predictor add information beyond condition? | Positive held-out gain or a supported nested comparison favors `M_joint` |
| `M_predictor` versus `M_joint` | Does condition add information beyond the theoretical predictor? | Positive held-out gain or a supported nested comparison favors `M_joint` |

The requested robustness display is a 2 × 2 analysis: select candidate predictors once by predictor-only likelihood and once by predictor-only z, and report each selected candidate using both likelihood and z. A fixed-`tr_24` HVE implementation exists in `analysis/model_comparison/reference_checks`, but its B23 ranking must be corrected to separate the 97-participant ordered and 168-participant order-independent samples. It is not completion of the full objective/ablation request. Any corresponding all-layer search remains exploratory and would require nested selection for a confirmatory predictive claim.

For paper-facing claims, `tr_24` avoids selecting a layer on the same data used to summarize it. If a data-selected layer is used for an out-of-sample claim, selection must occur inside each outer training split; held-out outcomes must never determine the selected candidate. Historical held-out-refit z values remain descriptive association summaries only.

## Standard figure grammar

Figure families share clear labels and named uncertainty units; their statistics are not interchangeable:

- improvement is plotted upward;
- the principal SBI layer plot uses signed z, three fold points and 95% bootstrap intervals across those three folds;
- its mean z-ceiling is 100%, with a gray ceiling interval and labeled nominal z significance lines, provided the z and ceiling have compatible fit scopes;
- likelihood/model-comparison companion plots retain their explicitly labeled participant-cluster intervals; fold-bootstrap versions are still required by the earlier email;
- the behavioral ceiling is shown as a horizontal reference with its uncertainty band when the plotted scale permits a valid same-observation normalization;
- MFCC39 and STRF24 appear before DNN representations when they are in the same panel;
- the title or subtitle states the model (`M_predictor`, `M_condition`, or `M_joint`), selection objective, representation, layer, and evaluation set;
- zero means “no gain” only in gain plots; raw-z and normalized-z plots use the nominal +/-1.96 references on the corresponding scale;
- line color, line type, and gray reference elements are explained in the legend or caption;
- raw loss is not used as the sole presentation scale when a gain representation is available.

## Work package 1: primary SBI/HVE result set

1. Deliver the requested SBI layerwise z presentation and complete HVE method displays; preserve the distinction between historical test-refit and revised training-fit z until a matched-scope current z/ceiling rerun is specified.
2. Retain the fixed-`tr_24` SBI and HVE fits for all three datasets and both HuBERT variants with `M_null`, `M_condition`, `M_predictor`, and `M_joint` as the common-layer summary discussed in the meeting.
3. Report predictor-only gain, predictor beyond condition, and condition beyond predictor with participant-cluster intervals.
4. Add the directly cross-validated behavioral ceiling on the same held-out observations. Do not divide by a ceiling computed from a different row set.
5. Produce the likelihood-selected and z-selected 2 × 2 robustness display on comparable samples. Showing layerwise z as the principal SBI display does not determine manuscript main/supplementary placement.
6. Preserve the September 1 combined-fold LRT as a supplementary association test; do not substitute it for frozen-model OOF evaluation.

## Work package 2: AN19 acoustic-control audit

The production reader now exposes diagnostic views of the existing AN19 acoustic HDF5 file without copying or changing it. The first-stage groups are:

- MFCC static coefficients, deltas, delta-deltas, C0 alone, and static coefficients without C0;
- STRF groups by temporal rate (2, 4, 8, 16), spectral scale (0.25, 0.5, 1.0), and direction.

Run these groups through the same pair construction, corpus scaling, DTW, participant folds, and GLMM code as the full baselines. Then check:

The first diagnostic batch did not satisfy the same-scaling requirement: its distance table records `none`, whereas the full baseline records `global_z`. The [September 17 rerun](../analysis/diagnostics/README.md) applies the full baseline's coordinate moments to all 14 subsets and reproduces the full MFCC/STRF scores. Random-effect fallback still differs across some groups/scopes; harmonize that structure before interpreting their ordering as controlled feature attribution.

1. whether the high result survives within each AN19 condition rather than merely separating experimental conditions;
2. whether any self-comparison, duplicate recording, item leakage, or missingness pattern drives the result;
3. whether one MFCC or STRF group accounts for most of the full-feature gain;
4. whether baseline distances are nearly collinear with HuBERT distances;
5. whether the result holds for predictor-only OOF gain as well as Wald z.

If a broad group dominates, run a second-stage individual-dimension and leave-one-group-out analysis. This audit is exploratory and should be labeled as such.

## Work package 3: Figures 1 and 2

The panel definitions, available assets, missing inputs, and draft captions are in `MAIN_FIGURE_SPEC.md`. Figure 1 explains the computational representation and language sample. Figure 2 validates that the representation captures production and perception structure before the SBI/HVE model-comparison results are introduced.

## Work package 4: standardized ablations

After the primary fixed-layer figures are frozen, render the following ablations in the same format:

- 3-D t-SNE versus full-dimensional HuBERT;
- HuBERT base versus ASR fine-tuned;
- Minkowski versus cosine local cost and declared `tau` values;
- historical mean-sequence-length versus DTW-path-length normalization;
- HuBERT, wav2vec 2.0, and Whisper results after source tables use the same model and metric definitions;
- HVE definitions, with unsupported dataset × measure cells shown rather than silently omitted.

## Work package 5: specific exposure understanding

This is a larger extension, not required to finish the current main result set. The target is not a participant's overall exposure accuracy. The predictor should ask whether correct perception of particular exposure segments or words changes recognition of related test words for that same participant.

A defensible implementation requires participant-level exposure responses, segment or subsegment alignment, a declared exposure-to-test relatedness function, and an analysis that separates item-specific transfer from general listener ability. Until those inputs and the estimand are fixed, this extension belongs in the plan rather than in the current results.

## Completion order

1. Freeze the SBI z display and the compatible z/ceiling scope, while retaining the meeting's common Tr-24 manuscript summary.
2. Run the AN19 acoustic audit and resolve or document the high baseline result.
3. Complete draft Figures 1 and 2 from verified source data.
4. Run the 2 × 2 objective/reporting analysis and standardized ablations.
5. Only then rebuild the broad presentation/report package.
