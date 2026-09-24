# Cross-talker Generalization: 15-minute Speaker Script

This script accompanies `cross_talker_generalization_full_presentation.pptx`.

Target duration: approximately **14 minutes 30 seconds**, leaving a short buffer for pauses and transitions.

## Slide 1 — Title

**Time: 0:00–0:35**

Hi everyone. I will give a brief recap of our cross-talker generalization project, including the question we are asking, the analysis framework we have now standardized, and the current results across the AN19, X21, and B23 datasets.

The central question is whether representations from an English-trained speech model can predict how human listeners generalize from previously heard speech to a new talker or test item.

## Slide 2 — X21 descriptive pattern

**Time: 0:35–1:30**

The descriptive observation behind the project is easiest to see in X21. The four colors separate the Control, Multi-talker, Single talker, and Talker-specific conditions, while the panels separate the four test talkers.

Within the conditions, higher model similarity is generally associated with higher listener accuracy. This is important because it suggests that similarity is not simply reproducing the average difference between experimental conditions.

This figure uses the HuBERT base Tr-24 representation as a descriptive visualization. The formal question is whether SBI improves participant-held-out prediction beyond condition, which I will return to later.

## Slide 3 — Two hypotheses

**Time: 1:30–2:25**

We currently distinguish two related theoretical predictors.

The first is SBI, or similarity-based inference. SBI is trial-specific: it compares the speech heard during exposure with test. The hypothesis is that a closer exposure–test match should support better generalization.

The second is HVE, or hypothesis-space variability. HVE summarizes the structure of the entire exposure set—for example, how dispersed the tokens are or how much their representations change in sequence.

## Slide 4 — Three datasets

**Time: 2:25–3:05**

We evaluate the account in three experiments: AN19 with 160 participants, X21 with 320, and B23 with 195.

The experiments differ in their talkers, stimuli, conditions, and response structures. B23 also uses aggregated count data rather than the same trial-level binary format as AN19 and X21.

The matrices on the right side illustrate that the model-derived talker geometry also differs across datasets. The goal is therefore not to force identical numerical results, but to apply the same analysis logic across all three datasets.

## Slide 5 — Representations

**Time: 3:05–3:50**

For HuBERT, we evaluate 18 feature spaces from both the base and ASR fine-tuned models. These include convolutional layers, and the even transformer layers through layer 24.

We also retain MFCC-39 and STRF-24 as acoustic baselines.

The correlation matrices show that distances from neighboring layers are related, but the layers are not interchangeable. This is why representation selection is an empirical part of the analysis rather than assuming in advance that the last layer must be best.

## Slide 6 — Constructing SBI

**Time: 3:50–4:50**

The SBI pipeline has four main steps.

First, we extract frame-level hidden states from each HuBERT layer. Second, following the existing project method, we reduce each layer to three dimensions using t-SNE. Third, we align exposure and test sequences with dynamic time warping, using minkawsiki distance. Finally, we standardize distance, so that a larger predictor value means greater exposure–test similarity.

The present primary pipeline preserves the historical mean-sequence-length normalization. Full-dimensional representations and alternative DTW normalization remain useful sensitivity analyses, but they are not replacing the main t-SNE analysis in this recap.

## Slide 7 — Constructing HVE

**Time: 4:50–5:55**

HVE is not a single measure. The current registry contains 17 candidate definitions.

Order-independent measures quantify properties such as overall dispersion, within- or between-type variability, and mean dissimilarity. Order-sensitive measures quantify changes between successive frames or tokens.

For the global order-sensitive definition, we concatenate the exposure tokens in their actual presentation order and include transitions across token boundaries. This makes trial order necessary for that global measure. By contrast, within-word or within-sentence order-sensitive measures can often be computed without knowing the order of separate trials.

The heatmap illustrates how candidate HVE definitions and layers are evaluated using the same held-out observations.

## Slide 8 — Statistical contract

**Time: 5:55–7:10**

This slide summarizes the statistical contract.

We fit three models: a predictor-only model, a condition-only model, and a joint model containing both the theoretical predictor and condition. All include the relevant random-effects structure.

Predictor selection is based on the three-fold held-out performance of the predictor-only model. Within each fold, the model is fitted to two participant folds. Its fixed effects are then frozen and used to predict the third fold. We compare the resulting out-of-fold log loss.

This is important because it prevents us from calling a z-value obtained after refitting the test data a cross-validated prediction. The ceiling shown on the right represents the strongest participant-held-out performance available from the human-response information used in this analysis.

## Slide 9 — SBI selection

**Time: 7:10–8:05**

The best SBI representation is not the same across datasets.

For AN19, transformer layer 10 is selected for both base and fine-tuned HuBERT. For X21, the selected layers are transformer 14 and 22. For B23, they are transformer 24 and 20.

The key point is not that any one layer is universally optimal. Instead, the predictive geometry is distributed across the HuBERT hierarchy. We also do not see a consistent advantage from ASR fine-tuning: sometimes the fine-tuned model is better, sometimes the base model is better, and often the difference is small.

## Slide 10 — SBI held-out results

**Time: 8:05–9:25**

This is the main SBI result.

The first comparison asks whether adding SBI to condition improves held-out prediction. The second asks whether adding condition to SBI improves prediction.

The benefit of SBI beyond condition is largest in AN19, smaller but still positive in X21, and close to zero in B23. Across the six selected base and fine-tuned representations, the participant-bootstrap confidence intervals for SBI beyond condition are above zero.

By contrast, the confidence intervals for condition beyond SBI include zero. This suggests that the selected SBI predictor captures much of the predictable structure represented by the experimental condition variable.

These estimates are conditional on the current predictor-selection procedure, so a fully nested selection analysis remains an important robustness step.

## Slide 11 — HVE selection

**Time: 9:25–10:25**

HVE produces a more heterogeneous picture.

The selected global order-sensitive layers differ across datasets and between the base and fine-tuned models. In B23, we also evaluate a broader set of order-independent HVE definitions.

There is an important sample-size distinction. The global-order B23 analysis currently uses 97 participants with usable order information, whereas the order-independent analysis uses 168 participants. Because these are different participant pools, we should not directly rank their log-loss values as if they came from the same held-out observations.

This illustrates why HVE results must always be reported together with the exact definition and analysis sample.

## Slide 12 — HVE held-out results

**Time: 10:25–11:40**

The held-out HVE results are much weaker than the SBI results.

For the selected global order-sensitive predictors, the confidence intervals for HVE beyond condition include zero. In B23, the selected order-independent HVE also does not improve the joint model; its estimated held-out gain is slightly negative.

Condition beyond HVE is positive in AN19 and X21, although the uncertainty is substantial. Overall, these results do not support the same stable predictive benefit that we observe for SBI.

This does not necessarily mean that exposure variability is irrelevant. It means that the current HVE definitions do not yet transport reliably to unseen participants under the present evaluation framework.

## Slide 13 — Combined-fold likelihood-ratio tests

**Time: 11:40–13:10**

Florian suggested adding a conventional nested-model comparison over the combined test-fold observations. We therefore use cross-fitted theoretical predictor values for all participants, fit the three GLMMs to the combined data, and run likelihood-ratio tests between the nested models.

For SBI beyond condition, these tests are significant in AN19 and X21, while B23 is around p equals .067. This broadly agrees with the held-out SBI pattern.

For HVE, the interpretation is more interesting. AN19 is not significant and X21 is approximately .053, but both B23 HVE families produce significant likelihood-ratio tests.

This is not identical to the out-of-fold result. The LRT asks whether the cross-fitted predictor is associated with behavior in the combined sample after model fitting. The out-of-fold comparison asks whether a model trained on other participants transports well enough to reduce prediction error. A predictor can show association without producing a stable held-out gain.

These p-values are currently selection-conditional and uncorrected, so I treat them as supplementary rather than replacing cross-validation.

## Slide 14 — Conclusions

**Time: 13:10–14:40**

To summarize, I would take three conclusions from the current results.

First, exposure–test similarity is the most consistent computational signal. SBI predicts unseen participants most clearly in AN19 and X21, with B23 closer to the boundary.

Second, ASR fine-tuning does not provide a consistent advantage over the self-supervised HuBERT base model.

Third, HVE is more sensitive to its mathematical definition, the available participant pool, and whether we evaluate association or held-out prediction.

The main next steps are to nest predictor selection inside the evaluation procedure, complete the planned full-dimensional and DTW-normalization sensitivities, and compare SBI, HVE, condition, MFCC, and STRF within a unified model-comparison framework.

So the current working conclusion is that similarity between exposure and test speech generalizes more reliably than variability within the exposure set. Thank you.

## If time is running short

To reduce the talk to approximately 12 minutes:

- On Slide 4, state only the three dataset names and participant counts.
- On Slide 5, omit the exact layer list.
- On Slide 7, describe only the distinction between order-independent and global order-sensitive HVE.
- On Slide 9, omit the exact selected-layer numbers.
- On Slide 13, report only the contrast between significant combined-fold B23 HVE association and weak held-out HVE prediction.

## Likely questions and short answers

### Why use 3-D t-SNE rather than the full-dimensional HuBERT representations?

The 3-D t-SNE pipeline is the established primary method in this project and matches the analysis our earlier work and current hypotheses were built around. Full-dimensional representations are planned as a sensitivity analysis rather than a replacement for the main analysis.

### Is the combined-fold LRT cross-validation?

The theoretical predictor values are cross-fitted, because each participant receives values selected without their own outcome fold. However, the final GLMM and likelihood-ratio test are fitted to the combined data. It is therefore a supplementary association test, not the same estimand as fully out-of-fold predictive gain.

### Why can HVE be significant in B23 in the LRT but not improve held-out prediction?

Association after combined-data model fitting and transport to unseen participants are different criteria. The B23 result suggests that HVE may contain behavioral signal, but that the estimated relationship is not yet stable enough to improve participant-held-out prediction.

### Does fine-tuning improve the model?

Not consistently. The selected base and fine-tuned representations perform similarly overall, and the direction of the difference changes across datasets.

### What is the most important remaining methodological issue?

The highest-priority robustness step is to nest representation and HVE-definition selection inside the outer participant-held-out evaluation, so that uncertainty from selection is included in the final performance estimate.
