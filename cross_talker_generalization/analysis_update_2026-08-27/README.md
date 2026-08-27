# Model-selection and global exposure-order HVE update

This report adds the newly specified order-sensitive HVE predictor. Complete exposure tokens are concatenated in actual presentation order, and adjacent-frame transitions include token boundaries.

- Figure 01 corrects SBI representation selection: HuBERT layers are ranked by held-out `M_predictor` log loss. MFCC39 and STRF24 are shown only as reference lines.
- Figure 02 evaluates the six selected SBI predictors in both downstream held-out comparisons, with paired participant-cluster bootstrap intervals.
- Figure 03a selects the HuBERT layer using held-out log loss from the predictor-only GLMM. Lower is better; condition is not used for selection. Because the same folds select and summarize the layer, this layer choice is exploratory.
- Figure 03b evaluates the selected predictor in the two downstream comparisons. Positive bars mean that the joint model has lower held-out loss than the reduced model. Error bars are paired participant-cluster bootstrap 95% intervals (10,000 resamples).
- Figure 03c shows the revised B23 order-independent HVE selection across 14 measures and 18 layers. Every cell uses the same 168 trained participants.
- Figure 03d evaluates the two selected B23 order-independent predictors in the downstream comparisons.
- Figures 03e and 03f show the complete revised AN19 and X21 HVE candidate searches.
- Figure 03g compares the downstream held-out results for the selected AN19, X21, and B23 comparable-sample predictors.

`hve_predictor_only_selected_methods.csv` applies the predictor-only criterion only within comparable participant sets. All AN19 and X21 HVE candidates were rebuilt from the revised actual-exposure tables. B23 is reported in two strata: 14 order-independent measures using all 168 trained participants, and global order-sensitive HVE using the 97 participants with recoverable presentation order. These strata are not ranked against each other because their held-out observations differ.

The revised HVE search contains 1,404 feature candidates. Its 5,616 predictor-only full/fold fits completed without failed, singular, or non-converged fits. All selected SBI and HVE downstream fits also converged and were non-singular.

## SBI layers selected by predictor-only held-out loss

| Dataset | Variant | Layer | Mean log loss |
|---|---|---:|---:|
| AN19 | base | tr_10 | 0.619245 |
| AN19 | ft | tr_10 | 0.636033 |
| X21 | base | tr_14 | 0.482788 |
| X21 | ft | tr_22 | 0.483093 |
| B23 | base | tr_24 | 0.568780 |
| B23 | ft | tr_20 | 0.569008 |

For all six selected SBI candidates, the participant-bootstrap interval for predictor beyond condition is above zero. The corresponding interval for condition beyond predictor includes zero in every case. These intervals condition on a layer selected from the same folds, so this is strong exploratory evidence rather than a selection-adjusted confirmatory test.

## HVE candidates selected by predictor-only held-out loss

| Dataset | Comparable selection set | Variant | Measure | Layer | Mean log loss |
|---|---|---|---|---:|---:|
| AN19 | all_exposure_participants | base | overall_order_sensitive | cnn_3 | 0.689779 |
| AN19 | all_exposure_participants | ft | overall_order_sensitive | cnn_3 | 0.689779 |
| X21 | all_exposure_participants | base | order_word | cnn_6 | 0.488145 |
| X21 | all_exposure_participants | ft | order_word | cnn_6 | 0.488145 |
| B23 | order_independent_168_participants | base | between_type_sentence | tr_12 | 0.567963 |
| B23 | order_independent_168_participants | ft | between_type_sentence | tr_4 | 0.568003 |
| B23 | global_order_97_participants | base | overall_order_sensitive | tr_6 | 0.568654 |
| B23 | global_order_97_participants | ft | overall_order_sensitive | tr_20 | 0.568186 |

For the new global order-sensitive HVE, every paired participant-bootstrap interval in Figure 03b includes zero. The current held-out evidence therefore does not establish a stable incremental benefit beyond condition. B23 fine-tuned has a positive point estimate, but its interval also crosses zero.

For the B23 order-independent winners, the joint model has slightly higher held-out loss than the condition-only model in both variants. The full-data likelihood-ratio result and the held-out predictive result therefore point in different directions; the report treats the held-out comparison as the predictive evidence.

AN19 and X21 include every analyzed participant with exposure. B23 order-independent HVE includes all 168 trained participants. Its global order-sensitive analysis includes the 97 participants whose public trial indices uniquely recover all 60 presentation positions. The remaining 71 are not assigned a guessed order.
