# To-do

This file tracks work that can still change the production results. Scientific definitions are kept in [`cross_talker_generalization/docs/SCIENTIFIC_SPEC.md`](cross_talker_generalization/docs/SCIENTIFIC_SPEC.md).

## September 11 meeting and follow-up email

Latest handoff (September 18): Zhengyang has paused further tau/k optimization research as well as the previously paused HVE work. The two X21 interactive maps are packaged in a new [two-slide presentation](outputs/sbi_parameter_maps_2026-09-18/presentation/README.md). The [remaining-request review](cross_talker_generalization/analysis_update_2026-09-18/NEXT_FLORIAN_TASKS.md) separates unfinished explicit requests, completed diagnostics, conditional suggestions and unanswered decisions. Recommended next task: explain the high AN19 result using the completed audits. No further optimization run was launched.

September 18 additional sensitivity: the [X21 landscape excluding Talker-specific](cross_talker_generalization/analysis_update_2026-09-18/x21_parameter_landscape_no_talker_specific/README.md) is complete, including freshly standardized predictors, 1,276 refits, static/interactive figures and selected-predictor value tables. This checks removal of the self-comparison condition on one training split; it does not close the cross-fold evaluation or beyond-condition comparison tasks.

Current scope (September 18): **HVE and cross-theory matching are paused at Zhengyang's request.** Focus on SBI. A [small joint tau/k search test](cross_talker_generalization/analysis_update_2026-09-18/sbi_optimizer_probe/README.md) is complete (101 seconds, 39 distinct fits; one layer/split/seed). The subsequent [X21 dense landscape](cross_talker_generalization/analysis_update_2026-09-18/x21_parameter_landscape/README.md) is also complete: 1,247 grid pairs plus 29 limit fits, all converged in 15.83 minutes, with checked 3-D surfaces and profiles. Both objectives select the same grid pair; tau remains boundary-limited. Cross-fold/layer validation and the multi-seed optimizer comparison remain open; the dense map is a single-training-split diagnostic.

September 18 update: the [conditional-ceiling correction](cross_talker_generalization/analysis_update_2026-09-18/README.md) has been computed for all three datasets. AN19/X21 now retain condition, and X21 retains sentence context; B23 reproduces the previous reference. All held-out rows have reference predictions. Zhengyang confirmed signed z for the optimization comparison. Matching theory-specific response coverage and constructing compatible z-ceilings remain open.

September 17 post-run update: the [completed diagnostic batch](cross_talker_generalization/analysis_update_2026-09-17/PROGRESS_REPORT.md) contains 396 specifications, 4,752 paired-fold ratios and 18 layer profiles (6 SBI, 12 HVE). Coverage and score-arithmetic checks pass; one small X21-FT historical-repeatability difference is retained. HVE coverage includes all existing Tr-24 definitions and two overall definitions at all registered layers. This does not complete the separate optimization, ceiling or manuscript-panel tasks below.

The [source-linked Chinese checklist](outputs/meeting_2026-09-11/TODO_CN.md) consolidates the complete meeting transcript and Florian's supplied follow-up email. It preserves the 14 meeting topics and adds four email work items: report the actual data split, confirm the diagnostic ratio, score training/test consistently, and produce layerwise SI diagnostics. The email text is retained there; its original date was not supplied.

The current prediction runner uses participant-level three-fold train-test CV, with no independent inner tuning split. Layer/method selection reuses those CV scores. The confirmatory configuration fixes tau and does not search k. Historical optimizer sources have now been identified; mapping each historical figure to its exact run and completing a matched objective comparison remain open. In the [September 15 exchange](cross_talker_generalization/analysis_update_2026-09-17/CORRESPONDENCE_2026-09-15.md), Florian accepted mean test log loss divided by mean training log loss. Nested CV remains unanswered.

The September 16 update consolidated the task records. Implementation began on September 17; see the [analysis update](cross_talker_generalization/analysis_update_2026-09-17/README.md) for corrected acoustic components, matched training/test scoring, SI ratios and the common-random-structure sensitivity. The [daily work log](cross_talker_generalization/analysis_update_2026-09-17/WORK_LOG_2026-09-17.md) records completed runs, checks and remaining coverage for the later report. Earlier completed boxes describe their original outputs; they do not certify completion of the new matched-objective comparison. Existing unresolved work remains in scope.

The earlier work-package rationale is in the [next analysis plan](cross_talker_generalization/docs/NEXT_ANALYSIS_PLAN.md); use the checklist above for the latest meeting/email priorities. Figure 1 and Figure 2 inputs are tracked separately in the [main-figure specification](cross_talker_generalization/docs/MAIN_FIGURE_SPEC.md).

The [numbered meeting/email requirements review](cross_talker_generalization/docs/FLORIAN_REQUIREMENTS_REVIEW.md) distinguishes verified requests from implementation choices. The requested layerwise z/three-fold CI/ceiling format is the principal **SBI result display**, not a designation of manuscript main figures.

## Annotated-report follow-up (September 9)

See the [illustrated report](cross_talker_generalization/analysis_update_2026-09-09/REPORT_FOR_FLORIAN.md) and [complete comment response](cross_talker_generalization/analysis_update_2026-09-09/REVIEW_RESPONSE.md). This update covers all supplied PDF annotations and pasted Slack messages; it is not a complete audit of inaccessible private Slack history.

- [x] Preserve the supplied world map, phonological feature illustration and map R code; apply its exact language groups/colors to the revised figures.
- [x] Redraw Figure 1a/b with first/last latent dimensions, white background and the annotated native example "The wife helped her husband"; replace the earlier example without silently changing its transcript.
- [x] Add English reference talkers to the AN19 matrix (42 total); provide mean distance and mean similarity over 138 shared words on linear scales, with language-first labels and gray language blocks.
- [x] Add control-test marginal distributions and shared legends, and revise both condition-specific and black pooled X21 curves.
- [x] Simplify the SBI z legend/layout and recover all-layer three-fold likelihood companions with acoustic baselines; export every supported revised HVE likelihood profile. These retained likelihood and notebook-z runs are not a matched objective comparison.
- [x] Verify X21 SBI/HVE use the same rows/structure; clarify that condition adds to HVE (p=.01168) and that its predictor-only association is negative (z=-5.5038).
- [x] Correct the direct AN19/X21 ceiling keys to include condition; retain X21 sentence context. Recompute all three datasets and compare against preserved results (September 18).
- [ ] Apply one shared reference on matched SBI/HVE training and test rows, folds and statistical scope. The historical SBI/HVE ceiling tables are already identical; no theory-specific ceiling definition is required. The September 18 all-behavioral-row reference does not yet complete this matching.
- [ ] Match AN19 SBI/HVE rows and random-effects structures before comparing theories; also control the random-effects fallback across AN19 layers. Preserve separate B23 participant strata.
- [x] Rerun acoustic component diagnostics using the full baseline's coordinate standardizer; see the September 17 update. A separate fixed participant/item random-intercept sensitivity now matches structures across these components; original fits remain available.
- [x] Run full-corpus AN19 automatic phone annotation without a pilot gate: FALCON aligned all 6,261 recordings from the user-selected `July/Nygaard_audio` directory (18,785 non-silence intended-phone intervals); independent PhoneticXEUS returned 5,872 nonempty and 389 empty IPA hypotheses. See the [annotation record](cross_talker_generalization/analysis_update_2026-09-09/an19_phone_alignment/README.md). These are machine outputs, not verified human labels.
- [ ] Complete Figure 2d's listener-response-to-phone alignment, lexical-neighbor definition and behavioral analysis. Automatic acoustic annotations alone do not provide listener error labels.
- [ ] Define and compute the first-ten-exposure analysis separately from control-test curves.
- [ ] Check tone annotation and weighting; obtain the supplied phonology heatmap's score table/code and repair its clipped label in the source before final compound-panel assembly.
- [ ] Publish reviewed figures to Git/Overleaf when explicitly requested; this report update saved local outputs only.

## Immediate display and analysis corrections (September 6 review)

- [x] Re-read the available full September 1 automatic transcript and supplied email chain; record the requested work with timestamps and sources.
- [x] Generate [SBI/HVE z review figures](cross_talker_generalization/analysis_update_2026-09-06/z_value_review/README.md), clearly separating stored notebook test-refit z/ceiling from revised HVE training-fold z.
- [ ] Specify and run current-definition z plus a compatible three-fold behavioral z-ceiling; do not normalize training-fit z with a test-refit ceiling.
- [ ] Produce a current-definition matched SBI z/likelihood/ceiling analysis across all 18 registered layers. Retained all-layer confirmatory likelihood scores were recovered on September 9, but use a different predictor from the notebook-z figures.
- [ ] Correct September 6 B23 HVE objective selection: keep the 97- and 168-participant samples separate, or refit all candidates on identical response rows.
- [x] Rerun all 14 AN19 component diagnostics with `global_z` scaling, matching the full acoustic baselines; retain the earlier unscaled distances as historical outputs.
- [ ] Provide fold-bootstrap companion intervals where earlier email requested three-fold uncertainty; retain participant-bootstrap results under explicit labels.
- [ ] Obtain the current Overleaf ablation inventory and the additional Figure 2 materials referenced in email; do not claim complete manuscript coverage from placeholders.

## 1. Freeze the primary analysis and figure grammar

- [x] Add `M_null` with the same blocking terms and random-effects structure as `M_predictor`, so predictor-only OOF gain can be expressed as `loss(M_null) - loss(M_predictor)`.
- [x] Rerun fixed-`tr_24` SBI and HVE for AN19, X21, and B23, for HuBERT base and ASR fine-tuned representations, using all four models.
- [x] Normalize predictor-only gain to the directly cross-validated behavioral ceiling on exactly the same held-out rows.
- [ ] Validate the fixed-`tr_24` 2 × 2 robustness analysis on comparable B23 samples; the existing implementation is not yet a valid cross-stratum ranking.
- [x] Use participant-cluster 95% bootstrap intervals and the same upward-is-better layout for the fixed-`tr_24` SBI/HVE and acoustic-control batch.
- [ ] Retain Tr-24 as the common manuscript summary discussed in the meeting, and supply layerwise z as the principal SBI display; do not confuse display format with manuscript figure placement.
- [ ] Audit every title, legend, and caption so it states the fitted model, optimization objective, evaluation set, and meaning of all gray/reference lines.

## 2. Diagnose the AN19 acoustic baselines

- [x] Add non-copying diagnostic views of MFCC static, delta, delta-delta, C0, and static-without-C0 dimensions in the production feature reader.
- [x] Add non-copying STRF diagnostic views grouped by rate, scale, and direction.
- [x] Complete first-stage acoustic groups under matched preprocessing (September 17); full MFCC/STRF distances and scores were reproduced unchanged.
- [x] Run a common-random-structure sensitivity across all 16 AN19 acoustic features and six condition-specific baseline analyses (September 17): 352 selected fits, no failures; all-condition feature rankings unchanged. This does not replace the registered primary analysis or complete the AN19 SBI/HVE theory comparison.
- [x] Check duplicate/self-pairs, physical recording identities, condition-specific missingness, and identical held-out trial sets.
- [x] Fit full MFCC39/STRF24 separately within all six AN19 conditions (September 17). Both improve held-out prediction in the four unflagged conditions; see the analysis update for the two flagged conditions.
- [x] Identify the boundary-singular components in Korean-exposure/Spanish-test and Spanish-exposure/Spanish-test: 20 selected fits have approximately zero participant variance, with item variance retained and no other convergence messages. Keep these flags and interpret the affected cells cautiously; no extra random effects were removed to suppress warnings.
- [x] Compare MFCC/STRF distances with HuBERT distances on identical physical pairs.
- [ ] If one group dominates, run individual-dimension and leave-one-group-out diagnostics.

## 3. Build manuscript Figures 1 and 2

- [x] Regenerate Figure 1a/b at `tr_24`: actual HuBERT frame values and same-space sentence, word and phone trajectories.
- [x] Select an annotated English-055 example. The September 9 revision supersedes the earlier "fell" example with "The wife helped her husband", word "wife", and W/AY1/F.
- [x] Register the language-to-branch table; resolve AN19 Hindi/Mandarin labels using Table I of the original paper.
- [x] Compute a PHOIBLE 2.0 consonant/vowel inventory Jaccard proposal, with inventory-source ranges (not confidence intervals).
- [ ] Agree whether the narrower inventory construct is suitable for Figure 1d; broader phonotactics/suprasegmental claims require additional sources.
- [x] Generate Figure 2a on the 138 shared words. The September 9 revision restores all 42 talkers, including six English references, as requested in the PDF review.
- [x] Register the full automatic AN19 phone annotations and generate Figure 2b's [descriptive intended-phone deviation candidate](cross_talker_generalization/analysis_update_2026-09-09/an19_phone_alignment/figures/figure2b_automatic_intended_phone_deviation.png), plus three legible companion pages. This is not a manually validated phone-production result.
- [ ] Assess the Figure 2b candidate's coverage sensitivity before manuscript claims: only 5,114 of 16,086 L2 phone intervals have all six usable English references; the strict duration/posterior sensitivity retains 197. Word coverage differs across talkers and phones. No manual pilot gate is imposed on full-corpus processing.
- [x] Build the initial AN19/X21 control-condition talker-level intelligibility validation in Figure 2c.
- [x] Add matched-word Figure 2c curves for AN19 and X21 controls, with participant-bootstrap intervals and explicit missing-reference counts.
- [x] Recover the complete Figure 2d caption from `references/manuscript_draft.pdf`; inspect the supplied Desktop notebooks.
- [x] Add AN19 automatic acoustic phone labels directly on the full corpus, following the decision to proceed without a small manual pilot. The selected audio source and all 6,261 recording hashes are recorded; two Somalian filename-number differences are mapped by identical audio SHA-256. Figure 2d's remaining work is tracked above.
- [ ] Resolve AN19 HW74's lexical-label conflict by checking the recording content: all six English files say "wade", all 36 L2 files say "wave", and corresponding features exist. Do not automatically alias two distinct words by matching their item number; Figure 2c provisionally excludes the 40 affected responses.
- [x] Produce the [illustrated report for Florian](cross_talker_generalization/analysis_update_2026-09-06/collaborator_report/REPORT_FOR_FLORIAN.md), separating completed panels, proposals and missing inputs.

## 4. Design the specific-exposure-understanding extension

- [ ] Inventory participant-level exposure responses and determine which datasets record item-specific exposure accuracy.
- [ ] Define segment/word relatedness between each correctly perceived exposure token and each test word.
- [ ] Specify a model that separates item-specific transfer from participants' general exposure and test ability.
- [ ] Treat this as a later extension until the required annotations and estimand are agreed.

## 5. Rerun predictor selection and model comparisons

- [x] Rank candidate SBI and HVE specifications by three-fold held-out total log loss from `M_predictor` (predictor plus random effects, without condition).
- [ ] Apply the comparable-sample check to every ranking, including the September 6 report; the August 27 search already separated B23 strata.
- [x] Record full-data log likelihood, deviance, AIC, observation count, and convergence status for auditability.
- [x] Produce both comparisons after selection: `M_condition` versus `M_joint`, and `M_predictor` versus `M_joint`.
- [x] Correct SBI selection using the existing true OOF `M_predictor` scores and replace the old best-layer labels in the dated analysis update.
- [x] Rerun the complete revised HVE candidate sets for AN19, X21, and B23 and publish corrected best-method labels in the dated analysis update.
- [ ] Replace or retire the remaining old best-method labels in the broader August 21 report package.

## 6. Complete the revised HVE analysis

- [x] Add `overall_order_sensitive`, defined by concatenating complete exposure tokens in presentation order and including cross-token frame transitions.
- [x] Recover participant-level order for AN19 and X21.
- [x] Integrate the B23 public stimulus lists and training table, using the actual stimulus filename to resolve the documented speaker-label discrepancy.
- [x] Keep unordered B23 HVE available when trial indices are incomplete, while marking only `overall_order_sensitive` unavailable for those participants.
- [x] Run `overall_order_sensitive` across all 18 t-SNE layers for base and fine-tuned HuBERT in all three datasets.
- [x] Fit all 14 modelable B23 order-independent HVE measures across 18 layers and both HuBERT variants using predictor-only selection.
- [x] Fit both downstream comparisons for the two selected B23 order-independent predictors and add participant-cluster bootstrap intervals.
- [x] Recompute the revised AN19/X21 HVE candidates and regenerate the complete cross-method HVE figures.

## 7. Multivariable model analyses

- [ ] Prespecify the backward-selection removal rule and stopping criterion.
- [ ] At HuBERT layer 24, fit the combined SBI + HVE + MFCC + STRF model and perform the prespecified backward selection.
- [ ] Separately fit all-layer SBI and all-layer HVE models to test whether layers retain nonredundant information.

## 8. Uncertainty and sensitivity analyses

- [x] Add participant-cluster bootstrap intervals for both held-out comparisons in the global order-sensitive HVE analysis.
- [x] Extend the paired participant-cluster bootstrap to every selected HVE analysis.
- [x] Add the same paired participant-cluster bootstrap intervals to the selected SBI analyses.
- [x] Add a combined-test-fold nested GLMM likelihood-ratio analysis using fold-specific cross-fitted predictor values, while retaining OOF log loss as the predictive evaluation.
- [ ] Treat data-driven layer selection as exploratory or evaluate it with nested participant-level cross-validation before making confirmatory claims.
- [ ] Run the planned full-dimensional HuBERT and DTW path-length-normalization sensitivity analyses for the key conclusions.

## 9. Revised report and release review

- [x] Build the dated methodological update `analysis_update_2026-08-27` without overwriting `analysis_update_2026-08-21`.
- [x] Review the update's numerical inventories, candidate diagnostics, selected-model diagnostics, source tables, labels, and figures.
- [x] Update the root documentation so the August 21 broad report is not presented as the authority for corrected selection results.
- [ ] Decide whether to promote the dated update or rebuild the broad report after the remaining multivariable and sensitivity-analysis decisions are resolved.

The September 6 batch, source tables and Figure 1/2 placeholders are in [`cross_talker_generalization/analysis_update_2026-09-06/`](cross_talker_generalization/analysis_update_2026-09-06/README.md). It is a partial analysis package, not completion of all manuscript requirements; read its correction notice and the z-value review before reusing figures.
