# Meeting and correspondence review: requested analyses and figures

**September 9 follow-up:** the [new illustrated report](../analysis/speech/REPORT_FOR_FLORIAN.md) and [comment-by-comment response](../analysis/speech/REVIEW_RESPONSE.md) incorporate all 20 supplied PDF annotations, the pasted Figure 1/AN19 Slack exchange and the supplied map/heatmap files. They supersede the older example, 36-talker matrix and provisional palette below. Full private Slack history remains unverified. The updated report also corrects the ceiling explanation and X21 HVE interpretation, while separating figure changes from analyses still to run.

Updated 2026-09-06. This is a requirements review, not a claim that the manuscript figures are complete. Start with the [z-value review package](../analysis/model_comparison/reference_checks/z_value_review/README.md) for the figures generated during this review. Executable outstanding work is tracked in the [project TODO](../../TODO.md).

## Evidence and an important distinction

**Follow-up source recovery:** the [illustrated report for Florian](../analysis/model_comparison/reference_checks/collaborator_report/REPORT_FOR_FLORIAN.md) now includes computed 1a/b/c and 2a/c, plus an explicitly narrower 1d inventory proposal. The supplied Desktop notebooks contain methods/S-curves, not the missing AN19 phone-error results. The local manuscript supplies the complete 2d caption and an ablation outline; it is not verified as the current Overleaf version. It also requests a Bonferroni reference in Figure 5: the testing family must be specified before replacing the currently labeled nominal z lines.

This review uses the complete available timestamped automatic transcript of the September 1 meeting, the supplied September 2 email chain, the Figure 1/2 captions supplied in the conversation, and earlier Slack/email messages quoted by Zhengyang. The transcript contains recognition errors and no reliable speaker labels; timestamps below locate discussion, not certified verbatim quotations. Private correspondence and the transcript are not copied into the public repository. The current Overleaf ablation list and the additional Figure 2 materials mentioned in email were not supplied, so their complete contents have not been verified.

The meeting does **not** support saying that Florian requested only z or only likelihood:

- **12:56–14:49:** show uncertainty across layers; use Tr-24 as the common primary manuscript representation, with layer trends in additional analyses rather than claims about a uniquely best layer.
- **30:32–35:00:** retain a ceiling with uncertainty, distinguish predictor z from model likelihood, and optimize for each objective and display both outcomes. The September 2 email explicitly repeats this last request.
- **05:56–08:27:** the likelihood-based display should be upward-is-better and use item-by-condition behavioral self-prediction as its reference. A held-out empirical ceiling can be surpassed.
- **Zhengyang's latest clarification:** layerwise z, three-fold intervals, normalized ceiling, and significance references are the main **SBI result display**, not a designation of the manuscript's main figures.

Accordingly, the **SBI display is z-first**. This does not renumber or replace manuscript Figures 1/2, decide main-versus-supplementary placement, or remove the common Tr-24 summary and likelihood/model-comparison analyses. A likelihood gain must never be renamed z, and a ceiling for one statistic must not normalize another statistic.

## 1. Separate optimization from evaluation and reporting

**Request:** optimize the theoretical predictor without experimental condition; examine likelihood-based and z-based objectives, and report both metrics for each. This applies to both SBI and HVE. Sources: September 2 email, first point; meeting 30:32–35:00; quoted Slack clarification.

**Required outputs:** a two-by-two objective/reporting comparison, plus layerwise profiles. Identify what was optimized: layer, HVE definition, distance parameter, or similarity transformation. Do not call a fixed-parameter run a parameter optimization. Fitting each GLMM still uses likelihood; a z-based outer search ranks theoretical-predictor configurations by the signed coefficient z, not by changing `glmer`'s fitting objective.

**What exists:** August 27 predictor-only held-out selection and September 6 fixed-Tr-24 HVE objective/reporting plots. These are exploratory searches, not selection-independent validation. September 6 did not complete every layer/parameter ablation requested by the email.

**Correction needed:** the September 6 HVE selector pools candidates with different observations in B23. Global order-sensitive HVE uses 97 participants; order-independent HVE uses 168. Total log loss cannot rank those sets against each other. Retain separate strata or refit on identical rows. Dividing by trial count alone does not make different participant samples comparable. The earlier August 27 report already distinguished these strata.

**Acceptance:** identical participant/item response keys and response counts within each candidate comparison; explicit objective and fit/evaluation scope; both reporting metrics; no choice using held-out outcomes described as independent validation. Register the fold-local parameter search before rerunning it.

## 2. Use one clear figure style, including z across layers

**Request:** identical formats for ablations; uncertainty on all points/bars; behavioral reference and an explanation of every line. Sources: August 28 email, September 2 email second point, meeting 12:56–14:49 and 30:32–31:11. The precise z-first format below follows Zhengyang's latest instruction and earlier example.

**Required z plots:** AN19, X21 and B23; HuBERT base and ASR-FT; MFCC39 and STRF24 first on the x-axis with a gap before HuBERT layers; signed z; three individual fold points; mean and 95% fold-bootstrap interval; mean ceiling at 100% with a gray 95% band; labeled nominal z reference lines. Show every supported HVE definition rather than only overall variability or the winning method.

The available registered layer set has **18 feature spaces**, not all 32 network outputs: CNN-2–6, Tr-0, and even Tr-2–24. Missing layers must not be interpolated or described as measured. Use “HuBERT base” for the non-ASR-fine-tuned variant, not as a claim that the checkpoint is the smaller HuBERT Base architecture.

**What exists now:** the new review script redraws the stored notebook-style SBI and HVE z tables with the requested normalized format, retaining their historical-data labels. It separately plots revised HVE training-fold z across all available layers and methods. Those new-definition z values are not divided by the old test-refit ceiling.

**Still required:** a current-definition, matched-scope three-fold z-and-ceiling rerun, especially for revised B23 exposure pools and global order-sensitive HVE; all-layer current SBI coefficient outputs; standardized ablations. The [z plotting specification](Z_VALUE_FIGURE_SPEC.md) explains the exact formulas and the remaining fit-scope decision. Existing z/ceiling ratios are descriptive evidence ratios, not percent accuracy or percent variance explained.

## 3. Complete manuscript Figures 1 and 2 before treating the package as paper-ready

**Request:** the two compound figures explain the method and validate speech production/perception structure before the theory comparison. Sources: September 2 email third point and follow-up, supplied captions; meeting opening and 35:17–35:59.

| Panel | Requested content | Existing material | What remains |
|---|---|---|---|
| 1a | Recording → HuBERT frame vectors → 3-D t-SNE | New Tr-24 waveform/full-feature/t-SNE panel | Review example and layout; checkpoint and fit scope recorded |
| 1b | Segment, word and sentence trajectories in one space | New English-055 "fell" F/EH1/L three-scale panel | Review annotated example; one-frame EH1 explicitly shown as a point |
| 1c | All represented L1s, colored by language branch | New checked 17-label language table and coverage plot | Distinguish available corpus from exposure-only coverage |
| 1d | L1-to-English sound-structure similarity | New computed PHOIBLE inventory proposal | Agree on the narrower construct or supply broader sound-structure sources |
| 2a | Word-matched AN19 pronunciation similarity by L1 | New 36-L2-talker subset, 138 shared words | Review similarity/log-color labeling; six English references excluded |
| 2b | Selected phoneme deviations from English reference, talker/L1 means and CI | Full automatic FALCON alignment now available for 6,261 AN19 recordings from `July/Nygaard_audio`; independent PhoneticXEUS hypotheses retained separately | [Automatic candidate generated](../analysis/speech/an19_phone_alignment/README.md): all-phone tables and twelve preselected panels, with 5,114 six-reference-complete L2 intervals. Coverage-sensitive, not manually verified; single-talker L1 groups have no CI. |
| 2c | Control intelligibility versus English-reference similarity: AN19 KOR/SPA and X21 Mandarin | New matched-word curves with participant-bootstrap uncertainty | Descriptive association, not CV GLMM; 40 AN19 responses provisionally excluded for HW74 wave/wade label conflict, not established missing audio |
| 2d | AN19 control segment errors versus deviation and lexical constraint | Complete caption recovered; automatic acoustic phone intervals now available | Still requires listener-response pronunciation alignment, lexicon/minimal-pair-neighbor rule and behavioral model; recognizer errors cannot replace listener errors. |

The draft images are in [figure_drafts](../analysis/model_comparison/reference_checks/figure_drafts/README.md). Their presence does not mean these eight panels are complete. No synthetic phonological scores or illustrative error rates may fill the missing panels. The detailed panel specification is [MAIN_FIGURE_SPEC.md](MAIN_FIGURE_SPEC.md).

## 4. Explain the high AN19 MFCC/STRF result

**Request:** investigate whether it is a bug and identify the acoustic features carrying the association. Sources: September 2 email fourth point; meeting 27:31–30:32.

**What exists:** full-baseline fits, component groups, distance correlations, pair checks, and pooled-model loss broken down by condition. These do not yet establish a causal feature contribution or exclude every preprocessing/confounding issue.

**Newly verified issue:** the existing diagnostic distance table records `coordinate_scaling=none`; the full-baseline table records `global_z`. Therefore the diagnostic groups were **not** run through identical preprocessing. The earlier completion claim was too strong. Do not interpret the reported component ordering as a controlled feature ablation. This mismatch does not by itself show that the original high full-baseline z is wrong.

**Next outputs:** rerun component groups with matched scaling, folds, physical recording pairs and rows; plot z and likelihood side by side; check within-condition slopes; add single-dimension/leave-one-group-out analyses where useful. Report negative as well as positive effects. Correlation with HuBERT distances is a separate diagnostic, not proof of behavioral explanatory equivalence.

## 5. Plan an item-specific exposure-understanding extension

**Request:** test whether correctly understanding particular exposure sounds/words helps recognition of related test words, not merely whether good listeners are good in both phases. Sources: September 2 email fifth point; meeting 20:38–25:09.

**Status:** extension design, not a completed analysis and not a prerequisite for claiming the already observed SBI association.

**Next outputs:** an inventory of participant-level exposure responses, target/response segment alignments, an exposure-to-test phonological-relatedness rule, and a model separating item-specific transfer from general listener ability. Do not substitute a correlation between overall training and test accuracy.

## Additional requirements from earlier exchanges

6. **Retain both nested comparisons.** `M_joint` versus `M_condition` asks whether SBI/HVE adds information beyond condition; `M_joint` versus `M_predictor` asks whether condition adds information beyond SBI/HVE. Do not reoptimize the theoretical predictor separately for each comparison. September 1 combined-fold LRT outputs exist, but layer/method selection from the same data makes their inference selection-conditional. A joint predictor's z is not the omnibus test of a multi-level condition factor. Sources: quoted Slack and nested-test email; meeting 16:46–20:17.
7. **Keep conditional S-curves.** Show colored condition curves and quantile-binned accuracy/CI within each test talker, including X21; retain a pooled-binning companion. Curve offsets illustrate residual condition effects but do not replace a joint-model test. Source: user's supplied plotting example; meeting 23:08–24:31.
8. **Show genuine matched-content talker distance matrices.** For X21, average same-sentence DTW across the 32 sentences for each talker pair, then assemble the matrix. Keep AN19 matched-word and B23 matched-sentence matrices separately labeled. A matrix of correlations between distance vectors answers a different question. Source: user's explicit matrix clarification; existing August 21 source tables.
9. **Make HVE definitions consistent across datasets.** Distinguish within-token frame transitions (no trial order required) from global exposure-order transitions (actual token order, including boundaries). Repeated-type measures need repeated instances, not just many frames. State root/no-root equations, type/token weighting, and unsupported cells; never fill unavailable definitions with zeros. Sources: earlier quoted Slack/email and user's global-order decision.
10. **Make reproduction and documentation usable.** Distinguish supplied HDF5 reproduction from audio-to-feature regeneration; record extraction/reduction seeds, checkpoints and software; reconcile collaborators' methods before asserting exact replication. Keep technical documentation concise, avoid duplicated scientific narrative, preserve Florian's prose, and track unfinished work in TODO rather than labeling every gap an inherent limitation. Sources: earlier quoted Yuhao/Florian correspondence.
11. **Keep the other open TODO tasks visible.** The combined Tr-24 SBI + HVE + acoustic model, backward-selection rule, all-layer multivariable analyses, and sensitivity analyses remain pending. They were present in the local TODO; this review does not attribute every one to a verified sentence in the September email. Obtain the current Overleaf ablation inventory before declaring the requested figure set exhaustive.

## Immediate execution order

1. Review the z figures and freeze their statistical scope using [Z_VALUE_FIGURE_SPEC.md](Z_VALUE_FIGURE_SPEC.md). The current review does not silently replace genuine CV with test-fold GLMM refits.
2. Rerun current-definition z and a matched ceiling; recover all-layer SBI inputs/results; separate B23 participant strata; correct acoustic preprocessing before further attribution.
3. Deliver all supported SBI/HVE layer panels plus Tr-24 summaries, both likelihood/z optimization views, conditional S-curves and matched-content matrices, with one navigable index.
4. Review the newly computed Figures 1a/b/c and 2a/c; agree on the narrower 1d proposal and obtain the specific annotations needed for 2b/d. Reconcile the local manuscript with the current Overleaf rather than treating the draft as authoritative.
5. Render standardized ablations and both nested-model comparisons, then rebuild the presentation. Keep the exposure-understanding extension as a separately scoped work package.

This ordering is an implementation plan. It does not assert that Florian rejected likelihood reporting, that all figure panels have been generated, or that the current result set establishes a particular publication outcome.
