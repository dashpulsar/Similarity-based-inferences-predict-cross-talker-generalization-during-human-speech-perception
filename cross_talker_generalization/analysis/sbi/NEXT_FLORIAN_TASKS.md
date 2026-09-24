# Remaining requests after the parameter-map handoff

Status: September 18, 2026. Zhengyang has paused further tau/k optimization research after the full-condition and Talker-specific-excluded X21 maps. HVE remains paused under his earlier instruction. No additional model runs or collaborator messages were initiated for this handoff.

This review uses the saved [September 11 meeting transcript](../../../outputs/meeting_2026-09-11/TRANSCRIPT_EN.md), its [source-linked task record](../../../outputs/meeting_2026-09-11/TODO_CN.md), the [September 15 email exchange](../diagnostics/CORRESPONDENCE_2026-09-15.md), and the earlier [PDF/Slack response record](../speech/REVIEW_RESPONSE.md). It does not claim access to any newer private correspondence.

## Priorities that remain active

The suggested order below is a handoff recommendation. The meeting explicitly prioritized the AN19 high-control result and the two optimization objectives; the latter is now paused at Zhengyang's request.

### 1. Explain the unusually strong AN19 result

**Scope clarified by Zhengyang on September 18:** explain why MFCC/STRF rival or exceed HuBERT CNN/Transformer results in AN19 but not X21. Demonstrating baseline predictive value alone does not answer this question. The [targeted comparison](../acoustic_baselines/AN19_BASELINE_EXPLANATION.md) verifies the historical ranking, contrasts current matched-response results, and examines item-level versus condition-level associations. The causal explanation remains open; no new GLMM fits were run for that comparison.

**Meeting: 28:50–29:12.** Florian called the high AN19 control result one of the two most important issues. The displayed context suggests the acoustic baselines, but the exact plotted point must be identified rather than treating “control” and “MFCC/STRF” as interchangeable terms.

Completed: corrected acoustic-component scaling, reproduced full MFCC/STRF distances, fitted the baselines within conditions, and checked a common random-effects structure. Full-baseline scores remained unchanged, and four unflagged conditions retained predictive information. These checks narrow the explanation but do not finish it.

Remaining deliverable: a concise, figure-linked account of which point prompted the concern, what checks ruled out, what remains uncertain, and whether any scientific interpretation changes. Use the existing [diagnostic summary](../diagnostics/PROGRESS_REPORT.md) before scheduling new computations. This is the recommended next task.

### 2. Finalize the SBI layerwise z/ceiling presentation

**Later clarification:** Florian's training/test diagnostic email requests likelihood-based SI figures and explicitly excludes z from that normalization. It does not require replacing the historical z/ceiling figures or changing the CV split. Preserve those figures with their test-fold-refit definition. Matching a new training-z ceiling is conditional on deciding to replace them, rather than a mandatory task arising from that email. The existing six SBI likelihood figures have now been checked and interpreted in the [September 18 diagnostic review](../diagnostics/sbi_review/README.md). Their arithmetic and source hashes pass. X21 has substantial opposing fold differences also seen without SBI, which must accompany any near-one mean-ratio statement.

**Meeting: 36:22–38:56; earlier figure feedback also specifies the format.** Preserve the three fold points, 95% intervals, mean ceiling at 100%, gray ceiling interval, significance reference line and comparable axes. Simplify the legend and unnecessary negative-axis whitespace, then supply the style/code to Wei-Kai.

Completed: historical displays exist; the direct likelihood-based behavioral reference now retains condition and X21 sentence context. That correction does not generate a Wald-z ceiling.

Remaining: define and compute compatible current z and z-ceiling values, then redraw the SBI profiles. Matching their response sample and fitting/evaluation convention is a calculation requirement for meaningful normalization, not an additional research objective attributed to Florian. Preserve the original figures until replacements are ready. Parameter re-optimization across all layers remains part of the paused work.

### 3. Finish the Figure 1 method schematic

**Meeting: 00:00–02:22 and 07:15–07:33.** Explain the transformation for a general reader, simplify the 3-D trajectory, and match the segment colors and example across panels.

Completed: the English-055 example and its actual representations exist. The [source audit](../diagnostics/source_validation/FIGURE_INPUT_AUDIT.md) confirms a 20 ms Tr-24 frame stride and a 25 ms convolutional front-end receptive field.

Remaining: replace approximate duration/T timing in the displayed transformer traces with the verified frame timing, update the caption, and finish the layout. Existing intermediate CNN arrays were pooled, so their displayed timing needs separate wording. No new feature extraction is required merely to revise this illustration.

### 4. Check the language-reference sources and clarify matrix presentation

**Meeting: 02:32–10:27 and 13:22–18:44.** Verify the map and phonological-reference definitions, feature values, database provenance and selection. Make the distance/similarity colorbar meaning clear and inspect a version with versus without English talkers.

Completed: the supplied figures and R code are retained; the current AN19 matrix has 138 shared words and 42 talkers, independently checked against its inputs.

Remaining: obtain or verify the heatmap's underlying score table/calculation, resolve its tone/weighting interpretation and clipped label, and review the matrix color scale. Decisions about keeping both traditional-reference panels, a formal traditional/model association, or within-/between-L1 significance tests remain separate choices. A PHOIBLE inventory overlap proposal does not substantiate all phonotactic and suprasegmental claims in the supplied illustration.

### 5. Organize the result story and specify layer comparisons

**Meeting: 41:29–43:46.** Start with fixed Tr-24, then show changes across layers. Florian asked for statistical evidence for layer differences and discussed pairing scores by fold.

Remaining: choose the actual layer contrasts and a justified test. The automatic transcript does not reliably recover the test name at 42:55–43:10. Check meeting chat or ask the collaborators before recording a specific test as agreed. Do not treat folds as independent by default or change the fold count based on this passage alone.

## Earlier manuscript gaps still open

These predate the September 11 meeting and are preserved in the [main TODO](../../../TODO.md); absence from the latest discussion does not mark them completed.

- **Figure 2d:** align listener responses to intended phones, define lexical neighborhoods and analyze errors. Automatic acoustic phone boundaries have been generated; they do not supply listener error labels.
- **Figure 2b:** assess the effect of unequal word/phone/reference coverage before making manuscript claims from automatic annotations.
- **Early exposure analysis:** define the first-ten-exposure analysis separately from control-test curves.
- **AN19 HW74:** resolve the “wade” versus “wave” content mismatch by recording-content verification. Do not merge different words just because their item numbers match.
- **Manuscript integration:** obtain the current Overleaf panel/ablation inventory before asserting that all requested panels are complete. Share or publish reviewed outputs only when authorized.

## Completed diagnostics and material ready to share

- Florian accepted test/train mean log loss on September 15. The September 17 batch has 18 layer profiles (six SBI, twelve HVE) with normalized, same-convention predictions, plus saved fold scores and documentation. Interpret the observed gaps and share the agreed plotting recipe; there is no need to relaunch the completed batch merely to produce those figures.
- The X21 full-condition and Talker-specific-excluded parameter landscapes are complete. A two-slide [PPTX/HTML package](../../../outputs/sbi_parameter_maps_2026-09-18/presentation/README.md) collects them. This satisfies a representative surface demonstration, while broader optimization validation remains open.

## Paused or awaiting a decision

- **Paused:** multi-seed optimizer comparisons, further tau/k searches, the full across-fold/layer z-versus-likelihood optimization comparison, and HVE follow-up. The current maps cover one layer and training split.
- **Unanswered:** nested train/tune/test CV for data-selected layers/methods. Florian's September 15 reply accepts the loss ratio but does not answer the separate nested-CV question. Preserve train-test until a decision is made.
- **Conditional suggestions:** adding other datasets to segment-to-English plots, formal L1 within-/between-group tests, and a model of actual exposure comprehension. Check data availability and agree on the question before implementation.

No new heavy run is needed to begin the AN19 explanation or the Figure 1/source review.
