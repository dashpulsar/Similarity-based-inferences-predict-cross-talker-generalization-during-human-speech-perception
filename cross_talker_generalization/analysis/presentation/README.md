# Presentation figures and speaking notes

Start with the **12-page concise update**: [figure PDF](../../../output/pdf/cross_talker_figures_concise.pdf), [bilingual speaker notes](SPEAKER_NOTES_CONCISE_BILINGUAL.md) ([PDF](../../../output/pdf/cross_talker_speaker_notes_concise_bilingual.pdf)), and [selection/page correspondence](CONCISE_EDITION.md). This is the current presentation; the complete reference collection below is preserved separately.

## Complete reference collection

Use [the figure PDF](../../../output/pdf/cross_talker_figures_only.pdf) and [the bilingual speaker notes](SPEAKER_NOTES_BILINGUAL.md) ([PDF](../../../output/pdf/cross_talker_speaker_notes_bilingual.pdf)). The figure PDF contains no cover, explanatory prose, external captions, names or dates; axes, panel labels and necessary legends remain. Bookmarks match the numbered speaking notes.

The 49-page collection has a 19-page main sequence, followed by the remaining layer/method panels. HVE z and likelihood pages now use identical method order. AN19 phone-specific pages use the same instance weighting and pairwise exponential similarity as the overall segment summary, with talker-bootstrap CIs; they replace the old word-type-weighted distance view.

## Evidence coverage

The HLP Slack connection was successfully verified. The available direct-message history from September 6 onward was read, including all three complete reply threads: 23 messages in total, nine from Florian. A two-page DM search was exhausted to check for replies outside the newly started threads. The latest returned reply was on September 10 (Asia/Shanghai). All 20 annotations in the locally supplied report_Sep_4-tfj.pdf were also checked. No raw private-message export or credentials are included in this package.

## Review decisions applied

1. Method illustration: first/last latent dimensions with an intervening ellipsis, white background, panels a/b, and a clearly identified L1-English example without a start legend (page 1). The example is explicitly changed to The wife helped her husband, not a silent a/the correction. [Discussion](https://hlplab.slack.com/archives/D0BHN5WAR24/p1788705101469909).
2. Language context: supplied world map and feature heatmap, consistent language groups/colors, and inventory overlap (pages 2-4). The clipped heatmap row label is recovered from existing PDF text by extending the page box; no scores or wording are invented.
3. Pronunciation matrices: English reference talkers included, language blocks, common talker order, and separate linear distance/similarity colors (pages 5-6).
4. AN19 segments: automatic intended-phone alignment; same word/context comparison; equal segment instances within each L2 talker; descending mean similarity; talker-bootstrap CIs and simplified labels (pages 7-10). [Averaging clarification](https://hlplab.slack.com/archives/D0BHN5WAR24/p1788974045909979), [CI/layout clarification](https://hlplab.slack.com/archives/D0BHN5WAR24/p1788978312362729).
5. Control responses: marginal similarity distributions, shared language colors, and explicit test/control interpretation (page 11). HW74 remains excluded from these matched-content comparisons because of the recorded wave/wade mismatch; no automatic relabeling. [Follow-up](https://hlplab.slack.com/archives/D0BHN5WAR24/p1788976621464199).
6. Statistical displays: SBI normalized-z profiles retained separately from genuine held-out likelihood; complete supported HVE method profiles in matched panel order. Historical z and ceiling limitations are explained in the notes, not hidden behind a predictive-performance claim.
7. Behavioral curves and comparison: condition-specific/pooled X21 curves, explicit nested-model comparison directions, and corpus coverage as supplementary material (pages 17-19 and 49). No unsupported result is added to fill an unavailable panel.

This is a presentation package, not certification that all analyses are final. Figure 2d remains unavailable; historical z/ceiling and current held-out/model-comparison results have distinct scopes, explained in the notes. Known-mismatched acoustic ablation rankings are excluded. No ASR, t-SNE, DTW, GLMM or predictor search was rerun.

## Regenerate

Use the scientific Python environment for panel preparation, then Python with PyMuPDF, Pillow and ReportLab for assembly. The bilingual PDF uses the installed Windows Arial and SimHei fonts.

    python cross_talker_generalization/scripts/build_figures_only_panels.py
    python cross_talker_generalization/scripts/build_figures_only_package.py --render

## Page index

| PDF page | Figure |
|---:|---|
| 1 | Speech to latent trajectories |
| 2 | Language backgrounds |
| 3 | Inventory overlap with English |
| 4 | Supplied phonological-feature comparison |
| 5 | AN19 talker distance matrix |
| 6 | AN19 talker similarity matrix |
| 7 | AN19 all-segment similarity |
| 8 | AN19 intended phones: IH, IY, AE and EH |
| 9 | AN19 intended phones: UH, UW, TH and DH |
| 10 | AN19 intended phones: S, SH, R and L |
| 11 | English-reference similarity and control accuracy |
| 12 | AN19 SBI: historical non-ASR-FT z |
| 13 | X21 SBI: historical non-ASR-FT z |
| 14 | B23 SBI: historical non-ASR-FT z |
| 15 | X21 HVE: training-fold z, definitions 1-4 |
| 16 | X21 HVE: held-out likelihood, definitions 1-4 |
| 17 | X21 condition-specific curves |
| 18 | X21 pooled curve |
| 19 | X21 selected-model nested comparisons |
| 20 | AN19 SBI: historical ASR-FT z |
| 21 | X21 SBI: historical ASR-FT z |
| 22 | B23 SBI: historical ASR-FT z |
| 23 | AN19 SBI: base held-out likelihood |
| 24 | AN19 SBI: ft held-out likelihood |
| 25 | X21 SBI: base held-out likelihood |
| 26 | X21 SBI: ft held-out likelihood |
| 27 | B23 SBI: base held-out likelihood |
| 28 | B23 SBI: ft held-out likelihood |
| 29 | AN19 HVE: z, definition group 1 |
| 30 | AN19 HVE: z, definition group 2 |
| 31 | X21 HVE: z, definition group 2 |
| 32 | X21 HVE: z, definition group 3 |
| 33 | X21 HVE: z, definition group 4 |
| 34 | X21 HVE: z, definition group 5 |
| 35 | B23 HVE: z, definition group 1 |
| 36 | B23 HVE: z, definition group 2 |
| 37 | B23 HVE: z, definition group 3 |
| 38 | B23 HVE: z, definition group 4 |
| 39 | AN19 HVE: loglik, definition group 1 |
| 40 | AN19 HVE: loglik, definition group 2 |
| 41 | X21 HVE: loglik, definition group 2 |
| 42 | X21 HVE: loglik, definition group 3 |
| 43 | X21 HVE: loglik, definition group 4 |
| 44 | X21 HVE: loglik, definition group 5 |
| 45 | B23 HVE: loglik, definition group 1 |
| 46 | B23 HVE: loglik, definition group 2 |
| 47 | B23 HVE: loglik, definition group 3 |
| 48 | B23 HVE: loglik, definition group 4 |
| 49 | Supplement: corpus talker coverage |
