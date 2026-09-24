# Concise figure update

Use this 12-page edition for the current discussion: [figures](../../../output/pdf/cross_talker_figures_concise.pdf), [bilingual notes PDF](../../../output/pdf/cross_talker_speaker_notes_concise_bilingual.pdf), and [editable notes](SPEAKER_NOTES_CONCISE_BILINGUAL.md). The 49-page collection remains unchanged for reference.

## What changed

- Removed original pages 8-10 and 23-49 from this presentation, following the latest user request.
- Replaced the six SBI pages with one three-dataset figure: overlay non-ASR-FT/ASR-FT, show MFCC/STRF once, and retain normalized z only. Raw z was the same information on another scale. The three folds, 95% intervals, 100% ceiling band and nominal significance references remain.
- Put inventory overlap above the supplied feature heatmap; put the requested distance/similarity matrices side by side.
- Keep the overall segment summary, not individual-phone details; keep both condition-specific and pooled curves because they answer different questions. Keep the two original X21 HVE examples, not the entire method catalogue.

This selection is an editorial decision, not a claim that Florian rejected every omitted analysis. In particular, his PDF explicitly requested likelihood plots; the exhaustive appendix is omitted here under the newer user instruction. No data, old PDFs, model results, or source figures were deleted. No model was rerun.

The prior review of 23 Slack messages and 20 PDF annotations remains the source context. Historical statistical limitations remain in the spoken notes. The source feature heatmap and its unknown score construction are not relabeled as a newly computed result.

## Page correspondence

| New page | Topic | Original pages |
|---:|---|---|
| 1 | Speech to latent trajectories | 1 |
| 2 | Language backgrounds | 2 |
| 3 | Phonological inventory and feature comparisons | 3, 4 |
| 4 | AN19 distance and similarity matrices | 5, 6 |
| 5 | AN19 all-segment similarity | 7 |
| 6 | Similarity to English and control accuracy | 11 |
| 7 | SBI across datasets and model variants | 12, 13, 14, 20, 21, 22 |
| 8 | X21 HVE: four retained definitions, z | 15 |
| 9 | X21 HVE: the same definitions, held-out likelihood | 16 |
| 10 | X21 condition-specific curves | 17 |
| 11 | X21 pooled curves | 18 |
| 12 | X21 nested-model comparisons | 19 |

## Rebuild

Use the scientific environment for the first command and the PDF environment for the second.

    python cross_talker_generalization/scripts/build_concise_figure_package.py --prepare-sbi
    python cross_talker_generalization/scripts/build_concise_figure_package.py --render
