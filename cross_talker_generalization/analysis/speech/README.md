# Speech representations and phoneme analyses

For the current **12-page figures-only update**, use the [figure PDF](../../../output/pdf/cross_talker_figures_concise.pdf) and [English/Chinese speaker notes](../presentation/SPEAKER_NOTES_CONCISE_BILINGUAL.md). The [page index and selection](../presentation/CONCISE_EDITION.md) distinguish this concise presentation from the earlier illustrated reports and complete 49-page collection. This newer package checks the available HLP private discussion from September 6 onward, including all three complete reply threads; the historical access statements below describe the earlier report preparation.

Start with [the illustrated report](REPORT_FOR_FLORIAN.md) or [the updated PDF with automatic AN19 phone annotations](../../../output/pdf/cross_talker_analysis_report_for_florian_with_phone_annotations.pdf). It uses first-person English without a name/date byline. The new 22-page edition adds full-corpus annotation methods and three legible Figure 2b candidate pages.

For replying directly to the review, use [the revised PDF with comment replies](../../../output/pdf/replies_report_Sep_4-tfj.pdf). The original ten pages retain their topic order, with updated text/figures and re-anchored comments. Eight supplementary pages follow. All 20 original annotation texts, authors, IDs and types are preserved, with 20 native linked replies under the author label `Response`. Open the PDF's comments panel to read the threads. The unchanged reviewed PDF is embedded as an attachment for comparison; this is a correspondence/review copy, not a clean manuscript PDF.

That 18-page reply edition is preserved unchanged and predates the new automatic phoneme analysis. Use the 22-page PDF above for the latest analysis; the original review threads have not been overwritten.

[REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) lists all 20 PDF annotations, the additional pasted Slack requests, completed edits and remaining analyses. The full private Slack history was not accessible; this update includes the supplied messages and attachments, not unseen correspondence.

## Figures

- Latest AN19 all-segment summary: [single-page PDF](../../../output/pdf/AN19_all_segments_similarity.pdf) and [PNG](../../../output/figures/an19_phonemes/AN19_all_segments_similarity.png). Each eligible segment instance has equal weight within a talker; similarity is averaged after pairwise exponential conversion, and L1 groups are sorted from more to less similar. Individual talkers, group means and 95% talker-bootstrap CIs are shown; single-talker groups have no CI. The legend and methods text have been removed from the figure and the x-axis shortened, following Florian's latest feedback. This standalone update is not yet embedded in the 22-page PDF above; it replaces the earlier equal-phone-type all-segment distance summary. See the [method and source tables](an19_phone_alignment/README.md).
- New Figure 1a/b representation example; original language world map and its source R; matching supplementary coverage; revised inventory comparison and supplied phonological feature illustration.
- Two 42-talker AN19 matrices: distance and similarity, both linear, with English talkers and language blocks.
- [Automatic AN19 phoneme annotations and Figure 2b candidate](an19_phone_alignment/README.md), using the user-selected `July/Nygaard_audio` recordings. Complete alignment covers 6,261 recordings; independently recognized IPA has 389 explicitly retained empty outputs. The new plots are coverage-sensitive automatic estimates, not verified phone transcriptions.
- AN19/X21 control-test curves with marginal similarity distributions and common language colors.
- Six SBI z figures and six genuine three-fold held-out likelihood profiles, including acoustic baselines first. These retain different historical predictor/fitting definitions, explicitly labeled, rather than claiming a new matched objective comparison.
- Eleven pages of revised HVE likelihood profiles covering all supported methods and layers; B23 samples remain separate.
- X21 condition-specific and black pooled curves with revised titles. Existing intervals are retained without inventing their undocumented resampling unit.

Open [figures](figures) for PNG/SVG/PDF exports and [tables](tables) for source data and generation details. The supplied files are preserved under [sources](sources), with checksums. The feature heatmap remains a supplied illustration because its score table/code were not supplied.

## Regenerate

From the repository root, use the project scientific environment:

```powershell
python cross_talker_generalization/scripts/build_september9_data_panels.py
python cross_talker_generalization/scripts/build_september9_condition_curves.py
python cross_talker_generalization/scripts/build_september9_method_panel.py
python cross_talker_generalization/scripts/build_september9_statistical_panels.py
python cross_talker_generalization/scripts/build_collaborator_report_pdf.py --source cross_talker_generalization/analysis/speech/REPORT_FOR_FLORIAN.md --output output/pdf/cross_talker_analysis_report_for_florian_with_phone_annotations.pdf --render --preview-dir tmp/pdfs/florian_phone_annotations --image-max-height 350
```

The statistical script recovers original all-layer SBI fold tables from the linked worktree; compact copies are supplied under tables. HVE uses the existing revised-selection runs. The other builders use the retained report tables, original SVGs, supplied figures and existing HDF5 data. None reruns GLMMs, feature extraction or t-SNE.

Phonetic model preparation and the full-corpus commands are documented in the [alignment record](an19_phone_alignment/README.md) and [IPA recognition record](an19_phonetic_xeus/README.md). These are separate from the figure-only commands above. PDF rendering additionally needs PyMuPDF in the Python environment used for the report; it does not require either phonetic model.

No earlier reports, data or supplied files were deleted. No Git commit/push, Slack message or Overleaf upload was made.

## Regenerate the PDF reply edition

Supply the original annotated PDF, without overwriting it:

```powershell
python cross_talker_generalization/scripts/build_in_place_review_pdf.py --original "path/to/report_Sep_4-tfj.pdf" --revised output/pdf/cross_talker_analysis_report_for_florian_reviewed.pdf
```

This command deliberately uses the retained 18-page clean template, not the newer 22-page phoneme report: the original review anchors and page mapping belong to that template. The builder replaces the original page content, relocates comments and adds native reply links. It checks that original annotation text/metadata/types are unchanged, that each original comment has one reply, and that the embedded source is byte-identical to the original file. Replies are editable in `tables/pdf_review_replies.json`; anchor and source checks are recorded in `tables/pdf_in_place_revision_audit.json`. The final 18-page output was visually checked with annotations visible.
