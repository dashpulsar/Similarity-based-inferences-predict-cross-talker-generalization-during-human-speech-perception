# Response to the annotated report and supplied Slack messages

The [direct PDF reply edition](../../output/pdf/replies_report_Sep_4-tfj.pdf) applies the earlier display revisions to the original ten-page topic sequence, repositions the original comments and adds a linked first-person reply to each. Supplementary figures/details follow on pages 11-18. The original marked PDF is preserved unchanged as an embedded attachment. The comment inventory below describes the reviewed source; original comment text/authorship is not rewritten.

The later [22-page analysis PDF](../../output/pdf/cross_talker_analysis_report_for_florian_with_phone_annotations.pdf) adds the full automatic AN19 phoneme analysis and Figure 2b candidate. The preserved 18-page comment-thread edition predates this addition; its original comments and replies have not been changed.

## Sources and coverage

I reviewed all 20 PDF annotations in `report_Sep_4-tfj.pdf`: 19 textual comments and one uncaptioned ink mark accompanying the model-comparison question. I also incorporated the Slack exchange pasted by Zhengyang about Figure 1 and AN19 phoneme labels, the five supplied map/heatmap files, and the earlier meeting/email decisions already documented in the project.

This is **not a complete audit of the private Slack conversation**. The live connector continued to return a different workspace; direct access to the supplied hlplab channel and DM failed. I have not inferred missing messages or attachment contents. The map and feature heatmap were ultimately obtained from the user-supplied local files, not from a successful private-message download.

## PDF comments, in page order

| ID / page | Request or question | Response and status |
|---|---|---|
| P01 / 1 | Should SBI/HVE have the same ceiling? | Corrected the explanation. Historical files are identical; share one behavioral reference for matched rows, folds, statistic and fitting procedure. Current item-by-condition ceiling implementation still needs correction for AN19/X21. |
| P02 / 1 | Explain coordinate scaling | Added the equation and the consequence for DTW weighting/alignment. Full baselines used global_z; components used none across the same 5,459 pairs. Matched rerun remains open. |
| P03 / 3 | Coverage figure belongs in SI; match world-map colors; store in Git and Overleaf | Added the SI figure, using exact colors/groups from the supplied R. Files are saved locally; no commit/push or Overleaf upload has been made. |
| P04 / 4 | Inventory panel may fit Figure 1; consider tones; match colors; remove counts | Matched colors and removed inventory counts. Tone extension remains a separate metric decision and missingness check; existing values are unchanged. |
| P05 / 4 | Change inventory x-axis wording | Applied: L1-L2 phonological inventory similarity with English. |
| P06 / 4 | Put inventory summary above supplied feature examples | Included both in the report, on adjacent pages for legibility. Final compound assembly awaits the feature heatmap score table/code and repair of its clipped row label. |
| P07 / 5 | Restore English talkers; gray language blocks; black language-first labels; similarity ordering | Applied to both 42-talker matrices. English first, other groups by mean distance to English; average-linkage clustering within each language. |
| P08 / 5 | Show distance and similarity separately, simplify legend, use linear colors | Applied. Mean DTW and mean exp(-DTW), k=1, each over the same 138 words. Both use linear scales and the same talker order. |
| P09 / 6 | Relation to Kim et al.; marginal density; world-map colors | Added distributions over unique target recordings and matched colors. Described a conceptual extension rather than exact replication, citing the paper. |
| P10 / 6 | Shared legend and dataset-only titles | Applied to AN19/X21 control curves. |
| P11 / 6 | Similarity to L1-English x-axis | Applied. |
| P12 / 6 | Verify wave/wade in recordings | Retained the exclusion pending content verification. Previous local ASR screening is not ground truth; no automatic alias or relabeling. |
| P13 / 6 | Exposure or control-test performance? Consider first ten exposure sentences | Explicitly control-condition test responses. Early exposure is a distinct planned analysis with a prespecified presentation window, not a relabeling of existing points. |
| P14 / 7 | Three-entry legend below ceiling | Applied to six SBI z plots: significance reference, ceiling, individual folds. Remaining explanations moved to captions. |
| P15 / 7 | Narrower figure and Feature space x-axis | Applied. |
| P16 / 7 | Add log-likelihood plots | Added six all-layer SBI profiles with acoustic baselines and eleven HVE pages, with genuine three-fold intervals. These are retained runs; no new objective optimization or incompatible ceiling normalization is claimed. |
| P17 / 8 | CMN and language-first talker titles | Applied to condition-specific and pooled X21 figures. |
| P18 / 8 | Single black pooled line and points | Added the pooled companion for all four test talkers, preserving the existing numerical curves and uncertainty geometry. |
| P19 / 9 | Why does condition add little to HVE? Are samples identical? | Replaced the tiny plot with exact X21 statistics and model-sample explanation. Condition adds to HVE at p=.01168. Predictor-only HVE is negatively associated (z=-5.5038); p=.05318 tests HVE beyond condition. X21 rows/structure match, but AN19 and B23 comparisons require the qualifications below. |
| P20 / 9 | Ink mark over model-comparison figure | Reviewed with P19; it contains no additional textual instruction. |

## Additional pasted Slack comments

| ID | Request | Response |
|---|---|---|
| S01 | First/last dimensions only, with vertical ellipsis between panels | New Figure 1a uses Dimension 1 and Dimension 1024 with a separate ellipsis gap. |
| S02 | Waveform/features/t-SNE are one panel a; trajectory is b and top-aligned | Applied; white background, no start legend or ambiguous black connectors. |
| S03 | Check a/the window; use a clear native example with no label overlap | Changed the example explicitly to native talker 055, The wife helped her husband, with wife /w aɪ f/. The original first-sentence manifests all say a window; no word was silently changed. |
| S04 | Use the previously sent world map, phonology examples and code | Preserved the five supplied files unchanged, recorded hashes and parsed the palette from the R. The provided R generates the map, not the feature-score heatmap. |
| S05 | Missing AN19 phone labels: consider ASR or another dataset | Ran the full AN19 corpus without a pilot gate: FALCON aligned all 6,261 recordings and PhoneticXEUS independently recognized 5,872 nonempty IPA hypotheses (389 empty outputs retained). Both read the user-selected `July/Nygaard_audio` directory, with explicit filename/hash mapping. The [annotation record](an19_phone_alignment/README.md) distinguishes intended-phone timing from recognized pronunciation and listener errors. |

## Statistical findings behind the revisions

Latest all-segment follow-up: the [new AN19 similarity plot](../../output/pdf/AN19_all_segments_similarity.pdf) averages every eligible segment **instance** equally within each L2 talker, rather than giving each word/phone type equal weight. I apply `exp(-DTW)` to each cached pair before the English-reference and L2-instance means, retain k = 1 from Figure 2a, and sort L1 groups by descending mean similarity. Following Florian's latest clarification, I restored 95% talker-bootstrap CIs (1,000 samples, seed 20260909) for the Korean and Spanish means; the other groups each have one talker and no estimated CI. I removed the top legend, methods subtitle and in-figure footnotes, and shortened the x-axis to "Mean segment similarity to L1-English". Symbol definitions and methods remain in the accompanying text. The 5,114-instance sample and original pair distances are unchanged; the superseded equal-type all-segment plot is retained but is not the current result. Other figures' uncertainty intervals are unaffected.

- X21: exact SBI/HVE row equality after canonicalizing the redundant HVE word suffix, 16,477 responses and 320 participants; common condition-only logLik=-5484.311388. Both use fold and test-talker blocking plus participant/item random intercepts.
- X21 selected HVE: predictor-only z=-5.503837, coefficient p=3.7161e-8; joint z=-1.951614. HVE beyond condition LRT p=.053179; condition beyond HVE p=.011680. Negative association does not support the expected positive HVE effect. Base/FT selected HVE inputs are identical, not independent confirmations.
- AN19: SBI 5,685 rows versus HVE 5,760. The 75 additional HVE rows concern hung (21), mote (20), main (19), route (15). The SBI random-effects fallback omitted talker; HVE retained it. This mismatch is separate from the 40 wave/wade control responses in Figure 2c.
- B23: SBI/order-independent HVE use 168 participants and 10,080 count rows (54,096 word trials); global-order HVE uses 97 participants and 5,820 rows (31,234 trials).
- Historical SBI/HVE ceiling CSVs have exactly identical nine fold values. The current direct ceiling uses item/talker keys without condition for AN19/X21 (`src/ctg/ceiling_cv.py`); this still needs the meeting's item-by-condition construction.
- Full acoustic distances standardize each coordinate before DTW; diagnostic subsets did not. This changes the metric and possibly the path, not merely a final scalar predictor scale.
- The retained AN19 layerwise SBI runs also vary their talker random-intercept fallback across layers/folds. They are recovered results, not a controlled common-structure comparison. Final fits converged, but that does not remove this design difference.

Validation: the six SBI likelihood profiles contain 360 fold scores (20 feature spaces each); HVE contains 4,212 scores, covering 7 AN19, 17 X21 and 15 B23 definitions across 18 layers, two variants and three folds. All exported scores are finite and match their source tables. The retained six-run SBI diagnostics (1,296 fits) and revised HVE diagnostics (5,616 fits) report successful, non-singular final fits with convergence status `ok`.

Sources: `analysis_update_2026-09-01/tables/crossfitted_lrt_results.csv`, its model `coefficients.csv`, `diagnostics.csv` and `crossfitted_predictors.csv`; the August 21 two ceiling tables; `src/ctg/ceiling_cv.py`; September 6 acoustic distance tables. New likelihood source tables retain fold counts and predictor definitions. No GLMM was refitted for this report.

## Remaining work

The report does not complete all prior TODOs. Still open: matched z/likelihood predictor construction and both optimization objectives; matched behavioral ceilings; corrected AN19/B23 comparisons; acoustic rerun; early-exposure curves; Figure 2d's listener-response phone alignment and lexical-error analysis; tone sensitivity; heatmap score provenance; and any relevant private Slack messages not included in the pasted exchange. Full-corpus automatic acoustic annotation is now complete; it is not a substitute for listener-response labels. Git and Overleaf publication require a separate explicit publication step.
