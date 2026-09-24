# Main Figure 1 and Figure 2 specification

This document converts the current manuscript captions into reproducible panel requirements. It deliberately marks missing inputs instead of filling gaps with illustrative or inferred values.

The latest computed panels and captions are in the [report for Florian](../analysis/speech/REPORT_FOR_FLORIAN.md). Its [review response](../analysis/speech/REVIEW_RESPONSE.md) supersedes the earlier example and matrix specification below: Figure 1a/b now uses the native "wife" example; Figure 1c uses the supplied world map and palette; Figure 2a includes all 42 talkers, with separate linear distance/similarity panels. AN19 phone panels remain an alignment/validation task, with an alternative annotated dataset also under consideration. Earlier source inspection is documented in [SOURCE_REVIEW.md](../analysis/model_comparison/reference_checks/collaborator_report/SOURCE_REVIEW.md).

## Figure 1: obtaining and interpreting latent speech trajectories

### Panel a — computational transformation

Show one speech recording as a waveform, its frame-wise representation from a declared HuBERT layer, and its 3-D t-SNE trajectory. A recording with `T` output frames is represented by a matrix in `R^(T × 3)`. The panel must state the HuBERT checkpoint, layer, t-SNE fit scope, and whether the plotted example is base or ASR fine-tuned.

Status: regenerated at Tr-24 from the actual HuBERT-large full-dimensional and corpus-level t-SNE stores. The English-055 waveform, checkpoint revision, fit scope and selected HDF5 leaf are recorded with the new panel.

### Panel b — segment, word, and sentence “hyper-noodles”

Show trajectories for verified segment, word, and sentence intervals in the same 3-D space. The caption placeholders `XXX` must be replaced only after selecting tokens with valid interval annotations. Labels should make clear that duration is represented by the number and ordering of frames, not by a fourth geometric dimension.

Status: computed in the same coordinate space for "A boy fell from a window", "fell", and F/EH1/L. The brief EH1 interval maps to one frame and is shown as a point, not an invented continuous trajectory.

### Panel c — first-language backgrounds

Show every L1 background represented across AN19, X21, and B23, colored by a single declared genealogical branch mapping. The panel must distinguish a language, a dataset accent label, and a talker. Repeated languages across datasets should not be counted as new branches.

Status: computed for 17 language labels across the available corpora. AN19 Table I resolves the original "Indian" and "Chinese" labels as Hindi and Mandarin. The exported table preserves original labels and distinguishes corpus coverage from actual experimental exposure.

### Panel d — sound-system similarity to English

Define and plot a reproducible language-level distance to English. [PHOIBLE 2.0](https://phoible.org/) provides phonological segment inventories and distinctive features, plus genealogical metadata. It does not by itself provide word-level phonotactic distributions. Therefore, a panel based only on PHOIBLE must be labeled as inventory/feature similarity; any claim about phonotactics or broader suprasegmental structure requires an additional named data source and metric.

Status: a computed proposal uses PHOIBLE 2.0 consonant/vowel inventory Jaccard overlap, averaged across available inventory pairs and all nine English inventories. All 16 L2 labels are covered. Lines show source ranges, not confidence intervals. It is not a dialect-specific, distinctive-feature or phonotactic metric; acceptance of this narrower construct remains pending.

### Draft caption

**Figure 1 | Computational approach for obtaining latent speech representations.** **a,** A speech recording is transformed into a continuous sequence of latent feature vectors by a specified layer of an English-trained HuBERT model. A separately fitted t-SNE transformation maps the frame-wise representation to three dimensions, producing a `T × 3` trajectory for a recording with `T` output frames. **b,** Verified segment, word, and sentence intervals form time-ordered trajectories in this space. **c,** First-language backgrounds represented in AN19, X21, and B23, colored by genealogical branch. **d,** Language-level phonological inventory/feature similarity to English under the declared PHOIBLE 2.0 distance. Exact example tokens and the panel-d metric will be inserted after the pending source-data checks.

## Figure 2: representation validation before theory tests

### Panel a — AN19 talker pronunciation similarity

Plot word-matched pronunciation similarity for the requested 36 AN19 L2 talkers, ordered and annotated by L1 and gender. Explicitly decide whether the six L1-English reference talkers appear in a separate block. Each off-diagonal cell must average DTW distance over the same verified shared-word set. The diagonal should be visually separated because self-distance is structurally zero.

Status: the new panel selects the 36 L2 talkers from the 42-talker source and masks the diagonal. Its similarity is the mean of per-word `exp(-distance)` across 138 shared words, shown on a logarithmic color scale. This is not `exp(-mean distance)`; a mean-distance matrix is a separate statistic.

### Panel b — segment-level L2-to-L1 production deviation

For selected phoneme types, compare each L2 talker's segment trajectory with the declared L1-English reference distribution. Plot talker estimates, language-group means, and 95% participant/talker bootstrap intervals. Selection of phonemes must be declared independently of the plotted effect size.

Status: deferred. Zhengyang confirms AN19 has no phone-level labels. This panel requires a new, separately agreed alignment/verification effort and an outcome-independent phoneme-selection rule, not merely locating an existing annotation file.

### Panel c — pre-exposure intelligibility

Test whether each L2 talker's pronunciation similarity to L1-English predicts L1-English listeners' control-condition intelligibility. Include AN19 Korean- and Spanish-accented control tests and X21 Mandarin-accented control tests. The observational unit, aggregation level, and uncertainty unit must be stated; do not treat repeated listener trials as independent talkers.

Status: new matched-word control curves use 1,880/1,920 AN19 responses and all 4,117 X21 responses. The 40 AN19 responses to "wave" are provisionally excluded because English HW74 files say "wade", while L2 files say "wave"; audio and features exist, so this is a lexical-label conflict, not established recording absence. Similarity compares the target with six AN19 or five X21 English talkers; the scale is outcome-independent. Logistic curves and quantile-bin accuracy have participant-bootstrap intervals. These descriptive fits are not cross-validated GLMM tests. Existing conditional S-curves remain distinct complementary results.

### Panel d — segment perception errors and lexical constraint

Model AN19 control-condition segment errors as a function of the segment-level L2-to-L1 deviation from panel b and lexical constraint, defined as the number of minimal-pair word neighbors distinguished by the target segment. The final model must specify how transcription errors are aligned to target segments and how zero-neighbor and missing-lexicon cases are handled.

Status: the complete caption has now been recovered from `references/manuscript_draft.pdf`. AN19 phone labels are confirmed absent, so 2d is deferred along with 2b. In addition to new verified intervals, it requires response alignment, pronunciation lexicon, neighbor computation and final modeling.

### Draft caption

**Figure 2 | HuBERT latent representations capture production and perception structure in L2 speech.** **a,** Word-matched pronunciation similarity among the 36 AN19 talkers, grouped by first-language background. **b,** Segment-level deviation of selected L2-English productions from L1-English reference pronunciations; points represent talkers, vertical lines show language-group means, and intervals show 95% bootstrap confidence intervals. **c,** Association between L2-to-L1 pronunciation similarity and pre-exposure intelligibility for AN19 and X21 control conditions. **d,** AN19 control-condition segment errors as a function of representational deviation and lexical constraint. The exact segment set, reference construction, and lexical-neighbor definition remain pending and must be filled from verified source data before release.

## Shared production rules

- Every panel must have a CSV source table and provenance record.
- Panel titles should state what is estimated, not the conclusion.
- Color encodes the same concept consistently across panels; dataset, L1 branch, condition, and model variant must not reuse one palette ambiguously.
- Confidence intervals must name their resampling unit.
- Draft placeholders are acceptable in an internal layout but cannot appear in a report labeled final.
