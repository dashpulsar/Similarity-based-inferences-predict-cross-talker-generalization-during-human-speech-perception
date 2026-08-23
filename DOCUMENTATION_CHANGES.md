# Documentation changes after review

This record makes the post-review edits visible without obscuring the reviewed wording. The text baseline is commit `798c2df` (`redundancy removal`), which contains the last documentation edits made before the implementation and B23 source audits described below.

On 2026-08-23, an initial cleanup rewrote substantial parts of `README.md` and `PROJECT_DESCRIPTION.md`. Those broad rewrites were merged and then reconsidered. The two files have now been restored to the `798c2df` wording except for the targeted changes listed here.

## README.md

### Superseded versions

Previous text:

> Superseded implementations and reports are recoverably archived under `recycle_bin/` and are not used at runtime.

Current treatment: the `recycle_bin` directory and its references are removed from the main branch. The README now states that superseded tracked versions remain available through Git history.

### Statistical optimization and OOF comparison

Previous text:

> The primary predictive quantity is: `OOF gain = log loss(M_condition) - log loss(M_joint)`

Implementation audit: the code fits `M_condition`, `M_predictor`, and `M_joint`, but the report builder currently ranks feature spaces by the condition-only versus joint OOF log-loss difference. It does not persist or use the full-data likelihood of `M_predictor` as the intended optimization criterion.

Current treatment: the README now separates predictor-only likelihood optimization from the additional condition-only versus joint OOF comparison. Implementing and prespecifying the intended criterion is recorded as Priority 1 in `TODO.md`. Existing numerical results were not rerun during this documentation correction.

### B23 multi-talker HVE

Previous text:

> B23 multi-talker actual-exposure HVE remains blocked because the repository does not contain the required sentence-to-talker assignment.

Source audit: the authors' public OSF project (`10.17605/OSF.IO/T83XK`) contains `BBP-2023-StimLists.xlsx` and `BBP-2023-TrainingData.xlsx`. These files provide multi-talker sentence-to-recording mappings. The current production pipeline has not yet imported or validated them.

Current treatment: B23 multi-talker HVE is described as pending implementation rather than permanently unavailable. The current count of 14 B23 methods is also explained separately: in the existing single-talker pools, `within_type_sentence` and `mean_dissimilarity_sentence` are mathematically undefined because each sentence type has one recording.

### Terminology

- “explicit sensitivity profiles” was changed to “prespecified sensitivity analyses”;
- “validated AN19 talker-matrix sources” was changed to a literal description of checked matched-content talker-distance summaries;
- “provenance-backed method schematic” was changed to a method schematic with recorded build inputs.

These wording changes respond only to the phrases identified as unclear in review.

## PROJECT_DESCRIPTION.md

The overview, theoretical framework, representation discussion, dataset table, dataset descriptions, computational-construction section, output list, and interpretation constraints have been restored to the `798c2df` wording.

Only two substantive passages remain changed:

1. The B23 paragraph now records that the OSF exposure mappings have been located but are not yet integrated.
2. The behavioral-modeling section now distinguishes the intended predictor-only likelihood optimization from the separate condition-only versus joint OOF comparison.

## Structural changes

- `TODO.md` was added to keep remediable gaps separate from intrinsic limitations.
- The tracked `recycle_bin` policy file was removed; Git history is the archive for superseded tracked content.
- The standalone manuscript-strategy document was removed from the working tree so that project documentation does not duplicate paper claims. It remains available in Git history.

## Deferred consolidation

Some duplication remains between `README.md` and `PROJECT_DESCRIPTION.md`. It has deliberately not been removed in this correction because doing so would require another substantive rewrite of reviewed prose. Any further movement or deletion should be proposed as a small, reviewable diff rather than performed as a broad rewrite.
