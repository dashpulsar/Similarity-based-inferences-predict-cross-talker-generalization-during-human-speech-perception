# Checks performed

- Recomputed the control word-reference DTW and descriptive curves using eight worker threads and 1,000 participant bootstrap samples. Existing HDF5 files were read-only.
- Verified every generated scientific output against its recorded SHA-256.
- Checked the 36-by-36 AN19 matrix: symmetric, finite off-diagonal values, masked diagonal and 138 shared words in the source.
- Checked 17 language labels across 57 corpus-specific talker entries and coverage of all 16 L2 labels by the proposed inventory metric.
- Reconstructed similarity from exported distance/scale columns. Matched 1,880/1,920 AN19 and 4,117/4,117 X21 control responses, using six and five English references respectively.
- Verified 1,000 successful logistic bootstrap fits in each accent group and valid probability intervals.
- Investigated HW74: all 42 recordings and Tr-24 features exist. English labels are wade; L2 labels are wave. Local ASR screening does not justify merging them. No source labels or features were changed.
- Rendered and visually inspected all ten PDF pages, then rechecked the revised pages after the naming audit and annotation clarification. No off-page text/image blocks were detected.

Run `python cross_talker_generalization/scripts/validate_collaborator_panels.py` to repeat the numerical checks. This checks internal consistency; it does not validate the unresolved word identity or make the descriptive fits confirmatory.

Unrelated pre-existing audit CSV whitespace warnings remain in the working tree. No Git index changes, commits, pushes or deletions were made for this report.
