# Final verification report

Generated: 2026-08-21 (Asia/Shanghai)

## Status

Pass. Before this verification file is added, the report contains 40 PNG files, 40 SVG files, 29 CSV files, 2 provenance JSON files, and 4 other Markdown files.

## Build-time invariants

- Figure 00 percentages are fold Wald z / mean behavioral-ceiling z × 100, not predictive accuracy or variance explained.
- AN19 base/FT talker matrices are 42×42 with 138 shared words per off-diagonal cell.
- X21 base/FT talker matrices are 11×11 with all 32 matched experimental sentences per off-diagonal cell.
- B23 base/FT talker matrices are 4×4 with 120 shared sentences per off-diagonal cell.
- All six matrices are complete and symmetric with zero diagonals; X21/B23 item-level DTW rows reaggregate to matrix means.
- Variability covers 6 AN19, 16 X21, and 14 B23 computable measures.
- Signed compatibility profiles and true participant-held-out OOF variability profiles are both included.
- An unidentified B23 multi-talker actual-exposure HVE value is not replaced by a proxy.

## Code tests

The release run of `python -m unittest discover -s cross_talker_generalization/tests -v` passed 15/15 tests. Final `provenance.json` recomputes SHA-256 for the complete package after this file is written.
