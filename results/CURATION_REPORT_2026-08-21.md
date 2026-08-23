# Result curation report

## Outcome

The reviewed package at
`cross_talker_generalization/final_report_2026-08-21/` is now the single authoritative
result entry point. The top-level `results/` directory has been reduced to report
dependencies, direct compatibility-analysis inputs, validated AN19 matrix sources, and
one method schematic with recorded build inputs.

Superseded tracked outputs remain available through Git history.

## Curation decision

The original `results/` tree contained 346 files (approximately 1.53 GiB), almost all
generated on 2026-08-14 or 2026-08-15. Date alone was not used as the removal rule,
because the final-report builder intentionally reads several historical compatibility
summaries. Instead, each result group was checked against report-builder dependencies,
provenance, and its independent scientific value.

### Retained analysis/source files

| Section | Files | Approximate size | Reason |
|---|---:|---:|---|
| `derived/` | 52 | 982.49 MiB | Direct compatibility inputs plus complete AN19 talker validation |
| `figures/` | 59 | 43.85 MiB | S-curve report sources, AN19 matrix sources, and the X21 method schematic |
| `statistics/` | 98 | 5.47 MiB | Direct inputs to compatibility Figure 00 and variability panels |

These 209 analysis/source files are accompanied by this report, `README.md`, and the
generated `RETAINED_FILES_SHA256.csv` manifest.

### Removed from the working tree

| Archive section | Files | Approximate size | Reason |
|---|---:|---:|---|
| `derived/` | 46 | 455.83 MiB | Superseded or duplicate upstream distance/condition tables |
| `figures/` | 87 | 76.72 MiB | Replaced distance-correlation, z/ceiling, and variability plots |
| `other/` | 3 | negligible | Old logs and files from unused result shells |

The corresponding tracked versions remain recoverable from Git history.

## Post-curation verification

The production test suite passed 15/15 tests after the move.

A complete report was then rebuilt from the curated structure into a temporary directory.
The rebuild produced the same 116-file path set as the reviewed release, and every file
listed in the rebuilt `provenance.json` passed its SHA-256 check.

Comparison with the reviewed release showed:

- 40/40 PNG figures were byte-identical;
- 28/29 CSV tables were byte-identical;
- the only changed CSV was `s_curve_figure_manifest.csv`, whose output-file column embeds
  the temporary absolute build path;
- 16/40 SVG files were byte-identical; the remaining SVGs contained regenerated
  Matplotlib metadata/object identifiers, while their corresponding PNG renderings were
  byte-identical;
- all Markdown report files were byte-identical;
- `provenance_variability.json` differed only because the rebuild used a temporary output
  directory.

The six matched-content talker matrices were independently checked after the rebuild:

| Dataset | Shape per model variant | Matched items per off-diagonal cell |
|---|---:|---:|
| AN19 | 42 x 42 | 138 words |
| X21 | 11 x 11 | 32 experimental sentences |
| B23 | 4 x 4 | 120 sentences |

Every matrix was complete and symmetric with a zero diagonal. The 4,960 X21/B23
item-level DTW rows reaggregated to the corresponding matrix cells with a maximum absolute
difference of `5.329e-15`.

## Known pre-existing lineage gap

The X21 compatibility run metadata names
`results/derived/X21-same-content-glmm-input.csv` and
`results/derived/X21-same-content-glmm-input-ft.csv` as earlier fold-source files. They
were already absent before this curation. The legacy-axis-z model inputs actually used for
the retained summaries, the summaries themselves, and their provenance remain present.
This does not prevent final-report rebuilding, but it does prevent claiming a fully closed
from-scratch lineage for that historical compatibility run until the two earlier fold
sources are recovered or regenerated.

## Use policy

Use the final report for current figures and claims. Use `results/` only to rebuild or
audit the explicitly retained components. Do not restore superseded Git-history outputs
as current evidence without a new validation run and a new provenance-bearing output directory.

For GitHub publication, large HDF5 stores, direct compatibility GLMM inputs, and
item/recording-level AN19 validation tables remain local and are ignored rather than
deleted. Git publishes compact pair summaries, statistics, provenance, report-source
figures, and the reviewed final report. The retained-file SHA-256 manifest continues to
inventory the complete local retained set.
