# Results directory policy

## Authoritative release

The authoritative result package is
[`cross_talker_generalization/analysis_update_2026-08-21/`](../cross_talker_generalization/analysis_update_2026-08-21/).
Use that package for interpretation, presentation, figure selection, and manuscript-facing
tables. It contains the reviewed figures, their source tables, verification records, and
file-level SHA-256 provenance.

This `results/` directory is not a second report. It is a curated source layer retained
because parts of the final-report builder still read historical notebook-compatible
summaries or validated intermediate products. Most retained files were generated on
2026-08-14 or 2026-08-15. Their age is recorded, but modification time alone is not used
as a scientific validity criterion.

## What remains here

| Path | Why it is retained | Evidential role |
|---|---|---|
| `statistics/` | Direct input to Figure 00 and the compatibility variability panels | Historical held-out-refit association only; not true OOF prediction |
| `figures/*notebook-s-curves*` | Source figures and trial tables copied into the final report | Descriptive S-curves |
| `figures/AN19-*-talker-similarity-*` | Validated AN19 Tr-24 matrices used by the report builder | Matched-content talker-distance source |
| `figures/X21-base-tr12-representation-schematic-v1/` | Method schematic with recorded source inputs and build parameters | Supporting illustration; not a behavioral result |
| `derived/AN19-talker-validation-*-tr24/` | Complete local AN19 validation products; Git publishes compact pair summaries and provenance | Direct source for the 42 x 42 report matrices |
| selected files at `derived/` root | Local direct inputs and published provenance for compatibility GLMM summaries | Audit and rerun support for compatibility analyses only |

The participant-held-out SBI and HVE products are under
[`cross_talker_generalization/artifacts/`](../cross_talker_generalization/artifacts/),
and their reviewed presentation is in the final report. They should not be inferred from
the compatibility z-value directories in `results/statistics/`.

## Reliability boundary

- A file is **primary** only when the final report identifies it as a true OOF result or
  uses it to construct a verified matched-content matrix.
- Files labeled **compatibility** reproduce the earlier notebook analysis. Their Wald z
  values come from models refit on held-out responses and therefore measure association
  stability, not frozen-model prediction of unseen responses.
- Retention does not mean endorsement as a current headline result. It means that the
  file remains necessary for report reconstruction, auditing, or a clearly identified
  supporting purpose.
- Two X21 compatibility metadata files name the earlier fold-source inputs
  `X21-same-content-glmm-input.csv` and `X21-same-content-glmm-input-ft.csv`. Those two
  files were already absent before this curation. The actual legacy-axis-z model inputs,
  result summaries, and their provenance remain available. This pre-existing lineage gap
  must be resolved before claiming a full from-scratch reproduction of that compatibility
  run.

## Superseded material

Superseded tracked versions remain available through Git history. They are not maintained
as a second results tree on the main branch.

`RETAINED_FILES_SHA256.csv` records the role, size, original modification time, and SHA-256
digest of every retained file other than the manifest itself.
The selection criteria, before/after counts, and post-move rebuild checks are recorded in
[`CURATION_REPORT_2026-08-21.md`](CURATION_REPORT_2026-08-21.md).

New analyses must write to a new, provenance-bearing directory. Do not overwrite the
reviewed final report or reuse a retained compatibility directory for a new estimand.

## Git distribution boundary

GitHub contains this policy, the curation report, the retained-file manifest, compact
statistics and provenance, report-source figures, and the AN19 talker-pair summaries used
by the report builder. Large direct GLMM inputs and item/recording-level AN19 validation
tables remain local and are ignored by Git. `RETAINED_FILES_SHA256.csv` inventories both
published and local-only retained files so that the complete workspace can still be
audited.
