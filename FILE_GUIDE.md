# File guide

## Repository root

| Path | Purpose | Status |
|---|---|---|
| `README.md` | Main entry point and shortest verification instructions | Production |
| `PROJECT_DESCRIPTION.md` | Research overview for project readers | Production |
| `TECHNICAL_DOCUMENTATION.md` | Complete experimental, computational, statistical, and terminology reference | Production |
| `cross_talker_generalization/` | The only production codebase, configuration set, test suite, and report package | Production |
| `data/` | Behavioral data and tracked manifests; large feature stores are local-only and read-only | Production input |
| `results/` | Published summaries plus local-only compatibility inputs and validated supporting sources | Secondary result sources |
| `references/` | Prior method paper and current manuscript reference files | Reference |
| `recycle_bin/` | Local-only superseded content; Git tracks only its policy README | Recoverable archive |
| `.vscode/` | Editor environment settings | Auxiliary |

An empty `glmm_prediction/` shell may temporarily remain at the root. All of its files have been archived. If the empty directory is still visible, VS Code or a notebook kernel is holding a Windows directory handle; it is not a second production codebase.

## Production analysis directory

Key contents of `cross_talker_generalization/`:

| Path | Description |
|---|---|
| `configs/project.json` | Three datasets, 15 feature stores, 18 layers, fold count, and random seed |
| `configs/confirmatory.json` | Primary profile: `tau=2`, mean-length DTW, negative standardized distance, OOF log loss |
| `configs/notebook_compatibility.json` | Historical `exp(-k d)` and held-out-refit z compatibility settings |
| `configs/sensitivities/` | Path-length, `tau=1/3`, and min-distance variants |
| `src/ctg/` | Audit, pairing, DTW, HVE, folds, model input, plotting, and report builders |
| `R/` | R/lme4 GLMM and compatibility-ceiling scripts |
| `scripts/` | PowerShell analysis entry points and the final report builder |
| `tests/` | Python unit and integration tests |
| `docs/RUNBOOK.md` | Commands from input audit through final figures |
| `docs/SCIENTIFIC_SPEC.md` | Concise scientific estimand and limitation specification |
| `docs/LEGACY_AUDIT.md` | Audit of differences between historical text and implementation |
| `docs/VALIDATION_REPORT.md` | Numerical and statistical checks that were actually run |
| `artifacts/` | Audit tables, intermediate data, models, and smoke-test figures |
| `final_report_2026-08-21/` | The current formal report package |

## Final report package

The production report contains:

- `README.md`: figure semantics, principal findings, and interpretation limits;
- `PRESENTATION_OUTLINE.md`: recommended presentation order;
- `FINAL_VERIFICATION_REPORT.md`: release-package validation;
- `MERGE_VERIFICATION.md`: report-component merge record;
- `VARIABILITY_NOTES.md`: detailed variability interpretation;
- `figures/`: PNG and SVG figures;
- `tables/`: figure source data, audit tables, and manifests;
- `provenance.json` and `provenance_variability.json`: environment, parameters, and hashes.

Figures should be interpreted together with their source tables and README. In particular, the percentages in Figure 00 and the compatibility variability figure rescale Wald z by mean ceiling z. They are not the percentage of human behavior predicted. Participant-held-out prediction is reported through OOF log-loss gain.

## Result authority and retained sources

`cross_talker_generalization/final_report_2026-08-21/` is the single authoritative
release for current figures and conclusions. `results/` is a dependency and audit layer,
not an alternative report. Its own [README](results/README.md) identifies every retained
category and distinguishes true OOF evidence from historical held-out-refit compatibility
statistics. The detailed selection and rebuild record is
[`results/CURATION_REPORT_2026-08-21.md`](results/CURATION_REPORT_2026-08-21.md).

Outputs that were neither required by the report builder nor independently useful were
moved to `recycle_bin/results_old_20260821/`. The archive contains no production runtime
dependency and can be restored selectively if an old computation must be inspected.

## Path conventions

- Relative paths in `configs/*.json` are resolved relative to the configuration file, not the shell working directory.
- Run `scripts/*.ps1` or `python -m ctg.cli` from the repository root.
- Production code has no runtime dependency on `recycle_bin/`.
- HDF5 inputs remain locally in `data/features/` and are excluded from Git hosting.
- Large participant/item-level CSV intermediates under `results/derived/` are also
  local-only; compact summaries, statistics, and provenance remain tracked.
- New runs should use new artifact/result directories rather than overwrite auditable outputs.

## Archive and recovery

No project file was permanently deleted during reorganization. Local archive payloads are
excluded from Git hosting; their policy is documented in
[`recycle_bin/README.md`](recycle_bin/README.md). In the complete local workspace, restore
an individual legacy file to a temporary review directory rather than restoring an entire
legacy tree to the repository root.
