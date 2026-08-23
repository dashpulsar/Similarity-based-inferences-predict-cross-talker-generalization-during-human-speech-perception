# File guide

This document describes where files belong. Scientific rationale is kept in [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md), technical definitions in [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md), executable commands in the [runbook](cross_talker_generalization/docs/RUNBOOK.md), and open work in [TODO.md](TODO.md).

## Repository root

| Path | Purpose | Status |
|---|---|---|
| `README.md` | Entry point, setup, and current analysis-status notice | Production |
| `PROJECT_DESCRIPTION.md` | Scientific scope, predictors, representations, and dataset designs | Production |
| `TECHNICAL_DOCUMENTATION.md` | Equations, data contracts, statistical semantics, and implementation details | Production |
| `DOCUMENTATION_CHANGES.md` | Post-review wording changes, previous claims, and reasons | Review record |
| `TODO.md` | Remediable analysis and documentation gaps | Active |
| `cross_talker_generalization/` | Production code, configuration, tests, artifacts, and report package | Production |
| `data/` | Behavioral data, manifests, stimulus sources, and local feature-store location | Production input |
| `results/` | Compact report dependencies and compatibility inputs | Supporting input |
| `references/` | Prior papers and manuscript reference material | Reference |
| `.vscode/` | Editor settings | Auxiliary |

Superseded versions are retained in Git history. They are not duplicated in a separate archive directory on the main branch.

## Production analysis directory

| Path | Description |
|---|---|
| `configs/project.json` | Dataset, feature-store, layer, fold, and seed registry |
| `configs/confirmatory.json` | Current DTW, aggregation, transformation, and cross-validation settings |
| `configs/notebook_compatibility.json` | Historical `exp(-k d)` and held-out-refit z settings |
| `configs/sensitivities/` | Alternative path normalization, `tau`, and multi-talker aggregation settings |
| `src/ctg/` | Auditing, pairing, DTW, HVE, folds, model inputs, plotting, and report builders |
| `R/` | R/lme4 GLMM and compatibility-ceiling scripts |
| `scripts/` | PowerShell analysis entry points |
| `tests/` | Unit and project-contract tests |
| `docs/RUNBOOK.md` | Commands from input audit through report generation |
| `docs/SCIENTIFIC_SPEC.md` | Concise estimand and model specification |
| `docs/LEGACY_AUDIT.md` | Differences between historical descriptions and implementations |
| `docs/VALIDATION_REPORT.md` | Numerical and statistical checks already run |
| `artifacts/` | Intermediate tables, model outputs, and smoke-test products |
| `final_report_2026-08-21/` | Current reviewed report package |

## Current report package

`cross_talker_generalization/final_report_2026-08-21/` is the current source for presentation figures and their underlying tables. It contains:

- `README.md`: figure descriptions and interpretation notes;
- `PRESENTATION_OUTLINE.md`: suggested presentation order;
- `FINAL_VERIFICATION_REPORT.md`: package-level checks;
- `VARIABILITY_NOTES.md`: HVE-specific notes;
- `figures/`: PNG and SVG outputs;
- `tables/`: figure source data and manifests;
- `provenance*.json`: parameters, environment versions, and file hashes.

The report predates the corrections listed in [TODO.md](TODO.md): its “best” feature-space labels use condition-only versus joint OOF log-loss gain, and its B23 HVE coverage does not include the public multi-talker stimulus mapping. Treat it as the current reviewed output package, not as evidence that those two open tasks are complete.

## Result authority and retained sources

The dated report directory is the presentation layer. The top-level `results/` directory is not an alternative report; it contains compact sources still used by the report builder, including historical compatibility summaries. [results/README.md](results/README.md) records the retained categories, and [results/CURATION_REPORT_2026-08-21.md](results/CURATION_REPORT_2026-08-21.md) records the previous selection process.

Presence or file timestamp alone does not establish statistical status. Check the source table, model definition, and provenance. In particular, held-out-refit Wald z is an association statistic, whereas frozen participant-held-out log loss is an out-of-sample prediction score.

## Path and output conventions

- Paths in JSON configuration files are resolved relative to the configuration file.
- Run `scripts/*.ps1` or `python -m ctg.cli` from the repository root.
- HDF5 feature inputs belong under `data/features/` and are excluded from Git.
- Large participant/item-level derived CSV files are local-only; compact summaries and provenance remain tracked.
- Write new analyses and report builds to new directories until they have been audited; do not overwrite the reviewed report in place.
