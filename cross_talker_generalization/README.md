# Cross-talker generalization: production analysis pipeline

This directory is the repository's production analysis codebase. It reads behavioral data, manifests, and precomputed HDF5 features from the root `data/` directory and writes audits, intermediate tables, models, and figures to `artifacts/` or root-level `results/`. Superseded versions are available through Git history.

## Coverage

- Same-content talker similarity for AN19, X21, and B23;
- 18 layers from HuBERT base and HuBERT ASR fine-tuned;
- Primary 3-D t-SNE reproduction and full-dimensional sensitivity analysis;
- MFCC39 and STRF24 acoustic baselines;
- Minkowski DTW with explicit `tau` and `mean_sequence_length` / `path_length` normalization;
- Up to 16 registered HVE measures on identifiable actual-exposure pools;
- Participant-disjoint three-fold GLMMs, behavioral ceilings, S-curves, and talker distance matrices;
- A unified cross-dataset final report builder.

## Statistical semantics

`confirmatory_v1` is the default profile. Folds are participant-disjoint, predictor scaling is estimated on training folds only, and GLMMs are frozen before population-level prediction for unseen participants. The current report compares condition-only and joint models with held-out binomial log loss. Predictor-only GLMM likelihood, the intended theoretical-predictor optimization criterion, is not yet implemented as the report-selection rule; see [`../TODO.md`](../TODO.md).

`notebook_compatibility_v1` reproduces `exp(-k d)` and held-out-refit Wald z. These z values describe association stability across participant subsets; they are not cross-validated predictive performance. Figure 00 and the compatibility variability panel preserve this view; OOF figures report the separate incremental-prediction comparison.

## Quick start

Run from the repository root:

```powershell
conda env create -f cross_talker_generalization\environment.yml
conda activate cross-talker-generalization
$env:PYTHONPATH = "$PWD\cross_talker_generalization\src"
python -m ctg.cli audit `
  --project cross_talker_generalization\configs\project.json `
  --output cross_talker_generalization\artifacts\audit
```

Run one SBI analysis:

```powershell
& .\cross_talker_generalization\scripts\run_similarity.ps1 `
  -Dataset X21 -Store X21_hubert_base_tsne -Features tr_24 -Jobs 8
```

Omit `-Features tr_24` to process all registered layers and generate the layer-distance correlation matrix. Use `scripts/run_variability.ps1` for HVE.

Build a new final report:

```powershell
python -m ctg.cli build-report `
  --repository . `
  --output cross_talker_generalization\final_report_rebuild
```

## Documentation

- Execution guide: [docs/RUNBOOK.md](docs/RUNBOOK.md)
- Scientific specification: [docs/SCIENTIFIC_SPEC.md](docs/SCIENTIFIC_SPEC.md)
- Historical implementation audit: [docs/LEGACY_AUDIT.md](docs/LEGACY_AUDIT.md)
- Numerical validation: [docs/VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md)
- Repository-wide technical reference: [../TECHNICAL_DOCUMENTATION.md](../TECHNICAL_DOCUMENTATION.md)
- Current production report: [final_report_2026-08-21/](final_report_2026-08-21/)

## Current B23 ingestion boundary

The public B23 OSF archive contains the multi-talker stimulus lists and training data, but the production exposure builder does not yet ingest them. Current multi-talker HVE cells therefore remain marked `blocked`, while four single-talker pools are available. Integrating and validating the public mapping is tracked in [`../TODO.md`](../TODO.md).

This pipeline does not re-extract HuBERT or refit t-SNE. Supplied full-dimensional, 3-D t-SNE, MFCC39, and STRF24 HDF5 files are read-only inputs.
