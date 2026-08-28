# Cross-talker generalization: production analysis pipeline

This directory is the repository's production analysis codebase. It reads behavioral data, manifests, and precomputed HDF5 features from the root `data/` directory and writes audits, intermediate tables, models, and figures to `artifacts/` or root-level `results/`. Superseded versions are available through Git history.

## Coverage

- Same-content talker similarity for AN19, X21, and B23;
- 18 layers from HuBERT base and HuBERT ASR fine-tuned;
- Primary 3-D t-SNE reproduction and full-dimensional sensitivity analysis;
- MFCC39 and STRF24 acoustic baselines;
- Minkowski DTW with explicit `tau` and `mean_sequence_length` / `path_length` normalization;
- Up to 17 registered HVE measures on identifiable actual-exposure presentations;
- Participant-disjoint three-fold GLMMs, behavioral ceilings, S-curves, and talker distance matrices;
- A unified cross-dataset final report builder.

## Statistical semantics

`confirmatory_v1` is the default profile. Folds are participant-disjoint, predictor scaling is estimated on training folds only, and GLMMs are frozen before population-level prediction for unseen participants. Candidate theoretical predictors are selected by held-out log loss from the predictor-only GLMM. Condition-only, predictor-only, and joint models are then compared without re-optimizing the selected predictor.

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
- Reviewed broad August 21 analysis: [analysis_update_2026-08-21/](analysis_update_2026-08-21/)
- Corrected SBI/HVE selection and downstream comparison update: [analysis_update_2026-08-27/](analysis_update_2026-08-27/)

## B23 exposure source status

The production builder integrates the public B23 stimulus lists and participant-level training data for all eight trained conditions. It uses actual filenames to map recordings. Order-independent HVE remains available for every trained participant; the global order-sensitive measure is unavailable when the public trial indices are incomplete.

This pipeline does not re-extract HuBERT or refit t-SNE. Supplied full-dimensional, 3-D t-SNE, MFCC39, and STRF24 HDF5 files are read-only inputs.
