# Cross-talker generalization: production analysis pipeline

This directory is the repository's only production analysis codebase. It reads behavioral data, manifests, and precomputed HDF5 features from the root `data/` directory and writes audits, intermediate tables, models, and figures to `artifacts/` or root-level `results/`. Superseded code and notebooks are archived under `recycle_bin/` and are not runtime dependencies.

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

`confirmatory_v1` is the default profile. Folds are participant-disjoint, predictor scaling is estimated on training folds only, and GLMMs are frozen before population-level prediction for unseen participants. Performance is evaluated with binomial log loss.

`notebook_compatibility_v1` reproduces `exp(-k d)` and held-out-refit Wald z. These z values describe association stability across participant subsets; they are not cross-validated predictive performance. Figure 00 and the compatibility variability panel preserve this view, while the OOF figures and tables contain the primary predictive evidence.

## Quick start

Run from the repository root:

```powershell
$env:PYTHONPATH = "$PWD\cross_talker_generalization\src"
conda run --no-capture-output -n BayesPCN python -m ctg.cli audit `
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
conda run --no-capture-output -n BayesPCN python -m ctg.cli build-report `
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

## Known data boundary

The repository does not contain the true B23 20/20/20 sentence-to-talker assignment for multi-talker exposure, including the noSPA presentation error. The corresponding actual-exposure HVE cells are explicitly marked `blocked`. Four single-talker pools remain identifiable. Recovering the assignment would require adding exposure mappings, not rewriting the DTW/HVE core.

This pipeline does not re-extract HuBERT or refit t-SNE. Supplied full-dimensional, 3-D t-SNE, MFCC39, and STRF24 HDF5 files are read-only inputs.
