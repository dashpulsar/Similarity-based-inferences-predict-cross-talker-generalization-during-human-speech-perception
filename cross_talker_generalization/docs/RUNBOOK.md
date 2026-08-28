# Runbook

All paths declared in JSON are resolved relative to the JSON file. The commands below are run from the repository root.

## 0. Environment and audit

```powershell
conda env create -f cross_talker_generalization\environment.yml
conda activate cross-talker-generalization
$env:PYTHONPATH = "$PWD\cross_talker_generalization\src"
python -m ctg.cli audit `
  --project cross_talker_generalization\configs\project.json `
  --output cross_talker_generalization\artifacts\audit
```

The audit checks three behavioral datasets, three manifests, and 15 HDF5 stores. It uses fast fingerprints for large files by default. Add `--hash-large-files` before creating a release archive to compute full SHA-256 hashes.

## 1. SBI: automated entry point

Smoke test:

```powershell
& .\cross_talker_generalization\scripts\run_similarity.ps1 `
  -Dataset X21 -Store X21_hubert_base_tsne -Features tr_24 -Jobs 8
```

All 18 layers:

```powershell
& .\cross_talker_generalization\scripts\run_similarity.ps1 `
  -Dataset X21 -Store X21_hubert_base_tsne -Jobs 8
```

All store IDs are listed in `configs/project.json`. Full-dimensional HuBERT and acoustic stores are standardized per dimension using corpus-global moments; 3-D t-SNE retains its supplied coordinates.

The script runs `build-pairs` → `make-folds` → optional `fit-standardizers` → `compute-distances` → `aggregate` → `make-model-input` → `fit-glmm` → `plot-profile`. Multi-layer runs also call `plot-distance-correlations`.

## 2. SBI: step-by-step execution

```powershell
python -m ctg.cli build-pairs `
  --dataset X21 --project cross_talker_generalization\configs\project.json `
  --output cross_talker_generalization\artifacts\derived\X21-pairs

python -m ctg.cli compute-distances `
  --pairs cross_talker_generalization\artifacts\derived\X21-pairs\pairs.csv `
  --store X21_hubert_base_tsne --features tr_24 `
  --project cross_talker_generalization\configs\project.json `
  --profile cross_talker_generalization\configs\confirmatory.json `
  --jobs 8 `
  --output cross_talker_generalization\artifacts\derived\X21-distances.csv

python -m ctg.cli aggregate `
  --cells cross_talker_generalization\artifacts\derived\X21-pairs\cells.csv `
  --distances cross_talker_generalization\artifacts\derived\X21-distances.csv `
  --profile cross_talker_generalization\configs\confirmatory.json `
  --output cross_talker_generalization\artifacts\derived\X21-predictors.csv
```

`distances.csv` retains accumulated DTW cost, path length, frame counts, normalized distance, and `exp(-d)`. The confirmatory model uses raw distance and does not search for a `k` that maximizes Wald z.

Alternative analysis settings are under `configs/sensitivities/`: `path_length.json`, `min_distance.json`, `tau_1.json`, and `tau_3.json`. Each setting must use a distinct output path.

## 3. HVE / exposure variability

```powershell
& .\cross_talker_generalization\scripts\run_variability.ps1 `
  -Dataset X21 -Store X21_hubert_base_tsne `
  -Features tr_24 -Measures overall within_token_word mean_dissimilarity_word `
  -Jobs 8
```

Omit `-Measures` to run every registered measure that is defined for the dataset. AN19 has isolated-word exposure, so sentence/phoneme measures are unsupported. B23 uses the normalized participant-level public exposure table under `data/exposure_presentations/`; only the global order-sensitive measure is unavailable for participants whose public trial indices are incomplete.

X21 defaults to presentation weighting: 16 tokens in Single-talker and Talker-specific conditions each appear five times. The task table also retains the unique-token structure.

## 4. Compatibility behavioral ceiling

```powershell
python -m ctg.cli make-ceiling-input `
  --project cross_talker_generalization\configs\project.json --dataset X21 `
  --folds cross_talker_generalization\artifacts\derived\X21-folds.csv `
  --output cross_talker_generalization\artifacts\derived\X21-ceiling-input.csv

python -m ctg.cli fit-ceiling-compatibility `
  --input cross_talker_generalization\artifacts\derived\X21-ceiling-input.csv `
  --output cross_talker_generalization\artifacts\models\X21-ceiling-compatibility
```

Each fold's item log odds are estimated from the other two participant folds. The three resulting z values are held-out-refit association statistics, not frozen-model OOF predictions.

## 5. Figures

```powershell
python -m ctg.cli plot-s-curves `
  --input cross_talker_generalization\artifacts\derived\X21-model-input.csv `
  --feature tr_24 --bins 10 `
  --output cross_talker_generalization\artifacts\figures\X21-tr24-s-curves

python -m ctg.cli plot-distance-correlations `
  --input cross_talker_generalization\artifacts\derived\X21-distances.csv `
  --output cross_talker_generalization\artifacts\figures\X21-distance-correlations
```

S-curve points are trial-count-weighted accuracy in predictor quantile bins with Wilson 95% intervals. Curves are descriptive binomial logistic fits. Correlation matrices accept physical pair/cell raw distances and reject tables replicated by participant, fold, or response.

## 6. Tests

```powershell
$env:PYTHONPATH = "$PWD\cross_talker_generalization\src"
python -m unittest discover `
  -s cross_talker_generalization\tests -v
```

## 7. Final report

The report builder merges SBI, acoustic baselines, all available variability profiles, S-curves, and talker matrices. The output directory must not exist:

```powershell
$env:PYTHONPATH = "$PWD\cross_talker_generalization\src"
python -m ctg.cli build-report `
  --repository . `
  --output cross_talker_generalization\final_report_rebuild
```

The reviewed broad August 21 package is `cross_talker_generalization/analysis_update_2026-08-21/`. Build to a new directory, inspect `FINAL_VERIFICATION_REPORT.md`, file counts, and provenance, and only then promote it.

## Parallelism

DTW and HVE are parallelized by feature layer. Every worker owns one HDF5 handle and restricts BLAS to one thread to prevent `jobs × BLAS threads` oversubscription. The default is eight jobs. Reduce the job count for full-dimensional layers if memory use is high.
