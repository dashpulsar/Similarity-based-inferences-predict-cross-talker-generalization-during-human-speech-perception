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

The audit checks three behavioral datasets, three manifests, 15 physical HDF5 stores, and the virtual AN19 acoustic-diagnostic view of the existing baseline store. It uses fast fingerprints for large files by default. Add `--hash-large-files` before creating a release archive to compute full SHA-256 hashes.

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

## 3. AN19 acoustic-control diagnostic

The diagnostic store reads declared subsets of the existing acoustic HDF5 arrays; it does not create or modify a feature file. Run all first-stage MFCC and STRF groups with:

```powershell
& .\cross_talker_generalization\scripts\run_similarity.ps1 `
  -Dataset AN19 -Store AN19_acoustic_diagnostic -Jobs 8
```

To run one group first, add for example `-Features mfcc_static13` or `-Features strf_rate_8`. The available groups are declared under `AN19_acoustic_diagnostic.feature_subsets` in `configs/project.json`. Each group is standardized from the complete AN19 acoustic corpus and then uses the same pair table, DTW profile, participant folds, and GLMM code as the full MFCC39/STRF24 baselines.

This run diagnoses which feature family carries the AN19 result. It does not by itself test duplicate recordings, condition separation, or individual dimensions; those checks remain listed in the root `TODO.md`.

The September 17 correction explicitly maps `acoustic_subset` to `global_z` in the main and sensitivity profiles. Earlier component distance files marked `coordinate_scaling=none` remain historical outputs. Use a new output prefix for corrected runs; the [September 17 update](../analysis_update_2026-09-17/README.md) records the matched-standardizer rerun.

### Optional common-random-structure sensitivity

The default GLMM policy remains `registered`. To reproduce the declared AN19 participant/item-only sensitivity using the corrected 16-feature input and existing six within-condition inputs:

```powershell
python cross_talker_generalization\scripts\run_an19_common_structure.py --jobs 4
```

The [runner](../scripts/run_an19_common_structure.py) writes a separate `analysis_update_2026-09-17/common_structure/` package and preserves the primary results. It fixes the existing AN19 fallback structure, `(1 | participant_id) + (1 | analysis_item_id)`, before fitting; no further term is removed automatically. Generic `fit-glmm` and `fit-glmm-parallel` calls also expose `--random-policy participant_item`; use this option only for a declared sensitivity in a distinct output directory. Omitting it retains the registered policy.

Both policies now export `variance_components.csv` alongside coefficients, diagnostics and matched split scores. Policy and source hashes enter cache validation. The sensitivity's 352 selected fits include 20 participant-variance boundary fits; consult the [result metadata](../analysis_update_2026-09-17/common_structure/result_metadata.json) and [work log](../analysis_update_2026-09-17/WORK_LOG_2026-09-17.md) for their scope. Boundary variance, additional convergence messages and fit failure are separate diagnostics. Rebuild comparison tables without refitting by adding `--report-only`.

## 4. HVE / exposure variability

```powershell
& .\cross_talker_generalization\scripts\run_variability.ps1 `
  -Dataset X21 -Store X21_hubert_base_tsne `
  -Features tr_24 -Measures overall within_token_word mean_dissimilarity_word `
  -Jobs 8
```

Omit `-Measures` to run every registered measure that is defined for the dataset. AN19 has isolated-word exposure, so sentence/phoneme measures are unsupported. B23 uses the normalized participant-level public exposure table under `data/exposure_presentations/`; only the global order-sensitive measure is unavailable for participants whose public trial indices are incomplete.

X21 defaults to presentation weighting: 16 tokens in Single-talker and Talker-specific conditions each appear five times. The task table also retains the unique-token structure.

## 5. Compatibility behavioral ceiling

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

### Matched training/test prediction diagnostics

New `fit-glmm` and `fit-glmm-parallel` runs also write `train_test_scores.csv`. Training and held-out rows are scored by the same training model with fixed-effect predictions and frozen training scaling. `total_trials` counts word responses, including all responses represented by grouped binomial rows. Existing `cv_metrics.csv` is unchanged.

The approved SI diagnostic is `mean_test_log_loss / mean_training_log_loss`, as recorded in the [September 15 correspondence](../analysis_update_2026-09-17/CORRESPONDENCE_2026-09-15.md). Both scores use the same training-fitted model, training mean/SD and `re.form=NA`; their denominators are word-response counts. Compute the ratio separately for each paired fold, then show the mean, three fold points, a ratio=1 reference, and a 95% percentile interval from all 27 ordered bootstrap resamples of the three fold ratios. The interval describes fold variability. The [scientific specification](SCIENTIFIC_SPEC.md#matched-trainingtest-diagnostic-scores) defines invalid-pair handling and interpretation.

After the model runs listed in the manifest have completed, generate the SI package with the [approved ratio plotting script](../scripts/build_train_test_ratio_figures.py):

```powershell
python cross_talker_generalization\scripts\build_train_test_ratio_figures.py `
  --manifest cross_talker_generalization\analysis_update_2026-09-17\diagnostic_inputs.json `
  --output cross_talker_generalization\analysis_update_2026-09-17\si_diagnostics
```

The manifest declares each run's model directory, predictor family, variant, HVE measure and participant stratum. Preserve distinct B23 participant strata rather than pooling them. The builder retains invalid pairs and fit warnings, and writes paired-fold tables, three-fold summaries, a figure inventory and source hashes. Fewer than three valid ratios do not produce a three-fold mean/interval. These are fixed-predictor diagnostics, separate from the pending optimization-objective comparison. They do not change the current train-test design; nested CV remains unconfirmed. Raw-loss ratios are not algebraically equivalent to baseline-gain ratios. The earlier `build_train_test_diagnostics.py` remains available for separate training/test loss plots; z/ceiling plots keep their existing meaning. See the [work log](../analysis_update_2026-09-17/WORK_LOG_2026-09-17.md) for actual run completion and coverage.

## 6. Figures

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

## 7. Tests

```powershell
$env:PYTHONPATH = "$PWD\cross_talker_generalization\src"
python -m unittest discover `
  -s cross_talker_generalization\tests -v
```

## 8. Final report

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
