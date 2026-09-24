# Optimizer audit

Reviewed on September 17, 2026. This review inspected source code and saved notebook output; it did not rerun the historical analyses.

## What the current pipeline does

The current [analysis configuration](../configs/confirmatory.json) fixes `tau=2`, sets `select_tau=false` and `select_k=false`, and uses negative training-standardized distance for SBI. It contains no Optuna search. The [GLMM runner](../R/fit_confirmatory.R) uses participant-level three-fold train–test evaluation: fit on two folds, freeze the model and predictor scaling, and predict the third. There is currently no separate inner tuning split. The downstream [report selection function](../src/ctg/report_core.py), `_select_by_predictor_oof`, ranks layers or measures using their predictor-only held-out log loss. A selected winner has not received an additional independent test.

The historical Optuna search tuned the theoretical predictor, whereas `bobyqa` fitted each GLMM. These are two different optimization steps.

## Historical source mapping

`LEGACY` denotes the earlier sibling repository containing `glmm_prediction/`. `ORIGINAL` denotes the original `cross-talker-ASR` project. These are local historical sources, not additional dependencies of the current release. Notebook cell numbers below are one-based and include markdown cells.

| Source | Parameter search | Data used for selection and evaluation |
| --- | --- | --- |
| **LEGACY:** `glmm_prediction/nygaard_glmm.ipynb`, cells 4–6, calling `glmm_prediction/project_utils.py::process_layer_nygaard_l2` | Explicit `TPESampler(seed=42)`; `k` between 0.001 and 2; 20 trials by default; minimize `-z_validation + 0.1*k²`. DTW `tau` is fixed at 2 in the notebook calls. | Rotating one training, one validation and one test fold. The validation and test GLMMs are fitted separately on their own responses. |
| **ORIGINAL:** `cross-validation/final_3fold_nygaard1.ipynb`, cell 22 | Explicit TPE seed 42; `k` between 0.001 and 2; 50 trials in the call; minimize `-z_validation + 0.1*k²`. | Same three-way rotation and target refitting. This cell has no saved execution count, so its presence does not establish that it generated reported results. |
| **ORIGINAL:** `cross-validation/final_3fold_nygaard.ipynb`, cells 15 and 22 | `create_study(direction="minimize")`, with no explicit sampler or seed. Cell 15: `k` between 0.001 and 10, 20 trials. Cell 22: `k` between 0.001 and 0.5, 100 trials, plus validity filters. Both maximize signed training z. | Two training folds and one test fold; test GLMM refitted. A second, “corrected” pass uses the mean of the three selected k values. Saved execution counts: 11 and 30. |
| **ORIGINAL:** `cross-validation/final_3fold_bradlow.ipynb`, cell 12 | No explicit sampler or seed; `k` between 0.001 and 5; 20 trials; maximize signed training z. | Two training folds and one test fold; test refit, followed by a mean-k pass. Saved execution count: 57. |
| **ORIGINAL:** `cross-validation/final_3fold_xie5.ipynb`, cell 10 | SciPy `minimize_scalar(method="bounded")`; `k` between 0.001 and 2; minimize `-z_validation + alpha*k²`, with `alpha=0.1` by default. | One training, one validation and one test fold; validation and test GLMMs refitted. Saved execution count: 18. |
| **ORIGINAL:** `Reduction/variability_pipeline_tau.py` and `Reduction/dtw_pipeline_tau.py` | SciPy bounded scalar search; `tau` between 0.5 and 4; minimize `-z_training + alpha*tau²`, with `alpha=0.1` by default. Nonpositive z values are rejected. | Two training folds and one test fold; test GLMM refitted, then a second pass uses the mean selected tau across folds. |

The penalties matter: the regularized searches optimize a combination of z and parameter size. Describing them as pure z maximization would omit part of the objective. Other historical HVE notebooks contain additional variants; the table identifies the inspected implementations and does not establish that every historical HVE figure used one of these two scripts.

For the explicitly specified TPE runs, Optuna constructs density models for better and remaining objective values and uses these to propose subsequent trials. This description comes from the [official TPE documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html). It does not establish which sampler or library version was used in historical calls that omitted an explicit sampler.

## Exact evidence for the AN19 Optuna path

In **LEGACY** `glmm_prediction/project_utils.py`:

- Lines 671–677 define the k range and the penalized validation-z objective.
- Line 680 sets the default trial count to 20; lines 698–708 implement the three-way rotation and `TPESampler(seed=42)`.
- Lines 752 and 772 calculate `exp(-k * raw_distance)`; lines 769 and 780 apply training-derived scaling to both datasets.
- Lines 795–804 fit separate training and target models with `similarity_scaled + (1 + similarity_scaled | SubjectID)`.
- Lines 809–812 extract z and log likelihood from those separately fitted models.

The caller in `nygaard_glmm.ipynb`, cell 5, leaves the 20-trial default unchanged. Cell 6 contains successful saved output, including 54 result rows for 18 layers. However, the output refers to `../preprocessing/...` while the current cell source refers to `../data/features/...`; the saved output therefore predates at least some source edits. Also, the inspected `project_utils.py` references `optuna` without importing it. An import in the notebook does not supply that name to an imported module's global namespace. This historical path needs repair before a clean rerun can be claimed.

For HVE, **ORIGINAL** `Reduction/variability_pipeline_tau.py`, lines 248–262, contains the tau objective and search; lines 102–109 show test refitting; lines 280–300 implement the cross-fold mean-tau pass. The equivalent sections of `Reduction/dtw_pipeline_tau.py` are lines 378–395 and 419–447.

## Consequences for the next analyses

The three-way historical split and the current train–test split answer different questions. Historical target-refit z values summarize the association within that target sample. Frozen-model held-out likelihood evaluates predictions from a model fitted without those target responses. Both can be reported with their definitions, but their scores should remain distinguishable.

The historical mean-k and mean-tau passes also share tuning information across folds. They should remain separate from an independent held-out prediction analysis. The historical `optimism` quantity was `(mean_training_log_likelihood - mean_test_log_likelihood) / abs(mean_training_log_likelihood)`, using separately refitted models. It cannot be reused directly as the requested frozen-model training/test diagnostic.

Completed: identification of concrete algorithms, parameter ranges, penalties, seeds where explicitly specified, and split/refit conventions in the sources above.

Remaining work:

1. Link the specific historical figures discussed at the meeting to their exact source version, inputs and saved optimization results. Saved notebook output alone is insufficient for that link.
2. Compare z-based and likelihood-based predictor selection on matched data, folds, preprocessing and random-effects structures. The current fixed-parameter results do not complete that comparison.
3. Check stability across optimization starts or seeds, focusing on objective values and predictions as well as parameter values. Similar predictions can arise from different near-equivalent parameters.
4. Agree on whether selection analyses should use an inner tuning split. Preserve an untouched outer test set for evaluating a data-selected configuration. The separate diagnostic ratio was accepted in the [September 15 correspondence](CORRESPONDENCE_2026-09-15.md): mean test log loss divided by mean training log loss.

## Source fingerprints

These are the first 16 hexadecimal characters of each inspected file's SHA-256 hash. They identify the inspected snapshots; they do not certify which snapshot generated an older figure.

| Source file | SHA-256 prefix |
| --- | --- |
| LEGACY `glmm_prediction/project_utils.py` | `e805fb02e3e27827` |
| LEGACY `glmm_prediction/nygaard_glmm.ipynb` | `85420614da25a0bd` |
| ORIGINAL `cross-validation/final_3fold_nygaard1.ipynb` | `2c4d84cccd24c0cd` |
| ORIGINAL `cross-validation/final_3fold_nygaard.ipynb` | `ac5ec34558f23b75` |
| ORIGINAL `cross-validation/final_3fold_bradlow.ipynb` | `70e3bc2c04e23977` |
| ORIGINAL `cross-validation/final_3fold_xie5.ipynb` | `a0b63ae998aa33b8` |
| ORIGINAL `Reduction/variability_pipeline_tau.py` | `ac6655f0c4c45804` |
| ORIGINAL `Reduction/dtw_pipeline_tau.py` | `fd6584d8ff955a12` |
