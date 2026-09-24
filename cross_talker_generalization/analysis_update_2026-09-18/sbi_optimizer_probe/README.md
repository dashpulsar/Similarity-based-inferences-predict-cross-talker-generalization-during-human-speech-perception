# Joint tau/k optimization: small feasibility test

## Result

Joint search of tau and k is operational with the existing hard-DTW definition. The run finished in **101.21 seconds**, including preparation, nine initial parameter pairs, and a tiny three-method/two-objective comparison. There were **39 distinct GLMM fits**, all converged with no singular estimates. No held-out responses were scored. This establishes feasibility for this AN19 word-level example, not a preferred optimizer or final SBI results.

| Method | Best tau | Best k | Signed training z | Training GLMM log-likelihood |
| --- | ---: | ---: | ---: | ---: |
| Grid | 1.000000 | 0.001000 | 4.755987 | -1534.443085 |
| Optuna-TPE | 1.043301 | 0.001351 | 4.735487 | -1534.531817 |
| Differential evolution | 1.124847 | 0.003335 | 4.690628 | -1534.722775 |

Both objective searches selected the same pair within each method in this tiny run. This does not establish equivalence of z and likelihood objectives in general. The grid's best pair lies on **both lower bounds**. It is the best evaluated pair within the pilot, not a demonstrated global optimum. The grid visits that corner first, so its flat best-so-far line must not be interpreted as evidence that grid search generally converges instantly.

## Scope and definitions

- AN19, non-ASR-fine-tuned HuBERT, stored 3-D t-SNE, Tr-24 only.
- Existing outer fold 0 was excluded; folds 1 and 2 supplied 3,791 valid word responses from 80 participants.
- The same responses were retained for every candidate. Control and unavailable-predictor rows retain the existing SBI exclusion policy.
- Recomputed 5,394 physical word pairs needed by these training responses. No audio extraction or t-SNE recomputation.
- DTW uses Minkowski exponent tau and mean-sequence-length normalization, without coordinate standardization of t-SNE.
- Existing aggregation retained: average physical distances within each source talker, then equally across source talkers, then compute `exp(-k * mean_distance)`. This is not averaging pair-level similarities. It is a declared pilot using the current mapping, not certification of every historical notebook's aggregation.
- Every candidate similarity is standardized using training rows only.
- Fixed binomial-logit model: `similarity_z + (1 | participant_id) + (1 | analysis_item_id) + (1 | test_talker_id)`. No condition predictor, no automatic random-structure fallback, no k penalty, no absolute-z objective.
- Inner fit: lme4 1.1-35.5, `bobyqa`, `maxfun=20000`, Laplace approximation (`nAGQ=1`). This fixed structure is a feasibility specification, not a silent replacement for historical random-slope models or production fallback policies.
- Outer objectives: maximize signed training Wald z; separately maximize the fitted training GLMM `logLik`. These are not held-out prediction scores. No joint gradient-based GLMM/SBI estimator was implemented.

## Search settings

The preliminary box was **tau in [1, 3], k in [0.001, 2]**. These bounds are for a bounded feasibility run and require review before a final experiment. The nine-point probe used tau `{1,2,3}` and k `{0.001,0.05,2}`.

Each method/objective received **12 candidate proposals**, using tau and `log10(k)` as coordinates:

- Grid: three equally spaced tau values and four log-spaced k values. No local refinement yet.
- Optuna 3.6.1: TPE, seed 42, five startup trials. This deliberately small setting differs from historical search budgets and is not a historical replication.
- SciPy 1.13.1: differential evolution, seed 42, population multiplier 3 (six members), one generation after initialization, no polishing, zero relative/absolute convergence tolerance.

There were 72 comparison proposals plus nine initial probe pairs. Exact repeated candidates reused fits; repeated tau reused distances. In this run the two objective searches revisited the same candidates. Cached objective values were used only when proposed; previous candidates were not supplied as extra observations to a search algorithm. **Recorded elapsed times are cache-dependent and cannot rank algorithm speed fairly.** A proper timing comparison would standardize cache/startup/resource conditions and repeat seeds.

## Cost and stopping decision

Initial median timing:

- New tau: **0.121 seconds** for word-pair DTW plus aggregation/output, using four DTW threads and preloaded features.
- GLMM evaluation, including starting R: **2.374 seconds**.
- Feature loading and kernel warmup: **1.343 seconds**.

A larger illustrative schedule (three methods × two objectives × three seeds × 40 proposals = 720 evaluations) is estimated at **about 30 minutes for this one layer and one training split**, assuming serial fits and a new tau on every call. The estimate is approximate: caching/parallelism can reduce it, while difficult fits and longer sequences can increase it. It should not be transferred directly to X21/B23.

The script enforced a ten-minute wall-time budget and a 90-second timeout for an individual R subprocess. The small comparison finished well within that cap. **No larger search was launched**, consistent with Zhengyang's request to defer expensive execution. HVE remains paused.

## Validation and plots

- Recomputed tau=2 distances reproduce the stored reference within `1.42e-14`.
- All candidate tables contain identical response identities/counts and exclude fold 0.
- Reconstructed `exp(-k*d)` means/SDs agree with R; reported z agrees with coefficient/SE.
- Trace scores, running best values and twelve-proposal coverage were checked.
- All 39 fits converged; no singular estimates. Existing R locale warnings and a Matrix build-version warning are retained in per-fit logs.
- Both PNGs were visually checked for axes, labels and clipping.

[Parameter probe](parameter_probe.png): nine computed pairs, with no interpolated surface or inferred optimum.

[Search progress](optimizer_progress.png): best training objective against candidate count, one panel per objective. No confidence intervals are shown because there is only one seed and training split.

Detailed tables: [method summary](method_summary.csv), [all fits](evaluations.csv), [search trace](search_trace.csv), [DTW timing](dtw_timings.csv), and [validation](validation.json). Input/code hashes and package versions are in [manifest.json](manifest.json).

## Next decision

Check the lower-bound behavior before expanding the benchmark. For small k, `exp(-k*d)` approaches `1-k*d`; after standardization, this approaches negative standardized distance. Similar scores over small k could therefore indicate weak identification of k, rather than an optimizer failure. This mathematical observation has not been established as the explanation for the present result. The next bounded check can examine this region, then repeat a modest search across seeds/splits if worthwhile. No superiority claim, inferential p-value, ceiling-normalized result, or independent predictive validation follows from this pilot.

## Reproduce

Run from the repository root, supply an installed Rscript executable, and use a **fresh** output directory (the runner rejects an existing directory):

```text
python cross_talker_generalization/scripts/probe_sbi_joint_optimization.py --output <fresh-directory> --rscript <Rscript-executable> --budget-seconds 600 --workers 4 --benchmark
python cross_talker_generalization/scripts/summarize_sbi_optimizer_probe.py --input <fresh-directory>
```

No production results, previous figures, HDF5 contents, or Git history were modified.
