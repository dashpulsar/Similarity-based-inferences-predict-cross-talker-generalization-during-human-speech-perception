# X21 dense parameter landscape — run record

Started September 18, following Zhengyang's explicit instruction to prioritize X21. The interrupted AN19 batch remains in `../sbi_parameter_landscape/`, labeled not for analysis. HVE remains paused.

## Scope

- HuBERT non-ASR-fine-tuned Tr-24, stored 3-D t-SNE.
- Training participant folds 1 and 2: 10,969 word responses, 213 participants. Fold 0 is not scored or used to select parameters.
- Preserve X21 keyword time slicing and current talker aggregation. New tau values trigger exact DTW recomputation; k values reuse distances.
- 29 tau values in [0.5,4], step 0.125; 43 k values (41 log-spaced values over [1e-6,2] plus exact 0.001 and 0.05).
- 1,247 grid fits plus 29 standardized-negative-distance limit fits. The latter represent the small-k limit after standardization, not literal k=0 constant similarity.
- Predictor-only binomial-logit GLMM with test-talker fixed blocking and participant/item random intercepts, matching the registered X21 structure; no condition term or penalty. bobyqa, nAGQ=1, maxfun=20000, no model-structure fallback.
- Optimize/display signed training Wald z and fitted training GLMM log likelihood. No ceiling normalization, held-out prediction claim, or optimizer-superiority test in this grid.

## Computation and safeguards

Eight independent R workers, one numerical thread per process; each worker retains R/lme4 in memory across its k sweep. Distances are prepared with four threads. The already-running eight-worker batch was retained after the CPU discussion to avoid discarding work. No claim of a measured 16-worker speedup is made.

30-minute overall budget and 10-minute per-tau subprocess limit. Partial tables are saved after each fit. The Windows x64 Rscript executable is called directly, allowing a timeout to terminate the actual fit process rather than a wrapper.

Numerical standardization uses an affine-equivalent shifted exponential: `a=-k*(d-min(d))`, then `expm1(a)` for exponent span below 0.1, otherwise `exp(a)`. This avoids cancellation at small k and all-range expm1 saturation at large k. Twelve checks of the actual R expressions passed, including large-distance underflow examples and the linear limit.

## Interpretation rules

tau<1 is a separately marked non-metric sensitivity (triangle inequality need not hold). The main metric-domain interpretation uses tau>=1. A maximum on a searched boundary leaves behavior outside that boundary unresolved. Dense-grid maxima are evaluated points, not proofs of global optimality. Connected 3-D surfaces and contour lines interpolate visually between fitted points; they are not additional fits.

Fit failures and singular estimates are retained in tables and excluded from the eligible surface. A broad near-best region is descriptive and is not a confidence region. This training-only landscape requires separate fold/held-out validation before making general predictive claims.

The runner records completion in `manifest.json`. Final figures and numerical interpretation are generated only after coverage and calculation checks pass.
