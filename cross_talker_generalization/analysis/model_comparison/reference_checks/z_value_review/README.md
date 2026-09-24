# SBI and HVE z-value review

Open [the figure gallery](index.html). This review makes **layerwise z the main SBI result display**. It does not designate manuscript main figures or replace Figures 1/2, the Tr-24 summary, or nested-model comparisons.

Read the [numbered Florian requirements report](../../../../docs/FLORIAN_REQUIREMENTS_REVIEW.md) for the meeting/email evidence, complete available figure inventory, and unfinished work. The [plot specification](../../../../docs/Z_VALUE_FIGURE_SPEC.md) defines every statistic and line.

## What was generated

1. **SBI:** six dataset-by-variant figures. Each contains the requested normalized-z display and a raw-z companion. MFCC39 and STRF24 are the first two points, followed by 18 registered HuBERT layers. Gray dots are the three fold estimates, black points/lines are means with 95% fold-bootstrap intervals, the dashed black line and gray band show the behavioral-ceiling mean and interval, and orange dotted lines mark nominal z = +/-1.96.
2. **Historical HVE:** paginated layer profiles for every method present in the stored notebook-style table, with the same z/ceiling format. Negative effects remain negative. These figures preserve the historical exposure definitions and are not a substitute for the revised B23 analysis.
3. **Revised HVE:** paginated raw-z profiles from the revised actual-exposure fits, including global exposure order and every available definition. B23's 97-participant global-order and 168-participant order-independent samples are explicitly labeled. These are three **training-fold** z estimates from `M_predictor`. No compatible current z-ceiling is available, so no historical ceiling is inserted.

All fold data, raw-z summaries, source/formula information for revised HVE, coverage, and figure paths are in [tables](tables/figure_manifest.csv). [validation.json](validation.json) records source hashes and counts. PNG and SVG versions are supplied.

## What this does not claim

The SBI and historical HVE normalized plots are **redraws of stored notebook test-fold-refit statistics**, not newly calculated current-definition results. Their dates and scope are intentionally visible. This plotting run fits zero new GLMMs and does not re-extract features or run t-SNE. It does not certify that historical ceiling formulas and samples meet the proposed matched-scope rerun specification.

The revised HVE plots use the newer definitions but a different statistical scope. Dividing their training-fit z by a historical test-refit ceiling would be misleading. A current-definition normalized z rerun remains in TODO; its fit scope must be settled explicitly rather than silently changing genuine held-out prediction into a test-fold association fit.

The 95% intervals enumerate all 27 ordered bootstrap samples of the three fold estimates. They describe variation across three folds, not a well-powered population uncertainty estimate. Nominal +/-1.96 lines are per-coefficient references, not multiplicity-adjusted tests of the mean curve. A z/ceiling ratio is not percentage accuracy or variance explained.

## Other figure families are still required

- Manuscript [Figure 1/2 drafts](../figure_drafts/README.md): several panels are incomplete; see the panel-by-panel report.
- [Conditional S-curves](../../../reference/figures/s_curves_tr24) and [their source manifest](../../../reference/tables/s_curve_figure_manifest.csv): retain X21's multiple conditions and talkers.
- [Matched-content talker distance table](../../../reference/tables/talker_distance_tr24.csv): X21 cells average its 32 matched sentences; AN19 and B23 have separate content sets. This is not a distance-correlation matrix.
- [Combined-fold nested comparisons](../../pooled_lrt/README.md): supplementary association tests, not z layer plots.
- [Earlier September 6 likelihood figures](../README.md): separate predictive analyses, with newly identified selection/preprocessing issues explicitly flagged. They must not replace the requested SBI z presentation.

## Reproduce the review figures

From the repository root, with NumPy, pandas and Matplotlib available:

```powershell
python cross_talker_generalization/scripts/build_z_value_review.py
```

This reads the August 21 compact z tables and August 27 revised HVE coefficient/diagnostic tables. It never reads the large HDF5 feature stores or row-level OOF prediction files. Local environment names are not part of the command.
