# Analyses and results

Results are organized by research question. Add new results to the relevant topic directory rather than creating a dated update folder. The code, model settings, and numerical results are unchanged by this reorganization.

| Topic | Start here |
| --- | --- |
| SBI parameter maps | [All X21 conditions](sbi/parameter_maps/x21_all_conditions/README.md), [without Talker-specific](sbi/parameter_maps/x21_without_talker_specific/README.md) |
| Training/test diagnostics | [Six SBI figures and their interpretation](diagnostics/sbi_review/README.md) |
| AN19 acoustic baselines | [Explanation and evidence](acoustic_baselines/AN19_BASELINE_EXPLANATION.md) |
| Speech representations and phonemes | [Figures, annotation methods, and report](speech/README.md) |
| Model comparisons | [Predictor selection](model_comparison/selection/README.md), [pooled GLMM comparisons](model_comparison/pooled_lrt/README.md) |
| Presentation | [Concise figure package and speaking notes](presentation/CONCISE_EDITION.md) |

## Supporting material

- `sbi/`: optimizer audit, feasibility checks, and parameter maps. The X21 maps cover one layer and one training split; they do not complete cross-fold optimization.
- `diagnostics/`: matched training/test scores, model exports, and source validation.
- `acoustic_baselines/`: component and common-structure checks. The explanation remains provisional.
- `hve/`: retained global-order and train/test results; further HVE work remains paused.
- `model_comparison/`: selection, pooled tests, behavioral references, and earlier fixed-layer checks.
- `speech/`: shared figure sources, automatic phone annotations, and report tables.
- `presentation/`: figure selections and bilingual notes.
- `reference/`: historical figures and tables still required by report builders. These are supporting sources, not a replacement for the current analyses.

The original dated batches are preserved locally in `recycle_bin/analysis_history/`. They are excluded from Git; the complete published pre-cleanup snapshot is commit `42cfa42618b404dcd041f8e0d9e3d8cccb15393b`. Historical provenance records retain their original execution paths and hashes. Current commands and document links use the topic-based layout.

This is an organizational change, not a new experiment. Dated observations inside retained scientific notes still refer to the runs described in those notes.
