# File guide

Start with [the analysis index](cross_talker_generalization/analysis/README.md) for figures and results. Scientific rationale belongs in [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md), methods in [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md), commands in the [runbook](cross_talker_generalization/docs/RUNBOOK.md), and outstanding work in [TODO.md](TODO.md).

## Results by topic

All retained analysis batches have been consolidated under `cross_talker_generalization/analysis/`.

| Directory | Contents |
| --- | --- |
| `sbi/` | Parameter maps, optimizer checks, and SBI interpretation |
| `diagnostics/` | Training/test scores, six SBI diagnostic figures, and validation |
| `acoustic_baselines/` | AN19 explanation, component analyses, and model-structure checks |
| `speech/` | Representation figures, phoneme annotations, source tables, and report |
| `model_comparison/` | Predictor selection, pooled GLMM tests, and behavioral references |
| `hve/` | Existing variability results; further work remains paused |
| `presentation/` | Figure selections and figure assembly metadata |
| `reference/` | Historical tables and figures required by existing builders; supporting material only |

The [analysis index](cross_talker_generalization/analysis/README.md) links directly to the relevant results. New work should update these topic directories, not create another dated analysis-update folder.

## Code and inputs

| Path | Purpose |
| --- | --- |
| `cross_talker_generalization/src/ctg/` | Python analysis package |
| `cross_talker_generalization/R/` | GLMM fitting and statistical checks |
| `cross_talker_generalization/scripts/` | Analysis and figure-generation commands |
| `cross_talker_generalization/configs/` | Dataset paths, model profiles, and annotation settings |
| `cross_talker_generalization/tests/` | Implementation checks |
| `cross_talker_generalization/docs/` | Technical specifications, progress, and audit records |
| `cross_talker_generalization/artifacts/` | Pipeline intermediates and model exports |
| `data/` | Behavioral data, manifests, exposure records, and local HDF5 feature stores |
| `results/` | Existing compatibility inputs still used by report builders |
| `output/`, `outputs/` | Standalone PDFs, figures, presentations, and interactive HTML packages |
| `references/` | Papers and manuscript reference material |

Run commands from the repository root. JSON configuration paths retain their documented resolution rules. Large feature stores, worker caches, and participant-level intermediate files remain local.

## Original batches and recovery

The original dated directories are retained as scientific archives under `recycle_bin/analysis_history/`. This local archive is excluded from Git and is not an input directory for the active analysis scripts. The published pre-cleanup snapshot is commit `42cfa42618b404dcd041f8e0d9e3d8cccb15393b`.

All 1,407 tracked files in those batches were copied and checked by SHA-256 before the originals were moved. Active path references were then updated. Historical provenance retains the original execution paths and hashes; it should not be interpreted as a new analysis run.

After migration, all 475 retained figure assets remain byte-identical. Across 718 result CSVs, changes are confined to path fields; no other cell values changed. The 59 Python checks and 119 entry-page links passed, and the relocated diagnostic input/score checks passed without fitting models.

Private communications, meeting transcripts and personal bilingual notes are not part of the public project or its local analysis archive.
