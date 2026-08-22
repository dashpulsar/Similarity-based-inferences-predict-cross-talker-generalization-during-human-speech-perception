# Testing theories of cross-talker generalization in human speech perception

This repository contains an investigation of how listeners generalize speech recognition from recently experienced speech to new words, new talkers, and new accents. It asks whether relationships among speech exemplars—--measured in either acoustic or learned latent speech-representation spaces—--predict human behavior.

The project combines modern DNN-based automatic speech recognition (ASR) systems trained through self-supervised learning (SSL) with theories of human speech perception. Specifically, we use the latent representations learned by different layers of the HuBERT model to approximate the latent representations of speech inputs learned by human listeners. This allows us to apply a general model of human speech perception (exemplar theory) to those representations, and to test competing hypotheses about the mechanisms that afford generalization during human speech perception. The use of modern ASR models allows us to test these hypotheses against the data from a comparatively unconstrained tasks---listeners' transcriptions of spoken words and sentences---and to go beyond qualitative hypothesis tests (whether a hypothesis can in principle explain the direction of the observed effects) towards quantitative hypothesis tests (whether a hypothesis can explain a non-trivial amount of human behavior, or perhaps even captures most of the observed behavior, pre-empting the need for less parsimonious explanations).

We address these questions for three distinct behavioral data sets previously elicited in separate perception experiments.

The production codebase is [cross_talker_generalization/](cross_talker_generalization/). Superseded implementations and reports are recoverably archived under `recycle_bin/` and are not used at runtime.

## Research questions

The analyses compare (multiple implementations of) two theories of generalization during human speech perception:

1. **Exposure variability (HVE):** does a listener perceive speech during the test phase more accurately when the exposure phase of the experiment contained speech tokens that varied substantially from each other.
2. **Similarity-based inference (SBI):** does a listener perceive speech during the test phase more accurately when exposure talker(s) and a test talker produce acoustically or representationally similar speech?

Additionally, we ask:

3. Can either HVE or SBI or both explain variability beyond that previous explained by the experimental conditions (the design factors) of the experiments that elicited the behavioral data sets we model?

We compare the answer to these questions:

* across different neural layers of the HuBERT model.
* across different variants of the HuBERT model (ASR-fine tuned vs. not)
* for acoustic (MFCCs) and perceptual (STRFs) baseline measures

## Conceptual analysis flow

```text
Speech recordings
    ├── HuBERT base / HuBERT ASR fine-tuned (18 registered layers)
    ├── MFCC39
    └── STRF24
            │
            ▼
Frame-level variable-length representations
            │
            ├── Same-content exposure–test DTW distance ──► SBI predictor
            │
            └── Dispersion within the heard exposure pool ─► HVE predictors
                                                             │
                                                             ▼
                        Participant-disjoint three-fold GLMM evaluation
                                                             │
                              ┌──────────────────────────────┴─────────────────────────────┐
                              ▼                                                            ▼
                 Full-data association statistics                         Frozen-model out-of-fold (OOF) prediction
              coefficient, CI, Wald z, LRT                         held-out binomial log-loss gain
```

The distinction at the bottom is essential. Primary predictive claims use models fit on training participants and frozen before scoring unseen participants.

## Representations and distance computation

The project analyzes HuBERT large base and an ASR fine-tuned variant. Each model contributes 18 registered feature spaces:

- CNN layers 2–6;
- Transformer layer 0;
- Even-numbered Transformer layers 2–24.

The established analysis uses supplied frame-level 3-D t-SNE sequences. Full-dimensional HuBERT representations are retained for key sensitivity analyses. MFCC39 and STRF24 provide acoustic/perceptual control comparisons.

Variable-length sequences are aligned with dynamic time warping (DTW). The main reproduction setting uses Minkowski `tau=2` and divides cumulative path cost by mean sequence length:

```text
d = minimum accumulated DTW cost / ((n_frames_left + n_frames_right) / 2)
```

Path-length normalization, alternative `tau` values, multi-talker aggregation, and predictor transformations are implemented as explicit sensitivity profiles. Raw costs, path lengths, frame counts, and normalized distances are retained for auditability.

## Two predictor families

### Similarity-based inference (SBI)

SBI compares exposure and test talkers producing matched linguistic content. The recoverable predictor is a **same-content talker proxy**: it does not guarantee that the comparison recording is the exact token heard by a participant. Multi-talker conditions use mean raw distance by default, with minimum distance available as a sensitivity analysis.

Descriptive figures may show bounded similarity `exp(-k d)`. Confirmatory models use negative DTW distance standardized with training-fold moments only, so larger predictor values indicate greater similarity without tuning `k` against held-out behavior.

### High exposure variability (HVE)

HVE describes dispersion within the exposure speech set rather than similarity between exposure and test speech. The registry contains 16 measures: `overall`, plus five measure families at sentence, word, and phoneme levels:

- within-token dispersion;
- within-type dispersion;
- between-type dispersion;
- adjacent-frame order dispersion;
- within-type mean DTW dissimilarity.

Availability is dataset-dependent. AN19 supports six measures, X21 supports all 16, and B23 currently supports 14 measures for identifiable exposure pools. B23 multi-talker actual-exposure HVE remains blocked because the repository does not contain the required sentence-to-talker assignment.

## Statistical contract

Participants---not trials---are assigned to three folds using fixed seed `230519`. Each predictor is evaluated with condition-only, predictor-only, and joint GLMMs.

The primary predictive quantity is:

```text
OOF gain = log loss(M_condition) - log loss(M_joint)
```

A positive value means that the computational predictor improves prediction for unseen participants beyond experimental condition. Full-data coefficients, confidence intervals, Wald z, and likelihood-ratio tests remain useful association summaries. Historical held-out-refit z values are preserved only in clearly labeled compatibility figures.

AN19 and X21 use Bernoulli responses. B23 retains sentence-level correct/incorrect keyword counts as a count-binomial outcome rather than collapsing each sentence to one binary trial.

## Repository layout

```text
.
├── cross_talker_generalization/   production code, configs, tests, artifacts, and report
│   ├── configs/                   dataset registry and analysis profiles
│   ├── src/ctg/                   Python analysis package
│   ├── R/                         lme4 GLMM fitting scripts
│   ├── scripts/                   automated PowerShell and report entry points
│   ├── docs/                      runbook, scientific specification, and validation records
│   ├── tests/                     unit and project-contract tests
│   ├── artifacts/                 refactored-pipeline intermediate and model products
│   └── final_report_2026-08-21/  reviewed figures, source tables, and provenance
├── data/                           tracked manifests plus local read-only feature stores
├── results/                        published summaries plus local compatibility inputs
├── references/                     prior paper and manuscript reference files
└── recycle_bin/                    recoverable archive of superseded project versions
```

See [FILE_GUIDE.md](FILE_GUIDE.md) for file-level navigation and [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) for exact estimands, equations, and implementation details.

## Installation and requirements

The tested Python environment is 3.9.18. Python dependencies are pinned in [pyproject.toml](cross_talker_generalization/pyproject.toml) and [environment.yml](cross_talker_generalization/environment.yml). GLMM fitting additionally requires R 4.4.1 and `lme4` 1.1-35.5.

Create the project environment from the versioned specification, then activate it from the repository root:

```powershell
conda env create -f cross_talker_generalization\environment.yml
conda activate cross-talker-generalization
$env:PYTHONPATH = "$PWD\cross_talker_generalization\src"
python -m ctg.cli --help
```

If the environment already exists, only the activation and `PYTHONPATH` commands are needed. The commands below assume that environment is active.

## Quick verification

Run the production tests:

```powershell
$env:PYTHONPATH = "$PWD\cross_talker_generalization\src"
python -m unittest discover `
  -s cross_talker_generalization\tests -v
```

Audit the registered behavioral data, manifests, and feature stores:

```powershell
python -m ctg.cli audit `
  --project cross_talker_generalization\configs\project.json `
  --output cross_talker_generalization\artifacts\audit-current
```

Run an X21 Tr-24 SBI analysis:

```powershell
& .\cross_talker_generalization\scripts\run_similarity.ps1 `
  -Dataset X21 -Store X21_hubert_base_tsne -Features tr_24 -Jobs 8
```

Build a new final report without overwriting the reviewed release:

```powershell
python -m ctg.cli build-report `
  --repository . `
  --output cross_talker_generalization\final_report_rebuild
```

Complete commands and output contracts are documented in the [runbook](cross_talker_generalization/docs/RUNBOOK.md).

## Results and presentation materials

The reviewed report package is [cross_talker_generalization/final_report_2026-08-21/](cross_talker_generalization/final_report_2026-08-21/). It contains:

- PNG and SVG figures;
- figure-level CSV source data;
- complete variability profiles;
- all-talker matched-content distance matrices;
- S-curves by talker and condition;
- a presentation outline;
- build verification and SHA-256 provenance.

This report package is the single authoritative entry point for current results. The
top-level [`results/`](results/) directory is deliberately narrower: it contains only
inputs still required to rebuild the report, compatibility-only notebook summaries,
validated AN19 talker-matrix sources, and one provenance-backed method schematic.

## Reproducibility policy

Production stages use explicit configuration, stable IDs, participant-disjoint saved folds, deterministic seeds, one HDF5 handle per parallel worker, tidy intermediate tables, and fail-closed validation. New runs should write to new output directories rather than overwrite audited products. Historical provenance retains the paths under which those computations were originally executed, even when the surrounding project directory is later renamed.

### GitHub large-file policy

The Git repository contains the production implementation, tests, configuration,
manifests, compact result summaries, provenance, and the reviewed final report. It does
not contain the large HDF5 feature stores, participant/item-level derived CSV files,
full local execution artifacts, or recoverable archive payloads. These files remain in
their documented local paths and are ignored rather than deleted.

To reproduce computational stages that require those inputs, place the externally stored
files under `data/features/` and the local result-source paths recorded in
[`results/README.md`](results/README.md). The tracked manifests and provenance records
provide filenames, identities, and hashes for auditing.

## Documentation index

- [Project description](PROJECT_DESCRIPTION.md)
- [File guide](FILE_GUIDE.md)
- [Technical documentation](TECHNICAL_DOCUMENTATION.md)
- [Production code README](cross_talker_generalization/README.md)
- [Runbook](cross_talker_generalization/docs/RUNBOOK.md)
- [Scientific specification](cross_talker_generalization/docs/SCIENTIFIC_SPEC.md)
- [Historical implementation audit](cross_talker_generalization/docs/LEGACY_AUDIT.md)
- [Validation report](cross_talker_generalization/docs/VALIDATION_REPORT.md)
- [Final report](cross_talker_generalization/final_report_2026-08-21/README.md)
- [Retained-result policy](results/README.md)
- [Result curation and rebuild record](results/CURATION_REPORT_2026-08-21.md)
