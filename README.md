# Testing theories of cross-talker generalization in human speech perception

This repository investigates how listeners generalize speech recognition from recently experienced speech to new words, talkers, and accents. It asks whether relationships among speech exemplars, measured in acoustic or learned speech-representation spaces, predict human behavior.

The project uses latent representations from English-trained HuBERT models as inputs to exemplar-based computational analyses. It evaluates two theoretical predictors against transcription behavior from three previously published experiments:

1. **High exposure variability (HVE):** is test performance related to variability among the speech tokens presented during exposure?
2. **Similarity-based inference (SBI):** is test performance related to the similarity between speech from the exposure talker or talkers and speech from the test talker?
3. Do HVE or SBI explain behavioral variation beyond the experimental condition labels?

The analyses compare HuBERT layers and model variants with MFCC39 and STRF24 baselines. The scientific rationale and dataset designs are described in [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md); equations and implementation details are in [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).

## Current analysis status

The pipeline fits condition-only, predictor-only, and joint generalized linear mixed models (GLMMs). The present report code ranks feature spaces using participant-held-out improvement of the joint model over the condition-only model. This is a useful additional comparison, but it is **not** the planned optimization criterion for the theoretical predictors. The planned criterion is the likelihood of the predictor-only GLMM, without the original condition predictor. That criterion is not yet implemented as the report-selection rule; it is the first item in [TODO.md](TODO.md).

The public B23 archive contains the multi-talker stimulus lists needed to reconstruct actual exposure. The current pipeline has not yet imported those lists, so the released B23 HVE outputs cover only the previously reconstructed single-talker exposure pools. This is an implementation task, not a permanent data limitation; see [TODO.md](TODO.md).

## Repository map

| Path | Purpose |
|---|---|
| [`cross_talker_generalization/`](cross_talker_generalization/) | Production code, configuration, tests, intermediate artifacts, and report builder |
| [`cross_talker_generalization/final_report_2026-08-21/`](cross_talker_generalization/final_report_2026-08-21/) | Current figures, source tables, and report notes |
| [`data/`](data/) | Behavioral data, stimulus manifests, and local feature-store location |
| [`results/`](results/) | Compact retained inputs needed by the report builder |
| [`references/`](references/) | Papers and manuscript reference material |

For file-level navigation, use [FILE_GUIDE.md](FILE_GUIDE.md). Superseded versions are available through Git history and are not maintained in a separate archive directory on the main branch.

## Installation

The tested environment uses Python 3.9.18. Python dependencies are specified in [pyproject.toml](cross_talker_generalization/pyproject.toml) and [environment.yml](cross_talker_generalization/environment.yml). GLMM fitting requires R 4.4.1 and `lme4` 1.1-35.5.

```powershell
conda env create -f cross_talker_generalization\environment.yml
conda activate cross-talker-generalization
$env:PYTHONPATH = "$PWD\cross_talker_generalization\src"
python -m ctg.cli --help
```

## Verification and execution

Run the production tests from the repository root:

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

Analysis and report commands are documented in the [runbook](cross_talker_generalization/docs/RUNBOOK.md).

## Data and GitHub policy

The repository tracks code, configuration, tests, compact results, figures, and provenance. Large HDF5 feature stores and participant/item-level derived files are excluded from Git. Place externally stored feature files under `data/features/`; the required filenames and hashes are recorded in manifests and provenance files.

## Documentation

- [Project description](PROJECT_DESCRIPTION.md): scientific scope, predictors, representations, and datasets
- [Technical documentation](TECHNICAL_DOCUMENTATION.md): exact computational and statistical definitions
- [File guide](FILE_GUIDE.md): repository navigation and result authority
- [To-do list](TODO.md): remediable analysis and documentation gaps
- [Runbook](cross_talker_generalization/docs/RUNBOOK.md): executable commands
- [Validation report](cross_talker_generalization/docs/VALIDATION_REPORT.md): checks already completed
- [Current report](cross_talker_generalization/final_report_2026-08-21/README.md): figures and interpretation notes
