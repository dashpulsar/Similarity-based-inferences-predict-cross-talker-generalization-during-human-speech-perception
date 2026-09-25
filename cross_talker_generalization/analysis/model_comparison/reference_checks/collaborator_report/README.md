# Collaborator report and completed validation panels

This directory retains scientific figures, source checks and numerical tables.

## New outputs

- Figure 1a-b: actual HuBERT-large Tr-24 waveform/full-feature/t-SNE example and verified sentence, word and phone intervals.
- Figure 1c: corpus language coverage, with original AN19 language labels checked against the study paper.
- Figure 1d: a computed PHOIBLE 2.0 segment-inventory-overlap **proposal**, not the full phonotactic analysis.
- Figure 2a: the requested 36-L2-talker submatrix, using the existing 138-word matched comparison.
- Figure 2c: newly computed word-to-English-reference DTW, control-condition binned accuracy, logistic curves and participant-bootstrap intervals for AN19 and X21.

Figure 2b and 2d are **not complete**. Their remaining source requirements are documented in the source review rather than represented by fabricated results or placeholder charts. [SOURCE_REVIEW.md](SOURCE_REVIEW.md) records what was found in the supplied notebooks and local papers.

## Reproduce

Use the project's Python scientific environment, with `statsmodels` available. The original HDF5 inputs are read-only. This script reads the large stores but does not rerun extraction or t-SNE:

```powershell
python cross_talker_generalization/scripts/build_collaborator_panels.py --jobs 8 --bootstrap 1000
```

The verified example WAV is retained under `sources/`; if absent, hydrate the original Git LFS audio or supply `--reference-audio PATH_TO_REAL_WAV`. The script checks the original LFS SHA-256 before accepting a fallback file. PHOIBLE v2.0 downloads once into `artifacts/external/`; the selected rows and derived inventory comparisons are exported with attribution under `tables/`.

`provenance.json` records source hashes, the full-feature example leaf, parameters, and generated scientific outputs. Source tables record 40 provisionally excluded AN19 answers: HW74 is labeled "wade" in English but "wave" in L2 recordings, so a valid same-word reference is not yet established. Audio and features exist; this is not established recording absence. All X21 control answers are mapped. The curve fit is a descriptive binomial GLM, not a new confirmatory GLMM. Participant bootstrap intervals condition on the observed items/talkers and are not the three-fold intervals used in the SBI z display.

The read-only naming audit is `scripts/audit_an19_hw74.py`. Optional `--asr` uses an already cached local Whisper large-v3-turbo model without a lexical prompt; this is a screening aid, not ground-truth transcription. No automatic wave/wade alias is applied. AN19's lack of existing phone labels has been confirmed by Zhengyang; 2b/d are deferred pending a separate annotation decision.
