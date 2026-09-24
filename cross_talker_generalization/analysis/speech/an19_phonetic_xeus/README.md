# AN19 unconstrained IPA recognition

This analysis runs [PhoneticXEUS](https://huggingface.co/changelinglab/PhoneticXeus) on all **6,261 recordings** in the AN19 manifest. Unlike the accompanying FALCON forced alignment, the recognizer receives **only the waveform**, never the intended word or a canonical phoneme sequence. Its outputs are automatic hypotheses about the pronunciation, not human phonetic labels.

The model is a released 2026 multilingual phone recognizer ([paper](https://arxiv.org/abs/2603.29042)). This is a current public model choice, not a claim that it is universally best or validated on AN19. The pinned model card declares Apache-2.0 licensing.

The final UTF-8 run in `nygaard_audio/` processed all **6,261 recordings** successfully: **5,872 nonempty hypotheses and 389 empty hypotheses**. Every model logit was finite, and every emitted token matches the pinned UTF-8 vocabulary. The full inference loop took 260.79 seconds on the local GPU; this excludes setup, model loading, and subsequent audit exports. These are execution/coverage checks, not a phonetic accuracy estimate.

## Run

Use a separate Python 3.11 environment with compatible PyTorch and torchaudio, Transformers 4.45.2, huggingface-hub 0.25.2, tokenizers 0.20.3, safetensors, NumPy, soundfile 0.12.1, PyYAML, and typeguard 2.13.3. Do not upgrade an existing analysis environment. A short environment/model-cache path avoids Windows path-length failures.

From the repository root, download and verify the fixed public model with:

```powershell
python -X utf8 cross_talker_generalization/scripts/run_an19_phonetic_xeus.py --download --prepare-only --output-dir cross_talker_generalization/analysis/speech/an19_phonetic_xeus/setup_checks
```

This command downloads and verifies the fixed snapshot and audits repository input availability; it does not load the model or use repository audio for inference. Replacing `--prepare-only` with `--check-load-only` additionally loads the model on CPU without audio inference. Setup records go into `setup_checks/`, leaving completed runs untouched. These are setup checks, not a pilot subset or manual review gate. The full-corpus run uses the external source below. `--device cpu` is available explicitly.

Full inference requires both external-source arguments; omitting them now raises an error rather than using repository audio. This CLI-only guard was added after the completed run, without changing inference functions. Because the resume key includes the script hash, the current script **cannot resume the saved `nygaard_audio/` run**. Use the fresh reproduction directory below. The exact executed source is retained in `nygaard_audio/executed_runner_source.py.txt` and matches the recorded source hash. It is a provenance snapshot, not a standalone entry point: its repository lookup assumes the original `scripts/` location, so do not execute it from this nested output folder.

To reproduce the full analysis with the current guarded script, supply the requested AN19 audio root and audited mapping together, writing to a new directory:

```powershell
python -X utf8 cross_talker_generalization/scripts/run_an19_phonetic_xeus.py `
  --audio-root $An19AudioRoot `
  --source-manifest cross_talker_generalization/analysis/speech/an19_phone_alignment/tables/nygaard_audio_source_manifest.csv `
  --output-dir cross_talker_generalization/analysis/speech/an19_phonetic_xeus/nygaard_audio_reproduction
```

Here `$An19AudioRoot` is the local `July/Nygaard_audio` directory. This mode requires all 6,261 recording IDs, checks each file's SHA-256, rejects paths outside that directory, and **never falls back** to repository audio. Repeating this reproduction command with the same script, inputs, and configuration can resume its own matching outputs in `nygaard_audio_reproduction/`; it does not resume or overwrite the completed `nygaard_audio/` results. The source-manifest hash is part of the resume key.

The earlier repository-source pass was interrupted after 144 successful recordings when the source preference changed; those files remain in this directory for traceability and are not the final full-corpus output. The completed replacement-source outputs are kept separately under `nygaard_audio/`.

The source audit found that all 6,261 selected external files are byte-identical to the corresponding repository audio. Two filenames differ; their unique matching hashes determine the mapping. The full run still reads the requested external directory directly and starts afresh. This check finds no byte-level discrepancy between these copies; it does not establish phonetic annotation accuracy.

The first external pass exposed an upstream Windows encoding issue: the released loader reads the IPA vocabulary without specifying UTF-8. Under a CP936 locale, numeric token IDs and model scores remain intact but some IPA text becomes corrupted. That pass is preserved under `nygaard_audio_cp936_decoding/` and **must not be used as an IPA transcription table**. The final full-corpus pass under `nygaard_audio/` runs in UTF-8 mode and checks every loaded vocabulary label; the original 144 interrupted repository-source outputs also predate this correction.

## Reproducibility and code review

- Model revision: `8d83dee94817a07dc150f87d08f7e0ee01bdb66d`.
- Float32 safetensors: 2,300,089,432 bytes; SHA-256 `ad58bf20a60e9d0380327bd8b2d0e8e90a9b8de2adccbfb479f9b21ea85eda18`.
- The runner verifies a fixed hash over all 39 bundled source/configuration/model-card files **before** permitting the model's custom Python code to run.
- `python -X utf8` is required on non-UTF-8 Windows locales. The runner fails explicitly if text decoding is not UTF-8 and verifies the loaded vocabulary against the pinned UTF-8 JSON. No model weight or bundled source file is modified.
- The reviewed loading route supplies an explicit local folder and uses safetensors. Optional upstream network-download and pickle-loading routes are not used; Hugging Face offline mode is enabled before loading.
- Git-LFS pointer WAVs resolve to their existing local objects and are checked against their declared SHA-256. No original recording or pointer is modified.
- Audio is averaged to mono, resampled to 16 kHz with torchaudio, and passed one utterance at a time, without trimming or lexical prompting. The model's own frontend performs its configured normalization.
- Greedy CTC decoding follows the released implementation. Float32 inference fixes seeds, disables TF32, and requests deterministic algorithms. Exact numerical agreement across different hardware/library versions is not guaranteed; versions are recorded.

## Outputs and interpretation

`ipa_hypotheses.csv` contains one row per recording, including intended-word **metadata**, unconstrained IPA, output status, and audio hashes. `recordings/*.json` additionally retains the token sequence, emitted CTC frame runs, and raw posterior diagnostics. `input_audit.json`, `model_integrity.json`, `run_provenance.json`, and `summary.json` document coverage and reproducibility.

The mean token maximum posterior is an **uncalibrated model score**, not the probability that the phonetic transcription is correct. CTC emission frames are **not phonetic segment boundaries**. FALCON and PhoneticXEUS answer different questions: FALCON places boundaries for a supplied target sequence; PhoneticXEUS supplies an independently decoded phone-sequence hypothesis. Neither automatically supplies human perceptual-error labels or resolves the existing *wave/wade* filename conflict.

No recognition result is claimed until `summary.json` records completed outputs. Setup and CPU loading alone are not an analysis run.

For the saved `nygaard_audio/` run, `python -X utf8 cross_talker_generalization/analysis/speech/an19_phonetic_xeus/export_audit.py` produces the complete quality table, the HW74/*cheif*/two renamed-file case table, and the list of empty hypotheses. This exporter targets that saved run; it does not automatically switch to `nygaard_audio_reproduction/`. Decoded sample counts and duration are checked against the authoritative audio-source audit. Every final inference also checks all logits for finite values. Empty hypotheses are retained as missing recognition outputs, not filled with the intended word or canonical phones.
