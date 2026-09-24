# AN19 automatic phoneme annotations

For sharing the figures alone: [three-page PDF with brief captions](../../../../output/pdf/AN19_phoneme_figures.pdf), [overview PNG](../../../../output/figures/an19_phonemes/AN19_phoneme_overview.png), and [individual PNG pages](../../../../output/figures/an19_phonemes). These use the existing numerical results unchanged and retain the automatic-annotation and coverage qualifications. The PDF source is [AN19_PHONEME_FIGURES.md](AN19_PHONEME_FIGURES.md).

For Florian's requested average across all segment instances, use the [single-page similarity PDF](../../../../output/pdf/AN19_all_segments_similarity.pdf), [PNG](../../../../output/figures/an19_phonemes/AN19_all_segments_similarity.png), or [editable SVG](../../../../output/figures/an19_phonemes/AN19_all_segments_similarity.svg). I convert each cached matched-phone distance to `exp(-DTW)` (k = 1, as in Figure 2a), average distinct reference recordings within each English speaker, and then average the six English speakers for each L2 segment instance. I average **all eligible instances equally within each L2 talker**, without intermediate word-type or phone-type means. The 5,114 instances cover 36 L2 talkers, with 136-150 instances per talker; the existing six-reference inclusion rule is unchanged. Dots show individual talkers and open diamonds show equal-talker L1-group means, sorted from higher to lower similarity. For a single-talker group, these coincide. Horizontal bars now show **95% percentile confidence intervals for the L1-group mean**, from 1,000 bootstrap samples of L2 talker means within each L1 (seed 20260909). Reference speakers and within-talker instance means remain fixed. The Korean and Spanish groups each contain 12 talkers; the other 12 language groups each contain one talker, so no group CI is estimated for them. These intervals describe uncertainty in group means, not the range of individual talkers or uncertainty in automatic phone boundaries. Following Florian's latest feedback, I removed the in-figure legend, methods subtitle and footer; the x-axis reads only "Mean segment similarity to L1-English". SBI fold intervals and the earlier per-phone panels are unchanged. Similarity is not a probability of correct perception, and coverage varies across talkers.

Regenerate with `python cross_talker_generalization/scripts/build_an19_all_segment_panel.py`. The script reads existing pair distances only, verifies their reference matching and hashes, and exports instance, talker and language summaries under `tables/figure2b_all_segments_similarity_*`. No alignment, HuBERT, t-SNE, DTW or GLMM is rerun. The previous `AN19_all_segments.*` files and `figure2b_all_segments_*` tables without the `similarity` infix are retained **as superseded equal-type distance summaries**, not the requested instance-weighted result.

The production run reads audio directly from the user-selected **`July/Nygaard_audio`** directory. It never falls back to audio under the repository's `data` folder. The original recording manifest supplies stable recording IDs and intended-word metadata only; the existing HDF5 files supply HuBERT features for the later distance calculation.

## What is being estimated

| Output | Input to the model | Meaning |
|---|---|---|
| FALCON phone intervals | Recording and the target word's canonical pronunciation | Estimated timing of **intended** phones; not evidence that the speaker produced every canonical phone correctly |
| [PhoneticXEUS IPA hypotheses](../an19_phonetic_xeus/README.md) | Recording only; no target word | Independently recognized pronunciation; still a machine hypothesis, not a human transcription |
| Figure 2b phone deviations | FALCON intervals and existing Tr-24 3-D t-SNE trajectories | Same-word, same-intended-phone distance from English reference talkers |

The full corpus is processed without a pilot subset or a manual-review prerequisite. Automatic coverage, hash and timestamp checks do not establish phonetic accuracy. In particular, FALCON's successful alignment of an expected pronunciation cannot establish the absence of a substitution or deletion. The two model outputs are retained separately.

## Authoritative source and naming audit

All **6,261 recordings from 42 talkers** were found and decoded in the selected directory: approximately 57.24 minutes of speech including surrounding silence. All 6,261 audio files are byte-identical to their corresponding previously resolved repository recordings. Thus the audit found no audio-content difference between these two copies; this is not a blanket certification of every repository file.

Most files match by complete filename, ignoring case. Two Somalian-talker files have different item numbers across the directories:

| Canonical manifest filename | Actual file in `Nygaard_audio` | Match |
|---|---|---|
| `IF6ew15 gave.wav` | `International accents/Int F6 Somalian/IF6ew16 gave.wav` | Identical SHA-256 |
| `IF6ew16 death.wav` | `International accents/Int F6 Somalian/IF6ew15 death.wav` | Identical SHA-256 |

These are explicit recording-location mappings, not guessed word substitutions. Neither set of files was renamed. The [source manifest](tables/nygaard_audio_source_manifest.csv) preserves both filenames and their hashes; the [source audit](nygaard_audio_source_audit.json) records all counts.

Other source details remain visible:

- `ef2ew32 cheif.WAV` retains its original filename and `word` label. A documented, recording-specific override supplies **chief** as `annotation_word`, supported by the other 41 EW32 recordings and behavioral target labels. It does not certify the spoken pronunciation.
- English HW74 targets are labeled **wade**, whereas L2 HW74 targets are labeled **wave**. Both sets receive their own target-conditioned annotations, but HW74 is excluded from same-word Figure 2b comparisons until the lexical conflict is resolved.
- `sf4ew52 vote.wav` has an overlong WAV header in this directory too. The model uses the 14,916 samples actually decoded at 22,050 Hz, not the 18,910 claimed by the header.
- Two English **wrong** filenames contain identical audio. Both registered recording IDs remain present. Speaker-plus-item or speaker-plus-word alone is not a unique recording identifier.

The earlier [input audit](INPUT_AUDIT.md) describes the preliminary repository-source investigation. The selected-source audit above supersedes its audio-location instructions. Matching audio hashes also permit reuse of the existing HuBERT time axes; no features or t-SNE coordinates are regenerated.

## Files to use

The selected-source batch is complete: **6,261/6,261 recordings aligned**, with **18,785 non-silence intended-phone intervals** (31,307 intervals including silence), and no alignment failures. All recordings were inferred afresh from `Nygaard_audio`; none was reused from the preliminary run. Independent PhoneticXEUS recognition processed the same 6,261 recordings, returning **5,872 nonempty IPA hypotheses and 389 empty hypotheses**. The empty outputs remain explicitly recorded.

Use the **`nygaard_audio/`** subdirectory for current alignment outputs:

- `alignment_summary.json`: completion counts, source/configuration hashes, checkpoint, software and device information.
- `tables/phone_intervals.csv`: canonical phone, reduced model phone class, start/end time, recording ID and descriptive quality flags.
- `tables/recording_alignment_status.csv`: one row per recording, including original and annotation words, actual external filename, audio hash and success status.
- `tables/failed_recordings.csv`: every failed recording; never silently removed from the denominator.
- `recordings/*.json`: individual results, full target sequences and diagnostics.
- `textgrids/*.TextGrid`: Praat-compatible automatic target-phone, model-class and target-word tiers.

Results directly under this parent folder's `recordings/`, `textgrids/` and alignment tables were produced during the preliminary repository-source pass. They are preserved for traceability, **not** used as the production annotation source. Figure 2b's builder explicitly requires the completed `nygaard_audio/` run and its selected-source hashes.

## Alignment method

I use [FALCON](https://github.com/MLSpeech/FALCON), a publicly released 2026 phoneme aligner, with its TIMIT-English checkpoint because the recordings contain English speech. This is a recent model choice, not a claim that it is universally best or already validated on AN19. The [paper](https://arxiv.org/abs/2606.25460) evaluates phonetic alignment; it does not establish the accuracy of these AN19 annotations.

The target pronunciation is the first variant in the pinned CMU pronunciation dictionary, fixed without selecting variants based on human outcomes. Stress-marked ARPAbet, stressless canonical targets and FALCON's reduced LH39 targets are stored separately. Initial and final silence targets are included.

Each complete decoded waveform is averaged to mono and resampled to 16 kHz using `torchaudio.functional.resample`. There is no VAD, trimming or padded multi-utterance inference. The official bidirectional model and decoder run on each entire recording. Boundary conversion follows the pinned implementation, approximately 10.084 ms per model frame; it is not rounded to a nominal 10 ms grid.

Shorter-than-20-ms intervals and mean target posterior below 0.2 are flags, not automatic proof of failure or calibrated error probabilities. Nonfinite, reversed, zero-length or out-of-audio boundaries are explicit failures, not silently repaired.

## Reproduce

Use the project scientific Python environment with matching PyTorch/torchaudio. The recorded run uses Python 3.9, PyTorch 2.4.1 and torchaudio 2.4.1. FALCON and CMUdict are pinned at:

- FALCON code: `3d6e3d876ebaf2472757720d92222d55dd7410f8` (MIT).
- FALCON weights: `MLSpeech/FALCON-weights`, revision `b94a36298af110fd78a6083ca4108f2a288125f7`, `falcon_timit_english.pt`.
- Checkpoint SHA-256: `a8f223c6ebdc56506a358ce74748b588c983ae68a53da45d9bd1ce7c1f3af016`.
- CMUdict revision: `74790861f652b15e4ac49015a90074ad62a27690`.

Place the pinned repositories in `cross_talker_generalization/artifacts/vendor/FALCON` and `.../cmudict`. The runner verifies their revisions. Legacy FALCON dependencies are isolated under `artifacts/phone_alignment_runtime`: `hydra-core==0.11.3`, `omegaconf==1.4.1`, `memory_profiler==0.61.0` and `boltons==20.0.0`; the existing scientific environment is not upgraded. Model downloads go to the standard Hugging Face cache. These dependencies and weights are local execution assets, not included with the source repository.

From the repository root, with your own path to the selected audio folder:

```powershell
python cross_talker_generalization/scripts/run_an19_falcon_alignment.py --audio-root "path/to/July/Nygaard_audio" --workers 8 --cpu-threads 2 --device cuda
python cross_talker_generalization/scripts/build_an19_automatic_phone_panel.py --jobs 8
```

The accompanying relative-path source manifest is required. The source-audit script can recheck this local directory against the retained preliminary audio hashes; it never rewrites the canonical stimulus manifest. Seeds and deterministic CUDA settings are recorded, but exact floating-point identity across arbitrary hardware/software versions is not promised.

## What remains separate

The [Figure 2b overview](figures/figure2b_automatic_intended_phone_deviation.png) and three companion pages are now generated. All-phone estimates and reference counts are in `tables/figure2b_*`. The 67,084 matched distances yield 5,114 L2 intended-phone intervals with all six usable English references, from 16,086 L2 intervals. Single-talker language groups have no group confidence interval. The preselected DH panel is explicitly empty because the corpus target lexicon contains no intended DH tokens.

Coverage matters: 3,167 of all 18,785 speech-phone intervals contain no HuBERT frame center. Some talker/phone means are based on only one or two word types. Requiring each target and reference interval to be at least 20 ms with mean target posterior at least 0.2 leaves only 197 six-reference-complete intervals. That score is not calibrated accuracy, but this attrition prevents claiming quality-robust phonetic effects from the current candidate. See the [full automatic checks](nygaard_audio/QC.md); no missing interval is interpolated or replaced with an invented one-frame segment.

Figure 2d concerns **listeners' segment-level errors**, not a recognizer's errors. It additionally requires alignment of target and listener-response pronunciations, a defined lexical-neighbor calculation, and the associated behavioral model. Producing automatic acoustic phone labels does not by itself complete that panel.
