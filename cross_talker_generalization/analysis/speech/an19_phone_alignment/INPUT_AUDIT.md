# AN19 audio inputs for automatic phone alignment

All **6,261 registered recordings are available locally**. The main worktree contains Git LFS pointers, but the corresponding real audio is present in the shared Git LFS object store. I checked every audio SHA256 against its pointer and confirmed that the full-dimensional and 3-D Tr-24 stores contain the corresponding recording.

The scope is 42 talkers and 156 word labels: 899 English recordings, 1,800 Korean-accented recordings, 1,800 Spanish-accented recordings, and 1,762 recordings from the other L1 groups. Only 1,056 recordings are marked as behavior-used; the proposed full alignment covers the entire corpus.

## Use the recording manifest, not speaker-word keys

The input list is `data/manifests/AN19-stimulus-manifest.csv`. Use `recording_id` as the identity and `source_wav_relpath` to locate its audio. Sixty rows have collisions in the older speaker-word key system.

To resolve a pointer without modifying the working tree:

1. Read its `oid sha256:` value.
2. Locate the shared Git directory with `git rev-parse --path-format=absolute --git-common-dir`.
3. Read `lfs/objects/<first two hash characters>/<next two>/<complete hash>` inside that directory, and verify its SHA256.

The current canonical manifest hash and the audio-collection hash both match the values stored when the full HuBERT features were created. The collection hash includes each relative recording path and the SHA256 of its actual audio bytes; it is not a hash of pointer files.

## Two source details to preserve

- `ef3ew31 wrong.WAV` and `ef3hw34 wrong.WAV` are byte-identical. There are consequently 6,260 distinct audio hashes but 6,261 registered recording identities. The same automatic alignment can be reused for this pair when the transcript is also identical, while retaining both manifest rows and their analysis identities.
- `sf4ew52 vote.wav` overstates its data length in the WAV header: 18,910 frames are declared but 14,916 frames are decodable at 22,050 Hz. Use the actual decoded samples, without padding to the declared duration. The existing HuBERT frame count agrees with the decoded length.

All recordings are mono. Most are sampled at 22,050 Hz; `IF2hw69 wail.wav` and `im3hw68 route.wav` are sampled at 44,100 Hz. The decodable corpus duration is 3,434.189 seconds, approximately 57.24 minutes. The older HDF5 header-based duration is 0.181 seconds longer because of the `vote` recording.

The English `wade` versus L2 `wave` naming issue is not an audio-availability problem. An aligner given either word will attempt to fit that transcript; success alone would not establish the actual pronunciation.

## Preserve the original time axis

The archived extraction script and helper match the script hashes recorded in the full HDF5 file. They load each entire recording, resample to 16 kHz, normalize through the model's feature extractor, and run HuBERT without VAD or silence trimming.

The full features are at `<speaker_id>/<recording_id>/tr_24`; the reduced features are at `tr_24/<speaker_id>/<recording_id>`. Every recording has matching full/reduced frame counts and matching stored source-feature hash attributes. This audit did not regenerate HuBERT features or rehash all feature-array bytes.

For Tr-24, the convolutional front end has a 400-sample receptive field and a 320-sample stride at 16 kHz. Frame centers occur approximately at **12.5 ms + 20 ms × frame index**. All 6,261 observed sequence lengths agree with this geometry and the actual decoded audio length.

Phone boundaries should therefore be mapped to those frame centers on the original, untrimmed recording time axis. Intervals shorter than the temporal resolution can contain no frame centers; flag them rather than silently forcing a one-frame crop. Cropping the existing Tr-24/t-SNE trajectories does not require a new feature extraction or t-SNE fit.

These checks establish that the inputs are usable. They do not establish phone labels or automatic alignment accuracy. The resulting annotations must remain identified as automatic until checked independently.

Exact counts, hashes, exceptions, and verification scope are recorded in [input_audio_audit.json](tables/input_audio_audit.json).

## Annotation transcript correction: `cheif`

One filename contains `cheif`: `ef2ew32 cheif.WAV` (`AN19-rec-8e01733e25fc77ea52f1`). The other 41 talkers' recordings at the same easy-word slot, EW32, all say `chief` in their filenames, including the other five English talkers. The behavioral table likewise contains 240 exposure responses with the intended target `chief` and none with `cheif`.

This supports using **`chief` as the intended annotation transcript for this one recording**, while retaining `cheif` as its original source label. No audio, filename, manifest label, or recording ID is changed. The correction resolves a spelling transposition, not a verified phonetic transcription of the recording.
