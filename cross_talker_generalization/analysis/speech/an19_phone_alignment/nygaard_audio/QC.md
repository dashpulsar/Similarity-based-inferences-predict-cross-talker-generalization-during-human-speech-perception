# AN19 automatic alignment: full-batch QC

The selected `July/Nygaard_audio` run covers **all 6,261 registered recording identities**: 42 talkers, including six English reference talkers. No recording failed. I checked the completed outputs without running another alignment or modifying the source recordings.

## Coverage and source identity

- 6,261 recording JSON files and 6,261 three-tier TextGrids match the canonical manifest exactly.
- The output contains 31,307 intervals: 18,785 speech-phone intervals and 12,522 imposed initial/final silence intervals.
- Recording counts are English 899, Korean 1,800, Spanish 1,800, and other L1 backgrounds 1,762.
- Every selected external audio file was rehashed after the run and matches the selected-source manifest. The runner requires the selected audio root, checks that each resolved path stays inside it, and has no repository-audio fallback.
- Every recording remains `human_verified=false`. No human verification is implied by complete coverage.
- Original manifest words and recording IDs are unchanged. The only annotation-transcript correction is `cheif` to `chief` for `AN19-rec-8e01733e25fc77ea52f1`; the original label is retained.
- All 42 `hw74` recordings retain the unresolved lexical-identity flag for the English `wade` versus L2 `wave` issue.

## Interval checks

Every interval has finite, positive duration and lies within the actual decoded audio duration. Each recording is partitioned exactly from zero to its decoded duration, without gaps or overlaps. JSON and CSV agree; TextGrid boundaries agree within the written nine-decimal precision (0.5 ns). In particular, the overstated header of `sf4ew52 vote.wav` is not used to extend or pad its audio.

There are 37 distinct canonical speech-phone labels. Canonical targets and FALCON's model labels remain separate: the observed merges are `aa -> ao`, `l -> el`, and `n -> en`. The canonical labels have not been replaced by merged model labels.

## Descriptive quality flags

These flags are frequent and must remain visible when interpreting the resulting phone analyses. They are **not calibrated error rates**, and I have not used them as an acceptance gate or to discard recordings.

| Check | Count |
|---|---:|
| Speech intervals shorter than 20 ms | 6,879 / 18,785 (36.6%) |
| Speech intervals with mean target posterior below 0.20 | 11,280 / 18,785 (60.0%) |
| Speech intervals meeting both flags | 3,098 |
| Speech intervals with no sampled posterior frame | 162 |
| Mean-posterior argmax differs from supplied model target, where available | 11,875 |
| Recordings containing a short speech interval | 4,849 |
| Recordings containing a low-posterior speech interval | 5,982 |

The raw `speech_argmax_differs_target` count in the JSON is 12,037 because it also counts the 162 null argmax values as unequal; the table above separates missing values from actual argmax disagreements. All 162 null posterior intervals are speech intervals.

The corresponding counts by source accent are:

| Source group | Speech intervals | Shorter than 20 ms | Mean target posterior below 0.20 |
|---|---:|---:|---:|
| English | 2,699 | 942 | 1,639 |
| Korean | 5,400 | 1,977 | 3,233 |
| Spanish | 5,400 | 2,098 | 3,290 |
| Other L1 backgrounds | 5,286 | 1,862 | 3,118 |

## Consequences for existing HuBERT features

Using the audited HuBERT frame centers, `(199.5 + 320 * frame_index) / 16000`, and actual decoded/resampled lengths:

- **3,167 speech intervals contain no HuBERT frame center**, across 2,912 recordings.
- A further 5,413 intervals contain exactly one center.
- A zero-center interval must remain missing in a frame-based phone calculation. It should not be assigned an artificial frame by stretching the time axis or forcing a one-frame crop.

These counts use the previously established convolution timing, not a new feature extraction. One frame also cannot describe a within-phone trajectory; downstream measures need to state their own minimum-frame requirements.

## What this check establishes

The pinned official FALCON implementation and decoder are unchanged. The run uses the first CMU dictionary pronunciation, preserves its original stressed form, and aligns stress-stripped canonical targets plus initial/final silence. The official timestamp conversion is retained; it is not an exact 10 ms time grid.

This establishes complete, internally consistent **automatic target-conditioned annotations** with identified source audio. It does not establish that each target phone was actually produced, that the phone boundaries are accurate, or that low model scores represent human perception errors. The supplied word constrains forced alignment, so the `wave/wade` distinction remains unresolved by alignment alone.

The audio checks establish unchanged bytes relative to the pre-run source audit. The alignment runner and this QC perform no HuBERT extraction or HDF5 writes; full pre/post HDF5 file checksums were not recomputed. Exact source/table hashes, all canonical counts, and the 162 null-posterior interval identities are in [qc_summary.json](qc_summary.json).
