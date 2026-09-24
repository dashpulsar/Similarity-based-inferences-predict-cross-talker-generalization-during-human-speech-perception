# Figure-input audit: frame timing and AN19 word coverage

Checked on September 17, 2026. This read-only audit addresses the numerical questions in meeting tasks 9 and 13. No features, distances, figures, or annotations were regenerated.

## 1. Tr-24 frame timing

**The HuBERT transformer output advances by 20 ms per frame.** Its convolutional front end has a 25 ms receptive field. Tr-24 representations also contain contextual information through the transformer; the 25 ms value describes the convolutional front end, not the complete contextual extent of a Tr-24 vector.

The Figure 1 example uses `facebook/hubert-large-ll60k`, revision `ff022d095678a2995f3c49bab18a96a9e553f782`. Its locally cached [configuration](C:/Users/Alex/.cache/huggingface/hub/models--facebook--hubert-large-ll60k/snapshots/ff022d095678a2995f3c49bab18a96a9e553f782/config.json) gives convolution strides `[5, 2, 2, 2, 2, 2, 2]` and kernels `[10, 3, 3, 3, 3, 2, 2]`; its [preprocessor configuration](C:/Users/Alex/.cache/huggingface/hub/models--facebook--hubert-large-ll60k/snapshots/ff022d095678a2995f3c49bab18a96a9e553f782/preprocessor_config.json) specifies 16,000 Hz. The current full-dimensional HDF5 attributes independently record the same checkpoint, revision, and sample rate.

Starting with stride `J = 1` and receptive field `R = 1`, each convolution gives `R_new = R + (kernel - 1) * J` and `J_new = J * stride`. The final values are 320 samples and 400 samples: 20 ms and 25 ms at 16 kHz. The nominal frame centers, relative to the extracted audio segment, are `(199.5 + 320 * frame_index) / 16000` seconds.

The [Figure 1 source](../../scripts/build_september9_method_panel.py) and its [stored metadata](../../analysis_update_2026-09-09/tables/figure_1ab_method_metadata.json) identify sentence `X21.ENG.ENG_M_055.HT1_S002`, “The wife helped her husband.” Direct HDF5 inspection confirms:

- Full `tr_24` array: 66 frames × 1,024 dimensions.
- Corresponding t-SNE array: 66 frames × 3 dimensions.
- Manifest sentence duration: 1.340 seconds.
- Selected full/reduced arrays exactly reproduce the metadata's SHA-256 values after conversion to float64.

### Existing schematic approximation

The plotting code currently sets `time = arange(T) * duration / T`. For this example, that gives **20.30303 ms** between displayed samples, with the first displayed point at zero. Interval slicing uses `round(T * boundary / duration)`. These are duration-based approximations rather than the nominal 20 ms frame-center grid. The x-axis itself is labeled “Time (s)”; the current source does not label frames as 24 ms or 4 ms.

For the displayed word *wife*, both the existing slicing rule and the nominal center-in-interval rule select the same frames: `/w/` `[4, 11)` (7 frames), `/aɪ/` `[11, 17)` (6), and `/f/` `[17, 20)` (3), using zero-based indices. Thus this audit identifies a time-coordinate refinement without establishing an annotation change for this example.

**Remaining figure work:** when revising the schematic, use the nominal frame centers for the transformer traces and explicitly retain the chosen interval-to-frame convention. This audit does not mark the figure revision complete.

Suggested concise English caption wording for that revision: “HuBERT Tr-24 provides 1,024-dimensional contextual representations at 20 ms intervals. The convolutional front end has a 25 ms receptive field. Corpus-level t-SNE maps each frame to three dimensions, and lines connect adjacent frames.” The transformer representations incorporate information from the input sequence beyond the front-end window; the term *contextual* preserves that distinction without adding a numerical receptive-field claim for Tr-24.

### CNN layers need separate wording

The archived [extraction helper](<C:/Users/Alex/Documents/GitHub/Similarity-based inferences predict cross-talker generalization during human speech perception/recycle_bin/project_reorganization_2026-08-21/legacy_root/preprocessing/feature_utils.py>) matches the HDF5's recorded helper checksum. It adaptively average-pools intermediate CNN outputs to the final CNN's temporal length before storage. Direct inspection confirms that every stored selected CNN layer in this example has 66 frames × 512 dimensions.

| Stored key, zero-based CNN index | Native pre-pooling stride | Native convolution receptive field |
|---|---:|---:|
| `cnn_2` | 1.25 ms | 2.5 ms |
| `cnn_3` | 2.5 ms | 5 ms |
| `cnn_4` | 5 ms | 10 ms |
| `cnn_5` | 10 ms | 15 ms |
| `cnn_6` | 20 ms | 25 ms |

These native strides must not be assigned to the already pooled HDF5 arrays. The pooled intermediate CNN vectors also should not be described as having exactly the last CNN's 25 ms receptive field. This check concerns the pinned Figure 1 checkpoint and its verified extraction helper; it does not certify every legacy feature file.

The matching archived [X21 extractor](<C:/Users/Alex/Documents/GitHub/Similarity-based inferences predict cross-talker generalization during human speech perception/recycle_bin/project_reorganization_2026-08-21/legacy_root/code/feature_extraction/extract_x21_hubert.py>) crops sentence audio at 22.05 kHz, resamples to 16 kHz, calls the pinned `HubertModel` in evaluation mode, and stores the selected hidden states. Both this script and the helper match the checksums recorded in the current HDF5; the similarly named helper at the sibling worktree's top level has a different checksum and was not used as the provenance authority.

## 2. AN19 matrix: 138 words, 42 talkers

**The current matrix averages the same 138 word labels for every one of 861 unordered talker pairs.** It has 42 talkers: six English-reference talkers and 36 L2 talkers. The count 178 is unsupported by the inspected matrix inputs. This audit does not determine whether that number was misspoken or mistranscribed in the meeting.

The [current panel builder](../../scripts/build_september9_data_panels.py) reads the retained [talker-pair summary](../../../results/derived/AN19-talker-validation-base-tr24/talker_pair_summary.csv). Independent checks found:

1. The summary has 861 unique unordered pairs, covering all `42 * 41 / 2` combinations; every `n_shared_words` value is 138.
2. The retained [word-level source in the sibling worktree](<C:/Users/Alex/Documents/GitHub/Similarity-based inferences predict cross-talker generalization during human speech perception/results/derived/AN19-talker-validation-base-tr24/word_pair_summary.csv>) has 118,818 unique pair–word rows (`861 * 138`), exactly 138 words per pair, and no duplicate pair–word keys. It accounts for 118,859 physical recording pairs because some word cells contain more than one physical comparison.
3. The [current stimulus manifest](../../../data/manifests/AN19-stimulus-manifest.csv) has 6,261 rows, 42 talkers, and 156 word labels in the union. Intersecting the word sets across all 42 talkers gives exactly the same 138 labels as the word-level source. Hashing the sorted word list with the original compact-JSON convention reproduces `2037cb683ed108329b4971547ad56f23cda0a3e7a1db34afbca826d960fe6129`.
4. Averaging the word-level distances and similarities reproduces the talker-pair summary to CSV floating-point precision (maximum differences `1.07e-14` and `9.54e-17`, respectively).
5. The current [distance matrix](../../analysis_update_2026-09-09/tables/figure_2a_42_talker_distance_matrix.csv) and [similarity matrix](../../analysis_update_2026-09-09/tables/figure_2a_42_talker_similarity_matrix.csv) are both 42 × 42, have all 1,722 off-diagonal cells populated, and have 42 masked diagonal cells. They reproduce their corresponding pair-summary values to floating-point precision (maximum differences `3.56e-15` and `3.39e-21`).

The aggregation is physical comparisons averaged within a word, then equal weighting across 138 words. Similarity uses the mean of `exp(-distance)` at the physical-comparison level, followed by those averaging steps. These checks verify the retained plot's source/counts/aggregation; they do not constitute a new raw-audio naming audit or a new DTW run.

## Recorded source checksums

Checksums below are SHA-256. HDF5 files were opened read-only; the two array checksums cover only the selected Figure 1 arrays converted to float64, not the entire multi-gigabyte stores.

| Source | SHA-256 |
|---|---|
| Pinned model `config.json` | `52e48f7f44bf7f1be327cc3e82368681acf0cc8c5538d757eae9e6685bfe2b16` |
| Pinned `preprocessor_config.json` | `a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335` |
| Archived X21 extractor, matching HDF5 attribute | `91becc9fe0b50576558ef6628df311851a96137c58e03fef6a48ba47f7ced3cf` |
| Archived feature helper, matching HDF5 attribute | `8ed80151505e733ea309dd41a08bf1435062df6008b3b29e0ca1a5c1b0cf6875` |
| Current method-panel script | `33834f677af8b0d5941afbf890e9c5ecb18d73341b1b962bbdf42bd1b2d56b6a` |
| Current method metadata | `5e0982d905da5c824a6a4b8480873afc548afff88de5c5f669fa757fb6c89fcd` |
| Selected full Tr-24 array, float64 | `2fb76e5bdafdb50343f851770565c8fdcdfc28ec07eaf8c237193516ce3d60f6` |
| Selected t-SNE Tr-24 array, float64 | `a71592bf2ddc4fcd43d45191f2ec7e7c42229fb002cf344e51393156e3055dba` |
| Current data-panel script | `9a8ba88812da36cb39e4903c4aa45626a3b96ecd3b5be487ee72ba83ab0d25fe` |
| Talker-pair summary | `b20f919d58973de12bd625cdb1c860a520783686fc52ba12e484110fa0a5bf7e` |
| Retained word-pair summary | `1c3e13bd8215f368306d5a4cb82b21ffc5495484beaaa5bb56aa82d75aa6db57` |
| Current AN19 manifest, file bytes | `2557a4870750a438debc09def1e402f96fc9ae1cce5d2bb6078fc29626f0cfaa` |
| Current 42-talker distance matrix | `ee35779aab39d14ecbc909aa3f65ea702d8e173681e10295114e4086a6d94834` |
| Current 42-talker similarity matrix | `62942a05c4c3ee513dd1b6b336e5177c552408d2a6054046fc4d01b7d9e1f8ef` |

The manifest's file-byte checksum uses a different serialization from the historical canonical-row checksum recorded in older provenance; those two hashes are not interchangeable.
