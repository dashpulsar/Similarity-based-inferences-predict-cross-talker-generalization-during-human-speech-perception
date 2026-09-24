# Source checks for the collaborator report

## Recovered notebooks

Zhengyang supplied the `Nature-Human-Behavior` folder of the earlier `cross-talker-ASR` project. The original files were inspected read-only, not executed or overwritten.

- `Figure_fit_reg.ipynb`, cell 0: X21 Tr-24 similarity/accuracy curves by test talker, with condition-specific and pooled binning. Its input is `cross-validation/all_layers_similarity_df_d.csv`. This is the requested S-curve style, not the AN19 segment-error analysis.
- Cell 2: English-talker waveform, word/phone intervals, feature display and sentence trajectory. Its t-SNE example is Tr-12, not Tr-24.
- Cell 4: English/Mandarin sentence DTW visualization. Each sentence's axes are standardized separately in that illustration. This is not the same distance construction as the current unscaled corpus t-SNE analysis.
- `Figure_create.ipynb`: further method illustrations and X21/AN19/B23 descriptive curve variants. It does not supply AN19 phoneme-boundary or minimal-pair-neighbor tables.
- `extract_features.py` loads `facebook/wav2vec2-lv-60-espeak-cv-ft` and writes `hidden_features.npy`. Therefore that script does not establish HuBERT provenance for the notebook's feature image. The replacement Figure 1a-b reads both full and reduced Tr-24 values from the declared HuBERT HDF5 stores.

The original English WAV supplied alongside the notebook matches the SHA-256 recorded in the repository's Git LFS pointer. A copy is retained under `sources/` for reproducing the new waveform; the repository's original LFS pointer was not replaced. The new script checks the hash before using this fallback.

## Local manuscript and original AN19 paper

`references/manuscript_draft.pdf` supplies the complete Figure 2d caption: lexical constraint is the number of minimal-pair neighbors distinguished by the target segment. It also specifies MA1 (ASR fine-tuning), MA2 (architecture), and MA3 (dimensionality reduction, including UMAP and PCA), plus a similarity-measure ablation. These are draft requests, not evidence that every ablation has been run.

The same draft's Figure 5 caption asks for a Bonferroni-corrected line. The current SBI review uses nominal +/-1.96 lines. The multiple-testing family must be specified before a corrected line is treated as the paper's final threshold; a nominal per-coefficient line is not a test of the mean fold z. The caption's phrase “proportion of explainable variance” also does not describe a normalized z ratio.

The draft says boundaries were checked and are on OSF, but contains `[URL/DOI]` rather than a resolvable archive address. That sentence cannot establish that those annotations are present locally. The AN19 raw-data folder contains 6,261 WAV entries, a behavioral workbook and the study paper; no TextGrid, EAF, CTM or LAB files were found there. Its production manifest has no phone-interval field, and both IPA response columns are empty. See `tables/an19_segment_input_audit.json` for the concrete inventory.

**Subsequent clarification:** Zhengyang confirms AN19 has no phone-level labels. Panels 2b/2d are therefore deferred pending a separately agreed alignment/verification effort; the draft's generic OSF statement must not be interpreted as proof that AN19 labels already exist.

**HW74 naming audit:** all six English files are labeled `wade`, while the 12 Korean, 12 mixed-L1 and 12 Spanish files are labeled `wave`. Their Tr-24 features exist. English files can be resolved to actual audio through the local Git LFS object store even though the current worktree contains pointers. Thus the initial phrase "missing English recordings" was too strong: same-word matching is blocked by a lexical-label conflict. Matching by item number alone would potentially compare two different words. No alias, source rename, behavioral-target change or feature overwrite has been applied. See `tables/an19_hw74_naming_audit.csv`; automatic ASR screening, if present, is diagnostic rather than a verified transcript.

The executed audit located real audio and Tr-24 features for all 42 HW74 files. Local, unprompted Whisper large-v3-turbo screening returned "Wade" for five English speakers and "weight" for ef3; it returned "Wave" for Korean km3/km6 and "weight" for Spanish sm4/sm5. This does **not** verify any transcript, but it argues against silently assuming all six English filenames are mere wave typos. Human listening/content verification remains necessary; the current Figure 2c exclusion is retained.

Alexander & Nygaard (2019), Table I (PDF page 5), identifies the source speakers from India and China as Hindi and Mandarin speakers. These clarify the old manifest labels `Indian` and `Chinese`; `Somalian` is displayed as `Somali`. Original labels are preserved in the exported mapping. Broad language-to-branch labels are used for display, not to assign a talker's dialect-specific phonological inventory.

## PHOIBLE proposal

Figure 1d is a newly computed, explicitly proposed **consonant/vowel inventory overlap** panel. It does not silently replace the caption's broader phonological/phonotactic construct. It uses the [PHOIBLE v2.0 release](https://github.com/phoible/dev/tree/v2.0), not the moving development branch, and retains the selected inventory rows and every English-reference pairing.

For each pair of inventories, overlap is `|A intersection B| / |A union B|` over exact segment symbols, excluding tone inventories. The point is the mean across all available source inventory pairs; the horizontal line is their min-max range, **not** a confidence interval. This avoids choosing one favorable reference inventory, but does not solve dialect selection or symbol-transcription differences. Portuguese and Persian/Farsi coverage is at the language level, not a certified Brazilian or speaker-specific inventory.

Source: Moran, Steven & McCloy, Daniel (eds.). 2019. [PHOIBLE 2.0](https://phoible.org/). The selected inventory data and derivatives retain the source's CC BY-SA 3.0 attribution/share-alike conditions. No phonotactic scores were inferred from an inventory alone.
