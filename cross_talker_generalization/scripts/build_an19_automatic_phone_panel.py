"""Compare AN19 intended-phone intervals in existing HuBERT Tr-24 t-SNE space.

Consumes the completed full-corpus automatic alignment; does not alter audio,
annotations, or feature stores. Run with --jobs 8. Outputs are descriptive,
automatic-target-conditioned comparisons, not verified production transcriptions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
OUT = PROJECT / "analysis/speech/an19_phone_alignment"
ALIGNMENT = OUT / "nygaard_audio"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
sys.path.insert(0, str(PROJECT / "src"))
from ctg.metrics import dtw_distance

# Fixed before inspection of computed deviation values, not ranked by results.
DISPLAY_PHONES = ("ih", "iy", "ae", "eh", "uh", "uw", "th", "dh", "s", "sh", "r", "l")
SEED = 20260909
SOURCE_HASHES = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source(path):
    SOURCE_HASHES[path.relative_to(ROOT).as_posix()] = digest(path)
    return path


def table(frame, name):
    frame.to_csv(TABLES / ("figure2b_" + name + ".csv"), index=False)


def boolean(series):
    if series.dtype == bool:
        return series
    return series.fillna(False).astype(str).str.lower().isin(["true", "1"])


def palette():
    path = PROJECT / "analysis/speech/sources/language_capitals_worldmap.R"
    code = source(path).read_text(encoding="utf-8-sig")
    block = re.search(r"family_colors\s*<-\s*c\((.*?)\n\)", code, flags=re.S).group(1)
    colors = dict(re.findall(r'"([^"\n]+)"\s*=\s*"(#[0-9a-fA-F]{6})"', block))
    path = PROJECT / "analysis/speech/tables/figure_si_talker_language_map.csv"
    languages = pd.read_csv(source(path))
    return colors, languages.loc[languages.dataset.eq("AN19")].set_index("speaker_id")


def load_inputs():
    status_path = ALIGNMENT / "tables/recording_alignment_status.csv"
    phones_path = ALIGNMENT / "tables/phone_intervals.csv"
    progress = json.loads((ALIGNMENT / "progress.json").read_text(encoding="utf-8"))
    if progress.get("processed") != 6261:
        raise ValueError("The complete 6,261-recording alignment attempt must finish before analysis")
    status = pd.read_csv(source(status_path), keep_default_na=False)
    phones = pd.read_csv(source(phones_path))
    manifest = pd.read_csv(source(ROOT / "data/manifests/AN19-stimulus-manifest.csv"))
    if len(status) != 6261 or not status.recording_id.is_unique:
        raise ValueError("Expected one alignment-status row per canonical recording")
    source_manifest = pd.read_csv(source(TABLES / "nygaard_audio_source_manifest.csv"))
    source_audit = json.loads(source(OUT / "nygaard_audio_source_audit.json").read_text(encoding="utf-8"))
    alignment_summary = json.loads(source(ALIGNMENT / "alignment_summary.json").read_text(encoding="utf-8"))
    if (len(source_manifest) != 6261 or not source_manifest.recording_id.is_unique
            or not source_manifest.status.eq("available").all()
            or not boolean(source_manifest.bytes_equal_to_repo).all()):
        raise ValueError("User-selected audio must be fully mapped and byte-identical to the feature source")
    if digest(TABLES / "nygaard_audio_source_manifest.csv") != source_audit["mapping_csv_sha256"]:
        raise ValueError("User-selected source mapping differs from the completed source audit")
    joined = status[["recording_id", "sha256"]].merge(
        source_manifest[["recording_id", "sha256"]], on="recording_id", validate="one_to_one", suffixes=("_alignment", "_source"))
    if len(joined) != 6261 or not joined.sha256_alignment.eq(joined.sha256_source).all():
        raise ValueError("Alignment recording hashes do not match the user-selected source")
    phones = phones.loc[~boolean(phones.is_silence)].copy()
    if phones.duplicated(["recording_id", "phone_index"]).any():
        raise ValueError("Duplicate phone interval identity")
    phones["original_word"] = phones.word.str.lower()
    # Preserve source labels while honoring an explicit annotation-only spelling
    # correction if the completed aligner exports an annotation transcript.
    transcript_column = next((c for c in ("annotation_word", "annotation_transcript", "alignment_word", "target_word") if c in status), None)
    if transcript_column:
        lookup = status.set_index("recording_id")[transcript_column]
        phones["comparison_word"] = phones.recording_id.map(lookup).replace("", np.nan).fillna(phones.original_word).str.lower()
    else:
        phones["comparison_word"] = phones.original_word
    override_path = PROJECT / "configs/an19_annotation_transcript_overrides.json"
    if override_path.exists():
        override = json.loads(source(override_path).read_text(encoding="utf-8"))
        # Support a recording-ID mapping or an explicit list of override records;
        # reject unrecognized entries instead of inferring lexical aliases.
        rows = override.get("overrides", override) if isinstance(override, dict) else override
        if isinstance(rows, dict):
            rows = [dict(recording_id=k, **v) if isinstance(v, dict)
                    else dict(recording_id=k, annotation_transcript=v) for k, v in rows.items()]
        for row in rows:
            recording_id = row.get("recording_id")
            corrected = row.get("annotation_word", row.get("annotation_transcript", row.get("target_word", row.get("corrected_word"))))
            if recording_id is None or corrected is None:
                raise ValueError("Unrecognized annotation-transcript override entry")
            phones.loc[phones.recording_id.eq(recording_id), "comparison_word"] = str(corrected).lower()
    metadata = manifest[["recording_id", "source_filename"]].copy()
    metadata["lexical_identity_unresolved"] = metadata.source_filename.str.lower().str.contains("hw74")
    phones = phones.merge(metadata, on="recording_id", validate="many_to_one")
    audio_hash = status.set_index("recording_id").sha256
    phones["audio_sha256"] = phones.recording_id.map(audio_hash)
    for column in ("shorter_than_hubert_stride", "low_target_posterior"):
        phones[column] = boolean(phones[column])
    phones = phones.sort_values(["recording_id", "phone_index"])
    sequences = phones.groupby("recording_id").canonical_phone.agg(" ".join)
    phones["canonical_sequence"] = phones.recording_id.map(sequences)
    phones["canonical_position"] = phones.groupby("recording_id").cumcount()
    phones["interval_id"] = phones.recording_id + ":phone:" + phones.phone_index.astype(str)
    return phones, status, alignment_summary


def crop_features(phones):
    path = ROOT / "data/features/an19_hubert_full_corpus_tsne_3d.h5"
    source(path)
    arrays, crop_rows = {}, []
    with h5py.File(path, "r") as store:
        for number, (recording_id, group) in enumerate(phones.groupby("recording_id"), 1):
            speaker_id = group.speaker_id.iloc[0]
            seq = np.asarray(store["tr_24"][speaker_id][recording_id], dtype=np.float64)
            if seq.ndim != 2 or seq.shape[1] != 3 or not np.isfinite(seq).all():
                raise ValueError(f"Invalid feature sequence: {recording_id}")
            centers = (199.5 + 320 * np.arange(len(seq))) / 16000
            for row in group.itertuples():
                indices = np.flatnonzero((centers >= row.start_seconds) & (centers < row.end_seconds))
                if len(indices):
                    arrays[row.interval_id] = np.ascontiguousarray(seq[indices])
                crop_rows.append(dict(interval_id=row.interval_id, n_feature_frames=len(indices),
                                      first_feature_frame=int(indices[0]) if len(indices) else None,
                                      last_feature_frame=int(indices[-1]) if len(indices) else None,
                                      feature_frame_indices=" ".join(str(i) for i in indices),
                                      no_feature_center=not bool(len(indices))))
            if number % 1000 == 0:
                print(f"Cropped {number} recordings", flush=True)
    return phones.merge(pd.DataFrame(crop_rows), on="interval_id", validate="one_to_one"), arrays


def calculate_pairs(phones, arrays, jobs):
    match_columns = ["comparison_word", "canonical_sequence", "canonical_position", "canonical_phone"]
    usable = phones.loc[~phones.lexical_identity_unresolved & ~phones.no_feature_center].copy()
    english = usable.loc[usable.accent.eq("ENG")].copy()
    # Identical audio at two manifest paths is not two independently produced
    # English reference tokens. Retain the manifest identities in the crop table.
    english = english.sort_values("interval_id").drop_duplicates(
        ["speaker_id", "audio_sha256", *match_columns])
    english_groups = {key: group for key, group in english.groupby(match_columns, sort=False)}
    target = usable.loc[~usable.accent.eq("ENG")].copy()
    requests = []
    for row in target.itertuples():
        key = tuple(getattr(row, name) for name in match_columns)
        refs = english_groups.get(key)
        if refs is None:
            continue
        for ref in refs.itertuples():
            requests.append((row.interval_id, ref.interval_id, ref.speaker_id,
                             bool(row.shorter_than_hubert_stride or row.low_target_posterior or
                                  ref.shorter_than_hubert_stride or ref.low_target_posterior)))
    def compare(request):
        target_id, ref_id, speaker_id, qc_flag = request
        distance = dtw_distance(arrays[target_id], arrays[ref_id], tau=2,
                                normalization="mean_sequence_length").distance
        return dict(target_interval_id=target_id, reference_interval_id=ref_id,
                    english_speaker_id=speaker_id, raw_distance=float(distance),
                    either_interval_qc_flag=qc_flag)
    if not requests:
        raise ValueError("No comparable intended-phone reference pairs")
    compare(requests[0])  # Compile the CPU kernel before creating worker threads.
    print(f"Computing {len(requests)} same-word/same-position DTW pairs on {jobs} threads", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pairs = pd.DataFrame(pool.map(compare, requests))
    if not np.isfinite(pairs.raw_distance).all() or (pairs.raw_distance < 0).any():
        raise ValueError("Invalid phone-pair distance")
    return pairs, english


def target_estimates(phones, pairs):
    target = phones.loc[~phones.accent.eq("ENG")].copy()
    per_speaker = pairs.groupby(["target_interval_id", "english_speaker_id"]).raw_distance.mean()
    estimate = per_speaker.groupby("target_interval_id").agg(["mean", "size"]).rename(
        columns={"mean": "mean_reference_dtw", "size": "n_english_reference_speakers"})
    target = target.merge(estimate, left_on="interval_id", right_index=True, how="left", validate="one_to_one")
    target["n_english_reference_speakers"] = target.n_english_reference_speakers.fillna(0).astype(int)
    target["six_reference_complete"] = target.n_english_reference_speakers.eq(6)
    target["exclusion_reason"] = np.select(
        [target.lexical_identity_unresolved, target.no_feature_center,
         target.n_english_reference_speakers.eq(0), target.n_english_reference_speakers.between(1, 5)],
        ["unresolved_hw74_identity", "no_feature_center", "no_matched_reference", "fewer_than_six_english_speakers"],
        default="included_six_references")
    clean_pairs = pairs.loc[~pairs.either_interval_qc_flag]
    clean_per_speaker = clean_pairs.groupby(["target_interval_id", "english_speaker_id"]).raw_distance.mean()
    clean_estimate = clean_per_speaker.groupby("target_interval_id").agg(["mean", "size"]).rename(
        columns={"mean": "qc_filtered_reference_dtw", "size": "qc_filtered_n_english_speakers"})
    target = target.merge(clean_estimate, left_on="interval_id", right_index=True, how="left", validate="one_to_one")
    target["qc_filtered_n_english_speakers"] = target.qc_filtered_n_english_speakers.fillna(0).astype(int)
    target["qc_six_reference_complete"] = target.qc_filtered_n_english_speakers.eq(6)
    if target.n_english_reference_speakers.max() > 6:
        raise ValueError("Unexpected English speaker count")
    return target


def summaries(target, languages, n_boot):
    rng = np.random.default_rng(SEED)
    words_all, talkers_all, groups_all = [], [], []
    for analysis, flag, column in (("six_reference_complete", "six_reference_complete", "mean_reference_dtw"),
                                    ("qc_filtered_six_reference_complete", "qc_six_reference_complete", "qc_filtered_reference_dtw")):
        selected = target.loc[target[flag]].copy()
        # Repeated positions/tokens first average within word type, then words get
        # equal weight within talker/phone; participants never weight production.
        words = selected.groupby(["speaker_id", "canonical_phone", "comparison_word"])[column].agg(
            ["mean", "size"]).reset_index().rename(columns={"mean": "mean_distance", "size": "n_phone_intervals"})
        words["analysis"] = analysis
        talkers = words.groupby(["speaker_id", "canonical_phone"]).agg(
            mean_distance=("mean_distance", "mean"), n_word_types=("comparison_word", "nunique"),
            n_phone_intervals=("n_phone_intervals", "sum")).reset_index()
        talkers["language"] = talkers.speaker_id.map(languages.language)
        talkers["branch"] = talkers.speaker_id.map(languages.branch)
        talkers["analysis"] = analysis
        for (phone, language, branch), group in talkers.groupby(["canonical_phone", "language", "branch"]):
            values = group.mean_distance.to_numpy()
            low = high = np.nan
            if len(values) > 1:
                boot = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
                low, high = np.quantile(boot, [.025, .975])
            groups_all.append(dict(analysis=analysis, canonical_phone=phone, language=language,
                                   branch=branch, mean_distance=float(values.mean()), ci_low=low, ci_high=high,
                                   n_talkers=len(values), n_word_types_min=int(group.n_word_types.min()),
                                   n_word_types_max=int(group.n_word_types.max()),
                                   ci_status="not_estimable_one_talker" if len(values) == 1 else "talker_bootstrap_95pct"))
        words_all.append(words); talkers_all.append(talkers)
    return pd.concat(words_all, ignore_index=True), pd.concat(talkers_all, ignore_index=True), pd.DataFrame(groups_all)


def draw(groups, talkers, language_map, colors, target):
    groups = groups.loc[groups.analysis.eq("six_reference_complete")]
    talkers = talkers.loc[talkers.analysis.eq("six_reference_complete")]
    language_order = language_map.loc[~language_map.language.eq("English"), ["language", "branch"]].drop_duplicates().sort_values(["branch", "language"])
    names = language_order.language.tolist()
    color_lookup = language_order.set_index("language").branch.map(colors).to_dict()
    intended_counts = target.canonical_phone.value_counts()
    def board(selected, rows, size, stem):
        fig, axes = plt.subplots(rows, 4, figsize=size, sharey=True, squeeze=False)
        compact = rows == 1
        fig.subplots_adjust(left=.13 if compact else .105, right=.99, bottom=.12 if compact else .08,
                            top=.81 if compact else .89, wspace=.13, hspace=.24)
        for phone, ax in zip(selected, axes.flat):
            # Identical jitter for each phone in the overview and its companion.
            rng = np.random.default_rng(SEED + DISPLAY_PHONES.index(phone))
            means = groups.loc[groups.canonical_phone.eq(phone)].set_index("language")
            for y, language in enumerate(names):
                points = talkers.loc[talkers.canonical_phone.eq(phone) & talkers.language.eq(language)]
                color = color_lookup[language]
                if len(points):
                    jitter = rng.uniform(-.14, .14, len(points))
                    ax.scatter(points.mean_distance, y + jitter, color=color, alpha=.42, s=12, edgecolors="none")
                if language in means.index:
                    row = means.loc[language]
                    if row.n_talkers > 1:
                        ax.plot([row.ci_low, row.ci_high], [y, y], color=color, lw=2)
                    ax.scatter(row.mean_distance, y, color=color, marker="o" if row.n_talkers > 1 else "D", s=30, zorder=4)
            ax.set_title(phone.upper(), fontsize=14, fontweight="bold")
            ax.set_yticks(range(len(names)), names, fontsize=12 if compact else 10)
            ax.tick_params(axis="x", labelsize=11)
            ax.set_ylim(len(names)-.4, -.6)
            ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.grid(axis="x", color="#eeeeee", lw=.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if means.empty:
                message = (f"No intended {phone.upper()} tokens\nin this corpus lexicon"
                           if intended_counts.get(phone, 0) == 0 else "No six-reference-complete\nestimates")
                ax.set_xticks([])
                ax.spines["bottom"].set_visible(False)
                ax.text(.5, .5, message, transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, wrap=True)
        fig.suptitle("AN19 | Intended-phone deviation from L1-English", fontsize=18 if compact else 20, y=.98)
        fig.text(.5, .912 if compact else .946, "AUTOMATIC TARGET-PHONE ALIGNMENT - NOT MANUALLY VERIFIED",
                 ha="center", fontsize=12, color="#8c2d2d", fontweight="bold")
        fig.text(.5, .023 if compact else .018,
                 "Mean DTW distance to six L1-English talkers (HuBERT Tr-24, 3-D t-SNE)", ha="center", fontsize=13)
        handles = [Line2D([], [], marker="o", linestyle="none", color="#777", markersize=4, label="Individual L2 talker"),
                   Line2D([], [], marker="o", color="#333", markersize=6, label="L1 mean and 95% talker-bootstrap CI"),
                   Line2D([], [], marker="D", linestyle="none", color="#333", markersize=5, label="Single talker: no group CI")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .884 if compact else .928),
                   ncol=3, frameon=False, fontsize=11)
        for ext in ("png", "svg"):
            fig.savefig(FIGURES / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)
    board(DISPLAY_PHONES, 3, (19.5, 14.3), "figure2b_automatic_intended_phone_deviation")
    for page in range(3):
        board(DISPLAY_PHONES[page*4:(page+1)*4], 1, (15.5, 7.0),
              f"figure2b_automatic_intended_phone_deviation_{page+1:02d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    design = {"predeclared_display_phones": DISPLAY_PHONES, "selection": "Fixed before deviation computation",
              "bootstrap": args.bootstrap, "seed": SEED, "dtw_tau": 2,
              "dtw_normalization": "mean_sequence_length", "coordinate_scaling": "none",
              "feature_space": "HuBERT base Tr-24, existing 3-D corpus t-SNE",
              "primary_coverage": "All six English reference speakers required for each target phone interval"}
    (TABLES / "figure2b_analysis_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    colors, languages = palette()
    phones, status, alignment_summary = load_inputs()
    phones, arrays = crop_features(phones)
    table(phones, "phone_frame_mapping")
    pairs, english = calculate_pairs(phones, arrays, args.jobs)
    table(pairs, "matched_phone_pairs")
    table(english[["interval_id", "recording_id", "speaker_id", "comparison_word", "canonical_sequence",
                   "canonical_position", "canonical_phone", "audio_sha256"]], "english_reference_tokens")
    target = target_estimates(phones, pairs)
    table(target, "target_phone_estimates")
    word, talker, groups = summaries(target, languages, args.bootstrap)
    table(word, "word_type_estimates"); table(talker, "talker_phone_estimates"); table(groups, "language_phone_estimates")
    coverage = target.groupby(["canonical_phone", "n_english_reference_speakers", "exclusion_reason"]).size().reset_index(name="n_target_phone_intervals")
    table(coverage, "reference_coverage")
    display = pd.DataFrame({"canonical_phone": DISPLAY_PHONES})
    display = display.merge(target.groupby("canonical_phone").agg(
        n_target_intervals=("interval_id", "size"), n_six_reference_intervals=("six_reference_complete", "sum"),
        n_qc_six_reference_intervals=("qc_six_reference_complete", "sum")),
        on="canonical_phone", how="left").fillna(0)
    table(display, "predeclared_phone_coverage")
    draw(groups, talker, languages, colors, target)
    notes = {
        "design": design, "status": "Automatically aligned intended-phone comparisons; not manually verified",
        "audio_input": "User-selected July/Nygaard_audio; full fresh annotation read from nygaard_audio subfolder",
        "audio_to_hubert_compatibility": "All 6,261 selected source audio hashes match the original feature-source audio; two path renames resolved by unique same-speaker byte hash",
        "alignment_summary_path": (ALIGNMENT / "alignment_summary.json").relative_to(ROOT).as_posix(),
        "alignment_model_config": alignment_summary.get("config", {}),
        "alignment_model_config_sha256": alignment_summary.get("config_sha256"),
        "alignment_recordings_attempted": int(len(status)), "alignment_recordings_aligned": int(status.status.eq("aligned").sum()),
        "speech_phone_intervals": int(len(phones)), "speech_intervals_without_hubert_center": int(phones.no_feature_center.sum()),
        "l2_phone_intervals": int(len(target)), "same_content_physical_phone_pairs": int(len(pairs)),
        "six_reference_complete_target_intervals": int(target.six_reference_complete.sum()),
        "qc_filtered_six_reference_complete_target_intervals": int(target.qc_six_reference_complete.sum()),
        "exclusions": target.exclusion_reason.value_counts().to_dict(),
        "all_supported_canonical_phones": sorted(target.canonical_phone.unique().tolist()),
        "comparison_key": ["intended word", "complete canonical phone sequence", "canonical phone position", "canonical phone symbol"],
        "frame_centers_seconds": "(199.5 + 320 * frame_index) / 16000; include center in [phone_start, phone_end)",
        "empty_crop_policy": "Exclude and count; no interpolation or forced one-frame segment",
        "english_aggregation": "Dedupe identical audio within speaker/word/phone position; mean remaining physical reference distances within speaker, then equally across English speakers",
        "l2_aggregation": "Mean intended-phone intervals within each word type, then equal-weight word types within talker/phone, then equal-weight talkers within L1/phone",
        "ci": "Percentile bootstrap of L2 talkers within L1/phone, 1,000 replicates by default; no CI for one-talker groups",
        "word_coverage": "Available six-reference-complete word types can differ by talker/phone; exported minimum/maximum word counts should be inspected",
        "qc_sensitivity": "Exclude pairs if either interval is shorter than 20 ms or flagged low target posterior; then require all six English speakers again",
        "lexical_conflict": "Exclude every HW74 wave/wade recording; annotation-only chief spelling override read explicitly if provided",
        "interpretation": "Distances quantify HuBERT trajectory differences inside automatically assigned intended-phone intervals. They do not identify actual substitutions, deletions, or verified produced phone categories",
        "human_verified": False, "source_audio_modified": False, "feature_extraction_rerun": False,
        "source_hashes": SOURCE_HASHES,
    }
    source(Path(__file__).resolve())
    notes["source_hashes"] = SOURCE_HASHES
    notes["output_hashes"] = {path.relative_to(OUT).as_posix(): digest(path)
                              for path in [*TABLES.glob("figure2b_*.csv"), *FIGURES.glob("figure2b_*")]}
    (TABLES / "figure2b_metadata.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(json.dumps({key: notes[key] for key in ("speech_phone_intervals", "same_content_physical_phone_pairs",
                                                "six_reference_complete_target_intervals", "exclusions")}, indent=2), flush=True)
    print("Figure 2b automatic intended-phone panel complete.", flush=True)


if __name__ == "__main__":
    main()
