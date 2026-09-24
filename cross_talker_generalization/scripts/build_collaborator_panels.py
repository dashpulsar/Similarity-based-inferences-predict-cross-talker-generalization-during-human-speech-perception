"""Build data-backed Figure 1/2 review panels; never infer missing phone times.

Python requirements: numpy, pandas, scipy, matplotlib, h5py, statsmodels, numba.
Run from any directory. PHOIBLE v2.0 is downloaded to artifacts/external once.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import urllib.request

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.io import wavfile
import statsmodels.api as sm

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from ctg.features import slice_by_time
from ctg.metrics import dtw_distance

OUT = PROJECT / "analysis/model_comparison/reference_checks/collaborator_report"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SOURCES = {}
LAYER = "tr_24"
SEED = 230519
PHOIBLE_URL = "https://raw.githubusercontent.com/phoible/dev/v2.0/data/phoible.csv"
# AN19 paper Table I explicitly identifies Hindi and Mandarin; keep original labels.
LANGUAGES = {
    "English": ("English", "eng", "Germanic"),
    "Korean": ("Korean", "kor", "Koreanic"),
    "Albanian": ("Albanian", "als", "Albanian"),
    "Dutch": ("Dutch", "nld", "Germanic"),
    "French": ("French", "fra", "Romance"),
    "German": ("German", "deu", "Germanic"),
    "Japanese": ("Japanese", "jpn", "Japonic"),
    "Somalian": ("Somali", "som", "Cushitic"),
    "Romanian": ("Romanian", "ron", "Romance"),
    "Bengali": ("Bengali", "ben", "Indo-Aryan"),
    "Russian": ("Russian", "rus", "Slavic"),
    "Chinese": ("Mandarin", "cmn", "Sinitic"),
    "Mandarin": ("Mandarin", "cmn", "Sinitic"),
    "Turkish": ("Turkish", "tur", "Turkic"),
    "Indian": ("Hindi", "hin", "Indo-Aryan"),
    "Spanish": ("Spanish", "spa", "Romance"),
    "Brazilian Portuguese": ("Brazilian Portuguese", "por", "Romance"),
    "Farsi": ("Farsi", "pes", "Iranian"),
}
BRANCHES = sorted({x[2] for x in LANGUAGES.values()})
BRANCH_COLORS = dict(zip(BRANCHES, plt.get_cmap("tab20").colors[:len(BRANCHES)]))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source(path):
    SOURCES[path.relative_to(REPO).as_posix()] = sha(path)
    return path


def read(path, **kwargs):
    return pd.read_csv(source(path), **kwargs)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}", flush=True)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


def trajectories(reference_audio=None):
    manifest = read(REPO / "data/manifests/X21-stimulus-manifest.csv")
    key = "X21.ENG.ENG_M_055.HT1_S001"
    row = manifest.set_index("segment_id").loc[key]
    reduced_path = REPO / "data/features/x21_hubert_full_corpus_tsne_3d.h5"
    full_path = REPO / "data/features/x21_hubert_full_corpus_features.h5"
    with h5py.File(source(reduced_path), "r") as f:
        reduced = np.asarray(f[LAYER][row.speaker_id][key], dtype=float)
        params = json.loads(f.attrs["tsne_parameters_json"])
        model = str(f.attrs["model_id"])
        revision = str(f.attrs["model_revision"])
        scope = str(f.attrs["corpus_scope"])
    with h5py.File(full_path, "r") as f:
        full = np.asarray(f[row.speaker_id][key][LAYER], dtype=float)
    assert full.shape == (len(reduced), 1024)
    SOURCES[full_path.relative_to(REPO).as_posix()] = {
        "selected_leaf": f"{row.speaker_id}/{key}/{LAYER}",
        "selected_array_sha256_float64": hashlib.sha256(full.tobytes()).hexdigest(),
        "file_size_bytes": full_path.stat().st_size,
    }
    audio_path = REPO / row.source_wav_relpath
    if audio_path.stat().st_size < 1024 and audio_path.read_bytes().startswith(b"version https://git-lfs"):
        expected = re.search(r"oid sha256:([0-9a-f]+)", audio_path.read_text()).group(1)
        cached = OUT / "sources/ALL_055_M_ENG_ENG_HT1.wav"
        candidate = reference_audio or cached
        if not candidate.exists() or sha(candidate) != expected:
            raise ValueError("Example WAV is an LFS pointer; supply --reference-audio with the matching real recording")
        cached.parent.mkdir(exist_ok=True)
        if candidate.resolve() != cached.resolve():
            shutil.copyfile(candidate, cached)
        audio_path = cached
    rate, audio = wavfile.read(source(audio_path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio[round(row.start_seconds * rate):round(row.end_seconds * rate)]
    t = np.arange(len(reduced)) * float(row.duration_seconds) / len(reduced)
    intervals = json.loads(row.word_intervals_json)
    phones = json.loads(row.phone_intervals_json)
    word = next(x for x in intervals if x["normalized"] == "FELL")
    word_seq = slice_by_time(reduced, word["start_seconds"], word["end_seconds"], row.duration_seconds)
    word_phones = [x for x in phones if x["start_seconds"] >= word["start_seconds"] - 1e-5
                   and x["end_seconds"] <= word["end_seconds"] + 1e-5]
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(grid[0, 0])
    ax.plot(np.arange(len(audio)) / rate, audio / max(1, np.abs(audio).max()), color="#285c83", lw=.7)
    ax.set(xlabel="Time (s)", ylabel="Relative amplitude", title="a  Recording")
    tidy(ax)
    ax = fig.add_subplot(grid[0, 1])
    # Display all dimensions. No feature selection or transformation of analysis values.
    im = ax.imshow(full.T, aspect="auto", origin="lower", extent=[0, row.duration_seconds, 1, 1024], cmap="RdBu_r")
    fig.colorbar(im, ax=ax, shrink=.65, label="Activation")
    ax.set(xlabel="Time (s)", ylabel="Latent dimension", title="HuBERT-large, Tr-24: T x 1024")
    bounds = (reduced.min(axis=0), reduced.max(axis=0))
    def ax3(cell, title, points, context=False):
        a = fig.add_subplot(cell, projection="3d")
        if context:
            a.plot(*reduced.T, color="#b8b8b8", alpha=.45, lw=1)
        a.plot(*points.T, color="#20639b", lw=1.8)
        a.scatter(*points[0], color="#20639b", marker="o", s=22)
        a.scatter(*points[-1], color="#20639b", marker="x", s=25)
        a.set(xlabel="t-SNE 1", ylabel="t-SNE 2", zlabel="t-SNE 3", title=title)
        a.set_xlim(bounds[0][0], bounds[1][0]); a.set_ylim(bounds[0][1], bounds[1][1]); a.set_zlim(bounds[0][2], bounds[1][2])
        a.tick_params(labelsize=7)
        a.set_box_aspect(np.maximum(bounds[1] - bounds[0], 1e-4))
        a.view_init(elev=23, azim=-58)
        return a
    ax3(grid[0, 2], "Corpus-level t-SNE: T x 3", reduced)
    ax3(grid[1, 0], 'b  Sentence: "A boy fell from a window"', reduced)
    ax3(grid[1, 1], 'Word: "fell"', word_seq, True)
    ax = ax3(grid[1, 2], 'Segments in "fell"', word_seq, True)
    # Replace blue word line with color-coded phoneme trajectories, in the same space.
    ax.lines[1].set_alpha(0)
    phone_rows = []
    for phone, color in zip(word_phones, ["#b53636", "#1f8a63", "#7045a2", "#e0a32d"]):
        seq = slice_by_time(reduced, phone["start_seconds"], phone["end_seconds"], row.duration_seconds)
        ax.plot(*seq.T, color=color, lw=3, label=phone["normalized"], marker=".", markersize=3)
        ax.scatter(*seq.T, color=color, s=23, depthshade=False)
        phone_rows.append({**phone, "n_frames": len(seq)})
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Figure 1a-b | From speech to sentence, word and segment trajectories", fontsize=15)
    save(fig, "figure_1ab_representations_tr24")
    pd.DataFrame(dict(frame=np.arange(len(t)), time_seconds=t, x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2])).to_csv(TABLES / "figure_1ab_frames.csv", index=False)
    pd.DataFrame(phone_rows).to_csv(TABLES / "figure_1b_phone_intervals.csv", index=False)
    (TABLES / "figure_1ab_metadata.json").write_text(json.dumps({
        "speaker": row.speaker_id, "L1": "English", "sex": "male", "sentence": row.sentence_normalized,
        "word": "FELL", "phone_labels": [x["normalized"] for x in word_phones], "layer": LAYER,
        "checkpoint": model, "revision": revision, "tsne_scope": scope, "tsne_parameters": params,
        "frames": len(t), "dimensions": full.shape[1], "display_scaling": "none",
        "frame_timing": "uniform manifest-relative positions; interval rounding follows ctg.features.slice_by_time",
        "start_marker": "circle", "end_marker": "cross"}, indent=2), encoding="utf-8")


def language_panels():
    records = []
    for dataset in ("AN19", "X21", "B23"):
        manifest = read(REPO / f"data/manifests/{dataset}-stimulus-manifest.csv")
        if dataset == "B23":
            manifest["source_l1_language"] = manifest.accent.map({
                "BRP": "Brazilian Portuguese", "FAR": "Farsi", "SPA": "Spanish", "TUR": "Turkish"})
        talkers = manifest[["speaker_id", "source_l1_language"]].drop_duplicates()
        for row in talkers.itertuples():
            if row.source_l1_language not in LANGUAGES:
                raise ValueError(f"Unregistered language {row.source_l1_language}")
            language, iso, branch = LANGUAGES[row.source_l1_language]
            records.append(dict(dataset=dataset, speaker_id=row.speaker_id, original_label=row.source_l1_language,
                                language=language, iso6393=iso, branch=branch))
    languages = pd.DataFrame(records)
    languages.to_csv(TABLES / "figure_1c_talker_language_map.csv", index=False)
    counts = languages.groupby(["branch", "language", "iso6393", "dataset"]).speaker_id.nunique().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    names = [x[1] for x in counts.index]
    for j, dataset in enumerate(("AN19", "X21", "B23")):
        for i, row in enumerate(counts.itertuples(index=False, name=None)):
            count = int(counts.iloc[i].get(dataset, 0))
            if count:
                ax.scatter(j, i, s=180, color=BRANCH_COLORS[counts.index[i][0]])
                ax.text(j, i, str(count), va="center", ha="center", color="black", fontsize=9, weight="bold")
    ax.set_xticks(range(3), ["AN19", "X21", "B23"])
    ax.set_yticks(range(len(names)), [f"{n}  [{b}]" for b, n, _ in counts.index])
    ax.set(xlim=(-.5, 2.5), ylim=(len(names)-.5, -.5), title="Figure 1c | L1 backgrounds in the available speech corpora")
    ax.text(.5, -.11, "Numbers = unique talkers in each corpus; English talkers are reference recordings.\nCorpus coverage includes recordings not used in experimental exposure.", transform=ax.transAxes, ha="center", fontsize=9)
    tidy(ax)
    save(fig, "figure_1c_language_coverage")
    counts.to_csv(TABLES / "figure_1c_language_counts.csv")
    cache = PROJECT / "artifacts/external/phoible_v2.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print("Downloading PHOIBLE v2.0", flush=True)
        partial = cache.with_suffix(".csv.partial")
        with urllib.request.urlopen(PHOIBLE_URL, timeout=120) as response, partial.open("wb") as target:
            expected_size = response.headers.get("Content-Length")
            while block := response.read(1024 * 1024):
                target.write(block)
        if expected_size is not None and partial.stat().st_size != int(expected_size):
            raise ValueError("Incomplete PHOIBLE download")
        partial.replace(cache)
    phoible = read(cache, usecols=["InventoryID", "ISO6393", "LanguageName", "Phoneme", "SegmentClass", "Source"])
    if phoible.InventoryID.nunique() != 3020:
        raise ValueError("Expected 3,020 inventories in PHOIBLE 2.0; check source completeness")
    wanted = set(languages.iso6393)
    # PHOIBLE v2 may encode Persian by the macro-language code rather than pes.
    if "pes" not in set(phoible.ISO6393):
        languages.loc[languages.iso6393.eq("pes"), "iso6393"] = "fas"
        wanted.discard("pes"); wanted.add("fas")
    selected = phoible.loc[phoible.ISO6393.isin(wanted) & phoible.SegmentClass.isin(["consonant", "vowel"])]
    selected.to_csv(TABLES / "figure_1d_phoible_segment_subset.csv", index=False)
    inventories = {(iso, inv): set(g.Phoneme) for (iso, inv), g in selected.groupby(["ISO6393", "InventoryID"])}
    refs = [(inv, items) for (iso, inv), items in inventories.items() if iso == "eng"]
    if not refs:
        raise ValueError("PHOIBLE English inventory missing")
    pairs = []
    for (iso, inv), items in inventories.items():
        if iso == "eng":
            continue
        for ref, ref_items in refs:
            pairs.append(dict(iso6393=iso, inventory_id=inv, english_inventory_id=ref,
                              n_segments=len(items), intersection=len(items & ref_items), union=len(items | ref_items),
                              jaccard=len(items & ref_items)/len(items | ref_items)))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(TABLES / "figure_1d_inventory_pairs.csv", index=False)
    summary = []
    for (language, iso, branch), group in languages.groupby(["language", "iso6393", "branch"]):
        if iso == "eng":
            continue
        values = pairs.loc[pairs.iso6393.eq(iso)]
        summary.append(dict(language=language, iso6393=iso, branch=branch,
                            mean_jaccard=values.jaccard.mean(), min_jaccard=values.jaccard.min(),
                            max_jaccard=values.jaccard.max(), n_inventories=values.inventory_id.nunique(),
                            n_english_inventories=len(refs), status="computed" if len(values) else "not_covered"))
    summary = pd.DataFrame(summary).sort_values("mean_jaccard", na_position="first")
    summary.to_csv(TABLES / "figure_1d_inventory_similarity.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    for i, row in enumerate(summary.itertuples()):
        if row.status == "computed":
            ax.plot([row.min_jaccard, row.max_jaccard], [i, i], color=BRANCH_COLORS[row.branch], lw=2)
            ax.scatter(row.mean_jaccard, i, color=BRANCH_COLORS[row.branch], s=45)
        else:
            ax.text(.01, i, "Not covered", va="center", color="#777")
    ax.set_yticks(range(len(summary)), [f"{r.language} (inventories: {r.n_inventories})" for r in summary.itertuples()])
    ax.set(xlabel="Consonant/vowel inventory Jaccard overlap with English", xlim=(0, 1),
           title="Figure 1d | Proposed segment-inventory comparison (PHOIBLE 2.0)")
    ax.text(.5, -.13, "Point = mean across inventory pairs; line = min-max across sources, NOT a 95% CI.\nLanguage-level comparison; not dialect-specific and not a phonotactic metric.",
            transform=ax.transAxes, ha="center", fontsize=9)
    tidy(ax); save(fig, "figure_1d_inventory_similarity_proposal")
    return languages


def talker_matrix(languages):
    path = REPO / "results/derived/AN19-talker-validation-base-tr24/talker_pair_summary.csv"
    pairs = read(path)
    talkers = languages.loc[languages.dataset.eq("AN19") & ~languages.language.eq("English")].sort_values(["branch", "language", "speaker_id"])
    keys = talkers.speaker_id.tolist()
    assert len(keys) == 36 and pairs.n_shared_words.eq(138).all()
    matrix = pd.DataFrame(np.nan, index=keys, columns=keys)
    for row in pairs.itertuples():
        if row.talker_a_id in keys and row.talker_b_id in keys:
            matrix.loc[row.talker_a_id, row.talker_b_id] = row.similarity_exp_k1
            matrix.loc[row.talker_b_id, row.talker_a_id] = row.similarity_exp_k1
    assert np.isfinite(matrix.to_numpy()[~np.eye(len(keys), dtype=bool)]).all()
    matrix.to_csv(TABLES / "figure_2a_36_l2_talker_matrix.csv", index_label="talker_id")
    talkers.to_csv(TABLES / "figure_2a_talker_order.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#eeeeee")
    image = ax.imshow(matrix, norm=LogNorm(), cmap=cmap)
    labels = [f"{r.speaker_id.rsplit('.',1)[-1]} {r.language}" for r in talkers.itertuples()]
    ax.set_xticks(range(36), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(36), labels, fontsize=7)
    for tick, (_, row) in zip(ax.get_yticklabels(), talkers.iterrows()):
        tick.set_color(BRANCH_COLORS[row.branch])
    ax.set_title("Figure 2a | Word-matched pronunciation similarity: 36 AN19 L2 talkers", pad=15)
    fig.colorbar(image, ax=ax, shrink=.75, label="Mean exp(-DTW distance) across 138 shared words (log color scale)")
    save(fig, "figure_2a_36_l2_talker_similarity")


def control_pairs(dataset, jobs):
    behavior = read(REPO / f"data/preprocessed data/{dataset}-behavioral-data.csv")
    behavior = behavior.loc[behavior.phase.eq("test") & behavior.exposure_test_condition_id.str.contains("Control")].copy()
    manifest = read(REPO / f"data/manifests/{dataset}-stimulus-manifest.csv")
    feature_path = REPO / f"data/features/{dataset.lower()}_hubert_full_corpus_tsne_3d.h5"
    queries = behavior[["item_id", "item_talker", "response_expected"]].drop_duplicates()
    queries["query_id"] = queries.item_id + "::" + queries.response_expected.str.lower()
    # Include the lexical target: AN19 W-number keys alone can collide across lists.
    behavior["query_id"] = behavior.item_id + "::" + behavior.response_expected.str.lower()
    sequences, requests, missing = {}, [], []
    with h5py.File(source(feature_path), "r") as h5:
        root = h5[LAYER]
        for query in queries.itertuples():
            refs = []
            if dataset == "AN19":
                targets = manifest.loc[manifest.speaker_id.eq(query.item_talker) & manifest.word.str.lower().eq(query.response_expected.lower())]
                english = manifest.loc[manifest.accent.eq("ENG") & manifest.word.str.lower().eq(query.response_expected.lower())]
                if len(targets) != 1 or english.speaker_id.nunique() != 6:
                    reason = "ambiguous target recording" if len(targets) != 1 else f"English reference pool has {english.speaker_id.nunique()} of 6 talkers"
                    missing.append({"query_id": query.query_id, "reason": reason}); continue
                target = targets.iloc[0]
                tkey = target.recording_id
                sequences[tkey] = np.asarray(root[target.speaker_id][tkey], dtype=float)
                for ref in english.itertuples():
                    rkey = ref.recording_id
                    sequences[rkey] = np.asarray(root[ref.speaker_id][rkey], dtype=float)
                    refs.append((ref.speaker_id, rkey))
            else:
                segment_id = query.item_id.rsplit(".W", 1)[0]
                targets = manifest.loc[manifest.segment_id.eq(segment_id)]
                if len(targets) != 1:
                    missing.append({"query_id": query.query_id, "reason": "missing sentence"}); continue
                target = targets.iloc[0]
                def crop(row):
                    intervals = [x for x in json.loads(row.word_intervals_json)
                                 if x["normalized"].lower() == query.response_expected.lower()]
                    if len(intervals) != 1:
                        raise ValueError("ambiguous keyword interval")
                    interval = intervals[0]
                    seq = np.asarray(root[row.speaker_id][row.segment_id], dtype=float)
                    return slice_by_time(seq, interval["start_seconds"], interval["end_seconds"], row.duration_seconds)
                english = manifest.loc[manifest.accent.eq("ENG") & manifest.sentence_code.eq(target.sentence_code)]
                if english.speaker_id.nunique() != 5:
                    missing.append({"query_id": query.query_id, "reason": "incomplete English pool"}); continue
                tkey = query.query_id
                try:
                    sequences[tkey] = crop(target)
                    for _, ref in english.iterrows():
                        rkey = ref.segment_id + "::" + query.response_expected.lower()
                        sequences[rkey] = crop(ref)
                        refs.append((ref.speaker_id, rkey))
                except ValueError as error:
                    missing.append({"query_id": query.query_id, "reason": str(error)}); continue
            for speaker, rkey in refs:
                requests.append((query.query_id, tkey, speaker, rkey))
    def compute(request):
        query_id, tkey, speaker, rkey = request
        distance = dtw_distance(sequences[tkey], sequences[rkey], tau=2, normalization="mean_sequence_length").distance
        return dict(query_id=query_id, target_key=tkey, english_speaker=speaker, reference_key=rkey, raw_distance=distance)
    # Compile the numba kernel before threads start.
    if requests:
        compute(requests[0])
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pairs = pd.DataFrame(executor.map(compute, requests))
    pairs.to_csv(TABLES / f"figure_2c_{dataset.lower()}_word_reference_pairs.csv", index=False)
    pd.DataFrame(missing, columns=["query_id", "reason"]).to_csv(TABLES / f"figure_2c_{dataset.lower()}_unmapped_items.csv", index=False)
    # Equal weight per English speaker, averaging any multiple recordings first.
    per_speaker = pairs.groupby(["query_id", "english_speaker"]).raw_distance.mean()
    values = per_speaker.groupby("query_id").agg(["mean", "size"]).rename(columns={"mean": "mean_reference_dtw", "size": "n_english_speakers"})
    result = behavior.merge(values, left_on="query_id", right_index=True, how="left", validate="many_to_one")
    result["mapping_status"] = np.where(result.mean_reference_dtw.notna(), "available", "unmapped")
    # Scale against unique target recordings, not their outcome-frequency weights.
    scale = float(values.mean_reference_dtw.median())
    result["similarity"] = np.exp(-result.mean_reference_dtw / scale)
    result["distance_scale_median_unique_targets"] = scale
    result.to_csv(TABLES / f"figure_2c_{dataset.lower()}_control_trials.csv", index=False)
    return result


def control_curves(jobs, n_boot):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    estimates, binned, curves, checks = [], [], [], []
    rng = np.random.default_rng(SEED)
    for dataset, ax in zip(("AN19", "X21"), axes):
        trials = control_pairs(dataset, jobs)
        valid = trials.loc[trials.mapping_status.eq("available")].copy()
        checks.append(dict(dataset=dataset, total_trials=len(trials), mapped_trials=len(valid),
                           participants=valid.participant_id.nunique(), test_talkers=valid.item_talker.nunique(),
                           unique_targets=valid.query_id.nunique()))
        for accent, group in valid.groupby("item_accent"):
            group = group.copy()
            color = {"KOR": "#a54d80", "SPA": "#c38327", "CMN": "#237cb1"}[accent]
            group["bin"] = pd.qcut(group.similarity, 10, labels=False, duplicates="drop")
            ids = group.participant_id.unique()
            for bin_id, points in group.groupby("bin"):
                totals = points.groupby("participant_id").agg(correct=("response_correct", "sum"), incorrect=("response_incorrect", "sum")).reindex(ids, fill_value=0)
                choices = rng.integers(0, len(totals), (n_boot, len(totals)))
                numerator = totals.correct.to_numpy()[choices].sum(axis=1)
                denominator = (totals.correct + totals.incorrect).to_numpy()[choices].sum(axis=1)
                values = numerator[denominator > 0] / denominator[denominator > 0]
                low, high = np.quantile(values, [.025, .975])
                mean = points.response_correct.sum()/(points.response_correct + points.response_incorrect).sum()
                x = points.similarity.mean()
                ax.errorbar(x, mean, yerr=[[mean-low], [high-mean]], fmt="o", color=color, alpha=.7, capsize=2)
                binned.append(dict(dataset=dataset, accent=accent, bin=bin_id, mean_similarity=x, accuracy=mean, ci_low=low, ci_high=high, n_trials=len(points)))
            design = sm.add_constant(group.similarity.to_numpy())
            fit = sm.GLM(group.response_correct.to_numpy(), design, family=sm.families.Binomial()).fit()
            grid = np.linspace(group.similarity.min(), group.similarity.max(), 100)
            predicted = fit.predict(sm.add_constant(grid))
            # Participant bootstrap for descriptive curve uncertainty. No claim of GLMM CV.
            chunks = {p: g for p, g in group.groupby("participant_id")}
            pred_boot = []
            for _ in range(n_boot):
                sampled = pd.concat([chunks[p] for p in rng.choice(list(chunks), len(chunks), replace=True)], ignore_index=True)
                try:
                    model = sm.GLM(sampled.response_correct.to_numpy(), sm.add_constant(sampled.similarity.to_numpy()), family=sm.families.Binomial()).fit()
                    if model.converged:
                        pred_boot.append(model.predict(sm.add_constant(grid)))
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if len(pred_boot) < .95*n_boot:
                raise ValueError("More than 5% curve bootstrap fits failed")
            lo, hi = np.quantile(pred_boot, [.025, .975], axis=0)
            ax.plot(grid, predicted, color=color, lw=2, label={"KOR":"Korean", "SPA":"Spanish", "CMN":"Mandarin"}[accent])
            ax.fill_between(grid, lo, hi, color=color, alpha=.12)
            curves.extend(dict(dataset=dataset, accent=accent, similarity=x, accuracy=y, ci_low=l, ci_high=h) for x,y,l,h in zip(grid,predicted,lo,hi))
            estimates.append(dict(dataset=dataset, accent=accent, intercept=fit.params[0], slope=fit.params[1],
                                  n_trials=len(group), n_participants=group.participant_id.nunique(),
                                  n_unique_targets=group.query_id.nunique(), bootstrap_successes=len(pred_boot),
                                  model="descriptive binomial logistic; participant bootstrap; no random effects"))
        ax.set(xlabel="Similarity to English reference productions", ylabel="Control-condition word accuracy", ylim=(0, 1), title=f"{dataset} | Before informative L2 exposure")
        ax.legend(title="Test talker L1", frameon=False); tidy(ax)
    fig.suptitle("Figure 2c | Matched-word English-reference similarity and intelligibility", fontsize=14)
    save(fig, "figure_2c_control_word_similarity")
    pd.DataFrame(estimates).to_csv(TABLES / "figure_2c_descriptive_models.csv", index=False)
    pd.DataFrame(binned).to_csv(TABLES / "figure_2c_binned_accuracy.csv", index=False)
    pd.DataFrame(curves).to_csv(TABLES / "figure_2c_logistic_curves.csv", index=False)
    pd.DataFrame(checks).to_csv(TABLES / "figure_2c_mapping_audit.csv", index=False)


def segment_input_audit():
    raw = REPO / "data/raw_data/alexander_nygaard19"
    manifest = read(REPO / "data/manifests/AN19-stimulus-manifest.csv")
    behavior = read(REPO / "data/preprocessed data/AN19-behavioral-data.csv")
    files = list(raw.rglob("*"))
    annotation_files = [p.relative_to(REPO).as_posix() for p in files if p.suffix.lower() in {".textgrid", ".eaf", ".lab", ".ctm"}]
    audit = {
        "search_scope": raw.relative_to(REPO).as_posix(), "recordings": int(len(manifest)),
        "annotation_files_found": annotation_files,
        "wav_entries_are_lfs_pointers": int(sum(p.suffix.lower() == ".wav" and p.stat().st_size < 200 for p in files if p.is_file())),
        "manifest_phone_interval_fields": [x for x in manifest.columns if "phon" in x.lower() or "segment" in x.lower()],
        "expected_ipa_nonmissing": int(behavior["response_expected.ipa"].notna().sum()),
        "provided_ipa_nonmissing": int(behavior["response_provided.ipa"].notna().sum()),
        "orthographic_responses_nonmissing": int(behavior.response_provided.notna().sum()),
        "figure_2b_status": "awaiting verified AN19 phone boundaries",
        "figure_2d_status": "awaiting phone deviations plus lexical/response alignment decisions",
        "note": "No arbitrary equal-duration phone segmentation or inferred acoustic errors were generated."
    }
    (TABLES / "an19_segment_input_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    control = behavior.loc[behavior.phase.eq("test") & behavior.exposure_test_condition_id.str.contains("Control")]
    targets = control[["item_talker", "response_expected"]].drop_duplicates()
    needed = manifest.merge(targets, left_on=["speaker_id", "word"], right_on=["item_talker", "response_expected"], how="inner")
    reference = manifest.loc[manifest.accent.eq("ENG") & manifest.word.isin(targets.response_expected)]
    pd.concat([needed[manifest.columns], reference], ignore_index=True).drop_duplicates("recording_id")[
        ["recording_id", "speaker_id", "word", "source_wav_relpath"]].to_csv(TABLES / "an19_priority_recordings_for_phone_annotation.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--reference-audio", type=Path, help="Actual X21 English example WAV if the repository holds an LFS pointer")
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    trajectories(args.reference_audio)
    languages = language_panels()
    talker_matrix(languages)
    control_curves(args.jobs, args.bootstrap)
    segment_input_audit()
    source(Path(__file__).resolve())
    result = {"source_hashes": SOURCES, "seed": SEED, "bootstrap": args.bootstrap, "jobs": args.jobs,
              "layer": LAYER, "dtw_tau": 2, "dtw_normalization": "mean_sequence_length",
              "coordinate_scaling": "none", "phoible_url": PHOIBLE_URL,
              "phoible_license": "CC BY-SA 3.0; Moran & McCloy (eds.), PHOIBLE 2.0 (2019)",
              "outputs": {p.relative_to(OUT).as_posix():sha(p) for p in [*TABLES.glob('*'), *FIGURES.glob('*')]}}
    (OUT / "provenance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Panel generation complete; 2b and 2d remain data-dependent, not filled with proxies.", flush=True)


if __name__ == "__main__":
    main()
