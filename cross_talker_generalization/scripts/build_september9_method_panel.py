"""Draw the revised Figure 1a-b from existing HuBERT arrays and annotations.

No model extraction, dimensionality reduction, realignment, or speech recognition
is performed. Dependencies: numpy, matplotlib, scipy, h5py, PyMuPDF.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cross_talker_generalization/analysis_update_2026-09-09"
KEY = "X21.ENG.ENG_M_055.HT1_S002"
LAYER = "tr_24"
COLORS = {"W": "#CC6677", "AY1": "#4477AA", "F": "#228833"}
LABELS = {"W": "/w/", "AY1": "/aɪ/", "F": "/f/"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_indices(interval, frames, duration):
    """Identical rounding and endpoint clipping to ctg.features.slice_by_time."""
    start = max(0, min(round(frames * interval["start_seconds"] / duration), frames - 1))
    end = max(start + 1, min(round(frames * interval["end_seconds"] / duration), frames))
    return start, end


def axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#777777")
    ax.tick_params(labelsize=10, color="#777777")


def space_style(ax, points):
    ax.set_facecolor("white")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")
        axis.line.set_color("#888888")
    bounds = np.ptp(points, axis=0)
    for setter, low, high in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), points.min(0), points.max(0)):
        pad = max((high - low) * .1, .01)
        setter(low - pad, high + pad)
    ax.set_box_aspect(np.maximum(bounds, .01))
    ax.tick_params(labelsize=8, pad=0)
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", zlabel="t-SNE 3")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_fontsize(10)
        axis.labelpad = 3
        axis.set_major_locator(MaxNLocator(nbins=3))
    ax.view_init(elev=24, azim=-58)


def main():
    figures = OUT / "figures"
    tables = OUT / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data/manifests/X21-stimulus-manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(row for row in rows if row["segment_id"] == KEY)
    assert row["sentence_normalized"] == "THE WIFE HELPED HER HUSBAND"
    sentence1 = [r for r in rows if r["accent"] == "ENG" and r["sentence_code"] == "HT1_S001"]
    # Depending on manifest version sentence_code can omit the half prefix.
    if not sentence1:
        sentence1 = [r for r in rows if r["accent"] == "ENG" and r["segment_id"].endswith("HT1_S001")]
    assert len(sentence1) == 5
    duration = float(row["duration_seconds"])
    reduced_path = ROOT / "data/features/x21_hubert_full_corpus_tsne_3d.h5"
    full_path = ROOT / "data/features/x21_hubert_full_corpus_features.h5"
    with h5py.File(reduced_path, "r") as store:
        reduced = np.asarray(store[LAYER][row["speaker_id"]][KEY], dtype=float)
        tsne_parameters = json.loads(store.attrs["tsne_parameters_json"])
        model = str(store.attrs["model_id"])
        revision = str(store.attrs["model_revision"])
        scope = str(store.attrs["corpus_scope"])
    with h5py.File(full_path, "r") as store:
        full = np.asarray(store[row["speaker_id"]][KEY][LAYER], dtype=float)
    assert full.shape == (len(reduced), 1024)
    assert np.isfinite(full).all() and np.isfinite(reduced).all()
    original_audio = ROOT / row["source_wav_relpath"]
    cached_audio = ROOT / "cross_talker_generalization/analysis_update_2026-09-06/collaborator_report/sources/ALL_055_M_ENG_ENG_HT1.wav"
    audio_path = original_audio
    if original_audio.stat().st_size < 1024:
        pointer = original_audio.read_text(encoding="utf-8")
        expected_sha = re.search(r"oid sha256:([0-9a-f]+)", pointer).group(1)
        assert sha(cached_audio) == expected_sha, "Cached audio does not match original LFS object"
        audio_path = cached_audio
    rate, audio = wavfile.read(audio_path)
    if audio.ndim == 2:
        audio = audio.astype(float).mean(axis=1)
    audio = audio[round(float(row["start_seconds"]) * rate):round(float(row["end_seconds"]) * rate)].astype(float)
    audio /= max(1, np.abs(audio).max())
    words = json.loads(row["word_intervals_json"])
    word = next(w for w in words if w["normalized"] == "WIFE")
    phones = [p for p in json.loads(row["phone_intervals_json"])
              if p["start_seconds"] >= word["start_seconds"] - 1e-6 and p["end_seconds"] <= word["end_seconds"] + 1e-6]
    assert [p["normalized"] for p in phones] == ["W", "AY1", "F"]
    word_start, word_end = interval_indices(word, len(reduced), duration)
    word_points = reduced[word_start:word_end]
    phone_ranges = [interval_indices(p, len(reduced), duration) for p in phones]
    assert min(end - start for start, end in phone_ranges) >= 3

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(19, 5.4), facecolor="white")
    fig.text(.015, .94, "a", fontsize=20, weight="bold")
    fig.text(.046, .947, "Speech to latent representation", fontsize=16, weight="bold")
    fig.text(.735, .94, "b", fontsize=20, weight="bold")
    fig.text(.769, .947, 'Segments in the word "wife"', fontsize=16, weight="bold")
    ax_audio = fig.add_axes([.045, .28, .18, .46])
    ax_audio.plot(np.arange(len(audio)) / rate, audio, color="#335577", linewidth=.65)
    ax_audio.set(xlim=(0, duration), ylim=(-1.05, 1.05), xlabel="Time (s)", ylabel="Relative amplitude")
    ax_audio.set_title("Recording", fontsize=13, pad=18)
    axes_style(ax_audio)
    fig.text(.135, .12, '"The wife helped her husband"', ha="center", fontsize=11)
    fig.text(.135, .071, "L1-English, male, talker 055", ha="center", fontsize=10, color="#555555")
    time = np.arange(len(full)) * duration / len(full)
    vmin, vmax = full[:, [0, 1023]].min(), full[:, [0, 1023]].max()
    margin = (vmax - vmin) * .12
    ax_first = fig.add_axes([.307, .575, .135, .145])
    ax_last = fig.add_axes([.307, .28, .135, .145])
    for ax, dimension in ((ax_first, 0), (ax_last, 1023)):
        ax.plot(time, full[:, dimension], color="#335577", linewidth=1.5)
        ax.set(xlim=(0, duration), ylim=(vmin - margin, vmax + margin), yticks=[])
        ax.set_title(f"Dimension {dimension + 1}", fontsize=11, loc="left", pad=5)
        axes_style(ax)
    ax_first.set_xticklabels([])
    ax_last.set_xlabel("Time (s)")
    fig.text(.3745, .507, "\u22ee", ha="center", va="center", fontsize=22)
    fig.text(.3745, .819, "HuBERT, Tr-24", ha="center", fontsize=13)
    fig.text(.3745, .12, "1,024 latent dimensions", ha="center", fontsize=11)
    fig.text(.3745, .071, "Raw activations; first and last shown", ha="center", fontsize=10, color="#555555")
    for start, end in ((.237, .29), (.454, .49)):
        fig.add_artist(FancyArrowPatch((start, .51), (end, .51), transform=fig.transFigure,
                                      arrowstyle="-|>", mutation_scale=16, color="#335577", linewidth=1.5))
    sentence_ax = fig.add_axes([.496, .23, .211, .59], projection="3d")
    sentence_ax.plot(*reduced.T, color="#335577", linewidth=1.8)
    space_style(sentence_ax, reduced)
    fig.text(.6015, .819, "Corpus-level t-SNE", ha="center", fontsize=13)
    fig.text(.6015, .12, "One 3-D vector per frame", ha="center", fontsize=11)
    fig.text(.6015, .071, "Lines join adjacent frames", ha="center", fontsize=10, color="#555555")
    word_ax = fig.add_axes([.754, .23, .23, .59], projection="3d")
    phone_assignment = {}
    for phone, (start, end) in zip(phones, phone_ranges):
        for index in range(start, end):
            phone_assignment[index] = phone["normalized"]
    segments = np.stack((word_points[:-1], word_points[1:]), axis=1)
    segment_colors = [COLORS[phone_assignment[index]] for index in range(word_start, word_end - 1)]
    word_ax.add_collection3d(Line3DCollection(segments, colors=segment_colors, linewidths=3))
    for phone, (start, end) in zip(phones, phone_ranges):
        word_ax.scatter(*reduced[start:end].T, color=COLORS[phone["normalized"]], s=15, depthshade=False)
    space_style(word_ax, word_points)
    # Phone names are in their own strip, never floating over a trajectory.
    for x, phone in zip((.80, .865, .93), phones):
        fig.text(x, .135, LABELS[phone["normalized"]], color=COLORS[phone["normalized"]], ha="center", fontsize=15, weight="bold")
    fig.text(.865, .071, "Same coordinates; zoomed to the word", ha="center", fontsize=10, color="#555555")
    for extension in ("png", "svg", "pdf"):
        kwargs = {"metadata": {"Author": "", "CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
        fig.savefig(figures / f"figure_1ab_method.{extension}", dpi=220, facecolor="white", **kwargs)
    plt.close(fig)
    phone_metadata = [{**phone, "first_frame_zero_based": start, "end_frame_exclusive": end, "frame_count": end - start,
                       "display_label": LABELS[phone["normalized"]]} for phone, (start, end) in zip(phones, phone_ranges)]
    textgrid = ROOT / row["source_textgrid_relpath"]
    metadata = {
        "segment_id": KEY, "speaker": row["speaker_id"], "L1": row["source_l1_language"],
        "sentence": row["sentence_normalized"], "word": "WIFE", "word_interval": word, "phones": phone_metadata,
        "changed_example": "The first example was replaced, not relabeled. All five native HT1_S001 manifest transcripts say A BOY FELL FROM A WINDOW; this illustration uses the annotated HT1_S002 instead.",
        "sentence1_manifest_audit": [{"segment_id": r["segment_id"], "sentence": r["sentence_normalized"]} for r in sentence1],
        "audio_verification": "Existing audio checked against original Git LFS SHA-256. No claim of manual listening or new ASR verification.",
        "audio_file": audio_path.relative_to(ROOT).as_posix(), "audio_sha256": sha(audio_path),
        "manifest_file": manifest_path.relative_to(ROOT).as_posix(), "manifest_sha256": sha(manifest_path),
        "textgrid_file": textgrid.relative_to(ROOT).as_posix(), "textgrid_sha256": sha(textgrid),
        "layer": LAYER, "checkpoint": model, "checkpoint_revision": revision, "frames": len(reduced),
        "dimensions": full.shape[1], "shown_dimensions_one_based": [1, 1024],
        "tsne_parameters": tsne_parameters, "tsne_corpus_scope": scope,
        "full_leaf": f"{row['speaker_id']}/{KEY}/{LAYER}",
        "reduced_leaf": f"{LAYER}/{row['speaker_id']}/{KEY}",
        "full_selected_array_float64_sha256": hashlib.sha256(full.tobytes()).hexdigest(),
        "reduced_selected_array_float64_sha256": hashlib.sha256(reduced.tobytes()).hexdigest(),
        "display_scaling": "No coordinate or activation standardization. Audio normalized by its maximum absolute amplitude for display only. Panel b zooms its axes without changing coordinates.",
        "timing": "Manifest-relative boundaries converted to frames with existing round(frames*time/duration) rule; no new alignment.",
        "connection_rule": "Only adjacent frames within this sentence/word are joined. A word edge crossing a phone boundary uses the preceding frame's phone color; no black linking segments.",
        "output_sha256": {p.name: sha(p) for p in figures.glob("figure_1ab_method.*")},
    }
    (tables / "figure_1ab_method_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"segment": KEY, "frames": len(reduced), "phone_frames": [p["frame_count"] for p in phone_metadata]}, indent=2))


if __name__ == "__main__":
    main()
