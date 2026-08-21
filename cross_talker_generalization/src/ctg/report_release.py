from __future__ import annotations

import concurrent.futures
import os
import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from .report_core import COLORS, LAYERS, _collect_sbi
from .report_extended import (
    DATASETS,
    DISPLAY_LAYERS,
    VARIANTS,
    _collect_notebook_ceiling_profiles,
    _fold_bootstrap95,
    _save_figure,
    _talker_sort,
    build_extended_report,
)
from .metrics import dtw_distance
from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


VARIABILITY_Z_DIR = "{dataset}-{variant}-variability-z-notebook-simplified-tau2-v1"
CEILING_Z_DIR = "{dataset}-behavioral-ceiling-notebook-folds-v1"
TALKER_H5 = {
    ("X21", "base"): "x21_hubert_full_corpus_tsne_3d.h5",
    ("X21", "ft"): "x21_hubert_full_corpus_tsne_3d_ft.h5",
    ("B23", "base"): "bradlow23_tsne_3d.h5",
    ("B23", "ft"): "bradlow23_tsne_3d_ft.h5",
}


def _association_profiles_reframed(
    repository: Path, destination: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replot the historical notebook statistic without presenting it as prediction."""

    source, ceilings = _collect_notebook_ceiling_profiles(repository)
    sns.set_theme(style="ticks", context="notebook")
    figure, axes = plt.subplots(3, 2, figsize=(19, 13), sharex=True, sharey=False)
    ordered_keys = ("mfcc39", "strf24_legacy", *LAYERS)
    positions = np.array([0, 1, *range(3, 3 + len(LAYERS))], dtype=float)
    labels = ["MFCC\n(39-D)", "STRF\n(24-D)", *[DISPLAY_LAYERS[key] for key in LAYERS]]
    jitter = np.array([-0.10, 0.0, 0.10])

    for row, dataset in enumerate(DATASETS):
        ceiling_values = ceilings.loc[ceilings["dataset_id"].eq(dataset), "z_test"].to_numpy(float)
        ceiling_mean = float(ceiling_values.mean())
        ceiling_percent = ceiling_values / ceiling_mean * 100.0
        _, ceiling_low, ceiling_high = _fold_bootstrap95(ceiling_percent)
        significance = 1.96 / ceiling_mean * 100.0
        for column, variant in enumerate(VARIANTS):
            axis = axes[row, column]
            axis.axvspan(-0.45, 1.45, color="#f2f2f2", zorder=0)
            axis.axhspan(ceiling_low, ceiling_high, color="#cfcfcf", alpha=0.8, zorder=0)
            axis.axhline(100, color="#222222", linestyle="--", linewidth=1.25, zorder=1)
            axis.axhline(significance, color="#8c8c8c", linestyle=":", linewidth=1.1, zorder=1)
            axis.axhline(-significance, color="#8c8c8c", linestyle=":", linewidth=1.1, zorder=1)

            subset = source.loc[
                source["dataset_id"].eq(dataset)
                & (source["source"].eq(variant) | source["source"].isin(["mfcc39", "strf24_legacy"]))
            ].copy()
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for position, key in zip(positions, ordered_keys):
                values = subset.loc[subset["layer"].eq(key), "percent_ceiling"].to_numpy(float)
                if len(values) != 3:
                    raise ValueError(f"{dataset} {variant} {key}: expected three folds, found {len(values)}")
                point_color = "#555555" if key in {"mfcc39", "strf24_legacy"} else COLORS[variant]
                axis.scatter(position + jitter, values, s=26, color=point_color, alpha=0.55, zorder=2)
                mean, low, high = _fold_bootstrap95(values)
                means.append(mean)
                lows.append(low)
                highs.append(high)
                axis.errorbar(
                    position,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="o",
                    color=point_color,
                    markersize=5,
                    linewidth=1.5,
                    capsize=3,
                    zorder=3,
                )
            axis.plot(positions[:2], means[:2], color="#555555", linewidth=1.35, zorder=2.5)
            axis.plot(positions[2:], means[2:], color=COLORS[variant], linewidth=1.65, zorder=2.5)
            axis.set_title(f"{dataset} - HuBERT {'base' if variant == 'base' else 'ASR fine-tuned'}")
            axis.set_ylabel("Held-out-refit Wald z / mean ceiling z (%)")
            axis.set_ylim(min(-32, min(lows) - 4), max(120, ceiling_high + 4, max(highs) + 4))
            axis.text(
                0.99,
                0.02,
                "Association statistic - not OOF accuracy or variance explained",
                transform=axis.transAxes,
                va="bottom",
                ha="right",
                fontsize=8.7,
                color="#8b1a1a",
                fontweight="bold",
            )
            sns.despine(ax=axis)

    for axis in axes[-1, :]:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=50, ha="right", fontsize=9)
        axis.set_xlabel("Feature representation")
    axes[0, 0].legend(
        handles=[
            Patch(facecolor="#f2f2f2", edgecolor="none", label="Acoustic-baseline columns"),
            Patch(facecolor="#cfcfcf", edgecolor="none", label="Ceiling fold-bootstrap 95% CI"),
            Line2D([0], [0], color="#222222", linestyle="--", label="Mean behavioral ceiling = 100%"),
            Line2D([0], [0], color="#8c8c8c", linestyle=":", label="|Wald z| = 1.96 reference"),
        ],
        fontsize=8.2,
        loc="upper left",
    )
    figure.suptitle(
        "Legacy notebook association profile relative to the 3-fold behavioral ceiling\n"
        "The percentage rescales Wald z only; it is not a percentage of human behavior predicted",
        fontsize=18,
        fontweight="bold",
        y=1.015,
    )
    figure.tight_layout()
    _save_figure(figure, destination)
    return source, ceilings


def _acoustic_baseline_audit(
    fold_z: pd.DataFrame, ceilings: pd.DataFrame, sbi: pd.DataFrame
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    feature_labels = {"mfcc39": "MFCC (39-D)", "strf24_legacy": "STRF (24-D)"}
    for dataset in DATASETS:
        ceiling = ceilings.loc[ceilings["dataset_id"].eq(dataset), "z_test"].astype(float)
        ceiling_mean = float(ceiling.mean())
        for feature, label in feature_labels.items():
            values = fold_z.loc[
                fold_z["dataset_id"].eq(dataset) & fold_z["source"].eq(feature), "z_test"
            ].astype(float)
            predictive = sbi.loc[
                sbi["dataset_id"].eq(dataset) & sbi["feature_key"].eq(feature)
            ].iloc[0]
            records.append(
                {
                    "dataset_id": dataset,
                    "baseline": label,
                    "fold_1_z": values.iloc[0],
                    "fold_2_z": values.iloc[1],
                    "fold_3_z": values.iloc[2],
                    "mean_wald_z": float(values.mean()),
                    "mean_ceiling_z": ceiling_mean,
                    "mean_z_percent_ceiling": float(values.mean() / ceiling_mean * 100.0),
                    "true_oof_log_loss_gain_vs_condition_only": float(
                        predictive["oof_gain_joint_vs_condition"]
                    ),
                    "interpretation": (
                        "Wald-z association magnitude; not accuracy, variance explained, or an OOF score"
                    ),
                }
            )
    return pd.DataFrame(records)


def _oof_hubert_profiles(sbi: pd.DataFrame, destination: Path) -> pd.DataFrame:
    """One question only: does the model improve prediction for unseen participants?"""

    source = sbi.loc[sbi["family"].eq("HuBERT t-SNE")].copy()
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), sharex=True, sharey=False)
    x = np.arange(len(LAYERS))
    for panel, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        data = source.loc[source["dataset_id"].eq(dataset)]
        for variant in VARIANTS:
            layer = data.loc[data["variant"].eq(variant)].set_index("feature_key").reindex(LAYERS)
            axis.plot(
                x,
                layer["oof_gain_joint_vs_condition"],
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                color=COLORS[variant],
                label="HuBERT base" if variant == "base" else "HuBERT ASR fine-tuned",
            )
        axis.axhline(0, color="#222222", linewidth=1.0)
        axis.set_title(dataset, fontweight="bold")
        axis.set_ylabel("" if panel else "OOF log-loss gain")
        axis.set_xticks(x)
        axis.set_xticklabels([DISPLAY_LAYERS[key] for key in LAYERS], rotation=58, ha="right", fontsize=8)
        axis.set_xlabel("HuBERT layer")
        axis.text(
            0.02,
            0.97,
            "above 0 = better prediction\nbelow 0 = worse prediction",
            transform=axis.transAxes,
            va="top",
            fontsize=8.5,
            color="#444444",
        )
        sns.despine(ax=axis)
    axes[0].legend(loc="best", fontsize=9)
    figure.suptitle(
        "True participant-held-out prediction across HuBERT layers\n"
        "Acoustic baselines are intentionally excluded here and compared separately in Figure 02",
        fontsize=15.5,
        fontweight="bold",
        y=1.035,
    )
    figure.tight_layout()
    _save_figure(figure, destination)
    return source


def _collect_variability_profiles(repository: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    statistics = repository / "results" / "statistics"
    rows: list[pd.DataFrame] = []
    ceilings: list[pd.DataFrame] = []
    for dataset in DATASETS:
        ceiling_path = statistics / CEILING_Z_DIR.format(dataset=dataset) / "behavioral_ceiling_z.csv"
        ceiling = pd.read_csv(ceiling_path)
        ceiling["z_test"] = pd.to_numeric(ceiling["z_test"], errors="raise")
        ceiling_mean = float(ceiling["z_test"].mean())
        ceiling["ceiling_mean_z"] = ceiling_mean
        ceiling["ceiling_percent"] = ceiling["z_test"] / ceiling_mean * 100.0
        ceilings.append(ceiling[["dataset_id", "fold", "z_test", "ceiling_mean_z", "ceiling_percent"]])
        for variant in VARIANTS:
            path = (
                statistics
                / VARIABILITY_Z_DIR.format(dataset=dataset, variant=variant)
                / "heldout_refit_z.csv"
            )
            frame = pd.read_csv(path)
            frame = frame.loc[frame["measure"].eq("overall")].copy()
            frame["z_test"] = pd.to_numeric(frame["z_test"], errors="raise")
            frame["percent_ceiling"] = frame["z_test"].abs() / ceiling_mean * 100.0
            frame["source_path"] = str(path.resolve())
            rows.append(frame)
    return pd.concat(rows, ignore_index=True), pd.concat(ceilings, ignore_index=True)


def _variability_ceiling_profiles(
    repository: Path, destination: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source, ceilings = _collect_variability_profiles(repository)
    sns.set_theme(style="ticks", context="notebook")
    figure, axes = plt.subplots(3, 2, figsize=(18.5, 12.5), sharex=True, sharey=False)
    x = np.arange(len(LAYERS))
    jitter = np.array([-0.09, 0.0, 0.09])
    for row, dataset in enumerate(DATASETS):
        ceiling_values = ceilings.loc[ceilings["dataset_id"].eq(dataset), "z_test"].to_numpy(float)
        ceiling_percent = ceiling_values / ceiling_values.mean() * 100.0
        _, ceiling_low, ceiling_high = _fold_bootstrap95(ceiling_percent)
        for column, variant in enumerate(VARIANTS):
            axis = axes[row, column]
            axis.axhspan(ceiling_low, ceiling_high, color="#d4d4d4", alpha=0.8, zorder=0)
            axis.axhline(100, color="#222222", linestyle="--", linewidth=1.2, zorder=1)
            data = source.loc[
                source["dataset_id"].eq(dataset) & source["model_variant"].eq(variant)
            ]
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for position, layer in zip(x, LAYERS):
                values = data.loc[data["layer"].eq(layer), "percent_ceiling"].to_numpy(float)
                if len(values) != 3:
                    raise ValueError(f"{dataset} {variant} {layer}: expected three variability folds")
                axis.scatter(position + jitter, values, s=24, color=COLORS[variant], alpha=0.42, zorder=2)
                mean, low, high = _fold_bootstrap95(values)
                means.append(mean)
                lows.append(low)
                highs.append(high)
                axis.errorbar(
                    position,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="o",
                    color=COLORS[variant],
                    markersize=4.6,
                    linewidth=1.4,
                    capsize=2.5,
                    zorder=3,
                )
            axis.plot(x, means, color=COLORS[variant], linewidth=1.6, zorder=2.5)
            axis.set_title(f"{dataset} - HuBERT {'base' if variant == 'base' else 'ASR fine-tuned'}")
            axis.set_ylabel("|Held-out-refit Wald z| / mean ceiling z (%)")
            axis.set_ylim(0, max(115, max(highs) + 5, ceiling_high + 4))
            if dataset == "B23":
                axis.text(
                    0.01,
                    0.03,
                    "Compatibility extension: identified single-talker exposure only",
                    transform=axis.transAxes,
                    fontsize=8.1,
                    color="#8b1a1a",
                )
            sns.despine(ax=axis)
    for axis in axes[-1, :]:
        axis.set_xticks(x)
        axis.set_xticklabels([DISPLAY_LAYERS[key] for key in LAYERS], rotation=55, ha="right", fontsize=8.5)
        axis.set_xlabel("HuBERT layer")
    axes[0, 0].legend(
        handles=[
            Patch(facecolor="#d4d4d4", edgecolor="none", label="Ceiling fold-bootstrap 95% CI"),
            Line2D([0], [0], color="#222222", linestyle="--", label="Mean behavioral ceiling = 100%"),
        ],
        loc="upper left",
        fontsize=8.5,
    )
    figure.suptitle(
        "Overall exposure variability across HuBERT layers\n"
        "Notebook-compatible association magnitude; three folds and behavioral-ceiling normalization",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    figure.tight_layout()
    _save_figure(figure, destination)
    return source, ceilings


def _content_key(dataset: str, unit_id: str, attributes: h5py.AttributeManager) -> str | None:
    if dataset == "X21":
        if str(attributes.get("manifest_role", "")) != "experimental_sentence":
            return None
        return str(attributes["sentence_code"])
    if dataset == "B23":
        parts = unit_id.split("-", 2)
        if len(parts) != 3:
            raise ValueError(f"cannot parse B23 unit ID: {unit_id}")
        return parts[2]
    raise ValueError(dataset)


def _load_matched_sequences(path: Path, dataset: str) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        root = handle["tr_24"]
        for speaker in sorted(root.keys(), key=_talker_sort):
            items: dict[str, np.ndarray] = {}
            for unit_id in root[speaker].keys():
                dataset_value = root[speaker][unit_id]
                key = _content_key(dataset, unit_id, dataset_value.attrs)
                if key is None:
                    continue
                if key in items:
                    raise ValueError(f"duplicate matched-content key {speaker}/{key}")
                items[key] = np.asarray(dataset_value, dtype=np.float64)
            result[speaker] = items
    expected = 32 if dataset == "X21" else 120
    counts = {speaker: len(items) for speaker, items in result.items()}
    if set(counts.values()) != {expected}:
        raise ValueError(f"{dataset}: unexpected per-talker item counts {counts}")
    common = set.intersection(*(set(items) for items in result.values()))
    if len(common) != expected:
        raise ValueError(f"{dataset}: expected {expected} globally matched items, found {len(common)}")
    return result


def _calculate_pair_distances(
    talker_a: str,
    talker_b: str,
    sequences: dict[str, dict[str, np.ndarray]],
) -> tuple[str, str, list[tuple[str, float]]]:
    keys = sorted(set(sequences[talker_a]) & set(sequences[talker_b]))
    values = [
        (
            key,
            dtw_distance(
                sequences[talker_a][key],
                sequences[talker_b][key],
                tau=2.0,
                normalization="mean_sequence_length",
            ).distance,
        )
        for key in keys
    ]
    return talker_a, talker_b, values


def _derive_small_dataset_talker_distances(
    repository: Path, dataset: str, variant: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = repository / "data" / "features" / TALKER_H5[(dataset, variant)]
    sequences = _load_matched_sequences(path, dataset)
    talkers = sorted(sequences, key=_talker_sort)
    first = talkers[0]
    first_item = next(iter(sequences[first].values()))
    dtw_distance(first_item, first_item, tau=2.0, normalization="mean_sequence_length")
    pairs = [(talkers[i], talkers[j]) for i in range(len(talkers)) for j in range(i + 1, len(talkers))]
    workers = max(1, min(len(pairs), os.cpu_count() or 1, 32))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(lambda pair: _calculate_pair_distances(pair[0], pair[1], sequences), pairs)
        )

    item_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    n_items = len(next(iter(sequences.values())))
    scope = "32 experimental sentences" if dataset == "X21" else "120 corpus sentences"
    for talker in talkers:
        summary_rows.append(
            {
                "dataset_id": dataset,
                "variant": variant,
                "layer": "tr_24",
                "talker_a_id": talker,
                "talker_b_id": talker,
                "n_matched_items": n_items,
                "mean_raw_distance": 0.0,
                "item_scope": scope,
                "tau": 2.0,
                "dtw_normalization": "mean_sequence_length",
            }
        )
    for talker_a, talker_b, values in results:
        mean_distance = float(np.mean([distance for _, distance in values]))
        for left, right in ((talker_a, talker_b), (talker_b, talker_a)):
            summary_rows.append(
                {
                    "dataset_id": dataset,
                    "variant": variant,
                    "layer": "tr_24",
                    "talker_a_id": left,
                    "talker_b_id": right,
                    "n_matched_items": len(values),
                    "mean_raw_distance": mean_distance,
                    "item_scope": scope,
                    "tau": 2.0,
                    "dtw_normalization": "mean_sequence_length",
                }
            )
        for content_id, distance in values:
            item_rows.append(
                {
                    "dataset_id": dataset,
                    "variant": variant,
                    "layer": "tr_24",
                    "talker_a_id": talker_a,
                    "talker_b_id": talker_b,
                    "content_id": content_id,
                    "raw_distance": distance,
                    "item_scope": scope,
                    "tau": 2.0,
                    "dtw_normalization": "mean_sequence_length",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(item_rows)


def _load_an19_complete_distances(repository: Path, variant: str) -> pd.DataFrame:
    source = (
        repository
        / "results"
        / "derived"
        / f"AN19-talker-validation-{'base' if variant == 'base' else 'ft'}-tr24"
        / "talker_pair_summary.csv"
    )
    pairs = pd.read_csv(source)
    talkers = sorted(set(pairs["talker_a_id"]) | set(pairs["talker_b_id"]), key=_talker_sort)
    rows: list[dict[str, object]] = []
    for talker in talkers:
        rows.append(
            {
                "dataset_id": "AN19",
                "variant": variant,
                "layer": "tr_24",
                "talker_a_id": talker,
                "talker_b_id": talker,
                "n_matched_items": 138,
                "mean_raw_distance": 0.0,
                "item_scope": "138 globally common words",
                "tau": 2.0,
                "dtw_normalization": "mean_sequence_length",
            }
        )
    for row in pairs.itertuples(index=False):
        for left, right in ((row.talker_a_id, row.talker_b_id), (row.talker_b_id, row.talker_a_id)):
            rows.append(
                {
                    "dataset_id": "AN19",
                    "variant": variant,
                    "layer": "tr_24",
                    "talker_a_id": left,
                    "talker_b_id": right,
                    "n_matched_items": int(row.n_shared_words),
                    "mean_raw_distance": float(row.raw_distance),
                    "item_scope": "138 globally common words",
                    "tau": 2.0,
                    "dtw_normalization": "mean_sequence_length",
                }
            )
    return pd.DataFrame(rows)


def _display_talker(dataset: str, value: str) -> str:
    parts = value.split(".")
    if dataset == "AN19":
        return parts[-1]
    if dataset == "X21":
        language = parts[1] if len(parts) > 1 else ""
        number = parts[-1].rsplit("_", 1)[-1]
        return f"{language}-{number}"
    if dataset == "B23":
        return parts[1] if len(parts) > 1 else value
    return value


def _group_code(value: str) -> str:
    parts = value.split(".")
    return parts[1] if len(parts) > 1 else value


def _matched_content_talker_matrices(
    repository: Path, destination: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    item_tables = []
    for variant in VARIANTS:
        summaries.append(_load_an19_complete_distances(repository, variant))
    for dataset in ("X21", "B23"):
        for variant in VARIANTS:
            summary, items = _derive_small_dataset_talker_distances(repository, dataset, variant)
            summaries.append(summary)
            item_tables.append(items)
    source = pd.concat(summaries, ignore_index=True)
    item_source = pd.concat(item_tables, ignore_index=True)

    sns.set_theme(style="ticks", context="notebook")
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(19.5, 20),
        gridspec_kw={"height_ratios": [2.9, 1.25, 0.9]},
    )
    image = None
    for row, dataset in enumerate(DATASETS):
        dataset_data = source.loc[source["dataset_id"].eq(dataset)]
        off_diagonal = dataset_data.loc[
            ~dataset_data["talker_a_id"].eq(dataset_data["talker_b_id"]), "mean_raw_distance"
        ].to_numpy(float)
        vmin, vmax = float(off_diagonal.min()), float(off_diagonal.max())
        for column, variant in enumerate(VARIANTS):
            axis = axes[row, column]
            data = dataset_data.loc[dataset_data["variant"].eq(variant)]
            talkers = sorted(data["talker_a_id"].unique(), key=_talker_sort)
            matrix = data.pivot(
                index="talker_a_id", columns="talker_b_id", values="mean_raw_distance"
            ).reindex(index=talkers, columns=talkers)
            display = matrix.to_numpy(float).copy()
            np.fill_diagonal(display, np.nan)
            color_map = plt.colormaps["viridis"].copy()
            color_map.set_bad("#eeeeee")
            image = axis.imshow(display, cmap=color_map, vmin=vmin, vmax=vmax, interpolation="nearest")
            labels = [_display_talker(dataset, talker) for talker in talkers]
            label_size = 6 if dataset == "AN19" else 8.5
            axis.set_xticks(np.arange(len(talkers)), labels=labels, rotation=90, fontsize=label_size)
            axis.set_yticks(np.arange(len(talkers)), labels=labels, fontsize=label_size)
            count = int(data.loc[~data["talker_a_id"].eq(data["talker_b_id"]), "n_matched_items"].iloc[0])
            unit = "words" if dataset == "AN19" else "sentences"
            axis.set_title(
                f"{dataset} - HuBERT {'base' if variant == 'base' else 'ASR fine-tuned'} - Tr-24\n"
                f"mean of {count} matched-{unit} DTW distances"
            )
            axis.set_xlabel("Talker B")
            axis.set_ylabel("Talker A")
            groups = [_group_code(talker) for talker in talkers]
            for boundary in [index for index in range(1, len(groups)) if groups[index] != groups[index - 1]]:
                axis.axhline(boundary - 0.5, color="white", linewidth=1.2)
                axis.axvline(boundary - 0.5, color="white", linewidth=1.2)
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
            colorbar.set_label("Mean DTW distance (lower = more similar)")
    figure.suptitle(
        "Complete all-talker matched-content distance matrices\n"
        "Each cell is the mean of DTW distances for the same word/sentence spoken by both talkers; diagonal = 0 (gray)",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    figure.subplots_adjust(left=0.07, right=0.96, bottom=0.05, top=0.94, hspace=0.30, wspace=0.22)
    _save_figure(figure, destination, dpi=280)
    _individual_talker_matrix_figures(source, destination.parent)
    return source, item_source


def _individual_talker_matrix_figures(source: pd.DataFrame, destination: Path) -> None:
    """Create presentation-friendly per-dataset versions of the complete matrices."""

    specifications = {
        "AN19": ((18.5, 9.4), "figure_05b1_an19_complete_talker_distance_matrix"),
        "X21": ((15.5, 7.0), "figure_05b2_x21_complete_talker_distance_matrix"),
        "B23": ((12.0, 5.4), "figure_05b3_b23_complete_talker_distance_matrix"),
    }
    for dataset in DATASETS:
        figure_size, file_name = specifications[dataset]
        data_set = source.loc[source["dataset_id"].eq(dataset)]
        off_diagonal = data_set.loc[
            ~data_set["talker_a_id"].eq(data_set["talker_b_id"]), "mean_raw_distance"
        ].to_numpy(float)
        vmin, vmax = float(off_diagonal.min()), float(off_diagonal.max())
        figure, axes = plt.subplots(1, 2, figsize=figure_size)
        for axis, variant in zip(axes, VARIANTS):
            data = data_set.loc[data_set["variant"].eq(variant)]
            talkers = sorted(data["talker_a_id"].unique(), key=_talker_sort)
            matrix = data.pivot(
                index="talker_a_id", columns="talker_b_id", values="mean_raw_distance"
            ).reindex(index=talkers, columns=talkers)
            display = matrix.to_numpy(float).copy()
            np.fill_diagonal(display, np.nan)
            color_map = plt.colormaps["viridis"].copy()
            color_map.set_bad("#eeeeee")
            image = axis.imshow(display, cmap=color_map, vmin=vmin, vmax=vmax, interpolation="nearest")
            labels = [_display_talker(dataset, talker) for talker in talkers]
            label_size = 6.3 if dataset == "AN19" else 9
            axis.set_xticks(np.arange(len(talkers)), labels=labels, rotation=90, fontsize=label_size)
            axis.set_yticks(np.arange(len(talkers)), labels=labels, fontsize=label_size)
            count = int(data.loc[~data["talker_a_id"].eq(data["talker_b_id"]), "n_matched_items"].iloc[0])
            unit = "words" if dataset == "AN19" else "sentences"
            axis.set_title(
                f"HuBERT {'base' if variant == 'base' else 'ASR fine-tuned'} - Tr-24\n"
                f"mean across {count} matched {unit}"
            )
            axis.set_xlabel("Talker B")
            axis.set_ylabel("Talker A")
            groups = [_group_code(talker) for talker in talkers]
            for boundary in [index for index in range(1, len(groups)) if groups[index] != groups[index - 1]]:
                axis.axhline(boundary - 0.5, color="white", linewidth=1.2)
                axis.axvline(boundary - 0.5, color="white", linewidth=1.2)
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
            colorbar.set_label("Mean DTW distance (lower = more similar)")
        figure.suptitle(
            f"{dataset}: complete all-talker matched-content distance matrix\n"
            "Diagonal = 0 (gray); color range is shared across base and fine-tuned panels",
            fontsize=15.5,
            fontweight="bold",
            y=1.01,
        )
        figure.tight_layout()
        _save_figure(figure, destination / file_name, dpi=280)


def _write_readme(output: Path, audit: pd.DataFrame) -> None:
    audit_lines = []
    for row in audit.itertuples(index=False):
        audit_lines.append(
            f"- {row.dataset_id} {row.baseline}: mean z={row.mean_wald_z:.2f}, "
            f"z/ceiling={row.mean_z_percent_ceiling:.1f}%, "
            f"true OOF log-loss gain={row.true_oof_log_loss_gain_vs_condition_only:.5g}."
        )
    text = f"""# Final experimental report and figures (2026-08-21)

This directory is the current production report package. Historical reports are recoverably archived under the repository's `recycle_bin/`.

## Four principal corrections

1. `figure_00_notebook_ceiling_normalized_profiles` reproduces the historical three-fold Wald-z/ceiling display while explicitly labeling it as a held-out-refit association statistic, not OOF predictive accuracy or variance explained.
2. `figure_01_sbi_layer_profiles` asks only whether HuBERT similarity improves prediction for unseen participants. MFCC, STRF, and a z=1.96 reference line are not mixed into this figure; acoustic OOF comparisons appear in Figure 02.
3. `figure_05b_experiment_talker_distance_matrices` contains genuine all-talker matrices: AN19 is 42×42 with 138 shared words per off-diagonal cell, X21 is 11×11 with all 32 matched experimental sentences, and B23 is 4×4 with 120 shared sentences. DTW uses `tau=2` and historical mean-sequence-length normalization.
4. The overall-only compatibility variability figures are retained, and every computable variability method plus true OOF results is included in the Figure 03 series.

## Why MFCC/STRF are high in Figure 00

The plotting code does not inflate these values. They come directly from stored GLMM `z_test` values, and the historical AN19 notebooks show the same high acoustic-baseline z values. Wald z is coefficient divided by standard error, so large samples, small standard errors, and acoustic–behavior association can all make z large. Dividing by behavioral-ceiling z only rescales z; it does not state how much human behavior is predicted. Acoustic predictive value should be assessed with participant-held-out OOF log-loss gain in Figure 02.

Audited values:

{os.linesep.join(audit_lines)}

## Figure 05b source data

- Prefer the dataset-specific `figure_05b1_an19_*`, `figure_05b2_x21_*`, and `figure_05b3_b23_*` panels for presentation. `figure_05b_experiment_*` is the six-matrix overview.
- `tables/talker_distance_tr24.csv` is the long table for all six dataset × model matrices, including zero diagonals.
- `tables/talker_distance_item_level_x21_b23.csv` contains sentence-level DTW distances for every undirected X21/B23 talker pair. AN19 word-level details remain in `results/derived/AN19-talker-validation-*/`; the report uses the validated 138-word summary rather than recomputing it.

## Variability interpretation boundary

The earlier `figure_03a_variability_ceiling_normalized_profiles` displays only `overall` and takes `abs(z_test)`. It is retained strictly as a compatibility output and is not the primary variability figure. Complete sign-preserving Figure 03 profiles and true OOF results are described below. B23 covers only identifiable actual single-talker exposure and is not equivalent in evidential status to AN19/X21.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _write_slide_outline(output: Path) -> None:
    text = """# Presentation outline (final)

1. Start with Figure 00 and state that it is a historical notebook-compatible association statistic, not predictive accuracy.
2. Follow with Figure 01: a HuBERT participant-held-out OOF layer profile in which values above zero improve prediction.
3. Use Figure 02 to compare the best OOF gains for MFCC, STRF, HuBERT base, and HuBERT ASR fine-tuned.
4. Use the Figure 03 core panels to separate compatibility three-fold variability associations from true OOF evidence.
5. Present Figure 05b and state that every matrix cell averages item-wise, matched-content DTW distances; X21 uses 11 talkers and all 32 matched experimental sentences.

Do not describe z/ceiling percentages as the percentage of human behavior explained. Call them a descriptive rescaling relative to behavioral-ceiling Wald z.
"""
    (output / "PRESENTATION_OUTLINE.md").write_text(text, encoding="utf-8")


def build_release_report(repository: str | Path, output_dir: str | Path) -> dict[str, object]:
    repository = Path(repository).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report directory: {output}")

    base_metadata = build_extended_report(repository, output)
    figures = output / "figures"
    tables = output / "tables"

    sbi = _collect_sbi(repository / "cross_talker_generalization")
    fold_z, ceiling_z = _association_profiles_reframed(
        repository, figures / "figure_00_notebook_ceiling_normalized_profiles"
    )
    atomic_write_csv(tables / "notebook_fold_z_percent_ceiling.csv", fold_z)
    atomic_write_csv(tables / "notebook_behavioral_ceiling_z.csv", ceiling_z)
    audit = _acoustic_baseline_audit(fold_z, ceiling_z, sbi)
    atomic_write_csv(tables / "figure00_acoustic_baseline_audit.csv", audit)

    oof_source = _oof_hubert_profiles(sbi, figures / "figure_01_sbi_layer_profiles")
    atomic_write_csv(tables / "figure01_hubert_true_oof_layer_profiles.csv", oof_source)

    variability_source, variability_ceiling = _variability_ceiling_profiles(
        repository, figures / "figure_03a_variability_ceiling_normalized_profiles"
    )
    atomic_write_csv(tables / "variability_overall_fold_z_percent_ceiling.csv", variability_source)
    atomic_write_csv(tables / "variability_behavioral_ceiling_z.csv", variability_ceiling)
    for suffix in (".png", ".svg"):
        original = figures / f"figure_03_hve_overall_profiles{suffix}"
        copied = figures / f"figure_03b_variability_true_oof_profiles{suffix}"
        shutil.copy2(original, copied)

    talker_summary, talker_items = _matched_content_talker_matrices(
        repository, figures / "figure_05b_experiment_talker_distance_matrices"
    )
    atomic_write_csv(tables / "talker_distance_tr24.csv", talker_summary)
    atomic_write_csv(tables / "talker_distance_item_level_x21_b23.csv", talker_items)

    _write_readme(output, audit)
    _write_slide_outline(output)

    hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "provenance.json":
            hashes[path.relative_to(output).as_posix()] = sha256_file(path)
    metadata = {
        **runtime_record(),
        "status": "complete",
        "stage": "report_release",
        "parent_stage": base_metadata.get("stage"),
        "primary_representation": "3-D t-SNE",
        "dtw_tau": 2.0,
        "dtw_normalization": "mean_sequence_length",
        "x21_talker_matrix": "11 talkers x 32 globally matched experimental sentences",
        "an19_talker_matrix": "42 talkers x 138 globally common words",
        "b23_talker_matrix": "4 talkers x 120 globally matched corpus sentences",
        "figure00_semantics": "held-out-refit Wald-z association rescaled by mean ceiling z; not OOF prediction",
        "figure01_semantics": "participant-held-out frozen-model OOF log-loss gain; HuBERT only",
        "figure03a_semantics": "mean(abs(held-out-refit variability z))/mean ceiling z; notebook-compatible association",
        "compatibility_models_rerun_for_release": False,
        "files_sha256": hashes,
    }
    atomic_write_json(output / "provenance.json", metadata)
    return metadata
