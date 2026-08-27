from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns

from .report_core import (
    COLORS,
    LAYERS,
    _b23_hve,
    _ceiling_figure,
    _collect_all_hve,
    _collect_hve,
    _collect_sbi,
    _collect_sbi_exp,
    _correlation_composite,
    _diagnostics,
    _hve_heatmaps,
    _sbi_profiles,
    _select_by_predictor_oof,
)
from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


DATASETS = ("AN19", "X21", "B23")
VARIANTS = ("base", "ft")
DISPLAY_LAYERS = {
    **{f"cnn_{number}": f"CNN-{number}" for number in range(2, 7)},
    **{f"tr_{number}": f"Tr-{number}" for number in range(0, 25, 2)},
}
MODEL_ORDER = ("MFCC (39-D)", "STRF (24-D)", "HuBERT base", "HuBERT FT")
MODEL_COLORS = {
    "MFCC (39-D)": "#7f7f7f",
    "STRF (24-D)": "#9467bd",
    "HuBERT base": "#1f77b4",
    "HuBERT FT": "#d62728",
}


NOTEBOOK_Z_DIRS = {
    ("AN19", "base"): "AN19-base-original-structure-z-notebook-folds-v2",
    ("AN19", "ft"): "AN19-ft-original-structure-z-notebook-folds-v2",
    ("AN19", "mfcc39"): "AN19-mfcc39-original-structure-z-notebook-folds-v2",
    ("AN19", "strf24_legacy"): "AN19-strf24-original-structure-z-notebook-folds-v2",
    ("X21", "base"): "X21-base-original-structure-z-legacy-axis-z-v5",
    ("X21", "ft"): "X21-ft-original-structure-z-legacy-axis-z-v5",
    ("X21", "mfcc39"): "X21-mfcc39-original-structure-z-legacy-axis-z-v5",
    ("X21", "strf24_legacy"): "X21-strf24-original-structure-z-legacy-axis-z-v5",
    ("B23", "base"): "B23-base-original-structure-z-notebook-folds-v4",
    ("B23", "ft"): "B23-ft-original-structure-z-notebook-folds-v4",
    ("B23", "mfcc39"): "B23-mfcc39-original-structure-z-notebook-folds-v4",
    ("B23", "strf24_legacy"): "B23-strf24-original-structure-z-notebook-folds-v4",
}
CEILING_Z_DIRS = {
    dataset: f"{dataset}-behavioral-ceiling-notebook-folds-v1" for dataset in DATASETS
}


def _save_figure(figure: plt.Figure, destination: Path, *, dpi: int = 280) -> None:
    figure.savefig(destination.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def _fold_bootstrap95(values: np.ndarray) -> tuple[float, float, float]:
    """Exact nonparametric bootstrap interval for the three displayed folds."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    indices = np.indices((len(values),) * len(values)).reshape(len(values), -1).T
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, float(low), float(high)


def _collect_notebook_ceiling_profiles(repository: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    ceilings: list[pd.DataFrame] = []
    statistics = repository / "results" / "statistics"
    for dataset in DATASETS:
        ceiling_path = statistics / CEILING_Z_DIRS[dataset] / "behavioral_ceiling_z.csv"
        ceiling = pd.read_csv(ceiling_path)
        ceiling["fold"] = pd.to_numeric(ceiling["fold"], errors="raise").astype(int)
        ceiling["z_test"] = pd.to_numeric(ceiling["z_test"], errors="raise")
        ceiling_mean = float(ceiling["z_test"].mean())
        ceiling["ceiling_mean_z"] = ceiling_mean
        ceiling["ceiling_percent"] = ceiling["z_test"] / ceiling_mean * 100.0
        ceilings.append(ceiling[["dataset_id", "fold", "z_test", "ceiling_mean_z", "ceiling_percent"]])
        for source in ("base", "ft", "mfcc39", "strf24_legacy"):
            path = statistics / NOTEBOOK_Z_DIRS[(dataset, source)] / "heldout_refit_z.csv"
            frame = pd.read_csv(path)
            frame = frame.loc[frame["type"].astype(str).eq("corrected")].copy()
            if source in {"mfcc39", "strf24_legacy"}:
                frame["layer"] = source
            frame["source"] = source
            frame["percent_ceiling"] = pd.to_numeric(frame["z_test"], errors="raise") / ceiling_mean * 100.0
            frame["ceiling_mean_z"] = ceiling_mean
            frame["source_path"] = str(path.resolve())
            rows.append(
                frame[[
                    "dataset_id", "model_variant", "representation", "layer", "fold", "type",
                    "source", "z_test", "percent_ceiling", "ceiling_mean_z", "source_path",
                ]]
            )
    return pd.concat(rows, ignore_index=True), pd.concat(ceilings, ignore_index=True)


def _ceiling_normalized_profiles(
    repository: Path, destination: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source, ceilings = _collect_notebook_ceiling_profiles(repository)
    sns.set_theme(style="ticks", context="notebook")
    figure, axes = plt.subplots(3, 2, figsize=(19, 13), sharex=True, sharey=False)
    ordered_keys = ("mfcc39", "strf24_legacy", *LAYERS)
    positions = np.array([0, 1, *range(3, 3 + len(LAYERS))], dtype=float)
    labels = ["MFCC\n(39-D)", "STRF\n(24-D)", *[DISPLAY_LAYERS[key] for key in LAYERS]]
    jitter = np.array([-0.10, 0.0, 0.10])
    for row, dataset in enumerate(DATASETS):
        ceiling_values = ceilings.loc[ceilings["dataset_id"].eq(dataset), "z_test"].to_numpy(float)
        ceiling_mean = float(np.mean(ceiling_values))
        ceiling_percent = ceiling_values / ceiling_mean * 100.0
        _, ceiling_low, ceiling_high = _fold_bootstrap95(ceiling_percent)
        significance = 1.96 / ceiling_mean * 100.0
        for column, variant in enumerate(VARIANTS):
            axis = axes[row, column]
            axis.axhspan(ceiling_low, ceiling_high, color="#d9d9d9", alpha=0.75, zorder=0)
            axis.axhline(100, color="#222222", linestyle="--", linewidth=1.2, zorder=1)
            axis.axhline(significance, color="#888888", linestyle=":", linewidth=1.2, zorder=1)
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
                    raise ValueError(f"{dataset} {variant} {key}: expected three fold values, found {len(values)}")
                axis.scatter(position + jitter, values, s=25, color="#8c8c8c", alpha=0.65, zorder=2)
                mean, low, high = _fold_bootstrap95(values)
                means.append(mean)
                lows.append(low)
                highs.append(high)
                axis.errorbar(
                    position, mean, yerr=[[mean - low], [high - mean]], fmt="o", color="black",
                    markersize=4.8, linewidth=1.4, capsize=3, zorder=3,
                )
            axis.plot(positions[:2], means[:2], color="black", linewidth=1.4, zorder=2.5)
            axis.plot(positions[2:], means[2:], color="black", linewidth=1.6, zorder=2.5)
            axis.set_title(f"{dataset} - HuBERT {'base' if variant == 'base' else 'ASR fine-tuned'}")
            axis.set_ylabel("Fold Wald z / mean ceiling z (%)")
            axis.set_ylim(min(-32, min(lows) - 4), max(120, ceiling_high + 4, max(highs) + 4))
            axis.text(
                0.01, 0.97, "dots: folds; point/error bar: mean and fold-bootstrap 95% CI",
                transform=axis.transAxes, va="top", ha="left", fontsize=8.5, color="#555555",
            )
            sns.despine(ax=axis)
    for axis in axes[-1, :]:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=50, ha="right", fontsize=9)
        axis.set_xlabel("Feature representation")
    figure.suptitle(
        "Notebook-compatible association strength relative to the 3-fold behavioral ceiling\n"
        "Acoustic baselines are shown first; gray band is the ceiling's fold-bootstrap 95% CI",
        fontsize=18, fontweight="bold", y=1.01,
    )
    figure.tight_layout()
    _save_figure(figure, destination)
    return source, ceilings


def _best_gain_source(sbi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        data = sbi.loc[sbi["dataset_id"].eq(dataset)]
        selectors = {
            "MFCC (39-D)": data["feature_key"].eq("mfcc39"),
            "STRF (24-D)": data["feature_key"].eq("strf24_legacy"),
            "HuBERT base": data["family"].eq("HuBERT t-SNE") & data["variant"].eq("base"),
            "HuBERT FT": data["family"].eq("HuBERT t-SNE") & data["variant"].eq("ft"),
        }
        for model in MODEL_ORDER:
            selected = _select_by_predictor_oof(data.loc[selectors[model]])
            rows.append({
                "dataset_id": dataset,
                "model": model,
                "feature_key": selected["feature_key"],
                "oof_gain": selected["oof_gain_joint_vs_condition"],
                "z_value": selected["z_value"],
                "p_value": selected["p_value"],
                "predictor_oof_total_log_loss": selected["predictor_oof_total_log_loss"],
                "predictor_oof_total_trials": selected["predictor_oof_total_trials"],
            })
    return pd.DataFrame(rows)


def _best_gain_facets(sbi: pd.DataFrame, destination: Path) -> pd.DataFrame:
    source = _best_gain_source(sbi)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for axis, dataset in zip(axes, DATASETS):
        data = source.loc[source["dataset_id"].eq(dataset)].set_index("model").reindex(MODEL_ORDER).reset_index()
        values = data["oof_gain"].to_numpy(float)
        x = np.arange(len(data))
        axis.bar(x, values, color=[MODEL_COLORS[model] for model in data["model"]], width=0.72)
        axis.axhline(0, color="black", linewidth=0.9)
        axis.set_title(dataset, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(["MFCC", "STRF", "HuBERT\nbase", "HuBERT\nFT"], rotation=0)
        span = max(float(np.max(np.abs(values))), 1e-6)
        axis.set_ylim(min(float(np.min(values)) - span * 0.22, -span * 0.08), float(np.max(values)) + span * 0.28)
        for position, row in enumerate(data.itertuples(index=False)):
            value_label = f"{row.oof_gain:.5f}" if abs(row.oof_gain) >= 0.001 else f"{row.oof_gain:.2e}"
            layer_label = f"\n{DISPLAY_LAYERS.get(row.feature_key, '')}" if row.feature_key in DISPLAY_LAYERS else ""
            offset = span * 0.035
            axis.text(
                position, row.oof_gain + (offset if row.oof_gain >= 0 else -offset), value_label + layer_label,
                ha="center", va="bottom" if row.oof_gain >= 0 else "top", fontsize=8.5,
            )
        sns.despine(ax=axis)
    axes[0].set_ylabel("OOF log-loss gain\nvs condition-only")
    figure.suptitle(
        "Predictive gain within each dataset\nIndependent y-axis scales reveal X21 and B23 results",
        fontsize=16, fontweight="bold", y=1.04,
    )
    figure.tight_layout()
    _save_figure(figure, destination)
    return source


def _clear_hve_profiles(hve: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 8.5), sharex=True)
    x = np.arange(len(LAYERS))
    for row, dataset in enumerate(("AN19", "X21")):
        for variant in VARIANTS:
            subset = hve.loc[hve["dataset_id"].eq(dataset) & hve["variant"].eq(variant)].copy()
            subset["layer"] = subset["feature_key"].str.split("::").str[0]
            subset = subset.set_index("layer").reindex(LAYERS)
            label = "HuBERT base" if variant == "base" else "HuBERT ASR fine-tuned"
            axes[row, 0].plot(x, subset["z_value"], marker="o", markersize=4, color=COLORS[variant], label=label)
            axes[row, 1].plot(
                x, subset["oof_gain_joint_vs_condition"], marker="o", markersize=4,
                color=COLORS[variant], label=label,
            )
        axes[row, 0].axhline(0, color="black", linewidth=0.8)
        axes[row, 0].axhline(1.96, color="0.45", linewidth=0.8, linestyle="--")
        axes[row, 0].axhline(-1.96, color="0.45", linewidth=0.8, linestyle="--")
        axes[row, 1].axhline(0, color="black", linewidth=0.8)
        axes[row, 0].set_ylabel(f"{dataset}\nWald z for overall HVE")
        axes[row, 1].set_ylabel(f"{dataset}\nOOF log-loss gain")
        axes[row, 0].legend(fontsize=9, loc="best")
        axes[row, 0].text(0.01, 0.04, "positive z: more variability -> higher accuracy", transform=axes[row, 0].transAxes, fontsize=8.5)
        axes[row, 1].text(0.01, 0.04, "positive gain: improves held-out prediction", transform=axes[row, 1].transAxes, fontsize=8.5)
    axes[0, 0].set_title("Association after controlling exposure-test condition")
    axes[0, 1].set_title("Incremental participant-held-out prediction")
    for axis in axes[-1, :]:
        axis.set_xticks(x)
        axis.set_xticklabels([DISPLAY_LAYERS[key] for key in LAYERS], rotation=55, ha="right", fontsize=8.5)
        axis.set_xlabel("HuBERT layer")
    figure.suptitle(
        "Overall actual-exposure variability (HVE)\nB23 is excluded because HVE is confounded with condition",
        fontsize=17, fontweight="bold", y=1.01,
    )
    figure.tight_layout()
    _save_figure(figure, destination)


def _short_talker(value: str) -> str:
    parts = str(value).split(".")
    if len(parts) >= 3:
        return f"{parts[-2]}:{parts[-1]}"
    return str(value)


def _talker_sort(value: str) -> tuple[int, str]:
    text = str(value)
    group_order = {
        "ENG": 0, "KOR": 1, "SPA": 2, "MIX": 3, "CMN": 0,
        "BRP": 0, "FAR": 1, "TUR": 2,
    }
    parts = text.split(".")
    group = parts[1] if len(parts) > 1 else text
    return group_order.get(group, 9), text


def _talker_distance_matrices(repository: Path, destination: Path) -> pd.DataFrame:
    derived = repository / "cross_talker_generalization" / "artifacts" / "derived"
    frames = []
    for dataset in DATASETS:
        pairs = pd.read_csv(derived / f"{dataset}-pairs" / "pairs.csv")
        for variant in VARIANTS:
            distances = pd.read_csv(
                derived / f"{dataset}-{dataset}_hubert_{variant}_tsne-confirmatory-distances.csv",
                usecols=["pair_id", "feature_key", "raw_distance"],
            )
            distances = distances.loc[distances["feature_key"].eq("tr_24")]
            merged = pairs[["pair_id", "test_speaker_id", "source_speaker_id"]].merge(
                distances, on="pair_id", how="inner", validate="one_to_one"
            )
            summary = merged.groupby(["test_speaker_id", "source_speaker_id"], as_index=False).agg(
                mean_raw_distance=("raw_distance", "mean"),
                n_content_pairs=("pair_id", "nunique"),
            )
            summary.insert(0, "variant", variant)
            summary.insert(0, "dataset_id", dataset)
            frames.append(summary)
    source = pd.concat(frames, ignore_index=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(3, 2, figsize=(18, 19), gridspec_kw={"height_ratios": [2.7, 1.2, 1.0]})
    for row, dataset in enumerate(DATASETS):
        dataset_data = source.loc[source["dataset_id"].eq(dataset)]
        vmin = float(dataset_data["mean_raw_distance"].min())
        vmax = float(dataset_data["mean_raw_distance"].max())
        for column, variant in enumerate(VARIANTS):
            axis = axes[row, column]
            data = dataset_data.loc[dataset_data["variant"].eq(variant)]
            tests = sorted(data["test_speaker_id"].unique(), key=_talker_sort)
            sources = sorted(data["source_speaker_id"].unique(), key=_talker_sort)
            matrix = data.pivot(index="test_speaker_id", columns="source_speaker_id", values="mean_raw_distance").reindex(
                index=tests, columns=sources
            )
            sns.heatmap(
                matrix, ax=axis, cmap="magma_r", vmin=vmin, vmax=vmax,
                annot=dataset == "B23", fmt=".2f", linewidths=0.25,
                cbar=column == 1, cbar_kws={"label": "Mean raw DTW distance"},
            )
            axis.set_title(f"{dataset} - {'base' if variant == 'base' else 'ASR fine-tuned'} - Tr-24")
            axis.set_xlabel("Exposure/reference talker")
            axis.set_ylabel("Test talker")
            axis.set_xticklabels([_short_talker(value.get_text()) for value in axis.get_xticklabels()], rotation=70, ha="right", fontsize=7.5)
            axis.set_yticklabels([_short_talker(value.get_text()) for value in axis.get_yticklabels()], rotation=0, fontsize=7.5)
    figure.suptitle(
        "Talker-to-talker distance in the experimental same-content comparisons\n"
        "Rows are test talkers; columns are exposure/reference talkers; color scales are shared within dataset",
        fontsize=17, fontweight="bold", y=1.005,
    )
    figure.tight_layout()
    _save_figure(figure, destination, dpi=260)
    return source


def _an19_complete_talker_similarity(repository: Path, destination: Path) -> None:
    matrices = []
    for variant, directory in (
        ("base", "AN19-base-tr24-talker-similarity-heatmap-log-v3"),
        ("ft", "AN19-ft-tr24-talker-similarity-heatmap-log-v1"),
    ):
        path = repository / "results" / "figures" / directory / "similarity_matrix.csv"
        matrix = pd.read_csv(path).set_index("talker_id")
        matrices.append((variant, matrix))
    off_diagonal = np.concatenate([
        matrix.to_numpy(float)[~np.eye(len(matrix), dtype=bool)] for _, matrix in matrices
    ])
    norm = LogNorm(vmin=float(off_diagonal.min()), vmax=float(off_diagonal.max()))
    sns.set_theme(style="ticks", context="notebook")
    figure, axes = plt.subplots(1, 2, figsize=(19, 9.5))
    for axis, (variant, matrix) in zip(axes, matrices):
        display = matrix.to_numpy(float).copy()
        np.fill_diagonal(display, float(off_diagonal.max()))
        image = axis.imshow(display, cmap="inferno", norm=norm, interpolation="nearest")
        labels = [value.rsplit(".", 1)[-1] for value in matrix.index]
        axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=90, fontsize=6)
        axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=6)
        axis.set_title(f"HuBERT {'base' if variant == 'base' else 'ASR fine-tuned'} - Tr-24")
        axis.set_xlabel("Talker")
        axis.set_ylabel("Talker")
        for boundary in (6, 18, 30):
            axis.axhline(boundary - 0.5, color="white", linewidth=1.0)
            axis.axvline(boundary - 0.5, color="white", linewidth=1.0)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Mean same-word similarity exp(-distance), shared log scale")
    figure.suptitle(
        "AN19 complete-corpus same-word similarity among 42 talkers\n"
        "Groups: L1-English, L1-Spanish, L1-Korean, mixed-L1",
        fontsize=17, fontweight="bold",
    )
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.10, top=0.88, wspace=0.20)
    _save_figure(figure, destination, dpi=280)


def _clear_transform_sensitivity(primary: pd.DataFrame, exp: pd.DataFrame, destination: Path) -> pd.DataFrame:
    raw = primary.loc[primary["family"].eq("HuBERT t-SNE")].copy()
    raw["transform"] = "Primary: standardized negative DTW distance"
    historical = exp.loc[exp["family"].eq("HuBERT t-SNE exp(k=1)")].copy()
    historical["transform"] = "Sensitivity: exp(-DTW distance), fixed k=1"
    source = pd.concat([raw, historical], ignore_index=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=True)
    x = np.arange(len(LAYERS))
    definitions = (
        ("Primary: standardized negative DTW distance", "#1f77b4", "o"),
        ("Sensitivity: exp(-DTW distance), fixed k=1", "#e6550d", "s"),
    )
    for row, dataset in enumerate(DATASETS):
        for column, variant in enumerate(VARIANTS):
            axis = axes[row, column]
            data = source.loc[source["dataset_id"].eq(dataset) & source["variant"].eq(variant)]
            for definition, color, marker in definitions:
                values = data.loc[data["transform"].eq(definition)].set_index("feature_key").reindex(LAYERS)
                axis.plot(
                    x, values["oof_gain_joint_vs_condition"], color=color, marker=marker,
                    markersize=4, linewidth=1.6, label=definition,
                )
                invalid = ~values["numerically_valid"].eq(True) & values["oof_gain_joint_vs_condition"].notna()
                axis.scatter(
                    x[invalid], values.loc[invalid, "oof_gain_joint_vs_condition"], marker="x",
                    s=70, linewidths=2, color="black", zorder=5,
                    label="Unstable fit - do not interpret" if definition.startswith("Sensitivity") else None,
                )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"{dataset} - {'base' if variant == 'base' else 'ASR fine-tuned'}")
            axis.set_ylabel("OOF log-loss gain\n(positive = better)")
            if row == 0 and column == 0:
                axis.legend(fontsize=8.5, loc="best")
    for axis in axes[-1, :]:
        axis.set_xticks(x)
        axis.set_xticklabels([DISPLAY_LAYERS[key] for key in LAYERS], rotation=55, ha="right", fontsize=8.5)
        axis.set_xlabel("HuBERT layer")
    figure.suptitle(
        "Sensitivity to the distance-to-similarity transform\n"
        "This tests robustness of the predictor definition; it is not a comparison of base vs fine-tuned HuBERT",
        fontsize=17, fontweight="bold", y=1.01,
    )
    figure.tight_layout()
    _save_figure(figure, destination)
    return source


def _copy_notebook_scurves(repository: Path, figures: Path, tables: Path) -> pd.DataFrame:
    destination = figures / "s_curves_tr24"
    destination.mkdir(parents=True, exist_ok=True)
    table_destination = tables / "s_curve_trials_tr24"
    table_destination.mkdir(parents=True, exist_ok=True)
    records = []
    specifications = []
    for variant in VARIANTS:
        specifications.extend([
            (
                "AN19", variant, f"AN19-{variant}-tr24-notebook-s-curves-notebook-folds-v2",
                ("an19_s_curves_24_test_talkers", "an19_s_curves_24_test_talkers_pooled",
                 "an19_s_curves_by_test_accent", "an19_s_curves_by_test_accent_pooled"),
                "s_curve_trials.csv",
            ),
            (
                "X21", variant, f"X21-{variant}-tr24-notebook-s-curves-legacy-axis-z-v5",
                ("x21_s_curves_by_condition", "x21_s_curves_pooled"),
                "s_curve_trials.csv",
            ),
            (
                "B23", variant, f"B23-{variant}-tr24-notebook-s-curves-notebook-folds-v4",
                ("b23_s_curves_by_condition", "b23_s_curves_pooled"),
                "s_curve_word_trials.csv",
            ),
        ])
    for dataset, variant, directory, stems, trial_name in specifications:
        source_directory = repository / "results" / "figures" / directory
        for stem in stems:
            for suffix in (".png", ".svg"):
                source = source_directory / f"{stem}{suffix}"
                target = destination / f"{dataset}_{variant}_{stem}{suffix}"
                shutil.copy2(source, target)
                records.append({
                    "dataset_id": dataset, "variant": variant, "layer": "tr_24",
                    "figure_kind": stem, "output_file": str(target.resolve()),
                    "source_file": str(source.resolve()), "source_sha256": sha256_file(source),
                })
        trial_source = source_directory / trial_name
        trial_target = table_destination / f"{dataset}_{variant}_{trial_name}"
        shutil.copy2(trial_source, trial_target)
    return pd.DataFrame(records)


def _write_readme(output: Path) -> None:
    text = """# Cross-talker generalization: extended figure package

This package incorporates the figure updates requested on 2026-08-21. It is generated without overwriting an existing report directory.

## Recommended presentation figures

1. `figure_00_notebook_ceiling_normalized_profiles`: notebook-compatible layout. MFCC and STRF appear first; gray points are folds; black points and intervals are fold means and fold-bootstrap 95% intervals; 100% is mean behavioral-ceiling z and the gray band is its fold interval. With three folds, the interval primarily shows between-fold dispersion.
2. `figure_02_best_predictive_gain`: dataset-specific y-axis ranges keep X21 and B23 visible. Every dataset includes MFCC, STRF, HuBERT base, and HuBERT ASR fine-tuned.
3. `figure_05b_experiment_talker_distance_matrices`: Tr-24 matched-content raw DTW distances among all talkers.
4. `figure_05c_an19_complete_corpus_talker_similarity`: AN19 42×42 same-word full-corpus validation matrix.
5. `s_curves_tr24/`: complete Tr-24 `exp(-k*d)` S-curves. X21/B23 use one main talker plus three side panels; AN19 contains 24 test-talker panels and test-accent summaries. Condition-wise and pooled versions are provided.

## Figure 03 semantics

HVE describes dispersion within the speech heard during exposure; it does not compare exposure speech with test speech. The association panel asks whether HVE relates to accuracy beyond original condition. The prediction panel asks whether HVE improves participant-held-out prediction. AN19/X21 currently show mostly negative z and near-zero OOF gain. Identifiable B23 HVE is confounded with condition, so an incremental GLMM is not estimable.

## Figure 09 semantics

This sensitivity asks whether conclusions change when DTW distance is represented by the primary `-(d-mean_train)/sd_train` predictor versus fixed `exp(-d)`. The y-axis remains held-out log-loss gain. Black crosses mark numerically unstable exponential-predictor fits. This is a supplementary figure.

## Statistical distinction

Figure 00 is a notebook-compatible three-fold Wald-z/ceiling association display, not cross-validated prediction. It redraws existing held-out-refit results without rerunning those fits. True participant-held-out frozen-model prediction appears in Figures 01, 02, and 04.

## S-curve interpretation

S-curves are descriptive: quantile-bin points and 95% intervals plus unadjusted logistic curves. They are not hierarchical GLMM conditional-effect plots. X21 control and Talker-specific cells may contain missing, constant, or self-comparison predictors and must be interpreted with panel annotations.

## Current strength of evidence

The analysis turns SBI/HVE accounts into testable predictors across three experiments and yields strong cross-validated incremental SBI prediction in AN19. The current results do not support a claim of consistent evidence across all datasets: X21 gains are small, B23 SBI is near zero, and HVE lacks robust positive evidence. A defensible summary is strong AN19 support, weak X21 replication, and a B23 boundary condition.

Highest-priority extensions:

1. Prespecify Tr-24 or select a layer with nested CV; do not treat the best layer chosen from held-out results as an unbiased headline.
2. Add participant-cluster bootstrap 95% intervals for OOF gain, calibration, and key coefficients, plus a cross-dataset hierarchical or meta-analytic summary.
3. Add mechanistic validation using the AN19 talker matrix, L2-to-L1 phoneme deviation, control intelligibility, and error-type prediction.
4. Run any claimed Whisper/wav2vec, UMAP/PCA, or full-dimensional ablations, or remove unimplemented claims.
5. Collect a prospectively designed experiment in which model predictions determine talker or stimulus selection before behavioral data collection.

## Additional useful figures

- Cross-dataset forest plot with prespecified-layer effects, 95% intervals, pooled effect, and heterogeneity.
- OOF calibration plots stratified by dataset and condition.
- Exposure–test cell residual plots after controlling condition.
- Ceiling-gap plots using a common predictive metric and uncertainty interval.
- Prospective stimulus maps with preregistered predictions.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _write_slide_outline(output: Path) -> None:
    text = """# Presentation outline

## 1. Research question and three-dataset design

- Can exemplar similarity in ASR representations predict human cross-talker generalization?
- State that the reporting emphasis uses 3-D t-SNE, historical mean-sequence-length DTW normalization, and prespecified Tr-24 descriptive figures.

## 2. Notebook-compatible result and behavioral ceiling

- Figure: `figures/figure_00_notebook_ceiling_normalized_profiles.png`.
- MFCC/STRF appear first; 100% is mean three-fold behavioral-ceiling z; the gray band is its fold-bootstrap interval.
- This is a compatibility association display, not cross-validated prediction.

## 3. True participant-held-out prediction

- Figures: `figure_01_sbi_layer_profiles.png` and `figure_02_best_predictive_gain.png`.
- AN19 has clear incremental prediction, X21 is small, and B23 is near zero. Dataset-specific scales keep all baselines visible.

## 4. Tr-24 S-curves

- Use `X21_base_x21_s_curves_by_condition.png` with the pooled version as a comparison.
- Use `X21_ft_*` for the fine-tuned representation.
- Quantile points and logistic curves are descriptive and do not replace hierarchical GLMM/OOF results.

## 5. Talker representation structure

- Experimental matched-content matrices: `figure_05b_experiment_talker_distance_matrices.png`.
- AN19 42×42 validation: `figure_05c_an19_complete_corpus_talker_similarity.png`.

## 6. Exposure variability (HVE)

- Use the complete Figure 03 core profiles.
- Separate association after condition adjustment from participant-held-out incremental prediction. AN19/X21 currently lack stable positive evidence; B23 is condition-confounded.

## 7. Method sensitivity

- `figure_09_predictor_transform_sensitivity.png` compares standardized negative distance with fixed `exp(-d)`. Use as supplementary material.

## 8. Supported conclusion

- AN19 strongly supports incremental SBI prediction.
- X21 is a weak replication with a small effect.
- B23 is near zero under the current design and defines a boundary condition.
- Do not claim consistent support across all three datasets.

## 9. Priority extensions

1. Prespecify Tr-24 or use nested CV for layer selection.
2. Add participant-cluster bootstrap intervals and direct HuBERT–baseline comparisons.
3. Add cross-dataset hierarchical synthesis and heterogeneity analysis.
4. Add a prospective experiment or genuinely independent test set.
"""
    (output / "PRESENTATION_OUTLINE.md").write_text(text, encoding="utf-8")


def build_extended_report(repository: str | Path, output_dir: str | Path) -> dict[str, object]:
    repository = Path(repository).resolve()
    project_root = repository / "cross_talker_generalization"
    output = Path(output_dir).resolve()
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    sbi = _collect_sbi(project_root)
    sbi_exp = _collect_sbi_exp(project_root)
    hve = _collect_hve(project_root)
    hve_all = _collect_all_hve(project_root, hve)
    diagnostics = _diagnostics(project_root, pd.concat([sbi, sbi_exp], ignore_index=True), hve_all)
    atomic_write_csv(tables / "sbi_all_results.csv", sbi)
    atomic_write_csv(tables / "sbi_exp_k1_sensitivity_results.csv", sbi_exp)
    atomic_write_csv(tables / "hve_overall_results.csv", hve)
    atomic_write_csv(tables / "hve_all_results.csv", hve_all)
    atomic_write_csv(tables / "diagnostic_summary.csv", diagnostics)

    fold_z, ceiling_z = _ceiling_normalized_profiles(
        repository, figures / "figure_00_notebook_ceiling_normalized_profiles"
    )
    atomic_write_csv(tables / "notebook_fold_z_percent_ceiling.csv", fold_z)
    atomic_write_csv(tables / "notebook_behavioral_ceiling_z.csv", ceiling_z)

    _sbi_profiles(sbi, figures / "figure_01_sbi_layer_profiles")
    best = _best_gain_facets(sbi, figures / "figure_02_best_predictive_gain")
    atomic_write_csv(tables / "best_predictive_gain.csv", best)
    _clear_hve_profiles(hve, figures / "figure_03_hve_overall_profiles")
    ceiling = _ceiling_figure(project_root, sbi, figures / "figure_04_cv_ceiling")
    atomic_write_csv(tables / "cv_ceiling_comparison.csv", ceiling)
    _correlation_composite(project_root, figures / "figure_05a_layer_distance_correlations")
    talker_distances = _talker_distance_matrices(
        repository, figures / "figure_05b_experiment_talker_distance_matrices"
    )
    atomic_write_csv(tables / "talker_distance_tr24.csv", talker_distances)
    _an19_complete_talker_similarity(
        repository, figures / "figure_05c_an19_complete_corpus_talker_similarity"
    )
    b23 = _b23_hve(project_root, figures / "figure_06_b23_hve_descriptive")
    atomic_write_csv(tables / "b23_hve_descriptive.csv", b23)
    _hve_heatmaps(hve_all, "AN19", figures / "figure_07_an19_hve_sensitivity")
    _hve_heatmaps(hve_all, "X21", figures / "figure_08_x21_hve_sensitivity")
    transform = _clear_transform_sensitivity(
        sbi, sbi_exp, figures / "figure_09_predictor_transform_sensitivity"
    )
    atomic_write_csv(tables / "predictor_transform_sensitivity.csv", transform)
    scurve_manifest = _copy_notebook_scurves(repository, figures, tables)
    atomic_write_csv(tables / "s_curve_figure_manifest.csv", scurve_manifest)
    _write_readme(output)
    _write_slide_outline(output)

    hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "provenance.json":
            hashes[path.relative_to(output).as_posix()] = sha256_file(path)
    metadata = {
        **runtime_record(),
        "status": "complete",
        "stage": "report_extended",
        "primary_representation": "3-D t-SNE",
        "dtw_normalization": "mean_sequence_length",
        "primary_evaluation": "participant-held-out frozen-model prediction",
        "notebook_z_figure_estimand": "existing heldout-refit Wald z divided by mean three-fold ceiling z",
        "compatibility_models_rerun_for_release": False,
        "notebook_z_figure_is_cross_validated_prediction": False,
        "s_curve_layer": "tr_24",
        "files_sha256": hashes,
    }
    atomic_write_json(output / "provenance.json", metadata)
    return metadata
