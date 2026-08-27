from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from .report_core import COLORS, HVE_MEASURES, LAYERS, _collect_all_hve, _collect_hve
from .report_extended import DISPLAY_LAYERS, _fold_bootstrap95, _save_figure
from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


DATASETS = ("AN19", "X21", "B23")
VARIANTS = ("base", "ft")
VARIABILITY_Z_DIR = "{dataset}-{variant}-variability-z-notebook-simplified-tau2-v1"
CEILING_Z_DIR = "{dataset}-behavioral-ceiling-notebook-folds-v1"

MEASURE_LABELS = {
    "overall": "Overall frame dispersion",
    "overall_order_sensitive": "Overall exposure-order transitions",
    "within_token_sentence": "Within sentence",
    "within_type_sentence": "Within sentence type",
    "between_type_sentence": "Between sentence types",
    "order_sentence": "Adjacent frames: sentence",
    "mean_dissimilarity_sentence": "Within-type DTW: sentence",
    "within_token_word": "Within word",
    "within_type_word": "Within word type",
    "between_type_word": "Between word types",
    "order_word": "Adjacent frames: word",
    "mean_dissimilarity_word": "Within-type DTW: word",
    "within_token_phoneme": "Within phoneme",
    "within_type_phoneme": "Within phoneme type",
    "between_type_phoneme": "Between phoneme types",
    "order_phoneme": "Adjacent frames: phoneme",
    "mean_dissimilarity_phoneme": "Within-type DTW: phoneme",
}

CORE_MEASURES = {
    "AN19": ("within_token_word", "between_type_word", "order_word"),
    "X21": ("within_token_sentence", "between_type_sentence", "order_sentence"),
    "B23": ("within_token_sentence", "between_type_sentence", "order_sentence"),
}

EXPECTED_MEASURES = {
    "AN19": tuple(
        measure
        for measure in HVE_MEASURES
        if measure in {"overall", "overall_order_sensitive"} or measure.endswith("_word")
    ),
    "X21": tuple(HVE_MEASURES),
    "B23": tuple(
        measure
        for measure in HVE_MEASURES
        if measure not in {"within_type_sentence", "mean_dissimilarity_sentence"}
    ),
}


def _collect_association_folds(repository: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    statistics = repository / "results" / "statistics"
    frames: list[pd.DataFrame] = []
    ceiling_frames: list[pd.DataFrame] = []
    for dataset in DATASETS:
        ceiling_path = statistics / CEILING_Z_DIR.format(dataset=dataset) / "behavioral_ceiling_z.csv"
        ceiling = pd.read_csv(ceiling_path)
        ceiling["fold"] = pd.to_numeric(ceiling["fold"], errors="raise").astype(int)
        ceiling["z_test"] = pd.to_numeric(ceiling["z_test"], errors="raise")
        if len(ceiling) != 3:
            raise ValueError(f"{dataset}: expected three behavioral-ceiling folds, found {len(ceiling)}")
        ceiling_mean = float(ceiling["z_test"].mean())
        ceiling["ceiling_mean_z"] = ceiling_mean
        ceiling["ceiling_percent"] = ceiling["z_test"] / ceiling_mean * 100.0
        ceiling["source_path"] = str(ceiling_path.resolve())
        ceiling_frames.append(ceiling)

        for variant in VARIANTS:
            source_path = (
                statistics
                / VARIABILITY_Z_DIR.format(dataset=dataset, variant=variant)
                / "heldout_refit_z.csv"
            )
            frame = pd.read_csv(source_path)
            frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
            frame["z_test"] = pd.to_numeric(frame["z_test"], errors="raise")
            frame["percent_ceiling"] = frame["z_test"] / ceiling_mean * 100.0
            frame["absolute_percent_ceiling"] = frame["percent_ceiling"].abs()
            frame["source_path"] = str(source_path.resolve())
            frames.append(frame)

    association = pd.concat(frames, ignore_index=True)
    ceilings = pd.concat(ceiling_frames, ignore_index=True)
    expected_layers = set(LAYERS)
    for dataset in DATASETS:
        expected_measures = set(EXPECTED_MEASURES[dataset])
        for variant in VARIANTS:
            subset = association.loc[
                association["dataset_id"].eq(dataset)
                & association["model_variant"].eq(variant)
            ]
            observed_measures = set(subset["measure"])
            if observed_measures != expected_measures:
                raise ValueError(
                    f"{dataset} {variant}: measure inventory mismatch; "
                    f"missing={sorted(expected_measures - observed_measures)}, "
                    f"extra={sorted(observed_measures - expected_measures)}"
                )
            counts = subset.groupby(["measure", "layer"])["fold"].agg(["count", "nunique"])
            if set(subset["layer"]) != expected_layers or not (
                counts["count"].eq(3).all() and counts["nunique"].eq(3).all()
            ):
                raise ValueError(f"{dataset} {variant}: incomplete three-fold layer inventory")
    return association, ceilings


def _summarize_association(association: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in association.groupby(
        ["dataset_id", "model_variant", "representation", "measure", "layer"], sort=False
    ):
        values = group.sort_values("fold")["percent_ceiling"].to_numpy(float)
        mean, low, high = _fold_bootstrap95(values)
        rows.append(
            {
                "dataset_id": keys[0],
                "model_variant": keys[1],
                "representation": keys[2],
                "measure": keys[3],
                "layer": keys[4],
                "mean_percent_ceiling": mean,
                "fold_bootstrap95_low": low,
                "fold_bootstrap95_high": high,
                "mean_z_test": float(group["z_test"].mean()),
                "min_fold_z_test": float(group["z_test"].min()),
                "max_fold_z_test": float(group["z_test"].max()),
                "n_folds": len(values),
            }
        )
    return pd.DataFrame(rows)


def _ceiling_context(ceilings: pd.DataFrame, dataset: str) -> tuple[float, float, float]:
    rows = ceilings.loc[ceilings["dataset_id"].eq(dataset)]
    values = rows["ceiling_percent"].to_numpy(float)
    _, low, high = _fold_bootstrap95(values)
    ceiling_mean = float(rows["ceiling_mean_z"].iloc[0])
    significance = 1.96 / ceiling_mean * 100.0
    return low, high, significance


def _limits_for_dataset(
    association: pd.DataFrame, ceilings: pd.DataFrame, dataset: str, measures: tuple[str, ...]
) -> tuple[float, float]:
    values = association.loc[
        association["dataset_id"].eq(dataset) & association["measure"].isin(measures),
        "percent_ceiling",
    ].to_numpy(float)
    ceiling_low, ceiling_high, significance = _ceiling_context(ceilings, dataset)
    lower = min(float(values.min()), -significance, 0.0) - 4.0
    upper = max(float(values.max()), ceiling_high, 100.0) + 4.0
    step = 10.0 if max(abs(lower), abs(upper)) < 150 else 25.0
    return math.floor(lower / step) * step, math.ceil(upper / step) * step


def _draw_association_axis(
    axis: plt.Axes,
    association: pd.DataFrame,
    ceilings: pd.DataFrame,
    *,
    dataset: str,
    measure: str,
    y_limits: tuple[float, float],
    show_x_labels: bool,
) -> None:
    x = np.arange(len(LAYERS), dtype=float)
    ceiling_low, ceiling_high, significance = _ceiling_context(ceilings, dataset)
    axis.axhspan(ceiling_low, ceiling_high, color="#d4d4d4", alpha=0.75, zorder=0)
    axis.axhline(100.0, color="#222222", linestyle="--", linewidth=1.0, zorder=1)
    axis.axhline(0.0, color="#222222", linewidth=0.9, zorder=1)
    axis.axhline(significance, color="#8c8c8c", linestyle=":", linewidth=0.9, zorder=1)
    axis.axhline(-significance, color="#8c8c8c", linestyle=":", linewidth=0.9, zorder=1)

    for variant, offset, marker in (("base", -0.10, "o"), ("ft", 0.10, "s")):
        data = association.loc[
            association["dataset_id"].eq(dataset)
            & association["model_variant"].eq(variant)
            & association["measure"].eq(measure)
        ]
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for position, layer in zip(x, LAYERS):
            values = data.loc[data["layer"].eq(layer)].sort_values("fold")["percent_ceiling"].to_numpy(float)
            if len(values) != 3:
                raise ValueError(f"{dataset} {variant} {measure} {layer}: expected three folds")
            axis.scatter(
                position + offset + np.array([-0.025, 0.0, 0.025]),
                values,
                s=14,
                color=COLORS[variant],
                alpha=0.28,
                linewidths=0,
                zorder=2,
            )
            mean, low, high = _fold_bootstrap95(values)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        means_array = np.asarray(means)
        axis.plot(
            x + offset,
            means_array,
            color=COLORS[variant],
            marker=marker,
            markersize=3.5,
            linewidth=1.45,
            zorder=3,
        )
        axis.errorbar(
            x + offset,
            means_array,
            yerr=[means_array - np.asarray(lows), np.asarray(highs) - means_array],
            fmt="none",
            color=COLORS[variant],
            linewidth=1.0,
            capsize=2.0,
            zorder=3,
        )

    axis.set_title(MEASURE_LABELS[measure], fontsize=10.5, fontweight="bold")
    axis.set_ylim(*y_limits)
    axis.set_xlim(-0.55, len(LAYERS) - 0.45)
    axis.set_xticks(x)
    if show_x_labels:
        axis.set_xticklabels([DISPLAY_LAYERS[layer] for layer in LAYERS], rotation=58, ha="right", fontsize=7)
    else:
        axis.set_xticklabels([])
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.65, zorder=0)
    sns.despine(ax=axis)


def _association_legend() -> list[object]:
    return [
        Line2D([0], [0], color=COLORS["base"], marker="o", label="HuBERT base"),
        Line2D([0], [0], color=COLORS["ft"], marker="s", label="HuBERT ASR fine-tuned"),
        Patch(facecolor="#d4d4d4", edgecolor="none", label="Behavioral-ceiling fold-bootstrap 95% CI"),
        Line2D([0], [0], color="#222222", linestyle="--", label="Mean behavioral ceiling = 100%"),
        Line2D([0], [0], color="#8c8c8c", linestyle=":", label="|Wald z| = 1.96 reference"),
    ]


def _plot_core_association(
    association: pd.DataFrame, ceilings: pd.DataFrame, destination: Path
) -> None:
    sns.set_theme(style="ticks", context="notebook")
    figure, axes = plt.subplots(3, 3, figsize=(20.5, 13.2), sharex=False, sharey=False)
    for row, dataset in enumerate(DATASETS):
        measures = CORE_MEASURES[dataset]
        y_limits = _limits_for_dataset(association, ceilings, dataset, measures)
        for column, measure in enumerate(measures):
            _draw_association_axis(
                axes[row, column],
                association,
                ceilings,
                dataset=dataset,
                measure=measure,
                y_limits=y_limits,
                show_x_labels=row == len(DATASETS) - 1,
            )
            if column == 0:
                axes[row, column].set_ylabel(
                    f"{dataset}\nSigned held-out-refit z / mean ceiling z (%)", fontsize=9.5
                )
    figure.legend(
        handles=_association_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        fontsize=8.7,
        frameon=False,
    )
    figure.suptitle(
        "Exposure variability: notebook-core layer profiles\n"
        "Three participant folds; signed association statistics and behavioral-ceiling normalization",
        fontsize=17,
        fontweight="bold",
        y=0.998,
    )
    figure.text(
        0.5,
        0.006,
        "AN19 'Between word types' is the registered type-centroid definition, not the legacy notebook's pooled token-centroid BetweenWord. "
        "B23 is a compatibility extension for identified single-talker exposure conditions.",
        ha="center",
        fontsize=8.3,
        color="#7f1d1d",
    )
    figure.tight_layout(rect=(0.02, 0.035, 0.995, 0.93))
    _save_figure(figure, destination, dpi=300)


def _plot_all_association_methods(
    association: pd.DataFrame,
    ceilings: pd.DataFrame,
    *,
    dataset: str,
    destination: Path,
) -> None:
    measures = EXPECTED_MEASURES[dataset]
    columns = 3 if len(measures) <= 6 else 4
    rows = math.ceil(len(measures) / columns)
    sns.set_theme(style="ticks", context="notebook")
    figure, axes = plt.subplots(rows, columns, figsize=(5.1 * columns, 4.15 * rows), squeeze=False)
    y_limits = _limits_for_dataset(association, ceilings, dataset, measures)
    for index, axis in enumerate(axes.flat):
        if index >= len(measures):
            axis.axis("off")
            continue
        _draw_association_axis(
            axis,
            association,
            ceilings,
            dataset=dataset,
            measure=measures[index],
            y_limits=y_limits,
            show_x_labels=True,
        )
        if index % columns == 0:
            axis.set_ylabel("Signed z / mean ceiling z (%)", fontsize=8.5)
    figure.legend(
        handles=_association_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=5,
        fontsize=8.5,
        frameon=False,
    )
    subtitle = {
        "AN19": "7 supported overall and word-recording measures",
        "X21": "all 17 registered measures",
        "B23": "15 modelable measures; two repeated-sentence measures lack coverage",
    }[dataset]
    figure.suptitle(
        f"{dataset} exposure variability across HuBERT layers\n{subtitle}",
        fontsize=17,
        fontweight="bold",
        y=0.998,
    )
    figure.text(
        0.5,
        0.006,
        "Fold dots, fold mean, and exact three-fold bootstrap 95% CI. Signed values preserve effect direction; this is association, not OOF prediction.",
        ha="center",
        fontsize=8.3,
    )
    figure.tight_layout(rect=(0.02, 0.025, 0.995, 0.93))
    _save_figure(figure, destination, dpi=300)


def _collect_true_oof(project_root: Path) -> pd.DataFrame:
    overall = _collect_hve(project_root)
    result = _collect_all_hve(project_root, overall)
    result["oof_gain_joint_vs_condition"] = pd.to_numeric(
        result["oof_gain_joint_vs_condition"], errors="raise"
    )
    expected = {"AN19": set(EXPECTED_MEASURES["AN19"]), "X21": set(EXPECTED_MEASURES["X21"])}
    for dataset, methods in expected.items():
        subset = result.loc[result["dataset_id"].eq(dataset)]
        if set(subset["measure"]) != methods:
            raise ValueError(f"{dataset}: true-OOF variability measure inventory is incomplete")
        counts = subset.groupby(["variant", "measure", "layer"]).size()
        if not counts.eq(1).all():
            raise ValueError(f"{dataset}: duplicate or missing true-OOF variability layer rows")
    return result


def _draw_oof_axis(
    axis: plt.Axes,
    oof: pd.DataFrame,
    *,
    dataset: str,
    measure: str,
    y_limits: tuple[float, float],
    show_x_labels: bool,
) -> None:
    x = np.arange(len(LAYERS), dtype=float)
    axis.axhline(0.0, color="#222222", linewidth=0.9)
    for variant, marker in (("base", "o"), ("ft", "s")):
        values = (
            oof.loc[
                oof["dataset_id"].eq(dataset)
                & oof["variant"].eq(variant)
                & oof["measure"].eq(measure)
            ]
            .set_index("layer")
            .reindex(LAYERS)["oof_gain_joint_vs_condition"]
        )
        if values.isna().any():
            raise ValueError(f"{dataset} {variant} {measure}: incomplete OOF layer profile")
        axis.plot(
            x,
            values.to_numpy(float),
            color=COLORS[variant],
            marker=marker,
            markersize=3.5,
            linewidth=1.5,
        )
    axis.set_title(MEASURE_LABELS[measure], fontsize=10.5, fontweight="bold")
    axis.set_ylim(*y_limits)
    axis.set_xlim(-0.55, len(LAYERS) - 0.45)
    axis.set_xticks(x)
    if show_x_labels:
        axis.set_xticklabels([DISPLAY_LAYERS[layer] for layer in LAYERS], rotation=58, ha="right", fontsize=7)
    else:
        axis.set_xticklabels([])
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.65)
    sns.despine(ax=axis)


def _oof_limits(oof: pd.DataFrame, dataset: str, measures: tuple[str, ...]) -> tuple[float, float]:
    values = oof.loc[
        oof["dataset_id"].eq(dataset) & oof["measure"].isin(measures),
        "oof_gain_joint_vs_condition",
    ].to_numpy(float)
    margin = max((float(values.max()) - float(values.min())) * 0.08, 0.00005)
    return float(values.min()) - margin, float(values.max()) + margin


def _plot_core_oof(oof: pd.DataFrame, destination: Path) -> None:
    datasets = ("AN19", "X21")
    sns.set_theme(style="ticks", context="notebook")
    figure, axes = plt.subplots(2, 3, figsize=(20.5, 8.7), squeeze=False)
    for row, dataset in enumerate(datasets):
        measures = CORE_MEASURES[dataset]
        y_limits = _oof_limits(oof, dataset, measures)
        for column, measure in enumerate(measures):
            _draw_oof_axis(
                axes[row, column],
                oof,
                dataset=dataset,
                measure=measure,
                y_limits=y_limits,
                show_x_labels=row == len(datasets) - 1,
            )
            if column == 0:
                axes[row, column].set_ylabel(f"{dataset}\nOOF log-loss gain", fontsize=9.5)
    figure.legend(
        handles=[
            Line2D([0], [0], color=COLORS["base"], marker="o", label="HuBERT base"),
            Line2D([0], [0], color=COLORS["ft"], marker="s", label="HuBERT ASR fine-tuned"),
            Line2D([0], [0], color="#222222", label="No predictive gain"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Exposure variability: true participant-held-out prediction\n"
        "Above zero improves prediction beyond condition; below zero worsens it",
        fontsize=17,
        fontweight="bold",
        y=0.998,
    )
    figure.text(
        0.5,
        0.008,
        "B23 is absent because the current release does not integrate multi-talker exposure; estimability must be reassessed after that integration.",
        ha="center",
        fontsize=8.5,
        color="#7f1d1d",
    )
    figure.tight_layout(rect=(0.02, 0.035, 0.995, 0.91))
    _save_figure(figure, destination, dpi=300)


def _plot_all_oof_methods(oof: pd.DataFrame, *, dataset: str, destination: Path) -> None:
    measures = EXPECTED_MEASURES[dataset]
    columns = 3 if len(measures) <= 6 else 4
    rows = math.ceil(len(measures) / columns)
    sns.set_theme(style="ticks", context="notebook")
    figure, axes = plt.subplots(rows, columns, figsize=(5.1 * columns, 4.0 * rows), squeeze=False)
    y_limits = _oof_limits(oof, dataset, measures)
    for index, axis in enumerate(axes.flat):
        if index >= len(measures):
            axis.axis("off")
            continue
        _draw_oof_axis(
            axis,
            oof,
            dataset=dataset,
            measure=measures[index],
            y_limits=y_limits,
            show_x_labels=True,
        )
        if index % columns == 0:
            axis.set_ylabel("OOF log-loss gain", fontsize=8.5)
    figure.legend(
        handles=[
            Line2D([0], [0], color=COLORS["base"], marker="o", label="HuBERT base"),
            Line2D([0], [0], color=COLORS["ft"], marker="s", label="HuBERT ASR fine-tuned"),
            Line2D([0], [0], color="#222222", label="No predictive gain"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        f"{dataset}: all exposure-variability methods in true participant-held-out prediction",
        fontsize=17,
        fontweight="bold",
        y=0.997,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.995, 0.91))
    _save_figure(figure, destination, dpi=300)


def _availability_table(association: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        observed = set(association.loc[association["dataset_id"].eq(dataset), "measure"])
        oof_observed = set(oof.loc[oof["dataset_id"].eq(dataset), "measure"])
        for measure in HVE_MEASURES:
            rows.append(
                {
                    "dataset_id": dataset,
                    "measure": measure,
                    "display_label": MEASURE_LABELS[measure],
                    "three_fold_heldout_refit_z_available": measure in observed,
                    "true_participant_heldout_oof_available": measure in oof_observed,
                    "status_note": (
                        "available"
                        if measure in observed
                        else (
                            "AN19 has isolated-word recordings only"
                            if dataset == "AN19"
                            else "undefined: one physical token per sentence type in identified B23 exposure pools"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_readme(output: Path) -> None:
    text = """# Complete Figure 03 variability package

This component adds complete variability figures without overwriting earlier compatibility outputs.

## Recommended presentation order

1. `figure_03a_variability_true_oof_core_profiles`: participant-held-out condition-incremental OOF log-loss gain. This is separate from the planned predictor-only likelihood optimization. B23 is absent because the public multi-talker stimulus mapping is not integrated in the current release.
2. `figure_03b_variability_ceiling_normalized_core_profiles`: notebook-compatible three-fold Wald-z/behavioral-ceiling display with the sign of z preserved.
3. Figures 03c–03e: every currently computable variability method for AN19, X21, and B23.
4. Figures 03f–03g: true OOF results for all methods in AN19 and X21.

## Differences from the earlier Figure 03

- The package is not restricted to `overall`.
- It shows 6 AN19, 16 X21, and 14 B23 computed methods. For the current B23 single-talker pools, `within_type_sentence` and `mean_dissimilarity_sentence` are undefined because there is one recording per sentence type; this count is separate from multi-talker condition coverage.
- It does not use `abs(z)`: positive values associate greater variability with greater accuracy; negative values indicate the opposite direction.
- Three-fold association figures retain fold points, fold means, exact fold-bootstrap 95% intervals, and behavioral-ceiling 95% bands.
- Held-out-refit Wald z is explicitly separated from frozen-model OOF prediction.

## Historical measure that is not claimed as reproduced

Current AN19 `between_type_word` first combines token centroids within word type. The historical notebook's `BetweenWord` pooled all talker × word token centroids. These are different estimands. The new figure uses `Between word types` and does not claim reproduction of historical `BetweenWord`. Exact reproduction would require adding that measure and rerunning the AN19 variability GLMM.

All three-fold association results use `tau=2`. Percentages are Wald z rescaled by mean behavioral-ceiling z, not the percentage of human behavior explained.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def build_variability_report(repository: str | Path, output_dir: str | Path) -> dict[str, object]:
    repository_path = Path(repository).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report directory: {output}")
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True)
    tables.mkdir(parents=True)

    association, ceilings = _collect_association_folds(repository_path)
    summary = _summarize_association(association)
    project_root = repository_path / "cross_talker_generalization"
    oof = _collect_true_oof(project_root)
    availability = _availability_table(association, oof)

    atomic_write_csv(tables / "variability_all_fold_z_percent_ceiling.csv", association)
    atomic_write_csv(tables / "variability_layer_summary.csv", summary)
    atomic_write_csv(tables / "variability_behavioral_ceiling_z.csv", ceilings)
    atomic_write_csv(tables / "variability_true_oof_all_methods.csv", oof)
    atomic_write_csv(tables / "variability_method_availability.csv", availability)

    _plot_core_oof(oof, figures / "figure_03a_variability_true_oof_core_profiles")
    _plot_core_association(
        association, ceilings, figures / "figure_03b_variability_ceiling_normalized_core_profiles"
    )
    for figure_id, dataset in (("03c", "AN19"), ("03d", "X21"), ("03e", "B23")):
        _plot_all_association_methods(
            association,
            ceilings,
            dataset=dataset,
            destination=figures / f"figure_{figure_id}_{dataset.lower()}_all_variability_methods",
        )
    _plot_all_oof_methods(
        oof,
        dataset="AN19",
        destination=figures / "figure_03f_an19_all_variability_methods_true_oof",
    )
    _plot_all_oof_methods(
        oof,
        dataset="X21",
        destination=figures / "figure_03g_x21_all_variability_methods_true_oof",
    )
    _write_readme(output)

    source_files = sorted(set(association["source_path"]) | set(ceilings["source_path"]))
    metadata = {
        **runtime_record(),
        "stage": "report_variability",
        "output_directory": str(output),
        "old_outputs_modified": False,
        "association_sign_policy": "signed_z_test; absolute value retained only as a source-table audit column",
        "association_warning": "held-out-refit Wald z is not frozen OOF prediction",
        "true_oof_datasets": ["AN19", "X21"],
        "b23_oof_status": "not_identifiable_exposure_variability_confounded_with_condition",
        "method_counts": {dataset: len(EXPECTED_MEASURES[dataset]) for dataset in DATASETS},
        "source_files_sha256": {path: sha256_file(Path(path)) for path in source_files},
    }
    atomic_write_json(output / "provenance.json", metadata)
    return metadata
