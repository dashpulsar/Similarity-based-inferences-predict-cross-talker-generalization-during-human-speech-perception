"""Build a compact report for the global exposure-order HVE analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("AN19", "X21", "B23")
VARIANTS = ("base", "ft")
LAYERS = (
    "cnn_2", "cnn_3", "cnn_4", "cnn_5", "cnn_6", "tr_0", "tr_2", "tr_4",
    "tr_6", "tr_8", "tr_10", "tr_12", "tr_14", "tr_16", "tr_18", "tr_20",
    "tr_22", "tr_24",
)
COLORS = {"base": "#1f77b4", "ft": "#d62728"}
DATASET_ORDER = {dataset: index for index, dataset in enumerate(DATASETS)}


def _sort_dataset_variant(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep report tables in the same study order used by the figures."""
    result = frame.copy()
    result["_dataset_order"] = result["dataset_id"].map(DATASET_ORDER)
    return result.sort_values(["_dataset_order", "variant"]).drop(columns="_dataset_order")


def _cluster_bootstrap_ci(
    predictions: pd.DataFrame,
    *,
    reduced_model: str,
    full_model: str = "M_joint",
    seed: int,
    draws: int = 10000,
) -> tuple[float, float]:
    subset = predictions.loc[
        predictions["model_id"].isin([reduced_model, full_model])
    ].copy()
    losses = subset.groupby(["participant_id", "model_id"])["log_loss"].sum().unstack()
    trials = (
        subset.assign(
            trials=subset["response_correct"] + subset["response_incorrect"]
        )
        .groupby(["participant_id", "model_id"])["trials"]
        .sum()
        .unstack()
    )
    if losses.isna().any().any() or trials.isna().any().any():
        raise ValueError("paired OOF predictions are incomplete for participant bootstrap")
    if not trials[reduced_model].equals(trials[full_model]):
        raise ValueError("reduced and joint predictions use different participant trial counts")
    differences = (losses[reduced_model] - losses[full_model]).to_numpy(float)
    weights = trials[reduced_model].to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    estimates = differences[indices].sum(axis=1) / weights[indices].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _collect(project: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = []
    selected = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            directory = project / "artifacts/models" / f"{dataset}-HVE-{variant}-order-sensitive"
            metrics = pd.read_csv(directory / "cv_metrics.csv")
            oof = metrics.loc[metrics["scope"].eq("oof_all")]
            means = oof.pivot(index="feature_key", columns="model_id", values="mean_log_loss")
            totals = oof.loc[oof["model_id"].eq("M_predictor")].set_index("feature_key")
            frame = means.join(
                totals[["total_log_loss", "total_trials"]].rename(
                    columns={
                        "total_log_loss": "predictor_oof_total_log_loss",
                        "total_trials": "predictor_oof_total_trials",
                    }
                )
            ).reset_index()
            frame["layer"] = frame["feature_key"].str.split("::").str[0]
            frame["dataset_id"] = dataset
            frame["variant"] = variant
            frame["predictor_beyond_condition_oof_gain"] = frame["M_condition"] - frame["M_joint"]
            frame["condition_beyond_predictor_oof_gain"] = frame["M_predictor"] - frame["M_joint"]
            if frame["predictor_oof_total_trials"].nunique() != 1:
                raise ValueError(f"{dataset}/{variant}: layers use different held-out trial counts")
            profiles.append(frame)

            winner = frame.sort_values(
                ["predictor_oof_total_log_loss", "layer"], ascending=[True, True]
            ).iloc[0].to_dict()
            coefficients = pd.read_csv(directory / "coefficients.csv")
            coefficient = coefficients.loc[
                coefficients["scope"].eq("full")
                & coefficients["model_id"].eq("M_joint")
                & coefficients["term"].eq("variability_z")
                & coefficients["feature_key"].eq(winner["feature_key"])
            ]
            if len(coefficient) == 1:
                for column in ("estimate", "std_error", "z_value", "p_value", "conf_low", "conf_high"):
                    winner[f"joint_{column}"] = coefficient.iloc[0][column]
            comparisons = pd.read_csv(directory / "likelihood_ratio_tests.csv")
            comparisons = comparisons.loc[comparisons["feature_key"].eq(winner["feature_key"])]
            for row in comparisons.itertuples(index=False):
                prefix = str(row.comparison_id)
                winner[f"{prefix}_lrt_chisq"] = row.chisq
                winner[f"{prefix}_lrt_df"] = row.df
                winner[f"{prefix}_lrt_p"] = row.p_value
            predictions = pd.read_csv(directory / "oof_predictions.csv")
            predictions = predictions.loc[
                predictions["feature_key"].eq(winner["feature_key"])
            ]
            seed_offset = DATASETS.index(dataset) * 10 + VARIANTS.index(variant)
            for comparison_name, reduced_model in (
                ("predictor_beyond_condition", "M_condition"),
                ("condition_beyond_predictor", "M_predictor"),
            ):
                low, high = _cluster_bootstrap_ci(
                    predictions,
                    reduced_model=reduced_model,
                    seed=230519 + seed_offset,
                )
                winner[f"{comparison_name}_oof_ci_low"] = low
                winner[f"{comparison_name}_oof_ci_high"] = high
            diagnostics = pd.read_csv(directory / "diagnostics.csv")
            diagnostic = diagnostics.loc[diagnostics["feature_key"].eq(winner["feature_key"])]
            winner["all_fits_converged"] = diagnostic["convergence"].fillna("").eq("ok").all()
            winner["any_singular_fit"] = diagnostic["singular"].astype(str).str.lower().eq("true").any()
            selected.append(winner)
    return pd.concat(profiles, ignore_index=True), pd.DataFrame(selected)


def _collect_sbi_selection(project: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(project / "analysis_update_2026-08-21/tables/sbi_all_results.csv")
    hubert = source.loc[source["family"].eq("HuBERT t-SNE")].copy()
    selected = hubert.loc[
        hubert.groupby(["dataset_id", "variant"])["M_predictor"].idxmin()
    ]
    selected = _sort_dataset_variant(selected)
    selected = selected.assign(
        selection_rule="minimum three-fold held-out mean log loss from M_predictor"
    )
    for index, row in selected.iterrows():
        directory = (
            project / "artifacts/models"
            / f"{row['dataset_id']}-{row['dataset_id']}_hubert_{row['variant']}_tsne-confirmatory"
        )
        metrics = pd.read_csv(directory / "cv_metrics.csv")
        oof = metrics.loc[
            metrics["scope"].eq("oof_all")
            & metrics["feature_key"].eq(row["feature_key"])
        ].set_index("model_id")
        for model_id in ("M_condition", "M_predictor", "M_joint"):
            selected.loc[index, model_id] = float(oof.loc[model_id, "mean_log_loss"])
        selected.loc[index, "predictor_beyond_condition_oof_gain"] = (
            selected.loc[index, "M_condition"] - selected.loc[index, "M_joint"]
        )
        selected.loc[index, "condition_beyond_predictor_oof_gain"] = (
            selected.loc[index, "M_predictor"] - selected.loc[index, "M_joint"]
        )
        predictions = pd.read_csv(directory / "oof_predictions.csv")
        predictions = predictions.loc[predictions["feature_key"].eq(row["feature_key"])]
        for offset, (comparison_name, reduced_model) in enumerate(
            (("predictor_beyond_condition", "M_condition"),
             ("condition_beyond_predictor", "M_predictor"))
        ):
            low, high = _cluster_bootstrap_ci(
                predictions,
                reduced_model=reduced_model,
                seed=230919 + DATASETS.index(row["dataset_id"]) * 10
                + VARIANTS.index(row["variant"]) + offset,
            )
            selected.loc[index, f"{comparison_name}_oof_ci_low"] = low
            selected.loc[index, f"{comparison_name}_oof_ci_high"] = high
        diagnostics = pd.read_csv(directory / "diagnostics.csv")
        selected.loc[index, "all_fits_converged"] = (
            diagnostics["convergence"].fillna("").eq("ok").all()
        )
        selected.loc[index, "any_singular_fit"] = (
            diagnostics["singular"].astype(str).str.lower().eq("true").any()
        )
    return source, selected


def _collect_b23_order_independent(
    project: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates: list[pd.DataFrame] = []
    selected: list[dict[str, object]] = []
    for variant in VARIANTS:
        selection_dir = project / "artifacts/models" / f"B23-HVE-{variant}-selection"
        metrics = pd.read_csv(selection_dir / "cv_metrics.csv")
        frame = metrics.loc[
            metrics["scope"].eq("oof_all") & metrics["model_id"].eq("M_predictor")
        ].copy()
        frame["layer"] = frame["feature_key"].str.split("::").str[0]
        frame["measure"] = frame["feature_key"].str.split("::").str[1]
        frame = frame.loc[~frame["measure"].eq("overall_order_sensitive")]
        if frame["total_trials"].nunique() != 1:
            raise ValueError(f"B23/{variant}: order-independent candidates use different trials")
        frame["dataset_id"] = "B23"
        frame["variant"] = variant
        frame["selection_stratum"] = "order_independent_168_participants"
        frame = frame.rename(
            columns={
                "total_log_loss": "predictor_oof_total_log_loss",
                "total_trials": "predictor_oof_total_trials",
                "mean_log_loss": "M_predictor",
            }
        )
        candidates.append(frame)
        winner = frame.sort_values(
            ["predictor_oof_total_log_loss", "feature_key"]
        ).iloc[0].to_dict()

        comparison_dir = project / "artifacts/models" / f"B23-HVE-{variant}-selected"
        comparison_metrics = pd.read_csv(comparison_dir / "cv_metrics.csv")
        oof = comparison_metrics.loc[comparison_metrics["scope"].eq("oof_all")]
        means = oof.set_index("model_id")["mean_log_loss"]
        if set(means.index) != {"M_condition", "M_predictor", "M_joint"}:
            raise ValueError(f"B23/{variant}: selected comparison lacks a registered model")
        for model_id in means.index:
            winner[model_id] = float(means[model_id])
        winner["predictor_beyond_condition_oof_gain"] = (
            winner["M_condition"] - winner["M_joint"]
        )
        winner["condition_beyond_predictor_oof_gain"] = (
            winner["M_predictor"] - winner["M_joint"]
        )
        predictions = pd.read_csv(comparison_dir / "oof_predictions.csv")
        for offset, (comparison_name, reduced_model) in enumerate(
            (("predictor_beyond_condition", "M_condition"),
             ("condition_beyond_predictor", "M_predictor"))
        ):
            low, high = _cluster_bootstrap_ci(
                predictions,
                reduced_model=reduced_model,
                seed=230719 + VARIANTS.index(variant) * 10 + offset,
            )
            winner[f"{comparison_name}_oof_ci_low"] = low
            winner[f"{comparison_name}_oof_ci_high"] = high
        diagnostics = pd.read_csv(comparison_dir / "diagnostics.csv")
        winner["all_fits_converged"] = diagnostics["convergence"].fillna("").eq("ok").all()
        winner["any_singular_fit"] = diagnostics["singular"].astype(str).str.lower().eq("true").any()
        selected.append(winner)
    return pd.concat(candidates, ignore_index=True), pd.DataFrame(selected)


def _collect_an19_x21_hve(
    project: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates: list[pd.DataFrame] = []
    selected: list[dict[str, object]] = []
    for dataset in ("AN19", "X21"):
        for variant in VARIANTS:
            selection_dir = (
                project / "artifacts/models" / f"{dataset}-HVE-{variant}-revised-selection"
            )
            metrics = pd.read_csv(selection_dir / "cv_metrics.csv")
            frame = metrics.loc[
                metrics["scope"].eq("oof_all") & metrics["model_id"].eq("M_predictor")
            ].copy()
            frame["layer"] = frame["feature_key"].str.split("::").str[0]
            frame["measure"] = frame["feature_key"].str.split("::").str[1]
            if frame["total_trials"].nunique() != 1:
                raise ValueError(f"{dataset}/{variant}: HVE candidates use different trials")
            frame["dataset_id"] = dataset
            frame["variant"] = variant
            frame["selection_stratum"] = "all_exposure_participants"
            frame = frame.rename(
                columns={
                    "total_log_loss": "predictor_oof_total_log_loss",
                    "total_trials": "predictor_oof_total_trials",
                    "mean_log_loss": "M_predictor",
                }
            )
            candidates.append(frame)
            winner = frame.sort_values(
                ["predictor_oof_total_log_loss", "feature_key"]
            ).iloc[0].to_dict()

            if dataset == "AN19":
                comparison_dir = (
                    project / "artifacts/models" / f"AN19-HVE-{variant}-order-sensitive"
                )
            else:
                comparison_dir = project / "artifacts/models" / f"X21-HVE-{variant}-selected"
            comparison_metrics = pd.read_csv(comparison_dir / "cv_metrics.csv")
            oof = comparison_metrics.loc[
                comparison_metrics["scope"].eq("oof_all")
                & comparison_metrics["feature_key"].eq(winner["feature_key"])
            ]
            means = oof.set_index("model_id")["mean_log_loss"]
            for model_id in ("M_condition", "M_predictor", "M_joint"):
                winner[model_id] = float(means[model_id])
            winner["predictor_beyond_condition_oof_gain"] = (
                winner["M_condition"] - winner["M_joint"]
            )
            winner["condition_beyond_predictor_oof_gain"] = (
                winner["M_predictor"] - winner["M_joint"]
            )
            predictions = pd.read_csv(comparison_dir / "oof_predictions.csv")
            predictions = predictions.loc[predictions["feature_key"].eq(winner["feature_key"])]
            for offset, (comparison_name, reduced_model) in enumerate(
                (("predictor_beyond_condition", "M_condition"),
                 ("condition_beyond_predictor", "M_predictor"))
            ):
                low, high = _cluster_bootstrap_ci(
                    predictions,
                    reduced_model=reduced_model,
                    seed=230819 + DATASETS.index(dataset) * 10 + VARIANTS.index(variant) + offset,
                )
                winner[f"{comparison_name}_oof_ci_low"] = low
                winner[f"{comparison_name}_oof_ci_high"] = high
            diagnostics = pd.read_csv(comparison_dir / "diagnostics.csv")
            diagnostic = diagnostics.loc[diagnostics["feature_key"].eq(winner["feature_key"])]
            winner["all_fits_converged"] = diagnostic["convergence"].fillna("").eq("ok").all()
            winner["any_singular_fit"] = diagnostic["singular"].astype(str).str.lower().eq("true").any()
            selected.append(winner)
    return pd.concat(candidates, ignore_index=True), pd.DataFrame(selected)


def _collect_hve_selection(
    order_profiles: pd.DataFrame,
    order_selected: pd.DataFrame,
    complete_candidates: pd.DataFrame,
    complete_selected: pd.DataFrame,
    b23_candidates: pd.DataFrame,
    b23_selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete_candidates = complete_candidates.copy()
    complete_candidates["result_source"] = "2026-08-27 revised actual-exposure HVE fits"
    selected = complete_selected.copy()
    selected["result_source"] = "2026-08-27 revised actual-exposure HVE fits"
    b23_order_candidates = b23_candidates.copy()
    b23_order_candidates["result_source"] = "2026-08-27 revised B23 order-independent fits"
    b23_global_candidates = order_profiles.loc[order_profiles["dataset_id"].eq("B23")].copy()
    b23_global_candidates["measure"] = "overall_order_sensitive"
    b23_global_candidates["selection_stratum"] = "global_order_97_participants"
    b23_global_candidates["result_source"] = "2026-08-27 participant-order HVE fits"
    candidates = pd.concat(
        [complete_candidates, b23_order_candidates, b23_global_candidates],
        ignore_index=True, sort=False,
    )
    b23_global_selected = order_selected.loc[order_selected["dataset_id"].eq("B23")].copy()
    b23_global_selected["measure"] = "overall_order_sensitive"
    b23_global_selected["selection_stratum"] = "global_order_97_participants"
    b23_global_selected["result_source"] = "2026-08-27 participant-order HVE fits"
    selected = pd.concat(
        [selected, b23_selected, b23_global_selected], ignore_index=True, sort=False
    )
    selected["_dataset_order"] = selected["dataset_id"].map(DATASET_ORDER)
    selected["_stratum_order"] = selected["selection_stratum"].map(
        {"all_exposure_participants": 0, "order_independent_168_participants": 0,
         "global_order_97_participants": 1}
    )
    selected = selected.sort_values(
        ["_dataset_order", "_stratum_order", "variant"]
    ).drop(columns=["_dataset_order", "_stratum_order"])
    selected["selection_rule"] = "minimum three-fold held-out mean log loss from M_predictor"
    selected["scope_note"] = selected["selection_stratum"].map(
        {
            "all_exposure_participants": "all modelable HVE measures in the reviewed candidate inventory",
            "order_independent_168_participants": "14 order-independent HVE measures x 18 layers",
            "global_order_97_participants": "global order-sensitive HVE x 18 layers",
        }
    )
    return candidates, selected


def _save(figure: plt.Figure, prefix: Path) -> None:
    figure.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def _selection_figure(profiles: pd.DataFrame, selected: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True)
    x = np.arange(len(LAYERS))
    for axis, dataset in zip(axes, DATASETS):
        for variant in VARIANTS:
            values = (
                profiles.loc[
                    profiles["dataset_id"].eq(dataset) & profiles["variant"].eq(variant)
                ]
                .set_index("layer")
                .reindex(LAYERS)
            )
            axis.plot(
                x, values["M_predictor"], marker="o", markersize=4,
                linewidth=1.7, color=COLORS[variant], label=variant,
            )
            winner = selected.loc[
                selected["dataset_id"].eq(dataset) & selected["variant"].eq(variant)
            ].iloc[0]
            winner_x = LAYERS.index(winner["layer"])
            axis.scatter(
                [winner_x], [winner["M_predictor"]], marker="*", s=180,
                color=COLORS[variant], edgecolor="black", linewidth=0.6, zorder=4,
            )
        axis.set_title(dataset, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(LAYERS, rotation=58, ha="right", fontsize=8)
        axis.set_xlabel("HuBERT layer")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out mean log loss of $M_{predictor}$\n(lower is better)")
    axes[0].legend(title="HuBERT", frameon=False)
    figure.suptitle(
        "Global exposure-order HVE: predictor-only cross-validated layer selection\n"
        "Stars mark the selected layer; data-driven layer selection is exploratory",
        fontweight="bold", y=1.04,
    )
    figure.tight_layout()
    _save(figure, destination)


def _sbi_selection_figure(source: pd.DataFrame, selected: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True)
    x = np.arange(len(LAYERS))
    baseline_styles = {
        "mfcc39": ("MFCC39", "#7f7f7f", "--"),
        "strf24_legacy": ("STRF24", "#9467bd", ":"),
    }
    for axis, dataset in zip(axes, DATASETS):
        data = source.loc[source["dataset_id"].eq(dataset)]
        for variant in VARIANTS:
            values = (
                data.loc[data["family"].eq("HuBERT t-SNE") & data["variant"].eq(variant)]
                .set_index("feature_key")
                .reindex(LAYERS)
            )
            axis.plot(
                x, values["M_predictor"], marker="o", markersize=4,
                linewidth=1.7, color=COLORS[variant], label=variant,
            )
            winner = selected.loc[
                selected["dataset_id"].eq(dataset) & selected["variant"].eq(variant)
            ].iloc[0]
            winner_x = LAYERS.index(winner["feature_key"])
            axis.scatter(
                [winner_x], [winner["M_predictor"]], marker="*", s=180,
                color=COLORS[variant], edgecolor="black", linewidth=0.6, zorder=4,
            )
        for feature, (label, color, linestyle) in baseline_styles.items():
            value = data.loc[data["feature_key"].eq(feature), "M_predictor"]
            if len(value) == 1:
                axis.axhline(
                    float(value.iloc[0]), color=color, linestyle=linestyle,
                    linewidth=1.4, label=label,
                )
        axis.set_title(dataset, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(LAYERS, rotation=58, ha="right", fontsize=8)
        axis.set_xlabel("HuBERT layer")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out mean log loss of $M_{predictor}$\n(lower is better)")
    axes[0].legend(title="Representation", frameon=False, fontsize=8)
    figure.suptitle(
        "SBI representation selection using the predictor-only GLMM\n"
        "Acoustic baselines are references, not HuBERT layers",
        fontweight="bold", y=1.04,
    )
    figure.tight_layout()
    _save(figure, destination)


def _comparison_figure(selected: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), sharey=True)
    labels = ("Predictor beyond\ncondition", "Condition beyond\npredictor")
    columns = (
        "predictor_beyond_condition_oof_gain",
        "condition_beyond_predictor_oof_gain",
    )
    offsets = {"base": -0.18, "ft": 0.18}
    for axis, dataset in zip(axes, DATASETS):
        subset = selected.loc[selected["dataset_id"].eq(dataset)]
        for variant in VARIANTS:
            row = subset.loc[subset["variant"].eq(variant)].iloc[0]
            values = np.asarray([row[column] for column in columns], dtype=float)
            low = np.asarray(
                [
                    row["predictor_beyond_condition_oof_ci_low"],
                    row["condition_beyond_predictor_oof_ci_low"],
                ],
                dtype=float,
            )
            high = np.asarray(
                [
                    row["predictor_beyond_condition_oof_ci_high"],
                    row["condition_beyond_predictor_oof_ci_high"],
                ],
                dtype=float,
            )
            axis.bar(
                np.arange(2) + offsets[variant], values,
                width=0.34, color=COLORS[variant], label=variant,
                yerr=np.vstack([values - low, high - values]), capsize=3,
            )
            for position, value in zip(np.arange(2) + offsets[variant], values):
                axis.annotate(
                    f"{value:.4f}", xy=(position, value), xytext=(0, 4 if value >= 0 else -11),
                    textcoords="offset points", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=7,
                )
        axis.axhline(0, color="black", linewidth=0.9)
        axis.set_title(dataset, fontweight="bold")
        axis.set_xticks(np.arange(2))
        axis.set_xticklabels(labels, fontsize=9)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out mean log-loss reduction\n(positive = joint model predicts better)")
    axes[0].legend(title="HuBERT", frameon=False)
    figure.suptitle(
        "Downstream comparisons for predictor-only selected candidates\n"
        "Error bars are paired participant-cluster bootstrap 95% CIs",
        fontweight="bold", y=1.02,
    )
    figure.tight_layout()
    _save(figure, destination)


def _hve_selection_heatmap(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    dataset: str,
    destination: Path,
) -> None:
    measures = sorted(candidates["measure"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(18, 8.2), sharex=True, sharey=True)
    matrices: dict[str, pd.DataFrame] = {}
    for variant in VARIANTS:
        matrix = (
            candidates.loc[candidates["variant"].eq(variant)]
            .pivot(index="measure", columns="layer", values="M_predictor")
            .reindex(index=measures, columns=LAYERS)
        )
        matrices[variant] = matrix - float(matrix.min().min())
    vmax = max(float(matrix.max().max()) for matrix in matrices.values())
    for axis, variant in zip(axes, VARIANTS):
        matrix = matrices[variant]
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        winner = selected.loc[selected["variant"].eq(variant)].iloc[0]
        axis.scatter(
            [LAYERS.index(winner["layer"])], [measures.index(winner["measure"])],
            marker="*", s=210, color="white", edgecolor="black", linewidth=0.8,
        )
        axis.set_title(f"{dataset} HuBERT {variant}", fontweight="bold")
        axis.set_xticks(np.arange(len(LAYERS)))
        axis.set_xticklabels(LAYERS, rotation=58, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(measures)))
        axis.set_yticklabels(measures, fontsize=8)
        axis.set_xlabel("HuBERT layer")
    axes[0].set_ylabel("HVE measure")
    colorbar_axis = figure.add_axes([0.9, 0.18, 0.015, 0.68])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Excess held-out mean log loss above the best candidate")
    figure.suptitle(
        f"{dataset} HVE selection with predictor-only cross-validation\n"
        "White stars mark selected candidates; every displayed cell uses the same held-out observations",
        fontweight="bold", y=0.99,
    )
    figure.subplots_adjust(left=0.17, right=0.87, bottom=0.18, top=0.88, wspace=0.08)
    _save(figure, destination)


def _b23_comparison_figure(selected: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    labels = ("Predictor beyond\ncondition", "Condition beyond\npredictor")
    columns = (
        "predictor_beyond_condition_oof_gain",
        "condition_beyond_predictor_oof_gain",
    )
    offsets = {"base": -0.18, "ft": 0.18}
    for variant in VARIANTS:
        row = selected.loc[selected["variant"].eq(variant)].iloc[0]
        values = np.asarray([row[column] for column in columns], dtype=float)
        low = np.asarray([
            row["predictor_beyond_condition_oof_ci_low"],
            row["condition_beyond_predictor_oof_ci_low"],
        ], dtype=float)
        high = np.asarray([
            row["predictor_beyond_condition_oof_ci_high"],
            row["condition_beyond_predictor_oof_ci_high"],
        ], dtype=float)
        axis.bar(
            np.arange(2) + offsets[variant], values, width=0.34,
            color=COLORS[variant], label=variant,
            yerr=np.vstack([values - low, high - values]), capsize=3,
        )
    axis.axhline(0, color="black", linewidth=0.9)
    axis.set_xticks(np.arange(2))
    axis.set_xticklabels(labels)
    axis.set_ylabel("Held-out mean log-loss reduction\n(positive = joint model predicts better)")
    axis.set_title(
        "B23 selected order-independent HVE: downstream comparisons\n"
        "Paired participant-cluster bootstrap 95% CIs",
        fontweight="bold",
    )
    axis.legend(title="HuBERT", frameon=False)
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save(figure, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=args.overwrite)
    tables.mkdir(parents=True, exist_ok=args.overwrite)
    profiles, selected = _collect(project)
    sbi_source, sbi_selected = _collect_sbi_selection(project)
    complete_candidates, complete_selected = _collect_an19_x21_hve(project)
    b23_candidates, b23_selected = _collect_b23_order_independent(project)
    hve_candidates, hve_selected = _collect_hve_selection(
        profiles, selected, complete_candidates, complete_selected,
        b23_candidates, b23_selected,
    )
    profiles.to_csv(tables / "order_sensitive_hve_layer_profiles.csv", index=False)
    selected.to_csv(tables / "order_sensitive_hve_selected_models.csv", index=False)
    sbi_selected.to_csv(tables / "sbi_predictor_only_selected_layers.csv", index=False)
    hve_candidates.to_csv(tables / "hve_predictor_only_candidate_scores.csv", index=False)
    hve_selected.to_csv(tables / "hve_predictor_only_selected_methods.csv", index=False)
    b23_candidates.to_csv(
        tables / "b23_order_independent_hve_candidate_scores.csv", index=False
    )
    b23_selected.to_csv(
        tables / "b23_order_independent_hve_selected_models.csv", index=False
    )
    _sbi_selection_figure(
        sbi_source, sbi_selected, figures / "figure_01_sbi_predictor_only_selection"
    )
    _comparison_figure(
        sbi_selected, figures / "figure_02_sbi_selected_downstream_comparisons"
    )
    _selection_figure(profiles, selected, figures / "figure_03a_order_sensitive_hve_selection")
    _comparison_figure(selected, figures / "figure_03b_order_sensitive_hve_comparisons")
    _hve_selection_heatmap(
        b23_candidates, b23_selected,
        dataset="B23",
        destination=figures / "figure_03c_b23_order_independent_hve_selection",
    )
    _b23_comparison_figure(
        b23_selected,
        figures / "figure_03d_b23_order_independent_hve_comparisons",
    )
    for figure_id, dataset in (("03e", "AN19"), ("03f", "X21")):
        _hve_selection_heatmap(
            complete_candidates.loc[complete_candidates["dataset_id"].eq(dataset)],
            complete_selected.loc[complete_selected["dataset_id"].eq(dataset)],
            dataset=dataset,
            destination=figures / f"figure_{figure_id}_{dataset.lower()}_complete_hve_selection",
        )
    comparable_selected = pd.concat(
        [complete_selected, b23_selected], ignore_index=True, sort=False
    )
    _comparison_figure(
        comparable_selected,
        figures / "figure_03g_selected_hve_downstream_comparisons",
    )
    sbi_rows = "\n".join(
        f"| {row.dataset_id} | {row.variant} | {row.feature_key} | {row.M_predictor:.6f} |"
        for row in sbi_selected.itertuples(index=False)
    )
    hve_rows = "\n".join(
        f"| {row.dataset_id} | {row.selection_stratum} | {row.variant} | {row.measure} | {row.layer} | {row.M_predictor:.6f} |"
        for row in hve_selected.itertuples(index=False)
    )
    readme = f"""# Model-selection and global exposure-order HVE update

This report adds the newly specified order-sensitive HVE predictor. Complete exposure tokens are concatenated in actual presentation order, and adjacent-frame transitions include token boundaries.

- Figure 01 corrects SBI representation selection: HuBERT layers are ranked by held-out `M_predictor` log loss. MFCC39 and STRF24 are shown only as reference lines.
- Figure 02 evaluates the six selected SBI predictors in both downstream held-out comparisons, with paired participant-cluster bootstrap intervals.
- Figure 03a selects the HuBERT layer using held-out log loss from the predictor-only GLMM. Lower is better; condition is not used for selection. Because the same folds select and summarize the layer, this layer choice is exploratory.
- Figure 03b evaluates the selected predictor in the two downstream comparisons. Positive bars mean that the joint model has lower held-out loss than the reduced model. Error bars are paired participant-cluster bootstrap 95% intervals (10,000 resamples).
- Figure 03c shows the revised B23 order-independent HVE selection across 14 measures and 18 layers. Every cell uses the same 168 trained participants.
- Figure 03d evaluates the two selected B23 order-independent predictors in the downstream comparisons.
- Figures 03e and 03f show the complete revised AN19 and X21 HVE candidate searches.
- Figure 03g compares the downstream held-out results for the selected AN19, X21, and B23 comparable-sample predictors.

`hve_predictor_only_selected_methods.csv` applies the predictor-only criterion only within comparable participant sets. All AN19 and X21 HVE candidates were rebuilt from the revised actual-exposure tables. B23 is reported in two strata: 14 order-independent measures using all 168 trained participants, and global order-sensitive HVE using the 97 participants with recoverable presentation order. These strata are not ranked against each other because their held-out observations differ.

The revised HVE search contains 1,404 feature candidates. Its 5,616 predictor-only full/fold fits completed without failed, singular, or non-converged fits. All selected SBI and HVE downstream fits also converged and were non-singular.

## SBI layers selected by predictor-only held-out loss

| Dataset | Variant | Layer | Mean log loss |
|---|---|---:|---:|
{sbi_rows}

For all six selected SBI candidates, the participant-bootstrap interval for predictor beyond condition is above zero. The corresponding interval for condition beyond predictor includes zero in every case. These intervals condition on a layer selected from the same folds, so this is strong exploratory evidence rather than a selection-adjusted confirmatory test.

## HVE candidates selected by predictor-only held-out loss

| Dataset | Comparable selection set | Variant | Measure | Layer | Mean log loss |
|---|---|---|---|---:|---:|
{hve_rows}

For the new global order-sensitive HVE, every paired participant-bootstrap interval in Figure 03b includes zero. The current held-out evidence therefore does not establish a stable incremental benefit beyond condition. B23 fine-tuned has a positive point estimate, but its interval also crosses zero.

For the B23 order-independent winners, the joint model has slightly higher held-out loss than the condition-only model in both variants. The full-data likelihood-ratio result and the held-out predictive result therefore point in different directions; the report treats the held-out comparison as the predictive evidence.

AN19 and X21 include every analyzed participant with exposure. B23 order-independent HVE includes all 168 trained participants. Its global order-sensitive analysis includes the 97 participants whose public trial indices uniquely recover all 60 presentation positions. The remaining 71 are not assigned a guessed order.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    metadata = {
        "status": "complete",
        "analysis": "predictor_only_selection_and_downstream_model_comparisons",
        "selection_rule": "minimum three-fold held-out total log loss from M_predictor",
        "datasets": list(DATASETS),
        "variants": list(VARIANTS),
        "selected_sbi_models": sbi_selected[["dataset_id", "variant", "feature_key"]].to_dict("records"),
        "selected_hve_models": hve_selected[
            ["dataset_id", "selection_stratum", "variant", "feature_key"]
        ].to_dict("records"),
    }
    (output / "provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
