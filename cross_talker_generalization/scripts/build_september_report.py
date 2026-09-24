from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("AN19", "X21", "B23")
COLORS = {
    "MFCC39": "#777777",
    "STRF24": "#8f63b8",
    "HuBERT base": "#2878b5",
    "HuBERT ASR-FT": "#d43f3a",
    "base": "#2878b5",
    "ft": "#d43f3a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cross_talker_generalization/analysis/model_comparison/reference_checks"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=230519)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_specs(repository: Path) -> list[dict[str, object]]:
    root = repository / "cross_talker_generalization"
    derived = root / "artifacts" / "derived"
    models = root / "artifacts" / "models"
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for variant in ("base", "ft"):
            store = f"{dataset}_hubert_{variant}_tsne"
            run_id = f"{dataset}-{store}-confirmatory-tr24-20260906"
            rows.append(
                {
                    "dataset_id": dataset,
                    "family": "SBI",
                    "variant": variant,
                    "method": "HuBERT base" if variant == "base" else "HuBERT ASR-FT",
                    "model_dir": models / run_id,
                    "input_path": derived / f"{run_id}-model-input.csv",
                }
            )
        acoustic_id = f"{dataset}-{dataset}_acoustic-confirmatory-full-20260906"
        rows.append(
            {
                "dataset_id": dataset,
                "family": "SBI",
                "variant": "acoustic",
                "method": "acoustic",
                "model_dir": models / acoustic_id,
                "input_path": derived / f"{acoustic_id}-model-input.csv",
            }
        )
        for variant in ("base", "ft"):
            run_id = f"{dataset}-HVE-{variant}-tr24-20260906"
            rows.append(
                {
                    "dataset_id": dataset,
                    "family": "HVE",
                    "variant": variant,
                    "method": variant,
                    "model_dir": models / run_id,
                    "input_path": derived / f"{run_id}-model-input.csv",
                }
            )
    diagnostic_id = "AN19-AN19_acoustic_diagnostic-confirmatory"
    rows.append(
        {
            "dataset_id": "AN19",
            "family": "ACOUSTIC_DIAGNOSTIC",
            "variant": "diagnostic",
            "method": "diagnostic",
            "model_dir": models / diagnostic_id,
            "input_path": derived / f"{diagnostic_id}-model-input.csv",
        }
    )
    return rows


def require_run(spec: dict[str, object]) -> None:
    model_dir = Path(spec["model_dir"])
    input_path = Path(spec["input_path"])
    required = (
        model_dir / "oof_predictions.csv",
        model_dir / "cv_metrics.csv",
        model_dir / "coefficients.csv",
        model_dir / "diagnostics.csv",
        input_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("run is incomplete:\n" + "\n".join(missing))


def fold_bootstrap(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def participant_bootstrap(
    participant: pd.DataFrame,
    numerator_columns: tuple[str, ...],
    denominator_column: str,
    rng: np.random.Generator,
    n_boot: int,
    ratio: bool = False,
) -> tuple[float, float, float]:
    numerator = participant.loc[:, numerator_columns].sum().to_numpy(float)
    denominator = float(participant[denominator_column].sum())
    point_values = numerator / denominator
    point = float(point_values[0] / point_values[1] * 100.0) if ratio else float(point_values[0])
    boot_numerator = np.zeros((n_boot, len(numerator_columns)), dtype=float)
    boot_denominator = np.zeros(n_boot, dtype=float)
    for _, frame in participant.groupby("fold", sort=True):
        values = frame.loc[:, numerator_columns].to_numpy(float)
        denominators = frame[denominator_column].to_numpy(float)
        sampled = rng.integers(0, len(frame), size=(n_boot, len(frame)))
        boot_numerator += values[sampled].sum(axis=1)
        boot_denominator += denominators[sampled].sum(axis=1)
    scaled = boot_numerator / boot_denominator[:, None]
    boot = scaled[:, 0] / scaled[:, 1] * 100.0 if ratio else scaled[:, 0]
    low, high = np.quantile(boot[np.isfinite(boot)], [0.025, 0.975])
    return point, float(low), float(high)


def wide_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "dataset_id",
        "feature_key",
        "fold",
        "analysis_row_id",
        "participant_id",
        "analysis_item_id",
        "condition_id",
        "response_correct",
        "response_incorrect",
    ]
    wide = predictions.pivot(index=identifiers, columns="model_id", values="log_loss").reset_index()
    wide.columns.name = None
    required = {"M_null", "M_condition", "M_predictor", "M_joint"}
    missing = required.difference(wide.columns)
    if missing:
        raise ValueError(f"OOF predictions lack models {sorted(missing)}")
    wide["n_trials"] = wide["response_correct"] + wide["response_incorrect"]
    return wide


def attach_ceiling(
    wide: pd.DataFrame,
    model_input: pd.DataFrame,
    ceiling: pd.DataFrame,
    feature_key: str,
) -> pd.DataFrame:
    source = model_input.loc[model_input["feature_key"].astype(str).eq(feature_key)].copy()
    source["analysis_row_id"] = np.arange(1, len(source) + 1)
    source = source[["analysis_row_id", "behavior_item_id", "response_expected"]]
    joined = wide.merge(
        source,
        on="analysis_row_id",
        how="left",
        validate="one_to_one",
    )
    if joined["behavior_item_id"].isna().any():
        raise ValueError(f"could not recover behavior items for {feature_key}")
    ceiling_rows = ceiling[
        ["participant_id", "fold", "item_id", "response_expected", "log_loss"]
    ].rename(
        columns={"item_id": "behavior_item_id", "log_loss": "ceiling_log_loss"}
    )
    joined = joined.merge(
        ceiling_rows,
        on=["participant_id", "fold", "behavior_item_id", "response_expected"],
        how="left",
        validate="many_to_one",
    )
    if joined["ceiling_log_loss"].isna().any():
        raise ValueError(f"ceiling rows do not match {feature_key}")
    return joined


def summarize_feature(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> list[dict[str, float | str]]:
    comparisons = {
        "predictor_only": ("M_null", "M_predictor"),
        "predictor_beyond_condition": ("M_condition", "M_joint"),
        "condition_beyond_predictor": ("M_predictor", "M_joint"),
    }
    rows: list[dict[str, float | str]] = []
    for comparison, (reduced, full) in comparisons.items():
        per_participant = (
            frame.assign(gain=frame[reduced] - frame[full])
            .groupby(["fold", "participant_id"], as_index=False)
            .agg(gain=("gain", "sum"), n_trials=("n_trials", "sum"))
        )
        point, low, high = participant_bootstrap(
            per_participant, ("gain",), "n_trials", rng, n_boot
        )
        rows.append(
            {
                "comparison": comparison,
                "gain": point,
                "ci_low": low,
                "ci_high": high,
            }
        )
    ratio_participant = (
        frame.assign(
            predictor_gain=frame["M_null"] - frame["M_predictor"],
            ceiling_gain=frame["M_null"] - frame["ceiling_log_loss"],
        )
        .groupby(["fold", "participant_id"], as_index=False)
        .agg(
            predictor_gain=("predictor_gain", "sum"),
            ceiling_gain=("ceiling_gain", "sum"),
            n_trials=("n_trials", "sum"),
        )
    )
    point, low, high = participant_bootstrap(
        ratio_participant,
        ("predictor_gain", "ceiling_gain"),
        "n_trials",
        rng,
        n_boot,
        ratio=True,
    )
    ceiling_by_fold = (
        frame.assign(ceiling_gain=frame["M_null"] - frame["ceiling_log_loss"])
        .groupby("fold", as_index=False)
        .agg(ceiling_gain=("ceiling_gain", "sum"), n_trials=("n_trials", "sum"))
    )
    ceiling_folds = (
        ceiling_by_fold["ceiling_gain"] / ceiling_by_fold["n_trials"]
    ).to_numpy(float)
    ceiling_folds = ceiling_folds / ceiling_folds.mean() * 100.0
    ceiling_low, ceiling_high = fold_bootstrap(
        ceiling_folds, rng, max(n_boot, 5000)
    )
    rows.append(
        {
            "comparison": "predictor_only_percent_ceiling",
            "gain": point,
            "ci_low": low,
            "ci_high": high,
            "ceiling_ci_low": ceiling_low,
            "ceiling_ci_high": ceiling_high,
        }
    )
    return rows


def collect(repository: Path, specs: list[dict[str, object]], n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    metric_rows: list[dict[str, object]] = []
    z_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    input_paths: list[Path] = []
    for spec in specs:
        require_run(spec)
        model_dir = Path(spec["model_dir"])
        input_path = Path(spec["input_path"])
        input_paths.extend(
            [
                model_dir / "oof_predictions.csv",
                model_dir / "cv_metrics.csv",
                model_dir / "coefficients.csv",
                model_dir / "diagnostics.csv",
                input_path,
            ]
        )
        predictions = pd.read_csv(model_dir / "oof_predictions.csv")
        model_input = pd.read_csv(input_path)
        coefficients = pd.read_csv(model_dir / "coefficients.csv")
        metrics = pd.read_csv(model_dir / "cv_metrics.csv")
        ceiling_path = (
            repository
            / "cross_talker_generalization"
            / "analysis/model_comparison/reference_checks"
            / "ceilings"
            / str(spec["dataset_id"])
            / "oof_predictions.csv"
        )
        ceiling = pd.read_csv(ceiling_path)
        input_paths.append(ceiling_path)
        wide = wide_predictions(predictions)
        for feature_key, feature in wide.groupby("feature_key", sort=True):
            feature = attach_ceiling(feature, model_input, ceiling, str(feature_key))
            for summary in summarize_feature(feature, rng, n_boot):
                metric_rows.append(
                    {
                        **{key: spec[key] for key in ("dataset_id", "family", "variant", "method")},
                        "feature_key": feature_key,
                        **summary,
                        "n_rows": int(len(feature)),
                        "n_participants": int(feature["participant_id"].nunique()),
                        "total_trials": int(feature["n_trials"].sum()),
                    }
                )
            if spec["family"] == "ACOUSTIC_DIAGNOSTIC":
                for condition, condition_frame in feature.groupby("condition_id", sort=True):
                    for comparison, reduced, full in (
                        ("predictor_only", "M_null", "M_predictor"),
                        ("predictor_beyond_condition", "M_condition", "M_joint"),
                    ):
                        participant = (
                            condition_frame.assign(gain=condition_frame[reduced] - condition_frame[full])
                            .groupby(["fold", "participant_id"], as_index=False)
                            .agg(gain=("gain", "sum"), n_trials=("n_trials", "sum"))
                        )
                        point, low, high = participant_bootstrap(
                            participant, ("gain",), "n_trials", rng, n_boot
                        )
                        condition_rows.append(
                            {
                                "dataset_id": spec["dataset_id"],
                                "feature_key": feature_key,
                                "condition_id": condition,
                                "comparison": comparison,
                                "gain": point,
                                "ci_low": low,
                                "ci_high": high,
                            }
                        )
        term = "variability_z" if spec["family"] == "HVE" else "similarity_z"
        for feature_key in sorted(coefficients["feature_key"].astype(str).unique()):
            full = coefficients.loc[
                coefficients["feature_key"].astype(str).eq(feature_key)
                & coefficients["scope"].eq("full")
                & coefficients["model_id"].eq("M_predictor")
                & coefficients["term"].eq(term)
            ]
            folds = coefficients.loc[
                coefficients["feature_key"].astype(str).eq(feature_key)
                & coefficients["scope"].eq("cv_train")
                & coefficients["model_id"].eq("M_predictor")
                & coefficients["term"].eq(term),
                "z_value",
            ].to_numpy(float)
            if len(full) != 1 or len(folds) != 3:
                raise ValueError(f"unexpected z inventory for {model_dir.name}/{feature_key}")
            low, high = fold_bootstrap(folds, rng, max(n_boot, 5000))
            oof_loss = metrics.loc[
                metrics["feature_key"].astype(str).eq(feature_key)
                & metrics["scope"].eq("oof_all")
                & metrics["model_id"].eq("M_predictor"),
                "total_log_loss",
            ]
            z_rows.append(
                {
                    **{key: spec[key] for key in ("dataset_id", "family", "variant", "method")},
                    "feature_key": feature_key,
                    "full_z": float(full.iloc[0]["z_value"]),
                    "full_estimate": float(full.iloc[0]["estimate"]),
                    "full_conf_low": float(full.iloc[0]["conf_low"]),
                    "full_conf_high": float(full.iloc[0]["conf_high"]),
                    "mean_cv_train_z": float(np.mean(folds)),
                    "cv_train_z_ci_low": low,
                    "cv_train_z_ci_high": high,
                    "predictor_oof_total_log_loss": float(oof_loss.iloc[0]),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(z_rows), pd.DataFrame(condition_rows), input_paths


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / "figures" / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def errorbar_points(axis, data: pd.DataFrame, x_column: str, y_column: str, order: list[str], colors):
    for index, label in enumerate(order):
        row = data.loc[data[x_column].eq(label)]
        if row.empty:
            continue
        row = row.iloc[0]
        axis.errorbar(
            index,
            row[y_column],
            yerr=[[row[y_column] - row["ci_low"]], [row["ci_high"] - row[y_column]]],
            fmt="o",
            color=colors.get(label, "#333333"),
            capsize=3,
            markersize=7,
            linewidth=1.5,
        )
    axis.set_xticks(range(len(order)), order, rotation=35, ha="right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)


def plot_sbi(metrics: pd.DataFrame, output: Path) -> None:
    sbi = metrics.loc[
        metrics["family"].eq("SBI")
        & metrics["comparison"].eq("predictor_only_percent_ceiling")
    ].copy()
    sbi["display"] = np.where(
        sbi["variant"].eq("acoustic"),
        sbi["feature_key"].map({"mfcc39": "MFCC39", "strf24_legacy": "STRF24"}),
        sbi["method"],
    )
    order = ["MFCC39", "STRF24", "HuBERT base", "HuBERT ASR-FT"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=False)
    for axis, dataset in zip(axes, DATASETS):
        frame = sbi.loc[sbi["dataset_id"].eq(dataset)]
        ceiling = frame.iloc[0]
        axis.axhspan(
            ceiling["ceiling_ci_low"], ceiling["ceiling_ci_high"],
            color="#d8d8d8", alpha=0.7, zorder=0,
        )
        axis.axhline(100, color="#444444", linestyle="--", linewidth=1)
        axis.axhline(0, color="#888888", linestyle=":", linewidth=1)
        errorbar_points(axis, frame, "display", "gain", order, COLORS)
        axis.set_title(dataset, fontweight="bold")
        axis.set_ylabel("Predictor-only OOF gain (% of behavioral ceiling)")
    fig.suptitle("Fixed Tr-24 SBI prediction; acoustic baselines shown first", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output, "figure_01_fixed_tr24_sbi_predictor_gain")
    sbi.to_csv(output / "tables" / "figure_01_fixed_tr24_sbi_predictor_gain.csv", index=False)


def plot_downstream(metrics: pd.DataFrame, output: Path) -> None:
    frame = metrics.loc[
        metrics["family"].eq("SBI")
        & metrics["comparison"].isin(["predictor_beyond_condition", "condition_beyond_predictor"])
    ].copy()
    frame["display"] = np.where(
        frame["variant"].eq("acoustic"),
        frame["feature_key"].map({"mfcc39": "MFCC39", "strf24_legacy": "STRF24"}),
        frame["method"],
    )
    order = ["MFCC39", "STRF24", "HuBERT base", "HuBERT ASR-FT"]
    comparison_labels = {
        "predictor_beyond_condition": "Predictor beyond condition",
        "condition_beyond_predictor": "Condition beyond predictor",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for axis, dataset in zip(axes, DATASETS):
        data = frame.loc[frame["dataset_id"].eq(dataset)]
        offsets = {"predictor_beyond_condition": -0.11, "condition_beyond_predictor": 0.11}
        colors = {"predictor_beyond_condition": "#247a3c", "condition_beyond_predictor": "#cc6b20"}
        for comparison in comparison_labels:
            subset = data.loc[data["comparison"].eq(comparison)]
            for index, label in enumerate(order):
                row = subset.loc[subset["display"].eq(label)]
                if row.empty:
                    continue
                row = row.iloc[0]
                x = index + offsets[comparison]
                axis.errorbar(
                    x,
                    row["gain"],
                    yerr=[[row["gain"] - row["ci_low"]], [row["ci_high"] - row["gain"]]],
                    fmt="o",
                    color=colors[comparison],
                    capsize=3,
                    label=comparison_labels[comparison] if index == 0 else None,
                )
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(order)), order, rotation=35, ha="right")
        axis.set_title(dataset, fontweight="bold")
        axis.set_ylabel("Participant-held-out log-loss gain per response")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Fixed Tr-24 SBI: distinct information in predictor and condition", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output, "figure_02_fixed_tr24_sbi_downstream_comparisons")
    frame.to_csv(output / "tables" / "figure_02_fixed_tr24_sbi_downstream_comparisons.csv", index=False)


def hve_measure(feature_key: str) -> str:
    return feature_key.split("::", 1)[1] if "::" in feature_key else feature_key


def plot_hve(metrics: pd.DataFrame, z_values: pd.DataFrame, output: Path) -> pd.DataFrame:
    hve = metrics.loc[
        metrics["family"].eq("HVE")
        & metrics["comparison"].eq("predictor_only_percent_ceiling")
    ].copy()
    hve["measure"] = hve["feature_key"].map(hve_measure)
    z_hve = z_values.loc[z_values["family"].eq("HVE")].copy()
    z_hve["measure"] = z_hve["feature_key"].map(hve_measure)
    selections = []
    for keys, group in z_hve.groupby(["dataset_id", "variant"], sort=True):
        loss_row = group.sort_values(["predictor_oof_total_log_loss", "feature_key"]).iloc[0]
        z_row = group.sort_values(["mean_cv_train_z", "feature_key"], ascending=[False, True]).iloc[0]
        for objective, row in (("OOF likelihood", loss_row), ("training-fold z", z_row)):
            selections.append(
                {
                    "dataset_id": keys[0],
                    "variant": keys[1],
                    "selection_objective": objective,
                    "feature_key": row["feature_key"],
                    "measure": row["measure"],
                }
            )
    selection = pd.DataFrame(selections)
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=False)
    for axis, dataset in zip(axes, DATASETS):
        data = hve.loc[hve["dataset_id"].eq(dataset)]
        order = sorted(data["measure"].unique())
        axis.axhspan(
            data["ceiling_ci_low"].min(), data["ceiling_ci_high"].max(),
            color="#d8d8d8", alpha=0.6, zorder=0,
        )
        for offset, variant in ((-0.12, "base"), (0.12, "ft")):
            subset = data.loc[data["variant"].eq(variant)]
            for index, measure in enumerate(order):
                row = subset.loc[subset["measure"].eq(measure)]
                if row.empty:
                    continue
                row = row.iloc[0]
                axis.errorbar(
                    index + offset,
                    row["gain"],
                    yerr=[[row["gain"] - row["ci_low"]], [row["ci_high"] - row["gain"]]],
                    fmt="o",
                    color=COLORS[variant],
                    capsize=2,
                    label=variant if index == 0 else None,
                )
        axis.axhline(0, color="#777777", linestyle=":", linewidth=1)
        axis.axhline(100, color="#444444", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(order)), [value.replace("_", " ") for value in order], rotation=35, ha="right")
        axis.set_ylabel("Predictor-only OOF gain\n(% of behavioral ceiling)")
        axis.set_title(dataset, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(title="HuBERT", frameon=False)
    fig.suptitle("Fixed Tr-24 HVE definitions", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output, "figure_03_fixed_tr24_hve_methods")

    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=False)
    for axis, dataset in zip(axes, DATASETS):
        data = hve.loc[hve["dataset_id"].eq(dataset)]
        order = sorted(data["measure"].unique())
        for offset, variant in ((-0.12, "base"), (0.12, "ft")):
            subset = data.loc[data["variant"].eq(variant)]
            for index, measure in enumerate(order):
                row = subset.loc[subset["measure"].eq(measure)]
                if row.empty:
                    continue
                row = row.iloc[0]
                axis.errorbar(
                    index + offset,
                    row["gain"],
                    yerr=[[row["gain"] - row["ci_low"]], [row["ci_high"] - row["gain"]]],
                    fmt="o",
                    color=COLORS[variant],
                    capsize=2,
                    label=variant if index == 0 else None,
                )
        low = float(data["ci_low"].min())
        high = float(data["ci_high"].max())
        margin = max((high - low) * 0.12, 0.2)
        axis.set_ylim(low - margin, high + margin)
        axis.axhline(0, color="#777777", linestyle=":", linewidth=1)
        axis.set_xticks(range(len(order)), [value.replace("_", " ") for value in order], rotation=35, ha="right")
        axis.set_ylabel("Predictor-only OOF gain\n(% of behavioral ceiling)")
        axis.set_title(dataset, fontweight="bold")
        axis.text(
            0.995,
            0.97,
            "100% behavioral ceiling is off scale",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#666666",
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(title="HuBERT", frameon=False)
    fig.suptitle("Fixed Tr-24 HVE definitions — zoomed result range", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output, "figure_03b_fixed_tr24_hve_methods_zoomed")
    hve.to_csv(output / "tables" / "figure_03_fixed_tr24_hve_methods.csv", index=False)
    selection.to_csv(output / "tables" / "hve_selection_by_objective.csv", index=False)
    return selection


def plot_objective_matrix(
    metrics: pd.DataFrame,
    z_values: pd.DataFrame,
    selection: pd.DataFrame,
    output: Path,
) -> None:
    metric = metrics.loc[
        metrics["family"].eq("HVE")
        & metrics["comparison"].eq("predictor_only_percent_ceiling")
    ]
    z_hve = z_values.loc[z_values["family"].eq("HVE")]
    rows = selection.merge(
        metric[["dataset_id", "variant", "feature_key", "gain", "ci_low", "ci_high"]],
        on=["dataset_id", "variant", "feature_key"],
        how="left",
        validate="many_to_one",
    ).merge(
        z_hve[
            [
                "dataset_id",
                "variant",
                "feature_key",
                "mean_cv_train_z",
                "cv_train_z_ci_low",
                "cv_train_z_ci_high",
            ]
        ],
        on=["dataset_id", "variant", "feature_key"],
        how="left",
        validate="many_to_one",
    )
    rows["cell"] = rows["dataset_id"] + "\n" + rows["variant"]
    order = [f"{dataset}\n{variant}" for dataset in DATASETS for variant in ("base", "ft")]
    objectives = ["OOF likelihood", "training-fold z"]
    objective_colors = {"OOF likelihood": "#355c9a", "training-fold z": "#b54b4b"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for objective_index, objective in enumerate(objectives):
        subset = rows.loc[rows["selection_objective"].eq(objective)]
        offset = -0.09 if objective_index == 0 else 0.09
        for index, cell in enumerate(order):
            row = subset.loc[subset["cell"].eq(cell)]
            if row.empty:
                continue
            row = row.iloc[0]
            axes[0].errorbar(
                index + offset,
                row["gain"],
                yerr=[[row["gain"] - row["ci_low"]], [row["ci_high"] - row["gain"]]],
                fmt="o",
                color=objective_colors[objective],
                capsize=3,
                label=objective if index == 0 else None,
            )
            axes[1].errorbar(
                index + offset,
                row["mean_cv_train_z"],
                yerr=[
                    [row["mean_cv_train_z"] - row["cv_train_z_ci_low"]],
                    [row["cv_train_z_ci_high"] - row["mean_cv_train_z"]],
                ],
                fmt="o",
                color=objective_colors[objective],
                capsize=3,
            )
    axes[0].axhline(0, color="#777777", linestyle=":")
    axes[0].axhline(100, color="#444444", linestyle="--")
    axes[0].set_ylabel("Reported OOF gain (% ceiling)")
    axes[0].set_title("Report as predictive gain")
    axes[1].axhline(0, color="#777777", linestyle=":")
    axes[1].set_ylabel("Reported predictor z (mean training folds)")
    axes[1].set_title("Report as fixed-effect evidence")
    for axis in axes:
        axis.set_xticks(range(len(order)), order)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(title="Selection objective", frameon=False)
    fig.suptitle("HVE optimization objective × reporting metric (fixed Tr-24)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output, "figure_04_hve_objective_reporting_matrix")

    fig, axis = plt.subplots(figsize=(10, 5.2))
    for objective_index, objective in enumerate(objectives):
        subset = rows.loc[rows["selection_objective"].eq(objective)]
        offset = -0.09 if objective_index == 0 else 0.09
        for index, cell in enumerate(order):
            row = subset.loc[subset["cell"].eq(cell)]
            if row.empty:
                continue
            row = row.iloc[0]
            axis.errorbar(
                index + offset,
                row["gain"],
                yerr=[[row["gain"] - row["ci_low"]], [row["ci_high"] - row["gain"]]],
                fmt="o",
                color=objective_colors[objective],
                capsize=3,
                label=objective if index == 0 else None,
            )
    axis.axhline(0, color="#777777", linestyle=":")
    axis.set_xticks(range(len(order)), order)
    axis.set_ylabel("Reported OOF gain (% ceiling)")
    axis.set_title("HVE likelihood-versus-z selection — zoomed predictive report", fontweight="bold")
    axis.text(0.995, 0.97, "100% behavioral ceiling is off scale", transform=axis.transAxes, ha="right", va="top", fontsize=8, color="#666666")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(title="Selection objective", frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "figure_04b_hve_objective_predictive_gain_zoomed")
    rows.to_csv(output / "tables" / "figure_04_hve_objective_reporting_matrix.csv", index=False)


def plot_hve_downstream(metrics: pd.DataFrame, selection: pd.DataFrame, output: Path) -> None:
    downstream = metrics.loc[
        metrics["family"].eq("HVE")
        & metrics["comparison"].isin(["predictor_beyond_condition", "condition_beyond_predictor"])
    ].copy()
    rows = selection.merge(
        downstream,
        on=["dataset_id", "variant", "feature_key"],
        how="left",
        validate="one_to_many",
    )
    cell_order = [
        (variant, objective)
        for variant in ("base", "ft")
        for objective in ("OOF likelihood", "training-fold z")
    ]
    comparison_labels = {
        "predictor_beyond_condition": "Predictor beyond condition",
        "condition_beyond_predictor": "Condition beyond predictor",
    }
    comparison_colors = {
        "predictor_beyond_condition": "#247a3c",
        "condition_beyond_predictor": "#cc6b20",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=False)
    for axis, dataset in zip(axes, DATASETS):
        data = rows.loc[rows["dataset_id"].eq(dataset)]
        for comparison, label in comparison_labels.items():
            subset = data.loc[data["comparison"].eq(comparison)]
            offset = -0.10 if comparison == "predictor_beyond_condition" else 0.10
            for index, (variant, objective) in enumerate(cell_order):
                row = subset.loc[
                    subset["variant"].eq(variant)
                    & subset["selection_objective"].eq(objective)
                ]
                if row.empty:
                    continue
                row = row.iloc[0]
                axis.errorbar(
                    index + offset,
                    row["gain"],
                    yerr=[[row["gain"] - row["ci_low"]], [row["ci_high"] - row["gain"]]],
                    fmt="o",
                    color=comparison_colors[comparison],
                    capsize=3,
                    label=label if index == 0 else None,
                )
        axis.axhline(0, color="#666666", linestyle="--", linewidth=1)
        axis.set_xticks(
            range(len(cell_order)),
            [f"{variant}\n{'likelihood' if objective == 'OOF likelihood' else 'z'}" for variant, objective in cell_order],
        )
        axis.set_title(dataset, fontweight="bold")
        axis.set_ylabel("Participant-held-out log-loss gain per response")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Fixed Tr-24 HVE: downstream comparisons after each selection objective", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output, "figure_04c_selected_hve_downstream_comparisons")
    rows.to_csv(output / "tables" / "figure_04c_selected_hve_downstream_comparisons.csv", index=False)


def plot_acoustic(metrics: pd.DataFrame, z_values: pd.DataFrame, conditions: pd.DataFrame, output: Path):
    data = metrics.loc[
        metrics["family"].eq("ACOUSTIC_DIAGNOSTIC")
        & metrics["comparison"].eq("predictor_only_percent_ceiling")
    ].copy()
    z = z_values.loc[z_values["family"].eq("ACOUSTIC_DIAGNOSTIC")].copy()
    data = data.merge(
        z[["feature_key", "mean_cv_train_z", "cv_train_z_ci_low", "cv_train_z_ci_high"]],
        on="feature_key",
        validate="one_to_one",
    )
    order = [
        "mfcc_c0",
        "mfcc_static_spectral12",
        "mfcc_static13",
        "mfcc_delta13",
        "mfcc_delta_delta13",
        "strf_rate_2",
        "strf_rate_4",
        "strf_rate_8",
        "strf_rate_16",
        "strf_scale_0p25",
        "strf_scale_0p5",
        "strf_scale_1p0",
        "strf_direction_positive",
        "strf_direction_negative",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axes[0].axhspan(
        data["ceiling_ci_low"].min(), data["ceiling_ci_high"].max(),
        color="#d8d8d8", alpha=0.6, zorder=0,
    )
    for index, key in enumerate(order):
        row = data.loc[data["feature_key"].eq(key)]
        if row.empty:
            continue
        row = row.iloc[0]
        color = "#666666" if key.startswith("mfcc") else "#8f63b8"
        axes[0].errorbar(
            index,
            row["gain"],
            yerr=[[row["gain"] - row["ci_low"]], [row["ci_high"] - row["gain"]]],
            fmt="o",
            color=color,
            capsize=3,
        )
        axes[1].errorbar(
            index,
            row["mean_cv_train_z"],
            yerr=[
                [row["mean_cv_train_z"] - row["cv_train_z_ci_low"]],
                [row["cv_train_z_ci_high"] - row["mean_cv_train_z"]],
            ],
            fmt="o",
            color=color,
            capsize=3,
        )
    axes[0].axhline(0, color="#777777", linestyle=":")
    axes[0].axhline(100, color="#444444", linestyle="--")
    axes[0].set_ylabel("Predictor-only OOF gain (% ceiling)")
    axes[1].axhline(0, color="#777777", linestyle=":")
    axes[1].set_ylabel("Predictor z (mean training folds)")
    axes[1].set_xticks(range(len(order)), order, rotation=35, ha="right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("AN19 acoustic-baseline component audit", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output, "figure_05a_an19_acoustic_component_audit")

    fig, axis = plt.subplots(figsize=(15, 5.5))
    for index, key in enumerate(order):
        row = data.loc[data["feature_key"].eq(key)]
        if row.empty:
            continue
        row = row.iloc[0]
        color = "#666666" if key.startswith("mfcc") else "#8f63b8"
        axis.errorbar(index, row["gain"], yerr=[[row["gain"] - row["ci_low"]], [row["ci_high"] - row["gain"]]], fmt="o", color=color, capsize=3)
    axis.axhline(0, color="#777777", linestyle=":")
    axis.set_ylabel("Predictor-only OOF gain (% ceiling)")
    axis.set_xticks(range(len(order)), [value.replace("_", " ") for value in order], rotation=35, ha="right")
    axis.set_title("AN19 acoustic component audit — zoomed predictive gain", fontweight="bold")
    axis.text(0.995, 0.97, "100% behavioral ceiling is off scale", transform=axis.transAxes, ha="right", va="top", fontsize=8, color="#666666")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, output, "figure_05a2_an19_acoustic_component_gain_zoomed")
    data.to_csv(output / "tables" / "figure_05a_an19_acoustic_component_audit.csv", index=False)

    heat = conditions.loc[conditions["comparison"].eq("predictor_only")].pivot(
        index="feature_key", columns="condition_id", values="gain"
    ).reindex(order)
    fig, axis = plt.subplots(figsize=(12, 8))
    values = heat.to_numpy(float)
    limit = np.nanmax(np.abs(values))
    image = axis.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_yticks(range(len(heat.index)), heat.index)
    axis.set_xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
    axis.set_title("AN19 predictor-only OOF gain by experimental condition", fontweight="bold")
    fig.colorbar(image, ax=axis, label="Log-loss gain per response")
    fig.tight_layout()
    save_figure(fig, output, "figure_05b_an19_acoustic_gain_by_condition")
    conditions.to_csv(output / "tables" / "figure_05b_an19_acoustic_gain_by_condition.csv", index=False)


def audit_acoustic_inputs(repository: Path, output: Path) -> list[Path]:
    root = repository / "cross_talker_generalization"
    derived = root / "artifacts" / "derived"
    paths = {
        "HuBERT base Tr-24": derived / "AN19-AN19_hubert_base_tsne-confirmatory-tr24-20260906-distances.csv",
        "HuBERT ASR-FT Tr-24": derived / "AN19-AN19_hubert_ft_tsne-confirmatory-tr24-20260906-distances.csv",
        "MFCC39": derived / "AN19-AN19_acoustic-confirmatory-full-20260906-distances.csv",
        "STRF24": derived / "AN19-AN19_acoustic-confirmatory-full-20260906-distances.csv",
    }
    frames = []
    for label, path in paths.items():
        frame = pd.read_csv(path)
        if label == "MFCC39":
            frame = frame.loc[frame["feature_key"].eq("mfcc39")]
        elif label == "STRF24":
            frame = frame.loc[frame["feature_key"].eq("strf24_legacy")]
        else:
            frame = frame.loc[frame["feature_key"].eq("tr_24")]
        frames.append(frame[["pair_id", "raw_distance"]].rename(columns={"raw_distance": label}))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="pair_id", how="inner", validate="one_to_one")
    correlation = merged.drop(columns="pair_id").corr(method="spearman")
    correlation.to_csv(output / "tables" / "figure_05c_an19_distance_spearman.csv")
    merged.to_csv(output / "tables" / "figure_05c_an19_matched_pair_distances.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = correlation.columns.tolist()
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlation.iloc[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.6 else "black")
    axis.set_title("AN19 matched physical-pair distance correlations\n(Spearman, n = {:,})".format(len(merged)), fontweight="bold")
    fig.colorbar(image, ax=axis, label="Spearman correlation")
    fig.tight_layout()
    save_figure(fig, output, "figure_05c_an19_acoustic_hubert_distance_correlations")

    pairs_path = derived / "AN19-pairs" / "pairs.csv"
    pairs = pd.read_csv(pairs_path)
    input_path = derived / "AN19-AN19_acoustic-confirmatory-full-20260906-model-input.csv"
    model_input = pd.read_csv(input_path)
    audit = pd.DataFrame(
        [
            {"check": "pair rows", "value": len(pairs)},
            {"check": "duplicate pair_id rows", "value": int(pairs["pair_id"].duplicated().sum())},
            {"check": "identical test/source physical unit", "value": int((pairs["test_unit_id"] == pairs["source_unit_id"]).sum())},
            {"check": "identical test/source speaker", "value": int((pairs["test_speaker_id"] == pairs["source_speaker_id"]).sum())},
            {"check": "pairs shared by all four representations", "value": len(merged)},
        ]
    )
    audit.to_csv(output / "tables" / "an19_acoustic_pair_integrity_audit.csv", index=False)
    status = (
        model_input.groupby(["feature_key", "condition_id", "predictor_status"], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
    )
    status.to_csv(output / "tables" / "an19_acoustic_predictor_status_by_condition.csv", index=False)
    return [*set(paths.values()), pairs_path, input_path]


def summarize_diagnostics(specs: list[dict[str, object]], output: Path) -> pd.DataFrame:
    rows = []
    for spec in specs:
        model_dir = Path(spec["model_dir"])
        diagnostics = pd.read_csv(model_dir / "diagnostics.csv")
        rows.append(
            {
                "run_id": model_dir.name,
                "dataset_id": spec["dataset_id"],
                "family": spec["family"],
                "n_fits": len(diagnostics),
                "fit_ok": int(diagnostics["fit_ok"].fillna(False).sum()),
                "convergence_ok": int(diagnostics["convergence"].fillna("").eq("ok").sum()),
                "singular": int(diagnostics["singular"].fillna(False).sum()),
                "warnings": int(diagnostics["warnings"].fillna("").astype(str).str.len().gt(0).sum()),
                "errors": int(diagnostics["error"].fillna("").astype(str).str.len().gt(0).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "tables" / "run_diagnostics_summary.csv", index=False)
    return summary


def build_readme(
    output: Path,
    metrics: pd.DataFrame,
    z_values: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    sbi = metrics.loc[
        metrics["family"].eq("SBI")
        & metrics["comparison"].eq("predictor_only_percent_ceiling")
    ].copy()
    sbi["display"] = np.where(
        sbi["variant"].eq("acoustic"),
        sbi["feature_key"].map({"mfcc39": "MFCC39", "strf24_legacy": "STRF24"}),
        sbi["method"],
    )
    acoustic = metrics.loc[
        metrics["family"].eq("ACOUSTIC_DIAGNOSTIC")
        & metrics["comparison"].eq("predictor_only_percent_ceiling")
    ].sort_values("gain", ascending=False)
    top = acoustic.head(5)
    lines = [
        "# September 6 analysis update",
        "",
        "> **Review correction:** these are likelihood-based companion analyses, not the principal SBI z display or a completed manuscript figure set. See [z-value review](z_value_review/README.md) and [requirements report](../docs/FLORIAN_REQUIREMENTS_REVIEW.md). B23 HVE selection mixes 97- and 168-participant samples and needs correction. Acoustic component distances used `none` scaling while full baselines used `global_z`; their ranking is not a controlled ablation. Figure 1d and Figure 2b/d remain placeholders.",
        "",
        "This package implements the first analysis batch requested after the September meeting. All predictive gains use models fitted on two participant folds and frozen before scoring the third. `M_null` is now included, so predictor-only gain is oriented upward. Confidence intervals resample participants within folds.",
        "",
        "## Main findings",
        "",
        "The fixed Tr-24 HuBERT predictor has a positive predictor-only held-out gain in all six dataset-by-variant cells. It also adds positive held-out gain beyond condition in all six cells. By contrast, the confidence interval for condition beyond the HuBERT predictor includes zero in all six cells.",
        "",
        "No positive fixed-Tr-24 HVE predictor-only interval excludes zero. Four gain intervals are entirely negative: worse held-out prediction than the reference, not necessarily a negative predictor coefficient. Signed z is a separate statistic. The B23 winner labels in Figure 4 require comparable-sample correction before interpretation.",
        "",
        "| Dataset | Predictor | OOF gain (% ceiling) | 95% participant-cluster CI |",
        "|---|---|---:|---:|",
    ]
    for dataset in DATASETS:
        for label in ("MFCC39", "STRF24", "HuBERT base", "HuBERT ASR-FT"):
            row = sbi.loc[sbi["dataset_id"].eq(dataset) & sbi["display"].eq(label)].iloc[0]
            lines.append(f"| {dataset} | {label} | {row.gain:.2f} | [{row.ci_low:.2f}, {row.ci_high:.2f}] |")
    lines.extend(
        [
        "",
        "## Figures",
        "",
        "- `figure_01_fixed_tr24_sbi_predictor_gain`: fixed Tr-24 SBI and the two acoustic baselines, normalized to the directly cross-validated behavioral ceiling on matched rows.",
        "- `figure_02_fixed_tr24_sbi_downstream_comparisons`: predictor beyond condition and condition beyond predictor.",
        "- `figure_03_fixed_tr24_hve_methods`: all modelable fixed-Tr-24 HVE definitions.",
        "- `figure_03b_fixed_tr24_hve_methods_zoomed`: the same HVE estimates on a readable local scale; the full-scale companion retains the ceiling.",
        "- `figure_04_hve_objective_reporting_matrix`: likelihood-versus-z selection crossed with gain-versus-z reporting.",
        "- `figure_04b_hve_objective_predictive_gain_zoomed`: a presentation-scale view of the predictive half of Figure 4.",
        "- `figure_04c_selected_hve_downstream_comparisons`: predictor-beyond-condition and condition-beyond-predictor gains for both HVE selection objectives.",
        "- `figure_05a_an19_acoustic_component_audit`: MFCC and STRF group diagnostics.",
        "- `figure_05a2_an19_acoustic_component_gain_zoomed`: presentation-scale acoustic component gains.",
        "- `figure_05b_an19_acoustic_gain_by_condition`: where the AN19 acoustic predictive gain occurs across conditions.",
        "- `figure_05c_an19_acoustic_hubert_distance_correlations`: Spearman correlations on the same 5,459 physical train-test pairs.",
        "- `figure_drafts/`: review drafts of manuscript Figures 1 and 2. Figure 2a/2c and the language inventory use current sources; panels requiring PHOIBLE metric decisions or segment-level response alignment remain visibly marked as missing.",
        "",
        "## Leading AN19 acoustic groups",
        "",
        "| Feature group | OOF gain (% ceiling) | 95% participant-cluster CI |",
        "|---|---:|---:|",
        ]
    )
    for row in top.itertuples(index=False):
        lines.append(f"| `{row.feature_key}` | {row.gain:.1f} | [{row.ci_low:.1f}, {row.ci_high:.1f}] |")
    lines.extend(
        [
            "",
            "These group analyses are diagnostic. They identify where to run individual-dimension and leave-one-group-out checks; they do not yet establish that the full AN19 baseline is free of condition encoding or recording leakage.",
            "The pair audit found no duplicate pair identifiers, no self-recording comparisons, and no same-speaker train-test comparisons. Control rows have no exposure predictor by design, and 75 exposed-test rows per feature have incomplete source mappings; the fitted models therefore use the same 5,685 non-control rows for every representation.",
            "MFCC39 and STRF24 correlate moderately with HuBERT Tr-24 distances on identical pairs (Spearman 0.38–0.46). The pooled-model gain from MFCC delta, MFCC delta-delta, and STRF rate-16 is positive within each of the six modeled AN19 exposure conditions, so simple between-condition separation is not a sufficient explanation. Separate within-condition refits and individual-dimension/leave-one-group-out tests remain necessary.",
            "",
            "## Inferential boundaries",
            "",
            "The fixed Tr-24 panels are common-layer likelihood summaries, not substitutes for the SBI layerwise z display. HVE selection is exploratory and the B23 cross-sample ranking needs correction. Z intervals bootstrap the three training-fold z values; predictive intervals resample participants. These are different uncertainty procedures.",
            "",
            "## Run validation",
            "",
            f"All {int(diagnostics['n_fits'].sum()):,} requested GLMM fits completed; {int(diagnostics['convergence_ok'].sum()):,} reported normal convergence, with {int(diagnostics['singular'].sum()):,} singular fits, {int(diagnostics['warnings'].sum()):,} warnings, and {int(diagnostics['errors'].sum()):,} errors. See `tables/run_diagnostics_summary.csv`.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    output = (repository / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    specs = run_specs(repository)
    metrics, z_values, conditions, inputs = collect(repository, specs, args.bootstrap, args.seed)
    metrics.to_csv(output / "tables" / "all_model_gain_summary.csv", index=False)
    z_values.to_csv(output / "tables" / "all_predictor_z_summary.csv", index=False)
    conditions.to_csv(output / "tables" / "an19_acoustic_condition_summary.csv", index=False)
    plot_sbi(metrics, output)
    plot_downstream(metrics, output)
    selection = plot_hve(metrics, z_values, output)
    plot_objective_matrix(metrics, z_values, selection, output)
    plot_hve_downstream(metrics, selection, output)
    plot_acoustic(metrics, z_values, conditions, output)
    inputs.extend(audit_acoustic_inputs(repository, output))
    diagnostics = summarize_diagnostics(specs, output)
    build_readme(output, metrics, z_values, diagnostics)
    inputs.extend(
        [
            repository / "cross_talker_generalization/scripts/build_september_report.py",
            repository / "cross_talker_generalization/configs/project.json",
            repository / "cross_talker_generalization/configs/confirmatory.json",
            repository / "cross_talker_generalization/R/fit_confirmatory.R",
        ]
    )
    unique_inputs = sorted(set(inputs))
    provenance = {
        "status": "complete",
        "analysis": "fixed_tr24_sbi_hve_and_an19_acoustic_diagnostics",
        "seed": args.seed,
        "participant_cluster_bootstrap_replicates": args.bootstrap,
        "inputs": {str(path.relative_to(repository)): sha256(path) for path in unique_inputs},
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"built {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
