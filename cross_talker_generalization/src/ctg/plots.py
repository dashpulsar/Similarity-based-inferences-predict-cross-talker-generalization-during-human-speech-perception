from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


def _layer_order(value: str) -> tuple[int, int]:
    value = value.split("::", 1)[0]
    if value.startswith("cnn_"):
        return 0, int(value.split("_")[1])
    if value.startswith("tr_"):
        return 1, int(value.split("_")[1])
    return 2, 0


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(numeric), np.nan)
    valid = np.flatnonzero(np.isfinite(numeric))
    if len(valid) == 0:
        return pd.Series(adjusted, index=values.index)
    order = valid[np.argsort(numeric[valid])]
    ranked = numeric[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return pd.Series(adjusted, index=values.index)


def plot_glmm_profile(model_dir: str | Path, output_prefix: str | Path) -> pd.DataFrame:
    source = Path(model_dir)
    coefficients = pd.read_csv(source / "coefficients.csv")
    lrts = pd.read_csv(source / "likelihood_ratio_tests.csv")
    metrics = pd.read_csv(source / "cv_metrics.csv")
    predictor_terms = [term for term in ("similarity_z", "variability_z", "predictor_z") if term in set(coefficients["term"])]
    if len(predictor_terms) != 1:
        raise ValueError(f"could not identify one predictor term; found {predictor_terms}")
    predictor_term = predictor_terms[0]
    coefficient = coefficients.loc[
        coefficients["scope"].eq("full")
        & coefficients["model_id"].eq("M_joint")
        & coefficients["term"].eq(predictor_term),
        [
            "dataset_id", "feature_key", "estimate", "std_error", "z_value",
            "conf_low", "conf_high", "p_value",
        ],
    ]
    lrt = lrts[["feature_key", "chisq", "df", "p_value"]].rename(
        columns={"p_value": "lrt_p_value"}
    )
    overall = metrics.loc[metrics["scope"].eq("oof_all")]
    wide = overall.pivot(index="feature_key", columns="model_id", values="mean_log_loss")
    wide["oof_log_loss_gain_joint_vs_condition"] = wide["M_condition"] - wide["M_joint"]
    summary = coefficient.merge(lrt, on="feature_key", how="left").merge(
        wide.reset_index(), on="feature_key", how="left"
    )
    summary = summary.sort_values(
        "feature_key", key=lambda values: values.map(_layer_order)
    ).reset_index(drop=True)
    summary["coefficient_q_bh"] = _benjamini_hochberg(summary["p_value"])
    summary["lrt_q_bh"] = _benjamini_hochberg(summary["lrt_p_value"])
    summary["x"] = np.arange(len(summary))

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2))
    axes[0].errorbar(
        summary["x"].to_numpy(),
        summary["estimate"].to_numpy(),
        yerr=np.vstack(
            [
                (summary["estimate"] - summary["conf_low"]).to_numpy(),
                (summary["conf_high"] - summary["estimate"]).to_numpy(),
            ]
        ),
        color="#1f4e79",
        marker="o",
        capsize=3,
    )
    axes[0].axhline(0, color="0.4", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Joint GLMM coefficient (95% CI)")
    axes[0].set_title("Similarity beyond condition")

    axes[1].plot(summary["x"], summary["z_value"], marker="o", color="#bd5d19")
    axes[1].axhline(0, color="0.4", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Joint-model Wald z")
    axes[1].set_title("Association profile")

    axes[2].plot(summary["x"], -np.log10(summary["lrt_p_value"].clip(lower=1e-300)), marker="o", color="#7a3e9d")
    axes[2].axhline(-np.log10(0.05), color="0.4", linewidth=1, linestyle="--")
    axes[2].set_ylabel("-log10 LRT p")
    axes[2].set_title("M_condition vs M_joint")

    axes[3].plot(
        summary["x"],
        summary["oof_log_loss_gain_joint_vs_condition"],
        marker="o",
        color="#2e7d32",
    )
    axes[3].axhline(0, color="0.4", linewidth=1, linestyle="--")
    axes[3].set_ylabel("Held-out log-loss gain")
    axes[3].set_title("Frozen-model prediction")

    for axis in axes:
        axis.set_xticks(summary["x"])
        axis.set_xticklabels(summary["feature_key"], rotation=60, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(str(summary["dataset_id"].iloc[0]))
    fig.tight_layout()
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(prefix) + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(str(prefix) + ".svg", bbox_inches="tight")
    plt.close(fig)
    atomic_write_csv(str(prefix) + "-source.csv", summary.drop(columns="x"))
    atomic_write_json(
        str(prefix) + ".provenance.json",
        {
            **runtime_record(),
            "stage": "glmm_profile_figure",
            "model_directory": str(source.resolve()),
            "source_sha256": {
                name: sha256_file(source / name)
                for name in ("coefficients.csv", "likelihood_ratio_tests.csv", "cv_metrics.csv")
            },
            "predictor_term": predictor_term,
            "feature_keys": summary["feature_key"].tolist(),
            "multiplicity": "Benjamini-Hochberg within plotted feature family",
        },
    )
    return summary


def _fit_logistic(x: np.ndarray, correct: np.ndarray, incorrect: np.ndarray):
    def objective(beta):
        eta = beta[0] + beta[1] * x
        log_p = -np.logaddexp(0.0, -eta)
        log_one_minus_p = -np.logaddexp(0.0, eta)
        return float(-(correct * log_p + incorrect * log_one_minus_p).sum())

    prevalence = (correct.sum() + 0.5) / (correct.sum() + incorrect.sum() + 1.0)
    start = np.asarray([math.log(prevalence / (1 - prevalence)), 0.0])
    result = minimize(objective, start, method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(result.message)
    return result.x


def _wilson(correct: float, total: float, z: float = 1.96):
    if total <= 0:
        return np.nan, np.nan
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def plot_descriptive_s_curves(
    model_input_path: str | Path,
    feature_key: str,
    output_prefix: str | Path,
    bins: int = 10,
) -> pd.DataFrame:
    data = pd.read_csv(model_input_path)
    data = data.loc[
        data["feature_key"].eq(feature_key)
        & data["predictor_status"].eq("available")
        & data["raw_distance"].notna()
    ].copy()
    if data.empty:
        raise ValueError(f"no available rows for {feature_key}")
    mean = data["raw_distance"].mean()
    scale = data["raw_distance"].std(ddof=1)
    data["similarity_z"] = -(data["raw_distance"] - mean) / scale
    conditions = sorted(data["condition_id"].unique())
    ncols = min(4, max(2, int(math.ceil(math.sqrt(len(conditions))))))
    nrows = int(math.ceil(len(conditions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False, sharex=True, sharey=True)
    source_rows = []

    for axis, condition in zip(axes.flat, conditions):
        subset = data.loc[data["condition_id"].eq(condition)].copy()
        if subset["similarity_z"].nunique() < 2:
            grouped = pd.DataFrame(
                {
                    "similarity_z": [subset["similarity_z"].iloc[0]],
                    "correct": [subset["response_correct"].sum()],
                    "incorrect": [subset["response_incorrect"].sum()],
                    "n_rows": [len(subset)],
                }
            )
            axis.axhline(
                grouped["correct"].iloc[0]
                / (grouped["correct"].iloc[0] + grouped["incorrect"].iloc[0]),
                color="#1f4e79",
                linewidth=2,
            )
            axis.text(
                0.5,
                0.08,
                "Predictor is constant\n(self-comparison)",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                color="0.35",
            )
        else:
            beta = _fit_logistic(
                subset["similarity_z"].to_numpy(),
                subset["response_correct"].to_numpy(),
                subset["response_incorrect"].to_numpy(),
            )
            grid = np.linspace(subset["similarity_z"].min(), subset["similarity_z"].max(), 200)
            probability = 1.0 / (1.0 + np.exp(-(beta[0] + beta[1] * grid)))
            axis.plot(grid, probability, color="#1f4e79", linewidth=2)
            subset["quantile_bin"] = pd.qcut(subset["similarity_z"], q=bins, duplicates="drop")
            grouped = (
                subset.groupby("quantile_bin", observed=True)
                .agg(
                    similarity_z=("similarity_z", "mean"),
                    correct=("response_correct", "sum"),
                    incorrect=("response_incorrect", "sum"),
                    n_rows=("participant_id", "size"),
                )
                .reset_index(drop=True)
            )
        grouped["total"] = grouped["correct"] + grouped["incorrect"]
        grouped["proportion"] = grouped["correct"] / grouped["total"]
        intervals = [_wilson(c, n) for c, n in zip(grouped["correct"], grouped["total"])]
        grouped["ci_low"] = [value[0] for value in intervals]
        grouped["ci_high"] = [value[1] for value in intervals]
        grouped["condition_id"] = condition
        grouped["feature_key"] = feature_key
        source_rows.append(grouped)
        axis.errorbar(
            grouped["similarity_z"].to_numpy(),
            grouped["proportion"].to_numpy(),
            yerr=np.vstack(
                [
                    (grouped["proportion"] - grouped["ci_low"]).to_numpy(),
                    (grouped["ci_high"] - grouped["proportion"]).to_numpy(),
                ]
            ),
            fmt="o",
            color="black",
            markersize=4,
            capsize=2,
        )
        axis.set_title(condition)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes.flat[len(conditions):]:
        axis.set_visible(False)
    fig.supxlabel("Similarity: -z(raw DTW distance)")
    fig.supylabel("Proportion correct")
    fig.suptitle(f"Descriptive response curves - {feature_key}\nPoints: quantile bins with Wilson 95% CI")
    fig.tight_layout(rect=(0.035, 0.035, 1, 0.95))
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(prefix) + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(str(prefix) + ".svg", bbox_inches="tight")
    plt.close(fig)
    source = pd.concat(source_rows, ignore_index=True)
    atomic_write_csv(str(prefix) + "-source.csv", source)
    atomic_write_json(
        str(prefix) + ".provenance.json",
        {
            **runtime_record(),
            "stage": "descriptive_s_curves",
            "model_input_path": str(Path(model_input_path).resolve()),
            "model_input_sha256": sha256_file(model_input_path),
            "feature_key": feature_key,
            "bins_requested": int(bins),
            "point_interval": "Wilson 95% binomial confidence interval",
            "curve": "descriptive fixed-effect binomial logistic fit within condition",
        },
    )
    return source
