from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .plots import _benjamini_hochberg, plot_descriptive_s_curves
from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


LAYERS = (
    "cnn_2", "cnn_3", "cnn_4", "cnn_5", "cnn_6", "tr_0", "tr_2", "tr_4",
    "tr_6", "tr_8", "tr_10", "tr_12", "tr_14", "tr_16", "tr_18", "tr_20",
    "tr_22", "tr_24",
)
COLORS = {"base": "#1f77b4", "ft": "#d62728", "mfcc39": "#7f7f7f", "strf24_legacy": "#9467bd"}
HVE_MEASURES = (
    "overall",
    "within_token_sentence", "within_type_sentence", "between_type_sentence",
    "order_sentence", "mean_dissimilarity_sentence",
    "within_token_word", "within_type_word", "between_type_word",
    "order_word", "mean_dissimilarity_word",
    "within_token_phoneme", "within_type_phoneme", "between_type_phoneme",
    "order_phoneme", "mean_dissimilarity_phoneme",
)


def _model_summary(model_dir: Path, *, dataset: str, family: str, variant: str, term: str) -> pd.DataFrame:
    coefficients = pd.read_csv(model_dir / "coefficients.csv")
    lrts = pd.read_csv(model_dir / "likelihood_ratio_tests.csv")
    metrics = pd.read_csv(model_dir / "cv_metrics.csv")
    coefficient = coefficients.loc[
        coefficients["scope"].eq("full")
        & coefficients["model_id"].eq("M_joint")
        & coefficients["term"].eq(term),
        ["feature_key", "estimate", "std_error", "z_value", "p_value", "conf_low", "conf_high"],
    ].copy()
    coefficient["coefficient_q_bh"] = _benjamini_hochberg(coefficient["p_value"])
    lrt = lrts[["feature_key", "chisq", "df", "p_value"]].rename(columns={"p_value": "lrt_p_value"})
    overall = metrics.loc[metrics["scope"].eq("oof_all")].pivot(
        index="feature_key", columns="model_id", values="mean_log_loss"
    )
    overall["oof_gain_joint_vs_condition"] = overall["M_condition"] - overall["M_joint"]
    features = pd.read_csv(model_dir / "feature_manifest.csv")[["feature_key"]].drop_duplicates()
    result = features.merge(coefficient, on="feature_key", how="left").merge(lrt, on="feature_key", how="left").merge(
        overall.reset_index(), on="feature_key", how="left"
    )
    result.insert(0, "variant", variant)
    result.insert(0, "family", family)
    result.insert(0, "dataset_id", dataset)
    result["model_directory"] = str(model_dir.resolve())
    diagnostics = pd.read_csv(model_dir / "diagnostics.csv")
    diagnostics["diagnostic_row_valid"] = (
        diagnostics["fit_ok"].astype(bool)
        & diagnostics["convergence"].fillna("").eq("ok")
        & ~diagnostics["singular"].map(lambda value: str(value).strip().lower() == "true")
    )
    diagnostic_valid = diagnostics.groupby("feature_key")["diagnostic_row_valid"].all()
    result["diagnostics_valid"] = result["feature_key"].map(diagnostic_valid).fillna(False)
    finite = np.isfinite(result[["estimate", "std_error", "z_value"]]).all(axis=1)
    result["numerically_valid"] = (
        result["diagnostics_valid"]
        & finite
        & result["std_error"].gt(0)
        & result["estimate"].abs().le(20)
        & result["z_value"].abs().le(20)
    )
    result["numerical_note"] = "ok"
    result.loc[~result["diagnostics_valid"], "numerical_note"] = "failed_or_nonconverged_fit"
    result.loc[~finite, "numerical_note"] = "missing_or_nonfinite_full_joint_coefficient"
    result.loc[
        finite & (result["estimate"].abs().gt(20) | result["z_value"].abs().gt(20)),
        "numerical_note",
    ] = "quasi_separation_or_scale_saturation"
    return result


def _collect_sbi(root: Path) -> pd.DataFrame:
    records = []
    for dataset in ("AN19", "X21", "B23"):
        for variant in ("base", "ft"):
            model = root / "artifacts" / "models" / f"{dataset}-{dataset}_hubert_{variant}_tsne-confirmatory"
            records.append(_model_summary(model, dataset=dataset, family="HuBERT t-SNE", variant=variant, term="similarity_z"))
        model = root / "artifacts" / "models" / f"{dataset}-{dataset}_acoustic-confirmatory"
        records.append(_model_summary(model, dataset=dataset, family="Acoustic baseline", variant="baseline", term="similarity_z"))
    return pd.concat(records, ignore_index=True)


def _collect_sbi_exp(root: Path) -> pd.DataFrame:
    records = []
    for dataset in ("AN19", "X21", "B23"):
        for variant in ("base", "ft"):
            model = root / "artifacts" / "models" / f"{dataset}-SBI-exp-{variant}"
            records.append(
                _model_summary(
                    model, dataset=dataset, family="HuBERT t-SNE exp(k=1)",
                    variant=variant, term="similarity_z",
                )
            )
        model = root / "artifacts" / "models" / f"{dataset}-SBI-exp-acoustic"
        records.append(
            _model_summary(
                model, dataset=dataset, family="Acoustic exp(k=1)",
                variant="baseline", term="similarity_z",
            )
        )
    return pd.concat(records, ignore_index=True)


def _collect_hve(root: Path) -> pd.DataFrame:
    paths = {
        ("AN19", "base"): root / "artifacts" / "models" / "AN19-HVE-base",
        ("AN19", "ft"): root / "artifacts" / "models" / "AN19-HVE-ft",
        ("X21", "base"): root / "artifacts" / "models" / "X21-X21_hubert_base_tsne-variability-confirmatory",
        ("X21", "ft"): root / "artifacts" / "models" / "X21-X21_hubert_ft_tsne-variability-confirmatory",
    }
    return pd.concat(
        [
            _model_summary(path, dataset=dataset, family="HVE overall", variant=variant, term="variability_z")
            for (dataset, variant), path in paths.items()
        ],
        ignore_index=True,
    )


def _collect_all_hve(root: Path, overall: pd.DataFrame) -> pd.DataFrame:
    frames = [overall.copy()]
    for dataset in ("AN19", "X21"):
        for variant in ("base", "ft"):
            model = root / "artifacts" / "models" / f"{dataset}-HVE-{variant}-supp"
            if model.is_dir():
                frames.append(
                    _model_summary(
                        model,
                        dataset=dataset,
                        family="HVE supplemental",
                        variant=variant,
                        term="variability_z",
                    )
                )
    result = pd.concat(frames, ignore_index=True)
    split = result["feature_key"].str.split("::", n=1, expand=True)
    result["layer"] = split[0]
    result["measure"] = split[1].fillna("overall")
    result["coefficient_q_bh_all_hve"] = result.groupby(
        ["dataset_id", "variant"], group_keys=False
    )["p_value"].transform(_benjamini_hochberg)
    return result


def _diagnostics(root: Path, sbi: pd.DataFrame, hve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for directory in sorted(set(sbi["model_directory"]) | set(hve["model_directory"])):
        data = pd.read_csv(Path(directory) / "diagnostics.csv")
        rows.append(
            {
                "model_directory": directory,
                "dataset_id": data["dataset_id"].iloc[0],
                "n_fits": len(data),
                "fit_failures": int((~data["fit_ok"].astype(bool)).sum()),
                "singular_fits": int(
                    data["singular"].map(lambda value: str(value).strip().lower() == "true").sum()
                ),
                "non_ok_convergence": int((data["convergence"].fillna("") != "ok").sum()),
                "fallback_fits": int(
                    data.get("random_structure", pd.Series(dtype=str)).eq("participant_item_intercepts").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _sbi_profiles(sbi: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=True)
    x = np.arange(len(LAYERS))
    for row, dataset in enumerate(("AN19", "X21", "B23")):
        subset = sbi.loc[(sbi["dataset_id"] == dataset) & (sbi["family"] == "HuBERT t-SNE")]
        for variant in ("base", "ft"):
            layer = subset.loc[subset["variant"].eq(variant)].set_index("feature_key").reindex(LAYERS)
            axes[row, 0].plot(x, layer["z_value"], marker="o", markersize=4, label=variant, color=COLORS[variant])
            axes[row, 1].plot(
                x, layer["oof_gain_joint_vs_condition"], marker="o", markersize=4,
                label=variant, color=COLORS[variant],
            )
        baseline = sbi.loc[(sbi["dataset_id"] == dataset) & (sbi["family"] == "Acoustic baseline")]
        for feature in ("mfcc39", "strf24_legacy"):
            value = baseline.loc[baseline["feature_key"].eq(feature)]
            if not value.empty:
                label = "MFCC39" if feature == "mfcc39" else "STRF24"
                axes[row, 0].axhline(value["z_value"].iloc[0], linestyle=":", linewidth=1.6, color=COLORS[feature], label=label)
                axes[row, 1].axhline(value["oof_gain_joint_vs_condition"].iloc[0], linestyle=":", linewidth=1.6, color=COLORS[feature], label=label)
        axes[row, 0].axhline(0, color="black", linewidth=0.8)
        axes[row, 0].axhline(1.96, color="0.4", linewidth=0.8, linestyle="--")
        axes[row, 1].axhline(0, color="black", linewidth=0.8)
        axes[row, 0].set_ylabel(f"{dataset}\nJoint Wald z")
        axes[row, 1].set_ylabel("OOF log-loss gain")
        axes[row, 0].legend(ncol=4, fontsize=9, loc="best")
    axes[0, 0].set_title("Association profile")
    axes[0, 1].set_title("True participant-held-out prediction")
    for axis in axes[-1, :]:
        axis.set_xticks(x)
        axis.set_xticklabels(LAYERS, rotation=60, ha="right", fontsize=9)
    figure.suptitle("Same-content similarity: HuBERT base/FT and acoustic baselines", y=1.01)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def _best_gain(sbi: pd.DataFrame, destination: Path) -> pd.DataFrame:
    rows = []
    for dataset in ("AN19", "X21", "B23"):
        data = sbi.loc[sbi["dataset_id"].eq(dataset)]
        for label, selector in (
            ("HuBERT base", (data["family"].eq("HuBERT t-SNE") & data["variant"].eq("base"))),
            ("HuBERT FT", (data["family"].eq("HuBERT t-SNE") & data["variant"].eq("ft"))),
            ("MFCC39", data["feature_key"].eq("mfcc39")),
            ("STRF24", data["feature_key"].eq("strf24_legacy")),
        ):
            selected = data.loc[selector].sort_values("oof_gain_joint_vs_condition", ascending=False).iloc[0]
            rows.append(
                {
                    "dataset_id": dataset, "model": label, "feature_key": selected["feature_key"],
                    "oof_gain": selected["oof_gain_joint_vs_condition"], "z_value": selected["z_value"],
                    "p_value": selected["p_value"],
                }
            )
    source = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=source, x="dataset_id", y="oof_gain", hue="model", ax=axis, palette="Set2")
    axis.axhline(0, color="black", linewidth=0.9)
    axis.set_xlabel("")
    axis.set_ylabel("Largest OOF log-loss reduction\njoint vs condition-only")
    axis.set_title("Auxiliary incremental-prediction comparison\n(HuBERT layer selected on this same OOF summary)")
    axis.legend(title="")
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    return source


def _hve_profiles(hve: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    x = np.arange(len(LAYERS))
    for row, dataset in enumerate(("AN19", "X21")):
        for variant in ("base", "ft"):
            subset = hve.loc[(hve["dataset_id"] == dataset) & hve["variant"].eq(variant)].copy()
            subset["layer"] = subset["feature_key"].str.split("::").str[0]
            subset = subset.set_index("layer").reindex(LAYERS)
            axes[row, 0].plot(x, subset["z_value"], marker="o", markersize=4, color=COLORS[variant], label=variant)
            axes[row, 1].plot(x, subset["oof_gain_joint_vs_condition"], marker="o", markersize=4, color=COLORS[variant], label=variant)
        axes[row, 0].axhline(0, color="black", linewidth=0.8)
        axes[row, 0].axhline(1.96, color="0.4", linewidth=0.8, linestyle="--")
        axes[row, 1].axhline(0, color="black", linewidth=0.8)
        axes[row, 0].set_ylabel(f"{dataset}\nJoint Wald z")
        axes[row, 1].set_ylabel("OOF log-loss gain")
        axes[row, 0].legend()
    axes[0, 0].set_title("HVE overall association")
    axes[0, 1].set_title("HVE overall held-out prediction")
    for axis in axes[-1, :]:
        axis.set_xticks(x)
        axis.set_xticklabels(LAYERS, rotation=60, ha="right", fontsize=9)
    figure.suptitle("Actual-exposure variability (current B23 multi-talker mapping not integrated)", y=1.01)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def _hve_heatmaps(hve: pd.DataFrame, dataset: str, destination: Path) -> None:
    subset = hve.loc[hve["dataset_id"].eq(dataset)].copy()
    present_measures = [measure for measure in HVE_MEASURES if measure in set(subset["measure"])]
    labels = {
        "overall": "overall",
        **{
            f"{kind}_{unit}": f"{unit} · {kind.replace('_', ' ')}"
            for unit in ("sentence", "word", "phoneme")
            for kind in (
                "within_token", "within_type", "between_type", "order", "mean_dissimilarity"
            )
        },
    }
    z_limit = max(1.96, float(np.nanmax(np.abs(subset["z_value"]))))
    gain_limit = max(1e-6, float(np.nanmax(np.abs(subset["oof_gain_joint_vs_condition"]))))
    height = 7 if dataset == "AN19" else 12
    figure, axes = plt.subplots(2, 2, figsize=(18, height), sharex=True, sharey=True)
    for row, variant in enumerate(("base", "ft")):
        data = subset.loc[subset["variant"].eq(variant)]
        z = data.pivot(index="measure", columns="layer", values="z_value").reindex(
            index=present_measures, columns=LAYERS
        )
        gain = data.pivot(
            index="measure", columns="layer", values="oof_gain_joint_vs_condition"
        ).reindex(index=present_measures, columns=LAYERS)
        z.index = [labels[value] for value in z.index]
        gain.index = z.index
        sns.heatmap(
            z, ax=axes[row, 0], cmap="vlag", center=0, vmin=-z_limit, vmax=z_limit,
            cbar_kws={"label": "Joint Wald z"},
        )
        sns.heatmap(
            gain, ax=axes[row, 1], cmap="vlag", center=0,
            vmin=-gain_limit, vmax=gain_limit,
            cbar_kws={"label": "OOF log-loss gain"},
        )
        axes[row, 0].set_ylabel(f"{variant}\nvariability definition")
        axes[row, 1].set_ylabel("")
        for axis in axes[row, :]:
            axis.set_xlabel("")
            axis.tick_params(axis="x", rotation=60, labelsize=8)
            axis.tick_params(axis="y", rotation=0, labelsize=8)
    axes[0, 0].set_title("Association")
    axes[0, 1].set_title("True participant-held-out prediction")
    figure.suptitle(f"{dataset}: HVE sensitivity across definitions and layers", y=1.01)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def _ceiling_figure(root: Path, sbi: pd.DataFrame, destination: Path) -> pd.DataFrame:
    rows = []
    for dataset in ("AN19", "X21", "B23"):
        ceiling = pd.read_csv(root / "artifacts" / "models" / f"{dataset}-ceiling-cv" / "cv_metrics.csv")
        ceiling_loss = ceiling.loc[ceiling["scope"].eq("oof_all"), "mean_log_loss"].iloc[0]
        dataset_sbi = sbi.loc[(sbi["dataset_id"] == dataset) & (sbi["family"] == "HuBERT t-SNE")]
        best = dataset_sbi.sort_values("M_joint").iloc[0]
        condition_loss = dataset_sbi["M_condition"].iloc[0]
        rows.extend(
            [
                {"dataset_id": dataset, "model": "Condition only", "mean_log_loss": condition_loss, "feature_key": ""},
                {"dataset_id": dataset, "model": "Lowest-loss HuBERT joint (auxiliary selection)", "mean_log_loss": best["M_joint"], "feature_key": best["feature_key"]},
                {"dataset_id": dataset, "model": "Behavioral ceiling", "mean_log_loss": ceiling_loss, "feature_key": "cross-fitted item rate"},
            ]
        )
    source = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=source, x="dataset_id", y="mean_log_loss", hue="model", palette="colorblind", ax=axis)
    axis.set_xlabel("")
    axis.set_ylabel("Held-out mean log loss\n(lower is better)")
    axis.set_title("True cross-validated performance and behavioral ceiling")
    axis.legend(title="")
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    return source


def _b23_hve(root: Path, destination: Path) -> pd.DataFrame:
    pools = pd.read_csv(root / "artifacts" / "derived" / "B23-exposure" / "exposure_pools.csv")
    frames = []
    for variant in ("base", "ft"):
        data = pd.read_csv(root / "artifacts" / "derived" / f"B23-HVE-{variant}-values.csv")
        data = data.loc[data["measure"].eq("overall") & data["value_status"].eq("available")].copy()
        data["variant"] = variant
        frames.append(data)
    source = pd.concat(frames, ignore_index=True).merge(
        pools[["pool_id", "exposure_talker_set"]], on="pool_id", how="left"
    )
    source["layer"] = pd.Categorical(source["feature_key"], categories=LAYERS, ordered=True)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
    for axis, variant in zip(axes, ("base", "ft")):
        subset = source.loc[source["variant"].eq(variant)]
        sns.lineplot(
            data=subset, x="layer", y="value", hue="exposure_talker_set",
            marker="o", linewidth=1.5, ax=axis,
        )
        axis.set_title(variant)
        axis.set_xlabel("")
        axis.set_ylabel("Overall actual-exposure variability")
        axis.tick_params(axis="x", rotation=60, labelsize=8)
        axis.legend(title="Single talker", fontsize=8)
    figure.suptitle("B23 HVE descriptive only: public multi-talker mapping not yet integrated")
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    return source


def _correlation_composite(root: Path, destination: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15, 18))
    for row, dataset in enumerate(("AN19", "X21", "B23")):
        for column, variant in enumerate(("base", "ft")):
            prefix = root / "artifacts" / "figures" / f"{dataset}-{dataset}_hubert_{variant}_tsne-confirmatory-distance-correlations-matrix.csv"
            matrix = pd.read_csv(prefix).set_index("feature_key").reindex(index=LAYERS, columns=LAYERS)
            sns.heatmap(matrix, vmin=-1, vmax=1, center=0, cmap="vlag", square=True, cbar=row == 0 and column == 1, ax=axes[row, column])
            axes[row, column].set_title(f"{dataset} {variant}")
            axes[row, column].tick_params(labelsize=6)
    figure.suptitle("Spearman correlations among layer-wise raw DTW distances", y=0.995)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def _transform_sensitivity(primary: pd.DataFrame, exp: pd.DataFrame, destination: Path) -> pd.DataFrame:
    raw = primary.loc[primary["family"].eq("HuBERT t-SNE")].copy()
    raw["transform"] = "-z(DTW distance)"
    historical = exp.loc[exp["family"].eq("HuBERT t-SNE exp(k=1)")].copy()
    historical["transform"] = "z(exp(-DTW distance)), k=1"
    source = pd.concat([raw, historical], ignore_index=True)
    figure, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=True)
    x = np.arange(len(LAYERS))
    for row, dataset in enumerate(("AN19", "X21", "B23")):
        for column, variant in enumerate(("base", "ft")):
            axis = axes[row, column]
            data = source.loc[source["dataset_id"].eq(dataset) & source["variant"].eq(variant)]
            for transform, color, marker in (
                ("-z(DTW distance)", "#1f77b4", "o"),
                ("z(exp(-DTW distance)), k=1", "#d62728", "s"),
            ):
                values = data.loc[data["transform"].eq(transform)].set_index("feature_key").reindex(LAYERS)
                valid = values["numerically_valid"].eq(True)
                axis.plot(
                    x, values["oof_gain_joint_vs_condition"], color=color, marker=marker,
                    markersize=4, linewidth=1.5, label=transform,
                )
                invalid = ~valid & values["oof_gain_joint_vs_condition"].notna()
                axis.scatter(
                    x[invalid], values.loc[invalid, "oof_gain_joint_vs_condition"],
                    marker="x", s=70, linewidths=2, color="black", zorder=5,
                    label="invalid exp fit" if transform.startswith("z(exp") else None,
                )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"{dataset} {variant}")
            axis.set_ylabel("OOF log-loss gain")
            if row == 0 and column == 0:
                axis.legend(fontsize=9)
    for axis in axes[-1, :]:
        axis.set_xticks(x)
        axis.set_xticklabels(LAYERS, rotation=60, ha="right", fontsize=9)
    figure.suptitle("Predictor-transform sensitivity (x = invalid or saturated exp fit)", y=1.01)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    return source


def build_report_core(root: str | Path, output_dir: str | Path) -> dict[str, object]:
    root = Path(root).resolve()
    output = Path(output_dir).resolve()
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    sbi = _collect_sbi(root)
    sbi_exp = _collect_sbi_exp(root)
    hve = _collect_hve(root)
    hve_all = _collect_all_hve(root, hve)
    diagnostics = _diagnostics(root, pd.concat([sbi, sbi_exp], ignore_index=True), hve_all)
    atomic_write_csv(tables / "sbi_all_results.csv", sbi)
    atomic_write_csv(tables / "sbi_exp_k1_sensitivity_results.csv", sbi_exp)
    atomic_write_csv(tables / "hve_overall_results.csv", hve)
    atomic_write_csv(tables / "hve_all_results.csv", hve_all)
    atomic_write_csv(tables / "diagnostic_summary.csv", diagnostics)
    _sbi_profiles(sbi, figures / "figure_01_sbi_layer_profiles")
    best = _best_gain(sbi, figures / "figure_02_best_predictive_gain")
    atomic_write_csv(tables / "best_predictive_gain.csv", best)
    _hve_profiles(hve, figures / "figure_03_hve_overall_profiles")
    ceiling = _ceiling_figure(root, sbi, figures / "figure_04_cv_ceiling")
    atomic_write_csv(tables / "cv_ceiling_comparison.csv", ceiling)
    _correlation_composite(root, figures / "figure_05_layer_distance_correlations")
    b23 = _b23_hve(root, figures / "figure_06_b23_hve_descriptive")
    atomic_write_csv(tables / "b23_hve_descriptive.csv", b23)
    _hve_heatmaps(hve_all, "AN19", figures / "figure_07_an19_hve_sensitivity")
    _hve_heatmaps(hve_all, "X21", figures / "figure_08_x21_hve_sensitivity")
    transform = _transform_sensitivity(sbi, sbi_exp, figures / "figure_09_predictor_transform_sensitivity")
    atomic_write_csv(tables / "predictor_transform_sensitivity.csv", transform)

    selected_curves = []
    hubert = sbi.loc[sbi["family"].eq("HuBERT t-SNE")]
    for (dataset, variant), group in hubert.groupby(["dataset_id", "variant"]):
        for basis, column, ascending in (
            ("best_oof_gain", "oof_gain_joint_vs_condition", False),
            ("strongest_association", "p_value", True),
        ):
            row = group.sort_values(column, ascending=ascending).iloc[0]
            selected_curves.append(
                {
                    "dataset_id": dataset,
                    "variant": variant,
                    "feature_key": row["feature_key"],
                    "selection_basis": basis,
                    "z_value": row["z_value"],
                    "p_value": row["p_value"],
                    "oof_gain": row["oof_gain_joint_vs_condition"],
                }
            )
    selected_curves = pd.DataFrame(selected_curves)
    atomic_write_csv(tables / "selected_scurves.csv", selected_curves)
    for row in selected_curves.drop_duplicates(["dataset_id", "variant", "feature_key"]).itertuples(index=False):
        input_path = root / "artifacts" / "derived" / f"{row.dataset_id}-{row.dataset_id}_hubert_{row.variant}_tsne-confirmatory-model-input.csv"
        plot_descriptive_s_curves(
            input_path, row.feature_key,
            figures / f"figure_scurve_{row.dataset_id}_{row.variant}_{row.feature_key}", bins=10,
        )

    hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "provenance.json":
            hashes[path.relative_to(output).as_posix()] = sha256_file(path)
    metadata = {
        **runtime_record(),
        "status": "complete",
        "stage": "report_core",
        "primary_representation": "3-D t-SNE",
        "dtw_normalization": "mean_sequence_length",
        "primary_similarity_predictor": "negative training-fold-standardized DTW distance",
        "historical_transform_sensitivity": "training-fold-standardized exp(-DTW distance), fixed k=1",
        "historical_transform_is_primary": False,
        "evaluation": "participant-held-out frozen-model prediction",
        "legacy_heldout_refit_results_included": False,
        "files_sha256": hashes,
    }
    atomic_write_json(output / "provenance.json", metadata)
    return metadata
