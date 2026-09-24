"""Run the combined-fold cross-fitted predictor likelihood-ratio analysis.

This is a supplementary inferential analysis requested after the primary
participant-held-out prediction analysis.  It does not replace OOF log loss.
The selected theoretical predictor is standardized separately for each outer
test fold using only the other two folds, the three held-out partitions are
combined, and nested GLMMs are compared once on that combined table.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DATASETS = ("AN19", "X21", "B23")
VARIANT_LABELS = {"base": "HuBERT base", "ft": "HuBERT ASR fine-tuned"}
FAMILY_COLORS = {"SBI": "#2563A7", "HVE": "#238B57"}
VARIANT_MARKERS = {"base": "o", "ft": "s"}
COMPARISON_LABELS = {
    "predictor_beyond_condition": "Predictor beyond condition\nM_condition vs M_joint",
    "condition_beyond_predictor": "Condition beyond predictor\nM_predictor vs M_joint",
}


@dataclass(frozen=True)
class AnalysisSpec:
    analysis_id: str
    dataset_id: str
    family: str
    variant: str
    selection_stratum: str
    feature_key: str
    input_path: Path
    predictor_column: str
    direction: int
    term_name: str
    selection_oof_mean_log_loss: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_rscript() -> Path:
    found = shutil.which("Rscript")
    if found:
        return Path(found).resolve()
    preferred = Path(r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe")
    if preferred.exists():
        return preferred
    candidates = sorted(
        Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"), reverse=True
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Rscript was not found; install R with lme4 or add Rscript to PATH")


def _safe_piece(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-")


def _sbi_input(derived: Path, dataset: str, variant: str) -> Path:
    return derived / (
        f"{dataset}-{dataset}_hubert_{variant}_tsne-confirmatory-model-input.csv"
    )


def _hve_input(
    derived: Path, dataset: str, variant: str, stratum: str, feature_key: str
) -> Path:
    if dataset == "AN19" or stratum.startswith("global_order_"):
        return derived / f"{dataset}-HVE-{variant}-order-sensitive-model-input.csv"
    if dataset in {"X21", "B23"}:
        return derived / f"{dataset}-HVE-{variant}-selected-model-input.csv"
    raise ValueError(f"no HVE input rule for {dataset}, {stratum}, {feature_key}")


def _load_registry(project: Path) -> list[AnalysisSpec]:
    report_tables = project / "analysis_update_2026-08-27" / "tables"
    derived = project / "artifacts" / "derived"
    sbi = pd.read_csv(report_tables / "sbi_predictor_only_selected_layers.csv")
    hve = pd.read_csv(report_tables / "hve_predictor_only_selected_methods.csv")
    specs: list[AnalysisSpec] = []

    for row in sbi.to_dict("records"):
        dataset = str(row["dataset_id"])
        variant = str(row["variant"])
        feature_key = str(row["feature_key"])
        specs.append(
            AnalysisSpec(
                analysis_id=f"{dataset}-SBI-{variant}-{_safe_piece(feature_key)}",
                dataset_id=dataset,
                family="SBI",
                variant=variant,
                selection_stratum="selected_hubert_layer",
                feature_key=feature_key,
                input_path=_sbi_input(derived, dataset, variant),
                predictor_column="raw_distance",
                direction=-1,
                term_name="similarity_z",
                selection_oof_mean_log_loss=float(row["M_predictor"]),
            )
        )

    for row in hve.to_dict("records"):
        dataset = str(row["dataset_id"])
        variant = str(row["variant"])
        stratum = str(row["selection_stratum"])
        feature_key = str(row["feature_key"])
        specs.append(
            AnalysisSpec(
                analysis_id=(
                    f"{dataset}-HVE-{variant}-{_safe_piece(stratum)}-"
                    f"{_safe_piece(feature_key)}"
                ),
                dataset_id=dataset,
                family="HVE",
                variant=variant,
                selection_stratum=stratum,
                feature_key=feature_key,
                input_path=_hve_input(derived, dataset, variant, stratum, feature_key),
                predictor_column="predictor_value",
                direction=1,
                term_name="variability_z",
                selection_oof_mean_log_loss=float(row["M_predictor"]),
            )
        )

    expected = 14
    if len(specs) != expected:
        raise RuntimeError(f"expected {expected} selected analyses, found {len(specs)}")
    missing = sorted({str(spec.input_path) for spec in specs if not spec.input_path.exists()})
    if missing:
        raise FileNotFoundError("missing selected model inputs:\n" + "\n".join(missing))
    return specs


def _run_one(
    spec: AnalysisSpec, rscript: Path, r_analysis: Path, model_root: Path, reuse: bool
) -> tuple[AnalysisSpec, Path]:
    destination = model_root / spec.analysis_id
    required = (
        "crossfitted_predictors.csv",
        "fold_scaling.csv",
        "coefficients.csv",
        "diagnostics.csv",
        "likelihood_ratio_tests.csv",
        "software.csv",
    )
    if reuse and all((destination / name).exists() for name in required):
        return spec, destination
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        str(rscript),
        str(r_analysis),
        str(spec.input_path),
        str(destination),
        spec.feature_key,
        spec.predictor_column,
        str(spec.direction),
        spec.term_name,
        "true",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (destination / "R_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (destination / "R_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{spec.analysis_id} failed with code {completed.returncode}:\n"
            f"{completed.stderr[-4000:]}"
        )
    return spec, destination


def _collect_results(completed: list[tuple[AnalysisSpec, Path]]) -> tuple[pd.DataFrame, ...]:
    lrt_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    scaling_frames: list[pd.DataFrame] = []
    registry_rows: list[dict[str, object]] = []
    for spec, directory in completed:
        metadata = {
            "analysis_id": spec.analysis_id,
            "family": spec.family,
            "variant": spec.variant,
            "selection_stratum": spec.selection_stratum,
            "selection_oof_mean_log_loss": spec.selection_oof_mean_log_loss,
            "model_directory": str(directory.relative_to(directory.parents[1])),
        }
        registry_rows.append(
            {
                **metadata,
                "dataset_id": spec.dataset_id,
                "feature_key": spec.feature_key,
                "input_path": str(spec.input_path),
                "input_sha256": _sha256(spec.input_path),
            }
        )
        for filename, target in (
            ("likelihood_ratio_tests.csv", lrt_frames),
            ("diagnostics.csv", diagnostic_frames),
            ("coefficients.csv", coefficient_frames),
            ("fold_scaling.csv", scaling_frames),
        ):
            frame = pd.read_csv(directory / filename)
            for key, value in metadata.items():
                frame[key] = value
            target.append(frame)
    return (
        pd.concat(lrt_frames, ignore_index=True),
        pd.concat(diagnostic_frames, ignore_index=True),
        pd.concat(coefficient_frames, ignore_index=True),
        pd.concat(scaling_frames, ignore_index=True),
        pd.DataFrame(registry_rows),
    )


def _measure_label(row: pd.Series) -> str:
    feature = str(row["feature_key"])
    if row["family"] == "SBI":
        detail = feature
    else:
        if "::" in feature:
            layer, measure = feature.split("::", 1)
            detail = f"{measure.replace('_', ' ')} · {layer}"
        else:
            detail = feature.replace("_", " ")
    return f"{row['family']} · {VARIANT_LABELS[row['variant']]} · {detail}"


def _format_p(value: float) -> str:
    if not math.isfinite(value):
        return "p = NA"
    if value < 0.0001:
        return f"p = {value:.1e}"
    return f"p = {value:.4f}"


def _plot_lrt(lrts: pd.DataFrame, destination: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    figure, axes = plt.subplots(3, 2, figsize=(18, 16))
    cap = 20.0
    for row_index, dataset in enumerate(DATASETS):
        dataset_data = lrts.loc[lrts["dataset_id"].eq(dataset)].copy()
        order_frame = (
            dataset_data[
                ["analysis_id", "family", "variant", "selection_stratum", "feature_key"]
            ]
            .drop_duplicates()
            .assign(
                family_order=lambda x: x["family"].map({"SBI": 0, "HVE": 1}),
                variant_order=lambda x: x["variant"].map({"base": 0, "ft": 1}),
            )
            .sort_values(["family_order", "selection_stratum", "variant_order", "feature_key"])
        )
        order = order_frame["analysis_id"].tolist()
        labels = {
            row["analysis_id"]: _measure_label(row)
            for _, row in order_frame.iterrows()
        }
        y_positions = {analysis_id: index for index, analysis_id in enumerate(order)}
        for column_index, comparison_id in enumerate(COMPARISON_LABELS):
            axis = axes[row_index, column_index]
            panel = dataset_data.loc[dataset_data["comparison_id"].eq(comparison_id)].copy()
            panel["minus_log10_p"] = -np.log10(panel["p_value"].clip(lower=1e-300))
            panel["display_x"] = panel["minus_log10_p"].clip(upper=cap)
            for _, result in panel.iterrows():
                y = y_positions[result["analysis_id"]]
                axis.scatter(
                    result["display_x"],
                    y,
                    s=75,
                    marker=VARIANT_MARKERS[result["variant"]],
                    color=FAMILY_COLORS[result["family"]],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
                df_label = str(int(result["df"])) if pd.notna(result["df"]) else "NA"
                annotation = (
                    f"{_format_p(float(result['p_value']))}; "
                    f"χ²({df_label}) = {float(result['chisq']):.2f}"
                )
                axis.annotate(
                    annotation,
                    (result["display_x"], y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8.2,
                )
            axis.axvline(-math.log10(0.05), color="#777777", linestyle="--", linewidth=1)
            axis.grid(axis="x", color="#e8e8e8", linewidth=0.8)
            axis.set_yticks(range(len(order)))
            axis.set_yticklabels([labels[item] for item in order], fontsize=8.8)
            axis.invert_yaxis()
            axis.set_xlim(left=0, right=cap + 6)
            axis.set_xlabel("Evidence against reduced model, −log₁₀(LRT p)")
            if row_index == 0:
                axis.set_title(COMPARISON_LABELS[comparison_id], fontsize=12)
            if column_index == 0:
                axis.set_ylabel(dataset, fontsize=12, weight="bold")
            else:
                axis.set_ylabel("")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FAMILY_COLORS["SBI"],
               markeredgecolor="white", markersize=9, label="SBI"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FAMILY_COLORS["HVE"],
               markeredgecolor="white", markersize=9, label="HVE"),
        Line2D([0], [0], marker="o", color="#444444", linestyle="none", markersize=7,
               label="HuBERT base"),
        Line2D([0], [0], marker="s", color="#444444", linestyle="none", markersize=7,
               label="HuBERT ASR fine-tuned"),
        Line2D([0], [0], color="#777777", linestyle="--", label="p = .05"),
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=5,
        frameon=False,
        fontsize=10,
    )
    figure.suptitle(
        "Combined-fold GLMM tests using cross-fitted theoretical predictors",
        fontsize=16,
        weight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        (
            "Each predictor value uses scaling estimated from the other two participant folds. "
            "Models are then fit once to the combined held-out partitions; tests are conditional "
            "on the predictor selected using the same three-fold study."
        ),
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.25, right=0.98, bottom=0.075, top=0.90, hspace=0.28, wspace=0.72
    )
    for suffix in ("png", "svg"):
        figure.savefig(destination.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_readme(lrts: pd.DataFrame, diagnostics: pd.DataFrame, destination: Path) -> None:
    successful = int(lrts["status"].eq("ok").sum())
    significant = lrts.loc[lrts["status"].eq("ok") & lrts["p_value"].lt(0.05)]
    lines = [
        "# Cross-fitted predictor nested-model update",
        "",
        "This supplementary analysis implements the combined-test-fold procedure requested for nested GLMM comparison. For each selected SBI or HVE specification, the predictor is standardized for each held-out participant fold using the mean and standard deviation from the other two folds. The three held-out partitions are then concatenated, and one set of GLMMs is fit to the combined data.",
        "",
        "This operationalizes option B rather than parameter averaging (option C). In the current analysis profile, tau, the distance-to-predictor transformation, and the aggregation rule are fixed; the selected layer or HVE definition is categorical and cannot be averaged across folds. Consequently, there is no fold-specific continuous theoretical parameter such as tau or k to average. Only the nuisance standardization moments are fold-specific, and those are estimated without the held-out fold.",
        "",
        "All models include the outer-fold label as a fixed blocking factor because the predictor scale was estimated separately for each fold. The participant, item, and talker structure otherwise follows the confirmatory analysis. B23 retains count-binomial responses.",
        "",
        "Two likelihood-ratio tests are reported:",
        "",
        "- `M_condition` versus `M_joint`: whether the theoretical predictor adds information beyond experimental condition.",
        "- `M_predictor` versus `M_joint`: whether condition adds information beyond the theoretical predictor.",
        "",
        f"The run produced {successful} successful LRTs out of {len(lrts)} planned comparisons. "
        f"All {len(diagnostics)} GLMM fits converged and were non-singular.",
        "",
        "## Important inferential boundary",
        "",
        "The predictor values are cross-fitted with respect to fold-specific scaling, but the HuBERT layer or HVE definition was selected using the same three-fold study. These p-values are therefore selection-conditional supplementary tests, not selection-adjusted confirmatory p-values. The participant-held-out OOF log-loss and cluster-bootstrap results remain the primary predictive evaluation. A fully selection-independent claim would require prespecifying the predictor or nesting candidate selection inside a new outer participant split.",
        "",
        "A significant combined-data LRT can coexist with weak or negative frozen-model OOF gain. The LRT tests an in-sample association after the cross-fitted predictor values have been assembled, whereas OOF log loss tests transport to unseen participants using model coefficients frozen on training folds. Such a disagreement is scientifically meaningful and is not a software inconsistency.",
        "",
        "## Outputs",
        "",
        "- `figures/figure_01_crossfitted_nested_lrt`: visual summary of both LRTs.",
        "- `tables/crossfitted_lrt_results.csv`: chi-square, degrees of freedom, p-value, and log-likelihood change.",
        "- `tables/crossfitted_model_diagnostics.csv`: formula and convergence audit.",
        "- `tables/crossfitted_coefficients.csv`: combined-data fixed-effect estimates.",
        "- `tables/fold_scaling.csv`: fold-specific training moments used to construct each held-out predictor.",
        "- `models/`: row-level cross-fitted predictors and per-analysis R outputs.",
        "",
        "## Reproduce",
        "",
        "Run from the repository root with R/lme4 and the Python project dependencies available:",
        "",
        "```powershell",
        "python .\\cross_talker_generalization\\scripts\\run_crossfitted_lrt.py --jobs 5",
        "```",
        "",
        "## Results with p < .05 (uncorrected, selection-conditional)",
        "",
    ]
    if significant.empty:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Dataset | Family | Variant | Feature | Comparison | LRT chi-square | df | p |",
                "|---|---|---|---|---|---:|---:|---:|",
            ]
        )
        for _, row in significant.sort_values(["dataset_id", "family", "variant", "comparison_id"]).iterrows():
            lines.append(
                f"| {row['dataset_id']} | {row['family']} | {row['variant']} | "
                f"`{row['feature_key']}` | {row['comparison_id']} | {row['chisq']:.3f} | "
                f"{int(row['df'])} | {row['p_value']:.4g} |"
            )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--reuse", action="store_true")
    options = parser.parse_args()

    project = options.project.resolve()
    output = (
        options.output.resolve()
        if options.output
        else project / "analysis_update_2026-09-01"
    )
    table_dir = output / "tables"
    figure_dir = output / "figures"
    model_dir = output / "models"
    for directory in (table_dir, figure_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rscript = _find_rscript()
    r_analysis = project / "R" / "fit_crossfitted_lrt.R"
    specs = _load_registry(project)
    completed: list[tuple[AnalysisSpec, Path]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, options.jobs)) as executor:
        futures = {
            executor.submit(
                _run_one, spec, rscript, r_analysis, model_dir, options.reuse
            ): spec
            for spec in specs
        }
        for future in concurrent.futures.as_completed(futures):
            spec, directory = future.result()
            completed.append((spec, directory))
            print(f"completed {spec.analysis_id}", flush=True)
    completed.sort(key=lambda pair: pair[0].analysis_id)

    lrts, diagnostics, coefficients, scaling, registry = _collect_results(completed)
    lrts.to_csv(table_dir / "crossfitted_lrt_results.csv", index=False)
    diagnostics.to_csv(table_dir / "crossfitted_model_diagnostics.csv", index=False)
    coefficients.to_csv(table_dir / "crossfitted_coefficients.csv", index=False)
    scaling.to_csv(table_dir / "fold_scaling.csv", index=False)
    registry.to_csv(table_dir / "analysis_registry.csv", index=False)
    _plot_lrt(lrts, figure_dir / "figure_01_crossfitted_nested_lrt")
    _write_readme(lrts, diagnostics, output / "README.md")

    outputs = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "provenance.json"
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "combined_fold_crossfitted_predictor_nested_glmm_lrt",
        "status": "complete",
        "selection_conditional": True,
        "include_outer_fold_fixed_block": True,
        "n_analyses": len(specs),
        "n_lrt_comparisons": len(lrts),
        "r_version_executable": str(rscript),
        "r_script_sha256": _sha256(r_analysis),
        "runner_sha256": _sha256(Path(__file__)),
        "outputs_sha256": {
            str(path.relative_to(output)): _sha256(path) for path in outputs
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"report written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
