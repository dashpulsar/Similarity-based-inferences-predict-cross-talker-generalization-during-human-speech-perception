"""Redraw stored z statistics without refitting models or changing their meaning.

Run from the repository root. Historical test-refit and revised training-fit
results are deliberately kept separate; only the former has a stored z ceiling.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


DATASETS = ("AN19", "X21", "B23")
LAYERS = [*(f"cnn_{i}" for i in range(2, 7)), *(f"tr_{i}" for i in range(0, 25, 2))]
COLORS = {"base": "#20639b", "ft": "#b43b35"}
MEASURES = [
    "overall", "overall_order_sensitive",
    *[f"{prefix}_{unit}" for unit in ("sentence", "word", "phoneme")
      for prefix in ("within_token", "within_type", "between_type", "order", "mean_dissimilarity")],
]


def fold_ci(values):
    """Exact percentile bootstrap of the mean over three folds (27 samples)."""
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Expected exactly three finite fold z values")
    draws = values[np.array(list(itertools.product(range(3), repeat=3)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def normalized(values, ceiling):
    fold_ci(ceiling)
    mean = float(np.mean(ceiling))
    if mean <= 0:
        raise ValueError("Mean ceiling z must be positive")
    return np.asarray(values, dtype=float) * 100 / mean


def label(key):
    return {"mfcc39": "MFCC\n(39-D)", "strf24_legacy": "STRF\n(24-D)"}.get(
        key, key.replace("cnn_", "CNN-").replace("tr_", "Tr-")
    )


def measure_label(key):
    if key == "overall":
        return "Overall frame dispersion"
    if key == "overall_order_sensitive":
        return "Global exposure-order transitions"
    prefix, unit = key.rsplit("_", 1)
    return {
        "within_token": "Within token", "within_type": "Within type",
        "between_type": "Between types", "order": "Adjacent frames",
        "mean_dissimilarity": "Within-type DTW",
    }[prefix] + f": {unit}"


def check_folds(frame, keys):
    counts = frame.groupby(keys, dropna=False)["fold"].agg(["count", "nunique"])
    if not counts["count"].eq(3).all() or not counts["nunique"].eq(3).all():
        raise ValueError(f"Incomplete/duplicate folds for {keys}")
    if not np.isfinite(frame["z_value"].to_numpy(float)).all():
        raise ValueError("Nonfinite z values")


def read_csv(path, repository, sources):
    frame = pd.read_csv(path)
    relative = path.relative_to(repository).as_posix()
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    frame["source_file"] = relative
    # Historical absolute paths are not portable and disclose local machine names.
    frame = frame.drop(columns=["source_path"], errors="ignore")
    return frame


def collect(repository, sources):
    old = repository / "cross_talker_generalization/analysis/reference/tables"
    sbi = read_csv(old / "notebook_fold_z_percent_ceiling.csv", repository, sources)
    hve = read_csv(old / "variability_all_fold_z_percent_ceiling.csv", repository, sources)
    ceiling = read_csv(old / "notebook_behavioral_ceiling_z.csv", repository, sources)
    hve_ceiling = read_csv(old / "variability_behavioral_ceiling_z.csv", repository, sources)
    for frame in (sbi, hve, ceiling, hve_ceiling):
        frame.rename(columns={"z_test": "z_value"}, inplace=True)
        frame["scope"] = "historical_test_fold_refit"
    check_folds(sbi, ["dataset_id", "source", "layer"])
    check_folds(hve, ["dataset_id", "model_variant", "measure", "layer"])
    check_folds(ceiling, ["dataset_id"])
    check_folds(hve_ceiling, ["dataset_id"])
    current = []
    models = repository / "cross_talker_generalization/artifacts/models"
    for dataset in DATASETS:
        for variant in ("base", "ft"):
            suffix = "selection" if dataset == "B23" else "revised-selection"
            folder = models / f"{dataset}-HVE-{variant}-{suffix}"
            frame = read_csv(folder / "coefficients.csv", repository, sources)
            frame = frame.loc[frame.scope.eq("cv_train") & frame.model_id.eq("M_predictor")
                              & frame.term.eq("variability_z")].copy()
            frame["model_variant"] = variant
            frame[["layer", "measure"]] = frame.feature_key.str.split("::", expand=True)
            diagnostics = read_csv(folder / "diagnostics.csv", repository, sources)
            diagnostics = diagnostics.loc[diagnostics.scope.eq("cv_train")
                                          & diagnostics.model_id.eq("M_predictor")]
            audit_columns = ["feature_key", "fold", "fit_ok", "singular", "convergence", "formula", "n_rows"]
            frame = frame.merge(diagnostics[audit_columns], on=["feature_key", "fold"],
                                how="left", validate="one_to_one")
            if not (frame.fit_ok.astype(str).str.lower().eq("true").all()
                    and frame.singular.astype(str).str.lower().eq("false").all()
                    and frame.convergence.eq("ok").all()):
                raise ValueError(f"Unaccepted fit in {folder.name}")
            frame["scope"] = "revised_training_two_folds"
            frame["sample_stratum"] = np.where(
                (dataset == "B23") & frame.measure.eq("overall_order_sensitive"),
                "global_order_97_participants", "order_independent_168_participants" if dataset == "B23"
                else "all_exposure_participants")
            current.append(frame)
    current = pd.concat(current, ignore_index=True)
    check_folds(current, ["dataset_id", "model_variant", "measure", "layer"])
    for _, group in current.groupby(["dataset_id", "model_variant", "measure"]):
        if set(group.layer) != set(LAYERS):
            raise ValueError("Revised HVE is missing a registered layer")
    return sbi, hve, ceiling, hve_ceiling, current


def references(ax, ceiling=None, percent=False):
    ax.axhline(0, color="#777777", linewidth=.7, zorder=0)
    if ceiling is not None:
        values = normalized(ceiling, ceiling) if percent else ceiling
        center, low, high = fold_ci(values)
        ax.axhspan(low, high, color="#bbbbbb", alpha=.4, zorder=-2)
        ax.axhline(center, color="black", linestyle="--", linewidth=1, zorder=-1)
    scale = 100 / np.mean(ceiling) if percent else 1
    for sign in (1, -1):
        ax.axhline(sign * 1.96 * scale, color="#b56e00", linestyle=":", linewidth=1, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Predictor z / mean ceiling z (%)" if percent else "Predictor Wald z")


def plot_points(ax, group, keys, positions, color, percent, ceiling, connect=True, offset=0):
    means, low, high = [], [], []
    for key, position in zip(keys, positions):
        part = group.loc[group.layer.eq(key)].sort_values("fold")
        values = part.z_value.to_numpy(float)
        if percent:
            values = normalized(values, ceiling)
        mean, lo, hi = fold_ci(values)
        means.append(mean)
        low.append(lo)
        high.append(hi)
        ax.scatter(position + offset + np.array([-.08, 0, .08]), values,
                   s=11, color="#777777", alpha=.5, zorder=2)
    ax.errorbar(np.asarray(positions) + offset, means,
                yerr=[np.array(means) - low, np.array(high) - means], color=color,
                marker="o", markersize=3.5, linestyle="-" if connect else "none",
                linewidth=1.3, elinewidth=1.1, capsize=2, zorder=3)


def legend(ceiling=True, variants=False):
    handles = [Line2D([], [], marker="o", color="#777777", linestyle="none", markersize=4,
                      label="Individual folds"),
               Line2D([], [], color="black", marker="o", markersize=4,
                      label="Mean and 95% fold-bootstrap CI")]
    if variants:
        handles += [Line2D([], [], color=color, label="Base" if v == "base" else "ASR-FT")
                    for v, color in COLORS.items()]
    if ceiling:
        handles += [Patch(facecolor="#bbbbbb", alpha=.4, label="Ceiling 95% fold-bootstrap CI"),
                    Line2D([], [], color="black", linestyle="--", label="Mean ceiling")]
    handles += [Line2D([], [], color="#b56e00", linestyle=":", label="Nominal z = +/-1.96")]
    return handles


def save(fig, output, name, title, scope, manifest):
    for extension in ("png", "svg"):
        fig.savefig(output / "figures" / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"figure": name, "title": title, "scope": scope,
                     "png": f"figures/{name}.png", "svg": f"figures/{name}.svg"})


def sbi_figures(sbi, ceilings, output, manifest):
    keys = ["mfcc39", "strf24_legacy", *LAYERS]
    positions = [0, 1, *range(3, len(LAYERS) + 3)]
    for dataset in DATASETS:
        ceiling = ceilings.loc[ceilings.dataset_id.eq(dataset)].sort_values("fold").z_value.to_numpy()
        for variant in ("base", "ft"):
            group = sbi.loc[sbi.dataset_id.eq(dataset) & sbi.source.isin([variant, "mfcc39", "strf24_legacy"])]
            fig, axes = plt.subplots(2, 1, figsize=(13.5, 8), layout="constrained")
            for ax, percent in zip(axes, (True, False)):
                references(ax, ceiling, percent)
                plot_points(ax, group, keys[:2], positions[:2], "black", percent, ceiling, connect=False)
                plot_points(ax, group, keys[2:], positions[2:], "black", percent, ceiling)
                ax.set_xticks(positions, [label(k) for k in keys], rotation=45, ha="right", fontsize=9)
                ax.set_xlabel("Feature space (18 registered HuBERT layers)")
                ax.margins(x=.02, y=.12)
            title = f"{dataset} | SBI | HuBERT {'base' if variant == 'base' else 'ASR-FT'}"
            fig.suptitle(title + "\nStored notebook test-fold refits: format review, not revised-model results", fontsize=13)
            axes[0].legend(handles=legend(), loc="upper left", fontsize=7, ncol=2)
            save(fig, output, f"sbi_{dataset.lower()}_{variant}_z", title,
                 "historical_test_fold_refit", manifest)


def hve_figures(frame, ceilings, output, manifest, historical):
    for dataset in DATASETS:
        subset = frame.loc[frame.dataset_id.eq(dataset)]
        measures = [m for m in MEASURES if m in set(subset.measure)]
        ceiling = (ceilings.loc[ceilings.dataset_id.eq(dataset)].sort_values("fold").z_value.to_numpy()
                   if historical else None)
        for start in range(0, len(measures), 4):
            fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
            for ax, measure in zip(axes.flat, measures[start:start + 4]):
                references(ax, ceiling, historical)
                for variant, offset in (("base", -.12), ("ft", .12)):
                    group = subset.loc[subset.measure.eq(measure) & subset.model_variant.eq(variant)]
                    plot_points(ax, group, LAYERS, np.arange(len(LAYERS)), COLORS[variant],
                                historical, ceiling, offset=offset)
                title = measure_label(measure)
                if dataset == "B23" and not historical:
                    title += "\n97 participants (ordered)" if measure == "overall_order_sensitive" else "\n168 participants"
                ax.set_title(title, fontsize=11)
                ax.set_xticks(range(len(LAYERS)), [label(k) for k in LAYERS], rotation=55, ha="right", fontsize=8)
                ax.set_xlabel("HuBERT layer")
                ax.margins(y=.13)
            for ax in list(axes.flat)[len(measures[start:start + 4]):]:
                ax.set_visible(False)
            status = "Stored notebook test-refit z / ceiling: historical definitions" if historical else (
                "Revised HVE: predictor-only training-fold z; no compatible z ceiling available")
            title = f"{dataset} | HVE definitions | page {start // 4 + 1}"
            fig.suptitle(title + "\n" + status, fontsize=13)
            fig.legend(handles=legend(historical, True), loc="outside lower center", ncol=4, fontsize=8)
            save(fig, output, f"hve_{'historical' if historical else 'revised'}_{dataset.lower()}_{start // 4 + 1:02d}",
                 title, "historical_test_fold_refit" if historical else "revised_training_two_folds", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = repository / "cross_talker_generalization/analysis/model_comparison/reference_checks/z_value_review"
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    sources, manifest = {}, []
    sbi, historical_hve, ceiling, hve_ceiling, revised = collect(repository, sources)
    for name, frame in (("historical_sbi_fold_z", sbi), ("historical_hve_fold_z", historical_hve),
                        ("historical_ceiling_fold_z", ceiling), ("historical_hve_ceiling_fold_z", hve_ceiling),
                        ("revised_hve_training_fold_z", revised)):
        frame.to_csv(output / "tables" / f"{name}.csv", index=False)
    sbi_figures(sbi, ceiling, output, manifest)
    hve_figures(historical_hve, hve_ceiling, output, manifest, True)
    hve_figures(revised, None, output, manifest, False)
    summaries = []
    for family, frame, keys in (
        ("SBI", sbi, ["dataset_id", "source", "layer", "scope"]),
        ("HVE", historical_hve, ["dataset_id", "model_variant", "measure", "layer", "scope"]),
        ("HVE", revised, ["dataset_id", "model_variant", "measure", "layer", "scope"]),
    ):
        for key, part in frame.groupby(keys):
            mean, low, high = fold_ci(part.z_value)
            summaries.append(dict(zip(keys, key), family=family, mean_z=mean, ci_low=low, ci_high=high,
                                  n_folds=3))
    pd.DataFrame(summaries).to_csv(output / "tables/z_summary.csv", index=False)
    coverage = revised.groupby(["dataset_id", "model_variant", "measure", "sample_stratum"]).agg(
        n_layers=("layer", "nunique"), n_fold_values=("z_value", "size")).reset_index()
    coverage.to_csv(output / "tables/revised_hve_coverage.csv", index=False)
    availability = []
    for dataset in DATASETS:
        for measure in MEASURES:
            available = not coverage.loc[coverage.dataset_id.eq(dataset) & coverage.measure.eq(measure)].empty
            availability.append({"dataset_id": dataset, "measure": measure,
                                 "status": "plotted" if available else "no_estimate_in_selected_sources",
                                 "note": "Not an assumed zero" if not available else "Both variants, 18 layers, 3 folds"})
    pd.DataFrame(availability).to_csv(output / "tables/hve_method_availability.csv", index=False)
    pd.DataFrame(manifest).to_csv(output / "tables/figure_manifest.csv", index=False)
    sources[Path(__file__).resolve().relative_to(repository).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    validation = {"figures": len(manifest), "sbi_historical_fold_values": len(sbi),
                  "hve_historical_fold_values": len(historical_hve), "hve_revised_fold_values": len(revised),
                  "bootstrap_resamples": 27, "new_glmm_fits": 0, "source_sha256": sources}
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    index = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>SBI and HVE z review</title>",
             "<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px}img{width:100%}article{margin:40px 0}small{color:#555}</style>",
             "<h1>SBI and HVE z-value review</h1><p>The SBI display format is not a designation of manuscript main figures.</p>",
             "<p>Historical test-refit z/ceiling and revised HVE training-fold z are separate analyses. No new GLMM was fitted. ",
             "See <a href='README.md'>README</a> and <a href='../../docs/FLORIAN_REQUIREMENTS_REVIEW.md'>requirements review</a>.</p>",
             "<p><a href='#historical_test_fold_refit'>Stored z/ceiling displays</a> | "
             "<a href='#revised_training_two_folds'>Revised HVE z</a> | "
             "<a href='#other'>Other requested figure families</a></p>"]
    for scope in ("historical_test_fold_refit", "revised_training_two_folds"):
        index.append(f"<h2 id='{scope}'>{html.escape(scope.replace('_', ' '))}</h2>")
        for entry in manifest:
            if entry["scope"] != scope:
                continue
            index.append(f"<article><h3>{html.escape(entry['title'])}</h3><a href='{entry['svg']}'>SVG</a> | "
                         f"<a href='{entry['png']}'>PNG</a><img loading='lazy' src='{entry['png']}' "
                         f"alt='{html.escape(entry['title'])}'></article>")
    index.append("<h2 id='other'>Other requested figure families (existing outputs, not rerun here)</h2>")
    companion = []
    project = repository / "cross_talker_generalization"
    groups = [
        ("Manuscript Figure 1/2: incomplete drafts, not final panels",
         project / "analysis/model_comparison/reference_checks/figure_drafts", "*.png"),
        ("Conditional and pooled S-curves: August 21 descriptive analysis",
         project / "analysis/reference/figures/s_curves_tr24", "*.png"),
        ("Matched-content talker distance matrices: August 21",
         project / "analysis/reference/figures", "figure_05b[123]*.png"),
        ("Combined-fold nested-model comparisons: September 1, selection-conditional",
         project / "analysis/model_comparison/pooled_lrt/figures", "*.png"),
    ]
    for title, directory, pattern in groups:
        index.append(f"<h3>{html.escape(title)}</h3><ul>")
        for path in sorted(directory.glob(pattern)):
            relative = Path(os.path.relpath(path, output)).as_posix()
            companion.append({"family": title, "path": relative, "status": "existing_not_rerun"})
            index.append(f"<li><a href='{html.escape(relative)}'>{html.escape(path.stem)}</a></li>")
        index.append("</ul>")
    pd.DataFrame(companion).to_csv(output / "tables/companion_figure_index.csv", index=False)
    (output / "index.html").write_text("\n".join(index) + "</html>", encoding="utf-8")
    print(json.dumps({k: v for k, v in validation.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
