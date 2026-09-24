"""Recompute AN19 acoustic components with the full baseline's coordinate scaling.

Run with the project Python environment. All new products go into the dated
diagnostic directory; the September 6 results remain untouched. ``prepare``
recomputes distances and model inputs, ``fit`` fits the registered GLMMs, and
``report`` compares old and matched-preprocessing diagnostic results.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ctg.aggregate import aggregate_predictors
from ctg.config import load_profile, load_project
from ctg.distances import compute_distances
from ctg.model_input import make_model_input
from ctg.parallel_glmm import fit_glmm_parallel
from ctg.provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


LABELS = {
    "mfcc39": "MFCC: all 39",
    "strf24_legacy": "STRF: all 24",
    "mfcc_static13": "MFCC: static 13",
    "mfcc_static_spectral12": "MFCC: static without C0",
    "mfcc_c0": "MFCC: C0",
    "mfcc_delta13": "MFCC: delta 13",
    "mfcc_delta_delta13": "MFCC: delta-delta 13",
    "strf_rate_2": "STRF: rate 2",
    "strf_rate_4": "STRF: rate 4",
    "strf_rate_8": "STRF: rate 8",
    "strf_rate_16": "STRF: rate 16",
    "strf_scale_0p25": "STRF: scale 0.25",
    "strf_scale_0p5": "STRF: scale 0.5",
    "strf_scale_1p0": "STRF: scale 1.0",
    "strf_direction_positive": "STRF: positive direction",
    "strf_direction_negative": "STRF: negative direction",
}
OLD_COMPONENT_RUN = "AN19-AN19_acoustic_diagnostic-confirmatory"
OLD_FULL_RUN = "AN19-AN19_acoustic-confirmatory-full-20260906"


def derive_component_moments(mean, scale, indices):
    """Select coordinates *after* adopting the full baseline's fitted moments."""
    mean, scale = np.asarray(mean), np.asarray(scale)
    selected = np.asarray(indices, dtype=int)
    if mean.ndim != 1 or scale.shape != mean.shape:
        raise ValueError("Full standardizer arrays must be matching vectors")
    if selected.size == 0 or selected.min() < 0 or selected.max() >= len(mean):
        raise ValueError("Component indices do not match full feature space")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Full standardizer must contain finite means and positive scales")
    return mean[selected], scale[selected]


def _matching_behavior_rows(frame, reference):
    columns = [c for c in reference.columns if c not in
               {"feature_key", "layer", "raw_distance", "similarity_exp_k"}]
    left = frame[columns].reset_index(drop=True).fillna("")
    right = reference[columns].reset_index(drop=True).fillna("")
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def prepare(output, jobs):
    project = load_project(PROJECT / "configs/project.json")
    original_profile = load_profile(PROJECT / "configs/confirmatory.json")
    derived = PROJECT / "artifacts/derived"
    pairs_path = derived / "AN19-pairs/pairs.csv"
    old_distances = derived / f"{OLD_COMPONENT_RUN}-distances.csv"
    old_full_distances = derived / f"{OLD_FULL_RUN}-distances.csv"
    old_meta = json.loads(Path(str(old_distances) + ".provenance.json").read_text())
    full_meta = json.loads(Path(str(old_full_distances) + ".provenance.json").read_text())
    if old_meta["pairs_sha256"] != sha256_file(pairs_path):
        raise ValueError("Physical pairs changed since the old component run")
    if full_meta["pairs_sha256"] != sha256_file(pairs_path):
        raise ValueError("Full and component physical pairs do not match")
    if full_meta["coordinate_scaling"] != "global_z":
        raise ValueError("Full baseline did not use the expected global standardizer")
    component = project.store("AN19_acoustic_diagnostic")
    subsets = dict(component.feature_subsets)
    subsets.update({"mfcc39": ("mfcc39", tuple(range(39))),
                    "strf24_legacy": ("strf24_legacy", tuple(range(24)))})
    spec = replace(component, store_id="AN19_acoustic_matched", feature_subsets=subsets)
    standardizers = output / "standardizers"
    standardizers.mkdir(parents=True, exist_ok=True)
    records = []
    for feature, (source, indices) in subsets.items():
        parent = derived / "AN19_acoustic-standardizers" / f"AN19_acoustic__{source}.npz"
        with np.load(parent) as bundle:
            mean, scale = derive_component_moments(bundle["mean"], bundle["scale"], indices)
            n_frames = int(bundle["frame_count"])
        target = standardizers / f"{spec.store_id}__{feature}.npz"
        np.savez(target, mean=mean, scale=scale, frame_count=np.asarray(n_frames))
        old_path = derived / "AN19_acoustic_diagnostic-standardizers" / f"{component.store_id}__{feature}.npz"
        old_mean_error = old_scale_error = None
        if old_path.is_file():
            with np.load(old_path) as old:
                old_mean_error = float(np.max(np.abs(old["mean"] - mean)))
                old_scale_error = float(np.max(np.abs(old["scale"] - scale)))
        records.append({"feature_key": feature, "source_feature": source,
                        "dimensions": len(indices), "frame_count": n_frames,
                        "parent_sha256": sha256_file(parent), "sliced_sha256": sha256_file(target),
                        "old_fitted_mean_max_abs_difference": old_mean_error,
                        "old_fitted_scale_max_abs_difference": old_scale_error})
    atomic_write_csv(output / "standardizer_audit.csv", pd.DataFrame(records))
    profile = dict(original_profile)
    profile["profile_id"] = "an19_matched_acoustic_components_20260917"
    profile["coordinate_scaling"] = dict(original_profile["coordinate_scaling"], acoustic_subset="global_z")
    atomic_write_json(output / "profile.json", profile)
    distances_path = output / "distances.csv"
    distances = compute_distances(pairs_path=pairs_path, spec=spec,
                                  feature_keys=list(LABELS), profile=profile,
                                  output_path=distances_path, jobs=jobs,
                                  standardizer_dir=str(standardizers))
    previous_full = pd.read_csv(old_full_distances)
    comparison = distances.merge(previous_full, on=["feature_key", "pair_id"],
                                 suffixes=("_new", "_old"), validate="one_to_one")
    if len(comparison) != len(previous_full):
        raise ValueError("Recomputed full baselines lost physical pairs")
    full_errors = {}
    for key, rows in comparison.groupby("feature_key"):
        np.testing.assert_allclose(rows.raw_distance_new, rows.raw_distance_old, rtol=1e-12, atol=1e-12)
        full_errors[key] = float(np.max(np.abs(rows.raw_distance_new - rows.raw_distance_old)))
    old_components = pd.read_csv(old_distances)
    change = distances.merge(old_components, on=["feature_key", "pair_id"],
                             suffixes=("_new", "_old"), validate="one_to_one")
    change_summary = []
    for key, rows in change.groupby("feature_key"):
        change_summary.append({"feature_key": key, "n_pairs": len(rows),
                               "old_scaling": "none", "new_scaling": "global_z",
                               "pearson_old_new": rows.raw_distance_new.corr(rows.raw_distance_old),
                               "spearman_old_new": rows.raw_distance_new.corr(rows.raw_distance_old, method="spearman"),
                               "old_mean_distance": rows.raw_distance_old.mean(),
                               "new_mean_distance": rows.raw_distance_new.mean()})
    atomic_write_csv(output / "distance_scaling_comparison.csv", pd.DataFrame(change_summary))
    predictors_path = output / "predictors.csv"
    aggregate_predictors(cells_path=derived / "AN19-pairs/cells.csv", distances_path=distances_path,
                         profile=profile, output_path=predictors_path)
    model = make_model_input(spec=project.dataset("AN19"), predictors_path=predictors_path,
                             folds_path=derived / "AN19-folds.csv", output_path=output / "model_input.csv")
    old_model = pd.read_csv(derived / f"{OLD_FULL_RUN}-model-input.csv")
    reference = old_model.loc[old_model.feature_key.eq("mfcc39")]
    for feature, rows in model.groupby("feature_key"):
        _matching_behavior_rows(rows, reference)
    checks = {**runtime_record(), "stage": "matched_acoustic_preparation", "status": "complete",
              "historical_component_scaling": old_meta["coordinate_scaling"],
              "historical_full_scaling": full_meta["coordinate_scaling"],
              "new_scaling": "global_z", "full_distance_max_abs_difference": full_errors,
              "same_behavior_rows_and_folds_all_features": True,
              "n_pairs_per_feature": int(len(comparison) / 2), "feature_count": len(subsets),
              "model_rows_per_feature": len(reference),
              "available_rows_per_feature": int(reference.predictor_status.eq("available").sum()),
              "feature_h5_sha256": sha256_file(spec.path),
              "pair_table_sha256": sha256_file(pairs_path),
              "fold_table_sha256": sha256_file(derived / "AN19-folds.csv"),
              "old_component_distances_sha256": sha256_file(old_distances),
              "old_full_distances_sha256": sha256_file(old_full_distances),
              "model_input_sha256": sha256_file(output / "model_input.csv"),
              "note": "The earlier standardizers existed but acoustic_subset fell through to none. This rerun uses exact full-baseline moments. It does not explain the high full-baseline result by itself."}
    atomic_write_json(output / "preparation_checks.json", checks)
    print(json.dumps(checks, indent=2), flush=True)


def _exact_fold_interval(values):
    """Percentile interval from enumerating all resamples of the three folds."""
    values = np.asarray(values, dtype=float)
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("Expected three finite fold values")
    means = np.asarray([np.mean(sample) for sample in product(values, repeat=3)])
    return np.quantile(means, [0.025, 0.975])


def report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def extract(directory, version):
        coefficients = pd.read_csv(directory / "coefficients.csv")
        z = coefficients.loc[coefficients.scope.eq("cv_train") &
                             coefficients.model_id.eq("M_predictor") &
                             coefficients.term.eq("similarity_z"),
                             ["feature_key", "fold", "z_value"]]
        metrics = pd.read_csv(directory / "cv_metrics.csv")
        loss = metrics.loc[metrics.scope.eq("oof_fold") & metrics.model_id.eq("M_predictor"),
                           ["feature_key", "fold", "mean_log_loss", "total_trials"]]
        result = z.merge(loss, on=["feature_key", "fold"], validate="one_to_one")
        result["version"] = version
        return result

    old_models = PROJECT / "artifacts/models"
    frames = [extract(output / "models", "matched_global_z"),
              extract(old_models / OLD_COMPONENT_RUN, "historical_none"),
              extract(old_models / OLD_FULL_RUN, "historical_full_global_z")]
    fold_table = pd.concat(frames, ignore_index=True)
    atomic_write_csv(output / "fold_results.csv", fold_table)
    summaries = []
    for (version, feature), rows in fold_table.groupby(["version", "feature_key"]):
        record = {"version": version, "feature_key": feature, "n_folds": len(rows)}
        for metric in ("z_value", "mean_log_loss"):
            low, high = _exact_fold_interval(rows[metric])
            record.update({f"{metric}_mean": rows[metric].mean(),
                           f"{metric}_ci_low": low, f"{metric}_ci_high": high})
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    atomic_write_csv(output / "summary.csv", summary)
    diagnostics = pd.read_csv(output / "models/diagnostics.csv")
    scope_summary = diagnostics.groupby(["scope", "fold", "model_id"], dropna=False).agg(
        n_random_structures=("random_structure", "nunique"),
        n_features=("feature_key", "nunique"),
        n_failures=("fit_ok", lambda x: int((~x.astype(bool)).sum())),
        n_singular=("singular", lambda x: int(x.astype(bool).sum())),
        n_non_ok_convergence=("convergence", lambda x: int(x.ne("ok").sum())),
    ).reset_index()
    atomic_write_csv(output / "random_structure_check.csv", scope_summary)
    positions = np.arange(len(LABELS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
    current = summary.loc[summary.version.eq("matched_global_z")].set_index("feature_key").loc[list(LABELS)]
    for axis, metric in zip(axes, ("z_value", "mean_log_loss")):
        mean = current[f"{metric}_mean"].to_numpy()
        low, high = current[f"{metric}_ci_low"].to_numpy(), current[f"{metric}_ci_high"].to_numpy()
        axis.errorbar(mean, positions, xerr=np.stack((mean-low, high-mean)),
                      fmt="o", color="#1f658c", capsize=3, markersize=6, label="Matched coordinates")
        for index, feature in enumerate(LABELS):
            folds = fold_table.loc[fold_table.version.eq("matched_global_z") & fold_table.feature_key.eq(feature)]
            axis.scatter(folds[metric], np.full(3, index) + np.array([-.09, 0, .09]),
                         s=15, color="#777777", alpha=.55, zorder=1)
            old = summary.loc[summary.version.eq("historical_none") & summary.feature_key.eq(feature)]
            if not old.empty:
                axis.scatter(old[f"{metric}_mean"], [index], marker="x", s=45, color="#bd7647",
                             label="Earlier unscaled components" if feature == "mfcc_static13" else None)
        axis.axhspan(-.4, 1.4, color="#eeeeee", alpha=.45, zorder=0)
        axis.axhline(1.5, color="#dddddd", linewidth=.8)
        axis.axhline(6.5, color="#dddddd", linewidth=.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=.15)
    axes[0].set_yticks(positions, list(LABELS.values()))
    axes[0].invert_yaxis()
    axes[0].axvline(1.96, color="#777777", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Predictor z-value (training folds)")
    axes[1].set_xlabel("Mean held-out log loss per word response")
    axes[0].set_title("A. Predictor-only association")
    axes[1].set_title("B. Frozen-model prediction")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.055), frameon=False, fontsize=9)
    fig.suptitle("AN19 acoustic components: matched coordinate standardization", y=.985)
    fig.tight_layout(rect=[0, 0, 1, .955])
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"matched_acoustic_components.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    atomic_write_json(output / "report_metadata.json", {
        **runtime_record(), "script_sha256": sha256_file(Path(__file__)),
        "figure": "matched_acoustic_components",
        "association": "M_predictor coefficient for negative training-standardized DTW distance; fitted on two participant folds",
        "prediction": "M_predictor frozen fixed-effects-only prediction in the third participant fold; mean word-response log loss",
        "uncertainty": "95% percentile interval over all 27 ordered bootstrap resamples of three fold summaries; equal fold weights",
        "interval_caveat": "Only three folds; training samples overlap. Intervals describe variation across these folds, not an independently replicated population-level inferential test.",
        "reference_line": "z=1.96, nominal two-sided Wald threshold for an individual coefficient, without multiple-comparison correction",
        "ceiling": "No ceiling normalization: no new matched z ceiling was fitted in this diagnostic",
        "scope": "Prespecified acoustic component diagnostic, not the historical test-refit z figure or a retuned SBI analysis",
        "component_design": "Each subset recomputes its own DTW alignment using the same parent coordinate moments; subset scores are not additive contributions to the full-space DTW score",
        "rows": "All 16 spaces share 5685 available word responses and the original participant folds; untrained controls have no exposure predictor",
        "models": "M_null, M_condition, M_predictor, M_joint; all-model fitting preserves the registered shared-within-scope random-structure fallback",
        "all_features_share_random_structure_per_scope": bool(scope_summary.n_random_structures.eq(1).all()),
        "selected_fit_failures": int(scope_summary.n_failures.sum()),
        "selected_singular_fits": int(scope_summary.n_singular.sum()),
        "selected_non_ok_convergence": int(scope_summary.n_non_ok_convergence.sum()),
    })
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "report"))
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "analysis/acoustic_baselines/components")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args.output, args.jobs)
    elif args.stage == "fit":
        fit_glmm_parallel(input_path=args.output / "model_input.csv", output_dir=args.output / "models",
                          jobs=args.jobs, model_set="all")
    else:
        report(args.output)


if __name__ == "__main__":
    main()
