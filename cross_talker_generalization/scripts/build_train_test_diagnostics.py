"""Summarize matched training/test scores; leave the requested ratio undefined.

Example (from the repository root):
  python cross_talker_generalization/scripts/build_train_test_diagnostics.py \
    --runs cross_talker_generalization/analysis/diagnostics/train_test

Fits are produced with the existing ctg fit-glmm-parallel command. This script
only reads those results and creates tables plus direct train/test loss plots.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ctg.provenance import atomic_write_csv, atomic_write_json, sha256_file


def validate_scores(scores: pd.DataFrame) -> None:
    keys = ["run_label", "dataset_id", "feature_key", "fold", "model_id"]
    if scores.duplicated(keys + ["split"]).any():
        raise ValueError("Duplicated split scores")
    if not scores.groupby(keys)["split"].agg(set).map(lambda x: x == {"train", "test"}).all():
        raise ValueError("Each fitted model must have both train and test scores")
    if not scores.groupby(keys[:-2] + ["model_id"])["fold"].agg(set).map(lambda x: x == {0, 1, 2}).all():
        raise ValueError("Expected all three folds for each feature/model")
    if not scores["prediction_convention"].eq("fixed_effects_only_re_form_NA").all():
        raise ValueError("Inconsistent random-effects scoring convention")
    if not scores["score_status"].eq("ok").all():
        raise ValueError("Some split scores failed; inspect the source diagnostics")
    if not (scores["total_trials"] > 0).all():
        raise ValueError("Nonpositive response counts")
    if not np.isfinite(scores["mean_log_loss"]).all():
        raise ValueError("Nonfinite mean log loss")
    if not scores["mean_log_loss"].ge(0).all():
        raise ValueError("Predictive log loss must be nonnegative")
    np.testing.assert_allclose(scores["mean_log_loss"], scores["total_log_loss"] / scores["total_trials"])
    np.testing.assert_allclose(scores["mean_log_likelihood"], -scores["mean_log_loss"])
    for col in ["train_predictor_mean", "train_predictor_sd", "random_structure", "formula"]:
        if not scores.groupby(keys)[col].nunique(dropna=False).eq(1).all():
            raise ValueError(f"Train and test differ in {col}")


def three_fold_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Exact enumeration of the 3^3 bootstrap samples of three fold scores."""
    values = np.asarray(values, dtype=float)
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("Exactly three finite fold values required")
    means = np.array([values[list(indices)].mean() for indices in product(range(3), repeat=3)])
    low, high = np.quantile(means, [.025, .975])
    return float(values.mean()), float(low), float(high)


def feature_order(key: str) -> tuple[int, int]:
    if key == "mfcc39":
        return 0, 0
    if key == "strf24_legacy":
        return 0, 1
    prefix, _, value = key.partition("_")
    return {"cnn": 1, "projection": 2, "tr": 3}.get(prefix, 2), int(value) if value.isdigit() else 0


def label(key: str) -> str:
    return {"mfcc39": "MFCC39", "strf24_legacy": "STRF24", "projection": "Projection"}.get(
        key, key.replace("cnn_", "CNN-").replace("tr_", "Tr-")
    )


def build(runs: Path) -> None:
    manifest = json.loads((runs / "run_inputs.json").read_text(encoding="utf-8"))
    frames, diagnostics, checks = [], [], []
    for source in manifest["inputs"]:
        name = source["run_label"]
        directory = runs / name
        scores = pd.read_csv(directory / "train_test_scores.csv")
        scores["run_label"] = name
        frames.append(scores)
        diag = pd.read_csv(directory / "diagnostics.csv")
        diag["run_label"] = name
        diagnostics.append(diag)
        test = scores.loc[scores["split"].eq("test")]
        old = pd.read_csv(directory / "cv_metrics.csv").query("scope == 'oof_fold'")
        joined = test.merge(old, on=["dataset_id", "feature_key", "fold", "model_id"],
                            validate="one_to_one", suffixes=("_new", "_existing"))
        assert len(joined) == len(test) == len(old)
        np.testing.assert_allclose(joined.mean_log_loss_new, joined.mean_log_loss_existing)
        cv_diag = diag.loc[diag.scope.eq("cv_train")]
        assert len(cv_diag) * 2 == len(scores), "Missing score row for a fitted/failed CV model"
        checks.append(dict(run_label=name, n_features=scores.feature_key.nunique(),
                           n_score_rows=len(scores), existing_oof_scores_match=True,
                           singular_cv_fits=int(cv_diag.singular.astype(str).str.lower().eq("true").sum()),
                           nonconverged_cv_fits=int(cv_diag.convergence.ne("ok").sum()),
                           score_file_sha256=sha256_file(directory / "train_test_scores.csv")))
    scores = pd.concat(frames, ignore_index=True)
    validate_scores(scores)
    tables, figures = runs / "tables", runs / "figures"
    atomic_write_csv(tables / "matched_train_test_scores.csv", scores)
    atomic_write_csv(tables / "fit_diagnostics.csv", pd.concat(diagnostics, ignore_index=True))
    atomic_write_csv(tables / "validation.csv", pd.DataFrame(checks))
    summary = []
    grouping = ["run_label", "dataset_id", "feature_key", "model_id", "split"]
    for keys, group in scores.groupby(grouping):
        mean, low, high = three_fold_ci(group.sort_values("fold").mean_log_loss.to_numpy())
        summary.append(dict(zip(grouping, keys), fold_mean_log_loss=mean,
                            fold_bootstrap_ci_low=low, fold_bootstrap_ci_high=high,
                            response_weighted_mean_log_loss=group.total_log_loss.sum() / group.total_trials.sum(),
                            n_folds=len(group)))
    summary = pd.DataFrame(summary)
    atomic_write_csv(tables / "three_fold_summary.csv", summary)
    # Keep paired differences available without choosing the pending ratio.
    paired = scores.pivot(index=["run_label", "dataset_id", "feature_key", "model_id", "fold"],
                          columns="split", values="mean_log_loss").reset_index()
    paired["test_minus_train_mean_log_loss"] = paired["test"] - paired["train"]
    atomic_write_csv(tables / "paired_train_test_losses.csv", paired)
    figures.mkdir(parents=True, exist_ok=True)
    plot_records = []
    for name in manifest["inputs"]:
        run_label = name["run_label"]
        neural = summary.loc[summary.run_label.eq(run_label) & summary.model_id.eq("M_predictor")]
        # A layer profile needs multiple layers; fixed-Tr-24 reruns remain in the tables.
        if neural.feature_key.nunique() < 3 or not neural.feature_key.str.startswith("tr_").any():
            continue
        dataset = neural.dataset_id.iloc[0]
        acoustic = summary.loc[summary.run_label.eq(f"{dataset}-acoustic") & summary.model_id.eq("M_predictor")]
        selected = pd.concat([acoustic, neural], ignore_index=True)
        keys = sorted(selected.feature_key.unique(), key=feature_order)
        xs = np.arange(len(keys), dtype=float)
        if keys[:2] == ["mfcc39", "strf24_legacy"]:
            xs[2:] += .7
        fig, ax = plt.subplots(figsize=(10.5, 4.2), layout="constrained")
        for split, color, caption in [("train", "#236b8e", "Training"), ("test", "#bf6419", "Held-out test")]:
            part = selected.loc[selected.split.eq(split)].set_index("feature_key").loc[keys]
            mean = part.fold_mean_log_loss.to_numpy()
            lower = part.fold_bootstrap_ci_low.to_numpy()
            upper = part.fold_bootstrap_ci_high.to_numpy()
            display_x = xs + (-.07 if split == "train" else .07)
            ax.errorbar(display_x, mean, yerr=[mean-lower, upper-mean], color=color,
                        fmt="s" if split == "train" else "o", ms=4, capsize=2, label=caption)
            # Keep acoustic baselines visually separate from the layer trajectory.
            start = 2 if keys[:2] == ["mfcc39", "strf24_legacy"] else 0
            ax.plot(display_x[start:], mean[start:], color=color, lw=1.2,
                    linestyle="--" if split == "train" else "-")
            fold_rows = scores.loc[scores.run_label.isin([run_label, f"{dataset}-acoustic"]) &
                                  scores.model_id.eq("M_predictor") & scores.split.eq(split)]
            for i, key in enumerate(keys):
                ys = fold_rows.loc[fold_rows.feature_key.eq(key)].sort_values("fold").mean_log_loss.to_numpy()
                ax.scatter(display_x[i] + np.array([-.025, 0, .025]), ys, s=10, color=color, alpha=.35)
        variant = "ASR-FT" if run_label.endswith("-ft") else "base"
        ax.set(title=f"{dataset} | HuBERT {variant} | SBI", xlabel="Feature space",
               ylabel="Mean log loss per word response")
        ax.set_xticks(xs, [label(key) for key in keys], rotation=45, ha="right", fontsize=8)
        shared = scores.loc[scores.dataset_id.eq(dataset) & scores.model_id.eq("M_predictor"), "mean_log_loss"]
        lo, hi = shared.min(), shared.max()
        margin = max((hi - lo) * .07, .001)
        ax.set_ylim(lo - margin, hi + margin)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        relevant = scores.loc[scores.run_label.isin([run_label, f"{dataset}-acoustic"]) &
                              scores.model_id.eq("M_predictor")]
        if relevant.singular.astype(str).str.lower().eq("true").any() or relevant.convergence.ne("ok").any():
            fig.suptitle("Diagnostic draft: some fits have warnings (see fit_diagnostics.csv)", fontsize=9)
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(figures / f"{run_label}_train_test_loss.{ext}", dpi=180)
        plt.close(fig)
        plot_records.append(dict(run_label=run_label, figure=f"{run_label}_train_test_loss",
                                 score="matched mean log loss", ratio="not selected"))
    atomic_write_csv(tables / "figure_inventory.csv", pd.DataFrame(plot_records))
    atomic_write_json(tables / "summary_validation.json", dict(
        passed=True, runs=len(checks), score_rows=len(scores),
        unique_feature_runs=len(scores[["run_label", "feature_key"]].drop_duplicates()),
        uncertainty="Exact enumeration of all 27 resamples of three fold scores; descriptive fold variability",
        ratio_status="Not chosen; figures display separate training and test losses",
        source_checks=checks))
    print(f"Validated {len(checks)} runs and {len(scores)} split scores; saved {len(plot_records)} layer profiles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    build(parser.parse_args().runs.resolve())
