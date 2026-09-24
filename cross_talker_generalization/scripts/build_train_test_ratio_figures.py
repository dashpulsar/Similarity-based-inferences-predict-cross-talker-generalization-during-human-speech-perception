"""Plot the approved per-response test/train log-loss ratio from matched scores.

python cross_talker_generalization/scripts/build_train_test_ratio_figures.py \
  --manifest cross_talker_generalization/analysis/diagnostics/train_test/run_inputs.json \
  --output cross_talker_generalization/analysis/diagnostics/si_diagnostics

Repeat --manifest for additional non-overlapping runs. Each JSON has an inputs
list: run_label, optional model_dir (relative to the manifest), path (input
provenance), family (SBI/HVE), variant (base/ft/acoustic), measure, and stratum.
HVE requires an explicit participant stratum; feature_key may be layer::measure.
One source per feature/profile is required. This script never selects predictors.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ctg.provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file

KEYS = ["dataset_id", "feature_key", "fold", "model_id"]
MODELS = ["M_null", "M_condition", "M_predictor", "M_joint"]
MATCHED = ["train_predictor_mean", "train_predictor_sd", "random_structure", "formula",
           "prediction_convention", "likelihood_convention", "predictor_column",
           "predictor_term", "predictor_direction"]
META = ["run_label", "family", "variant", "measure", "stratum", "layer"]
PREDICTION = "fixed_effects_only_re_form_NA"
LIKELIHOOD = "word_response_bernoulli_no_binomial_coefficient"


def exact_three_fold_ci(values):
    """Percentile interval over all 27 paired-fold resamples of fold ratios."""
    values = np.asarray(values, dtype=float)
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("Exactly three finite ratios are required")
    samples = [values[list(indices)].mean() for indices in product(range(3), repeat=3)]
    lo, hi = np.quantile(samples, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def truth(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}


def same(a, b):
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    return bool(a == b)


def paired_ratios(scores):
    """Keep invalid pairs and fit warnings, with reasons; never silently drop them."""
    required = KEYS + ["split", "mean_log_loss", "total_log_loss", "total_trials",
                       "score_status", "fit_ok", "singular", "convergence"] + MATCHED
    absent = sorted(set(required) - set(scores.columns))
    if absent:
        raise ValueError(f"Missing score columns: {absent}")
    if scores.duplicated(KEYS + ["split"]).any():
        raise ValueError("Duplicate split scores for the same fitted model")
    if not set(scores.split) <= {"train", "test"}:
        raise ValueError("Unexpected split label")
    if not set(scores.fold.dropna()) <= {0, 1, 2}:
        raise ValueError("Expected folds 0, 1, and 2")
    if not set(scores.model_id) <= set(MODELS):
        raise ValueError("Unexpected model_id")
    rows = []
    # An absent fold or model stays visible, including wholly missing model fits.
    for (dataset, feature), feature_scores in scores.groupby(["dataset_id", "feature_key"]):
        for fold, model in product(range(3), MODELS):
            group = feature_scores.loc[feature_scores.fold.eq(fold) & feature_scores.model_id.eq(model)]
            row = dict(dataset_id=dataset, feature_key=feature, fold=fold, model_id=model)
            reasons, warnings = [], []
            splits = {split: part.iloc[0] for split, part in group.groupby("split")}
            for split in ["train", "test"]:
                if split not in splits:
                    reasons.append(f"missing_{split}_score")
                    continue
                item = splits[split]
                for field in ["mean_log_loss", "total_log_loss", "total_trials"]:
                    row[f"{split}_{field}"] = item[field]
                if item.score_status != "ok":
                    reasons.append(f"{split}_score_status:{item.score_status}")
                if not truth(item.fit_ok):
                    reasons.append(f"{split}_fit_failed")
                count, total, loss = item.total_trials, item.total_log_loss, item.mean_log_loss
                if not np.isfinite(count) or count <= 0:
                    reasons.append(f"{split}_invalid_response_count")
                if not np.isfinite(loss) or loss < 0:
                    reasons.append(f"{split}_invalid_mean_log_loss")
                if not np.isfinite(total) or total < 0:
                    reasons.append(f"{split}_invalid_total_log_loss")
                if np.isfinite(count) and count > 0 and np.isfinite(total) and np.isfinite(loss):
                    if not np.isclose(loss, total / count, rtol=1e-8, atol=1e-12):
                        reasons.append(f"{split}_response_normalization_mismatch")
                if item.prediction_convention != PREDICTION:
                    reasons.append(f"{split}_unexpected_prediction_convention")
                if item.likelihood_convention != LIKELIHOOD:
                    reasons.append(f"{split}_unexpected_likelihood_convention")
                if truth(item.singular):
                    warnings.append("singular_fit")
                if item.convergence != "ok":
                    warnings.append(f"convergence:{item.convergence}")
            if set(splits) == {"train", "test"}:
                for field in MATCHED:
                    row[field] = splits["train"][field]
                    if not same(splits["train"][field], splits["test"][field]):
                        reasons.append(f"train_test_mismatch:{field}")
                train_loss = splits["train"].mean_log_loss
                if not np.isfinite(train_loss) or train_loss <= 0:
                    reasons.append("nonpositive_or_nonfinite_denominator")
            ratio = np.nan
            if not reasons:
                ratio = splits["test"].mean_log_loss / splits["train"].mean_log_loss
                if not np.isfinite(ratio):
                    reasons.append("nonfinite_ratio")
                    ratio = np.nan
            row.update(ratio=ratio, ratio_status="invalid" if reasons else "ok",
                       invalid_reason="; ".join(sorted(set(reasons))),
                       fit_warning="; ".join(sorted(set(warnings))),
                       has_fit_warning=bool(warnings))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(ratios):
    groups = META + ["dataset_id", "feature_key", "model_id"]
    rows = []
    for keys, group in ratios.groupby(groups, dropna=False):
        valid = group.ratio_status.eq("ok") & np.isfinite(group.ratio)
        all_three = set(group.loc[valid, "fold"]) == {0, 1, 2} and valid.sum() == 3
        mean, low, high = exact_three_fold_ci(group.ratio.to_numpy()) if all_three else (np.nan,) * 3
        rows.append(dict(zip(groups, keys), mean_fold_ratio=mean, ci95_low=low, ci95_high=high,
                         n_valid_folds=int(valid.sum()), n_expected_folds=3,
                         n_warning_folds=int(group.has_fit_warning.sum()),
                         summary_status="ok" if all_three else "incomplete_no_three_fold_summary",
                         invalid_reason="; ".join(sorted(set(group.invalid_reason.dropna()) - {""}))))
    return pd.DataFrame(rows)


def resolve_path(value, parent):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def add_metadata(frame, source):
    frame = frame.copy()
    family = source.get("family", "SBI").upper()
    if family not in {"SBI", "HVE"}:
        raise ValueError(f"Unknown predictor family: {family}")
    if family == "HVE" and not source.get("stratum"):
        raise ValueError("HVE inputs require an explicit participant stratum")
    layers, measures = [], []
    for feature in frame.feature_key:
        layer, sep, encoded = str(feature).partition("::")
        supplied = source.get("measure")
        if supplied and sep and supplied != encoded:
            raise ValueError(f"Manifest measure differs from feature_key: {feature}")
        measure = supplied or encoded or ("similarity" if family == "SBI" else "")
        if not measure:
            raise ValueError("HVE measure is required in the manifest or layer::measure")
        layers.append(layer)
        measures.append(measure)
    run_label = source["run_label"]
    inferred = "acoustic" if run_label.endswith("-acoustic") else "ft" if run_label.endswith("-ft") else "base"
    for key, value in dict(run_label=run_label, family=family, variant=source.get("variant", inferred),
                           stratum=source.get("stratum", "all_participants")).items():
        frame[key] = value
    frame["measure"], frame["layer"] = measures, layers
    return frame


def feature_order(key):
    if key in {"mfcc39", "strf24_legacy", "strf24"}:
        return 0, 0 if key == "mfcc39" else 1
    prefix, _, value = key.partition("_")
    return {"cnn": 1, "projection": 2, "tr": 3}.get(prefix, 2), int(value) if value.isdigit() else 0


def label(key):
    return {"mfcc39": "MFCC39", "strf24_legacy": "STRF24", "strf24": "STRF24",
            "projection": "Projection"}.get(key, key.replace("cnn_", "CNN-").replace("tr_", "Tr-"))


def slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-")


def profile_response_counts(ratios, selected):
    """Check counts within a plotted profile; count agreement is not identity proof."""
    source_keys = selected[["run_label", "feature_key"]].drop_duplicates()
    selected_rows = ratios.loc[ratios.model_id.eq("M_predictor")].merge(
        source_keys, on=["run_label", "feature_key"], validate="many_to_one")
    fields = ["train_total_trials", "test_total_trials"]
    consistent = selected_rows.groupby("fold")[fields].nunique(dropna=False).le(1).all().all()
    return bool(consistent)


def make_plots(ratios, summary, output):
    primary = summary.loc[summary.model_id.eq("M_predictor")]
    records = []
    for (dataset, family, measure, stratum), pool in primary.groupby(["dataset_id", "family", "measure", "stratum"]):
        for variant, neural in pool.loc[~pool.variant.eq("acoustic")].groupby("variant"):
            neural_layers = neural.layer.nunique()
            record = dict(dataset_id=dataset, family=family, measure=measure, stratum=stratum,
                          variant=variant, n_neural_layers=neural_layers,
                          layers="; ".join(sorted(neural.layer.unique(), key=feature_order)))
            if neural_layers < 3:
                records.append(dict(record, plot_status="table_only_insufficient_layer_coverage", figure=""))
                continue
            selected = pd.concat([pool.loc[pool.variant.eq("acoustic")], neural], ignore_index=True)
            if selected.layer.duplicated().any():
                raise ValueError(f"Multiple sources for one plotted layer: {record}")
            counts_match = profile_response_counts(ratios, selected)
            keys = sorted(selected.layer.unique(), key=feature_order)
            selected = selected.set_index("layer").loc[keys]
            baseline_n = sum(feature_order(key)[0] == 0 for key in keys)
            xs = np.arange(len(keys), dtype=float)
            xs[baseline_n:] += .6 if baseline_n else 0
            fig, ax = plt.subplots(figsize=(max(9, len(keys) * .39), 4.7), layout="constrained")
            ax.axhline(1., ls="--", color="#808080", lw=1., zorder=0)
            means = selected.mean_fold_ratio.to_numpy()
            if len(keys) > baseline_n:
                ax.plot(xs[baseline_n:], means[baseline_n:], lw=1., color="#444444", zorder=2)
            for i, (key, item) in enumerate(selected.iterrows()):
                fold_rows = ratios.loc[ratios.run_label.eq(item.run_label) & ratios.feature_key.eq(item.feature_key) &
                                       ratios.model_id.eq("M_predictor")].sort_values("fold")
                for _, fold_row in fold_rows.iterrows():
                    if fold_row.ratio_status == "ok":
                        ax.scatter(xs[i] + (fold_row.fold - 1) * .06, fold_row.ratio,
                                   s=14, color="#aaaaaa", zorder=2)
                if item.summary_status == "ok":
                    color = "#ba6b14" if item.n_warning_folds else "#151515"
                    mean = item.mean_fold_ratio
                    ax.errorbar(xs[i], mean, yerr=[[mean - item.ci95_low], [item.ci95_high - mean]],
                                fmt="o", ms=4, capsize=2, color=color, zorder=3)
                else:
                    ax.text(xs[i], .98, "NA", transform=ax.get_xaxis_transform(), ha="center",
                            va="top", color="#b83333", fontsize=8)
            variant_label = {"base": "base", "ft": "ASR-FT"}.get(variant, variant)
            title = f"{dataset} | HuBERT {variant_label} | {family}"
            if family == "HVE":
                title += f"\n{measure} | {stratum}"
            if not counts_match:
                title += "\nResponse counts differ across features"
            ax.set(title=title, xlabel="Feature space", ylabel="Mean test log loss / mean training log loss")
            ax.set_title(title, pad=36)
            ax.set_xticks(xs, [label(key) for key in keys], rotation=45, ha="right", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            handles = [Line2D([], [], color="#151515", marker="o", linestyle="none", label="Fold mean and 95% CI"),
                       Line2D([], [], color="#aaaaaa", marker="o", linestyle="none", markersize=4, label="Individual folds")]
            if selected.n_warning_folds.gt(0).any():
                handles.append(Line2D([], [], color="#ba6b14", marker="o", linestyle="none", label="Fit warning"))
            ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower center",
                      bbox_to_anchor=(.5, 1.01), ncol=len(handles))
            name = "_".join(map(slug, [dataset, family, variant, measure, stratum, "test_train_ratio"]))
            for ext in ["png", "pdf", "svg"]:
                fig.savefig(output / f"{name}.{ext}", dpi=180, facecolor="white")
            plt.close(fig)
            records.append(dict(record, plot_status="layer_profile", figure=name,
                                n_warning_features=int(selected.n_warning_folds.gt(0).sum()),
                                n_incomplete_features=int(selected.summary_status.ne("ok").sum()),
                                per_fold_response_counts_match=counts_match))
    return pd.DataFrame(records)


def build(manifests, output):
    output = Path(output).resolve()
    all_ratios, inventories, seen_labels, seen_dirs = [], [], set(), set()
    manifest_hashes = []
    for manifest_path in manifests:
        manifest_path = Path(manifest_path).resolve()
        manifest_hashes.append(dict(path=str(manifest_path), sha256=sha256_file(manifest_path)))
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))["inputs"]
        for source in entries:
            name = source["run_label"]
            directory = resolve_path(source.get("model_dir", name), manifest_path.parent)
            if name in seen_labels or directory in seen_dirs:
                raise ValueError(f"Duplicate run label or model directory: {name}")
            seen_labels.add(name)
            seen_dirs.add(directory)
            score_path = directory / "train_test_scores.csv"
            scores = pd.read_csv(score_path)
            ratios = add_metadata(paired_ratios(scores), source)
            all_ratios.append(ratios)
            record = dict(run_label=name, model_dir=str(directory), score_file=str(score_path),
                          score_sha256=sha256_file(score_path), n_score_rows=len(scores),
                          n_ratio_rows=len(ratios), n_features=scores.feature_key.nunique(),
                          n_invalid_ratio_rows=int(ratios.ratio_status.ne("ok").sum()),
                          n_warning_ratio_rows=int(ratios.has_fit_warning.sum()))
            if source.get("path"):
                input_path = resolve_path(source["path"], manifest_path.parent)
                record.update(input_file=str(input_path), input_manifest_sha256=source.get("sha256", ""),
                              input_file_available=input_path.exists())
                if input_path.exists():
                    record["input_sha256"] = sha256_file(input_path)
                    if source.get("sha256") and source["sha256"] != record["input_sha256"]:
                        raise ValueError(f"Input hash differs from manifest: {input_path}")
            inventories.append(record)
    if not all_ratios:
        raise ValueError("No input runs")
    ratios = pd.concat(all_ratios, ignore_index=True)
    duplicate_key = ["dataset_id", "family", "variant", "measure", "stratum", "layer", "model_id", "fold"]
    if ratios.duplicated(duplicate_key).any():
        raise ValueError("Multiple sources cover the same feature/profile; select one explicitly in the manifests")
    summary = summarize(ratios)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    inventory = make_plots(ratios, summary, figures)
    atomic_write_csv(output / "tables/paired_fold_ratios.csv", ratios)
    atomic_write_csv(output / "tables/three_fold_ratio_summary.csv", summary)
    atomic_write_csv(output / "tables/source_inventory.csv", pd.DataFrame(inventories))
    atomic_write_csv(output / "tables/figure_inventory.csv", inventory)
    atomic_write_json(output / "tables/build_record.json", dict(
        **runtime_record(), source_manifests=manifest_hashes,
        source_script_sha256=sha256_file(Path(__file__)), n_runs=len(inventories), n_ratio_rows=len(ratios),
        n_invalid_ratio_rows=int(ratios.ratio_status.ne("ok").sum()),
        n_warning_ratio_rows=int(ratios.has_fit_warning.sum()),
        ratio="(test total negative log likelihood / test word responses) / (train total negative log likelihood / train word responses)",
        approval="Florian's September 15 reply, supplied by Zhengyang: proposed test/train mean log-loss ratio accepted",
        interpretation="Above 1 means worse held-out loss under matched scoring; a descriptive generalization diagnostic, not a significance test",
        uncertainty="Equal-weight mean of paired fold ratios; percentile 95% interval over all 27 resamples of three folds. Descriptive fold variability; overlapping training folds are not independent replications.",
        normalization="Per word response; grouped-binomial row counts are not denominators",
        prediction_convention=PREDICTION, likelihood_convention=LIKELIHOOD,
        caveat="A raw-loss ratio is not algebraically equivalent to a ratio of gains relative to a baseline. No independent tuning split is introduced by this diagnostic.",
        incomplete_policy="Invalid pairs retained with reasons; no three-fold CI/mean unless all three ratios are valid; fewer than three neural layers remain table-only"))
    plotted = int(inventory.plot_status.eq("layer_profile").sum()) if len(inventory) else 0
    print(f"Saved {len(ratios)} paired fold ratios from {len(inventories)} runs; {plotted} layer profiles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.manifest, args.output)
