"""Verify saved SBI score pairs and intervals, without fitting any models."""
from pathlib import Path
from itertools import product
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
TABLES = PROJECT / "analysis/diagnostics/si_diagnostics/tables"
paired = pd.read_csv(TABLES / "paired_fold_ratios.csv")
paired = paired.loc[paired.family.eq("SBI")].copy()
summary = pd.read_csv(TABLES / "three_fold_ratio_summary.csv")
summary = summary.loc[summary.family.eq("SBI")].copy()
inventory = pd.read_csv(TABLES / "source_inventory.csv")
inventory = inventory.loc[inventory.run_label.isin(paired.run_label.unique())]
key = ["dataset_id", "feature_key", "fold", "model_id"]
matched = ["train_predictor_mean", "train_predictor_sd", "random_structure", "formula",
           "prediction_convention", "likelihood_convention", "predictor_column", "predictor_term", "predictor_direction"]
assert len(inventory) == 9 and len(paired) == 1368 and len(summary) == 456
assert paired.ratio_status.eq("ok").all() and not paired.has_fit_warning.astype(bool).any()
checks = []
for row in inventory.itertuples():
    score_path, input_path = Path(row.score_file), Path(row.input_file)
    assert hashlib.sha256(score_path.read_bytes()).hexdigest() == row.score_sha256
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == row.input_sha256
    scores = pd.read_csv(score_path)
    assert not scores.duplicated(key + ["split"]).any()
    tr = scores.loc[scores.split.eq("train")].set_index(key).sort_index()
    te = scores.loc[scores.split.eq("test")].set_index(key).sort_index()
    pd.testing.assert_frame_equal(tr[matched], te[matched])
    assert tr.prediction_convention.eq("fixed_effects_only_re_form_NA").all()
    assert tr.likelihood_convention.eq("word_response_bernoulli_no_binomial_coefficient").all()
    assert tr.predictor_column.eq("raw_distance").all() and tr.predictor_direction.eq(-1).all()
    for split in [tr, te]:
        np.testing.assert_allclose(split.mean_log_loss, split.total_log_loss / split.total_trials, rtol=1e-12)
        assert split.fit_ok.all() and not split.singular.any() and split.convergence.eq("ok").all()
    saved = paired.loc[paired.run_label.eq(row.run_label)].set_index(key).sort_index()
    pd.testing.assert_index_equal(saved.index, tr.index)
    np.testing.assert_allclose(saved.ratio, te.mean_log_loss / tr.mean_log_loss, rtol=1e-12)
    for label, split in [("train", tr), ("test", te)]:
        for field in ["mean_log_loss", "total_log_loss", "total_trials"]:
            np.testing.assert_allclose(saved[f"{label}_{field}"], split[field], rtol=1e-12)
    checks.append(dict(run=row.run_label, score_sha256=row.score_sha256,
                       input_sha256=row.input_sha256, score_pairs=len(saved), status="passed"))

group = ["run_label", "dataset_id", "feature_key", "model_id"]
for identity, data in paired.groupby(group):
    assert set(data.fold) == {0, 1, 2} and len(data) == 3
    ratios = data.sort_values("fold").ratio.to_numpy()
    bootstrap = [np.mean(ratios[list(i)]) for i in product(range(3), repeat=3)]
    low, high = np.quantile(bootstrap, [.025, .975])
    mask = np.ones(len(summary), dtype=bool)
    for column, value in zip(group, identity):
        mask &= summary[column].eq(value).to_numpy()
    record = summary.loc[mask]
    assert len(record) == 1
    np.testing.assert_allclose(record[["mean_fold_ratio", "ci95_low", "ci95_high"]].iloc[0],
                               [ratios.mean(), low, high], rtol=1e-12)

predictor = summary.loc[summary.model_id.eq("M_predictor")]
neural = predictor.loc[predictor.variant.isin(["base", "ft"])]
ranges = []
for dataset, rows in neural.groupby("dataset_id"):
    fold_rows = paired.loc[paired.dataset_id.eq(dataset) & paired.variant.isin(["base", "ft"]) & paired.model_id.eq("M_predictor")]
    ranges.append(dict(dataset=dataset, n_layers_variants=len(rows), mean_ratio_min=rows.mean_fold_ratio.min(),
                       mean_ratio_max=rows.mean_fold_ratio.max(), single_fold_min=fold_rows.ratio.min(),
                       single_fold_max=fold_rows.ratio.max()))
fold_accuracy = []
for dataset in ["AN19", "X21", "B23"]:
    source = inventory.loc[inventory.run_label.eq(dataset + "-acoustic")].iloc[0]
    frame = pd.read_csv(source.input_file)
    frame = frame.loc[frame.feature_key.eq("mfcc39") & frame.predictor_status.eq("available")]
    for fold, rows in frame.groupby("fold"):
        total = int((rows.response_correct + rows.response_incorrect).sum())
        fold_accuracy.append(dict(dataset=dataset, fold=int(fold), word_responses=total,
                                  participants=int(rows.participant_id.nunique()),
                                  accuracy=float(rows.response_correct.sum()/total)))
x21 = paired.loc[paired.dataset_id.eq("X21") & paired.variant.eq("base") & paired.layer.eq("tr_24"),
                 ["fold", "model_id", "train_mean_log_loss", "test_mean_log_loss", "ratio"]]
result = dict(status="passed", model_fits_run=0, source_runs=checks, all_model_pairs=len(paired),
              all_model_summaries=len(summary), plotted_unique_specifications=len(predictor),
              plotted_fold_ratios=len(paired.loc[paired.model_id.eq("M_predictor")]),
              predictor_intervals_containing_one=int(((predictor.ci95_low <= 1) & (predictor.ci95_high >= 1)).sum()),
              neural_ranges=ranges, fold_accuracy=fold_accuracy,
              x21_base_tr24_model_comparison=x21.to_dict(orient="records"),
              limitation="This checks saved score arithmetic, pairing, metadata and hashes, plus source-code review. It does not rerun model fitting or regenerate probabilities.")
(OUT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
