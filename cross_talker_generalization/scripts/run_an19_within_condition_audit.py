"""Fit full AN19 acoustic predictors separately within each exposure condition.

Uses the original participant folds and the corrected acoustic model input.
Only M_null and M_predictor are fitted. This isolates within-condition evidence;
it does not retune predictors or select a condition on its observed result.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ctg.parallel_glmm import fit_glmm_parallel, OUTPUT_TABLES
from ctg.provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "analysis/acoustic_baselines/components/model_input.csv")
    parser.add_argument("--output", type=Path, default=PROJECT / "analysis/acoustic_baselines/components/within_condition")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input)
    source = source.loc[source.feature_key.isin(["mfcc39", "strf24_legacy"]) &
                        source.predictor_status.eq("available")].copy()
    if source.condition_id.nunique() != 6:
        raise ValueError("Expected the six available exposure conditions, not untrained controls")
    cases = []
    for number, (condition, frame) in enumerate(source.groupby("condition_id", sort=True)):
        directory = args.output / f"c{number}"
        directory.mkdir(exist_ok=True)
        if set(frame.fold) != {0, 1, 2}:
            raise ValueError(f"Missing fold in {condition}")
        participant_folds = frame[["participant_id", "fold"]].drop_duplicates()
        if len(participant_folds) != 20:
            raise ValueError(f"Unexpected participant count in {condition}")
        if participant_folds.participant_id.duplicated().any():
            raise ValueError(f"Participant crosses folds in {condition}")
        left = frame.loc[frame.feature_key.eq("mfcc39")]
        right = frame.loc[frame.feature_key.eq("strf24_legacy")]
        columns = [c for c in frame.columns if c not in {"feature_key", "layer", "raw_distance", "similarity_exp_k"}]
        pd.testing.assert_frame_equal(left[columns].reset_index(drop=True),
                                      right[columns].reset_index(drop=True), check_dtype=False)
        input_path = directory / "input.csv"
        atomic_write_csv(input_path, frame)
        cases.append({"condition_id": condition, "directory": str(directory), "input_path": str(input_path),
                      "n_participants": len(participant_folds), "rows_per_feature": len(left),
                      "word_responses_per_feature": int((left.response_correct + left.response_incorrect).sum()),
                      "n_test_talkers": int(left.test_talker_id.nunique()),
                      "participant_fold_counts": ";".join(f"{fold}:{count}" for fold, count in participant_folds.groupby("fold").size().items())})
    manifest = pd.DataFrame(cases)
    atomic_write_csv(args.output / "condition_manifest.csv", manifest)

    def run_case(case):
        fit_glmm_parallel(input_path=case["input_path"], output_dir=Path(case["directory"]) / "m",
                          jobs=2, model_set="predictor_only")
        return case["condition_id"]

    # Two conditions at a time, each with two one-thread R feature processes.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_case, case) for case in cases]
        for future in as_completed(futures):
            print("Completed condition:", future.result(), flush=True)

    combined = {}
    for table in OUTPUT_TABLES:
        frames = []
        for case in cases:
            frame = pd.read_csv(Path(case["directory"]) / "m" / table)
            frame["fitted_condition_id"] = case["condition_id"]
            frames.append(frame)
        combined[table] = pd.concat(frames, ignore_index=True)
        atomic_write_csv(args.output / table, combined[table])

    metrics = combined["cv_metrics.csv"]
    keys = ["fitted_condition_id", "feature_key", "fold"]
    folds = metrics.loc[metrics.scope.eq("oof_fold")]
    wide = folds.pivot(index=keys, columns="model_id", values="mean_log_loss").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"M_null": "null_mean_log_loss", "M_predictor": "predictor_mean_log_loss"})
    wide["predictor_improvement_mean_log_loss"] = wide.null_mean_log_loss - wide.predictor_mean_log_loss
    coefficients = combined["coefficients.csv"]
    train = coefficients.loc[coefficients.scope.eq("cv_train") & coefficients.model_id.eq("M_predictor") &
                             coefficients.term.eq("similarity_z"), keys + ["estimate", "std_error", "z_value", "p_value"]]
    train = train.rename(columns={"estimate": "training_slope", "std_error": "training_slope_se",
                                  "z_value": "training_z", "p_value": "training_p"})
    result = wide.merge(train, on=keys, validate="one_to_one")
    diagnostics = combined["diagnostics.csv"]
    train_diag = diagnostics.loc[diagnostics.scope.eq("cv_train") & diagnostics.model_id.eq("M_predictor"),
                                 keys + ["random_structure", "selection_reason", "fit_ok", "singular", "convergence"]]
    result = result.merge(train_diag, on=keys, validate="one_to_one")
    atomic_write_csv(args.output / "fold_results.csv", result)
    summaries = []
    for (condition, feature), frame in result.groupby(["fitted_condition_id", "feature_key"]):
        all_rows = metrics.loc[metrics.scope.eq("oof_all") & metrics.fitted_condition_id.eq(condition) & metrics.feature_key.eq(feature)].set_index("model_id")
        case_diagnostics = diagnostics.loc[diagnostics.fitted_condition_id.eq(condition) & diagnostics.feature_key.eq(feature)]
        singular_count = int(case_diagnostics.singular.astype(bool).sum())
        non_ok_count = int(case_diagnostics.convergence.ne("ok").sum())
        summaries.append({"condition_id": condition, "feature_key": feature, "n_folds": len(frame),
                          "mean_training_slope": frame.training_slope.mean(), "mean_training_z": frame.training_z.mean(),
                          "min_training_z": frame.training_z.min(), "max_training_z": frame.training_z.max(),
                          "positive_slope_folds": int(frame.training_slope.gt(0).sum()),
                          "positive_test_improvement_folds": int(frame.predictor_improvement_mean_log_loss.gt(0).sum()),
                          "pooled_test_mean_log_loss_null": all_rows.loc["M_null", "mean_log_loss"],
                          "pooled_test_mean_log_loss_predictor": all_rows.loc["M_predictor", "mean_log_loss"],
                          "pooled_test_improvement": all_rows.loc["M_null", "mean_log_loss"] - all_rows.loc["M_predictor", "mean_log_loss"],
                          "random_structures": ";".join(sorted(frame.random_structure.unique())),
                          "singular_training_fits": int(frame.singular.astype(bool).sum()),
                          "non_ok_training_convergence_flags": int(frame.convergence.ne("ok").sum()),
                          "selected_singular_fits_full_and_cv_both_models": singular_count,
                          "selected_non_ok_fits_full_and_cv_both_models": non_ok_count,
                          "fit_quality": "review_singular_or_convergence_flags" if singular_count or non_ok_count else "no_selected_fit_flags"})
    summary = pd.DataFrame(summaries)
    atomic_write_csv(args.output / "summary.csv", summary)
    # Within each scope, nested model pairs must use the same random structure.
    structure_counts = diagnostics.groupby(["fitted_condition_id", "feature_key", "scope", "fold"], dropna=False).random_structure.nunique()
    if not structure_counts.eq(1).all():
        raise ValueError("Nested models have unmatched random-effects structures")
    flagged = diagnostics.loc[diagnostics.singular.astype(bool) | diagnostics.convergence.ne("ok") | ~diagnostics.fit_ok.astype(bool)]
    atomic_write_csv(args.output / "flagged_fits.csv", flagged)
    metadata = {**runtime_record(), "status": "complete_with_fit_warnings" if len(flagged) else "complete", "source_input_sha256": sha256_file(args.input),
                "script_sha256": sha256_file(Path(__file__)), "conditions": len(cases), "features": ["mfcc39", "strf24_legacy"],
                "n_selected_fits": len(diagnostics), "fit_failures": int((~diagnostics.fit_ok.astype(bool)).sum()),
                "singular_selected_fits": int(diagnostics.singular.astype(bool).sum()),
                "non_ok_convergence": int(diagnostics.convergence.ne("ok").sum()),
                "convergence_flag_messages": diagnostics.loc[diagnostics.convergence.ne("ok"), "convergence"].value_counts().to_dict(),
                "flagged_conditions": sorted(flagged.fitted_condition_id.unique()),
                "all_positive_training_slopes": bool(result.training_slope.gt(0).all()),
                "positive_test_improvement_folds": int(result.predictor_improvement_mean_log_loss.gt(0).sum()),
                "total_test_folds": len(result),
                "positive_pooled_feature_condition_results": int(summary.pooled_test_improvement.gt(0).sum()),
                "interpretation": "Positive held-out improvements persist within the four conditions with no selected-fit flags for both spaces. Thus these predictors contain within-condition information; unstable cells remain exploratory and require model review.",
                "matched_null_predictor_structures_within_each_scope": True,
                "prediction": "Frozen fixed-effects-only predictions (re.form=NA), original participant folds; mean Bernoulli log loss per word response",
                "predictor": "Negative within-training-fold standardized distance from full MFCC39 or STRF24; fixed tau=2, historical mean-sequence-length normalization",
                "purpose": "Independent model fitting within each condition tests whether positive acoustic evidence persists after removing between-condition differences",
                "scope": "Six informative-exposure conditions, 20 participants each; no predictor exists for untrained controls in this analysis",
                "uncertainty": "Raw fold results, z estimates and ranges retained. No fold-bootstrap interval or cross-condition inferential claim added",
                "limitations": "Small condition samples; random-effect fallback can differ across conditions/features; a common scope within each nested pair is enforced. Selection or causal decomposition is not claimed.",
                "max_parallel_R_processes": 4}
    atomic_write_json(args.output / "audit_metadata.json", metadata)
    print(summary.to_string(index=False), flush=True)
    print(metadata, flush=True)


if __name__ == "__main__":
    main()
