"""AN19 sensitivity: apply one declared participant/item structure to every feature.

The structure is the existing AN19 fallback, fixed before this run; no tuning or
random-effect deletion uses test responses. Primary registered outputs stay intact.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ctg.parallel_glmm import OUTPUT_TABLES, fit_glmm_parallel
from ctg.provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


def _bool(series):
    return series.fillna(False).map(lambda value: str(value).lower() == "true")


def compare_case(case):
    keys = ["feature_key", "scope", "fold", "model_id"]
    tables = []
    for policy, directory in (("registered", Path(case["original_directory"])),
                              ("participant_item", Path(case["directory"]))):
        diagnostics = pd.read_csv(directory / "diagnostics.csv")
        coefficients = pd.read_csv(directory / "coefficients.csv")
        predictor = coefficients.loc[coefficients.term.eq("similarity_z"),
                                     keys + ["estimate", "std_error", "z_value", "p_value"]]
        table = diagnostics.merge(predictor, on=keys, how="left", validate="one_to_one")
        boundary_message = table.convergence.fillna("").str.fullmatch(
            r"boundary \(singular\) fit: see help\('isSingular'\)"
        )
        table["nonboundary_convergence_message"] = table.convergence.ne("ok") & ~boundary_message
        table["boundary_only_diagnostic"] = _bool(table.fit_ok) & _bool(table.singular) & ~table.nonboundary_convergence_message
        metrics = pd.read_csv(directory / "cv_metrics.csv")
        metrics = metrics.loc[metrics.scope.eq("oof_fold"),
                              ["feature_key", "fold", "model_id", "total_trials", "mean_log_loss"]]
        metrics["scope"] = "cv_train"
        table = table.merge(metrics, on=keys, how="left", validate="one_to_one")
        variance_file = directory / "variance_components.csv"
        table["variance_components_available"] = variance_file.is_file()
        if variance_file.is_file():
            variance = pd.read_csv(variance_file)
            for group in ("participant_id", "analysis_item_id", "test_talker_id"):
                component = variance.loc[variance.group.eq(group), keys + ["variance"]]
                component = component.rename(columns={"variance": f"variance_{group}"})
                table = table.merge(component, on=keys, how="left", validate="one_to_one")
        for group in ("participant_id", "analysis_item_id", "test_talker_id"):
            column = f"variance_{group}"
            if column not in table:
                table[column] = np.nan
        table["policy"] = policy
        table["case_id"] = case["case_id"]
        table["condition_id"] = case["condition_id"]
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def report(cases, output):
    long = pd.concat([compare_case(case) for case in cases], ignore_index=True)
    atomic_write_csv(output / "comparison_all_fits.csv", long)
    rows = []
    for (case, feature, policy), frame in long.groupby(["case_id", "feature_key", "policy"]):
        train = frame.loc[frame.scope.eq("cv_train") & frame.model_id.eq("M_predictor")]
        null = frame.loc[frame.scope.eq("cv_train") & frame.model_id.eq("M_null")]
        prediction_loss = np.average(train.mean_log_loss, weights=train.total_trials)
        null_loss = np.average(null.mean_log_loss, weights=null.total_trials)
        paired_loss = train[["fold", "mean_log_loss"]].merge(
            null[["fold", "mean_log_loss"]], on="fold", suffixes=("_predictor", "_null"), validate="one_to_one"
        )
        selected = frame.loc[frame.model_id.eq("M_predictor")]
        boundary = selected[["variance_participant_id", "variance_analysis_item_id"]].lt(1e-8).any(axis=1)
        rows.append(dict(case_id=case, condition_id=frame.condition_id.iloc[0], feature_key=feature,
                         policy=policy, mean_training_slope=train.estimate.mean(),
                         mean_training_z=train.z_value.mean(), pooled_test_mean_log_loss=prediction_loss,
                         pooled_test_mean_log_loss_null=null_loss,
                         pooled_test_improvement_mean_log_loss=null_loss - prediction_loss,
                         positive_test_improvement_folds=int(paired_loss.mean_log_loss_null.gt(paired_loss.mean_log_loss_predictor).sum()),
                         positive_training_slope_folds=int(train.estimate.gt(0).sum()),
                         fitted_structures=";".join(sorted(frame.random_structure.unique())),
                         failed_fits=int((~_bool(frame.fit_ok)).sum()),
                         singular_fits=int(_bool(frame.singular).sum()),
                         convergence_message_fits=int(frame.convergence.ne("ok").sum()),
                         nonboundary_convergence_message_fits=int(frame.nonboundary_convergence_message.sum()),
                         boundary_only_fits=int(frame.boundary_only_diagnostic.sum()),
                         predictor_boundary_fits=int(boundary.sum()) if frame.variance_components_available.all() else np.nan,
                         variance_components_available=bool(frame.variance_components_available.all())))
    summary = pd.DataFrame(rows)
    atomic_write_csv(output / "comparison_summary.csv", summary)
    index = ["case_id", "condition_id", "feature_key"]
    values = ["mean_training_slope", "mean_training_z", "pooled_test_mean_log_loss", "singular_fits",
              "convergence_message_fits", "predictor_boundary_fits"]
    comparison = summary.pivot(index=index, columns="policy", values=values)
    comparison.columns = [f"{metric}_{policy}" for metric, policy in comparison.columns]
    for metric in ("mean_training_slope", "mean_training_z", "pooled_test_mean_log_loss"):
        comparison[f"delta_{metric}"] = comparison[f"{metric}_participant_item"] - comparison[f"{metric}_registered"]
    atomic_write_csv(output / "registered_vs_common.csv", comparison.reset_index())
    variances = []
    for case in cases:
        frame = pd.read_csv(Path(case["directory"]) / "variance_components.csv")
        frame["case_id"] = case["case_id"]
        frame["condition_id"] = case["condition_id"]
        variances.append(frame)
    variance = pd.concat(variances, ignore_index=True)
    atomic_write_csv(output / "variance_components.csv", variance)
    flagged = variance.loc[_bool(variance.singular) | variance.convergence.ne("ok") |
                           _bool(variance.variance_below_1e_8)]
    atomic_write_csv(output / "flagged_variance_components.csv", flagged)
    fixed = long.loc[long.policy.eq("participant_item")]
    if not fixed.random_structure.eq("participant_item_intercepts").all():
        raise ValueError("Declared structure was not preserved")
    if any(sha256_file(Path(case["input_path"])) != case["input_sha256"] for case in cases):
        raise ValueError("An original input changed during the sensitivity run")
    atomic_write_json(output / "result_metadata.json", {
        **runtime_record(), "status": "complete_with_fit_warnings" if len(flagged) else "complete",
        "case_count": len(cases), "common_policy_fit_count": len(fixed),
        "common_policy_fit_failures": int((~_bool(fixed.fit_ok)).sum()),
        "common_policy_singular_fits": int(_bool(fixed.singular).sum()),
        "common_policy_convergence_message_fits": int(fixed.convergence.ne("ok").sum()),
        "common_policy_nonboundary_convergence_message_fits": int(fixed.nonboundary_convergence_message.sum()),
        "common_policy_boundary_only_diagnostic_fits": int(fixed.boundary_only_diagnostic.sum()),
        "policy": "participant_item", "structure": "(1 | participant_id) + (1 | analysis_item_id)",
        "structure_rationale": "Existing AN19 registered fallback fixed across features and folds before this sensitivity run; not selected using test performance",
        "preserved": "Original predictors, response rows, participant folds, predictor direction, frozen train-fold scaling and fixed-effect-only scoring",
        "scope": "Sensitivity analysis; primary registered outputs not replaced",
        "variance_boundary_note": "Variance below 1e-8 is an explicit descriptive flag. isSingular(tol=1e-4), optimizer convergence messages, and fit failures are retained separately. Zero random-intercept variance is not automatically an optimizer failure.",
        "original_variance_note": "Previous primary files did not save variance components; missing old variances remain missing and are not reconstructed from z-values",
        "documentation": ["https://lme4.github.io/lme4/reference/isSingular.html", "https://lme4.github.io/lme4/reference/VarCorr.html"],
        "script_sha256": sha256_file(Path(__file__)),
        "r_script_sha256": sha256_file(PROJECT / "R/fit_confirmatory.R"),
    })
    print(comparison.reset_index().to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "analysis/acoustic_baselines/components/model_input.csv")
    parser.add_argument("--output", type=Path, default=PROJECT / "analysis/acoustic_baselines/common_structure")
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input)
    if source.feature_key.nunique() != 16 or set(source.dataset_id) != {"AN19"}:
        raise ValueError("Expected all sixteen AN19 acoustic features")
    original = args.input.parent
    cases = [dict(case_id="all_conditions", condition_id="all", directory=str(args.output / "all"),
                  original_directory=str(original / "models"), input_path=str(args.input), model_set="all")]
    conditions = pd.read_csv(original / "within_condition/condition_manifest.csv")
    if len(conditions) != 6:
        raise ValueError("Expected six original condition inputs")
    for number, row in enumerate(conditions.itertuples(index=False)):
        cases.append(dict(case_id=f"c{number}", condition_id=row.condition_id,
                          directory=str(args.output / f"c{number}"),
                          original_directory=str(Path(row.directory) / "m"),
                          input_path=str(row.input_path), model_set="predictor_only"))
    for case in cases:
        case["input_sha256"] = sha256_file(Path(case["input_path"]))
    atomic_write_csv(args.output / "case_manifest.csv", pd.DataFrame(cases))
    atomic_write_json(args.output / ("report_record.json" if args.report_only else "run_record.json"), {
        **runtime_record(), "status": "report_only" if args.report_only else "started",
        "command_arguments": sys.argv, "random_policy": "participant_item", "max_R_workers": args.jobs,
        "source_script_sha256": sha256_file(PROJECT / "R/fit_confirmatory.R"),
        "runner_sha256": sha256_file(Path(__file__)), "source_input_sha256": sha256_file(args.input),
        "cases": cases,
    })
    def run(case, jobs):
        fit_glmm_parallel(input_path=case["input_path"], output_dir=case["directory"], jobs=jobs,
                          model_set=case["model_set"], random_policy="participant_item")
        print("Finished sensitivity", case["case_id"], flush=True)
    if not args.report_only:
        run(cases[0], args.jobs)
        workers = max(1, args.jobs // 2)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run, case, min(2, args.jobs)) for case in cases[1:]]
            for future in futures:
                future.result()
    report(cases, args.output)


if __name__ == "__main__":
    main()
