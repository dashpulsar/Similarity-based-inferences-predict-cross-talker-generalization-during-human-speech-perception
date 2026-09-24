"""Compare completed Sep17 HVE Tr-24 runs with retained Sep6 all-model fits.

Read-only for model inputs/outputs; writes only the named validation JSON.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from ctg.provenance import atomic_write_json, runtime_record, sha256_file


def compare_columns(old, new, keys, numeric, exact):
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError(f"Duplicate comparison keys: {keys}")
    joined = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True,
                       validate="one_to_one")
    key_match = bool(joined._merge.eq("both").all())
    deltas, near, equality, nonidentical, outside_counts, affected = {}, {}, {}, {}, {}, {}
    outside_any = np.zeros(len(joined), dtype=bool)
    for column in numeric:
        a, b = joined[column + "_old"].to_numpy(float), joined[column + "_new"].to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        deltas[column] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None
        near[column] = bool(np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True))
        different = ~(a == b) & ~(np.isnan(a) & np.isnan(b))
        outside = ~np.isclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)
        nonidentical[column] = int(different.sum())
        outside_counts[column] = int(outside.sum())
        affected[column] = sorted(joined.loc[different, "feature_key"].unique())
        outside_any |= outside
    for column in exact:
        equality[column] = bool(joined[column + "_old"].fillna("<NA>").astype(str).eq(
            joined[column + "_new"].fillna("<NA>").astype(str)).all())
    return dict(compared_rows=len(joined), identical_keys=key_match,
                numeric_max_absolute_difference=deltas, numeric_within_tolerance=near,
                numeric_nonidentical_rows=nonidentical, numeric_outside_tolerance_rows=outside_counts,
                numeric_nonidentical_feature_keys=affected,
                first_20_outside_tolerance_keys=joined.loc[outside_any, keys].head(20).astype(object).where(
                    pd.notna(joined.loc[outside_any, keys].head(20)), None).to_dict("records"),
                exact_column_match=equality,
                passed=key_match and all(near.values()) and all(equality.values()))


def main():
    manifest = PROJECT / "analysis_update_2026-09-17/hve_tt/run_inputs.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))["inputs"]
    records = []
    for entry in entries:
        dataset = entry["stratum"].split("_")[0]
        old_dir = PROJECT / f"artifacts/models/{dataset}-HVE-{entry['variant']}-tr24-20260906"
        new_dir = Path(entry["model_dir"])
        record = dict(run_label=entry["run_label"], stratum=entry["stratum"],
                      old_directory=str(old_dir), new_directory=str(new_dir))
        if not (new_dir / "provenance.json").is_file():
            record.update(status="not_compared_incomplete_new_run", passed=None)
            records.append(record)
            continue
        old_provenance = json.loads((old_dir / "provenance.json").read_text(encoding="utf-8"))
        new_provenance = json.loads((new_dir / "provenance.json").read_text(encoding="utf-8"))
        if old_provenance.get("status") != "complete" or new_provenance.get("status") != "complete":
            record.update(status="not_compared_incomplete_provenance", passed=None)
            records.append(record)
            continue
        if old_provenance["parameters"]["model_set"] != "all" or new_provenance["parameters"]["model_set"] != "all":
            raise ValueError("This comparison requires all four models, not selection-only fits")
        features = set(new_provenance["feature_keys"])
        old_input_path, new_input_path = Path(old_provenance["input_path"]), Path(entry["path"])
        if sha256_file(old_input_path) != old_provenance["input_sha256"]:
            raise ValueError(f"Retained source hash mismatch: {old_input_path}")
        if sha256_file(new_input_path) != new_provenance["input_sha256"]:
            raise ValueError(f"New source hash mismatch: {new_input_path}")
        old_input, new_input = pd.read_csv(old_input_path), pd.read_csv(new_input_path)
        def available(frame):
            return frame.loc[frame.feature_key.isin(features) & frame.predictor_status.eq("available") &
                             np.isfinite(frame.predictor_value)].copy()
        old_input, new_input = available(old_input), available(new_input)
        keys = ["feature_key", "participant_id", "fold", "condition_id", "analysis_item_id", "test_talker_id"]
        checks = {"available_input": compare_columns(old_input, new_input, keys,
                  ["predictor_value"], ["response_correct", "response_incorrect"])}
        sample_ok = checks["available_input"]["passed"]
        files = {}
        for name in ["cv_metrics.csv", "diagnostics.csv"]:
            old_file, new_file = old_dir / name, new_dir / name
            if sha256_file(old_file) != old_provenance["outputs_sha256"][name]:
                raise ValueError(f"Retained output hash mismatch: {old_file}")
            if sha256_file(new_file) != new_provenance["outputs_sha256"][name]:
                raise ValueError(f"New output hash mismatch: {new_file}")
            files[name] = dict(old_sha256=sha256_file(old_file), new_sha256=sha256_file(new_file))
            old, new = pd.read_csv(old_file), pd.read_csv(new_file)
            old = old.loc[old.feature_key.isin(features)]
            keys = ["dataset_id", "feature_key", "scope", "fold", "model_id"]
            if name == "cv_metrics.csv":
                checks[name] = compare_columns(old, new, keys, ["total_log_loss", "mean_log_loss"],
                                                ["n_rows", "total_trials"])
            else:
                checks[name] = compare_columns(old, new, keys, ["log_likelihood", "deviance", "AIC"],
                    ["formula", "random_structure", "selection_reason", "optimizer", "fit_ok", "singular",
                     "convergence", "n_rows", "n_observations"])
        record.update(status="compared" if sample_ok else "sample_or_predictor_mismatch_not_a_replication_claim",
                      passed=all(check["passed"] for check in checks.values()),
                      exact_response_and_formula_identity=(checks["available_input"]["identical_keys"] and
                          all(checks["available_input"]["exact_column_match"].values()) and
                          checks["diagnostics.csv"]["identical_keys"] and
                          all(checks["diagnostics.csv"]["exact_column_match"].values())),
                      numeric_repeatability_within_tolerance=all(
                          all(check["numeric_within_tolerance"].values()) for check in checks.values()),
                      features=sorted(features), participant_count=new_input.participant_id.nunique(),
                      matched_available_rows=len(new_input), checks=checks, output_hashes=files,
                      old_input_sha256=old_provenance["input_sha256"], new_input_sha256=new_provenance["input_sha256"],
                      old_R_script_sha256=old_provenance["r_script_sha256"],
                      new_R_script_sha256=new_provenance["r_script_sha256"])
        records.append(record)
        print(record["run_label"], record["status"], "passed=", record["passed"], flush=True)
    completed = [record for record in records if record["passed"] is not None]
    result = {**runtime_record(), "scope": "Completed HVE Tr-24 runs versus retained September6 all-four-model fits; no model rerun",
              "numeric_tolerance": dict(rtol=1e-10, atol=1e-12),
              "passed_completed_comparisons": all(record["passed"] for record in completed),
              "completed_run_comparisons": len(completed), "not_yet_compared_runs": len(records) - len(completed),
              "B23_rule": "Restrict retained combined output to the new stratum's feature keys and compare available participant/fold/response/predictor rows before a replication claim",
              "training_score_note": "September6 outputs predate train_test_scores.csv; training split predictive loss cannot be compared with those outputs. Held-out loss and training/full fitted-model diagnostics are checked separately.",
              "interpretation": "Exact response/fold/formula identity and numerical-repeatability tolerance are separate checks. Small numerical differences remain reported, without refitting or relaxing the tolerance to force agreement.",
              "script_sha256": sha256_file(Path(__file__)), "runs": records}
    destination = Path(__file__).with_name("hve_repeated_tr24_score_validation.json")
    atomic_write_json(destination, result)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
