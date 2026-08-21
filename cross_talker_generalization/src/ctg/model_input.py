from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DatasetSpec
from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


def _talker_set(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return ", ".join(sorted({part.strip() for part in str(value).split(",") if part.strip()}))


def make_model_input(
    *,
    spec: DatasetSpec,
    predictors_path: str | Path,
    folds_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    predictors_path = Path(predictors_path).resolve()
    folds_path = Path(folds_path).resolve()
    predictors = pd.read_csv(predictors_path)
    # Empty exposure sets round-trip through CSV as NaN. Normalize both sides
    # to the same canonical string so no-exposure/control cells retain one
    # explicit unavailable predictor row per feature instead of failing joins.
    if "exposure_talker_set" in predictors:
        predictors["exposure_talker_set"] = predictors["exposure_talker_set"].map(_talker_set)
    folds = pd.read_csv(folds_path)
    behavior = pd.read_csv(spec.behavior)
    behavior = behavior.loc[behavior["phase"].eq("test")].copy()
    behavior["exposure_talker_set"] = behavior["exposure_talkers"].map(_talker_set)
    behavior = behavior.rename(
        columns={
            "exposure_test_condition_id": "condition_id",
            "item_id": "behavior_item_id",
            "item_talker": "behavior_test_talker_id",
        }
    )
    join_columns = [
        "condition_id",
        "behavior_item_id",
        "response_expected",
        "exposure_talker_set",
    ]
    predictor_columns = join_columns + [
        "feature_key",
        "layer",
        "analysis_item_id",
        "test_talker_id",
        "exposure_set_id",
        "raw_distance",
        "similarity_exp_k",
        "predictor_status",
        "predictor_reason",
        "talker_aggregation",
    ]
    missing = [column for column in predictor_columns if column not in predictors]
    if missing:
        raise ValueError(f"predictor table lacks columns: {missing}")
    duplicate_keys = join_columns + ["feature_key"]
    if predictors.duplicated(duplicate_keys).any():
        examples = predictors.loc[predictors.duplicated(duplicate_keys, keep=False), duplicate_keys].head()
        raise ValueError(f"predictor join keys are not unique:\n{examples}")

    merged = behavior.merge(
        predictors[predictor_columns], on=join_columns, how="left", validate="many_to_many"
    )
    # Each behavioral row must map once per feature key.
    expected_layers = predictors["feature_key"].nunique()
    if len(merged) != len(behavior) * expected_layers:
        raise ValueError(
            f"behavior-predictor merge yielded {len(merged)} rows; expected {len(behavior) * expected_layers}"
        )
    merged = merged.merge(
        folds[["participant_id", "fold"]], on="participant_id", how="left", validate="many_to_one"
    )
    if merged["fold"].isna().any():
        raise ValueError("some participants have no fold")
    if not merged["behavior_test_talker_id"].eq(merged["test_talker_id"]).all():
        bad = merged.loc[
            ~merged["behavior_test_talker_id"].eq(merged["test_talker_id"]),
            ["behavior_item_id", "behavior_test_talker_id", "test_talker_id"],
        ].head()
        raise ValueError(f"test talker mismatch:\n{bad}")
    merged["dataset_id"] = spec.dataset_id
    merged["response_correct"] = merged["response_correct"].astype(int)
    merged["response_incorrect"] = merged["response_incorrect"].astype(int)
    merged["fold"] = merged["fold"].astype(int)
    output_columns = [
        "dataset_id",
        "feature_key",
        "layer",
        "participant_id",
        "fold",
        "condition_id",
        "exposure_set_id",
        "analysis_item_id",
        "behavior_item_id",
        "test_talker_id",
        "response_expected",
        "response_correct",
        "response_incorrect",
        "raw_distance",
        "similarity_exp_k",
        "predictor_status",
        "predictor_reason",
        "talker_aggregation",
    ]
    result = merged[output_columns].sort_values(
        ["feature_key", "participant_id", "behavior_item_id", "response_expected"]
    ).reset_index(drop=True)
    destination = Path(output_path)
    atomic_write_csv(destination, result)
    provenance = {
        **runtime_record(),
        "stage": "model_input",
        "dataset_id": spec.dataset_id,
        "behavior_path": str(spec.behavior),
        "behavior_sha256": sha256_file(spec.behavior),
        "predictors_path": str(predictors_path),
        "predictors_sha256": sha256_file(predictors_path),
        "folds_path": str(folds_path),
        "folds_sha256": sha256_file(folds_path),
        "output_rows": int(len(result)),
        "feature_keys": sorted(result["feature_key"].unique()),
        "predictor_status_counts": {
            str(key): int(value) for key, value in result["predictor_status"].value_counts(dropna=False).items()
        },
    }
    atomic_write_json(str(destination) + ".provenance.json", provenance)
    return result
