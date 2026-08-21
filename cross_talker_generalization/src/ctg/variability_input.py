from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DatasetSpec
from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


def make_variability_model_input(
    *,
    spec: DatasetSpec,
    variability_path: str | Path,
    participant_pools_path: str | Path,
    folds_path: str | Path,
    output_path: str | Path,
    measures: list[str] | None = None,
) -> pd.DataFrame:
    variability_path = Path(variability_path).resolve()
    participant_pools_path = Path(participant_pools_path).resolve()
    folds_path = Path(folds_path).resolve()
    values = pd.read_csv(variability_path)
    if measures is not None:
        unknown = set(measures).difference(values["measure"].unique())
        if unknown:
            raise ValueError(f"unknown variability measures: {sorted(unknown)}")
        values = values.loc[values["measure"].isin(measures)].copy()
    participants = pd.read_csv(participant_pools_path)
    folds = pd.read_csv(folds_path)
    behavior = pd.read_csv(spec.behavior)
    behavior = behavior.loc[behavior["phase"].eq("test")].copy()
    behavior = behavior.rename(
        columns={
            "exposure_test_condition_id": "condition_id",
            "item_id": "behavior_item_id",
            "item_talker": "test_talker_id",
        }
    )
    participant_design = participants[
        ["participant_id", "condition_id", "pool_id", "pool_status", "pool_reason"]
    ]
    behavior = behavior.merge(
        participant_design,
        on=["participant_id", "condition_id"],
        how="left",
        validate="many_to_one",
    )
    if behavior["pool_id"].isna().any():
        raise ValueError("some behavioral participants have no exposure-pool mapping")

    value_columns = [
        "pool_id",
        "feature_key",
        "layer",
        "measure",
        "value",
        "value_status",
        "status_reason",
        "tau",
        "coordinate_scaling",
    ]
    missing = [column for column in value_columns if column not in values]
    if missing:
        raise ValueError(f"variability table lacks columns: {missing}")
    if values.duplicated(["pool_id", "feature_key", "measure"]).any():
        raise ValueError("variability pool x feature x measure rows must be unique")
    merged = behavior.merge(values[value_columns], on="pool_id", how="left", validate="many_to_many")
    n_groups = values[["feature_key", "measure"]].drop_duplicates().shape[0]
    if len(merged) != len(behavior) * n_groups:
        raise ValueError(
            f"behavior-variability merge yielded {len(merged)} rows; expected {len(behavior) * n_groups}"
        )
    merged = merged.merge(
        folds[["participant_id", "fold"]], on="participant_id", how="left", validate="many_to_one"
    )
    if merged["fold"].isna().any():
        raise ValueError("some participants have no fold")
    merged["registered_layer"] = merged["feature_key"]
    merged["feature_key"] = merged["registered_layer"] + "::" + merged["measure"]
    merged["analysis_item_id"] = (
        merged["behavior_item_id"].astype(str) + "::" + merged["response_expected"].astype(str)
    )
    merged["predictor_value"] = merged["value"]
    merged["predictor_status"] = merged["value_status"]
    merged["predictor_reason"] = merged["status_reason"].fillna(merged["pool_reason"])
    merged["dataset_id"] = spec.dataset_id
    merged["fold"] = merged["fold"].astype(int)
    columns = [
        "dataset_id",
        "feature_key",
        "registered_layer",
        "measure",
        "participant_id",
        "fold",
        "condition_id",
        "pool_id",
        "analysis_item_id",
        "behavior_item_id",
        "test_talker_id",
        "response_expected",
        "response_correct",
        "response_incorrect",
        "predictor_value",
        "predictor_status",
        "predictor_reason",
        "tau",
        "coordinate_scaling",
    ]
    result = merged[columns].sort_values(
        ["feature_key", "participant_id", "behavior_item_id", "response_expected"]
    ).reset_index(drop=True)
    destination = Path(output_path)
    atomic_write_csv(destination, result)
    provenance = {
        **runtime_record(),
        "stage": "variability_model_input",
        "dataset_id": spec.dataset_id,
        "behavior_sha256": sha256_file(spec.behavior),
        "variability_sha256": sha256_file(variability_path),
        "participant_pools_sha256": sha256_file(participant_pools_path),
        "folds_sha256": sha256_file(folds_path),
        "output_rows": int(len(result)),
        "analysis_groups": int(result["feature_key"].nunique()),
        "requested_measures": measures,
        "status_counts": {
            str(key): int(value) for key, value in result["predictor_status"].value_counts().items()
        },
    }
    atomic_write_json(str(destination) + ".provenance.json", provenance)
    return result
