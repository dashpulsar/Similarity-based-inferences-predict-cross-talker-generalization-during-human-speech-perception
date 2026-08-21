from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import distance_to_similarity
from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


def aggregate_predictors(
    *,
    cells_path: str | Path,
    distances_path: str | Path,
    profile: dict[str, Any],
    output_path: str | Path,
) -> pd.DataFrame:
    cells_path = Path(cells_path).resolve()
    distances_path = Path(distances_path).resolve()
    cells = pd.read_csv(cells_path)
    distances = pd.read_csv(distances_path)
    if cells.empty:
        raise ValueError("cell table is empty")
    if distances.empty:
        raise ValueError("distance table is empty")
    for column in ("cell_id", "cell_status", "source_talker_id", "pair_id"):
        if column not in cells:
            raise ValueError(f"cell table lacks {column}")
    for column in ("pair_id", "feature_key", "raw_distance"):
        if column not in distances:
            raise ValueError(f"distance table lacks {column}")
    if distances.duplicated(["pair_id", "feature_key"]).any():
        raise ValueError("distance pair_id x feature_key rows must be unique")

    aggregation = str(profile["talker_aggregation"])
    if aggregation not in {"mean_distance", "min_distance"}:
        raise ValueError(f"unsupported talker aggregation {aggregation!r}")
    fixed_k = float(profile.get("fixed_k", 1.0))
    layers = sorted(distances["feature_key"].unique())
    distance_lookup = distances.set_index(["feature_key", "pair_id"])["raw_distance"]
    metadata_columns = [
        "dataset_id",
        "condition_id",
        "behavior_item_id",
        "analysis_item_id",
        "response_expected",
        "test_talker_id",
        "exposure_talker_set",
        "exposure_set_id",
        "n_expected_source_talkers",
        "n_participants",
    ]
    records: list[dict[str, Any]] = []

    for cell_id, mapping in cells.groupby("cell_id", sort=True, dropna=False):
        first = mapping.iloc[0]
        base = {column: first.get(column) for column in metadata_columns}
        base["cell_id"] = cell_id
        expected = int(first["n_expected_source_talkers"])
        statuses = set(mapping["cell_status"].astype(str))
        for layer in layers:
            record = {
                **base,
                "feature_key": layer,
                "layer": layer,
                "talker_aggregation": aggregation,
                "similarity_k": fixed_k,
            }
            if expected == 0 or statuses == {"no_exposure"}:
                records.append(
                    {
                        **record,
                        "n_available_source_talkers": 0,
                        "raw_distance": np.nan,
                        "similarity_exp_k": np.nan,
                        "predictor_status": "no_exposure",
                        "predictor_reason": "untrained control has no exposure talker similarity",
                    }
                )
                continue
            unavailable = mapping.loc[~mapping["cell_status"].eq("available")]
            if not unavailable.empty:
                reasons = sorted(set(unavailable["status_reason"].dropna().astype(str)))
                records.append(
                    {
                        **record,
                        "n_available_source_talkers": int(
                            mapping.loc[mapping["cell_status"].eq("available"), "source_talker_id"].nunique()
                        ),
                        "raw_distance": np.nan,
                        "similarity_exp_k": np.nan,
                        "predictor_status": "incomplete_source_mapping",
                        "predictor_reason": " | ".join(reasons),
                    }
                )
                continue

            source_values: list[float] = []
            missing_pairs: list[str] = []
            for source_talker, source_rows in mapping.groupby("source_talker_id", sort=True):
                pair_values: list[float] = []
                for pair_id in source_rows["pair_id"].astype(str):
                    key = (layer, pair_id)
                    if key not in distance_lookup.index:
                        missing_pairs.append(pair_id)
                    else:
                        pair_values.append(float(distance_lookup.loc[key]))
                if pair_values:
                    source_values.append(float(np.mean(pair_values)))
            if missing_pairs or len(source_values) != expected:
                records.append(
                    {
                        **record,
                        "n_available_source_talkers": len(source_values),
                        "raw_distance": np.nan,
                        "similarity_exp_k": np.nan,
                        "predictor_status": "missing_distance",
                        "predictor_reason": f"missing {len(set(missing_pairs))} pair distances",
                    }
                )
                continue
            raw_distance = (
                float(np.mean(source_values))
                if aggregation == "mean_distance"
                else float(np.min(source_values))
            )
            records.append(
                {
                    **record,
                    "n_available_source_talkers": len(source_values),
                    "raw_distance": raw_distance,
                    "similarity_exp_k": distance_to_similarity(raw_distance, fixed_k),
                    "predictor_status": "available",
                    "predictor_reason": None,
                }
            )

    result = pd.DataFrame(records).sort_values(["feature_key", "cell_id"]).reset_index(drop=True)
    destination = Path(output_path)
    atomic_write_csv(destination, result)
    provenance = {
        **runtime_record(),
        "stage": "aggregate_predictors",
        "estimand": "same_content_talker_proxy",
        "profile_id": profile.get("profile_id"),
        "cells_path": str(cells_path),
        "cells_sha256": sha256_file(cells_path),
        "distances_path": str(distances_path),
        "distances_sha256": sha256_file(distances_path),
        "talker_aggregation": aggregation,
        "fixed_k": fixed_k,
        "output_rows": int(len(result)),
        "status_counts": {
            str(key): int(value)
            for key, value in result["predictor_status"].value_counts().items()
        },
    }
    atomic_write_json(str(destination) + ".provenance.json", provenance)
    return result
