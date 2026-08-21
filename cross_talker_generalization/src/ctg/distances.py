from __future__ import annotations

from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import FeatureStoreSpec
from .features import FeatureStore, apply_standardizer, slice_by_time
from .metrics import distance_to_similarity, dtw_distance
from .provenance import (
    atomic_write_csv,
    atomic_write_json,
    runtime_record,
    sha256_file,
)
from .standardize import load_standardizer


def _number_or_none(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _compute_feature_key(
    pairs: pd.DataFrame,
    spec: FeatureStoreSpec,
    feature_key: str,
    metric: str,
    tau: float,
    normalization: str,
    scaling: str,
    standardizer_dir: str | None,
) -> pd.DataFrame:
    if scaling == "global_z":
        if standardizer_dir is None:
            raise ValueError(
                f"{spec.store_id}/{feature_key} requires a corpus-wide standardizer"
            )
        mean, scale = load_standardizer(standardizer_dir, spec.store_id, feature_key)
    elif scaling == "none":
        mean = scale = None
    else:
        raise ValueError(f"unsupported coordinate scaling {scaling!r}")

    output: list[dict[str, Any]] = []
    with threadpool_limits(limits=1), FeatureStore(spec) as store:
        for row in pairs.itertuples(index=False):
            test = store.read(row.test_speaker_id, row.test_unit_id, feature_key)
            source = store.read(row.source_speaker_id, row.source_unit_id, feature_key)
            test = slice_by_time(
                test,
                _number_or_none(row.test_start_seconds),
                _number_or_none(row.test_end_seconds),
                _number_or_none(row.test_duration_seconds),
            )
            source = slice_by_time(
                source,
                _number_or_none(row.source_start_seconds),
                _number_or_none(row.source_end_seconds),
                _number_or_none(row.source_duration_seconds),
            )
            test = apply_standardizer(test, mean, scale)
            source = apply_standardizer(source, mean, scale)
            result = dtw_distance(
                test,
                source,
                tau=tau,
                metric=metric,
                normalization=normalization,
            )
            output.append(
                {
                    "pair_id": row.pair_id,
                    "dataset_id": row.dataset_id,
                    "store_id": spec.store_id,
                    "model_variant": spec.variant,
                    "representation": spec.kind,
                    "feature_key": feature_key,
                    "layer": feature_key,
                    "distance_metric": metric,
                    "tau": tau,
                    "dtw_normalization": normalization,
                    "coordinate_scaling": scaling,
                    "raw_distance": result.distance,
                    "dtw_raw_cost": result.raw_cost,
                    "dtw_path_length": result.path_length,
                    "test_frames": result.left_frames,
                    "source_frames": result.right_frames,
                    "similarity_exp_k1": distance_to_similarity(result.distance),
                }
            )
    return pd.DataFrame(output)


def compute_distances(
    *,
    pairs_path: str | Path,
    spec: FeatureStoreSpec,
    feature_keys: list[str],
    profile: dict[str, Any],
    output_path: str | Path,
    jobs: int,
    standardizer_dir: str | None = None,
) -> pd.DataFrame:
    pairs_path = Path(pairs_path).resolve()
    pairs = pd.read_csv(pairs_path)
    required = {
        "pair_id",
        "dataset_id",
        "test_speaker_id",
        "test_unit_id",
        "source_speaker_id",
        "source_unit_id",
    }
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"pair table lacks columns: {sorted(missing)}")
    if pairs["pair_id"].duplicated().any():
        raise ValueError("pair table must contain unique pair_id values")
    datasets = set(pairs["dataset_id"].dropna().astype(str))
    if datasets != {spec.dataset}:
        raise ValueError(f"pair dataset {datasets} does not match store dataset {spec.dataset}")

    metric = str(profile["distance_metric"])
    tau = float(profile["tau"])
    normalization = str(profile["dtw_normalization"])
    scaling_map = profile.get("coordinate_scaling", {})
    scaling = str(scaling_map.get(spec.kind, "none"))
    unknown = set(feature_keys).difference(_feature_keys(spec))
    if unknown:
        raise ValueError(f"feature keys absent from store: {sorted(unknown)}")

    frames = Parallel(n_jobs=min(max(1, jobs), len(feature_keys)), backend="loky", verbose=5)(
        delayed(_compute_feature_key)(
            pairs,
            spec,
            feature_key,
            metric,
            tau,
            normalization,
            scaling,
            standardizer_dir,
        )
        for feature_key in feature_keys
    )
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["feature_key", "pair_id"]).reset_index(drop=True)
    destination = Path(output_path)
    atomic_write_csv(destination, result)
    file_stat = spec.path.stat()
    provenance = {
        **runtime_record(),
        "stage": "compute_distances",
        "profile_id": profile.get("profile_id"),
        "profile_path": profile.get("_source_path"),
        "store_id": spec.store_id,
        "store_path": str(spec.path),
        "store_quick_fingerprint": {
            "size": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
        },
        "pairs_path": str(pairs_path),
        "pairs_sha256": sha256_file(pairs_path),
        "feature_keys": feature_keys,
        "pair_count": int(len(pairs)),
        "output_rows": int(len(result)),
        "distance_metric": metric,
        "tau": tau,
        "dtw_normalization": normalization,
        "coordinate_scaling": scaling,
        "jobs": int(jobs),
    }
    atomic_write_json(str(destination) + ".provenance.json", provenance)
    return result


def _feature_keys(spec: FeatureStoreSpec) -> set[str]:
    with FeatureStore(spec) as store:
        return set(store.feature_keys())
