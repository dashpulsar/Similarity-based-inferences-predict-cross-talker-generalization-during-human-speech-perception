from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import DatasetSpec, FeatureStoreSpec
from .features import FeatureStore, apply_standardizer, slice_by_time
from .metrics import Token, VARIABILITY_NAMES, compute_variability
from .provenance import (
    atomic_write_csv,
    atomic_write_json,
    runtime_record,
    sha256_file,
    stable_id,
)
from .standardize import load_standardizer
from .textgrid_io import parse_long_textgrid


TASK_COLUMNS = [
    "pool_id",
    "dataset_id",
    "token_id",
    "type_id",
    "linguistic_unit",
    "speaker_id",
    "unit_id",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
]


def _talkers(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return sorted({part.strip() for part in str(value).split(",") if part.strip()})


def _norm(value: object) -> str:
    return str(value).strip().casefold()


def _task(
    pool_id: str,
    dataset_id: str,
    token_id: str,
    type_id: str,
    unit: str,
    speaker: str,
    unit_id: str,
    interval: tuple[float, float, float] | None = None,
):
    start, end, duration = interval or (None, None, None)
    return {
        "pool_id": pool_id,
        "dataset_id": dataset_id,
        "token_id": token_id,
        "type_id": type_id,
        "linguistic_unit": unit,
        "speaker_id": speaker,
        "unit_id": unit_id,
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": duration,
    }


def _participant_design(behavior: pd.DataFrame) -> pd.DataFrame:
    test = behavior.loc[behavior["phase"].eq("test")].copy()
    columns = ["participant_id", "exposure_test_condition_id", "exposure_talkers"]
    result = test[columns].drop_duplicates()
    counts = result.groupby("participant_id").size()
    if not counts.eq(1).all():
        raise ValueError("participants must have one exposure design")
    return result


def _build_an19(spec: DatasetSpec, behavior: pd.DataFrame, manifest: pd.DataFrame):
    exposure = behavior.loc[behavior["phase"].eq("exposure")].copy()
    manifest = manifest.copy()
    manifest["_word_norm"] = manifest["word"].map(_norm)
    lookup = manifest.set_index("recording_id", drop=False)
    participant_recordings: dict[str, list[str]] = {}
    for participant, rows in exposure.groupby("participant_id", sort=True):
        recordings = []
        for row in rows.itertuples(index=False):
            matched = manifest.loc[
                manifest["behavior_item_id"].eq(row.item_id)
                & manifest["_word_norm"].eq(_norm(row.response_expected))
            ]
            if len(matched) != 1:
                raise ValueError(
                    f"AN19 exposure row maps to {len(matched)} recordings: {participant}/{row.item_id}/{row.response_expected}"
                )
            recordings.append(str(matched.iloc[0]["recording_id"]))
        if len(recordings) != 144 or len(set(recordings)) != 144:
            raise ValueError(f"AN19 participant {participant} does not have 144 unique exposure tokens")
        participant_recordings[str(participant)] = sorted(recordings)

    pool_by_signature: dict[tuple[str, ...], str] = {}
    tasks: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    for signature in sorted(set(tuple(values) for values in participant_recordings.values())):
        pool_id = stable_id("AN19V", *signature)
        pool_by_signature[signature] = pool_id
        members = sorted(
            participant for participant, values in participant_recordings.items() if tuple(values) == signature
        )
        pool_rows.append(
            {
                "pool_id": pool_id,
                "dataset_id": "AN19",
                "pool_status": "available",
                "pool_reason": None,
                "estimand": "actual_heard_exposure_variability",
                "n_participants": len(members),
                "n_presentations": len(signature),
            }
        )
        for recording_id in signature:
            row = lookup.loc[recording_id]
            tasks.append(
                _task(
                    pool_id,
                    "AN19",
                    recording_id,
                    str(row["word"]).casefold(),
                    "word",
                    str(row["speaker_id"]),
                    recording_id,
                )
            )

    participant_rows = []
    design = _participant_design(behavior).set_index("participant_id")
    for participant in sorted(behavior["participant_id"].unique()):
        condition = str(design.loc[participant, "exposure_test_condition_id"])
        if participant in participant_recordings:
            pool_id = pool_by_signature[tuple(participant_recordings[participant])]
            status, reason = "available", None
        else:
            pool_id = "AN19V:no_exposure"
            status, reason = "no_exposure", "untrained control"
        participant_rows.append(
            {
                "participant_id": participant,
                "dataset_id": "AN19",
                "condition_id": condition,
                "pool_id": pool_id,
                "pool_status": status,
                "pool_reason": reason,
            }
        )
    pool_rows.append(
        {
            "pool_id": "AN19V:no_exposure",
            "dataset_id": "AN19",
            "pool_status": "no_exposure",
            "pool_reason": "untrained control",
            "estimand": "actual_heard_exposure_variability",
            "n_participants": 40,
            "n_presentations": 0,
        }
    )
    return pd.DataFrame(tasks), pd.DataFrame(pool_rows), pd.DataFrame(participant_rows)


def _interval_tasks_for_segment(
    pool_id: str,
    dataset_id: str,
    presentation_id: str,
    row: pd.Series,
) -> list[dict[str, Any]]:
    duration = float(row["duration_seconds"])
    result = [
        _task(
            pool_id,
            dataset_id,
            f"{presentation_id}:sentence",
            str(row.get("sentence_code", row.get("sentence_normalized", row["segment_id"]))),
            "sentence",
            str(row["speaker_id"]),
            str(row["segment_id"]),
        )
    ]
    for unit, column in (("word", "word_intervals_json"), ("phoneme", "phone_intervals_json")):
        if column not in row or pd.isna(row[column]):
            continue
        for interval in json.loads(row[column]):
            label = str(interval.get("normalized", interval.get("label", ""))).strip()
            if not label or _norm(label) in {"sp", "sil", ""}:
                continue
            index = interval.get("index", len(result))
            result.append(
                _task(
                    pool_id,
                    dataset_id,
                    f"{presentation_id}:{unit}:{index}",
                    label,
                    unit,
                    str(row["speaker_id"]),
                    str(row["segment_id"]),
                    (float(interval["start_seconds"]), float(interval["end_seconds"]), duration),
                )
            )
    return result


def _build_x21(spec: DatasetSpec, behavior: pd.DataFrame, manifest: pd.DataFrame):
    by_segment = manifest.set_index("segment_id", drop=False)
    test = behavior.loc[behavior["phase"].eq("test")].copy()
    participant_rows = []
    pool_definitions: dict[str, dict[str, Any]] = {}

    for participant, rows in test.groupby("participant_id", sort=True):
        conditions = rows["exposure_test_condition_id"].unique()
        exposure_sets = rows["exposure_talkers"].unique()
        if len(conditions) != 1 or len(exposure_sets) != 1:
            raise ValueError(f"X21 participant {participant} has multiple designs")
        segments = sorted({str(item).rsplit(".W", 1)[0] for item in rows["item_id"]})
        sets = {str(by_segment.loc[segment, "experimental_set"]) for segment in segments}
        if len(sets) != 1:
            raise ValueError(f"X21 participant {participant} spans test sets {sets}")
        test_set = next(iter(sets))
        exposure_set = "set2" if test_set == "set1" else "set1"
        talkers = _talkers(exposure_sets[0])
        condition = str(conditions[0])
        multiplicity = 5 if condition in {"X21.Single_talker", "X21.Talker_specific"} else 1
        signature = (exposure_set, tuple(talkers), multiplicity)
        pool_id = stable_id("X21V", exposure_set, ",".join(talkers), multiplicity)
        pool_definitions[pool_id] = {
            "pool_id": pool_id,
            "dataset_id": "X21",
            "pool_status": "available",
            "pool_reason": None,
            "estimand": "actual_heard_presentation_weighted_exposure_variability",
            "exposure_sentence_set": exposure_set,
            "exposure_talker_set": ", ".join(talkers),
            "presentation_multiplicity": multiplicity,
        }
        participant_rows.append(
            {
                "participant_id": participant,
                "dataset_id": "X21",
                "condition_id": condition,
                "pool_id": pool_id,
                "pool_status": "available",
                "pool_reason": None,
            }
        )

    tasks: list[dict[str, Any]] = []
    pool_rows = []
    participant_frame = pd.DataFrame(participant_rows)
    for pool_id, definition in sorted(pool_definitions.items()):
        talkers = _talkers(definition["exposure_talker_set"])
        selected = manifest.loc[
            manifest["speaker_id"].isin(talkers)
            & manifest["experimental_set"].eq(definition["exposure_sentence_set"])
        ].sort_values(["speaker_id", "sentence_code"])
        expected_segments = 16 * len(talkers)
        if len(selected) != expected_segments:
            raise ValueError(f"X21 pool {pool_id} has {len(selected)} segments, expected {expected_segments}")
        multiplicity = int(definition["presentation_multiplicity"])
        for _, segment in selected.iterrows():
            for repetition in range(multiplicity):
                presentation = f"{segment['segment_id']}:presentation{repetition + 1}"
                tasks.extend(_interval_tasks_for_segment(pool_id, "X21", presentation, segment))
        pool_rows.append(
            {
                **definition,
                "n_participants": int(participant_frame["pool_id"].eq(pool_id).sum()),
                "n_presentations": int(len(selected) * multiplicity),
            }
        )
    return pd.DataFrame(tasks), pd.DataFrame(pool_rows), participant_frame


def _textgrid_intervals(row: pd.Series, repo_root: Path):
    path = repo_root / str(row["source_textgrid_relpath"])
    grid = parse_long_textgrid(path)
    start = float(row["start_seconds"])
    end = float(row["end_seconds"])
    duration = float(row["duration_seconds"])
    result: dict[str, list[dict[str, Any]]] = {"word": [], "phoneme": []}
    for unit, tier_name in (("word", "Speaker - word"), ("phoneme", "Speaker - phone")):
        if tier_name not in grid:
            raise ValueError(f"missing B23 tier {tier_name!r} in {path}")
        for interval in grid[tier_name]:
            label = str(interval.text).strip().upper()
            midpoint = (interval.xmin + interval.xmax) / 2.0
            if not label or label in {"SP", "SIL"} or not (start - 1e-9 <= midpoint <= end + 1e-9):
                continue
            if interval.xmin < start - 1e-6 or interval.xmax > end + 1e-6:
                raise ValueError(f"B23 interval {interval.index} extends beyond {row['segment_id']}")
            result[unit].append(
                {
                    "index": interval.index,
                    "label": label,
                    "start": max(0.0, float(interval.xmin) - start),
                    "end": min(duration, float(interval.xmax) - start),
                    "duration": duration,
                }
            )
    return result


def _build_b23(spec: DatasetSpec, behavior: pd.DataFrame, manifest: pd.DataFrame):
    design = _participant_design(behavior)
    participant_rows = []
    pool_definitions: dict[str, dict[str, Any]] = {}
    for row in design.itertuples(index=False):
        condition = str(row.exposure_test_condition_id)
        talkers = _talkers(row.exposure_talkers)
        if condition == "B23.a_control":
            pool_id, status, reason = "B23V:no_exposure", "no_exposure", "untrained control"
        elif condition.startswith("B23.c_no"):
            pool_id = stable_id("B23V", condition, "blocked")
            status = "blocked"
            reason = "public B23 multi-talker stimulus mapping not yet integrated or validated"
        elif condition.startswith("B23.b_") and len(talkers) == 1:
            pool_id = stable_id("B23V", talkers[0])
            status, reason = "available", None
            pool_definitions[pool_id] = {
                "pool_id": pool_id,
                "dataset_id": "B23",
                "pool_status": status,
                "pool_reason": reason,
                "estimand": "actual_single_talker_exposure_variability",
                "exposure_talker_set": talkers[0],
            }
        else:
            raise ValueError(f"unexpected B23 design {condition}/{talkers}")
        participant_rows.append(
            {
                "participant_id": row.participant_id,
                "dataset_id": "B23",
                "condition_id": condition,
                "pool_id": pool_id,
                "pool_status": status,
                "pool_reason": reason,
            }
        )

    participant_frame = pd.DataFrame(participant_rows)
    pool_rows = [
        {
            "pool_id": "B23V:no_exposure",
            "dataset_id": "B23",
            "pool_status": "no_exposure",
            "pool_reason": "untrained control",
            "estimand": "actual_exposure_variability",
            "n_participants": int(participant_frame["pool_id"].eq("B23V:no_exposure").sum()),
            "n_presentations": 0,
        }
    ]
    for pool_id in sorted(participant_frame.loc[participant_frame["pool_status"].eq("blocked"), "pool_id"].unique()):
        rows = participant_frame.loc[participant_frame["pool_id"].eq(pool_id)]
        pool_rows.append(
            {
                "pool_id": pool_id,
                "dataset_id": "B23",
                "pool_status": "blocked",
                "pool_reason": rows["pool_reason"].iloc[0],
                "estimand": "actual_multi_talker_exposure_variability",
                "n_participants": int(rows["participant_id"].nunique()),
                "n_presentations": np.nan,
            }
        )

    tasks = []
    repo_root = spec.manifest.parents[2]
    interval_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for pool_id, definition in sorted(pool_definitions.items()):
        talker = definition["exposure_talker_set"]
        selected = manifest.loc[
            manifest["speaker_id"].eq(talker) & manifest["corpus_role"].eq("training_sentence")
        ].sort_values("segment_id")
        if len(selected) != 60:
            raise ValueError(f"B23 single-talker pool {talker} has {len(selected)} training sentences")
        for _, segment in selected.iterrows():
            presentation = str(segment["segment_id"])
            tasks.append(
                _task(
                    pool_id,
                    "B23",
                    f"{presentation}:sentence",
                    str(segment["sentence_normalized"]),
                    "sentence",
                    str(segment["speaker_id"]),
                    str(segment["segment_id"]),
                )
            )
            cache_key = str(segment["segment_id"])
            intervals = interval_cache.setdefault(cache_key, _textgrid_intervals(segment, repo_root))
            for unit in ("word", "phoneme"):
                for interval in intervals[unit]:
                    tasks.append(
                        _task(
                            pool_id,
                            "B23",
                            f"{presentation}:{unit}:{interval['index']}",
                            interval["label"],
                            unit,
                            str(segment["speaker_id"]),
                            str(segment["segment_id"]),
                            (interval["start"], interval["end"], interval["duration"]),
                        )
                    )
        pool_rows.append(
            {
                **definition,
                "n_participants": int(participant_frame["pool_id"].eq(pool_id).sum()),
                "n_presentations": 60,
            }
        )
    return pd.DataFrame(tasks), pd.DataFrame(pool_rows), participant_frame


def build_exposure_tables(spec: DatasetSpec, output_dir: str | Path):
    behavior = pd.read_csv(spec.behavior)
    manifest = pd.read_csv(spec.manifest)
    if spec.dataset_id == "AN19":
        tasks, pools, participants = _build_an19(spec, behavior, manifest)
    elif spec.dataset_id == "X21":
        tasks, pools, participants = _build_x21(spec, behavior, manifest)
    elif spec.dataset_id == "B23":
        tasks, pools, participants = _build_b23(spec, behavior, manifest)
    else:  # pragma: no cover
        raise ValueError(spec.dataset_id)
    tasks = tasks[TASK_COLUMNS].sort_values(["pool_id", "linguistic_unit", "token_id"]).reset_index(drop=True)
    pools = pools.sort_values("pool_id").reset_index(drop=True)
    participants = participants.sort_values("participant_id").reset_index(drop=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(destination / "exposure_tasks.csv", tasks)
    atomic_write_csv(destination / "exposure_pools.csv", pools)
    atomic_write_csv(destination / "participant_pools.csv", participants)
    provenance = {
        **runtime_record(),
        "stage": "build_exposure_tables",
        "dataset_id": spec.dataset_id,
        "behavior_sha256": sha256_file(spec.behavior),
        "manifest_sha256": sha256_file(spec.manifest),
        "task_rows": int(len(tasks)),
        "pool_status_counts": {
            str(key): int(value) for key, value in pools["pool_status"].value_counts().items()
        },
        "participant_rows": int(len(participants)),
    }
    atomic_write_json(destination / "provenance.json", provenance)
    return tasks, pools, participants, provenance


def _optional(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _compute_variability_key(
    tasks: pd.DataFrame,
    pools: pd.DataFrame,
    spec: FeatureStoreSpec,
    feature_key: str,
    tau: float,
    scaling: str,
    standardizer_dir: str | None,
):
    if scaling == "global_z":
        if standardizer_dir is None:
            raise ValueError(f"{spec.store_id}/{feature_key} requires global standardization")
        mean, scale = load_standardizer(standardizer_dir, spec.store_id, feature_key)
    elif scaling == "none":
        mean = scale = None
    else:
        raise ValueError(scaling)
    output = []
    with threadpool_limits(limits=1), FeatureStore(spec) as store:
        sequence_cache: dict[tuple[object, ...], np.ndarray] = {}
        for pool in pools.itertuples(index=False):
            pool_tasks = tasks.loc[tasks["pool_id"].eq(pool.pool_id)]
            for name in VARIABILITY_NAMES:
                unit = "all" if name == "overall" else name.rsplit("_", 1)[-1]
                if pool.pool_status != "available":
                    output.append(
                        {
                            "dataset_id": pool.dataset_id,
                            "store_id": spec.store_id,
                            "model_variant": spec.variant,
                            "feature_key": feature_key,
                            "layer": feature_key,
                            "pool_id": pool.pool_id,
                            "measure": name,
                            "linguistic_unit": unit,
                            "tau": tau,
                            "coordinate_scaling": scaling,
                            "value": np.nan,
                            "value_status": pool.pool_status,
                            "status_reason": pool.pool_reason,
                        }
                    )
                    continue
                if name == "overall":
                    # Use each exposure frame once.  Connected-speech datasets
                    # use complete sentence tokens; AN19 has only isolated-word
                    # recordings, for which the word token is the complete signal.
                    overall_unit = (
                        "sentence"
                        if pool_tasks["linguistic_unit"].eq("sentence").any()
                        else "word"
                    )
                    selected = pool_tasks.loc[pool_tasks["linguistic_unit"].eq(overall_unit)]
                else:
                    selected = pool_tasks.loc[pool_tasks["linguistic_unit"].eq(unit)]
                if selected.empty:
                    output.append(
                        {
                            "dataset_id": pool.dataset_id,
                            "store_id": spec.store_id,
                            "model_variant": spec.variant,
                            "feature_key": feature_key,
                            "layer": feature_key,
                            "pool_id": pool.pool_id,
                            "measure": name,
                            "linguistic_unit": unit,
                            "tau": tau,
                            "coordinate_scaling": scaling,
                            "value": np.nan,
                            "value_status": "unsupported_linguistic_unit",
                            "status_reason": f"no {unit} annotations registered for this dataset",
                        }
                    )
                    continue
                tokens = []
                for task in selected.itertuples(index=False):
                    cache_key = (
                        task.speaker_id,
                        task.unit_id,
                        _optional(task.start_seconds),
                        _optional(task.end_seconds),
                        _optional(task.duration_seconds),
                    )
                    if cache_key not in sequence_cache:
                        sequence = store.read(task.speaker_id, task.unit_id, feature_key)
                        sequence = slice_by_time(sequence, cache_key[2], cache_key[3], cache_key[4])
                        sequence_cache[cache_key] = apply_standardizer(sequence, mean, scale)
                    tokens.append(Token(str(task.token_id), str(task.type_id), sequence_cache[cache_key]))
                value = compute_variability(name, tokens, tau=tau)
                output.append(
                    {
                        "dataset_id": pool.dataset_id,
                        "store_id": spec.store_id,
                        "model_variant": spec.variant,
                        "feature_key": feature_key,
                        "layer": feature_key,
                        "pool_id": pool.pool_id,
                        "measure": name,
                        "linguistic_unit": unit,
                        "tau": tau,
                        "coordinate_scaling": scaling,
                        "value": value,
                        "value_status": "available" if np.isfinite(value) else "mathematically_undefined",
                        "status_reason": None if np.isfinite(value) else "measure has no eligible token pairs",
                    }
                )
    return pd.DataFrame(output)


def compute_exposure_variability(
    *,
    tasks_path: str | Path,
    pools_path: str | Path,
    spec: FeatureStoreSpec,
    feature_keys: list[str],
    profile: dict[str, Any],
    output_path: str | Path,
    jobs: int,
    standardizer_dir: str | None = None,
):
    tasks_path = Path(tasks_path).resolve()
    pools_path = Path(pools_path).resolve()
    tasks = pd.read_csv(tasks_path)
    pools = pd.read_csv(pools_path)
    if set(tasks["dataset_id"].unique()) != {spec.dataset} or set(pools["dataset_id"].unique()) != {spec.dataset}:
        raise ValueError("exposure tables and feature store refer to different datasets")
    scaling = str(profile.get("coordinate_scaling", {}).get(spec.kind, "none"))
    tau = float(profile["tau"])
    frames = Parallel(n_jobs=min(max(1, jobs), len(feature_keys)), backend="loky", verbose=5)(
        delayed(_compute_variability_key)(
            tasks, pools, spec, key, tau, scaling, standardizer_dir
        )
        for key in feature_keys
    )
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["feature_key", "pool_id", "measure"]
    ).reset_index(drop=True)
    destination = Path(output_path)
    atomic_write_csv(destination, result)
    provenance = {
        **runtime_record(),
        "stage": "compute_exposure_variability",
        "profile_id": profile.get("profile_id"),
        "store_id": spec.store_id,
        "tasks_sha256": sha256_file(tasks_path),
        "pools_sha256": sha256_file(pools_path),
        "feature_keys": feature_keys,
        "measures": list(VARIABILITY_NAMES),
        "tau": tau,
        "coordinate_scaling": scaling,
        "output_rows": int(len(result)),
        "value_status_counts": {
            str(key): int(value) for key, value in result["value_status"].value_counts().items()
        },
    }
    atomic_write_json(str(destination) + ".provenance.json", provenance)
    return result
