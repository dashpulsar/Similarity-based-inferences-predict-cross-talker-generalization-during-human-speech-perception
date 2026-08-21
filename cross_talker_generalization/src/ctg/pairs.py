from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DatasetSpec
from .provenance import (
    atomic_write_csv,
    atomic_write_json,
    runtime_record,
    sha256_file,
    stable_id,
)


PAIR_COLUMNS = [
    "pair_id",
    "dataset_id",
    "linguistic_unit",
    "content_label",
    "test_speaker_id",
    "test_unit_id",
    "test_start_seconds",
    "test_end_seconds",
    "test_duration_seconds",
    "source_speaker_id",
    "source_unit_id",
    "source_start_seconds",
    "source_end_seconds",
    "source_duration_seconds",
]


def _split_talkers(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return sorted({part.strip() for part in str(value).split(",") if part.strip()})


def _talker_set(talkers: Iterable[str]) -> str:
    return ", ".join(sorted(talkers))


def _norm(value: object) -> str:
    return str(value).strip().casefold()


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _pair_row(
    *,
    dataset_id: str,
    linguistic_unit: str,
    content_label: str,
    test_speaker_id: str,
    test_unit_id: str,
    source_speaker_id: str,
    source_unit_id: str,
    test_interval: tuple[float, float, float] | None = None,
    source_interval: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    test_start, test_end, test_duration = test_interval or (None, None, None)
    source_start, source_end, source_duration = source_interval or (None, None, None)
    pair_id = stable_id(
        f"{dataset_id}P",
        linguistic_unit,
        content_label,
        test_speaker_id,
        test_unit_id,
        test_start,
        test_end,
        source_speaker_id,
        source_unit_id,
        source_start,
        source_end,
    )
    return {
        "pair_id": pair_id,
        "dataset_id": dataset_id,
        "linguistic_unit": linguistic_unit,
        "content_label": content_label,
        "test_speaker_id": test_speaker_id,
        "test_unit_id": test_unit_id,
        "test_start_seconds": test_start,
        "test_end_seconds": test_end,
        "test_duration_seconds": test_duration,
        "source_speaker_id": source_speaker_id,
        "source_unit_id": source_unit_id,
        "source_start_seconds": source_start,
        "source_end_seconds": source_end,
        "source_duration_seconds": source_duration,
    }


def _base_cell(
    *,
    dataset_id: str,
    condition_id: str,
    behavior_item_id: str,
    analysis_item_id: str,
    response_expected: str,
    test_talker_id: str,
    talkers: list[str],
    n_participants: int,
) -> dict[str, object]:
    talker_text = _talker_set(talkers)
    exposure_set_id = stable_id(f"{dataset_id}E", talker_text)
    cell_id = stable_id(
        f"{dataset_id}C",
        condition_id,
        behavior_item_id,
        analysis_item_id,
        response_expected,
        test_talker_id,
        talker_text,
    )
    return {
        "cell_id": cell_id,
        "dataset_id": dataset_id,
        "condition_id": condition_id,
        "behavior_item_id": behavior_item_id,
        "analysis_item_id": analysis_item_id,
        "response_expected": response_expected,
        "test_talker_id": test_talker_id,
        "exposure_talker_set": talker_text,
        "exposure_set_id": exposure_set_id,
        "n_expected_source_talkers": len(talkers),
        "n_participants": int(n_participants),
    }


def _design_cells(behavior: pd.DataFrame) -> pd.DataFrame:
    test = behavior.loc[behavior["phase"].eq("test")].copy()
    columns = [
        "participant_id",
        "exposure_test_condition_id",
        "item_id",
        "response_expected",
        "item_talker",
        "exposure_talkers",
    ]
    missing = [column for column in columns if column not in test.columns]
    if missing:
        raise ValueError(f"behavior table lacks columns: {missing}")
    test["exposure_talker_set"] = test["exposure_talkers"].map(
        lambda value: _talker_set(_split_talkers(value))
    )
    keys = [
        "exposure_test_condition_id",
        "item_id",
        "response_expected",
        "item_talker",
        "exposure_talker_set",
    ]
    counts = test.groupby(keys, dropna=False)["participant_id"].nunique().rename("n_participants")
    result = counts.reset_index()
    result["exposure_talkers"] = result["exposure_talker_set"]
    return result


def _append_empty_cell(
    cells: list[dict[str, object]], base: dict[str, object], status: str, reason: str
) -> None:
    cells.append(
        {
            **base,
            "pair_id": None,
            "source_talker_id": None,
            "source_unit_id": None,
            "cell_status": status,
            "status_reason": reason,
        }
    )


def _build_an19(behavior: pd.DataFrame, manifest: pd.DataFrame):
    pairs: dict[str, dict[str, object]] = {}
    cells: list[dict[str, object]] = []
    manifest = manifest.copy()
    manifest["_word_norm"] = manifest["word"].map(_norm)

    for design in _design_cells(behavior).itertuples(index=False):
        expected = str(design.response_expected)
        test_matches = manifest.loc[
            manifest["behavior_item_id"].eq(design.item_id)
            & manifest["_word_norm"].eq(_norm(expected))
        ]
        talkers = _split_talkers(design.exposure_talkers)
        fallback_analysis_item = str(design.item_id)
        base = _base_cell(
            dataset_id="AN19",
            condition_id=str(design.exposure_test_condition_id),
            behavior_item_id=str(design.item_id),
            analysis_item_id=fallback_analysis_item,
            response_expected=expected,
            test_talker_id=str(design.item_talker),
            talkers=talkers,
            n_participants=int(design.n_participants),
        )
        if len(test_matches) != 1:
            _append_empty_cell(
                cells,
                base,
                "missing_test_mapping",
                f"expected one manifest row for item+word, found {len(test_matches)}",
            )
            continue

        test_row = test_matches.iloc[0]
        base = _base_cell(
            dataset_id="AN19",
            condition_id=str(design.exposure_test_condition_id),
            behavior_item_id=str(design.item_id),
            analysis_item_id=str(test_row["canonical_recording_item_id"]),
            response_expected=expected,
            test_talker_id=str(test_row["speaker_id"]),
            talkers=talkers,
            n_participants=int(design.n_participants),
        )
        if not talkers:
            _append_empty_cell(cells, base, "no_exposure", "untrained control")
            continue

        for source_talker in talkers:
            source_matches = manifest.loc[
                manifest["speaker_id"].eq(source_talker)
                & manifest["_word_norm"].eq(_norm(expected))
            ]
            if source_matches.empty:
                cells.append(
                    {
                        **base,
                        "pair_id": None,
                        "source_talker_id": source_talker,
                        "source_unit_id": None,
                        "cell_status": "missing_source_mapping",
                        "status_reason": "source talker has no registered recording of this word",
                    }
                )
                continue
            for _, source_row in source_matches.sort_values("recording_id").iterrows():
                pair = _pair_row(
                    dataset_id="AN19",
                    linguistic_unit="word",
                    content_label=expected.casefold(),
                    test_speaker_id=str(test_row["speaker_id"]),
                    test_unit_id=str(test_row["recording_id"]),
                    source_speaker_id=source_talker,
                    source_unit_id=str(source_row["recording_id"]),
                )
                pairs[pair["pair_id"]] = pair
                cells.append(
                    {
                        **base,
                        "pair_id": pair["pair_id"],
                        "source_talker_id": source_talker,
                        "source_unit_id": str(source_row["recording_id"]),
                        "cell_status": "available",
                        "status_reason": None,
                    }
                )
    return pd.DataFrame(pairs.values(), columns=PAIR_COLUMNS), pd.DataFrame(cells)


def _interval(row: pd.Series, expected: str) -> tuple[float, float, float]:
    intervals = json.loads(row["word_intervals_json"])
    matches = [
        interval
        for interval in intervals
        if _norm(interval.get("normalized", interval.get("label", ""))) == _norm(expected)
        and bool(interval.get("is_behavior_keyword", False))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one keyword interval for {expected!r} in {row['segment_id']}, found {len(matches)}"
        )
    match = matches[0]
    return float(match["start_seconds"]), float(match["end_seconds"]), float(row["duration_seconds"])


def _x21_segment_from_item(item_id: str) -> str:
    if ".W" not in item_id:
        raise ValueError(f"cannot recover X21 segment from {item_id!r}")
    return item_id.rsplit(".W", 1)[0]


def _build_x21(behavior: pd.DataFrame, manifest: pd.DataFrame):
    pairs: dict[str, dict[str, object]] = {}
    cells: list[dict[str, object]] = []
    by_segment = manifest.set_index("segment_id", drop=False)

    for design in _design_cells(behavior).itertuples(index=False):
        expected = str(design.response_expected)
        talkers = _split_talkers(design.exposure_talkers)
        segment_id = _x21_segment_from_item(str(design.item_id))
        base = _base_cell(
            dataset_id="X21",
            condition_id=str(design.exposure_test_condition_id),
            behavior_item_id=str(design.item_id),
            analysis_item_id=str(design.item_id),
            response_expected=expected,
            test_talker_id=str(design.item_talker),
            talkers=talkers,
            n_participants=int(design.n_participants),
        )
        if segment_id not in by_segment.index:
            _append_empty_cell(cells, base, "missing_test_mapping", segment_id)
            continue
        test_row = by_segment.loc[segment_id]
        if isinstance(test_row, pd.DataFrame):
            raise ValueError(f"duplicate X21 segment_id {segment_id}")
        try:
            test_interval = _interval(test_row, expected)
        except ValueError as exc:
            _append_empty_cell(cells, base, "missing_test_interval", str(exc))
            continue

        for source_talker in talkers:
            source_matches = manifest.loc[
                manifest["speaker_id"].eq(source_talker)
                & manifest["sentence_code"].eq(test_row["sentence_code"])
            ]
            if len(source_matches) != 1:
                cells.append(
                    {
                        **base,
                        "pair_id": None,
                        "source_talker_id": source_talker,
                        "source_unit_id": None,
                        "cell_status": "missing_source_mapping",
                        "status_reason": f"expected one same-sentence source, found {len(source_matches)}",
                    }
                )
                continue
            source_row = source_matches.iloc[0]
            try:
                source_interval = _interval(source_row, expected)
            except ValueError as exc:
                cells.append(
                    {
                        **base,
                        "pair_id": None,
                        "source_talker_id": source_talker,
                        "source_unit_id": str(source_row["segment_id"]),
                        "cell_status": "missing_source_interval",
                        "status_reason": str(exc),
                    }
                )
                continue
            pair = _pair_row(
                dataset_id="X21",
                linguistic_unit="keyword",
                content_label=f"{test_row['sentence_code']}::{expected.casefold()}",
                test_speaker_id=str(test_row["speaker_id"]),
                test_unit_id=str(test_row["segment_id"]),
                source_speaker_id=source_talker,
                source_unit_id=str(source_row["segment_id"]),
                test_interval=test_interval,
                source_interval=source_interval,
            )
            pairs[pair["pair_id"]] = pair
            cells.append(
                {
                    **base,
                    "pair_id": pair["pair_id"],
                    "source_talker_id": source_talker,
                    "source_unit_id": str(source_row["segment_id"]),
                    "cell_status": "available",
                    "status_reason": None,
                }
            )
    return pd.DataFrame(pairs.values(), columns=PAIR_COLUMNS), pd.DataFrame(cells)


def _build_b23(behavior: pd.DataFrame, manifest: pd.DataFrame):
    pairs: dict[str, dict[str, object]] = {}
    cells: list[dict[str, object]] = []

    for design in _design_cells(behavior).itertuples(index=False):
        talkers = _split_talkers(design.exposure_talkers)
        item_rows = manifest.loc[manifest["behavior_item_id"].eq(design.item_id)]
        actual = item_rows.loc[item_rows["is_actual_test_talker"].eq(True)]
        base = _base_cell(
            dataset_id="B23",
            condition_id=str(design.exposure_test_condition_id),
            behavior_item_id=str(design.item_id),
            analysis_item_id=str(design.item_id),
            response_expected=str(design.response_expected),
            test_talker_id=str(design.item_talker),
            talkers=talkers,
            n_participants=int(design.n_participants),
        )
        if len(item_rows) != 4 or len(actual) != 1:
            _append_empty_cell(
                cells,
                base,
                "missing_test_mapping",
                f"expected four same-sentence rows and one actual test row; found {len(item_rows)}/{len(actual)}",
            )
            continue
        test_row = actual.iloc[0]
        if not talkers:
            _append_empty_cell(cells, base, "no_exposure", "untrained control")
            continue
        for source_talker in talkers:
            source = item_rows.loc[item_rows["speaker_id"].eq(source_talker)]
            if len(source) != 1:
                cells.append(
                    {
                        **base,
                        "pair_id": None,
                        "source_talker_id": source_talker,
                        "source_unit_id": None,
                        "cell_status": "missing_source_mapping",
                        "status_reason": f"expected one same-sentence source, found {len(source)}",
                    }
                )
                continue
            source_row = source.iloc[0]
            pair = _pair_row(
                dataset_id="B23",
                linguistic_unit="sentence",
                content_label=str(test_row["sentence_normalized"]),
                test_speaker_id=str(test_row["speaker_id"]),
                test_unit_id=str(test_row["segment_id"]),
                source_speaker_id=source_talker,
                source_unit_id=str(source_row["segment_id"]),
            )
            pairs[pair["pair_id"]] = pair
            cells.append(
                {
                    **base,
                    "pair_id": pair["pair_id"],
                    "source_talker_id": source_talker,
                    "source_unit_id": str(source_row["segment_id"]),
                    "cell_status": "available",
                    "status_reason": None,
                }
            )
    return pd.DataFrame(pairs.values(), columns=PAIR_COLUMNS), pd.DataFrame(cells)


def build_pair_tables(spec: DatasetSpec, output_dir: str | Path):
    behavior = pd.read_csv(spec.behavior)
    manifest = pd.read_csv(spec.manifest)
    if spec.dataset_id == "AN19":
        pairs, cells = _build_an19(behavior, manifest)
    elif spec.dataset_id == "X21":
        pairs, cells = _build_x21(behavior, manifest)
    elif spec.dataset_id == "B23":
        pairs, cells = _build_b23(behavior, manifest)
    else:  # pragma: no cover
        raise ValueError(spec.dataset_id)

    if not pairs.empty:
        pairs = pairs.drop_duplicates("pair_id").sort_values("pair_id").reset_index(drop=True)
    cells = cells.sort_values(["cell_id", "source_talker_id", "source_unit_id"], na_position="last").reset_index(drop=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pair_path = destination / "pairs.csv"
    cell_path = destination / "cells.csv"
    atomic_write_csv(pair_path, pairs)
    atomic_write_csv(cell_path, cells)
    status_counts = cells["cell_status"].value_counts(dropna=False).to_dict()
    provenance = {
        **runtime_record(),
        "stage": "build_pairs",
        "dataset_id": spec.dataset_id,
        "estimand": "same_content_talker_proxy",
        "behavior_path": str(spec.behavior),
        "behavior_sha256": sha256_file(spec.behavior),
        "manifest_path": str(spec.manifest),
        "manifest_sha256": sha256_file(spec.manifest),
        "pair_count": int(len(pairs)),
        "cell_mapping_rows": int(len(cells)),
        "unique_cells": int(cells["cell_id"].nunique()),
        "cell_status_counts": {str(key): int(value) for key, value in status_counts.items()},
    }
    atomic_write_json(destination / "provenance.json", provenance)
    return pairs, cells, provenance
