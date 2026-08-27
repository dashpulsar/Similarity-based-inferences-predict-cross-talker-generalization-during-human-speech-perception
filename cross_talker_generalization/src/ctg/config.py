from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    behavior: Path
    manifest: Path
    expected_behavior_rows: int
    expected_test_rows: int
    expected_participants: int
    expected_manifest_units: int
    expected_test_talkers: int
    outcome: str
    similarity_unit: str
    exposure_presentations: Path | None = None


@dataclass(frozen=True)
class FeatureStoreSpec:
    store_id: str
    dataset: str
    path: Path
    kind: str
    variant: str
    expected_units: int


@dataclass(frozen=True)
class ProjectConfig:
    source_path: Path
    schema_version: int
    seed: int
    n_folds: int
    default_jobs: int
    layers: tuple[str, ...]
    datasets: Mapping[str, DatasetSpec]
    feature_stores: Mapping[str, FeatureStoreSpec]

    @property
    def root(self) -> Path:
        return self.source_path.parent

    def dataset(self, dataset_id: str) -> DatasetSpec:
        key = dataset_id.strip().upper()
        if key not in self.datasets:
            raise ValueError(f"unknown dataset {dataset_id!r}")
        return self.datasets[key]

    def store(self, store_id: str) -> FeatureStoreSpec:
        if store_id not in self.feature_stores:
            accepted = ", ".join(sorted(self.feature_stores))
            raise ValueError(f"unknown feature store {store_id!r}; choose from {accepted}")
        return self.feature_stores[store_id]


def _absolute(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def load_project(path: str | Path) -> ProjectConfig:
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    base = source.parent
    datasets = {
        dataset_id: DatasetSpec(
            dataset_id=dataset_id,
            behavior=_absolute(base, values["behavior"]),
            manifest=_absolute(base, values["manifest"]),
            expected_behavior_rows=int(values["expected_behavior_rows"]),
            expected_test_rows=int(values["expected_test_rows"]),
            expected_participants=int(values["expected_participants"]),
            expected_manifest_units=int(values["expected_manifest_units"]),
            expected_test_talkers=int(values["expected_test_talkers"]),
            outcome=str(values["outcome"]),
            similarity_unit=str(values["similarity_unit"]),
            exposure_presentations=(
                _absolute(base, values["exposure_presentations"])
                if values.get("exposure_presentations")
                else None
            ),
        )
        for dataset_id, values in raw["datasets"].items()
    }
    stores = {
        store_id: FeatureStoreSpec(
            store_id=store_id,
            dataset=str(values["dataset"]),
            path=_absolute(base, values["path"]),
            kind=str(values["kind"]),
            variant=str(values["variant"]),
            expected_units=int(values["expected_units"]),
        )
        for store_id, values in raw["feature_stores"].items()
    }
    return ProjectConfig(
        source_path=source,
        schema_version=int(raw["schema_version"]),
        seed=int(raw["seed"]),
        n_folds=int(raw["n_folds"]),
        default_jobs=int(raw["default_jobs"]),
        layers=tuple(raw["layers"]),
        datasets=datasets,
        feature_stores=stores,
    )


def load_profile(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    profile = json.loads(source.read_text(encoding="utf-8"))
    profile["_source_path"] = str(source)
    return profile
