from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .config import FeatureStoreSpec


class FeatureStore(AbstractContextManager["FeatureStore"]):
    """Uniform reader for the three HDF5 layouts in the repository."""

    def __init__(self, spec: FeatureStoreSpec):
        self.spec = spec
        self._handle: h5py.File | None = None

    def __enter__(self) -> "FeatureStore":
        self._handle = h5py.File(self.spec.path, "r")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def h5(self) -> h5py.File:
        if self._handle is None:
            raise RuntimeError("FeatureStore must be used as a context manager")
        return self._handle

    @property
    def attrs(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in self.h5.attrs.items():
            if isinstance(value, bytes):
                result[key] = value.decode("utf-8", errors="replace")
            elif isinstance(value, np.generic):
                result[key] = value.item()
            else:
                result[key] = value
        return result

    def top_keys(self) -> list[str]:
        return list(self.h5.keys())

    def feature_keys(self) -> tuple[str, ...]:
        if self.spec.kind == "hubert_tsne":
            return tuple(self.h5.keys())
        for speaker in self.h5.keys():
            if speaker.startswith("__"):
                continue
            units = self.h5[speaker]
            if not units:
                continue
            first_unit = units[next(iter(units.keys()))]
            return tuple(first_unit.keys())
        return ()

    def unit_count(self, feature_key: str | None = None) -> int:
        if self.spec.kind == "hubert_tsne":
            key = feature_key or next(iter(self.h5.keys()))
            return sum(len(group) for group in self.h5[key].values())
        return sum(
            len(group)
            for name, group in self.h5.items()
            if not name.startswith("__")
        )

    def iter_unit_keys(self, feature_key: str) -> Iterator[tuple[str, str]]:
        if self.spec.kind == "hubert_tsne":
            root = self.h5[feature_key]
            for speaker in sorted(root.keys()):
                for unit in sorted(root[speaker].keys()):
                    yield speaker, unit
            return
        for speaker in sorted(key for key in self.h5.keys() if not key.startswith("__")):
            for unit in sorted(self.h5[speaker].keys()):
                yield speaker, unit

    def read(self, speaker_id: str, unit_id: str, feature_key: str) -> np.ndarray:
        if self.spec.kind == "hubert_tsne":
            dataset = self.h5[feature_key][speaker_id][unit_id]
        else:
            dataset = self.h5[speaker_id][unit_id][feature_key]
        array = np.asarray(dataset, dtype=np.float64)
        if array.ndim != 2 or min(array.shape) == 0:
            raise ValueError(
                f"invalid feature shape {array.shape} for {speaker_id}/{unit_id}/{feature_key}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite features for {speaker_id}/{unit_id}/{feature_key}")
        return array

    def iter_sequences(self, feature_key: str) -> Iterator[np.ndarray]:
        for speaker, unit in self.iter_unit_keys(feature_key):
            yield self.read(speaker, unit, feature_key)


def slice_by_time(
    sequence: np.ndarray,
    start_seconds: float | None,
    end_seconds: float | None,
    duration_seconds: float | None,
) -> np.ndarray:
    """Crop a feature sequence using manifest-relative time boundaries."""

    if start_seconds is None or end_seconds is None or duration_seconds is None:
        return sequence
    if not (0 <= start_seconds < end_seconds <= duration_seconds + 1e-6):
        raise ValueError(
            f"invalid interval ({start_seconds}, {end_seconds}) for duration {duration_seconds}"
        )
    frames = sequence.shape[0]
    start = int(round(frames * start_seconds / duration_seconds))
    end = int(round(frames * end_seconds / duration_seconds))
    start = max(0, min(start, frames - 1))
    end = max(start + 1, min(end, frames))
    return sequence[start:end]


def apply_standardizer(
    sequence: np.ndarray, mean: np.ndarray | None, scale: np.ndarray | None
) -> np.ndarray:
    if mean is None and scale is None:
        return sequence
    if mean is None or scale is None:
        raise ValueError("mean and scale must be supplied together")
    if mean.shape != (sequence.shape[1],) or scale.shape != mean.shape:
        raise ValueError("standardizer dimension does not match feature sequence")
    return (sequence - mean) / scale
