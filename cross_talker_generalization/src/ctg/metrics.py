from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _validate_tau(tau: float) -> float:
    if isinstance(tau, (bool, np.bool_)):
        raise ValueError("tau must be a finite positive number")
    value = float(tau)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("tau must be a finite positive number")
    return value


def _validate_sequence(value: np.ndarray) -> np.ndarray:
    sequence = np.asarray(value, dtype=np.float64)
    if sequence.ndim != 2 or min(sequence.shape) == 0:
        raise ValueError(f"expected non-empty (frames, dimensions), got {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise ValueError("feature sequences must be finite")
    return sequence


def _dtw_kernel(left: np.ndarray, right: np.ndarray, tau: float, metric_code: int):
    width = right.shape[0] + 1
    previous_cost = np.full(width, np.inf, dtype=np.float64)
    current_cost = np.full(width, np.inf, dtype=np.float64)
    previous_steps = np.zeros(width, dtype=np.int64)
    current_steps = np.zeros(width, dtype=np.int64)
    previous_cost[0] = 0.0

    for i in range(left.shape[0]):
        current_cost[0] = np.inf
        current_steps[0] = 0
        for j in range(right.shape[0]):
            if metric_code == 0:
                powered = 0.0
                for d in range(left.shape[1]):
                    powered += abs(left[i, d] - right[j, d]) ** tau
                local = powered ** (1.0 / tau)
            else:
                dot = 0.0
                norm_left = 0.0
                norm_right = 0.0
                for d in range(left.shape[1]):
                    dot += left[i, d] * right[j, d]
                    norm_left += left[i, d] * left[i, d]
                    norm_right += right[j, d] * right[j, d]
                local = 1.0 if norm_left == 0.0 or norm_right == 0.0 else 1.0 - dot / ((norm_left * norm_right) ** 0.5)

            column = j + 1
            # Prefer the shorter path when costs tie; diagonal is considered first.
            best_cost = previous_cost[column - 1]
            best_steps = previous_steps[column - 1]
            candidate_cost = previous_cost[column]
            candidate_steps = previous_steps[column]
            if candidate_cost < best_cost or (
                candidate_cost == best_cost and candidate_steps < best_steps
            ):
                best_cost, best_steps = candidate_cost, candidate_steps
            candidate_cost = current_cost[column - 1]
            candidate_steps = current_steps[column - 1]
            if candidate_cost < best_cost or (
                candidate_cost == best_cost and candidate_steps < best_steps
            ):
                best_cost, best_steps = candidate_cost, candidate_steps
            current_cost[column] = local + best_cost
            current_steps[column] = best_steps + 1

        previous_cost, current_cost = current_cost, previous_cost
        previous_steps, current_steps = current_steps, previous_steps
    return previous_cost[-1], previous_steps[-1]


if njit is not None:
    _dtw_fast = njit(cache=True, nogil=True)(_dtw_kernel)
else:  # pragma: no cover
    _dtw_fast = _dtw_kernel


@dataclass(frozen=True)
class DtwResult:
    distance: float
    raw_cost: float
    path_length: int
    left_frames: int
    right_frames: int
    normalization: str


def dtw_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    tau: float = 2.0,
    metric: str = "minkowski",
    normalization: str = "mean_sequence_length",
) -> DtwResult:
    tau = _validate_tau(tau)
    left = _validate_sequence(left)
    right = _validate_sequence(right)
    if left.shape[1] != right.shape[1]:
        raise ValueError("feature dimensions do not match")
    if metric not in {"minkowski", "cosine"}:
        raise ValueError("metric must be minkowski or cosine")
    if normalization not in {"mean_sequence_length", "path_length", "none"}:
        raise ValueError("unknown DTW normalization")
    raw_cost, path_length = _dtw_fast(left, right, tau, 0 if metric == "minkowski" else 1)
    if normalization == "mean_sequence_length":
        denominator = (left.shape[0] + right.shape[0]) / 2.0
    elif normalization == "path_length":
        denominator = float(path_length)
    else:
        denominator = 1.0
    return DtwResult(
        distance=float(raw_cost / denominator),
        raw_cost=float(raw_cost),
        path_length=int(path_length),
        left_frames=int(left.shape[0]),
        right_frames=int(right.shape[0]),
        normalization=normalization,
    )


def distance_to_similarity(distance: float | np.ndarray, k: float = 1.0):
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be positive")
    values = np.asarray(distance, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("distance must be finite and non-negative")
    result = np.exp(-k * values)
    return float(result) if result.ndim == 0 else result


def generalized_dispersion(sequence: np.ndarray, tau: float = 2.0) -> float:
    sequence = _validate_sequence(sequence)
    tau = _validate_tau(tau)
    center = sequence.mean(axis=0, keepdims=True)
    return float(np.power(np.abs(sequence - center), tau).sum(axis=1).mean())


def consecutive_dispersion(sequence: np.ndarray, tau: float = 2.0) -> float:
    sequence = _validate_sequence(sequence)
    tau = _validate_tau(tau)
    if sequence.shape[0] < 2:
        return float("nan")
    return float(np.power(np.abs(np.diff(sequence, axis=0)), tau).sum(axis=1).mean())


@dataclass(frozen=True)
class Token:
    token_id: str
    type_id: str
    sequence: np.ndarray
    presentation_index: int | None = None


UNITS = ("sentence", "word", "phoneme")
VARIABILITY_NAMES = (
    "overall",
    "overall_order_sensitive",
    *(f"within_token_{unit}" for unit in UNITS),
    *(f"within_type_{unit}" for unit in UNITS),
    *(f"between_type_{unit}" for unit in UNITS),
    *(f"order_{unit}" for unit in UNITS),
    *(f"mean_dissimilarity_{unit}" for unit in UNITS),
)


def compute_variability(name: str, tokens: Sequence[Token], tau: float = 2.0) -> float:
    if name not in VARIABILITY_NAMES:
        raise ValueError(f"unknown variability measure {name!r}")
    if not tokens:
        raise ValueError("at least one token is required")
    checked = [
        Token(t.token_id, t.type_id, _validate_sequence(t.sequence), t.presentation_index)
        for t in tokens
    ]
    if len({t.token_id for t in checked}) != len(checked):
        raise ValueError("token IDs must be unique")
    if len({t.sequence.shape[1] for t in checked}) != 1:
        raise ValueError("token dimensions do not match")

    if name == "overall":
        return generalized_dispersion(np.vstack([t.sequence for t in checked]), tau)
    if name == "overall_order_sensitive":
        indices = [token.presentation_index for token in checked]
        if any(index is None for index in indices):
            raise ValueError("overall_order_sensitive requires a presentation index for every token")
        numeric_indices = [int(index) for index in indices if index is not None]
        if len(set(numeric_indices)) != len(numeric_indices):
            raise ValueError("presentation indices must be unique within an exposure sequence")
        ordered = [
            token
            for _, token in sorted(
                zip(numeric_indices, checked), key=lambda pair: pair[0]
            )
        ]
        return consecutive_dispersion(np.vstack([token.sequence for token in ordered]), tau)
    if name.startswith("within_token_"):
        return float(np.mean([generalized_dispersion(t.sequence, tau) for t in checked]))
    if name.startswith("order_"):
        values = [consecutive_dispersion(t.sequence, tau) for t in checked]
        values = [value for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else float("nan")

    by_type: dict[str, list[Token]] = defaultdict(list)
    for token in checked:
        by_type[token.type_id].append(token)

    if name.startswith("within_type_"):
        values = []
        for instances in by_type.values():
            if len(instances) > 1:
                centers = np.vstack([token.sequence.mean(axis=0) for token in instances])
                values.append(generalized_dispersion(centers, tau))
        return float(np.mean(values)) if values else float("nan")
    if name.startswith("between_type_"):
        centers = np.vstack(
            [
                np.vstack([token.sequence.mean(axis=0) for token in instances]).mean(axis=0)
                for instances in by_type.values()
            ]
        )
        return generalized_dispersion(centers, tau)
    if name.startswith("mean_dissimilarity_"):
        type_values = []
        for instances in by_type.values():
            pair_values = [
                dtw_distance(left.sequence, right.sequence, tau=tau).distance
                for left, right in combinations(instances, 2)
            ]
            if pair_values:
                type_values.append(float(np.mean(pair_values)))
        return float(np.mean(type_values)) if type_values else float("nan")
    raise AssertionError(name)
