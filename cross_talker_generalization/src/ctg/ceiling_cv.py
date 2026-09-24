from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import DatasetSpec
from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


X21_ITEM = re.compile(
    r"^X21\.[^.]+\.[^.]+\.(?P<sentence>HT1_S\d{3})\.W(?P<keyword>.+)$"
)


def _prepare_keys(dataset_id: str, rows: pd.DataFrame) -> list[str]:
    required = ["item_id", "item_talker", "response_expected",
                "exposure_test_condition_id", "exposure_test_condition.original"]
    if rows[required].isna().any().any():
        raise ValueError("missing behavioral ceiling cell identifiers")
    rows["ceiling_condition_id"] = rows["exposure_test_condition_id"].astype(str)
    if dataset_id == "AN19":
        rows["ceiling_content_id"] = rows["response_expected"].astype(str)
        rows["ceiling_talker_id"] = rows["item_talker"].astype(str)
        return ["ceiling_condition_id", "ceiling_content_id", "ceiling_talker_id"]
    if dataset_id == "X21":
        parsed = rows["item_id"].astype(str).map(X21_ITEM.fullmatch)
        if parsed.isna().any():
            raise ValueError("an X21 item ID could not be parsed")
        rows["ceiling_content_id"] = parsed.map(lambda value: value.group("keyword"))
        rows["ceiling_sentence_id"] = parsed.map(lambda value: value.group("sentence"))
        rows["ceiling_talker_id"] = rows["item_talker"].astype(str)
        return ["ceiling_condition_id", "ceiling_sentence_id", "ceiling_content_id", "ceiling_talker_id"]
    if dataset_id == "B23":
        rows["ceiling_content_id"] = rows["item_id"].astype(str).str.rsplit(".", n=1).str[-1]
        rows["ceiling_talker_id"] = rows["item_talker"].astype(str)
        rows["ceiling_condition_id"] = rows["exposure_test_condition.original"].astype(str)
        return ["ceiling_condition_id", "ceiling_content_id", "ceiling_talker_id"]
    raise ValueError(dataset_id)


def compute_cross_validated_ceiling(
    *, spec: DatasetSpec, folds_path: str | Path, output_dir: str | Path,
    missing_cell_policy: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training-only item reference, conditional on experimental condition.

    Default behavior rejects unseen training cells. The diagnostic-only
    mark_unavailable policy retains those responses with missing predictions;
    partial-coverage metrics must not normalize models on the complete sample.
    """
    if missing_cell_policy not in {"error", "mark_unavailable"}:
        raise ValueError("missing_cell_policy must be error or mark_unavailable")
    folds_path = Path(folds_path).resolve()
    output_dir = Path(output_dir)
    folds = pd.read_csv(folds_path)
    if folds["participant_id"].duplicated().any() or folds["fold"].isna().any():
        raise ValueError("duplicate participants or missing fold assignments")
    if not folds["fold"].isin([0, 1, 2]).all():
        raise ValueError("expected integer folds 0, 1, 2")
    if set(folds["fold"].astype(int)) != {0, 1, 2}:
        raise ValueError("expected folds 0, 1, 2")
    behavior = pd.read_csv(spec.behavior)
    rows = behavior.loc[behavior["phase"].eq("test")].copy()
    counts = rows[["response_correct", "response_incorrect"]]
    if (counts.isna().any().any() or not np.isfinite(counts.to_numpy()).all()
            or (counts < 0).any().any() or (counts.sum(axis=1) <= 0).any()):
        raise ValueError("invalid behavioral response counts")
    rows = rows.merge(
        folds[["participant_id", "fold"]], on="participant_id", how="left", validate="many_to_one"
    )
    if rows["fold"].isna().any():
        raise ValueError("behavior participant absent from fold table")
    rows["fold"] = rows["fold"].astype(int)
    keys = _prepare_keys(spec.dataset_id, rows)
    predictions: list[pd.DataFrame] = []
    for fold_id in (0, 1, 2):
        training = rows.loc[rows["fold"] != fold_id]
        held_out = rows.loc[rows["fold"] == fold_id].copy()
        lookup = training.groupby(keys, as_index=False, observed=True).agg(
            training_correct=("response_correct", "sum"),
            training_incorrect=("response_incorrect", "sum"),
        )
        # Jeffreys smoothing prevents infinite loss while becoming negligible
        # for well-sampled items. No held-out response enters this estimate.
        lookup["predicted_probability"] = (
            (lookup["training_correct"] + 0.5)
            / (lookup["training_correct"] + lookup["training_incorrect"] + 1.0)
        )
        held_out = held_out.merge(lookup, on=keys, how="left", validate="many_to_one")
        if missing_cell_policy == "error" and held_out["predicted_probability"].isna().any():
            missing = held_out.loc[held_out["predicted_probability"].isna(), keys].drop_duplicates().head()
            raise ValueError(f"held-out ceiling cells absent from training fold {fold_id}:\n{missing}")
        held_out["prediction_status"] = np.where(
            held_out["predicted_probability"].notna(), "available", "unseen_training_cell"
        )
        probability = held_out["predicted_probability"].clip(1e-12, 1 - 1e-12)
        held_out["log_loss"] = -(
            held_out["response_correct"] * np.log(probability)
            + held_out["response_incorrect"] * np.log1p(-probability)
        )
        held_out["brier_sum"] = (
            held_out["response_correct"] * (1.0 - probability) ** 2
            + held_out["response_incorrect"] * probability**2
        )
        held_out["n_trials"] = held_out["response_correct"] + held_out["response_incorrect"]
        held_out["dataset_id"] = spec.dataset_id
        predictions.append(held_out)
    prediction = pd.concat(predictions, ignore_index=True)
    output_columns = [
        "dataset_id", "participant_id", "fold", "item_id", "item_talker",
        "response_expected", "response_correct", "response_incorrect", "n_trials",
        *keys, "training_correct", "training_incorrect", "predicted_probability",
        "log_loss", "brier_sum", "prediction_status", "exposure_test_condition_id",
    ]
    prediction = prediction[output_columns].sort_values(
        ["fold", "participant_id", "item_id", "response_expected"]
    ).reset_index(drop=True)
    metric_rows = []
    for fold_id in (0, 1, 2, None):
        selected = prediction if fold_id is None else prediction.loc[prediction["fold"].eq(fold_id)]
        available = selected.loc[selected["prediction_status"].eq("available")]
        total_trials = int(available["n_trials"].sum())
        metric_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "model_id": "cross_validated_behavioral_ceiling",
                "scope": "oof_all" if fold_id is None else "oof_fold",
                "fold": np.nan if fold_id is None else fold_id,
                "n_rows": int(len(available)),
                "total_trials": total_trials,
                "requested_n_rows": int(len(selected)),
                "requested_total_trials": int(selected["n_trials"].sum()),
                "missing_rows": int(len(selected) - len(available)),
                "status": "complete" if len(available) == len(selected) else "partial_coverage",
                "total_log_loss": float(available["log_loss"].sum()) if total_trials else np.nan,
                "mean_log_loss": float(available["log_loss"].sum() / total_trials) if total_trials else np.nan,
                "mean_brier": float(available["brier_sum"].sum() / total_trials) if total_trials else np.nan,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "oof_predictions.csv", prediction)
    atomic_write_csv(output_dir / "cv_metrics.csv", metrics)
    atomic_write_json(
        output_dir / "provenance.json",
        {
            **runtime_record(),
            "status": "complete" if prediction["prediction_status"].eq("available").all() else "partial_coverage",
            "stage": "cross_validated_behavioral_ceiling",
            "interpretation": "direct held-out prediction; no held-out refit",
            "dataset_id": spec.dataset_id,
            "behavior_path": str(spec.behavior),
            "behavior_sha256": sha256_file(spec.behavior),
            "folds_path": str(folds_path),
            "folds_sha256": sha256_file(folds_path),
            "item_keys": keys,
            "cell_definition_version": "condition_and_token_context_v2",
            "missing_cell_policy": missing_cell_policy,
            "missing_prediction_rows": int(prediction["prediction_status"].ne("available").sum()),
            "source_sha256": sha256_file(Path(__file__)),
            "smoothing": "Jeffreys Beta(0.5, 0.5)",
            "prediction_rows": int(len(prediction)),
        },
    )
    return prediction, metrics
