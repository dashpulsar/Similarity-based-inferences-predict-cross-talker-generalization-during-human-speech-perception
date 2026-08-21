from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DatasetSpec
from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


X21_ITEM = re.compile(
    r"^X21\.[^.]+\.[^.]+\.(?P<sentence>HT1_S\d{3})\.W(?P<keyword>.+)$"
)


def _item_logodds(training: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    stats = training.groupby(keys, as_index=False, observed=True).agg(
        n_correct=("response_correct", "sum"),
        n_incorrect=("response_incorrect", "sum"),
    )
    denominator = stats["n_correct"] + stats["n_incorrect"]
    probability = (stats["n_correct"] / denominator).clip(0.01, 0.99)
    stats["ceiling_raw"] = np.log(probability / (1.0 - probability))
    return stats[keys + ["ceiling_raw"]]


def _dataset_columns(dataset_id: str, rows: pd.DataFrame) -> tuple[list[str], list[str]]:
    if dataset_id == "AN19":
        rows["Keyword"] = rows["response_expected"].astype(str)
        rows["TestTalker"] = rows["item_talker"].astype(str)
        rows["SubjectID"] = rows["participant_id"].astype(str)
        return ["Keyword", "TestTalker"], ["SubjectID", "Keyword", "TestTalker"]
    if dataset_id == "X21":
        parsed = rows["item_id"].astype(str).map(X21_ITEM.fullmatch)
        if parsed.isna().any():
            example = rows.loc[parsed.isna(), "item_id"].iloc[0]
            raise ValueError(f"cannot parse X21 item ID: {example}")
        rows["Keyword"] = parsed.map(lambda value: value.group("keyword"))
        rows["SentenceID"] = parsed.map(lambda value: value.group("sentence"))
        rows["TestTalkerID"] = rows["item_talker"].astype(str)
        return ["Keyword", "TestTalkerID"], ["Keyword", "TestTalkerID", "SentenceID"]
    if dataset_id == "B23":
        rows["Sentence"] = rows["item_id"].astype(str).str.rsplit(".", n=1).str[-1]
        rows["TestTalker"] = rows["item_talker"].astype(str)
        rows["Condition"] = rows["exposure_test_condition.original"].astype(str)
        return ["Condition", "Sentence", "TestTalker"], ["Condition", "Sentence", "TestTalker"]
    raise ValueError(f"unsupported dataset: {dataset_id}")


def make_compatibility_ceiling_input(
    *,
    spec: DatasetSpec,
    folds_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Build the leakage-free item ceiling used by the historical 3-fold plot.

    For each participant fold, item accuracy is estimated using the other two
    folds only.  The resulting training-item log odds are then attached to the
    held-out responses and scaled with training-fold moments.  The subsequent
    GLMM is intentionally a held-out refit and is therefore a compatibility
    association statistic, not an out-of-sample prediction score.
    """

    folds_path = Path(folds_path).resolve()
    folds = pd.read_csv(folds_path)
    required_folds = {"participant_id", "fold"}
    if not required_folds.issubset(folds.columns):
        raise ValueError(f"fold table lacks columns: {sorted(required_folds - set(folds.columns))}")
    if folds["participant_id"].duplicated().any():
        raise ValueError("fold table has duplicate participant IDs")
    if set(folds["fold"].astype(int)) != {0, 1, 2}:
        raise ValueError("expected folds 0, 1, 2")

    rows = pd.read_csv(spec.behavior)
    rows = rows.loc[rows["phase"].eq("test")].copy()
    rows = rows.merge(
        folds[["participant_id", "fold"]], on="participant_id", how="left", validate="many_to_one"
    )
    if rows["fold"].isna().any():
        raise ValueError("behavior contains participants absent from fold table")
    rows["fold"] = rows["fold"].astype(int)
    keys, aggregate = _dataset_columns(spec.dataset_id, rows)

    output: list[pd.DataFrame] = []
    scaling: dict[str, dict[str, float | int]] = {}
    for fold_id in (0, 1, 2):
        training = rows.loc[rows["fold"] != fold_id].copy()
        held_out = rows.loc[rows["fold"] == fold_id].copy()
        lookup = _item_logodds(training, keys)
        training = training.merge(lookup, on=keys, how="left", validate="many_to_one")
        held_out = held_out.merge(lookup, on=keys, how="left", validate="many_to_one")
        if held_out["ceiling_raw"].isna().any():
            missing = held_out.loc[held_out["ceiling_raw"].isna(), keys].drop_duplicates().head()
            raise ValueError(f"held-out ceiling items absent from training fold {fold_id}:\n{missing}")
        center = float(training["ceiling_raw"].mean())
        spread = float(training["ceiling_raw"].std(ddof=1))
        if not math.isfinite(spread) or spread <= 0:
            raise ValueError(f"invalid ceiling scale in fold {fold_id}")
        held_out["ceiling_z"] = (held_out["ceiling_raw"] - center) / (2.0 * spread)
        grouped = held_out.groupby(aggregate, as_index=False, sort=False, observed=True).agg(
            ceiling_z=("ceiling_z", "first"),
            numCorrect=("response_correct", "sum"),
            numIncorrect=("response_incorrect", "sum"),
        )
        grouped.insert(0, "fold", fold_id)
        grouped.insert(0, "dataset_id", spec.dataset_id)
        output.append(grouped)
        scaling[str(fold_id)] = {
            "training_rows": int(len(training)),
            "heldout_rows": int(len(held_out)),
            "training_ceiling_mean": center,
            "training_ceiling_sd": spread,
        }

    result = pd.concat(output, ignore_index=True)
    if not np.isfinite(result["ceiling_z"]).all():
        raise ValueError("ceiling input contains non-finite values")
    destination = Path(output_path)
    atomic_write_csv(destination, result)
    atomic_write_json(
        str(destination) + ".provenance.json",
        {
            **runtime_record(),
            "stage": "notebook_compatibility_behavioral_ceiling_input",
            "publication_status": "compatibility_only",
            "interpretation": "heldout_refit_wald_z_is_association_stability_not_prediction",
            "dataset_id": spec.dataset_id,
            "behavior_path": str(spec.behavior),
            "behavior_sha256": sha256_file(spec.behavior),
            "folds_path": str(folds_path),
            "folds_sha256": sha256_file(folds_path),
            "output_rows": int(len(result)),
            "scaling_by_fold": scaling,
        },
    )
    return result
