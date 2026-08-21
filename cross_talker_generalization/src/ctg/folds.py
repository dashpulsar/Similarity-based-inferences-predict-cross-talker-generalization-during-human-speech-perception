from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import DatasetSpec
from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


def _talker_set(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return ", ".join(sorted({part.strip() for part in str(value).split(",") if part.strip()}))


def make_participant_folds(
    spec: DatasetSpec,
    *,
    seed: int,
    n_folds: int,
    output_path: str | Path,
) -> pd.DataFrame:
    behavior = pd.read_csv(spec.behavior)
    test = behavior.loc[behavior["phase"].eq("test")].copy()
    test["exposure_talker_set"] = test["exposure_talkers"].map(_talker_set)
    columns = [
        "participant_id",
        "exposure_test_condition_id",
        "exposure_talker_set",
        "test_talkers",
    ]
    participant = test[columns].drop_duplicates()
    per_participant = participant.groupby("participant_id").size()
    if not per_participant.eq(1).all():
        bad = per_participant[~per_participant.eq(1)].index.tolist()[:10]
        raise ValueError(f"participants do not have one design assignment: {bad}")
    participant = participant.reset_index(drop=True)
    participant["stratum"] = (
        participant["exposure_test_condition_id"].astype(str)
        + "|"
        + participant["exposure_talker_set"].astype(str)
        + "|"
        + participant["test_talkers"].astype(str)
    )
    counts = participant["stratum"].value_counts()
    if int(counts.min()) < n_folds:
        raise ValueError(
            f"smallest design stratum has {int(counts.min())} participants, fewer than {n_folds} folds"
        )
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    participant["fold"] = -1
    for fold, (_, heldout) in enumerate(
        splitter.split(participant, participant["stratum"])
    ):
        participant.loc[heldout, "fold"] = fold
    if set(participant["fold"]) != set(range(n_folds)):
        raise AssertionError("fold assignment failed")
    participant = participant.sort_values("participant_id").reset_index(drop=True)
    destination = Path(output_path)
    atomic_write_csv(destination, participant)
    provenance = {
        **runtime_record(),
        "stage": "participant_folds",
        "dataset_id": spec.dataset_id,
        "behavior_path": str(spec.behavior),
        "behavior_sha256": sha256_file(spec.behavior),
        "seed": int(seed),
        "n_folds": int(n_folds),
        "n_participants": int(len(participant)),
        "n_strata": int(participant["stratum"].nunique()),
        "fold_counts": {
            str(key): int(value) for key, value in participant["fold"].value_counts().sort_index().items()
        },
    }
    atomic_write_json(str(destination) + ".provenance.json", provenance)
    return participant
