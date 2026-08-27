"""Prepare compact participant-level exposure tables from public study files.

The generated CSVs contain one row per presentation and are the only public
source derivatives needed by the production HVE pipeline. Run this script from
the repository root when the upstream sources need to be refreshed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
import pyreadr


SOURCES = {
    "x21_training": "https://osf.io/download/8wtn2/",
    "b23_lists": "https://osf.io/download/x2dmg/",
    "b23_training": "https://osf.io/download/gt7bj/",
}


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as target:
        target.write(response.read())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _x21_table(root: Path, source: Path) -> pd.DataFrame:
    behavior = pd.read_csv(root / "data/preprocessed data/X21-behavioral-data.csv")
    manifest = pd.read_csv(root / "data/manifests/X21-stimulus-manifest.csv")
    raw = pyreadr.read_r(source)[None]
    columns = [
        "WorkerID", "Trial", "Filename", "Condition2", "TrainingTestSet",
        "ListNum", "PresentationBlock", "Experiment",
    ]
    rows = raw[columns].drop_duplicates().copy()
    participant_lookup = {
        str(value).removeprefix("X21."): str(value)
        for value in behavior["participant_id"].astype(str)
    }
    rows = rows.loc[rows["WorkerID"].astype(str).isin(participant_lookup)].copy()
    rows["participant_id"] = rows["WorkerID"].astype(str).map(participant_lookup)
    if rows["WorkerID"].nunique() != len(participant_lookup):
        raise ValueError("X21 public training data do not cover all production participants")
    counts = rows.groupby("WorkerID").agg(rows=("Trial", "size"), trials=("Trial", "nunique"))
    if not counts.eq(80).all().all():
        raise ValueError("each X21 participant must have 80 uniquely indexed presentations")

    lookup = manifest.set_index(["source_speaker_code", "sentence_code"])["segment_id"]
    pattern = re.compile(r"ALL_\d+_[MF]_([A-Z]+)_ENG_(HT\d_S\d+)$")

    def segment_id(filename: object) -> str:
        stem = Path(str(filename).replace("\\", "/")).name.removesuffix(".wav")
        match = pattern.fullmatch(stem)
        if match is None:
            raise ValueError(f"unrecognized X21 exposure filename: {filename}")
        parts = stem.split("_")
        speaker = f"{parts[3]}_{parts[2]}_{parts[1]}"
        sentence = match.group(2)
        return str(lookup.loc[(speaker, sentence)])

    result = pd.DataFrame(
        {
            "dataset_id": "X21",
            "participant_id": rows["participant_id"],
            "presentation_index": rows["Trial"].astype(int),
            "segment_id": rows["Filename"].map(segment_id),
            "source_filename": rows["Filename"].astype(str),
            "source_condition": rows["Condition2"].astype(str),
            "source_list": rows["ListNum"].astype(str),
            "source_trial": rows["Trial"].astype(int),
            "order_status": "available",
        }
    )
    return result.sort_values(["participant_id", "presentation_index"]).reset_index(drop=True)


def _b23_filenames(lists_path: Path, sheet: str) -> list[str]:
    raw = pd.read_excel(lists_path, sheet_name=sheet, header=None)
    filenames = []
    for row in raw.itertuples(index=False, name=None):
        candidates = [
            str(value).strip()
            for value in row
            if isinstance(value, str) and str(value).strip().lower().endswith(".wav")
        ]
        if candidates:
            filenames.append(candidates[-1])
    if len(filenames) != 60:
        raise ValueError(f"B23 sheet {sheet} contains {len(filenames)} audio rows, expected 60")
    return filenames


def _b23_table(root: Path, lists_path: Path, training_path: Path) -> pd.DataFrame:
    behavior = pd.read_csv(root / "data/preprocessed data/B23-behavioral-data.csv")
    manifest = pd.read_csv(root / "data/manifests/B23-stimulus-manifest.csv")
    training = pd.read_excel(training_path, sheet_name="all_training_data")
    participant_lookup = {
        str(value).removeprefix("B23."): str(value)
        for value in behavior.loc[
            ~behavior["exposure_test_condition_id"].eq("B23.a_control"), "participant_id"
        ].astype(str)
    }
    if set(training["id"].astype(str)) != set(participant_lookup):
        raise ValueError("B23 public training IDs do not match production trained participants")
    counts = training.groupby("id").size()
    if not counts.eq(60).all():
        raise ValueError("each B23 trained participant must have 60 presentation rows")

    sheets = {
        "BRP": "ST1-BRP", "FAR": "ST2-FAR", "SPA": "ST3-SPA", "TUR": "ST4-TUR",
        "noBRP": "MT1_noBRP", "noFAR": "MT2-noFAR", "noTUR": "MT3_noTUR",
        "noSPA": "MT4-noSPA",
    }
    filename_by_group_sentence: dict[tuple[str, str], str] = {}
    for group, sheet in sheets.items():
        for filename in _b23_filenames(lists_path, sheet):
            sentence = Path(filename).stem.split("_", 6)[-1].strip().upper()
            key = (group, sentence)
            if key in filename_by_group_sentence:
                raise ValueError(f"duplicate B23 group/sentence mapping: {key}")
            filename_by_group_sentence[key] = filename

    manifest_lookup = manifest.set_index(["source_speaker_code", "sentence_normalized"])["segment_id"]
    filename_pattern = re.compile(r"ALL_\d+_[MF]_(PBR|FAR|SPA|TUR)(?:_ENG)?_HT[12]_(.+)\.wav$")

    def mapped(row: pd.Series) -> tuple[str, str]:
        key = (str(row["group"]), str(row["sentence"]).strip().upper())
        filename = filename_by_group_sentence[key]
        match = filename_pattern.fullmatch(filename)
        if match is None:
            raise ValueError(f"unrecognized B23 exposure filename: {filename}")
        segment = str(manifest_lookup.loc[(match.group(1), match.group(2).strip().upper())])
        return filename, segment

    mappings = training.apply(mapped, axis=1, result_type="expand")
    mappings.columns = ["source_filename", "segment_id"]
    rows = pd.concat([training.reset_index(drop=True), mappings], axis=1)
    valid_order = rows.groupby("id")["trial"].transform(
        lambda values: len(values) == 60
        and values.nunique() == 60
        and set(values.astype(int)) == set(range(1, 61))
    )
    rows["presentation_index"] = rows["trial"].where(valid_order)
    rows["order_status"] = valid_order.map(
        {True: "available", False: "unavailable_duplicate_or_missing_trial_index"}
    )
    result = pd.DataFrame(
        {
            "dataset_id": "B23",
            "participant_id": rows["id"].astype(str).map(participant_lookup),
            "presentation_index": rows["presentation_index"].astype("Int64"),
            "segment_id": rows["segment_id"].astype(str),
            "source_filename": rows["source_filename"].astype(str),
            "source_condition": rows["group"].astype(str),
            "source_list": rows["group"].astype(str),
            "source_trial": rows["trial"],
            "order_status": rows["order_status"],
        }
    )
    return result.sort_values(
        ["participant_id", "presentation_index", "segment_id"], na_position="last"
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / "data/exposure_presentations"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ctg-public-exposure-") as temporary:
        temporary_path = Path(temporary)
        downloaded = {}
        for name, url in SOURCES.items():
            suffix = ".rds" if name == "x21_training" else ".xlsx"
            path = temporary_path / f"{name}{suffix}"
            _download(url, path)
            downloaded[name] = path
        x21 = _x21_table(root, downloaded["x21_training"])
        b23 = _b23_table(root, downloaded["b23_lists"], downloaded["b23_training"])
        x21.to_csv(destination / "X21-exposure-presentations.csv", index=False)
        b23.to_csv(destination / "B23-exposure-presentations.csv", index=False)
        provenance = {
            "sources": {
                name: {"url": SOURCES[name], "sha256": _sha256(path)}
                for name, path in downloaded.items()
            },
            "outputs": {
                "X21-exposure-presentations.csv": {
                    "rows": len(x21), "participants": x21["participant_id"].nunique()
                },
                "B23-exposure-presentations.csv": {
                    "rows": len(b23),
                    "participants": b23["participant_id"].nunique(),
                    "order_status_counts": b23.groupby("participant_id")["order_status"].first().value_counts().to_dict(),
                    "unique_segment_count_by_participant": b23.groupby("participant_id")["segment_id"].nunique().value_counts().sort_index().to_dict(),
                },
            },
        }
        (destination / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
