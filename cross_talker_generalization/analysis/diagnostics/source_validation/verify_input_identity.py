"""Read-only checks of input identities; writes only the JSON verification report.

Run with the project scientific Python environment. No fits, input edits, or
feature calculations are performed. The report contains counts/hashes, not
participant identifiers.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
UPDATE = HERE.parent
IDENTITY = ["participant_id", "fold", "condition_id", "analysis_item_id",
            "test_talker_id", "response_correct", "response_incorrect"]
CHECKS: list[dict] = []


def check(name, passed, **details):
    CHECKS.append(dict(check=name, passed=bool(passed), **details))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(frame):
    return frame[IDENTITY].sort_values(IDENTITY).reset_index(drop=True)


def identity_hash(frame):
    return hashlib.sha256(identity(frame).to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def counts(frame):
    return dict(rows=len(frame), participants=int(frame.participant_id.nunique()),
                word_responses=int((frame.response_correct + frame.response_incorrect).sum()),
                rows_by_fold={str(k): int(v) for k, v in frame.groupby("fold").size().items()},
                participants_by_fold={str(k): int(v) for k, v in frame.groupby("fold").participant_id.nunique().items()})


def inventory(entry, predictor, retain_feature=None):
    path = Path(entry["path"])
    actual = file_hash(path)
    expected = entry.get("input_sha256", entry.get("source_sha256", entry.get("sha256")))
    check(entry["run_label"] + ": input file hash", actual == expected,
          actual_sha256=actual, recorded_sha256=expected)
    columns = ["feature_key", *IDENTITY, predictor, "predictor_status"]
    parts = {}
    statuses = {}
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100000):
        for key, group in chunk.groupby("feature_key", sort=False):
            statuses.setdefault(key, Counter()).update(group.predictor_status.fillna("MISSING"))
            keep = group.predictor_status.eq("available") & np.isfinite(group[predictor])
            parts.setdefault(key, []).append(group.loc[keep, IDENTITY + [predictor]])
    summaries, first, retained = {}, None, None
    for key in sorted(parts):
        group = pd.concat(parts.pop(key), ignore_index=True)
        this_identity = identity(group)
        if first is None:
            first = this_identity
        check(entry["run_label"] + f": {key} same usable responses", this_identity.equals(first))
        pf = group[["participant_id", "fold"]].drop_duplicates()
        check(entry["run_label"] + f": {key} stable participant folds",
              not pf.participant_id.duplicated().any() and set(pf.fold) == {0, 1, 2})
        summaries[key] = dict(**counts(group), predictor_status_counts=dict(statuses[key]),
                              usable_identity_sha256=identity_hash(group))
        if key == retain_feature:
            retained = group.sort_values(IDENTITY).reset_index(drop=True)
    return dict(run_label=entry["run_label"], path=str(path), file_sha256=actual,
                feature_count=len(summaries), features=summaries), first, retained


def difference(a, b):
    # Multiset comparison preserves repeated behavioral rows if any exist.
    left = Counter(a.itertuples(index=False, name=None))
    right = Counter(b.itertuples(index=False, name=None))
    return dict(left_only_rows=sum((left-right).values()),
                right_only_rows=sum((right-left).values()),
                shared_rows=sum((left & right).values()))


def main():
    retained_entries = json.loads((UPDATE / "train_test/run_inputs.json").read_text())["inputs"]
    retained_entries = {entry["run_label"]: entry for entry in retained_entries}
    expanded = json.loads((UPDATE / "tt_all/run_inputs.json").read_text())["inputs"]
    sbi, sbi_refs, comparisons = [], {}, []
    expected_layers = {*(f"cnn_{i}" for i in range(2, 7)), *(f"tr_{i}" for i in range(0, 25, 2))}
    for entry in expanded:
        label = entry["run_label"].removesuffix("-all")
        result, ref, current = inventory(entry, "raw_distance", "tr_24")
        check(label + ": all 18 registered layers", set(result["features"]) == expected_layers)
        prior, prior_ref, previous = inventory(retained_entries[label], "raw_distance", "tr_24")
        equal = current.equals(previous)
        numeric = np.array_equal(current.raw_distance.to_numpy(), previous.raw_distance.to_numpy())
        check(label + ": Tr24 usable response and distance equality", equal,
              identities_equal=ref.equals(prior_ref), predictor_values_exact=numeric,
              max_absolute_predictor_difference=float(np.max(np.abs(current.raw_distance.to_numpy()-previous.raw_distance.to_numpy()))))
        # Full-column comparison includes unavailable/status rows and metadata.
        raw_current = []
        for chunk in pd.read_csv(entry["path"], chunksize=100000):
            raw_current.append(chunk.loc[chunk.feature_key.eq("tr_24")])
        a = pd.concat(raw_current, ignore_index=True).sort_values(IDENTITY).reset_index(drop=True)
        b = pd.read_csv(retained_entries[label]["path"]).sort_values(IDENTITY).reset_index(drop=True)
        full_equal = a.equals(b)
        check(label + ": Tr24 complete input equality", full_equal,
              new_rows=len(a), retained_rows=len(b),
              differing_columns=[c for c in a.columns if c not in b or not a[c].equals(b[c])])
        sbi.append(result)
        sbi_refs[label] = ref
    for dataset in ("X21", "B23"):
        acoustic, ref, _ = inventory(retained_entries[f"{dataset}-acoustic"], "raw_distance")
        sbi.append(acoustic)
        for variant in ("base", "ft"):
            label = f"{dataset}-{variant}"
            delta = difference(ref, sbi_refs[label])
            comparisons.append(dict(dataset=dataset, acoustic_vs_neural_variant=variant,
                                    acoustic_counts=counts(ref), neural_counts=counts(sbi_refs[label]),
                                    equal=ref.equals(sbi_refs[label]), **delta))
    hve, refs, manifest_hashes = [], {}, {}
    for batch in ("hve_tt", "hve_global"):
        entries = json.loads((UPDATE / batch / "run_inputs.json").read_text())["inputs"]
        for entry in entries:
            result, ref, _ = inventory(entry, "predictor_value")
            response_path = Path(entry["path"]).with_name("response_set.csv")
            saved = pd.read_csv(response_path)
            response_hash = file_hash(response_path)
            check(entry["run_label"] + ": recorded response-set file/hash",
                  ref.equals(saved) and response_hash == entry["response_set_sha256"],
                  actual_response_file_sha256=response_hash,
                  canonical_lf_identity_sha256=identity_hash(ref))
            key = (entry["stratum"], entry["variant"])
            if key in refs:
                check(entry["run_label"] + ": Tr24/global same response identities", refs[key].equals(ref))
                check(entry["run_label"] + ": Tr24/global manifest response hashes",
                      manifest_hashes[key] == entry["response_set_sha256"])
            else:
                refs[key] = ref
                manifest_hashes[key] = entry["response_set_sha256"]
            check(entry["run_label"] + ": manifest counts",
                  len(ref) == entry["rows_per_feature"]
                  and ref.participant_id.nunique() == entry["participant_count"]
                  and counts(ref)["word_responses"] == entry["responses_per_feature"]
                  and result["feature_count"] == entry["feature_count"])
            hve.append(dict(batch=batch, stratum=entry["stratum"], variant=entry["variant"], **result))
    for stratum in sorted({key[0] for key in refs}):
        check(stratum + ": base/ft response identities", refs[(stratum, "base")].equals(refs[(stratum, "ft")]))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Input identities, folds, availability and preserved Tr24 predictors; no GLMM fits or input mutations",
                  script_sha256=file_hash(Path(__file__)),
                  passed=all(row["passed"] for row in CHECKS), checks=CHECKS,
                  sbi_inventory=sbi, acoustic_neural_comparisons=comparisons, hve_inventory=hve)
    destination = HERE / "input_identity_validation.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(report=str(destination), checks=len(CHECKS),
                          failures=[row for row in CHECKS if not row["passed"]],
                          acoustic_neural_comparisons=comparisons), indent=2), flush=True)


if __name__ == "__main__":
    main()
