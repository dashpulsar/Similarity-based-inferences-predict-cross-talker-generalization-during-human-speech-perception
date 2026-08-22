from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectConfig
from .features import FeatureStore
from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


def _check(record: dict[str, Any], observed: object, expected: object) -> None:
    record["observed"] = observed
    record["expected"] = expected
    record["status"] = "pass" if observed == expected else "fail"


def run_audit(
    project: ProjectConfig, output_dir: str | Path, *, hash_large_files: bool = False
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    conditions: list[pd.DataFrame] = []
    warnings: list[str] = []

    for dataset_id, spec in project.datasets.items():
        for label, path in (("behavior", spec.behavior), ("manifest", spec.manifest)):
            records.append(
                {
                    "scope": dataset_id,
                    "component": label,
                    "check": "path_exists",
                    "observed": path.is_file(),
                    "expected": True,
                    "status": "pass" if path.is_file() else "fail",
                    "detail": str(path),
                }
            )
        if not spec.behavior.is_file() or not spec.manifest.is_file():
            continue
        behavior = pd.read_csv(spec.behavior)
        manifest = pd.read_csv(spec.manifest)
        test = behavior.loc[behavior["phase"].eq("test")]
        checks = [
            ("behavior_rows", len(behavior), spec.expected_behavior_rows),
            ("test_rows", len(test), spec.expected_test_rows),
            ("participants", behavior["participant_id"].nunique(), spec.expected_participants),
            ("manifest_units", len(manifest), spec.expected_manifest_units),
            ("test_talkers", test["item_talker"].nunique(), spec.expected_test_talkers),
            (
                "count_identity",
                bool((behavior["response_correct"] + behavior["response_incorrect"] > 0).all()),
                True,
            ),
        ]
        for name, observed, expected in checks:
            record = {
                "scope": dataset_id,
                "component": "dataset",
                "check": name,
                "detail": "",
            }
            _check(record, int(observed) if isinstance(observed, bool) is False and hasattr(observed, "item") else observed, expected)
            records.append(record)

        summary = (
            test.groupby("exposure_test_condition_id", dropna=False)
            .agg(
                n_rows=("participant_id", "size"),
                n_participants=("participant_id", "nunique"),
                n_items=("item_id", "nunique"),
                n_test_talkers=("item_talker", "nunique"),
            )
            .reset_index()
            .rename(columns={"exposure_test_condition_id": "condition_id"})
        )
        summary.insert(0, "dataset_id", dataset_id)
        conditions.append(summary)

    for store_id, spec in project.feature_stores.items():
        if not spec.path.is_file():
            records.append(
                {
                    "scope": store_id,
                    "component": "feature_store",
                    "check": "path_exists",
                    "observed": False,
                    "expected": True,
                    "status": "fail",
                    "detail": str(spec.path),
                }
            )
            continue
        with FeatureStore(spec) as store:
            keys = store.feature_keys()
            observed_units = store.unit_count(keys[0] if keys else None)
            attrs = store.attrs
            expected_keys = (
                set(project.layers)
                if spec.kind in {"hubert_tsne", "hubert_full"}
                else {"mfcc39", "strf24_legacy"}
            )
            store_checks = [
                ("dataset_id", str(attrs.get("dataset_id")), spec.dataset),
                ("unit_count", observed_units, spec.expected_units),
                ("feature_keys", set(keys), expected_keys),
            ]
            for name, observed, expected in store_checks:
                record = {
                    "scope": store_id,
                    "component": "feature_store",
                    "check": name,
                    "detail": str(spec.path),
                }
                if isinstance(observed, set):
                    _check(record, ",".join(sorted(observed)), ",".join(sorted(expected)))
                else:
                    _check(record, observed, expected)
                records.append(record)
            staging = [key for key in store.top_keys() if key.startswith("__")]
            if staging:
                warnings.append(f"{store_id}: ignored staging roots {staging}")
            file_stat = spec.path.stat()
            records.append(
                {
                    "scope": store_id,
                    "component": "feature_store",
                    "check": "file_fingerprint",
                    "observed": sha256_file(spec.path) if hash_large_files else "not_computed",
                    "expected": "sha256" if hash_large_files else "quick_fingerprint",
                    "status": "pass",
                    "detail": f"size={file_stat.st_size};mtime_ns={file_stat.st_mtime_ns};manifest_sha256={attrs.get('manifest_sha256', '')}",
                }
            )

    audit = pd.DataFrame(records)
    condition_frame = pd.concat(conditions, ignore_index=True) if conditions else pd.DataFrame()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(destination / "audit_summary.csv", audit)
    atomic_write_csv(destination / "conditions.csv", condition_frame)
    failed = audit.loc[audit["status"].eq("fail")]
    report = {
        **runtime_record(),
        "stage": "audit",
        "project_config": str(project.source_path),
        "project_config_sha256": sha256_file(project.source_path),
        "status": "pass" if failed.empty else "fail",
        "n_checks": int(len(audit)),
        "n_failed": int(len(failed)),
        "failed_checks": failed.to_dict("records"),
        "warnings": warnings,
        "large_file_hashes_computed": bool(hash_large_files),
    }
    atomic_write_json(destination / "audit_report.json", report)
    return report
