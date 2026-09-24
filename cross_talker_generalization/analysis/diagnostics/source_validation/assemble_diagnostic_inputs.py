"""Assemble the completed September 17 batch, then audit its ratio outputs.

Run --assemble only after all merged model directories are complete. Next run
scripts/build_train_test_ratio_figures.py with diagnostic_inputs.json, then run
this script with --validate. Neither mode fits models or selects predictors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

UPDATE = Path(__file__).resolve().parents[1]
PACKAGE = UPDATE.parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
from ctg.provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file

MODELS = {"M_null", "M_condition", "M_predictor", "M_joint"}
KEYS = ["dataset_id", "feature_key", "model_id"]
EXPECTED_RUNS, EXPECTED_FEATURES, EXPECTED_HVE = 25, 396, 282


def truth(series):
    return series.astype(str).str.lower().isin(["true", "1", "1.0"])


def resolve(path, parent):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def inputs_from_batches():
    result, sources = [], []
    for batch in ["train_test", "tt_all", "hve_tt", "hve_global"]:
        batch_dir = {"hve_tt": PACKAGE / "analysis/hve/train_test",
                     "hve_global": PACKAGE / "analysis/hve/global_order"}.get(batch, UPDATE / batch)
        manifest = batch_dir / "run_inputs.json"
        sources.append(dict(path=str(manifest), sha256=sha256_file(manifest)))
        for original in json.loads(manifest.read_text(encoding="utf-8"))["inputs"]:
            source = dict(original)
            name = source["run_label"]
            if batch == "train_test" and name not in {
                "AN19-base", "AN19-ft", "AN19-acoustic", "X21-acoustic", "B23-acoustic"
            }:
                continue
            directory = resolve(source.get("model_dir", name), manifest.parent)
            source["model_dir"] = Path(os.path.relpath(directory, UPDATE)).as_posix()
            source["path"] = str(resolve(source["path"], manifest.parent))
            if batch == "tt_all":
                source["run_label"] = name.removesuffix("-all")
            if batch in {"train_test", "tt_all"}:
                source.update(family="SBI", measure="similarity", stratum="all_participants",
                              variant="acoustic" if name.endswith("-acoustic") else
                              "ft" if "-ft" in name else "base")
            else:
                if source.get("family") != "HVE" or not source.get("stratum"):
                    raise ValueError("HVE family and original participant stratum must be explicit")
            claimed = source.get("input_sha256") or source.get("sha256") or source.get("source_sha256")
            source["sha256"] = claimed
            if not claimed:
                raise ValueError(f"No input checksum registered for {name}")
            result.append(source)
    if len(result) != EXPECTED_RUNS or len({s["run_label"] for s in result}) != EXPECTED_RUNS:
        raise ValueError("Expected exactly 25 uniquely named diagnostic runs")
    return result, sources


def audit_runs(inputs):
    # Preflight every merged run before reading results or changing the manifest.
    missing = [str(UPDATE / source["model_dir"] / name) for source in inputs
               for name in ["train_test_scores.csv", "cv_metrics.csv", "diagnostics.csv"]
               if not (UPDATE / source["model_dir"] / name).is_file()]
    if missing:
        raise FileNotFoundError("Merged runs are incomplete:\n" + "\n".join(missing))
    records, statuses, structures = [], Counter(), Counter()
    for source in inputs:
        directory = UPDATE / source["model_dir"]
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError(f"Input checksum mismatch: {source['run_label']}")
        scores = pd.read_csv(directory / "train_test_scores.csv")
        metrics = pd.read_csv(directory / "cv_metrics.csv")
        diag = pd.read_csv(directory / "diagnostics.csv")
        features = set(pd.read_csv(source["path"], usecols=["feature_key"]).feature_key.unique())
        if set(scores.feature_key) != features:
            raise ValueError(f"Missing or extra scored features: {source['run_label']}")
        if source.get("feature_count", len(features)) != len(features):
            raise ValueError("Registered feature count differs from model input")
        if scores.duplicated(KEYS + ["fold", "split"]).any() or len(scores) != len(features) * 24:
            raise ValueError("Expected four models, three folds and two splits per feature")
        for _, group in scores.groupby(["dataset_id", "feature_key"]):
            if set(group.model_id) != MODELS:
                raise ValueError("Incomplete model set")
            for _, model in group.groupby("model_id"):
                if set(zip(model.fold, model.split)) != {(f, s) for f in range(3) for s in ["train", "test"]}:
                    raise ValueError("Incomplete split/fold score grid")
        ok = scores.score_status.eq("ok")
        np.testing.assert_allclose(scores.loc[ok, "mean_log_loss"],
                                   scores.loc[ok, "total_log_loss"] / scores.loc[ok, "total_trials"],
                                   rtol=1e-10, atol=1e-12)
        test = scores.loc[scores.split.eq("test")]
        fold_old = metrics.loc[metrics.scope.eq("oof_fold")]
        fold_join = test.merge(fold_old, on=KEYS + ["fold"], validate="one_to_one", suffixes=("_score", "_cv"))
        if len(fold_join) != len(test):
            raise ValueError("Missing fold-level CV metrics")
        for field in ["total_trials", "total_log_loss", "mean_log_loss"]:
            np.testing.assert_allclose(fold_join[f"{field}_score"], fold_join[f"{field}_cv"],
                                       rtol=1e-10, atol=1e-12, equal_nan=True)
        pooled = test.groupby(KEYS, as_index=False)[["total_trials", "total_log_loss"]].sum(min_count=3)
        pooled["mean_log_loss"] = pooled.total_log_loss / pooled.total_trials
        pooled_join = pooled.merge(metrics.loc[metrics.scope.eq("oof_all")], on=KEYS,
                                   validate="one_to_one", suffixes=("_score", "_cv"))
        if len(pooled_join) != len(features) * 4:
            raise ValueError("Missing pooled CV metrics")
        for field in ["total_trials", "total_log_loss", "mean_log_loss"]:
            np.testing.assert_allclose(pooled_join[f"{field}_score"], pooled_join[f"{field}_cv"],
                                       rtol=1e-10, atol=1e-12, equal_nan=True)
        cv = diag.loc[diag.scope.eq("cv_train")]
        if len(cv) != len(features) * 12 or cv.duplicated(KEYS + ["fold"]).any():
            raise ValueError("CV selected-fit diagnostics do not cover the score grid")
        statuses.update(scores.score_status.astype(str))
        structures.update(cv.random_structure.astype(str))
        records.append(dict(run_label=source["run_label"], family=source["family"],
                            stratum=source["stratum"], feature_count=len(features),
                            score_rows=len(scores), selected_cv_fit_count=len(cv),
                            selected_all_scope_fit_count=len(diag),
                            failed_cv_fits=int((~truth(cv.fit_ok)).sum()),
                            singular_cv_fits=int(truth(cv.singular).sum()),
                            nonconverged_cv_fits=int(cv.convergence.ne("ok").sum()),
                            score_status_counts=dict(Counter(scores.score_status.astype(str))),
                            pooled_oof_loss_matches=True, fold_oof_loss_matches=True,
                            input_sha256=source["sha256"],
                            score_sha256=sha256_file(directory / "train_test_scores.csv"),
                            cv_metrics_sha256=sha256_file(directory / "cv_metrics.csv"),
                            diagnostics_sha256=sha256_file(directory / "diagnostics.csv")))
    total = sum(r["feature_count"] for r in records)
    hve = sum(r["feature_count"] for r in records if r["family"] == "HVE")
    if (total, hve) != (EXPECTED_FEATURES, EXPECTED_HVE):
        raise ValueError(f"Expected 396 total/282 HVE feature specifications; found {total}/{hve}")
    return dict(run_count=len(records), feature_count=total, hve_feature_count=hve,
                score_rows=sum(r["score_rows"] for r in records),
                selected_cv_fit_count=sum(r["selected_cv_fit_count"] for r in records),
                selected_all_scope_fit_count=sum(r["selected_all_scope_fit_count"] for r in records),
                score_status_counts=dict(statuses), cv_random_structure_counts=dict(structures), runs=records)


def assemble():
    inputs, sources = inputs_from_batches()
    validation = audit_runs(inputs)
    target = UPDATE / "diagnostic_inputs.json"
    snapshot = UPDATE / "diagnostic_inputs_sbi.json"
    if target.exists() and not snapshot.exists():
        current = json.loads(target.read_text(encoding="utf-8"))
        if len(current["inputs"]) == 9 and all(s.get("family") == "SBI" for s in current["inputs"]):
            atomic_write_json(snapshot, current)
    atomic_write_json(target, dict(scope="Approved matched test/train mean-log-loss diagnostic; fixed parameters, no selection; HVE participant strata kept separate",
                                  source_manifests=sources, inputs=inputs))
    atomic_write_json(UPDATE / "source_validation/assembly_validation.json", dict(
        **runtime_record(), passed=True, diagnostic_manifest_sha256=sha256_file(target), **validation))
    print("Assembled 25 runs / 396 specifications. Next rebuild ratio figures, then use --validate.")


def validate():
    manifest = UPDATE / "diagnostic_inputs.json"
    inputs = json.loads(manifest.read_text(encoding="utf-8"))["inputs"]
    if len(inputs) != EXPECTED_RUNS:
        raise ValueError("Assemble the complete manifest first")
    validation = audit_runs(inputs)
    tables = UPDATE / "si_diagnostics/tables"
    build = json.loads((tables / "build_record.json").read_text(encoding="utf-8"))
    recorded_hashes = [r["sha256"] for r in build["source_manifests"]]
    relocation = build.get("layout_relocation", {})
    manifest_relocated = (
        recorded_hashes == [relocation.get("original_manifest_sha256")]
        and sha256_file(manifest) == relocation.get("equivalent_manifest_sha256")
        and relocation.get("input_definitions_checked") == EXPECTED_RUNS
    )
    if recorded_hashes != [sha256_file(manifest)] and not manifest_relocated:
        raise ValueError("Ratio outputs do not match the current manifest or its verified path relocation")
    inventory = pd.read_csv(tables / "source_inventory.csv").set_index("run_label")
    for source in validation["runs"]:
        if inventory.loc[source["run_label"], "score_sha256"] != source["score_sha256"]:
            raise ValueError("Scores changed after ratio generation")
    ratios = pd.read_csv(tables / "paired_fold_ratios.csv")
    summary = pd.read_csv(tables / "three_fold_ratio_summary.csv")
    figures = pd.read_csv(tables / "figure_inventory.csv")
    if len(ratios) != EXPECTED_FEATURES * 12 or len(summary) != EXPECTED_FEATURES * 4:
        raise ValueError("Ratio tables do not cover all specifications/models/folds")
    profiles = summary.loc[summary.model_id.eq("M_predictor")].groupby(
        ["dataset_id", "family", "variant", "measure", "stratum"], as_index=False).agg(
            n_layers=("layer", "nunique"), min_mean_ratio=("mean_fold_ratio", "min"),
            max_mean_ratio=("mean_fold_ratio", "max"), warning_folds=("n_warning_folds", "sum"),
            valid_folds=("n_valid_folds", "sum"))
    atomic_write_csv(UPDATE / "source_validation/profile_summary.csv", profiles)
    atomic_write_json(UPDATE / "source_validation/final_validation.json", dict(
        **runtime_record(), passed=True, diagnostic_manifest_sha256=sha256_file(manifest),
        manifest_relocated=manifest_relocated,
        n_ratio_rows=len(ratios), n_invalid_ratio_rows=int(ratios.ratio_status.ne("ok").sum()),
        n_warning_ratio_rows=int(truth(ratios.has_fit_warning).sum()),
        layer_profile_count=int(figures.plot_status.eq("layer_profile").sum()),
        table_only_profile_count=int(figures.plot_status.ne("layer_profile").sum()),
        interpretation="Checks establish output coverage and matched arithmetic; they do not establish absence of overfitting or independent selection validity.",
        **validation))
    print("Saved final_validation.json and profile_summary.csv.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--assemble", action="store_true")
    action.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    assemble() if args.assemble else validate()
