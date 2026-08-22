from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pandas as pd

from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file, stable_id


OUTPUT_TABLES = (
    "coefficients.csv",
    "diagnostics.csv",
    "likelihood_ratio_tests.csv",
    "oof_predictions.csv",
    "cv_metrics.csv",
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RCPP_PARALLEL_NUM_THREADS",
)


def _slug(value: str) -> str:
    # The repository root itself is long. R on Windows still encounters the
    # classic MAX_PATH boundary in file(), so internal run directories must be
    # deliberately short; feature_manifest.csv preserves the readable key.
    return f"f-{stable_id('g', value, length=10).split(':', 1)[1]}"


def _valid_cached_run(
    run_dir: Path,
    *,
    input_hash: str,
    script_hash: str,
    parameters: dict[str, Any],
) -> bool:
    provenance_path = run_dir / "provenance.json"
    if not provenance_path.is_file():
        return False
    try:
        import json

        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (
        provenance.get("status") != "complete"
        or provenance.get("input_sha256") != input_hash
        or provenance.get("r_script_sha256") != script_hash
        or provenance.get("parameters") != parameters
    ):
        return False
    recorded = provenance.get("outputs_sha256", {})
    return all(
        (run_dir / name).is_file() and recorded.get(name) == sha256_file(run_dir / name)
        for name in OUTPUT_TABLES
    )


def _fit_one(
    *,
    feature_key: str,
    input_frame: pd.DataFrame,
    run_dir: Path,
    rscript: str,
    script: Path,
    predictor_column: str,
    direction: int,
    term: str,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "model_input.csv"
    atomic_write_csv(input_path, input_frame)
    input_hash = sha256_file(input_path)
    script_hash = sha256_file(script)
    parameters: dict[str, Any] = {
        "predictor_column": predictor_column,
        "direction": int(direction),
        "term": term,
    }
    if _valid_cached_run(
        run_dir, input_hash=input_hash, script_hash=script_hash, parameters=parameters
    ):
        return {"feature_key": feature_key, "status": "cached", "run_dir": str(run_dir)}

    environment = dict(os.environ)
    for name in THREAD_ENVIRONMENT:
        environment[name] = "1"
    completed = subprocess.run(
        [
            rscript,
            str(script),
            str(input_path.resolve()),
            str(run_dir.resolve()),
            predictor_column,
            str(direction),
            term,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        atomic_write_json(
            run_dir / "provenance.json",
            {
                **runtime_record(),
                "status": "failed",
                "feature_key": feature_key,
                "input_sha256": input_hash,
                "r_script_sha256": script_hash,
                "parameters": parameters,
                "returncode": int(completed.returncode),
                "stdout": completed.stdout[-20000:],
                "stderr": completed.stderr[-20000:],
            },
        )
        raise RuntimeError(
            f"GLMM failed for {feature_key}: "
            + (completed.stderr.strip() or completed.stdout.strip())[-4000:]
        )
    missing = [name for name in OUTPUT_TABLES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"GLMM output missing for {feature_key}: {missing}")
    outputs = {name: sha256_file(run_dir / name) for name in OUTPUT_TABLES}
    diagnostics = pd.read_csv(run_dir / "diagnostics.csv")
    atomic_write_json(
        run_dir / "provenance.json",
        {
            **runtime_record(),
            "status": "complete",
            "stage": "parallel_confirmatory_glmm_feature",
            "feature_key": feature_key,
            "input_rows": int(len(input_frame)),
            "input_sha256": input_hash,
            "r_script_sha256": script_hash,
            "parameters": parameters,
            "outputs_sha256": outputs,
            "fit_status_counts": {
                str(key): int(value)
                for key, value in diagnostics["fit_ok"].value_counts(dropna=False).items()
            },
        "singular_fits": int(
            diagnostics["singular"].map(lambda value: str(value).strip().lower() == "true").sum()
        ),
            "non_ok_convergence": int((diagnostics["convergence"].fillna("") != "ok").sum()),
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-20000:],
        },
    )
    return {"feature_key": feature_key, "status": "computed", "run_dir": str(run_dir)}


def fit_glmm_parallel(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    jobs: int,
    predictor_column: str = "raw_distance",
    direction: int = -1,
    term: str = "similarity_z",
    rscript: str | None = None,
) -> pd.DataFrame:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    data = pd.read_csv(input_path)
    required = {"feature_key", predictor_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"model input lacks columns: {sorted(missing)}")
    features = sorted(data["feature_key"].dropna().astype(str).unique())
    if not features:
        raise ValueError("model input has no feature keys")
    system_r = Path(r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe")
    executable = rscript or (str(system_r) if system_r.is_file() else shutil.which("Rscript"))
    if not executable:
        raise FileNotFoundError("Rscript was not found")
    script = Path(__file__).resolve().parents[2] / "R" / "fit_confirmatory.R"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir / "r"
    runs_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(jobs, len(features))) as executor:
        futures = {}
        for feature in features:
            frame = data.loc[data["feature_key"].astype(str).eq(feature)].copy()
            run_dir = runs_root / _slug(feature)
            future = executor.submit(
                _fit_one,
                feature_key=feature,
                input_frame=frame,
                run_dir=run_dir,
                rscript=str(executable),
                script=script,
                predictor_column=predictor_column,
                direction=direction,
                term=term,
            )
            futures[future] = feature
        for future in as_completed(futures):
            feature = futures[future]
            try:
                record = future.result()
                records.append(record)
                print(f"GLMM feature {feature}: {record['status']}", flush=True)
            except Exception as error:  # aggregate all failures before aborting
                errors.append(f"{feature}: {error}")
                print(f"GLMM feature {feature}: failed", flush=True)
    if errors:
        raise RuntimeError("parallel GLMM failures:\n" + "\n".join(errors))

    manifest = pd.DataFrame(records).sort_values("feature_key").reset_index(drop=True)
    atomic_write_csv(output_dir / "feature_manifest.csv", manifest)
    for filename in OUTPUT_TABLES:
        frames = [pd.read_csv(Path(row.run_dir) / filename) for row in manifest.itertuples(index=False)]
        combined = pd.concat(frames, ignore_index=True)
        sort_columns = [
            column
            for column in ("feature_key", "scope", "fold", "model_id", "analysis_row_id", "term")
            if column in combined.columns
        ]
        if sort_columns:
            combined = combined.sort_values(sort_columns, na_position="first").reset_index(drop=True)
        atomic_write_csv(output_dir / filename, combined)

    software_frames = [
        pd.read_csv(Path(row.run_dir) / "software.csv") for row in manifest.itertuples(index=False)
    ]
    software = pd.concat(software_frames, ignore_index=True).drop_duplicates(
        ["R_version", "lme4_version", "dataset_id", "talker_strategy", "predictor_column", "predictor_direction", "predictor_term"]
    )
    software["parallel_jobs"] = min(jobs, len(features))
    software["feature_count"] = len(features)
    atomic_write_csv(output_dir / "software.csv", software)

    top_outputs = [*OUTPUT_TABLES, "software.csv", "feature_manifest.csv"]
    diagnostics = pd.read_csv(output_dir / "diagnostics.csv")
    provenance = {
        **runtime_record(),
        "status": "complete",
        "stage": "parallel_confirmatory_glmm",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "r_script": str(script),
        "r_script_sha256": sha256_file(script),
        "r_executable": str(executable),
        "parameters": {
            "predictor_column": predictor_column,
            "direction": int(direction),
            "term": term,
            "jobs": int(jobs),
        },
        "feature_keys": features,
        "feature_status_counts": {
            str(key): int(value) for key, value in manifest["status"].value_counts().items()
        },
        "selected_fit_diagnostics": {
            "fits": int(len(diagnostics)),
            "fit_failures": int((~diagnostics["fit_ok"].astype(bool)).sum()),
            "singular": int(diagnostics["singular"].fillna(False).astype(bool).sum()),
            "non_ok_convergence": int((diagnostics["convergence"].fillna("") != "ok").sum()),
        },
        "outputs_sha256": {name: sha256_file(output_dir / name) for name in top_outputs},
    }
    atomic_write_json(output_dir / "provenance.json", provenance)
    return manifest
