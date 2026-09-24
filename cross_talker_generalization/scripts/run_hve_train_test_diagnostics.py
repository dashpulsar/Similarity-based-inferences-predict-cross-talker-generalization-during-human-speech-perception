"""Score revised HVE inputs without selecting layers or methods.

Default: all available Tr-24 definitions. --global-profiles scores the two
prespecified overall definitions at all registered layers. These are diagnostic
coverage choices, not winners selected by behavioral outcomes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ctg.parallel_glmm import fit_glmm_parallel
from ctg.provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


def prepare(output: Path, datasets: list[str], global_profiles: bool, exclude_tr24: bool = False) -> list[dict]:
    entries = []
    expected_n = {"AN19": 120, "X21": 320, "B23": 168}
    for dataset in datasets:
        for variant in ("base", "ft"):
            if global_profiles:
                suffix = "revised" if dataset == "B23" else "revised-selection"
                source = PROJECT / f"artifacts/derived/{dataset}-HVE-{variant}-{suffix}-model-input.csv"
                chunks = []
                for chunk in pd.read_csv(source, chunksize=100000):
                    mask = chunk.measure.isin(["overall", "overall_order_sensitive"])
                    if exclude_tr24:
                        mask &= chunk.registered_layer.ne("tr_24")
                    chunks.append(chunk.loc[mask])
                data = pd.concat(chunks, ignore_index=True)
            else:
                source = PROJECT / f"artifacts/derived/{dataset}-HVE-{variant}-tr24-20260906-model-input.csv"
                data = pd.read_csv(source)
                if set(data.registered_layer) != {"tr_24"}:
                    raise ValueError("Expected the precomputed Tr-24 HVE input")
            strata = {"available_exposure": data}
            if dataset == "B23":
                strata = {
                    "complete_order": data.loc[data.measure.eq("overall_order_sensitive")].copy(),
                    "available_exposure": data.loc[~data.measure.eq("overall_order_sensitive")].copy(),
                }
            for stratum, frame in strata.items():
                n_expected = 97 if dataset == "B23" and stratum == "complete_order" else expected_n[dataset]
                usable = frame.loc[frame.predictor_status.eq("available") & np.isfinite(frame.predictor_value)].copy()
                reference = None
                for feature, group in usable.groupby("feature_key", sort=True):
                    identity = group[["participant_id", "fold", "condition_id", "analysis_item_id",
                                      "test_talker_id", "response_correct", "response_incorrect"]]
                    identity = identity.sort_values(list(identity.columns)).reset_index(drop=True)
                    if reference is None:
                        reference = identity
                    else:
                        pd.testing.assert_frame_equal(identity, reference)
                    pf = group[["participant_id", "fold"]].drop_duplicates()
                    if pf.participant_id.duplicated().any() or len(pf) != n_expected or set(pf.fold) != {0, 1, 2}:
                        raise ValueError(f"Unexpected participant folds: {dataset} {feature}")
                if reference is None:
                    raise ValueError("No usable HVE responses")
                name = f"{dataset}-{variant}-{n_expected}"
                directory = output / name
                input_path = directory / "input.csv"
                atomic_write_csv(input_path, frame)
                response_path = directory / "response_set.csv"
                atomic_write_csv(response_path, reference)
                entries.append(dict(
                    run_label=f"HVE-{'global-' if global_profiles else ''}{name}", path=str(input_path.resolve()),
                    model_dir=str((directory / "m").resolve()), family="HVE", variant=variant,
                    stratum=f"{dataset}_{n_expected}_{stratum}", source_input=str(source),
                    source_sha256=sha256_file(source), input_sha256=sha256_file(input_path),
                    response_set_sha256=sha256_file(response_path),
                    same_responses_across_features=True, participant_count=n_expected,
                    rows_per_feature=len(reference),
                    responses_per_feature=int((reference.response_correct + reference.response_incorrect).sum()),
                    feature_count=usable.feature_key.nunique(), measures=sorted(usable.measure.unique()),
                    layers=sorted(usable.registered_layer.unique()),
                ))
                print("Prepared", name, entries[-1]["feature_count"], "features", flush=True)
    atomic_write_json(output / "run_inputs.json", dict(
        **runtime_record(), inputs=entries, selection="none", model_set="all",
        predictor_column="predictor_value", direction=1, term="variability_z",
        random_policy="registered", scope="global_profiles" if global_profiles else "all_Tr24_measures",
        exclude_tr24=exclude_tr24, runner_sha256=sha256_file(Path(__file__)),
    ))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=["AN19", "X21", "B23"], default=["AN19", "X21", "B23"])
    parser.add_argument("--global-profiles", action="store_true")
    parser.add_argument("--exclude-tr24", action="store_true", help="Reuse the separately scored Tr-24 inputs when preparing global profiles")
    parser.add_argument("--jobs", type=int, default=4, help="R workers per run; two runs proceed concurrently")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.exclude_tr24 and not args.global_profiles:
        parser.error("--exclude-tr24 requires --global-profiles")
    entries = prepare(args.output.resolve(), args.datasets, args.global_profiles, args.exclude_tr24)
    if args.prepare_only:
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(fit_glmm_parallel, input_path=e["path"], output_dir=e["model_dir"],
                        jobs=args.jobs, predictor_column="predictor_value", direction=1,
                        term="variability_z", model_set="all"): e["run_label"] for e in entries
        }
        for future in as_completed(futures):
            print("DONE", futures[future], len(future.result()), flush=True)
    print("HVE MATCHED-SCORE BATCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
