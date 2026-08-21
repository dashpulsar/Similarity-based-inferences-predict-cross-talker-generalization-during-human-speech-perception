from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from .aggregate import aggregate_predictors
from .final_report import build_final_report
from .audit import run_audit
from .ceiling import make_compatibility_ceiling_input
from .ceiling_cv import compute_cross_validated_ceiling
from .config import load_profile, load_project
from .correlations import plot_distance_correlations
from .distances import compute_distances
from .exposure import build_exposure_tables, compute_exposure_variability
from .features import FeatureStore
from .folds import make_participant_folds
from .model_input import make_model_input
from .parallel_glmm import fit_glmm_parallel
from .pairs import build_pair_tables
from .plots import plot_descriptive_s_curves, plot_glmm_profile
from .provenance import atomic_write_json, runtime_record, sha256_file
from .standardize import fit_standardizers
from .variability_input import make_variability_model_input


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ctg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="audit behavior, manifests, and feature stores")
    audit.add_argument("--project", type=Path, required=True)
    audit.add_argument("--output", type=Path, default=Path("artifacts/audit"))
    audit.add_argument("--hash-large-files", action="store_true")

    pairs = sub.add_parser("build-pairs", help="build deterministic same-content comparison tables")
    pairs.add_argument("--project", type=Path, required=True)
    pairs.add_argument("--dataset", choices=("AN19", "X21", "B23"), required=True)
    pairs.add_argument("--output", type=Path, required=True)

    standardizers = sub.add_parser("fit-standardizers", help="fit corpus-wide feature z-scaling")
    standardizers.add_argument("--project", type=Path, required=True)
    standardizers.add_argument("--store", required=True)
    standardizers.add_argument("--features", nargs="+")
    standardizers.add_argument("--jobs", type=int)
    standardizers.add_argument("--output", type=Path, required=True)

    distance = sub.add_parser("compute-distances", help="compute DTW for a physical pair table")
    distance.add_argument("--project", type=Path, required=True)
    distance.add_argument("--profile", type=Path, required=True)
    distance.add_argument("--pairs", type=Path, required=True)
    distance.add_argument("--store", required=True)
    distance.add_argument("--features", nargs="+")
    distance.add_argument("--layers", nargs="+", dest="features")
    distance.add_argument("--standardizer-dir", type=Path)
    distance.add_argument("--jobs", type=int)
    distance.add_argument("--output", type=Path, required=True)

    aggregate = sub.add_parser("aggregate", help="aggregate pair distances to condition-item predictors")
    aggregate.add_argument("--profile", type=Path, required=True)
    aggregate.add_argument("--cells", type=Path, required=True)
    aggregate.add_argument("--distances", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)

    folds = sub.add_parser("make-folds", help="create deterministic participant-disjoint folds")
    folds.add_argument("--project", type=Path, required=True)
    folds.add_argument("--dataset", choices=("AN19", "X21", "B23"), required=True)
    folds.add_argument("--output", type=Path, required=True)

    model_input = sub.add_parser("make-model-input", help="join predictors to behavioral rows")
    model_input.add_argument("--project", type=Path, required=True)
    model_input.add_argument("--dataset", choices=("AN19", "X21", "B23"), required=True)
    model_input.add_argument("--predictors", type=Path, required=True)
    model_input.add_argument("--folds", type=Path, required=True)
    model_input.add_argument("--output", type=Path, required=True)

    exposure = sub.add_parser("build-exposure", help="build actual-exposure pool and token tables")
    exposure.add_argument("--project", type=Path, required=True)
    exposure.add_argument("--dataset", choices=("AN19", "X21", "B23"), required=True)
    exposure.add_argument("--output", type=Path, required=True)

    variability = sub.add_parser("compute-variability", help="compute the 16 registered HVE measures")
    variability.add_argument("--project", type=Path, required=True)
    variability.add_argument("--profile", type=Path, required=True)
    variability.add_argument("--tasks", type=Path, required=True)
    variability.add_argument("--pools", type=Path, required=True)
    variability.add_argument("--store", required=True)
    variability.add_argument("--features", nargs="+")
    variability.add_argument("--standardizer-dir", type=Path)
    variability.add_argument("--jobs", type=int)
    variability.add_argument("--output", type=Path, required=True)

    variability_input = sub.add_parser("make-variability-input", help="join HVE values to behavioral rows")
    variability_input.add_argument("--project", type=Path, required=True)
    variability_input.add_argument("--dataset", choices=("AN19", "X21", "B23"), required=True)
    variability_input.add_argument("--variability", type=Path, required=True)
    variability_input.add_argument("--participant-pools", type=Path, required=True)
    variability_input.add_argument("--folds", type=Path, required=True)
    variability_input.add_argument("--measures", nargs="+")
    variability_input.add_argument("--output", type=Path, required=True)

    fit = sub.add_parser("fit-glmm", help="fit confirmatory GLMMs with frozen-fold prediction")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--rscript", default=None)
    fit.add_argument("--predictor-column", default="raw_distance")
    fit.add_argument("--direction", type=int, choices=(-1, 1), default=-1)
    fit.add_argument("--term", default="similarity_z")

    fit_parallel = sub.add_parser(
        "fit-glmm-parallel", help="fit independent feature GLMMs in parallel and merge outputs"
    )
    fit_parallel.add_argument("--input", type=Path, required=True)
    fit_parallel.add_argument("--output", type=Path, required=True)
    fit_parallel.add_argument("--jobs", type=int, default=4)
    fit_parallel.add_argument("--rscript", default=None)
    fit_parallel.add_argument("--predictor-column", default="raw_distance")
    fit_parallel.add_argument("--direction", type=int, choices=(-1, 1), default=-1)
    fit_parallel.add_argument("--term", default="similarity_z")

    ceiling_input = sub.add_parser(
        "make-ceiling-input", help="build cross-fitted behavioral ceiling input"
    )
    ceiling_input.add_argument("--project", type=Path, required=True)
    ceiling_input.add_argument("--dataset", choices=("AN19", "X21", "B23"), required=True)
    ceiling_input.add_argument("--folds", type=Path, required=True)
    ceiling_input.add_argument("--output", type=Path, required=True)

    ceiling_fit = sub.add_parser(
        "fit-ceiling-compatibility", help="fit notebook-compatible held-out ceiling z"
    )
    ceiling_fit.add_argument("--input", type=Path, required=True)
    ceiling_fit.add_argument("--output", type=Path, required=True)
    ceiling_fit.add_argument("--rscript", default=None)

    ceiling_cv = sub.add_parser(
        "compute-ceiling-cv", help="compute direct participant-held-out behavioral ceiling predictions"
    )
    ceiling_cv.add_argument("--project", type=Path, required=True)
    ceiling_cv.add_argument("--dataset", choices=("AN19", "X21", "B23"), required=True)
    ceiling_cv.add_argument("--folds", type=Path, required=True)
    ceiling_cv.add_argument("--output", type=Path, required=True)

    profile_plot = sub.add_parser("plot-profile", help="plot coefficient, LRT, and held-out layer profiles")
    profile_plot.add_argument("--model-dir", type=Path, required=True)
    profile_plot.add_argument("--output", type=Path, required=True)

    curves = sub.add_parser("plot-s-curves", help="plot descriptive sigmoid curves and quantile-bin CIs")
    curves.add_argument("--input", type=Path, required=True)
    curves.add_argument("--feature", required=True)
    curves.add_argument("--bins", type=int, default=10)
    curves.add_argument("--output", type=Path, required=True)

    correlations = sub.add_parser(
        "plot-distance-correlations", help="plot listwise-complete raw-distance correlations"
    )
    correlations.add_argument("--input", type=Path, required=True)
    correlations.add_argument("--observation-cols", nargs="+")
    correlations.add_argument("--output", type=Path, required=True)

    report = sub.add_parser("build-report", help="build the final cross-dataset report package")
    report.add_argument("--repository", type=Path, default=Path("."))
    report.add_argument("--output", type=Path, required=True)
    return parser


def _record_r_run(
    *,
    stage: str,
    input_path: Path,
    output_dir: Path,
    script: Path,
    executable: str,
    parameters: dict[str, object],
) -> None:
    output_files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "provenance.json"
    )
    atomic_write_json(
        output_dir / "provenance.json",
        {
            **runtime_record(),
            "stage": stage,
            "input_path": str(input_path.resolve()),
            "input_sha256": sha256_file(input_path),
            "r_script": str(script.resolve()),
            "r_script_sha256": sha256_file(script),
            "r_executable": str(executable),
            "parameters": parameters,
            "outputs_sha256": {path.name: sha256_file(path) for path in output_files},
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        report = run_audit(
            load_project(args.project), args.output, hash_large_files=args.hash_large_files
        )
        print(f"audit status={report['status']} checks={report['n_checks']} failed={report['n_failed']}")
        return 0 if report["status"] == "pass" else 1
    if args.command == "build-pairs":
        project = load_project(args.project)
        pairs, cells, provenance = build_pair_tables(project.dataset(args.dataset), args.output)
        print(
            f"{args.dataset}: {len(pairs)} unique pairs, {cells['cell_id'].nunique()} cells, "
            f"statuses={provenance['cell_status_counts']}"
        )
        return 0
    if args.command == "fit-standardizers":
        project = load_project(args.project)
        spec = project.store(args.store)
        with FeatureStore(spec) as store:
            available = list(store.feature_keys())
        features = args.features or available
        unknown = set(features).difference(available)
        if unknown:
            raise ValueError(f"features absent from store: {sorted(unknown)}")
        records = fit_standardizers(
            spec, features, args.output, args.jobs or project.default_jobs
        )
        print(f"wrote {len(records)} corpus standardizers")
        return 0
    if args.command == "compute-distances":
        project = load_project(args.project)
        spec = project.store(args.store)
        with FeatureStore(spec) as store:
            available = list(store.feature_keys())
        features = args.features or available
        result = compute_distances(
            pairs_path=args.pairs,
            spec=spec,
            feature_keys=features,
            profile=load_profile(args.profile),
            output_path=args.output,
            jobs=args.jobs or project.default_jobs,
            standardizer_dir=str(args.standardizer_dir) if args.standardizer_dir else None,
        )
        print(f"wrote {len(result)} distance rows to {args.output}")
        return 0
    if args.command == "aggregate":
        result = aggregate_predictors(
            cells_path=args.cells,
            distances_path=args.distances,
            profile=load_profile(args.profile),
            output_path=args.output,
        )
        print(f"wrote {len(result)} predictor rows to {args.output}")
        return 0
    if args.command == "make-folds":
        project = load_project(args.project)
        result = make_participant_folds(
            project.dataset(args.dataset),
            seed=project.seed,
            n_folds=project.n_folds,
            output_path=args.output,
        )
        print(f"wrote folds for {len(result)} participants")
        return 0
    if args.command == "make-model-input":
        project = load_project(args.project)
        result = make_model_input(
            spec=project.dataset(args.dataset),
            predictors_path=args.predictors,
            folds_path=args.folds,
            output_path=args.output,
        )
        print(f"wrote {len(result)} model rows to {args.output}")
        return 0
    if args.command == "build-exposure":
        project = load_project(args.project)
        tasks, pools, participants, provenance = build_exposure_tables(
            project.dataset(args.dataset), args.output
        )
        print(
            f"{args.dataset}: {len(tasks)} exposure token tasks, {len(pools)} pools, "
            f"{len(participants)} participant mappings; statuses={provenance['pool_status_counts']}"
        )
        return 0
    if args.command == "compute-variability":
        project = load_project(args.project)
        spec = project.store(args.store)
        with FeatureStore(spec) as store:
            available = list(store.feature_keys())
        features = args.features or available
        result = compute_exposure_variability(
            tasks_path=args.tasks,
            pools_path=args.pools,
            spec=spec,
            feature_keys=features,
            profile=load_profile(args.profile),
            output_path=args.output,
            jobs=args.jobs or project.default_jobs,
            standardizer_dir=str(args.standardizer_dir) if args.standardizer_dir else None,
        )
        print(f"wrote {len(result)} variability rows to {args.output}")
        return 0
    if args.command == "make-variability-input":
        project = load_project(args.project)
        result = make_variability_model_input(
            spec=project.dataset(args.dataset),
            variability_path=args.variability,
            participant_pools_path=args.participant_pools,
            folds_path=args.folds,
            output_path=args.output,
            measures=args.measures,
        )
        print(f"wrote {len(result)} variability model rows to {args.output}")
        return 0
    if args.command == "fit-glmm":
        system_r = Path(r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe")
        executable = args.rscript or (str(system_r) if system_r.is_file() else shutil.which("Rscript"))
        if not executable:
            raise FileNotFoundError("Rscript was not found")
        script = Path(__file__).resolve().parents[2] / "R" / "fit_confirmatory.R"
        args.output.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(executable), str(script), str(args.input.resolve()), str(args.output.resolve()),
                args.predictor_column, str(args.direction), args.term,
            ],
            check=True,
        )
        _record_r_run(
            stage="confirmatory_glmm",
            input_path=args.input,
            output_dir=args.output,
            script=script,
            executable=str(executable),
            parameters={
                "predictor_column": args.predictor_column,
                "direction": args.direction,
                "term": args.term,
            },
        )
        print(f"GLMM outputs written to {args.output}")
        return 0
    if args.command == "fit-glmm-parallel":
        result = fit_glmm_parallel(
            input_path=args.input,
            output_dir=args.output,
            jobs=args.jobs,
            predictor_column=args.predictor_column,
            direction=args.direction,
            term=args.term,
            rscript=args.rscript,
        )
        print(f"parallel GLMM completed for {len(result)} feature groups in {args.output}")
        return 0
    if args.command == "make-ceiling-input":
        project = load_project(args.project)
        result = make_compatibility_ceiling_input(
            spec=project.dataset(args.dataset),
            folds_path=args.folds,
            output_path=args.output,
        )
        print(f"wrote {len(result)} ceiling rows to {args.output}")
        return 0
    if args.command == "fit-ceiling-compatibility":
        system_r = Path(r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe")
        executable = args.rscript or (str(system_r) if system_r.is_file() else shutil.which("Rscript"))
        if not executable:
            raise FileNotFoundError("Rscript was not found")
        script = Path(__file__).resolve().parents[2] / "R" / "fit_ceiling_compatibility.R"
        args.output.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(executable), str(script), str(args.input.resolve()), str(args.output.resolve())],
            check=True,
        )
        _record_r_run(
            stage="notebook_compatibility_behavioral_ceiling",
            input_path=args.input,
            output_dir=args.output,
            script=script,
            executable=str(executable),
            parameters={
                "interpretation": "heldout refit association statistic, not prediction"
            },
        )
        print(f"compatibility ceiling outputs written to {args.output}")
        return 0
    if args.command == "compute-ceiling-cv":
        project = load_project(args.project)
        predictions, metrics = compute_cross_validated_ceiling(
            spec=project.dataset(args.dataset), folds_path=args.folds, output_dir=args.output
        )
        overall = metrics.loc[metrics["scope"].eq("oof_all"), "mean_log_loss"].iloc[0]
        print(
            f"{args.dataset} ceiling: {len(predictions)} held-out rows, "
            f"mean log loss={overall:.6f}"
        )
        return 0
    if args.command == "plot-profile":
        summary = plot_glmm_profile(args.model_dir, args.output)
        print(f"plotted {len(summary)} feature rows to {args.output}.png/.svg")
        return 0
    if args.command == "plot-s-curves":
        source = plot_descriptive_s_curves(args.input, args.feature, args.output, args.bins)
        print(f"plotted {len(source)} binned points to {args.output}.png/.svg")
        return 0
    if args.command == "plot-distance-correlations":
        result = plot_distance_correlations(
            args.input, args.output, observation_columns=args.observation_cols
        )
        print(f"wrote {len(result)} correlation cells to {args.output}.*")
        return 0
    if args.command == "build-report":
        metadata = build_final_report(args.repository, args.output)
        print(f"final report complete with {len(metadata['files_sha256'])} files in {args.output}")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
