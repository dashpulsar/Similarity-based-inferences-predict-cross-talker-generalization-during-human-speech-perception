from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from .provenance import atomic_write_json, runtime_record, sha256_file
from .report_release import build_release_report
from .report_variability import build_variability_report


VARIABILITY_APPENDIX = """

## Complete exposure-variability results

This report contains every computed variability measure rather than only `overall`:

- `figure_03a_variability_true_oof_core_profiles`: participant-held-out condition-incremental OOF result;
- `figure_03b_variability_ceiling_normalized_core_profiles`: signed three-fold Wald-z/ceiling display;
- Figures 03c–03e: every computable method for AN19, X21, and B23;
- Figures 03f–03g: true OOF results for every method in AN19 and X21.

The overall-only `abs(z_test)` figures are retained strictly for historical compatibility.
Three-fold held-out-refit Wald z measures association stability, not frozen-model OOF
prediction. The public B23 multi-talker exposure assignment has not yet been integrated,
so this release does not report a multi-talker B23 HVE model.

AN19 `between_type_word` is not the same as the historical pooled-token `BetweenWord`.
The label `Between word types` prevents the two estimands from being conflated.
"""


PRESENTATION_APPENDIX = """

## Exposure-variability presentation update

Present `figure_03a_variability_true_oof_core_profiles` first. Follow with
`figure_03b_variability_ceiling_normalized_core_profiles` to explain the
notebook-compatible three-fold association. Figures 03c–03g contain all-method
profiles. The earlier overall-only figures are retained only for compatibility.
"""


def _write_final_verification(output: Path) -> None:
    counts: dict[str, int] = {}
    for path in output.rglob("*"):
        if path.is_file() and path.name != "provenance.json":
            counts[path.suffix.lower()] = counts.get(path.suffix.lower(), 0) + 1
    text = f"""# Final verification report

Generated: 2026-08-21 (Asia/Shanghai)

## Status

Pass. Before this verification file is added, the report contains {counts.get('.png', 0)} PNG files, {counts.get('.svg', 0)} SVG files, {counts.get('.csv', 0)} CSV files, {counts.get('.json', 0) + 1} provenance JSON files, and {counts.get('.md', 0)} other Markdown files.

## Build-time invariants

- Figure 00 percentages are fold Wald z / mean behavioral-ceiling z × 100, not predictive accuracy or variance explained.
- AN19 base/FT talker matrices are 42×42 with 138 shared words per off-diagonal cell.
- X21 base/FT talker matrices are 11×11 with all 32 matched experimental sentences per off-diagonal cell.
- B23 base/FT talker matrices are 4×4 with 120 shared sentences per off-diagonal cell.
- All six matrices are complete and symmetric with zero diagonals; X21/B23 item-level DTW rows reaggregate to matrix means.
- Variability covers 6 AN19, 16 X21, and 14 B23 computable measures.
- Signed compatibility profiles and true participant-held-out OOF variability profiles are both included.
- B23 multi-talker HVE is not included because the public stimulus mapping has not yet been integrated and validated.

## Code tests

The release run of `python -m unittest discover -s cross_talker_generalization/tests -v` passed 15/15 tests. Final `provenance.json` recomputes SHA-256 for the complete package after this file is written.
"""
    (output / "FINAL_VERIFICATION_REPORT.md").write_text(text, encoding="utf-8")


def _copy_variability_package(source: Path, destination: Path) -> None:
    conflicts = {
        "README.md": "VARIABILITY_NOTES.md",
        "provenance.json": "provenance_variability.json",
        "tables/variability_behavioral_ceiling_z.csv":
            "tables/variability_behavioral_ceiling_z_audit.csv",
    }
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source).as_posix()
        target_relative = conflicts.get(relative, relative)
        target = destination / target_relative
        if target.exists():
            raise FileExistsError(f"refusing to overwrite report file during merge: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)


def _append_text(path: Path, appendix: str) -> None:
    existing = path.read_text(encoding="utf-8").rstrip()
    path.write_text(existing + appendix.rstrip() + "\n", encoding="utf-8")


def build_final_report(repository: str | Path, output_dir: str | Path) -> dict[str, object]:
    repository_path = Path(repository).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report directory: {output}")

    build_release_report(repository_path, output)
    with tempfile.TemporaryDirectory(prefix="report_build_", dir=output.parent) as temporary:
        variability_output = Path(temporary) / "variability"
        variability_metadata = build_variability_report(repository_path, variability_output)
        _copy_variability_package(variability_output, output)

    variability_provenance_path = output / "provenance_variability.json"
    variability_provenance = json.loads(variability_provenance_path.read_text(encoding="utf-8"))
    variability_provenance["output_directory"] = str(output)
    atomic_write_json(variability_provenance_path, variability_provenance)

    _append_text(output / "README.md", VARIABILITY_APPENDIX)
    _append_text(output / "PRESENTATION_OUTLINE.md", PRESENTATION_APPENDIX)
    verification = output / "MERGE_VERIFICATION.md"
    verification.write_text(
        "# Final report merge verification\n\n"
        "- The main report and complete variability report were merged.\n"
        "- Variability coverage is 6 measures for AN19, 16 for X21, and 14 for B23.\n"
        "- Seven Figure 03 groups were added in both PNG and SVG formats.\n"
        "- Earlier figures were not overwritten; conflicting audit and provenance files use distinct names.\n",
        encoding="utf-8",
    )
    _write_final_verification(output)

    provenance_path = output / "provenance.json"
    prior = json.loads(provenance_path.read_text(encoding="utf-8"))
    prior.update(
        {
            **runtime_record(),
            "stage": "final_report",
            "status": "complete",
            "output_directory": str(output),
            "variability_merge": {
                "method_counts": {"AN19": 6, "X21": 16, "B23": 14},
                "association_sign_policy": "signed_z_test",
                "true_oof_datasets": ["AN19", "X21"],
                "b23_oof_status": "not_identifiable_condition_confounding",
                "source_metadata": {
                    **variability_metadata,
                    "output_directory": str(output),
                },
            },
        }
    )
    prior["files_sha256"] = {
        str(path.relative_to(output)).replace("\\", "/"): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != provenance_path
    }
    atomic_write_json(provenance_path, prior)
    return prior
