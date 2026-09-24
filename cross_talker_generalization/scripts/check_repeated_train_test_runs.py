"""Check repeated Tr-24 scores when extending fixed-parameter SBI to all layers."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ctg.provenance import atomic_write_json, runtime_record, sha256_file


def compare(old_path, new_path, keys):
    old = pd.read_csv(old_path).query("feature_key == 'tr_24'").sort_values(keys).reset_index(drop=True)
    new = pd.read_csv(new_path).query("feature_key == 'tr_24'").sort_values(keys).reset_index(drop=True)
    if not len(old) or old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError("Empty or duplicated repeated-run comparison keys")
    pd.testing.assert_frame_equal(old[keys], new[keys], check_dtype=False)
    common = sorted(set(old.columns) & set(new.columns))
    numeric, nonnumeric = [], []
    max_difference = {}
    for field in common:
        if pd.api.types.is_numeric_dtype(old[field]) and pd.api.types.is_numeric_dtype(new[field]):
            # Boolean status is compared exactly, not through approximate arithmetic.
            if pd.api.types.is_bool_dtype(old[field]):
                pd.testing.assert_series_equal(old[field], new[field], check_dtype=False)
                nonnumeric.append(field)
            else:
                np.testing.assert_allclose(old[field], new[field], rtol=1e-10, atol=1e-12, equal_nan=True,
                                           err_msg=f"Repeated-run mismatch: {field}")
                finite = np.isfinite(old[field]) & np.isfinite(new[field])
                max_difference[field] = float((old.loc[finite, field] - new.loc[finite, field]).abs().max()) if finite.any() else 0.
                numeric.append(field)
        else:
            pd.testing.assert_series_equal(old[field], new[field], check_dtype=False)
            nonnumeric.append(field)
    return dict(old_file=str(old_path), new_file=str(new_path), old_sha256=sha256_file(old_path),
                new_sha256=sha256_file(new_path), compared_rows=len(old), common_columns=common,
                numeric_columns=numeric, exact_nonnumeric_columns=nonnumeric,
                numeric_max_absolute_difference=max_difference,
                additional_new_columns=sorted(set(new.columns) - set(old.columns)), passed=True)


def build(update):
    records = []
    for name in ["X21-base", "X21-ft", "B23-base", "B23-ft"]:
        record = dict(run_label=name, checks={})
        for filename, keys in [
            ("train_test_scores.csv", ["dataset_id", "feature_key", "fold", "model_id", "split"]),
            ("cv_metrics.csv", ["dataset_id", "feature_key", "scope", "fold", "model_id"]),
        ]:
            record["checks"][filename] = compare(update / "train_test" / name / filename,
                                                   update / "tt_all" / name / filename, keys)
        records.append(record)
    result = dict(**runtime_record(), passed=True, scope="Repeated Tr-24 fits only; four expanded SBI runs",
                  numeric_tolerance=dict(rtol=1e-10, atol=1e-12), runs=records,
                  interpretation="Training/test scores, prediction/scaling conventions, formulas and all shared metrics reproduce within stated tolerance. Added metadata columns are listed separately.")
    destination = update / "source_validation/repeated_tr24_score_validation.json"
    atomic_write_json(destination, result)
    print(f"Passed repeated Tr-24 score checks for {len(records)} runs; saved {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", type=Path, required=True)
    build(parser.parse_args().update.resolve())
