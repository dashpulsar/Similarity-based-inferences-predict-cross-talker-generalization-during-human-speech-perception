"""Read saved model inputs, calculate descriptive contrasts, and export deck data.

No GLMM fitting or parameter selection. Run from the repository root.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "cross_talker_generalization"
sources = {}

def read(path):
    sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)

historical = read(PROJECT / "analysis_update_2026-09-06/z_value_review/tables/historical_sbi_fold_z.csv")
labels = ["MFCC39", "STRF24", "HuBERT base Tr-24", "HuBERT FT Tr-24"]
keys = [("mfcc39", "mfcc39"), ("strf24_legacy", "strf24_legacy"), ("base", "tr_24"), ("ft", "tr_24")]
z = {}
for dataset in ["AN19", "X21"]:
    z[dataset] = []
    for source, layer in keys:
        rows = historical.loc[historical.dataset_id.eq(dataset) & historical.source.eq(source) & historical.layer.eq(layer)]
        assert len(rows) == 3 and rows.fold.nunique() == 3
        z[dataset].append(float(rows.z_value.mean()))

diagnostics = []
for dataset in ["AN19", "X21"]:
    reference = None
    for store in ["acoustic", "hubert_base_tsne", "hubert_ft_tsne"]:
        run = f"{dataset}-{dataset}_{store}-confirmatory-{'full' if store == 'acoustic' else 'tr24'}-20260906"
        frame = read(PROJECT / "artifacts/derived" / f"{run}-model-input.csv")
        for feature, group in frame.loc[frame.predictor_status.eq("available")].groupby("feature_key"):
            cols = ["participant_id", "fold", "condition_id", "analysis_item_id", "response_correct", "response_incorrect"]
            identity = group[cols].sort_values(cols).reset_index(drop=True)
            if reference is None:
                reference = identity
            else:
                pd.testing.assert_frame_equal(reference, identity)
            assert (group.response_correct + group.response_incorrect).eq(1).all()
            subsets = ["all"] if dataset == "AN19" else ["all", "no_talker_specific", "exposure_only_no_talker_specific"]
            for subset in subsets:
                selected = group.copy()
                if subset != "all":
                    selected = selected.loc[~selected.condition_id.eq("X21.Talker_specific")]
                if subset.startswith("exposure_only"):
                    selected = selected.loc[~selected.condition_id.eq("X21.Control")]
                cells = selected.groupby(["condition_id", "analysis_item_id", "response_expected"], as_index=False).agg(
                    distance=("raw_distance", "mean"), accuracy=("response_correct", "mean"), n=("response_correct", "size"))
                record = dict(dataset=dataset, store=store, feature=feature, subset=subset,
                              n_responses=len(selected), n_cells=len(cells), n_words=int(cells.response_expected.nunique()))
                for name, grouping in [("raw", None), ("condition", "condition_id"), ("word", "response_expected"), ("recording", "analysis_item_id")]:
                    x, y = -cells.distance, cells.accuracy
                    if grouping:
                        x = x - x.groupby(cells[grouping]).transform("mean")
                        y = y - y.groupby(cells[grouping]).transform("mean")
                    record[f"r_{name}"] = float(x.corr(y))
                design = np.column_stack([np.ones(len(cells)), pd.get_dummies(cells[["response_expected", "condition_id"]], drop_first=True).to_numpy(float)])
                values = np.column_stack([-cells.distance, cells.accuracy])
                residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
                record["r_word_and_condition"] = float(np.corrcoef(residual.T)[0, 1])
                diagnostics.append(record)

output = {"labels": labels, "historical_z": z, "descriptive": diagnostics, "source_sha256": sources,
          "method": "Equal-weight condition-by-recording cell correlations. Predictor is negative raw distance. Within-word centering subtracts the lexical word mean from both predictor and accuracy. Word+condition residuals use simultaneous ordinary least squares projection as a descriptive check, not a GLMM."}
Path(__file__).with_name("evidence.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps({"historical_z": z, "an19": [r for r in diagnostics if r['dataset'] == 'AN19']}, indent=2))
