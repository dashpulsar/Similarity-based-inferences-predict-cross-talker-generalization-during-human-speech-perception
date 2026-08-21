from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .provenance import atomic_write_csv, atomic_write_json, runtime_record, sha256_file


REGISTERED_LAYER_ORDER = (
    "cnn_2", "cnn_3", "cnn_4", "cnn_5", "cnn_6", "tr_0", "tr_2", "tr_4",
    "tr_6", "tr_8", "tr_10", "tr_12", "tr_14", "tr_16", "tr_18", "tr_20",
    "tr_22", "tr_24",
)
FORBIDDEN_OBSERVATION_COLUMNS = {
    "participant_id", "fold", "response_correct", "response_incorrect"
}


def _feature_order(values: pd.Series) -> list[str]:
    present = set(values.astype(str))
    registered = [value for value in REGISTERED_LAYER_ORDER if value in present]
    return registered + sorted(present.difference(registered))


def plot_distance_correlations(
    input_path: str | Path,
    output_prefix: str | Path,
    *,
    observation_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create a listwise-complete layer-by-layer raw-distance correlation matrix."""

    input_path = Path(input_path).resolve()
    data = pd.read_csv(input_path)
    if "raw_distance" not in data or "feature_key" not in data:
        raise ValueError("input requires raw_distance and feature_key")
    if FORBIDDEN_OBSERVATION_COLUMNS.intersection(data.columns):
        raise ValueError("use physical-pair/cell distances, not participant-replicated model rows")
    if observation_columns is None:
        if "pair_id" in data:
            observation_columns = ["pair_id"]
        elif "cell_id" in data:
            observation_columns = ["cell_id"]
        else:
            raise ValueError("declare observation columns when pair_id/cell_id is absent")
    missing = set(observation_columns).difference(data.columns)
    if missing:
        raise ValueError(f"observation columns absent from input: {sorted(missing)}")
    if FORBIDDEN_OBSERVATION_COLUMNS.intersection(observation_columns):
        raise ValueError("participant/fold/response columns cannot identify acoustic observations")
    if "predictor_status" in data:
        data = data.loc[data["predictor_status"].eq("available")].copy()
    data["raw_distance"] = pd.to_numeric(data["raw_distance"], errors="coerce")
    if (data["raw_distance"].dropna() < 0).any():
        raise ValueError("raw distances must be non-negative")
    duplicate_key = observation_columns + ["feature_key"]
    if data.duplicated(duplicate_key).any():
        examples = data.loc[data.duplicated(duplicate_key, keep=False), duplicate_key].head()
        raise ValueError(f"observation-feature keys are not unique:\n{examples}")

    data["observation_id"] = data[observation_columns].astype(str).agg("\x1f".join, axis=1)
    wide = data.pivot(index="observation_id", columns="feature_key", values="raw_distance")
    features = _feature_order(data["feature_key"])
    if len(features) < 2:
        raise ValueError("at least two feature layers are required for a correlation matrix")
    wide = wide.reindex(columns=features).dropna(axis=0, how="any")
    if len(wide) < 3:
        raise ValueError("fewer than three listwise-complete acoustic observations")
    if (wide.nunique(axis=0) < 2).any():
        bad = wide.columns[wide.nunique(axis=0) < 2].tolist()
        raise ValueError(f"constant distance feature(s): {bad}")

    rho = wide.corr(method="spearman").to_numpy()
    p_value = np.zeros_like(rho)
    for left_index in range(len(features)):
        for right_index in range(left_index + 1, len(features)):
            result = spearmanr(
                wide.iloc[:, left_index].to_numpy(), wide.iloc[:, right_index].to_numpy()
            )
            p_value[left_index, right_index] = float(result.pvalue)
            p_value[right_index, left_index] = float(result.pvalue)
    matrix = pd.DataFrame(rho, index=features, columns=features)
    long_rows = []
    for left_index, left in enumerate(features):
        for right_index, right in enumerate(features):
            long_rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "rho": float(rho[left_index, right_index]),
                    "p_value": float(p_value[left_index, right_index]),
                    "n_complete": int(len(wide)),
                }
            )
    long = pd.DataFrame(long_rows)

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(str(prefix) + "-long.csv", long)
    atomic_write_csv(str(prefix) + "-matrix.csv", matrix.reset_index(names="feature_key"))
    atomic_write_csv(str(prefix) + "-complete-cases.csv", wide.reset_index())

    sns.set_theme(style="white", context="paper")
    size = max(5.5, 0.42 * len(features) + 2.0)
    figure, axis = plt.subplots(figsize=(size, size))
    sns.heatmap(
        matrix, vmin=-1, vmax=1, center=0, cmap="vlag", square=True,
        cbar_kws={"label": "Spearman rho"}, ax=axis,
    )
    axis.set_title(f"Raw-distance correlations (listwise n={len(wide):,})")
    axis.set_xlabel("")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(str(prefix) + ".png", dpi=300)
    figure.savefig(str(prefix) + ".svg")
    plt.close(figure)

    metadata = {
        **runtime_record(),
        "stage": "raw_distance_correlations",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "observation_columns": observation_columns,
        "distance_column": "raw_distance",
        "correlation": "spearman",
        "missing_policy": "listwise_complete_across_all_features",
        "n_input_rows": int(len(data)),
        "n_complete_observations": int(len(wide)),
        "feature_keys": features,
    }
    atomic_write_json(str(prefix) + ".provenance.json", metadata)
    return long
