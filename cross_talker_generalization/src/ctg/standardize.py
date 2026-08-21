from __future__ import annotations

from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
from threadpoolctl import threadpool_limits

from .config import FeatureStoreSpec
from .features import FeatureStore
from .provenance import atomic_write_json, runtime_record, sha256_file


def _moments_for_key(spec: FeatureStoreSpec, feature_key: str, destination: Path):
    count = 0
    total = None
    total_sq = None
    with threadpool_limits(limits=1), FeatureStore(spec) as store:
        for sequence in store.iter_sequences(feature_key):
            if total is None:
                total = np.zeros(sequence.shape[1], dtype=np.float64)
                total_sq = np.zeros(sequence.shape[1], dtype=np.float64)
            if sequence.shape[1] != total.shape[0]:
                raise ValueError(f"dimension changed inside {spec.store_id}/{feature_key}")
            total += sequence.sum(axis=0)
            total_sq += np.square(sequence).sum(axis=0)
            count += sequence.shape[0]
    if count == 0 or total is None or total_sq is None:
        raise ValueError(f"no frames in {spec.store_id}/{feature_key}")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    path = destination / f"{spec.store_id}__{feature_key}.npz"
    np.savez(path, mean=mean, scale=scale, frame_count=np.asarray(count, dtype=np.int64))
    return {
        "feature_key": feature_key,
        "path": str(path),
        "frame_count": int(count),
        "dimension": int(mean.shape[0]),
        "sha256": sha256_file(path),
    }


def fit_standardizers(
    spec: FeatureStoreSpec,
    feature_keys: list[str],
    output_dir: str | Path,
    jobs: int,
):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = Parallel(n_jobs=min(max(1, jobs), len(feature_keys)), backend="loky")(
        delayed(_moments_for_key)(spec, key, destination) for key in feature_keys
    )
    provenance = {
        **runtime_record(),
        "stage": "fit_global_standardizers",
        "store_id": spec.store_id,
        "store_path": str(spec.path),
        "scaling_scope": "all_registered_frames_in_feature_store",
        "ddof": 0,
        "records": records,
    }
    atomic_write_json(destination / f"{spec.store_id}__provenance.json", provenance)
    return records


def load_standardizer(directory: str | Path, store_id: str, feature_key: str):
    path = Path(directory) / f"{store_id}__{feature_key}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing global standardizer {path}; run ctg fit-standardizers first"
        )
    bundle = np.load(path)
    return np.asarray(bundle["mean"], dtype=np.float64), np.asarray(bundle["scale"], dtype=np.float64)
