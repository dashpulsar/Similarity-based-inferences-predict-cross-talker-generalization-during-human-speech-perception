"""Check the generated figure data without refitting models or changing inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cross_talker_generalization/analysis/model_comparison/reference_checks/collaborator_report"


def main():
    provenance = json.loads((OUT / "provenance.json").read_text())
    for relative, expected in provenance["outputs"].items():
        assert hashlib.sha256((OUT / relative).read_bytes()).hexdigest() == expected, relative
    matrix = pd.read_csv(OUT / "tables/figure_2a_36_l2_talker_matrix.csv", index_col=0).to_numpy()
    assert matrix.shape == (36, 36)
    assert np.allclose(matrix, matrix.T, equal_nan=True)
    assert np.isnan(np.diag(matrix)).all()
    off_diag = matrix[~np.eye(36, dtype=bool)]
    assert np.isfinite(off_diag).all() and ((off_diag > 0) & (off_diag <= 1)).all()
    language = pd.read_csv(OUT / "tables/figure_1c_talker_language_map.csv")
    assert language.language.nunique() == 17 and len(language) == 57
    inventories = pd.read_csv(OUT / "tables/figure_1d_inventory_similarity.csv")
    assert len(inventories) == 16 and inventories.status.eq("computed").all()
    assert (inventories.min_jaccard <= inventories.mean_jaccard).all()
    assert (inventories.mean_jaccard <= inventories.max_jaccard).all()
    for dataset, n, mapped, refs in (("an19",1920,1880,6), ("x21",4117,4117,5)):
        trials = pd.read_csv(OUT / f"tables/figure_2c_{dataset}_control_trials.csv")
        valid = trials.loc[trials.mapping_status.eq("available")]
        assert len(trials) == n and len(valid) == mapped
        assert valid.n_english_speakers.eq(refs).all()
        assert valid.similarity.between(0, 1).all()
        assert np.allclose(valid.similarity,
                           np.exp(-valid.mean_reference_dtw/valid.distance_scale_median_unique_targets))
        if dataset == "an19":
            assert set(trials.loc[trials.mapping_status.eq("unmapped"), "response_expected"].str.lower()) == {"wave"}
    models = pd.read_csv(OUT / "tables/figure_2c_descriptive_models.csv")
    assert len(models) == 3 and models.bootstrap_successes.eq(1000).all()
    for name in ("binned_accuracy", "logistic_curves"):
        frame = pd.read_csv(OUT / f"tables/figure_2c_{name}.csv")
        assert frame.ci_low.between(0, 1).all() and frame.ci_high.between(0, 1).all()
        assert (frame.ci_low <= frame.ci_high).all()
    naming = pd.read_csv(OUT / "tables/an19_hw74_naming_audit.csv")
    assert len(naming) == 42 and naming.real_audio_available.all() and naming.tr24_feature_present.all()
    print("PASS: output hashes, symmetric 36-talker matrix, language/inventory coverage, "
          "control mapping, similarity reconstruction, bootstrap counts/intervals and HW74 audit.")


if __name__ == "__main__":
    main()
