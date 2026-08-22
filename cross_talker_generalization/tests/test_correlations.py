import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ctg.correlations import plot_distance_correlations


class CorrelationTests(unittest.TestCase):
    def test_listwise_complete_spearman_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "distances.csv"
            rows = []
            for pair_id, left, right in (
                ("a", 1.0, 2.0), ("b", 2.0, 4.0), ("c", 3.0, 6.0), ("d", 4.0, None)
            ):
                rows.append({"pair_id": pair_id, "feature_key": "cnn_2", "raw_distance": left})
                rows.append({"pair_id": pair_id, "feature_key": "cnn_3", "raw_distance": right})
            pd.DataFrame(rows).to_csv(source, index=False)
            result = plot_distance_correlations(source, root / "correlations")
            cross = result.loc[
                result["feature_left"].eq("cnn_2") & result["feature_right"].eq("cnn_3")
            ].iloc[0]
            self.assertEqual(cross["n_complete"], 3)
            self.assertAlmostEqual(cross["rho"], 1.0)
            self.assertTrue((root / "correlations.png").is_file())
            self.assertTrue((root / "correlations-matrix.csv").is_file())

    def test_rejects_participant_replicated_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "distances.csv"
            pd.DataFrame(
                {
                    "pair_id": ["a", "a"],
                    "feature_key": ["cnn_2", "cnn_3"],
                    "raw_distance": [1.0, 2.0],
                    "participant_id": ["p1", "p1"],
                }
            ).to_csv(source, index=False)
            with self.assertRaises(ValueError):
                plot_distance_correlations(source, Path(temporary) / "bad")


if __name__ == "__main__":
    unittest.main()
