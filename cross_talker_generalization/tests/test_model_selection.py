import unittest

import pandas as pd

from ctg.report_core import _select_by_predictor_oof


class PredictorSelectionTests(unittest.TestCase):
    def test_selects_smallest_heldout_predictor_only_log_loss(self):
        candidates = pd.DataFrame(
            {
                "feature_key": ["tr_0", "tr_2", "tr_4"],
                "predictor_oof_total_log_loss": [102.0, 98.0, 101.0],
                "predictor_oof_total_trials": [400, 400, 400],
            }
        )
        self.assertEqual(_select_by_predictor_oof(candidates)["feature_key"], "tr_2")

    def test_rejects_candidate_scores_from_different_heldout_samples(self):
        candidates = pd.DataFrame(
            {
                "feature_key": ["tr_0", "tr_2"],
                "predictor_oof_total_log_loss": [90.0, 80.0],
                "predictor_oof_total_trials": [400, 350],
            }
        )
        with self.assertRaises(ValueError):
            _select_by_predictor_oof(candidates)


if __name__ == "__main__":
    unittest.main()
