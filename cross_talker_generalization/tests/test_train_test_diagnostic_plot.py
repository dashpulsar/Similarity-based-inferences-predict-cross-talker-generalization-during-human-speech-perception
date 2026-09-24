"""Numerical checks for summaries of the newly paired prediction scores."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

script = Path(__file__).resolve().parents[1] / "scripts/build_train_test_diagnostics.py"
spec = importlib.util.spec_from_file_location("train_test_diagnostics", script)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class TrainTestSummaryTests(unittest.TestCase):
    def setUp(self):
        self.scores = pd.DataFrame([
            dict(run_label="AN19-base", dataset_id="AN19", feature_key="tr_24", fold=fold,
                 model_id="M_predictor", split=split, total_trials=10, total_log_loss=4.,
                 mean_log_loss=.4, mean_log_likelihood=-.4,
                 prediction_convention="fixed_effects_only_re_form_NA", score_status="ok",
                 train_predictor_mean=fold + 2., train_predictor_sd=1.,
                 random_structure="item_and_participant", formula="y ~ x + (1|p) + (1|i)")
            for fold in range(3) for split in ["train", "test"]
        ])

    def test_matched_pairs_are_accepted(self):
        review.validate_scores(self.scores)

    def test_duplicate_or_missing_split_is_rejected(self):
        for bad in [pd.concat([self.scores, self.scores.iloc[:1]]), self.scores.iloc[1:]]:
            with self.assertRaises(ValueError):
                review.validate_scores(bad)

    def test_test_specific_scaling_is_rejected(self):
        self.scores.loc[1, "train_predictor_mean"] = 999.
        with self.assertRaises(ValueError):
            review.validate_scores(self.scores)

    def test_response_normalization_is_checked(self):
        self.scores["total_trials"] = 1
        with self.assertRaises(AssertionError):
            review.validate_scores(self.scores)

    def test_three_fold_interval_is_reproducible(self):
        np.testing.assert_allclose(review.three_fold_ci([1., 2., 3.]),
                                   [2., 1.2166666666666666, 2.783333333333333])
        with self.assertRaises(ValueError):
            review.three_fold_ci([1., 2.])


if __name__ == "__main__":
    unittest.main()
