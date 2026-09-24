"""Scientific arithmetic and grouping checks for paired train/test loss ratios."""
import importlib.util
from itertools import product
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

script = Path(__file__).resolve().parents[1] / "scripts/build_train_test_ratio_figures.py"
spec = importlib.util.spec_from_file_location("ratio_figures", script)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class RatioTests(unittest.TestCase):
    def setUp(self):
        self.scores = pd.DataFrame([
            dict(dataset_id="AN19", feature_key="tr_24", fold=fold, model_id=model,
                 split=split, total_trials=30 if split == "train" else 15,
                 mean_log_loss=loss, total_log_loss=loss * (30 if split == "train" else 15),
                 score_status="ok", fit_ok=True, singular=False, convergence="ok",
                 train_predictor_mean=fold + 1., train_predictor_sd=2.,
                 random_structure="participant_item", formula="y ~ x + (1|p) + (1|i)",
                 prediction_convention=review.PREDICTION, likelihood_convention=review.LIKELIHOOD,
                 predictor_column="raw_distance", predictor_term="similarity_z", predictor_direction=-1)
            for fold, model in product(range(3), review.MODELS)
            for split, loss in [("train", .4), ("test", .6)]
        ])

    def test_all_four_models_have_paired_response_normalized_ratios(self):
        paired = review.paired_ratios(self.scores)
        self.assertEqual(len(paired), 12)
        self.assertTrue(paired.ratio_status.eq("ok").all())
        np.testing.assert_allclose(paired.ratio, 1.5)
        self.assertTrue(paired.train_total_trials.eq(30).all())

    def test_better_training_gives_ratio_above_one(self):
        paired = review.paired_ratios(self.scores)
        self.assertTrue(paired.ratio.gt(1).all())

    def test_zero_and_nonfinite_denominators_are_retained_as_invalid(self):
        for value in [0., np.nan, np.inf]:
            scores = self.scores.copy()
            scores.loc[0, "mean_log_loss"] = value
            scores.loc[0, "total_log_loss"] = value * 30
            paired = review.paired_ratios(scores)
            invalid = paired.loc[paired.ratio_status.eq("invalid")]
            self.assertEqual(len(paired), 12)
            self.assertEqual(len(invalid), 1)
            self.assertIn("nonpositive_or_nonfinite_denominator", invalid.invalid_reason.iloc[0])
            self.assertTrue(invalid.ratio.isna().all())

    def test_zero_test_loss_is_valid(self):
        self.scores.loc[1, ["mean_log_loss", "total_log_loss"]] = 0.
        self.assertEqual(review.paired_ratios(self.scores).ratio.iloc[0], 0.)

    def test_duplicate_scores_raise(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            review.paired_ratios(pd.concat([self.scores, self.scores.iloc[:1]]))

    def test_missing_split_and_missing_fold_remain_visible(self):
        paired = review.paired_ratios(self.scores.iloc[1:])
        self.assertIn("missing_train_score", paired.invalid_reason.iloc[0])
        paired = review.paired_ratios(self.scores.loc[self.scores.fold.ne(2)])
        self.assertEqual(len(paired), 12)
        self.assertEqual(paired.ratio_status.eq("invalid").sum(), 4)

    def test_different_scaling_and_conventions_invalidate_pair(self):
        for field, value in [("train_predictor_sd", 999.), ("random_structure", "other"),
                             ("predictor_direction", 1), ("prediction_convention", "conditional"),
                             ("likelihood_convention", "include_binomial_constant")]:
            scores = self.scores.copy()
            scores.loc[1, field] = value
            paired = review.paired_ratios(scores)
            self.assertIn(f"train_test_mismatch:{field}", paired.invalid_reason.iloc[0])

    def test_response_count_arithmetic_is_checked(self):
        self.scores.loc[1, "total_trials"] = 1
        self.assertIn("response_normalization_mismatch", review.paired_ratios(self.scores).invalid_reason.iloc[0])

    def test_fit_warnings_do_not_silently_remove_finite_scores(self):
        self.scores.loc[0, "singular"] = True
        self.scores.loc[1, "convergence"] = "gradient"
        row = review.paired_ratios(self.scores).iloc[0]
        self.assertEqual(row.ratio_status, "ok")
        self.assertTrue(row.has_fit_warning)
        self.assertIn("singular_fit", row.fit_warning)
        self.assertIn("convergence:gradient", row.fit_warning)

    def test_failed_fit_cannot_get_valid_ratio(self):
        self.scores.loc[0, "fit_ok"] = False
        self.assertIn("train_fit_failed", review.paired_ratios(self.scores).invalid_reason.iloc[0])

    def test_exact_bootstrap_is_over_paired_ratios_not_ratio_of_means(self):
        train = np.array([.1, .2, .4])
        test = np.array([.2, .2, .2])
        mean, low, high = review.exact_three_fold_ci(test / train)
        brute = [np.mean((test / train)[list(indices)]) for indices in product(range(3), repeat=3)]
        np.testing.assert_allclose([low, high], np.quantile(brute, [.025, .975]))
        self.assertAlmostEqual(mean, (2 + 1 + .5) / 3)
        self.assertNotAlmostEqual(mean, test.mean() / train.mean())
        with self.assertRaises(ValueError):
            review.exact_three_fold_ci([1., 2.])

    def test_incomplete_fold_set_has_no_three_fold_ci(self):
        paired = review.paired_ratios(self.scores.iloc[1:])
        paired = review.add_metadata(paired, dict(run_label="AN19-base"))
        summary = review.summarize(paired)
        bad = summary.loc[summary.model_id.eq("M_null")].iloc[0]
        self.assertEqual(bad.n_valid_folds, 2)
        self.assertTrue(np.isnan(bad.mean_fold_ratio))
        self.assertTrue(np.isnan(bad.ci95_low))

    def test_hve_measure_and_participant_stratum_are_explicit(self):
        paired = review.paired_ratios(self.scores)
        paired["feature_key"] = "tr_24::overall_order_sensitive"
        with self.assertRaisesRegex(ValueError, "stratum"):
            review.add_metadata(paired, dict(run_label="HVE", family="HVE"))
        result = review.add_metadata(paired, dict(run_label="HVE", family="HVE", stratum="97_with_order"))
        self.assertTrue(result.layer.eq("tr_24").all())
        self.assertTrue(result.measure.eq("overall_order_sensitive").all())
        self.assertTrue(result.stratum.eq("97_with_order").all())
        with self.assertRaisesRegex(ValueError, "measure differs"):
            review.add_metadata(paired, dict(run_label="HVE", family="HVE", stratum="97", measure="other"))

    def test_baselines_sort_before_layers(self):
        self.assertEqual(sorted(["tr_24", "cnn_2", "strf24_legacy", "mfcc39"], key=review.feature_order),
                         ["mfcc39", "strf24_legacy", "cnn_2", "tr_24"])


if __name__ == "__main__":
    unittest.main()
