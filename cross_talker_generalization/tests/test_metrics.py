import unittest

import numpy as np

from ctg.metrics import (
    Token,
    compute_variability,
    distance_to_similarity,
    dtw_distance,
)


class DtwTests(unittest.TestCase):
    def test_identical_sequences_have_zero_distance(self):
        sequence = np.asarray([[0.0, 1.0], [2.0, 3.0]])
        result = dtw_distance(sequence, sequence)
        self.assertEqual(result.distance, 0.0)
        self.assertEqual(result.raw_cost, 0.0)
        self.assertEqual(result.path_length, 2)

    def test_normalization_is_explicit(self):
        left = np.asarray([[0.0], [2.0]])
        right = np.asarray([[0.0], [1.0], [2.0]])
        legacy = dtw_distance(left, right, normalization="mean_sequence_length")
        published = dtw_distance(left, right, normalization="path_length")
        self.assertEqual(legacy.raw_cost, published.raw_cost)
        self.assertEqual(legacy.path_length, 3)
        self.assertAlmostEqual(legacy.distance, 1.0 / 2.5)
        self.assertAlmostEqual(published.distance, 1.0 / 3.0)

    def test_similarity_transform(self):
        self.assertAlmostEqual(distance_to_similarity(2.0, 0.5), np.exp(-1.0))
        with self.assertRaises(ValueError):
            distance_to_similarity(-1.0)

    def test_invalid_tau_is_rejected(self):
        sequence = np.ones((2, 2))
        for tau in (0, -1, np.nan, np.inf, True):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                dtw_distance(sequence, sequence, tau=tau)


class VariabilityTests(unittest.TestCase):
    def test_overall_rootless_dispersion(self):
        tokens = [
            Token("a", "x", np.asarray([[0.0], [2.0]])),
            Token("b", "y", np.asarray([[2.0], [4.0]])),
        ]
        self.assertAlmostEqual(compute_variability("overall", tokens, tau=2), 2.0)

    def test_between_type_equal_weights_types(self):
        tokens = [
            Token("a1", "a", np.asarray([[0.0]])),
            Token("a2", "a", np.asarray([[2.0]])),
            Token("b1", "b", np.asarray([[5.0]])),
        ]
        # Type centroids are 1 and 5; their rootless squared dispersion is 4.
        self.assertAlmostEqual(
            compute_variability("between_type_word", tokens, tau=2), 4.0
        )


if __name__ == "__main__":
    unittest.main()
