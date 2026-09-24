"""Numerical checks for exact full-baseline coordinate slicing."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_an19_matched_acoustic_audit.py"
spec = importlib.util.spec_from_file_location("matched_acoustic_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


class MatchedAcousticTests(unittest.TestCase):
    def test_standardize_then_slice_matches_slice_then_standardize(self):
        rng = np.random.default_rng(3)
        frames = rng.normal(size=(15, 39))
        mean, scale = rng.normal(size=39), np.exp(rng.normal(size=39))
        indices = [0, 4, 12, 27]
        selected_mean, selected_scale = audit.derive_component_moments(mean, scale, indices)
        np.testing.assert_array_equal(
            ((frames - mean) / scale)[:, indices],
            (frames[:, indices] - selected_mean) / selected_scale,
        )


    def test_reject_invalid_indices_and_scales(self):
        with self.assertRaisesRegex(ValueError, "indices"):
            audit.derive_component_moments(np.zeros(2), np.ones(2), [2])
        with self.assertRaisesRegex(ValueError, "positive"):
            audit.derive_component_moments(np.zeros(2), np.zeros(2), [0])


    def test_fold_interval_is_deterministic_and_contains_mean(self):
        low, high = audit._exact_fold_interval([1., 2., 3.])
        self.assertTrue(low <= 2 <= high)
        np.testing.assert_array_equal([low, high], audit._exact_fold_interval([1., 2., 3.]))


if __name__ == "__main__":
    unittest.main()
