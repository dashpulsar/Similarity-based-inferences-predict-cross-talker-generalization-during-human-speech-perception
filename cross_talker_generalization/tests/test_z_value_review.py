"""Small numerical checks for the figure summaries, not new model tests."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


script = Path(__file__).resolve().parents[1] / "scripts/build_z_value_review.py"
spec = importlib.util.spec_from_file_location("z_value_review", script)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_fold_bootstrap_matches_all_27_resamples():
    values = [-3., 2., 7.]
    samples = [sum(sample) / 3 for sample in itertools.product(values, repeat=3)]
    mean, low, high = review.fold_ci(values)
    assert mean == 2
    np.testing.assert_allclose([low, high], np.quantile(samples, [.025, .975]))


def test_ceiling_normalization_preserves_sign_and_centers_ceiling():
    ceiling = np.array([10., 20., 30.])
    np.testing.assert_allclose(review.normalized([-4., 0., 30.], ceiling), [-20., 0., 150.])
    assert review.normalized(ceiling, ceiling).mean() == 100


@pytest.mark.parametrize("values", [[1., 2.], [1., 2., 3., 4.], [1., np.nan, 3.]])
def test_invalid_fold_sets_are_not_silently_plotted(values):
    with pytest.raises(ValueError):
        review.fold_ci(values)


def test_nonpositive_ceiling_is_rejected():
    with pytest.raises(ValueError):
        review.normalized([1., 2., 3.], [-1., 0., 1.])


def test_duplicate_fold_is_rejected():
    frame = pd.DataFrame({"layer": ["tr_24"] * 3, "fold": [0, 0, 2], "z_value": [1., 2., 3.]})
    with pytest.raises(ValueError):
        review.check_folds(frame, ["layer"])
