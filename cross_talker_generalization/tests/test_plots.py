import unittest

import pandas as pd

from ctg.plots import _benjamini_hochberg


class PlotSourceTests(unittest.TestCase):
    def test_benjamini_hochberg_is_monotone_in_rank(self):
        values = pd.Series([0.01, 0.04, 0.03, float("nan")])
        adjusted = _benjamini_hochberg(values)
        self.assertAlmostEqual(adjusted.iloc[0], 0.03)
        self.assertAlmostEqual(adjusted.iloc[1], 0.04)
        self.assertAlmostEqual(adjusted.iloc[2], 0.04)
        self.assertTrue(pd.isna(adjusted.iloc[3]))


if __name__ == "__main__":
    unittest.main()
