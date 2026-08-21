import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from ctg.config import FeatureStoreSpec
from ctg.features import FeatureStore, slice_by_time


class FeatureStoreTests(unittest.TestCase):
    def test_layer_first_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reduced.h5"
            with h5py.File(path, "w") as h5:
                h5.create_dataset("tr_24/spk/unit", data=np.arange(12).reshape(4, 3))
            spec = FeatureStoreSpec("test", "X21", path, "hubert_tsne", "base", 1)
            with FeatureStore(spec) as store:
                self.assertEqual(store.feature_keys(), ("tr_24",))
                self.assertEqual(store.unit_count("tr_24"), 1)
                np.testing.assert_array_equal(
                    store.read("spk", "unit", "tr_24"), np.arange(12).reshape(4, 3)
                )

    def test_speaker_first_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full.h5"
            with h5py.File(path, "w") as h5:
                h5.create_dataset("spk/unit/cnn_2", data=np.ones((2, 5)))
                h5.create_group("__staging__")
            spec = FeatureStoreSpec("test", "AN19", path, "hubert_full", "base", 1)
            with FeatureStore(spec) as store:
                self.assertEqual(store.feature_keys(), ("cnn_2",))
                self.assertEqual(store.unit_count(), 1)

    def test_time_slice_uses_manifest_relative_bounds(self):
        sequence = np.arange(20).reshape(10, 2)
        cropped = slice_by_time(sequence, 0.2, 0.6, 1.0)
        np.testing.assert_array_equal(cropped, sequence[2:6])


if __name__ == "__main__":
    unittest.main()
