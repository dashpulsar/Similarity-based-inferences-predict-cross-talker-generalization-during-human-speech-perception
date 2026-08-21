import tempfile
import unittest
from pathlib import Path

from ctg.config import load_project
from ctg.exposure import build_exposure_tables
from ctg.pairs import build_pair_tables


PROJECT = Path(__file__).resolve().parents[1] / "configs" / "project.json"


class RealProjectContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = load_project(PROJECT)

    def test_registered_paths_exist(self):
        for spec in self.project.datasets.values():
            self.assertTrue(spec.behavior.is_file())
            self.assertTrue(spec.manifest.is_file())
        for spec in self.project.feature_stores.values():
            self.assertTrue(spec.path.is_file())

    def test_pair_inventories(self):
        expected = {
            "AN19": (5459, 1104),
            "X21": (4532, 3296),
            "B23": (240, 540),
        }
        with tempfile.TemporaryDirectory() as temporary:
            for dataset_id, (pair_count, cell_count) in expected.items():
                with self.subTest(dataset=dataset_id):
                    pairs, cells, _ = build_pair_tables(
                        self.project.dataset(dataset_id), Path(temporary) / dataset_id
                    )
                    self.assertEqual(len(pairs), pair_count)
                    self.assertEqual(cells["cell_id"].nunique(), cell_count)

    def test_actual_exposure_inventories_and_blocked_cells(self):
        expected = {
            "AN19": (864, 7, 160, {"available": 6, "no_exposure": 1}),
            "X21": (41147, 22, 320, {"available": 22}),
            "B23": (
                5601,
                9,
                195,
                {"available": 4, "blocked": 4, "no_exposure": 1},
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            for dataset_id, (task_count, pool_count, participant_count, statuses) in expected.items():
                with self.subTest(dataset=dataset_id):
                    tasks, pools, participants, provenance = build_exposure_tables(
                        self.project.dataset(dataset_id), Path(temporary) / dataset_id
                    )
                    self.assertEqual(len(tasks), task_count)
                    self.assertEqual(len(pools), pool_count)
                    self.assertEqual(len(participants), participant_count)
                    self.assertEqual(provenance["pool_status_counts"], statuses)


if __name__ == "__main__":
    unittest.main()
