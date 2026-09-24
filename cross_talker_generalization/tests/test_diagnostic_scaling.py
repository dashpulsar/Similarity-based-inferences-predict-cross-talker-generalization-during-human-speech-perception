"""Keep acoustic subsets on the coordinate scale of the full baselines."""
import json
from pathlib import Path
import unittest


class AcousticScalingTests(unittest.TestCase):
    def test_acoustic_subsets_share_the_full_baseline_scaling(self):
        config = Path(__file__).resolve().parents[1] / "configs"
        profiles = [config / "confirmatory.json", *sorted((config / "sensitivities").glob("*.json"))]
        for path in profiles:
            with self.subTest(profile=path.name):
                settings = json.loads(path.read_text(encoding="utf-8"))["coordinate_scaling"]
                self.assertEqual(settings["acoustic"], "global_z")
                self.assertEqual(settings["acoustic_subset"], settings["acoustic"])


if __name__ == "__main__":
    unittest.main()
