"""A declared structural sensitivity must never reuse a different-policy fit."""
from pathlib import Path
import json
import tempfile
import unittest

from ctg.parallel_glmm import OUTPUT_TABLES, _valid_cached_run, fit_glmm_parallel
from ctg.provenance import sha256_file
from ctg.cli import _parser


class PolicyCacheChecks(unittest.TestCase):
    def test_cache_requires_matching_policy_and_variance_table(self):
        with tempfile.TemporaryDirectory() as scratch:
            run = Path(scratch)
            parameters = {"predictor_column": "raw_distance", "direction": -1,
                          "term": "similarity_z", "model_set": "all", "random_policy": "registered"}
            for name in OUTPUT_TABLES:
                (run / name).write_text("example\n", encoding="utf-8")
            provenance = {"status": "complete", "input_sha256": "input", "r_script_sha256": "script",
                          "parameters": parameters,
                          "outputs_sha256": {name: sha256_file(run / name) for name in OUTPUT_TABLES}}
            (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            def valid(params):
                return _valid_cached_run(run, input_hash="input", script_hash="script", parameters=params)
            self.assertTrue(valid(parameters))
            self.assertFalse(valid({**parameters, "random_policy": "participant_item"}))
            self.assertIn("variance_components.csv", OUTPUT_TABLES)
            (run / "variance_components.csv").unlink()
            self.assertFalse(valid(parameters))

    def test_reject_unknown_policy_before_any_fit(self):
        with self.assertRaisesRegex(ValueError, "random_policy"):
            fit_glmm_parallel(input_path="absent.csv", output_dir="unused", jobs=1, random_policy="automatic_test_selection")

    def test_cli_default_and_explicit_policy(self):
        for command in ("fit-glmm", "fit-glmm-parallel"):
            tokens = [command, "--input", "input.csv", "--output", "models"]
            self.assertEqual(_parser().parse_args(tokens).random_policy, "registered")
            self.assertEqual(_parser().parse_args(tokens + ["--random-policy", "participant_item"]).random_policy,
                             "participant_item")


if __name__ == "__main__":
    unittest.main()
