import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluation_bank as bank


class EvaluationBankTests(unittest.TestCase):
    def test_round_trip_and_fail_closed_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bank"
            fine = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2) / 10.0
            periodic = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
            manifest_path, digest = bank.save_frozen_evaluation_bank(
                root,
                bank_id="unit",
                scenario="fangonda_base",
                m=4,
                seed=17,
                periodic_demand_mode="binomial",
                fine_grid_brownian_increments=fine,
                periodic_review_demands=periodic,
                inputs={"config_sha256": "a" * 64},
            )
            loaded = bank.load_frozen_evaluation_bank(
                manifest_path,
                digest,
                expected_scenario="fangonda_base",
                expected_m=4,
                expected_num_paths=3,
                expected_seed=17,
                expected_periodic_demand_mode="binomial",
            )
            np.testing.assert_array_equal(loaded["fine_grid_brownian_increments"], fine)
            np.testing.assert_array_equal(loaded["periodic_review_demands"], periodic)
            with self.assertRaisesRegex(bank.EvaluationBankError, "manifest SHA-256 mismatch"):
                bank.load_bank_manifest(manifest_path, "0" * 64)

    def test_array_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bank"
            manifest_path, digest = bank.save_frozen_evaluation_bank(
                root,
                bank_id="unit",
                scenario="s",
                m=2,
                seed=1,
                periodic_demand_mode="binomial",
                fine_grid_brownian_increments=np.zeros((2, 2, 1), dtype=np.float32),
                periodic_review_demands=np.zeros((1, 2, 1), dtype=np.float32),
                inputs={},
            )
            payload = json.loads(manifest_path.read_text())
            fine_path = root / payload["arrays"]["fine_grid_brownian_increments"]["path"]
            values = np.load(fine_path, allow_pickle=False)
            values[0, 0, 0] = 1.0
            with fine_path.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
            with self.assertRaisesRegex(bank.EvaluationBankError, "file hash mismatch"):
                bank.load_bank_manifest(manifest_path, digest)

    def test_refuses_overwrite_and_negative_periodic_demand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bank"
            with self.assertRaisesRegex(bank.EvaluationBankError, "negative demand"):
                bank.save_frozen_evaluation_bank(
                    root,
                    bank_id="bad",
                    scenario="s",
                    m=1,
                    seed=1,
                    periodic_demand_mode="binomial",
                    fine_grid_brownian_increments=np.zeros((1, 1, 1), dtype=np.float32),
                    periodic_review_demands=-np.ones((1, 1, 1), dtype=np.float32),
                    inputs={},
                )


if __name__ == "__main__":
    unittest.main()
