import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from checkpoint_continuation import (
    BLOCKWISE_SCENARIO_PARAMETER_FILES,
    CheckpointContinuationError,
    M110_BLOCKWISE_MATCHED_MODE,
    validate_checkpoint_continuation,
)
try:
    from evaluate import compute_paired_difference_summary
except ModuleNotFoundError:
    compute_paired_difference_summary = None


def base_config(config_dir, lambda_param, chi_eta=0.75, loss_mode=None):
    net_config = {
        "num_time_interval": 2,
        "num_neurons": 512,
        "num_layers": 4,
        "activation": "GELU",
        "init_method": "xavier_uniform",
        "consistency_coef": 10.0,
        "train_seed": 0,
        "sampling_mode": "rollout",
        "proposal_sampler": None,
        "rollout_pool_size": 32768,
        "sampling_seed": 0,
        "uniform_z_radius_mult": 1.0,
        "uniform_q_radius_frac": 0.1,
        "total_inventory_buffer": 20.0,
        "q_low_values": [0.0, 1.0, 5.0],
        "boundary_eps_frac": 0.05,
        "mix_rollout": 1.0,
        "mix_box": 0.0,
        "mix_shortage": 0.0,
        "shortage_q_frac": 0.3,
        "shortage_z_frac": 0.15,
        "lognormal_delta_z": 0.01,
        "lognormal_delta_q": 0.001,
        "lognormal_r_min": 20,
        "initial_state_mode": "paper_70_30",
        "initial_perturb_prob": 0.3,
        "initial_q_perturb_frac": 0.05,
        "initial_z_perturb_frac": 0.0125,
        "initial_perturb_minus_prob": 0.2,
        "initial_perturb_plus_prob": 0.1,
        "initial_perturb_minus_frac": 0.1,
        "initial_perturb_plus_frac": 0.05,
        "initial_dirichlet_alpha": 1.0,
    }
    if loss_mode is not None:
        net_config["loss_mode"] = loss_mode
    return {
        "eqn_config": {
            "lambda_param": lambda_param,
            "alpha": 0.5,
            "num_time_interval": 2,
            "dim": 8,
            "eta_mode": "weekly_time_varying",
            "chi_eta": chi_eta,
            "s_eta": 0.0,
            "fairness_spec": "demand_weighted_alpha_fair_v1",
            "fairness_includes_hf": True,
            "fairness_period_h_factor": False,
            "policy": "inventory_scaled_proportional",
            "config_dir": str(config_dir),
        },
        "net_config": net_config,
    }


class CheckpointContinuationTests(unittest.TestCase):
    def write_run(self, root, name, lambda_param, chi_eta=0.75, loss_mode=None):
        run_dir = Path(root) / name
        params_dir = run_dir / "params"
        logs_dir = run_dir / "logs"
        params_dir.mkdir(parents=True)
        logs_dir.mkdir()
        np.savetxt(params_dir / "p.csv", np.full(7, 0.69), delimiter=",")
        np.savetxt(params_dir / "c.csv", np.linspace(1.0, 0.4, 8), delimiter=",")
        config = base_config(params_dir, lambda_param, chi_eta=chi_eta, loss_mode=loss_mode)
        (run_dir / "config_used.json").write_text(json.dumps(config), encoding="utf-8")
        for interval in range(2):
            (logs_dir / f"y_network{interval}.pth").write_bytes(b"placeholder")
            (logs_dir / f"z_network{interval}.pth").write_bytes(b"placeholder")
            if loss_mode == "martingale_qgrad":
                (logs_dir / f"qgrad_network{interval}.pth").write_bytes(b"placeholder")
        return run_dir

    def test_validate_accepts_lambda_difference_with_matching_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_run(tmp, "source", lambda_param=1.0)
            target = self.write_run(tmp, "target", lambda_param=0.5)
            target_config = json.loads((target / "config_used.json").read_text(encoding="utf-8"))
            meta = validate_checkpoint_continuation(
                source_checkpoint_dir=source / "logs",
                target_run_dir=target,
                target_config=target_config,
                expected_source_lambda=1.0,
            )
            self.assertEqual(meta["source_lambda"], 1.0)
            self.assertEqual(meta["target_lambda"], 0.5)
            self.assertEqual(len(meta["checkpoint_pairs"]), 2)
            self.assertEqual(meta["initialization_mode"], "lambda_homotopy")
            self.assertFalse(meta["optimizer_state_reused"])
            self.assertTrue(meta["same_interval_initialization"])
            self.assertEqual(meta["continuation_target_source"], "target_next_interval")
            self.assertEqual(len(meta["source_checkpoint_hash"]), 64)
            for interval, pair in enumerate(meta["checkpoint_pairs"]):
                self.assertEqual(pair["source_interval"], interval)
                self.assertEqual(pair["target_interval"], interval)
                self.assertEqual(len(pair["y_checkpoint_sha256"]), 64)
                self.assertEqual(len(pair["z_checkpoint_sha256"]), 64)

    def test_validate_preserves_audited_logical_mount_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_root = root / "real_mount"
            real_root.mkdir()
            source = self.write_run(real_root, "source", lambda_param=1.0)
            target = self.write_run(real_root, "target", lambda_param=0.5)
            logical_root = root / "audited_mount"
            logical_root.symlink_to(real_root, target_is_directory=True)
            target_config = json.loads((target / "config_used.json").read_text(encoding="utf-8"))

            meta = validate_checkpoint_continuation(
                source_checkpoint_dir=logical_root / "source" / "logs",
                target_run_dir=logical_root / "target",
                target_config=target_config,
                expected_source_lambda=1.0,
            )

            self.assertEqual(meta["source_run_dir"], str(logical_root / "source"))
            self.assertEqual(meta["source_checkpoint_dir"], str(logical_root / "source" / "logs"))
            self.assertEqual(meta["target_run_dir"], str(logical_root / "target"))
            self.assertTrue(
                all(pair["y_checkpoint"].startswith(str(logical_root)) for pair in meta["checkpoint_pairs"])
            )

    def test_validate_rejects_chi_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_run(tmp, "source", lambda_param=1.0, chi_eta=1.75)
            target = self.write_run(tmp, "target", lambda_param=0.5, chi_eta=0.75)
            target_config = json.loads((target / "config_used.json").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(CheckpointContinuationError, "chi_eta"):
                validate_checkpoint_continuation(
                    source_checkpoint_dir=source / "logs",
                    target_run_dir=target,
                    target_config=target_config,
                    expected_source_lambda=1.0,
                )

    def test_validate_l2_requires_and_records_qgrad_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_run(tmp, "source", lambda_param=1.0, loss_mode="martingale_qgrad")
            target = self.write_run(tmp, "target", lambda_param=0.5, loss_mode="martingale_qgrad")
            target_config = json.loads((target / "config_used.json").read_text(encoding="utf-8"))
            meta = validate_checkpoint_continuation(
                source_checkpoint_dir=source / "logs",
                target_run_dir=target,
                target_config=target_config,
                expected_source_lambda=1.0,
            )
            self.assertEqual(meta["loss_mode"], "martingale_qgrad")
            for pair in meta["checkpoint_pairs"]:
                self.assertIn("qgrad_checkpoint", pair)
                self.assertEqual(len(pair["qgrad_checkpoint_sha256"]), 64)

    def test_validate_rejects_loss_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_run(tmp, "source", lambda_param=1.0, loss_mode="ce_consistency")
            target = self.write_run(tmp, "target", lambda_param=0.5, loss_mode="martingale_qgrad")
            target_config = json.loads((target / "config_used.json").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(CheckpointContinuationError, "loss_mode"):
                validate_checkpoint_continuation(
                    source_checkpoint_dir=source / "logs",
                    target_run_dir=target,
                    target_config=target_config,
                    expected_source_lambda=1.0,
                )

    def write_matched_blockwise_run(self, root, name, *, m, mu_shift=0.0):
        run_dir = Path(root) / name
        params_dir = run_dir / "params"
        logs_dir = run_dir / "logs"
        params_dir.mkdir(parents=True)
        logs_dir.mkdir()
        for index, filename in enumerate(BLOCKWISE_SCENARIO_PARAMETER_FILES):
            values = np.linspace(1.0 + index, 2.0 + index, 8)
            if filename == "mu.csv":
                values = values + mu_shift
            np.savetxt(params_dir / filename, values, delimiter=",")
        config = base_config(params_dir, 0.25, chi_eta=1.75, loss_mode="ce_consistency")
        config["eqn_config"].update(
            {
                "num_time_interval": m,
                "I0": 142.0,
                "K": 7,
                "total_time": 22.0,
                "calibration_version": "test-v1",
                "periodic_demand_mode": "binomial",
            }
        )
        config["net_config"].update(
            {
                "num_time_interval": m,
                "consistency_coef": 30.0,
                "h_init_method": "xavier_uniform",
                "g_init_method": "kaiming_uniform",
                "g_output_bias": True,
            }
        )
        (run_dir / "config_used.json").write_text(json.dumps(config), encoding="utf-8")
        for interval in range(m):
            (logs_dir / f"y_network{interval}.pth").write_bytes(f"y-{interval}".encode())
            (logs_dir / f"z_network{interval}.pth").write_bytes(f"z-{interval}".encode())
        return run_dir

    def test_matched_blockwise_accepts_exact_scenario_and_records_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_matched_blockwise_run(tmp, "source", m=110)
            target = self.write_matched_blockwise_run(tmp, "target", m=1100)
            target_config = json.loads((target / "config_used.json").read_text(encoding="utf-8"))
            meta = validate_checkpoint_continuation(
                source_checkpoint_dir=source / "logs",
                target_run_dir=target,
                target_config=target_config,
                expected_source_lambda=0.25,
                initialization_mode=M110_BLOCKWISE_MATCHED_MODE,
            )
            self.assertEqual(len(meta["checkpoint_pairs"]), 110)
            self.assertEqual(meta["checkpoint_pairs"][0]["target_interval"], 9)
            self.assertEqual(meta["checkpoint_pairs"][-1]["target_interval"], 1099)
            self.assertEqual(meta["blockwise_mapping_mode"], M110_BLOCKWISE_MATCHED_MODE)
            self.assertEqual(set(meta["scenario_parameter_hashes"]), set(BLOCKWISE_SCENARIO_PARAMETER_FILES))

    def test_matched_blockwise_rejects_scenario_parameter_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_matched_blockwise_run(tmp, "source", m=110)
            target = self.write_matched_blockwise_run(tmp, "target", m=1100, mu_shift=0.1)
            target_config = json.loads((target / "config_used.json").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(CheckpointContinuationError, "scenario parameter mu.csv"):
                validate_checkpoint_continuation(
                    source_checkpoint_dir=source / "logs",
                    target_run_dir=target,
                    target_config=target_config,
                    expected_source_lambda=0.25,
                    initialization_mode=M110_BLOCKWISE_MATCHED_MODE,
                )

    def test_matched_blockwise_rejects_scenario_parameter_shape_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_matched_blockwise_run(tmp, "source", m=110)
            target = self.write_matched_blockwise_run(tmp, "target", m=1100)
            mu_path = target / "params" / "mu.csv"
            np.savetxt(mu_path, np.loadtxt(mu_path, delimiter=",").reshape(2, 4), delimiter=",")
            target_config = json.loads((target / "config_used.json").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(CheckpointContinuationError, "mu.csv shape"):
                validate_checkpoint_continuation(
                    source_checkpoint_dir=source / "logs",
                    target_run_dir=target,
                    target_config=target_config,
                    expected_source_lambda=0.25,
                    initialization_mode=M110_BLOCKWISE_MATCHED_MODE,
                )

    def test_paired_difference_includes_scaled_static(self):
        if compute_paired_difference_summary is None:
            self.skipTest("evaluate.py dependencies are unavailable")
        costs = {
            "Deep_Splitting": {"total_objectives": np.array([1.0, 2.0, 3.0])},
            "Static": {"total_objectives": np.array([2.0, 3.0, 4.0])},
            "Scaled_Static": {"total_objectives": np.array([4.0, 5.0, 6.0])},
        }
        df = compute_paired_difference_summary(costs, subtrahend_policy=("Static", "Scaled_Static"))
        self.assertEqual(set(df["Policy_B"]), {"Static", "Scaled_Static"})
        self.assertTrue((df["Mean_Difference"] < 0).all())


if __name__ == "__main__":
    unittest.main()
