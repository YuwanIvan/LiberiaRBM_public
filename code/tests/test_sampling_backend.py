import csv
import os
import tempfile
import types
import unittest
from types import SimpleNamespace

import numpy as np
import torch

import equation as eqn
from solver import DSPL_solver


def to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: to_namespace(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(value) for value in obj]
    return obj


class SamplingBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = self.temp_dir.name
        self._write_test_parameters()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_test_parameters(self):
        np.savetxt(
            os.path.join(self.config_dir, "mu.csv"),
            np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32),
            delimiter=",",
        )
        np.savetxt(
            os.path.join(self.config_dir, "gamma.csv"),
            np.zeros((2, 2), dtype=np.float32),
            delimiter=",",
        )
        np.savetxt(
            os.path.join(self.config_dir, "p.csv"),
            np.array([0.25], dtype=np.float32),
            delimiter=",",
        )
        np.savetxt(
            os.path.join(self.config_dir, "c.csv"),
            np.array([1.0, 1.5], dtype=np.float32),
            delimiter=",",
        )
        np.savetxt(
            os.path.join(self.config_dir, "d.csv"),
            np.array([1.0, 2.5], dtype=np.float32),
            delimiter=",",
        )
        np.savetxt(
            os.path.join(self.config_dir, "h.csv"),
            np.array([0.5, 0.75], dtype=np.float32),
            delimiter=",",
        )

    def _build_solver(self, sampling_mode="box", **net_overrides):
        config_dict = {
            "eqn_config": {
                "eqn_name": "LiberiaHJB",
                "config_dir": self.config_dir,
                "total_time": 1.0,
                "dim": 2,
                "num_time_interval": 2,
                "K": 1,
                "lambda_param": 0.1,
                "alpha": 0.5,
                "I0": 10.0,
                "policy": "static_proportional",
                "mu_file": "mu.csv",
                "gamma_file": "gamma.csv",
                "p_file": "p.csv",
                "c_file": "c.csv",
                "d_file": "d.csv",
                "h_file": "h.csv",
            },
            "net_config": {
                "final_learning_rate": 1e-3,
                "inner_learning_rate": 1e-3,
                "final_iterations": 1,
                "inner_iterations": 1,
                "num_time_interval": 2,
                "print_interval": 1,
                "num_neurons": 4,
                "gamma": 0.5,
                "a_lowbound": False,
                "final_milestones": [1],
                "inner_milestones": [1],
                "early_stopping": False,
                "grad_clip": False,
                "grad_norm": 1.0,
                "patience": 2,
                "min_delta": 0.0,
                "activation": "ReLU",
                "num_layers": 2,
                "slope": 0.2,
                "prev_init": False,
                "lambd_const": 0.1,
                "valid_size": 4,
                "batch_size": 4,
                "loss_type": 2,
                "sampling_mode": sampling_mode,
                "rollout_pool_size": 8,
                "sampling_seed": 7,
                "mix_rollout": 0.5,
                "mix_box": 0.25,
                "mix_shortage": 0.25,
                "shortage_q_frac": 0.3,
                "shortage_z_frac": 0.15,
                "consistency_coef": 0.0,
            },
        }
        config_dict["net_config"].update(net_overrides)
        config = to_namespace(config_dict)
        bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
        return DSPL_solver(config, bsde)

    def _assert_sample_contract(self, solver, mode):
        Y_n, Y_n1, a_ref, L_inc, W_inc = solver.sample(5, 1, mode=mode)
        expected_dtype = solver.bsde.mu.dtype
        expected_device = solver.device

        self.assertEqual(Y_n.shape, (5, solver.state_dim))
        self.assertEqual(Y_n1.shape, (5, solver.state_dim))
        self.assertEqual(a_ref.shape, (5, solver.dim))
        self.assertEqual(L_inc.shape, (5, solver.dim))
        self.assertEqual(W_inc.shape, (5, solver.dim))

        for tensor in (Y_n, Y_n1, a_ref, L_inc, W_inc):
            self.assertEqual(tensor.dtype, expected_dtype)
            self.assertEqual(tensor.device, expected_device)

    def test_box_sampling_contract(self):
        solver = self._build_solver("box")
        self._assert_sample_contract(solver, mode="box")

    def test_rollout_sampling_contract(self):
        solver = self._build_solver("rollout")
        self._assert_sample_contract(solver, mode="rollout")

    def test_shortage_sampling_contract(self):
        solver = self._build_solver("mixture")
        self._assert_sample_contract(solver, mode="shortage")

    def test_lognormal_proposal_sampling_contract(self):
        solver = self._build_solver(
            "lognormal_proposal",
            rollout_pool_size=32,
            lognormal_r_min=4,
        )
        self._assert_sample_contract(solver, mode="lognormal_proposal")

    def test_lognormal_proposal_is_seed_deterministic(self):
        solver_a = self._build_solver(
            "lognormal_proposal",
            rollout_pool_size=32,
            lognormal_r_min=4,
            sampling_seed=123,
        )
        solver_b = self._build_solver(
            "lognormal_proposal",
            rollout_pool_size=32,
            lognormal_r_min=4,
            sampling_seed=123,
        )

        sample_a = solver_a.sample_states(1, 16, mode="lognormal_proposal")
        sample_b = solver_b.sample_states(1, 16, mode="lognormal_proposal")

        self.assertTrue(torch.allclose(sample_a, sample_b))
        self.assertTrue(torch.all(sample_a >= 0.0))

    def test_lognormal_proposal_uniform_fallback_for_near_zero_rollout_coords(self):
        solver = self._build_solver(
            "lognormal_proposal",
            rollout_pool_size=32,
            lognormal_r_min=4,
        )
        states, metadata = solver.sample_states(
            1,
            64,
            mode="lognormal_proposal",
            return_metadata=True,
        )

        thresholds = solver.state_sampler._lognormal_fallback_thresholds().to(states.device)
        z_thresholds = thresholds[:solver.dim]

        self.assertEqual(metadata["mode"], "lognormal_proposal")
        self.assertEqual(metadata["uniform_coord_count"], solver.dim)
        self.assertEqual(metadata["lognormal_coord_count"], 1)
        self.assertTrue(torch.all(states[:, :solver.dim] >= 0.0))
        self.assertTrue(
            torch.all(states[:, :solver.dim] <= z_thresholds.unsqueeze(0) + 1e-7)
        )
        self.assertTrue(torch.all(states[:, solver.dim] > 0.0))

    def test_rollout_sampling_uses_time_indexed_pool(self):
        solver = self._build_solver("rollout")

        pool_t0 = solver.state_sampler.get_rollout_state_pool(0)
        pool_t1 = solver.state_sampler.get_rollout_state_pool(1)

        expected_t0 = torch.tensor([0.0, 0.0, 10.0], device=solver.device, dtype=solver.bsde.mu.dtype)
        expected_t1 = torch.tensor([0.0, 0.0, 8.5], device=solver.device, dtype=solver.bsde.mu.dtype)

        self.assertTrue(torch.allclose(pool_t0, expected_t0.unsqueeze(0).expand_as(pool_t0)))
        self.assertTrue(torch.allclose(pool_t1, expected_t1.unsqueeze(0).expand_as(pool_t1)))

        sampled_t0 = solver.sample_states(0, 4, mode="rollout")
        sampled_t1 = solver.sample_states(1, 4, mode="rollout")

        self.assertTrue(torch.allclose(sampled_t0, expected_t0.unsqueeze(0).expand_as(sampled_t0)))
        self.assertTrue(torch.allclose(sampled_t1, expected_t1.unsqueeze(0).expand_as(sampled_t1)))
        self.assertFalse(torch.allclose(sampled_t0, sampled_t1))

    def test_rollout_pool_can_start_from_q_sensitivity_initial_mixture(self):
        solver = self._build_solver(
            "rollout",
            rollout_pool_size=5000,
            sampling_seed=123,
            initial_state_mode="q_sensitivity_mixture",
            initial_perturb_minus_prob=0.20,
            initial_perturb_plus_prob=0.10,
            initial_perturb_minus_frac=0.10,
            initial_perturb_plus_frac=0.05,
            initial_dirichlet_alpha=1.0,
        )

        pool_t0 = solver.state_sampler.get_rollout_state_pool(0).detach().cpu()
        labels = solver.state_sampler.initial_state_component_labels

        self.assertEqual(pool_t0.shape, (5000, solver.state_dim))
        self.assertAlmostEqual(float((labels == 0).double().mean()), 0.70, delta=0.03)
        self.assertAlmostEqual(float((labels == 1).double().mean()), 0.20, delta=0.03)
        self.assertAlmostEqual(float((labels == 2).double().mean()), 0.10, delta=0.03)

        redistributed = pool_t0[labels == 1]
        self.assertGreater(redistributed.shape[0], 0)
        torch.testing.assert_close(
            redistributed.sum(dim=1),
            torch.full((redistributed.shape[0],), float(solver.bsde.I0), dtype=redistributed.dtype),
            atol=1e-5,
            rtol=1e-6,
        )
        self.assertTrue(torch.all(redistributed[:, : solver.dim] >= 0.0))
        self.assertTrue(torch.all(redistributed[:, solver.dim] <= float(solver.bsde.I0) + 1e-6))

        q_plus = pool_t0[labels == 2]
        self.assertGreater(q_plus.shape[0], 0)
        torch.testing.assert_close(q_plus[:, : solver.dim], torch.zeros_like(q_plus[:, : solver.dim]))
        self.assertTrue(torch.all(q_plus[:, solver.dim] >= float(solver.bsde.I0) - 1e-6))
        self.assertTrue(torch.any(q_plus[:, solver.dim] > float(solver.bsde.I0)))

        local_sum = redistributed[:, : solver.dim].sum(dim=1)
        positive = local_sum > 1e-10
        weights = redistributed[positive, : solver.dim] / local_sum[positive].unsqueeze(1)
        torch.testing.assert_close(
            weights.mean(dim=0),
            torch.full((solver.dim,), 1.0 / float(solver.dim), dtype=weights.dtype),
            atol=0.03,
            rtol=0.0,
        )

        summary = solver.state_sampler.initial_state_summary
        self.assertEqual(summary["initial_state_mode"], "q_sensitivity_mixture")
        self.assertGreater(summary["initial_q_above_q0_share"], 0.0)

    def test_mixture_sampling_respects_requested_ratio(self):
        solver = self._build_solver(
            "mixture",
            mix_rollout=0.5,
            mix_box=0.3,
            mix_shortage=0.2,
        )
        states, metadata = solver.sample_states(1, 10, mode="mixture", return_metadata=True)

        self.assertEqual(states.shape, (10, solver.state_dim))
        self.assertEqual(metadata["rollout_count"], 5)
        self.assertEqual(metadata["box_count"], 3)
        self.assertEqual(metadata["shortage_count"], 2)
        self.assertEqual(int((metadata["source_labels"] == 1).sum().item()), 5)
        self.assertEqual(int((metadata["source_labels"] == 0).sum().item()), 3)
        self.assertEqual(int((metadata["source_labels"] == 2).sum().item()), 2)
        self.assertEqual(
            metadata["rollout_count"] + metadata["box_count"] + metadata["shortage_count"],
            10,
        )

    def test_shortage_sampling_respects_low_inventory_bounds(self):
        solver = self._build_solver(
            "mixture",
            shortage_q_frac=0.2,
            shortage_z_frac=0.1,
        )
        states = solver.sample_states(1, 32, mode="shortage")
        z_states = states[:, :solver.dim]
        q_states = states[:, solver.dim:]

        self.assertLessEqual(q_states.max().item(), 0.2 * solver.bsde.I0 + 1e-6)
        self.assertLessEqual(z_states.max().item(), 0.1 * solver._compute_box_z_max(1) + 1e-6)

    def test_prev_init_defaults_to_unconditional_reuse(self):
        solver = self._build_solver("box", prev_init=True)
        decision = solver._get_interval_init_decision(0)

        self.assertTrue(decision["reuse_prev_init"])
        self.assertEqual(decision["reason"], "prev_init_unconditional")

    def test_conditional_prev_init_reuses_when_adjacent_params_match(self):
        solver = self._build_solver(
            "box",
            prev_init=True,
            prev_init_same_params_only=True,
        )
        decision = solver._get_interval_init_decision(0)

        self.assertTrue(decision["reuse_prev_init"])
        self.assertEqual(decision["reason"], "params_match_within_tol")
        self.assertTrue(decision["mu_matches"])
        self.assertTrue(decision["gamma_matches"])

    def test_conditional_prev_init_skips_reuse_when_mu_changes(self):
        solver = self._build_solver(
            "box",
            prev_init=True,
            prev_init_same_params_only=True,
        )
        solver.bsde.mu[0, 0] = solver.bsde.mu[1, 0] + 1e-3
        decision = solver._get_interval_init_decision(0)

        self.assertFalse(decision["reuse_prev_init"])
        self.assertEqual(decision["reason"], "params_changed_beyond_tol")
        self.assertFalse(decision["mu_matches"])
        self.assertTrue(decision["gamma_matches"])

    def test_conditional_prev_init_skips_reuse_when_gamma_changes(self):
        solver = self._build_solver(
            "box",
            prev_init=True,
            prev_init_same_params_only=True,
        )
        solver.bsde.Gamma[0, 0, 0] = solver.bsde.Gamma[1, 0, 0] + 1e-3
        decision = solver._get_interval_init_decision(0)

        self.assertFalse(decision["reuse_prev_init"])
        self.assertEqual(decision["reason"], "params_changed_beyond_tol")
        self.assertTrue(decision["mu_matches"])
        self.assertFalse(decision["gamma_matches"])

    def test_conditional_prev_init_skips_reuse_when_eta_changes(self):
        solver = self._build_solver(
            "box",
            prev_init=True,
            prev_init_same_params_only=True,
        )
        solver.bsde.eta_schedule[0, 0] = solver.bsde.eta_schedule[1, 0] + 1e-3
        decision = solver._get_interval_init_decision(0)

        self.assertFalse(decision["reuse_prev_init"])
        self.assertEqual(decision["reason"], "params_changed_beyond_tol")
        self.assertTrue(decision["mu_matches"])
        self.assertTrue(decision["gamma_matches"])
        self.assertFalse(decision["eta_matches"])

    def test_conditional_prev_init_tolerance_absorbs_tiny_numeric_noise(self):
        solver = self._build_solver(
            "box",
            prev_init=True,
            prev_init_same_params_only=True,
            prev_init_param_tol=1e-6,
        )
        solver.bsde.mu[0, 0] = solver.bsde.mu[1, 0] + 5e-7
        solver.bsde.Gamma[0, 0, 0] = solver.bsde.Gamma[1, 0, 0] + 5e-7
        decision = solver._get_interval_init_decision(0)

        self.assertTrue(decision["reuse_prev_init"])
        self.assertEqual(decision["reason"], "params_match_within_tol")

    def test_reinitialized_interval_uses_final_budget(self):
        solver = self._build_solver(
            "box",
            prev_init=True,
            prev_init_same_params_only=True,
            final_iterations=7,
            inner_iterations=3,
            final_learning_rate=5e-3,
            inner_learning_rate=1e-3,
            final_milestones=[2, 5],
            inner_milestones=[1],
            early_stopping=True,
        )
        solver.bsde.mu[0, 0] = solver.bsde.mu[1, 0] + 1e-3
        decision = solver._get_interval_init_decision(0)
        schedule = solver._get_interval_training_schedule(decision)

        self.assertFalse(decision["reuse_prev_init"])
        self.assertEqual(schedule["budget_type"], "final")
        self.assertEqual(schedule["iterations"], 7)
        self.assertEqual(schedule["learning_rate"], 5e-3)
        self.assertEqual(schedule["milestones"], [2, 5])
        self.assertFalse(schedule["early_stopping"])

    def test_non_reinitialized_interval_keeps_inner_budget(self):
        solver = self._build_solver(
            "box",
            prev_init=True,
            prev_init_same_params_only=True,
            final_iterations=7,
            inner_iterations=3,
            final_learning_rate=5e-3,
            inner_learning_rate=1e-3,
            final_milestones=[2, 5],
            inner_milestones=[1],
            early_stopping=True,
        )
        decision = solver._get_interval_init_decision(0)
        schedule = solver._get_interval_training_schedule(decision)

        self.assertTrue(decision["reuse_prev_init"])
        self.assertEqual(schedule["budget_type"], "inner")
        self.assertEqual(schedule["iterations"], 3)
        self.assertEqual(schedule["learning_rate"], 1e-3)
        self.assertEqual(schedule["milestones"], [1])
        self.assertTrue(schedule["early_stopping"])

    def test_training_smoke_runs_for_all_sampling_modes(self):
        for mode in ("box", "rollout", "mixture", "lognormal_proposal"):
            with self.subTest(mode=mode):
                solver = self._build_solver(
                    mode,
                    mix_rollout=0.5,
                    mix_box=0.25,
                    mix_shortage=0.25,
                )
                run_dir = tempfile.TemporaryDirectory()
                try:
                    logs_dir = os.path.join(run_dir.name, "logs")
                    os.makedirs(logs_dir, exist_ok=True)
                    history_path = os.path.join(run_dir.name, f"{mode}_history.csv")

                    solver.train(
                        final_iterations=1,
                        inner_iterations=1,
                        save_path_logs=logs_dir + os.sep,
                        save_path_cppweights=None,
                        training_history_path=history_path,
                    )

                    self.assertTrue(os.path.exists(history_path))
                    self.assertTrue(os.path.exists(os.path.join(logs_dir, "y_network1.pth")))
                    self.assertTrue(os.path.exists(os.path.join(logs_dir, "z_network1.pth")))
                    self.assertTrue(os.path.exists(os.path.join(logs_dir, "y_network0.pth")))
                    self.assertTrue(os.path.exists(os.path.join(logs_dir, "z_network0.pth")))
                finally:
                    run_dir.cleanup()

    def test_training_smoke_runs_with_consistency_regularization(self):
        solver = self._build_solver(
            "mixture",
            mix_rollout=0.5,
            mix_box=0.25,
            mix_shortage=0.25,
            consistency_coef=1e-2,
        )
        run_dir = tempfile.TemporaryDirectory()
        try:
            logs_dir = os.path.join(run_dir.name, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            history_path = os.path.join(run_dir.name, "history.csv")
            summary_path = os.path.join(run_dir.name, "interval_summary.csv")

            solver.train(
                final_iterations=1,
                inner_iterations=1,
                save_path_logs=logs_dir + os.sep,
                save_path_cppweights=None,
                training_history_path=history_path,
                interval_summary_path=summary_path,
            )

            self.assertTrue(os.path.exists(history_path))
            with open(history_path, newline="") as f:
                history_rows = list(csv.reader(f))
            self.assertTrue(history_rows)
            for row in history_rows:
                self.assertGreaterEqual(len(row), 17)
                self.assertAlmostEqual(float(row[2]), float(row[11]) + float(row[15]), delta=1e-4)
                self.assertAlmostEqual(float(row[3]), float(row[12]) + float(row[16]), delta=1e-4)

            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, newline="") as f:
                summary_rows = list(csv.DictReader(f))
            self.assertTrue(summary_rows)
            for row in summary_rows:
                self.assertIn("best_val_residual_loss", row)
                self.assertIn("best_val_consistency_loss", row)
                self.assertIn("best_val_weighted_consistency_loss", row)
                self.assertIn("final_val_residual_loss", row)
                self.assertIn("final_val_consistency_loss", row)
                self.assertIn("final_val_weighted_consistency_loss", row)
                self.assertEqual(float(row["consistency_coef"]), 1e-2)

            self.assertTrue(os.path.exists(os.path.join(logs_dir, "y_network1.pth")))
            self.assertTrue(os.path.exists(os.path.join(logs_dir, "z_network1.pth")))
            self.assertTrue(os.path.exists(os.path.join(logs_dir, "y_network0.pth")))
            self.assertTrue(os.path.exists(os.path.join(logs_dir, "z_network0.pth")))
        finally:
            run_dir.cleanup()

    def test_final_interval_never_early_stops_but_inner_can(self):
        solver = self._build_solver(
            "box",
            early_stopping=True,
            patience=0,
            final_iterations=3,
            inner_iterations=3,
        )

        interval_call_counts = {solver.num_time_interval - 1: 0, 0: 0}

        def scripted_loss_fn(self, **kwargs):
            interval_idx = kwargs["n"]
            call_idx = interval_call_counts[interval_idx]
            interval_call_counts[interval_idx] += 1
            iteration_idx = call_idx // 2
            is_validation_call = (call_idx % 2) == 1

            if is_validation_call:
                scripted_value = 1.0 + float(iteration_idx)
            else:
                scripted_value = 0.5 + float(iteration_idx)

            base = kwargs["H_n"].sum() * 0.0 + kwargs["D_n"].sum() * 0.0
            return base + scripted_value

        solver.loss_fn = types.MethodType(scripted_loss_fn, solver)

        run_dir = tempfile.TemporaryDirectory()
        try:
            logs_dir = os.path.join(run_dir.name, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            history_path = os.path.join(run_dir.name, "history.csv")
            summary_path = os.path.join(run_dir.name, "interval_summary.csv")

            history = solver.train(
                final_iterations=3,
                inner_iterations=3,
                save_path_logs=logs_dir + os.sep,
                save_path_cppweights=None,
                training_history_path=history_path,
                interval_summary_path=summary_path,
            )

            final_rows = [row for row in history if row[0] == solver.num_time_interval - 1]
            inner_rows = [row for row in history if row[0] == 0]

            self.assertTrue(final_rows)
            self.assertTrue(inner_rows)

            final_summary = final_rows[-1]
            inner_summary = inner_rows[-1]

            self.assertEqual(final_summary[1], 2)
            self.assertEqual(final_summary[8], 2)
            self.assertFalse(final_summary[9])

            self.assertEqual(inner_summary[1], 1)
            self.assertEqual(inner_summary[8], 1)
            self.assertTrue(inner_summary[9])

            self.assertTrue(os.path.exists(history_path))
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, newline="") as f:
                summary_rows = list(csv.DictReader(f))

            self.assertEqual(len(summary_rows), 2)
            summary_by_interval = {int(row["interval"]): row for row in summary_rows}
            self.assertEqual(float(summary_by_interval[1]["best_val_loss"]), 1.0)
            self.assertEqual(int(float(summary_by_interval[1]["best_iter"])), 0)
            self.assertEqual(float(summary_by_interval[0]["best_val_loss"]), 1.0)
            self.assertEqual(int(float(summary_by_interval[0]["best_iter"])), 0)
            self.assertEqual(summary_by_interval[0]["early_stopped"], "True")
            self.assertEqual(summary_by_interval[1]["early_stopped"], "False")
            self.assertTrue(os.path.exists(os.path.join(logs_dir, "y_network1.pth")))
            self.assertTrue(os.path.exists(os.path.join(logs_dir, "z_network1.pth")))
            self.assertTrue(os.path.exists(os.path.join(logs_dir, "y_network0.pth")))
            self.assertTrue(os.path.exists(os.path.join(logs_dir, "z_network0.pth")))
        finally:
            run_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
