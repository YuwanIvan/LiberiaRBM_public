import math
import unittest

import numpy as np
import torch

import equation as eqn


def scalar_hamiltonian(a, x, mu, eta, lambda_param, alpha):
    del eta
    if mu <= 1e-12:
        utility = 0.0
    else:
        utility = mu * (max(a, 0.0) / mu) ** (1.0 - alpha) / (1.0 - alpha)
    return a * x - lambda_param * utility


class DemandWeightedAlphaFairnessTests(unittest.TestCase):
    def test_alpha_half_utility_identity(self):
        allocation = np.array([[4.0, 0.25, 9.0], [0.0, 16.0, 1.0]], dtype=float)
        mu = np.array([1.0, 4.0, 0.25], dtype=float)

        actual_np = eqn.alpha_fair_utility_numpy(allocation, mu, 0.5)
        expected = 2.0 * np.sqrt(allocation * mu.reshape(1, -1)).sum(axis=1)
        np.testing.assert_allclose(actual_np, expected, rtol=1e-12, atol=1e-12)

        actual_torch = eqn.alpha_fair_utility_torch(
            torch.tensor(allocation, dtype=torch.float64),
            torch.tensor(mu, dtype=torch.float64),
            0.5,
        )
        np.testing.assert_allclose(actual_torch.numpy(), expected, rtol=1e-12, atol=1e-12)

    def test_fairness_only_allocation_is_demand_proportional(self):
        mu = np.array([2.0, 5.0, 3.0, 7.0], dtype=float)
        total_supply = 11.0
        proportional = total_supply * mu / mu.sum()
        baseline = eqn.alpha_fair_utility_numpy(proportional, mu, 0.5)

        rng = np.random.default_rng(123)
        for _ in range(200):
            candidate = rng.dirichlet(np.ones_like(mu)) * total_supply
            value = eqn.alpha_fair_utility_numpy(candidate, mu, 0.5)
            self.assertLessEqual(float(value), float(baseline) + 1e-10)

    def test_closed_form_hamiltonian_minimizer_matches_dense_grid(self):
        rng = np.random.default_rng(321)
        alpha = 0.5
        lambda_param = 0.37
        for _ in range(40):
            mu = float(rng.uniform(0.05, 8.0))
            eta = float(rng.uniform(0.05, 6.0))
            x = float(rng.uniform(-2.0, 5.0))
            grad = torch.tensor([[x]], dtype=torch.float64)
            inventory = torch.tensor([10.0], dtype=torch.float64)
            mu_t = torch.tensor([mu], dtype=torch.float64)
            eta_t = torch.tensor([eta], dtype=torch.float64)

            closed = float(
                eqn.build_feasible_ds_allocation(
                    grad,
                    inventory,
                    mu_t,
                    eta_t,
                    lambda_param,
                    alpha,
                    delta_t=1.0,
                )[0, 0]
            )
            grid = np.linspace(0.0, eta, 20001)
            values = np.array(
                [scalar_hamiltonian(a, x, mu, eta, lambda_param, alpha) for a in grid],
                dtype=float,
            )
            grid_best = float(grid[int(np.argmin(values))])
            self.assertAlmostEqual(closed, grid_best, delta=max(eta / 20000.0 * 2.0, 2e-3))

    def test_threshold_continuity(self):
        mu = torch.tensor([3.5], dtype=torch.float64)
        eta = torch.tensor([2.25], dtype=torch.float64)
        lambda_param = 0.8
        alpha = 0.5
        x = lambda_param * (float(mu[0]) / float(eta[0])) ** alpha
        allocation = eqn.build_feasible_ds_allocation(
            torch.tensor([[x]], dtype=torch.float64),
            torch.tensor([10.0], dtype=torch.float64),
            mu,
            eta,
            lambda_param,
            alpha,
            delta_t=1.0,
        )
        self.assertAlmostEqual(float(allocation[0, 0]), float(eta[0]), places=10)

        interior_formula = float(mu[0]) * (lambda_param / x) ** (1.0 / alpha)
        self.assertAlmostEqual(interior_formula, float(eta[0]), places=10)

    def test_lambda_zero_bang_bang_behavior(self):
        grad = torch.tensor([[-2.0, 0.0, 0.3]], dtype=torch.float64)
        eta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        allocation = eqn.build_feasible_ds_allocation(
            grad,
            torch.tensor([100.0], dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
            eta,
            lambda_param=0.0,
            alpha=0.5,
            delta_t=1.0,
        )
        expected_old_lambda_zero = torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(allocation, expected_old_lambda_zero)

        zero_inventory = eqn.build_feasible_ds_allocation(
            grad,
            torch.tensor([0.0], dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
            eta,
            lambda_param=0.0,
            alpha=0.5,
            delta_t=1.0,
        )
        torch.testing.assert_close(zero_inventory, torch.zeros_like(zero_inventory))

        zero_mu = eqn.build_feasible_ds_allocation(
            torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float64),
            torch.tensor([100.0], dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            eta,
            lambda_param=0.0,
            alpha=0.5,
            delta_t=1.0,
        )
        torch.testing.assert_close(
            zero_mu,
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        )

    def test_zero_mu_has_no_nan_or_inf_and_uses_linear_hamiltonian(self):
        allocation_np = np.array([[3.0, 0.0, 2.0]], dtype=float)
        mu_np = np.array([0.0, 0.0, 4.0], dtype=float)
        utility = eqn.alpha_fair_utility_numpy(allocation_np, mu_np, 0.5)
        self.assertTrue(np.all(np.isfinite(utility)))
        self.assertAlmostEqual(float(utility[0]), 2.0 * math.sqrt(2.0 * 4.0))

        action = eqn.build_feasible_ds_allocation(
            torch.tensor([[-10.0, 0.0, 0.2, 0.2]], dtype=torch.float64),
            torch.tensor([100.0], dtype=torch.float64),
            torch.tensor([0.0, 0.0, 1e-12, 2.0], dtype=torch.float64),
            torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64),
            lambda_param=0.5,
            alpha=0.5,
            delta_t=1.0,
        )
        self.assertEqual(float(action[0, 0]), 1.0)
        # At exactly zero gradient the documented optimal selector is a=0.
        self.assertEqual(float(action[0, 1]), 0.0)
        self.assertEqual(float(action[0, 2]), 0.0)
        self.assertTrue(torch.isfinite(action).all())

    def test_strict_fairness_parameter_validation(self):
        with self.assertRaisesRegex(ValueError, "0 < alpha < 1"):
            eqn.alpha_fair_utility_numpy(np.ones(2), np.ones(2), 1.0)
        with self.assertRaisesRegex(ValueError, "lambda_param must be nonnegative"):
            eqn.build_feasible_ds_allocation(
                torch.ones(1, 2, dtype=torch.float64),
                torch.tensor([10.0], dtype=torch.float64),
                torch.ones(2, dtype=torch.float64),
                torch.ones(2, dtype=torch.float64),
                lambda_param=-0.1,
                alpha=0.5,
                delta_t=1.0,
            )

    def test_fluid_qcp_objective_equivalence_alpha_half(self):
        allocation = np.array([[4.0, 0.0], [1.0, 9.0]], dtype=float)
        period_mu = np.array([[9.0, 0.0], [4.0, 1.0]], dtype=float)
        y = np.where(period_mu > 0.0, np.sqrt(allocation * period_mu), 0.0)
        y[period_mu <= 0.0] = 0.0

        self.assertTrue(np.all(y**2 <= allocation * period_mu + 1e-12))
        cost = 17.0
        lambda_param = 0.25
        qcp_objective = cost - 2.0 * lambda_param * y.sum()
        utility_objective = cost - lambda_param * eqn.alpha_fair_utility_numpy(allocation, period_mu, 0.5).sum()
        self.assertAlmostEqual(qcp_objective, utility_objective, places=12)

    def test_discrete_continuous_scaling(self):
        h = 0.4
        period_mu = np.array([2.0, 5.0, 7.0], dtype=float)
        period_allocation = np.array([1.0, 3.0, 2.0], dtype=float)
        rate_mu = period_mu / h
        rate_allocation = period_allocation / h

        discrete = eqn.alpha_fair_utility_numpy(period_allocation, period_mu, 0.5)
        continuous_integral = h * eqn.alpha_fair_utility_numpy(rate_allocation, rate_mu, 0.5)
        self.assertAlmostEqual(float(discrete), float(continuous_integral), places=12)

    def test_projection_diagnostics_report_scale(self):
        raw = torch.tensor([[3.0, 1.0], [0.5, 0.5]], dtype=torch.float64)
        projected, _, diagnostics = eqn.project_rate_to_inventory_torch(
            raw,
            torch.tensor([2.0, 10.0], dtype=torch.float64),
            delta_t=1.0,
            return_diagnostics=True,
        )
        self.assertEqual(diagnostics["projection_count"], 1)
        self.assertAlmostEqual(diagnostics["min_projection_scale"], 0.5)
        torch.testing.assert_close(projected[0], torch.tensor([1.5, 0.5], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
