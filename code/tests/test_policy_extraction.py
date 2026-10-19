import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import equation as eqn
import policy_extraction as pext
from evaluate import create_deep_splitting_policy, simulate_periodic_review_trajectories


class ConstantGradNet(torch.nn.Module):
    def __init__(self, grad):
        super().__init__()
        self.register_buffer("grad", torch.tensor(grad, dtype=torch.float32))

    def forward(self, state):
        return self.grad.unsqueeze(0).expand(state.shape[0], -1)


def fake_bsde(num_intervals=6, dim=3, delta_t=0.2, eta_value=4.0, grad_value=-1.0):
    del grad_value
    mu = torch.ones(num_intervals, dim, dtype=torch.float32)
    eta_schedule = torch.full((num_intervals, dim), float(eta_value), dtype=torch.float32)
    return SimpleNamespace(
        dim=dim,
        num_time_interval=num_intervals,
        total_time=float(num_intervals) * float(delta_t),
        I0=10.0,
        mu=mu,
        eta=eta_schedule.max(dim=0).values,
        eta_schedule=eta_schedule,
        p=torch.full((dim - 1,), 0.5, dtype=torch.float32),
        c=torch.ones(dim, dtype=torch.float32),
        d=torch.ones(dim, dtype=torch.float32),
        h=torch.zeros(dim, dtype=torch.float32),
        u=torch.tensor(1.0, dtype=torch.float32),
        exposure=torch.zeros(dim, dtype=torch.float32),
        lambda_param=0.0,
        alpha=0.5,
        delta_t=float(delta_t),
        eta_at_fine_index=lambda n: eta_schedule[min(max(int(n), 0), num_intervals - 1)],
    )


class PolicyExtractionTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.bsde = fake_bsde()
        self.nets = torch.nn.ModuleList([ConstantGradNet([-1.0, -1.0, -1.0]) for _ in range(6)])
        self.config = SimpleNamespace(net_config=SimpleNamespace(input_mode="raw", q0_input_scale=10.0))
        self.extractor = pext.DeepSplittingPolicyExtractor(self.nets, self.bsde, self.device, self.config)
        self.Z = torch.zeros(2, 3)
        self.Q = torch.tensor([10.0, 1.0])

    def test_fine_network_index_mapping(self):
        self.assertEqual(pext.fine_indices_for_period(0, 5), (0, 1, 2, 3, 4))
        self.assertEqual(pext.fine_indices_for_period(3, 5), (15, 16, 17, 18, 19))

    def test_e0_matches_current_policy_call_sum(self):
        policy = create_deep_splitting_policy(
            self.nets,
            self.bsde,
            self.device,
            config=self.config,
            extraction_mode="E0",
        )
        result = self.extractor.extract_weekly_action("E0", 0, self.Z, self.Q, 3, 0.6)
        manual = torch.zeros_like(result.tentative_quantity)
        for fine_idx in (0, 1, 2):
            manual += policy(fine_idx, self.Z, self.Q) * 0.2
        self.assertTrue(torch.allclose(result.tentative_quantity, manual, rtol=0.0, atol=1e-7))

    def test_e1_only_queries_first_network(self):
        result = self.extractor.extract_weekly_action("E1", 1, self.Z, self.Q, 3, 0.6)
        self.assertEqual(result.query_count, 1)
        self.assertEqual(result.fine_indices, (3, 4, 5))
        expected_first_rate = self.extractor.raw_rate(3, self.Z, self.Q)
        self.assertTrue(torch.allclose(result.tentative_quantity, expected_first_rate * 0.6))
        self.assertTrue(torch.all(result.raw_rates[1:] == 0.0))

    def test_e2_keeps_z_fixed_and_q_monotone(self):
        result = self.extractor.extract_weekly_action("E2", 0, self.Z, self.Q, 3, 0.6)
        self.assertFalse(result.used_step_transition)
        self.assertTrue(torch.allclose(result.virtual_z[1:], self.Z.unsqueeze(0).expand(3, -1, -1)))
        q_path = result.virtual_q.detach().numpy()
        self.assertTrue(np.all(np.diff(q_path, axis=0) <= 1e-6))
        self.assertTrue(torch.all(result.tentative_quantity.sum(dim=1) <= self.Q + 1e-6))

    def test_e3_uses_step_transition_with_zero_delta_w(self):
        calls = []
        original = eqn.step_transition_torch

        def wrapped_step_transition(Z, Q, allocation_rate, delta_w, delta_t, mu_t, p, clamp_inventory=True):
            calls.append(delta_w.detach().clone())
            return original(Z, Q, allocation_rate, delta_w, delta_t, mu_t, p, clamp_inventory=clamp_inventory)

        with mock.patch("policy_extraction.eqn.step_transition_torch", side_effect=wrapped_step_transition):
            result = self.extractor.extract_weekly_action("E3", 0, self.Z, self.Q, 3, 0.6)

        self.assertTrue(result.used_step_transition)
        self.assertTrue(result.delta_w_is_zero)
        self.assertEqual(len(calls), 3)
        for delta_w in calls:
            self.assertTrue(torch.count_nonzero(delta_w).item() == 0)
        self.assertFalse(torch.allclose(result.virtual_z[-1], self.Z))

    def test_e2_e3_do_not_report_future_information_use(self):
        for mode in ("E1", "E2", "E3"):
            result = self.extractor.extract_weekly_action(mode, 0, self.Z, self.Q, 3, 0.6)
            self.assertFalse(result.uses_future_information)

    def test_discrete_review_simulator_uses_attached_extractor(self):
        policy = create_deep_splitting_policy(
            self.nets,
            self.bsde,
            self.device,
            config=self.config,
            extraction_mode="E2",
        )
        mu_review = np.ones((2, 3), dtype=np.float32)
        gamma_review = np.zeros((2, 3, 3), dtype=np.float32)
        demands = np.zeros((2, 2, 3), dtype=np.float32)
        costs = simulate_periodic_review_trajectories(
            policy,
            self.bsde,
            2,
            self.device,
            mu_review,
            gamma_review,
            precomputed_demands=demands,
            return_history=True,
        )
        self.assertIn("virtual_q_hist", costs)
        self.assertIn("fine_projection_needed_hist", costs)
        self.assertEqual(costs["extraction_query_count_hist"].tolist(), [3, 3])
        self.assertTrue(np.all(np.diff(costs["virtual_q_hist"], axis=1) <= 1e-6))


if __name__ == "__main__":
    unittest.main()
