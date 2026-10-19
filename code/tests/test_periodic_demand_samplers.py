import unittest

import numpy as np

try:
    import evaluate as eval_mod
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    eval_mod = None


@unittest.skipIf(eval_mod is None, "torch is required to import evaluate.py")
class PeriodicDemandSamplerTests(unittest.TestCase):
    def setUp(self):
        self.exposure = np.array([219] + [28] * 7, dtype=float)
        self.theta = 0.05
        self.mu = self.exposure * self.theta
        self.gamma_diag = self.exposure * self.theta * (1.0 - self.theta)

    def test_binomial_sampler_matches_integer_marginals(self):
        rng = np.random.default_rng(123)
        draws = eval_mod.sample_periodic_demand(
            self.mu,
            np.diag(self.gamma_diag),
            1.0,
            20000,
            rng,
            mode="binomial",
            exposure=self.exposure,
        )

        self.assertTrue(np.all(draws >= 0))
        self.assertTrue(np.allclose(draws, np.round(draws)))
        np.testing.assert_allclose(draws.mean(axis=0), self.mu, rtol=0.03, atol=0.08)
        np.testing.assert_allclose(draws.var(axis=0), self.gamma_diag, rtol=0.08, atol=0.12)

    def test_gaussian_copula_binomial_preserves_marginals_and_positive_dependence(self):
        rng = np.random.default_rng(456)
        rho = 0.6
        std = np.sqrt(self.gamma_diag)
        cov = rho * np.outer(std, std)
        np.fill_diagonal(cov, self.gamma_diag)

        draws = eval_mod.sample_periodic_demand(
            self.mu,
            cov,
            1.0,
            30000,
            rng,
            mode="gaussian_copula_binomial",
            exposure=self.exposure,
        )

        np.testing.assert_allclose(draws.mean(axis=0), self.mu, rtol=0.035, atol=0.10)
        corr = np.corrcoef(draws[:, 1], draws[:, 2])[0, 1]
        self.assertGreater(corr, 0.25)

    def test_beta_binomial_inflates_cha_variance_independently(self):
        rng = np.random.default_rng(789)
        gamma = self.gamma_diag.copy()
        gamma[1:] *= 3.0

        draws = eval_mod.sample_periodic_demand(
            self.mu,
            np.diag(gamma),
            1.0,
            40000,
            rng,
            mode="beta_binomial",
            exposure=self.exposure,
        )

        np.testing.assert_allclose(draws.mean(axis=0), self.mu, rtol=0.04, atol=0.10)
        np.testing.assert_allclose(draws.var(axis=0)[1:], gamma[1:], rtol=0.12, atol=0.2)
        corr = np.corrcoef(draws[:, 1], draws[:, 2])[0, 1]
        self.assertLess(abs(corr), 0.05)


if __name__ == "__main__":
    unittest.main()
