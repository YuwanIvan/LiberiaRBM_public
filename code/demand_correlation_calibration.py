"""Calibrate Gaussian-copula latent correlation to realized count correlation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.stats import binom, multivariate_normal, norm


CALIBRATION_METHOD = "exact_discrete_threshold_bivariate_normal_cdf_v1"
TAIL_PROBABILITY_TOLERANCE = 1e-14
CDF_MAX_POINTS = 1_000_000
CDF_ABSOLUTE_TOLERANCE = 1e-12
CDF_RELATIVE_TOLERANCE = 1e-12


class CorrelationCalibrationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrelationCalibrationError(message)


def nearest_correlation(matrix: np.ndarray, *, min_eigenvalue: float = 1e-10, iterations: int = 20) -> np.ndarray:
    """Return a symmetric positive-definite correlation matrix."""

    value = np.asarray(matrix, dtype=np.float64).copy()
    _require(value.ndim == 2 and value.shape[0] == value.shape[1], "Correlation matrix must be square.")
    value = 0.5 * (value + value.T)
    np.fill_diagonal(value, 1.0)
    for _ in range(int(iterations)):
        eigenvalues, eigenvectors = np.linalg.eigh(value)
        value = (eigenvectors * np.maximum(eigenvalues, min_eigenvalue)) @ eigenvectors.T
        scale = np.sqrt(np.maximum(np.diag(value), min_eigenvalue))
        value = value / np.outer(scale, scale)
        value = np.clip(0.5 * (value + value.T), -0.9999, 0.9999)
        np.fill_diagonal(value, 1.0)
    return value


@lru_cache(maxsize=1024)
def _active_binomial_thresholds(n: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Return CDF values and Gaussian thresholds with non-negligible mass on both sides.

    A binomial inverse CDF is a step function.  The old Gauss--Hermite rule
    integrated across those jumps and could bias a realized-correlation result
    by roughly one percentage point.  Splitting at the exact count thresholds
    removes that source of numerical error.
    """

    counts = np.arange(int(n), dtype=np.int64)
    cdf = binom.cdf(counts, int(n), float(p))
    survival = binom.sf(counts, int(n), float(p))
    active = (cdf > TAIL_PROBABILITY_TOLERANCE) & (survival > TAIL_PROBABILITY_TOLERANCE)
    active_cdf = np.asarray(cdf[active], dtype=np.float64)
    active_thresholds = np.asarray(norm.ppf(active_cdf), dtype=np.float64)
    _require(active_cdf.size > 0, "Binomial threshold grid is numerically empty.")
    _require(np.isfinite(active_thresholds).all(), "Binomial threshold grid contains nonfinite values.")
    active_cdf.setflags(write=False)
    active_thresholds.setflags(write=False)
    return active_cdf, active_thresholds


def realized_binomial_correlation(
    n1: int,
    p1: float,
    n2: int,
    p2: float,
    latent_rho: float,
) -> float:
    """Compute Corr(Bin(n1,p1), Bin(n2,p2)) under a Gaussian copula.

    For integer-valued nonnegative variables,

        Cov(X, Y) = sum_i sum_j [P(X <= i, Y <= j) - P(X <= i)P(Y <= j)].

    Each binomial CDF value maps to one Gaussian threshold, so the joint terms
    are bivariate-normal CDF evaluations.  This respects every discrete jump
    explicitly instead of applying a global quadrature rule to an inverse-CDF
    step function.
    """

    n1, n2 = int(n1), int(n2)
    p1, p2 = float(p1), float(p2)
    rho = float(latent_rho)
    _require(n1 > 0 and n2 > 0, "Binomial exposure must be positive.")
    _require(0.0 < p1 < 1.0 and 0.0 < p2 < 1.0, "Binomial probabilities must be strictly between zero and one.")
    _require(abs(rho) < 1.0, "Latent correlation must be strictly between -1 and 1.")
    if rho == 0.0:
        return 0.0
    cdf1, thresholds1 = _active_binomial_thresholds(n1, p1)
    cdf2, thresholds2 = _active_binomial_thresholds(n2, p2)
    points = np.column_stack(
        (
            np.repeat(thresholds1, thresholds2.size),
            np.tile(thresholds2, thresholds1.size),
        )
    )
    joint_cdf = np.asarray(
        multivariate_normal.cdf(
            points,
            mean=np.zeros(2, dtype=np.float64),
            cov=np.asarray([[1.0, rho], [rho, 1.0]], dtype=np.float64),
            maxpts=CDF_MAX_POINTS,
            abseps=CDF_ABSOLUTE_TOLERANCE,
            releps=CDF_RELATIVE_TOLERANCE,
        ),
        dtype=np.float64,
    )
    _require(np.isfinite(joint_cdf).all(), "Bivariate-normal CDF returned nonfinite values.")
    independent_cdf = np.repeat(cdf1, cdf2.size) * np.tile(cdf2, cdf1.size)
    covariance = float(np.sum(joint_cdf - independent_cdf))
    variance1 = n1 * p1 * (1.0 - p1)
    variance2 = n2 * p2 * (1.0 - p2)
    return covariance / np.sqrt(variance1 * variance2)


def solve_latent_binomial_correlation(
    n1: int,
    p1: float,
    n2: int,
    p2: float,
    target_rho: float,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 60,
) -> dict[str, float | int]:
    """Invert the monotone latent-to-realized correlation map by bisection."""

    target = float(target_rho)
    _require(0.0 <= target < 1.0, "This paper calibration expects target correlation in [0,1).")
    if target == 0.0:
        return {"latent_rho": 0.0, "realized_rho": 0.0, "iterations": 0}
    lower, upper = 0.0, 0.9999
    upper_value = realized_binomial_correlation(n1, p1, n2, p2, upper)
    _require(
        upper_value >= target - tolerance,
        f"Target realized correlation {target:g} is unattainable; maximum is approximately {upper_value:g}.",
    )
    midpoint = 0.5 * (lower + upper)
    observed = 0.0
    for iteration in range(1, int(max_iterations) + 1):
        midpoint = 0.5 * (lower + upper)
        observed = realized_binomial_correlation(n1, p1, n2, p2, midpoint)
        if abs(observed - target) <= tolerance:
            return {"latent_rho": midpoint, "realized_rho": observed, "iterations": iteration}
        if observed < target:
            lower = midpoint
        else:
            upper = midpoint
    return {"latent_rho": midpoint, "realized_rho": observed, "iterations": int(max_iterations)}


def calibrate_weekly_latent_matrices(
    mu_review: np.ndarray,
    exposure: np.ndarray,
    review_dt: float,
    target_rho: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate and PSD-project one latent matrix per review period."""

    mu = np.asarray(mu_review, dtype=np.float64)
    exposure_int = np.asarray(exposure, dtype=np.int64).reshape(-1)
    _require(mu.ndim == 2 and mu.shape[1] == exposure_int.shape[0], "mu/exposure shape mismatch.")
    probabilities = mu * float(review_dt) / exposure_int.reshape(1, -1)
    _require(np.all((probabilities > 0.0) & (probabilities < 1.0)), "All binomial probabilities must be in (0,1).")
    periods, dim = mu.shape
    matrices = np.repeat(np.eye(dim, dtype=np.float64)[None, :, :], periods, axis=0)
    pair_rows: list[dict[str, Any]] = []
    solution_cache: dict[tuple[tuple[int, float], tuple[int, float]], dict[str, float | int]] = {}
    for week in range(periods):
        raw = np.eye(dim, dtype=np.float64)
        for left in range(dim):
            for right in range(left + 1, dim):
                marginal_left = (int(exposure_int[left]), float(probabilities[week, left]))
                marginal_right = (int(exposure_int[right]), float(probabilities[week, right]))
                cache_key = tuple(sorted((marginal_left, marginal_right)))
                solved = solution_cache.get(cache_key)
                if solved is None:
                    solved = solve_latent_binomial_correlation(
                        marginal_left[0],
                        marginal_left[1],
                        marginal_right[0],
                        marginal_right[1],
                        target_rho,
                    )
                    solution_cache[cache_key] = solved
                raw[left, right] = raw[right, left] = float(solved["latent_rho"])
        projected = nearest_correlation(raw)
        matrices[week] = projected
        for left in range(dim):
            for right in range(left + 1, dim):
                realized = realized_binomial_correlation(
                    int(exposure_int[left]),
                    float(probabilities[week, left]),
                    int(exposure_int[right]),
                    float(probabilities[week, right]),
                    float(projected[left, right]),
                )
                pair_rows.append(
                    {
                        "week": week,
                        "left": left,
                        "right": right,
                        "target_realized_rho": float(target_rho),
                        "latent_rho": float(projected[left, right]),
                        "predicted_realized_rho": float(realized),
                        "absolute_error": float(abs(realized - target_rho)),
                    }
                )
    errors = np.asarray([row["absolute_error"] for row in pair_rows], dtype=np.float64)
    return matrices, {
        "numerical_method": {
            "name": CALIBRATION_METHOD,
            "tail_probability_tolerance": TAIL_PROBABILITY_TOLERANCE,
            "bivariate_normal_cdf_max_points": CDF_MAX_POINTS,
            "bivariate_normal_cdf_absolute_tolerance": CDF_ABSOLUTE_TOLERANCE,
            "bivariate_normal_cdf_relative_tolerance": CDF_RELATIVE_TOLERANCE,
        },
        "target_realized_rho": float(target_rho),
        "mean_absolute_pairwise_error": float(np.mean(errors)),
        "maximum_absolute_pairwise_error": float(np.max(errors)),
        "minimum_latent_eigenvalue": float(min(np.min(np.linalg.eigvalsh(matrix)) for matrix in matrices)),
        "pairs": pair_rows,
    }
