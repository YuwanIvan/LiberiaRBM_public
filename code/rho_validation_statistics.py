"""Scenario-independent statistical gates for realized-rho validation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import norm


VALIDATION_SCHEMA_VERSION = "fangonda-realized-rho-validation-v2"
MARGINAL_FAMILYWISE_ALPHA = 0.05
MARGINAL_CORRECTION = "bonferroni_two_sided_normal"


def simultaneous_z_limit(
    num_comparisons: int,
    *,
    familywise_alpha: float = MARGINAL_FAMILYWISE_ALPHA,
) -> float:
    comparisons = int(num_comparisons)
    alpha = float(familywise_alpha)
    if comparisons <= 0:
        raise ValueError("num_comparisons must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("familywise_alpha must be in (0,1)")
    return float(norm.ppf(1.0 - alpha / (2.0 * comparisons)))


def binomial_marginal_diagnostics(
    draws: np.ndarray,
    exposure: np.ndarray,
    probabilities: np.ndarray,
    *,
    week: int,
    z_limit: float,
) -> list[dict[str, Any]]:
    """Evaluate mean and variance errors using their binomial sampling scales."""

    values = np.asarray(draws, dtype=np.float64)
    trials = np.asarray(exposure, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != trials.size or probs.size != trials.size:
        raise ValueError("draws, exposure, and probabilities have incompatible shapes")
    if values.shape[0] <= 3:
        raise ValueError("at least four paths are required")
    if np.any(trials <= 0) or np.any((probs <= 0.0) | (probs >= 1.0)):
        raise ValueError("binomial trials and probabilities are invalid")
    if not np.isfinite(values).all():
        raise ValueError("draws contain nonfinite values")

    num_paths = int(values.shape[0])
    expected_mean = trials * probs
    expected_variance = trials * probs * (1.0 - probs)
    observed_mean = np.mean(values, axis=0)
    observed_variance = np.var(values, axis=0, ddof=1)
    mean_standard_error = np.sqrt(expected_variance / num_paths)

    variance_squared = expected_variance**2
    fourth_central_moment = (
        3.0 * variance_squared
        + expected_variance * (1.0 - 6.0 * probs * (1.0 - probs))
    )
    sample_variance_variance = (
        fourth_central_moment
        - ((num_paths - 3.0) / (num_paths - 1.0)) * variance_squared
    ) / num_paths
    variance_standard_error = np.sqrt(np.maximum(sample_variance_variance, 0.0))
    if np.any(mean_standard_error <= 0.0) or np.any(variance_standard_error <= 0.0):
        raise ValueError("binomial marginal standard error is not positive")

    rows: list[dict[str, Any]] = []
    for location in range(trials.size):
        mean_absolute_error = float(abs(observed_mean[location] - expected_mean[location]))
        variance_absolute_error = float(abs(observed_variance[location] - expected_variance[location]))
        mean_z = mean_absolute_error / float(mean_standard_error[location])
        variance_z = variance_absolute_error / float(variance_standard_error[location])
        rows.append(
            {
                "week": int(week),
                "location": int(location),
                "num_paths": num_paths,
                "binomial_trials": int(trials[location]),
                "binomial_probability": float(probs[location]),
                "expected_mean": float(expected_mean[location]),
                "observed_mean": float(observed_mean[location]),
                "mean_absolute_error": mean_absolute_error,
                "mean_relative_error": mean_absolute_error / float(expected_mean[location]),
                "mean_standard_error": float(mean_standard_error[location]),
                "mean_z_score": float(mean_z),
                "mean_simultaneous_passed": bool(mean_z <= z_limit),
                "expected_variance": float(expected_variance[location]),
                "observed_variance": float(observed_variance[location]),
                "variance_absolute_error": variance_absolute_error,
                "variance_relative_error": variance_absolute_error / float(expected_variance[location]),
                "variance_standard_error": float(variance_standard_error[location]),
                "variance_z_score": float(variance_z),
                "variance_simultaneous_passed": bool(variance_z <= z_limit),
            }
        )
    return rows


def summarize_marginal_diagnostics(
    rows: Sequence[dict[str, Any]],
    *,
    z_limit: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("marginal diagnostics are empty")
    worst_mean = max(rows, key=lambda row: float(row["mean_z_score"]))
    worst_variance = max(rows, key=lambda row: float(row["variance_z_score"]))
    return {
        "mean_relative_error_max": float(max(float(row["mean_relative_error"]) for row in rows)),
        "variance_relative_error_max": float(
            max(float(row["variance_relative_error"]) for row in rows)
        ),
        "mean_z_score_max": float(worst_mean["mean_z_score"]),
        "variance_z_score_max": float(worst_variance["variance_z_score"]),
        "simultaneous_z_limit": float(z_limit),
        "mean_simultaneous_passed": bool(float(worst_mean["mean_z_score"]) <= z_limit),
        "variance_simultaneous_passed": bool(float(worst_variance["variance_z_score"]) <= z_limit),
        "worst_mean_cell": dict(worst_mean),
        "worst_variance_cell": dict(worst_variance),
    }
