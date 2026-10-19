"""Diagnostics aligned with demand-weighted alpha-fair allocation ratios."""

from __future__ import annotations

import numpy as np


def gini_coefficient(values, eps=1e-12):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= eps:
        return 0.0
    sorted_arr = np.sort(arr)
    n = sorted_arr.size
    weights = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(weights * sorted_arr) / (n * total)) - (n + 1.0) / n)


def allocation_concentration(a_hist, dt=1.0, eps=1e-12):
    """Raw cumulative shipment concentration, not the alpha-fair objective."""
    a_hist = np.asarray(a_hist, dtype=np.float64)
    cumulative_alloc = (a_hist * float(dt)).sum(axis=0)
    mean_alloc = np.mean(cumulative_alloc, axis=1)
    std_alloc = np.std(cumulative_alloc, axis=1)
    cv_per_path = std_alloc / np.maximum(mean_alloc, eps)
    gini_per_path = np.array([gini_coefficient(row, eps=eps) for row in cumulative_alloc])
    return {
        "cv": float(np.nanmean(cv_per_path)),
        "gini": float(np.nanmean(gini_per_path)),
        "cv_per_path": cv_per_path,
        "gini_per_path": gini_per_path,
    }


def demand_adjusted_fairness_diagnostics(a_hist, mu_hist, eta_hist=None, dt=1.0, eps=1e-12):
    """Coverage-ratio diagnostics for f[n,k] = a[n,k] / mu[n,k].

    ``a_hist`` is expected to be ``(T, paths, K)`` allocation rates and
    ``mu_hist`` is ``(T, K)`` demand rates. Demand-zero cells are masked out of
    all coverage-ratio metrics.
    """
    a = np.asarray(a_hist, dtype=np.float64)
    mu = np.asarray(mu_hist, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"a_hist must have shape (T, paths, K), got {a.shape}.")
    if mu.shape != (a.shape[0], a.shape[2]):
        raise ValueError(f"mu_hist shape {mu.shape} must match (T, K)={(a.shape[0], a.shape[2])}.")

    active = mu > float(eps)
    weights_tk = np.where(active, mu * float(dt), 0.0)
    weights = weights_tk[:, None, :]
    weight_total = weights.sum(axis=(0, 2))
    safe_mu = np.where(active, mu, 1.0)
    f = np.where(active[:, None, :], a / safe_mu[:, None, :], np.nan)

    weighted_sum = np.nansum(np.where(np.isfinite(f), weights * f, 0.0), axis=(0, 2))
    mean_f = weighted_sum / np.maximum(weight_total, eps)
    centered = np.where(np.isfinite(f), f - mean_f[None, :, None], 0.0)
    var_f = np.sum(weights * centered * centered, axis=(0, 2)) / np.maximum(weight_total, eps)
    cv_f = np.sqrt(np.maximum(var_f, 0.0)) / np.maximum(np.abs(mean_f), eps)

    max_f = np.nanmax(f, axis=(0, 2))
    min_f = np.nanmin(f, axis=(0, 2))
    gap_f = max_f - min_f

    allocation_qty = a * float(dt)
    total_allocation = allocation_qty.sum(axis=(0, 2))
    benchmark_f = total_allocation / np.maximum(weight_total, eps)
    benchmark_centered = np.where(np.isfinite(f), f - benchmark_f[None, :, None], 0.0)
    benchmark_rmse = np.sqrt(
        np.sum(weights * benchmark_centered * benchmark_centered, axis=(0, 2))
        / np.maximum(weight_total, eps)
    )
    benchmark_l1 = (
        np.sum(weights * np.abs(benchmark_centered), axis=(0, 2))
        / np.maximum(weight_total, eps)
    )

    finite_actions = np.isfinite(a)
    zero_mask = finite_actions & (a <= eps)
    cap_mask = np.zeros_like(zero_mask, dtype=bool)
    if eta_hist is not None:
        eta = np.asarray(eta_hist, dtype=np.float64)
        if eta.shape != (a.shape[0], a.shape[2]):
            raise ValueError(f"eta_hist shape {eta.shape} must match (T, K)={(a.shape[0], a.shape[2])}.")
        cap_mask = finite_actions & (eta[:, None, :] > eps) & (a >= eta[:, None, :] - eps)
    interior_mask = finite_actions & (~zero_mask) & (~cap_mask)
    denom = max(int(finite_actions.sum()), 1)

    return {
        "demand_weighted_mean_f": float(np.nanmean(mean_f)),
        "demand_weighted_var_f": float(np.nanmean(var_f)),
        "demand_weighted_cv_f": float(np.nanmean(cv_f)),
        "planned_coverage_ratio_gap": float(np.nanmean(gap_f)),
        "demand_proportional_benchmark_rmse": float(np.nanmean(benchmark_rmse)),
        "demand_proportional_benchmark_l1": float(np.nanmean(benchmark_l1)),
        "action_zero_fraction": float(zero_mask.sum() / denom),
        "action_interior_fraction": float(interior_mask.sum() / denom),
        "action_eta_cap_fraction": float(cap_mask.sum() / denom),
        "mean_f_per_path": mean_f,
        "cv_f_per_path": cv_f,
        "coverage_gap_per_path": gap_f,
        "benchmark_rmse_per_path": benchmark_rmse,
    }
