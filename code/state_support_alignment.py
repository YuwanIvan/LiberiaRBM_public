"""Align evaluation states with the fine-grid networks that consume them.

Training support is logged by computational interval.  Fine-grid evaluation
already has one state per computational interval, while periodic-review E3
evaluation stores the within-review virtual states used to query each network.
This module keeps that mapping explicit and testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


ALIGNMENT_SCHEMA_VERSION = "fine-interval-state-support-v2"


def fine_intervals_for_review(review_interval: int, fine_per_review: int) -> tuple[int, ...]:
    if int(review_interval) < 0:
        raise ValueError("review_interval must be nonnegative")
    if int(fine_per_review) <= 0:
        raise ValueError("fine_per_review must be positive")
    start = int(review_interval) * int(fine_per_review)
    return tuple(range(start, start + int(fine_per_review)))


def _variables(z: np.ndarray, q: np.ndarray) -> list[tuple[str, np.ndarray]]:
    values: list[tuple[str, np.ndarray]] = [("Q", q)]
    for idx in range(z.shape[1]):
        values.append((f"Z_{idx}", z[:, idx]))
    if z.shape[1] > 1:
        cha = z[:, 1:]
        values.extend(
            [
                ("aggregate_CHA_inventory", np.sum(cha, axis=1)),
                ("minimum_CHA_inventory", np.min(cha, axis=1)),
                ("maximum_CHA_inventory", np.max(cha, axis=1)),
            ]
        )
    return values


def _summary_row(
    *,
    training_interval: int,
    source: str,
    variable: str,
    values: np.ndarray,
    envelope: Mapping[tuple[int, str], Mapping[str, Any]],
    review_interval: int | None,
    fine_offset: int | None,
    alignment_mode: str,
) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    env = envelope.get((int(training_interval), str(variable)))
    if finite.size == 0:
        summary = {key: np.nan for key in ("min", "p01", "p05", "p50", "p95", "p99", "max")}
        count = 0
        outside_p01_p99 = np.nan
        outside_min_max = np.nan
        mean_distance_p01_p99 = np.nan
        max_distance_p01_p99 = np.nan
        mean_distance_min_max = np.nan
        max_distance_min_max = np.nan
        low_q_coverage = np.nan
        stockout_boundary_coverage = np.nan
    else:
        summary = {
            "min": float(np.min(finite)),
            "p01": float(np.percentile(finite, 1)),
            "p05": float(np.percentile(finite, 5)),
            "p50": float(np.percentile(finite, 50)),
            "p95": float(np.percentile(finite, 95)),
            "p99": float(np.percentile(finite, 99)),
            "max": float(np.max(finite)),
        }
        count = int(finite.size)
        if env is not None:
            env_min = float(env.get("min", np.nan))
            env_max = float(env.get("max", np.nan))
            env_p01 = float(env.get("p01", np.nan))
            env_p99 = float(env.get("p99", np.nan))
            outside_p01_p99 = float(np.mean((finite < env_p01) | (finite > env_p99)))
            outside_min_max = float(np.mean((finite < env_min) | (finite > env_max)))
            p_span = max(env_p99 - env_p01, 1e-12)
            mm_span = max(env_max - env_min, 1e-12)
            p_distance = (
                np.maximum(env_p01 - finite, 0.0)
                + np.maximum(finite - env_p99, 0.0)
            ) / p_span
            mm_distance = (
                np.maximum(env_min - finite, 0.0)
                + np.maximum(finite - env_max, 0.0)
            ) / mm_span
            mean_distance_p01_p99 = float(np.mean(p_distance))
            max_distance_p01_p99 = float(np.max(p_distance))
            mean_distance_min_max = float(np.mean(mm_distance))
            max_distance_min_max = float(np.max(mm_distance))
        else:
            outside_p01_p99 = np.nan
            outside_min_max = np.nan
            mean_distance_p01_p99 = np.nan
            max_distance_p01_p99 = np.nan
            mean_distance_min_max = np.nan
            max_distance_min_max = np.nan
        if variable == "Q":
            env_p05 = float(env.get("p05", np.nan)) if env is not None else np.nan
            low_q_coverage = float(np.mean(finite <= env_p05)) if np.isfinite(env_p05) else np.nan
            stockout_boundary_coverage = float(np.mean(finite <= 1e-8))
        else:
            low_q_coverage = np.nan
            stockout_boundary_coverage = np.nan
    return {
        # Keep the legacy column name because formal validators and aggregators
        # use it as the training-envelope key.
        "interval": int(training_interval),
        "training_interval": int(training_interval),
        "review_interval": "" if review_interval is None else int(review_interval),
        "fine_offset": "" if fine_offset is None else int(fine_offset),
        "alignment_mode": str(alignment_mode),
        "alignment_schema_version": ALIGNMENT_SCHEMA_VERSION,
        "source": str(source),
        "variable": str(variable),
        "count": count,
        **summary,
        "outside_training_p01_p99_fraction": outside_p01_p99,
        "outside_training_min_max_fraction": outside_min_max,
        "mean_normalized_distance_outside_training_p01_p99": mean_distance_p01_p99,
        "max_normalized_distance_outside_training_p01_p99": max_distance_p01_p99,
        "mean_normalized_distance_outside_training_min_max": mean_distance_min_max,
        "max_normalized_distance_outside_training_min_max": max_distance_min_max,
        "low_Q_coverage": low_q_coverage,
        "stockout_boundary_coverage": stockout_boundary_coverage,
    }


def fine_grid_state_support_rows(
    result: Mapping[str, Any],
    *,
    source: str,
    envelope: Mapping[tuple[int, str], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not result or "z_hist" not in result or "i_hist" not in result:
        return []
    z_hist = np.asarray(result["z_hist"])
    i_hist = np.asarray(result["i_hist"])
    if z_hist.ndim != 3 or i_hist.ndim != 2:
        return []
    horizon = min(z_hist.shape[0], i_hist.shape[0]) - 1
    rows: list[dict[str, Any]] = []
    for interval in range(max(horizon, 0)):
        for variable, values in _variables(z_hist[interval], i_hist[interval]):
            rows.append(
                _summary_row(
                    training_interval=interval,
                    source=source,
                    variable=variable,
                    values=values,
                    envelope=envelope or {},
                    review_interval=None,
                    fine_offset=None,
                    alignment_mode="fine_grid_exact_interval",
                )
            )
    return rows


def periodic_state_support_rows(
    result: Mapping[str, Any],
    *,
    num_fine_intervals: int,
    source: str,
    envelope: Mapping[tuple[int, str], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Map every periodic-review query state to its consuming fine network."""

    if not result or "z_hist" not in result or "i_hist" not in result:
        return []
    z_hist = np.asarray(result["z_hist"])
    i_hist = np.asarray(result["i_hist"])
    if z_hist.ndim != 3 or i_hist.ndim != 2:
        return []
    review_intervals = min(z_hist.shape[0], i_hist.shape[0]) - 1
    if review_intervals <= 0 or int(num_fine_intervals) % review_intervals != 0:
        raise ValueError(
            "Fine-grid interval count must be an integer multiple of periodic-review intervals"
        )
    fine_per_review = int(num_fine_intervals) // review_intervals
    virtual_z = np.asarray(result.get("virtual_z_hist", []))
    virtual_q = np.asarray(result.get("virtual_q_hist", []))
    virtual_available = (
        virtual_z.ndim == 4
        and virtual_q.ndim == 3
        and virtual_z.shape[:2] == (review_intervals, fine_per_review + 1)
        and virtual_q.shape[:2] == (review_intervals, fine_per_review + 1)
    )
    rows: list[dict[str, Any]] = []
    for review_interval in range(review_intervals):
        for fine_offset, training_interval in enumerate(
            fine_intervals_for_review(review_interval, fine_per_review)
        ):
            if virtual_available:
                z = virtual_z[review_interval, fine_offset]
                q = virtual_q[review_interval, fine_offset]
                alignment_mode = "periodic_virtual_state_exact_fine_interval"
            else:
                z = z_hist[review_interval]
                q = i_hist[review_interval]
                alignment_mode = "periodic_start_state_repeated_across_fine_intervals"
            for variable, values in _variables(z, q):
                rows.append(
                    _summary_row(
                        training_interval=training_interval,
                        source=source,
                        variable=variable,
                        values=values,
                        envelope=envelope or {},
                        review_interval=review_interval,
                        fine_offset=fine_offset,
                        alignment_mode=alignment_mode,
                    )
                )
    return rows
