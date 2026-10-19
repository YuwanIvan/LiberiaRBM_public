"""Pure builders for source tables used by the M=110 presentation workflow."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    if finite.size == 1:
        return float(finite[0]), 0.0
    return float(np.mean(finite)), float(np.std(finite, ddof=1) / np.sqrt(finite.size))


def build_periodic_location_diagnostic_rows(
    results_by_policy: Mapping[str, Mapping[str, object]],
    referral_probability: object,
    distance: object,
    *,
    eps: float = 1e-12,
) -> dict[str, list[dict[str, float | int | str]]]:
    """Build CHA fill/travel rows from already-simulated periodic histories."""

    if not results_by_policy:
        raise ValueError("Periodic location diagnostics require at least one policy.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")

    p = np.asarray(referral_probability, dtype=float).reshape(-1)
    distance_array = np.asarray(distance, dtype=float).reshape(-1)
    if distance_array.size != p.size + 1:
        raise ValueError("distance must contain HF plus one entry per CHA referral probability.")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(distance_array)):
        raise ValueError("Referral probabilities and distances must be finite.")

    cha_order = sorted(
        range(p.size),
        key=lambda idx: (-distance_array[idx + 1], idx + 1),
    )
    local_rows: list[dict[str, float | int | str]] = []
    travel_rows: list[dict[str, float | int | str]] = []

    for policy_name, result in results_by_policy.items():
        if not str(policy_name).strip():
            raise ValueError("Policy names must be nonempty.")
        demand = np.asarray(result["d_hist"], dtype=float)
        unmet = np.asarray(result["l_hist"], dtype=float)
        if demand.shape != unmet.shape or demand.ndim != 3:
            raise ValueError(f"{policy_name}: d_hist/l_hist must share [week,path,location] shape.")
        if demand.shape[2] != distance_array.size:
            raise ValueError(f"{policy_name}: location dimension disagrees with distance.")
        if demand.shape[1] <= 0:
            raise ValueError(f"{policy_name}: periodic histories contain no paths.")
        if not np.all(np.isfinite(demand)) or not np.all(np.isfinite(unmet)):
            raise ValueError(f"{policy_name}: periodic histories contain nonfinite values.")

        cha_demand = demand[:, :, 1:]
        cha_unmet = unmet[:, :, 1:]
        cha_service = np.maximum(cha_demand - cha_unmet, 0.0)
        demand_by_path_location = cha_demand.sum(axis=0)
        service_by_path_location = cha_service.sum(axis=0)
        local_fill_by_path_location = service_by_path_location / np.maximum(
            demand_by_path_location,
            eps,
        )

        total_cha_demand_by_path = cha_demand.sum(axis=(0, 2))
        referral_travel_by_path_location = (
            cha_unmet * p[None, None, :] * distance_array[None, None, 1:]
        ).sum(axis=0) / np.maximum(total_cha_demand_by_path[:, None], eps)

        for distance_order, cha_idx in enumerate(cha_order, start=1):
            local_mean, local_se = _mean_se(local_fill_by_path_location[:, cha_idx])
            travel_mean, travel_se = _mean_se(referral_travel_by_path_location[:, cha_idx])
            common: dict[str, float | int | str] = {
                "policy": str(policy_name),
                "distance_order": distance_order,
                "cha_index": cha_idx + 1,
                "distance_miles": float(distance_array[cha_idx + 1]),
            }
            local_rows.append(
                {
                    **common,
                    "local_fill_mean": local_mean,
                    "local_fill_se": local_se,
                }
            )
            travel_rows.append(
                {
                    **common,
                    "travel_burden_contribution_mean": travel_mean,
                    "travel_burden_contribution_se": travel_se,
                }
            )

    return {
        "cha_local_fill_by_location.csv": local_rows,
        "referral_travel_burden_by_cha.csv": travel_rows,
    }
