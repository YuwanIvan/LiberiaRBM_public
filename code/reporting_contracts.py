"""Publication-facing names and pure reporting aggregations.

This module deliberately contains no simulation or policy-selection logic.  It
turns histories produced by the existing evaluators into compact, human-facing
tables while retaining the internal policy and location identities needed for
provenance.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


INVENTORY_SCHEMA_VERSION = "inventory-trajectory-by-policy-location-v1"

# The one canonical source for human-facing policy names.  Internal policy keys
# remain unchanged in numerical outputs and provenance.
POLICY_DISPLAY_NAMES: dict[str, str] = {
    "E3_mean_rollout": "Our Policy",
    "Status_Quo": "80/20",
    "Static": "Static",
    "Scaled_Static": "Fairness Only",
    "Myopic": "Myopic",
    "Fluid_Optimal": "Fluid-optimal",
    "E0_frozen_average": "E0 diagnostic",
}

POLICY_ROLES: dict[str, str] = {
    "E3_mean_rollout": "primary",
    "Status_Quo": "benchmark",
    "Static": "benchmark",
    "Scaled_Static": "benchmark",
    "Myopic": "benchmark",
    "Fluid_Optimal": "benchmark",
    "E0_frozen_average": "diagnostic",
}

BASELINE_REPORT_POLICIES = (
    "E3_mean_rollout",
    "Status_Quo",
    "Static",
    "Scaled_Static",
    "Myopic",
    "Fluid_Optimal",
)
SENSITIVITY_REPORT_POLICIES = (
    "E3_mean_rollout",
    "Scaled_Static",
    "Fluid_Optimal",
)
INVENTORY_0P9_EXTRA_POLICIES = ("Static",)
DIAGNOSTIC_REPORT_POLICIES = ("E0_frozen_average",)


@dataclass(frozen=True)
class InventoryLocation:
    location_type: str
    location_index: int
    location_key: str
    location_display_name: str
    distance_miles: float | None
    panel_order: int


# Internal downstream indices are untouched.  Only panel_order determines the
# publication layout:
# Warehouse | HF   | CHA1
# CHA5      | CHA6 | CHA2
# CHA3      | CHA7 | CHA4
INVENTORY_LOCATIONS: tuple[InventoryLocation, ...] = (
    InventoryLocation("warehouse", -1, "warehouse", "Warehouse", None, 0),
    InventoryLocation("hf", 0, "hf", "HF", None, 1),
    InventoryLocation("cha", 1, "cha1", "CHA 1", 11.0, 2),
    InventoryLocation("cha", 5, "cha5", "CHA 5", 5.0, 3),
    InventoryLocation("cha", 6, "cha6", "CHA 6", 5.0, 4),
    InventoryLocation("cha", 2, "cha2", "CHA 2", 3.8, 5),
    InventoryLocation("cha", 3, "cha3", "CHA 3", 3.0, 6),
    InventoryLocation("cha", 7, "cha7", "CHA 7", 0.5, 7),
    InventoryLocation("cha", 4, "cha4", "CHA 4", 0.25, 8),
)

INVENTORY_COLUMNS = (
    "schema_version",
    "setting_id",
    "evaluation_run_id",
    "training_run_id",
    "stream_id",
    "policy_key",
    "internal_policy_name",
    "policy_display_name",
    "policy_role",
    "location_type",
    "location_index",
    "location_key",
    "location_display_name",
    "distance_miles",
    "panel_order",
    "week",
    "state_timing",
    "inventory_mean",
    "inventory_sample_sd",
    "inventory_se",
    "inventory_ci95_lower",
    "inventory_ci95_upper",
    "num_paths",
    "evaluation_mode",
    "evaluation_bank_id",
    "evaluation_bank_sha256",
    "training_branch",
    "training_commit",
    "config_hash",
    "checkpoint_certification_sha256",
    "evaluation_branch",
    "evaluation_commit",
)


def policy_display_name(policy_key: str) -> str:
    """Return the exact publication-facing name for a known internal policy."""

    try:
        return POLICY_DISPLAY_NAMES[str(policy_key)]
    except KeyError as exc:
        raise KeyError(f"No canonical display name for policy {policy_key!r}") from exc


def policy_role(policy_key: str) -> str:
    """Return primary, benchmark, or diagnostic without changing policy logic."""

    try:
        return POLICY_ROLES[str(policy_key)]
    except KeyError as exc:
        raise KeyError(f"No canonical role for policy {policy_key!r}") from exc


def add_objective_gap_vs_our_policy(
    frame: Any,
    *,
    policy_column: str = "Policy",
    objective_column: str = "Total_Objective_Mean",
) -> Any:
    """Add an absolute objective difference with an explicit reference.

    The returned column is ``objective_gap_vs_our_policy`` and is defined as
    ``row objective - Our Policy objective``.  Lower objectives are better, so
    a positive value means the row is worse than Our Policy.  The legacy
    order-dependent ``Performance_Gap_Percent`` column is not changed.
    """

    matches = frame.loc[frame[policy_column] == "E3_mean_rollout", objective_column]
    if len(matches) != 1:
        raise ValueError("Expected exactly one E3_mean_rollout row for the objective-gap reference")
    reference = float(matches.iloc[0])
    frame["objective_gap_vs_our_policy"] = frame[objective_column].astype(float) - reference
    return frame


def add_operational_cost_summary(frame: Any, costs_by_policy: Mapping[str, Mapping[str, Any]]) -> Any:
    """Add pathwise-derived operational mean/SE as reporting-only fields."""

    means: dict[str, float] = {}
    ses: dict[str, float] = {}
    for policy_key, costs in costs_by_policy.items():
        untreated = np.asarray(
            costs.get("untreated_costs", costs.get("stockout_costs")), dtype=np.float64
        )
        travel = np.asarray(
            costs.get("referral_travel_costs", costs.get("travel_costs")), dtype=np.float64
        )
        if untreated.shape != travel.shape or untreated.ndim != 1 or untreated.size == 0:
            raise ValueError(f"{policy_key}: invalid pathwise operational-cost components")
        values = untreated + travel
        means[policy_key] = float(np.mean(values, dtype=np.float64))
        ses[policy_key] = (
            float(np.std(values, ddof=1, dtype=np.float64) / math.sqrt(values.size))
            if values.size > 1
            else math.nan
        )
    frame["Operational_Cost_Mean"] = frame["Policy"].map(means)
    frame["Operational_Cost_SE"] = frame["Policy"].map(ses)
    if frame["Operational_Cost_Mean"].isna().any():
        raise ValueError("Operational-cost reporting fields could not be aligned to policy rows")
    return frame


def _summary(values: np.ndarray) -> tuple[float, float, float, float, float, int]:
    """Return mean, sample SD, SE, and untrimmed normal 95% interval."""

    path_values = np.asarray(values, dtype=np.float64).reshape(-1)
    count = int(path_values.size)
    if count == 0:
        raise ValueError("Inventory aggregation requires at least one evaluation path")
    if not np.all(np.isfinite(path_values)):
        raise ValueError("Inventory histories must contain only finite values")
    mean = float(np.mean(path_values, dtype=np.float64))
    if count == 1:
        sample_sd = math.nan
        se = math.nan
        lower = math.nan
        upper = math.nan
    else:
        sample_sd = float(np.std(path_values, ddof=1, dtype=np.float64))
        se = sample_sd / math.sqrt(count)
        lower = mean - 1.96 * se
        upper = mean + 1.96 * se
    return mean, sample_sd, se, lower, upper, count


def _state_timing(location: InventoryLocation, week: int) -> str:
    if week == 0:
        return "initial_state_before_week_1_shipment"
    if location.location_type == "warehouse":
        return "warehouse_remaining_after_outbound_allocation_through_week_t"
    if location.location_type == "hf":
        return "end_of_week_after_shipment_own_demand_and_referral_demand"
    return "end_of_week_after_shipment_and_local_demand"


def aggregate_inventory_trajectories(
    results_by_policy: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate periodic-review warehouse and downstream inventory histories.

    ``i_hist[t, path]`` is central warehouse inventory remaining after outbound
    allocation through week ``t``.  ``z_hist[t, path, 0]`` is HF end-of-week
    on-hand inventory after shipment, own demand, and referral demand;
    ``z_hist[t, path, 1:8]`` contains CHA end-of-week on-hand inventory after
    shipment and local demand.  At week 0 both histories are the initial state
    before the week-1 shipment.

    This helper never modifies or saves the raw histories.  Every path slice is
    converted to float64 before its mean, sample SD (ddof=1), SE, and untrimmed
    normal 95% interval are computed.
    """

    rows: list[dict[str, Any]] = []
    for policy_key, result in results_by_policy.items():
        display = policy_display_name(policy_key)
        role = policy_role(policy_key)
        warehouse = np.asarray(result["i_hist"])
        downstream = np.asarray(result["z_hist"])
        if warehouse.ndim != 2:
            raise ValueError(f"{policy_key}: i_hist must have shape (week, path)")
        if downstream.ndim != 3 or downstream.shape[2] != 8:
            raise ValueError(f"{policy_key}: z_hist must have shape (week, path, 8)")
        if warehouse.shape[:2] != downstream.shape[:2]:
            raise ValueError(f"{policy_key}: i_hist and z_hist week/path shapes differ")

        for location in INVENTORY_LOCATIONS:
            source = warehouse if location.location_type == "warehouse" else downstream[:, :, location.location_index]
            for week in range(source.shape[0]):
                mean, sample_sd, se, lower, upper, count = _summary(
                    np.asarray(source[week], dtype=np.float64)
                )
                rows.append(
                    {
                        "schema_version": INVENTORY_SCHEMA_VERSION,
                        "setting_id": metadata.get("setting_id", "NA"),
                        "evaluation_run_id": metadata.get("evaluation_run_id", "NA"),
                        "training_run_id": metadata.get("training_run_id", "NA"),
                        "stream_id": metadata.get("stream_id", "NA"),
                        "policy_key": policy_key,
                        "internal_policy_name": policy_key,
                        "policy_display_name": display,
                        "policy_role": role,
                        "location_type": location.location_type,
                        "location_index": location.location_index,
                        "location_key": location.location_key,
                        "location_display_name": location.location_display_name,
                        "distance_miles": location.distance_miles,
                        "panel_order": location.panel_order,
                        "week": week,
                        "state_timing": _state_timing(location, week),
                        "inventory_mean": mean,
                        "inventory_sample_sd": sample_sd,
                        "inventory_se": se,
                        "inventory_ci95_lower": lower,
                        "inventory_ci95_upper": upper,
                        "num_paths": count,
                        "evaluation_mode": metadata.get("evaluation_mode", "periodic_review"),
                        "evaluation_bank_id": metadata.get("evaluation_bank_id", "NA"),
                        "evaluation_bank_sha256": metadata.get("evaluation_bank_sha256", "NA"),
                        "training_branch": metadata.get("training_branch", "NA"),
                        "training_commit": metadata.get("training_commit", "NA"),
                        "config_hash": metadata.get("config_hash", "NA"),
                        "checkpoint_certification_sha256": metadata.get(
                            "checkpoint_certification_sha256", "NA"
                        ),
                        "evaluation_branch": metadata.get("evaluation_branch", "NA"),
                        "evaluation_commit": metadata.get("evaluation_commit", "NA"),
                    }
                )
    return rows


def write_inventory_trajectory_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write a new unified inventory CSV without overwriting an existing file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in INVENTORY_COLUMNS})
    return output
