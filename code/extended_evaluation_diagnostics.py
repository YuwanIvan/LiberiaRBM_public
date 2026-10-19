"""Pure, opt-in diagnostics for periodic-review evaluation histories.

This module does not simulate, select policies, or load checkpoints.  It only
validates histories already produced by the periodic evaluator, computes the
CHA local-unmet mismatch diagnostics, builds compact location tables,
and writes versioned additive artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "extended-evaluation-diagnostics-v1"
STATE_HISTORY_SCHEMA_VERSION = "periodic-review-state-histories-v1"
MISMATCH_SCHEMA_VERSION = "cha-local-unmet-mismatch-v2"
LOCATION_SCHEMA_VERSION = "periodic-review-location-metrics-v2"
VALIDATION_SCHEMA_VERSION = "extended-evaluation-validation-v3"
MANIFEST_SCHEMA_VERSION = "extended-evaluation-artifact-manifest-v1"
DEFAULT_EPS = 1e-12
RETAINED_POLICIES = ("E3_mean_rollout", "Scaled_Static")
ATTRIBUTION_LABEL = "evaluator-defined proportional referral-service attribution"
ATTRIBUTION_FORMULA = (
    "component_hf_load / total_hf_load * total_hf_service, bounded by component_hf_load"
)

MISMATCH_COLUMNS = (
    "schema_version",
    "setting_id",
    "evaluation_run_id",
    "stream_id",
    "evaluation_bank_id",
    "evaluation_bank_sha256",
    "evaluation_commit",
    "policy",
    "num_weeks",
    "num_paths",
    "num_chas",
    "eps",
    "MismatchFreq",
    "S_bar",
    "S_positive",
    "R_pool",
    "R_U_positive_mean",
    "positive_mismatch_events",
    "positive_unmet_events",
    "path_week_count",
    "same_cha_invariant_violation_count",
)

COMMON_LOCATION_COLUMNS = (
    "schema_version",
    "setting_id",
    "evaluation_run_id",
    "stream_id",
    "evaluation_bank_id",
    "evaluation_bank_sha256",
    "evaluation_commit",
    "policy",
    "location_index",
    "location_type",
    "distance_to_hf",
    "week",
    "num_paths",
    "state_timing",
    "demand_semantics",
)

WEEKLY_LOCATION_COLUMNS = COMMON_LOCATION_COLUMNS + (
    "shipment_mean",
    "shipment_se",
    "demand_mean",
    "demand_se",
    "own_hf_demand_mean",
    "own_hf_demand_se",
    "referral_inflow_mean",
    "referral_inflow_se",
    "hf_load_mean",
    "hf_load_se",
    "end_inventory_mean",
    "end_inventory_se",
    "local_service_mean",
    "local_service_se",
    "local_unmet_mean",
    "local_unmet_se",
    "local_fill",
    "referral_generated_mean",
    "referral_generated_se",
    "own_hf_service_attributed_mean",
    "own_hf_service_attributed_se",
    "referral_served_attributed_mean",
    "referral_served_attributed_se",
    "mismatch_inventory_participation_incidence",
    "mismatch_unmet_participation_incidence",
    "mismatch_participation_incidence",
    "attribution_method",
)

LOCATION_TOTAL_COLUMNS = COMMON_LOCATION_COLUMNS + (
    "total_shipment_mean",
    "total_shipment_se",
    "total_demand_mean",
    "total_demand_se",
    "total_own_hf_demand_mean",
    "total_own_hf_demand_se",
    "total_referral_inflow_mean",
    "total_referral_inflow_se",
    "total_hf_load_mean",
    "total_hf_load_se",
    "terminal_leftover_mean",
    "terminal_leftover_se",
    "total_local_service_mean",
    "total_local_service_se",
    "total_local_unmet_mean",
    "total_local_unmet_se",
    "local_fill",
    "total_referral_generated_mean",
    "total_referral_generated_se",
    "total_own_hf_service_attributed_mean",
    "total_own_hf_service_attributed_se",
    "total_referral_served_attributed_mean",
    "total_referral_served_attributed_se",
    "mismatch_inventory_participation_incidence",
    "mismatch_unmet_participation_incidence",
    "mismatch_participation_incidence",
    "attribution_method",
)

REQUIRED_METADATA = (
    "setting_id",
    "evaluation_run_id",
    "stream_id",
    "evaluation_bank_id",
    "evaluation_bank_sha256",
    "evaluation_commit",
)


class ExtendedEvaluationError(RuntimeError):
    """Raised when an extended-evaluation contract fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExtendedEvaluationError(message)


def _array(result: Mapping[str, Any], key: str) -> np.ndarray:
    require(key in result, f"Missing required periodic history: {key}")
    value = np.asarray(result[key])
    require(np.issubdtype(value.dtype, np.number), f"{key} must be numeric")
    require(np.all(np.isfinite(value)), f"{key} contains nonfinite values")
    return value


def validate_periodic_histories(
    result: Mapping[str, Any], *, policy: str = "unknown", eps: float = DEFAULT_EPS
) -> dict[str, Any]:
    """Validate shape, timing, nonnegativity, and the same-CHA invariant."""

    require(math.isfinite(eps) and eps >= 0.0, "eps must be finite and nonnegative")
    q_hist = _array(result, "q_hist")
    d_hist = _array(result, "d_hist")
    l_hist = _array(result, "l_hist")
    z_hist = _array(result, "z_hist")
    require(q_hist.ndim == 3, f"{policy}: q_hist must have [week,path,location] shape")
    require(d_hist.shape == q_hist.shape, f"{policy}: d_hist shape differs from q_hist")
    require(l_hist.shape == q_hist.shape, f"{policy}: l_hist shape differs from q_hist")
    weeks, paths, locations = q_hist.shape
    require(weeks > 0 and paths > 0 and locations >= 2, f"{policy}: invalid history dimensions")
    require(
        z_hist.shape == (weeks + 1, paths, locations),
        f"{policy}: z_hist must have shape (weeks+1,paths,locations)",
    )
    mismatch_validation = validate_mismatch_histories(z_hist, l_hist, policy=policy, eps=eps)
    minima = {
        "q_hist": float(np.min(q_hist)),
        "d_hist": float(np.min(d_hist)),
        "l_hist": float(np.min(l_hist)),
        "z_hist": float(np.min(z_hist)),
    }
    for key, minimum in minima.items():
        require(minimum >= -eps, f"{policy}: {key} violates nonnegativity: {minimum}")

    return {
        "num_weeks": int(weeks),
        "num_paths": int(paths),
        "num_locations": int(locations),
        "num_chas": int(locations - 1),
        "eps": float(eps),
        "minima": minima,
        "same_cha_invariant_violation_count": mismatch_validation["same_cha_invariant_violation_count"],
        "timing": "l_hist[t] aligned with end-of-week z_hist[t+1]",
    }


def validate_mismatch_histories(
    z_hist: np.ndarray, l_hist: np.ndarray, *, policy: str = "unknown", eps: float = DEFAULT_EPS
) -> dict[str, Any]:
    """Validate only the two histories needed by the mismatch definition."""

    require(math.isfinite(eps) and eps >= 0.0, "eps must be finite and nonnegative")
    z_hist = np.asarray(z_hist)
    l_hist = np.asarray(l_hist)
    require(z_hist.ndim == 3 and l_hist.ndim == 3, f"{policy}: mismatch histories must be 3D")
    weeks, paths, locations = l_hist.shape
    require(weeks > 0 and paths > 0 and locations >= 2, f"{policy}: invalid mismatch dimensions")
    require(z_hist.shape == (weeks + 1, paths, locations), f"{policy}: mismatch z/l timing or shape differs")
    for label, values in (("z_hist", z_hist), ("l_hist", l_hist)):
        require(np.issubdtype(values.dtype, np.number) and np.all(np.isfinite(values)), f"{policy}: invalid {label}")
        require(float(np.min(values)) >= -eps, f"{policy}: {label} violates nonnegativity")
    cha_inventory = z_hist[1:, :, 1:]
    cha_unmet = l_hist[:, :, 1:]
    violation = (cha_inventory > eps) & (cha_unmet > eps)
    if np.any(violation):
        week, path, cha_zero = (int(v) for v in np.argwhere(violation)[0])
        raise ExtendedEvaluationError(
            f"{policy}: same-CHA invariant violation at week={week + 1}, path={path}, "
            f"location_index={cha_zero + 1}, inventory={cha_inventory[week, path, cha_zero]:.17g}, "
            f"local_unmet={cha_unmet[week, path, cha_zero]:.17g}, eps={eps:.17g}"
        )
    return {"num_weeks": weeks, "num_paths": paths, "num_chas": locations - 1, "eps": float(eps), "same_cha_invariant_violation_count": 0}


def compute_mismatch_metrics(
    z_hist: np.ndarray, l_hist: np.ndarray, *, policy: str = "unknown", eps: float = DEFAULT_EPS
) -> dict[str, Any]:
    """Compute mismatch incidence from eps masks and severity from raw nonnegative sums."""

    validation = validate_mismatch_histories(z_hist, l_hist, policy=policy, eps=eps)
    z = np.asarray(z_hist, dtype=np.float64)[1:, :, 1:]
    l = np.asarray(l_hist, dtype=np.float64)[:, :, 1:]
    inventory_total = np.maximum(z, 0.0).sum(axis=2, dtype=np.float64)
    unmet_total = np.maximum(l, 0.0).sum(axis=2, dtype=np.float64)
    mismatch = (z > eps).any(axis=2) & (l > eps).any(axis=2)
    severity = np.minimum(inventory_total, unmet_total)
    mismatch_count = int(np.count_nonzero(mismatch))
    unmet_mask = unmet_total > 0.0
    unmet_count = int(np.count_nonzero(unmet_mask))
    total_unmet = float(unmet_total.sum(dtype=np.float64))
    mismatch_frequency = float(np.mean(mismatch, dtype=np.float64))
    s_bar = float(np.mean(severity, dtype=np.float64))
    s_positive = float(np.mean(severity[mismatch], dtype=np.float64)) if mismatch_count else math.nan
    r_pool = float(severity.sum(dtype=np.float64) / total_unmet) if total_unmet > 0.0 else math.nan
    r_u_positive_mean = (
        float(np.mean(severity[unmet_mask] / unmet_total[unmet_mask], dtype=np.float64))
        if unmet_count
        else math.nan
    )
    require(0.0 <= mismatch_frequency <= 1.0, f"{policy}: MismatchFreq outside [0,1]")
    require(s_bar >= 0.0, f"{policy}: S_bar is negative")
    if math.isfinite(s_positive):
        require(s_positive >= 0.0, f"{policy}: S_positive is negative")
    for name, value in (("R_pool", r_pool), ("R_U_positive_mean", r_u_positive_mean)):
        if math.isfinite(value):
            require(-eps <= value <= 1.0 + eps, f"{policy}: {name} outside [0,1]")
    return {
        "MismatchFreq": mismatch_frequency,
        "S_bar": s_bar,
        "S_positive": s_positive,
        "R_pool": r_pool,
        "R_U_positive_mean": r_u_positive_mean,
        "positive_mismatch_events": mismatch_count,
        "positive_unmet_events": unmet_count,
        "path_week_count": int(mismatch.size),
        **{key: validation[key] for key in ("num_weeks", "num_paths", "num_chas", "eps")},
        "same_cha_invariant_violation_count": 0,
    }


def _summary(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    require(values.size > 0 and np.all(np.isfinite(values)), "Summary input must be finite and nonempty")
    mean = float(np.mean(values, dtype=np.float64))
    se = float(np.std(values, ddof=1, dtype=np.float64) / math.sqrt(values.size)) if values.size > 1 else math.nan
    return mean, se


def _path_mean_fill(service: np.ndarray, demand: np.ndarray, *, eps: float) -> float:
    """Accepted CHA convention: mean of path-level ratios, zero demand -> zero."""

    service_by_path = np.asarray(service, dtype=np.float64).sum(axis=0)
    demand_by_path = np.asarray(demand, dtype=np.float64).sum(axis=0)
    return float(np.mean(service_by_path / np.maximum(demand_by_path, eps), dtype=np.float64))


def _location_base(metadata: Mapping[str, Any], policy: str, location: int, distance: np.ndarray) -> dict[str, Any]:
    return {
        "schema_version": LOCATION_SCHEMA_VERSION,
        **{key: metadata[key] for key in REQUIRED_METADATA},
        "policy": policy,
        "location_index": location,
        "location_type": "hf" if location == 0 else "cha",
        "distance_to_hf": "" if location == 0 else float(distance[location]),
        "demand_semantics": "location-native demand: HF own demand only; CHA local demand",
    }


def build_location_metrics(
    result: Mapping[str, Any],
    metadata: Mapping[str, Any],
    referral_probabilities: Sequence[float],
    distances: Sequence[float],
    *,
    policy: str,
    eps: float = DEFAULT_EPS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build aggregate and weekly HF/CHA tables without retaining q_hist/d_hist."""

    validation = validate_periodic_histories(result, policy=policy, eps=eps)
    for key in REQUIRED_METADATA:
        require(metadata.get(key) not in (None, ""), f"Missing extended metadata: {key}")
    q = np.maximum(np.asarray(result["q_hist"], dtype=np.float64), 0.0)
    d = np.maximum(np.asarray(result["d_hist"], dtype=np.float64), 0.0)
    l = np.maximum(np.asarray(result["l_hist"], dtype=np.float64), 0.0)
    z = np.maximum(np.asarray(result["z_hist"], dtype=np.float64), 0.0)
    weeks, paths, locations = q.shape
    p = np.asarray(referral_probabilities, dtype=np.float64).reshape(-1)
    distance = np.asarray(distances, dtype=np.float64).reshape(-1)
    require(p.shape == (locations - 1,), "Referral probabilities must have one value per CHA")
    require(distance.shape == (locations,), "Distances must have one value for HF and every CHA")
    require(np.all(np.isfinite(p)) and np.all((p >= 0.0) & (p <= 1.0)), "Invalid referral probabilities")
    require(np.all(np.isfinite(distance)) and np.all(distance >= 0.0), "Invalid distances")

    cha_demand = d[:, :, 1:]
    cha_unmet = l[:, :, 1:]
    cha_service = np.maximum(cha_demand - cha_unmet, 0.0)
    referrals = cha_unmet * p[None, None, :]
    hf_load = d[:, :, 0] + referrals.sum(axis=2)
    hf_short = l[:, :, 0]
    hf_service_total = np.minimum(np.maximum(hf_load - hf_short, 0.0), hf_load)
    hf_safe = np.maximum(hf_load, eps)
    own_hf_service = np.where(hf_load > eps, d[:, :, 0] / hf_safe * hf_service_total, 0.0)
    referral_service = np.where(
        hf_load[:, :, None] > eps,
        referrals / hf_safe[:, :, None] * hf_service_total[:, :, None],
        0.0,
    )
    own_hf_service = np.minimum(own_hf_service, d[:, :, 0])
    referral_service = np.minimum(referral_service, referrals)

    cha_end_inventory = z[1:, :, 1:]
    total_cha_inventory_positive = (cha_end_inventory > eps).any(axis=2)
    total_cha_unmet_positive = (cha_unmet > eps).any(axis=2)

    weekly_rows: list[dict[str, Any]] = []
    total_rows: list[dict[str, Any]] = []
    for location in range(locations):
        base = _location_base(metadata, policy, location, distance)
        is_hf = location == 0
        cha = location - 1
        shipment_total = q[:, :, location].sum(axis=0)
        demand_total = d[:, :, location].sum(axis=0)
        terminal = z[-1, :, location]
        total = {
            **base,
            "week": "all",
            "num_paths": paths,
            "state_timing": "aggregate_weeks_1_through_N; terminal_inventory_after_week_N",
        }
        for prefix, values in (
            ("total_shipment", shipment_total),
            ("total_demand", demand_total),
            ("terminal_leftover", terminal),
        ):
            total[f"{prefix}_mean"], total[f"{prefix}_se"] = _summary(values)
        if is_hf:
            for prefix, values in (
                ("total_own_hf_demand", d[:, :, 0].sum(axis=0)),
                ("total_referral_inflow", referrals.sum(axis=2).sum(axis=0)),
                ("total_hf_load", hf_load.sum(axis=0)),
            ):
                total[f"{prefix}_mean"], total[f"{prefix}_se"] = _summary(values)
            own_total = own_hf_service.sum(axis=0)
            total["total_own_hf_service_attributed_mean"], total["total_own_hf_service_attributed_se"] = _summary(own_total)
            total["attribution_method"] = ATTRIBUTION_LABEL
        else:
            service_total = cha_service[:, :, cha].sum(axis=0)
            unmet_total = cha_unmet[:, :, cha].sum(axis=0)
            referral_total = referrals[:, :, cha].sum(axis=0)
            referral_served_total = referral_service[:, :, cha].sum(axis=0)
            for prefix, values in (
                ("total_local_service", service_total),
                ("total_local_unmet", unmet_total),
                ("total_referral_generated", referral_total),
                ("total_referral_served_attributed", referral_served_total),
            ):
                total[f"{prefix}_mean"], total[f"{prefix}_se"] = _summary(values)
            total["local_fill"] = _path_mean_fill(cha_service[:, :, cha], cha_demand[:, :, cha], eps=eps)
            inventory_side = (cha_end_inventory[:, :, cha] > eps) & total_cha_unmet_positive
            unmet_side = (cha_unmet[:, :, cha] > eps) & total_cha_inventory_positive
            total["mismatch_inventory_participation_incidence"] = float(np.mean(inventory_side))
            total["mismatch_unmet_participation_incidence"] = float(np.mean(unmet_side))
            total["mismatch_participation_incidence"] = float(np.mean(inventory_side | unmet_side))
            total["attribution_method"] = ATTRIBUTION_LABEL
        total_rows.append({key: total.get(key, "") for key in LOCATION_TOTAL_COLUMNS})

        for week in range(weeks):
            row = {
                **base,
                "week": week + 1,
                "num_paths": paths,
                "state_timing": "shipment_before_demand; demand/service within week; inventory_end_of_week",
            }
            for prefix, values in (
                ("shipment", q[week, :, location]),
                ("demand", d[week, :, location]),
                ("end_inventory", z[week + 1, :, location]),
            ):
                row[f"{prefix}_mean"], row[f"{prefix}_se"] = _summary(values)
            if is_hf:
                for prefix, values in (
                    ("own_hf_demand", d[week, :, 0]),
                    ("referral_inflow", referrals[week].sum(axis=1)),
                    ("hf_load", hf_load[week]),
                ):
                    row[f"{prefix}_mean"], row[f"{prefix}_se"] = _summary(values)
                row["own_hf_service_attributed_mean"], row["own_hf_service_attributed_se"] = _summary(
                    own_hf_service[week]
                )
                row["attribution_method"] = ATTRIBUTION_LABEL
            else:
                for prefix, values in (
                    ("local_service", cha_service[week, :, cha]),
                    ("local_unmet", cha_unmet[week, :, cha]),
                    ("referral_generated", referrals[week, :, cha]),
                    ("referral_served_attributed", referral_service[week, :, cha]),
                ):
                    row[f"{prefix}_mean"], row[f"{prefix}_se"] = _summary(values)
                row["local_fill"] = _path_mean_fill(cha_service[week : week + 1, :, cha], cha_demand[week : week + 1, :, cha], eps=eps)
                inventory_side = (cha_end_inventory[week, :, cha] > eps) & total_cha_unmet_positive[week]
                unmet_side = (cha_unmet[week, :, cha] > eps) & total_cha_inventory_positive[week]
                row["mismatch_inventory_participation_incidence"] = float(np.mean(inventory_side))
                row["mismatch_unmet_participation_incidence"] = float(np.mean(unmet_side))
                row["mismatch_participation_incidence"] = float(np.mean(inventory_side | unmet_side))
                row["attribution_method"] = ATTRIBUTION_LABEL
            weekly_rows.append({key: row.get(key, "") for key in WEEKLY_LOCATION_COLUMNS})

    require(len(total_rows) == locations, f"{policy}: location total row count mismatch")
    require(len(weekly_rows) == weeks * locations, f"{policy}: weekly location row count mismatch")
    # Exact aggregate reconciliation for shipment and demand means.
    for row in total_rows:
        location = int(row["location_index"])
        weekly = [item for item in weekly_rows if int(item["location_index"]) == location]
        require(
            np.isclose(sum(float(item["shipment_mean"]) for item in weekly), float(row["total_shipment_mean"]), atol=1e-10),
            f"{policy}: shipment aggregate does not reconcile for location {location}",
        )
        require(
            np.isclose(sum(float(item["demand_mean"]) for item in weekly), float(row["total_demand_mean"]), atol=1e-10),
            f"{policy}: demand aggregate does not reconcile for location {location}",
        )
    require(validation["same_cha_invariant_violation_count"] == 0, f"{policy}: invariant validation failed")
    return total_rows, weekly_rows


def _atomic_path(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)


def _publish_no_replace(temp: Path, path: Path) -> None:
    """Publish atomically while preserving the no-overwrite contract."""

    try:
        os.link(temp, path)
    except FileExistsError as exc:
        raise ExtendedEvaluationError(f"Refusing to overwrite extended artifact: {path}") from exc
    finally:
        temp.unlink(missing_ok=True)


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    fd, temp = _atomic_path(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        _publish_no_replace(Path(temp), path)
    except Exception:
        Path(temp).unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd, temp = _atomic_path(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        _publish_no_replace(Path(temp), path)
    except Exception:
        Path(temp).unlink(missing_ok=True)
        raise


def _write_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    fd, temp = _atomic_path(path)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **payload)
        _publish_no_replace(Path(temp), path)
    except Exception:
        Path(temp).unlink(missing_ok=True)
        raise


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_extended_evaluation_artifacts(
    results_by_policy: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    metadata: Mapping[str, Any],
    referral_probabilities: Sequence[float],
    distances: Sequence[float],
    *,
    enabled: bool = False,
    policies: Sequence[str] = RETAINED_POLICIES,
    eps: float = DEFAULT_EPS,
) -> dict[str, str]:
    """Write additive artifacts; return immediately with no side effect when disabled."""

    if not enabled:
        return {}
    require(tuple(policies) == RETAINED_POLICIES, f"Raw retention policy set must be exactly {RETAINED_POLICIES}")
    for key in REQUIRED_METADATA:
        require(metadata.get(key) not in (None, ""), f"Missing extended metadata: {key}")
    missing = [policy for policy in policies if policy not in results_by_policy]
    require(not missing, f"Missing retained policy histories: {missing}")
    output = Path(output_dir)
    require(output.is_dir(), f"Evaluation output directory does not exist: {output}")

    mismatch_rows: list[dict[str, Any]] = []
    location_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "PASS",
        "eps": eps,
        "retained_policies": list(policies),
        "state_timing": "l_hist[t] aligned with end-of-week z_hist[t+1]",
        "attribution_label": ATTRIBUTION_LABEL,
        "attribution_formula": ATTRIBUTION_FORMULA,
        "demand_semantics": "Generic demand columns mean location-native demand: own HF demand at HF, local demand at CHA; HF load additionally includes referrals.",
        "mismatch_eps_treatment": "Incidence uses location-level values > eps. Severity S_bar/S_positive and R_pool/R_U_positive_mean use untruncated CHA inventory and local-unmet sums; validated negative roundoff within eps is clipped to zero, not positive values <= eps.",
        "location_mismatch_participation": {
            "inventory_side": "location inventory > eps and any CHA local unmet > eps",
            "unmet_side": "location local unmet > eps and any CHA inventory > eps",
            "combined": "inventory_side OR unmet_side",
            "severity_decomposition": "not computed",
        },
        "policies": {},
    }
    state_payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(STATE_HISTORY_SCHEMA_VERSION),
        "policies": np.asarray(policies),
        "eps": np.asarray(eps, dtype=np.float64),
    }
    for policy in policies:
        result = results_by_policy[policy]
        metrics = compute_mismatch_metrics(result["z_hist"], result["l_hist"], policy=policy, eps=eps)
        mismatch_rows.append(
            {
                "schema_version": MISMATCH_SCHEMA_VERSION,
                **{key: metadata[key] for key in REQUIRED_METADATA},
                "policy": policy,
                **metrics,
            }
        )
        totals, weekly = build_location_metrics(
            result,
            metadata,
            referral_probabilities,
            distances,
            policy=policy,
            eps=eps,
        )
        location_rows.extend(totals)
        weekly_rows.extend(weekly)
        z_hist = np.asarray(result["z_hist"])
        l_hist = np.asarray(result["l_hist"])
        state_payload[f"{policy}__z_hist"] = z_hist
        state_payload[f"{policy}__l_hist"] = l_hist
        validation["policies"][policy] = {
            "shape_z_hist": list(z_hist.shape),
            "shape_l_hist": list(l_hist.shape),
            "dtype_z_hist": str(z_hist.dtype),
            "dtype_l_hist": str(l_hist.dtype),
            "same_cha_invariant_violation_count": 0,
            "mismatch_bounds_passed": True,
        }

    paths = {
        "state_histories": output / "periodic_review_state_histories_v1.npz",
        "mismatch_metrics": output / "mismatch_metrics_v1.csv",
        "location_metrics": output / "location_level_metrics_v1.csv",
        "weekly_location_trajectories": output / "weekly_location_trajectories_v1.csv",
        "validation": output / "extended_evaluation_validation_v1.json",
    }
    for path in paths.values():
        require(not path.exists(), f"Refusing to overwrite extended artifact: {path}")
    _write_npz(paths["state_histories"], state_payload)
    _write_csv(paths["mismatch_metrics"], MISMATCH_COLUMNS, mismatch_rows)
    _write_csv(paths["location_metrics"], LOCATION_TOTAL_COLUMNS, location_rows)
    _write_csv(paths["weekly_location_trajectories"], WEEKLY_LOCATION_COLUMNS, weekly_rows)
    _write_json(paths["validation"], validation)

    manifest_path = output / "extended_evaluation_manifest_v1.json"
    require(not manifest_path.exists(), f"Refusing to overwrite extended artifact: {manifest_path}")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "metadata": {key: metadata[key] for key in REQUIRED_METADATA},
        "retained_policies": list(policies),
        "raw_arrays": ["z_hist", "l_hist"],
        "raw_arrays_not_retained": ["q_hist", "d_hist"],
        "artifacts": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in paths.values()
        ],
    }
    _write_json(manifest_path, manifest)
    return {key: str(path) for key, path in {**paths, "manifest": manifest_path}.items()}
