"""
Weekly Fangonda empirical comparative-statics scenarios.

This registry intentionally excludes repeated baseline points inside each family.
Use ``fangonda_base`` as the shared reference run, then compare against
non-baseline deviations in each family.
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

BASELINE_NAME = "fangonda_base"
FAIRNESS_SPEC = "demand_weighted_alpha_fair_v1"
CALIBRATION_VERSION = "fangonda_paper_2026_07_normalized_costs_dwaf_v1"

BASELINE_SPEC: Dict[str, Any] = {
    "name": BASELINE_NAME,
    "family": "baseline",
    "calibration_version": CALIBRATION_VERSION,
    "corr_rho": 0.0,
    "I0": 142.0,
    "I0_expected_demand_multiplier": None,
    "lambda_param": 0.0,
    "alpha": 0.5,
    "fairness_spec": FAIRNESS_SPEC,
    "fairness_includes_hf": True,
    "fairness_period_h_factor": False,
    "cha_variance_scale": 1.0,
    "N_h": 219,
    "N_c": 28,
    "cost_units": "one untreated under-five malaria episode",
    "hourly_opportunity_cost": 0.68,
    "motorcycle_taxi_fare_per_mile": 0.15,
    "normalization_caregiver_hours": 64.0,
    "referral_caregiver_hours": 8.0,
    "eta_scale": 1.0,
}

SCENARIO_PATCHES: Dict[str, Dict[str, Any]] = {
    BASELINE_NAME: {},
    "fangonda_kappa_0p05": {"family": "travel", "motorcycle_taxi_fare_per_mile": 0.05},
    "fangonda_kappa_0p25": {"family": "travel", "motorcycle_taxi_fare_per_mile": 0.25},
    "fangonda_rho_realized_0p5": {
        "family": "correlation_calibrated",
        "corr_rho": 0.5,
        "rho_interpretation": "realized_discrete_count_pearson",
    },
    "fangonda_rho_realized_0p7": {
        "family": "correlation_calibrated",
        "corr_rho": 0.7,
        "rho_interpretation": "realized_discrete_count_pearson",
    },
    "fangonda_inventory_0p6_expected": {
        "family": "inventory",
        "I0_expected_demand_multiplier": 0.6,
        "hold_eta_schedule_from_baseline": True,
    },
    "fangonda_inventory_0p8_expected": {
        "family": "inventory",
        "I0_expected_demand_multiplier": 0.8,
        "hold_eta_schedule_from_baseline": True,
    },
    "fangonda_var_2x": {"family": "variance", "cha_variance_scale": 2.0},
    "fangonda_var_5x": {"family": "variance", "cha_variance_scale": 5.0},
    "fangonda_hf_exposure_146": {"family": "hf_demand", "N_h": 146},
    "fangonda_hf_exposure_199": {"family": "hf_demand", "N_h": 199},
    "fangonda_hf_exposure_243": {"family": "hf_demand", "N_h": 243},
    "fangonda_hf_exposure_292": {"family": "hf_demand", "N_h": 292},
}

SCENARIO_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "baseline": (BASELINE_NAME,),
    "travel": ("fangonda_kappa_0p05", "fangonda_kappa_0p25"),
    "correlation_calibrated": ("fangonda_rho_realized_0p5", "fangonda_rho_realized_0p7"),
    "inventory": ("fangonda_inventory_0p6_expected", "fangonda_inventory_0p8_expected"),
    "variance": ("fangonda_var_2x", "fangonda_var_5x"),
    "hf_demand": (
        "fangonda_hf_exposure_146",
        "fangonda_hf_exposure_199",
        "fangonda_hf_exposure_243",
        "fangonda_hf_exposure_292",
    ),
}


SCENARIO_NAMES: Tuple[str, ...] = tuple(SCENARIO_PATCHES.keys())
NON_BASELINE_SCENARIO_NAMES: Tuple[str, ...] = tuple(
    name for name in SCENARIO_NAMES if name != BASELINE_NAME
)
NON_BASELINE_FAMILY_NAMES: Tuple[str, ...] = tuple(
    family for family in SCENARIO_FAMILIES.keys() if family != "baseline"
)


def load_base_dict() -> Dict[str, Any]:
    path = os.path.join(_PKG_DIR, "base.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_scenarios(include_baseline: bool = True) -> Tuple[str, ...]:
    if include_baseline:
        return SCENARIO_NAMES
    return NON_BASELINE_SCENARIO_NAMES


def list_families(include_baseline: bool = False) -> Tuple[str, ...]:
    if include_baseline:
        return tuple(SCENARIO_FAMILIES.keys())
    return NON_BASELINE_FAMILY_NAMES


def get_family_members(family_name: str, include_baseline: bool = False) -> Tuple[str, ...]:
    if family_name not in SCENARIO_FAMILIES:
        raise KeyError(
            f"Unknown family {family_name!r}. Expected one of {list(SCENARIO_FAMILIES.keys())}"
        )
    if family_name == "baseline":
        return SCENARIO_FAMILIES["baseline"]
    members = SCENARIO_FAMILIES[family_name]
    if include_baseline:
        return (BASELINE_NAME,) + members
    return members


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in patch.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def resolve_scenario_spec(scenario_name: str) -> Dict[str, Any]:
    if scenario_name not in SCENARIO_PATCHES:
        raise KeyError(
            f"Unknown scenario {scenario_name!r}. Expected one of {list(SCENARIO_PATCHES.keys())}"
        )
    spec = _deep_merge(BASELINE_SPEC, SCENARIO_PATCHES[scenario_name])
    spec["name"] = scenario_name
    return spec


def periodic_demand_mode_for_spec(resolved_spec: Mapping[str, Any]) -> str:
    """Return the demand model implied by one resolved Fangonda scenario."""

    family = str(resolved_spec.get("family", ""))
    corr_rho = float(resolved_spec.get("corr_rho", 0.0))
    variance_scale = float(resolved_spec.get("cha_variance_scale", 1.0))
    if family in {"correlation", "correlation_calibrated"} and not math.isclose(corr_rho, 0.0):
        return "gaussian_copula_binomial"
    if family == "variance" and not math.isclose(variance_scale, 1.0):
        return "beta_binomial"
    return "binomial"


def periodic_demand_mode_for_scenario(scenario_name: str) -> str:
    """Resolve a registry scenario and return its implied demand model."""

    return periodic_demand_mode_for_spec(resolve_scenario_spec(scenario_name))


def iter_scenario_specs(
    family_name: str | None = None, include_baseline: bool = False
) -> Iterator[Dict[str, Any]]:
    if family_name is None:
        scenario_names: Iterable[str] = list_scenarios(include_baseline=include_baseline)
    else:
        scenario_names = get_family_members(family_name, include_baseline=include_baseline)
    for scenario_name in scenario_names:
        yield resolve_scenario_spec(scenario_name)
