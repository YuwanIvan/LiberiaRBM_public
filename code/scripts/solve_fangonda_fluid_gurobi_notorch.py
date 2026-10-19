#!/usr/bin/env python3
"""Torch-free Fangonda deterministic fluid LP/QCP solver.

This script requires Gurobi but does not require PyTorch. It reads
Fangonda CSV/config inputs, solves the deterministic fluid benchmark,
and writes allocation_plan.npz for policy evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.fangonda_weekly import scenarios as scenario_registry
from fairness_diagnostics import demand_adjusted_fairness_diagnostics
import rho_shared_bundle
import rho_validation_bank


AUDITED_RHO_CALIBRATION_METHOD = "exact_discrete_threshold_bivariate_normal_cdf_v1"
AUDITED_RHO_VALIDATION_SCHEMA = "fangonda-realized-rho-validation-v2"

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - Gurobi is an optional dependency
    gp = None
    GRB = None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


LOCATION_LABELS = ("HF", "CHA 1", "CHA 2", "CHA 3", "CHA 4", "CHA 5", "CHA 6", "CHA 7")
ETA_MODES = ("constant_peak", "weekly_time_varying")
FAIRNESS_SPEC = "demand_weighted_alpha_fair_v1"
FAIRNESS_INCLUDES_HF = True
FAIRNESS_PERIOD_H_FACTOR = False
MU_RATE_EVIDENCE = (
    "Fangonda mu.csv stores weekly demand rates over a 22-week horizon.",
    "Discrete-review evaluation converts each row to period demand by multiplying by review_dt.",
    "The paper calibration has 22 raw weekly rows, so the default fluid grid is weekly.",
)


@dataclass
class FluidContext:
    raw_config: Dict[str, Any]
    config_path: Path
    output_dir: Path
    scenario_manifest: Optional[Dict[str, Any]]
    materialized: Optional[Dict[str, str]]
    total_time: float
    dim: int
    q0: float
    z0: np.ndarray
    lambda_param: float
    alpha: float
    mu_raw: np.ndarray
    gamma_raw: np.ndarray
    mu_review: np.ndarray
    gamma_review: np.ndarray
    review_dt: float
    p: np.ndarray
    u: np.ndarray
    r: np.ndarray
    distance: np.ndarray
    c_tilde: np.ndarray
    eta: np.ndarray
    eta_schedule_raw: np.ndarray
    eta_review: np.ndarray
    eta_mode: str
    chi_eta: float
    s_eta: float
    calibration_version: Optional[str]
    exposure: Optional[np.ndarray]
    theta: Optional[np.ndarray]
    warnings: List[str]


@dataclass
class DemandResolution:
    interpretation: str
    period_demand: np.ndarray
    diagnostics: Dict[str, Any]
    warnings: List[str]


@dataclass
class FluidSolution:
    status: int
    status_name: str
    runtime: float
    objective: Optional[float]
    allocation: np.ndarray
    warehouse: np.ndarray
    local_inventory: np.ndarray
    shortage: np.ndarray
    lexicographic: Optional[Dict[str, Any]] = None


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def load_base_config_notorch() -> Dict[str, Any]:
    return scenario_registry.load_base_dict()


def load_baseline_arrays_notorch() -> Dict[str, np.ndarray]:
    baseline_dir = PROJECT_ROOT / "configs" / "fangonda_weekly"
    required = ("theta", "exposure", "mu", "gamma", "p", "distance", "u", "r", "c", "h", "eta")
    arrays: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for key in required:
        path = baseline_dir / f"{key}.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        data = np.loadtxt(path, delimiter=",")
        arr = np.asarray(data, dtype=np.float64)
        arrays[key] = np.atleast_2d(arr) if key in {"mu", "gamma"} else arr.reshape(-1)
    if missing:
        raise FileNotFoundError(
            "The Torch-free fluid solver is torch-free and expects Fangonda CSVs to exist. "
            f"Missing: {missing}"
        )
    return arrays


def write_csv_array(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 1:
        np.savetxt(path, arr.reshape(-1, 1), delimiter=",", fmt="%.12g")
    elif arr.ndim == 2:
        np.savetxt(path, arr, delimiter=",", fmt="%.12g")
    elif arr.ndim == 3:
        np.savetxt(path, arr.reshape(arr.shape[0], -1), delimiter=",", fmt="%.12g")
    else:
        raise ValueError(f"Unsupported array rank for CSV export: {arr.ndim}.")


def gamma_diag_np(gamma: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(gamma, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == dim:
        return arr.copy()
    if arr.ndim == 2 and arr.shape[1] == dim * dim:
        return np.diagonal(arr.reshape(arr.shape[0], dim, dim), axis1=1, axis2=2).copy()
    if arr.ndim == 3 and arr.shape[1:] == (dim, dim):
        return np.diagonal(arr, axis1=1, axis2=2).copy()
    raise ValueError(f"Unexpected gamma shape {arr.shape} for dim={dim}.")


def build_full_covariance_np(gamma_diag: np.ndarray, rho: float) -> np.ndarray:
    rows, dim = gamma_diag.shape
    out = np.zeros((rows, dim, dim), dtype=np.float64)
    for n in range(rows):
        variances = np.clip(gamma_diag[n], 0.0, None)
        std = np.sqrt(variances)
        cov = float(rho) * np.outer(std, std)
        np.fill_diagonal(cov, variances)
        out[n] = cov
    return out


def normalized_cost_arrays_np(distance: np.ndarray, p: np.ndarray, spec: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    w = float(spec["hourly_opportunity_cost"])
    kappa = float(spec["motorcycle_taxi_fare_per_mile"])
    referral_hours = float(spec["referral_caregiver_hours"])
    normalization_hours = float(spec.get("normalization_caregiver_hours", 64.0))
    if w <= 0.0 or normalization_hours <= 0.0:
        raise ValueError("w and normalization_hours must be positive for normalized costs.")
    distance = np.asarray(distance, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    dim = p.shape[0] + 1
    r = np.zeros(dim, dtype=np.float64)
    r[1:] = (referral_hours * w + 2.0 * kappa * distance[1:]) / (normalization_hours * w)
    u = np.asarray([1.0], dtype=np.float64)
    c = np.zeros(dim, dtype=np.float64)
    c[0] = 1.0
    c[1:] = (1.0 - p) + p * r[1:]
    return {"u": u, "r": r, "c": c}


def compute_paper_eta_np(
    mu: np.ndarray,
    gamma: np.ndarray,
    p: np.ndarray,
    I0: float,
    total_time: float,
    slack_multiplier: float = 1.0,
) -> tuple[np.ndarray, Dict[str, Any]]:
    mu_arr = np.asarray(mu, dtype=np.float64)
    rows, dim = mu_arr.shape
    gamma_diag = gamma_diag_np(gamma, dim)
    p_arr = np.asarray(p, dtype=np.float64).reshape(-1)
    step = float(total_time) / float(rows)
    total_expected = float(mu_arr.sum() * step)
    rho_raw = 1.0 - float(I0) / total_expected
    rho_bar = max(0.0, rho_raw)
    eta = np.zeros(dim, dtype=np.float64)
    eta[1:] = float(slack_multiplier) * np.max(
        mu_arr[:, 1:] + 2.0 * np.sqrt(np.clip(gamma_diag[:, 1:], 0.0, None)),
        axis=0,
    )
    referral_mean = rho_bar * np.sum(mu_arr[:, 1:] * p_arr.reshape(1, -1), axis=1)
    referral_var = (rho_bar ** 2) * np.sum(gamma_diag[:, 1:] * (p_arr.reshape(1, -1) ** 2), axis=1)
    eta[0] = float(slack_multiplier) * np.max(
        mu_arr[:, 0] + referral_mean + 2.0 * np.sqrt(np.clip(gamma_diag[:, 0] + referral_var, 0.0, None))
    )
    return eta, {"rho_bar": rho_bar, "rho_raw": rho_raw, "total_expected_demand": total_expected}


def compute_weekly_eta_schedule_np(
    mu: np.ndarray,
    gamma: np.ndarray,
    p: np.ndarray,
    I0: float,
    total_time: float,
    chi_eta: float = 1.75,
    s_eta: float = 0.0,
) -> tuple[np.ndarray, Dict[str, Any]]:
    mu_arr = np.asarray(mu, dtype=np.float64)
    rows, dim = mu_arr.shape
    gamma_diag = gamma_diag_np(gamma, dim)
    p_arr = np.asarray(p, dtype=np.float64).reshape(-1)
    chi_eta = float(chi_eta)
    s_eta = float(s_eta)
    if chi_eta < 0.0 or s_eta < 0.0:
        raise ValueError(f"chi_eta and s_eta must be nonnegative, got {chi_eta}, {s_eta}.")
    review_length = float(total_time) / float(rows)
    dbar = float(mu_arr.sum() * review_length)
    rho_raw = 1.0 - float(I0) / dbar
    rho_q = max(0.0, rho_raw)
    diag = np.clip(gamma_diag, 0.0, None)
    schedule = np.zeros((rows, dim), dtype=np.float64)
    schedule[:, 1:] = chi_eta * mu_arr[:, 1:] + s_eta * np.sqrt(diag[:, 1:])
    referral_mean = rho_q * np.sum(mu_arr[:, 1:] * p_arr.reshape(1, -1), axis=1)
    referral_var = (rho_q ** 2) * np.sum(diag[:, 1:] * (p_arr.reshape(1, -1) ** 2), axis=1)
    schedule[:, 0] = chi_eta * (mu_arr[:, 0] + referral_mean) + s_eta * np.sqrt(
        np.clip(diag[:, 0] + referral_var, 0.0, None)
    )
    return schedule, {
        "rho_Q": rho_q,
        "rho_raw": rho_raw,
        "total_expected_demand": dbar,
        "review_length": review_length,
        "chi_eta": chi_eta,
        "s_eta": s_eta,
    }


def compute_eta_schedule_notorch(arrays: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
    eqn_config = config["eqn_config"]
    eta_mode = str(eqn_config.get("eta_mode", eqn_config.get("eta_profile", "constant_peak")))
    chi_eta = float(eqn_config.get("chi_eta", 1.75))
    s_eta = float(eqn_config.get("s_eta", 0.0))
    mu = np.asarray(arrays["mu"], dtype=np.float64)
    rows, dim = mu.shape
    if eta_mode == "constant_peak":
        schedule = np.repeat(np.asarray(arrays["eta"], dtype=np.float64).reshape(1, dim), rows, axis=0)
        details = {
            "eta_mode": eta_mode,
            "eta_profile": eta_mode,
            "eta_schedule_rule": "constant_peak_broadcast_from_eta_csv",
            "chi_eta": chi_eta,
            "s_eta": s_eta,
        }
    elif eta_mode == "weekly_time_varying":
        schedule, formula_details = compute_weekly_eta_schedule_np(
            mu,
            np.asarray(arrays["gamma"], dtype=np.float64),
            np.asarray(arrays["p"], dtype=np.float64),
            float(eqn_config["I0"]),
            float(eqn_config["total_time"]),
            chi_eta=chi_eta,
            s_eta=s_eta,
        )
        details = {
            "eta_mode": eta_mode,
            "eta_profile": eta_mode,
            "eta_schedule_rule": "weekly_time_varying_formula",
            "eta_schedule_file_role": str(eqn_config.get("eta_schedule_file_role", "active_input")),
            **formula_details,
        }
    else:
        raise ValueError(f"eta_mode must be one of {ETA_MODES}, got {eta_mode!r}.")
    details.update(
        {
            "eta_schedule_shape": list(schedule.shape),
            "eta_min_by_location": np.min(schedule, axis=0).tolist(),
            "eta_max_by_location": np.max(schedule, axis=0).tolist(),
            "eta_min_by_week": np.min(schedule, axis=1).tolist(),
            "eta_max_by_week": np.max(schedule, axis=1).tolist(),
            "all_weeks_identical": bool(np.allclose(schedule, schedule[0:1])),
            "cha_profiles_identical": bool(dim <= 2 or np.allclose(schedule[:, 1:], schedule[:, 1:2])),
        }
    )
    return schedule, details


def materialize_run_notorch(
    run_name: str,
    spec: Mapping[str, Any],
    args: argparse.Namespace,
    materialize_root: Path,
) -> Dict[str, str]:
    run_dir = materialize_root / run_name
    params_dir = run_dir / "params"
    run_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    config = load_base_config_notorch()
    arrays = {key: np.array(value, copy=True) for key, value in load_baseline_arrays_notorch().items()}
    eqn_config = config["eqn_config"]
    net_config = config["net_config"]
    dim = int(eqn_config["dim"])
    total_time = float(eqn_config["total_time"])
    theta = arrays["theta"].reshape(-1)
    exposure = arrays["exposure"].reshape(-1)
    p = arrays["p"].reshape(-1)
    baseline_mu = np.asarray(arrays["mu"], dtype=np.float64)
    baseline_gamma_diag = gamma_diag_np(arrays["gamma"], dim)
    baseline_eta = np.asarray(arrays["eta"], dtype=np.float64).reshape(-1)

    if not np.isclose(float(spec["N_h"]), exposure[0]):
        exposure[0] = float(spec["N_h"])
        arrays["exposure"] = exposure
        arrays["mu"][:, 0] = theta * exposure[0]
        gamma_diag = gamma_diag_np(arrays["gamma"], dim)
        gamma_diag[:, 0] = exposure[0] * theta * (1.0 - theta)
        arrays["gamma"] = gamma_diag

    costs = normalized_cost_arrays_np(arrays["distance"], p, spec)
    arrays["u"] = costs["u"]
    arrays["r"] = costs["r"]
    arrays["c"] = costs["c"]

    gamma_diag = gamma_diag_np(arrays["gamma"], dim)
    var_scale = float(spec["cha_variance_scale"])
    gamma_diag[:, 1:] *= var_scale
    rho = float(spec["corr_rho"])
    arrays["gamma"] = gamma_diag if np.isclose(rho, 0.0) else build_full_covariance_np(gamma_diag, rho)
    rho_metadata: Dict[str, Any] = {}
    rho_calibration_report = getattr(args, "rho_calibration_report", None)
    rho_calibration_report_sha256 = getattr(args, "rho_calibration_report_sha256", None)
    rho_validation_report = getattr(args, "rho_validation_report", None)
    rho_validation_report_sha256 = getattr(args, "rho_validation_report_sha256", None)
    rho_calibration_finalization = getattr(args, "rho_calibration_finalization", None)
    rho_calibration_finalization_sha256 = getattr(args, "rho_calibration_finalization_sha256", None)
    rho_validation_finalization = getattr(args, "rho_validation_finalization", None)
    rho_validation_finalization_sha256 = getattr(args, "rho_validation_finalization_sha256", None)
    rho_shared_bundle_manifest = getattr(args, "rho_shared_bundle_manifest", None)
    rho_shared_bundle_manifest_sha256 = getattr(args, "rho_shared_bundle_manifest_sha256", None)
    rho_shared_bundle_finalization = getattr(args, "rho_shared_bundle_finalization", None)
    rho_shared_bundle_finalization_sha256 = getattr(args, "rho_shared_bundle_finalization_sha256", None)
    rho_args = (
        rho_calibration_report,
        rho_calibration_report_sha256,
        rho_validation_report,
        rho_validation_report_sha256,
        rho_calibration_finalization,
        rho_calibration_finalization_sha256,
        rho_validation_finalization,
        rho_validation_finalization_sha256,
    )
    bundle_args = (
        rho_shared_bundle_manifest,
        rho_shared_bundle_manifest_sha256,
        rho_shared_bundle_finalization,
        rho_shared_bundle_finalization_sha256,
    )
    if any(value is not None for value in bundle_args) and not all(value is not None for value in bundle_args):
        raise ValueError("All shared rho bundle paths and hashes must be supplied together.")
    if any(value is not None for value in bundle_args) and any(value is not None for value in rho_args):
        raise ValueError("Shared rho bundle inputs cannot be mixed with direct cluster artifact paths.")
    bundle_resolution: Dict[str, Any] = {}
    if all(value is not None for value in bundle_args):
        if spec.get("rho_interpretation") != "realized_discrete_count_pearson":
            raise ValueError("A shared rho bundle is valid only for calibrated realized-correlation scenarios.")
        bundle_resolution = rho_shared_bundle.validate_bundle(
            rho_shared_bundle_manifest,
            rho_shared_bundle_manifest_sha256,
            finalization_path=rho_shared_bundle_finalization,
            finalization_sha256=rho_shared_bundle_finalization_sha256,
            scenario=str(spec["name"]),
        )
        rho_calibration_report = bundle_resolution["calibration_report_path"]
        rho_calibration_report_sha256 = bundle_resolution["calibration_report_sha256"]
        rho_validation_report = bundle_resolution["validation_report_path"]
        rho_validation_report_sha256 = bundle_resolution["validation_report_sha256"]
    if bundle_resolution or any(value is not None for value in rho_args):
        if not bundle_resolution and not all(value is not None for value in rho_args):
            raise ValueError("All rho calibration and validation paths/hashes must be supplied together.")
        if spec.get("rho_interpretation") != "realized_discrete_count_pearson":
            raise ValueError("Rho calibration artifacts require a calibrated realized-correlation scenario.")
        calibration_path = Path(rho_calibration_report).expanduser().resolve()
        validation_path = Path(rho_validation_report).expanduser().resolve()
        if file_sha256(calibration_path) != rho_calibration_report_sha256:
            raise ValueError("Rho calibration report SHA-256 mismatch.")
        if file_sha256(validation_path) != rho_validation_report_sha256:
            raise ValueError("Rho validation report SHA-256 mismatch.")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if calibration.get("schema_version") != "fangonda-realized-rho-calibration-v1" or calibration.get("frozen") is not True:
            raise ValueError("Rho calibration report is not frozen v1.")
        if calibration.get("numerical_method", {}).get("name") != AUDITED_RHO_CALIBRATION_METHOD:
            raise ValueError("Rho calibration report does not use the audited exact discrete-threshold method.")
        if (
            validation.get("schema_version") != AUDITED_RHO_VALIDATION_SCHEMA
            or validation.get("frozen") is not True
            or validation.get("validation_passed") is not True
        ):
            raise ValueError("Rho calibration lacks a passing 200,000-path validator-v2 report.")
        if calibration.get("scenario") != spec["name"] or not np.isclose(
            float(calibration.get("target_realized_rho")), rho
        ):
            raise ValueError("Rho calibration scenario or target mismatch.")
        if validation.get("calibration_report", {}).get("sha256") != rho_calibration_report_sha256:
            raise ValueError("Rho validation does not certify the selected calibration.")
        if not bundle_resolution:
            rho_validation_bank.validate_slurm_finalization(
                rho_calibration_finalization,
                rho_calibration_finalization_sha256,
                artifact_path=calibration_path,
                artifact_sha256=rho_calibration_report_sha256,
            )
            rho_validation_bank.validate_slurm_finalization(
                rho_validation_finalization,
                rho_validation_finalization_sha256,
                artifact_path=validation_path,
                artifact_sha256=rho_validation_report_sha256,
            )
        if int(validation.get("num_paths", -1)) != 200000 or int(validation.get("seed", -1)) != 2026082203:
            raise ValueError("Rho validation did not use the formal 200,000-path bank contract.")
        bank_record = validation.get("validation_bank", {})
        if bundle_resolution:
            if (
                bank_record.get("sha256") != bundle_resolution["bank_manifest_sha256"]
                or bank_record.get("array_file_sha256") != bundle_resolution["bank_array_file_sha256"]
                or bank_record.get("array_sha256") != bundle_resolution["bank_array_sha256"]
            ):
                raise ValueError("Shared rho bundle does not preserve the certified validation-bank evidence.")
        else:
            bank_payload, _ = rho_validation_bank.load_bank(
                bank_record.get("path", ""),
                bank_record.get("sha256", ""),
                expected_seed=2026082203,
                expected_review_intervals=gamma_diag.shape[0],
                expected_num_paths=200000,
                expected_dim=dim,
            )
            if (
                bank_record.get("array_file_sha256") != bank_payload["array"]["file_sha256"]
                or bank_record.get("array_sha256") != bank_payload["array"]["array_sha256"]
            ):
                raise ValueError("Rho validation report does not pin the actual validation-bank draws.")
            bank_final = validation.get("validation_bank_slurm_finalization", {})
            rho_validation_bank.validate_slurm_finalization(
                bank_final.get("path", ""),
                bank_final.get("sha256", ""),
                artifact_path=bank_record.get("path", ""),
                artifact_sha256=bank_record.get("sha256", ""),
            )
        matrix_record = calibration["latent_matrices"]
        matrix_path = calibration_path.parent / matrix_record["path"]
        if file_sha256(matrix_path) != matrix_record["file_sha256"]:
            raise ValueError("Latent rho matrix file SHA-256 mismatch.")
        matrices = np.load(matrix_path, allow_pickle=False)
        if array_sha256(matrices) != matrix_record["array_sha256"]:
            raise ValueError("Latent rho matrix content SHA-256 mismatch.")
        if matrices.shape != (gamma_diag.shape[0], dim, dim):
            raise ValueError("Latent rho matrix shape mismatch.")
        calibrated_gamma = np.zeros_like(matrices, dtype=np.float64)
        for week in range(gamma_diag.shape[0]):
            std = np.sqrt(np.clip(gamma_diag[week], 0.0, None))
            calibrated_gamma[week] = matrices[week] * np.outer(std, std)
            np.fill_diagonal(calibrated_gamma[week], gamma_diag[week])
        arrays["gamma"] = calibrated_gamma
        rho_metadata = {
            "rho_interpretation": "realized_discrete_count_pearson",
            "rho_calibration_report": {"path": str(calibration_path), "sha256": rho_calibration_report_sha256},
            "rho_validation_report": {"path": str(validation_path), "sha256": rho_validation_report_sha256},
            "latent_correlation_matrix_file_sha256": matrix_record["file_sha256"],
            "latent_correlation_matrix_array_sha256": matrix_record["array_sha256"],
        }
        if bundle_resolution:
            rho_metadata["rho_shared_bundle"] = {
                "path": bundle_resolution["bundle_manifest_path"],
                "sha256": bundle_resolution["bundle_manifest_sha256"],
            }
    elif spec.get("rho_interpretation") == "realized_discrete_count_pearson":
        raise ValueError("Calibrated-rho scenarios require calibration and validation artifacts.")

    raw_dt = total_time / float(arrays["mu"].shape[0])
    total_expected = float(arrays["mu"].sum() * raw_dt)
    i0_multiplier = spec.get("I0_expected_demand_multiplier")
    I0 = float(spec["I0"]) if i0_multiplier is None else float(i0_multiplier) * total_expected
    eqn_config["I0"] = I0

    arrays["eta"] = baseline_eta.copy()
    eta_rule = "baseline_fixed"
    eta_details = {"rho_bar": None, "rho_raw": None, "total_expected_demand": float(baseline_mu.sum() * raw_dt)}
    if spec["family"] == "hf_demand":
        recomputed_eta, eta_details = compute_paper_eta_np(arrays["mu"], gamma_diag, p, I0, total_time)
        arrays["eta"][0] = recomputed_eta[0]
        arrays["eta"][1:] = baseline_eta[1:]
        eta_rule = "recompute_hf_eta_only"
    elif spec["family"] == "shipment_bounds":
        arrays["eta"] = baseline_eta * float(spec["eta_scale"])
        eta_rule = "scale_baseline_eta"

    lambda_param = (
        float(args.lambda_param)
        if getattr(args, "lambda_param", None) is not None
        else float(spec["lambda_param"])
    )
    alpha = float(args.alpha) if getattr(args, "alpha", None) is not None else float(spec["alpha"])
    eta_mode = str(
        getattr(args, "eta_mode", None)
        or getattr(args, "eta_profile", None)
        or eqn_config.get("eta_mode", "constant_peak")
    )
    eqn_config.update(
        {
            "config_dir": str(params_dir.resolve()),
            "lambda_param": lambda_param,
            "alpha": alpha,
            "calibration_version": spec.get("calibration_version"),
            "fairness_spec": FAIRNESS_SPEC,
            "fairness_includes_hf": FAIRNESS_INCLUDES_HF,
            "fairness_period_h_factor": FAIRNESS_PERIOD_H_FACTOR,
            "eta_mode": eta_mode,
            "eta_profile": eta_mode,
            "eta_schedule_file": "eta_schedule.csv",
            "eta_schedule_file_role": "active_input",
            "chi_eta": float(args.chi_eta) if getattr(args, "chi_eta", None) is not None else float(eqn_config.get("chi_eta", 1.75)),
            "s_eta": float(args.s_eta) if getattr(args, "s_eta", None) is not None else float(eqn_config.get("s_eta", 0.0)),
        }
    )
    if getattr(args, "num_time_interval", None) is not None:
        eqn_config["num_time_interval"] = int(args.num_time_interval)
    net_config["num_time_interval"] = int(eqn_config["num_time_interval"])
    net_config["lambd_const"] = lambda_param
    if spec["family"] in {"correlation", "correlation_calibrated"} and not np.isclose(rho, 0.0):
        eqn_config["periodic_demand_mode"] = "gaussian_copula_binomial"
    elif spec["family"] == "variance" and not np.isclose(var_scale, 1.0):
        eqn_config["periodic_demand_mode"] = "beta_binomial"
    else:
        eqn_config["periodic_demand_mode"] = "binomial"

    eta_arrays: Mapping[str, np.ndarray] = arrays
    eta_config: Mapping[str, Any] = config
    if spec.get("hold_eta_schedule_from_baseline") and str(spec.get("family")) == "inventory":
        baseline_arrays = load_baseline_arrays_notorch()
        eta_arrays = {key: np.array(value, copy=True) for key, value in baseline_arrays.items()}
        eta_config_dict = json.loads(json.dumps(config, default=json_default))
        eta_config_dict["eqn_config"]["I0"] = float(scenario_registry.BASELINE_SPEC["I0"])
        eta_config = eta_config_dict
    eta_schedule, eta_metadata = compute_eta_schedule_notorch(eta_arrays, eta_config)
    if spec.get("hold_eta_schedule_from_baseline") and str(spec.get("family")) == "inventory":
        eta_metadata["eta_schedule_rule"] = (
            str(eta_metadata.get("eta_schedule_rule", ""))
            + "_held_fixed_from_baseline_for_inventory"
        )
        eta_metadata["eta_held_fixed_from_baseline"] = True
        eta_metadata["eta_held_fixed_baseline_I0"] = float(scenario_registry.BASELINE_SPEC["I0"])
    else:
        eta_metadata["eta_held_fixed_from_baseline"] = False
    arrays["eta_schedule"] = eta_schedule
    for key in ("theta", "exposure", "mu", "gamma", "p", "distance", "u", "r", "c", "h", "eta", "eta_schedule"):
        write_csv_array(params_dir / f"{key}.csv", arrays[key])

    manifest_spec = dict(spec)
    manifest_spec["lambda_param"] = lambda_param
    manifest_spec["alpha"] = alpha
    metadata = {
        "gamma_storage": "diagonal" if np.asarray(arrays["gamma"]).ndim == 2 else "full_flattened",
        "demand_mode": eqn_config["periodic_demand_mode"],
            "cost_units": str(spec["cost_units"]),
            "calibration_version": spec.get("calibration_version"),
            "untreated_cost": float(arrays["u"][0]),
            "p": arrays["p"].tolist(),
            "effective_shortage_costs": arrays["c"].tolist(),
            "I0": I0,
        "total_expected_demand": total_expected,
        "hf_expected_demand": float(arrays["mu"][:, 0].sum() * raw_dt),
        "cha_expected_demand": float(arrays["mu"][:, 1:].sum() * raw_dt),
        "eta_rule": eta_rule,
        "eta_rho_bar": eta_details["rho_bar"],
        "eta_rho_raw": eta_details["rho_raw"],
        "lambda_param": lambda_param,
        "alpha": alpha,
        "fairness_spec": FAIRNESS_SPEC,
        "fairness_includes_hf": FAIRNESS_INCLUDES_HF,
        "fairness_period_h_factor": FAIRNESS_PERIOD_H_FACTOR,
        **eta_metadata,
        **rho_metadata,
    }
    manifest = {
        "scenario_name": run_name,
        "family": str(spec["family"]),
        "baseline_reference": scenario_registry.BASELINE_NAME,
        "resolved_spec": manifest_spec,
        "metadata": metadata,
        "created_by": "solve_fangonda_fluid_gurobi_notorch.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_name": socket.gethostname(),
        "git_commit": git_commit(),
    }
    config_path = run_dir / "config_scenario.json"
    manifest_path = run_dir / "scenario_manifest.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=json_default)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=json_default)
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "params_dir": str(params_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve Fangonda deterministic fluid LP/QCP artifacts without importing Torch."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config_path", "--config-path", default=None, help="Existing config_scenario.json path.")
    source.add_argument(
        "--scenario",
        default=None,
        choices=scenario_registry.list_scenarios(include_baseline=True),
        help="Fangonda scenario to materialize before solving. Defaults to fangonda_base.",
    )
    parser.add_argument("--run-name", default=None, help="Run name for materialized scenario inputs.")
    parser.add_argument(
        "--materialize-output-root",
        default=None,
        help="Root for materialized scenario inputs. Defaults to <output-dir>/_materialized.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="fluid_artifacts/fangonda_dwaf_v1/fangonda_base/fluid_eta",
        help="Directory for solver outputs.",
    )
    parser.add_argument(
        "--variant-name",
        default=None,
        help="Optional variant label written to summaries. Defaults to Fluid-LP/QCP with eta suffix.",
    )
    parser.add_argument(
        "--eta_mode",
        "--eta-mode",
        choices=ETA_MODES,
        default=None,
        help="Eta mode used when materializing a scenario before solving.",
    )
    parser.add_argument("--eta_profile", "--eta-profile", choices=ETA_MODES, default=None)
    parser.add_argument("--chi_eta", "--chi-eta", type=float, default=None)
    parser.add_argument("--s_eta", "--s-eta", type=float, default=None)
    parser.add_argument("--lambda_param", "--lambda-param", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rho-calibration-report", default=None)
    parser.add_argument("--rho-calibration-report-sha256", default=None)
    parser.add_argument("--rho-calibration-finalization", default=None)
    parser.add_argument("--rho-calibration-finalization-sha256", default=None)
    parser.add_argument("--rho-validation-report", default=None)
    parser.add_argument("--rho-validation-report-sha256", default=None)
    parser.add_argument("--rho-validation-finalization", default=None)
    parser.add_argument("--rho-validation-finalization-sha256", default=None)
    parser.add_argument("--rho-shared-bundle-manifest", default=None)
    parser.add_argument("--rho-shared-bundle-manifest-sha256", default=None)
    parser.add_argument("--rho-shared-bundle-finalization", default=None)
    parser.add_argument("--rho-shared-bundle-finalization-sha256", default=None)
    parser.add_argument(
        "--num_time_interval",
        "--num-time-interval",
        type=int,
        default=None,
        help="Override num_time_interval when materializing a scenario before solving.",
    )

    rate_group = parser.add_mutually_exclusive_group()
    rate_group.add_argument("--with-rate-bound", action="store_true", dest="with_rate_bound")
    rate_group.add_argument("--no-rate-bound", action="store_false", dest="with_rate_bound")
    parser.set_defaults(with_rate_bound=True)

    mu_group = parser.add_mutually_exclusive_group()
    mu_group.add_argument("--mu-is-rate", action="store_const", const="rate", dest="mu_interpretation")
    mu_group.add_argument(
        "--mu-is-period-demand",
        action="store_const",
        const="period_demand",
        dest="mu_interpretation",
    )
    parser.set_defaults(mu_interpretation="rate")

    parser.add_argument("--review_num_intervals", "--review-num-intervals", type=int, default=None)
    parser.add_argument("--review_dt", "--review-dt", type=float, default=None)
    parser.add_argument("--num-paths", "--num_paths", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gurobi-threads", type=int, default=None)
    parser.add_argument("--gurobi-time-limit", type=float, default=None)
    parser.add_argument("--gurobi-output", type=int, choices=(0, 1), default=1)
    parser.set_defaults(lambda0_lexicographic_tiebreak=True)
    lambda0_group = parser.add_mutually_exclusive_group()
    lambda0_group.add_argument(
        "--lambda0-lexicographic-tiebreak",
        dest="lambda0_lexicographic_tiebreak",
        action="store_true",
        help="For lambda=0, minimize operational cost and then maximize fairness within tolerance.",
    )
    lambda0_group.add_argument(
        "--no-lambda0-lexicographic-tiebreak",
        dest="lambda0_lexicographic_tiebreak",
        action="store_false",
        help="Diagnostic only: keep the raw cost-minimizing LP solution at lambda=0.",
    )
    parser.add_argument("--lambda0-cost-tolerance", type=float, default=1e-7)
    return parser.parse_args()


def load_or_materialize_config(args: argparse.Namespace) -> tuple[Dict[str, Any], Path, Optional[Dict[str, str]]]:
    if args.config_path:
        config_path = Path(args.config_path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as f:
            return json.load(f), config_path, None

    scenario_name = args.scenario or scenario_registry.BASELINE_NAME
    run_name = args.run_name or scenario_name
    output_dir = Path(args.output_dir).expanduser()
    materialize_root = (
        Path(args.materialize_output_root).expanduser()
        if args.materialize_output_root
        else output_dir / "_materialized"
    )
    spec = scenario_registry.resolve_scenario_spec(scenario_name)
    materialized = materialize_run_notorch(
        run_name,
        spec,
        args,
        materialize_root,
    )
    config_path = Path(materialized["config_path"]).resolve()
    with config_path.open(encoding="utf-8") as f:
        return json.load(f), config_path, materialized


def load_scenario_manifest(config_path: Path) -> Optional[Dict[str, Any]]:
    manifest_path = config_path.parent / "scenario_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(filename: str, config_path: Path, config_dir: Optional[str]) -> Path:
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path

    candidates: List[Path] = []
    if config_dir:
        candidates.append(Path(config_dir).expanduser() / filename)
    candidates.extend(
        [
            config_path.parent / filename,
            config_path.parent / "params" / filename,
            PROJECT_ROOT / filename,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else path.resolve()


def load_csv_vector(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    return np.asarray(data, dtype=np.float64).reshape(-1)


def load_csv_matrix(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    arr = np.asarray(data, dtype=np.float64)
    return np.atleast_2d(arr)


def load_optional_vector(eqn_config: Mapping[str, Any], attr_name: str, config_path: Path) -> Optional[np.ndarray]:
    filename = eqn_config.get(attr_name)
    if not filename:
        return None
    path = resolve_config_path(str(filename), config_path, eqn_config.get("config_dir"))
    if not path.is_file():
        return None
    return load_csv_vector(path)


def load_optional_matrix(eqn_config: Mapping[str, Any], attr_name: str, config_path: Path) -> Optional[np.ndarray]:
    filename = eqn_config.get(attr_name)
    if not filename:
        return None
    path = resolve_config_path(str(filename), config_path, eqn_config.get("config_dir"))
    if not path.is_file():
        return None
    return load_csv_matrix(path)


def load_mu_gamma(eqn_config: Mapping[str, Any], config_path: Path) -> tuple[np.ndarray, np.ndarray]:
    dim = int(eqn_config["dim"])
    config_dir = eqn_config.get("config_dir")
    mu_path = resolve_config_path(str(eqn_config["mu_file"]), config_path, config_dir)
    gamma_path = resolve_config_path(str(eqn_config["gamma_file"]), config_path, config_dir)
    mu_raw = load_csv_matrix(mu_path)
    gamma_flat = load_csv_matrix(gamma_path)

    if mu_raw.shape[0] != gamma_flat.shape[0]:
        raise ValueError(
            f"mu rows ({mu_raw.shape[0]}) and gamma rows ({gamma_flat.shape[0]}) must match."
        )
    if mu_raw.shape[1] != dim:
        raise ValueError(f"mu has {mu_raw.shape[1]} columns, expected dim={dim}.")

    if gamma_flat.shape[1] == dim:
        gamma_raw = np.zeros((gamma_flat.shape[0], dim, dim), dtype=np.float64)
        for n in range(gamma_flat.shape[0]):
            gamma_raw[n] = np.diag(gamma_flat[n])
    elif gamma_flat.shape[1] == dim * dim:
        gamma_raw = gamma_flat.reshape(gamma_flat.shape[0], dim, dim)
    else:
        raise ValueError(
            f"gamma has {gamma_flat.shape[1]} columns; expected {dim} or {dim * dim}."
        )
    return mu_raw, gamma_raw


def resample_matrix_to_review_grid(matrix_raw: np.ndarray, num_review_intervals: int) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(matrix_raw, dtype=np.float64))
    num_raw = int(arr.shape[0])
    num_review = int(num_review_intervals)
    if num_review <= 0:
        raise ValueError("review interval count must be positive.")
    if num_review == num_raw:
        return arr.copy()
    if num_raw == 1:
        return np.repeat(arr, num_review, axis=0)
    if num_review % num_raw == 0:
        repeat_factor = num_review // num_raw
        return np.repeat(arr, repeat_factor, axis=0)
    if num_raw % num_review == 0:
        block = num_raw // num_review
        return arr.reshape(num_review, block, arr.shape[1]).mean(axis=1)
    raise ValueError(
        "Review grid must exactly nest or be nested by the raw eta grid. "
        f"Got eta_raw={num_raw}, review={num_review}."
    )


def infer_review_num_intervals(
    total_time: float,
    review_num_intervals: Optional[int],
    review_dt: Optional[float],
    default_num_intervals: int,
) -> int:
    if review_num_intervals is not None and review_dt is not None:
        raise ValueError("Specify at most one of --review_num_intervals and --review_dt.")
    if review_num_intervals is not None:
        return int(review_num_intervals)
    if review_dt is not None:
        scaled = float(total_time) / float(review_dt)
        rounded = int(round(scaled))
        if not np.isclose(scaled, rounded, atol=1e-10, rtol=0.0):
            raise ValueError(f"review_dt={review_dt} does not divide total_time={total_time}.")
        return rounded
    return int(default_num_intervals)


def resample_parameters_to_review_grid(
    mu_raw: np.ndarray,
    gamma_raw: np.ndarray,
    num_review_intervals: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_raw = int(mu_raw.shape[0])
    num_review = int(num_review_intervals)
    if gamma_raw.shape[0] != num_raw:
        raise ValueError("mu_raw and gamma_raw must have the same number of rows.")
    if num_review <= 0:
        raise ValueError("review interval count must be positive.")
    if num_review == num_raw:
        return mu_raw.copy(), gamma_raw.copy()
    if num_review % num_raw == 0:
        repeat_factor = num_review // num_raw
        return np.repeat(mu_raw, repeat_factor, axis=0), np.repeat(gamma_raw, repeat_factor, axis=0)
    if num_raw % num_review == 0:
        block = num_raw // num_review
        mu_review = mu_raw.reshape(num_review, block, -1).mean(axis=1)
        gamma_review = gamma_raw.reshape(num_review, block, gamma_raw.shape[1], gamma_raw.shape[2]).mean(axis=1)
        return mu_review, gamma_review
    raise ValueError(
        "Review grid must exactly nest or be nested by the raw grid. "
        f"Got raw={num_raw}, review={num_review}."
    )


def build_context(args: argparse.Namespace) -> FluidContext:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config, config_path, materialized = load_or_materialize_config(args)
    eqn_config = raw_config["eqn_config"]

    total_time = float(eqn_config["total_time"])
    dim = int(eqn_config["dim"])
    lambda_param = float(eqn_config.get("lambda_param", 0.0))
    alpha = float(eqn_config.get("alpha", 0.5))
    chi_eta = float(eqn_config.get("chi_eta", 1.75))
    s_eta = float(eqn_config.get("s_eta", 0.0))
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}.")
    if lambda_param < 0.0:
        raise ValueError(f"lambda_param must be nonnegative, got {lambda_param}.")
    if chi_eta < 0.0 or s_eta < 0.0:
        raise ValueError(f"chi_eta and s_eta must be nonnegative, got {chi_eta}, {s_eta}.")
    config_dir = eqn_config.get("config_dir")
    mu_raw, gamma_raw = load_mu_gamma(eqn_config, config_path)
    review_intervals = infer_review_num_intervals(
        total_time,
        args.review_num_intervals,
        args.review_dt,
        default_num_intervals=mu_raw.shape[0],
    )
    mu_review, gamma_review = resample_parameters_to_review_grid(mu_raw, gamma_raw, review_intervals)
    review_dt = total_time / float(review_intervals)

    p = load_csv_vector(resolve_config_path(str(eqn_config["p_file"]), config_path, config_dir))
    c_tilde = load_csv_vector(resolve_config_path(str(eqn_config["c_file"]), config_path, config_dir))
    eta = load_csv_vector(resolve_config_path(str(eqn_config["eta_file"]), config_path, config_dir))
    eta_schedule_raw = load_optional_matrix(eqn_config, "eta_schedule_file", config_path)
    u = load_optional_vector(eqn_config, "u_file", config_path)
    r = load_optional_vector(eqn_config, "r_file", config_path)
    distance = load_optional_vector(eqn_config, "distance_file", config_path)
    exposure = load_optional_vector(eqn_config, "exposure_file", config_path)
    theta = load_optional_vector(eqn_config, "theta_file", config_path)

    if p.shape[0] != dim - 1:
        raise ValueError(f"p has {p.shape[0]} entries, expected {dim - 1}.")
    if c_tilde.shape[0] != dim:
        raise ValueError(f"c has {c_tilde.shape[0]} entries, expected {dim}.")
    if eta.shape[0] != dim:
        raise ValueError(f"eta has {eta.shape[0]} entries, expected {dim}.")
    if eta_schedule_raw is None:
        eta_schedule_raw = np.repeat(eta.reshape(1, dim), mu_raw.shape[0], axis=0)
    if eta_schedule_raw.shape[1] != dim:
        raise ValueError(
            f"eta_schedule has {eta_schedule_raw.shape[1]} columns, expected dim={dim}."
        )
    eta_review = resample_matrix_to_review_grid(eta_schedule_raw, review_intervals)
    if eta_review.shape != (review_intervals, dim):
        raise ValueError(
            f"eta_review has shape {eta_review.shape}, expected {(review_intervals, dim)}."
        )
    if u is None:
        u = np.asarray([c_tilde[0]], dtype=np.float64)
    if r is None:
        r = np.zeros(dim, dtype=np.float64)
    if distance is None:
        distance = np.zeros(dim, dtype=np.float64)

    warnings: List[str] = []
    if exposure is not None and exposure.shape[0] != dim:
        warnings.append(f"Ignoring exposure vector with shape {exposure.shape}; expected {dim}.")
        exposure = None
    if theta is not None and theta.shape[0] != mu_raw.shape[0]:
        warnings.append(f"theta has {theta.shape[0]} rows but mu has {mu_raw.shape[0]}; kept for metadata only.")
    eta_mode = str(eqn_config.get("eta_mode", "constant_peak"))
    if eta_mode == "weekly_time_varying" and np.allclose(eta_review, eta.reshape(1, dim)):
        warnings.append("eta_mode=weekly_time_varying but eta_review is identical to the constant eta vector.")

    return FluidContext(
        raw_config=raw_config,
        config_path=config_path,
        output_dir=output_dir,
        scenario_manifest=load_scenario_manifest(config_path),
        materialized=materialized,
        total_time=total_time,
        dim=dim,
        q0=float(eqn_config["I0"]),
        z0=np.zeros(dim, dtype=np.float64),
        lambda_param=lambda_param,
        alpha=alpha,
        mu_raw=mu_raw,
        gamma_raw=gamma_raw,
        mu_review=mu_review,
        gamma_review=gamma_review,
        review_dt=review_dt,
        p=p.astype(np.float64),
        u=u.astype(np.float64),
        r=r.astype(np.float64),
        distance=distance.astype(np.float64),
        c_tilde=c_tilde.astype(np.float64),
        eta=eta.astype(np.float64),
        eta_schedule_raw=eta_schedule_raw.astype(np.float64),
        eta_review=eta_review.astype(np.float64),
        eta_mode=eta_mode,
        chi_eta=chi_eta,
        s_eta=s_eta,
        calibration_version=str(eqn_config.get("calibration_version", "")) or None,
        exposure=exposure,
        theta=theta,
        warnings=warnings,
    )


def resolve_period_demand(context: FluidContext, interpretation: str) -> DemandResolution:
    if interpretation == "rate":
        demand = context.mu_review * context.review_dt
    elif interpretation == "period_demand":
        demand = context.mu_review.copy()
    else:
        raise ValueError(f"Unsupported mu interpretation: {interpretation}")

    manifest_total = None
    if context.scenario_manifest:
        metadata = context.scenario_manifest.get("metadata", {})
        if isinstance(metadata, dict):
            manifest_total = metadata.get("total_expected_demand")
    rate_totals = (context.mu_review * context.review_dt).sum(axis=0)
    diagnostics = {
        "mu_interpretation": interpretation,
        "code_evidence": list(MU_RATE_EVIDENCE),
        "review_period_length_h": context.review_dt,
        "num_review_periods": int(context.mu_review.shape[0]),
        "direct_mu_sum_by_location": context.mu_review.sum(axis=0).tolist(),
        "rate_based_total_mean_demand_by_location": rate_totals.tolist(),
        "rate_based_total_mean_demand_all_locations": float(rate_totals.sum()),
        "fluid_total_mean_demand_by_location": demand.sum(axis=0).tolist(),
        "fluid_total_mean_demand_all_locations": float(demand.sum()),
        "scenario_manifest_total_expected_demand": manifest_total,
    }
    warnings: List[str] = []
    if interpretation == "period_demand":
        warnings.append("mu was explicitly treated as period demand; Fangonda paper configs normally store rates.")
    if manifest_total is not None and not np.isclose(float(manifest_total), float(rate_totals.sum()), atol=1e-8):
        warnings.append("Scenario manifest total expected demand differs from mu_review * review_dt.")
    return DemandResolution(
        interpretation=interpretation,
        period_demand=np.asarray(demand, dtype=np.float64),
        diagnostics=diagnostics,
        warnings=warnings,
    )


def gurobi_status_name(status: int) -> str:
    if GRB is None:
        return str(status)
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, str(status))


def solve_fluid_model(
    context: FluidContext,
    demand: np.ndarray,
    with_rate_bound: bool,
    args: argparse.Namespace,
) -> FluidSolution:
    if gp is None or GRB is None:
        raise ImportError("gurobipy is required but is not importable.")

    periods, dim = demand.shape
    if dim != context.dim:
        raise ValueError(f"demand dim {dim} does not match config dim {context.dim}.")
    if context.lambda_param > 0.0 and not np.isclose(context.alpha, 0.5):
        raise ValueError("The deterministic Fangonda QCP implementation requires alpha=0.5.")

    model_name = "fangonda_fluid_qcp" if context.lambda_param > 0.0 else "fangonda_fluid_lp"
    model = gp.Model(model_name)
    model.Params.OutputFlag = int(args.gurobi_output)
    model.Params.LogFile = str(context.output_dir / "gurobi.log")
    if args.gurobi_threads is not None:
        model.Params.Threads = int(args.gurobi_threads)
    if args.gurobi_time_limit is not None:
        model.Params.TimeLimit = float(args.gurobi_time_limit)
    if float(getattr(args, "lambda0_cost_tolerance", 0.0)) < 0.0:
        raise ValueError("--lambda0-cost-tolerance must be nonnegative.")

    a = model.addVars(periods, dim, lb=0.0, name="a")
    q = model.addVars(periods, lb=0.0, name="Q")
    z = model.addVars(periods, dim, lb=0.0, name="Z")
    ell = model.addVars(periods, dim, lb=0.0, name="ell")
    y = None
    if context.lambda_param > 0.0:
        y = model.addVars(periods, dim, lb=0.0, name="y")

    for n in range(periods):
        q_prev = context.q0 if n == 0 else q[n - 1]
        model.addConstr(q[n] == q_prev - gp.quicksum(a[n, k] for k in range(dim)), name=f"warehouse_{n + 1}")

        for k in range(1, dim):
            z_prev = float(context.z0[k]) if n == 0 else z[n - 1, k]
            model.addConstr(
                z[n, k] == z_prev + a[n, k] - float(demand[n, k]) + ell[n, k],
                name=f"local_cha_{n + 1}_{k}",
            )

        z0_prev = float(context.z0[0]) if n == 0 else z[n - 1, 0]
        referrals = gp.quicksum(float(context.p[k - 1]) * ell[n, k] for k in range(1, dim))
        model.addConstr(
            z[n, 0] == z0_prev + a[n, 0] - float(demand[n, 0]) - referrals + ell[n, 0],
            name=f"local_hf_{n + 1}",
        )

        if with_rate_bound:
            for k in range(dim):
                model.addConstr(
                    a[n, k] <= float(context.eta_review[n, k]) * context.review_dt,
                    name=f"rate_bound_{n + 1}_{k}",
                )

        if y is not None:
            for k in range(dim):
                mu_nk = max(float(demand[n, k]), 0.0)
                if mu_nk <= 1e-12:
                    model.addConstr(y[n, k] == 0.0, name=f"fairness_zero_demand_{n + 1}_{k}")
                else:
                    model.addQConstr(
                        y[n, k] * y[n, k] <= mu_nk * a[n, k],
                        name=f"fairness_qc_{n + 1}_{k}",
                    )

    objective = gp.quicksum(
        float(context.c_tilde[k]) * ell[n, k]
        for n in range(periods)
        for k in range(dim)
    )
    if y is not None:
        objective -= 2.0 * context.lambda_param * gp.quicksum(
            y[n, k]
            for n in range(periods)
            for k in range(dim)
        )
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if int(model.SolCount) <= 0:
        raise RuntimeError(f"Gurobi produced no solution; status={gurobi_status_name(int(model.Status))}.")
    primary_status = int(model.Status)
    primary_status_name = gurobi_status_name(primary_status)
    primary_runtime = float(model.Runtime)
    primary_objective = float(model.ObjVal)
    solution_objective = primary_objective
    lexicographic: Optional[Dict[str, Any]] = None

    if (
        np.isclose(float(context.lambda_param), 0.0, rtol=0.0, atol=1e-15)
        and bool(getattr(args, "lambda0_lexicographic_tiebreak", True))
    ):
        cost_tolerance = float(getattr(args, "lambda0_cost_tolerance", 1e-7))
        tie_y = model.addVars(periods, dim, lb=0.0, name="lambda0_fairness_y")
        for n in range(periods):
            for k in range(dim):
                mu_nk = max(float(demand[n, k]), 0.0)
                if mu_nk <= 1e-12:
                    model.addConstr(tie_y[n, k] == 0.0, name=f"lambda0_fairness_zero_demand_{n + 1}_{k}")
                else:
                    model.addQConstr(
                        tie_y[n, k] * tie_y[n, k] <= mu_nk * a[n, k],
                        name=f"lambda0_fairness_qc_{n + 1}_{k}",
                    )
        model.addConstr(
            objective <= primary_objective + cost_tolerance,
            name="lambda0_primary_cost_tolerance",
        )
        fairness_tiebreak_objective = 2.0 * gp.quicksum(
            tie_y[n, k]
            for n in range(periods)
            for k in range(dim)
        )
        model.setObjective(fairness_tiebreak_objective, GRB.MAXIMIZE)
        model.optimize()
        if int(model.SolCount) <= 0:
            raise RuntimeError(
                "Gurobi produced no lambda=0 lexicographic tie-break solution; "
                f"status={gurobi_status_name(int(model.Status))}."
            )
        realized_cost = float(sum(
            float(context.c_tilde[k]) * ell[n, k].X
            for n in range(periods)
            for k in range(dim)
        ))
        tie_fairness_proxy = float(model.ObjVal)
        lexicographic = {
            "enabled": True,
            "primary_status": primary_status,
            "primary_status_name": primary_status_name,
            "primary_runtime": primary_runtime,
            "primary_operational_cost": primary_objective,
            "cost_tolerance": cost_tolerance,
            "realized_operational_cost_after_tiebreak": realized_cost,
            "cost_tolerance_slack_used": realized_cost - primary_objective,
            "tie_breaker": "maximize demand-weighted alpha-fair utility among cost-optimal lambda=0 solutions",
            "tie_breaker_fairness_proxy": tie_fairness_proxy,
            "secondary_status": int(model.Status),
            "secondary_status_name": gurobi_status_name(int(model.Status)),
            "secondary_runtime": float(model.Runtime),
        }
        solution_objective = realized_cost

    allocation = np.zeros((periods, dim), dtype=np.float64)
    warehouse = np.zeros(periods, dtype=np.float64)
    local_inventory = np.zeros((periods, dim), dtype=np.float64)
    shortage = np.zeros((periods, dim), dtype=np.float64)
    for n in range(periods):
        warehouse[n] = float(q[n].X)
        for k in range(dim):
            allocation[n, k] = float(a[n, k].X)
            local_inventory[n, k] = float(z[n, k].X)
            shortage[n, k] = float(ell[n, k].X)

    return FluidSolution(
        status=int(model.Status),
        status_name=gurobi_status_name(int(model.Status)),
        runtime=float(model.Runtime) + (primary_runtime if lexicographic else 0.0),
        objective=solution_objective,
        allocation=allocation,
        warehouse=warehouse,
        local_inventory=local_inventory,
        shortage=shortage,
        lexicographic=lexicographic,
    )


def validate_solution(
    context: FluidContext,
    demand: np.ndarray,
    solution: FluidSolution,
    with_rate_bound: bool,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    a = solution.allocation
    q = solution.warehouse
    z = solution.local_inventory
    ell = solution.shortage
    periods, dim = demand.shape

    warehouse_residuals: List[float] = []
    local_residuals: List[float] = []
    for n in range(periods):
        q_prev = context.q0 if n == 0 else q[n - 1]
        warehouse_residuals.append(float(q[n] - (q_prev - a[n].sum())))
        for k in range(1, dim):
            z_prev = context.z0[k] if n == 0 else z[n - 1, k]
            local_residuals.append(float(z[n, k] - (z_prev + a[n, k] - demand[n, k] + ell[n, k])))
        z0_prev = context.z0[0] if n == 0 else z[n - 1, 0]
        referral = float(np.dot(context.p, ell[n, 1:]))
        local_residuals.append(float(z[n, 0] - (z0_prev + a[n, 0] - demand[n, 0] - referral + ell[n, 0])))

    min_variable = float(min(np.min(a), np.min(q), np.min(z), np.min(ell))) if periods > 0 else 0.0
    rate_bound_violation = 0.0
    if with_rate_bound:
        rhs = context.eta_review * context.review_dt
        rate_bound_violation = max(0.0, float(np.max(a - rhs)))
    allocation_excess = float(a.sum() - context.q0)
    checks = {
        "all_variables_nonnegative": bool(min_variable >= -tol),
        "min_variable_value": min_variable,
        "maximum_warehouse_balance_residual": float(np.max(np.abs(warehouse_residuals))) if warehouse_residuals else 0.0,
        "maximum_local_balance_residual": float(np.max(np.abs(local_residuals))) if local_residuals else 0.0,
        "maximum_rate_bound_violation": rate_bound_violation,
        "rate_bound_satisfied": bool((not with_rate_bound) or rate_bound_violation <= tol),
        "total_allocation": float(a.sum()),
        "total_allocation_minus_q0": allocation_excess,
        "total_allocation_within_initial_warehouse": bool(allocation_excess <= tol),
    }
    checks["passed"] = bool(
        checks["all_variables_nonnegative"]
        and checks["maximum_warehouse_balance_residual"] <= tol
        and checks["maximum_local_balance_residual"] <= tol
        and checks["rate_bound_satisfied"]
        and checks["total_allocation_within_initial_warehouse"]
    )
    return checks


def alpha_fair_utility_numpy(
    allocation: np.ndarray,
    demand: np.ndarray,
    alpha: float,
    mu_epsilon: float = 1e-12,
) -> np.ndarray:
    """Demand-weighted alpha-fair period utility with explicit zero-demand masking."""
    a = np.asarray(allocation, dtype=np.float64)
    mu = np.asarray(demand, dtype=np.float64)
    if a.shape != mu.shape:
        raise ValueError(f"allocation shape {a.shape} must match demand shape {mu.shape}.")
    if not (0.0 <= float(alpha) < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}.")
    active = mu > float(mu_epsilon)
    safe_ratio = np.zeros_like(a, dtype=np.float64)
    safe_ratio[active] = np.maximum(a[active], 0.0) / mu[active]
    contribution = np.zeros_like(a, dtype=np.float64)
    contribution[active] = mu[active] * np.power(safe_ratio[active], 1.0 - float(alpha))
    return contribution.sum(axis=1) / (1.0 - float(alpha))


def fluid_objective_metrics(
    context: FluidContext,
    demand: np.ndarray,
    solution: FluidSolution,
    with_rate_bound: bool,
) -> Dict[str, Any]:
    shortage_cost = float(np.sum(solution.shortage * context.c_tilde.reshape(1, -1)))
    fairness_by_period = alpha_fair_utility_numpy(solution.allocation, demand, context.alpha)
    fairness_utility = float(np.sum(fairness_by_period))
    weighted_fairness = float(context.lambda_param * fairness_utility)
    reconstructed_objective = shortage_cost - weighted_fairness
    objective_value = float(solution.objective) if solution.objective is not None else np.nan
    reconstruction_error = (
        float(reconstructed_objective - objective_value)
        if np.isfinite(objective_value)
        else np.nan
    )

    total_allocation = float(np.sum(solution.allocation))
    hf_share = (
        float(np.sum(solution.allocation[:, 0]) / total_allocation)
        if total_allocation > 1e-12
        else np.nan
    )

    cap_quantity = context.eta_review * context.review_dt
    cap_active = cap_quantity > 1e-12
    if with_rate_bound and np.any(cap_active):
        cap_binding_fraction = float(
            np.mean(solution.allocation[cap_active] >= cap_quantity[cap_active] - 1e-6)
        )
    else:
        cap_binding_fraction = np.nan

    ratio_diagnostics = demand_adjusted_fairness_diagnostics(
        solution.allocation[:, None, :],
        demand,
        eta_hist=cap_quantity if with_rate_bound else None,
        dt=1.0,
    )
    scalar_ratio_diagnostics = {
        key: value
        for key, value in ratio_diagnostics.items()
        if np.asarray(value).ndim == 0
    }
    return {
        "fluid_shortage_cost": shortage_cost,
        "fluid_fairness_utility": fairness_utility,
        "fluid_weighted_fairness_utility": weighted_fairness,
        "fluid_objective_reconstructed": reconstructed_objective,
        "objective_reconstruction_error": reconstruction_error,
        "fluid_hf_shipment_share": hf_share,
        "eta_cap_binding_fraction": cap_binding_fraction,
        "demand_adjusted_allocation_ratio_diagnostics": ratio_diagnostics,
        **{f"fluid_{key}": value for key, value in scalar_ratio_diagnostics.items()},
    }


def location_label(index: int) -> str:
    return LOCATION_LABELS[index] if index < len(LOCATION_LABELS) else f"loc_{index}"


def variant_name(context: FluidContext, with_rate_bound: bool, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    family = "Fluid-QCP" if context.lambda_param > 0.0 else "Fluid-LP"
    return f"{family}-eta" if with_rate_bound else family


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def scenario_name(context: FluidContext) -> Optional[str]:
    if context.scenario_manifest:
        name = context.scenario_manifest.get("scenario_name")
        if name:
            return str(name)
    if context.materialized:
        name = context.materialized.get("run_name")
        if name:
            return str(name)
    return None


def build_summary(
    context: FluidContext,
    demand: np.ndarray,
    solution: FluidSolution,
    validation: Mapping[str, Any],
    demand_resolution: DemandResolution,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    scenario = scenario_name(context)
    final_local = solution.local_inventory[-1] if solution.local_inventory.size else context.z0
    metrics = fluid_objective_metrics(context, demand, solution, args.with_rate_bound)
    summary = {
        "scenario_name": scenario,
        "variant_name": variant_name(context, args.with_rate_bound, args.variant_name),
        "with_rate_bound": bool(args.with_rate_bound),
        "gurobi_status": solution.status,
        "gurobi_status_name": solution.status_name,
        "gurobi_runtime": solution.runtime,
        "fluid_objective": solution.objective,
        "mu_interpretation": demand_resolution.interpretation,
        "period_length_h": context.review_dt,
        "num_periods": int(demand.shape[0]),
        "dim": context.dim,
        "q0": context.q0,
        "z0": context.z0.tolist(),
        "lambda_param": context.lambda_param,
        "alpha": context.alpha,
        "calibration_version": context.calibration_version,
        "fairness_spec": FAIRNESS_SPEC,
        "fairness_includes_hf": FAIRNESS_INCLUDES_HF,
        "fairness_period_h_factor": FAIRNESS_PERIOD_H_FACTOR,
        "u": context.u.tolist(),
        "referral_costs_r": context.r.tolist(),
        "distance_miles": context.distance.tolist(),
        "p": context.p.tolist(),
        "effective_shortage_costs": context.c_tilde.tolist(),
        "eta_mode": context.eta_mode,
        "eta_profile": context.eta_mode,
        "chi_eta": context.chi_eta,
        "s_eta": context.s_eta,
        "eta": context.eta.tolist(),
        "eta_schedule_raw_shape": list(context.eta_schedule_raw.shape),
        "eta_review_shape": list(context.eta_review.shape),
        "eta_review": context.eta_review.tolist(),
        "eta_min_by_location": np.min(context.eta_review, axis=0).tolist(),
        "eta_max_by_location": np.max(context.eta_review, axis=0).tolist(),
        "eta_min_by_period": np.min(context.eta_review, axis=1).tolist(),
        "eta_max_by_period": np.max(context.eta_review, axis=1).tolist(),
        "rate_bound_rhs_eta_h": (context.eta_review * context.review_dt).tolist() if args.with_rate_bound else None,
        "total_mean_demand_by_location": demand.sum(axis=0).tolist(),
        "aggregate_total_mean_demand": float(demand.sum()),
        "total_allocation_by_location": solution.allocation.sum(axis=0).tolist(),
        "total_shortage_by_location": solution.shortage.sum(axis=0).tolist(),
        "final_warehouse_inventory": float(solution.warehouse[-1]) if solution.warehouse.size else context.q0,
        "final_local_inventories": final_local.tolist(),
        "lambda0_lexicographic": solution.lexicographic or {"enabled": False},
        **metrics,
        "validation": dict(validation),
        "warnings": list(context.warnings) + list(demand_resolution.warnings),
        "config_path": str(context.config_path),
        "output_dir": str(context.output_dir),
        "gurobi_log_path": str(context.output_dir / "gurobi.log"),
        "materialized": context.materialized,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }
    if context.scenario_manifest:
        metadata = context.scenario_manifest.get("metadata")
        if isinstance(metadata, dict):
            summary["scenario_metadata"] = metadata
    return summary


def allocation_rows(context: FluidContext, demand: np.ndarray, solution: FluidSolution) -> Iterable[Dict[str, Any]]:
    for n in range(solution.allocation.shape[0]):
        for k in range(context.dim):
            yield {
                "period": n + 1,
                "time_start": n * context.review_dt,
                "time_end": (n + 1) * context.review_dt,
                "location_index": k,
                "location_name": location_label(k),
                "allocation_quantity": solution.allocation[n, k],
                "allocation_rate": solution.allocation[n, k] / context.review_dt,
                "period_mean_demand": demand[n, k],
                "eta": context.eta_review[n, k],
                "eta_h": context.eta_review[n, k] * context.review_dt,
                "fluid_shortage": solution.shortage[n, k],
                "fluid_end_local_inventory": solution.local_inventory[n, k],
                "fluid_end_warehouse_inventory": solution.warehouse[n],
            }


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def flatten_for_csv(payload: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            row[key] = json.dumps(value, default=json_default)
        else:
            row[key] = value
    return row


def update_parent_comparison(output_dir: Path) -> Optional[Path]:
    parent = output_dir.parent
    rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []
    for path in sorted(parent.glob("*/comparison_row.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
    if not rows:
        return None
    out_path = parent / "fluid_comparison.csv"
    write_csv(out_path, rows, fieldnames)
    return out_path


def write_outputs(
    context: FluidContext,
    demand: np.ndarray,
    demand_resolution: DemandResolution,
    solution: FluidSolution,
    summary: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Path]:
    out = context.output_dir
    paths = {
        "allocation_csv": out / "allocation_plan.csv",
        "allocation_npz": out / "allocation_plan.npz",
        "summary_json": out / "fluid_solution_summary.json",
        "diagnostics_json": out / "fluid_diagnostics.json",
        "comparison_row_csv": out / "comparison_row.csv",
    }

    allocation_fieldnames = [
        "period",
        "time_start",
        "time_end",
        "location_index",
        "location_name",
        "allocation_quantity",
        "allocation_rate",
        "period_mean_demand",
        "eta",
        "eta_h",
        "fluid_shortage",
        "fluid_end_local_inventory",
        "fluid_end_warehouse_inventory",
    ]
    write_csv(paths["allocation_csv"], allocation_rows(context, demand, solution), allocation_fieldnames)

    scenario = scenario_name(context) or ""
    np.savez_compressed(
        paths["allocation_npz"],
        allocation_plan=solution.allocation,
        warehouse_inventory=solution.warehouse,
        local_inventory=solution.local_inventory,
        shortage=solution.shortage,
        period_mean_demand=demand,
        mu_review=context.mu_review,
        gamma_review=context.gamma_review,
        eta=context.eta,
        eta_schedule_raw=context.eta_schedule_raw,
        eta_review=context.eta_review,
        eta_mode=np.asarray([context.eta_mode]),
        eta_profile=np.asarray([context.eta_mode]),
        chi_eta=np.asarray([context.chi_eta], dtype=np.float64),
        s_eta=np.asarray([context.s_eta], dtype=np.float64),
        p=context.p,
        u=context.u,
        r=context.r,
        distance=context.distance,
        c=context.c_tilde,
        c_tilde=context.c_tilde,
        q0=np.asarray([context.q0], dtype=np.float64),
        z0=context.z0,
        calibration_version=np.asarray([context.calibration_version or ""]),
        review_dt=np.asarray([context.review_dt], dtype=np.float64),
        lambda_param=np.asarray([context.lambda_param], dtype=np.float64),
        alpha=np.asarray([context.alpha], dtype=np.float64),
        fairness_spec=np.asarray([FAIRNESS_SPEC]),
        fairness_includes_hf=np.asarray([FAIRNESS_INCLUDES_HF]),
        fairness_period_h_factor=np.asarray([FAIRNESS_PERIOD_H_FACTOR]),
        objective_value=np.asarray([solution.objective if solution.objective is not None else np.nan], dtype=np.float64),
        lambda0_lexicographic_enabled=np.asarray([
            bool((solution.lexicographic or {}).get("enabled", False))
        ]),
        lambda0_primary_operational_cost=np.asarray([
            (solution.lexicographic or {}).get("primary_operational_cost", np.nan)
        ], dtype=np.float64),
        lambda0_cost_tolerance=np.asarray([
            (solution.lexicographic or {}).get("cost_tolerance", np.nan)
        ], dtype=np.float64),
        lambda0_realized_operational_cost_after_tiebreak=np.asarray([
            (solution.lexicographic or {}).get("realized_operational_cost_after_tiebreak", np.nan)
        ], dtype=np.float64),
        lambda0_tie_breaker_fairness_proxy=np.asarray([
            (solution.lexicographic or {}).get("tie_breaker_fairness_proxy", np.nan)
        ], dtype=np.float64),
        fluid_shortage_cost=np.asarray([summary.get("fluid_shortage_cost", np.nan)], dtype=np.float64),
        fluid_fairness_utility=np.asarray([summary.get("fluid_fairness_utility", np.nan)], dtype=np.float64),
        fluid_weighted_fairness_utility=np.asarray([summary.get("fluid_weighted_fairness_utility", np.nan)], dtype=np.float64),
        fluid_objective_reconstructed=np.asarray([summary.get("fluid_objective_reconstructed", np.nan)], dtype=np.float64),
        objective_reconstruction_error=np.asarray([summary.get("objective_reconstruction_error", np.nan)], dtype=np.float64),
        fluid_hf_shipment_share=np.asarray([summary.get("fluid_hf_shipment_share", np.nan)], dtype=np.float64),
        eta_cap_binding_fraction=np.asarray([summary.get("eta_cap_binding_fraction", np.nan)], dtype=np.float64),
        solver_status=np.asarray([solution.status], dtype=np.int64),
        solver_status_name=np.asarray([solution.status_name]),
        solve_time=np.asarray([solution.runtime], dtype=np.float64),
        with_rate_bound=np.asarray([bool(summary["with_rate_bound"])]),
        scenario_name=np.asarray([scenario]),
        mu_interpretation=np.asarray([demand_resolution.interpretation]),
    )

    with paths["summary_json"].open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)
    with paths["diagnostics_json"].open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=json_default)

    comparison = {
        "scenario_name": summary.get("scenario_name"),
        "variant_name": summary.get("variant_name"),
        "with_rate_bound": summary.get("with_rate_bound"),
        "gurobi_status_name": summary.get("gurobi_status_name"),
        "fluid_objective": summary.get("fluid_objective"),
        "fluid_shortage_cost": summary.get("fluid_shortage_cost"),
        "fluid_fairness_utility": summary.get("fluid_fairness_utility"),
        "fluid_weighted_fairness_utility": summary.get("fluid_weighted_fairness_utility"),
        "fluid_objective_reconstructed": summary.get("fluid_objective_reconstructed"),
        "objective_reconstruction_error": summary.get("objective_reconstruction_error"),
        "fluid_hf_shipment_share": summary.get("fluid_hf_shipment_share"),
        "eta_cap_binding_fraction": summary.get("eta_cap_binding_fraction"),
        "fluid_demand_weighted_mean_f": summary.get("fluid_demand_weighted_mean_f"),
        "fluid_demand_weighted_cv_f": summary.get("fluid_demand_weighted_cv_f"),
        "fluid_planned_coverage_ratio_gap": summary.get("fluid_planned_coverage_ratio_gap"),
        "fluid_demand_proportional_benchmark_rmse": summary.get("fluid_demand_proportional_benchmark_rmse"),
        "total_mean_demand": summary.get("aggregate_total_mean_demand"),
        "total_allocation": summary.get("validation", {}).get("total_allocation"),
        "config_path": str(context.config_path),
        "output_dir": str(context.output_dir),
        "allocation_plan_path": str(paths["allocation_npz"]),
    }
    flat = flatten_for_csv(comparison)
    write_csv(paths["comparison_row_csv"], [flat], list(flat.keys()))
    comparison_path = update_parent_comparison(out)
    if comparison_path is not None:
        paths["comparison_csv"] = comparison_path
    return paths


def print_result(summary: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    print(
        json.dumps(
            {
                "scenario": summary.get("scenario_name"),
                "variant": summary.get("variant_name"),
                "gurobi_status": summary.get("gurobi_status_name"),
                "fluid_objective": summary.get("fluid_objective"),
                "fluid_shortage_cost": summary.get("fluid_shortage_cost"),
                "fluid_fairness_utility": summary.get("fluid_fairness_utility"),
                "fluid_weighted_fairness_utility": summary.get("fluid_weighted_fairness_utility"),
                "objective_reconstruction_error": summary.get("objective_reconstruction_error"),
                "fluid_hf_shipment_share": summary.get("fluid_hf_shipment_share"),
                "eta_cap_binding_fraction": summary.get("eta_cap_binding_fraction"),
                "total_mean_demand": summary.get("aggregate_total_mean_demand"),
                "total_allocation": summary.get("validation", {}).get("total_allocation"),
                "output_dir": summary.get("output_dir"),
                "allocation_plan": str(paths["allocation_npz"]),
                "warnings": summary.get("warnings", []),
            },
            indent=2,
            default=json_default,
        )
    )
    print("Saved files:")
    for path in paths.values():
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    if args.num_paths <= 0:
        raise ValueError("--num-paths must be positive.")
    context = build_context(args)
    demand_resolution = resolve_period_demand(context, args.mu_interpretation)
    solution = solve_fluid_model(context, demand_resolution.period_demand, args.with_rate_bound, args)
    validation = validate_solution(context, demand_resolution.period_demand, solution, args.with_rate_bound)
    summary = build_summary(context, demand_resolution.period_demand, solution, validation, demand_resolution, args)
    diagnostics = {
        "demand_diagnostics": demand_resolution.diagnostics,
        "solution_validation": validation,
        "fluid_objective_metrics": {
            key: summary.get(key)
            for key in (
                "fluid_shortage_cost",
                "fluid_fairness_utility",
                "fluid_weighted_fairness_utility",
                "fluid_objective_reconstructed",
                "objective_reconstruction_error",
                "fluid_hf_shipment_share",
                "eta_cap_binding_fraction",
            )
        },
        "lambda0_lexicographic": summary.get("lambda0_lexicographic", {"enabled": False}),
        "demand_adjusted_allocation_ratio_diagnostics": summary.get(
            "demand_adjusted_allocation_ratio_diagnostics", {}
        ),
        "allocation_shape": list(solution.allocation.shape),
        "warehouse_shape": list(solution.warehouse.shape),
        "local_inventory_shape": list(solution.local_inventory.shape),
        "shortage_shape": list(solution.shortage.shape),
        "warnings": summary["warnings"],
    }
    paths = write_outputs(context, demand_resolution.period_demand, demand_resolution, solution, summary, diagnostics)
    print_result(summary, paths)


if __name__ == "__main__":
    main()
