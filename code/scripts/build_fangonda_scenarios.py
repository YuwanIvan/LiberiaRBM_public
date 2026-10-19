#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.fangonda_weekly import scenarios as scenario_registry
from scripts.build_fangonda_calibration import (
    compute_calibration,
    compute_fangonda_eta,
    compute_normalized_cost_arrays,
    default_data_dir,
)
from equation import (
    ETA_MODES,
    FAIRNESS_INCLUDES_HF,
    FAIRNESS_PERIOD_H_FACTOR,
    FAIRNESS_SPEC,
    compute_weekly_time_varying_eta_schedule,
)
from Net import G_INIT_METHOD_CHOICES, INIT_METHOD_CHOICES
import demand_correlation_calibration as correlation_calibration
import evaluation_bank
import rho_validation_bank
import rho_validation_statistics

SAMPLING_MODE_CHOICES = (
    "box",
    "rollout",
    "shortage",
    "mixture",
    "mixed",
    "lognormal_proposal",
    "lognormal",
)
REFERENCE_POLICY_CHOICES = (
    "static_proportional",
    "inventory_scaled_proportional",
    "myopic",
    "status_quo",
    "fluid_optimal_rollout",
    "reference_mixture",
    "even",
)
PROPOSAL_SAMPLER_CHOICES = (
    "empirical",
    "uniform_tube",
    "mixture_uniform_lowq_boundary",
    "truncated_lognormal",
)
ACTIVATION_CHOICES = (
    "ReLU",
    "LeakyReLU",
    "GELU",
    "SELU",
    "SiLU",
    "CELU",
)
LOSS_MODE_CHOICES = (
    "ce_consistency",
    "ce_no_consistency",
    "martingale_qgrad",
)


def parse_int_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected boolean value such as 0/1 or true/false, got {value!r}."
    )

NET_CONFIG_OVERRIDES = (
    ("num_neurons", int),
    ("activation", str),
    ("init_method", str),
    ("h_init_method", str),
    ("g_init_method", str),
    ("g_output_bias", parse_bool),
    ("g_prior_fluid_plan_path", str),
    ("g_prior_epsilon_a", float),
    ("g_prior_lower_bound", float),
    ("g_prior_upper_bound", float),
    ("g_prior_cap_offset", float),
    ("final_min_iterations", int),
    ("inner_min_iterations", int),
    ("final_learning_rate", float),
    ("inner_learning_rate", float),
    ("final_milestones", parse_int_list),
    ("inner_milestones", parse_int_list),
    ("gamma", float),
    ("inner_gamma", float),
    ("final_gamma", float),
    ("grad_norm", float),
    ("h_grad_norm", float),
    ("g_grad_norm", float),
    ("grad_clip_rule", str),
    ("final_grad_clip_rule", str),
    ("inner_grad_clip_rule", str),
    ("input_mode", str),
    ("q0_input_scale", float),
    ("h_output_scale", float),
    ("checkpoint_reparam_load_mode", str),
    ("state_support_sample_size", int),
    ("proposal_sampler", str),
    ("sampling_mode", str),
    ("rollout_pool_size", int),
    ("sampling_seed", int),
    ("uniform_z_radius_mult", float),
    ("uniform_q_radius_frac", float),
    ("total_inventory_buffer", float),
    ("mix_rollout", float),
    ("mix_box", float),
    ("mix_shortage", float),
    ("shortage_q_frac", float),
    ("shortage_z_frac", float),
    ("lognormal_delta_z", float),
    ("lognormal_delta_q", float),
    ("lognormal_r_min", int),
    ("initial_state_mode", str),
    ("initial_perturb_prob", float),
    ("initial_q_perturb_frac", float),
    ("initial_z_perturb_frac", float),
    ("initial_perturb_minus_prob", float),
    ("initial_perturb_plus_prob", float),
    ("initial_perturb_minus_frac", float),
    ("initial_perturb_plus_frac", float),
    ("initial_dirichlet_alpha", float),
    ("consistency_coef", float),
    ("gradient_assembly_mode", str),
    ("h_base_grad_multiplier", float),
    ("g_base_grad_multiplier", float),
    ("h_consistency_coef", float),
    ("g_consistency_coef", float),
    ("prev_init_same_params_only", parse_bool),
    ("state_support_logging", parse_bool),
    ("train_seed", int),
    ("validation_every", int),
    ("loss_mode", str),
    ("component_seed_isolation", parse_bool),
    ("h_network_seed", int),
    ("g_network_seed", int),
    ("qgrad_network_seed", int),
    ("save_initial_checkpoints", parse_bool),
)


def project_root() -> Path:
    return PROJECT_ROOT


def default_output_root() -> Path:
    env_output = os.environ.get("OUTPUT_DIR", "").strip()
    if env_output:
        return Path(env_output).expanduser()
    return project_root() / "output_fangonda_dwaf_v1"


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root()),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _load_csv_matrix(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    return np.atleast_2d(np.asarray(data, dtype=float))


def _load_csv_vector(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    return np.asarray(data, dtype=float).reshape(-1)


def load_baseline_bundle() -> Dict[str, Any]:
    root = project_root()
    baseline_dir = root / "configs" / "fangonda_weekly"
    config_template = scenario_registry.load_base_dict()
    if not (baseline_dir / "mu.csv").is_file():
        calibration = compute_calibration(data_dir=default_data_dir())
        for key in ("theta", "exposure", "mu", "gamma", "p", "distance", "u", "r", "c", "h", "eta"):
            _write_csv(baseline_dir / f"{key}.csv", calibration[key])
    arrays = {
        "theta": _load_csv_vector(baseline_dir / "theta.csv"),
        "exposure": _load_csv_vector(baseline_dir / "exposure.csv"),
        "mu": _load_csv_matrix(baseline_dir / "mu.csv"),
        "gamma": _load_csv_matrix(baseline_dir / "gamma.csv"),
        "p": _load_csv_vector(baseline_dir / "p.csv"),
        "distance": _load_csv_vector(baseline_dir / "distance.csv"),
        "u": _load_csv_vector(baseline_dir / "u.csv"),
        "r": _load_csv_vector(baseline_dir / "r.csv"),
        "c": _load_csv_vector(baseline_dir / "c.csv"),
        "h": _load_csv_vector(baseline_dir / "h.csv"),
        "eta": _load_csv_vector(baseline_dir / "eta.csv"),
    }
    return {
        "project_root": root,
        "baseline_dir": baseline_dir,
        "config_template": config_template,
        "arrays": arrays,
    }


def _write_csv(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 1:
        np.savetxt(path, arr.reshape(-1, 1), delimiter=",", fmt="%.12g")
    elif arr.ndim == 2:
        np.savetxt(path, arr, delimiter=",", fmt="%.12g")
    elif arr.ndim == 3:
        np.savetxt(path, arr.reshape(arr.shape[0], -1), delimiter=",", fmt="%.12g")
    else:
        raise ValueError(f"Unsupported array rank for CSV export: {arr.ndim}")


def _gamma_to_diag(gamma: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(gamma, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == dim:
        return arr.copy()
    if arr.ndim == 2 and arr.shape[1] == dim * dim:
        return np.diagonal(arr.reshape(arr.shape[0], dim, dim), axis1=1, axis2=2).copy()
    if arr.ndim == 3 and arr.shape[1:] == (dim, dim):
        return np.diagonal(arr, axis1=1, axis2=2).copy()
    raise ValueError(f"Unexpected gamma shape {arr.shape} for dim={dim}.")


def _build_full_covariance(gamma_diag: np.ndarray, rho: float) -> np.ndarray:
    num_rows, dim = gamma_diag.shape
    gamma_full = np.zeros((num_rows, dim, dim), dtype=float)
    for t in range(num_rows):
        variances = np.clip(gamma_diag[t], 0.0, None)
        std = np.sqrt(variances)
        cov = float(rho) * np.outer(std, std)
        np.fill_diagonal(cov, variances)
        gamma_full[t] = cov
    return gamma_full


def apply_realized_rho_calibration(
    patched_bundle: Dict[str, Any],
    resolved_spec: Dict[str, Any],
    *,
    calibration_report_path: Path,
    calibration_report_sha256: str,
    validation_report_path: Path,
    validation_report_sha256: str,
) -> None:
    """Replace scalar latent rho with a validated week-specific matrix."""

    if resolved_spec.get("rho_interpretation") != "realized_discrete_count_pearson":
        raise ValueError("Rho calibration artifacts are only valid for calibrated-rho scenarios.")
    calibration_report_path = calibration_report_path.expanduser().resolve()
    validation_report_path = validation_report_path.expanduser().resolve()
    if evaluation_bank.file_sha256(calibration_report_path) != calibration_report_sha256:
        raise ValueError("Rho calibration report SHA-256 mismatch.")
    if evaluation_bank.file_sha256(validation_report_path) != validation_report_sha256:
        raise ValueError("Rho validation report SHA-256 mismatch.")
    calibration = json.loads(calibration_report_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_report_path.read_text(encoding="utf-8"))
    if calibration.get("schema_version") != "fangonda-realized-rho-calibration-v1" or calibration.get("frozen") is not True:
        raise ValueError("Rho calibration report is not frozen v1.")
    if calibration.get("numerical_method", {}).get("name") != correlation_calibration.CALIBRATION_METHOD:
        raise ValueError("Rho calibration report does not use the audited exact discrete-threshold method.")
    if (
        validation.get("schema_version") != rho_validation_statistics.VALIDATION_SCHEMA_VERSION
        or validation.get("frozen") is not True
        or validation.get("validation_passed") is not True
    ):
        raise ValueError("Rho calibration has not passed the 200,000-path validator-v2 gate.")
    if calibration.get("scenario") != resolved_spec["name"]:
        raise ValueError("Rho calibration scenario identity mismatch.")
    if not np.isclose(float(calibration.get("target_realized_rho")), float(resolved_spec["corr_rho"])):
        raise ValueError("Rho calibration target differs from the scenario target.")
    validation_link = validation.get("calibration_report", {})
    if (
        not rho_validation_bank.shared_paths_equivalent(
            validation_link.get("path", ""),
            calibration_report_path,
        )
        or validation_link.get("sha256") != calibration_report_sha256
    ):
        raise ValueError("Rho validation does not certify the selected calibration report.")
    calibration_final = validation.get("calibration_slurm_finalization", {})
    rho_validation_bank.validate_slurm_finalization(
        calibration_final.get("path", ""),
        calibration_final.get("sha256", ""),
        artifact_path=calibration_report_path,
        artifact_sha256=calibration_report_sha256,
    )
    if int(validation.get("num_paths", -1)) != 200000 or int(validation.get("seed", -1)) != 2026082203:
        raise ValueError("Rho validation did not use the frozen 200,000-path bank contract.")
    matrix_record = calibration["latent_matrices"]
    matrix_path = calibration_report_path.parent / matrix_record["path"]
    if evaluation_bank.file_sha256(matrix_path) != matrix_record["file_sha256"]:
        raise ValueError("Calibrated latent matrix file SHA-256 mismatch.")
    matrices = np.load(matrix_path, allow_pickle=False)
    if evaluation_bank.array_sha256(matrices) != matrix_record["array_sha256"]:
        raise ValueError("Calibrated latent matrix content SHA-256 mismatch.")
    dim = patched_bundle["arrays"]["mu"].shape[1]
    gamma_diag = _gamma_to_diag(patched_bundle["arrays"]["gamma"], dim)
    bank_record = validation.get("validation_bank", {})
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
    if matrices.shape != (gamma_diag.shape[0], dim, dim):
        raise ValueError("Calibrated latent matrices do not match the scenario gamma grid.")
    gamma = np.zeros_like(matrices, dtype=np.float64)
    for week in range(gamma_diag.shape[0]):
        std = np.sqrt(np.clip(gamma_diag[week], 0.0, None))
        gamma[week] = matrices[week] * np.outer(std, std)
        np.fill_diagonal(gamma[week], gamma_diag[week])
    patched_bundle["arrays"]["gamma"] = gamma
    patched_bundle["metadata"].update(
        {
            "gamma_storage": "full_flattened",
            "correlation_mode": "week_specific_latent_matrix_calibrated_to_realized_counts",
            "rho_interpretation": "realized_discrete_count_pearson",
            "rho_calibration_report": {"path": str(calibration_report_path), "sha256": calibration_report_sha256},
            "rho_validation_report": {"path": str(validation_report_path), "sha256": validation_report_sha256},
            "latent_correlation_matrix_file_sha256": matrix_record["file_sha256"],
            "latent_correlation_matrix_array_sha256": matrix_record["array_sha256"],
        }
    )
def _compute_cost_arrays(distance: np.ndarray, p: np.ndarray, spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
    w = float(spec["hourly_opportunity_cost"])
    kappa = float(spec["motorcycle_taxi_fare_per_mile"])
    referral_hours = float(spec["referral_caregiver_hours"])
    normalization_hours = float(spec.get("normalization_caregiver_hours", 64.0))
    cost_arrays = compute_normalized_cost_arrays(
        distance,
        p,
        w=w,
        kappa=kappa,
        referral_hours=referral_hours,
        normalization_caregiver_hours=normalization_hours,
    )
    return {
        "u": cost_arrays["u"],
        "r": cost_arrays["r"],
        "c": cost_arrays["c"],
    }


def compute_eta_schedule_for_config(arrays: Dict[str, np.ndarray], config: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
    eqn_cfg = config["eqn_config"]
    eta_mode = str(eqn_cfg.get("eta_mode", eqn_cfg.get("eta_profile", "constant_peak")))
    chi_eta = float(eqn_cfg.get("chi_eta", 1.75))
    s_eta = float(eqn_cfg.get("s_eta", 0.0))
    if eta_mode not in ETA_MODES:
        raise ValueError(f"eta_mode must be one of {ETA_MODES}, got {eta_mode!r}.")
    num_rows, dim = np.asarray(arrays["mu"]).shape
    if eta_mode == "constant_peak":
        schedule = np.repeat(np.asarray(arrays["eta"], dtype=float).reshape(1, dim), num_rows, axis=0)
        details = {
            "eta_mode": eta_mode,
            "eta_profile": eta_mode,
            "chi_eta": chi_eta,
            "s_eta": s_eta,
            "eta_schedule_rule": "constant_peak_broadcast_from_eta_csv",
            "eta_schedule_shape": list(schedule.shape),
        }
    else:
        schedule, formula_details = compute_weekly_time_varying_eta_schedule(
            arrays["mu"],
            arrays["gamma"],
            arrays["p"],
            float(eqn_cfg["I0"]),
            float(eqn_cfg["total_time"]),
            chi_eta=chi_eta,
            s_eta=s_eta,
            return_details=True,
        )
        details = {
            "eta_mode": eta_mode,
            "eta_profile": eta_mode,
            "chi_eta": chi_eta,
            "s_eta": s_eta,
            "eta_schedule_rule": "weekly_time_varying_formula",
            "eta_schedule_file_role": str(eqn_cfg.get("eta_schedule_file_role", "active_input")),
            "eta_schedule_shape": list(schedule.shape),
            **formula_details,
        }
    details.update(
        {
            "eta_min_by_location": np.min(schedule, axis=0).tolist(),
            "eta_max_by_location": np.max(schedule, axis=0).tolist(),
            "eta_min_by_week": np.min(schedule, axis=1).tolist(),
            "eta_max_by_week": np.max(schedule, axis=1).tolist(),
            "all_weeks_identical": bool(np.allclose(schedule, schedule[0:1])),
            "cha_profiles_identical": bool(dim <= 2 or np.allclose(schedule[:, 1:], schedule[:, 1:2])),
        }
    )
    return np.asarray(schedule, dtype=float), details


def apply_scenario_patch(bundle: Dict[str, Any], resolved_spec: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "project_root": bundle["project_root"],
        "baseline_dir": bundle["baseline_dir"],
        "config_template": copy.deepcopy(bundle["config_template"]),
        "arrays": {key: np.array(val, copy=True) for key, val in bundle["arrays"].items()},
    }

    arrays = out["arrays"]
    config = out["config_template"]
    eqn_cfg = config["eqn_config"]
    net_cfg = config["net_config"]

    dim = arrays["mu"].shape[1]
    baseline_mu = np.asarray(bundle["arrays"]["mu"], dtype=float)
    baseline_gamma_diag = _gamma_to_diag(bundle["arrays"]["gamma"], dim)
    baseline_eta = np.asarray(bundle["arrays"]["eta"], dtype=float).reshape(-1)
    theta = np.asarray(arrays["theta"], dtype=float).reshape(-1)
    exposure = np.asarray(arrays["exposure"], dtype=float).reshape(-1)
    p = np.asarray(arrays["p"], dtype=float).reshape(-1)
    if not np.allclose(p, 0.69, rtol=0.0, atol=1e-10):
        raise ValueError(
            "Fangonda paper workflow expects p=0.69 for every CHA. "
            f"Got {p.tolist()}; old p=0.89 artifacts/configs are legacy."
        )
    distance_miles = np.asarray(arrays["distance"], dtype=float).reshape(-1)

    family = resolved_spec["family"]
    N_h = float(resolved_spec["N_h"])
    if not np.isclose(N_h, exposure[0]):
        exposure[0] = N_h
        arrays["exposure"] = exposure
        arrays["mu"][:, 0] = theta * N_h
        gamma_diag = _gamma_to_diag(arrays["gamma"], dim)
        gamma_diag[:, 0] = N_h * theta * (1.0 - theta)
        arrays["gamma"] = gamma_diag

    cost_arrays = _compute_cost_arrays(distance_miles, p, resolved_spec)
    arrays["u"] = cost_arrays["u"]
    arrays["r"] = cost_arrays["r"]
    arrays["c"] = cost_arrays["c"]

    gamma_diag = _gamma_to_diag(arrays["gamma"], dim)
    var_scale = float(resolved_spec["cha_variance_scale"])
    gamma_diag[:, 1:] *= var_scale

    rho = float(resolved_spec["corr_rho"])
    if np.isclose(rho, 0.0):
        arrays["gamma"] = gamma_diag
        gamma_storage = "diagonal"
    else:
        arrays["gamma"] = _build_full_covariance(gamma_diag, rho)
        gamma_storage = "full_flattened"

    total_time = float(eqn_cfg["total_time"])
    raw_dt = total_time / float(arrays["mu"].shape[0])
    total_expected_demand = float(arrays["mu"].sum() * raw_dt)
    i0_multiplier = resolved_spec.get("I0_expected_demand_multiplier")
    if i0_multiplier is None:
        I0 = float(resolved_spec["I0"])
        i0_rule = "fixed"
    else:
        I0 = float(i0_multiplier) * total_expected_demand
        i0_rule = f"{float(i0_multiplier):g}_times_expected_demand"
    eqn_cfg["I0"] = I0

    arrays["eta"] = baseline_eta.copy()
    eta_rule = "baseline_fixed"
    eta_details = {
        "rho_bar": None,
        "rho_raw": None,
        "total_expected_demand": float(baseline_mu.sum() * raw_dt),
    }
    if family == "hf_demand":
        recomputed_eta, eta_details = compute_fangonda_eta(
            arrays["mu"],
            gamma_diag,
            p,
            I0,
            total_time,
        )
        arrays["eta"][0] = recomputed_eta[0]
        arrays["eta"][1:] = baseline_eta[1:]
        eta_rule = "recompute_hf_eta_only"
    elif family == "shipment_bounds":
        arrays["eta"] = baseline_eta * float(resolved_spec["eta_scale"])
        eta_rule = "scale_baseline_eta"

    lambda_param = float(resolved_spec["lambda_param"])
    alpha = float(resolved_spec["alpha"])
    eqn_cfg["lambda_param"] = lambda_param
    eqn_cfg["alpha"] = alpha
    eqn_cfg["calibration_version"] = resolved_spec.get("calibration_version")
    eqn_cfg["fairness_spec"] = FAIRNESS_SPEC
    eqn_cfg["fairness_includes_hf"] = FAIRNESS_INCLUDES_HF
    eqn_cfg["fairness_period_h_factor"] = FAIRNESS_PERIOD_H_FACTOR
    net_cfg["lambd_const"] = lambda_param
    net_cfg["num_time_interval"] = int(eqn_cfg["num_time_interval"])
    eqn_cfg["I0"] = I0
    eqn_cfg["periodic_demand_mode"] = scenario_registry.periodic_demand_mode_for_spec(resolved_spec)

    out["resolved_spec"] = copy.deepcopy(resolved_spec)
    out["metadata"] = {
        "gamma_storage": gamma_storage,
        "correlation_mode": "all_off_diagonal" if not np.isclose(rho, 0.0) else "diagonal_only",
        "demand_mode": eqn_cfg["periodic_demand_mode"],
        "calibration_version": resolved_spec.get("calibration_version"),
        "N_h": float(exposure[0]),
        "N_c": float(exposure[1]),
        "cost_units": str(resolved_spec["cost_units"]),
        "hourly_opportunity_cost": float(resolved_spec["hourly_opportunity_cost"]),
        "motorcycle_taxi_fare_per_mile": float(resolved_spec["motorcycle_taxi_fare_per_mile"]),
        "normalization_caregiver_hours": float(resolved_spec["normalization_caregiver_hours"]),
        "referral_caregiver_hours": float(resolved_spec["referral_caregiver_hours"]),
        "untreated_cost": float(arrays["u"][0]),
        "distance_miles": distance_miles.tolist(),
        "referral_costs": arrays["r"].tolist(),
        "p": arrays["p"].tolist(),
        "effective_shortage_costs": arrays["c"].tolist(),
        "I0": I0,
        "I0_rule": i0_rule,
        "total_expected_demand": total_expected_demand,
        "hf_expected_demand": float(arrays["mu"][:, 0].sum() * raw_dt),
        "cha_expected_demand": float(arrays["mu"][:, 1:].sum() * raw_dt),
        "eta_rule": eta_rule,
        "eta_scale": float(resolved_spec["eta_scale"]),
        "eta_rho_bar": eta_details["rho_bar"],
        "eta_rho_raw": eta_details["rho_raw"],
        "lambda_param": lambda_param,
        "alpha": alpha,
        "fairness_spec": FAIRNESS_SPEC,
        "fairness_includes_hf": FAIRNESS_INCLUDES_HF,
        "fairness_period_h_factor": FAIRNESS_PERIOD_H_FACTOR,
    }
    return out


def apply_training_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.num_time_interval is not None:
        cfg["eqn_config"]["num_time_interval"] = int(args.num_time_interval)
        cfg["net_config"]["num_time_interval"] = int(args.num_time_interval)
    if args.num_layers is not None:
        cfg["net_config"]["num_layers"] = int(args.num_layers)
    if getattr(args, "reference_policy", None) is not None:
        cfg["eqn_config"]["policy"] = str(args.reference_policy)
    if getattr(args, "reference_fluid_plan_path", None) is not None:
        cfg["eqn_config"]["reference_fluid_plan_path"] = str(args.reference_fluid_plan_path)
    if getattr(args, "reference_mixture", None) is not None:
        cfg["eqn_config"]["reference_mixture"] = str(args.reference_mixture)
    if getattr(args, "reference_mixture_counts", None) is not None:
        cfg["eqn_config"]["reference_mixture_counts"] = str(args.reference_mixture_counts)
    if getattr(args, "eta_mode", None) is not None:
        cfg["eqn_config"]["eta_mode"] = str(args.eta_mode)
        cfg["eqn_config"]["eta_profile"] = str(args.eta_mode)
    if getattr(args, "eta_profile", None) is not None:
        cfg["eqn_config"]["eta_mode"] = str(args.eta_profile)
        cfg["eqn_config"]["eta_profile"] = str(args.eta_profile)
    cfg["eqn_config"].setdefault("eta_mode", cfg["eqn_config"].get("eta_profile", "constant_peak"))
    cfg["eqn_config"].setdefault("eta_profile", cfg["eqn_config"]["eta_mode"])
    cfg["eqn_config"].setdefault("chi_eta", 1.75)
    cfg["eqn_config"].setdefault("s_eta", 0.0)
    cfg["eqn_config"].setdefault("eta_schedule_file_role", "active_input")
    if getattr(args, "chi_eta", None) is not None:
        cfg["eqn_config"]["chi_eta"] = float(args.chi_eta)
    if getattr(args, "s_eta", None) is not None:
        cfg["eqn_config"]["s_eta"] = float(args.s_eta)
    if getattr(args, "lambda_param", None) is not None:
        cfg["eqn_config"]["lambda_param"] = float(args.lambda_param)
        cfg["net_config"]["lambd_const"] = float(args.lambda_param)
    if getattr(args, "alpha", None) is not None:
        cfg["eqn_config"]["alpha"] = float(args.alpha)
    cfg["eqn_config"]["fairness_spec"] = FAIRNESS_SPEC
    cfg["eqn_config"]["fairness_includes_hf"] = FAIRNESS_INCLUDES_HF
    cfg["eqn_config"]["fairness_period_h_factor"] = FAIRNESS_PERIOD_H_FACTOR
    cfg["eqn_config"].setdefault("eta_schedule_file", "eta_schedule.csv")
    if args.final_iterations is not None:
        cfg["net_config"]["final_iterations"] = int(args.final_iterations)
    if args.inner_iterations is not None:
        cfg["net_config"]["inner_iterations"] = int(args.inner_iterations)
    if getattr(args, "train_final_interval_only", False):
        cfg["net_config"]["train_final_interval_only"] = True
    if args.batch_size is not None:
        cfg["net_config"]["batch_size"] = int(args.batch_size)
    if args.valid_size is not None:
        cfg["net_config"]["valid_size"] = int(args.valid_size)
    for key, cast in NET_CONFIG_OVERRIDES:
        value = getattr(args, key, None)
        if value is not None:
            cfg["net_config"][key] = cast(value)
    eval_config: Dict[str, Any] = dict(cfg.get("eval_config", {}))
    eval_seed = getattr(args, "eval_seed", None)
    if eval_seed is not None:
        eval_config["eval_seed"] = int(eval_seed)
        eval_config["fine_grid_seed"] = int(eval_seed)
        eval_config["periodic_seed"] = int(eval_seed)
    eval_num_paths = getattr(args, "eval_num_paths", None)
    if eval_num_paths is not None:
        eval_config["eval_num_paths"] = int(eval_num_paths)
    eval_ext_num_paths = getattr(args, "eval_ext_num_paths", None)
    if eval_ext_num_paths is not None:
        eval_config["eval_ext_num_paths"] = int(eval_ext_num_paths)
    if eval_config:
        cfg["eval_config"] = eval_config
    return cfg


def build_manifest(run_name: str, family: str, resolved_spec: Dict[str, Any], patched_bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scenario_name": run_name,
        "family": family,
        "baseline_reference": scenario_registry.BASELINE_NAME,
        "resolved_spec": resolved_spec,
        "transform_notes": {
            "cost_rule": (
                "All costs are normalized by one untreated under-five malaria episode: "
                "u.csv stores 1, r.csv stores normalized referral costs, c.csv stores "
                "effective normalized shortage costs, and distance.csv stores miles to the HF."
            ),
            "gamma_rule": (
                "For variance scenarios, scale CHA diagonal variances only. For correlation scenarios, "
                "write full covariance rows with non-diagonal entries rho*sqrt(var_i*var_j)."
            ),
            "eta_rule": (
                "Eta is held at the baseline unless the paper recomputation table says otherwise: "
                "HF exposure recomputes eta0 only; shipment-bound scenarios scale baseline eta."
            ),
            "fairness_rule": "eqn_config.lambda_param and net_config.lambd_const kept synchronized",
            "inventory_rule": (
                "If I0_expected_demand_multiplier is set, I0 is that multiplier times expected total demand; otherwise fixed I0 is used"
            ),
            "hf_demand_rule": "HF exposure scenarios recompute mu0 and Gamma00 from theta and N_h.",
        },
        "metadata": patched_bundle.get("metadata", {}),
        "git_commit": git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_name": socket.gethostname(),
    }


def materialize_run(run_name: str, resolved_spec: Dict[str, Any], args: argparse.Namespace) -> Dict[str, str]:
    output_root = Path(args.output_root).expanduser()
    run_dir = output_root / run_name
    params_dir = run_dir / "params"
    run_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    baseline_bundle = load_baseline_bundle()
    patched_bundle = apply_scenario_patch(baseline_bundle, resolved_spec)
    rho_arguments = (
        getattr(args, "rho_calibration_report", None),
        getattr(args, "rho_calibration_report_sha256", None),
        getattr(args, "rho_validation_report", None),
        getattr(args, "rho_validation_report_sha256", None),
    )
    if any(value is not None for value in rho_arguments):
        if not all(value is not None for value in rho_arguments):
            raise ValueError("All rho calibration and validation paths/hashes must be supplied together.")
        apply_realized_rho_calibration(
            patched_bundle,
            resolved_spec,
            calibration_report_path=Path(args.rho_calibration_report),
            calibration_report_sha256=str(args.rho_calibration_report_sha256),
            validation_report_path=Path(args.rho_validation_report),
            validation_report_sha256=str(args.rho_validation_report_sha256),
        )
    elif resolved_spec.get("rho_interpretation") == "realized_discrete_count_pearson":
        raise ValueError("Calibrated-rho scenarios require frozen calibration and 200,000-path validation artifacts.")
    config = apply_training_overrides(patched_bundle["config_template"], args)
    manifest_spec = copy.deepcopy(resolved_spec)
    if getattr(args, "lambda_param", None) is not None:
        manifest_spec["lambda_param"] = float(args.lambda_param)
    if getattr(args, "alpha", None) is not None:
        manifest_spec["alpha"] = float(args.alpha)
    config["eqn_config"]["config_dir"] = str(params_dir.resolve())
    config["net_config"]["num_time_interval"] = int(config["eqn_config"]["num_time_interval"])
    eta_arrays = patched_bundle["arrays"]
    eta_config = config
    if (
        manifest_spec.get("hold_eta_schedule_from_baseline")
        and str(manifest_spec.get("family")) == "inventory"
    ):
        eta_arrays = {key: np.array(value, copy=True) for key, value in baseline_bundle["arrays"].items()}
        eta_config = copy.deepcopy(config)
        eta_config["eqn_config"]["I0"] = float(scenario_registry.BASELINE_SPEC["I0"])
    eta_schedule, eta_metadata = compute_eta_schedule_for_config(eta_arrays, eta_config)
    if (
        manifest_spec.get("hold_eta_schedule_from_baseline")
        and str(manifest_spec.get("family")) == "inventory"
    ):
        eta_metadata["eta_schedule_rule"] = (
            str(eta_metadata.get("eta_schedule_rule", ""))
            + "_held_fixed_from_baseline_for_inventory"
        )
        eta_metadata["eta_held_fixed_from_baseline"] = True
        eta_metadata["eta_held_fixed_baseline_I0"] = float(scenario_registry.BASELINE_SPEC["I0"])
    else:
        eta_metadata["eta_held_fixed_from_baseline"] = False
    patched_bundle["arrays"]["eta_schedule"] = eta_schedule
    patched_bundle["metadata"].update(eta_metadata)
    patched_bundle["metadata"].update(
        {
            "lambda_param": float(config["eqn_config"].get("lambda_param", 0.0)),
            "alpha": float(config["eqn_config"].get("alpha", 0.5)),
            "eta_profile": str(config["eqn_config"].get("eta_profile", config["eqn_config"].get("eta_mode", "constant_peak"))),
            "chi_eta": float(config["eqn_config"].get("chi_eta", 1.75)),
            "s_eta": float(config["eqn_config"].get("s_eta", 0.0)),
            "reference_policy": str(config["eqn_config"].get("policy", "")),
            "reference_fluid_plan_path": config["eqn_config"].get("reference_fluid_plan_path"),
            "reference_mixture": config["eqn_config"].get("reference_mixture"),
            "reference_mixture_counts": config["eqn_config"].get("reference_mixture_counts"),
        }
    )

    for key in ("theta", "exposure", "mu", "gamma", "p", "distance", "u", "r", "c", "h", "eta", "eta_schedule"):
        _write_csv(params_dir / f"{key}.csv", patched_bundle["arrays"][key])

    config_path = run_dir / "config_scenario.json"
    manifest_path = run_dir / "scenario_manifest.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            build_manifest(run_name, manifest_spec["family"], manifest_spec, patched_bundle),
            f,
            indent=2,
        )

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "params_dir": str(params_dir),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize Fangonda weekly comparative-statics inputs under OUTPUT_DIR/<RUN_NAME>/."
    )
    parser.add_argument("--scenario", required=True, help="Scenario name from configs.fangonda_weekly.scenarios")
    parser.add_argument("--rho_calibration_report", default=None)
    parser.add_argument("--rho_calibration_report_sha256", default=None)
    parser.add_argument("--rho_validation_report", default=None)
    parser.add_argument("--rho_validation_report_sha256", default=None)
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional override for the output run directory name. Defaults to the scenario name.",
    )
    parser.add_argument(
        "--output_root",
        default=str(default_output_root()),
        help="Base output directory for run-local configs and parameter snapshots.",
    )
    parser.add_argument("--num_time_interval", type=int, default=None, help="Optional num_time_interval override.")
    parser.add_argument("--num_layers", type=int, default=None, help="Optional num_layers override.")
    parser.add_argument("--num_neurons", type=int, default=None, help="Optional num_neurons override.")
    parser.add_argument("--activation", choices=ACTIVATION_CHOICES, default=None, help="Optional activation override.")
    parser.add_argument(
        "--init_method",
        choices=INIT_METHOD_CHOICES,
        default=None,
        help="Optional network weight initialization override.",
    )
    parser.add_argument("--h_init_method", choices=INIT_METHOD_CHOICES, default=None)
    parser.add_argument("--g_init_method", choices=G_INIT_METHOD_CHOICES, default=None)
    parser.add_argument("--g_output_bias", type=parse_bool, default=None)
    parser.add_argument("--g_prior_fluid_plan_path", default=None)
    parser.add_argument("--g_prior_epsilon_a", type=float, default=None)
    parser.add_argument("--g_prior_lower_bound", type=float, default=None)
    parser.add_argument("--g_prior_upper_bound", type=float, default=None)
    parser.add_argument("--g_prior_cap_offset", type=float, default=None)
    parser.add_argument("--reference_policy", choices=REFERENCE_POLICY_CHOICES, default=None)
    parser.add_argument(
        "--reference_fluid_plan_path",
        default=None,
        help="Optional allocation_plan.npz used only for reference_policy=fluid_optimal_rollout training rollouts.",
    )
    parser.add_argument("--reference_mixture", choices=("R0", "R1", "R2", "R3", "R4"), default=None)
    parser.add_argument("--reference_mixture_counts", default=None)
    parser.add_argument("--eta_mode", choices=ETA_MODES, default=None)
    parser.add_argument("--eta_profile", choices=ETA_MODES, default=None)
    parser.add_argument("--chi_eta", type=float, default=None)
    parser.add_argument("--s_eta", type=float, default=None)
    parser.add_argument("--lambda_param", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--proposal_sampler", choices=PROPOSAL_SAMPLER_CHOICES, default=None)
    parser.add_argument("--final_learning_rate", type=float, default=None, help="Optional final_learning_rate override.")
    parser.add_argument("--inner_learning_rate", type=float, default=None, help="Optional inner_learning_rate override.")
    parser.add_argument("--final_milestones", type=str, default=None, help="Comma-separated final_milestones override.")
    parser.add_argument("--inner_milestones", type=str, default=None, help="Comma-separated inner_milestones override.")
    parser.add_argument("--gamma", type=float, default=None, help="Optional scheduler gamma override.")
    parser.add_argument("--inner_gamma", type=float, default=None, help="Optional inner-interval scheduler gamma override.")
    parser.add_argument("--final_gamma", type=float, default=None, help="Optional terminal-interval scheduler gamma override.")
    parser.add_argument("--grad_norm", type=float, default=None, help="Optional gradient clipping norm override.")
    parser.add_argument("--h_grad_norm", type=float, default=None, help="Optional H clip norm for separate_hg.")
    parser.add_argument("--g_grad_norm", type=float, default=None, help="Optional G clip norm for separate_hg.")
    parser.add_argument("--grad_clip_rule", choices=("joint", "separate_hg"), default=None)
    parser.add_argument("--final_grad_clip_rule", choices=("joint", "separate_hg"), default=None)
    parser.add_argument("--inner_grad_clip_rule", choices=("joint", "separate_hg"), default=None)
    parser.add_argument("--gradient_assembly_mode", choices=("standard", "hg_component"), default=None)
    parser.add_argument("--h_base_grad_multiplier", type=float, default=None)
    parser.add_argument("--g_base_grad_multiplier", type=float, default=None)
    parser.add_argument("--h_consistency_coef", type=float, default=None)
    parser.add_argument("--g_consistency_coef", type=float, default=None)
    parser.add_argument("--input_mode", choices=("raw", "q0"), default=None)
    parser.add_argument("--q0_input_scale", type=float, default=None)
    parser.add_argument("--h_output_scale", type=float, default=None)
    parser.add_argument("--checkpoint_reparam_load_mode", choices=("guard", "convert"), default=None)
    parser.add_argument("--state_support_logging", type=parse_bool, default=None)
    parser.add_argument("--state_support_sample_size", type=int, default=None)
    parser.add_argument("--final_iterations", type=int, default=None, help="Optional final_iterations override.")
    parser.add_argument("--inner_iterations", type=int, default=None, help="Optional inner_iterations override.")
    parser.add_argument("--final_min_iterations", type=int, default=None, help="Optional final_min_iterations override.")
    parser.add_argument("--inner_min_iterations", type=int, default=None, help="Optional inner_min_iterations override.")
    parser.add_argument(
        "--train_final_interval_only",
        action="store_true",
        help="Set net_config.train_final_interval_only for terminal-interval-only training.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch_size override.")
    parser.add_argument("--valid_size", type=int, default=None, help="Optional valid_size override.")
    parser.add_argument("--validation_every", type=int, default=None, help="Optional validation cadence override.")
    parser.add_argument("--loss_mode", choices=LOSS_MODE_CHOICES, default=None)
    parser.add_argument("--component_seed_isolation", type=parse_bool, default=None)
    parser.add_argument("--h_network_seed", type=int, default=None)
    parser.add_argument("--g_network_seed", type=int, default=None)
    parser.add_argument("--qgrad_network_seed", type=int, default=None)
    parser.add_argument("--save_initial_checkpoints", type=parse_bool, default=None)
    parser.add_argument("--sampling_mode", choices=SAMPLING_MODE_CHOICES, default=None)
    parser.add_argument("--rollout_pool_size", type=int, default=None)
    parser.add_argument("--sampling_seed", type=int, default=0)
    parser.add_argument("--uniform_z_radius_mult", type=float, default=None)
    parser.add_argument("--uniform_q_radius_frac", type=float, default=None)
    parser.add_argument("--total_inventory_buffer", type=float, default=None)
    parser.add_argument("--mix_rollout", type=float, default=None)
    parser.add_argument("--mix_box", type=float, default=None)
    parser.add_argument("--mix_shortage", type=float, default=None)
    parser.add_argument("--shortage_q_frac", type=float, default=None)
    parser.add_argument("--shortage_z_frac", type=float, default=None)
    parser.add_argument("--lognormal_delta_z", type=float, default=None)
    parser.add_argument("--lognormal_delta_q", type=float, default=None)
    parser.add_argument("--lognormal_r_min", type=int, default=None)
    parser.add_argument("--initial_state_mode", choices=("standard", "q_sensitivity_mixture", "paper_70_30"), default=None)
    parser.add_argument("--initial_perturb_prob", type=float, default=None)
    parser.add_argument("--initial_q_perturb_frac", type=float, default=None)
    parser.add_argument("--initial_z_perturb_frac", type=float, default=None)
    parser.add_argument("--initial_perturb_minus_prob", type=float, default=None)
    parser.add_argument("--initial_perturb_plus_prob", type=float, default=None)
    parser.add_argument("--initial_perturb_minus_frac", type=float, default=None)
    parser.add_argument("--initial_perturb_plus_frac", type=float, default=None)
    parser.add_argument("--initial_dirichlet_alpha", type=float, default=None)
    parser.add_argument("--consistency_coef", type=float, default=None)
    parser.add_argument(
        "--prev_init_same_params_only",
        type=parse_bool,
        default=None,
        help="Optional prev_init_same_params_only override. Accepts 0/1 or true/false.",
    )
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--eval_num_paths", type=int, default=None)
    parser.add_argument("--eval_ext_num_paths", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    resolved_spec = scenario_registry.resolve_scenario_spec(args.scenario)
    run_name = args.run_name or resolved_spec["name"]
    result = materialize_run(run_name, resolved_spec, args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
