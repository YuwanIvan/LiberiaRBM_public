#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

K = 7
BASELINE_I0 = 142.0
BASELINE_N_HF = 219
BASELINE_N_CHA = 28
BASELINE_P = 0.69
BASELINE_W = 0.68
BASELINE_KAPPA = 0.15
UNTREATED_COST = 1.0
NORMALIZATION_CAREGIVER_HOURS = 8.0 * 8.0
REFERRAL_HOURS = 8.0
ACTIVE_BEGIN_WEEK = 5
ACTIVE_END_WEEK = 26
EXPECTED_NORMALIZED_BASELINE_C = np.array(
    [1.0000, 0.4486, 0.4143, 0.4105, 0.3974, 0.4200, 0.4200, 0.3986],
    dtype=float,
)


def compute_fangonda_eta(
    mu: np.ndarray,
    gamma: np.ndarray,
    p: np.ndarray,
    I0: float,
    total_time: float = 22.0,
) -> tuple[np.ndarray, Dict[str, float]]:
    mu_arr = np.asarray(mu, dtype=float)
    gamma_diag = np.asarray(gamma, dtype=float)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    step = float(total_time) / float(mu_arr.shape[0])
    total_expected_demand = float(mu_arr.sum() * step)
    rho_raw = 1.0 - float(I0) / total_expected_demand
    rho_bar = max(0.0, rho_raw)
    eta = np.zeros(mu_arr.shape[1], dtype=float)
    eta[1:] = np.max(mu_arr[:, 1:] + 2.0 * np.sqrt(np.clip(gamma_diag[:, 1:], 0.0, None)), axis=0)
    referral_mean = rho_bar * np.sum(mu_arr[:, 1:] * p_arr.reshape(1, -1), axis=1)
    referral_var = (rho_bar**2) * np.sum(gamma_diag[:, 1:] * (p_arr.reshape(1, -1) ** 2), axis=1)
    eta[0] = np.max(
        mu_arr[:, 0]
        + referral_mean
        + 2.0 * np.sqrt(np.clip(gamma_diag[:, 0] + referral_var, 0.0, None))
    )
    return eta, {
        "rho_bar": float(rho_bar),
        "rho_raw": float(rho_raw),
        "total_expected_demand": float(total_expected_demand),
        "slack_multiplier": 1.0,
    }


def default_config_dir() -> Path:
    return PROJECT_ROOT / "configs" / "fangonda_weekly"


def default_data_dir() -> Path:
    return PROJECT_ROOT / "data"


def _write_csv(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    np.savetxt(path, arr, delimiter=",", fmt="%.12g")


def load_weekly_theta(data_dir: Path, n_cha: int = BASELINE_N_CHA) -> np.ndarray:
    cha_path = data_dir / "CHA_weekly.xlsx"
    cha_data = pd.read_excel(cha_path, sheet_name="CHA-weekly")
    cha_data = cha_data.sort_values(by=["CHA ID", "unique_id", "household visits"])
    cha_by_week = (
        cha_data[["CHA ID", "Malaria (RTD+)", "unique_id"]]
        .pivot(index="unique_id", columns="CHA ID", values="Malaria (RTD+)")
        .reset_index(drop=True)
    )
    weekly_cases = cha_by_week.sum(axis=1)
    theta = weekly_cases / float(K * n_cha)
    theta_use = theta.iloc[ACTIVE_BEGIN_WEEK - 1 : ACTIVE_END_WEEK].reset_index(drop=True)
    if len(theta_use) != 22:
        raise ValueError(f"Expected 22 active weeks, got {len(theta_use)}.")
    return theta_use.to_numpy(dtype=float)


def load_distances(data_dir: Path) -> np.ndarray:
    distance_path = data_dir / "distance.xlsx"
    distance = pd.read_excel(distance_path)[["distance"]].to_numpy(dtype=float).reshape(-1)
    if distance.shape[0] != K:
        raise ValueError(f"Expected {K} CHA distances, got {distance.shape[0]}.")
    return np.concatenate([[0.0], distance])


def compute_normalized_cost_arrays(
    distance: np.ndarray,
    p: np.ndarray,
    *,
    w: float = BASELINE_W,
    kappa: float = BASELINE_KAPPA,
    referral_hours: float = REFERRAL_HOURS,
    normalization_caregiver_hours: float = NORMALIZATION_CAREGIVER_HOURS,
) -> Dict[str, Any]:
    """Compute paper-normalized untreated, referral, and effective shortage costs."""
    distance_arr = np.asarray(distance, dtype=float).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    if distance_arr.shape[0] != p_arr.shape[0] + 1:
        raise ValueError(
            f"distance has {distance_arr.shape[0]} entries but p has {p_arr.shape[0]}; "
            "expected one HF entry plus one entry per CHA."
        )
    denominator = float(normalization_caregiver_hours) * float(w)
    if denominator <= 0.0:
        raise ValueError("Cost normalization denominator must be positive.")

    u = np.array([UNTREATED_COST], dtype=float)
    r = np.zeros_like(distance_arr, dtype=float)
    r[1:] = (float(referral_hours) * float(w) + 2.0 * float(kappa) * distance_arr[1:]) / denominator
    c = np.zeros_like(distance_arr, dtype=float)
    c[0] = UNTREATED_COST
    c[1:] = (1.0 - p_arr) * UNTREATED_COST + p_arr * r[1:]
    return {
        "u": u,
        "r": r,
        "c": c,
        "normalization_denominator": denominator,
    }


def validate_normalized_baseline_cost_vector(c: np.ndarray, atol: float = 1e-4) -> np.ndarray:
    """Assert the baseline normalized shortage-cost vector matches the paper TeX."""
    c_arr = np.asarray(c, dtype=float).reshape(-1)
    if not np.allclose(c_arr, EXPECTED_NORMALIZED_BASELINE_C, rtol=0.0, atol=atol):
        raise AssertionError(
            "Normalized Fangonda baseline c-vector mismatch. "
            f"Expected approximately {EXPECTED_NORMALIZED_BASELINE_C.tolist()}, got {c_arr.tolist()}."
        )
    return c_arr


def compute_calibration(
    *,
    data_dir: Path | None = None,
    n_hf: int = BASELINE_N_HF,
    n_cha: int = BASELINE_N_CHA,
    w: float = BASELINE_W,
    kappa: float = BASELINE_KAPPA,
    I0: float = BASELINE_I0,
) -> Dict[str, Any]:
    data_dir = data_dir or default_data_dir()
    theta = load_weekly_theta(data_dir, n_cha=n_cha)
    exposure = np.array([n_hf] + [n_cha] * K, dtype=float)
    mu = np.outer(theta, exposure)
    gamma = np.outer(theta * (1.0 - theta), exposure)
    p = np.full(K, BASELINE_P, dtype=float)
    distance = load_distances(data_dir)
    cost_arrays = compute_normalized_cost_arrays(distance, p, w=w, kappa=kappa)
    is_canonical_baseline = (
        int(n_hf) == BASELINE_N_HF
        and int(n_cha) == BASELINE_N_CHA
        and np.isclose(float(w), BASELINE_W)
        and np.isclose(float(kappa), BASELINE_KAPPA)
    )
    if is_canonical_baseline:
        validate_normalized_baseline_cost_vector(cost_arrays["c"])
    h = np.zeros(K + 1, dtype=float)
    eta, eta_details = compute_fangonda_eta(mu, gamma, p, I0, total_time=22.0)
    return {
        "theta": theta,
        "exposure": exposure,
        "mu": mu,
        "gamma": gamma,
        "p": p,
        "distance": distance,
        "u": cost_arrays["u"],
        "r": cost_arrays["r"],
        "c": cost_arrays["c"],
        "h": h,
        "eta": eta,
        "metadata": {
            "source": "CHA_weekly.xlsx weeks 5-26",
            "active_begin_week": ACTIVE_BEGIN_WEEK,
            "active_end_week": ACTIVE_END_WEEK,
            "N_h": int(n_hf),
            "N_c": int(n_cha),
            "K": K,
            "I0": float(I0),
            "p": BASELINE_P,
            "cost_units": "one untreated under-five malaria episode",
            "w": float(w),
            "kappa": float(kappa),
            "untreated_cost": UNTREATED_COST,
            "normalization_caregiver_hours": NORMALIZATION_CAREGIVER_HOURS,
            "normalization_denominator_formula": "64*w",
            "referral_hours": REFERRAL_HOURS,
            "distance_miles": distance.tolist(),
            "referral_costs": cost_arrays["r"].tolist(),
            "effective_shortage_costs": cost_arrays["c"].tolist(),
            "expected_normalized_baseline_c": EXPECTED_NORMALIZED_BASELINE_C.tolist(),
            "normalized_baseline_c_validation_atol": 1e-4,
            "total_expected_demand": float(mu.sum()),
            "hf_expected_demand": float(mu[:, 0].sum()),
            "cha_expected_demand": float(mu[:, 1:].sum()),
            "supply_coverage": float(I0 / mu.sum()),
            "eta_details": eta_details,
        },
    }


def write_theta_figure(theta: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.stairs(theta, np.arange(len(theta) + 1), color="black", linewidth=1.5, baseline=None)
    plt.ylabel(r"$\theta(n)$", fontsize=14)
    plt.xlabel(r"week $n$", fontsize=14)
    plt.xticks(np.arange(0, len(theta) + 2, 2), fontsize=12)
    plt.yticks(np.arange(0, 0.1, 0.02), fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def write_calibration(config_dir: Path, data_dir: Path, figure_path: Path | None = None) -> Dict[str, Any]:
    calibration = compute_calibration(data_dir=data_dir)
    for key in ("theta", "exposure", "mu", "gamma", "p", "distance", "u", "r", "c", "h", "eta"):
        _write_csv(config_dir / f"{key}.csv", calibration[key])
    with open(config_dir / "calibration_manifest.json", "w", encoding="utf-8") as f:
        json.dump(calibration["metadata"], f, indent=2)
    if figure_path is not None:
        write_theta_figure(calibration["theta"], figure_path)
    return calibration["metadata"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical Fangonda weekly calibration inputs.")
    parser.add_argument("--config-dir", default=str(default_config_dir()))
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument(
        "--theta-figure",
        default=str(default_data_dir() / "calibrated_theta.pdf"),
        help="Path for the calibrated theta figure. Use an empty string to skip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_path = Path(args.theta_figure).expanduser() if args.theta_figure else None
    metadata = write_calibration(
        Path(args.config_dir).expanduser(),
        Path(args.data_dir).expanduser(),
        figure_path,
    )
    print(json.dumps(metadata, indent=2))
    print("normalized_baseline_c =", np.round(metadata["effective_shortage_costs"], 4).tolist())


if __name__ == "__main__":
    main()
