"""Guards against mixing old fairness/calibration artifacts with DWAF v1 runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

EXPECTED_FAIRNESS_SPEC = "demand_weighted_alpha_fair_v1"
EXPECTED_FAIRNESS_INCLUDES_HF = True
EXPECTED_FAIRNESS_PERIOD_H_FACTOR = False
EXPECTED_FANGONDA_P = 0.69
EXPECTED_FANGONDA_COST_VECTOR = np.asarray(
    [1.0000, 0.448571, 0.414324, 0.410519, 0.397439, 0.420032, 0.420032, 0.398628],
    dtype=np.float64,
)
EXPECTED_CALIBRATION_VERSION = "fangonda_paper_2026_07_normalized_costs_dwaf_v1"
_DEFAULT_EXPECTED = object()


def _read_vector(config: Mapping[str, Any], run_dir: str | Path | None, key: str):
    eqn_config = config.get("eqn_config", {})
    filename = eqn_config.get(key)
    if not filename:
        return None
    candidates = []
    if run_dir is not None:
        candidates.append(Path(run_dir) / "params" / filename)
    config_dir = eqn_config.get("config_dir")
    if config_dir:
        candidates.append(Path(config_dir) / filename)
    candidates.append(Path(filename))
    for path in candidates:
        if path.is_file():
            return np.loadtxt(path, delimiter=",").reshape(-1)
    return None


def _assert_close(name: str, actual, expected, atol=1e-8):
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    if actual_arr.shape != expected_arr.shape or not np.allclose(actual_arr, expected_arr, rtol=0.0, atol=atol):
        raise ValueError(f"{name} mismatch: got {actual_arr.tolist()}, expected {expected_arr.tolist()}.")


def _npz_scalar(data, key: str, path: str | Path):
    if key not in data:
        raise ValueError(f"Fluid artifact {path} is missing required metadata field {key!r}.")
    return np.asarray(data[key]).reshape(-1)[0]


def _manifest_metadata(scenario_manifest: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if scenario_manifest is None:
        return {}
    if isinstance(scenario_manifest, Mapping):
        payload = dict(scenario_manifest)
    else:
        path = Path(scenario_manifest).expanduser()
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    metadata = payload.get("metadata", payload)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _expected_from_manifest_or_default(
    provided,
    metadata: Mapping[str, Any],
    manifest_key: str,
    fallback,
):
    if provided is not _DEFAULT_EXPECTED:
        return provided
    if manifest_key in metadata:
        return metadata[manifest_key]
    return fallback


def _validate_fairness_parameters(alpha, lambda_param=None) -> float:
    alpha_value = float(alpha)
    if not (0.0 < alpha_value < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha_value}.")
    if lambda_param is not None:
        lambda_value = float(lambda_param)
        if lambda_value < 0.0:
            raise ValueError(f"lambda_param must be nonnegative, got {lambda_value}.")
    return alpha_value


def validate_r2_reference_fluid_artifact(
    path: str | Path,
    *,
    expected_lambda_ref: float = 1e-4,
    expected_alpha: float = 0.5,
    expected_chi_eta: float = 1.75,
    expected_s_eta: float = 0.0,
    expected_review_intervals: int = 22,
    expected_p: float | Sequence[float] | object = _DEFAULT_EXPECTED,
    expected_cost_vector: Sequence[float] | object = _DEFAULT_EXPECTED,
    expected_calibration_version: str = EXPECTED_CALIBRATION_VERSION,
    scenario_manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Validate the lexicographic fluid QCP artifact used as R2 reference."""
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"R2 reference fluid artifact not found: {artifact_path}")
    manifest_meta = _manifest_metadata(scenario_manifest)
    expected_p = _expected_from_manifest_or_default(expected_p, manifest_meta, "p", EXPECTED_FANGONDA_P)
    expected_cost_vector = _expected_from_manifest_or_default(
        expected_cost_vector,
        manifest_meta,
        "effective_shortage_costs",
        EXPECTED_FANGONDA_COST_VECTOR,
    )
    with np.load(artifact_path, allow_pickle=False) as data:
        if "allocation_plan" not in data:
            raise ValueError(f"Fluid artifact {artifact_path} does not contain allocation_plan.")
        plan = np.asarray(data["allocation_plan"], dtype=np.float64)
        if plan.ndim != 2 or plan.shape[0] != int(expected_review_intervals):
            raise ValueError(
                f"R2 allocation_plan shape={plan.shape}; expected ({expected_review_intervals}, dim)."
            )
        _assert_close("lambda_param", _npz_scalar(data, "lambda_param", artifact_path), expected_lambda_ref, atol=1e-12)
        _assert_close("alpha", _npz_scalar(data, "alpha", artifact_path), expected_alpha, atol=1e-12)
        _assert_close("chi_eta", _npz_scalar(data, "chi_eta", artifact_path), expected_chi_eta, atol=1e-12)
        _assert_close("s_eta", _npz_scalar(data, "s_eta", artifact_path), expected_s_eta, atol=1e-12)
        _assert_close("review_dt", _npz_scalar(data, "review_dt", artifact_path), 1.0, atol=1e-12)
        p_actual = np.asarray(data["p"], dtype=np.float64)
        p_expected = (
            np.full_like(p_actual, float(expected_p), dtype=np.float64)
            if np.asarray(expected_p).ndim == 0
            else np.asarray(expected_p, dtype=np.float64)
        )
        _assert_close("p", p_actual, p_expected, atol=1e-10)
        _assert_close("cost vector", np.asarray(data["c"], dtype=np.float64), expected_cost_vector, atol=5e-4)
        fairness_spec = str(_npz_scalar(data, "fairness_spec", artifact_path))
        if fairness_spec != EXPECTED_FAIRNESS_SPEC:
            raise ValueError(
                f"Fluid artifact {artifact_path} fairness_spec={fairness_spec!r}; "
                f"expected {EXPECTED_FAIRNESS_SPEC!r}."
            )
        if bool(_npz_scalar(data, "fairness_includes_hf", artifact_path)) is not True:
            raise ValueError("R2 fluid artifact fairness_includes_hf must be true.")
        if bool(_npz_scalar(data, "fairness_period_h_factor", artifact_path)) is not False:
            raise ValueError("R2 fluid artifact fairness_period_h_factor must be false.")
        calibration_version = str(_npz_scalar(data, "calibration_version", artifact_path))
        if calibration_version != str(expected_calibration_version):
            raise ValueError(
                f"calibration_version={calibration_version!r}; expected {expected_calibration_version!r}."
            )
        eta_mode = str(_npz_scalar(data, "eta_mode", artifact_path)) if "eta_mode" in data else ""
        if eta_mode != "weekly_time_varying":
            raise ValueError(f"eta_mode={eta_mode!r}; expected 'weekly_time_varying'.")
        return {
            "path": str(artifact_path),
            "allocation_shape": list(plan.shape),
            "lambda_ref": float(expected_lambda_ref),
            "alpha": float(expected_alpha),
            "chi_eta": float(expected_chi_eta),
            "s_eta": float(expected_s_eta),
            "review_intervals": int(expected_review_intervals),
            "calibration_version": calibration_version,
            "fairness_spec": fairness_spec,
        }


def validate_fangonda_fluid_artifact(
    path: str | Path,
    *,
    expected_lambda: float,
    expected_review_intervals: int,
    expected_total_time: float = 22.0,
    expected_dim: int = 8,
    expected_alpha: float = 0.5,
    expected_chi_eta: float = 1.75,
    expected_s_eta: float = 0.0,
    expected_p: float | Sequence[float] | object = _DEFAULT_EXPECTED,
    expected_cost_vector: Sequence[float] | object = _DEFAULT_EXPECTED,
    expected_calibration_version: str = EXPECTED_CALIBRATION_VERSION,
    scenario_manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Validate a Fangonda deterministic fluid artifact for training/evaluation reuse."""
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Fangonda fluid artifact not found: {artifact_path}")
    expected_review_dt = float(expected_total_time) / float(expected_review_intervals)
    manifest_meta = _manifest_metadata(scenario_manifest)
    expected_p = _expected_from_manifest_or_default(expected_p, manifest_meta, "p", EXPECTED_FANGONDA_P)
    expected_cost_vector = _expected_from_manifest_or_default(
        expected_cost_vector,
        manifest_meta,
        "effective_shortage_costs",
        EXPECTED_FANGONDA_COST_VECTOR,
    )
    with np.load(artifact_path, allow_pickle=False) as data:
        if "allocation_plan" not in data:
            raise ValueError(f"Fluid artifact {artifact_path} does not contain allocation_plan.")
        plan = np.asarray(data["allocation_plan"], dtype=np.float64)
        expected_shape = (int(expected_review_intervals), int(expected_dim))
        if plan.shape != expected_shape:
            raise ValueError(f"allocation_plan shape={plan.shape}; expected {expected_shape}.")
        if np.any(plan < -1e-8):
            raise ValueError(f"allocation_plan contains negative shipment quantities: {artifact_path}")
        _assert_close("lambda_param", _npz_scalar(data, "lambda_param", artifact_path), expected_lambda, atol=1e-12)
        _assert_close("alpha", _npz_scalar(data, "alpha", artifact_path), expected_alpha, atol=1e-12)
        _assert_close("chi_eta", _npz_scalar(data, "chi_eta", artifact_path), expected_chi_eta, atol=1e-12)
        _assert_close("s_eta", _npz_scalar(data, "s_eta", artifact_path), expected_s_eta, atol=1e-12)
        _assert_close("review_dt", _npz_scalar(data, "review_dt", artifact_path), expected_review_dt, atol=1e-12)
        p_actual = np.asarray(data["p"], dtype=np.float64)
        p_expected = (
            np.full_like(p_actual, float(expected_p), dtype=np.float64)
            if np.asarray(expected_p).ndim == 0
            else np.asarray(expected_p, dtype=np.float64)
        )
        _assert_close("p", p_actual, p_expected, atol=1e-10)
        _assert_close("cost vector", np.asarray(data["c"], dtype=np.float64), expected_cost_vector, atol=5e-4)
        fairness_spec = str(_npz_scalar(data, "fairness_spec", artifact_path))
        if fairness_spec != EXPECTED_FAIRNESS_SPEC:
            raise ValueError(
                f"Fluid artifact {artifact_path} fairness_spec={fairness_spec!r}; "
                f"expected {EXPECTED_FAIRNESS_SPEC!r}."
            )
        if bool(_npz_scalar(data, "fairness_includes_hf", artifact_path)) is not True:
            raise ValueError("Fangonda fluid artifact fairness_includes_hf must be true.")
        if bool(_npz_scalar(data, "fairness_period_h_factor", artifact_path)) is not False:
            raise ValueError("Fangonda fluid artifact fairness_period_h_factor must be false.")
        calibration_version = str(_npz_scalar(data, "calibration_version", artifact_path))
        if calibration_version != str(expected_calibration_version):
            raise ValueError(
                f"calibration_version={calibration_version!r}; expected {expected_calibration_version!r}."
            )
        eta_mode = str(_npz_scalar(data, "eta_mode", artifact_path)) if "eta_mode" in data else ""
        if eta_mode != "weekly_time_varying":
            raise ValueError(f"eta_mode={eta_mode!r}; expected 'weekly_time_varying'.")
        solver_status_name = str(_npz_scalar(data, "solver_status_name", artifact_path)) if "solver_status_name" in data else ""
        if solver_status_name and solver_status_name != "OPTIMAL":
            raise ValueError(f"solver_status_name={solver_status_name!r}; expected 'OPTIMAL'.")
        return {
            "path": str(artifact_path),
            "allocation_shape": list(plan.shape),
            "lambda_param": float(expected_lambda),
            "alpha": float(expected_alpha),
            "chi_eta": float(expected_chi_eta),
            "s_eta": float(expected_s_eta),
            "review_intervals": int(expected_review_intervals),
            "review_dt": float(expected_review_dt),
            "calibration_version": calibration_version,
            "fairness_spec": fairness_spec,
            "solver_status_name": solver_status_name or None,
        }


def validate_run_metadata(
    config: Mapping[str, Any],
    *,
    run_dir: str | Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    expected_m: int | None = None,
    expected_eta_mode: str | None = "weekly_time_varying",
    expected_chi_eta: float | None = None,
    expected_s_eta: float | None = None,
    expected_lambda: float | None = None,
    expected_alpha: float | None = 0.5,
    expected_p: float | Sequence[float] | None = None,
    expected_cost_vector: Sequence[float] | None = None,
    expected_calibration_version: str | None = EXPECTED_CALIBRATION_VERSION,
) -> None:
    eqn_config = config.get("eqn_config", {})
    manifest_meta = {}
    if manifest is not None:
        manifest_meta = dict(manifest.get("metadata", manifest))
    elif run_dir is not None:
        manifest_path = Path(run_dir) / "scenario_manifest.json"
        if manifest_path.is_file():
            with open(manifest_path, encoding="utf-8") as f:
                manifest_meta = json.load(f).get("metadata", {})

    fairness_spec = eqn_config.get("fairness_spec")
    if fairness_spec != EXPECTED_FAIRNESS_SPEC:
        raise ValueError(f"fairness_spec={fairness_spec!r}; expected {EXPECTED_FAIRNESS_SPEC!r}.")
    if bool(eqn_config.get("fairness_includes_hf")) is not True:
        raise ValueError("fairness_includes_hf must be true.")
    if bool(eqn_config.get("fairness_period_h_factor")) is not False:
        raise ValueError("fairness_period_h_factor must be false.")

    alpha = float(eqn_config.get("alpha", np.nan))
    lambda_param = float(eqn_config.get("lambda_param", np.nan))
    _validate_fairness_parameters(alpha, lambda_param)
    if expected_alpha is not None and not np.isclose(alpha, float(expected_alpha), rtol=0.0, atol=1e-12):
        raise ValueError(f"alpha={alpha}; expected {expected_alpha}.")
    if expected_lambda is not None and not np.isclose(lambda_param, float(expected_lambda), rtol=0.0, atol=1e-12):
        raise ValueError(f"lambda_param={lambda_param}; expected {expected_lambda}.")

    if expected_m is not None:
        actual_m = int(eqn_config.get("num_time_interval", -1))
        if actual_m != int(expected_m):
            raise ValueError(f"num_time_interval={actual_m}; expected {expected_m}.")
    if expected_eta_mode is not None:
        eta_mode = str(eqn_config.get("eta_mode", eqn_config.get("eta_profile", "")))
        if eta_mode != str(expected_eta_mode):
            raise ValueError(f"eta_mode={eta_mode!r}; expected {expected_eta_mode!r}.")
    chi_eta = float(eqn_config.get("chi_eta", 0.0))
    s_eta = float(eqn_config.get("s_eta", 0.0))
    if chi_eta < 0.0 or s_eta < 0.0:
        raise ValueError(f"chi_eta and s_eta must be nonnegative, got {chi_eta}, {s_eta}.")
    if expected_chi_eta is not None and not np.isclose(chi_eta, float(expected_chi_eta), rtol=0.0, atol=1e-12):
        raise ValueError(f"chi_eta={chi_eta}; expected {expected_chi_eta}.")
    if expected_s_eta is not None and not np.isclose(s_eta, float(expected_s_eta), rtol=0.0, atol=1e-12):
        raise ValueError(f"s_eta={s_eta}; expected {expected_s_eta}.")

    if expected_p is None and "p" in manifest_meta:
        expected_p = manifest_meta["p"]
    if expected_cost_vector is None and "effective_shortage_costs" in manifest_meta:
        expected_cost_vector = manifest_meta["effective_shortage_costs"]

    if expected_p is not None:
        p = _read_vector(config, run_dir, "p_file")
        if p is not None:
            expected = (
                np.full_like(p, float(expected_p), dtype=np.float64)
                if np.asarray(expected_p).ndim == 0
                else np.asarray(expected_p, dtype=np.float64)
            )
            _assert_close("p", p, expected, atol=1e-10)
    if expected_cost_vector is not None:
        c = _read_vector(config, run_dir, "c_file")
        if c is not None:
            _assert_close("cost vector", c, expected_cost_vector, atol=5e-4)

    calibration_version = manifest_meta.get("calibration_version")
    if (
        expected_calibration_version is not None
        and calibration_version is not None
        and calibration_version != expected_calibration_version
    ):
        raise ValueError(
            f"calibration_version={calibration_version!r}; expected {expected_calibration_version!r}."
        )
