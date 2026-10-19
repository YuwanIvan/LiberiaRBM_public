"""Reusable validation-bank construction for Deep Splitting models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch

import equation as eqn


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _assert_close(name: str, actual, expected, atol=1e-7):
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    if actual_arr.shape != expected_arr.shape or not np.allclose(actual_arr, expected_arr, rtol=0.0, atol=atol):
        raise ValueError(f"validation bank {name} mismatch: got {actual_arr.tolist()}, expected {expected_arr.tolist()}.")


def _assert_equal(name: str, actual, expected):
    if actual != expected:
        raise ValueError(f"validation bank {name} mismatch: got {actual!r}, expected {expected!r}.")


def build_common_validation_bank(solver, intervals: Iterable[int] | None = None, size: int | None = None) -> Dict[str, Any]:
    """Sample complete one-step validation transitions from a configured solver.

    This uses ``solver.sample`` so state components, reference actions,
    regulator increments, and Brownian increments are drawn jointly from the
    same sampler/transition kernel used for training validation.
    """
    size = int(size if size is not None else solver.valid_size)
    if size <= 0:
        raise ValueError("validation bank size must be positive.")
    if intervals is None:
        intervals = range(int(solver.num_time_interval))
    intervals = [int(idx) for idx in intervals]
    if not intervals:
        raise ValueError("validation bank intervals must be nonempty.")

    current_states = []
    next_states = []
    reference_actions = []
    regulator_increments = []
    brownian_increments = []
    reference_policy_labels = []
    path_ids = []
    mu_rows = []
    gamma_rows = []
    eta_rows = []
    seed = getattr(solver, "sampling_seed", None)
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        for n in intervals:
            sample_out = solver.sample(size, n, return_metadata=True)
            Y_n, Y_n1, a_ref, L_inc, W_inc, sample_meta = sample_out
            current_states.append(Y_n.detach().cpu().numpy())
            next_states.append(Y_n1.detach().cpu().numpy())
            reference_actions.append(a_ref.detach().cpu().numpy())
            regulator_increments.append(L_inc.detach().cpu().numpy())
            brownian_increments.append(W_inc.detach().cpu().numpy())
            labels = sample_meta.get("reference_policy_labels")
            ids = sample_meta.get("path_ids")
            reference_policy_labels.append(
                np.asarray(labels, dtype=np.int64) if labels is not None else np.full((size,), -1, dtype=np.int64)
            )
            path_ids.append(
                np.asarray(ids, dtype=np.int64) if ids is not None else np.full((size,), -1, dtype=np.int64)
            )
            mu_rows.append(solver.bsde.mu[n].detach().cpu().numpy())
            gamma_rows.append(solver.bsde.Gamma[n].detach().cpu().numpy())
            eta_rows.append(solver.bsde.eta_at_fine_index(n).detach().cpu().numpy())
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    metadata = {
        "fairness_spec": eqn.FAIRNESS_SPEC,
        "fairness_includes_hf": eqn.FAIRNESS_INCLUDES_HF,
        "fairness_period_h_factor": eqn.FAIRNESS_PERIOD_H_FACTOR,
        "num_intervals": len(intervals),
        "solver_num_time_interval": int(solver.num_time_interval),
        "dim": int(solver.bsde.dim),
        "total_time": float(solver.bsde.total_time),
        "delta_t": float(solver.bsde.delta_t),
        "intervals": intervals,
        "size_per_interval": size,
        "sampling_mode": getattr(solver, "sampling_mode", None),
        "sampling_seed": seed,
        "proposal_sampler": getattr(solver, "proposal_sampler", None),
        "eta_mode": getattr(solver.bsde, "eta_mode", None),
        "chi_eta": getattr(solver.bsde, "chi_eta", None),
        "s_eta": getattr(solver.bsde, "s_eta", None),
        "alpha": getattr(solver.bsde, "alpha", None),
        "lambda_param": getattr(solver.bsde, "lambda_param", None),
        "calibration_version": getattr(solver.bsde, "calibration_version", None),
        "q0": float(getattr(solver.bsde, "I0", np.nan)),
        "p": _to_numpy(getattr(solver.bsde, "p")).tolist(),
        "c_tilde": _to_numpy(getattr(solver.bsde, "c_tilde")).tolist(),
        "state_sampling_note": "Generated by solver.sample; state components were not sampled independently.",
    }

    return {
        "current_states": np.stack(current_states, axis=0),
        "next_states": np.stack(next_states, axis=0),
        "reference_actions": np.stack(reference_actions, axis=0),
        "regulator_increments": np.stack(regulator_increments, axis=0),
        "brownian_increments": np.stack(brownian_increments, axis=0),
        "reference_policy_labels": np.stack(reference_policy_labels, axis=0),
        "path_ids": np.stack(path_ids, axis=0),
        "mu_rows": np.stack(mu_rows, axis=0),
        "gamma_rows": np.stack(gamma_rows, axis=0),
        "eta_rows": np.stack(eta_rows, axis=0),
        "metadata": metadata,
    }


def build_equal_reference_common_validation_bank(
    solvers_by_reference: Mapping[str, Any],
    intervals: Iterable[int] | None = None,
    size: int | None = None,
    references: Iterable[str] = ("R0", "R1", "R2", "R3", "R4"),
) -> Dict[str, Any]:
    """Build a common validation bank with equal held-out rows from R0--R4.

    Each contribution is a complete one-step transition sampled from that
    reference solver, preserving reference action, Brownian increment, label,
    and path id coherence.
    """
    references = tuple(str(ref) for ref in references)
    if not references:
        raise ValueError("references must be nonempty.")
    missing = [ref for ref in references if ref not in solvers_by_reference]
    if missing:
        raise ValueError(f"Missing solvers for references: {missing}.")
    first_solver = solvers_by_reference[references[0]]
    size = int(size if size is not None else first_solver.valid_size)
    if size <= 0:
        raise ValueError("validation bank size must be positive.")
    if size % len(references) != 0:
        raise ValueError(
            f"size={size} must be divisible by {len(references)} for equal reference contributions."
        )
    per_reference = size // len(references)
    partials = {
        ref: build_common_validation_bank(
            solvers_by_reference[ref],
            intervals=intervals,
            size=per_reference,
        )
        for ref in references
    }
    base = partials[references[0]]
    arrays = {}
    for key in (
        "current_states",
        "next_states",
        "reference_actions",
        "regulator_increments",
        "brownian_increments",
        "reference_policy_labels",
        "path_ids",
    ):
        arrays[key] = np.concatenate([partials[ref][key] for ref in references], axis=1)
    for key in ("mu_rows", "gamma_rows", "eta_rows"):
        arrays[key] = base[key]
    contribution_codes = []
    for ref_idx, _ref in enumerate(references):
        contribution_codes.append(np.full((per_reference,), ref_idx, dtype=np.int64))
    contribution_codes = np.concatenate(contribution_codes, axis=0)
    contribution_by_interval = np.broadcast_to(
        contribution_codes.reshape(1, -1),
        (arrays["current_states"].shape[0], contribution_codes.shape[0]),
    )
    metadata = dict(base["metadata"])
    metadata.update(
        {
            "common_validation_bank_type": "equal_reference_contributions",
            "reference_contributions": list(references),
            "reference_contribution_size": per_reference,
            "size_per_interval": size,
            "state_sampling_note": (
                "Equal held-out complete one-step transitions from R0--R4; "
                "state/action/transition labels are preserved."
            ),
        }
    )
    arrays["reference_contribution_codes"] = contribution_by_interval.astype(np.int64)
    arrays["metadata"] = metadata
    return arrays


def save_common_validation_bank(path: str | Path, bank: Mapping[str, Any]) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in bank.items()
        if key != "metadata"
    }
    arrays["metadata_json"] = np.asarray([json.dumps(bank.get("metadata", {}), default=_json_default)])
    np.savez_compressed(path, **arrays)
    return path


def load_common_validation_bank(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser()
    with np.load(path, allow_pickle=False) as data:
        out = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
        metadata_json = str(np.asarray(data["metadata_json"]).reshape(-1)[0])
    out["metadata"] = json.loads(metadata_json)
    return out


def validate_common_validation_bank_compatibility(bank: Mapping[str, Any], solver) -> None:
    """Fail fast if a common validation bank was generated for different primitives."""
    metadata = dict(bank.get("metadata", {}))
    _assert_equal("fairness_spec", metadata.get("fairness_spec"), eqn.FAIRNESS_SPEC)
    _assert_equal("fairness_includes_hf", bool(metadata.get("fairness_includes_hf")), eqn.FAIRNESS_INCLUDES_HF)
    _assert_equal("fairness_period_h_factor", bool(metadata.get("fairness_period_h_factor")), eqn.FAIRNESS_PERIOD_H_FACTOR)
    _assert_equal("solver_num_time_interval", int(metadata.get("solver_num_time_interval", -1)), int(solver.num_time_interval))
    _assert_equal("dim", int(metadata.get("dim", -1)), int(solver.bsde.dim))
    _assert_close("total_time", metadata.get("total_time"), float(solver.bsde.total_time), atol=1e-12)
    _assert_close("delta_t", metadata.get("delta_t"), float(solver.bsde.delta_t), atol=1e-12)
    _assert_close("alpha", metadata.get("alpha"), float(solver.bsde.alpha), atol=1e-12)
    _assert_close("lambda_param", metadata.get("lambda_param"), float(solver.bsde.lambda_param), atol=1e-12)
    _assert_close("q0", metadata.get("q0"), float(solver.bsde.I0), atol=1e-8)
    _assert_close("p", metadata.get("p"), _to_numpy(solver.bsde.p), atol=1e-7)
    _assert_close("c_tilde", metadata.get("c_tilde"), _to_numpy(solver.bsde.c_tilde), atol=1e-6)
    _assert_equal("eta_mode", metadata.get("eta_mode"), getattr(solver.bsde, "eta_mode", None))
    _assert_close("chi_eta", metadata.get("chi_eta"), float(solver.bsde.chi_eta), atol=1e-12)
    _assert_close("s_eta", metadata.get("s_eta"), float(solver.bsde.s_eta), atol=1e-12)
    expected_calibration = getattr(solver.bsde, "calibration_version", None)
    if expected_calibration is not None:
        _assert_equal("calibration_version", metadata.get("calibration_version"), expected_calibration)

    for key in ("sampling_mode", "proposal_sampler"):
        expected = getattr(solver, key, None)
        if expected is not None and metadata.get(key) is not None:
            _assert_equal(key, metadata.get(key), expected)

    intervals = [int(x) for x in metadata.get("intervals", [])]
    if len(intervals) != int(metadata.get("num_intervals", -1)):
        raise ValueError("validation bank intervals length does not match num_intervals metadata.")
    if int(bank["current_states"].shape[0]) != len(intervals):
        raise ValueError("validation bank current_states first dimension does not match intervals metadata.")
    if int(bank["current_states"].shape[1]) != int(metadata.get("size_per_interval", -1)):
        raise ValueError("validation bank size_per_interval mismatch.")
    for optional_key in ("reference_policy_labels", "path_ids", "reference_contribution_codes"):
        if optional_key in bank and bank[optional_key].shape[:2] != bank["current_states"].shape[:2]:
            raise ValueError(f"validation bank {optional_key} shape does not match current_states.")

    for bank_idx, interval in enumerate(intervals):
        if interval < 0 or interval >= int(solver.num_time_interval):
            raise ValueError(f"validation bank interval {interval} is outside solver grid.")
        _assert_close(f"mu_rows[{bank_idx}]", bank["mu_rows"][bank_idx], _to_numpy(solver.bsde.mu[interval]), atol=1e-7)
        _assert_close(f"gamma_rows[{bank_idx}]", bank["gamma_rows"][bank_idx], _to_numpy(solver.bsde.Gamma[interval]), atol=1e-7)
        _assert_close(
            f"eta_rows[{bank_idx}]",
            bank["eta_rows"][bank_idx],
            _to_numpy(solver.bsde.eta_at_fine_index(interval)),
            atol=1e-6,
        )
