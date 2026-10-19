"""Strict guards for cross-run checkpoint continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


class CheckpointContinuationError(ValueError):
    """Raised when a continuation source is incompatible with the target run."""


STATE_SAMPLER_KEYS = (
    "sampling_mode",
    "proposal_sampler",
    "rollout_pool_size",
    "sampling_seed",
    "uniform_z_radius_mult",
    "uniform_q_radius_frac",
    "total_inventory_buffer",
    "q_low_values",
    "boundary_eps_frac",
    "mix_rollout",
    "mix_box",
    "mix_shortage",
    "shortage_q_frac",
    "shortage_z_frac",
    "lognormal_delta_z",
    "lognormal_delta_q",
    "lognormal_r_min",
    "initial_state_mode",
    "initial_perturb_prob",
    "initial_q_perturb_frac",
    "initial_z_perturb_frac",
    "initial_perturb_minus_prob",
    "initial_perturb_plus_prob",
    "initial_perturb_minus_frac",
    "initial_perturb_plus_frac",
    "initial_dirichlet_alpha",
)

NETWORK_KEYS = (
    "num_neurons",
    "num_layers",
    "activation",
    "init_method",
    "h_init_method",
    "g_init_method",
    "g_output_bias",
    "input_mode",
    "q0_input_scale",
    "h_output_scale",
)
FAIRNESS_KEYS = ("fairness_spec", "fairness_includes_hf", "fairness_period_h_factor")
M110_BLOCKWISE_MODE = "m110_blockwise_coarse_init"
M110_BLOCKWISE_MATCHED_MODE = "m110_blockwise_matched_init"
BLOCKWISE_SCENARIO_PARAMETER_FILES = (
    "c.csv",
    "distance.csv",
    "eta.csv",
    "eta_schedule.csv",
    "exposure.csv",
    "gamma.csv",
    "h.csv",
    "mu.csv",
    "p.csv",
    "r.csv",
    "theta.csv",
    "u.csv",
)
BLOCKWISE_EQN_KEYS = (
    "I0",
    "K",
    "total_time",
    "calibration_version",
    "periodic_demand_mode",
    "reference_mixture",
    "reference_mixture_counts",
)


def config_loss_mode(net_config: Dict[str, Any]) -> str:
    mode = net_config.get("loss_mode")
    if mode:
        return str(mode)
    legacy = net_config.get("loss_type")
    if legacy is None or int(legacy) == 2:
        return "ce_consistency"
    if int(legacy) == 1:
        return "legacy_loss_type_1"
    return f"legacy_loss_type_{legacy}"


def config_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    for filename in ("config_used.json", "config_scenario.json"):
        path = run_dir / filename
        if path.is_file():
            return load_json(path)
    raise CheckpointContinuationError(f"Missing config_used.json/config_scenario.json under {run_dir}.")


def load_vector(path: Path) -> np.ndarray:
    if not path.is_file():
        raise CheckpointContinuationError(f"Missing required metadata vector: {path}")
    return np.loadtxt(path, delimiter=",").reshape(-1).astype(float)


def load_array(path: Path) -> np.ndarray:
    """Load a numeric parameter file without discarding its recorded shape."""

    if not path.is_file():
        raise CheckpointContinuationError(f"Missing required metadata array: {path}")
    return np.asarray(np.loadtxt(path, delimiter=","), dtype=float)


def resolve_params_dir(run_dir: Path, config: Dict[str, Any]) -> Path:
    run_params = run_dir / "params"
    if run_params.is_dir():
        return run_params
    config_dir = config.get("eqn_config", {}).get("config_dir")
    if not config_dir:
        raise CheckpointContinuationError(f"Cannot resolve params dir for {run_dir}.")
    path = Path(config_dir).expanduser()
    if not path.is_absolute():
        path = (run_dir / path).resolve()
    return path


def _as_float(value: Any, label: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise CheckpointContinuationError(f"{label} is not numeric: {value!r}") from exc
    if not np.isfinite(out):
        raise CheckpointContinuationError(f"{label} is not finite: {value!r}")
    return out


def _assert_equal(label: str, source: Any, target: Any) -> None:
    if source != target:
        raise CheckpointContinuationError(f"Checkpoint metadata mismatch for {label}: source={source!r}, target={target!r}.")


def _assert_close(label: str, source: Any, target: Any, atol: float = 1e-10) -> None:
    s = _as_float(source, f"source {label}")
    t = _as_float(target, f"target {label}")
    if not np.isclose(s, t, rtol=0.0, atol=atol):
        raise CheckpointContinuationError(f"Checkpoint metadata mismatch for {label}: source={s}, target={t}.")


def _assert_array_close(label: str, source: np.ndarray, target: np.ndarray, atol: float = 1e-8) -> None:
    if source.shape != target.shape:
        raise CheckpointContinuationError(
            f"Checkpoint metadata mismatch for {label} shape: source={source.shape}, target={target.shape}."
        )
    if not np.allclose(source, target, rtol=0.0, atol=atol):
        max_abs = float(np.max(np.abs(source - target)))
        raise CheckpointContinuationError(f"Checkpoint metadata mismatch for {label}: max_abs_diff={max_abs:g}.")


def _compare_keys(label: str, source: Dict[str, Any], target: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    compared: Dict[str, Any] = {}
    for key in keys:
        source_value = source.get(key)
        target_value = target.get(key)
        _assert_equal(f"{label}.{key}", source_value, target_value)
        compared[key] = target_value
    return compared


def _checkpoint_pairs(checkpoint_dir: Path, num_intervals: int, *, require_qgrad: bool = False) -> tuple[list[dict[str, Any]], str]:
    pairs = []
    missing = []
    for interval in range(num_intervals):
        y_path = checkpoint_dir / f"y_network{interval}.pth"
        z_path = checkpoint_dir / f"z_network{interval}.pth"
        qgrad_path = checkpoint_dir / f"qgrad_network{interval}.pth"
        if not y_path.is_file() or not z_path.is_file() or (require_qgrad and not qgrad_path.is_file()):
            missing.append(interval)
        y_hash = file_sha256(y_path) if y_path.is_file() else ""
        z_hash = file_sha256(z_path) if z_path.is_file() else ""
        qgrad_hash = file_sha256(qgrad_path) if qgrad_path.is_file() else ""
        pair_hash = config_hash(
            {
                "source_interval": interval,
                "target_interval": interval,
                "y_checkpoint_sha256": y_hash,
                "z_checkpoint_sha256": z_hash,
                "qgrad_checkpoint_sha256": qgrad_hash,
            }
        )
        pairs.append(
            {
                "interval": interval,
                "source_interval": interval,
                "target_interval": interval,
                "y_checkpoint": str(y_path),
                "z_checkpoint": str(z_path),
                "qgrad_checkpoint": str(qgrad_path) if require_qgrad else "",
                "y_checkpoint_sha256": y_hash,
                "z_checkpoint_sha256": z_hash,
                "qgrad_checkpoint_sha256": qgrad_hash,
                "source_checkpoint_hash": pair_hash,
            }
        )
    if missing:
        preview = ", ".join(str(item) for item in missing[:20])
        suffix = "..." if len(missing) > 20 else ""
        raise CheckpointContinuationError(
            f"Missing source checkpoint sets in {checkpoint_dir} for intervals: {preview}{suffix}"
        )
    overall_hash = config_hash(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "num_intervals": num_intervals,
            "pairs": [
                {
                    "source_interval": pair["source_interval"],
                    "target_interval": pair["target_interval"],
                    "y_checkpoint_sha256": pair["y_checkpoint_sha256"],
                    "z_checkpoint_sha256": pair["z_checkpoint_sha256"],
                    "qgrad_checkpoint_sha256": pair.get("qgrad_checkpoint_sha256", ""),
                }
                for pair in pairs
            ],
        }
    )
    return pairs, overall_hash


def _checkpoint_pairs_m110_blockwise(
    checkpoint_dir: Path,
    *,
    source_intervals: int,
    target_intervals: int,
    block_size: int,
    mapping_mode: str = M110_BLOCKWISE_MODE,
    require_qgrad: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    pairs = []
    missing = []
    for source_interval in range(source_intervals):
        target_interval = (source_interval + 1) * block_size - 1
        if target_interval >= target_intervals:
            raise CheckpointContinuationError(
                "Blockwise checkpoint mapping produced target interval "
                f"{target_interval} outside 0..{target_intervals - 1}."
            )
        y_path = checkpoint_dir / f"y_network{source_interval}.pth"
        z_path = checkpoint_dir / f"z_network{source_interval}.pth"
        qgrad_path = checkpoint_dir / f"qgrad_network{source_interval}.pth"
        if not y_path.is_file() or not z_path.is_file() or (require_qgrad and not qgrad_path.is_file()):
            missing.append(source_interval)
        y_hash = file_sha256(y_path) if y_path.is_file() else ""
        z_hash = file_sha256(z_path) if z_path.is_file() else ""
        qgrad_hash = file_sha256(qgrad_path) if qgrad_path.is_file() else ""
        pair_hash = config_hash(
            {
                "mapping": mapping_mode,
                "source_interval": source_interval,
                "target_interval": target_interval,
                "block_size": block_size,
                "y_checkpoint_sha256": y_hash,
                "z_checkpoint_sha256": z_hash,
                "qgrad_checkpoint_sha256": qgrad_hash,
            }
        )
        pairs.append(
            {
                "interval": target_interval,
                "source_interval": source_interval,
                "target_interval": target_interval,
                "initialization_source": "coarse_block_entry",
                "source_block_index": source_interval,
                "block_entry_interval": target_interval,
                "block_start_interval": source_interval * block_size,
                "block_end_interval": target_interval,
                "block_size": block_size,
                "y_checkpoint": str(y_path),
                "z_checkpoint": str(z_path),
                "qgrad_checkpoint": str(qgrad_path) if require_qgrad else "",
                "y_checkpoint_sha256": y_hash,
                "z_checkpoint_sha256": z_hash,
                "qgrad_checkpoint_sha256": qgrad_hash,
                "source_checkpoint_hash": pair_hash,
            }
        )
    if missing:
        preview = ", ".join(str(item) for item in missing[:20])
        suffix = "..." if len(missing) > 20 else ""
        raise CheckpointContinuationError(
            f"Missing source checkpoint sets in {checkpoint_dir} for M110 intervals: {preview}{suffix}"
        )
    overall_hash = config_hash(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "mapping": mapping_mode,
            "source_intervals": source_intervals,
            "target_intervals": target_intervals,
            "block_size": block_size,
            "pairs": [
                {
                    "source_interval": pair["source_interval"],
                    "target_interval": pair["target_interval"],
                    "y_checkpoint_sha256": pair["y_checkpoint_sha256"],
                    "z_checkpoint_sha256": pair["z_checkpoint_sha256"],
                    "qgrad_checkpoint_sha256": pair.get("qgrad_checkpoint_sha256", ""),
                }
                for pair in pairs
            ],
        }
    )
    return pairs, overall_hash


def validate_checkpoint_continuation(
    *,
    source_checkpoint_dir: str | Path,
    target_run_dir: str | Path,
    target_config: Dict[str, Any],
    expected_source_lambda: Optional[float] = None,
    initialization_mode: str = "lambda_homotopy",
) -> Dict[str, Any]:
    """Validate that source checkpoints can initialize the target run.

    Lambda is intentionally allowed to differ; the model, data, fairness, reference
    process, architecture, and state sampler must match exactly or near-exactly.
    """

    # Preserve the supplied logical mount path: resolving storage aliases
    # can produce a path unavailable on the host reading these records.
    checkpoint_dir = Path(source_checkpoint_dir).expanduser().absolute()
    if checkpoint_dir.name == "logs":
        source_run_dir = checkpoint_dir.parent
    else:
        source_run_dir = checkpoint_dir
        checkpoint_dir = source_run_dir / "logs"
    target_run_dir = Path(target_run_dir).expanduser().absolute()

    if not checkpoint_dir.is_dir():
        raise CheckpointContinuationError(f"Source checkpoint directory does not exist: {checkpoint_dir}")

    source_config = load_run_config(source_run_dir)
    source_eqn = source_config.get("eqn_config", {})
    target_eqn = target_config.get("eqn_config", {})
    source_net = source_config.get("net_config", {})
    target_net = target_config.get("net_config", {})
    source_loss_mode = config_loss_mode(source_net)
    target_loss_mode = config_loss_mode(target_net)
    _assert_equal("loss_mode", source_loss_mode, target_loss_mode)
    blockwise_mode = initialization_mode in {M110_BLOCKWISE_MODE, M110_BLOCKWISE_MATCHED_MODE}
    matched_blockwise_mode = initialization_mode == M110_BLOCKWISE_MATCHED_MODE

    source_lambda = _as_float(source_eqn.get("lambda_param"), "source lambda_param")
    target_lambda = _as_float(target_eqn.get("lambda_param"), "target lambda_param")
    if expected_source_lambda is not None:
        _assert_close("source lambda_param", source_lambda, expected_source_lambda, atol=1e-12)

    if blockwise_mode:
        source_m = int(source_eqn.get("num_time_interval"))
        target_m = int(target_eqn.get("num_time_interval"))
        source_net_m = int(source_net.get("num_time_interval"))
        target_net_m = int(target_net.get("num_time_interval"))
        if (source_m, source_net_m, target_m, target_net_m) != (110, 110, 1100, 1100):
            raise CheckpointContinuationError(
                "m110_blockwise_coarse_init requires source M=110 and target M=1100; "
                f"got source eqn/net M={source_m}/{source_net_m}, target eqn/net M={target_m}/{target_net_m}."
            )
        _assert_close("lambda_param", source_lambda, target_lambda, atol=1e-12)
        if matched_blockwise_mode:
            _assert_close("chi_eta", source_eqn.get("chi_eta"), target_eqn.get("chi_eta"), atol=1e-12)
            for key in BLOCKWISE_EQN_KEYS:
                _assert_equal(f"eqn_config.{key}", source_eqn.get(key), target_eqn.get(key))
        else:
            _assert_close("blockwise lambda_param", source_lambda, 0.25, atol=1e-12)
            _assert_close("source chi_eta", source_eqn.get("chi_eta"), 1.75, atol=1e-12)
            target_chi = _as_float(target_eqn.get("chi_eta"), "target chi_eta")
            allowed_chi = (1.25, 1.5, 1.75)
            if not any(np.isclose(target_chi, value, rtol=0.0, atol=1e-12) for value in allowed_chi):
                raise CheckpointContinuationError(
                    "m110_blockwise_coarse_init target chi_eta must be one of "
                    f"{list(allowed_chi)}; got {target_chi}."
                )
    else:
        _assert_equal("eqn_config.num_time_interval", source_eqn.get("num_time_interval"), target_eqn.get("num_time_interval"))
        _assert_equal("net_config.num_time_interval", source_net.get("num_time_interval"), target_net.get("num_time_interval"))
    _assert_equal("eqn_config.dim", source_eqn.get("dim"), target_eqn.get("dim"))
    _assert_close("alpha", source_eqn.get("alpha"), target_eqn.get("alpha"), atol=1e-12)
    if not blockwise_mode:
        _assert_close("chi_eta", source_eqn.get("chi_eta"), target_eqn.get("chi_eta"), atol=1e-12)
    _assert_close("s_eta", source_eqn.get("s_eta"), target_eqn.get("s_eta"), atol=1e-12)
    _assert_equal("eta_mode", source_eqn.get("eta_mode", source_eqn.get("eta_profile")), target_eqn.get("eta_mode", target_eqn.get("eta_profile")))
    _assert_equal("reference_policy", source_eqn.get("policy"), target_eqn.get("policy"))
    if not blockwise_mode:
        _assert_equal("net_config.train_seed", source_net.get("train_seed"), target_net.get("train_seed"))
    _assert_close("consistency_coef", source_net.get("consistency_coef", 0.0), target_net.get("consistency_coef", 0.0), atol=1e-12)
    if blockwise_mode:
        _assert_close("blockwise consistency_coef", target_net.get("consistency_coef", 0.0), 30.0, atol=1e-12)
    _compare_keys("fairness", source_eqn, target_eqn, FAIRNESS_KEYS)
    network = _compare_keys("net_config", source_net, target_net, NETWORK_KEYS)
    if blockwise_mode:
        state_sampler = {key: target_net.get(key) for key in STATE_SAMPLER_KEYS}
        if matched_blockwise_mode:
            comparable_sampler_keys = tuple(key for key in STATE_SAMPLER_KEYS if key != "sampling_seed")
            state_sampler = _compare_keys(
                "state_sampler",
                source_net,
                target_net,
                comparable_sampler_keys,
            )
    else:
        state_sampler = _compare_keys("state_sampler", source_net, target_net, STATE_SAMPLER_KEYS)

    state_dim = int(target_eqn["dim"]) + 1
    num_intervals = int(target_eqn["num_time_interval"])
    source_params = resolve_params_dir(source_run_dir, source_config)
    target_params = resolve_params_dir(target_run_dir, target_config)
    source_p = load_vector(source_params / "p.csv")
    target_p = load_vector(target_params / "p.csv")
    source_c = load_vector(source_params / "c.csv")
    target_c = load_vector(target_params / "c.csv")
    _assert_array_close("p vector", source_p, target_p, atol=1e-10)
    _assert_array_close("effective cost vector c", source_c, target_c, atol=1e-8)
    scenario_parameter_hashes: Dict[str, Dict[str, str]] = {}
    if matched_blockwise_mode:
        for filename in BLOCKWISE_SCENARIO_PARAMETER_FILES:
            source_path = source_params / filename
            target_path = target_params / filename
            source_values = load_array(source_path)
            target_values = load_array(target_path)
            _assert_array_close(f"scenario parameter {filename}", source_values, target_values, atol=1e-8)
            scenario_parameter_hashes[filename] = {
                "source_sha256": file_sha256(source_path),
                "target_sha256": file_sha256(target_path),
            }

    if blockwise_mode:
        checkpoint_pairs, source_checkpoint_hash = _checkpoint_pairs_m110_blockwise(
            checkpoint_dir,
            source_intervals=110,
            target_intervals=num_intervals,
            block_size=10,
            mapping_mode=initialization_mode,
            require_qgrad=target_loss_mode == "martingale_qgrad",
        )
    else:
        checkpoint_pairs, source_checkpoint_hash = _checkpoint_pairs(
            checkpoint_dir,
            num_intervals,
            require_qgrad=target_loss_mode == "martingale_qgrad",
        )
    metadata = {
        "initialization_mode": initialization_mode,
        "source_run_dir": str(source_run_dir),
        "source_run_path": str(source_run_dir),
        "source_checkpoint_dir": str(checkpoint_dir),
        "source_checkpoint_hash": source_checkpoint_hash,
        "target_run_dir": str(target_run_dir),
        "loss_mode": target_loss_mode,
        "source_lambda": source_lambda,
        "target_lambda": target_lambda,
        "optimizer_state_reused": False,
        "scheduler_state_reused": False,
        "early_stopping_state_reused": False,
        "same_interval_initialization": not blockwise_mode,
        "same_interval_overrides_prev_init": not blockwise_mode,
        "blockwise_coarse_initialization": bool(blockwise_mode),
        "blockwise_mapping_mode": initialization_mode if blockwise_mode else "",
        "block_size": 10 if blockwise_mode else "",
        "source_M": int(source_eqn.get("num_time_interval")) if blockwise_mode else num_intervals,
        "target_M": num_intervals,
        "source_checkpoint_load_count": len(checkpoint_pairs),
        "block_internal_initialization_source": "adjacent_target" if blockwise_mode else "",
        "continuation_target_source": "target_next_interval",
        "terminal_loss_source": "target_lambda_terminal_loss",
        "M": num_intervals,
        "chi_eta": float(target_eqn.get("chi_eta")),
        "s_eta": float(target_eqn.get("s_eta")),
        "alpha": float(target_eqn.get("alpha")),
        "fairness_spec": target_eqn.get("fairness_spec"),
        "fairness_includes_hf": target_eqn.get("fairness_includes_hf"),
        "fairness_period_h_factor": target_eqn.get("fairness_period_h_factor"),
        "reference_policy": target_eqn.get("policy"),
        "train_seed": int(target_net.get("train_seed")),
        "network_architecture": network,
        "state_dim": state_dim,
        "grad_diff_dim": int(target_eqn["dim"]),
        "state_sampler": state_sampler,
        "consistency_coef": float(target_net.get("consistency_coef", 0.0)),
        "p": target_p.tolist(),
        "effective_cost_vector": target_c.tolist(),
        "scenario_parameter_hashes": scenario_parameter_hashes,
        "checkpoint_pairs": checkpoint_pairs,
        "source_config_hash": config_hash(source_config),
        "target_config_hash": config_hash(target_config),
    }
    return metadata


def save_continuation_metadata(path: str | Path, metadata: Dict[str, Any]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
