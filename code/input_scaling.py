"""Function-preserving state-input scaling utilities for Deep Splitting.

The production training path uses physical state coordinates.  These helpers are
for diagnostics that compare an equivalent scaled input parameterization without
changing the represented H/G functions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


TRANSFORMATION_VERSION = "scale00_first_layer_column_transform_v1"
H_OUTPUT_TRANSFORMATION_VERSION = "reparam00_h_output_transform_v1"
DEFAULT_FIRST_WEIGHT_KEY = "linearlayers.0.weight"
DEFAULT_OUTPUT_WEIGHT_KEY = "linearlayers.-1.weight"


def state_coordinate_labels(state_dim: int) -> list[str]:
    """Return labels for the Deep Splitting state vector order ``(Z_0,...,Z_K,Q)``."""
    state_dim = int(state_dim)
    if state_dim < 2:
        raise ValueError(f"state_dim must be at least 2, got {state_dim}.")
    return [f"Z_{idx}" for idx in range(state_dim - 1)] + ["Q"]


def as_scale_tensor(
    scale_vector: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """Convert and validate a positive 1D scale vector."""
    if torch.is_tensor(scale_vector):
        scale = scale_vector.detach().clone()
        if device is not None or dtype is not None:
            scale = scale.to(device=device, dtype=dtype)
    else:
        scale = torch.tensor(np.asarray(scale_vector, dtype=np.float64), device=device, dtype=dtype or torch.float64)
    scale = scale.reshape(-1)
    if expected_dim is not None and int(scale.numel()) != int(expected_dim):
        raise ValueError(f"scale_vector has dim {scale.numel()}, expected {int(expected_dim)}.")
    if not torch.all(torch.isfinite(scale)):
        raise ValueError("scale_vector contains non-finite values.")
    if not torch.all(scale > 0):
        raise ValueError("scale_vector entries must be strictly positive.")
    return scale


def apply_state_input_scaling(states: torch.Tensor, scale_vector: Sequence[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Map physical state coordinates to scaled network inputs, ``x_scaled = x / s``."""
    scale = as_scale_tensor(scale_vector, device=states.device, dtype=states.dtype, expected_dim=states.shape[-1])
    return states / scale


def invert_state_input_scaling(states_scaled: torch.Tensor, scale_vector: Sequence[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Map scaled network inputs back to physical coordinates, ``x = x_scaled * s``."""
    scale = as_scale_tensor(scale_vector, device=states_scaled.device, dtype=states_scaled.dtype, expected_dim=states_scaled.shape[-1])
    return states_scaled * scale


def uniform_q0_scale_vector(state_dim: int, q0: float) -> np.ndarray:
    """Return S1: all coordinates scaled by the fixed physical inventory level Q0."""
    q0 = float(q0)
    if not np.isfinite(q0) or q0 <= 0.0:
        raise ValueError(f"q0 must be finite and positive, got {q0}.")
    return np.full(int(state_dim), q0, dtype=np.float64)


def summarize_state_bank(
    states: np.ndarray | torch.Tensor,
    *,
    labels: Sequence[str] | None = None,
) -> list[Dict[str, Any]]:
    """Summarize state-bank coordinate quantiles used to calibrate robust scaling."""
    if torch.is_tensor(states):
        arr = states.detach().cpu().numpy()
    else:
        arr = np.asarray(states)
    if arr.ndim < 2:
        raise ValueError(f"states must have at least two dimensions, got shape {arr.shape}.")
    arr = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
    if labels is None:
        labels = state_coordinate_labels(arr.shape[1])
    if len(labels) != arr.shape[1]:
        raise ValueError(f"labels length {len(labels)} does not match state_dim {arr.shape[1]}.")
    rows: list[Dict[str, Any]] = []
    for idx, label in enumerate(labels):
        values = arr[:, idx]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"No finite state-bank values for coordinate {label}.")
        rows.append(
            {
                "coordinate": idx,
                "label": str(label),
                "count": int(values.size),
                "min": float(np.min(values)),
                "p01": float(np.percentile(values, 1)),
                "p05": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
            }
        )
    return rows


def robust_scale_vector_from_states(
    states: np.ndarray | torch.Tensor,
    *,
    lower_bound: float = 1.0,
    labels: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[Dict[str, Any]]]:
    """Return S2: fixed per-coordinate robust scales ``max(p95(x_j), lower_bound)``."""
    lower_bound = float(lower_bound)
    if not np.isfinite(lower_bound) or lower_bound <= 0.0:
        raise ValueError(f"lower_bound must be finite and positive, got {lower_bound}.")
    rows = summarize_state_bank(states, labels=labels)
    scale = np.asarray([max(float(row["p95"]), lower_bound) for row in rows], dtype=np.float64)
    for row, value in zip(rows, scale):
        row["scale"] = float(value)
        row["scale_rule"] = f"max(p95,{lower_bound:g})"
    return scale, rows


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    return path


def save_scale_vectors(
    path: str | Path,
    *,
    vectors: Mapping[str, Sequence[float] | np.ndarray],
    metadata: Mapping[str, Any],
) -> Path:
    """Save scale vectors and metadata to JSON."""
    payload = {
        "metadata": dict(metadata),
        "vectors": {name: [float(x) for x in np.asarray(vector, dtype=np.float64).reshape(-1)] for name, vector in vectors.items()},
    }
    return write_json(path, payload)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def first_linear_layer(module: nn.Module) -> nn.Linear:
    """Return the first ``nn.Linear`` used for state input."""
    layers = getattr(module, "linearlayers", None)
    if layers is not None and len(layers) > 0 and isinstance(layers[0], nn.Linear):
        return layers[0]
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise ValueError(f"Could not find an nn.Linear first layer in {module.__class__.__name__}.")


def output_linear_layer(module: nn.Module) -> nn.Linear:
    """Return the final ``nn.Linear`` output layer."""
    layers = getattr(module, "linearlayers", None)
    if layers is not None:
        for layer in reversed(layers):
            if isinstance(layer, nn.Linear):
                return layer
    candidates = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if candidates:
        return candidates[-1]
    raise ValueError(f"Could not find an nn.Linear output layer in {module.__class__.__name__}.")


def transform_linear_input_columns(
    layer: nn.Linear,
    scale_vector: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    inverse: bool = False,
) -> None:
    """In-place first-layer column transform.

    If ``x_scaled = x / s``, the function-preserving transform is
    ``W_scaled[:, j] = W_raw[:, j] * s_j``.  ``inverse=True`` divides columns by
    ``s_j`` and returns the parameterization to raw input.
    """
    if layer.weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(layer.weight.shape)}.")
    scale = as_scale_tensor(
        scale_vector,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
        expected_dim=layer.weight.shape[1],
    )
    factor = torch.reciprocal(scale) if inverse else scale
    with torch.no_grad():
        layer.weight.mul_(factor.reshape(1, -1))


def transform_first_layer(
    module: nn.Module,
    scale_vector: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    inverse: bool = False,
) -> nn.Module:
    """In-place transform of a module's first input layer."""
    transform_linear_input_columns(first_linear_layer(module), scale_vector, inverse=inverse)
    return module


def transform_h_output_layer(
    module: nn.Module,
    h_output_scale: float,
    *,
    inverse: bool = False,
) -> nn.Module:
    """In-place H-output transform for ``H_physical(x)=H_hat(x)/s_H``.

    For a raw output layer ``W h + b``, the function-preserving scaled
    representation is ``s_H W h + s_H b``.  ``inverse=True`` divides by
    ``s_H`` and returns the output layer to raw physical units.  ``Y_Net`` has
    no output bias today, but the bias branch is included for checkpoint
    compatibility with earlier or future variants.
    """
    scale = float(h_output_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"h_output_scale must be finite and positive, got {h_output_scale}.")
    factor = 1.0 / scale if inverse else scale
    layer = output_linear_layer(module)
    with torch.no_grad():
        layer.weight.mul_(factor)
        if layer.bias is not None:
            layer.bias.mul_(factor)
    return module


def find_first_weight_key(state_dict: Mapping[str, torch.Tensor], preferred: str = DEFAULT_FIRST_WEIGHT_KEY) -> str:
    """Find the first-layer weight key in a saved H/G state dict."""
    if preferred in state_dict:
        return preferred
    candidates = [key for key, value in state_dict.items() if key.endswith(".0.weight") and torch.is_tensor(value) and value.ndim == 2]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Could not uniquely identify first-layer weight key; candidates={candidates}.")


def find_output_weight_key(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Find the final linear-layer weight key in a saved H state dict."""
    keys = [
        key
        for key, value in state_dict.items()
        if key.endswith(".weight") and torch.is_tensor(value) and value.ndim == 2
    ]
    if not keys:
        raise KeyError("Could not identify output-layer weight key in state dict.")

    def layer_index(key: str) -> tuple[int, str]:
        parts = key.split(".")
        if len(parts) >= 3 and parts[-1] == "weight":
            try:
                return int(parts[-2]), key
            except ValueError:
                return -1, key
        return -1, key

    return max(keys, key=layer_index)


def output_bias_key_for_weight(weight_key: str, state_dict: Mapping[str, torch.Tensor]) -> str | None:
    """Return the matching output bias key when the architecture has one."""
    bias_key = weight_key[:-len(".weight")] + ".bias"
    value = state_dict.get(bias_key)
    return bias_key if torch.is_tensor(value) else None


def transform_state_dict_first_layer(
    state_dict: Mapping[str, torch.Tensor],
    scale_vector: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    inverse: bool = False,
    first_weight_key: str | None = None,
) -> Dict[str, torch.Tensor]:
    """Return a copied state dict with only first-layer input columns transformed."""
    transformed = {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state_dict.items()
    }
    weight_key = first_weight_key or find_first_weight_key(transformed)
    weight = transformed[weight_key]
    if weight.ndim != 2:
        raise ValueError(f"{weight_key} must be 2D, got shape {tuple(weight.shape)}.")
    scale = as_scale_tensor(scale_vector, device=weight.device, dtype=weight.dtype, expected_dim=weight.shape[1])
    factor = torch.reciprocal(scale) if inverse else scale
    transformed[weight_key] = weight * factor.reshape(1, -1)
    return transformed


def transform_state_dict_h_output_layer(
    state_dict: Mapping[str, torch.Tensor],
    h_output_scale: float,
    *,
    inverse: bool = False,
    output_weight_key: str | None = None,
) -> Dict[str, torch.Tensor]:
    """Return a copied H state dict with only the output layer transformed."""
    scale = float(h_output_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"h_output_scale must be finite and positive, got {h_output_scale}.")
    transformed = {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state_dict.items()
    }
    weight_key = output_weight_key or find_output_weight_key(transformed)
    factor = 1.0 / scale if inverse else scale
    transformed[weight_key] = transformed[weight_key] * factor
    bias_key = output_bias_key_for_weight(weight_key, transformed)
    if bias_key is not None:
        transformed[bias_key] = transformed[bias_key] * factor
    return transformed


def transform_state_dict_between_input_scales(
    state_dict: Mapping[str, torch.Tensor],
    source_scale_vector: Sequence[float] | np.ndarray | torch.Tensor,
    target_scale_vector: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    first_weight_key: str | None = None,
) -> Dict[str, torch.Tensor]:
    """Convert a state dict from one input scale representation to another.

    If the source network expects ``x / s_source`` and the target network
    expects ``x / s_target``, the first-layer columns are multiplied by
    ``s_target / s_source``.
    """
    transformed = {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state_dict.items()
    }
    weight_key = first_weight_key or find_first_weight_key(transformed)
    weight = transformed[weight_key]
    source = as_scale_tensor(source_scale_vector, device=weight.device, dtype=weight.dtype, expected_dim=weight.shape[1])
    target = as_scale_tensor(target_scale_vector, device=weight.device, dtype=weight.dtype, expected_dim=weight.shape[1])
    transformed[weight_key] = weight * (target / source).reshape(1, -1)
    return transformed


def transform_h_state_dict_between_reparameterizations(
    state_dict: Mapping[str, torch.Tensor],
    *,
    source_input_scale: Sequence[float] | np.ndarray | torch.Tensor,
    target_input_scale: Sequence[float] | np.ndarray | torch.Tensor,
    source_h_output_scale: float = 1.0,
    target_h_output_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Convert an H checkpoint between input and output representations."""
    transformed = transform_state_dict_between_input_scales(
        state_dict,
        source_input_scale,
        target_input_scale,
    )
    output_factor = float(target_h_output_scale) / float(source_h_output_scale)
    if not np.isclose(output_factor, 1.0):
        transformed = transform_state_dict_h_output_layer(transformed, output_factor)
    return transformed


def transform_g_state_dict_between_reparameterizations(
    state_dict: Mapping[str, torch.Tensor],
    *,
    source_input_scale: Sequence[float] | np.ndarray | torch.Tensor,
    target_input_scale: Sequence[float] | np.ndarray | torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Convert a G checkpoint between input representations; G output is unchanged."""
    return transform_state_dict_between_input_scales(
        state_dict,
        source_input_scale,
        target_input_scale,
    )


def canonical_input_mode(mode: str | None) -> str:
    text = "raw" if mode is None else str(mode).strip().lower()
    aliases = {"none": "raw", "s0": "raw", "raw": "raw", "q0": "q0", "s1": "q0", "uniform_q0": "q0"}
    if text not in aliases:
        raise ValueError(f"input_mode must be raw or q0, got {mode!r}.")
    return aliases[text]


def input_scale_vector_for_mode(
    mode: str | None,
    *,
    state_dim: int,
    q0: float,
) -> np.ndarray:
    """Return the runtime state-input scale vector for a supported mode."""
    canonical = canonical_input_mode(mode)
    if canonical == "raw":
        return np.ones(int(state_dim), dtype=np.float64)
    if canonical == "q0":
        return uniform_q0_scale_vector(int(state_dim), float(q0))
    raise ValueError(f"Unsupported input_mode={mode!r}.")


def reparameterization_metadata(
    *,
    input_mode: str,
    state_dim: int,
    q0: float,
    h_output_scale: float,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build compact metadata describing the runtime function-preserving wrapper."""
    scale = input_scale_vector_for_mode(input_mode, state_dim=state_dim, q0=q0)
    metadata: Dict[str, Any] = {
        "input_mode": canonical_input_mode(input_mode),
        "input_scale_vector": [float(x) for x in scale],
        "q0_input_scale": float(q0),
        "h_output_scale": float(h_output_scale),
        "h_physical_formula": "H_physical(x)=H_hat(x_scaled)/h_output_scale",
        "g_output_scaling": "none",
        "state_coordinate_order": state_coordinate_labels(int(state_dim)),
        "first_layer_transformation_version": TRANSFORMATION_VERSION,
        "h_output_transformation_version": H_OUTPUT_TRANSFORMATION_VERSION,
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


@dataclass(frozen=True)
class PhysicalDHDetails:
    """Canonical physical-derivative audit payload."""

    dh: torch.Tensor
    physical_gradient: torch.Tensor
    scaled_gradient: torch.Tensor
    scaled_states: torch.Tensor
    scale_vector: torch.Tensor
    h_output_scale: float
    state_coordinate_order: list[str]


def _metadata_scale_vector(
    checkpoint_metadata: Mapping[str, Any] | None,
    *,
    state_dim: int,
    scale_vector: Sequence[float] | np.ndarray | torch.Tensor | None,
    q0: float | None = None,
    input_mode: str | None = None,
) -> Sequence[float] | np.ndarray | torch.Tensor:
    if scale_vector is not None:
        return scale_vector
    metadata = dict(checkpoint_metadata or {})
    if "input_scale_vector" in metadata:
        return metadata["input_scale_vector"]
    if input_mode is not None or "input_mode" in metadata:
        q0_value = q0 if q0 is not None else metadata.get("q0_input_scale", 1.0)
        return input_scale_vector_for_mode(
            input_mode if input_mode is not None else metadata.get("input_mode"),
            state_dim=state_dim,
            q0=float(q0_value),
        )
    return np.ones(int(state_dim), dtype=np.float64)


def _metadata_h_output_scale(
    checkpoint_metadata: Mapping[str, Any] | None,
    *,
    h_output_scale: float | None,
) -> float:
    if h_output_scale is not None:
        return float(h_output_scale)
    metadata = dict(checkpoint_metadata or {})
    return float(metadata.get("h_output_scale", 1.0))


def _metadata_coordinate_order(
    checkpoint_metadata: Mapping[str, Any] | None,
    *,
    state_dim: int,
) -> list[str]:
    metadata = dict(checkpoint_metadata or {})
    labels = metadata.get("state_coordinate_order")
    if labels is None:
        return state_coordinate_labels(int(state_dim))
    labels = [str(item) for item in labels]
    if len(labels) != int(state_dim):
        raise ValueError(
            f"state_coordinate_order has length {len(labels)}, expected {int(state_dim)}."
        )
    final_label = labels[-1].strip().lower()
    if final_label not in {"q", "i", "inventory", "central_inventory"} and not final_label.startswith("q"):
        raise ValueError(
            "state_coordinate_order must put the central inventory/Q coordinate last; "
            f"got {labels}."
        )
    return labels


def physical_state_gradient_details(
    y_net: nn.Module,
    physical_states: torch.Tensor,
    scale_vector: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    h_output_scale: float = 1.0,
    create_graph: bool = False,
    retain_graph: bool | None = None,
) -> PhysicalDHDetails:
    """Compute scaled and physical gradients for the canonical H wrapper."""
    if retain_graph is None:
        retain_graph = create_graph
    if physical_states.ndim != 2:
        raise ValueError(f"physical_states must be 2D, got shape {tuple(physical_states.shape)}.")
    scale = as_scale_tensor(
        scale_vector,
        device=physical_states.device,
        dtype=physical_states.dtype,
        expected_dim=physical_states.shape[-1],
    )
    h_scale = float(h_output_scale)
    if not np.isfinite(h_scale) or h_scale <= 0.0:
        raise ValueError(f"h_output_scale must be finite and positive, got {h_output_scale}.")
    scaled_states = (physical_states.detach().clone() / scale).requires_grad_(True)
    value = y_net(scaled_states) / h_scale
    grad_scaled = torch.autograd.grad(
        value.sum(),
        scaled_states,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )[0]
    grad_physical = grad_scaled / scale.reshape(1, -1)
    return PhysicalDHDetails(
        dh=torch.empty(0, device=physical_states.device, dtype=physical_states.dtype),
        physical_gradient=grad_physical,
        scaled_gradient=grad_scaled,
        scaled_states=scaled_states,
        scale_vector=scale,
        h_output_scale=h_scale,
        state_coordinate_order=state_coordinate_labels(int(physical_states.shape[-1])),
    )


def compute_physical_DH(
    H_network: nn.Module,
    physical_state: torch.Tensor,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    *,
    grad_diff_dim: int | None = None,
    scale_vector: Sequence[float] | np.ndarray | torch.Tensor | None = None,
    h_output_scale: float | None = None,
    input_mode: str | None = None,
    q0: float | None = None,
    create_graph: bool = False,
    retain_graph: bool | None = None,
    return_details: bool = False,
) -> torch.Tensor | PhysicalDHDetails:
    """Canonical physical ``D_k H = dH/dZ_k - dH/dQ`` conversion.

    ``physical_state`` is in physical coordinates ordered as
    ``[Z_0, ..., Z_K, Q]``.  The network is evaluated through the recorded
    runtime wrapper: ``x_scaled = x_physical / input_scale_vector`` and
    ``H_physical = H_network(x_scaled) / h_output_scale``.  G outputs are not
    H-scaled by this helper.
    """
    if physical_state.ndim != 2:
        raise ValueError(f"physical_state must be 2D, got shape {tuple(physical_state.shape)}.")
    state_dim = int(physical_state.shape[-1])
    if grad_diff_dim is None:
        grad_diff_dim = state_dim - 1
    grad_diff_dim = int(grad_diff_dim)
    if grad_diff_dim <= 0 or grad_diff_dim >= state_dim:
        raise ValueError(f"grad_diff_dim={grad_diff_dim} is incompatible with state_dim={state_dim}.")

    labels = _metadata_coordinate_order(checkpoint_metadata, state_dim=state_dim)
    resolved_scale = _metadata_scale_vector(
        checkpoint_metadata,
        state_dim=state_dim,
        scale_vector=scale_vector,
        q0=q0,
        input_mode=input_mode,
    )
    resolved_h_scale = _metadata_h_output_scale(
        checkpoint_metadata,
        h_output_scale=h_output_scale,
    )
    details = physical_state_gradient_details(
        H_network,
        physical_state,
        resolved_scale,
        h_output_scale=resolved_h_scale,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    q_grad = details.physical_gradient[:, grad_diff_dim : grad_diff_dim + 1].expand(-1, grad_diff_dim)
    dh = details.physical_gradient[:, :grad_diff_dim] - q_grad
    if return_details:
        return PhysicalDHDetails(
            dh=dh,
            physical_gradient=details.physical_gradient,
            scaled_gradient=details.scaled_gradient,
            scaled_states=details.scaled_states,
            scale_vector=details.scale_vector,
            h_output_scale=details.h_output_scale,
            state_coordinate_order=labels,
        )
    return dh


def checkpoint_transform_metadata(
    *,
    raw_checkpoint: str | Path | None,
    scale_name: str,
    scale_vector: Sequence[float] | np.ndarray | torch.Tensor,
    inverse: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build metadata for an independent transformed checkpoint artifact."""
    vector = np.asarray(scale_vector.detach().cpu().numpy() if torch.is_tensor(scale_vector) else scale_vector, dtype=np.float64).reshape(-1)
    metadata: Dict[str, Any] = {
        "transformation_version": TRANSFORMATION_VERSION,
        "raw_checkpoint": str(raw_checkpoint) if raw_checkpoint is not None else None,
        "scale_name": str(scale_name),
        "scale_vector": vector.tolist(),
        "state_coordinate_order": state_coordinate_labels(vector.shape[0]),
        "formula": "W_scaled[:, j] = W_raw[:, j] * s_j; bias unchanged; later layers unchanged",
        "inverse": bool(inverse),
        "g_output_scaling": "none",
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


def physical_state_gradient(
    y_net: nn.Module,
    physical_states: torch.Tensor,
    scale_vector: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    h_output_scale: float = 1.0,
    create_graph: bool = False,
    retain_graph: bool | None = None,
) -> torch.Tensor:
    """Compute ``dH/dx`` in physical units for a possibly scaled input network."""
    return physical_state_gradient_details(
        y_net,
        physical_states,
        scale_vector,
        h_output_scale=h_output_scale,
        create_graph=create_graph,
        retain_graph=retain_graph,
    ).physical_gradient


def physical_gradient_difference(
    y_net: nn.Module,
    physical_states: torch.Tensor,
    *,
    grad_diff_dim: int,
    scale_vector: Sequence[float] | np.ndarray | torch.Tensor,
    h_output_scale: float = 1.0,
    create_graph: bool = False,
    retain_graph: bool | None = None,
) -> torch.Tensor:
    """Compute physical ``D_k H = dH/dZ_k - dH/dQ``."""
    return compute_physical_DH(
        y_net,
        physical_states,
        grad_diff_dim=grad_diff_dim,
        scale_vector=scale_vector,
        h_output_scale=h_output_scale,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )


def physical_consistency_loss(
    y_net: nn.Module,
    grad_diff_prediction: torch.Tensor,
    physical_states: torch.Tensor,
    *,
    grad_diff_dim: int,
    scale_vector: Sequence[float] | np.ndarray | torch.Tensor,
    h_output_scale: float = 1.0,
    create_graph: bool,
) -> torch.Tensor:
    """Consistency loss comparing G to physical-unit DH under scaled inputs."""
    dh = physical_gradient_difference(
        y_net,
        physical_states,
        grad_diff_dim=grad_diff_dim,
        scale_vector=scale_vector,
        h_output_scale=h_output_scale,
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    return torch.mean((grad_diff_prediction - dh) ** 2)
