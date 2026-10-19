"""Evaluation utilities for trained Deep Splitting runs.

Two evaluation modes are supported:
1. fine-grid: simulate the reflected Brownian approximation on the training grid.
2. discrete-review: map fine-grid policies to a coarser review grid, then realize
   period-level demand in the original discrete inventory system.

Some public names still contain "periodic" for backward compatibility; in the
current paper workflow these refer to the discrete-review evaluator.
"""

import argparse
import glob
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog
from scipy.stats import betabinom, binom, nbinom, norm
from torch import nn

import equation as eqn
import policy_extraction as pext
from Net import GradDiffNet
from benchmarks import inventory_scaled_proportional_policy, myopic_policy, static_policy, status_quo_policy
from input_scaling import apply_state_input_scaling, input_scale_vector_for_mode

try:
    import munch
except ImportError:  # pragma: no cover - fallback for lightweight environments
    munch = None


def _truthy_config(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# Legacy "periodic" argument names are kept so older scripts still run.
DEFAULT_FINE_OUTPUT = "evaluation_results.csv"
DEFAULT_PERIODIC_OUTPUT = "discrete_evaluation_results.csv"
DEFAULT_FINE_PAIRED_OUTPUT = "paired_ds_vs_static_fine_grid.csv"
DEFAULT_PERIODIC_PAIRED_OUTPUT = "paired_ds_vs_static_discrete.csv"
PERIODIC_DEMAND_MODES = (
    "auto",
    "binomial",
    "gaussian_copula_binomial",
    "beta_binomial",
    "negative_binomial",
    "brownian_motion",
    "lognormal",
)


def to_config_namespace(obj):
    if munch is not None:
        return munch.munchify(obj)
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: to_config_namespace(val) for key, val in obj.items()})
    if isinstance(obj, list):
        return [to_config_namespace(val) for val in obj]
    return obj


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Deep Splitting policies on fine-grid and discrete-review settings."
    )
    parser.add_argument("--config_path", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--run_name", type=str, required=True, help="Run name used to load trained models")
    parser.add_argument("--num_paths", type=int, default=10000, help="Number of sample paths per evaluation mode")
    parser.add_argument(
        "--mode",
        choices=["fine_grid", "periodic_review", "discrete", "all"],
        default="fine_grid",
        help="Evaluation mode to run",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_FINE_OUTPUT,
        help="Output CSV for fine-grid mode, or the fine-grid CSV when --mode=all",
    )
    parser.add_argument(
        "--periodic_output_file",
        type=str,
        default=DEFAULT_PERIODIC_OUTPUT,
        help="Output CSV for discrete-review mode",
    )
    parser.add_argument(
        "--review_num_intervals",
        type=int,
        default=None,
        help="Number of discrete review intervals",
    )
    parser.add_argument(
        "--review_dt",
        type=float,
        default=None,
        help="Length of each discrete review interval",
    )
    parser.add_argument(
        "--periodic_demand_mode",
        choices=PERIODIC_DEMAND_MODES,
        default="auto",
        help="Demand model for discrete-review evaluation",
    )
    parser.add_argument(
        "--fine_grid_seed",
        type=int,
        default=0,
        help="Base random seed for fine-grid evaluation when common random numbers are enabled",
    )
    parser.add_argument(
        "--periodic_seed",
        type=int,
        default=0,
        help="Base random seed for discrete-review evaluation",
    )
    parser.add_argument(
        "--paired_output_file",
        type=str,
        default=DEFAULT_FINE_PAIRED_OUTPUT,
        help="Output CSV for the fine-grid paired DS-vs-Static comparison",
    )
    parser.add_argument(
        "--periodic_paired_output_file",
        type=str,
        default=DEFAULT_PERIODIC_PAIRED_OUTPUT,
        help="Output CSV for the discrete-review paired DS-vs-Static comparison",
    )
    parser.set_defaults(use_common_random_numbers=True)
    parser.add_argument(
        "--use_common_random_numbers",
        dest="use_common_random_numbers",
        action="store_true",
        help="Reuse identical exogenous randomness across all policies",
    )
    parser.add_argument(
        "--no_common_random_numbers",
        dest="use_common_random_numbers",
        action="store_false",
        help="Disable common random numbers and preserve policy-specific simulation randomness",
    )
    parser.add_argument(
        "--logs_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint directory",
    )
    parser.add_argument(
        "--num_time_interval",
        type=int,
        default=None,
        help="Override num_time_interval in both eqn_config and net_config",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Override num_layers in net_config",
    )
    parser.add_argument(
        "--fluid_plan_path",
        type=str,
        default=None,
        help="Optional allocation_plan.npz from the deterministic fluid LP/QCP solver.",
    )
    parser.add_argument(
        "--skip_fluid_benchmark",
        action="store_true",
        help="Omit the Fluid_Optimal benchmark policy even when lambda is positive.",
    )
    parser.add_argument(
        "--policy_extraction_modes",
        type=str,
        default=None,
        help="Comma-separated DS extraction modes for discrete-review evaluation, e.g. E0,E1,E2,E3.",
    )
    return parser.parse_args()


def load_config_from_path(config_path, num_time_interval=None, num_layers=None):
    with open(config_path) as json_data_file:
        config = json.load(json_data_file)

    # Prefer run-local params/ when evaluating materialized output directories.
    sibling_params_dir = os.path.join(os.path.dirname(os.path.abspath(config_path)), "params")
    if os.path.isdir(sibling_params_dir):
        config["eqn_config"]["config_dir"] = sibling_params_dir

    if num_time_interval is not None:
        config["eqn_config"]["num_time_interval"] = num_time_interval
        config["net_config"]["num_time_interval"] = num_time_interval
        print(f"Overriding num_time_interval to {num_time_interval}")

    if num_layers is not None:
        config["net_config"]["num_layers"] = num_layers
        print(f"Overriding num_layers to {num_layers}")

    return to_config_namespace(config)


def load_config(args):
    return load_config_from_path(
        args.config_path,
        num_time_interval=args.num_time_interval,
        num_layers=args.num_layers,
    )


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_bsde_to_device(bsde, device):
    bsde.mu = bsde.mu.to(device)
    bsde.Gamma = bsde.Gamma.to(device)
    bsde.p = bsde.p.to(device)
    bsde.c = bsde.c.to(device)
    bsde.c_tilde = bsde.c_tilde.to(device)
    bsde.u = bsde.u.to(device)
    bsde.r = bsde.r.to(device)
    bsde.d = bsde.d.to(device)
    bsde.distance = bsde.distance.to(device)
    bsde.exposure = bsde.exposure.to(device)
    bsde.theta = bsde.theta.to(device)
    bsde.h = bsde.h.to(device)
    bsde.eta = bsde.eta.to(device)
    if hasattr(bsde, "eta_schedule_raw"):
        bsde.eta_schedule_raw = bsde.eta_schedule_raw.to(device)
    if hasattr(bsde, "eta_schedule"):
        bsde.eta_schedule = bsde.eta_schedule.to(device)
    return bsde


def attach_cost_components(config, bsde, device):
    """Backward-compatible no-op; LiberiaHJB now loads cost components."""
    for attr in ("c", "c_tilde", "u", "r", "d", "distance", "exposure", "theta"):
        if hasattr(bsde, attr):
            setattr(bsde, attr, getattr(bsde, attr).to(device))
    return bsde


def resolve_logs_path(run_name, preferred_path=None, project_root=None, config_directory=None):
    """Find checkpoint logs, allowing explicit paths and run-local output folders."""
    candidates = []
    if preferred_path:
        candidates.append(preferred_path)
    if config_directory:
        candidates.append(os.path.join(config_directory, "logs"))

    if project_root:
        candidates.extend(
            [
                os.path.join(project_root, "output", run_name, "logs"),
                os.path.join(project_root, f"logs_{run_name}"),
            ]
        )

    candidates.append(f"logs_{run_name}")

    seen = set()
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            if os.path.isdir(path):
                return path

    return candidates[0] if candidates else f"logs_{run_name}"


def load_z_networks(config, device, logs_path):
    """Load one gradient network per training interval."""
    num_time_interval = config.eqn_config.num_time_interval
    state_dim = config.eqn_config.dim + 1
    grad_diff_dim = config.eqn_config.dim
    g_output_bias = _truthy_config(getattr(config.net_config, "g_output_bias", False))
    z_net_list = nn.ModuleList(
        [
            GradDiffNet(
                state_dim,
                grad_diff_dim,
                config.net_config.num_neurons,
                config.net_config.num_layers,
                config.net_config.activation,
                config.net_config.slope,
                output_bias=g_output_bias,
            ).to(device)
            for _ in range(num_time_interval)
        ]
    )

    loaded = 0
    for n in range(num_time_interval):
        z_net_path = os.path.join(logs_path, f"z_network{n}.pth")
        if os.path.exists(z_net_path):
            z_net_list[n].load_state_dict(torch.load(z_net_path, map_location=device))
            z_net_list[n].eval()
            loaded += 1
        else:
            print(f"Warning: Model file {z_net_path} not found. Skipping interval {n}.")

    print(f"Loaded {loaded}/{num_time_interval} z_network checkpoints from {logs_path}")
    return z_net_list


def cap_quantity_to_inventory_numpy(q, inventory, abs_tol=1e-8, rel_tol=1e-6, max_quantity=None):
    """Project period shipment quantities onto remaining inventory.

    Used after aggregating fine-grid rate decisions into one discrete shipment.
    """
    remaining = np.clip(np.asarray(inventory).reshape(-1), 0.0, None)
    tol = np.maximum(abs_tol, rel_tol * np.maximum(remaining, 1.0))
    q_projected = np.clip(np.asarray(q), 0.0, None)
    if max_quantity is not None:
        max_arr = np.asarray(max_quantity, dtype=q_projected.dtype)
        if max_arr.ndim == 1:
            max_arr = max_arr.reshape(1, -1)
        q_projected = np.minimum(q_projected, max_arr)

    inactive = remaining <= tol
    if np.any(inactive):
        q_projected[inactive, :] = 0.0

    q_sum = q_projected.sum(axis=1)
    needs_scale = q_sum > (remaining + tol)
    if np.any(needs_scale):
        safe_denom = np.maximum(q_sum[needs_scale], 1e-12)
        q_projected[needs_scale, :] = q_projected[needs_scale, :] * (
            remaining[needs_scale] / safe_denom
        )[:, None]

    return q_projected, needs_scale, inactive


def cap_rate_to_inventory_torch(
    a_rate,
    inventory,
    dt,
    abs_tol=1e-8,
    rel_tol=1e-6,
    eta_rate=None,
    return_diagnostics=False,
):
    """Project fine-grid allocation rates onto the one-step feasible set."""
    projected, tol, diagnostics = eqn.project_rate_to_inventory_torch(
        a_rate,
        inventory,
        dt,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        eta_rate=eta_rate,
        return_diagnostics=True,
    )

    if inventory.dim() == 2:
        inventory = inventory.squeeze(1)
    remaining = torch.clamp(inventory, min=0.0)
    raw = torch.clamp(a_rate, min=0.0)
    if eta_rate is not None:
        eta_rate = torch.as_tensor(eta_rate, device=raw.device, dtype=raw.dtype)
        if eta_rate.dim() == 0:
            eta_rate = eta_rate.expand(raw.shape[1])
        if eta_rate.dim() == 1:
            eta_rate = eta_rate.unsqueeze(0)
        raw = torch.minimum(raw, eta_rate.expand_as(raw))
    q_sum = raw.sum(dim=1) * dt
    needs_scale = q_sum > (remaining + tol)
    inactive = remaining <= tol
    if return_diagnostics:
        diagnostics = dict(diagnostics)
        diagnostics["projection_mask"] = needs_scale
        diagnostics["zero_inventory_mask"] = inactive
        return projected, needs_scale, inactive, diagnostics
    return projected, needs_scale, inactive


def cholesky_with_fallback_torch(Gamma_t):
    try:
        return torch.linalg.cholesky(Gamma_t), False
    except Exception:
        diag_gamma = torch.diag(Gamma_t)
        return torch.diag(torch.sqrt(torch.clamp(diag_gamma, min=0.0))), True


def precompute_fine_grid_brownian_increments(bsde, num_paths, device, seed=0):
    """Precompute path-indexed Brownian increments shared across policies."""
    dim = bsde.dim
    N = bsde.num_time_interval
    dt_sqrt = np.sqrt(bsde.delta_t)
    dtype = bsde.mu.dtype
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    w_increments = torch.empty((N, num_paths, dim), device=device, dtype=dtype)

    for n in range(N):
        xi = torch.randn((num_paths, dim), generator=generator, dtype=dtype).to(device)
        Gamma_t = bsde.Gamma[n, :, :].to(device)
        L_chol, _ = cholesky_with_fallback_torch(Gamma_t)
        w_increments[n] = (xi @ L_chol.T) * dt_sqrt

    return w_increments


def precompute_periodic_demands(
    mu_review,
    gamma_review,
    review_dt,
    num_paths,
    demand_mode="auto",
    seed=0,
    eps=1e-8,
    exposure=None,
):
    """Precompute path-indexed discrete demand draws shared across policies."""
    rng = np.random.default_rng(seed)
    num_review_intervals, dim = mu_review.shape
    demand_draws = np.zeros((num_review_intervals, num_paths, dim), dtype=np.float32)

    for n in range(num_review_intervals):
        demand_draws[n] = sample_periodic_demand(
            mu_review[n],
            gamma_review[n],
            review_dt,
            num_paths,
            rng,
            mode=demand_mode,
            eps=eps,
            exposure=exposure,
        ).astype(np.float32)

    return demand_draws


def create_deep_splitting_policy(z_net_list, bsde, device, config=None, extraction_mode=pext.FROZEN_AVERAGE):
    """Create a policy wrapper for the Deep Splitting model."""
    extraction_mode = pext.canonical_extraction_mode(extraction_mode)
    net_config = getattr(config, "net_config", SimpleNamespace()) if config is not None else SimpleNamespace()
    input_mode = getattr(net_config, "input_mode", "raw")
    q0_input_scale = float(getattr(net_config, "q0_input_scale", getattr(bsde, "I0", 142.0)))
    state_dim = int(getattr(bsde, "dim")) + 1
    input_scale = torch.tensor(
        input_scale_vector_for_mode(input_mode, state_dim=state_dim, q0=q0_input_scale),
        device=device,
        dtype=torch.float32,
    )

    def policy_fn(t, Z, I):
        batch_size = Z.shape[0]
        dim = Z.shape[1]

        if I.dim() == 0:
            I = I.unsqueeze(0)
        if I.dim() == 1 and I.shape[0] != batch_size:
            I = I.expand(batch_size)

        I_expanded = I.unsqueeze(1) if I.dim() == 1 else I
        Y = torch.cat([Z, I_expanded], dim=1)

        n = min(t, len(z_net_list) - 1)
        with torch.no_grad():
            grad_diff = z_net_list[n](apply_state_input_scaling(Y, input_scale))

        mu_t = bsde.mu[min(t, bsde.mu.shape[0] - 1), :].to(device)
        eta_t = (
            bsde.eta_at_fine_index(t).to(device=device, dtype=grad_diff.dtype)
            if hasattr(bsde, "eta_at_fine_index")
            else bsde.eta.to(device=device, dtype=grad_diff.dtype)
        )
        # DS uses the HJB closed-form rule and applies a fine-step inventory cap.
        return eqn.build_feasible_ds_allocation(
            grad_diff,
            I,
            mu_t,
            eta_t,
            bsde.lambda_param,
            bsde.alpha,
            bsde.delta_t,
        )

    extractor = pext.DeepSplittingPolicyExtractor(z_net_list, bsde, device, config=config)

    def extract_weekly_action(period_index, Z_start, Q_start, fine_per_review, review_dt):
        return extractor.extract_weekly_action(
            extraction_mode,
            period_index,
            Z_start,
            Q_start,
            fine_per_review,
            review_dt,
        )

    policy_fn.policy_kind = "deep_splitting"
    policy_fn.extraction_mode = extraction_mode
    policy_fn.extraction_label = pext.MODE_LABELS[extraction_mode]
    policy_fn.extract_weekly_action = extract_weekly_action
    policy_fn.extraction_engine = extractor
    return policy_fn


def create_static_policy_wrapper(bsde, device):
    """Static demand-tracking benchmark; simulator enforces final feasibility."""
    def policy_fn(t, Z, I):
        eta_profile = (
            bsde.eta_schedule.to(device)
            if hasattr(bsde, "eta_schedule")
            else bsde.eta.to(device)
        )
        return static_policy(t, Z, I, bsde.mu.to(device), eta_profile)

    return policy_fn


def create_scaled_static_policy_wrapper(bsde, device):
    """Scaled static benchmark from the paper."""
    q0 = float(bsde.I0)

    def policy_fn(t, Z, I):
        return inventory_scaled_proportional_policy(
            t,
            Z,
            I,
            bsde.mu.to(device),
            bsde.eta_schedule.to(device) if hasattr(bsde, "eta_schedule") else bsde.eta.to(device),
            float(bsde.delta_t),
            initial_inventory=q0,
        )

    return policy_fn


def create_status_quo_policy_wrapper(bsde, device):
    """Open-loop quota benchmark, evaluated at the fine-grid index."""
    I0 = float(bsde.I0)
    K = int(bsde.K)
    dt = float(bsde.delta_t)

    def policy_fn(t, Z, I):
        eta_profile = (
            bsde.eta_schedule.to(device)
            if hasattr(bsde, "eta_schedule")
            else bsde.eta.to(device)
        )
        return status_quo_policy(t, Z, I, eta_profile, I0, K, dt)

    return policy_fn


def _solve_fluid_lp_plan(bsde):
    mu_raw = bsde.mu_raw.detach().cpu().numpy().astype(np.float64)
    gamma_raw = bsde.Gamma_raw.detach().cpu().numpy().astype(np.float64)
    del gamma_raw
    periods, dim = mu_raw.shape
    h = float(bsde.total_time) / float(periods)
    demand = mu_raw * h
    if hasattr(bsde, "eta_for_review_grid"):
        eta_review = bsde.eta_for_review_grid(periods).detach().cpu().numpy().astype(np.float64)
    else:
        eta_review = np.repeat(
            bsde.eta.detach().cpu().numpy().reshape(1, -1).astype(np.float64),
            periods,
            axis=0,
        )
    c = bsde.c_tilde.detach().cpu().numpy().astype(np.float64)
    p = bsde.p.detach().cpu().numpy().astype(np.float64)
    q0 = float(bsde.I0)

    num_a = periods * dim
    num_q = periods
    num_z = periods * dim
    num_l = periods * dim
    off_a = 0
    off_q = off_a + num_a
    off_z = off_q + num_q
    off_l = off_z + num_z
    total_vars = off_l + num_l

    def a_idx(n, k):
        return off_a + n * dim + k

    def q_idx(n):
        return off_q + n

    def z_idx(n, k):
        return off_z + n * dim + k

    def l_idx(n, k):
        return off_l + n * dim + k

    objective = np.zeros(total_vars, dtype=np.float64)
    for n in range(periods):
        for k in range(dim):
            objective[l_idx(n, k)] = c[k]

    A_eq = []
    b_eq = []
    for n in range(periods):
        row = np.zeros(total_vars, dtype=np.float64)
        row[q_idx(n)] = 1.0
        if n > 0:
            row[q_idx(n - 1)] = -1.0
            rhs = 0.0
        else:
            rhs = q0
        for k in range(dim):
            row[a_idx(n, k)] = 1.0
        A_eq.append(row)
        b_eq.append(rhs)

        for k in range(1, dim):
            row = np.zeros(total_vars, dtype=np.float64)
            row[z_idx(n, k)] = 1.0
            if n > 0:
                row[z_idx(n - 1, k)] = -1.0
            row[a_idx(n, k)] = -1.0
            row[l_idx(n, k)] = -1.0
            A_eq.append(row)
            b_eq.append(-float(demand[n, k]))

        row = np.zeros(total_vars, dtype=np.float64)
        row[z_idx(n, 0)] = 1.0
        if n > 0:
            row[z_idx(n - 1, 0)] = -1.0
        row[a_idx(n, 0)] = -1.0
        row[l_idx(n, 0)] = -1.0
        for k in range(1, dim):
            row[l_idx(n, k)] = p[k - 1]
        A_eq.append(row)
        b_eq.append(-float(demand[n, 0]))

    bounds = []
    for n in range(periods):
        for k in range(dim):
            bounds.append((0.0, float(eta_review[n, k] * h)))
    bounds.extend((0.0, None) for _ in range(num_q + num_z + num_l))
    result = linprog(
        objective,
        A_eq=np.asarray(A_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Fluid LP failed: {result.message}")
    allocation = result.x[off_a:off_q].reshape(periods, dim)
    return allocation


def _unique_existing_paths(paths):
    seen = set()
    out = []
    for path in paths:
        if not path:
            continue
        norm = os.path.abspath(os.path.expanduser(str(path)))
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(norm):
            out.append(norm)
    return out


def find_fluid_plan_path(bsde, logs_path=None, config_path=None, explicit_path=None):
    """Find a deterministic fluid allocation artifact for paper benchmark evaluation."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    for attr in ("fluid_plan_path", "fluid_plan_file"):
        value = getattr(bsde, attr, None)
        if value:
            candidates.append(value)

    run_dir = None
    if logs_path:
        logs_abs = os.path.abspath(os.path.expanduser(str(logs_path)))
        if os.path.basename(logs_abs) == "logs":
            run_dir = os.path.dirname(logs_abs)
        else:
            run_dir = logs_abs
        candidates.extend(
            [
                os.path.join(run_dir, "allocation_plan.npz"),
                os.path.join(run_dir, "fluid_optimal", "allocation_plan.npz"),
                os.path.join(run_dir, "mean_demand_lp", "allocation_plan.npz"),
                os.path.join(run_dir, "mean_demand_qcp", "allocation_plan.npz"),
                os.path.join(run_dir, "md_lp", "allocation_plan.npz"),
            ]
        )
        candidates.extend(glob.glob(os.path.join(run_dir, "**", "allocation_plan.npz"), recursive=True))

    if config_path:
        config_dir = os.path.dirname(os.path.abspath(os.path.expanduser(str(config_path))))
        candidates.extend(
            [
                os.path.join(config_dir, "allocation_plan.npz"),
                os.path.join(config_dir, "fluid_optimal", "allocation_plan.npz"),
            ]
        )

    existing = _unique_existing_paths(candidates)
    return existing[0] if existing else None


def _fluid_npz_scalar(data, key, path):
    if key not in data:
        raise ValueError(f"Fluid plan {path} is missing required metadata field {key!r}.")
    arr = np.asarray(data[key])
    if arr.size != 1:
        raise ValueError(f"Fluid plan {path} metadata field {key!r} must be scalar, got shape {arr.shape}.")
    return arr.reshape(-1)[0]


def _fluid_npz_string(data, key, path):
    return str(_fluid_npz_scalar(data, key, path))


def _fluid_npz_float(data, key, path):
    return float(_fluid_npz_scalar(data, key, path))


def _fluid_metadata_array(data, key, path):
    if key not in data:
        raise ValueError(f"Fluid plan {path} is missing required metadata field {key!r}.")
    return np.asarray(data[key], dtype=np.float64)


def _tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _assert_fluid_close(path, name, actual, expected, atol=1e-7, rtol=0.0):
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    if actual_arr.shape != expected_arr.shape or not np.allclose(actual_arr, expected_arr, rtol=rtol, atol=atol):
        raise ValueError(
            f"Fluid plan {path} {name} mismatch: got {actual_arr.tolist()}, "
            f"expected {expected_arr.tolist()}."
        )


def validate_fluid_plan_metadata(path, data, bsde, plan, review_dt):
    """Strict compatibility check for deterministic fluid allocation artifacts."""
    fairness_spec = _fluid_npz_string(data, "fairness_spec", path)
    if fairness_spec != eqn.FAIRNESS_SPEC:
        raise ValueError(
            f"Fluid plan {path} fairness_spec={fairness_spec!r}; expected {eqn.FAIRNESS_SPEC!r}."
        )
    if bool(_fluid_npz_scalar(data, "fairness_includes_hf", path)) is not True:
        raise ValueError(f"Fluid plan {path} fairness_includes_hf must be true.")
    if bool(_fluid_npz_scalar(data, "fairness_period_h_factor", path)) is not False:
        raise ValueError(f"Fluid plan {path} fairness_period_h_factor must be false.")

    _assert_fluid_close(path, "lambda_param", _fluid_npz_float(data, "lambda_param", path), float(bsde.lambda_param), atol=1e-12)
    _assert_fluid_close(path, "alpha", _fluid_npz_float(data, "alpha", path), float(bsde.alpha), atol=1e-12)
    _assert_fluid_close(path, "p", _fluid_metadata_array(data, "p", path), _tensor_to_numpy(bsde.p), atol=1e-7)
    _assert_fluid_close(path, "c", _fluid_metadata_array(data, "c", path), _tensor_to_numpy(bsde.c_tilde), atol=1e-6)
    _assert_fluid_close(path, "q0", _fluid_npz_float(data, "q0", path), float(bsde.I0), atol=1e-8)
    _assert_fluid_close(path, "chi_eta", _fluid_npz_float(data, "chi_eta", path), float(bsde.chi_eta), atol=1e-12)
    _assert_fluid_close(path, "s_eta", _fluid_npz_float(data, "s_eta", path), float(bsde.s_eta), atol=1e-12)

    artifact_eta_mode = _fluid_npz_string(data, "eta_mode", path)
    expected_eta_mode = str(getattr(bsde, "eta_mode", getattr(bsde, "eta_profile", "")))
    if artifact_eta_mode != expected_eta_mode:
        raise ValueError(
            f"Fluid plan {path} eta_mode={artifact_eta_mode!r}; expected {expected_eta_mode!r}."
        )
    if "eta_profile" in data:
        artifact_eta_profile = _fluid_npz_string(data, "eta_profile", path)
        if artifact_eta_profile != expected_eta_mode:
            raise ValueError(
                f"Fluid plan {path} eta_profile={artifact_eta_profile!r}; expected {expected_eta_mode!r}."
            )

    eta_review = _fluid_metadata_array(data, "eta_review", path)
    expected_eta_review = _tensor_to_numpy(bsde.eta_for_review_grid(plan.shape[0]))
    _assert_fluid_close(path, "eta_review", eta_review, expected_eta_review, atol=2e-6, rtol=1e-7)
    eta_schedule_raw = _fluid_metadata_array(data, "eta_schedule_raw", path)
    _assert_fluid_close(
        path,
        "eta_schedule_raw",
        eta_schedule_raw,
        _tensor_to_numpy(bsde.eta_schedule_raw),
        atol=2e-6,
        rtol=1e-7,
    )

    calibration_version = _fluid_npz_string(data, "calibration_version", path)
    expected_calibration_version = getattr(bsde, "calibration_version", None)
    if expected_calibration_version is not None and calibration_version != str(expected_calibration_version):
        raise ValueError(
            f"Fluid plan {path} calibration_version={calibration_version!r}; "
            f"expected {expected_calibration_version!r}."
        )

    artifact_review_dt = _fluid_npz_float(data, "review_dt", path)
    _assert_fluid_close(path, "review_dt", artifact_review_dt, review_dt, atol=1e-12)


def load_fluid_allocation_plan(path, bsde):
    """Load allocation quantities from allocation_plan.npz and validate metadata."""
    data = np.load(path)
    if "allocation_plan" not in data:
        raise ValueError(f"Fluid plan {path} does not contain allocation_plan.")
    plan = np.asarray(data["allocation_plan"], dtype=np.float64)
    if plan.ndim != 2:
        raise ValueError(f"Fluid plan allocation must be 2D, got shape {plan.shape}.")
    if plan.shape[1] != int(bsde.dim):
        raise ValueError(f"Fluid plan dim mismatch: got {plan.shape[1]}, expected {int(bsde.dim)}.")
    if np.any(plan < -1e-8):
        raise ValueError(f"Fluid plan {path} contains negative allocations.")

    if "review_dt" in data:
        review_dt = float(np.asarray(data["review_dt"]).reshape(-1)[0])
    else:
        review_dt = float(bsde.total_time) / float(plan.shape[0])
    if review_dt <= 0.0:
        raise ValueError(f"Fluid plan {path} has nonpositive review_dt={review_dt}.")
    validate_fluid_plan_metadata(path, data, bsde, plan, review_dt)
    return np.clip(plan, 0.0, None), review_dt


def create_fluid_optimal_policy_wrapper(bsde, device, logs_path=None, config_path=None, fluid_plan_path=None):
    """Open-loop deterministic fluid benchmark."""
    resolved_plan_path = find_fluid_plan_path(
        bsde,
        logs_path=logs_path,
        config_path=config_path,
        explicit_path=fluid_plan_path,
    )
    if resolved_plan_path is not None:
        plan, review_dt = load_fluid_allocation_plan(resolved_plan_path, bsde)
        source = resolved_plan_path
    elif float(bsde.lambda_param) != 0.0:
        message = (
            "Fluid_Optimal for lambda>0 requires a pre-solved QCP allocation_plan.npz. "
            "Run scripts/solve_fangonda_fluid_gurobi_notorch.py for this scenario, or pass --fluid_plan_path."
        )

        def missing_fluid_policy(t, Z, I):
            del t, Z, I
            raise RuntimeError(message)

        missing_fluid_policy.fluid_source = "missing_qcp_plan"
        return missing_fluid_policy
    else:
        plan = _solve_fluid_lp_plan(bsde)
        review_dt = float(bsde.total_time) / float(plan.shape[0])
        source = "lp"

    review_intervals = plan.shape[0]
    plan_t = torch.tensor(plan / review_dt, device=device, dtype=bsde.mu.dtype)

    def policy_fn(t, Z, I):
        del I
        current_time = float(int(t)) * float(bsde.delta_t)
        review_idx = min(int(current_time // review_dt), review_intervals - 1)
        return plan_t[review_idx].unsqueeze(0).expand(Z.shape[0], -1)

    policy_fn.fluid_source = source
    return policy_fn


def create_myopic_policy_wrapper(bsde, device):
    """Reactive benchmark that ships to locations at or below the threshold."""
    eps = getattr(bsde, "myopic_epsilon", 0.0)

    def policy_fn(t, Z, I):
        eta_t = (
            bsde.eta_at_fine_index(t).to(device)
            if hasattr(bsde, "eta_at_fine_index")
            else bsde.eta.to(device)
        )
        return myopic_policy(Z, I, eta_t, eps=eps)

    return policy_fn


def create_policy_suite(
    bsde,
    device,
    z_net_list,
    logs_path=None,
    config_path=None,
    fluid_plan_path=None,
    skip_fluid_benchmark=False,
    config=None,
    ds_extraction_modes=None,
):
    """Return policies in reporting order; DS must remain first for gap reporting."""
    policies = {}
    if ds_extraction_modes is None:
        policies["Deep_Splitting"] = create_deep_splitting_policy(
            z_net_list,
            bsde,
            device,
            config=config,
            extraction_mode=pext.FROZEN_AVERAGE,
        )
    else:
        for mode in pext.parse_extraction_modes(ds_extraction_modes):
            policies[pext.extraction_policy_name(mode)] = create_deep_splitting_policy(
                z_net_list,
                bsde,
                device,
                config=config,
                extraction_mode=mode,
            )
    policies.update(
        {
            "Status_Quo": create_status_quo_policy_wrapper(bsde, device),
            "Static": create_static_policy_wrapper(bsde, device),
            "Scaled_Static": create_scaled_static_policy_wrapper(bsde, device),
            "Myopic": create_myopic_policy_wrapper(bsde, device),
        }
    )
    if not skip_fluid_benchmark:
        policies["Fluid_Optimal"] = create_fluid_optimal_policy_wrapper(
            bsde,
            device,
            logs_path=logs_path,
            config_path=config_path,
            fluid_plan_path=fluid_plan_path,
        )
    return policies


def prepare_evaluation_context(
    config_path,
    run_name,
    logs_path=None,
    num_time_interval=None,
    num_layers=None,
    device=None,
    fluid_plan_path=None,
    skip_fluid_benchmark=False,
    ds_extraction_modes=None,
):
    """Load config, equation object, checkpoints, and policy wrappers."""
    config = load_config_from_path(
        config_path,
        num_time_interval=num_time_interval,
        num_layers=num_layers,
    )
    fairness_spec = getattr(config.eqn_config, "fairness_spec", None)
    if fairness_spec is not None and fairness_spec != eqn.FAIRNESS_SPEC:
        raise ValueError(
            f"Config {config_path} has fairness_spec={fairness_spec!r}; expected {eqn.FAIRNESS_SPEC!r}."
        )
    if float(getattr(config.eqn_config, "lambda_param", 0.0)) > 0.0 and fairness_spec is None:
        raise ValueError(
            f"Config {config_path} is lambda-positive but has no fairness_spec metadata. "
            "Old lambda-positive checkpoints/results are incompatible with demand_weighted_alpha_fair_v1."
        )
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    device = device or get_device()
    move_bsde_to_device(bsde, device)
    attach_cost_components(config, bsde, device)

    project_root = os.path.dirname(os.path.abspath(config_path))
    logs_path = resolve_logs_path(
        run_name,
        preferred_path=logs_path,
        project_root=project_root,
        config_directory=os.path.dirname(os.path.abspath(config_path)),
    )
    bsde.config_path = os.path.abspath(config_path)
    bsde.logs_path = logs_path
    if fluid_plan_path:
        bsde.fluid_plan_path = fluid_plan_path
    z_net_list = load_z_networks(config, device, logs_path)
    policies = create_policy_suite(
        bsde,
        device,
        z_net_list,
        logs_path=logs_path,
        config_path=config_path,
        fluid_plan_path=fluid_plan_path,
        skip_fluid_benchmark=skip_fluid_benchmark,
        config=config,
        ds_extraction_modes=ds_extraction_modes,
    )

    return {
        "config": config,
        "bsde": bsde,
        "device": device,
        "logs_path": logs_path,
        "z_net_list": z_net_list,
        "policies": policies,
    }


def simulate_trajectories(
    policy_fn,
    bsde,
    num_paths,
    device,
    return_history=False,
    shared_w_increments=None,
    random_seed=None,
):
    """Simulate the fine-grid reflected Brownian approximation."""
    dim = bsde.dim
    N = bsde.num_time_interval
    dt = bsde.delta_t
    I0 = bsde.I0

    Z = torch.zeros(num_paths, dim, device=device)
    I = torch.ones(num_paths, 1, device=device) * I0

    total_stockout_cost = torch.zeros(num_paths, device=device)
    total_travel_cost = torch.zeros(num_paths, device=device)
    total_fairness_utility = torch.zeros(num_paths, device=device)
    total_weighted_fairness_utility = torch.zeros(num_paths, device=device)

    if return_history:
        z_hist = torch.zeros(N + 1, num_paths, dim, device=device)
        i_hist = torch.zeros(N + 1, num_paths, device=device)
        a_hist = torch.zeros(N, num_paths, dim, device=device)
        a_raw_hist = torch.zeros(N, num_paths, dim, device=device)
        d_hist = torch.zeros(N, num_paths, dim, device=device)
        l_hist = torch.zeros(N, num_paths, dim, device=device)
        mu_hist = torch.zeros(N, dim, device=device)
        eta_hist = torch.zeros(N, dim, device=device)
        inventory_cap_active_hist = torch.zeros(N, num_paths, device=device, dtype=torch.bool)
        zero_inventory_hist = torch.zeros(N, num_paths, device=device, dtype=torch.bool)
        projection_scale_hist = torch.ones(N, num_paths, device=device)

        z_hist[0] = Z
        i_hist[0] = I.squeeze(1)

    c = bsde.c.to(device)
    d = bsde.d.to(device)
    u_cost = bsde.u.to(device)
    h = bsde.h.to(device)
    local_generator = None
    if shared_w_increments is None and random_seed is not None:
        local_generator = torch.Generator(device="cpu")
        local_generator.manual_seed(int(random_seed))

    for n in range(N):
        mu_t = bsde.mu[n, :].to(device)
        eta_t = (
            bsde.eta_at_fine_index(n).to(device=device, dtype=Z.dtype)
            if hasattr(bsde, "eta_at_fine_index")
            else bsde.eta.to(device=device, dtype=Z.dtype)
        )
        Gamma_t = bsde.Gamma[n, :, :].to(device)
        p = bsde.p.to(device)

        if return_history:
            mu_hist[n] = mu_t
            eta_hist[n] = eta_t

        I_squeezed = I.squeeze(1) if I.dim() == 2 else I
        a_raw = policy_fn(n, Z, I_squeezed)
        # Fine-grid simulation applies a one-step cap to every policy.
        if return_history:
            a, cap_active, zero_inventory_mask, projection_diag = cap_rate_to_inventory_torch(
                a_raw,
                I_squeezed,
                dt,
                eta_rate=eta_t,
                return_diagnostics=True,
            )
        else:
            a, cap_active, zero_inventory_mask = cap_rate_to_inventory_torch(
                a_raw,
                I_squeezed,
                dt,
                eta_rate=eta_t,
            )

        # Common random numbers are optional but preferred for policy comparisons.
        if shared_w_increments is not None:
            W_increment = shared_w_increments[n].to(device=device, dtype=Z.dtype)
        else:
            if local_generator is not None:
                xi = torch.randn((num_paths, dim), generator=local_generator, dtype=Z.dtype).to(device)
            else:
                xi = torch.randn(num_paths, dim, device=device)
            L_chol, _ = cholesky_with_fallback_torch(Gamma_t)
            W_increment = (xi @ L_chol.T) * np.sqrt(dt)
        demand_increment = mu_t.unsqueeze(0) * dt - W_increment
        Z_new, I_new_flat, Delta_L = eqn.step_transition_torch(
            Z,
            I_squeezed,
            a,
            W_increment,
            dt,
            mu_t,
            p,
        )
        I_new = I_new_flat.unsqueeze(1)
        Delta_L_CHAs = Delta_L[:, 1:]

        stockout_cost_step = (
            u_cost * Delta_L[:, 0]
            + (u_cost * (1.0 - p).unsqueeze(0) * Delta_L_CHAs).sum(dim=1)
        )
        travel_cost_step = (
            d[1:].unsqueeze(0) * p.unsqueeze(0) * Delta_L_CHAs
        ).sum(dim=1)
        shortage_cost_step = (c.unsqueeze(0) * Delta_L).sum(dim=1)
        if not torch.allclose(stockout_cost_step + travel_cost_step, shortage_cost_step, rtol=1e-5, atol=1e-5):
            raise AssertionError("Untreated plus referral cost does not equal c · L.")
        total_stockout_cost += stockout_cost_step
        total_travel_cost += travel_cost_step

        fairness_utility_step = eqn.alpha_fair_utility_torch(a, mu_t, bsde.alpha)
        weighted_fairness_utility_step = bsde.lambda_param * fairness_utility_step
        # Fine-grid fairness is an instantaneous utility integrated over dt.
        total_fairness_utility += fairness_utility_step * dt
        total_weighted_fairness_utility += weighted_fairness_utility_step * dt

        Z = Z_new
        I = I_new

        if return_history:
            a_raw_hist[n] = a_raw
            a_hist[n] = a
            d_hist[n] = demand_increment
            l_hist[n] = Delta_L
            z_hist[n + 1] = Z
            i_hist[n + 1] = I.squeeze(1)
            inventory_cap_active_hist[n] = cap_active
            zero_inventory_hist[n] = zero_inventory_mask
            projection_scale_hist[n] = projection_diag["projection_scale"]

    terminal_cost = (Z * h.unsqueeze(0)).sum(dim=1)
    total_shortage_cost = total_stockout_cost + total_travel_cost
    total_objective = total_shortage_cost - total_weighted_fairness_utility + terminal_cost

    results = {
        "total_objectives": total_objective.detach().cpu().numpy(),
        "total_costs": total_objective.detach().cpu().numpy(),
        "stockout_costs": total_stockout_cost.detach().cpu().numpy(),
        "untreated_costs": total_stockout_cost.detach().cpu().numpy(),
        "travel_costs": total_travel_cost.detach().cpu().numpy(),
        "referral_travel_costs": total_travel_cost.detach().cpu().numpy(),
        "shortage_costs": total_shortage_cost.detach().cpu().numpy(),
        "fairness_utilities": total_fairness_utility.detach().cpu().numpy(),
        "weighted_fairness_utilities": total_weighted_fairness_utility.detach().cpu().numpy(),
        "fairness_objective_contributions": total_weighted_fairness_utility.detach().cpu().numpy(),
        "fairness_costs": total_weighted_fairness_utility.detach().cpu().numpy(),
        "terminal_costs": terminal_cost.detach().cpu().numpy(),
        "time_dt": dt,
        **eqn.fairness_metadata(),
    }

    if return_history:
        results.update(
            {
                "z_hist": z_hist.detach().cpu().numpy(),
                "i_hist": i_hist.detach().cpu().numpy(),
                "a_hist": a_hist.detach().cpu().numpy(),
                "a_raw_hist": a_raw_hist.detach().cpu().numpy(),
                "d_hist": d_hist.detach().cpu().numpy(),
                "l_hist": l_hist.detach().cpu().numpy(),
                "mu_hist": mu_hist.detach().cpu().numpy(),
                "eta_hist": eta_hist.detach().cpu().numpy(),
                "inventory_cap_active_hist": inventory_cap_active_hist.detach().cpu().numpy(),
                "zero_inventory_hist": zero_inventory_hist.detach().cpu().numpy(),
                "projection_scale_hist": projection_scale_hist.detach().cpu().numpy(),
                "inventory_projection_frequency": float(
                    inventory_cap_active_hist.float().mean().detach().cpu().item()
                ),
                "inventory_projection_mean_scale": float(
                    projection_scale_hist[inventory_cap_active_hist].mean().detach().cpu().item()
                )
                if bool(inventory_cap_active_hist.any().detach().cpu().item())
                else 1.0,
                "inventory_projection_min_scale": float(
                    projection_scale_hist[inventory_cap_active_hist].min().detach().cpu().item()
                )
                if bool(inventory_cap_active_hist.any().detach().cpu().item())
                else 1.0,
            }
        )

    return results


def load_raw_parameter_arrays(eqn_config):
    """Load raw demand-rate and covariance-rate arrays from the config directory."""
    mu_raw, gamma_raw = eqn.load_mu_gamma_raw(eqn_config, dtype=torch.float32)
    return mu_raw.detach().cpu().numpy(), gamma_raw.detach().cpu().numpy()


def infer_review_num_intervals(total_time, review_num_intervals=None, review_dt=None, default_num_intervals=None):
    """Resolve the discrete review grid from either N or review length."""
    if review_num_intervals is not None and review_dt is not None:
        raise ValueError("Specify at most one of review_num_intervals and review_dt.")

    if review_num_intervals is not None:
        return int(review_num_intervals)

    if review_dt is not None:
        scaled = float(total_time) / float(review_dt)
        rounded = int(round(scaled))
        if not np.isclose(scaled, rounded, atol=1e-10, rtol=0.0):
            raise ValueError(
                f"review_dt={review_dt} does not divide total_time={total_time} into an integer number of intervals."
            )
        return rounded

    if default_num_intervals is None:
        raise ValueError("A default review interval count is required when no review grid is specified.")
    return int(default_num_intervals)


def resample_parameters_to_review_grid(mu_raw, gamma_raw, total_time, num_review_intervals):
    """Align weekly/raw moment rates to the requested review grid.

    For coarser grids, rates are averaged; later multiplying by review_dt gives
    the sum of the raw-period means/covariances.
    """
    del total_time  # raw and review grids are assumed to partition the same horizon uniformly.
    num_review_intervals = int(num_review_intervals)
    num_raw_intervals = mu_raw.shape[0]

    if gamma_raw.shape[0] != num_raw_intervals:
        raise ValueError("mu_raw and gamma_raw must have the same number of rows.")
    if num_review_intervals <= 0:
        raise ValueError("num_review_intervals must be positive.")

    if num_review_intervals == num_raw_intervals:
        return mu_raw.copy(), gamma_raw.copy()

    if num_review_intervals % num_raw_intervals == 0:
        # Finer review grid: repeat the raw rate within each raw interval.
        repeat_factor = num_review_intervals // num_raw_intervals
        mu_review = np.repeat(mu_raw, repeat_factor, axis=0)
        gamma_review = np.repeat(gamma_raw, repeat_factor, axis=0)
        return mu_review, gamma_review

    if num_raw_intervals % num_review_intervals == 0:
        # Coarser review grid: average rates over nested raw intervals.
        block_factor = num_raw_intervals // num_review_intervals
        mu_review = mu_raw.reshape(num_review_intervals, block_factor, -1).mean(axis=1)
        gamma_review = gamma_raw.reshape(num_review_intervals, block_factor, gamma_raw.shape[1], gamma_raw.shape[2]).mean(axis=1)
        return mu_review, gamma_review

    raise ValueError(
        "Review grid must exactly nest the raw parameter grid or be exactly nested by it. "
        f"Got N_raw={num_raw_intervals}, N_review={num_review_intervals}."
    )


def prepare_periodic_review_parameters(config, review_num_intervals=None, review_dt=None):
    """Legacy name for preparing discrete-review moment rates."""
    mu_raw, gamma_raw = load_raw_parameter_arrays(config.eqn_config)
    num_review_intervals = infer_review_num_intervals(
        config.eqn_config.total_time,
        review_num_intervals=review_num_intervals,
        review_dt=review_dt,
        default_num_intervals=mu_raw.shape[0],
    )
    mu_review, gamma_review = resample_parameters_to_review_grid(
        mu_raw,
        gamma_raw,
        config.eqn_config.total_time,
        num_review_intervals,
    )
    return {
        "mu_review": mu_review,
        "gamma_review": gamma_review,
        "num_review_intervals": num_review_intervals,
    }


def prepare_discrete_review_parameters(config, review_num_intervals=None, review_dt=None):
    """Prepare period-level demand moments for the discrete model."""
    return prepare_periodic_review_parameters(
        config,
        review_num_intervals=review_num_intervals,
        review_dt=review_dt,
    )


def project_psd(mat, eps=1e-10):
    """Clip eigenvalues to make a symmetric matrix numerically PSD."""
    evals, evecs = np.linalg.eigh(mat)
    evals = np.clip(evals, eps, None)
    return (evecs * evals) @ evecs.T


def build_correlation_matrix(cov, eps=1e-10):
    """Convert covariance to a PSD correlation matrix for Gaussian copulas."""
    cov_psd = project_psd(cov, eps=eps)
    diag = np.sqrt(np.clip(np.diag(cov_psd), eps, None))
    corr = cov_psd / np.outer(diag, diag)
    corr = np.clip(corr, -1.0, 1.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return project_psd(corr, eps=eps)


def gamma_has_material_off_diagonal(gamma_n, tol=1e-10):
    gamma_arr = np.asarray(gamma_n, dtype=np.float64)
    off_diag = gamma_arr - np.diag(np.diag(gamma_arr))
    return not np.allclose(off_diag, 0.0, atol=tol, rtol=tol)


def infer_binomial_probabilities(mu_n, dt, exposure, eps=1e-8):
    mean = np.asarray(mu_n, dtype=np.float64) * float(dt)
    exposure_arr = np.asarray(exposure, dtype=np.float64).reshape(-1)
    if exposure_arr.shape[0] != mean.shape[0]:
        raise ValueError(f"exposure has {exposure_arr.shape[0]} values, expected {mean.shape[0]}.")
    if np.any(exposure_arr < -eps):
        raise ValueError("exposure must be nonnegative.")
    probs = np.divide(mean, np.maximum(exposure_arr, eps), out=np.zeros_like(mean), where=exposure_arr > eps)
    if np.any(probs < -1e-7) or np.any(probs > 1.0 + 1e-7):
        raise ValueError(f"Binomial probabilities outside [0,1]: {probs.tolist()}")
    return np.clip(probs, 0.0, 1.0), exposure_arr.astype(np.int64)


def sample_binomial_demand(mu_n, gamma_n, dt, num_paths, rng, exposure, eps=1e-8):
    """Sample independent binomial demand using exposure sizes and mu-implied theta."""
    del gamma_n
    probs, exposure_int = infer_binomial_probabilities(mu_n, dt, exposure, eps=eps)
    draws = rng.binomial(exposure_int.reshape(1, -1), probs.reshape(1, -1), size=(num_paths, len(probs)))
    return draws.astype(np.float32)


def sample_gaussian_copula_binomial_demand(mu_n, gamma_n, dt, num_paths, rng, exposure, eps=1e-8):
    """Sample binomial marginals coupled by a Gaussian copula."""
    probs, exposure_int = infer_binomial_probabilities(mu_n, dt, exposure, eps=eps)
    gamma_arr = np.asarray(gamma_n, dtype=np.float64) * float(dt)
    if not gamma_has_material_off_diagonal(gamma_arr, tol=eps):
        return sample_binomial_demand(mu_n, gamma_n, dt, num_paths, rng, exposure, eps=eps)
    corr = build_correlation_matrix(gamma_arr, eps=eps)
    gaussian_draws = rng.multivariate_normal(mean=np.zeros(len(probs)), cov=corr, size=num_paths)
    uniforms = np.clip(norm.cdf(gaussian_draws), eps, 1.0 - eps)
    draws = np.zeros((num_paths, len(probs)), dtype=np.float64)
    for k, (N_k, p_k) in enumerate(zip(exposure_int, probs)):
        if N_k > 0 and p_k > eps:
            draws[:, k] = binom.ppf(uniforms[:, k], N_k, p_k)
    return draws.astype(np.float32)


def sample_beta_binomial_demand(mu_n, gamma_n, dt, num_paths, rng, exposure, eps=1e-8):
    """Sample beta-binomial CHA marginals and binomial HF demand independently."""
    probs, exposure_int = infer_binomial_probabilities(mu_n, dt, exposure, eps=eps)
    gamma_arr = np.asarray(gamma_n, dtype=np.float64)
    target_variances = np.diag(gamma_arr) * float(dt)
    baseline_variances = exposure_int * probs * (1.0 - probs)
    draws = np.zeros((num_paths, len(probs)), dtype=np.float64)
    for k, (N_k, p_k) in enumerate(zip(exposure_int, probs)):
        if N_k <= 0 or p_k <= eps:
            continue
        if k == 0:
            draws[:, k] = rng.binomial(N_k, p_k, size=num_paths)
            continue
        base_var = max(float(baseline_variances[k]), eps)
        scale = max(float(target_variances[k]) / base_var, 1.0)
        if scale <= 1.0 + 1e-8:
            draws[:, k] = rng.binomial(N_k, p_k, size=num_paths)
            continue
        scale = min(scale, float(N_k) - 1e-8)
        concentration = (float(N_k) - scale) / (scale - 1.0)
        alpha = max(p_k * concentration, eps)
        beta = max((1.0 - p_k) * concentration, eps)
        draws[:, k] = betabinom.rvs(N_k, alpha, beta, size=num_paths, random_state=rng)
    return draws.astype(np.float32)


def sample_negative_binomial_demand(mu_n, gamma_n, dt, num_paths, rng, eps=1e-8):
    """Sample NB demand matched to period mean and diagonal variances."""
    mean = np.asarray(mu_n, dtype=np.float64) * float(dt)
    gamma_arr = np.asarray(gamma_n, dtype=np.float64)
    variances = np.diag(gamma_arr) * float(dt)
    dim = mean.shape[0]

    paper_p = np.zeros(dim, dtype=np.float64)
    r_param = np.zeros(dim, dtype=np.float64)
    scipy_p = np.ones(dim, dtype=np.float64)
    active = mean > eps

    if np.any(active & (variances <= mean + eps)):
        bad_idx = np.where(active & (variances <= mean + eps))[0].tolist()
        raise ValueError(
            "Strict overdispersion is required for NB sampling in discrete-review evaluation. "
            f"Locations failing var>mean: {bad_idx}"
        )

    paper_p[active] = 1.0 - mean[active] / variances[active]
    r_param[active] = mean[active] ** 2 / (variances[active] - mean[active])
    scipy_p[active] = 1.0 - paper_p[active]

    draws = np.zeros((num_paths, dim), dtype=np.float64)
    if not gamma_has_material_off_diagonal(gamma_arr, tol=eps):
        for k in range(dim):
            if active[k]:
                draws[:, k] = nbinom.rvs(r_param[k], scipy_p[k], size=num_paths, random_state=rng)
        return draws.astype(np.float32)

    corr = build_correlation_matrix(gamma_arr, eps=eps)
    gaussian_draws = rng.multivariate_normal(mean=np.zeros(dim), cov=corr, size=num_paths)
    uniforms = np.clip(norm.cdf(gaussian_draws), eps, 1.0 - eps)

    for k in range(dim):
        if active[k]:
            draws[:, k] = nbinom.ppf(uniforms[:, k], r_param[k], scipy_p[k])
    return draws.astype(np.float32)


def sample_brownian_motion_demand(mu_n, gamma_n, dt, num_paths, rng):
    """Sample Gaussian period demand; can be negative and is diagnostic only."""
    mean = np.asarray(mu_n, dtype=np.float64) * float(dt)
    cov = project_psd(np.asarray(gamma_n, dtype=np.float64) * float(dt))
    draws = rng.multivariate_normal(mean=mean, cov=cov, size=num_paths)
    return draws.astype(np.float32)


def sample_lognormal_demand(mu_n, gamma_n, dt, num_paths, rng, eps=1e-8):
    """Sample nonnegative continuous demand matched to target first and second moments."""
    mean = np.asarray(mu_n, dtype=np.float64) * float(dt)
    cov = project_psd(np.asarray(gamma_n, dtype=np.float64) * float(dt), eps=eps)
    dim = mean.shape[0]
    draws = np.zeros((num_paths, dim), dtype=np.float64)

    active = mean > eps
    if not np.any(active):
        return draws.astype(np.float32)

    mean_active = mean[active]
    cov_active = cov[np.ix_(active, active)]
    denom = np.outer(mean_active, mean_active)
    log_cov_arg = 1.0 + cov_active / np.maximum(denom, eps)
    log_cov = np.log(np.maximum(log_cov_arg, eps))
    log_cov = project_psd(log_cov, eps=eps)
    log_mean = np.log(mean_active) - 0.5 * np.diag(log_cov)

    normal_draws = rng.multivariate_normal(mean=log_mean, cov=log_cov, size=num_paths)
    draws[:, active] = np.exp(normal_draws)
    return draws.astype(np.float32)


def sample_periodic_demand(mu_n, gamma_n, dt, num_paths, rng, mode="lognormal", eps=1e-8, exposure=None):
    """Dispatch the configured period-demand sampler."""
    if mode == "binomial":
        if exposure is None:
            raise ValueError("binomial demand mode requires exposure.")
        return sample_binomial_demand(mu_n, gamma_n, dt, num_paths, rng, exposure, eps=eps)
    if mode == "gaussian_copula_binomial":
        if exposure is None:
            raise ValueError("gaussian_copula_binomial demand mode requires exposure.")
        return sample_gaussian_copula_binomial_demand(
            mu_n,
            gamma_n,
            dt,
            num_paths,
            rng,
            exposure,
            eps=eps,
        )
    if mode == "beta_binomial":
        if exposure is None:
            raise ValueError("beta_binomial demand mode requires exposure.")
        return sample_beta_binomial_demand(mu_n, gamma_n, dt, num_paths, rng, exposure, eps=eps)
    if mode == "negative_binomial":
        return sample_negative_binomial_demand(
            mu_n,
            gamma_n,
            dt,
            num_paths,
            rng,
            eps=eps,
        )
    if mode == "brownian_motion":
        return sample_brownian_motion_demand(mu_n, gamma_n, dt, num_paths, rng)
    if mode == "lognormal":
        return sample_lognormal_demand(mu_n, gamma_n, dt, num_paths, rng, eps=eps)
    raise ValueError(f"Unsupported discrete demand mode: {mode}")


def resolve_periodic_demand_mode(mode, bsde=None, gamma_review=None):
    if mode != "auto":
        return mode
    configured = getattr(bsde, "periodic_demand_mode", None) if bsde is not None else None
    if configured and configured != "auto":
        return str(configured)
    exposure = getattr(bsde, "exposure", None) if bsde is not None else None
    has_exposure = exposure is not None and int(torch.as_tensor(exposure).numel()) > 0
    if not has_exposure:
        return "lognormal"
    if gamma_review is not None and np.any(
        [
            gamma_has_material_off_diagonal(gamma_review[n])
            for n in range(np.asarray(gamma_review).shape[0])
        ]
    ):
        return "gaussian_copula_binomial"
    return "binomial"


def simulate_periodic_review_trajectories(
    policy_fn,
    bsde,
    num_paths,
    device,
    mu_review,
    gamma_review,
    demand_mode="auto",
    seed=0,
    eps=1e-8,
    return_history=False,
    precomputed_demands=None,
    exposure=None,
):
    """Simulate the discrete-review model using fixed-state subinterval averaging.

    The function name is legacy. It no longer performs virtual rollout: within
    each review period, the physical state is held fixed while fine-grid rates
    are queried and averaged into one shipment.

    Within each review period, policy rates are evaluated at the observed period-start
    state for every computational subinterval. The resulting shipment quantities are
    summed and then projected once onto the period-level inventory-feasible set.
    """
    rng = None if precomputed_demands is not None else np.random.default_rng(seed)

    num_review_intervals, dim = mu_review.shape
    num_fine_intervals = bsde.num_time_interval
    if num_fine_intervals % num_review_intervals != 0:
        raise ValueError(
            "Fine training grid must be an integer multiple of the discrete-review grid. "
            f"Got N_fine={num_fine_intervals}, N_review={num_review_intervals}."
        )
    print(f"num_fine_intervals: {num_fine_intervals}, num_review_intervals: {num_review_intervals}")

    fine_per_review = num_fine_intervals // num_review_intervals
    review_dt = bsde.total_time / num_review_intervals
    fine_dt = bsde.delta_t
    if hasattr(bsde, "eta_for_review_grid"):
        eta_review = bsde.eta_for_review_grid(num_review_intervals).detach().cpu().numpy().astype(np.float32)
    else:
        eta_review = np.repeat(
            bsde.eta.detach().cpu().numpy().reshape(1, -1).astype(np.float32),
            num_review_intervals,
            axis=0,
        )
    if precomputed_demands is not None:
        expected_shape = (num_review_intervals, num_paths, dim)
        if precomputed_demands.shape != expected_shape:
            raise ValueError(
                f"precomputed_demands must have shape {expected_shape}, got {precomputed_demands.shape}."
            )

    Z = np.zeros((num_paths, dim), dtype=np.float32)
    I = np.full((num_paths,), float(bsde.I0), dtype=np.float32)

    p = bsde.p.detach().cpu().numpy().astype(np.float32)
    c = bsde.c.detach().cpu().numpy().astype(np.float32)
    d = bsde.d.detach().cpu().numpy().astype(np.float32)
    h = bsde.h.detach().cpu().numpy().astype(np.float32)
    lambda_param = float(bsde.lambda_param)
    alpha = float(bsde.alpha)

    total_stockout_cost = np.zeros(num_paths, dtype=np.float64)
    total_travel_cost = np.zeros(num_paths, dtype=np.float64)
    total_fairness_utility = np.zeros(num_paths, dtype=np.float64)
    total_weighted_fairness_utility = np.zeros(num_paths, dtype=np.float64)
    cap_trigger_count = 0
    zero_inventory_period_count = 0
    zero_inventory_action_count = 0
    total_period_count = num_review_intervals * num_paths

    if return_history:
        z_hist = np.zeros((num_review_intervals + 1, num_paths, dim), dtype=np.float32)
        i_hist = np.zeros((num_review_intervals + 1, num_paths), dtype=np.float32)
        a_hist = np.zeros((num_review_intervals, num_paths, dim), dtype=np.float32)
        a_raw_hist = np.zeros((num_review_intervals, num_paths, dim), dtype=np.float32)
        fine_rate_hist = np.zeros((num_review_intervals, fine_per_review, num_paths, dim), dtype=np.float32)
        raw_fine_rate_hist = np.zeros((num_review_intervals, fine_per_review, num_paths, dim), dtype=np.float32)
        fine_quantity_hist = np.zeros((num_review_intervals, fine_per_review, num_paths, dim), dtype=np.float32)
        fine_projection_needed_hist = np.zeros((num_review_intervals, fine_per_review, num_paths), dtype=bool)
        fine_projection_scale_hist = np.ones((num_review_intervals, fine_per_review, num_paths), dtype=np.float32)
        virtual_q_hist = np.full((num_review_intervals, fine_per_review + 1, num_paths), np.nan, dtype=np.float32)
        virtual_z_hist = np.full(
            (num_review_intervals, fine_per_review + 1, num_paths, dim),
            np.nan,
            dtype=np.float32,
        )
        virtual_l_hist = np.zeros((num_review_intervals, fine_per_review, num_paths, dim), dtype=np.float32)
        action_regime_hist = np.full(
            (num_review_intervals, fine_per_review, num_paths, dim),
            -1,
            dtype=np.int8,
        )
        extraction_query_count_hist = np.zeros((num_review_intervals,), dtype=np.int16)
        q_hist = np.zeros((num_review_intervals, num_paths, dim), dtype=np.float32)
        q_raw_hist = np.zeros((num_review_intervals, num_paths, dim), dtype=np.float32)
        l_hist = np.zeros((num_review_intervals, num_paths, dim), dtype=np.float32)
        d_hist = np.zeros((num_review_intervals, num_paths, dim), dtype=np.float32)
        mu_hist = np.zeros((num_review_intervals, dim), dtype=np.float32)
        eta_hist = np.zeros((num_review_intervals, dim), dtype=np.float32)
        projection_needed_hist = np.zeros((num_review_intervals, num_paths), dtype=bool)
        zero_inventory_hist = np.zeros((num_review_intervals, num_paths), dtype=bool)
        projection_scale_hist = np.ones((num_review_intervals, num_paths), dtype=np.float32)
        z_hist[0] = Z
        i_hist[0] = I

    for n in range(num_review_intervals):
        alloc_period_raw = np.zeros((num_paths, dim), dtype=np.float32)
        z_period_start = torch.from_numpy(Z).to(device=device, dtype=bsde.mu.dtype)
        i_period_start = torch.from_numpy(I).to(device=device, dtype=bsde.mu.dtype)

        extract_weekly_action = getattr(policy_fn, "extract_weekly_action", None)
        if callable(extract_weekly_action):
            extraction = extract_weekly_action(
                n,
                z_period_start,
                i_period_start,
                fine_per_review,
                review_dt,
            )
            alloc_period_raw = (
                extraction.tentative_quantity.detach().cpu().numpy().astype(np.float32)
            )
            if return_history:
                fine_rate_hist[n] = extraction.effective_rates.detach().cpu().numpy().astype(np.float32)
                raw_fine_rate_hist[n] = extraction.raw_rates.detach().cpu().numpy().astype(np.float32)
                fine_quantity_hist[n] = extraction.feasible_quantities.detach().cpu().numpy().astype(np.float32)
                fine_projection_needed_hist[n] = extraction.fine_projection_mask.detach().cpu().numpy()
                fine_projection_scale_hist[n] = extraction.fine_projection_scale.detach().cpu().numpy().astype(np.float32)
                virtual_q_hist[n] = extraction.virtual_q.detach().cpu().numpy().astype(np.float32)
                virtual_z_hist[n] = extraction.virtual_z.detach().cpu().numpy().astype(np.float32)
                virtual_l_hist[n] = extraction.virtual_l.detach().cpu().numpy().astype(np.float32)
                action_regime_hist[n] = extraction.action_regimes.detach().cpu().numpy().astype(np.int8)
                extraction_query_count_hist[n] = int(extraction.query_count)
        else:
            # Query all fine-grid policy rules at the same period-start state.
            for j in range(fine_per_review):
                fine_idx = n * fine_per_review + j

                a_rate = (
                    policy_fn(fine_idx, z_period_start, i_period_start)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                if return_history:
                    fine_rate_hist[n, j] = a_rate
                    raw_fine_rate_hist[n, j] = a_rate
                    fine_quantity_hist[n, j] = a_rate * fine_dt
                    virtual_q_hist[n, j] = I
                    virtual_z_hist[n, j] = Z
                alloc_period_raw += a_rate * fine_dt
            if return_history:
                virtual_q_hist[n, fine_per_review] = I
                virtual_z_hist[n, fine_per_review] = Z
                extraction_query_count_hist[n] = fine_per_review

        # A period-level projection is still needed after aggregating substeps.
        alloc_period, needs_scale, zero_inventory_mask = cap_quantity_to_inventory_numpy(
            alloc_period_raw,
            I,
            max_quantity=eta_review[n] * review_dt,
        )
        cap_trigger_count += int(np.count_nonzero(needs_scale))
        zero_inventory_period_count += int(np.count_nonzero(zero_inventory_mask))
        zero_inventory_action_count += int(
            np.count_nonzero(zero_inventory_mask & (alloc_period_raw.sum(axis=1) > 1e-12))
        )

        z_plus = Z + alloc_period
        # Demand is realized once per review period, after the shipment arrives.
        if precomputed_demands is not None:
            xi_n = np.asarray(precomputed_demands[n], dtype=np.float32)
        else:
            xi_n = sample_periodic_demand(
                mu_review[n],
                gamma_review[n],
                review_dt,
                num_paths,
                rng,
                mode=demand_mode,
                eps=eps,
                exposure=exposure,
            ).astype(np.float32)

        L = np.zeros((num_paths, dim), dtype=np.float32)
        Z_new = np.zeros_like(z_plus)

        # CHA shortages create referral demand at the HF.
        cha_short = np.maximum(xi_n[:, 1:] - z_plus[:, 1:], 0.0)
        L[:, 1:] = cha_short
        Z_new[:, 1:] = np.maximum(z_plus[:, 1:] - xi_n[:, 1:], 0.0)

        hf_total_demand = xi_n[:, 0] + (cha_short * p[None, :]).sum(axis=1)
        hf_short = np.maximum(hf_total_demand - z_plus[:, 0], 0.0)
        L[:, 0] = hf_short
        Z_new[:, 0] = np.maximum(z_plus[:, 0] - hf_total_demand, 0.0)

        untreated_cost_step = (
            float(bsde.u.detach().cpu().item()) * L[:, 0]
            + (L[:, 1:] * float(bsde.u.detach().cpu().item()) * (1.0 - p)[None, :]).sum(axis=1)
        )
        referral_cost_step = (L[:, 1:] * d[None, 1:] * p[None, :]).sum(axis=1)
        shortage_cost_step = (L * c[None, :]).sum(axis=1)
        if not np.allclose(untreated_cost_step + referral_cost_step, shortage_cost_step, rtol=1e-5, atol=1e-5):
            raise AssertionError("Untreated plus referral cost does not equal c · L.")
        total_stockout_cost += untreated_cost_step
        total_travel_cost += referral_cost_step

        expected_demand_n = mu_review[n] * review_dt
        fairness_utility_step = eqn.alpha_fair_utility_numpy(alloc_period, expected_demand_n, alpha)
        weighted_fairness_utility_step = lambda_param * fairness_utility_step
        total_fairness_utility += fairness_utility_step
        total_weighted_fairness_utility += weighted_fairness_utility_step

        Z = Z_new
        I = np.clip(I - alloc_period.sum(axis=1), 0.0, None)

        if return_history:
            a_hist[n] = alloc_period / review_dt
            a_raw_hist[n] = alloc_period_raw / review_dt
            q_hist[n] = alloc_period
            q_raw_hist[n] = alloc_period_raw
            l_hist[n] = L
            d_hist[n] = xi_n
            mu_hist[n] = mu_review[n]
            eta_hist[n] = eta_review[n]
            projection_needed_hist[n] = needs_scale
            zero_inventory_hist[n] = zero_inventory_mask
            raw_total = alloc_period_raw.sum(axis=1)
            projected_total = alloc_period.sum(axis=1)
            projection_scale_hist[n] = np.divide(
                projected_total,
                np.maximum(raw_total, 1e-12),
                out=np.ones_like(projected_total, dtype=np.float32),
                where=raw_total > 1e-12,
            ).astype(np.float32)
            z_hist[n + 1] = Z
            i_hist[n + 1] = I

    terminal_cost = (Z * h[None, :]).sum(axis=1)
    total_shortage_cost = total_stockout_cost + total_travel_cost
    total_objective = total_shortage_cost - total_weighted_fairness_utility + terminal_cost

    results = {
        "total_objectives": total_objective,
        "total_costs": total_objective,
        "stockout_costs": total_stockout_cost,
        "untreated_costs": total_stockout_cost,
        "travel_costs": total_travel_cost,
        "referral_travel_costs": total_travel_cost,
        "shortage_costs": total_shortage_cost,
        "fairness_utilities": total_fairness_utility,
        "weighted_fairness_utilities": total_weighted_fairness_utility,
        "fairness_objective_contributions": total_weighted_fairness_utility,
        "fairness_costs": total_weighted_fairness_utility,
        "terminal_costs": terminal_cost,
        "inventory_cap_trigger_rate": cap_trigger_count / max(total_period_count, 1),
        "zero_inventory_positive_action_rate": zero_inventory_action_count / max(zero_inventory_period_count, 1),
        "time_dt": review_dt,
        **eqn.fairness_metadata(),
    }
    if return_history:
        results.update(
            {
                "z_hist": z_hist,
                "i_hist": i_hist,
                "a_hist": a_hist,
                "a_raw_hist": a_raw_hist,
                "fine_rate_hist": fine_rate_hist,
                "raw_fine_rate_hist": raw_fine_rate_hist,
                "fine_quantity_hist": fine_quantity_hist,
                "fine_projection_needed_hist": fine_projection_needed_hist,
                "fine_projection_scale_hist": fine_projection_scale_hist,
                "virtual_q_hist": virtual_q_hist,
                "virtual_z_hist": virtual_z_hist,
                "virtual_l_hist": virtual_l_hist,
                "action_regime_hist": action_regime_hist,
                "extraction_query_count_hist": extraction_query_count_hist,
                "q_hist": q_hist,
                "q_raw_hist": q_raw_hist,
                "l_hist": l_hist,
                "d_hist": d_hist,
                "mu_hist": mu_hist,
                "eta_hist": eta_hist,
                "projection_needed_hist": projection_needed_hist,
                "zero_inventory_hist": zero_inventory_hist,
                "projection_scale_hist": projection_scale_hist,
            }
        )
    return results


def simulate_open_loop_periodic_review_trajectories(
    allocation_plan,
    bsde,
    num_paths,
    device,
    mu_review,
    gamma_review,
    demand_mode="auto",
    seed=0,
    eps=1e-8,
    return_history=False,
    precomputed_demands=None,
    initial_z=None,
    initial_inventory=None,
    exposure=None,
):
    """Simulate a fixed open-loop period allocation plan in the discrete model.

    ``allocation_plan`` is interpreted as period shipment quantities with shape
    (num_review_intervals, K+1). It is not recomputed from the realized state.
    """
    del device  # Kept in the signature for consistency with other evaluators.
    rng = None if precomputed_demands is not None else np.random.default_rng(seed)

    allocation_plan = np.asarray(allocation_plan, dtype=np.float64)
    mu_review = np.asarray(mu_review, dtype=np.float64)
    gamma_review = np.asarray(gamma_review, dtype=np.float64)
    num_review_intervals, dim = mu_review.shape
    if allocation_plan.shape != (num_review_intervals, dim):
        raise ValueError(
            "allocation_plan must have shape "
            f"{(num_review_intervals, dim)}, got {allocation_plan.shape}."
        )
    if np.any(allocation_plan < -eps):
        raise ValueError("allocation_plan must be nonnegative up to numerical tolerance.")
    allocation_plan = np.clip(allocation_plan, 0.0, None)

    if precomputed_demands is not None:
        expected_shape = (num_review_intervals, num_paths, dim)
        if precomputed_demands.shape != expected_shape:
            raise ValueError(
                f"precomputed_demands must have shape {expected_shape}, got {precomputed_demands.shape}."
            )

    review_dt = float(bsde.total_time) / float(num_review_intervals)
    if hasattr(bsde, "eta_for_review_grid"):
        eta_review = bsde.eta_for_review_grid(num_review_intervals).detach().cpu().numpy().astype(np.float64)
    else:
        eta_review = np.repeat(
            bsde.eta.detach().cpu().numpy().reshape(1, -1).astype(np.float64),
            num_review_intervals,
            axis=0,
        )
    max_period_quantity = eta_review * review_dt
    max_eta_violation = float(np.max(allocation_plan - max_period_quantity)) if allocation_plan.size else 0.0
    if max_eta_violation > max(1e-6, 1e-8 * max(float(np.max(max_period_quantity)), 1.0)):
        raise ValueError(
            "Open-loop allocation plan exceeds eta schedule bounds: "
            f"max violation={max_eta_violation}."
        )
    p = bsde.p.detach().cpu().numpy().astype(np.float64)
    c = bsde.c.detach().cpu().numpy().astype(np.float64)
    d = bsde.d.detach().cpu().numpy().astype(np.float64)
    u_cost = float(bsde.u.detach().cpu().item())
    h = bsde.h.detach().cpu().numpy().astype(np.float64)
    lambda_param = float(bsde.lambda_param)
    alpha = float(bsde.alpha)

    if initial_z is None:
        Z = np.zeros((num_paths, dim), dtype=np.float64)
    else:
        z0 = np.asarray(initial_z, dtype=np.float64)
        if z0.shape == (dim,):
            Z = np.repeat(z0.reshape(1, dim), num_paths, axis=0)
        elif z0.shape == (num_paths, dim):
            Z = z0.copy()
        else:
            raise ValueError(f"initial_z must have shape {(dim,)} or {(num_paths, dim)}, got {z0.shape}.")
    if np.any(Z < -eps):
        raise ValueError("initial_z must be nonnegative up to numerical tolerance.")
    Z = np.clip(Z, 0.0, None)

    q0 = float(bsde.I0) if initial_inventory is None else float(initial_inventory)
    I = np.full((num_paths,), q0, dtype=np.float64)

    cumulative_plan = np.cumsum(allocation_plan.sum(axis=1), axis=0)
    max_warehouse_overdraw = float(np.max(cumulative_plan - q0)) if cumulative_plan.size else 0.0
    if max_warehouse_overdraw > max(1e-6, 1e-8 * max(q0, 1.0)):
        raise ValueError(
            "Open-loop allocation plan ships more than initial warehouse inventory: "
            f"max overdraw={max_warehouse_overdraw}."
        )

    total_stockout_cost = np.zeros(num_paths, dtype=np.float64)
    total_travel_cost = np.zeros(num_paths, dtype=np.float64)
    total_cha_shortage = np.zeros(num_paths, dtype=np.float64)
    total_hf_shortage = np.zeros(num_paths, dtype=np.float64)
    total_referral_demand = np.zeros(num_paths, dtype=np.float64)
    total_served_chas = np.zeros(num_paths, dtype=np.float64)
    total_served_hf = np.zeros(num_paths, dtype=np.float64)
    total_fairness_utility = np.zeros(num_paths, dtype=np.float64)
    total_weighted_fairness_utility = np.zeros(num_paths, dtype=np.float64)

    if return_history:
        z_hist = np.zeros((num_review_intervals + 1, num_paths, dim), dtype=np.float32)
        i_hist = np.zeros((num_review_intervals + 1, num_paths), dtype=np.float32)
        a_hist = np.zeros((num_review_intervals, num_paths, dim), dtype=np.float32)
        q_hist = np.zeros((num_review_intervals, num_paths, dim), dtype=np.float32)
        l_hist = np.zeros((num_review_intervals, num_paths, dim), dtype=np.float32)
        d_hist = np.zeros((num_review_intervals, num_paths, dim), dtype=np.float32)
        mu_hist = np.zeros((num_review_intervals, dim), dtype=np.float32)
        eta_hist = np.zeros((num_review_intervals, dim), dtype=np.float32)
        referral_hist = np.zeros((num_review_intervals, num_paths), dtype=np.float32)
        served_cha_hist = np.zeros((num_review_intervals, num_paths, dim - 1), dtype=np.float32)
        served_hf_hist = np.zeros((num_review_intervals, num_paths), dtype=np.float32)
        hf_total_demand_hist = np.zeros((num_review_intervals, num_paths), dtype=np.float32)
        z_hist[0] = Z.astype(np.float32)
        i_hist[0] = I.astype(np.float32)

    for n in range(num_review_intervals):
        alloc_period = allocation_plan[n].reshape(1, dim)
        alloc_period = np.repeat(alloc_period, num_paths, axis=0)
        I_after_alloc = I - alloc_period.sum(axis=1)
        if np.min(I_after_alloc) < -max(1e-6, 1e-8 * max(q0, 1.0)):
            raise ValueError(f"Open-loop allocation overdraws warehouse in period {n}.")
        I_after_alloc = np.clip(I_after_alloc, 0.0, None)
        z_plus = Z + alloc_period

        if precomputed_demands is not None:
            xi_n = np.asarray(precomputed_demands[n], dtype=np.float64)
        else:
            xi_n = sample_periodic_demand(
                mu_review[n],
                gamma_review[n],
                review_dt,
                num_paths,
                rng,
                mode=demand_mode,
                eps=eps,
                exposure=exposure,
            ).astype(np.float64)
        xi_n = np.clip(xi_n, 0.0, None)

        cha_demand = xi_n[:, 1:]
        cha_available = z_plus[:, 1:]
        served_chas = np.minimum(cha_available, cha_demand)
        cha_short = np.maximum(cha_demand - served_chas, 0.0)
        referrals = cha_short * p[None, :]

        hf_total_demand = xi_n[:, 0] + referrals.sum(axis=1)
        served_hf = np.minimum(z_plus[:, 0], hf_total_demand)
        hf_short = np.maximum(hf_total_demand - served_hf, 0.0)

        L = np.zeros((num_paths, dim), dtype=np.float64)
        L[:, 0] = hf_short
        L[:, 1:] = cha_short

        Z_new = np.zeros_like(z_plus)
        Z_new[:, 0] = np.maximum(z_plus[:, 0] - hf_total_demand, 0.0)
        Z_new[:, 1:] = np.maximum(cha_available - cha_demand, 0.0)

        stockout_cost_step = u_cost * hf_short + (cha_short * u_cost * (1.0 - p)[None, :]).sum(axis=1)
        travel_cost_step = (cha_short * d[None, 1:] * p[None, :]).sum(axis=1)
        shortage_cost_step = (L * c[None, :]).sum(axis=1)
        if not np.allclose(stockout_cost_step + travel_cost_step, shortage_cost_step, rtol=1e-5, atol=1e-5):
            raise AssertionError("Untreated plus referral cost does not equal c · L.")
        total_stockout_cost += stockout_cost_step
        total_travel_cost += travel_cost_step
        total_cha_shortage += cha_short.sum(axis=1)
        total_hf_shortage += hf_short
        total_referral_demand += referrals.sum(axis=1)
        total_served_chas += served_chas.sum(axis=1)
        total_served_hf += served_hf

        expected_demand_n = mu_review[n] * review_dt
        fairness_utility_step = eqn.alpha_fair_utility_numpy(alloc_period, expected_demand_n, alpha)
        weighted_fairness_utility_step = lambda_param * fairness_utility_step
        total_fairness_utility += fairness_utility_step
        total_weighted_fairness_utility += weighted_fairness_utility_step

        Z = Z_new
        I = I_after_alloc

        if return_history:
            a_hist[n] = (alloc_period / review_dt).astype(np.float32)
            q_hist[n] = alloc_period.astype(np.float32)
            l_hist[n] = L.astype(np.float32)
            d_hist[n] = xi_n.astype(np.float32)
            mu_hist[n] = mu_review[n].astype(np.float32)
            eta_hist[n] = eta_review[n].astype(np.float32)
            referral_hist[n] = referrals.sum(axis=1).astype(np.float32)
            served_cha_hist[n] = served_chas.astype(np.float32)
            served_hf_hist[n] = served_hf.astype(np.float32)
            hf_total_demand_hist[n] = hf_total_demand.astype(np.float32)
            z_hist[n + 1] = Z.astype(np.float32)
            i_hist[n + 1] = I.astype(np.float32)

    terminal_cost = (Z * h[None, :]).sum(axis=1)
    total_shortage_cost = total_stockout_cost + total_travel_cost
    total_objective = total_shortage_cost - total_weighted_fairness_utility + terminal_cost
    results = {
        "total_objectives": total_objective,
        "total_costs": total_objective,
        "stockout_costs": total_stockout_cost,
        "untreated_costs": total_stockout_cost,
        "travel_costs": total_travel_cost,
        "referral_travel_costs": total_travel_cost,
        "shortage_costs": total_shortage_cost,
        "fairness_utilities": total_fairness_utility,
        "weighted_fairness_utilities": total_weighted_fairness_utility,
        "fairness_objective_contributions": total_weighted_fairness_utility,
        "fairness_costs": total_weighted_fairness_utility,
        "terminal_costs": terminal_cost,
        "total_cha_shortage": total_cha_shortage,
        "total_hf_shortage": total_hf_shortage,
        "total_referral_demand": total_referral_demand,
        "total_served_chas": total_served_chas,
        "total_served_hf": total_served_hf,
        "final_warehouse_inventory": I.copy(),
        "final_local_inventory": Z.copy(),
        "time_dt": review_dt,
        "allocation_plan": allocation_plan,
        "inventory_cap_trigger_rate": 0.0,
        "zero_inventory_positive_action_rate": 0.0,
        **eqn.fairness_metadata(),
    }
    if return_history:
        results.update(
            {
                "z_hist": z_hist,
                "i_hist": i_hist,
                "a_hist": a_hist,
                "a_raw_hist": a_hist.copy(),
                "q_hist": q_hist,
                "l_hist": l_hist,
                "d_hist": d_hist,
                "mu_hist": mu_hist,
                "eta_hist": eta_hist,
                "referral_hist": referral_hist,
                "served_cha_hist": served_cha_hist,
                "served_hf_hist": served_hf_hist,
                "hf_total_demand_hist": hf_total_demand_hist,
            }
        )
    return results


def compute_statistics(costs):
    """Return Monte Carlo mean and standard error."""
    mean = np.mean(costs)
    std_err = np.std(costs, ddof=1) / np.sqrt(len(costs))
    return mean, std_err


def compute_paired_difference_summary(
    costs_by_policy,
    minuend_policy="Deep_Splitting",
    subtrahend_policy="Static",
):
    """Summarize pathwise objective differences under common random numbers."""
    if isinstance(subtrahend_policy, str):
        subtrahend_policies = [subtrahend_policy]
    else:
        subtrahend_policies = list(subtrahend_policy)
    if minuend_policy not in costs_by_policy:
        return None

    rows = []
    for benchmark_policy in subtrahend_policies:
        if benchmark_policy not in costs_by_policy:
            continue
        diff = np.asarray(costs_by_policy[minuend_policy]["total_objectives"]) - np.asarray(
            costs_by_policy[benchmark_policy]["total_objectives"]
        )
        mean_diff, se_diff = compute_statistics(diff)
        ci_half_width = 1.96 * se_diff
        rows.append(
            {
                "Difference_Formula": f"Objective_{minuend_policy} - Objective_{benchmark_policy}",
                "Policy_A": minuend_policy,
                "Policy_B": benchmark_policy,
                "Mean_Difference": mean_diff,
                "Standard_Error": se_diff,
                "CI95_Lower": mean_diff - ci_half_width,
                "CI95_Upper": mean_diff + ci_half_width,
                "Num_Paths": diff.shape[0],
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def summarize_cost_dict(policy_name, costs_dict):
    """Convert pathwise costs for one policy into report-ready statistics."""
    total_mean, total_se = compute_statistics(costs_dict["total_objectives"])
    untreated_mean, untreated_se = compute_statistics(costs_dict.get("untreated_costs", costs_dict["stockout_costs"]))
    referral_mean, referral_se = compute_statistics(
        costs_dict.get("referral_travel_costs", costs_dict["travel_costs"])
    )
    shortage_mean, shortage_se = compute_statistics(costs_dict["shortage_costs"])
    fairness_mean, fairness_se = compute_statistics(costs_dict["fairness_utilities"])
    weighted_fairness_values = costs_dict.get("weighted_fairness_utilities")
    if weighted_fairness_values is None:
        weighted_fairness_values = costs_dict.get("fairness_objective_contributions")
    if weighted_fairness_values is None:
        weighted_fairness_values = costs_dict.get("fairness_costs", costs_dict["fairness_utilities"])
    weighted_fairness_mean, weighted_fairness_se = compute_statistics(weighted_fairness_values)
    terminal_mean, terminal_se = compute_statistics(costs_dict["terminal_costs"])

    summary = {
        "Policy": policy_name,
        "Total objective": total_mean,
        "Total objective SE": total_se,
        "Untreated cost": untreated_mean,
        "Untreated cost SE": untreated_se,
        "Referral travel cost": referral_mean,
        "Referral travel cost SE": referral_se,
        "Fairness utility": fairness_mean,
        "Fairness utility SE": fairness_se,
        "Weighted fairness utility": weighted_fairness_mean,
        "Weighted fairness utility SE": weighted_fairness_se,
        "Total_Objective_Mean": total_mean,
        "Total_Objective_SE": total_se,
        "Untreated_Cost_Mean": untreated_mean,
        "Untreated_Cost_SE": untreated_se,
        "Referral_Travel_Cost_Mean": referral_mean,
        "Referral_Travel_Cost_SE": referral_se,
        "Stockout_Cost_Mean": untreated_mean,
        "Stockout_Cost_SE": untreated_se,
        "Travel_Cost_Mean": referral_mean,
        "Travel_Cost_SE": referral_se,
        "Shortage_Cost_Mean": shortage_mean,
        "Shortage_Cost_SE": shortage_se,
        "Fairness_Utility_Mean": fairness_mean,
        "Fairness_Utility_SE": fairness_se,
        "Weighted_Fairness_Utility_Mean": weighted_fairness_mean,
        "Weighted_Fairness_Utility_SE": weighted_fairness_se,
        "Terminal_Cost_Mean": terminal_mean,
        "Terminal_Cost_SE": terminal_se,
        **eqn.fairness_metadata(),
    }
    if "inventory_cap_trigger_rate" in costs_dict:
        summary["Inventory_Cap_Trigger_Rate"] = costs_dict["inventory_cap_trigger_rate"]
    if "zero_inventory_positive_action_rate" in costs_dict:
        summary["Zero_Inventory_Positive_Action_Rate"] = costs_dict["zero_inventory_positive_action_rate"]
    return summary


def add_performance_gaps(results):
    """Add percentage gaps relative to the first result, assumed to be DS."""
    if not results:
        return results

    baseline_total = results[0]["Total_Objective_Mean"]
    gap_denom = max(abs(baseline_total), 1e-12)
    results[0]["Performance_Gap_Percent"] = 0.0
    results[0]["Gap vs proposed policy"] = 0.0
    for result in results[1:]:
        gap = (
            (result["Total_Objective_Mean"] - baseline_total) / gap_denom
        ) * 100.0
        result["Performance_Gap_Percent"] = gap
        result["Gap vs proposed policy"] = gap
    return results


def evaluate_fine_grid_policies(
    policies,
    bsde,
    num_paths,
    device,
    output_path=None,
    use_common_random_numbers=True,
    common_random_seed=0,
    paired_output_path=None,
    return_costs=False,
    shared_w_increments=None,
    paired_minuend_policy="Deep_Splitting",
):
    """Evaluate every policy on the fine-grid Brownian approximation."""
    results = []
    costs_by_policy = {}
    if shared_w_increments is not None and not use_common_random_numbers:
        raise ValueError("A frozen fine-grid bank requires common random numbers.")
    if shared_w_increments is None and use_common_random_numbers:
        shared_w_increments = precompute_fine_grid_brownian_increments(
            bsde, num_paths, device, seed=common_random_seed
        )
    if shared_w_increments is not None:
        expected_shape = (int(bsde.num_time_interval), int(num_paths), int(bsde.dim))
        if tuple(shared_w_increments.shape) != expected_shape:
            raise ValueError(
                f"shared_w_increments must have shape {expected_shape}, got {tuple(shared_w_increments.shape)}."
            )

    for policy_name, policy_fn in policies.items():
        print(f"\nEvaluating {policy_name} policy on fine grid...")
        costs_dict = simulate_trajectories(
            policy_fn,
            bsde,
            num_paths,
            device,
            shared_w_increments=shared_w_increments,
        )
        costs_by_policy[policy_name] = costs_dict
        results.append(summarize_cost_dict(policy_name, costs_dict))

    add_performance_gaps(results)
    df = pd.DataFrame(results)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved fine-grid results to {output_path}")

    paired_df = None
    if use_common_random_numbers:
        paired_df = compute_paired_difference_summary(
            costs_by_policy,
            minuend_policy=paired_minuend_policy,
            subtrahend_policy=("Static", "Scaled_Static"),
        )
        if paired_df is not None and paired_output_path:
            paired_df.to_csv(paired_output_path, index=False)
            print(f"Saved fine-grid paired DS-vs-benchmark results to {paired_output_path}")

    if return_costs:
        return df, costs_by_policy, paired_df
    return df


def evaluate_discrete_review_policies(
    policies,
    bsde,
    num_paths,
    device,
    mu_review,
    gamma_review,
    demand_mode="auto",
    base_seed=0,
    output_path=None,
    use_common_random_numbers=True,
    paired_output_path=None,
    return_costs=False,
    shared_demands=None,
    paired_minuend_policy="Deep_Splitting",
):
    """Evaluate every policy in the original discrete-review model."""
    results = []
    costs_by_policy = {}
    review_dt = bsde.total_time / mu_review.shape[0]
    demand_mode = resolve_periodic_demand_mode(demand_mode, bsde=bsde, gamma_review=gamma_review)
    exposure = bsde.exposure.detach().cpu().numpy() if hasattr(bsde, "exposure") else None
    if shared_demands is not None and not use_common_random_numbers:
        raise ValueError("A frozen periodic-review bank requires common random numbers.")
    if shared_demands is None and use_common_random_numbers:
        # Shared period demands make DS-vs-benchmark differences pathwise paired.
        shared_demands = precompute_periodic_demands(
            mu_review,
            gamma_review,
            review_dt,
            num_paths,
            demand_mode=demand_mode,
            seed=base_seed,
            exposure=exposure,
        )
    if shared_demands is not None:
        expected_shape = (int(mu_review.shape[0]), int(num_paths), int(mu_review.shape[1]))
        if tuple(shared_demands.shape) != expected_shape:
            raise ValueError(
                f"shared_demands must have shape {expected_shape}, got {tuple(shared_demands.shape)}."
            )

    for idx, (policy_name, policy_fn) in enumerate(policies.items()):
        print(f"\nEvaluating {policy_name} policy on discrete-review grid...")
        costs_dict = simulate_periodic_review_trajectories(
            policy_fn,
            bsde,
            num_paths,
            device,
            mu_review,
            gamma_review,
            demand_mode=demand_mode,
            seed=base_seed + idx,
            precomputed_demands=shared_demands,
            exposure=exposure,
        )
        costs_by_policy[policy_name] = costs_dict
        results.append(summarize_cost_dict(policy_name, costs_dict))

    add_performance_gaps(results)
    df = pd.DataFrame(results)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved discrete-review results to {output_path}")

    paired_df = None
    if use_common_random_numbers:
        paired_df = compute_paired_difference_summary(
            costs_by_policy,
            minuend_policy=paired_minuend_policy,
            subtrahend_policy=("Static", "Scaled_Static"),
        )
        if paired_df is not None and paired_output_path:
            paired_df.to_csv(paired_output_path, index=False)
            print(f"Saved discrete-review paired DS-vs-benchmark results to {paired_output_path}")

    if return_costs:
        return df, costs_by_policy, paired_df
    return df


def evaluate_periodic_review_policies(*args, **kwargs):
    """Backward-compatible alias for the discrete-review evaluator."""
    return evaluate_discrete_review_policies(*args, **kwargs)


def print_summary_table(title, df):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(df.to_string(index=False))


def main():
    """CLI entry point used by scripts and ad-hoc evaluations."""
    print("Starting evaluation...")
    args = parse_arguments()

    context = prepare_evaluation_context(
        args.config_path,
        args.run_name,
        logs_path=args.logs_path,
        num_time_interval=args.num_time_interval,
        num_layers=args.num_layers,
        fluid_plan_path=args.fluid_plan_path,
        skip_fluid_benchmark=args.skip_fluid_benchmark,
        ds_extraction_modes=args.policy_extraction_modes,
    )
    config = context["config"]
    bsde = context["bsde"]
    device = context["device"]
    policies = context["policies"]

    print(f"Using device: {device}")

    if args.mode in ("fine_grid", "all"):
        df_fine, _, paired_fine_df = evaluate_fine_grid_policies(
            policies,
            bsde,
            args.num_paths,
            device,
            output_path=args.output_file,
            use_common_random_numbers=args.use_common_random_numbers,
            common_random_seed=args.fine_grid_seed,
            paired_output_path=args.paired_output_file,
            return_costs=True,
        )
        print_summary_table("Fine-Grid Evaluation Summary", df_fine)
        if paired_fine_df is not None:
            print_summary_table("Fine-Grid Paired DS vs Static", paired_fine_df)

    # "periodic_review" is accepted as a legacy alias for "discrete".
    if args.mode in ("periodic_review", "discrete", "all"):
        review_params = prepare_discrete_review_parameters(
            config,
            review_num_intervals=args.review_num_intervals,
            review_dt=args.review_dt,
        )
        if bsde.num_time_interval % review_params["num_review_intervals"] != 0:
            raise ValueError(
                "Discrete-review grid must divide the fine training grid exactly. "
                f"Got N_fine={bsde.num_time_interval}, N_review={review_params['num_review_intervals']}."
            )

        df_periodic, _, paired_periodic_df = evaluate_discrete_review_policies(
            policies,
            bsde,
            args.num_paths,
            device,
            review_params["mu_review"],
            review_params["gamma_review"],
            demand_mode=args.periodic_demand_mode,
            base_seed=args.periodic_seed,
            output_path=args.periodic_output_file,
            use_common_random_numbers=args.use_common_random_numbers,
            paired_output_path=args.periodic_paired_output_file,
            return_costs=True,
        )
        print_summary_table("Discrete-Review Evaluation Summary", df_periodic)
        if paired_periodic_df is not None:
            print_summary_table("Discrete-Review Paired DS vs Static", paired_periodic_df)


if __name__ == "__main__":
    main()
