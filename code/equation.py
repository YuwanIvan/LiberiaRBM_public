import os

import numpy as np
import pandas as pd
import torch


ETA_MODES = ("constant_peak", "weekly_time_varying")
REFERENCE_FLUID_POLICY = "fluid_optimal_rollout"
REFERENCE_MIXTURE_POLICY = "reference_mixture"
REFERENCE_POLICY_LABELS = {
    "R0": 0,
    "R1": 1,
    "R2": 2,
}
REFERENCE_POLICY_BY_LABEL = {
    0: "inventory_scaled_proportional",
    1: "static_proportional",
    2: REFERENCE_FLUID_POLICY,
}
REFERENCE_LABEL_BY_POLICY = {
    policy: label for label, policy in REFERENCE_POLICY_BY_LABEL.items()
}
FAIRNESS_SPEC = "demand_weighted_alpha_fair_v1"
FAIRNESS_INCLUDES_HF = True
FAIRNESS_PERIOD_H_FACTOR = False
DEFAULT_MU_EPSILON = 1e-8


def validate_fairness_parameters(alpha, lambda_param=None):
    """Validate the demand-weighted alpha-fair parameter range."""
    alpha_value = float(alpha)
    if not (0.0 < alpha_value < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha_value}.")
    if lambda_param is not None:
        lambda_value = float(lambda_param)
        if lambda_value < 0.0:
            raise ValueError(f"lambda_param must be nonnegative, got {lambda_value}.")
    return alpha_value


def fairness_metadata():
    """Metadata identifying the demand-weighted alpha-fair convention."""
    return {
        "fairness_spec": FAIRNESS_SPEC,
        "fairness_includes_hf": FAIRNESS_INCLUDES_HF,
        "fairness_period_h_factor": FAIRNESS_PERIOD_H_FACTOR,
    }


def alpha_fair_utility_torch(allocation, mu, alpha, mu_epsilon=DEFAULT_MU_EPSILON):
    """Demand-weighted alpha-fair utility with zero-demand locations masked out.

    ``allocation`` and ``mu`` must use the same unit: either period quantities
    for the discrete-review objective or rates for the continuous-time
    integrand. Locations with ``mu <= mu_epsilon`` contribute zero.
    """
    alpha = validate_fairness_parameters(alpha)
    allocation = torch.clamp(allocation, min=0.0)
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu, device=allocation.device, dtype=allocation.dtype)
    else:
        mu = mu.to(device=allocation.device, dtype=allocation.dtype)
    while mu.dim() < allocation.dim():
        mu = mu.unsqueeze(0)
    mu = mu.expand_as(allocation)
    active = mu > float(mu_epsilon)
    safe_mu = torch.where(active, mu, torch.ones_like(mu))
    ratio = torch.where(active, allocation / safe_mu, torch.zeros_like(allocation))
    contribution = torch.where(
        active,
        safe_mu * torch.pow(torch.clamp(ratio, min=0.0), 1.0 - alpha),
        torch.zeros_like(allocation),
    )
    return contribution.sum(dim=-1) / (1.0 - alpha)


def alpha_fair_utility_numpy(allocation, mu, alpha, mu_epsilon=DEFAULT_MU_EPSILON):
    """Numpy equivalent of :func:`alpha_fair_utility_torch`."""
    alpha = validate_fairness_parameters(alpha)
    allocation_arr = np.clip(np.asarray(allocation, dtype=np.float64), 0.0, None)
    mu_arr = np.asarray(mu, dtype=np.float64)
    if mu_arr.ndim < allocation_arr.ndim:
        expand_axes = (1,) * (allocation_arr.ndim - mu_arr.ndim)
        mu_arr = mu_arr.reshape(expand_axes + mu_arr.shape)
    mu_arr = np.broadcast_to(mu_arr, allocation_arr.shape)
    active = mu_arr > float(mu_epsilon)
    safe_mu = np.where(active, mu_arr, 1.0)
    ratio = np.where(active, allocation_arr / safe_mu, 0.0)
    contribution = np.where(
        active,
        safe_mu * np.power(np.clip(ratio, 0.0, None), 1.0 - alpha),
        0.0,
    )
    return contribution.sum(axis=-1) / (1.0 - alpha)


def demand_weighted_hamiltonian_value_torch(
    allocation,
    grad_diff,
    mu,
    lambda_param,
    alpha,
    mu_epsilon=DEFAULT_MU_EPSILON,
):
    """Return ``a dot x - lambda * Psi_alpha(a/mu; mu)`` componentwise summed."""
    validate_fairness_parameters(alpha, lambda_param)
    linear_term = (allocation * grad_diff).sum(dim=-1)
    if float(lambda_param) == 0.0:
        return linear_term
    utility = alpha_fair_utility_torch(allocation, mu, alpha, mu_epsilon=mu_epsilon)
    return linear_term - float(lambda_param) * utility


def resolve_config_path(filename, config_dir=None):
    """Resolve a config-relative or absolute path."""
    if os.path.isabs(filename):
        return filename
    if config_dir:
        config_path = os.path.join(config_dir, filename)
        if os.path.exists(config_path):
            return config_path
        if os.path.exists(filename):
            return filename
        return config_path

    if os.path.exists(filename):
        return filename

    for fallback_dir in (
        os.path.join("configs", "liberia"),
        os.path.join("configs", "liberia_empirical"),
    ):
        config_path = os.path.join(fallback_dir, filename)
        if os.path.exists(config_path):
            return config_path
    return filename


def load_mu_gamma_raw(eqn_config, dtype=torch.float32):
    """Load raw mu/Gamma rows before any time-grid expansion."""
    config_dir = getattr(eqn_config, "config_dir", None)
    dim = int(eqn_config.dim)

    mu_path = resolve_config_path(eqn_config.mu_file, config_dir)
    mu_raw = torch.tensor(pd.read_csv(mu_path, header=None).values, dtype=dtype)
    mu_raw_size = mu_raw.shape[0]

    gamma_path = resolve_config_path(eqn_config.gamma_file, config_dir)
    gamma_flat = pd.read_csv(gamma_path, header=None).values
    gamma_raw_size = gamma_flat.shape[0]
    gamma_num_cols = gamma_flat.shape[1]

    if mu_raw_size != gamma_raw_size:
        raise ValueError(
            f"mu file has {mu_raw_size} rows but gamma file has {gamma_raw_size} rows. They must match."
        )

    if gamma_num_cols == dim:
        gamma_raw = torch.zeros(gamma_raw_size, dim, dim, dtype=dtype)
        for t in range(gamma_raw_size):
            gamma_raw[t] = torch.diag(torch.tensor(gamma_flat[t, :], dtype=dtype))
    elif gamma_num_cols == dim * dim:
        gamma_raw = torch.tensor(gamma_flat.reshape(-1, dim, dim), dtype=dtype)
    else:
        raise ValueError(
            f"Gamma file has {gamma_num_cols} columns. Expected {dim} (diagonal only) "
            f"or {dim * dim} (full matrix flattened)."
        )

    return mu_raw, gamma_raw


def load_optional_vector(eqn_config, attr_name, config_dir=None, dtype=torch.float32):
    filename = getattr(eqn_config, attr_name, None)
    if not filename:
        return None
    path = resolve_config_path(filename, config_dir)
    if not os.path.exists(path):
        return None
    return torch.tensor(pd.read_csv(path, header=None).values.flatten(), dtype=dtype)


def _to_numpy_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _gamma_diagonal_array(gamma, dim):
    gamma_arr = np.asarray(_to_numpy_array(gamma), dtype=float)
    if gamma_arr.ndim == 2:
        if gamma_arr.shape[1] == dim:
            return gamma_arr
        if gamma_arr.shape[1] == dim * dim:
            return np.diagonal(gamma_arr.reshape(-1, dim, dim), axis1=1, axis2=2)
    if gamma_arr.ndim == 3 and gamma_arr.shape[1:] == (dim, dim):
        return np.diagonal(gamma_arr, axis1=1, axis2=2)
    raise ValueError(
        f"Gamma must be diagonal rows with {dim} columns or full {dim}x{dim} matrices."
    )


def compute_paper_eta(
    mu,
    gamma,
    p,
    I0,
    total_time,
    slack_multiplier=1.0,
    return_details=False,
):
    """Compute demand-scale shipment-rate bounds from the paper calibration."""
    mu_arr = np.asarray(_to_numpy_array(mu), dtype=float)
    if mu_arr.ndim != 2:
        raise ValueError("mu must be a 2D array with one row per time interval.")

    num_rows, dim = mu_arr.shape
    gamma_diag = _gamma_diagonal_array(gamma, dim)
    if gamma_diag.shape != mu_arr.shape:
        raise ValueError(
            f"Gamma diagonal shape {gamma_diag.shape} must match mu shape {mu_arr.shape}."
        )

    p_arr = np.asarray(_to_numpy_array(p), dtype=float).reshape(-1)
    if p_arr.shape[0] != dim - 1:
        raise ValueError(f"p must have {dim - 1} values, got {p_arr.shape[0]}.")

    step = float(total_time) / float(num_rows)
    total_expected_demand = float(mu_arr.sum() * step)
    if total_expected_demand <= 0.0:
        raise ValueError("Total expected demand must be positive to compute eta.")

    rho_raw = 1.0 - float(I0) / total_expected_demand
    rho_bar = max(0.0, rho_raw)
    m = float(slack_multiplier)

    eta = np.zeros(dim, dtype=float)
    eta[1:] = m * np.max(
        mu_arr[:, 1:] + 2.0 * np.sqrt(np.clip(gamma_diag[:, 1:], 0.0, None)),
        axis=0,
    )

    referral_mean = rho_bar * np.sum(mu_arr[:, 1:] * p_arr.reshape(1, -1), axis=1)
    referral_var = (rho_bar ** 2) * np.sum(
        gamma_diag[:, 1:] * (p_arr.reshape(1, -1) ** 2),
        axis=1,
    )
    eta[0] = m * np.max(
        mu_arr[:, 0]
        + referral_mean
        + 2.0 * np.sqrt(np.clip(gamma_diag[:, 0] + referral_var, 0.0, None))
    )

    if return_details:
        return eta, {
            "rho_bar": rho_bar,
            "rho_raw": rho_raw,
            "total_expected_demand": total_expected_demand,
            "slack_multiplier": m,
        }
    return eta


def compute_weekly_time_varying_eta_schedule(
    mu,
    gamma,
    p,
    I0,
    total_time,
    chi_eta=1.75,
    s_eta=0.0,
    return_details=False,
):
    """Compute the experimental weekly time-varying shipment-rate bounds."""
    mu_arr = np.asarray(_to_numpy_array(mu), dtype=float)
    if mu_arr.ndim != 2:
        raise ValueError("mu must be a 2D array with one row per review interval.")

    num_rows, dim = mu_arr.shape
    gamma_diag = _gamma_diagonal_array(gamma, dim)
    if gamma_diag.shape != mu_arr.shape:
        raise ValueError(
            f"Gamma diagonal shape {gamma_diag.shape} must match mu shape {mu_arr.shape}."
        )

    p_arr = np.asarray(_to_numpy_array(p), dtype=float).reshape(-1)
    if p_arr.shape[0] != dim - 1:
        raise ValueError(f"p must have {dim - 1} values, got {p_arr.shape[0]}.")
    chi_eta = float(chi_eta)
    s_eta = float(s_eta)
    if chi_eta < 0.0 or s_eta < 0.0:
        raise ValueError(f"chi_eta and s_eta must be nonnegative, got {chi_eta}, {s_eta}.")

    review_length = float(total_time) / float(num_rows)
    dbar = float(mu_arr.sum() * review_length)
    if dbar <= 0.0:
        raise ValueError("Total expected demand must be positive to compute eta schedule.")
    rho_raw = 1.0 - float(I0) / dbar
    rho_q = max(0.0, rho_raw)

    clipped_diag = np.clip(gamma_diag, 0.0, None)
    eta_schedule = np.zeros((num_rows, dim), dtype=float)
    eta_schedule[:, 1:] = chi_eta * mu_arr[:, 1:] + s_eta * np.sqrt(clipped_diag[:, 1:])

    referral_mean = rho_q * np.sum(mu_arr[:, 1:] * p_arr.reshape(1, -1), axis=1)
    referral_var = (rho_q ** 2) * np.sum(
        clipped_diag[:, 1:] * (p_arr.reshape(1, -1) ** 2),
        axis=1,
    )
    eta_schedule[:, 0] = (
        chi_eta * (mu_arr[:, 0] + referral_mean)
        + s_eta * np.sqrt(np.clip(clipped_diag[:, 0] + referral_var, 0.0, None))
    )

    if return_details:
        return eta_schedule, {
            "rho_Q": rho_q,
            "rho_raw": rho_raw,
            "total_expected_demand": dbar,
            "review_length": review_length,
            "chi_eta": chi_eta,
            "s_eta": s_eta,
        }
    return eta_schedule


def project_rate_to_inventory_torch(
    a_rate,
    inventory,
    delta_t,
    abs_tol=1e-8,
    rel_tol=1e-6,
    eta_rate=None,
    return_diagnostics=False,
):
    """Project per-path allocation rates onto the one-step inventory-feasible set.

    The production rule is the historical proportional projection: solve the
    componentwise Hamiltonian first, then scale all positive shipment rates on a
    path if their one-step total exceeds available warehouse inventory. This is
    not the exact coupled constrained Hamiltonian minimizer; diagnostics expose
    when this approximation is active.
    """
    if inventory.dim() == 2:
        inventory = inventory.squeeze(1)

    remaining = torch.clamp(inventory, min=0.0)
    tol = torch.maximum(
        torch.full_like(remaining, abs_tol),
        rel_tol * torch.maximum(remaining, torch.ones_like(remaining)),
    )

    projected = torch.clamp(a_rate, min=0.0)
    if eta_rate is not None:
        if not isinstance(eta_rate, torch.Tensor):
            eta_rate = torch.tensor(eta_rate, device=projected.device, dtype=projected.dtype)
        else:
            eta_rate = eta_rate.to(device=projected.device, dtype=projected.dtype)
        if eta_rate.dim() == 0:
            eta_rate = eta_rate.expand(projected.shape[1])
        if eta_rate.dim() == 1:
            eta_rate = eta_rate.unsqueeze(0)
        projected = torch.minimum(projected, eta_rate.expand_as(projected))

    inactive = remaining <= tol
    scale_factor = torch.ones_like(remaining)

    if inactive.any():
        projected[inactive] = 0.0

    q_sum = projected.sum(dim=1) * delta_t
    needs_scale = q_sum > (remaining + tol)
    if needs_scale.any():
        denom = torch.clamp(q_sum[needs_scale], min=1e-12)
        scale_values = remaining[needs_scale] / denom
        scale_factor[needs_scale] = scale_values
        scale = scale_values.unsqueeze(1)
        projected[needs_scale] = projected[needs_scale] * scale

    if not return_diagnostics:
        return projected, tol

    active_scale = scale_factor[needs_scale]
    diagnostics = {
        "projection_mask": needs_scale,
        "zero_inventory_mask": inactive,
        "projection_count": int(needs_scale.detach().sum().item()),
        "zero_inventory_count": int(inactive.detach().sum().item()),
        "projection_frequency": float(needs_scale.detach().float().mean().item())
        if needs_scale.numel()
        else 0.0,
        "projection_scale": scale_factor,
        "mean_projection_scale": float(active_scale.detach().mean().item())
        if active_scale.numel()
        else 1.0,
        "min_projection_scale": float(active_scale.detach().min().item())
        if active_scale.numel()
        else 1.0,
    }
    return projected, tol, diagnostics


def build_unprojected_ds_allocation(
    grad_diff,
    inventory,
    mu_t,
    eta,
    lambda_param,
    alpha,
    delta_t,
    mu_epsilon=DEFAULT_MU_EPSILON,
):
    """Recover the closed-form Deep Splitting allocation rate before inventory projection."""
    alpha = validate_fairness_parameters(alpha, lambda_param)
    batch_size, dim = grad_diff.shape
    device = grad_diff.device
    dtype = grad_diff.dtype

    if inventory.dim() == 2:
        inventory = inventory.squeeze(1)

    mu_t = mu_t.to(device=device, dtype=dtype)
    eta = eta.to(device=device, dtype=dtype)
    if eta.dim() == 0:
        eta = eta.expand(dim)
    if eta.dim() == 2:
        if eta.shape[0] != 1:
            raise ValueError(
                "build_feasible_ds_allocation expects a single eta row; "
                f"got eta with shape {tuple(eta.shape)}."
            )
        eta = eta.squeeze(0)
    eta_broadcast = eta.unsqueeze(0).expand(batch_size, -1)
    a_star = torch.zeros(batch_size, dim, device=device, dtype=dtype)

    remaining = torch.clamp(inventory, min=0.0)
    inventory_tol = torch.maximum(
        torch.full_like(remaining, 1e-8),
        1e-6 * torch.maximum(remaining, torch.ones_like(remaining)),
    )
    active_mask = remaining > inventory_tol

    if active_mask.any():
        grad_active = grad_diff[active_mask]
        eta_active = eta_broadcast[active_mask]
        mu_active = mu_t.unsqueeze(0).expand(grad_active.shape[0], -1)
        eta_positive = eta_active > 0.0
        positive_mu = mu_active > float(mu_epsilon)
        zero_mu_linear = eta_positive & (~positive_mu)

        if float(lambda_param) == 0.0:
            # For zero demand the fairness term is zero and the Hamiltonian is
            # linear in a. At x == 0 we use the inventory-conserving selector
            # a=0. Positive-mu lambda=0 keeps the historical bang-bang x<=0 rule.
            a_active = torch.where(
                (eta_positive & positive_mu & (grad_active <= 0.0))
                | (zero_mu_linear & (grad_active < 0.0)),
                eta_active,
                torch.zeros_like(eta_active),
            )
        else:
            a_active = torch.zeros_like(eta_active)
            # Zero-demand components are again linear; use the same documented
            # selector at x == 0 as in the lambda=0 branch.
            a_active = torch.where(
                zero_mu_linear & (grad_active < 0.0),
                eta_active,
                a_active,
            )
            threshold = torch.where(
                eta_positive & positive_mu,
                float(lambda_param)
                * torch.pow(mu_active / torch.clamp(eta_active, min=float(mu_epsilon)), alpha),
                torch.zeros_like(eta_active),
            )
            full_capacity = eta_positive & positive_mu & (grad_active <= threshold)
            a_active = torch.where(full_capacity, eta_active, a_active)
            interior = eta_positive & positive_mu & (grad_active > threshold)
            if interior.any():
                safe_grad = torch.clamp(grad_active, min=float(mu_epsilon))
                interior_value = mu_active * torch.pow(float(lambda_param) / safe_grad, 1.0 / alpha)
                a_active = torch.where(interior, interior_value, a_active)

        a_star[active_mask] = a_active

    a_star = torch.minimum(a_star, eta_broadcast)
    return torch.clamp(a_star, min=0.0)


def build_feasible_ds_allocation(
    grad_diff,
    inventory,
    mu_t,
    eta,
    lambda_param,
    alpha,
    delta_t,
    mu_epsilon=DEFAULT_MU_EPSILON,
):
    """Recover the Deep Splitting control and enforce inventory feasibility pathwise."""
    a_raw = build_unprojected_ds_allocation(
        grad_diff,
        inventory,
        mu_t,
        eta,
        lambda_param,
        alpha,
        delta_t,
        mu_epsilon=mu_epsilon,
    )
    a_projected, _ = project_rate_to_inventory_torch(
        a_raw,
        inventory,
        delta_t,
        eta_rate=eta,
    )
    return torch.clamp(a_projected, min=0.0)


def step_transition_torch(
    Z,
    Q,
    allocation_rate,
    delta_w,
    delta_t,
    mu_t,
    p,
    clamp_inventory=True,
):
    """One Euler StepTransition for the Liberia reflected inventory process.

    Inputs use the same units as the production fine-grid simulator:
    ``allocation_rate`` and ``mu_t`` are rates, ``delta_w`` is an inventory
    increment, and ``delta_t`` is the fine-step length.
    """
    if Q.dim() == 2:
        Q = Q.squeeze(1)

    mu_t = mu_t.to(device=Z.device, dtype=Z.dtype)
    p = p.to(device=Z.device, dtype=Z.dtype)
    delta_w = delta_w.to(device=Z.device, dtype=Z.dtype)

    X_next = Z + (allocation_rate - mu_t.unsqueeze(0)) * float(delta_t) + delta_w
    delta_l = torch.zeros_like(X_next)
    delta_l_cha = torch.relu(-X_next[:, 1:])
    delta_l[:, 1:] = delta_l_cha

    referral_load = torch.sum(p.unsqueeze(0) * delta_l_cha, dim=1)
    effective_hf_inventory = X_next[:, 0] - referral_load
    delta_l[:, 0] = torch.relu(-effective_hf_inventory)

    Z_next = torch.zeros_like(X_next)
    Z_next[:, 0] = X_next[:, 0] + delta_l[:, 0] - referral_load
    Z_next[:, 1:] = X_next[:, 1:] + delta_l[:, 1:]

    Q_next = Q - allocation_rate.sum(dim=1) * float(delta_t)
    if clamp_inventory:
        Q_next = torch.clamp(Q_next, min=0.0)
    return Z_next, Q_next, delta_l


def load_open_loop_allocation_plan(path, dim, total_time, require_fairness_spec=False):
    """Load deterministic open-loop allocation quantities from a fluid artifact."""
    resolved_path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Open-loop allocation plan not found: {resolved_path}")

    with np.load(resolved_path) as data:
        if "allocation_plan" not in data:
            raise ValueError(f"Allocation artifact {resolved_path} does not contain allocation_plan.")
        if "fairness_spec" in data:
            fairness_spec = str(np.asarray(data["fairness_spec"]).reshape(-1)[0])
            if fairness_spec != FAIRNESS_SPEC:
                raise ValueError(
                    f"Allocation artifact {resolved_path} fairness_spec={fairness_spec!r}; "
                    f"expected {FAIRNESS_SPEC!r}."
                )
        elif require_fairness_spec:
            raise ValueError(
                f"Allocation artifact {resolved_path} has no fairness_spec metadata and cannot be reused "
                f"for lambda-positive demand-weighted alpha-fair runs; expected {FAIRNESS_SPEC!r}."
            )
        plan = np.asarray(data["allocation_plan"], dtype=np.float64)
        if "review_dt" in data:
            review_dt = float(np.asarray(data["review_dt"]).reshape(-1)[0])
        else:
            review_dt = float(total_time) / float(plan.shape[0])

    if plan.ndim != 2:
        raise ValueError(f"Allocation plan must be 2D, got shape {plan.shape}.")
    if plan.shape[1] != int(dim):
        raise ValueError(f"Allocation plan has {plan.shape[1]} locations, expected {int(dim)}.")
    if plan.shape[0] <= 0:
        raise ValueError("Allocation plan must contain at least one review interval.")
    if review_dt <= 0.0:
        raise ValueError(f"Allocation plan has nonpositive review_dt={review_dt}.")
    if not np.isclose(review_dt * plan.shape[0], float(total_time), rtol=1e-7, atol=1e-7):
        raise ValueError(
            "Allocation plan review grid does not cover total_time: "
            f"rows={plan.shape[0]}, review_dt={review_dt}, total_time={total_time}."
        )
    if np.any(plan < -1e-8):
        raise ValueError(f"Allocation plan {resolved_path} contains negative shipment quantities.")

    return np.clip(plan, 0.0, None), review_dt, resolved_path


class Equation(object):
    """Base class for defining PDE-related functions."""

    def __init__(self, eqn_config):

        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)

    
    def f_tf(self, x, a_lowbound, mu, theta, cost,  z, u):
        """Generator function in the PDE"""
        raise NotImplementedError
    
    def g_tf(self, x):
        """Terminal condition of the PDE"""
        raise NotImplementedError

class HJB(Equation):

    def __init__(self, eqn_config):
        super(HJB, self).__init__(eqn_config)
        self.policy = eqn_config.policy
        self.overtime_cost = eqn_config.overtime_cost
    
    def generate_u_sample(self, num_sample):
        """Generate u_sample based on the specified policy."""

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.policy == "even":
            # Uniform distribution
            u_sample = torch.ones(num_sample, self.dim).to(device) * 1 / self.dim 

        elif self.policy == "minimal":
            # All-zero policy
            u_sample = torch.zeros(num_sample, self.dim).to(device) 

        elif self.policy == "best":
            # All zeros except one index set to 1
            u_sample = torch.zeros(num_sample, self.dim).to(device) 
            u_sample[:,1] = 1

        elif self.policy == "best_var":
            # All zeros except one index set to 1
            u_sample = torch.zeros(num_sample, self.dim).to(device)
            u_sample[:,0] = 1

        elif self.policy == "weighted_split_main":
            # Weighted split policy for the main test problem
            u_sample = torch.ones(num_sample, self.dim).to(device) * 0.037178922
            u_sample[:,0:3] = 0.159803922
        
        elif self.policy == "weighted_split_var1":
            # Weighted split policy for the variant 1
            alpha = 4/6
            base_weight = alpha * 0.025 + (1 - alpha) * 1/self.dim
            special_weight = alpha * 0.14 + (1 - alpha) * 1/self.dim

            u_sample = torch.ones(num_sample, self.dim).to(device) * base_weight
            u_sample[:,0] = special_weight
            u_sample[:,1] = special_weight
            u_sample[:,2] = special_weight
            u_sample[:,6] = special_weight
            u_sample[:,14] = special_weight

        elif self.policy == "weighted_split_var2":
            # Weighted split policy for the variant 2
            alpha = 3/6
            base_weight = alpha * 0.025 + (1 - alpha) * 1/self.dim
            special_weight = alpha * 0.14 + (1 - alpha) * 1/self.dim

            u_sample = torch.ones(num_sample, self.dim).to(device) * base_weight
            u_sample[:,0] = special_weight
            u_sample[:,1] = special_weight
            u_sample[:,2] = special_weight
            u_sample[:,6] = special_weight
            u_sample[:,14] = special_weight
        

        return u_sample

    def compute_mx(self, x, a_lowbound):
        """Compute mx based on input x and a lower bound."""
        sum_x = torch.sum(x, dim = 1, keepdim = True)
        sum_x_p = torch.clamp(sum_x, min = 0.0)
        sum_x_n = torch.clamp(sum_x, max = 0.0)
        return sum_x_p + a_lowbound * (torch.exp(sum_x_n) - 1)

        
    def f_tf(self, x, a_lowbound, mu, theta, cost,  z, u):
        """Generator function in the PDE."""
        mx = self.compute_mx(x, a_lowbound)
        first_term = torch.sum((mu - theta) * z * u, dim = 1, keepdim = True)
        second_term = torch.min(cost + (mu - theta) * z, dim = 1, keepdim = True)[0]
        return mx * (first_term - second_term)


    def g_tf(self, x):
        """Terminal condition of the PDE."""
        terminal = self.overtime_cost * torch.max(torch.sum(x, dim=1, keepdim = True), torch.tensor(0.0))
        return terminal


class LiberiaHJB(Equation):
    """
    Liberia Inventory Model HJB equation class.
    Implements the two-echelon inventory management system for healthcare distribution.
    """
    
    def __init__(self, eqn_config):
        super(LiberiaHJB, self).__init__(eqn_config)
        
        # Load parameters from config
        self.K = eqn_config.K  # Number of CHAs
        self.lambda_param = float(eqn_config.lambda_param)  # Fairness objective weight
        self.alpha = validate_fairness_parameters(
            eqn_config.alpha,
            self.lambda_param,
        )  # Fairness objective concavity parameter
        configured_fairness_spec = getattr(eqn_config, "fairness_spec", FAIRNESS_SPEC)
        if configured_fairness_spec != FAIRNESS_SPEC:
            raise ValueError(
                f"Unsupported fairness_spec={configured_fairness_spec!r}; expected {FAIRNESS_SPEC!r}."
            )
        self.fairness_spec = FAIRNESS_SPEC
        self.fairness_includes_hf = FAIRNESS_INCLUDES_HF
        self.fairness_period_h_factor = FAIRNESS_PERIOD_H_FACTOR
        self.calibration_version = getattr(eqn_config, "calibration_version", None)
        self.I0 = eqn_config.I0  # Initial unallocated inventory
        
        # Load data files
        config_dir = getattr(eqn_config, 'config_dir', None)
        self.config_dir = config_dir
        self.eta_schedule_file = getattr(eqn_config, "eta_schedule_file", None)
        self.eta_schedule_file_role = getattr(eqn_config, "eta_schedule_file_role", "active_input")
        mu_raw, gamma_raw = load_mu_gamma_raw(eqn_config, dtype=torch.float32)
        self.mu_raw = mu_raw.clone()
        self.Gamma_raw = gamma_raw.clone()
        mu_raw_size = mu_raw.shape[0]
        gamma_raw_size = gamma_raw.shape[0]
        self.raw_num_time_interval = int(mu_raw_size)
        self.review_length = float(self.total_time) / float(mu_raw_size)
        self.raw_delta_t = self.review_length
        
        # Repeat the parameter rows when the fine grid subdivides the source intervals.
        if self.num_time_interval % mu_raw_size == 0:
            repeat_factor = self.num_time_interval // mu_raw_size
            if repeat_factor == 1:
                # Use the source rows directly when the grids have equal size.
                self.mu = mu_raw
                self.Gamma = gamma_raw
            else:
                # Repeat each mu row repeat_factor times.
                # Consecutive fine intervals use the same weekly parameters.
                self.mu = mu_raw.repeat_interleave(repeat_factor, dim=0)
                # Repeat each Gamma row repeat_factor times.
                self.Gamma = gamma_raw.repeat_interleave(repeat_factor, dim=0)
                print(f"Data repetition: each of {mu_raw_size} rows repeated {repeat_factor} times to get {self.num_time_interval} intervals")
        else:
            raise ValueError(
                f"num_time_interval ({self.num_time_interval}) must be equal to or a multiple of "
                f"data file rows ({mu_raw_size}). Got {self.num_time_interval} intervals but data has {mu_raw_size} rows."
            )
        
        # Load p_k - shape: (K,)
        p_path = resolve_config_path(eqn_config.p_file, config_dir)
        self.p = torch.tensor(pd.read_csv(p_path, header=None).values.flatten(), dtype=torch.float32)

        # Current Fangonda configs store c directly as the effective shortage
        # vector, and store u/r/distance separately for objective decomposition.
        if not hasattr(eqn_config, 'c_file'):
            raise ValueError("Must provide c_file in config.")
        c_path = resolve_config_path(eqn_config.c_file, config_dir)
        c_data = pd.read_csv(c_path, header=None).values.flatten()  # (K+1,)
        if len(c_data) != self.dim:
            raise ValueError(f"c file has {len(c_data)} values, expected {self.dim} (K+1)")
        c_tensor = torch.tensor(c_data, dtype=torch.float32)

        u_tensor = load_optional_vector(eqn_config, "u_file", config_dir)
        r_tensor = load_optional_vector(eqn_config, "r_file", config_dir)
        distance_tensor = load_optional_vector(eqn_config, "distance_file", config_dir)
        exposure_tensor = load_optional_vector(eqn_config, "exposure_file", config_dir)
        theta_tensor = load_optional_vector(eqn_config, "theta_file", config_dir)

        if u_tensor is not None and r_tensor is not None:
            if u_tensor.numel() != 1:
                raise ValueError(f"u file has {u_tensor.numel()} values, expected 1")
            if r_tensor.numel() != self.dim:
                raise ValueError(f"r file has {r_tensor.numel()} values, expected {self.dim}")
            self.u = u_tensor.reshape(1)[0]
            self.r = r_tensor
            self.c = c_tensor
            self.c_tilde = c_tensor.clone()
            self.distance = (
                distance_tensor
                if distance_tensor is not None and distance_tensor.numel() == self.dim
                else torch.zeros(self.dim, dtype=torch.float32)
            )
            print(f"Loaded effective shortage costs from c_file: {self.c_tilde.tolist()}")
        else:
            # Backward compatibility for older configs: c_file stored untreated
            # costs and d_file stored referral costs.
            if not hasattr(eqn_config, 'd_file'):
                raise ValueError("Old-style configs must provide d_file when u_file/r_file are absent.")
            d_path = resolve_config_path(eqn_config.d_file, config_dir)
            d_data = pd.read_csv(d_path, header=None).values.flatten()
            if len(d_data) != self.dim:
                raise ValueError(f"d file has {len(d_data)} values, expected {self.dim} (K+1)")
            d_tensor = torch.tensor(d_data, dtype=torch.float32)
            self.u = c_tensor[0]
            self.r = d_tensor
            self.c = c_tensor
            self.c_tilde = torch.zeros(self.dim, dtype=torch.float32)
            self.c_tilde[0] = c_tensor[0]
            self.c_tilde[1:] = d_tensor[1:] * self.p + c_tensor[1:] * (1.0 - self.p)
            self.distance = d_tensor
            print(f"Computed legacy c_tilde from c_file and d_file.")

        # Historical code used bsde.d as the referral-cost vector.
        self.d = self.r
        if exposure_tensor is not None and exposure_tensor.numel() == self.dim:
            self.exposure = exposure_tensor
        else:
            self.exposure = torch.zeros(self.dim, dtype=torch.float32)
        self.theta = theta_tensor if theta_tensor is not None else torch.empty(0, dtype=torch.float32)
        self.periodic_demand_mode = getattr(eqn_config, "periodic_demand_mode", "auto")
        
        # Load h_k - shape: (K+1,)
        h_path = resolve_config_path(eqn_config.h_file, config_dir)
        self.h = torch.tensor(pd.read_csv(h_path, header=None).values.flatten(), dtype=torch.float32)
        
        # If eta_file or eta is explicitly provided, use it; otherwise compute from formula
        if hasattr(eqn_config, 'eta_file') and eqn_config.eta_file:
            # Load from file (K+1 values, one per location) - explicit override
            eta_path = resolve_config_path(eqn_config.eta_file, config_dir)
            eta_data = pd.read_csv(eta_path, header=None).values.flatten()
            if len(eta_data) != self.dim:
                raise ValueError(f"eta file has {len(eta_data)} values, expected {self.dim} (K+1)")
            self.eta = torch.tensor(eta_data, dtype=torch.float32)  # (K+1,)
            print(f"Loaded eta from file: {self.eta.tolist()}")
        elif hasattr(eqn_config, 'eta') and isinstance(eqn_config.eta, (list, tuple)):
            # Vector from config - explicit override
            if len(eqn_config.eta) != self.dim:
                raise ValueError(f"eta vector has {len(eqn_config.eta)} values, expected {self.dim} (K+1)")
            self.eta = torch.tensor(eqn_config.eta, dtype=torch.float32)  # (K+1,)
            print(f"Loaded eta from config vector: {self.eta.tolist()}")
        elif hasattr(eqn_config, 'eta') and eqn_config.eta is not None:
            # Scalar from config - explicit override (backward compatibility)
            eta_scalar = float(eqn_config.eta)
            # Expand to vector with same value for all locations
            self.eta = torch.ones(self.dim, dtype=torch.float32) * eta_scalar  # (K+1,)
            print(f"Using scalar eta={eta_scalar} for all locations")
        else:
            eta, eta_details = compute_paper_eta(
                self.mu,
                self.Gamma,
                self.p,
                self.I0,
                self.total_time,
                slack_multiplier=1.0,
                return_details=True,
            )
            self.eta = torch.tensor(eta, dtype=torch.float32)
            print(
                "Auto-computed eta from demand-scale paper formula "
                f"(m=1, rho_bar={eta_details['rho_bar']:.6f}): {self.eta.tolist()}"
            )

        self.eta_mode = getattr(eqn_config, "eta_mode", getattr(eqn_config, "eta_profile", "constant_peak"))
        self.eta_profile = self.eta_mode
        self.chi_eta = float(getattr(eqn_config, "chi_eta", 1.75))
        self.s_eta = float(getattr(eqn_config, "s_eta", 0.0))
        if self.eta_mode not in ETA_MODES:
            raise ValueError(f"eta_mode must be one of {ETA_MODES}, got {self.eta_mode!r}.")
        if self.chi_eta < 0.0 or self.s_eta < 0.0:
            raise ValueError(f"chi_eta and s_eta must be nonnegative, got {self.chi_eta}, {self.s_eta}.")
        self._build_eta_schedule()
        if self.eta_mode == "weekly_time_varying":
            self.eta_constant_input = self.eta.clone()
            self.eta = self.eta_schedule_raw.max(dim=0).values.clone()
            self.eta_vector_role = "peak_diagnostic_vector_max_over_review_schedule"
        else:
            self.eta_constant_input = self.eta.clone()
            self.eta_vector_role = "active_constant_rate_bound"
        self._print_eta_diagnostics()
        
        # Build reflection matrix R (equation 202-209)
        self.R = self._build_reflection_matrix()
        
        # Build Omega_1 and Omega_2 matrices (equations 372-383)
        self.Omega_1, self.Omega_2 = self._build_omega_matrices()
        
        # Policy for reference process
        self.policy = getattr(eqn_config, 'policy', 'even')  # Default to 'even' policy
        
        # Myopic policy threshold (epsilon) - default 0 (paper); tiny tolerance added in code
        self.myopic_epsilon = getattr(eqn_config, 'myopic_epsilon', 0.0)

        self.reference_fluid_plan_path = None
        self.reference_fluid_plan_quantity = None
        self.reference_fluid_rate_plan = None
        self.reference_fluid_review_dt = None
        self.reference_fluid_plan_metadata = {}
        self._load_reference_fluid_plan_if_needed(eqn_config, config_dir)
        self._build_reference_policy_caches()

    def _build_eta_schedule(self):
        """Build raw-review and fine-grid eta schedules for the selected eta mode."""
        repeat_factor = int(self.num_time_interval // self.raw_num_time_interval)
        self.eta_schedule_file_status = "not_configured"
        self.eta_schedule_file_path = None
        if self.eta_mode == "constant_peak":
            self.eta_schedule_raw = self.eta.detach().cpu().unsqueeze(0).repeat(
                self.raw_num_time_interval,
                1,
            ).to(dtype=torch.float32)
            self.eta_schedule_details = {
                "eta_rule": "constant_peak",
                "source": "eta",
                "repeat_factor": repeat_factor,
            }
        elif self.eta_mode == "weekly_time_varying":
            if self.raw_num_time_interval != 22:
                raise ValueError(
                    "weekly_time_varying eta is currently defined for the 22-week Fangonda grid; "
                    f"got {self.raw_num_time_interval} raw intervals."
                )
            if not np.isclose(float(self.review_length), 1.0, atol=1e-10, rtol=0.0):
                raise ValueError(
                    "weekly_time_varying eta expects one-week review intervals; "
                    f"got review_length={self.review_length}."
                )
            formula_schedule, details = compute_weekly_time_varying_eta_schedule(
                self.mu_raw,
                self.Gamma_raw,
                self.p,
                self.I0,
                self.total_time,
                chi_eta=self.chi_eta,
                s_eta=self.s_eta,
                return_details=True,
            )
            formula_tensor = torch.tensor(formula_schedule, dtype=torch.float32)
            file_tensor = self._load_eta_schedule_file_tensor()
            if file_tensor is not None and file_tensor.shape == (self.raw_num_time_interval, self.dim):
                self.eta_schedule_raw = file_tensor.to(dtype=torch.float32)
                file_np = self.eta_schedule_raw.detach().cpu().numpy()
                formula_np = formula_tensor.detach().cpu().numpy()
                max_abs_diff = float(np.max(np.abs(file_np - formula_np)))
                if np.allclose(file_np, formula_np, rtol=1e-7, atol=1e-7):
                    self.eta_schedule_file_status = "matches_computed_schedule"
                else:
                    self.eta_schedule_file_status = "active_file_differs_from_computed_schedule"
                    print(
                        "Warning: eta_schedule_file differs from the formula schedule "
                        f"(max_abs_diff={max_abs_diff:.6g}); using eta_schedule_file as the active schedule."
                    )
                source = "eta_schedule_file"
            else:
                self.eta_schedule_raw = formula_tensor
                max_abs_diff = 0.0
                source = "mu_gamma_formula"
            self.eta_schedule_details = {
                "eta_rule": "weekly_time_varying",
                "source": source,
                "eta_schedule_file_role": str(self.eta_schedule_file_role),
                "formula_source": "mu_gamma_formula",
                "formula_max_abs_diff": max_abs_diff,
                "repeat_factor": repeat_factor,
                **details,
            }
        else:  # pragma: no cover - guarded above
            raise ValueError(f"Unsupported eta_mode={self.eta_mode!r}.")

        if self.eta_schedule_raw.shape != (self.raw_num_time_interval, self.dim):
            raise ValueError(
                "eta_schedule_raw has wrong shape: "
                f"{tuple(self.eta_schedule_raw.shape)}, expected {(self.raw_num_time_interval, self.dim)}."
            )
        self.eta_schedule = self.eta_schedule_raw.repeat_interleave(repeat_factor, dim=0)
        if self.eta_schedule.shape != (self.num_time_interval, self.dim):
            raise ValueError(
                "eta_schedule has wrong shape: "
                f"{tuple(self.eta_schedule.shape)}, expected {(self.num_time_interval, self.dim)}."
            )

    def _load_eta_schedule_file_tensor(self):
        """Load a configured eta_schedule_file for use as the active weekly schedule."""
        if self.eta_mode != "weekly_time_varying":
            self.eta_schedule_file_status = "not_active_for_constant_peak"
            return None
        filename = self.eta_schedule_file
        if not filename:
            self.eta_schedule_file_status = "not_configured_formula_used"
            return None
        path = resolve_config_path(filename, self.config_dir)
        self.eta_schedule_file_path = path
        if not os.path.exists(path):
            self.eta_schedule_file_status = "missing_snapshot_computed_schedule_used"
            return None
        try:
            snapshot = pd.read_csv(path, header=None).values
        except Exception as exc:
            self.eta_schedule_file_status = f"unreadable_snapshot_computed_schedule_used: {exc}"
            print(f"Warning: could not read eta_schedule_file snapshot {path}: {exc}")
            return None
        if snapshot.shape != (self.raw_num_time_interval, self.dim):
            self.eta_schedule_file_status = (
                "shape_mismatch_snapshot_ignored_computed_schedule_used"
            )
            print(
                "Warning: eta_schedule_file snapshot shape "
                f"{snapshot.shape} differs from expected shape {(self.raw_num_time_interval, self.dim)}; "
                "using the computed eta schedule."
            )
            return None
        return torch.tensor(snapshot, dtype=torch.float32)

    def eta_at_fine_index(self, n):
        """Return the shipment-rate bound row for fine-grid interval n."""
        if n is None:
            return self.eta
        idx = max(0, min(int(n), int(self.eta_schedule.shape[0]) - 1))
        return self.eta_schedule[idx]

    def eta_at_review_index(self, n):
        """Return the shipment-rate bound row for raw review interval n."""
        idx = max(0, min(int(n), int(self.eta_schedule_raw.shape[0]) - 1))
        return self.eta_schedule_raw[idx]

    def eta_for_review_grid(self, num_review_intervals):
        """Return eta rates aligned to a requested nested review grid."""
        num_review_intervals = int(num_review_intervals)
        if num_review_intervals <= 0:
            raise ValueError("num_review_intervals must be positive.")
        raw_rows = int(self.eta_schedule_raw.shape[0])
        if num_review_intervals == raw_rows:
            return self.eta_schedule_raw
        if self.eta_mode == "constant_peak":
            return self.eta.detach().cpu().unsqueeze(0).repeat(num_review_intervals, 1).to(
                dtype=self.eta_schedule_raw.dtype
            )
        if num_review_intervals % raw_rows == 0:
            repeat_factor = num_review_intervals // raw_rows
            return self.eta_schedule_raw.repeat_interleave(repeat_factor, dim=0)
        if raw_rows % num_review_intervals == 0:
            block = raw_rows // num_review_intervals
            return self.eta_schedule_raw.reshape(num_review_intervals, block, self.dim).mean(dim=1)
        raise ValueError(
            "Review grid must nest the raw eta grid for weekly_time_varying eta. "
            f"Got raw_rows={raw_rows}, num_review_intervals={num_review_intervals}."
        )

    def eta_diagnostics(self):
        """Return JSON-serializable eta-schedule diagnostics."""
        eta_raw = self.eta_schedule_raw.detach().cpu().numpy()
        eta_fine = self.eta_schedule.detach().cpu().numpy()
        return {
            "eta_mode": str(self.eta_mode),
            "eta_profile": str(self.eta_profile),
            "chi_eta": float(self.chi_eta),
            "s_eta": float(self.s_eta),
            "eta_vector_role": str(getattr(self, "eta_vector_role", "")),
            "eta_peak_diagnostic_vector": self.eta.detach().cpu().numpy().tolist(),
            "eta_constant_input_vector": self.eta_constant_input.detach().cpu().numpy().tolist()
            if hasattr(self, "eta_constant_input")
            else None,
            "eta_schedule_file": self.eta_schedule_file,
            "eta_schedule_file_path": self.eta_schedule_file_path,
            "eta_schedule_file_status": self.eta_schedule_file_status,
            "eta_schedule_raw_shape": list(eta_raw.shape),
            "eta_schedule_shape": list(eta_fine.shape),
            "min_by_location": eta_raw.min(axis=0).tolist(),
            "max_by_location": eta_raw.max(axis=0).tolist(),
            "min_by_week": eta_raw.min(axis=1).tolist(),
            "max_by_week": eta_raw.max(axis=1).tolist(),
            "all_weeks_identical": bool(np.allclose(eta_raw, eta_raw[0:1])),
            "cha_profiles_identical": bool(
                eta_raw.shape[1] <= 2 or np.allclose(eta_raw[:, 1:], eta_raw[:, 1:2])
            ),
            "details": dict(self.eta_schedule_details),
        }

    def _print_eta_diagnostics(self):
        diag = self.eta_diagnostics()
        print(
            "Eta mode: "
            f"{diag['eta_mode']} | chi_eta={diag['chi_eta']} | s_eta={diag['s_eta']} | "
            f"raw shape={diag['eta_schedule_raw_shape']} | "
            f"fine shape={diag['eta_schedule_shape']}"
        )
        print(f"Eta min by location: {diag['min_by_location']}")
        print(f"Eta max by location: {diag['max_by_location']}")
        print(
            "Eta weekly min/max ranges: "
            f"min={min(diag['min_by_week']):.6f}..{max(diag['min_by_week']):.6f}, "
            f"max={min(diag['max_by_week']):.6f}..{max(diag['max_by_week']):.6f}"
        )

    def _load_reference_fluid_plan_if_needed(self, eqn_config, config_dir):
        """Attach a precomputed deterministic fluid rollout plan for training-state sampling."""
        plan_value = (
            getattr(eqn_config, "reference_fluid_plan_path", None)
            or getattr(eqn_config, "reference_fluid_plan_file", None)
        )
        if self.policy not in {REFERENCE_FLUID_POLICY, REFERENCE_MIXTURE_POLICY}:
            if plan_value:
                self.reference_fluid_plan_path = resolve_config_path(plan_value, config_dir)
            return

        if not plan_value:
            raise ValueError(
                f"policy={self.policy!r} requires reference_fluid_plan_path "
                "or reference_fluid_plan_file in eqn_config."
            )
        plan_path = resolve_config_path(plan_value, config_dir)
        plan_quantity, review_dt, resolved_path = load_open_loop_allocation_plan(
            plan_path,
            self.dim,
            self.total_time,
            require_fairness_spec=float(self.lambda_param) > 0.0,
        )
        total_quantity = float(plan_quantity.sum())
        inventory_tol = max(1e-4, 1e-6 * max(1.0, float(self.I0)))
        if total_quantity > float(self.I0) + inventory_tol:
            raise ValueError(
                "Reference fluid plan ships more than initial inventory: "
                f"total={total_quantity:.12g}, I0={float(self.I0):.12g}, path={resolved_path}."
            )

        eta_review = self.eta_for_review_grid(plan_quantity.shape[0]).detach().cpu().numpy()
        quantity_cap = eta_review * review_dt
        cap_tol = np.maximum(1e-5, 1e-6 * np.maximum(1.0, quantity_cap))
        violations = plan_quantity > quantity_cap + cap_tol
        if np.any(violations):
            n_idx, k_idx = np.argwhere(violations)[0]
            raise ValueError(
                "Reference fluid plan violates eta bound at "
                f"row={int(n_idx)}, location={int(k_idx)}: "
                f"a={plan_quantity[n_idx, k_idx]:.12g}, cap={quantity_cap[n_idx, k_idx]:.12g}, "
                f"path={resolved_path}."
            )

        rate_plan = plan_quantity / review_dt
        self.reference_fluid_plan_path = resolved_path
        self.reference_fluid_plan_quantity = torch.tensor(plan_quantity, dtype=torch.float32)
        self.reference_fluid_rate_plan = torch.tensor(rate_plan, dtype=torch.float32)
        self.reference_fluid_review_dt = float(review_dt)
        self.reference_fluid_plan_metadata = {
            "reference_policy": REFERENCE_FLUID_POLICY,
            "reference_fluid_plan_path": resolved_path,
            "reference_fluid_plan_shape": list(plan_quantity.shape),
            "reference_fluid_review_dt": float(review_dt),
            "reference_fluid_total_shipment": total_quantity,
            "reference_fluid_rate_min": float(rate_plan.min()) if rate_plan.size else 0.0,
            "reference_fluid_rate_max": float(rate_plan.max()) if rate_plan.size else 0.0,
            "reference_fluid_direct_fine_grid_match": bool(plan_quantity.shape[0] == self.num_time_interval),
            **fairness_metadata(),
        }
        print(
            "Loaded reference fluid rollout plan: "
            f"path={resolved_path}, shape={plan_quantity.shape}, review_dt={review_dt}, "
            f"total_shipment={total_quantity:.6f}"
        )

    def reference_fluid_rate_at_fine_index(self, n, device=None, dtype=None):
        """Return the open-loop fluid shipment rate row for fine interval n."""
        if self.reference_fluid_rate_plan is None or self.reference_fluid_review_dt is None:
            raise ValueError("Reference fluid rollout plan is not loaded.")
        if n is None:
            raise ValueError(f"{REFERENCE_FLUID_POLICY} requires a fine-grid time index n.")
        current_time = float(int(n)) * float(self.delta_t)
        review_idx = min(
            int(np.floor((current_time + 1e-12) / float(self.reference_fluid_review_dt))),
            int(self.reference_fluid_rate_plan.shape[0]) - 1,
        )
        row = self.reference_fluid_rate_plan[review_idx]
        if device is not None or dtype is not None:
            row = row.to(device=device if device is not None else row.device, dtype=dtype if dtype is not None else row.dtype)
        return row

    def _build_reference_policy_caches(self):
        """Precompute time-invariant reference-policy quantities used in training."""
        base_profile = torch.minimum(self.eta_schedule, self.mu).clamp(min=0.0)
        total_planned_capacity = base_profile.sum() * self.delta_t
        inventory_budget = torch.tensor(float(self.I0), dtype=base_profile.dtype)
        self._inventory_scaled_global_scale = torch.clamp(
            inventory_budget / (total_planned_capacity + 1e-8),
            max=1.0,
        )

        quota = torch.zeros(self.dim, dtype=self.eta_schedule.dtype)
        quota[0] = 0.8 * float(self.I0)
        if self.K > 0:
            quota[1:] = 0.2 * float(self.I0) / float(self.K)

        status_quo_rates = []
        for n_idx in range(int(self.num_time_interval)):
            if n_idx == 0:
                cum_capacity_n = torch.zeros(self.dim, dtype=self.eta_schedule.dtype)
            else:
                cum_capacity_n = self.eta_schedule[:n_idx].sum(dim=0) * self.delta_t
            cum_capacity_np1 = self.eta_schedule[: n_idx + 1].sum(dim=0) * self.delta_t
            cum_n = torch.minimum(cum_capacity_n, quota)
            cum_np1 = torch.minimum(cum_capacity_np1, quota)
            status_quo_rates.append(torch.clamp((cum_np1 - cum_n) / self.delta_t, min=0.0))
        self._status_quo_rate_schedule = torch.stack(status_quo_rates, dim=0)
    
    def _build_reflection_matrix(self):
        """Build the reflection matrix R (equation 202-209)."""
        R = torch.eye(self.dim, dtype=torch.float32)
        # First row: [1, -p_1, -p_2, ..., -p_K]
        R[0, 1:] = -self.p
        return R
    
    def _build_omega_matrices(self):
        """
        Build Omega_1 and Omega_2 matrices (equations 372-383).
        Omega_1: (K+1) x (K+2) matrix with -1 in last column
        Omega_2: (K+1) x (K+2) matrix with zeros in last column
        """
        K_plus_1 = self.dim
        K_plus_2 = K_plus_1 + 1
        
        # Omega_1: identity for first K+1 columns, -1 in last column
        Omega_1 = torch.zeros(K_plus_1, K_plus_2, dtype=torch.float32)
        Omega_1[:, :K_plus_1] = torch.eye(K_plus_1, dtype=torch.float32)
        Omega_1[:, K_plus_1] = -1.0
        
        # Omega_2: identity for first K+1 columns, 0 in last column
        Omega_2 = torch.zeros(K_plus_1, K_plus_2, dtype=torch.float32)
        Omega_2[:, :K_plus_1] = torch.eye(K_plus_1, dtype=torch.float32)
        
        return Omega_1, Omega_2
    
    def generate_u_sample_for_policy(self, policy, num_sample, n=None, Z_n=None, I_n=None, mu_t=None):
        """
        Generate reference policy samples for the reference process.
        Returns allocation rates a = (a_0, a_1, ..., a_K) for each sample.
        
        Args:
            num_sample: Number of samples
            n: Time interval index (optional, required for static_proportional,
               inventory_scaled_proportional, myopic, and status_quo policies)
            Z_n: Current inventory levels (num_sample, K+1) (optional, required for myopic policy)
            I_n: Current unallocated inventory (num_sample, 1) or (num_sample,) (optional, required for
               static_proportional, inventory_scaled_proportional, myopic, and status_quo policies)
            mu_t: Current demand drift (K+1,) (optional, required for
               static_proportional and inventory_scaled_proportional policies)
        
        Returns:
            u_sample: Allocation rates (num_sample, K+1)
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        policy = str(policy)
        
        # Current shipment-rate bound, broadcast to (num_sample, K+1).
        eta_vec = self.eta_at_fine_index(n).to(device=device)
        eta_expanded = eta_vec.unsqueeze(0).expand(num_sample, -1)
        
        if policy == REFERENCE_FLUID_POLICY:
            if n is None:
                raise ValueError(f"{REFERENCE_FLUID_POLICY} policy requires n parameter")
            a_vec = self.reference_fluid_rate_at_fine_index(
                n,
                device=device,
                dtype=eta_vec.dtype,
            )
            a_vec = torch.minimum(torch.clamp(a_vec, min=0.0), eta_vec)
            u_sample = a_vec.unsqueeze(0).expand(num_sample, -1).clone()
            if I_n is not None:
                if I_n.dim() == 2:
                    I_n_flat = I_n.squeeze(1)
                else:
                    I_n_flat = I_n
                u_sample[I_n_flat.to(device=device) <= 0.0] = 0.0

        elif policy == "static_proportional":
            # Static Proportional Policy (equation from paper)
            # φ^static_k(t, Z(t), I(t)) = min{η_k, μ_k(t)} if I(t) > 0, else 0
            if mu_t is None or n is None:
                raise ValueError("static_proportional policy requires n and mu_t parameters")
            if I_n is None:
                raise ValueError("static_proportional policy requires I_n parameter")
            
            # mu_t is (K+1,), broadcast to (num_sample, K+1)
            mu_t_expanded = mu_t.unsqueeze(0).expand(num_sample, -1).to(device)  # (num_sample, K+1)
            
            # I_n might be (num_sample, 1) or (num_sample,), handle both
            if I_n.dim() == 2:
                I_n_flat = I_n.squeeze(1)  # (num_sample,)
            else:
                I_n_flat = I_n  # (num_sample,)
            
            # Compute min{η_k, μ_k(t)} for each location
            allocation_rate = torch.min(eta_expanded, mu_t_expanded)  # (num_sample, K+1)
            
            # Set to 0 if I(t) <= 0
            mask_I_positive = I_n_flat > 0  # (num_sample,)
            u_sample = torch.zeros(num_sample, self.dim, device=device)
            u_sample[mask_I_positive] = allocation_rate[mask_I_positive]
        
        elif policy == "inventory_scaled_proportional":
            # Inventory-Scaled Proportional reference policy.
            # base_k(n) = min{eta_k, mu_k(n)}
            # C = sum_n sum_k base_k(n) * delta_t
            # rho_IS = min{1, I0 / C}
            # phi_k(n) = rho_IS * base_k(n). The shared reference-transition
            # projection enforces one-step warehouse feasibility.
            if mu_t is None or n is None:
                raise ValueError("inventory_scaled_proportional policy requires n and mu_t parameters")
            if I_n is None:
                raise ValueError("inventory_scaled_proportional policy requires I_n parameter")

            if I_n.dim() == 2:
                I_n_flat = I_n.squeeze(1)
            else:
                I_n_flat = I_n

            mu_vec = mu_t.to(device)
            base_rate = torch.minimum(eta_vec, mu_vec).clamp(min=0.0)

            global_scale = self._inventory_scaled_global_scale.to(
                device=device,
                dtype=base_rate.dtype,
            )

            allocation_rate = global_scale * base_rate.unsqueeze(0).expand(num_sample, -1)
            allocation_rate = torch.minimum(allocation_rate, eta_expanded)
            allocation_rate = torch.clamp(allocation_rate, min=0.0)

            mask_I_positive = I_n_flat > 0
            u_sample = torch.zeros(num_sample, self.dim, device=device)
            u_sample[mask_I_positive] = allocation_rate[mask_I_positive]

        elif policy == "myopic":
            # Myopic Policy (equation from paper)
            # φ^myopic_k(t, Z(t), I(t)) = η_k if Z_k(t) ≤ ε and I(t) > 0, else 0
            if Z_n is None:
                raise ValueError("myopic policy requires Z_n parameter")
            if I_n is None:
                raise ValueError("myopic policy requires I_n parameter")
            
            # I_n might be (num_sample, 1) or (num_sample,), handle both
            if I_n.dim() == 2:
                I_n_flat = I_n.squeeze(1)  # (num_sample,)
            else:
                I_n_flat = I_n  # (num_sample,)
            
            epsilon = getattr(self, 'myopic_epsilon', 0.0)
            tol = 1e-9
            # Z_n is (num_sample, K+1), epsilon is scalar
            mask_low_stock = Z_n <= (epsilon + tol)  # (num_sample, K+1)
            mask_I_positive = I_n_flat > 0  # (num_sample,)
            
            # Combine conditions: both must be true
            mask_allocate = mask_low_stock & mask_I_positive.unsqueeze(1)  # (num_sample, K+1)
            
            # Set allocation: η_k if condition is true, else 0
            u_sample = torch.where(mask_allocate, eta_expanded, torch.zeros_like(eta_expanded))

        elif policy == "status_quo":
            # Upfront quota: Q_0 = 0.8 I0, Q_k = 0.2 I0 / K; discrete step integrates continuous bang rate
            if n is None:
                raise ValueError("status_quo policy requires n parameter")
            if I_n is None:
                raise ValueError("status_quo policy requires I_n parameter")

            if I_n.dim() == 2:
                I_n_flat = I_n.squeeze(1)
            else:
                I_n_flat = I_n

            rate_schedule = self._status_quo_rate_schedule.to(device=device, dtype=eta_vec.dtype)
            n_idx = max(0, min(int(n), rate_schedule.shape[0] - 1))
            a_vec = rate_schedule[n_idx]

            sum_rate_dt = a_vec.sum() * self.delta_t
            base = a_vec.unsqueeze(0).expand(num_sample, -1)
            if sum_rate_dt > 1e-12:
                scale = torch.clamp(I_n_flat / (sum_rate_dt + 1e-12), max=1.0)
                u_sample = base * scale.unsqueeze(1)
            else:
                u_sample = torch.zeros(num_sample, self.dim, device=device)
        
        else:
            # Default: even distribution
            u_sample = eta_expanded / self.dim
        
        return u_sample

    def generate_u_sample(self, num_sample, n=None, Z_n=None, I_n=None, mu_t=None):
        if self.policy == REFERENCE_MIXTURE_POLICY:
            raise ValueError(
                f"{REFERENCE_MIXTURE_POLICY} requires path-level labels; "
                "call generate_u_sample_for_policy for each stored label instead."
            )
        return self.generate_u_sample_for_policy(
            self.policy,
            num_sample,
            n=n,
            Z_n=Z_n,
            I_n=I_n,
            mu_t=mu_t,
        )
    
    def F_function(self, t, i, grad_diff, mu_t, delta_t):
        """
        Compute the Hamiltonian minimized over feasible allocations.
        
        Args:
            t: time index (int)
            i: unallocated inventory (torch.Tensor, shape: (batch_size,))
            grad_diff: control-relevant gradient differences D V
                (torch.Tensor, shape: (batch_size, K+1))
            mu_t: demand drift at time t (torch.Tensor, shape: (K+1,))
            delta_t: time step size (float). Enforces constraint that total allocation
                    in time step doesn't exceed available inventory.
        
        Returns:
            F values (torch.Tensor, shape: (batch_size,))
        """
        # Compute optimal allocation a* for each sample
        a_star = build_feasible_ds_allocation(
            grad_diff,
            i,
            mu_t,
            self.eta_at_fine_index(t).to(device=grad_diff.device, dtype=grad_diff.dtype),
            self.lambda_param,
            self.alpha,
            delta_t,
        )
        
        return demand_weighted_hamiltonian_value_torch(
            a_star,
            grad_diff,
            mu_t,
            self.lambda_param,
            self.alpha,
        )
    
    def f_tf(self, x, a_lowbound, mu, theta, cost, z, u):
        """
        This method is kept for compatibility but not used in LiberiaHJB.
        The actual computation uses F_function instead.
        """
        raise NotImplementedError("LiberiaHJB uses F_function instead of f_tf")
    
    def g_tf(self, z):
        """
        Terminal condition (equation 324).
        g(z) = sum(h_k * z_k)
        
        Args:
            z: inventory levels (torch.Tensor, shape: (batch_size, K+1))
        
        Returns:
            Terminal cost (torch.Tensor, shape: (batch_size, 1))
        """
        device = z.device
        h_tensor = self.h.to(device)
        terminal = (z * h_tensor.unsqueeze(0)).sum(dim=1, keepdim=True)
        return terminal
