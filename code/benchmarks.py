import torch


def _ensure_tensor(value, device, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def _eta_at(t, eta, dim, device, dtype):
    eta_t = _ensure_tensor(eta, device, dtype)
    if eta_t.dim() == 0:
        return eta_t.expand(dim)
    if eta_t.dim() == 1:
        return eta_t
    if eta_t.dim() == 2:
        idx = max(0, min(int(t), eta_t.shape[0] - 1))
        return eta_t[idx]
    raise ValueError(f"eta must be scalar, vector, or time profile; got shape {tuple(eta_t.shape)}.")


def _eta_profile_like(eta, mu_profile, dim, device, dtype):
    eta_t = _ensure_tensor(eta, device, dtype)
    rows = int(mu_profile.shape[0])
    if eta_t.dim() == 0:
        return eta_t.expand(rows, dim)
    if eta_t.dim() == 1:
        return eta_t.unsqueeze(0).expand(rows, -1)
    if eta_t.dim() == 2:
        if eta_t.shape[0] == rows:
            return eta_t
        if eta_t.shape[0] == 1:
            return eta_t.expand(rows, -1)
    raise ValueError(
        f"eta profile rows must match mu profile rows ({rows}); got shape {tuple(eta_t.shape)}."
    )


def status_quo_policy(n, Z, I, eta, I0, K, dt, eps_scale=1e-12):
    """
    Status Quo (Upfront Quota) policy: open-loop push until per-location quotas are met.

    Q_0 = 0.8 * I0, Q_k = 0.2 * I0 / K for k = 1..K. Continuous-time rate is eta_k while
    t <= Q_k / eta_k, else 0. In discrete time, use the average rate that integrates the
    continuous policy over [t_n, t_{n+1}] so cumulative shipment per location does not
    exceed Q_k. If sum_k a_k * dt exceeds remaining central inventory I, scale down
    proportionally per path.

    Args:
        n: Time step index (int), interval [n*dt, (n+1)*dt).
        Z: Current local inventory (unused; kept for API consistency with other policies).
        I: Central inventory (batch,) or (batch, 1).
        eta: Per-location capacity (K+1,) tensor or array.
        I0: Initial central inventory (scalar) used to define quotas.
        K: Number of CHAs (HF is k=0; dim = K+1).
        dt: Time step length.
        eps_scale: Numerical floor for division.

    Returns:
        a: Allocation rates (batch_size, K+1).
    """
    batch_size = Z.shape[0]
    device = Z.device
    dtype = Z.dtype
    dim = K + 1

    if I.dim() == 2:
        I_flat = I.squeeze(1)
    else:
        I_flat = I

    eta_tensor = _ensure_tensor(eta, device, dtype)

    Q = torch.zeros(dim, device=device, dtype=dtype)
    Q[0] = 0.8 * I0
    if K > 0:
        Q[1:] = 0.2 * I0 / K

    if eta_tensor.dim() == 2:
        n_idx = max(0, min(int(n), eta_tensor.shape[0] - 1))
        if n_idx == 0:
            cum_capacity_n = torch.zeros(dim, device=device, dtype=dtype)
        else:
            cum_capacity_n = eta_tensor[:n_idx].sum(dim=0) * float(dt)
        cum_capacity_np1 = eta_tensor[: n_idx + 1].sum(dim=0) * float(dt)
        cum_n = torch.minimum(cum_capacity_n, Q)
        cum_np1 = torch.minimum(cum_capacity_np1, Q)
    else:
        eta_vec = _eta_at(n, eta_tensor, dim, device, dtype)
        t_n = float(n) * dt
        t_np1 = float(n + 1) * dt
        cum_n = torch.minimum(eta_vec * t_n, Q)
        cum_np1 = torch.minimum(eta_vec * t_np1, Q)
    a_vec = (cum_np1 - cum_n) / dt
    a_vec = torch.clamp(a_vec, min=0.0)

    sum_rate_dt = a_vec.sum() * dt
    a = a_vec.unsqueeze(0).expand(batch_size, -1).clone()
    if sum_rate_dt > eps_scale:
        scale = torch.clamp(I_flat / (sum_rate_dt + eps_scale), max=1.0)
        a = a * scale.unsqueeze(1)
    else:
        a = torch.zeros_like(a)

    return a


def static_policy(t, Z, I, mu, eta):
    """
    Static Proportional Policy for Liberia Inventory Model.
    
    If global inventory I(t) > 0:
        Supply rate a_k = min(eta, mu_k) for each location k
    If global inventory I(t) <= 0:
        Supply rate a = 0
    
    Args:
        t: Time index (int). Used to index mu if mu is time-dependent.
        Z: Current inventory levels (torch.Tensor, shape: [batch_size, dim])
           where dim = K+1 (1 HF + K CHAs)
        I: Global unallocated inventory (torch.Tensor, shape: [batch_size, 1] or [batch_size,])
        mu: Demand drift vector (torch.Tensor, shape: [K+1,] or [num_time_intervals, K+1])
           If time-dependent, should be indexed as mu[t, :] before passing
           If constant, should be shape [K+1,]
        eta: Per-location capacity (K+1,) tensor; elementwise min(eta_k, mu_k) is used.
             A scalar is broadcast to all locations (legacy).
    
    Returns:
        a: Allocation rates (torch.Tensor, shape: [batch_size, dim])
    """
    batch_size = Z.shape[0]
    dim = Z.shape[1]
    device = Z.device
    
    # Handle I shape: ensure it's [batch_size,]
    if I.dim() == 2:
        I = I.squeeze(1)
    
    # Handle mu: if it's 2D (time-dependent), index it; otherwise use as-is
    if mu.dim() == 2:
        mu_t = mu[t, :]  # (K+1,)
    else:
        mu_t = mu  # (K+1,)
    mu_t = mu_t.to(device=device, dtype=Z.dtype)
    
    # Broadcast mu_t to match batch size: (K+1,) -> (1, K+1) -> (batch_size, K+1)
    mu_expanded = mu_t.unsqueeze(0)  # (1, K+1)
    
    eta_t = _eta_at(t, eta, dim, device, Z.dtype)
    
    # Compute min(eta, mu_k) for each location
    a_proportional = torch.min(eta_t.unsqueeze(0), mu_expanded)
    
    # Conditional logic: if I > 0, use a_proportional; otherwise use 0
    # I > 0 gives shape [batch_size,], need to expand to [batch_size, 1] for broadcasting
    mask_I_positive = (I > 0).unsqueeze(1)  # (batch_size, 1)
    
    # Use torch.where to efficiently handle the conditional
    a = torch.where(mask_I_positive, a_proportional, torch.zeros_like(a_proportional))
    
    return a


def inventory_scaled_proportional_policy(t, Z, I, mu, eta, dt, eps=1e-8, initial_inventory=None):
    """
    Inventory-Scaled Proportional Policy.

    Reference-only policy used for rollout diagnostics. First form the capped
    demand base over the whole horizon
        base_k(t) = min(eta_k, mu_k(t)).
    Then compute one global scale
        s = min(1, Q0 / sum_t sum_k base_k(t) * dt),
    and use
        phi_k(t) = s * base_k(t).
    Finally project the one-step shipment so sum_k phi_k * dt <= I(t). If
    global inventory I(t) <= 0, then a = 0.

    This policy is "static" in the sense that it does not react to local
    inventory levels Z(t), but unlike the plain static policy it explicitly
    scales the full forecast-following shipment profile to the initial
    warehouse budget. If the capped base profile has enough total capacity, the
    planned cumulative shipment equals Q0.

    Args:
        t: Time index (int). Used to index mu if mu is time-dependent.
        Z: Current inventory levels (unused; kept for interface consistency).
        I: Global unallocated inventory (torch.Tensor, shape: [batch_size, 1] or [batch_size,]).
        mu: Demand drift vector (torch.Tensor, shape: [K+1,] or [num_time_intervals, K+1]).
        eta: Maximum supply capacity vector (torch.Tensor or scalar).
        dt: Time step size.
        eps: Small positive constant for numerical stability.
        initial_inventory: Optional scalar Q0. If omitted, the current I value
            is used as the budget, which is appropriate when evaluating the
            policy at the initial time.

    Returns:
        a: Allocation rates (torch.Tensor, shape: [batch_size, dim]).
    """
    batch_size = Z.shape[0]
    dim = Z.shape[1]
    device = Z.device
    dtype = Z.dtype

    if I.dim() == 2:
        I = I.squeeze(1)

    if mu.dim() == 2:
        mu_t = mu[t, :]
    else:
        mu_t = mu

    eta_t = _eta_at(t, eta, dim, device, dtype)

    mu_t = mu_t.to(device=device, dtype=dtype)
    if mu.dim() == 2:
        mu_profile = mu.to(device=device, dtype=dtype)
    else:
        mu_profile = mu_t.unsqueeze(0)
    eta_profile = _eta_profile_like(eta, mu_profile, dim, device, dtype)

    base_profile = torch.minimum(eta_profile, mu_profile).clamp(min=0.0)
    total_planned_capacity = base_profile.sum() * float(dt)
    if initial_inventory is None:
        inventory_budget = torch.clamp(I, min=0.0)
    else:
        inventory_budget = torch.as_tensor(initial_inventory, device=device, dtype=dtype)
        inventory_budget = inventory_budget.expand(batch_size)
    scale = torch.clamp(inventory_budget / (total_planned_capacity + eps), max=1.0)

    base = torch.minimum(eta_t, mu_t).clamp(min=0.0)
    a_scaled = scale.unsqueeze(1) * base.unsqueeze(0)

    one_step_shipment = a_scaled.sum(dim=1) * float(dt)
    feasible_scale = torch.clamp(I / (one_step_shipment + eps), max=1.0)
    a_scaled = a_scaled * feasible_scale.unsqueeze(1)

    mask_I_positive = (I > 0).unsqueeze(1)
    a = torch.where(mask_I_positive, a_scaled, torch.zeros_like(a_scaled))

    return a


def myopic_policy(Z, I, eta, eps=0.0):
    """
    Myopic Policy for Liberia Inventory Model.
    
    If global inventory I(t) > 0:
        If Z_k <= eps: a_k = eta (supply at maximum capacity)
        If Z_k > eps: a_k = 0 (no supply)
    If global inventory I(t) <= 0:
        a = 0
    
    Args:
        Z: Current inventory levels (torch.Tensor, shape: [batch_size, dim])
           where dim = K+1 (1 HF + K CHAs)
        I: Global unallocated inventory (torch.Tensor, shape: [batch_size, 1] or [batch_size,])
        eta: Maximum supply capacity (float or scalar tensor)
        eps: Threshold for inventory level (float, default: 0.0). A tiny tolerance is
             added so float noise does not break Z ≈ 0.
    
    Returns:
        a: Allocation rates (torch.Tensor, shape: [batch_size, dim])
    """
    batch_size = Z.shape[0]
    dim = Z.shape[1]
    device = Z.device
    
    # Handle I shape: ensure it's [batch_size,]
    if I.dim() == 2:
        I = I.squeeze(1)
    
    eta = _eta_at(0, eta, dim, device, Z.dtype)
    if not isinstance(eps, torch.Tensor):
        eps = torch.tensor(eps, device=device, dtype=Z.dtype)
    else:
        eps = eps.to(device=device, dtype=Z.dtype)
    
    tol = torch.tensor(1e-9, device=device, dtype=Z.dtype)
    mask_Z_low = Z <= (eps + tol)  # (batch_size, dim)
    
    # If Z_k <= eps, a_k = eta; otherwise a_k = 0
    a_myopic = torch.where(mask_Z_low, eta, torch.zeros_like(Z))
    
    # Conditional logic: if I > 0, use a_myopic; otherwise use 0
    # I > 0 gives shape [batch_size,], need to expand to [batch_size, 1] for broadcasting
    mask_I_positive = (I > 0).unsqueeze(1)  # (batch_size, 1)
    
    # Use torch.where to efficiently handle the conditional
    a = torch.where(mask_I_positive, a_myopic, torch.zeros_like(a_myopic))
    
    return a
