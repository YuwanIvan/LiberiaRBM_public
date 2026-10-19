"""Policy-extraction operators for discrete-review Deep Splitting evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import torch

import equation as eqn
from input_scaling import apply_state_input_scaling, input_scale_vector_for_mode


FROZEN_AVERAGE = "frozen_average"
PERIOD_START = "period_start"
Q_ROLLOUT = "q_rollout"
MEAN_ROLLOUT = "mean_rollout"

MODE_ALIASES: Dict[str, str] = {
    "E0": FROZEN_AVERAGE,
    "e0": FROZEN_AVERAGE,
    "frozen": FROZEN_AVERAGE,
    "frozen_average": FROZEN_AVERAGE,
    "frozen-state": FROZEN_AVERAGE,
    "frozen_state": FROZEN_AVERAGE,
    "E1": PERIOD_START,
    "e1": PERIOD_START,
    "period_start": PERIOD_START,
    "period-start": PERIOD_START,
    "first_network": PERIOD_START,
    "period_start_only": PERIOD_START,
    "E2": Q_ROLLOUT,
    "e2": Q_ROLLOUT,
    "q_rollout": Q_ROLLOUT,
    "q-aware": Q_ROLLOUT,
    "q_aware": Q_ROLLOUT,
    "q_aware_rollout": Q_ROLLOUT,
    "E3": MEAN_ROLLOUT,
    "e3": MEAN_ROLLOUT,
    "mean_rollout": MEAN_ROLLOUT,
    "mean-path": MEAN_ROLLOUT,
    "mean_path": MEAN_ROLLOUT,
    "certainty_equivalent": MEAN_ROLLOUT,
}

MODE_LABELS: Dict[str, str] = {
    FROZEN_AVERAGE: "E0",
    PERIOD_START: "E1",
    Q_ROLLOUT: "E2",
    MEAN_ROLLOUT: "E3",
}


def canonical_extraction_mode(mode: str) -> str:
    """Return the canonical extraction-mode string."""
    key = str(mode).strip()
    if key in MODE_ALIASES:
        return MODE_ALIASES[key]
    lowered = key.lower()
    if lowered in MODE_ALIASES:
        return MODE_ALIASES[lowered]
    valid = ", ".join(sorted(set(MODE_ALIASES.values())))
    raise ValueError(f"Unsupported policy extraction mode {mode!r}. Valid canonical modes: {valid}.")


def extraction_policy_name(mode: str) -> str:
    canonical = canonical_extraction_mode(mode)
    return f"{MODE_LABELS[canonical]}_{canonical}"


def fine_indices_for_period(period_index: int, fine_per_review: int) -> Tuple[int, ...]:
    """Map zero-based review period ``n`` to fine indices ``nr, ..., (n+1)r-1``."""
    n = int(period_index)
    r = int(fine_per_review)
    if n < 0:
        raise ValueError(f"period_index must be nonnegative, got {period_index}.")
    if r <= 0:
        raise ValueError(f"fine_per_review must be positive, got {fine_per_review}.")
    start = n * r
    return tuple(range(start, start + r))


def classify_action_regime_torch(rate: torch.Tensor, eta: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:
    """Classify action rates as 0=zero, 1=interior, 2=eta cap."""
    eta = eta.to(device=rate.device, dtype=rate.dtype)
    if eta.dim() == 1:
        eta = eta.unsqueeze(0)
    eta = eta.expand_as(rate)
    cap_tol = torch.maximum(
        torch.full_like(eta, float(tol)),
        1e-6 * torch.maximum(eta, torch.ones_like(eta)),
    )
    regimes = torch.ones(rate.shape, device=rate.device, dtype=torch.int8)
    regimes = torch.where(rate <= float(tol), torch.zeros_like(regimes), regimes)
    regimes = torch.where(
        (rate > float(tol)) & (rate >= eta - cap_tol),
        torch.full_like(regimes, 2),
        regimes,
    )
    return regimes


@dataclass
class WeeklyExtractionResult:
    """Tensor payload returned by one review-period extraction."""

    tentative_quantity: torch.Tensor
    effective_rates: torch.Tensor
    raw_rates: torch.Tensor
    feasible_quantities: torch.Tensor
    virtual_q: torch.Tensor
    virtual_z: torch.Tensor
    virtual_l: torch.Tensor
    fine_projection_mask: torch.Tensor
    fine_projection_scale: torch.Tensor
    action_regimes: torch.Tensor
    query_count: int
    fine_indices: Tuple[int, ...]
    mode: str
    used_step_transition: bool = False
    delta_w_is_zero: bool = True
    uses_future_information: bool = False


class DeepSplittingPolicyExtractor:
    """Read-only helper that evaluates trained fine-grid DS policies."""

    def __init__(self, z_net_list, bsde, device, config=None):
        self.z_net_list = z_net_list
        self.bsde = bsde
        self.device = device
        net_config = getattr(config, "net_config", SimpleNamespace()) if config is not None else SimpleNamespace()
        input_mode = getattr(net_config, "input_mode", "raw")
        q0_input_scale = float(getattr(net_config, "q0_input_scale", getattr(bsde, "I0", 142.0)))
        state_dim = int(getattr(bsde, "dim")) + 1
        self.input_scale = torch.tensor(
            input_scale_vector_for_mode(input_mode, state_dim=state_dim, q0=q0_input_scale),
            device=device,
            dtype=torch.float32,
        )

    def _normalize_state(self, Z: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Z = Z.to(device=self.device, dtype=self.bsde.mu.dtype)
        Q = Q.to(device=self.device, dtype=self.bsde.mu.dtype)
        if Q.dim() == 0:
            Q = Q.unsqueeze(0)
        if Q.dim() == 2:
            Q = Q.squeeze(1)
        if Q.shape[0] != Z.shape[0]:
            Q = Q.expand(Z.shape[0])
        return Z, Q

    def mu_at(self, fine_idx: int, dtype=None) -> torch.Tensor:
        idx = min(max(int(fine_idx), 0), int(self.bsde.mu.shape[0]) - 1)
        return self.bsde.mu[idx, :].to(device=self.device, dtype=dtype or self.bsde.mu.dtype)

    def eta_at(self, fine_idx: int, dtype=None) -> torch.Tensor:
        if hasattr(self.bsde, "eta_at_fine_index"):
            eta = self.bsde.eta_at_fine_index(int(fine_idx))
        else:
            eta = self.bsde.eta
        return eta.to(device=self.device, dtype=dtype or self.bsde.mu.dtype)

    def grad_diff(self, fine_idx: int, Z: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        Z, Q = self._normalize_state(Z, Q)
        state = torch.cat([Z, Q.unsqueeze(1)], dim=1)
        idx = min(max(int(fine_idx), 0), len(self.z_net_list) - 1)
        with torch.no_grad():
            return self.z_net_list[idx](apply_state_input_scaling(state, self.input_scale))

    def raw_rate(self, fine_idx: int, Z: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        Z, Q = self._normalize_state(Z, Q)
        grad = self.grad_diff(fine_idx, Z, Q)
        return eqn.build_unprojected_ds_allocation(
            grad,
            Q,
            self.mu_at(fine_idx, dtype=grad.dtype),
            self.eta_at(fine_idx, dtype=grad.dtype),
            self.bsde.lambda_param,
            self.bsde.alpha,
            self.bsde.delta_t,
        )

    def feasible_rate(
        self,
        raw_rate: torch.Tensor,
        Q: torch.Tensor,
        fine_idx: int,
        fine_dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = Q.to(device=raw_rate.device, dtype=raw_rate.dtype)
        if Q.dim() == 2:
            Q = Q.squeeze(1)
        projected, _, diagnostics = eqn.project_rate_to_inventory_torch(
            raw_rate,
            Q,
            fine_dt,
            eta_rate=self.eta_at(fine_idx, dtype=raw_rate.dtype),
            return_diagnostics=True,
        )
        return projected, diagnostics["projection_mask"], diagnostics["projection_scale"]

    def extract_weekly_action(
        self,
        mode: str,
        period_index: int,
        Z_start: torch.Tensor,
        Q_start: torch.Tensor,
        fine_per_review: int,
        review_dt: float,
    ) -> WeeklyExtractionResult:
        mode = canonical_extraction_mode(mode)
        fine_indices = fine_indices_for_period(period_index, fine_per_review)
        fine_dt = float(review_dt) / float(fine_per_review)
        Z_start, Q_start = self._normalize_state(Z_start, Q_start)
        paths, dim = Z_start.shape
        r = int(fine_per_review)
        dtype = Z_start.dtype

        raw_rates = torch.zeros(r, paths, dim, device=self.device, dtype=dtype)
        effective_rates = torch.zeros_like(raw_rates)
        quantities = torch.zeros_like(raw_rates)
        fine_projection_mask = torch.zeros(r, paths, device=self.device, dtype=torch.bool)
        fine_projection_scale = torch.ones(r, paths, device=self.device, dtype=dtype)
        action_regimes = torch.full((r, paths, dim), -1, device=self.device, dtype=torch.int8)
        virtual_q = torch.full((r + 1, paths), torch.nan, device=self.device, dtype=dtype)
        virtual_z = torch.full((r + 1, paths, dim), torch.nan, device=self.device, dtype=dtype)
        virtual_l = torch.zeros(r, paths, dim, device=self.device, dtype=dtype)
        virtual_q[0] = Q_start
        virtual_z[0] = Z_start

        if mode == FROZEN_AVERAGE:
            for pos, fine_idx in enumerate(fine_indices):
                raw = self.raw_rate(fine_idx, Z_start, Q_start)
                feasible, mask, scale = self.feasible_rate(raw, Q_start, fine_idx, fine_dt)
                raw_rates[pos] = raw
                effective_rates[pos] = feasible
                quantities[pos] = feasible * fine_dt
                fine_projection_mask[pos] = mask
                fine_projection_scale[pos] = scale
                action_regimes[pos] = classify_action_regime_torch(raw, self.eta_at(fine_idx, dtype=dtype))
                virtual_q[pos + 1] = Q_start
                virtual_z[pos + 1] = Z_start
            query_count = r
            used_step_transition = False

        elif mode == PERIOD_START:
            fine_idx = fine_indices[0]
            raw = self.raw_rate(fine_idx, Z_start, Q_start)
            raw_rates[0] = raw
            effective_rates[:] = raw.unsqueeze(0).expand_as(effective_rates)
            quantities[:] = effective_rates * fine_dt
            action_regimes[0] = classify_action_regime_torch(raw, self.eta_at(fine_idx, dtype=dtype))
            virtual_q[1:] = Q_start.unsqueeze(0).expand(r, -1)
            virtual_z[1:] = Z_start.unsqueeze(0).expand(r, -1, -1)
            query_count = 1
            used_step_transition = False

        elif mode == Q_ROLLOUT:
            Z_bar = Z_start
            Q_bar = Q_start
            for pos, fine_idx in enumerate(fine_indices):
                raw = self.raw_rate(fine_idx, Z_bar, Q_bar)
                feasible, mask, scale = self.feasible_rate(raw, Q_bar, fine_idx, fine_dt)
                quantity = feasible * fine_dt
                raw_rates[pos] = raw
                effective_rates[pos] = feasible
                quantities[pos] = quantity
                fine_projection_mask[pos] = mask
                fine_projection_scale[pos] = scale
                action_regimes[pos] = classify_action_regime_torch(raw, self.eta_at(fine_idx, dtype=dtype))
                Q_bar = torch.clamp(Q_bar - quantity.sum(dim=1), min=0.0)
                virtual_q[pos + 1] = Q_bar
                virtual_z[pos + 1] = Z_start
            query_count = r
            used_step_transition = False

        elif mode == MEAN_ROLLOUT:
            Z_bar = Z_start
            Q_bar = Q_start
            for pos, fine_idx in enumerate(fine_indices):
                raw = self.raw_rate(fine_idx, Z_bar, Q_bar)
                feasible, mask, scale = self.feasible_rate(raw, Q_bar, fine_idx, fine_dt)
                quantity = feasible * fine_dt
                raw_rates[pos] = raw
                effective_rates[pos] = feasible
                quantities[pos] = quantity
                fine_projection_mask[pos] = mask
                fine_projection_scale[pos] = scale
                action_regimes[pos] = classify_action_regime_torch(raw, self.eta_at(fine_idx, dtype=dtype))
                delta_w = torch.zeros_like(Z_bar)
                Z_bar, Q_bar, delta_l = eqn.step_transition_torch(
                    Z_bar,
                    Q_bar,
                    feasible,
                    delta_w,
                    fine_dt,
                    self.mu_at(fine_idx, dtype=dtype),
                    self.bsde.p,
                )
                virtual_l[pos] = delta_l
                virtual_q[pos + 1] = Q_bar
                virtual_z[pos + 1] = Z_bar
            query_count = r
            used_step_transition = True

        else:  # pragma: no cover - guarded by canonical_extraction_mode
            raise ValueError(f"Unsupported extraction mode {mode!r}.")

        tentative_quantity = quantities.sum(dim=0)
        return WeeklyExtractionResult(
            tentative_quantity=tentative_quantity,
            effective_rates=effective_rates,
            raw_rates=raw_rates,
            feasible_quantities=quantities,
            virtual_q=virtual_q,
            virtual_z=virtual_z,
            virtual_l=virtual_l,
            fine_projection_mask=fine_projection_mask,
            fine_projection_scale=fine_projection_scale,
            action_regimes=action_regimes,
            query_count=query_count,
            fine_indices=fine_indices,
            mode=mode,
            used_step_transition=used_step_transition,
            delta_w_is_zero=True,
            uses_future_information=False,
        )


def parse_extraction_modes(value: str | Iterable[str]) -> Tuple[str, ...]:
    """Parse comma-separated extraction modes, preserving order."""
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip() for item in value if str(item).strip()]
    modes = []
    for item in raw_items:
        canonical = canonical_extraction_mode(item)
        if canonical not in modes:
            modes.append(canonical)
    if not modes:
        modes = [FROZEN_AVERAGE]
    return tuple(modes)
