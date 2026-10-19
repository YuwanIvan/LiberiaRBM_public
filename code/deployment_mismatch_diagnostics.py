#!/usr/bin/env python
import argparse
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import equation as eqn
import evaluate as eval_mod


LADDER_SUMMARY_FILE = "deployment_ladder_summary.csv"
INTERVAL_DIAGNOSTICS_FILE = "deployment_interval_paired_diagnostics.csv"
REPLAY_TRACES_FILE = "deployment_replay_traces.csv"
INTRA_INTERVAL_REQUERY_FILE = "deployment_intra_interval_requery.csv"
NOTE_FILE = "deployment_mismatch_note.md"

LEVEL_A = "A"
LEVEL_B = "B"
LEVEL_C = "C"
LEVEL_D = "D"
PERIODIC_LEVELS = (LEVEL_B, LEVEL_C, LEVEL_D)

LEVEL_DESCRIPTIONS = {
    LEVEL_A: "Fine-grid Brownian evaluation with fine-step policy updates.",
    LEVEL_B: "Periodic-review cadence with fixed actions, frozen review-level dynamics, and Brownian substeps.",
    LEVEL_C: "Periodic-review cadence with fixed actions, fine-grid time-varying dynamics, and Brownian substeps.",
    LEVEL_D: "Periodic-review cadence with fixed actions and discrete periodic demand in the deployed environment.",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run deployment-mismatch diagnostics for a trained Deep Splitting policy."
    )
    parser.add_argument(
        "run_name",
        nargs="?",
        default=None,
        help="Run directory under output/. Optional when --use_latest_volatility_case is set.",
    )
    parser.add_argument(
        "--use_latest_volatility_case",
        action="store_true",
        help="Resolve the latest stylized volatility run automatically.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Base output directory containing run folders. Defaults to <project>/output.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for diagnostic outputs. Defaults to the selected run directory.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional explicit config_used.json path.",
    )
    parser.add_argument(
        "--logs_path",
        type=str,
        default=None,
        help="Optional explicit logs directory.",
    )
    parser.add_argument(
        "--num_paths",
        type=int,
        default=10000,
        help="Monte Carlo paths for the ladder evaluation.",
    )
    parser.add_argument(
        "--fine_grid_seed",
        type=int,
        default=0,
        help="Seed for fine-grid Brownian common random numbers.",
    )
    parser.add_argument(
        "--periodic_seed",
        type=int,
        default=0,
        help="Seed for periodic-review common random numbers.",
    )
    parser.add_argument(
        "--review_num_intervals",
        type=int,
        default=None,
        help="Override the periodic-review interval count.",
    )
    parser.add_argument(
        "--review_dt",
        type=float,
        default=None,
        help="Override the periodic-review interval length.",
    )
    parser.add_argument(
        "--periodic_demand_mode",
        choices=eval_mod.PERIODIC_DEMAND_MODES,
        default="negative_binomial",
        help="Periodic demand model for Level D replay diagnostics.",
    )
    parser.add_argument(
        "--num_replay_paths",
        type=int,
        default=5,
        help="Number of fixed CRN paths to export for replay traces.",
    )
    parser.add_argument(
        "--replay_path_ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit path ids for replay/export diagnostics.",
    )
    return parser.parse_args()


def project_root():
    return os.path.dirname(os.path.abspath(__file__))


def resolve_latest_run(output_root: str, prefix: str) -> str:
    candidates = []
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for entry in os.listdir(output_root):
        match = pattern.match(entry)
        if match and os.path.isdir(os.path.join(output_root, entry)):
            candidates.append((int(match.group(1)), entry))
    if not candidates:
        raise FileNotFoundError(f"No runs found under {output_root} matching prefix {prefix}*")
    candidates.sort()
    return candidates[-1][1]


def resolve_run_paths(args):
    base_output_root = args.output_root or os.path.join(project_root(), "output")
    if args.use_latest_volatility_case:
        run_name = resolve_latest_run(base_output_root, "stylized_2cha_case_a_volatility_")
    elif args.run_name:
        run_name = args.run_name
    else:
        raise ValueError("Provide run_name or set --use_latest_volatility_case.")

    run_dir = os.path.join(base_output_root, run_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = args.config_path or os.path.join(run_dir, "config_used.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logs_path = args.logs_path or os.path.join(run_dir, "logs")
    output_dir = args.output_dir or run_dir
    os.makedirs(output_dir, exist_ok=True)

    return run_name, run_dir, config_path, logs_path, output_dir


def select_replay_paths(num_paths: int, replay_path_ids: Optional[Sequence[int]], num_replay_paths: int) -> List[int]:
    if replay_path_ids:
        unique_ids = sorted(set(int(path_id) for path_id in replay_path_ids))
        if any(path_id < 0 or path_id >= num_paths for path_id in unique_ids):
            raise ValueError(f"Replay path ids must lie in [0, {num_paths - 1}].")
        return unique_ids
    return list(range(min(num_paths, int(num_replay_paths))))


def mean_se_ci(values: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        se = 0.0
    else:
        se = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    ci_half_width = 1.96 * se
    return mean, se, mean - ci_half_width, mean + ci_half_width


def precompute_standard_normals(num_steps: int, num_paths: int, dim: int, device, dtype, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    normals = torch.randn((num_steps, num_paths, dim), generator=generator, dtype=dtype)
    return normals.to(device)


def build_brownian_increments(standard_normals: torch.Tensor, covariance_schedule: torch.Tensor, dt: float):
    num_steps = standard_normals.shape[0]
    increments = torch.empty_like(standard_normals)
    sqrt_dt = math.sqrt(float(dt))
    for n in range(num_steps):
        chol, _ = eval_mod.cholesky_with_fallback_torch(covariance_schedule[n])
        increments[n] = (standard_normals[n] @ chol.T) * sqrt_dt
    return increments


def freeze_review_schedule(array: torch.Tensor, fine_per_review: int):
    return array.repeat_interleave(int(fine_per_review), dim=0)


def policy_subset(policies: Dict[str, callable]) -> Dict[str, callable]:
    return {
        "Deep_Splitting": policies["Deep_Splitting"],
        "Static": policies["Static"],
    }


def append_requery_rows(
    rows: List[dict],
    selected_idx_tensor: Optional[torch.Tensor],
    selected_path_ids: Sequence[int],
    evaluation_level: str,
    review_interval: int,
    fine_substep: int,
    fine_idx: int,
    Z: torch.Tensor,
    I: torch.Tensor,
    fixed_action: torch.Tensor,
    policy_fn,
    dt: float,
):
    if selected_idx_tensor is None or len(selected_path_ids) == 0:
        return

    Z_sel = Z.index_select(0, selected_idx_tensor)
    I_sel = I.index_select(0, selected_idx_tensor)
    fixed_sel = fixed_action.index_select(0, selected_idx_tensor)
    requery_raw = policy_fn(fine_idx, Z_sel, I_sel)
    requery_action, _, _ = eval_mod.cap_rate_to_inventory_torch(requery_raw, I_sel, dt)

    Z_np = Z_sel.detach().cpu().numpy()
    I_np = I_sel.detach().cpu().numpy()
    fixed_np = fixed_sel.detach().cpu().numpy()
    requery_np = requery_action.detach().cpu().numpy()

    for local_idx, path_id in enumerate(selected_path_ids):
        rows.append(
            {
                "path_id": int(path_id),
                "review_interval": int(review_interval),
                "fine_substep": int(fine_substep),
                "evaluation_level": evaluation_level,
                "fixed_action_a0": float(fixed_np[local_idx, 0]),
                "fixed_action_a1": float(fixed_np[local_idx, 1]),
                "fixed_action_a2": float(fixed_np[local_idx, 2]),
                "requeried_action_a0": float(requery_np[local_idx, 0]),
                "requeried_action_a1": float(requery_np[local_idx, 1]),
                "requeried_action_a2": float(requery_np[local_idx, 2]),
                "I_state_for_requery": float(I_np[local_idx]),
                "Z0_state_for_requery": float(Z_np[local_idx, 0]),
                "Z1_state_for_requery": float(Z_np[local_idx, 1]),
                "Z2_state_for_requery": float(Z_np[local_idx, 2]),
            }
        )


def simulate_periodic_fixed_action_policy(
    policy_fn,
    bsde,
    num_paths: int,
    device,
    evaluation_level: str,
    review_num_intervals: int,
    mu_review: torch.Tensor,
    gamma_review: torch.Tensor,
    shared_w_increments: Optional[torch.Tensor] = None,
    periodic_demands: Optional[torch.Tensor] = None,
    record_requery: bool = False,
    selected_path_ids: Optional[Sequence[int]] = None,
):
    dtype = bsde.mu.dtype
    dim = int(bsde.dim)
    fine_dt = float(bsde.delta_t)
    total_time = float(bsde.total_time)
    review_dt = total_time / int(review_num_intervals)
    num_fine_intervals = int(bsde.num_time_interval)
    fine_per_review = num_fine_intervals // int(review_num_intervals)
    eps = 1e-8

    if num_fine_intervals % int(review_num_intervals) != 0:
        raise ValueError(
            f"Fine grid must divide review grid exactly. Got N_fine={num_fine_intervals}, "
            f"N_review={review_num_intervals}."
        )

    fine_mu = bsde.mu.to(device=device, dtype=dtype)
    p = bsde.p.to(device=device, dtype=dtype)
    c_tilde = bsde.c_tilde.to(device=device, dtype=dtype)
    h = bsde.h.to(device=device, dtype=dtype)

    Z = torch.zeros((num_paths, dim), device=device, dtype=dtype)
    I = torch.full((num_paths,), float(bsde.I0), device=device, dtype=dtype)

    interval_start_z = torch.zeros((review_num_intervals, num_paths, dim), device=device, dtype=dtype)
    interval_start_i = torch.zeros((review_num_intervals, num_paths), device=device, dtype=dtype)
    interval_end_z = torch.zeros((review_num_intervals, num_paths, dim), device=device, dtype=dtype)
    interval_end_i = torch.zeros((review_num_intervals, num_paths), device=device, dtype=dtype)
    interval_actions = torch.zeros((review_num_intervals, num_paths, dim), device=device, dtype=dtype)
    interval_realized_demand = torch.zeros((review_num_intervals, num_paths, dim), device=device, dtype=dtype)
    interval_shortage_cost = torch.zeros((review_num_intervals, num_paths), device=device, dtype=dtype)
    interval_fairness_term = torch.zeros((review_num_intervals, num_paths), device=device, dtype=dtype)
    interval_total_cost = torch.zeros((review_num_intervals, num_paths), device=device, dtype=dtype)
    interval_stockouts = torch.zeros((review_num_intervals, num_paths, dim), device=device, dtype=dtype)

    total_shortage_cost = torch.zeros((num_paths,), device=device, dtype=dtype)
    total_fairness_cost = torch.zeros((num_paths,), device=device, dtype=dtype)

    requery_rows: List[dict] = []
    selected_idx_tensor = None
    selected_path_ids = list(selected_path_ids or [])
    if record_requery and selected_path_ids:
        selected_idx_tensor = torch.tensor(selected_path_ids, device=device, dtype=torch.long)

    for review_interval in range(review_num_intervals):
        fine_start = review_interval * fine_per_review
        interval_start_z[review_interval] = Z
        interval_start_i[review_interval] = I

        fixed_raw_action = policy_fn(fine_start, Z, I)
        fixed_action, _, _ = eval_mod.cap_rate_to_inventory_torch(fixed_raw_action, I, fine_dt)
        interval_actions[review_interval] = fixed_action

        interval_shortage = torch.zeros((num_paths,), device=device, dtype=dtype)
        interval_fairness = torch.zeros((num_paths,), device=device, dtype=dtype)
        interval_stockout = torch.zeros((num_paths, dim), device=device, dtype=dtype)
        interval_demand = torch.zeros((num_paths, dim), device=device, dtype=dtype)
        alloc_period = torch.zeros((num_paths, dim), device=device, dtype=dtype)

        if evaluation_level in (LEVEL_B, LEVEL_C):
            for fine_substep in range(fine_per_review):
                fine_idx = fine_start + fine_substep
                if record_requery:
                    append_requery_rows(
                        requery_rows,
                        selected_idx_tensor,
                        selected_path_ids,
                        evaluation_level,
                        review_interval,
                        fine_substep,
                        fine_idx,
                        Z,
                        I,
                        fixed_action,
                        policy_fn,
                        fine_dt,
                    )

                a_step, _, _ = eval_mod.cap_rate_to_inventory_torch(fixed_action, I, fine_dt)
                mu_t = mu_review[review_interval] if evaluation_level == LEVEL_B else fine_mu[fine_idx]
                W_increment = shared_w_increments[fine_idx]

                demand_increment = mu_t.unsqueeze(0) * fine_dt - W_increment
                interval_demand += demand_increment

                X_1 = Z + (a_step - mu_t.unsqueeze(0)) * fine_dt + W_increment
                Delta_L = torch.zeros_like(X_1)
                Delta_L_CHAs = torch.relu(-X_1[:, 1:])
                Delta_L[:, 1:] = Delta_L_CHAs

                referral_load = torch.sum(p.unsqueeze(0) * Delta_L_CHAs, dim=1)
                effective_HF_inventory = X_1[:, 0] - referral_load
                Delta_L[:, 0] = torch.relu(-effective_HF_inventory)

                Z_new = torch.zeros_like(X_1)
                Z_new[:, 0] = X_1[:, 0] + Delta_L[:, 0] - referral_load
                Z_new[:, 1:] = X_1[:, 1:] + Delta_L[:, 1:]

                I = torch.clamp(I - a_step.sum(dim=1) * fine_dt, min=0.0)
                Z = Z_new

                shortage_step = torch.sum(c_tilde.unsqueeze(0) * Delta_L, dim=1)
                interval_shortage += shortage_step

                fairness_step = float(bsde.lambda_param) * eqn.alpha_fair_utility_torch(
                    a_step,
                    mu_t,
                    float(bsde.alpha),
                    mu_epsilon=eps,
                )
                interval_fairness += fairness_step * fine_dt
                interval_stockout += Delta_L
                alloc_period += a_step * fine_dt

        elif evaluation_level == LEVEL_D:
            for fine_substep in range(fine_per_review):
                fine_idx = fine_start + fine_substep
                if record_requery:
                    append_requery_rows(
                        requery_rows,
                        selected_idx_tensor,
                        selected_path_ids,
                        evaluation_level,
                        review_interval,
                        fine_substep,
                        fine_idx,
                        Z,
                        I,
                        fixed_action,
                        policy_fn,
                        fine_dt,
                    )

                a_step, _, _ = eval_mod.cap_rate_to_inventory_torch(fixed_action, I, fine_dt)
                q_step = a_step * fine_dt
                Z = Z + q_step
                I = torch.clamp(I - q_step.sum(dim=1), min=0.0)
                alloc_period += q_step

            xi_n = periodic_demands[review_interval]
            interval_demand = xi_n

            L = torch.zeros_like(Z)
            Z_new = torch.zeros_like(Z)

            cha_short = torch.maximum(xi_n[:, 1:] - Z[:, 1:], torch.zeros_like(Z[:, 1:]))
            L[:, 1:] = cha_short
            Z_new[:, 1:] = torch.maximum(Z[:, 1:] - xi_n[:, 1:], torch.zeros_like(Z[:, 1:]))

            hf_total_demand = xi_n[:, 0] + torch.sum(cha_short * p.unsqueeze(0), dim=1)
            hf_short = torch.maximum(hf_total_demand - Z[:, 0], torch.zeros_like(Z[:, 0]))
            L[:, 0] = hf_short
            Z_new[:, 0] = torch.maximum(Z[:, 0] - hf_total_demand, torch.zeros_like(Z[:, 0]))

            expected_demand_n = mu_review[review_interval] * review_dt
            fairness_step = float(bsde.lambda_param) * eqn.alpha_fair_utility_torch(
                alloc_period,
                expected_demand_n,
                float(bsde.alpha),
                mu_epsilon=eps,
            )

            interval_shortage = torch.sum(L * c_tilde.unsqueeze(0), dim=1)
            interval_fairness = fairness_step
            interval_stockout = L
            Z = Z_new
        else:
            raise ValueError(f"Unsupported evaluation level: {evaluation_level}")

        interval_end_z[review_interval] = Z
        interval_end_i[review_interval] = I
        interval_realized_demand[review_interval] = interval_demand
        interval_shortage_cost[review_interval] = interval_shortage
        interval_fairness_term[review_interval] = interval_fairness
        interval_total_cost[review_interval] = interval_shortage - interval_fairness
        interval_stockouts[review_interval] = interval_stockout

        total_shortage_cost += interval_shortage
        total_fairness_cost += interval_fairness

    terminal_cost = torch.sum(Z * h.unsqueeze(0), dim=1)
    total_cost = total_shortage_cost - total_fairness_cost + terminal_cost

    return {
        "total_costs": total_cost.detach().cpu().numpy(),
        "shortage_costs": total_shortage_cost.detach().cpu().numpy(),
        "fairness_costs": total_fairness_cost.detach().cpu().numpy(),
        "terminal_costs": terminal_cost.detach().cpu().numpy(),
        "interval_start_z": interval_start_z.detach().cpu().numpy(),
        "interval_start_i": interval_start_i.detach().cpu().numpy(),
        "interval_end_z": interval_end_z.detach().cpu().numpy(),
        "interval_end_i": interval_end_i.detach().cpu().numpy(),
        "interval_actions": interval_actions.detach().cpu().numpy(),
        "interval_realized_demand": interval_realized_demand.detach().cpu().numpy(),
        "interval_shortage_cost": interval_shortage_cost.detach().cpu().numpy(),
        "interval_fairness_term": interval_fairness_term.detach().cpu().numpy(),
        "interval_total_cost": interval_total_cost.detach().cpu().numpy(),
        "interval_stockouts": interval_stockouts.detach().cpu().numpy(),
        "requery_rows": requery_rows,
        **eqn.fairness_metadata(),
    }


def summarize_ladder_level(level: str, ds_costs: dict, static_costs: dict):
    paired_total = np.asarray(ds_costs["total_costs"]) - np.asarray(static_costs["total_costs"])
    paired_shortage = np.asarray(ds_costs["shortage_costs"]) - np.asarray(static_costs["shortage_costs"])
    paired_fairness = np.asarray(ds_costs["fairness_costs"]) - np.asarray(static_costs["fairness_costs"])
    mean_diff, se_diff, ci_lower, ci_upper = mean_se_ci(paired_total)

    return {
        "evaluation_level": level,
        "level_description": LEVEL_DESCRIPTIONS[level],
        "ds_total_cost_mean": float(np.mean(ds_costs["total_costs"])),
        "static_total_cost_mean": float(np.mean(static_costs["total_costs"])),
        "mean_paired_difference": mean_diff,
        "standard_error": se_diff,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "mean_paired_shortage_cost_difference": float(np.mean(paired_shortage)),
        "mean_paired_fairness_term_difference": float(np.mean(paired_fairness)),
        "difference_formula": "Cost_Deep_Splitting - Cost_Static",
        "num_paths": int(len(paired_total)),
        **eqn.fairness_metadata(),
    }


def build_interval_paired_diagnostics(level: str, ds_costs: dict, static_costs: dict):
    ds_total = np.asarray(ds_costs["interval_total_cost"])
    st_total = np.asarray(static_costs["interval_total_cost"])
    ds_short = np.asarray(ds_costs["interval_shortage_cost"])
    st_short = np.asarray(static_costs["interval_shortage_cost"])
    ds_fair = np.asarray(ds_costs["interval_fairness_term"])
    st_fair = np.asarray(static_costs["interval_fairness_term"])
    ds_i_end = np.asarray(ds_costs["interval_end_i"])
    st_i_end = np.asarray(static_costs["interval_end_i"])
    ds_stock = np.asarray(ds_costs["interval_stockouts"])
    st_stock = np.asarray(static_costs["interval_stockouts"])

    review_num_intervals, num_paths = ds_total.shape
    rows = []
    for review_interval in range(review_num_intervals):
        for path_id in range(num_paths):
            rows.append(
                {
                    "path_id": int(path_id),
                    "review_interval": int(review_interval),
                    "evaluation_level": level,
                    "delta_total_cost": float(ds_total[review_interval, path_id] - st_total[review_interval, path_id]),
                    "delta_shortage_cost": float(ds_short[review_interval, path_id] - st_short[review_interval, path_id]),
                    "delta_fairness_term": float(ds_fair[review_interval, path_id] - st_fair[review_interval, path_id]),
                    "delta_central_inventory_end": float(ds_i_end[review_interval, path_id] - st_i_end[review_interval, path_id]),
                    "delta_stockout_node0": float(ds_stock[review_interval, path_id, 0] - st_stock[review_interval, path_id, 0]),
                    "delta_stockout_node1": float(ds_stock[review_interval, path_id, 1] - st_stock[review_interval, path_id, 1]),
                    "delta_stockout_node2": float(ds_stock[review_interval, path_id, 2] - st_stock[review_interval, path_id, 2]),
                }
            )
    return rows


def build_replay_traces(level: str, ds_costs: dict, static_costs: dict, selected_path_ids: Sequence[int]):
    rows = []
    for path_id in selected_path_ids:
        for review_interval in range(ds_costs["interval_total_cost"].shape[0]):
            ds_start_z = ds_costs["interval_start_z"][review_interval, path_id]
            st_start_z = static_costs["interval_start_z"][review_interval, path_id]
            ds_end_z = ds_costs["interval_end_z"][review_interval, path_id]
            st_end_z = static_costs["interval_end_z"][review_interval, path_id]
            ds_action = ds_costs["interval_actions"][review_interval, path_id]
            st_action = static_costs["interval_actions"][review_interval, path_id]
            realized_demand = ds_costs["interval_realized_demand"][review_interval, path_id]

            rows.append(
                {
                    "path_id": int(path_id),
                    "review_interval": int(review_interval),
                    "evaluation_level": level,
                    "I_start_DS": float(ds_costs["interval_start_i"][review_interval, path_id]),
                    "I_start_Static": float(static_costs["interval_start_i"][review_interval, path_id]),
                    "Z0_start_DS": float(ds_start_z[0]),
                    "Z1_start_DS": float(ds_start_z[1]),
                    "Z2_start_DS": float(ds_start_z[2]),
                    "Z0_start_Static": float(st_start_z[0]),
                    "Z1_start_Static": float(st_start_z[1]),
                    "Z2_start_Static": float(st_start_z[2]),
                    "a0_DS": float(ds_action[0]),
                    "a1_DS": float(ds_action[1]),
                    "a2_DS": float(ds_action[2]),
                    "a0_Static": float(st_action[0]),
                    "a1_Static": float(st_action[1]),
                    "a2_Static": float(st_action[2]),
                    "realized_demand_0": float(realized_demand[0]),
                    "realized_demand_1": float(realized_demand[1]),
                    "realized_demand_2": float(realized_demand[2]),
                    "I_end_DS": float(ds_costs["interval_end_i"][review_interval, path_id]),
                    "I_end_Static": float(static_costs["interval_end_i"][review_interval, path_id]),
                    "Z0_end_DS": float(ds_end_z[0]),
                    "Z1_end_DS": float(ds_end_z[1]),
                    "Z2_end_DS": float(ds_end_z[2]),
                    "Z0_end_Static": float(st_end_z[0]),
                    "Z1_end_Static": float(st_end_z[1]),
                    "Z2_end_Static": float(st_end_z[2]),
                    "shortage_cost_DS": float(ds_costs["interval_shortage_cost"][review_interval, path_id]),
                    "shortage_cost_Static": float(static_costs["interval_shortage_cost"][review_interval, path_id]),
                    "fairness_term_DS": float(ds_costs["interval_fairness_term"][review_interval, path_id]),
                    "fairness_term_Static": float(static_costs["interval_fairness_term"][review_interval, path_id]),
                }
            )
    return rows


def action_difference_summary(requery_df: pd.DataFrame, level: str, tol: float = 1e-4):
    level_df = requery_df[requery_df["evaluation_level"] == level]
    if level_df.empty:
        return {
            "share_changed": float("nan"),
            "mean_l1_gap": float("nan"),
            "max_l1_gap": float("nan"),
        }

    diffs = np.abs(
        level_df[["requeried_action_a0", "requeried_action_a1", "requeried_action_a2"]].to_numpy()
        - level_df[["fixed_action_a0", "fixed_action_a1", "fixed_action_a2"]].to_numpy()
    )
    l1_gap = diffs.sum(axis=1)
    changed = np.any(diffs > tol, axis=1)
    return {
        "share_changed": float(np.mean(changed)),
        "mean_l1_gap": float(np.mean(l1_gap)),
        "max_l1_gap": float(np.max(l1_gap)),
    }


def first_underperform_interval(interval_diag_df: pd.DataFrame, level: str):
    level_df = interval_diag_df[interval_diag_df["evaluation_level"] == level]
    if level_df.empty:
        return None
    grouped = (
        level_df.groupby("review_interval", as_index=False)["delta_total_cost"]
        .mean()
        .sort_values("review_interval")
    )
    positive = grouped[grouped["delta_total_cost"] > 0]
    if positive.empty:
        return None
    return int(positive.iloc[0]["review_interval"])


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator] + body)


def fmt(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{float(x):.6f}"


def write_note(
    output_path: str,
    ladder_df: pd.DataFrame,
    interval_diag_df: pd.DataFrame,
    requery_df: pd.DataFrame,
    selected_path_ids: Sequence[int],
):
    level_rows = [[level, LEVEL_DESCRIPTIONS[level]] for level in [LEVEL_A, LEVEL_B, LEVEL_C, LEVEL_D]]
    result_rows = []
    for _, row in ladder_df.iterrows():
        result_rows.append(
            [
                str(row["evaluation_level"]),
                fmt(row["ds_total_cost_mean"]),
                fmt(row["static_total_cost_mean"]),
                fmt(row["mean_paired_difference"]),
                fmt(row["standard_error"]),
                f"[{fmt(row['ci95_lower'])}, {fmt(row['ci95_upper'])}]",
                fmt(row["mean_paired_shortage_cost_difference"]),
                fmt(row["mean_paired_fairness_term_difference"]),
            ]
        )

    level_to_mean_diff = {
        row["evaluation_level"]: float(row["mean_paired_difference"])
        for _, row in ladder_df.iterrows()
    }

    harm_lower_update = level_to_mean_diff[LEVEL_B] - level_to_mean_diff[LEVEL_A]
    harm_frozen_dynamics = level_to_mean_diff[LEVEL_B] - level_to_mean_diff[LEVEL_C]
    harm_discrete_demand = level_to_mean_diff[LEVEL_D] - level_to_mean_diff[LEVEL_C]

    harm_components = {
        "lower action-update frequency": harm_lower_update,
        "frozen interval dynamics": harm_frozen_dynamics,
        "switching to discrete/NB demand": harm_discrete_demand,
    }
    positive_harms = {name: value for name, value in harm_components.items() if value > 0}
    dominant_driver = (
        max(positive_harms.items(), key=lambda kv: kv[1])[0]
        if positive_harms
        else "no positive deterioration identified"
    )

    interval_rows = []
    for level in PERIODIC_LEVELS:
        first_bad = first_underperform_interval(interval_diag_df, level)
        interval_rows.append([level, "none" if first_bad is None else str(first_bad)])

    requery_rows = []
    for level in PERIODIC_LEVELS:
        summary = action_difference_summary(requery_df, level)
        requery_rows.append(
            [
                level,
                fmt(summary["share_changed"]),
                fmt(summary["mean_l1_gap"]),
                fmt(summary["max_l1_gap"]),
            ]
        )

    note = "\n".join(
        [
            "# Deployment Mismatch Diagnostic Note",
            "",
            "## Levels",
            markdown_table(["Level", "Meaning"], level_rows),
            "",
            "## Main Paired Results",
            markdown_table(
                [
                    "Level",
                    "DS total cost",
                    "Static total cost",
                    "Mean paired diff",
                    "SE",
                    "95% CI",
                    "Paired shortage diff",
                    "Paired fairness diff",
                ],
                result_rows,
            ),
            "",
            "## Attribution",
            f"- Lower action-update frequency contribution (A -> B): {fmt(harm_lower_update)}",
            f"- Frozen interval dynamics contribution (B - C): {fmt(harm_frozen_dynamics)}",
            f"- Switching to discrete/NB demand contribution (C -> D): {fmt(harm_discrete_demand)}",
            f"- Largest deterioration driver in this run: {dominant_driver}",
            "",
            "## Interval Where DS Starts To Lose",
            markdown_table(["Level", "First interval with mean(delta_total_cost) > 0"], interval_rows),
            "",
            "## Intra-Interval Requery Signal",
            f"Selected replay paths: {', '.join(str(path_id) for path_id in selected_path_ids)}",
            markdown_table(
                ["Level", "Share changed", "Mean L1 gap", "Max L1 gap"],
                requery_rows,
            ),
            "",
            "A large requery-change share or large action gap indicates that Deep Splitting likely depends on higher-frequency state feedback that periodic-review execution suppresses.",
            "",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(note)


def save_dataframe(df: pd.DataFrame, output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return path


def main():
    args = parse_args()
    run_name, run_dir, config_path, logs_path, output_dir = resolve_run_paths(args)

    context = eval_mod.prepare_evaluation_context(
        config_path,
        run_name,
        logs_path=logs_path,
    )
    config = context["config"]
    bsde = context["bsde"]
    device = context["device"]
    policies = policy_subset(context["policies"])

    review_params = eval_mod.prepare_periodic_review_parameters(
        config,
        review_num_intervals=args.review_num_intervals,
        review_dt=args.review_dt,
    )
    review_num_intervals = int(review_params["num_review_intervals"])
    fine_per_review = int(bsde.num_time_interval // review_num_intervals)
    if bsde.num_time_interval % review_num_intervals != 0:
        raise ValueError(
            f"Review grid must divide the fine grid exactly. Got N_fine={bsde.num_time_interval}, "
            f"N_review={review_num_intervals}."
        )

    mu_review = torch.tensor(review_params["mu_review"], device=device, dtype=bsde.mu.dtype)
    gamma_review = torch.tensor(review_params["gamma_review"], device=device, dtype=bsde.mu.dtype)

    standard_normals = precompute_standard_normals(
        int(bsde.num_time_interval),
        int(args.num_paths),
        int(bsde.dim),
        device,
        bsde.mu.dtype,
        args.fine_grid_seed,
    )
    shared_w_fine = build_brownian_increments(standard_normals, bsde.Gamma.to(device=device, dtype=bsde.mu.dtype), bsde.delta_t)
    frozen_gamma_schedule = freeze_review_schedule(gamma_review, fine_per_review)
    shared_w_frozen = build_brownian_increments(standard_normals, frozen_gamma_schedule, bsde.delta_t)

    periodic_demands_np = eval_mod.precompute_periodic_demands(
        review_params["mu_review"],
        review_params["gamma_review"],
        float(bsde.total_time) / review_num_intervals,
        int(args.num_paths),
        demand_mode=args.periodic_demand_mode,
        seed=args.periodic_seed,
    )
    periodic_demands = torch.tensor(periodic_demands_np, device=device, dtype=bsde.mu.dtype)

    selected_path_ids = select_replay_paths(args.num_paths, args.replay_path_ids, args.num_replay_paths)

    level_costs = {LEVEL_A: {}, LEVEL_B: {}, LEVEL_C: {}, LEVEL_D: {}}

    for policy_name, policy_fn in policies.items():
        level_costs[LEVEL_A][policy_name] = eval_mod.simulate_trajectories(
            policy_fn,
            bsde,
            args.num_paths,
            device,
            shared_w_increments=shared_w_fine,
        )

        level_costs[LEVEL_B][policy_name] = simulate_periodic_fixed_action_policy(
            policy_fn,
            bsde,
            args.num_paths,
            device,
            LEVEL_B,
            review_num_intervals,
            mu_review,
            gamma_review,
            shared_w_increments=shared_w_frozen,
            selected_path_ids=selected_path_ids,
            record_requery=(policy_name == "Deep_Splitting"),
        )

        level_costs[LEVEL_C][policy_name] = simulate_periodic_fixed_action_policy(
            policy_fn,
            bsde,
            args.num_paths,
            device,
            LEVEL_C,
            review_num_intervals,
            mu_review,
            gamma_review,
            shared_w_increments=shared_w_fine,
            selected_path_ids=selected_path_ids,
            record_requery=(policy_name == "Deep_Splitting"),
        )

        level_costs[LEVEL_D][policy_name] = simulate_periodic_fixed_action_policy(
            policy_fn,
            bsde,
            args.num_paths,
            device,
            LEVEL_D,
            review_num_intervals,
            mu_review,
            gamma_review,
            periodic_demands=periodic_demands,
            selected_path_ids=selected_path_ids,
            record_requery=(policy_name == "Deep_Splitting"),
        )

    ladder_rows = []
    interval_rows = []
    replay_rows = []
    requery_rows = []

    for level in [LEVEL_A, LEVEL_B, LEVEL_C, LEVEL_D]:
        ds_costs = level_costs[level]["Deep_Splitting"]
        static_costs = level_costs[level]["Static"]
        ladder_rows.append(summarize_ladder_level(level, ds_costs, static_costs))

        if level in PERIODIC_LEVELS:
            interval_rows.extend(build_interval_paired_diagnostics(level, ds_costs, static_costs))
            replay_rows.extend(build_replay_traces(level, ds_costs, static_costs, selected_path_ids))
            requery_rows.extend(ds_costs["requery_rows"])

    ladder_df = pd.DataFrame(ladder_rows)
    interval_diag_df = pd.DataFrame(interval_rows)
    replay_df = pd.DataFrame(replay_rows)
    requery_df = pd.DataFrame(requery_rows)

    ladder_path = save_dataframe(ladder_df, output_dir, LADDER_SUMMARY_FILE)
    interval_path = save_dataframe(interval_diag_df, output_dir, INTERVAL_DIAGNOSTICS_FILE)
    replay_path = save_dataframe(replay_df, output_dir, REPLAY_TRACES_FILE)
    requery_path = save_dataframe(requery_df, output_dir, INTRA_INTERVAL_REQUERY_FILE)

    note_path = os.path.join(output_dir, NOTE_FILE)
    write_note(note_path, ladder_df, interval_diag_df, requery_df, selected_path_ids)

    print(f"Run: {run_name}")
    print(f"Run directory: {run_dir}")
    print(f"Review intervals: {review_num_intervals}")
    print(f"Replay path ids: {selected_path_ids}")
    print(f"Saved: {ladder_path}")
    print(f"Saved: {interval_path}")
    print(f"Saved: {replay_path}")
    print(f"Saved: {requery_path}")
    print(f"Saved: {note_path}")


if __name__ == "__main__":
    main()
