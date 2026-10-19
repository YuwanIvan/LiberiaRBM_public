#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import equation as eqn
import evaluate as eval_mod
from configs.fangonda_weekly import scenarios as scenario_registry
from scripts.build_fangonda_scenarios import materialize_run

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised in environments without Gurobi
    gp = None
    GRB = None


LOCATION_LABELS = ("HF", "CHA 1", "CHA 2", "CHA 3", "CHA 4", "CHA 5", "CHA 6", "CHA 7")
QUANTILES = (("p05", 0.05), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p95", 0.95))
MU_RATE_EVIDENCE = (
    "evaluate.load_raw_parameter_arrays documents raw arrays as demand-rate and covariance-rate arrays.",
    "evaluate.resample_parameters_to_review_grid averages rates across nested review grids.",
    "evaluate.sample_periodic_demand converts mu_n to period mean by multiplying by dt.",
    "equation.compute_paper_eta computes total expected demand as mu.sum() * step.",
)


@dataclass
class LPContext:
    raw_config: Dict[str, Any]
    config: Any
    config_path: Path
    output_dir: Path
    bsde: Any
    device: torch.device
    mu_review: np.ndarray
    gamma_review: np.ndarray
    review_dt: float
    q0: float
    z0: np.ndarray
    p: np.ndarray
    c: np.ndarray
    r: np.ndarray
    c_tilde: np.ndarray
    eta: np.ndarray
    eta_review: np.ndarray
    materialized: Optional[Dict[str, str]]
    warnings: List[str]


@dataclass
class DemandResolution:
    interpretation: str
    period_demand: np.ndarray
    diagnostics: Dict[str, Any]
    warnings: List[str]


@dataclass
class LPSolution:
    status: int
    status_name: str
    runtime: float
    objective: Optional[float]
    allocation: np.ndarray
    warehouse: np.ndarray
    local_inventory: np.ndarray
    shortage: np.ndarray


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: to_namespace(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(value) for value in obj]
    return obj


def location_label(index: int) -> str:
    if index < len(LOCATION_LABELS):
        return LOCATION_LABELS[index]
    return f"loc_{index}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a mean-demand deterministic compact LP benchmark with Gurobi."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--config_path", "--config-path", default=None, help="Existing config JSON path.")
    source_group.add_argument(
        "--scenario",
        default=None,
        help="Fangonda scenario to materialize before solving. Defaults to fangonda_base if no config is given.",
    )
    parser.add_argument("--run-name", default=None, help="Run name used when materializing a scenario.")
    parser.add_argument(
        "--materialize-output-root",
        default=None,
        help=(
            "Optional root directory for scenario materialization. The materialized "
            "config is written under <root>/<run-name> instead of <output-dir>/_materialized/<run-name>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="output_mean_demand_dwaf_v1/md_lp",
        help="Directory for this LP variant's outputs.",
    )

    rate_group = parser.add_mutually_exclusive_group()
    rate_group.add_argument("--with-rate-bound", action="store_true", dest="with_rate_bound")
    rate_group.add_argument("--no-rate-bound", action="store_false", dest="with_rate_bound")
    parser.set_defaults(with_rate_bound=False)

    mu_group = parser.add_mutually_exclusive_group()
    mu_group.add_argument("--mu-is-rate", action="store_const", const="rate", dest="mu_interpretation")
    mu_group.add_argument(
        "--mu-is-period-demand",
        action="store_const",
        const="period_demand",
        dest="mu_interpretation",
    )
    parser.set_defaults(mu_interpretation=None)

    parser.add_argument("--review_num_intervals", "--review-num-intervals", type=int, default=None)
    parser.add_argument("--review_dt", "--review-dt", type=float, default=None)
    parser.add_argument("--num_time_interval", "--num-time-interval", type=int, default=None)
    parser.add_argument("--num-paths", "--num_paths", type=int, default=10000)
    parser.add_argument(
        "--periodic_demand_mode",
        "--periodic-demand-mode",
        choices=eval_mod.PERIODIC_DEMAND_MODES,
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gurobi-threads", type=int, default=None)
    parser.add_argument("--gurobi-time-limit", type=float, default=None)
    parser.add_argument("--gurobi-output", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--solve-only",
        "--skip-simulation",
        action="store_true",
        help="Only solve the deterministic LP and skip stochastic open-loop simulation.",
    )
    parser.add_argument("--write-history-npz", action="store_true", help="Optionally save stochastic history arrays.")
    return parser.parse_args()


def build_materialize_args(args: argparse.Namespace, output_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=str(output_root),
        num_time_interval=args.num_time_interval,
        num_layers=None,
        final_iterations=None,
        inner_iterations=None,
        batch_size=None,
        valid_size=None,
        reference_policy=None,
        eval_seed=args.seed,
        eval_num_paths=args.num_paths,
        eval_ext_num_paths=args.num_paths,
    )


def load_or_materialize_config(args: argparse.Namespace) -> tuple[Dict[str, Any], Path, Optional[Dict[str, str]]]:
    output_dir = Path(args.output_dir).expanduser()
    if args.config_path:
        config_path = Path(args.config_path).expanduser()
        with open(config_path, encoding="utf-8") as f:
            return json.load(f), config_path, None

    scenario_name = args.scenario or scenario_registry.BASELINE_NAME
    run_name = args.run_name or f"{scenario_name}_mean_demand_lp_config"
    materialized_root = (
        Path(args.materialize_output_root).expanduser()
        if args.materialize_output_root
        else output_dir / "_materialized"
    )
    resolved_spec = scenario_registry.resolve_scenario_spec(scenario_name)
    materialized = materialize_run(
        run_name,
        resolved_spec,
        build_materialize_args(args, materialized_root),
    )
    config_path = Path(materialized["config_path"])
    with open(config_path, encoding="utf-8") as f:
        return json.load(f), config_path, materialized


def build_lp_context(args: argparse.Namespace) -> LPContext:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config, config_path, materialized = load_or_materialize_config(args)
    config = to_namespace(raw_config)
    device = torch.device("cpu")
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    eval_mod.move_bsde_to_device(bsde, device)
    eval_mod.attach_cost_components(config, bsde, device)

    review_params = eval_mod.prepare_discrete_review_parameters(
        config,
        review_num_intervals=args.review_num_intervals,
        review_dt=args.review_dt,
    )
    mu_review = np.asarray(review_params["mu_review"], dtype=np.float64)
    gamma_review = np.asarray(review_params["gamma_review"], dtype=np.float64)
    review_dt = float(bsde.total_time) / float(mu_review.shape[0])
    dim = int(bsde.dim)

    p = bsde.p.detach().cpu().numpy().astype(np.float64)
    c = bsde.u.detach().cpu().numpy().reshape(-1).astype(np.float64)
    r = bsde.r.detach().cpu().numpy().astype(np.float64)
    c_tilde = bsde.c_tilde.detach().cpu().numpy().astype(np.float64)
    eta = bsde.eta.detach().cpu().numpy().astype(np.float64)
    eta_review = bsde.eta_for_review_grid(mu_review.shape[0]).detach().cpu().numpy().astype(np.float64)
    z0 = np.zeros(dim, dtype=np.float64)
    q0 = float(bsde.I0)

    warnings: List[str] = []
    if bsde.num_time_interval % mu_review.shape[0] != 0:
        warnings.append(
            "Stochastic open-loop evaluation does not require fine-grid divisibility, "
            f"but existing policy aggregation would not divide exactly: N_fine={bsde.num_time_interval}, "
            f"N_review={mu_review.shape[0]}."
        )
    if q0 > 0.0 and z0[0] > 0.0:
        warnings.append(
            "Both warehouse inventory and initial HF inventory are positive; check for possible initial-inventory double counting."
        )

    return LPContext(
        raw_config=raw_config,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        bsde=bsde,
        device=device,
        mu_review=mu_review,
        gamma_review=gamma_review,
        review_dt=review_dt,
        q0=q0,
        z0=z0,
        p=p,
        c=c,
        r=r,
        c_tilde=c_tilde,
        eta=eta,
        eta_review=eta_review,
        materialized=materialized,
        warnings=warnings,
    )


def load_scenario_manifest(config_path: Path) -> Optional[Dict[str, Any]]:
    manifest_path = config_path.parent / "scenario_manifest.json"
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_period_demand(
    mu_review: np.ndarray,
    review_dt: float,
    requested_interpretation: Optional[str],
    config_path: Optional[Path] = None,
) -> DemandResolution:
    warnings: List[str] = []
    if requested_interpretation is None:
        interpretation = "rate"
        source = "inferred_from_existing_code"
    else:
        interpretation = requested_interpretation
        source = "explicit_cli_flag"

    if interpretation == "rate":
        period_demand = mu_review * float(review_dt)
    elif interpretation == "period_demand":
        period_demand = mu_review.copy()
        warnings.append(
            "CLI requested mu as period demand, but existing evaluation and eta code interpret mu as a rate."
        )
    else:
        raise ValueError(f"Unsupported mu interpretation: {interpretation}")

    direct_totals = mu_review.sum(axis=0)
    rate_totals = (mu_review * float(review_dt)).sum(axis=0)
    scenario_manifest = load_scenario_manifest(config_path) if config_path is not None else None
    manifest_total = None
    if scenario_manifest:
        manifest_total = scenario_manifest.get("metadata", {}).get("total_expected_demand")

    diagnostics = {
        "mu_interpretation": interpretation,
        "mu_interpretation_source": source,
        "code_evidence": list(MU_RATE_EVIDENCE),
        "review_period_length_h": float(review_dt),
        "num_review_periods": int(mu_review.shape[0]),
        "direct_mu_sum_by_location": direct_totals.tolist(),
        "direct_mu_sum_all_locations": float(direct_totals.sum()),
        "rate_based_total_mean_demand_by_location": rate_totals.tolist(),
        "rate_based_total_mean_demand_all_locations": float(rate_totals.sum()),
        "lp_total_mean_demand_by_location": period_demand.sum(axis=0).tolist(),
        "lp_total_mean_demand_all_locations": float(period_demand.sum()),
        "scenario_manifest_total_expected_demand": manifest_total,
    }
    if not np.allclose(direct_totals, rate_totals, rtol=1e-10, atol=1e-10):
        diagnostics["direct_vs_rate_total_difference_by_location"] = (direct_totals - rate_totals).tolist()
    if manifest_total is not None and not np.isclose(float(manifest_total), float(rate_totals.sum()), rtol=1e-8, atol=1e-8):
        warnings.append(
            "Scenario manifest total expected demand differs from mu_review * h total; check review-grid alignment."
        )
    return DemandResolution(
        interpretation=interpretation,
        period_demand=np.asarray(period_demand, dtype=np.float64),
        diagnostics=diagnostics,
        warnings=warnings,
    )


def build_and_solve_compact_lp(
    context: LPContext,
    demand: np.ndarray,
    with_rate_bound: bool,
    args: argparse.Namespace,
) -> LPSolution:
    if gp is None or GRB is None:
        raise ImportError("gurobipy is required to solve the LP but is not importable.")

    output_dir = context.output_dir
    dim = int(context.bsde.dim)
    periods = int(demand.shape[0])
    if demand.shape != (periods, dim):
        raise ValueError(f"demand shape {demand.shape} is inconsistent with dim={dim}.")

    lambda_param = float(context.bsde.lambda_param)
    alpha = float(context.bsde.alpha)
    model_name = "mean_demand_compact_qcp" if lambda_param > 0.0 else "mean_demand_compact_lp"
    model = gp.Model(model_name)
    model.Params.OutputFlag = int(args.gurobi_output)
    model.Params.LogFile = str(output_dir / "gurobi.log")
    if args.gurobi_threads is not None:
        model.Params.Threads = int(args.gurobi_threads)
    if args.gurobi_time_limit is not None:
        model.Params.TimeLimit = float(args.gurobi_time_limit)

    a = model.addVars(periods, dim, lb=0.0, name="a")
    Q = model.addVars(periods, lb=0.0, name="Q")
    Z = model.addVars(periods, dim, lb=0.0, name="Z")
    ell = model.addVars(periods, dim, lb=0.0, name="ell")
    y = None
    if lambda_param > 0.0:
        if not np.isclose(alpha, 0.5):
            raise ValueError("The deterministic fluid QCP implementation currently requires alpha=0.5.")
        y = model.addVars(periods, dim, lb=0.0, name="y")

    for n in range(periods):
        q_prev = context.q0 if n == 0 else Q[n - 1]
        model.addConstr(Q[n] == q_prev - gp.quicksum(a[n, k] for k in range(dim)), name=f"warehouse_{n + 1}")
        for k in range(1, dim):
            z_prev = float(context.z0[k]) if n == 0 else Z[n - 1, k]
            model.addConstr(
                Z[n, k] == z_prev + a[n, k] - float(demand[n, k]) + ell[n, k],
                name=f"local_cha_{n + 1}_{k}",
            )
        z0_prev = float(context.z0[0]) if n == 0 else Z[n - 1, 0]
        referral_shortage = gp.quicksum(float(context.p[k - 1]) * ell[n, k] for k in range(1, dim))
        model.addConstr(
            Z[n, 0]
            == z0_prev + a[n, 0] - float(demand[n, 0]) - referral_shortage + ell[n, 0],
            name=f"local_hf_{n + 1}",
        )
        if with_rate_bound:
            for k in range(dim):
                model.addConstr(
                    a[n, k] <= float(context.eta_review[n, k]) * context.review_dt,
                    name=f"rate_bound_{n + 1}_{k}",
                )
        if y is not None:
            for k in range(dim):
                mu_nk = max(float(demand[n, k]), 0.0)
                if mu_nk <= 1e-12:
                    model.addConstr(y[n, k] == 0.0, name=f"fairness_zero_mu_{n + 1}_{k}")
                else:
                    model.addQConstr(
                        y[n, k] * y[n, k] <= mu_nk * a[n, k],
                        name=f"fairness_qc_{n + 1}_{k}",
                    )

    objective = gp.quicksum(
        float(context.c_tilde[k]) * ell[n, k]
        for n in range(periods)
        for k in range(dim)
    )
    if y is not None:
        objective -= 2.0 * lambda_param * gp.quicksum(
            y[n, k]
            for n in range(periods)
            for k in range(dim)
        )
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    status = int(model.Status)
    status_name = _gurobi_status_name(status)
    has_solution = int(model.SolCount) > 0
    allocation = np.zeros((periods, dim), dtype=np.float64)
    warehouse = np.zeros(periods, dtype=np.float64)
    local_inventory = np.zeros((periods, dim), dtype=np.float64)
    shortage = np.zeros((periods, dim), dtype=np.float64)
    if has_solution:
        for n in range(periods):
            warehouse[n] = float(Q[n].X)
            for k in range(dim):
                allocation[n, k] = float(a[n, k].X)
                local_inventory[n, k] = float(Z[n, k].X)
                shortage[n, k] = float(ell[n, k].X)
        objective_value = float(model.ObjVal)
    else:
        objective_value = None

    return LPSolution(
        status=status,
        status_name=status_name,
        runtime=float(model.Runtime),
        objective=objective_value,
        allocation=allocation,
        warehouse=warehouse,
        local_inventory=local_inventory,
        shortage=shortage,
    )


def _gurobi_status_name(status: int) -> str:
    if GRB is None:
        return str(status)
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, str(status))


def extract_solution(solution: LPSolution) -> Dict[str, np.ndarray]:
    return {
        "allocation": solution.allocation,
        "warehouse": solution.warehouse,
        "local_inventory": solution.local_inventory,
        "shortage": solution.shortage,
    }


def validate_lp_solution(
    context: LPContext,
    demand: np.ndarray,
    solution: LPSolution,
    with_rate_bound: bool,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    a = solution.allocation
    Q = solution.warehouse
    Z = solution.local_inventory
    ell = solution.shortage
    dim = int(context.bsde.dim)
    periods = int(demand.shape[0])

    warehouse_residuals = []
    local_residuals = []
    for n in range(periods):
        q_prev = context.q0 if n == 0 else Q[n - 1]
        warehouse_residuals.append(Q[n] - (q_prev - a[n].sum()))
        for k in range(1, dim):
            z_prev = context.z0[k] if n == 0 else Z[n - 1, k]
            local_residuals.append(Z[n, k] - (z_prev + a[n, k] - demand[n, k] + ell[n, k]))
        z0_prev = context.z0[0] if n == 0 else Z[n - 1, 0]
        referral = float(np.dot(context.p, ell[n, 1:]))
        local_residuals.append(Z[n, 0] - (z0_prev + a[n, 0] - demand[n, 0] - referral + ell[n, 0]))

    min_variable = float(min(np.min(a), np.min(Q), np.min(Z), np.min(ell))) if periods > 0 else 0.0
    rate_bound_violation = 0.0
    if with_rate_bound:
        rate_bound_violation = max(0.0, float(np.max(a - context.eta_review * context.review_dt)))
    total_allocation_excess = float(a.sum() - context.q0)
    initial_double_count_warning = bool(context.q0 > tol and context.z0[0] > tol)

    checks = {
        "all_variables_nonnegative": bool(min_variable >= -tol),
        "min_variable_value": min_variable,
        "maximum_warehouse_balance_residual": float(np.max(np.abs(warehouse_residuals))) if warehouse_residuals else 0.0,
        "maximum_local_balance_residual": float(np.max(np.abs(local_residuals))) if local_residuals else 0.0,
        "maximum_rate_bound_violation": rate_bound_violation,
        "rate_bound_satisfied": bool((not with_rate_bound) or rate_bound_violation <= tol),
        "total_allocation": float(a.sum()),
        "total_allocation_minus_q0": total_allocation_excess,
        "total_allocation_within_initial_warehouse": bool(total_allocation_excess <= tol),
        "initial_inventory_appears_double_counted": initial_double_count_warning,
    }
    checks["passed"] = bool(
        checks["all_variables_nonnegative"]
        and checks["maximum_warehouse_balance_residual"] <= tol
        and checks["maximum_local_balance_residual"] <= tol
        and checks["rate_bound_satisfied"]
        and checks["total_allocation_within_initial_warehouse"]
    )
    return checks


def summarize_lp_solution(
    context: LPContext,
    demand: np.ndarray,
    solution: LPSolution,
    with_rate_bound: bool,
    validation: Mapping[str, Any],
    demand_resolution: DemandResolution,
) -> Dict[str, Any]:
    family = "MD-QCP" if float(context.bsde.lambda_param) > 0.0 else "MD-LP"
    variant = f"{family}-eta" if with_rate_bound else family
    allocation = solution.allocation
    shortage = solution.shortage
    final_warehouse = float(solution.warehouse[-1]) if solution.warehouse.size else context.q0
    final_local = solution.local_inventory[-1] if solution.local_inventory.size else context.z0.copy()
    return {
        "variant_name": variant,
        "with_rate_bound": bool(with_rate_bound),
        "gurobi_status": solution.status,
        "gurobi_status_name": solution.status_name,
        "gurobi_runtime": solution.runtime,
        "lp_objective": solution.objective,
        "mu_interpretation": demand_resolution.interpretation,
        "period_length_h": context.review_dt,
        "num_periods": int(demand.shape[0]),
        "dim": int(context.bsde.dim),
        "q0": context.q0,
        "z0": context.z0.tolist(),
        "lambda_param": float(context.bsde.lambda_param),
        "alpha": float(context.bsde.alpha),
        **eqn.fairness_metadata(),
        "u": float(context.c[0]),
        "referral_costs_r": context.r.tolist(),
        "p": context.p.tolist(),
        "effective_shortage_costs": context.c_tilde.tolist(),
        "eta_peak_diagnostic_vector": context.eta.tolist(),
        "eta_review": context.eta_review.tolist(),
        "rate_bound_rhs_eta_h": (context.eta_review * context.review_dt).tolist() if with_rate_bound else None,
        "total_mean_demand_by_location": demand.sum(axis=0).tolist(),
        "aggregate_total_mean_demand": float(demand.sum()),
        "total_allocation_by_location": allocation.sum(axis=0).tolist(),
        "total_shortage_by_location": shortage.sum(axis=0).tolist(),
        "final_warehouse_inventory": final_warehouse,
        "final_local_inventories": final_local.tolist(),
        "validation": dict(validation),
        "warnings": list(context.warnings) + list(demand_resolution.warnings),
        "config_path": str(context.config_path),
        "output_dir": str(context.output_dir),
    }


def array_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_se": float("nan")}
    out: Dict[str, float] = {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_se": float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }
    for name, quantile in QUANTILES:
        out[f"{prefix}_{name}"] = float(np.quantile(arr, quantile))
    return out


def location_distribution_stats(values: np.ndarray, prefix: str) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    stats: Dict[str, Any] = {}
    for k in range(arr.shape[1]):
        label = location_label(k).replace(" ", "_").lower()
        stats[f"{prefix}_{label}_mean"] = float(np.mean(arr[:, k]))
        stats[f"{prefix}_{label}_p05"] = float(np.quantile(arr[:, k], 0.05))
        stats[f"{prefix}_{label}_p50"] = float(np.quantile(arr[:, k], 0.50))
        stats[f"{prefix}_{label}_p95"] = float(np.quantile(arr[:, k], 0.95))
    return stats


def evaluate_lp_allocation(
    context: LPContext,
    solution: LPSolution,
    args: argparse.Namespace,
) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    demand_mode = eval_mod.resolve_periodic_demand_mode(
        args.periodic_demand_mode,
        bsde=context.bsde,
        gamma_review=context.gamma_review,
    )
    exposure = context.bsde.exposure.detach().cpu().numpy() if hasattr(context.bsde, "exposure") else None
    shared_demands = eval_mod.precompute_periodic_demands(
        context.mu_review,
        context.gamma_review,
        context.review_dt,
        args.num_paths,
        demand_mode=demand_mode,
        seed=args.seed,
        exposure=exposure,
    )
    sim = eval_mod.simulate_open_loop_periodic_review_trajectories(
        solution.allocation,
        context.bsde,
        args.num_paths,
        context.device,
        context.mu_review,
        context.gamma_review,
        demand_mode=demand_mode,
        seed=args.seed,
        return_history=args.write_history_npz,
        precomputed_demands=shared_demands,
        exposure=exposure,
    )

    shortage_cost = np.asarray(sim["shortage_costs"], dtype=np.float64)
    final_local = np.asarray(sim["final_local_inventory"], dtype=np.float64)
    final_warehouse = np.asarray(sim["final_warehouse_inventory"], dtype=np.float64)
    summary: Dict[str, Any] = {
        "num_paths": int(args.num_paths),
        "seed": int(args.seed),
        "periodic_demand_mode": demand_mode,
        "simulated_mean_cost": float(np.mean(shortage_cost)),
        "simulated_standard_error": float(np.std(shortage_cost, ddof=1) / math.sqrt(args.num_paths))
        if args.num_paths > 1
        else 0.0,
        "selected_cost_quantiles": {
            name: float(np.quantile(shortage_cost, quantile)) for name, quantile in QUANTILES
        },
        "total_allocation_by_location": solution.allocation.sum(axis=0).tolist(),
        "final_warehouse_inventory_mean": float(np.mean(final_warehouse)),
        "final_warehouse_inventory_p05": float(np.quantile(final_warehouse, 0.05)),
        "final_warehouse_inventory_p50": float(np.quantile(final_warehouse, 0.50)),
        "final_warehouse_inventory_p95": float(np.quantile(final_warehouse, 0.95)),
    }
    for metric in (
        "total_cha_shortage",
        "total_hf_shortage",
        "total_referral_demand",
        "total_served_chas",
        "total_served_hf",
        "stockout_costs",
        "travel_costs",
        "terminal_costs",
    ):
        summary.update(array_stats(sim[metric], metric))
    summary["final_local_inventory_distribution"] = location_distribution_stats(final_local, "final_local_inventory")
    return summary, sim


def allocation_dataframe(context: LPContext, demand: np.ndarray, solution: LPSolution) -> pd.DataFrame:
    rows = []
    dim = int(context.bsde.dim)
    for n in range(solution.allocation.shape[0]):
        for k in range(dim):
            rows.append(
                {
                    "period": n + 1,
                    "time_start": n * context.review_dt,
                    "time_end": (n + 1) * context.review_dt,
                    "location_index": k,
                    "location_name": location_label(k),
                    "allocation_quantity": solution.allocation[n, k],
                    "allocation_rate": solution.allocation[n, k] / context.review_dt,
                    "period_mean_demand": demand[n, k],
                    "eta": context.eta_review[n, k],
                    "eta_h": context.eta_review[n, k] * context.review_dt,
                    "lp_shortage": solution.shortage[n, k],
                    "lp_end_local_inventory": solution.local_inventory[n, k],
                    "lp_end_warehouse_inventory": solution.warehouse[n],
                }
            )
    return pd.DataFrame(rows)


def flatten_for_csv(payload: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            row[key] = json.dumps(value, default=json_default)
        else:
            row[key] = value
    return row


def build_comparison_row(
    context: LPContext,
    solution: LPSolution,
    lp_summary: Mapping[str, Any],
    simulation_summary: Optional[Mapping[str, Any]],
    with_rate_bound: bool,
    demand_resolution: DemandResolution,
) -> Dict[str, Any]:
    def sim_value(key: str) -> Optional[Any]:
        if simulation_summary is None:
            return None
        return simulation_summary.get(key)

    return {
        "variant_name": ("MD-QCP-eta" if with_rate_bound else "MD-QCP")
        if float(context.bsde.lambda_param) > 0.0
        else ("MD-LP-eta" if with_rate_bound else "MD-LP"),
        "with_rate_bound": bool(with_rate_bound),
        "mu_interpretation": demand_resolution.interpretation,
        "lp_objective": solution.objective,
        "simulated_mean_cost": sim_value("simulated_mean_cost"),
        "simulated_standard_error": sim_value("simulated_standard_error"),
        "total_cha_shortage": sim_value("total_cha_shortage_mean"),
        "total_hf_shortage": sim_value("total_hf_shortage_mean"),
        "total_referral_demand": sim_value("total_referral_demand_mean"),
        "final_warehouse_inventory": sim_value("final_warehouse_inventory_mean"),
        "total_allocation_by_location": json.dumps(lp_summary["total_allocation_by_location"]),
        "output_dir": str(context.output_dir),
    }


def write_outputs(
    context: LPContext,
    demand: np.ndarray,
    demand_resolution: DemandResolution,
    solution: LPSolution,
    lp_summary: Mapping[str, Any],
    lp_diagnostics: Mapping[str, Any],
    simulation_summary: Optional[Mapping[str, Any]],
    simulation_results: Optional[Mapping[str, np.ndarray]],
    comparison_row: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Path]:
    output_dir = context.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "allocation_csv": output_dir / "allocation_plan.csv",
        "allocation_npz": output_dir / "allocation_plan.npz",
        "lp_summary_json": output_dir / "lp_solution_summary.json",
        "lp_diagnostics_json": output_dir / "lp_diagnostics.json",
        "comparison_row_csv": output_dir / "comparison_row.csv",
    }

    allocation_dataframe(context, demand, solution).to_csv(paths["allocation_csv"], index=False)
    np.savez_compressed(
        paths["allocation_npz"],
        allocation_plan=solution.allocation,
        warehouse_inventory=solution.warehouse,
        local_inventory=solution.local_inventory,
        shortage=solution.shortage,
        period_mean_demand=demand,
        mu_review=context.mu_review,
        eta=context.eta,
        eta_review=context.eta_review,
        p=context.p,
        c=context.c,
        r=context.r,
        c_tilde=context.c_tilde,
        q0=np.array([context.q0]),
        z0=context.z0,
        review_dt=np.array([context.review_dt]),
        lambda_param=np.array([float(context.bsde.lambda_param)], dtype=np.float64),
        alpha=np.array([float(context.bsde.alpha)], dtype=np.float64),
        fairness_spec=np.asarray([eqn.FAIRNESS_SPEC]),
        fairness_includes_hf=np.asarray([eqn.FAIRNESS_INCLUDES_HF]),
        fairness_period_h_factor=np.asarray([eqn.FAIRNESS_PERIOD_H_FACTOR]),
    )
    with open(paths["lp_summary_json"], "w", encoding="utf-8") as f:
        json.dump(lp_summary, f, indent=2, default=json_default)
    with open(paths["lp_diagnostics_json"], "w", encoding="utf-8") as f:
        json.dump(lp_diagnostics, f, indent=2, default=json_default)
    if simulation_summary is not None:
        paths["simulation_summary_json"] = output_dir / "simulation_summary.json"
        paths["simulation_results_csv"] = output_dir / "simulation_results.csv"
        with open(paths["simulation_summary_json"], "w", encoding="utf-8") as f:
            json.dump(simulation_summary, f, indent=2, default=json_default)
        pd.DataFrame([flatten_for_csv(simulation_summary)]).to_csv(
            paths["simulation_results_csv"],
            index=False,
        )
    pd.DataFrame([comparison_row]).to_csv(paths["comparison_row_csv"], index=False)

    if args.write_history_npz and simulation_results is not None:
        history_keys = {
            key: value
            for key, value in simulation_results.items()
            if isinstance(value, np.ndarray)
            and key
            in {
                "z_hist",
                "i_hist",
                "a_hist",
                "q_hist",
                "l_hist",
                "d_hist",
                "mu_hist",
                "referral_hist",
                "served_cha_hist",
                "served_hf_hist",
                "hf_total_demand_hist",
            }
        }
        if history_keys:
            paths["simulation_history_npz"] = output_dir / "simulation_history.npz"
            np.savez_compressed(paths["simulation_history_npz"], **history_keys)

    update_parent_comparison(output_dir)
    return paths


def update_parent_comparison(output_dir: Path) -> Optional[Path]:
    parent = output_dir.parent
    rows = []
    for path in sorted(parent.glob("*/comparison_row.csv")):
        try:
            rows.append(pd.read_csv(path))
        except Exception as exc:  # pragma: no cover - defensive diagnostic only
            print(f"Warning: could not read comparison row {path}: {exc}")
    if not rows:
        return None
    combined = pd.concat(rows, ignore_index=True, sort=False)
    if "variant_name" in combined.columns:
        combined = combined.sort_values("variant_name").reset_index(drop=True)
    out_path = parent / "mean_demand_lp_comparison.csv"
    combined.to_csv(out_path, index=False)
    print("Mean-demand LP comparison:")
    print(combined.to_string(index=False))
    return out_path


def print_diagnostics(
    lp_summary: Mapping[str, Any],
    simulation_summary: Optional[Mapping[str, Any]],
    paths: Mapping[str, Path],
) -> None:
    sim_mean = None if simulation_summary is None else simulation_summary["simulated_mean_cost"]
    sim_se = None if simulation_summary is None else simulation_summary["simulated_standard_error"]
    print(
        json.dumps(
            {
                "variant": lp_summary["variant_name"],
                "gurobi_status": lp_summary["gurobi_status_name"],
                "lp_objective": lp_summary["lp_objective"],
                "mu_interpretation": lp_summary["mu_interpretation"],
                "period_length_h": lp_summary["period_length_h"],
                "total_mean_demand": lp_summary["aggregate_total_mean_demand"],
                "simulated_mean_cost": sim_mean,
                "simulated_standard_error": sim_se,
                "solve_only": simulation_summary is None,
                "output_dir": lp_summary["output_dir"],
                "comparison_csv": str(Path(lp_summary["output_dir"]).parent / "mean_demand_lp_comparison.csv"),
                "warnings": lp_summary["warnings"],
            },
            indent=2,
            default=json_default,
        )
    )
    print("Saved files:")
    for path in paths.values():
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    if args.num_paths <= 0:
        raise ValueError("--num-paths must be positive.")

    context = build_lp_context(args)
    demand_resolution = resolve_period_demand(
        context.mu_review,
        context.review_dt,
        args.mu_interpretation,
        context.config_path,
    )
    solution = build_and_solve_compact_lp(
        context,
        demand_resolution.period_demand,
        args.with_rate_bound,
        args,
    )
    solution_arrays = extract_solution(solution)
    validation = validate_lp_solution(
        context,
        demand_resolution.period_demand,
        solution,
        args.with_rate_bound,
    )
    lp_summary = summarize_lp_solution(
        context,
        demand_resolution.period_demand,
        solution,
        args.with_rate_bound,
        validation,
        demand_resolution,
    )
    lp_diagnostics = {
        "demand_diagnostics": demand_resolution.diagnostics,
        "solution_validation": validation,
        "solution_arrays_summary": {
            "allocation_shape": list(solution_arrays["allocation"].shape),
            "warehouse_shape": list(solution_arrays["warehouse"].shape),
            "local_inventory_shape": list(solution_arrays["local_inventory"].shape),
            "shortage_shape": list(solution_arrays["shortage"].shape),
        },
        "warnings": lp_summary["warnings"],
    }
    if args.solve_only:
        simulation_summary = None
        simulation_results = None
    else:
        simulation_summary, simulation_results = evaluate_lp_allocation(context, solution, args)
    comparison_row = build_comparison_row(
        context,
        solution,
        lp_summary,
        simulation_summary,
        args.with_rate_bound,
        demand_resolution,
    )
    paths = write_outputs(
        context,
        demand_resolution.period_demand,
        demand_resolution,
        solution,
        lp_summary,
        lp_diagnostics,
        simulation_summary,
        simulation_results,
        comparison_row,
        args,
    )
    print_diagnostics(lp_summary, simulation_summary, paths)


if __name__ == "__main__":
    main()
