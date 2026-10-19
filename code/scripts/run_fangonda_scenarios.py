#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.fangonda_weekly import scenarios as scenario_registry
from scripts.build_fangonda_scenarios import (
    ACTIVATION_CHOICES,
    ETA_MODES,
    NET_CONFIG_OVERRIDES,
    PROPOSAL_SAMPLER_CHOICES,
    REFERENCE_POLICY_CHOICES,
    SAMPLING_MODE_CHOICES,
    default_output_root,
    materialize_run,
    project_root,
)


PARAM_FILES = (
    "theta.csv",
    "exposure.csv",
    "mu.csv",
    "gamma.csv",
    "p.csv",
    "distance.csv",
    "u.csv",
    "r.csv",
    "c.csv",
    "h.csv",
    "eta.csv",
    "eta_schedule.csv",
)
PERIODIC_DEMAND_MODES = ("auto", "binomial", "gaussian_copula_binomial", "beta_binomial", "negative_binomial", "brownian_motion", "lognormal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate weekly Fangonda comparative-statics runs while reusing a shared baseline."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--baseline", action="store_true", help="Run or materialize the shared baseline only.")
    selection.add_argument("--family", choices=scenario_registry.list_families(), help="Run one non-baseline family.")
    selection.add_argument(
        "--scenario",
        choices=scenario_registry.list_scenarios(include_baseline=True),
        help="Run one named scenario.",
    )
    selection.add_argument("--all", action="store_true", help="Run all non-baseline weekly sensitivity scenarios.")

    parser.add_argument(
        "--output_root",
        default=str(default_output_root()),
        help="Base output directory containing run folders.",
    )
    parser.add_argument("--materialize_only", action="store_true", help="Write run-local configs/CSV files only.")
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Run evaluation_report.py against existing checkpoints only; never start training.",
    )
    parser.add_argument(
        "--force_evaluate",
        action="store_true",
        help="Run evaluation even when evaluation CSVs already exist.",
    )
    parser.add_argument(
        "--skip_baseline_bootstrap",
        action="store_true",
        help="Do not automatically ensure the shared baseline before non-baseline scenarios.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted training from existing run-local checkpoints when available.",
    )
    parser.add_argument("--num_time_interval", type=int, default=None, help="Optional num_time_interval override.")
    parser.add_argument("--num_layers", type=int, default=None, help="Optional num_layers override.")
    parser.add_argument("--num_neurons", type=int, default=None, help="Optional num_neurons override.")
    parser.add_argument("--activation", choices=ACTIVATION_CHOICES, default=None, help="Optional activation override.")
    parser.add_argument(
        "--init_method",
        choices=("xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal"),
        default=None,
        help="Optional network weight initialization override.",
    )
    parser.add_argument("--reference_policy", choices=REFERENCE_POLICY_CHOICES, default=None)
    parser.add_argument(
        "--reference_fluid_plan_path",
        default=None,
        help="Optional allocation_plan.npz used only for reference_policy=fluid_optimal_rollout training rollouts.",
    )
    parser.add_argument("--eta_mode", choices=ETA_MODES, default=None)
    parser.add_argument("--eta_profile", choices=ETA_MODES, default=None)
    parser.add_argument("--chi_eta", type=float, default=None)
    parser.add_argument("--s_eta", type=float, default=None)
    parser.add_argument("--lambda_param", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--proposal_sampler", choices=PROPOSAL_SAMPLER_CHOICES, default=None)
    parser.add_argument("--final_learning_rate", type=float, default=None, help="Optional final_learning_rate override.")
    parser.add_argument("--inner_learning_rate", type=float, default=None, help="Optional inner_learning_rate override.")
    parser.add_argument("--final_milestones", type=str, default=None, help="Comma-separated final_milestones override.")
    parser.add_argument("--inner_milestones", type=str, default=None, help="Comma-separated inner_milestones override.")
    parser.add_argument("--gamma", type=float, default=None, help="Optional scheduler gamma override.")
    parser.add_argument("--grad_norm", type=float, default=None, help="Optional gradient clipping norm override.")
    parser.add_argument("--final_iterations", type=int, default=None, help="Optional final_iterations override.")
    parser.add_argument("--inner_iterations", type=int, default=None, help="Optional inner_iterations override.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch_size override.")
    parser.add_argument("--valid_size", type=int, default=None, help="Optional valid_size override.")
    parser.add_argument("--sampling_mode", choices=SAMPLING_MODE_CHOICES, default=None)
    parser.add_argument("--rollout_pool_size", type=int, default=None)
    parser.add_argument("--sampling_seed", type=int, default=0)
    parser.add_argument("--uniform_z_radius_mult", type=float, default=None)
    parser.add_argument("--uniform_q_radius_frac", type=float, default=None)
    parser.add_argument("--total_inventory_buffer", type=float, default=None)
    parser.add_argument("--mix_rollout", type=float, default=None)
    parser.add_argument("--mix_box", type=float, default=None)
    parser.add_argument("--mix_shortage", type=float, default=None)
    parser.add_argument("--shortage_q_frac", type=float, default=None)
    parser.add_argument("--shortage_z_frac", type=float, default=None)
    parser.add_argument("--lognormal_delta_z", type=float, default=None)
    parser.add_argument("--lognormal_delta_q", type=float, default=None)
    parser.add_argument("--lognormal_r_min", type=int, default=None)
    parser.add_argument("--initial_state_mode", choices=("standard", "q_sensitivity_mixture", "paper_70_30"), default=None)
    parser.add_argument("--initial_perturb_prob", type=float, default=None)
    parser.add_argument("--initial_q_perturb_frac", type=float, default=None)
    parser.add_argument("--initial_z_perturb_frac", type=float, default=None)
    parser.add_argument("--initial_perturb_minus_prob", type=float, default=None)
    parser.add_argument("--initial_perturb_plus_prob", type=float, default=None)
    parser.add_argument("--initial_perturb_minus_frac", type=float, default=None)
    parser.add_argument("--initial_perturb_plus_frac", type=float, default=None)
    parser.add_argument("--initial_dirichlet_alpha", type=float, default=None)
    parser.add_argument("--consistency_coef", type=float, default=None)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--eval_num_paths", type=int, default=10000, help="Fine-grid and periodic-review Monte Carlo paths.")
    parser.add_argument("--eval_ext_num_paths", type=int, default=10000, help="Extended diagnostics Monte Carlo paths.")
    parser.add_argument(
        "--train_final_interval_only",
        action="store_true",
        help="Train only the terminal interval N-1 and skip evaluation.",
    )
    parser.add_argument(
        "--periodic_demand_mode",
        choices=PERIODIC_DEMAND_MODES,
        default="auto",
        help="Demand model passed through to evaluation_report.py for periodic-review evaluation.",
    )
    parser.add_argument(
        "--fluid_plan_path",
        default=None,
        help="Optional allocation_plan.npz passed through to evaluation_report.py for Fluid-Optimal.",
    )
    parser.add_argument(
        "--skip_extended_diagnostics",
        action="store_true",
        help="Skip the history-based diagnostics section inside evaluation_report.py.",
    )
    return parser.parse_args()


def run_dir(output_root: Path, run_name: str) -> Path:
    return output_root / run_name


def is_materialized(run_path: Path) -> bool:
    params_dir = run_path / "params"
    if not (run_path / "config_scenario.json").is_file():
        return False
    if not (run_path / "scenario_manifest.json").is_file():
        return False
    return all((params_dir / filename).is_file() for filename in PARAM_FILES)


def get_num_time_interval(run_path: Path) -> int | None:
    for config_name in ("config_used.json", "config_scenario.json"):
        config_path = run_path / config_name
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as f:
                config = json.load(f)
            return int(config["eqn_config"]["num_time_interval"])
    return None


def completed_checkpoint_count(run_path: Path) -> int:
    logs_dir = run_path / "logs"
    num_time_interval = get_num_time_interval(run_path)
    if num_time_interval is None or not logs_dir.is_dir():
        return 0
    return sum(
        1
        for n in range(num_time_interval)
        if (logs_dir / f"y_network{n}.pth").is_file()
        and (logs_dir / f"z_network{n}.pth").is_file()
    )


def is_trained(run_path: Path) -> bool:
    num_time_interval = get_num_time_interval(run_path)
    if num_time_interval is None:
        return False
    return completed_checkpoint_count(run_path) == num_time_interval


def is_partially_trained(run_path: Path) -> bool:
    num_time_interval = get_num_time_interval(run_path)
    if num_time_interval is None:
        return False
    completed = completed_checkpoint_count(run_path)
    return 0 < completed < num_time_interval


def can_resume_training(run_path: Path) -> bool:
    return (run_path / "config_used.json").is_file() and completed_checkpoint_count(run_path) > 0


def is_evaluated(run_path: Path) -> bool:
    return (run_path / "evaluation_results.csv").is_file() and (
        run_path / "periodic_review_evaluation_results.csv"
    ).is_file()


def build_namespace(args: argparse.Namespace, output_root: Path) -> SimpleNamespace:
    namespace = SimpleNamespace(
        output_root=str(output_root),
        num_time_interval=args.num_time_interval,
        num_layers=args.num_layers,
        eta_mode=args.eta_mode,
        eta_profile=args.eta_profile,
        chi_eta=args.chi_eta,
        s_eta=args.s_eta,
        lambda_param=args.lambda_param,
        alpha=args.alpha,
        reference_policy=args.reference_policy,
        reference_fluid_plan_path=args.reference_fluid_plan_path,
        final_iterations=args.final_iterations,
        inner_iterations=args.inner_iterations,
        batch_size=args.batch_size,
        valid_size=args.valid_size,
        eval_seed=args.eval_seed,
        eval_num_paths=args.eval_num_paths,
        eval_ext_num_paths=args.eval_ext_num_paths,
    )
    for key, _ in NET_CONFIG_OVERRIDES:
        setattr(namespace, key, getattr(args, key, None))
    return namespace


def has_net_config_overrides(args: argparse.Namespace) -> bool:
    return (
        args.reference_policy is not None
        or args.reference_fluid_plan_path is not None
        or args.eta_mode is not None
        or args.eta_profile is not None
        or args.chi_eta is not None
        or args.s_eta is not None
        or args.lambda_param is not None
        or args.alpha is not None
        or any(getattr(args, key, None) is not None for key, _ in NET_CONFIG_OVERRIDES)
    )


def read_sampler_config(run_path: Path) -> Dict[str, object]:
    config_path = run_path / "config_scenario.json"
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    net_config = config.get("net_config", {})
    keys = ("num_layers",) + tuple(key for key, _ in NET_CONFIG_OVERRIDES)
    sampler_config = {key: net_config.get(key) for key in keys if key in net_config}
    sampler_config["reference_policy"] = config.get("eqn_config", {}).get("policy")
    sampler_config["reference_fluid_plan_path"] = config.get("eqn_config", {}).get("reference_fluid_plan_path")
    sampler_config["eta_mode"] = config.get("eqn_config", {}).get("eta_mode", "constant_peak")
    sampler_config["eta_profile"] = config.get("eqn_config", {}).get("eta_profile", sampler_config["eta_mode"])
    sampler_config["chi_eta"] = config.get("eqn_config", {}).get("chi_eta", 1.75)
    sampler_config["s_eta"] = config.get("eqn_config", {}).get("s_eta", 0.0)
    sampler_config["lambda_param"] = config.get("eqn_config", {}).get("lambda_param", 0.0)
    sampler_config["alpha"] = config.get("eqn_config", {}).get("alpha", 0.5)
    return sampler_config


def explicit_sampler_overrides(args: argparse.Namespace) -> Dict[str, object]:
    requested: Dict[str, object] = {}
    if args.reference_policy is not None:
        requested["reference_policy"] = args.reference_policy
    if args.reference_fluid_plan_path is not None:
        requested["reference_fluid_plan_path"] = str(args.reference_fluid_plan_path)
    if args.eta_mode is not None:
        requested["eta_mode"] = args.eta_mode
        requested["eta_profile"] = args.eta_mode
    if args.eta_profile is not None:
        requested["eta_mode"] = args.eta_profile
        requested["eta_profile"] = args.eta_profile
    if args.chi_eta is not None:
        requested["chi_eta"] = float(args.chi_eta)
    if args.s_eta is not None:
        requested["s_eta"] = float(args.s_eta)
    if args.lambda_param is not None:
        requested["lambda_param"] = float(args.lambda_param)
    if args.alpha is not None:
        requested["alpha"] = float(args.alpha)
    if args.num_layers is not None:
        requested["num_layers"] = int(args.num_layers)
    for key, cast in NET_CONFIG_OVERRIDES:
        value = getattr(args, key, None)
        if value is not None:
            requested[key] = cast(value)
    return requested


def assert_materialized_sampler_matches(run_path: Path, args: argparse.Namespace) -> None:
    requested = explicit_sampler_overrides(args)
    if not requested:
        return
    existing = read_sampler_config(run_path)
    mismatches = {}
    for key, expected in requested.items():
        actual = existing.get(key)
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(
            "Existing materialized config has sampler/reference settings that differ from "
            f"explicit overrides: {mismatches}. Use a fresh --output_root or remove {run_path}."
        )


def read_scenario_manifest(run_path: Path) -> Dict[str, object]:
    manifest_path = run_path / "scenario_manifest.json"
    if not manifest_path.is_file():
        return {}
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def assert_materialized_scenario_matches(run_path: Path, spec: Dict[str, object]) -> None:
    manifest = read_scenario_manifest(run_path)
    existing_spec = manifest.get("resolved_spec")
    if existing_spec != spec:
        existing_version = None
        if isinstance(existing_spec, dict):
            existing_version = existing_spec.get("calibration_version")
        expected_version = spec.get("calibration_version")
        raise RuntimeError(
            "Existing materialized scenario inputs do not match the requested scenario spec. "
            f"Existing calibration_version={existing_version!r}, "
            f"expected calibration_version={expected_version!r}. "
            f"Use a fresh --output_root or remove {run_path}."
        )

    metadata = manifest.get("metadata", {})
    metadata_version = metadata.get("calibration_version") if isinstance(metadata, dict) else None
    expected_version = spec.get("calibration_version")
    if metadata_version != expected_version:
        raise RuntimeError(
            "Existing materialized scenario metadata does not match the requested calibration. "
            f"Existing metadata calibration_version={metadata_version!r}, "
            f"expected calibration_version={expected_version!r}. "
            f"Use a fresh --output_root or remove {run_path}."
        )


def effective_spec_for_args(spec: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    effective = dict(spec)
    if getattr(args, "lambda_param", None) is not None:
        effective["lambda_param"] = float(args.lambda_param)
    if getattr(args, "alpha", None) is not None:
        effective["alpha"] = float(args.alpha)
    return effective


def ensure_materialized(spec: Dict[str, object], args: argparse.Namespace, output_root: Path) -> Path:
    spec = effective_spec_for_args(spec, args)
    run_name = str(spec["name"])
    path = run_dir(output_root, run_name)
    if not is_materialized(path):
        result = materialize_run(run_name, spec, build_namespace(args, output_root))
        path = Path(result["run_dir"])
        print(f"Materialized scenario inputs under {path}")
    else:
        print(f"Reusing materialized inputs under {path}")
        print(f"Existing sampler config: {read_sampler_config(path)}")
        assert_materialized_scenario_matches(path, spec)
        assert_materialized_sampler_matches(path, args)
    return path


def run_command(cmd: List[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def train_run(run_name: str, output_root: Path, cwd: Path, args: argparse.Namespace, resume: bool = False) -> None:
    config_path = run_dir(output_root, run_name) / "config_scenario.json"
    cmd = [
        sys.executable,
        "-u",
        "main.py",
        f"--config_path={config_path}",
        f"--run_name={run_name}",
        f"--output_dir={output_root}",
    ]
    if args.num_time_interval is not None:
        cmd.extend(["--num_time_interval", str(args.num_time_interval)])
    if args.num_layers is not None:
        cmd.extend(["--num_layers", str(args.num_layers)])
    if args.reference_policy is not None:
        cmd.extend(["--reference_policy", str(args.reference_policy)])
    if args.reference_fluid_plan_path is not None:
        cmd.extend(["--reference_fluid_plan_path", str(args.reference_fluid_plan_path)])
    if args.eta_mode is not None:
        cmd.extend(["--eta_mode", str(args.eta_mode)])
    if args.eta_profile is not None:
        cmd.extend(["--eta_profile", str(args.eta_profile)])
    if args.chi_eta is not None:
        cmd.extend(["--chi_eta", str(args.chi_eta)])
    if args.s_eta is not None:
        cmd.extend(["--s_eta", str(args.s_eta)])
    if args.lambda_param is not None:
        cmd.extend(["--lambda_param", str(args.lambda_param)])
    if args.alpha is not None:
        cmd.extend(["--alpha", str(args.alpha)])
    for key, _ in NET_CONFIG_OVERRIDES:
        value = getattr(args, key, None)
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    if args.eval_seed is not None:
        cmd.extend(["--eval_seed", str(args.eval_seed)])
    cmd.extend(["--eval_num_paths", str(args.eval_num_paths)])
    cmd.extend(["--eval_ext_num_paths", str(args.eval_ext_num_paths)])
    if args.train_final_interval_only:
        cmd.append("--train_final_interval_only")
    if resume:
        cmd.append("--resume")
    run_command(cmd, cwd)


def evaluate_run(run_name: str, output_root: Path, cwd: Path, args: argparse.Namespace) -> None:
    if args.train_final_interval_only:
        print(f"Skipping evaluation for final-interval-only run: {run_name}")
        return
    cmd = [
        sys.executable,
        "-u",
        "evaluation_report.py",
        "-r",
        run_name,
        "--output_root",
        str(output_root),
        "--num_paths",
        str(args.eval_num_paths),
        "--ext_num_paths",
        str(args.eval_ext_num_paths),
        "--periodic_demand_mode",
        args.periodic_demand_mode,
    ]
    if args.eval_seed is not None:
        cmd.extend(["--fine_grid_seed", str(args.eval_seed)])
        cmd.extend(["--periodic_seed", str(args.eval_seed)])
    if args.skip_extended_diagnostics:
        cmd.append("--skip_extended_diagnostics")
    if args.fluid_plan_path:
        cmd.extend(["--fluid_plan_path", str(args.fluid_plan_path)])
    run_command(cmd, cwd)


def selected_specs(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.baseline:
        return [scenario_registry.resolve_scenario_spec(scenario_registry.BASELINE_NAME)]
    if args.family:
        return list(scenario_registry.iter_scenario_specs(args.family))
    if args.scenario:
        return [scenario_registry.resolve_scenario_spec(args.scenario)]
    if args.all:
        specs: List[Dict[str, object]] = []
        for family_name in scenario_registry.list_families():
            specs.extend(list(scenario_registry.iter_scenario_specs(family_name)))
        return specs
    raise ValueError("One selection mode is required.")


def maybe_ensure_baseline(args: argparse.Namespace, output_root: Path, cwd: Path) -> None:
    if args.baseline or args.skip_baseline_bootstrap or args.evaluate_only:
        return
    baseline_spec = scenario_registry.resolve_scenario_spec(scenario_registry.BASELINE_NAME)
    baseline_run_name = str(baseline_spec["name"])
    baseline_run_dir = ensure_materialized(baseline_spec, args, output_root)
    if is_trained(baseline_run_dir):
        if is_evaluated(baseline_run_dir) and not args.force_evaluate:
            print(f"Baseline already complete: {baseline_run_name}")
            return
        print(f"Baseline already trained; running evaluation only: {baseline_run_name}")
        if not args.materialize_only:
            evaluate_run(baseline_run_name, output_root, cwd, args)
        return
    if args.resume and is_partially_trained(baseline_run_dir):
        completed = completed_checkpoint_count(baseline_run_dir)
        expected = get_num_time_interval(baseline_run_dir)
        print(f"Resuming shared baseline: {baseline_run_name} ({completed}/{expected} intervals complete)")
        train_run(
            baseline_run_name,
            output_root,
            cwd,
            args,
            resume=can_resume_training(baseline_run_dir),
        )
        evaluate_run(baseline_run_name, output_root, cwd, args)
        return
    if args.materialize_only:
        return
    print(f"Training shared baseline: {baseline_run_name}")
    train_run(baseline_run_name, output_root, cwd, args)
    evaluate_run(baseline_run_name, output_root, cwd, args)


def main() -> None:
    args = parse_args()
    if args.materialize_only and args.evaluate_only:
        raise ValueError("--materialize_only and --evaluate_only cannot be used together.")
    cwd = project_root()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    maybe_ensure_baseline(args, output_root, cwd)

    for spec in selected_specs(args):
        run_name = str(spec["name"])
        path = ensure_materialized(spec, args, output_root)
        if args.materialize_only:
            continue
        if args.evaluate_only:
            if not is_trained(path):
                completed = completed_checkpoint_count(path)
                expected = get_num_time_interval(path)
                raise RuntimeError(
                    f"Evaluate-only requested for {run_name}, but checkpoints are incomplete "
                    f"({completed}/{expected})."
                )
            print(f"Evaluate-only requested; running evaluation without training: {run_name}")
            evaluate_run(run_name, output_root, cwd, args)
            continue
        if is_trained(path):
            if is_evaluated(path) and not args.force_evaluate:
                print(f"Skipping completed run: {run_name}")
                continue
            print(f"Run already trained; running evaluation only: {run_name}")
            evaluate_run(run_name, output_root, cwd, args)
            continue
        if args.resume and is_partially_trained(path):
            completed = completed_checkpoint_count(path)
            expected = get_num_time_interval(path)
            print(f"Resuming training for {run_name} ({completed}/{expected} intervals complete)")
            train_run(run_name, output_root, cwd, args, resume=can_resume_training(path))
            evaluate_run(run_name, output_root, cwd, args)
            continue
        print(f"Starting training for {run_name}")
        train_run(run_name, output_root, cwd, args)
        evaluate_run(run_name, output_root, cwd, args)


if __name__ == "__main__":
    main()
