#!/usr/bin/env python
# coding: utf-8

# # Evaluation Report Draft
# 
# This notebook generates training-loss figures and benchmark comparison tables for a selected run under `output/<RUN_NAME>`.
# 
# Sections:
# - Setup
# - Figure 1: training curves for intervals 0, T/2, T
# - Figure 2: final train/valid loss across T intervals
# - Benchmark evaluation and CSV output
# - LaTeX table export
# - Discrete-time evaluation and CSV output
# - LaTeX table export for continuous and discrete-time evaluation

# In[1]:


import argparse
import os
import json
import matplotlib
import pandas as pd
import importlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm

import evaluate as eval_mod
import evaluation_bank as evaluation_bank_mod
import extended_evaluation_diagnostics
import reporting_contracts
import state_support_alignment
from artifact_guards import validate_run_metadata
from fairness_diagnostics import (
    allocation_concentration,
    demand_adjusted_fairness_diagnostics,
)
from scripts.m110_presentation_diagnostics import build_periodic_location_diagnostic_rows

try:
    import munch
except ImportError:  # pragma: no cover - fallback for lightweight environments
    munch = None

import sys
# sys.argv = [
#     "evaluation_report.py",
#     "--run_name", "stylized_2cha_case_c_correlation_87396",
#     "--num_paths", "10000",
#     "--ext_num_paths", "10000",
#     "--review_num_intervals", "10",
#     "--periodic_demand_mode", "negative_binomial",
# ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate evaluation-report figures and tables for a run under <output_root>/<RUN_NAME>."
    )
    parser.add_argument("-r", "--run_name", help="Run directory name under output/")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Base output directory containing run folders. Defaults to <project>/output.",
    )
    parser.add_argument(
        "--source_run_dir",
        type=str,
        default=None,
        help=(
            "Optional absolute directory containing the frozen config and checkpoints. "
            "Evaluation outputs still go to <output_root>/<run_name>, so the source run "
            "is never modified."
        ),
    )
    parser.add_argument("--num_paths", type=int, default=10000, help="Monte Carlo paths for fine-grid and discrete-review tables")
    parser.add_argument("--ext_num_paths", type=int, default=10000, help="Monte Carlo paths for history-based diagnostics")
    parser.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="Optional suffix for discrete-review outputs, e.g. monthly, weekly, or daily.",
    )
    parser.add_argument(
        "--eval_scope",
        choices=["all", "fine", "periodic", "discrete", "extended"],
        default="all",
        help="Which evaluation sections to run.",
    )
    parser.add_argument(
        "--skip_extended_diagnostics",
        action="store_true",
        help="Skip the history-based diagnostics section after the main fine-grid and discrete-review tables.",
    )
    parser.add_argument(
        "--write_extended_evaluation_artifacts",
        action="store_true",
        help=(
            "Opt in to versioned CHA mismatch, compact location, and selected-policy "
            "z_hist/l_hist artifacts after all legacy periodic outputs are written."
        ),
    )
    parser.add_argument("--fine_grid_seed", type=int, default=0, help="Seed for fine-grid common random numbers")
    parser.add_argument("--periodic_seed", type=int, default=0, help="Seed for discrete-review common random numbers")
    parser.add_argument(
        "--evaluation_bank_manifest",
        type=str,
        default=None,
        help="Hash-pinned manifest containing realized fine-grid and periodic-review CRN arrays.",
    )
    parser.add_argument(
        "--evaluation_bank_sha256",
        type=str,
        default=None,
        help="Required SHA-256 for --evaluation_bank_manifest.",
    )
    parser.add_argument(
        "--evaluation_scenario",
        type=str,
        default=None,
        help="Exact scientific scenario identity used to validate a frozen evaluation bank.",
    )
    parser.add_argument("--review_num_intervals", type=int, default=None, help="Optional discrete-review interval count")
    parser.add_argument("--review_dt", type=float, default=None, help="Optional discrete-review interval length")
    parser.add_argument(
        "--fluid_plan_path",
        type=str,
        default=None,
        help="Optional allocation_plan.npz from the deterministic fluid LP/QCP solver.",
    )
    parser.add_argument(
        "--skip_fluid_benchmark",
        action="store_true",
        help="Omit the Fluid-Optimal benchmark when no compatible Gurobi artifact is available.",
    )
    parser.add_argument(
        "--periodic_demand_mode",
        choices=eval_mod.PERIODIC_DEMAND_MODES,
        default="auto",
        help="Demand model for discrete-review evaluation",
    )
    parser.add_argument(
        "--policy_extraction_modes",
        default=None,
        help="Comma-separated DS extraction modes for discrete-review evaluation, e.g. E0,E2,E3.",
    )
    parser.add_argument(
        "--primary_deep_splitting_policy",
        default=None,
        help=(
            "Exact policy label used for DS diagnostics and paired comparisons. "
            "The policy must be present in --policy_extraction_modes."
        ),
    )
    parser.add_argument(
        "--allow_missing_checkpoints",
        action="store_true",
        help="Allow evaluation to continue if run-local z_network checkpoints are incomplete.",
    )
    parser.add_argument("--setting_id", default=None, help="Frozen campaign setting identity for report provenance")
    parser.add_argument("--evaluation_run_id", default=None, help="Immutable evaluation-only run identity")
    parser.add_argument("--training_run_id", default=None, help="Immutable source training run identity")
    parser.add_argument("--stream_id", type=int, default=None, help="Frozen training stream identity")
    parser.add_argument("--training_branch", default=None, help="Recorded source training branch")
    parser.add_argument("--training_commit", default=None, help="Recorded source training commit")
    parser.add_argument(
        "--config_hash",
        default=None,
        help="Expected SHA-256 of the resolved source configuration object",
    )
    parser.add_argument(
        "--config_file_sha256",
        default=None,
        help="Expected file SHA-256 of config_used.json (distinct from the resolved config object hash)",
    )
    parser.add_argument(
        "--checkpoint_certification_sha256",
        default=None,
        help="SHA-256 of the frozen checkpoint/application certification",
    )
    parser.add_argument("--evaluation_branch", default=None, help="Evaluation repository branch")
    parser.add_argument("--evaluation_commit", default=None, help="Evaluation repository commit")
    parser.set_defaults(use_common_random_numbers=True)
    parser.add_argument(
        "--use_common_random_numbers",
        dest="use_common_random_numbers",
        action="store_true",
        help="Reuse identical exogenous randomness across policies during evaluation",
    )
    parser.add_argument(
        "--no_common_random_numbers",
        dest="use_common_random_numbers",
        action="store_false",
        help="Disable common random numbers and preserve policy-specific randomness",
    )
    return parser.parse_args()


ARGS = parse_args()
RUN_NAME = ARGS.run_name
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = ARGS.output_root or os.path.join(PROJECT_ROOT, "output")
BASE = os.path.join(OUTPUT_ROOT, RUN_NAME)
SOURCE_BASE = os.path.abspath(os.path.expanduser(ARGS.source_run_dir)) if ARGS.source_run_dir else BASE
OUTPUT_TAG = ARGS.output_tag.strip() if ARGS.output_tag else ""
WRITE_BASELINE_TABLES = RUN_NAME == "fangonda_base" and OUTPUT_TAG in ("", "weekly")
RUN_FINE_TABLE = ARGS.eval_scope in ("all", "fine")
RUN_PERIODIC_TABLE = ARGS.eval_scope in ("all", "periodic", "discrete")
RUN_TRAINING_DIAGNOSTICS = ARGS.eval_scope in ("all", "fine")
RUN_FINE_HISTORY = (not ARGS.skip_extended_diagnostics) and ARGS.eval_scope in ("all", "extended")
RUN_PERIODIC_HISTORY = (not ARGS.skip_extended_diagnostics) and ARGS.eval_scope in ("all", "periodic", "discrete", "extended")

SOURCE_RUN_NAME = os.path.basename(os.path.normpath(SOURCE_BASE))
TRAIN_CSV = os.path.join(SOURCE_BASE, f"{SOURCE_RUN_NAME}_training_history.csv")
TRAIN_SUMMARY_CSV = os.path.join(SOURCE_BASE, f"{SOURCE_RUN_NAME}_training_interval_summary.csv")
CONFIG = os.path.join(SOURCE_BASE, "config_used.json")
LOGS_PATH = os.path.join(SOURCE_BASE, "logs")
FIG_DIR = os.path.join(BASE, "figure")
PAPER_TABLE_DIR = os.path.join(PROJECT_ROOT, "tables", "fangonda")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PAPER_TABLE_DIR, exist_ok=True)

print("RUN_NAME:", RUN_NAME)
print("BASE:", BASE)
print("SOURCE_BASE:", SOURCE_BASE)
print("TRAIN_CSV:", TRAIN_CSV)
print("TRAIN_SUMMARY_CSV:", TRAIN_SUMMARY_CSV)
print("CONFIG:", CONFIG)
print("LOGS_PATH:", LOGS_PATH)
print("FIG_DIR:", FIG_DIR)
print("PAPER_TABLE_DIR:", PAPER_TABLE_DIR)
print("WRITE_BASELINE_TABLES:", WRITE_BASELINE_TABLES)

eval_mod = importlib.reload(eval_mod)


def require_complete_z_checkpoints(config_path, logs_path):
    with open(config_path, encoding="utf-8") as handle:
        raw_config = json.load(handle)
    expected = int(raw_config.get("eqn_config", {}).get("num_time_interval", 0))
    if expected <= 0:
        raise ValueError(f"Cannot determine num_time_interval from {config_path}")
    missing = [
        n for n in range(expected)
        if not os.path.isfile(os.path.join(logs_path, f"z_network{n}.pth"))
    ]
    if missing:
        preview = ",".join(str(n) for n in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)}/{expected} z_network checkpoints in {logs_path}: "
            f"{preview}{suffix}. Refusing to evaluate DS policy with random/unloaded networks. "
            "Pass --allow_missing_checkpoints only for explicit debugging."
        )


def tagged_output_name(filename, tag=OUTPUT_TAG):
    if not tag:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{tag}{ext}"


def tagged_csv_name(filename, tag=OUTPUT_TAG):
    return tagged_output_name(filename, tag)


def latex_escape(text):
    return str(text).replace("_", r"\_")


def policy_label(policy):
    if str(policy) in reporting_contracts.POLICY_DISPLAY_NAMES:
        return reporting_contracts.policy_display_name(str(policy))
    labels = {
        "Deep_Splitting": "Deep Splitting",
        "E0_frozen_average": "E0 frozen average",
        "E2_q_rollout": "E2 q rollout",
    }
    return labels.get(str(policy), str(policy).replace("_", " "))


DEEP_SPLITTING_POLICY_CANDIDATES = (
    "Deep_Splitting",
    "E0_frozen_average",
    "E2_q_rollout",
    "E3_mean_rollout",
)


def select_deep_splitting_policy(mapping, *, preferred=None, required=True):
    if preferred is not None:
        if preferred in mapping:
            return preferred
        available = ", ".join(str(name) for name in mapping)
        raise KeyError(f"Requested Deep Splitting policy {preferred!r} is absent. Available policies: {available}")
    for name in DEEP_SPLITTING_POLICY_CANDIDATES:
        if name in mapping:
            return name
    for name, policy_fn in mapping.items():
        if getattr(policy_fn, "extraction_mode", None) is not None:
            return name
    if required:
        available = ", ".join(str(name) for name in mapping)
        raise KeyError(f"No Deep Splitting policy found. Available policies: {available}")
    return None


def mean_se(m, se):
    return f"{m:.3f} ({se:.3f})"


def artifact_token(value):
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    return token.strip("_") or "policy"


def stack_pathwise_metric(costs_by_policy, policy_names, primary_key, *aliases):
    arrays = []
    for policy in policy_names:
        costs = costs_by_policy[policy]
        metric = None
        for key in (primary_key, *aliases):
            if key in costs:
                metric = costs[key]
                break
        if metric is None:
            raise KeyError(f"Missing pathwise metric {primary_key!r} for policy {policy!r}")
        arrays.append(np.asarray(metric, dtype=np.float64))
    return np.stack(arrays, axis=0)


def write_pathwise_cost_artifacts(costs_by_policy, base_dir, stem, *, deep_splitting_policy=None):
    if not costs_by_policy:
        return None
    policy_names = list(costs_by_policy)
    arrays = {
        "policies": np.asarray(policy_names),
        "policy_names": np.asarray(policy_names),
        "total_objectives": stack_pathwise_metric(costs_by_policy, policy_names, "total_objectives", "total_costs"),
        "untreated_costs": stack_pathwise_metric(costs_by_policy, policy_names, "untreated_costs", "stockout_costs"),
        "referral_travel_costs": stack_pathwise_metric(costs_by_policy, policy_names, "referral_travel_costs", "travel_costs"),
        "shortage_costs": stack_pathwise_metric(costs_by_policy, policy_names, "shortage_costs"),
        "fairness_utilities": stack_pathwise_metric(costs_by_policy, policy_names, "fairness_utilities"),
        "weighted_fairness_utilities": stack_pathwise_metric(
            costs_by_policy,
            policy_names,
            "weighted_fairness_utilities",
            "fairness_objective_contributions",
            "fairness_costs",
        ),
        "terminal_costs": stack_pathwise_metric(costs_by_policy, policy_names, "terminal_costs"),
    }
    arrays["operational_costs"] = arrays["untreated_costs"] + arrays["referral_travel_costs"]

    ds_policy = deep_splitting_policy if deep_splitting_policy in costs_by_policy else select_deep_splitting_policy(costs_by_policy, required=False)
    if ds_policy is not None:
        ds_idx = policy_names.index(ds_policy)
        ds_total = arrays["total_objectives"][ds_idx]
        for bench_idx, benchmark in enumerate(policy_names):
            if benchmark == ds_policy:
                continue
            arrays[f"paired_gap_{artifact_token(ds_policy)}_minus_{artifact_token(benchmark)}"] = (
                ds_total - arrays["total_objectives"][bench_idx]
            )

    npz_path = os.path.join(base_dir, tagged_output_name(f"{stem}_pathwise_costs.npz"))
    np.savez_compressed(npz_path, **arrays)

    rows = []
    for policy_idx, policy in enumerate(policy_names):
        num_paths = arrays["total_objectives"].shape[1]
        for path_idx in range(num_paths):
            rows.append(
                {
                    "policy": policy,
                    "policy_index": policy_idx,
                    "path": path_idx,
                    "total_objective": float(arrays["total_objectives"][policy_idx, path_idx]),
                    "operational_cost": float(arrays["operational_costs"][policy_idx, path_idx]),
                    "untreated_cost": float(arrays["untreated_costs"][policy_idx, path_idx]),
                    "referral_travel_cost": float(arrays["referral_travel_costs"][policy_idx, path_idx]),
                    "shortage_cost": float(arrays["shortage_costs"][policy_idx, path_idx]),
                    "fairness_utility": float(arrays["fairness_utilities"][policy_idx, path_idx]),
                    "weighted_fairness_utility": float(arrays["weighted_fairness_utilities"][policy_idx, path_idx]),
                    "terminal_cost": float(arrays["terminal_costs"][policy_idx, path_idx]),
                }
            )
    csv_path = os.path.join(base_dir, tagged_csv_name(f"{stem}_pathwise_costs.csv"))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Saved pathwise costs:", npz_path)
    print("Saved pathwise costs CSV:", csv_path)
    return npz_path, csv_path


def write_objective_decomposition_table(df, path):
    columns = [
        ("Policy", "Policy"),
        ("Total objective", "Total objective"),
        ("Untreated cost", "Untreated cost"),
        ("Referral travel cost", "Referral travel cost"),
        ("Fairness utility", "Fairness utility"),
        ("Gap vs proposed policy", "Gap vs proposed policy"),
    ]
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Policy": policy_label(row["Policy"]),
                "Total objective": mean_se(row["Total_Objective_Mean"], row["Total_Objective_SE"]),
                "Untreated cost": mean_se(
                    row.get("Untreated_Cost_Mean", row.get("Stockout_Cost_Mean")),
                    row.get("Untreated_Cost_SE", row.get("Stockout_Cost_SE")),
                ),
                "Referral travel cost": mean_se(
                    row.get("Referral_Travel_Cost_Mean", row.get("Travel_Cost_Mean")),
                    row.get("Referral_Travel_Cost_SE", row.get("Travel_Cost_SE")),
                ),
                "Fairness utility": mean_se(row["Fairness_Utility_Mean"], row["Fairness_Utility_SE"]),
                "Gap vs proposed policy": f"{row['Performance_Gap_Percent']:.1f}\\%",
            }
        )
    write_latex_tabular(rows, columns, path, column_spec="@{}lrrrrr@{}")


def write_operational_metrics_table(df, path):
    columns = [
        ("Policy", "Policy"),
        ("Coverage", "Overall Coverage"),
        ("LocalFill", "CHA Local Fill"),
        ("TravelBurden", "Referral Travel Burden"),
        ("HFShare", "HF shipment share"),
        ("WorstLocalFill", "Worst-CHA Local Fill"),
    ]
    rows = []
    for policy, row in df.iterrows():
        rows.append(
            {
                "Policy": policy_label(policy),
                "Coverage": f"{row['Coverage']:.3f}",
                "LocalFill": f"{row['LocalFill']:.3f}",
                "TravelBurden": f"{row['TravelBurden']:.3f}",
                "HFShare": f"{row['HFShare']:.3f}",
                "WorstLocalFill": f"{row.get('WorstLocalFill', row.get('Worst_Coverage')):.3f}",
            }
        )
    write_latex_tabular(rows, columns, path, column_spec="@{}lrrrrr@{}")


def write_latex_tabular(rows, columns, path, column_spec):
    lines = [f"\\begin{{tabular}}{{{column_spec}}}", "\\toprule"]
    lines.append(" & ".join(header for _, header in columns) + r" \\")
    lines.append("\\midrule")
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(latex_escape(value) if key == "Policy" else str(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("Saved:", path)


# ## Parameter Check: $\mu$, $\Gamma$, and inventory-demand ratio
# 
# This section plots the input parameter sequences for HF and CHA 1, and computes the ratio of initial inventory to expected total demand over the horizon.

# In[2]:
# Parameter check: HF / CHA 1 mu and Gamma, plus I0 / total demand

import numpy as np

with open(CONFIG) as f:
    config_preview_raw = json.load(f)
validate_run_metadata(config_preview_raw, run_dir=SOURCE_BASE)
config_preview = eval_mod.to_config_namespace(config_preview_raw)

params_dir = os.path.join(SOURCE_BASE, "params")
if os.path.isdir(params_dir):
    mu_path = os.path.join(params_dir, config_preview.eqn_config.mu_file)
    gamma_path = os.path.join(params_dir, config_preview.eqn_config.gamma_file)
else:
    cfg_dir = config_preview.eqn_config.config_dir
    mu_path = os.path.join(cfg_dir, config_preview.eqn_config.mu_file)
    gamma_path = os.path.join(cfg_dir, config_preview.eqn_config.gamma_file)

mu_raw = pd.read_csv(mu_path, header=None)
gamma_raw = pd.read_csv(gamma_path, header=None)

T = float(config_preview.eqn_config.total_time)
I0 = float(config_preview.eqn_config.I0)
dim = int(config_preview.eqn_config.dim)
n_raw = len(mu_raw)
dt_raw = T / n_raw
raw_t = np.arange(n_raw) * dt_raw

if gamma_raw.shape[1] == dim:
    gamma_hf = gamma_raw.iloc[:, 0].to_numpy()
    gamma_cha1 = gamma_raw.iloc[:, 1].to_numpy()
elif gamma_raw.shape[1] == dim * dim:
    gamma_tensor = gamma_raw.to_numpy().reshape(n_raw, dim, dim)
    gamma_hf = gamma_tensor[:, 0, 0]
    gamma_cha1 = gamma_tensor[:, 1, 1]
else:
    raise ValueError(
        f"Unexpected Gamma shape {gamma_raw.shape}. Expected {dim} columns or {dim * dim} columns."
    )

mu_hf = mu_raw.iloc[:, 0].to_numpy()
mu_cha1 = mu_raw.iloc[:, 1].to_numpy()
expected_total_demand = float(mu_raw.to_numpy().sum() * dt_raw)
inventory_demand_ratio = I0 / max(expected_total_demand, 1e-12)

fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

axes[0].plot(raw_t, mu_hf, linewidth=2, label=r"HF $\mu$")
axes[0].plot(raw_t, mu_cha1, linewidth=2, linestyle="--", label=r"CHA 1 $\mu$")
axes[0].set_ylabel(r"$\mu$")
axes[0].set_title("HF and CHA 1 demand-rate sequence")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(raw_t, gamma_hf, linewidth=2, label=r"HF $\Gamma$")
axes[1].plot(raw_t, gamma_cha1, linewidth=2, linestyle="--", label=r"CHA 1 $\Gamma$")
axes[1].set_xlabel("Time")
axes[1].set_ylabel(r"$\Gamma$")
axes[1].set_title("HF and CHA 1 variance sequence")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()

out_path = os.path.join(FIG_DIR, "mu_gamma_hf_cha1.png")
plt.savefig(out_path)
plt.show()
print("Saved:", out_path)

param_summary = pd.DataFrame({
    "metric": [
        "I0",
        "expected_total_demand_over_horizon",
        "I0_over_expected_total_demand",
        "num_parameter_points",
        "parameter_dt",
    ],
    "value": [
        I0,
        expected_total_demand,
        inventory_demand_ratio,
        n_raw,
        dt_raw,
    ],
})

print(param_summary.to_string(index=False))


# In[3]:

# Figure 1: training curves for intervals 0, T/2, T

def safe_upper_ylim(values, quantile=0.999):
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        return None
    upper = float(np.quantile(finite, quantile))
    if not np.isfinite(upper) or upper <= 0:
        return None
    return upper


def derive_interval_summary_from_history(train_df):
    idx = train_df.groupby("interval")["iter"].idxmax()
    summary_columns = [
        column
        for column in [
            "interval",
            "best_iter",
            "best_val_loss",
            "final_train_loss",
            "final_val_loss",
            "train_residual_loss",
            "val_residual_loss",
            "train_consistency_loss",
            "val_consistency_loss",
            "train_weighted_consistency_loss",
            "val_weighted_consistency_loss",
            "train_weighted_consistency_to_residual_ratio",
            "val_weighted_consistency_to_residual_ratio",
            "grad_norm",
            "grad_clipped",
            "loss_finite",
            "stop_iter",
            "early_stopped",
            "restored_best",
        ]
        if column in train_df.columns
    ]
    return train_df.loc[idx, summary_columns].copy()


def load_interval_training_summary(train_df):
    derived_summary_df = derive_interval_summary_from_history(train_df)
    if os.path.isfile(TRAIN_SUMMARY_CSV):
        summary_df = pd.read_csv(TRAIN_SUMMARY_CSV)
        print("Using interval training summary:", TRAIN_SUMMARY_CSV)

        summary_intervals = pd.to_numeric(summary_df.get("interval"), errors="coerce")
        derived_intervals = pd.to_numeric(derived_summary_df.get("interval"), errors="coerce")
        existing_intervals = set(summary_intervals.dropna().astype(int))
        missing_intervals = set(derived_intervals.dropna().astype(int)) - existing_intervals
        if missing_intervals:
            missing_rows = derived_summary_df[derived_intervals.astype("Int64").isin(missing_intervals)]
            summary_df = pd.concat([summary_df, missing_rows], ignore_index=True, sort=False)
            print(
                f"Filled {len(missing_rows)} missing interval summaries from training history."
            )
    else:
        print("Interval training summary not found; deriving summary from training history:", TRAIN_CSV)
        summary_df = derived_summary_df

    numeric_columns = [
        "interval",
        "best_iter",
        "best_train_loss",
        "best_val_loss",
        "best_train_residual_loss",
        "best_val_residual_loss",
        "best_train_consistency_loss",
        "best_val_consistency_loss",
        "best_train_weighted_consistency_loss",
        "best_val_weighted_consistency_loss",
        "final_train_loss",
        "final_val_loss",
        "final_train_residual_loss",
        "final_val_residual_loss",
        "final_train_consistency_loss",
        "final_val_consistency_loss",
        "final_train_weighted_consistency_loss",
        "final_val_weighted_consistency_loss",
        "final_train_weighted_consistency_to_residual_ratio",
        "final_val_weighted_consistency_to_residual_ratio",
        "final_grad_norm",
        "stop_iter",
        "consistency_coef",
    ]
    for column in numeric_columns:
        if column in summary_df.columns:
            summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")

    if "best_val_loss" not in summary_df.columns or summary_df["best_val_loss"].isna().all():
        sampled_idx = train_df.groupby("interval")["val_loss"].idxmin()
        summary_df = train_df.loc[sampled_idx, ["interval", "iter", "train_loss", "val_loss"]].copy()
        summary_df = summary_df.rename(
            columns={
                "iter": "best_iter",
                "train_loss": "best_train_loss",
                "val_loss": "best_val_loss",
            }
        )
        print("Warning: best_val_loss was unavailable; using minimum among recorded history rows.")

    summary_df = summary_df.dropna(subset=["interval"]).copy()
    summary_df["interval"] = summary_df["interval"].astype(int)
    summary_df = summary_df.drop_duplicates(subset=["interval"], keep="last")
    return summary_df.sort_values("interval")


if RUN_TRAINING_DIAGNOSTICS:
    train_df_raw = pd.read_csv(TRAIN_CSV, header=None)
    train_history_columns = [
        "interval",
        "iter",
        "train_loss",
        "val_loss",
        "elapsed",
        "best_val_loss",
        "best_iter",
        "final_val_loss",
        "stop_iter",
        "early_stopped",
        "restored_best",
        "train_residual_loss",
        "val_residual_loss",
        "train_consistency_loss",
        "val_consistency_loss",
        "train_weighted_consistency_loss",
        "val_weighted_consistency_loss",
        "train_weighted_consistency_to_residual_ratio",
        "val_weighted_consistency_to_residual_ratio",
        "grad_norm",
        "grad_clipped",
        "loss_finite",
    ]
    df = train_df_raw.iloc[:, :len(train_history_columns)].copy()
    df.columns = train_history_columns[: df.shape[1]]
    interval_summary_df = load_interval_training_summary(df)
    interval_summary_by_idx = interval_summary_df.set_index("interval", drop=False)

    final_interval_idx = df["interval"].max()
    middle_interval_idx = final_interval_idx // 2
    print(f"Final interval index: {final_interval_idx}")
    print(f"Middle interval index: {middle_interval_idx}")

    for itv in [0, middle_interval_idx, final_interval_idx]:
        sub = df[df["interval"] == itv].sort_values("iter")
        plt.figure(figsize=(5, 3))
        plt.plot(sub["iter"], sub["train_loss"], label="train")
        plt.plot(sub["iter"], sub["val_loss"], label="valid")
        if itv in interval_summary_by_idx.index:
            best_iter = interval_summary_by_idx.loc[itv, "best_iter"]
            if np.isfinite(best_iter):
                plt.axvline(best_iter, color="black", linestyle="--", linewidth=1, label="best valid")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        upper = safe_upper_ylim(sub["val_loss"])
        if upper is not None:
            plt.ylim(0, upper)
        # plt.title(f"Interval {itv}")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(FIG_DIR, f"training_curve_interval_{itv}.png")
        plt.savefig(out_path, dpi=150)
        plt.show()
        print("Saved:", out_path)

    # In[4]:
    # Figure 2: best validation loss across all intervals

    plt.figure(figsize=(5, 3))
    plt.plot(interval_summary_df["interval"], interval_summary_df["best_val_loss"], label="best valid")
    plt.xlabel("Interval")
    plt.ylabel("Best Validation Loss")
    upper = safe_upper_ylim(interval_summary_df["best_val_loss"])
    if upper is not None:
        plt.ylim(0, upper)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, "best_validation_loss_by_interval.png")
    plt.savefig(out_path, dpi=150)
    legacy_out_path = os.path.join(FIG_DIR, "final_loss_by_interval.png")
    plt.savefig(legacy_out_path, dpi=150)
    plt.show()
    print("Saved:", out_path)
    print("Saved:", legacy_out_path)

    decomp_columns = [
        "best_val_residual_loss",
        "best_val_weighted_consistency_loss",
    ]
    if all(column in interval_summary_df.columns for column in decomp_columns):
        decomp_df = interval_summary_df.copy()
        for column in ["best_val_loss", *decomp_columns]:
            decomp_df[column] = pd.to_numeric(decomp_df[column], errors="coerce")
        if decomp_df[decomp_columns].notna().any().any():
            plt.figure(figsize=(5, 3))
            plt.plot(decomp_df["interval"], decomp_df["best_val_loss"], label="total best valid")
            plt.plot(decomp_df["interval"], decomp_df["best_val_residual_loss"], label="residual")
            plt.plot(
                decomp_df["interval"],
                decomp_df["best_val_weighted_consistency_loss"],
                label="weighted consistency",
            )
            plt.xlabel("Interval")
            plt.ylabel("Best Validation Loss")
            upper = safe_upper_ylim(decomp_df["best_val_loss"])
            if upper is not None:
                plt.ylim(0, upper)
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(FIG_DIR, "best_validation_loss_decomposition_by_interval.png")
            plt.savefig(out_path, dpi=150)
            plt.show()
            print("Saved:", out_path)
else:
    print("Skipping training diagnostics for eval_scope:", ARGS.eval_scope)


# In[ ]:
# Benchmark evaluation and CSV output
# Note: This cell can be slow depending on num_paths.

NUM_PATHS = ARGS.num_paths

if not ARGS.allow_missing_checkpoints:
    require_complete_z_checkpoints(CONFIG, LOGS_PATH)

context = eval_mod.prepare_evaluation_context(
    CONFIG,
    SOURCE_RUN_NAME,
    logs_path=LOGS_PATH,
    fluid_plan_path=ARGS.fluid_plan_path,
    skip_fluid_benchmark=ARGS.skip_fluid_benchmark,
    ds_extraction_modes=ARGS.policy_extraction_modes,
)
config = context["config"]
bsde = context["bsde"]
device = context["device"]
z_net_list = context["z_net_list"]
policies = context["policies"]
DS_POLICY_NAME = select_deep_splitting_policy(
    policies,
    preferred=ARGS.primary_deep_splitting_policy,
)
print(f"Using {DS_POLICY_NAME} for Deep Splitting diagnostics.")

review_params = eval_mod.prepare_discrete_review_parameters(
    config,
    review_num_intervals=ARGS.review_num_intervals,
    review_dt=ARGS.review_dt,
)
resolved_periodic_demand_mode = eval_mod.resolve_periodic_demand_mode(
    ARGS.periodic_demand_mode,
    bsde=bsde,
    gamma_review=review_params["gamma_review"],
)

FROZEN_EVALUATION_BANK = None
FROZEN_FINE_INCREMENTS = None
FROZEN_PERIODIC_DEMANDS = None
if bool(ARGS.evaluation_bank_manifest) != bool(ARGS.evaluation_bank_sha256):
    raise ValueError("--evaluation_bank_manifest and --evaluation_bank_sha256 must be supplied together.")
if ARGS.evaluation_bank_manifest:
    if not ARGS.use_common_random_numbers:
        raise ValueError("Frozen evaluation banks require common random numbers.")
    scenario = str(ARGS.evaluation_scenario or "").strip()
    if not scenario:
        scenario_path = os.path.join(SOURCE_BASE, "scenario_manifest.json")
        if os.path.isfile(scenario_path):
            with open(scenario_path, encoding="utf-8") as handle:
                scenario = str(json.load(handle).get("resolved_spec", {}).get("name", "")).strip()
    if not scenario:
        raise ValueError("Frozen evaluation requires --evaluation_scenario or a run-local scenario_manifest.json.")
    FROZEN_EVALUATION_BANK = evaluation_bank_mod.load_frozen_evaluation_bank(
        ARGS.evaluation_bank_manifest,
        ARGS.evaluation_bank_sha256,
        expected_scenario=scenario,
        expected_m=int(bsde.num_time_interval),
        expected_num_paths=int(NUM_PATHS),
        expected_seed=int(ARGS.periodic_seed),
        expected_periodic_demand_mode=resolved_periodic_demand_mode,
    )
    if int(ARGS.fine_grid_seed) != int(ARGS.periodic_seed):
        raise ValueError("A frozen combined bank requires identical fine-grid and periodic seeds.")
    FROZEN_FINE_INCREMENTS = torch.from_numpy(
        FROZEN_EVALUATION_BANK["fine_grid_brownian_increments"]
    ).to(device=device, dtype=bsde.mu.dtype)
    FROZEN_PERIODIC_DEMANDS = FROZEN_EVALUATION_BANK["periodic_review_demands"]
    if FROZEN_PERIODIC_DEMANDS.shape[0] != int(review_params["num_review_intervals"]):
        raise ValueError("Frozen periodic-review bank interval count does not match the requested review grid.")
    bank_used_path = os.path.join(BASE, "evaluation_bank_used.json")
    with open(bank_used_path, "x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": evaluation_bank_mod.SCHEMA_VERSION,
                "manifest_path": FROZEN_EVALUATION_BANK["manifest_path"],
                "manifest_sha256": FROZEN_EVALUATION_BANK["manifest_sha256"],
                "bank_id": FROZEN_EVALUATION_BANK["manifest"]["bank_id"],
                "scenario": scenario,
                "m": int(bsde.num_time_interval),
                "num_paths": int(NUM_PATHS),
                "seed": int(ARGS.periodic_seed),
                "periodic_demand_mode": resolved_periodic_demand_mode,
                "arrays": FROZEN_EVALUATION_BANK["manifest"]["arrays"],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print("Loaded frozen evaluation bank:", bank_used_path)

if SOURCE_BASE != BASE:
    source_record_path = os.path.join(BASE, "evaluation_source_used.json")
    with open(source_record_path, "x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": "evaluation-source-used-v1",
                "source_run_dir": SOURCE_BASE,
                "source_run_name": SOURCE_RUN_NAME,
                "config_path": CONFIG,
                "config_sha256": evaluation_bank_mod.file_sha256(CONFIG),
                "checkpoint_dir": LOGS_PATH,
                "checkpoint_count": sum(
                    os.path.isfile(os.path.join(LOGS_PATH, f"z_network{n}.pth"))
                    for n in range(int(bsde.num_time_interval))
                ),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print("Recorded immutable evaluation source:", source_record_path)

if RUN_FINE_TABLE:
    DF_EVAL, FINE_COSTS_BY_POLICY, DF_EVAL_PAIRED = eval_mod.evaluate_fine_grid_policies(
        policies,
        bsde,
        NUM_PATHS,
        device,
        output_path=None,
        use_common_random_numbers=ARGS.use_common_random_numbers,
        common_random_seed=ARGS.fine_grid_seed,
        paired_output_path=os.path.join(BASE, "paired_ds_vs_static_fine_grid.csv"),
        return_costs=True,
        shared_w_increments=FROZEN_FINE_INCREMENTS,
        paired_minuend_policy=DS_POLICY_NAME,
    )
    reporting_contracts.add_operational_cost_summary(DF_EVAL, FINE_COSTS_BY_POLICY)
    if "E3_mean_rollout" in set(DF_EVAL["Policy"]):
        reporting_contracts.add_objective_gap_vs_our_policy(DF_EVAL)
    OUT_CSV = os.path.join(BASE, "evaluation_results.csv")
    OUT_CSV_ALIAS = os.path.join(BASE, "fine_grid_evaluation_results.csv")
    DF_EVAL.to_csv(OUT_CSV, index=False)
    DF_EVAL.to_csv(OUT_CSV_ALIAS, index=False)
    print("Saved:", OUT_CSV)
    print("Saved:", OUT_CSV_ALIAS)
    print(DF_EVAL.to_string(index=False))
    if DF_EVAL_PAIRED is not None:
        print(DF_EVAL_PAIRED.to_string(index=False))
    write_pathwise_cost_artifacts(
        FINE_COSTS_BY_POLICY,
        BASE,
        "fine_grid",
        deep_splitting_policy=DS_POLICY_NAME,
    )


# In[ ]:
# Export LaTeX table

def mean_se(m, se):
    return f"{m:.3f} ({se:.3f})"

if RUN_FINE_TABLE:
    DF_EVAL["Total objective"] = DF_EVAL.apply(lambda r: mean_se(r["Total_Objective_Mean"], r["Total_Objective_SE"]), axis=1)
    DF_EVAL["Untreated cost"] = DF_EVAL.apply(lambda r: mean_se(r["Untreated_Cost_Mean"], r["Untreated_Cost_SE"]), axis=1)
    DF_EVAL["Referral travel cost"] = DF_EVAL.apply(lambda r: mean_se(r["Referral_Travel_Cost_Mean"], r["Referral_Travel_Cost_SE"]), axis=1)
    DF_EVAL["Fairness utility"] = DF_EVAL.apply(lambda r: mean_se(r["Fairness_Utility_Mean"], r["Fairness_Utility_SE"]), axis=1)
    DF_EVAL["Gap vs proposed policy"] = DF_EVAL["Performance_Gap_Percent"].map(lambda v: f"{v:.1f}%")

    latex = DF_EVAL[[
        "Policy",
        "Total objective",
        "Untreated cost",
        "Referral travel cost",
        "Fairness utility",
        "Gap vs proposed policy",
    ]].to_latex(index=False)
    print(latex)

    DF_EVAL[[
        "Policy",
        "Total objective",
        "Untreated cost",
        "Referral travel cost",
        "Fairness utility",
        "Gap vs proposed policy",
    ]].reset_index(drop=True)


# In[ ]:
# ## Discrete-Setting Evaluation
# 
# This section evaluates the policies in the discrete-time model from `main.tex`.
# 
# When the discrete review period is coarser than the training grid, all policy
# rates are evaluated at the observed period-start state over the computational
# subintervals in that period. The resulting shipment quantities are averaged
# over the period and then projected once onto the discrete feasible set.
# 
# For the demand model, we use the shared discrete-review evaluator from
# `evaluate.py`, which resamples the raw parameter rows onto the requested
# review grid and then simulates the configured paper demand model.

# Discrete-time evaluation with fine-grid policy aggregation

DISCRETE_NUM_PATHS = ARGS.num_paths
DISCRETE_SEED = ARGS.periodic_seed

print("ARGS.run_name =", ARGS.run_name)
print("ARGS.num_paths =", ARGS.num_paths)
print("ARGS.review_num_intervals =", ARGS.review_num_intervals)
print("ARGS.review_dt =", ARGS.review_dt)
print("ARGS.discrete_demand_mode =", ARGS.periodic_demand_mode)

print("bsde.num_time_interval =", bsde.num_time_interval)
print("config eqn num_time_interval =", config.eqn_config.num_time_interval)
print("review_params['num_review_intervals'] =", review_params["num_review_intervals"])


if RUN_PERIODIC_TABLE:
    DF_EVAL_DISCRETE, DISCRETE_COSTS_BY_POLICY, DF_EVAL_DISCRETE_PAIRED = eval_mod.evaluate_discrete_review_policies(
        policies,
        bsde,
        DISCRETE_NUM_PATHS,
        device,
        review_params["mu_review"],
        review_params["gamma_review"],
        demand_mode=ARGS.periodic_demand_mode,
        base_seed=DISCRETE_SEED,
        output_path=None,
        use_common_random_numbers=ARGS.use_common_random_numbers,
        paired_output_path=os.path.join(BASE, tagged_csv_name("paired_ds_vs_static_discrete.csv")),
        return_costs=True,
        shared_demands=FROZEN_PERIODIC_DEMANDS,
        paired_minuend_policy=DS_POLICY_NAME,
    )
    reporting_contracts.add_operational_cost_summary(DF_EVAL_DISCRETE, DISCRETE_COSTS_BY_POLICY)
    if "E3_mean_rollout" in set(DF_EVAL_DISCRETE["Policy"]):
        reporting_contracts.add_objective_gap_vs_our_policy(DF_EVAL_DISCRETE)
    DISCRETE_OUT_CSV = os.path.join(BASE, tagged_csv_name("discrete_evaluation_results.csv"))
    DISCRETE_OUT_CSV_ALIAS = os.path.join(BASE, tagged_csv_name("periodic_review_evaluation_results.csv"))
    DF_EVAL_DISCRETE.to_csv(DISCRETE_OUT_CSV, index=False)
    DF_EVAL_DISCRETE.to_csv(DISCRETE_OUT_CSV_ALIAS, index=False)
    print("Saved:", DISCRETE_OUT_CSV)
    print("Saved compatibility alias:", DISCRETE_OUT_CSV_ALIAS)
    print(DF_EVAL_DISCRETE.to_string(index=False))
    if DF_EVAL_DISCRETE_PAIRED is not None:
        print(DF_EVAL_DISCRETE_PAIRED.to_string(index=False))
    write_pathwise_cost_artifacts(
        DISCRETE_COSTS_BY_POLICY,
        BASE,
        "periodic_review",
        deep_splitting_policy=DS_POLICY_NAME,
    )

def mean_se(m, se):
    return f"{m:.3f} ({se:.3f})"

if RUN_PERIODIC_TABLE:
    DF_EVAL_DISCRETE["Total objective"] = DF_EVAL_DISCRETE.apply(lambda r: mean_se(r["Total_Objective_Mean"], r["Total_Objective_SE"]), axis=1)
    DF_EVAL_DISCRETE["Untreated cost"] = DF_EVAL_DISCRETE.apply(lambda r: mean_se(r["Untreated_Cost_Mean"], r["Untreated_Cost_SE"]), axis=1)
    DF_EVAL_DISCRETE["Referral travel cost"] = DF_EVAL_DISCRETE.apply(lambda r: mean_se(r["Referral_Travel_Cost_Mean"], r["Referral_Travel_Cost_SE"]), axis=1)
    DF_EVAL_DISCRETE["Fairness utility"] = DF_EVAL_DISCRETE.apply(lambda r: mean_se(r["Fairness_Utility_Mean"], r["Fairness_Utility_SE"]), axis=1)
    DF_EVAL_DISCRETE["Gap vs proposed policy"] = DF_EVAL_DISCRETE["Performance_Gap_Percent"].map(lambda v: f"{v:.1f}%")

    latex = DF_EVAL_DISCRETE[[
        "Policy",
        "Total objective",
        "Untreated cost",
        "Referral travel cost",
        "Fairness utility",
        "Gap vs proposed policy",
    ]].to_latex(index=False)
    print(latex)
    if WRITE_BASELINE_TABLES:
        write_objective_decomposition_table(
            DF_EVAL_DISCRETE,
            os.path.join(PAPER_TABLE_DIR, "baseline_objective_decomposition.tex"),
        )

    DF_EVAL_DISCRETE[[
        "Policy",
        "Total objective",
        "Untreated cost",
        "Referral travel cost",
        "Fairness utility",
        "Gap vs proposed policy",
    ]].reset_index(drop=True)


# ## Extended Evaluation (Efficiency / Fairness / Seasonality / Inventory Reaction)
# 
# This section computes additional evaluation metrics aligned with the four ideas:
# 1) Efficiency (coverage), 2) Fairness (dispersion), 3) Seasonality (tracking $\mu(t)$), 4) Reaction to inventory.

# In[ ]:


if not RUN_FINE_HISTORY and not RUN_PERIODIC_HISTORY:
    print("Skipping extended diagnostics section by request or eval_scope.")
    sys.exit(0)


import numpy as np


def gini_coefficient(x, eps=1e-12):
    x = np.asarray(x)
    x = x[x >= 0]
    if x.size == 0:
        return np.nan
    x = x + eps
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))


def simulate_trajectories_with_history(policy_fn, bsde, num_paths, device, shared_w_increments=None):
    return eval_mod.simulate_trajectories(
        policy_fn,
        bsde,
        num_paths,
        device,
        return_history=True,
        shared_w_increments=shared_w_increments,
    )


def simulate_periodic_review_trajectories_with_history(
    policy_fn,
    bsde,
    num_paths,
    device,
    mu_review,
    gamma_review,
    demand_mode,
    seed,
    precomputed_demands=None,
    exposure=None,
):
    return eval_mod.simulate_periodic_review_trajectories(
        policy_fn,
        bsde,
        num_paths,
        device,
        mu_review,
        gamma_review,
        demand_mode=demand_mode,
        seed=seed,
        return_history=True,
        precomputed_demands=precomputed_demands,
        exposure=exposure,
    )


# In[ ]:


# Run extended evaluation (smaller num_paths recommended for history tracking)
EXT_NUM_PATHS = ARGS.ext_num_paths
shared_ext_w_increments = None
shared_periodic_demands = None
periodic_exposure = bsde.exposure.detach().cpu().numpy() if hasattr(bsde, "exposure") else None
if ARGS.use_common_random_numbers and RUN_FINE_HISTORY:
    if FROZEN_FINE_INCREMENTS is not None:
        if EXT_NUM_PATHS > NUM_PATHS:
            raise ValueError("Extended evaluation cannot use more paths than the frozen bank.")
        shared_ext_w_increments = FROZEN_FINE_INCREMENTS[:, :EXT_NUM_PATHS]
    else:
        shared_ext_w_increments = eval_mod.precompute_fine_grid_brownian_increments(
            bsde,
            EXT_NUM_PATHS,
            device,
            seed=ARGS.fine_grid_seed,
        )
if ARGS.use_common_random_numbers and RUN_PERIODIC_HISTORY:
    if FROZEN_PERIODIC_DEMANDS is not None:
        if EXT_NUM_PATHS > NUM_PATHS:
            raise ValueError("Extended evaluation cannot use more paths than the frozen bank.")
        shared_periodic_demands = FROZEN_PERIODIC_DEMANDS[:, :EXT_NUM_PATHS]
    else:
        shared_periodic_demands = eval_mod.precompute_periodic_demands(
            review_params["mu_review"],
            review_params["gamma_review"],
            bsde.total_time / review_params["num_review_intervals"],
            EXT_NUM_PATHS,
            demand_mode=resolved_periodic_demand_mode,
            seed=ARGS.periodic_seed,
            exposure=periodic_exposure,
        )

extended_results = {}
if RUN_FINE_HISTORY:
    for name, policy_fn in policies.items():
        print(f"Running extended evaluation for {name}...")
        extended_results[name] = simulate_trajectories_with_history(
            policy_fn,
            bsde,
            EXT_NUM_PATHS,
            device,
            shared_w_increments=shared_ext_w_increments,
        )

periodic_extended_results = {}
if RUN_PERIODIC_HISTORY:
    for idx, (name, policy_fn) in enumerate(policies.items()):
        print(f"Running discrete extended evaluation for {name}...")
        periodic_extended_results[name] = simulate_periodic_review_trajectories_with_history(
            policy_fn,
            bsde,
            EXT_NUM_PATHS,
            device,
            review_params["mu_review"],
            review_params["gamma_review"],
            resolved_periodic_demand_mode,
            DISCRETE_SEED + idx,
            precomputed_demands=shared_periodic_demands,
            exposure=periodic_exposure,
        )


def training_state_envelope(training_summary_path):
    if not os.path.isfile(training_summary_path):
        return {}
    train = pd.read_csv(training_summary_path)
    if "source" in train.columns:
        train = train[train["source"].astype(str) == "training_sampler"]
    envelope = {}
    for _, row in train.iterrows():
        if pd.isna(row.get("interval")) or pd.isna(row.get("variable")):
            continue
        envelope[(int(row["interval"]), str(row["variable"]))] = row
    return envelope


support_rows = []
support_envelope = training_state_envelope(os.path.join(SOURCE_BASE, "state_support_summary.csv"))
if DS_POLICY_NAME in extended_results:
    support_rows.extend(
        state_support_alignment.fine_grid_state_support_rows(
            extended_results[DS_POLICY_NAME],
            source="fine_grid_deep_splitting",
            envelope=support_envelope,
        )
    )
if DS_POLICY_NAME in periodic_extended_results:
    support_rows.extend(
        state_support_alignment.periodic_state_support_rows(
            periodic_extended_results[DS_POLICY_NAME],
            num_fine_intervals=int(bsde.num_time_interval),
            source="discrete_review_deep_splitting",
            envelope=support_envelope,
        )
    )
if support_rows:
    support_path = os.path.join(BASE, "evaluation_state_support_summary.csv")
    pd.DataFrame(support_rows).to_csv(support_path, index=False)
    print(f"Saved: {support_path}")


def compute_periodic_operational_metrics(res, bsde, eps=1e-12):
    q_hist = res["q_hist"]
    l_hist = res["l_hist"]
    d_hist = res["d_hist"]
    z_hist = res["z_hist"]
    i_hist = res["i_hist"]
    eta_hist = res.get("eta_hist")
    p = bsde.p.detach().cpu().numpy()
    distance = bsde.distance.detach().cpu().numpy()
    I0 = float(bsde.I0)
    num_weeks = q_hist.shape[0]
    num_paths = q_hist.shape[1]
    review_dt = float(bsde.total_time) / max(int(num_weeks), 1)

    cha_demand = d_hist[:, :, 1:]
    cha_local_unmet = l_hist[:, :, 1:]
    cha_local_service = np.maximum(cha_demand - cha_local_unmet, 0.0)
    referrals = cha_local_unmet * p[None, None, :]
    hf_load = d_hist[:, :, 0] + referrals.sum(axis=2)
    hf_short = l_hist[:, :, 0]
    hf_service_total = np.maximum(hf_load - hf_short, 0.0)
    hf_service_total = np.minimum(hf_service_total, hf_load)
    hf_load_safe = np.maximum(hf_load, eps)
    cha_hf_service = referrals / hf_load_safe[:, :, None] * hf_service_total[:, :, None]
    own_hf_service = d_hist[:, :, 0] / hf_load_safe * hf_service_total
    cha_hf_service = np.where(hf_load[:, :, None] > eps, cha_hf_service, 0.0)
    own_hf_service = np.where(hf_load > eps, own_hf_service, 0.0)
    cha_hf_service = np.minimum(cha_hf_service, referrals)
    own_hf_service = np.minimum(own_hf_service, d_hist[:, :, 0])

    total_demand = d_hist.sum(axis=(0, 2))
    total_service = hf_service_total.sum(axis=0) + cha_local_service.sum(axis=(0, 2))
    coverage = total_service / np.maximum(total_demand, eps)

    final_inventory = i_hist[-1] + z_hist[-1].sum(axis=1)
    utilization = 1.0 - final_inventory / max(I0, eps)

    cha_excess = z_hist[1:, :, 1:]
    mismatch = (cha_excess > eps).any(axis=2) & (cha_local_unmet > eps).any(axis=2)
    mismatch_freq = mismatch.mean(axis=0)

    shipments_total = q_hist.sum(axis=(0, 2))
    hf_share = q_hist[:, :, 0].sum(axis=0) / np.maximum(shipments_total, eps)

    cha_demand_total = cha_demand.sum(axis=(0, 2))
    local_fill = cha_local_service.sum(axis=(0, 2)) / np.maximum(cha_demand_total, eps)
    referral_served = cha_hf_service.sum(axis=(0, 2)) / np.maximum(cha_demand_total, eps)
    travel_burden = (referrals * distance[None, None, 1:]).sum(axis=(0, 2)) / np.maximum(cha_demand_total, eps)
    hf_coverage = hf_service_total.sum(axis=0) / np.maximum(hf_load.sum(axis=0), eps)

    cha_location_demand = cha_demand.sum(axis=0)
    cha_location_service = cha_local_service.sum(axis=0)
    local_fill_by_cha = cha_location_service / np.maximum(cha_location_demand, eps)
    gap_local_fill = np.nanmax(local_fill_by_cha, axis=1) - np.nanmin(local_fill_by_cha, axis=1)
    worst_local_fill = np.nanmin(local_fill_by_cha, axis=1)

    total_allocation_by_path = q_hist.sum(axis=(0, 2))

    depletion_week_by_path = np.full(num_paths, np.nan, dtype=float)
    depleted = i_hist <= eps
    for path_idx in range(num_paths):
        hits = np.flatnonzero(depleted[:, path_idx])
        if hits.size:
            depletion_week_by_path[path_idx] = float(hits[0])
    has_depletion = np.isfinite(depletion_week_by_path)

    finite_q = np.isfinite(q_hist)
    zero_action = finite_q & (q_hist <= eps)
    cap_action = np.zeros_like(zero_action, dtype=bool)
    if eta_hist is not None:
        eta_arr = np.asarray(eta_hist, dtype=float)
        if eta_arr.shape == (num_weeks, q_hist.shape[2]):
            period_cap = eta_arr[:, None, :] * review_dt
            cap_action = finite_q & (period_cap > eps) & (q_hist >= period_cap - 1e-6)
    interior_action = finite_q & (~zero_action) & (~cap_action)
    action_denom = max(int(finite_q.sum()), 1)

    projection_needed = np.asarray(res.get("projection_needed_hist", []), dtype=bool)
    projection_scale = np.asarray(res.get("projection_scale_hist", []), dtype=float)
    if projection_needed.shape == (num_weeks, num_paths):
        projection_frequency = float(np.mean(projection_needed))
        active_scale = (
            projection_scale[projection_needed]
            if projection_scale.shape == projection_needed.shape
            else np.asarray([], dtype=float)
        )
        mean_projection_scale = float(np.mean(active_scale)) if active_scale.size else 1.0
        min_projection_scale = float(np.min(active_scale)) if active_scale.size else 1.0
    else:
        projection_frequency = np.nan
        mean_projection_scale = np.nan
        min_projection_scale = np.nan

    return {
        "Coverage": float(np.nanmean(coverage)),
        "Utilization": float(np.nanmean(utilization)),
        "MismatchFreq": float(np.nanmean(mismatch_freq)),
        "HFShare": float(np.nanmean(hf_share)),
        "HF_Coverage": float(np.nanmean(hf_coverage)),
        "LocalFill": float(np.nanmean(local_fill)),
        "ReferralServed": float(np.nanmean(referral_served)),
        "TravelBurden": float(np.nanmean(travel_burden)),
        "GapLocalFill": float(np.nanmean(gap_local_fill)),
        "WorstLocalFill": float(np.nanmean(worst_local_fill)),
        "Gap_Coverage": float(np.nanmean(gap_local_fill)),
        "Worst_Coverage": float(np.nanmean(worst_local_fill)),
        "TotalAllocation": float(np.nanmean(total_allocation_by_path)),
        "TotalAllocationSE": float(np.nanstd(total_allocation_by_path, ddof=1) / np.sqrt(max(num_paths, 1))),
        "WarehouseDepletionWeek": float(np.nanmean(depletion_week_by_path)) if np.any(has_depletion) else np.nan,
        "WarehouseDepletionFraction": float(np.mean(has_depletion)),
        "InventoryProjectionFrequency": projection_frequency,
        "InventoryProjectionMeanScale": mean_projection_scale,
        "InventoryProjectionMinScale": min_projection_scale,
        "Action_Zero_Fraction": float(np.count_nonzero(zero_action) / action_denom),
        "Action_Interior_Fraction": float(np.count_nonzero(interior_action) / action_denom),
        "Action_EtaCap_Fraction": float(np.count_nonzero(cap_action) / action_denom),
    }


def build_periodic_operational_metric_table(results_by_policy):
    return pd.DataFrame({
        policy_name: compute_periodic_operational_metrics(res, bsde)
        for policy_name, res in results_by_policy.items()
    }).T


def eta_schedule_raw_array(bsde):
    if hasattr(bsde, "eta_schedule_raw"):
        return bsde.eta_schedule_raw.detach().cpu().numpy()
    rows = int(getattr(bsde, "raw_num_time_interval", bsde.num_time_interval))
    eta = bsde.eta.detach().cpu().numpy().reshape(1, -1)
    return np.repeat(eta, rows, axis=0)


def eta_schedule_fine_array(bsde, rows=None):
    if hasattr(bsde, "eta_schedule"):
        eta = bsde.eta_schedule.detach().cpu().numpy()
    else:
        eta = np.repeat(bsde.eta.detach().cpu().numpy().reshape(1, -1), bsde.num_time_interval, axis=0)
    if rows is None or rows == eta.shape[0]:
        return eta
    if rows < eta.shape[0]:
        return eta[:rows]
    return np.repeat(eta[-1:, :], rows - eta.shape[0], axis=0)


def write_eta_schedule_diagnostics(bsde, base_dir):
    eta_raw = eta_schedule_raw_array(bsde)
    rows, dim = eta_raw.shape
    all_weeks_identical = bool(np.allclose(eta_raw, eta_raw[0:1]))
    cha_profiles_identical = bool(dim <= 2 or np.allclose(eta_raw[:, 1:], eta_raw[:, 1:2]))
    labels = ["HF"] + [f"CHA {k}" for k in range(1, dim)]
    min_by_loc = eta_raw.min(axis=0)
    max_by_loc = eta_raw.max(axis=0)
    week_min = eta_raw.min(axis=1)
    week_max = eta_raw.max(axis=1)
    out_rows = []
    for n in range(rows):
        for k in range(dim):
            out_rows.append(
                {
                    "eta_mode": getattr(bsde, "eta_mode", "constant_peak"),
                    "raw_shape_rows": rows,
                    "raw_shape_cols": dim,
                    "week": n + 1,
                    "location_index": k,
                    "location": labels[k],
                    "eta": float(eta_raw[n, k]),
                    "min_by_location": float(min_by_loc[k]),
                    "max_by_location": float(max_by_loc[k]),
                    "min_by_week": float(week_min[n]),
                    "max_by_week": float(week_max[n]),
                    "all_weeks_identical": all_weeks_identical,
                    "cha_profiles_identical": cha_profiles_identical,
                }
            )
    path = os.path.join(base_dir, tagged_csv_name("eta_schedule_diagnostics.csv"))
    pd.DataFrame(out_rows).to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def write_periodic_eta_policy_diagnostics(results_by_policy, bsde, base_dir):
    q0 = max(float(bsde.I0), 1e-12)
    rows_early = []
    rows_weekly = []
    rows_total = []
    rows_inventory = []
    rows_action_mechanism = []
    fairness_by_policy = {}
    cutoffs = [1, 2, 4, 8]
    for policy_name, res in results_by_policy.items():
        q_hist = np.asarray(res["q_hist"], dtype=float)
        q_raw_hist = np.asarray(res.get("q_raw_hist", q_hist), dtype=float)
        i_hist = np.asarray(res["i_hist"], dtype=float)
        num_weeks, num_paths, dim = q_hist.shape
        labels = ["HF"] + [f"CHA {k}" for k in range(1, dim)]
        fine_rates = np.asarray(res.get("fine_rate_hist", []), dtype=float)
        raw_fine_rates = np.asarray(res.get("raw_fine_rate_hist", []), dtype=float)
        fine_quantities = np.asarray(res.get("fine_quantity_hist", []), dtype=float)
        fine_projection = np.asarray(res.get("fine_projection_needed_hist", []), dtype=bool)
        period_projection = np.asarray(res.get("projection_needed_hist", []), dtype=bool)
        fine_per_review = fine_rates.shape[1] if fine_rates.ndim == 4 else 0
        eta_fine = eta_schedule_fine_array(bsde)
        review_dt = float(res.get("time_dt", float(bsde.total_time) / max(num_weeks, 1)))
        eta_blocks = (
            eta_fine.reshape(num_weeks, fine_per_review, dim)
            if fine_per_review > 0 and eta_fine.shape == (num_weeks * fine_per_review, dim)
            else None
        )
        mu_hist = np.asarray(res.get("mu_hist", []), dtype=float)
        if mu_hist.shape == (num_weeks, dim):
            weekly_fairness = np.stack(
                [
                    eval_mod.eqn.alpha_fair_utility_numpy(
                        q_hist[n], mu_hist[n] * review_dt, float(bsde.alpha)
                    )
                    for n in range(num_weeks)
                ],
                axis=0,
            )
            fairness_by_policy[policy_name] = {
                "weekly": weekly_fairness,
                "cumulative": np.cumsum(weekly_fairness, axis=0),
            }

        for cutoff in cutoffs:
            week = min(int(cutoff), num_weeks)
            if week <= 0:
                continue
            cumulative = q_hist[:week].sum(axis=(0, 2))
            hf_cumulative = q_hist[:week, :, 0].sum(axis=0)
            cha_cumulative = q_hist[:week, :, 1:].sum(axis=(0, 2))
            rows_early.append(
                {
                    "policy": policy_name,
                    "through_week": week,
                    "cumulative_total_shipment_mean": float(np.mean(cumulative)),
                    "cumulative_hf_shipment_mean": float(np.mean(hf_cumulative)),
                    "cumulative_cha_shipment_mean": float(np.mean(cha_cumulative)),
                    "fraction_initial_inventory_shipped_mean": float(np.mean(cumulative / q0)),
                }
            )

        cumulative_total = np.zeros(num_paths, dtype=float)
        for n in range(num_weeks):
            q_n = q_hist[n]
            total_n = q_n.sum(axis=1)
            cumulative_total += total_n
            rows_total.append(
                {
                    "policy": policy_name,
                    "week": n + 1,
                    "mean_total_shipment": float(np.mean(total_n)),
                    "se_total_shipment": float(np.std(total_n, ddof=1) / np.sqrt(max(num_paths, 1))),
                    "mean_cumulative_total_shipment": float(np.mean(cumulative_total)),
                    "fraction_initial_inventory_shipped": float(np.mean(cumulative_total / q0)),
                }
            )
            for k in range(dim):
                values = q_n[:, k]
                rows_weekly.append(
                    {
                        "policy": policy_name,
                        "week": n + 1,
                        "location_index": k,
                        "location": labels[k],
                        "mean_shipment": float(np.mean(values)),
                        "se_shipment": float(np.std(values, ddof=1) / np.sqrt(max(num_paths, 1))),
                    }
                )

            mechanism = {
                "policy": policy_name,
                "week": n + 1,
                "mean_raw_period_quantity": float(np.mean(q_raw_hist[n].sum(axis=1))),
                "mean_projected_period_quantity": float(np.mean(q_hist[n].sum(axis=1))),
                "mean_inventory_start": float(np.mean(i_hist[n])),
                "mean_inventory_end": float(np.mean(i_hist[n + 1])),
                "mean_shipment_fraction_initial_inventory": float(np.mean(total_n / q0)),
                "mean_cumulative_shipment_fraction_initial_inventory": float(
                    np.mean(cumulative_total / q0)
                ),
                "inventory_depleted_fraction_end": float(np.mean(i_hist[n + 1] <= 1e-8)),
                "max_projected_period_quantity_minus_inventory": float(
                    np.max(total_n - i_hist[n])
                ),
                "period_projection_fraction": (
                    float(np.mean(period_projection[n]))
                    if period_projection.shape == (num_weeks, num_paths)
                    else np.nan
                ),
            }
            if (
                fine_rates.ndim == 4
                and raw_fine_rates.shape == fine_rates.shape
                and fine_rates.shape[0] == num_weeks
                and fine_rates.shape[2:] == (num_paths, dim)
                and eta_blocks is not None
            ):
                eta_week = eta_blocks[n][:, None, :]
                fine_dt = review_dt / fine_per_review
                finite = np.isfinite(fine_rates[n])
                positive_cap = eta_week > 1e-12
                cap_tolerance = np.maximum(1e-6, 1e-6 * np.maximum(eta_week, 1.0))
                cap_hit = finite & positive_cap & (fine_rates[n] >= eta_week - cap_tolerance)
                raw_cap_exceed = (
                    np.isfinite(raw_fine_rates[n])
                    & positive_cap
                    & (raw_fine_rates[n] > eta_week + cap_tolerance)
                )
                zero = finite & (fine_rates[n] <= 1e-8)
                interior = finite & (~zero) & (~cap_hit)
                denom = max(int(np.count_nonzero(finite)), 1)
                mechanism.update(
                    {
                        "fine_cap_hit_fraction": float(np.count_nonzero(cap_hit) / denom),
                        "raw_fine_cap_exceed_fraction": float(
                            np.count_nonzero(raw_cap_exceed) / denom
                        ),
                        "fine_interior_action_fraction": float(np.count_nonzero(interior) / denom),
                        "mean_raw_fine_rate": float(np.nanmean(raw_fine_rates[n])),
                        "mean_effective_fine_rate": float(np.nanmean(fine_rates[n])),
                        "fine_projection_fraction": (
                            float(np.mean(fine_projection[n]))
                            if fine_projection.shape == fine_rates.shape[:3]
                            else np.nan
                        ),
                        "max_effective_fine_rate_minus_eta": float(
                            np.nanmax(fine_rates[n] - eta_week)
                        ),
                        "max_rate_times_dt_quantity_error": (
                            float(
                                np.nanmax(
                                    np.abs(fine_quantities[n] - fine_rates[n] * fine_dt)
                                )
                            )
                            if fine_quantities.shape == fine_rates.shape
                            else np.nan
                        ),
                    }
                )
            else:
                mechanism.update(
                    {
                        "fine_cap_hit_fraction": np.nan,
                        "raw_fine_cap_exceed_fraction": np.nan,
                        "fine_interior_action_fraction": np.nan,
                        "mean_raw_fine_rate": np.nan,
                        "mean_effective_fine_rate": np.nan,
                        "fine_projection_fraction": np.nan,
                        "max_effective_fine_rate_minus_eta": np.nan,
                        "max_rate_times_dt_quantity_error": np.nan,
                    }
                )
            rows_action_mechanism.append(mechanism)

        for n in range(i_hist.shape[0]):
            values = i_hist[n]
            rows_inventory.append(
                {
                    "policy": policy_name,
                    "week": n,
                    "mean_warehouse_inventory": float(np.mean(values)),
                    "se_warehouse_inventory": float(np.std(values, ddof=1) / np.sqrt(max(num_paths, 1))),
                }
            )

    rows_fairness = []
    benchmark_fairness = fairness_by_policy.get("Scaled_Static")
    for policy_name, values in fairness_by_policy.items():
        weekly = values["weekly"]
        cumulative = values["cumulative"]
        for n in range(weekly.shape[0]):
            benchmark_cumulative_mean = (
                float(np.mean(benchmark_fairness["cumulative"][n]))
                if benchmark_fairness is not None
                else np.nan
            )
            cumulative_mean = float(np.mean(cumulative[n]))
            rows_fairness.append(
                {
                    "policy": policy_name,
                    "week": n + 1,
                    "weekly_fairness_utility_mean": float(np.mean(weekly[n])),
                    "cumulative_fairness_utility_mean": cumulative_mean,
                    "scaled_static_cumulative_fairness_utility_mean": benchmark_cumulative_mean,
                    "cumulative_fairness_retention_vs_scaled_static": (
                        cumulative_mean / benchmark_cumulative_mean
                        if np.isfinite(benchmark_cumulative_mean)
                        and abs(benchmark_cumulative_mean) > 1e-12
                        else np.nan
                    ),
                }
            )

    outputs = {
        "early_shipment_diagnostics.csv": rows_early,
        "weekly_shipments_by_policy_location.csv": rows_weekly,
        "total_shipment_trajectory_by_policy.csv": rows_total,
        "warehouse_inventory_trajectory_by_policy.csv": rows_inventory,
        "weekly_action_mechanism_diagnostics.csv": rows_action_mechanism,
        "weekly_fairness_trajectory_by_policy.csv": rows_fairness,
    }
    for filename, rows in outputs.items():
        path = os.path.join(base_dir, tagged_csv_name(filename))
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved: {path}")


def write_periodic_location_diagnostics(results_by_policy, bsde, base_dir, eps=1e-12):
    """Write presentation-source CHA diagnostics from the existing CRN histories.

    This intentionally reuses ``periodic_extended_results`` rather than running a
    second simulation.  The resulting location tables therefore have exactly the
    same policies, paths, demand draws, and extraction modes as the certified
    periodic-review trajectory outputs.
    """

    outputs = build_periodic_location_diagnostic_rows(
        results_by_policy,
        bsde.p.detach().cpu().numpy(),
        bsde.distance.detach().cpu().numpy(),
        eps=eps,
    )
    for filename, rows in outputs.items():
        path = os.path.join(base_dir, tagged_csv_name(filename))
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved: {path}")


def write_publication_inventory_trajectory(results_by_policy, base_dir):
    """Write the unified nine-location table from this periodic evaluation.

    The opt-in is the frozen ``--setting_id`` supplied by the stream-16
    campaign.  Older report invocations retain their existing outputs and do
    not acquire partially populated publication provenance.
    """

    if not ARGS.setting_id:
        return None
    required = {
        "evaluation_run_id": ARGS.evaluation_run_id,
        "training_run_id": ARGS.training_run_id,
        "stream_id": ARGS.stream_id,
        "training_branch": ARGS.training_branch,
        "training_commit": ARGS.training_commit,
        "config_hash": ARGS.config_hash,
        "config_file_sha256": ARGS.config_file_sha256,
        "checkpoint_certification_sha256": ARGS.checkpoint_certification_sha256,
        "evaluation_branch": ARGS.evaluation_branch,
        "evaluation_commit": ARGS.evaluation_commit,
    }
    missing = sorted(name for name, value in required.items() if value in (None, ""))
    if missing:
        raise ValueError(f"Frozen publication inventory metadata is incomplete: {missing}")
    actual_config_hash = evaluation_bank_mod.file_sha256(CONFIG)
    if actual_config_hash != ARGS.config_file_sha256:
        raise ValueError("Frozen publication inventory config-file hash does not match config_used.json")
    configured_stream = int(config.net_config.train_seed)
    if configured_stream != int(ARGS.stream_id):
        raise ValueError("Frozen publication inventory stream does not match the source config")
    if FROZEN_EVALUATION_BANK is None:
        raise ValueError("Frozen publication inventory requires an exact evaluation bank")
    metadata = {
        "setting_id": ARGS.setting_id,
        "evaluation_run_id": ARGS.evaluation_run_id,
        "training_run_id": ARGS.training_run_id,
        "stream_id": ARGS.stream_id,
        "evaluation_mode": "periodic_review",
        "evaluation_bank_id": FROZEN_EVALUATION_BANK["manifest"]["bank_id"],
        "evaluation_bank_sha256": FROZEN_EVALUATION_BANK["manifest_sha256"],
        "training_branch": ARGS.training_branch,
        "training_commit": ARGS.training_commit,
        "config_hash": ARGS.config_hash,
        "checkpoint_certification_sha256": ARGS.checkpoint_certification_sha256,
        "evaluation_branch": ARGS.evaluation_branch,
        "evaluation_commit": ARGS.evaluation_commit,
    }
    rows = reporting_contracts.aggregate_inventory_trajectories(results_by_policy, metadata)
    path = os.path.join(base_dir, tagged_csv_name("inventory_trajectory_by_policy_location.csv"))
    reporting_contracts.write_inventory_trajectory_csv(rows, path)
    print(f"Saved: {path}")
    return path


if RUN_PERIODIC_HISTORY:
    write_eta_schedule_diagnostics(bsde, BASE)
    periodic_operational_summary = build_periodic_operational_metric_table(periodic_extended_results)
    print("Discrete-review operational metrics")
    print(periodic_operational_summary.to_string())
    DISCRETE_OPERATIONAL_RESULTS_CSV = os.path.join(
        BASE,
        tagged_csv_name("discrete_auxiliary_metrics.csv"),
    )
    PERIODIC_OPERATIONAL_RESULTS_CSV_ALIAS = os.path.join(
        BASE,
        tagged_csv_name("periodic_review_auxiliary_metrics.csv"),
    )
    periodic_operational_summary.to_csv(DISCRETE_OPERATIONAL_RESULTS_CSV, index=True)
    periodic_operational_summary.to_csv(PERIODIC_OPERATIONAL_RESULTS_CSV_ALIAS, index=True)
    print(f"Saved: {DISCRETE_OPERATIONAL_RESULTS_CSV}")
    print(f"Saved compatibility alias: {PERIODIC_OPERATIONAL_RESULTS_CSV_ALIAS}")
    write_publication_inventory_trajectory(periodic_extended_results, BASE)
    write_periodic_eta_policy_diagnostics(periodic_extended_results, bsde, BASE)
    write_periodic_location_diagnostics(periodic_extended_results, bsde, BASE)
    if WRITE_BASELINE_TABLES:
        write_operational_metrics_table(
            periodic_operational_summary,
            os.path.join(PAPER_TABLE_DIR, "baseline_operational_metrics.tex"),
        )
    if ARGS.write_extended_evaluation_artifacts:
        if FROZEN_EVALUATION_BANK is None:
            raise ValueError("Extended evaluation artifacts require an exact frozen evaluation bank")
        extended_metadata = {
            "setting_id": ARGS.setting_id,
            "evaluation_run_id": ARGS.evaluation_run_id,
            "stream_id": ARGS.stream_id,
            "evaluation_bank_id": FROZEN_EVALUATION_BANK["manifest"]["bank_id"],
            "evaluation_bank_sha256": FROZEN_EVALUATION_BANK["manifest_sha256"],
            "evaluation_commit": ARGS.evaluation_commit,
        }
        written = extended_evaluation_diagnostics.write_extended_evaluation_artifacts(
            periodic_extended_results,
            BASE,
            extended_metadata,
            bsde.p.detach().cpu().numpy(),
            bsde.distance.detach().cpu().numpy(),
            enabled=True,
        )
        print("Saved opt-in extended evaluation artifacts:", json.dumps(written, sort_keys=True))
    # All downstream publication summaries are now materialized; raw periodic
    # histories are intentionally not serialized by the unified output.
    periodic_extended_results.clear()

if not RUN_FINE_HISTORY:
    sys.exit(0)


# ## Trajectory Diagnostics: $I(t)$ vs $\sum_k \mu_k(t)$ and $a_k(t)$ vs $\mu_k(t)$
# 
# This section visualizes the simulated trajectory summaries from $extended_results$.
# - Figure A compares the mean unallocated inventory $I(t)$ against the total demand rate $\sum_k \mu_k(t)$ for all policies on one axis.
# - Figure B compares the mean allocation process $a_k(t)$ against the corresponding demand rate $\mu_k(t)$ for each location and each policy.
# 

# In[ ]:


# Trajectory diagnostics plots

import math


def extend_mu_to_state_grid(mu_hist):
    """Extend mu from N points to N+1 state-grid points by repeating the last value."""
    return np.vstack([mu_hist, mu_hist[-1:]])

print("Mean I(t) and total demand rate by policy")
policy_order = [
    *DEEP_SPLITTING_POLICY_CANDIDATES,
    "Status_Quo",
    "Static",
    "Scaled_Static",
    "Fluid_Optimal",
    "Myopic",
]
policy_order = [name for name in policy_order if name in extended_results]
policy_colors = {
    "Deep_Splitting": "tab:blue",
    "E0_frozen_average": "tab:blue",
    "E2_q_rollout": "tab:cyan",
    "E3_mean_rollout": "tab:olive",
    "Status_Quo": "tab:red",
    "Static": "tab:orange",
    "Scaled_Static": "tab:purple",
    "Fluid_Optimal": "tab:brown",
    "Myopic": "tab:green",
}

N = bsde.num_time_interval
state_t = np.arange(N + 1) * bsde.delta_t
control_t = np.arange(N) * bsde.delta_t
location_labels = ["HF"] + [f"CHA {k}" for k in range(1, bsde.dim)]

# Figure A: I(t) vs sum_k mu_k(t), all policies on one axis
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
mu_sums = []
for name in policy_order:
    res = extended_results[name]
    i_mean = res["i_hist"].mean(axis=1)
    mu_state = extend_mu_to_state_grid(res["mu_hist"])
    mu_sum = mu_state.sum(axis=1)
    mu_sums.append(mu_sum)
    ax.plot(state_t, i_mean, color=policy_colors[name], linewidth=2, label=f"{name}")

# mu_stack = np.stack(mu_sums, axis=0)
# if np.allclose(mu_stack, mu_stack[0:1], rtol=1e-5, atol=1e-8):
#     ax.plot(state_t, mu_sums[0], color="black", linestyle="--", linewidth=1.5, label=r"$\sum_k \mu_k(t)$")
# else:
#     for name, mu_sum in zip(policy_order, mu_sums):
#         ax.plot(
#             state_t,
#             mu_sum,
#             color=policy_colors[name],
#             linestyle="--",
#             linewidth=1.5,
#             alpha=0.85,
#             label=f"{name}: " + r"$\sum_k \mu_k(t)$",
#         )

ax.set_xlabel("Time")
ax.set_ylabel("Level / rate")
# ax.set_title("Mean I(t) and total demand rate by policy")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()

out_path = os.path.join(FIG_DIR, "I_vs_mu_sum_by_policy.png")
plt.savefig(out_path)
plt.show()
print("Saved:", out_path)

# Figure B: a_k(t) vs mu_k(t), one multi-panel figure per policy
for name in policy_order:
    res = extended_results[name]
    a_mean = res["a_hist"].mean(axis=1)  # (N, K+1)
    mu_hist = res["mu_hist"]  # (N, K+1)

    ncols = 2
    nrows = math.ceil(bsde.dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.8 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for k in range(bsde.dim):
        ax = axes[k]
        ax.plot(control_t, a_mean[:, k], color=policy_colors[name], linewidth=2, label=rf"mean $a_{k}(t)$")
        ax.plot(control_t, mu_hist[:, k], color="black", linestyle="--", linewidth=1.2, label=rf"$\mu_{k}(t)$")
        ax.set_title(location_labels[k])
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for k in range(bsde.dim, len(axes)):
        axes[k].axis("off")

    for ax in axes[-ncols:]:
        if ax.has_data():
            ax.set_xlabel("Time")

    fig.suptitle(f"{name.replace('_', ' ')}: allocation trajectory vs demand rate", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(FIG_DIR, f"a_vs_mu_by_location_{name}.png")
    plt.savefig(out_path)
    plt.show()
    print("Saved:", out_path)


# ## Deep Splitting Threshold Diagnostics
# 
# This section aligns the diagnostics with the theoretical threshold rule for the Deep Splitting policy.
# 
# It focuses on whether the recovered signal $\partial_{z_k}V - \partial_iV$ is above or below the fairness threshold
# $\tau_k(t) = \lambda (\mu_k(t)/\eta_k(t))^{\alpha}$, and separates three mechanisms behind boundary behavior:
# 1. no available central inventory,
# 2. threshold-implied upper-bound allocation, and
# 3. interior allocation.

# In[ ]:


# Deep Splitting threshold-aligned diagnostics

DIAG_POLICY = DS_POLICY_NAME
ACTION_TOL = 1e-6
INVENTORY_TOL = 1e-6
RATIO_SAMPLE_PER_STEP = 100
EPS = 1e-12
RNG = np.random.default_rng(0)


def compute_deep_splitting_threshold_diagnostics(res, z_net_list, bsde, device,
                                                 action_tol=1e-6,
                                                 inventory_tol=1e-6,
                                                 ratio_sample_per_step=100,
                                                 eps=1e-12):
    a_hist = res["a_hist"]
    z_hist = res["z_hist"]
    i_hist = res["i_hist"]
    mu_hist = res["mu_hist"]

    N, P, K = a_hist.shape
    eta_hist = eta_schedule_fine_array(bsde, N)
    lambda_param = float(bsde.lambda_param)
    alpha = float(bsde.alpha)

    threshold = np.zeros((N, K))
    no_inventory_share = np.zeros(N)
    negative_grad_share = np.zeros(N)
    fairness_upper_share = np.zeros(N)
    interior_region_share = np.zeros(N)
    inventory_binding_share = np.zeros(N)

    upper_region_share_by_loc = np.zeros((N, K))
    actual_eta_share_by_loc = (a_hist >= (eta_hist[:, None, :] - action_tol)).mean(axis=1)
    actual_zero_share_by_loc = (a_hist <= action_tol).mean(axis=1)

    ratio_samples = {0: [], 1: []}
    median_ratio_by_loc = np.full((N, K), np.nan)

    with torch.no_grad():
        for n in range(N):
            z_n = torch.from_numpy(z_hist[n]).to(device=device, dtype=bsde.mu.dtype)
            i_n = torch.from_numpy(i_hist[n]).to(device=device, dtype=bsde.mu.dtype).unsqueeze(1)
            y_n = torch.cat([z_n, i_n], dim=1)

            grad_diff_n = z_net_list[min(n, len(z_net_list) - 1)](y_n).detach().cpu().numpy()

            if lambda_param > 0:
                valid = (mu_hist[n] > eps) & (eta_hist[n] > eps)
                threshold_n = np.zeros(K)
                threshold_n[valid] = lambda_param * np.power(mu_hist[n][valid] / eta_hist[n][valid], alpha)
            else:
                threshold_n = np.zeros(K)
            threshold[n] = threshold_n

            no_inventory_mask = i_hist[n][:, None] <= inventory_tol
            negative_grad_mask = (~no_inventory_mask) & (grad_diff_n <= 0.0)

            if lambda_param > 0:
                fairness_upper_mask = (~no_inventory_mask) & (grad_diff_n > 0.0) & (grad_diff_n <= threshold_n[None, :] + eps)
                interior_mask = (~no_inventory_mask) & (grad_diff_n > threshold_n[None, :] + eps)
                ratio_n = grad_diff_n / np.maximum(threshold_n[None, :], eps)

                for loc in [0, 1]:
                    loc_vals = ratio_n[:, loc]
                    take = min(ratio_sample_per_step, loc_vals.size)
                    if take > 0:
                        idx = RNG.choice(loc_vals.size, size=take, replace=False)
                        ratio_samples[loc].append(loc_vals[idx])
                    median_ratio_by_loc[n, loc] = np.nanmedian(loc_vals)
            else:
                fairness_upper_mask = np.zeros_like(negative_grad_mask)
                interior_mask = (~no_inventory_mask) & (grad_diff_n > 0.0)

            upper_region_mask = negative_grad_mask | fairness_upper_mask

            no_inventory_share[n] = no_inventory_mask.mean()
            negative_grad_share[n] = negative_grad_mask.mean()
            fairness_upper_share[n] = fairness_upper_mask.mean()
            interior_region_share[n] = interior_mask.mean()
            upper_region_share_by_loc[n] = upper_region_mask.mean(axis=0)

            alloc_step = a_hist[n].sum(axis=1) * bsde.delta_t
            inventory_prev = i_hist[n]
            positive_mask = inventory_prev > inventory_tol
            if np.any(positive_mask):
                inventory_binding_share[n] = np.mean(
                    np.abs(alloc_step[positive_mask] - inventory_prev[positive_mask]) <= inventory_tol
                )
            else:
                inventory_binding_share[n] = np.nan

    ratio_samples = {
        loc: np.concatenate(vals) if vals else np.array([])
        for loc, vals in ratio_samples.items()
    }

    actual_zero_share = (a_hist <= action_tol).mean(axis=(1, 2))
    actual_eta_share = (a_hist >= (eta_hist[:, None, :] - action_tol)).mean(axis=(1, 2))
    actual_interior_share = np.clip(1.0 - actual_zero_share - actual_eta_share, 0.0, 1.0)

    summary = pd.DataFrame({
        "metric": [
            "mean_no_inventory_share",
            "mean_negative_grad_share",
            "mean_positive_but_below_threshold_share",
            "mean_interior_region_share",
            "mean_actual_zero_share",
            "mean_actual_eta_share",
            "mean_actual_interior_share",
            "mean_inventory_binding_share",
            "mean_HF_upper_region_share",
            "mean_CHA1_upper_region_share",
        ],
        "value": [
            float(np.nanmean(no_inventory_share)),
            float(np.nanmean(negative_grad_share)),
            float(np.nanmean(fairness_upper_share)),
            float(np.nanmean(interior_region_share)),
            float(np.nanmean(actual_zero_share)),
            float(np.nanmean(actual_eta_share)),
            float(np.nanmean(actual_interior_share)),
            float(np.nanmean(inventory_binding_share)),
            float(np.nanmean(upper_region_share_by_loc[:, 0])),
            float(np.nanmean(upper_region_share_by_loc[:, 1])),
        ],
    })

    return {
        "threshold": threshold,
        "no_inventory_share": no_inventory_share,
        "negative_grad_share": negative_grad_share,
        "fairness_upper_share": fairness_upper_share,
        "interior_region_share": interior_region_share,
        "inventory_binding_share": inventory_binding_share,
        "upper_region_share_by_loc": upper_region_share_by_loc,
        "actual_zero_share": actual_zero_share,
        "actual_eta_share": actual_eta_share,
        "actual_interior_share": actual_interior_share,
        "actual_eta_share_by_loc": actual_eta_share_by_loc,
        "actual_zero_share_by_loc": actual_zero_share_by_loc,
        "ratio_samples": ratio_samples,
        "median_ratio_by_loc": median_ratio_by_loc,
        "summary": summary,
    }


threshold_diag = compute_deep_splitting_threshold_diagnostics(
    extended_results[DIAG_POLICY], z_net_list, bsde, device,
    action_tol=ACTION_TOL,
    inventory_tol=INVENTORY_TOL,
    ratio_sample_per_step=RATIO_SAMPLE_PER_STEP,
    eps=EPS,
)

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

if bsde.lambda_param > 0:
    hf_samples = threshold_diag["ratio_samples"][0]
    cha1_samples = threshold_diag["ratio_samples"][1]
    if hf_samples.size > 0 or cha1_samples.size > 0:
        combined = np.concatenate([x for x in [hf_samples, cha1_samples] if x.size > 0])
        lo, hi = np.quantile(combined, [0.01, 0.99])
        if hf_samples.size > 0:
            axes[0].hist(np.clip(hf_samples, lo, hi), bins=50, alpha=0.6, label="HF", color="tab:blue")
        if cha1_samples.size > 0:
            axes[0].hist(np.clip(cha1_samples, lo, hi), bins=50, alpha=0.6, label="CHA 1", color="tab:orange")
        axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        axes[0].axvline(1.0, color="tab:red", linestyle=":", linewidth=1.2)
        axes[0].set_title(r"Deep Splitting: ratio $\mathcal{D}_k V/\tau_k(t)$ for HF and CHA 1")
        axes[0].set_xlabel("Ratio to threshold")
        axes[0].set_ylabel("Count")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No ratio samples available", ha="center", va="center")
        axes[0].set_axis_off()
else:
    axes[0].text(0.5, 0.5, "lambda = 0: threshold ratio is not defined", ha="center", va="center")
    axes[0].set_axis_off()

axes[1].stackplot(
    control_t,
    threshold_diag["no_inventory_share"],
    threshold_diag["negative_grad_share"],
    threshold_diag["fairness_upper_share"],
    threshold_diag["interior_region_share"],
    labels=[
        "no central inventory",
        r"$\mathcal{D}_k V \leq 0$",
        r"$0 < \mathcal{D}_k V \leq \tau_k(t)$",
        r"$\mathcal{D}_k V > \tau_k(t)$",
    ],
    alpha=0.85,
)
axes[1].set_title("Deep Splitting: theoretical regimes over time")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Fraction")
axes[1].set_ylim(0.0, 1.0)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8, loc="upper right")

axes[2].plot(control_t, threshold_diag["actual_eta_share_by_loc"][:, 0], label="HF actual share near eta", linewidth=2)
axes[2].plot(control_t, threshold_diag["upper_region_share_by_loc"][:, 0], label="HF threshold-implied upper region", linestyle="--")
axes[2].plot(control_t, threshold_diag["actual_eta_share_by_loc"][:, 1], label="CHA 1 actual share near eta", linewidth=2)
axes[2].plot(control_t, threshold_diag["upper_region_share_by_loc"][:, 1], label="CHA 1 threshold-implied upper region", linestyle="--")
axes[2].plot(control_t, threshold_diag["inventory_binding_share"], label="inventory-binding share", linestyle=":", color="black")
axes[2].set_title("Deep Splitting: actual vs threshold-implied saturation")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Fraction")
axes[2].set_ylim(0.0, 1.0)
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=8)

plt.tight_layout()

out_path = os.path.join(FIG_DIR, "deep_splitting_threshold_diagnostics.png")
plt.savefig(out_path)
plt.show()
print("Saved:", out_path)

print(threshold_diag["summary"].to_string(index=False))


# In[ ]:


def compute_boundary_feasibility_diagnostics(res, bsde, action_tol=1e-6, inventory_tol=1e-6):
    a_hist = res["a_hist"]
    a_raw_hist = res.get("a_raw_hist", a_hist)
    zero_inventory_hist = res.get("zero_inventory_hist", res["i_hist"][:-1] <= inventory_tol)
    inventory_cap_active_hist = res.get("inventory_cap_active_hist")

    raw_ship = a_raw_hist.sum(axis=2) * bsde.delta_t
    projected_ship = a_hist.sum(axis=2) * bsde.delta_t
    zero_inventory_hist = zero_inventory_hist.astype(bool)
    positive_inventory_hist = ~zero_inventory_hist

    if inventory_cap_active_hist is None:
        inventory_cap_active_hist = np.abs(raw_ship - projected_ship) > action_tol
    else:
        inventory_cap_active_hist = inventory_cap_active_hist.astype(bool)

    zero_step_count = max(int(zero_inventory_hist.sum()), 1)
    positive_step_count = max(int(positive_inventory_hist.sum()), 1)
    raw_positive_at_zero = zero_inventory_hist & (raw_ship > action_tol)
    projected_positive_at_zero = zero_inventory_hist & (projected_ship > action_tol)

    return {
        "zero_inventory_share": float(zero_inventory_hist.mean()),
        "zero_inventory_positive_action_rate_raw": float(raw_positive_at_zero.sum() / zero_step_count),
        "zero_inventory_positive_action_rate_projected": float(projected_positive_at_zero.sum() / zero_step_count),
        "mean_raw_shipment_at_zero_inventory": float(raw_ship[zero_inventory_hist].mean()) if zero_inventory_hist.any() else 0.0,
        "max_raw_shipment_at_zero_inventory": float(raw_ship[zero_inventory_hist].max()) if zero_inventory_hist.any() else 0.0,
        "inventory_cap_trigger_rate": float(inventory_cap_active_hist.mean()),
        "inventory_cap_trigger_rate_positive_inventory": float(
            inventory_cap_active_hist[positive_inventory_hist].sum() / positive_step_count
        ),
    }


boundary_table = {}
for name, res in extended_results.items():
    boundary_table[name] = compute_boundary_feasibility_diagnostics(
        res,
        bsde,
        action_tol=ACTION_TOL,
        inventory_tol=INVENTORY_TOL,
    )

boundary_summary = pd.DataFrame(boundary_table).T
print(boundary_summary.to_string())

BOUNDARY_RESULTS_CSV = os.path.join(BASE, "boundary_feasibility_summary.csv")
boundary_summary.to_csv(BOUNDARY_RESULTS_CSV, index=True)
print(f"Saved: {BOUNDARY_RESULTS_CSV}")


# In[ ]:


# Efficiency / fairness / seasonality / inventory-reaction metrics aligned with main.tex

def get_result_dt(res):
    return float(res.get("time_dt", bsde.delta_t))


def compute_pathwise_coverage(res, eps=1e-12):
    l_hist = res["l_hist"]  # (T, P, K+1)
    d_hist = res["d_hist"]  # (T, P, K+1)
    p = bsde.p.detach().cpu().numpy()  # (K,)

    uncovered_per_path = l_hist[:, :, 0].sum(axis=0) + (
        l_hist[:, :, 1:] * (1.0 - p)[None, None, :]
    ).sum(axis=(0, 2))
    realized_demand_per_path = d_hist.sum(axis=(0, 2))
    coverage_per_path = 1.0 - uncovered_per_path / np.maximum(realized_demand_per_path, eps)
    return {
        "mean": float(np.nanmean(coverage_per_path)),
        "per_path": coverage_per_path,
    }


def compute_pathwise_allocation_dispersion(res, eps=1e-12):
    a_hist = res["a_hist"]  # (T, P, K+1)
    dt_hist = get_result_dt(res)
    return allocation_concentration(a_hist, dt=dt_hist, eps=eps)


def compute_demand_adjusted_fairness(res, eps=1e-12):
    return demand_adjusted_fairness_diagnostics(
        res["a_hist"],
        res["mu_hist"],
        eta_hist=res.get("eta_hist"),
        dt=get_result_dt(res),
        eps=eps,
    )


def compute_macro_seasonality_alignment(res, eps=1e-12):
    a_hist = res["a_hist"]  # (T, P, K+1)
    mu_hist = res["mu_hist"]  # (T, K+1)
    a_mean_t = a_hist.mean(axis=1)

    corrs = []
    for k in range(a_mean_t.shape[1]):
        x = a_mean_t[:, k]
        y = mu_hist[:, k]
        if np.std(x) < eps or np.std(y) < eps:
            corrs.append(np.nan)
        else:
            corrs.append(np.corrcoef(x, y)[0, 1])

    return {"mean_corr": float(np.nanmean(corrs)), "per_location_corr": corrs}


def compute_pooled_inventory_reaction(res, eps=1e-12):
    a_hist = res["a_hist"]  # (T, P, K+1)
    z_hist = res["z_hist"]  # (T+1, P, K+1)
    i_hist = res["i_hist"]  # (T+1, P)

    T_steps, P, K = a_hist.shape
    a_flat = a_hist.reshape(T_steps * P, K)
    z_flat = z_hist[:-1].reshape(T_steps * P, K)
    i_flat = i_hist[:-1].reshape(T_steps * P)

    corr_az = []
    for k in range(K):
        x = a_flat[:, k]
        y = z_flat[:, k]
        if np.std(x) < eps or np.std(y) < eps:
            corr_az.append(np.nan)
        else:
            corr_az.append(np.corrcoef(x, y)[0, 1])

    total_a = a_flat.sum(axis=1)
    if np.std(total_a) < eps or np.std(i_flat) < eps:
        corr_aI = np.nan
    else:
        corr_aI = np.corrcoef(total_a, i_flat)[0, 1]

    return {
        "corr_aZ_mean": float(np.nanmean(corr_az)),
        "corr_aZ_per_loc": corr_az,
        "corr_aI": float(corr_aI) if not np.isnan(corr_aI) else np.nan,
    }


def build_auxiliary_metric_table(results_by_policy):
    coverage_table = {}
    fairness_table = {}
    demand_fairness_table = {}
    seasonality_table = {}
    inventory_table = {}

    for name, res in results_by_policy.items():
        coverage_table[name] = compute_pathwise_coverage(res)
        fairness_table[name] = compute_pathwise_allocation_dispersion(res)
        demand_fairness_table[name] = compute_demand_adjusted_fairness(res)
        seasonality_table[name] = compute_macro_seasonality_alignment(res)
        inventory_table[name] = compute_pooled_inventory_reaction(res)

    summary = pd.DataFrame({
        "Coverage": {k: v["mean"] for k, v in coverage_table.items()},
        "Allocation_Concentration_CV": {k: v["cv"] for k, v in fairness_table.items()},
        "Allocation_Concentration_Gini": {k: v["gini"] for k, v in fairness_table.items()},
        "DemandWeighted_Mean_f": {k: v["demand_weighted_mean_f"] for k, v in demand_fairness_table.items()},
        "DemandWeighted_CV_f": {k: v["demand_weighted_cv_f"] for k, v in demand_fairness_table.items()},
        "Coverage_Ratio_Gap": {k: v["planned_coverage_ratio_gap"] for k, v in demand_fairness_table.items()},
        "DemandProportional_RMSE": {k: v["demand_proportional_benchmark_rmse"] for k, v in demand_fairness_table.items()},
        "Action_Zero_Fraction": {k: v["action_zero_fraction"] for k, v in demand_fairness_table.items()},
        "Action_Interior_Fraction": {k: v["action_interior_fraction"] for k, v in demand_fairness_table.items()},
        "Action_EtaCap_Fraction": {k: v["action_eta_cap_fraction"] for k, v in demand_fairness_table.items()},
        "Seasonality_Corr": {k: v["mean_corr"] for k, v in seasonality_table.items()},
        "Inventory_Corr_aZ": {k: v["corr_aZ_mean"] for k, v in inventory_table.items()},
        "Inventory_Corr_aI": {k: v["corr_aI"] for k, v in inventory_table.items()},
    })
    return summary, {
        "coverage": coverage_table,
        "allocation_concentration": fairness_table,
        "demand_adjusted_fairness": demand_fairness_table,
        "seasonality": seasonality_table,
        "inventory": inventory_table,
    }


fine_aux_summary, fine_aux_details = build_auxiliary_metric_table(extended_results)
periodic_aux_summary = periodic_operational_summary
periodic_aux_details = {"operational": periodic_operational_summary}

print("Fine-grid auxiliary metrics")
print(fine_aux_summary.to_string())
print("Discrete-review auxiliary metrics")
print(periodic_aux_summary.to_string())


# In[ ]:


# Optional: quick tables to export
import pandas as pd

extended_summary = fine_aux_summary.copy()
extended_summary["Zero_Inventory_Share"] = {k: v["zero_inventory_share"] for k, v in boundary_table.items()}
extended_summary["Zero_Inventory_Positive_Action_Rate_Raw"] = {
    k: v["zero_inventory_positive_action_rate_raw"] for k, v in boundary_table.items()
}
extended_summary["Inventory_Cap_Trigger_Rate"] = {
    k: v["inventory_cap_trigger_rate"] for k, v in boundary_table.items()
}

print(extended_summary.to_string())


# In[ ]:


# Save summary tables
FINE_AUX_RESULTS_CSV = os.path.join(BASE, "fine_grid_auxiliary_metrics.csv")
fine_aux_summary.to_csv(FINE_AUX_RESULTS_CSV, index=True)
print(f"Saved: {FINE_AUX_RESULTS_CSV}")

PERIODIC_AUX_RESULTS_CSV = os.path.join(BASE, tagged_csv_name("discrete_auxiliary_metrics.csv"))
periodic_aux_summary.to_csv(PERIODIC_AUX_RESULTS_CSV, index=True)
print(f"Saved: {PERIODIC_AUX_RESULTS_CSV}")

EXT_RESULTS_CSV = os.path.join(BASE, "extended_evaluation_results.csv")
extended_summary.to_csv(EXT_RESULTS_CSV, index=True)
print(f"Saved: {EXT_RESULTS_CSV}")

# In[ ]:

# Stylized Visualizations (Policy Heatmap, Trajectory, Shadow Prices)
print("Generating Stylized Visualizations (Heatmap, Trajectory, Shadow Prices)...")

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Read the model dimensions.
N = bsde.num_time_interval
dim = bsde.dim

# ---------------------------------------------------------
# Policy heatmaps for a0, a1 and a2 on a fixed state slice.
# ---------------------------------------------------------
print("Generating Plot 1: Combined Policy Heatmaps (a0, a1, a2)...")
t_idx = N//2
print(f"t_idx: {t_idx}")
I_fixed = 6.0  # Warehouse inventory for this illustrative state slice.
Z0_fixed = 4.0  # Fixed HF inventory.
print(f"I={I_fixed}, Z0={Z0_fixed}")

# Construct the Z1 and Z2 plotting grid.
z1_vals = np.linspace(0, 5, 100)
z2_vals = np.linspace(0, 5, 100)
Z1_grid, Z2_grid = np.meshgrid(z1_vals, z2_vals)

grid_size = Z1_grid.size
states_tensor = torch.zeros((grid_size, dim + 1), device=device, dtype=torch.float32)
states_tensor[:, 0] = Z0_fixed
if dim > 1:
    states_tensor[:, 1] = torch.tensor(Z1_grid.flatten(), device=device, dtype=torch.float32)
if dim > 2:
    states_tensor[:, 2] = torch.tensor(Z2_grid.flatten(), device=device, dtype=torch.float32)
states_tensor[:, -1] = I_fixed  

ds_policy_fn = policies[DS_POLICY_NAME]
with torch.no_grad():
    a_rates = ds_policy_fn(t_idx, states_tensor[:, :-1], states_tensor[:, -1])
    a0_rates = a_rates[:, 0].cpu().numpy().reshape(100, 100) if dim > 0 else np.zeros((100, 100))
    a1_rates = a_rates[:, 1].cpu().numpy().reshape(100, 100) if dim > 1 else np.zeros((100, 100))
    a2_rates = a_rates[:, 2].cpu().numpy().reshape(100, 100) if dim > 2 else np.zeros((100, 100))

# Each panel uses its own color scale; shared limits can be set below.
# a_max = float(bsde.eta.max().item())

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)

cp0 = axes[0].contourf(Z1_grid, Z2_grid, a0_rates, levels=30, cmap="Blues")
fig.colorbar(cp0, ax=axes[0], label=r"Supply rate to HF ($a_0$)")
axes[0].set_xlabel(r"CHA 1 inventory ($Z_1$)")
axes[0].set_ylabel(r"CHA 2 inventory ($Z_2$)")

if dim > 1:
    cp1 = axes[1].contourf(Z1_grid, Z2_grid, a1_rates, levels=30, cmap="Reds")
    fig.colorbar(cp1, ax=axes[1], label=r"Supply rate to CHA 1 ($a_1$)")
    axes[1].set_xlabel(r"CHA 1 inventory ($Z_1$)")
else:
    axes[1].set_visible(False)

if dim > 2:
    cp2 = axes[2].contourf(Z1_grid, Z2_grid, a2_rates, levels=30, cmap="Greens")
    fig.colorbar(cp2, ax=axes[2], label=r"Supply rate to CHA 2 ($a_2$)")
    axes[2].set_xlabel(r"CHA 1 inventory ($Z_1$)")
else:
    axes[2].set_visible(False)

plt.tight_layout()
plot1_path = os.path.join(FIG_DIR, "plot1_policy_heatmaps_a0_a1_a2.png")
plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved:", plot1_path)

# In[ ]:


# ---------------------------------------------------------
# Inventory trajectory for the path with the largest shortage at the selected location.
# ---------------------------------------------------------
print("Generating Plot 2: Single Path Trajectory...")
l_hist_ds = extended_results[DS_POLICY_NAME]["l_hist"]

# Select the path with the largest cumulative shortage at the selected location under Deep Splitting.
target_idx = 1 if dim > 1 else 0
cha1_shortage_sum = l_hist_ds[:, :, target_idx].sum(axis=0) 
worst_path_idx = np.argmax(cha1_shortage_sum)

time_axis_states = np.arange(N + 1) * bsde.delta_t

plt.figure(figsize=(8, 4))
# Plot warehouse inventory under Deep Splitting.
i_hist_ds_single = extended_results[DS_POLICY_NAME]["i_hist"][:, worst_path_idx]
plt.plot(time_axis_states, i_hist_ds_single, label=f"{policy_label(DS_POLICY_NAME)} (Dynamic Rationing)", color='tab:blue', linewidth=2.5)

# Plot the Static policy on the same demand path.
if "Static" in extended_results:
    i_hist_static_single = extended_results["Static"]["i_hist"][:, worst_path_idx]
    plt.plot(time_axis_states, i_hist_static_single, label="Static Proportional", color='tab:orange', linestyle='--', linewidth=2.5)

plt.xlabel("Time")
plt.ylabel("Central Inventory $I(t)$")
plt.title(f"Central Inventory Depletion on a Stressed Sample Path (Index: {worst_path_idx})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
plot2_path = os.path.join(FIG_DIR, "plot2_single_path_inventory.png")
plt.savefig(plot2_path, dpi=150)
plt.close()
print("Saved:", plot2_path)


# ---------------------------------------------------------
# Control-relevant gradient differences along the selected path.
# ---------------------------------------------------------
print("Generating Plot 3: Shadow Prices Evolution...")
z_path = extended_results[DS_POLICY_NAME]["z_hist"][:-1, worst_path_idx, :] # Shape: (N, dim)
i_path = extended_results[DS_POLICY_NAME]["i_hist"][:-1, worst_path_idx]    # Shape: (N,)

shadow_prices_cha1 = np.zeros(N)
shadow_prices_cha2 = np.zeros(N)

with torch.no_grad():
    for n in range(N):
        state_n = np.concatenate([z_path[n], [i_path[n]]])
        state_tensor = torch.tensor(state_n, device=device, dtype=torch.float32).unsqueeze(0)
        
        # Evaluate the control-relevant gradient-difference network.
        grad_diff_n = z_net_list[min(n, len(z_net_list) - 1)](state_tensor)
        
        if dim > 1:
            shadow_prices_cha1[n] = grad_diff_n[0, 1].item()
        if dim > 2:
            shadow_prices_cha2[n] = grad_diff_n[0, 2].item()

time_axis_controls = np.arange(N) * bsde.delta_t

plt.figure(figsize=(8, 4))
if dim > 1:
    plt.plot(time_axis_controls, shadow_prices_cha1, label=r"Control-relevant cost for CHA 1 ($D_1 V$)", color='tab:red', linewidth=2)
if dim > 2:
    plt.plot(time_axis_controls, shadow_prices_cha2, label=r"Control-relevant cost for CHA 2 ($D_2 V$)", color='tab:green', linewidth=2)

plt.xlabel("Time")
plt.ylabel("Gradient Difference / Opportunity Cost")
plt.title(f"Evolution of Control-Relevant Gradient Differences along Stressed Path {worst_path_idx}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
plot3_path = os.path.join(FIG_DIR, "plot3_shadow_prices.png")
plt.savefig(plot3_path, dpi=150)
plt.close()
print("Saved:", plot3_path)


# %%
