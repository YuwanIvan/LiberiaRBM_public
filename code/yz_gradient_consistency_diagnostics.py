#!/usr/bin/env python
import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import equation as eqn
import evaluate as eval_mod
from Net import GradDiffNet, Y_Net


SUMMARY_FILE = "yz_consistency_summary.csv"
COMPONENTWISE_FILE = "yz_componentwise_metrics.csv"
CONTROL_FILE = "yz_control_difference_metrics.csv"
RAW_SIGN_SUMMARY_FILE = "yz_grad_diff_sign_summary.csv"
STATES_FILE = "yz_sampled_states.csv"
NOTE_FILE = "yz_consistency_note.md"
HEATMAP_FILE = "yz_component_correlation_heatmap.png"
BARPLOT_FILE = "yz_regionwise_barplots.png"
SCATTER_FILE_TEMPLATE = "yz_control_difference_scatter_interval_{interval_idx}.png"
RAW_SCATTER_FILE_TEMPLATE = "yz_grad_diff_scatter_interval_{interval_idx}.png"
RAW_SCATTER_REGION_FILE_TEMPLATE = "yz_grad_diff_scatter_interval_{interval_idx}_region_{region_name}.png"
MATCHED_BOUNDARY_CONTROL_SCATTER_FILE_TEMPLATE = "yz_matched_boundary_control_scatter_interval_{interval_idx}.png"
MATCHED_BOUNDARY_SUMMARY_FILE = "yz_matched_boundary_summary.png"


@dataclass
class ZInterpretation:
    name: str
    description: str
    comparison_target_name: str
    code_references: List[str]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose consistency between trained Y_Net and Z_Net checkpoints."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Experiment directory containing config_used.json, params/, and logs/.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. cpu, cuda, or cuda:0. Defaults to auto.",
    )
    parser.add_argument(
        "--num_states",
        type=int,
        default=4000,
        help="Held-out diagnostic states per interval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible state sampling.",
    )
    parser.add_argument(
        "--interval_indices",
        type=str,
        default=None,
        help="Comma-separated interval indices, e.g. 0,199,399. If omitted, representative checkpoints are inferred.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for outputs. Defaults to the experiment directory.",
    )
    parser.add_argument(
        "--eps_local",
        type=float,
        default=1.0,
        help="Threshold for local boundary labels Z_k <= eps_local.",
    )
    parser.add_argument(
        "--eps_central",
        type=float,
        default=1.0,
        help="Threshold for central boundary label I <= eps_central.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
        help="Near-threshold band for control-sign diagnostics based on |delta_from_Y| <= tau.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for forward/autodiff passes.",
    )
    parser.add_argument(
        "--plot_mode",
        type=str,
        choices=["minimal", "full"],
        default="minimal",
        help="Plotting level. 'minimal' keeps the main diagnostic figures; 'full' preserves the richer plot set.",
    )
    parser.add_argument(
        "--include_terminal_interval_plots",
        action="store_true",
        help="In minimal mode, also generate plots for the terminal interval if it was requested.",
    )
    return parser.parse_args()


def resolve_device(device_arg: Optional[str]):
    if device_arg:
        return torch.device(device_arg)
    return eval_mod.get_device()


def parse_interval_indices(interval_indices_arg: Optional[str], logs_dir: str) -> List[int]:
    if interval_indices_arg:
        items = [item.strip() for item in interval_indices_arg.split(",") if item.strip()]
        if not items:
            raise ValueError("--interval_indices was provided but no indices were parsed.")
        return sorted({int(item) for item in items})

    pattern = re.compile(r"^y_network(\d+)\.pth$")
    y_indices = []
    z_indices = set()
    for name in os.listdir(logs_dir):
        y_match = pattern.match(name)
        if y_match:
            y_indices.append(int(y_match.group(1)))
        z_match = re.match(r"^z_network(\d+)\.pth$", name)
        if z_match:
            z_indices.add(int(z_match.group(1)))

    shared = sorted(idx for idx in y_indices if idx in z_indices)
    if not shared:
        raise FileNotFoundError(f"No matching y/z checkpoints found in {logs_dir}")
    if len(shared) <= 3:
        return shared
    mid = shared[len(shared) // 2]
    return [shared[0], mid, shared[-1]]


def resolve_paths(args):
    experiment_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    config_path = os.path.join(experiment_dir, "config_used.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config_used.json not found in {experiment_dir}")

    logs_dir = os.path.join(experiment_dir, "logs")
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"logs directory not found: {logs_dir}")

    output_dir = os.path.abspath(args.output_dir or experiment_dir)
    os.makedirs(output_dir, exist_ok=True)

    return experiment_dir, config_path, logs_dir, output_dir


def load_context(config_path: str, device):
    config = eval_mod.load_config_from_path(config_path)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    bsde = eval_mod.move_bsde_to_device(bsde, device)
    return config, bsde


def infer_z_interpretation() -> ZInterpretation:
    return ZInterpretation(
        name="control_relevant_gradient_difference_surrogate",
        description=(
            "Z_Net is treated as a learned surrogate for the control-relevant gradient differences "
            "D_kV = dV/dz_k - dV/dI, i.e. a K+1 vector indexed by location."
        ),
        comparison_target_name="autodiff_gradient_differences_from_Y_Net",
        code_references=[
            "solver.DSPL_solver.loss_fn",
            "equation.LiberiaHJB.F_function",
            "evaluate.create_deep_splitting_policy",
        ],
    )


def build_networks(config, state_dim: int, device):
    g_output_bias = str(getattr(config.net_config, "g_output_bias", False)).strip().lower() in {"1", "true", "yes", "on"}
    y_net = Y_Net(
        state_dim,
        config.net_config.num_neurons,
        config.net_config.num_layers,
        config.net_config.activation,
        config.net_config.slope,
    ).to(device)
    z_net = GradDiffNet(
        state_dim,
        config.eqn_config.dim,
        config.net_config.num_neurons,
        config.net_config.num_layers,
        config.net_config.activation,
        config.net_config.slope,
        output_bias=g_output_bias,
    ).to(device)
    return y_net, z_net


def load_interval_networks(config, logs_dir: str, interval_idx: int, device, state_dim: int):
    y_path = os.path.join(logs_dir, f"y_network{interval_idx}.pth")
    z_path = os.path.join(logs_dir, f"z_network{interval_idx}.pth")
    if not os.path.isfile(y_path):
        raise FileNotFoundError(f"Missing checkpoint: {y_path}")
    if not os.path.isfile(z_path):
        raise FileNotFoundError(f"Missing checkpoint: {z_path}")

    y_net, z_net = build_networks(config, state_dim, device)
    y_net.load_state_dict(torch.load(y_path, map_location=device))
    z_net.load_state_dict(torch.load(z_path, map_location=device))
    y_net.eval()
    z_net.eval()
    return y_net, z_net


def state_bounds_for_interval(bsde, interval_idx: int):
    remaining_time = (bsde.num_time_interval - int(interval_idx)) * float(bsde.delta_t)
    mu_t = bsde.mu[interval_idx, :]
    mu_max = float(torch.max(mu_t).item())
    z_max = max(mu_max * remaining_time * 2.0, 50.0)
    return float(z_max), float(bsde.I0)


def allocate_counts(num_states: int, dim: int):
    broad = max(1, int(round(0.4 * num_states)))
    central = max(1, int(round(0.2 * num_states)))
    remaining = max(num_states - broad - central, dim)
    local_counts = [remaining // dim] * dim
    for idx in range(remaining % dim):
        local_counts[idx] += 1
    return broad, central, local_counts


def sample_interval_states(
    bsde,
    interval_idx: int,
    num_states: int,
    rng: np.random.Generator,
    eps_local: float,
    eps_central: float,
):
    dim = int(bsde.dim)
    z_max, i_max = state_bounds_for_interval(bsde, interval_idx)
    broad_count, central_count, local_counts = allocate_counts(num_states, dim)

    rows = []
    state_id = 0

    def append_block(block_size: int, source_label: str, local_boundary_idx: Optional[int] = None, central_boundary: bool = False):
        nonlocal state_id
        if block_size <= 0:
            return
        Z = rng.uniform(0.0, z_max, size=(block_size, dim))
        I = rng.uniform(0.0, i_max, size=(block_size,))

        if local_boundary_idx is not None:
            Z[:, local_boundary_idx] = rng.uniform(0.0, eps_local, size=(block_size,))
        if central_boundary:
            I = rng.uniform(0.0, eps_central, size=(block_size,))

        for row_idx in range(block_size):
            z_row = Z[row_idx]
            i_val = I[row_idx]
            record = {
                "interval_idx": int(interval_idx),
                "state_id": int(state_id),
                "sample_source": source_label,
                "I": float(i_val),
                "region_all": 1,
                "region_central_boundary": int(i_val <= eps_central),
            }
            local_boundary_flags = []
            for k in range(dim):
                record[f"Z{k}"] = float(z_row[k])
                is_boundary = int(z_row[k] <= eps_local)
                record[f"region_local_boundary_{k}"] = is_boundary
                local_boundary_flags.append(is_boundary)
            record["region_interior"] = int((record["region_central_boundary"] == 0) and all(flag == 0 for flag in local_boundary_flags))
            rows.append(record)
            state_id += 1

    append_block(broad_count, "broad_global")
    append_block(central_count, "central_boundary_oversample", central_boundary=True)
    for k, count in enumerate(local_counts):
        append_block(count, f"local_boundary_{k}_oversample", local_boundary_idx=k)

    return pd.DataFrame(rows)


def sampled_state_columns(dim: int):
    return [f"Z{k}" for k in range(dim)] + ["I"]


def raw_component_names(dim: int):
    return [f"D_{k}V" for k in range(dim)]


def plot_interval_indices(interval_indices: Sequence[int], final_interval_idx: int, plot_mode: str, include_terminal_interval_plots: bool):
    if plot_mode != "minimal" or include_terminal_interval_plots:
        return list(interval_indices)
    return [interval_idx for interval_idx in interval_indices if interval_idx != final_interval_idx]


def region_masks(state_df: pd.DataFrame, dim: int):
    masks = {"all": np.ones(len(state_df), dtype=bool)}
    for k in range(dim):
        masks[f"local_boundary_{k}"] = state_df[f"region_local_boundary_{k}"].to_numpy(dtype=bool)
    masks["central_boundary"] = state_df["region_central_boundary"].to_numpy(dtype=bool)
    masks["interior"] = state_df["region_interior"].to_numpy(dtype=bool)
    return masks


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size <= 1:
        return float("nan")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_networks_on_states(y_net, z_net, state_array: np.ndarray, device, batch_size: int):
    z_preds = []
    grad_targets = []
    num_states = state_array.shape[0]

    for start in range(0, num_states, batch_size):
        end = min(start + batch_size, num_states)
        x = torch.tensor(state_array[start:end], dtype=torch.float32, device=device, requires_grad=True)
        y_val = y_net(x)
        grad = torch.autograd.grad(y_val.sum(), x, create_graph=False, retain_graph=False)[0]
        z_pred = z_net(x)
        grad_diff = grad[:, : z_pred.shape[1]] - grad[:, -1:].expand_as(z_pred)

        z_preds.append(z_pred.detach().cpu().numpy())
        grad_targets.append(grad_diff.detach().cpu().numpy())

    return np.vstack(z_preds), np.vstack(grad_targets)


def cosine_similarity_per_state(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    denom = pred_norm * target_norm
    cosine = np.full(pred.shape[0], np.nan, dtype=np.float64)
    mask = denom > 1e-12
    cosine[mask] = np.sum(pred[mask] * target[mask], axis=1) / denom[mask]
    return cosine


def compute_raw_metrics(pred: np.ndarray, target: np.ndarray):
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    cosine = cosine_similarity_per_state(pred, target)
    mean_cosine = float(np.nanmean(cosine)) if np.any(~np.isnan(cosine)) else float("nan")
    return {
        "mse": mse,
        "mae": mae,
        "mean_cosine_similarity": mean_cosine,
    }


def summarize_raw_region(interval_idx: int, region_name: str, pred: np.ndarray, target: np.ndarray):
    metrics = compute_raw_metrics(pred, target)
    component_corrs = [safe_corr(pred[:, j], target[:, j]) for j in range(pred.shape[1])]
    metrics["mean_component_correlation"] = float(np.nanmean(component_corrs)) if np.any(~np.isnan(component_corrs)) else float("nan")
    metrics["min_component_correlation"] = float(np.nanmin(component_corrs)) if np.any(~np.isnan(component_corrs)) else float("nan")

    rows = []
    for metric_name, value in metrics.items():
        rows.append(
            {
                "interval_idx": int(interval_idx),
                "region": region_name,
                "group": "raw_vector",
                "metric": metric_name,
                "value": value,
                "num_states": int(pred.shape[0]),
            }
        )
    return rows


def componentwise_rows(interval_idx: int, region_name: str, pred: np.ndarray, target: np.ndarray, dim: int):
    rows = []
    names = raw_component_names(dim)
    for j, name in enumerate(names):
        diff = pred[:, j] - target[:, j]
        rows.append(
            {
                "interval_idx": int(interval_idx),
                "region": region_name,
                "component_index": int(j),
                "component_name": name,
                "mse": float(np.mean(diff ** 2)),
                "mae": float(np.mean(np.abs(diff))),
                "correlation": safe_corr(pred[:, j], target[:, j]),
                "pred_mean": float(np.mean(pred[:, j])),
                "target_mean": float(np.mean(target[:, j])),
                "pred_std": float(np.std(pred[:, j])),
                "target_std": float(np.std(target[:, j])),
                "num_states": int(pred.shape[0]),
            }
        )
    return rows


def control_difference_rows(interval_idx: int, region_name: str, pred: np.ndarray, target: np.ndarray, dim: int, tau: float):
    rows = []
    summary_accumulator = []
    for k in range(dim):
        delta_pred = pred[:, k]
        delta_target = target[:, k]
        diff = delta_pred - delta_target

        sign_pred = np.sign(delta_pred)
        sign_target = np.sign(delta_target)
        sign_mismatch = sign_pred != sign_target
        near_threshold_mask = np.abs(delta_target) <= tau
        near_threshold_mismatch = sign_mismatch[near_threshold_mask] if np.any(near_threshold_mask) else np.array([], dtype=bool)

        row = {
            "interval_idx": int(interval_idx),
            "region": region_name,
            "delta_index": int(k),
            "delta_name": f"D_{k}V",
            "mse": float(np.mean(diff ** 2)),
            "mae": float(np.mean(np.abs(diff))),
            "correlation": safe_corr(delta_pred, delta_target),
            "sign_mismatch_rate": float(np.mean(sign_mismatch)),
            "near_threshold_share": float(np.mean(near_threshold_mask)),
            "near_threshold_sign_mismatch_rate": (
                float(np.mean(near_threshold_mismatch)) if near_threshold_mismatch.size > 0 else float("nan")
            ),
            "pred_mean": float(np.mean(delta_pred)),
            "target_mean": float(np.mean(delta_target)),
            "pred_std": float(np.std(delta_pred)),
            "target_std": float(np.std(delta_target)),
            "num_states": int(pred.shape[0]),
        }
        rows.append(row)
        summary_accumulator.append(row)

    summary_rows = []
    for metric_name in [
        "mse",
        "mae",
        "correlation",
        "sign_mismatch_rate",
        "near_threshold_share",
        "near_threshold_sign_mismatch_rate",
    ]:
        values = np.array([row[metric_name] for row in summary_accumulator], dtype=np.float64)
        summary_rows.append(
            {
                "interval_idx": int(interval_idx),
                "region": region_name,
                "group": "control_difference_avg",
                "metric": f"mean_{metric_name}",
                "value": float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan"),
                "num_states": int(pred.shape[0]),
            }
        )

    return rows, summary_rows


def raw_gradient_sign_rows(interval_idx: int, region_name: str, pred: np.ndarray, target: np.ndarray, dim: int):
    rows = []
    for component_index, component_name in enumerate(raw_component_names(dim)):
        autodiff_vals = target[:, component_index]
        z_net_vals = pred[:, component_index]
        rows.append(
            {
                "interval_idx": int(interval_idx),
                "region": region_name,
                "component_index": int(component_index),
                "component_name": component_name,
                "autodiff_positive_fraction": float(np.mean(autodiff_vals > 0.0)),
                "autodiff_negative_fraction": float(np.mean(autodiff_vals < 0.0)),
                "z_net_positive_fraction": float(np.mean(z_net_vals > 0.0)),
                "z_net_negative_fraction": float(np.mean(z_net_vals < 0.0)),
                "autodiff_mean": float(np.mean(autodiff_vals)),
                "z_net_mean": float(np.mean(z_net_vals)),
                "autodiff_median": float(np.median(autodiff_vals)),
                "z_net_median": float(np.median(z_net_vals)),
                "num_states": int(pred.shape[0]),
            }
        )
    return rows


def save_states_csv(state_frames: Sequence[pd.DataFrame], output_dir: str):
    states_df = pd.concat(state_frames, ignore_index=True)
    path = os.path.join(output_dir, STATES_FILE)
    states_df.to_csv(path, index=False)
    return states_df, path


def plot_component_correlation_heatmap(component_df: pd.DataFrame, output_dir: str):
    all_region = component_df[component_df["region"] == "all"].copy()
    if all_region.empty:
        return None

    heatmap_df = all_region.pivot(index="interval_idx", columns="component_name", values="correlation").sort_index()
    fig, ax = plt.subplots(figsize=(1.5 * heatmap_df.shape[1] + 2, 1.2 * heatmap_df.shape[0] + 2))
    image = ax.imshow(heatmap_df.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(heatmap_df.shape[1]))
    ax.set_xticklabels(list(heatmap_df.columns), rotation=45, ha="right")
    ax.set_yticks(range(heatmap_df.shape[0]))
    ax.set_yticklabels([str(idx) for idx in heatmap_df.index])
    ax.set_xlabel("Component")
    ax.set_ylabel("Interval")
    ax.set_title("Y/Z componentwise correlation (region=all)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = os.path.join(output_dir, HEATMAP_FILE)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_regionwise_barplots(summary_df: pd.DataFrame, output_dir: str):
    target_metrics = [
        ("raw_vector", "mse"),
        ("raw_vector", "mean_component_correlation"),
        ("control_difference_avg", "mean_sign_mismatch_rate"),
        ("control_difference_avg", "mean_near_threshold_sign_mismatch_rate"),
    ]
    subset = summary_df[
        summary_df.apply(lambda row: (row["group"], row["metric"]) in target_metrics, axis=1)
        & (summary_df["region"] != "all")
    ].copy()
    if subset.empty:
        return None

    subset["metric_label"] = subset["group"] + ":" + subset["metric"]
    agg = (
        subset.groupby(["region", "metric_label"], as_index=False)["value"]
        .mean()
        .sort_values(["metric_label", "region"])
    )
    regions = sorted(agg["region"].unique())
    metric_labels = list(dict.fromkeys(agg["metric_label"].tolist()))

    x = np.arange(len(regions))
    width = 0.8 / max(len(metric_labels), 1)
    fig, ax = plt.subplots(figsize=(2.0 * len(regions) + 4, 5))
    for idx, metric_label in enumerate(metric_labels):
        metric_slice = agg[agg["metric_label"] == metric_label].set_index("region").reindex(regions)
        values = metric_slice["value"].to_numpy(dtype=float)
        ax.bar(x + (idx - (len(metric_labels) - 1) / 2.0) * width, values, width=width, label=metric_label)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_ylabel("Mean across checked intervals")
    ax.set_title("Region-wise Y/Z consistency diagnostics")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, BARPLOT_FILE)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_control_scatter(interval_idx: int, control_df: pd.DataFrame, cached_control: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]], output_dir: str):
    interval_df = control_df[(control_df["interval_idx"] == interval_idx) & (control_df["region"] == "all")]
    if interval_df.empty:
        return None

    delta_indices = sorted(interval_df["delta_index"].unique())
    fig, axes = plt.subplots(1, len(delta_indices), figsize=(5 * len(delta_indices), 4), squeeze=False)

    for axis, delta_idx in zip(axes[0], delta_indices):
        delta_pred, delta_target = cached_control[(interval_idx, delta_idx)]
        axis.scatter(delta_target, delta_pred, s=8, alpha=0.25)
        min_val = float(min(np.min(delta_target), np.min(delta_pred)))
        max_val = float(max(np.max(delta_target), np.max(delta_pred)))
        axis.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
        axis.set_xlabel("D_kV from autodiff(Y_Net)")
        axis.set_ylabel("D_kV from Z_Net")
        axis.set_title(f"Interval {interval_idx}, D_{delta_idx}V")

    fig.tight_layout()
    path = os.path.join(output_dir, SCATTER_FILE_TEMPLATE.format(interval_idx=interval_idx))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_raw_gradient_scatter(
    interval_idx: int,
    region_name: str,
    raw_pairs: Tuple[np.ndarray, np.ndarray],
    dim: int,
    output_dir: str,
):
    z_pred, grad_y = raw_pairs
    component_names = raw_component_names(dim)
    num_components = len(component_names)
    fig, axes = plt.subplots(1, num_components, figsize=(5 * num_components, 4), squeeze=False)

    for axis, component_idx, component_name in zip(axes[0], range(num_components), component_names):
        x_vals = grad_y[:, component_idx]
        y_vals = z_pred[:, component_idx]
        axis.scatter(x_vals, y_vals, s=8, alpha=0.25)
        min_val = float(min(np.min(x_vals), np.min(y_vals)))
        max_val = float(max(np.max(x_vals), np.max(y_vals)))
        axis.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
        axis.set_xlabel(f"{component_name} from autodiff(Y_Net)")
        axis.set_ylabel(f"{component_name} from Z_Net")
        axis.set_title(f"Interval {interval_idx}, {component_name}, region={region_name}")

    fig.tight_layout()
    if region_name == "all":
        path = os.path.join(output_dir, RAW_SCATTER_FILE_TEMPLATE.format(interval_idx=interval_idx))
    else:
        path = os.path.join(
            output_dir,
            RAW_SCATTER_REGION_FILE_TEMPLATE.format(interval_idx=interval_idx, region_name=region_name),
        )
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_matched_boundary_control_scatter(
    interval_idx: int,
    matched_control_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    dim: int,
    output_dir: str,
):
    matched_indices = [k for k in range(dim) if (interval_idx, k) in matched_control_cache]
    if not matched_indices:
        return None

    fig, axes = plt.subplots(1, len(matched_indices), figsize=(5 * len(matched_indices), 4), squeeze=False)
    for axis, k in zip(axes[0], matched_indices):
        delta_pred, delta_target = matched_control_cache[(interval_idx, k)]
        axis.scatter(delta_target, delta_pred, s=8, alpha=0.25)
        min_val = float(min(np.min(delta_target), np.min(delta_pred)))
        max_val = float(max(np.max(delta_target), np.max(delta_pred)))
        axis.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
        axis.set_xlabel(f"D_{k}V from autodiff(Y_Net)")
        axis.set_ylabel(f"D_{k}V from Z_Net")
        axis.set_title(f"local_boundary_{k} with D_{k}V")

    fig.tight_layout()
    path = os.path.join(output_dir, MATCHED_BOUNDARY_CONTROL_SCATTER_FILE_TEMPLATE.format(interval_idx=interval_idx))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_matched_boundary_summary(control_df: pd.DataFrame, interval_indices: Sequence[int], dim: int, output_dir: str):
    matched_rows = []
    for k in range(dim):
        region_name = f"local_boundary_{k}"
        matched_slice = control_df[
            (control_df["region"] == region_name) & (control_df["delta_index"] == k) & (control_df["interval_idx"].isin(interval_indices))
        ].copy()
        if matched_slice.empty:
            continue
        matched_slice["matched_boundary"] = f"{region_name}->D_{k}V"
        matched_rows.append(matched_slice)

    if not matched_rows:
        return None

    matched_df = pd.concat(matched_rows, ignore_index=True)
    metric_specs = [
        ("correlation", "Correlation"),
        ("sign_mismatch_rate", "Sign mismatch rate"),
        ("near_threshold_sign_mismatch_rate", "Near-threshold sign mismatch rate"),
    ]
    boundaries = [f"local_boundary_{k}->D_{k}V" for k in range(dim)]

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(5 * len(metric_specs), 1.2 * len(interval_indices) + 3), squeeze=False)
    for axis, (metric_name, title) in zip(axes[0], metric_specs):
        heatmap_df = (
            matched_df.pivot(index="interval_idx", columns="matched_boundary", values=metric_name)
            .reindex(index=list(interval_indices))
            .reindex(columns=boundaries)
        )
        values = heatmap_df.to_numpy(dtype=float)
        if metric_name == "correlation":
            image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        else:
            image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(len(boundaries)))
        axis.set_xticklabels(boundaries, rotation=35, ha="right")
        axis.set_yticks(range(len(heatmap_df.index)))
        axis.set_yticklabels([str(idx) for idx in heatmap_df.index])
        axis.set_title(title)
        axis.set_xlabel("Matched boundary diagnostic")
        axis.set_ylabel("Interval")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = os.path.join(output_dir, MATCHED_BOUNDARY_SUMMARY_FILE)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator] + body)


def fmt(value):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "NA"
    return f"{float(value):.6f}"


def write_note(
    output_dir: str,
    interpretation: ZInterpretation,
    interval_indices: Sequence[int],
    summary_df: pd.DataFrame,
    component_df: pd.DataFrame,
    control_df: pd.DataFrame,
    raw_sign_df: pd.DataFrame,
    plot_mode: str,
):
    global_raw = summary_df[(summary_df["region"] == "all") & (summary_df["group"] == "raw_vector")]
    boundary_raw = summary_df[
        (summary_df["region"].isin(["central_boundary"] + sorted([region for region in summary_df["region"].unique() if region.startswith("local_boundary_")])))
        & (summary_df["group"] == "raw_vector")
    ]
    global_control = summary_df[(summary_df["region"] == "all") & (summary_df["group"] == "control_difference_avg")]
    boundary_control = summary_df[
        (summary_df["region"].isin(["central_boundary"] + sorted([region for region in summary_df["region"].unique() if region.startswith("local_boundary_")])))
        & (summary_df["group"] == "control_difference_avg")
    ]

    def mean_metric(df: pd.DataFrame, metric: str):
        metric_df = df[df["metric"] == metric]
        if metric_df.empty:
            return float("nan")
        return float(np.nanmean(metric_df["value"].to_numpy(dtype=float)))

    def mean_sign_metric(df: pd.DataFrame, column_name: str):
        if df.empty or column_name not in df.columns:
            return float("nan")
        values = df[column_name].to_numpy(dtype=float)
        return float(np.nanmean(values)) if values.size > 0 else float("nan")

    def positivity_assessment(pos_frac: float, neg_frac: float):
        if math.isnan(pos_frac) or math.isnan(neg_frac):
            return "insufficient data"
        if neg_frac >= max(pos_frac, 0.5):
            return "mostly nonpositive"
        if pos_frac >= max(neg_frac, 0.5):
            return "frequently positive"
        return "mixed sign"

    def region_dependence_assessment(all_pos: float, boundary_pos: float, interior_pos: float):
        values = [value for value in [all_pos, boundary_pos, interior_pos] if not math.isnan(value)]
        if len(values) <= 1:
            return "insufficient data"
        spread = max(values) - min(values)
        if spread >= 0.15:
            return "strong region dependence"
        if spread >= 0.05:
            return "moderate region dependence"
        return "limited region dependence"

    def mismatch_source_assessment(raw_corr: float, control_mismatch: float):
        if math.isnan(raw_corr) or math.isnan(control_mismatch):
            return "insufficient data"
        if raw_corr < 0.5:
            return "the main mismatch is already visible in the gradient-difference outputs"
        if control_mismatch > 0.25:
            return "the componentwise D V outputs are closer, but the mismatch becomes more pronounced in the policy-oriented sign checks"
        return "the gradient-difference mismatch appears limited, and policy-oriented checks do not add much extra distortion"

    all_raw_sign = raw_sign_df[raw_sign_df["region"] == "all"].copy()
    boundary_regions = ["central_boundary"] + sorted(
        [region for region in raw_sign_df["region"].unique() if region.startswith("local_boundary_")]
    )
    boundary_raw_sign = raw_sign_df[raw_sign_df["region"].isin(boundary_regions)].copy()
    interior_raw_sign = raw_sign_df[raw_sign_df["region"] == "interior"].copy()

    autodiff_all_pos = mean_sign_metric(all_raw_sign, "autodiff_positive_fraction")
    autodiff_all_neg = mean_sign_metric(all_raw_sign, "autodiff_negative_fraction")
    z_net_all_pos = mean_sign_metric(all_raw_sign, "z_net_positive_fraction")
    z_net_all_neg = mean_sign_metric(all_raw_sign, "z_net_negative_fraction")
    autodiff_boundary_pos = mean_sign_metric(boundary_raw_sign, "autodiff_positive_fraction")
    z_net_boundary_pos = mean_sign_metric(boundary_raw_sign, "z_net_positive_fraction")
    autodiff_interior_pos = mean_sign_metric(interior_raw_sign, "autodiff_positive_fraction")
    z_net_interior_pos = mean_sign_metric(interior_raw_sign, "z_net_positive_fraction")

    note_lines = [
        "# Y/Z Gradient Consistency Note",
        "",
        "## Interpretation",
        f"- Interpreted `Z_Net` as: {interpretation.description}",
        f"- Compared it against: `{interpretation.comparison_target_name}` computed by torch autodiff on `Y_Net`.",
        f"- Code basis for this interpretation: {', '.join(interpretation.code_references)}.",
        "",
        "## Intervals Checked",
        f"- {', '.join(str(idx) for idx in interval_indices)}",
        "",
        "## Plotting Scope",
        "- All CSV metrics and summaries are still written in full.",
        f"- Plotting mode used for this run: `{plot_mode}`.",
        "- In `minimal` mode, plotting is intentionally reduced to the main diagnostic figures.",
        "- The diagnostic figures focus on: (1) global gradient-difference sanity checks and (2) matched-boundary control-difference failures.",
        "",
        "## High-Level Summary",
        f"- Mean D V vector MSE on `all`: {fmt(mean_metric(global_raw, 'mse'))}",
        f"- Mean D V component correlation on `all`: {fmt(mean_metric(global_raw, 'mean_component_correlation'))}",
        f"- Mean control-difference sign mismatch on `all`: {fmt(mean_metric(global_control, 'mean_sign_mismatch_rate'))}",
        f"- Mean near-threshold control sign mismatch on `all`: {fmt(mean_metric(global_control, 'mean_near_threshold_sign_mismatch_rate'))}",
        "",
        "## Boundary Behavior",
        f"- Mean D V vector MSE across boundary regions: {fmt(mean_metric(boundary_raw, 'mse'))}",
        f"- Mean D V component correlation across boundary regions: {fmt(mean_metric(boundary_raw, 'mean_component_correlation'))}",
        f"- Mean control-difference sign mismatch across boundary regions: {fmt(mean_metric(boundary_control, 'mean_sign_mismatch_rate'))}",
        f"- Mean near-threshold control sign mismatch across boundary regions: {fmt(mean_metric(boundary_control, 'mean_near_threshold_sign_mismatch_rate'))}",
        "",
        "## Interpretation of Results",
        "- If boundary-region control mismatch is materially worse than global gradient-difference mismatch, then the policy-relevant object is less stable near the boundaries than global D V matching alone suggests.",
        "- If the policy-oriented sign diagnostics are worse than the componentwise D V metrics, then control extraction amplifies Y/Z inconsistency.",
        "",
        "## Gradient-Difference Sign Summary",
        f"- Mean autodiff positive fraction on `all`: {fmt(mean_sign_metric(all_raw_sign, 'autodiff_positive_fraction'))}",
        f"- Mean autodiff negative fraction on `all`: {fmt(mean_sign_metric(all_raw_sign, 'autodiff_negative_fraction'))}",
        f"- Mean Z_Net positive fraction on `all`: {fmt(mean_sign_metric(all_raw_sign, 'z_net_positive_fraction'))}",
        f"- Mean Z_Net negative fraction on `all`: {fmt(mean_sign_metric(all_raw_sign, 'z_net_negative_fraction'))}",
        f"- Mean autodiff positive fraction across boundary regions: {fmt(mean_sign_metric(boundary_raw_sign, 'autodiff_positive_fraction'))}",
        f"- Mean Z_Net positive fraction across boundary regions: {fmt(mean_sign_metric(boundary_raw_sign, 'z_net_positive_fraction'))}",
        f"- Mean autodiff positive fraction on `interior`: {fmt(mean_sign_metric(interior_raw_sign, 'autodiff_positive_fraction'))}",
        f"- Mean Z_Net positive fraction on `interior`: {fmt(mean_sign_metric(interior_raw_sign, 'z_net_positive_fraction'))}",
        "",
        "## Gradient-Difference Interpretation",
        f"- Autodiff D V on `all` looks: {positivity_assessment(autodiff_all_pos, autodiff_all_neg)}.",
        f"- Z_Net D V on `all` looks: {positivity_assessment(z_net_all_pos, z_net_all_neg)}.",
        f"- Autodiff D V positivity shows: {region_dependence_assessment(autodiff_all_pos, autodiff_boundary_pos, autodiff_interior_pos)}.",
        f"- Z_Net D V positivity shows: {region_dependence_assessment(z_net_all_pos, z_net_boundary_pos, z_net_interior_pos)}.",
        f"- Overall, {mismatch_source_assessment(mean_metric(global_raw, 'mean_component_correlation'), mean_metric(global_control, 'mean_sign_mismatch_rate'))}.",
        "",
    ]

    raw_table = []
    for interval_idx in interval_indices:
        interval_component = component_df[(component_df["interval_idx"] == interval_idx) & (component_df["region"] == "all")]
        interval_control = control_df[(control_df["interval_idx"] == interval_idx) & (control_df["region"] == "all")]
        raw_table.append(
            [
                str(interval_idx),
                fmt(interval_component["correlation"].mean()),
                fmt(interval_component["mse"].mean()),
                fmt(interval_control["sign_mismatch_rate"].mean()),
                fmt(interval_control["near_threshold_sign_mismatch_rate"].mean()),
            ]
        )
    note_lines.extend(
        [
            "## Interval Snapshot",
            markdown_table(
                [
                    "Interval",
                    "Mean component corr",
                    "Mean component MSE",
                    "Mean control sign mismatch",
                    "Mean near-threshold sign mismatch",
                ],
                raw_table,
            ),
            "",
        ]
    )

    raw_sign_table = []
    for interval_idx in interval_indices:
        interval_raw_sign = raw_sign_df[(raw_sign_df["interval_idx"] == interval_idx) & (raw_sign_df["region"] == "all")]
        raw_sign_table.append(
            [
                str(interval_idx),
                fmt(mean_sign_metric(interval_raw_sign, "autodiff_positive_fraction")),
                fmt(mean_sign_metric(interval_raw_sign, "autodiff_negative_fraction")),
                fmt(mean_sign_metric(interval_raw_sign, "z_net_positive_fraction")),
                fmt(mean_sign_metric(interval_raw_sign, "z_net_negative_fraction")),
            ]
        )
    note_lines.extend(
        [
            "## Gradient-Difference Sign Snapshot",
            markdown_table(
                [
                    "Interval",
                    "Autodiff pos frac",
                    "Autodiff neg frac",
                    "Z_Net pos frac",
                    "Z_Net neg frac",
                ],
                raw_sign_table,
            ),
            "",
        ]
    )

    note_path = os.path.join(output_dir, NOTE_FILE)
    with open(note_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(note_lines))
    return note_path


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    experiment_dir, config_path, logs_dir, output_dir = resolve_paths(args)
    device = resolve_device(args.device)
    config, bsde = load_context(config_path, device)
    interpretation = infer_z_interpretation()

    interval_indices = parse_interval_indices(args.interval_indices, logs_dir)
    state_dim = int(bsde.dim) + 1
    dim = int(bsde.dim)

    for interval_idx in interval_indices:
        if interval_idx < 0 or interval_idx >= int(bsde.num_time_interval):
            raise ValueError(
                f"Interval {interval_idx} is outside [0, {int(bsde.num_time_interval) - 1}]."
            )

    sampled_frames = []
    for interval_idx in interval_indices:
        sampled_frames.append(
            sample_interval_states(
                bsde,
                interval_idx,
                args.num_states,
                rng,
                float(args.eps_local),
                float(args.eps_central),
            )
        )
    states_df, states_path = save_states_csv(sampled_frames, output_dir)

    summary_rows = []
    component_rows = []
    control_rows = []
    raw_sign_rows = []
    cached_control_scatter = {}
    cached_matched_boundary_control = {}
    cached_raw_scatter = {}

    for interval_idx in interval_indices:
        interval_state_df = states_df[states_df["interval_idx"] == interval_idx].copy().reset_index(drop=True)
        state_array = interval_state_df[sampled_state_columns(dim)].to_numpy(dtype=np.float32)
        y_net, z_net = load_interval_networks(config, logs_dir, interval_idx, device, state_dim)

        z_pred, grad_y = evaluate_networks_on_states(
            y_net,
            z_net,
            state_array,
            device,
            int(args.batch_size),
        )

        masks = region_masks(interval_state_df, dim)
        for region_name, mask in masks.items():
            if not np.any(mask):
                continue
            pred_region = z_pred[mask]
            grad_region = grad_y[mask]

            summary_rows.extend(summarize_raw_region(interval_idx, region_name, pred_region, grad_region))
            component_rows.extend(componentwise_rows(interval_idx, region_name, pred_region, grad_region, dim))
            raw_sign_rows.extend(raw_gradient_sign_rows(interval_idx, region_name, pred_region, grad_region, dim))
            control_detail_rows, control_summary_rows = control_difference_rows(
                interval_idx,
                region_name,
                pred_region,
                grad_region,
                dim,
                float(args.tau),
            )
            control_rows.extend(control_detail_rows)
            summary_rows.extend(control_summary_rows)

            cached_raw_scatter[(interval_idx, region_name)] = (pred_region.copy(), grad_region.copy())
            if region_name == "all":
                for k in range(dim):
                    cached_control_scatter[(interval_idx, k)] = (
                        pred_region[:, k],
                        grad_region[:, k],
                    )
            if region_name.startswith("local_boundary_"):
                region_k = int(region_name.split("_")[-1])
                if region_k < dim:
                    cached_matched_boundary_control[(interval_idx, region_k)] = (
                        pred_region[:, region_k],
                        grad_region[:, region_k],
                    )

    summary_df = pd.DataFrame(summary_rows)
    component_df = pd.DataFrame(component_rows)
    control_df = pd.DataFrame(control_rows)
    raw_sign_df = pd.DataFrame(raw_sign_rows)

    summary_path = os.path.join(output_dir, SUMMARY_FILE)
    component_path = os.path.join(output_dir, COMPONENTWISE_FILE)
    control_path = os.path.join(output_dir, CONTROL_FILE)
    raw_sign_path = os.path.join(output_dir, RAW_SIGN_SUMMARY_FILE)

    summary_df.to_csv(summary_path, index=False)
    component_df.to_csv(component_path, index=False)
    control_df.to_csv(control_path, index=False)
    raw_sign_df.to_csv(raw_sign_path, index=False)

    plot_intervals = plot_interval_indices(
        interval_indices,
        int(bsde.num_time_interval) - 1,
        args.plot_mode,
        bool(args.include_terminal_interval_plots),
    )

    heatmap_path = None
    barplot_path = None
    scatter_paths = []
    raw_scatter_paths = []
    matched_boundary_scatter_paths = []
    matched_boundary_summary_path = None

    if args.plot_mode == "full":
        heatmap_path = plot_component_correlation_heatmap(component_df, output_dir)
        barplot_path = plot_regionwise_barplots(summary_df, output_dir)
        for interval_idx in plot_intervals:
            scatter_path = plot_control_scatter(interval_idx, control_df, cached_control_scatter, output_dir)
            if scatter_path is not None:
                scatter_paths.append(scatter_path)

        ordered_regions = ["all"] + [f"local_boundary_{k}" for k in range(dim)] + ["central_boundary", "interior"]
        for interval_idx in plot_intervals:
            for region_name in ordered_regions:
                raw_pairs = cached_raw_scatter.get((interval_idx, region_name))
                if raw_pairs is None:
                    continue
                raw_scatter_paths.append(
                    plot_raw_gradient_scatter(interval_idx, region_name, raw_pairs, dim, output_dir)
                )
    else:
        for interval_idx in plot_intervals:
            raw_pairs = cached_raw_scatter.get((interval_idx, "all"))
            if raw_pairs is not None:
                raw_scatter_paths.append(
                    plot_raw_gradient_scatter(interval_idx, "all", raw_pairs, dim, output_dir)
                )

    for interval_idx in plot_intervals:
        matched_boundary_scatter_path = plot_matched_boundary_control_scatter(
            interval_idx,
            cached_matched_boundary_control,
            dim,
            output_dir,
        )
        if matched_boundary_scatter_path is not None:
            matched_boundary_scatter_paths.append(matched_boundary_scatter_path)

    matched_boundary_summary_path = plot_matched_boundary_summary(control_df, plot_intervals, dim, output_dir)

    note_path = write_note(
        output_dir,
        interpretation,
        interval_indices,
        summary_df,
        component_df,
        control_df,
        raw_sign_df,
        args.plot_mode,
    )

    print(f"Experiment directory: {experiment_dir}")
    print(f"Intervals checked: {interval_indices}")
    print(f"Interpreted Z_Net as: {interpretation.name}")
    print(f"Saved: {states_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {component_path}")
    print(f"Saved: {control_path}")
    print(f"Saved: {raw_sign_path}")
    print(f"Saved: {note_path}")
    if heatmap_path:
        print(f"Saved: {heatmap_path}")
    if barplot_path:
        print(f"Saved: {barplot_path}")
    for scatter_path in scatter_paths:
        print(f"Saved: {scatter_path}")
    for raw_scatter_path in raw_scatter_paths:
        print(f"Saved: {raw_scatter_path}")
    for matched_boundary_scatter_path in matched_boundary_scatter_paths:
        print(f"Saved: {matched_boundary_scatter_path}")
    if matched_boundary_summary_path:
        print(f"Saved: {matched_boundary_summary_path}")


if __name__ == "__main__":
    main()
