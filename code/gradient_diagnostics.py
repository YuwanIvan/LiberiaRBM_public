"""Gradient-scale diagnostics for Deep Splitting training.

The utilities in this module are intentionally side-effect-light: component
gradient probes use ``torch.autograd.grad`` and never call ``optimizer.step``.
The production training path only pays this cost when diagnostics are enabled.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


QUANTILES: Tuple[Tuple[str, float], ...] = (
    ("p50", 50.0),
    ("p75", 75.0),
    ("p90", 90.0),
    ("p95", 95.0),
    ("p99", 99.0),
)

DIAGNOSTIC_FIELDNAMES: Tuple[str, ...] = (
    "interval",
    "phase",
    "iteration",
    "sample_index",
    "budget_type",
    "learning_rate",
    "optimizer_param_group_count",
    "grad_clip_enabled",
    "grad_clip_rule",
    "clip_norm",
    "clipped_indicator",
    "clipping_multiplier_total",
    "clipping_multiplier_h",
    "clipping_multiplier_g",
    "pre_clip_grad_norm_total",
    "pre_clip_grad_norm_h",
    "pre_clip_grad_norm_g",
    "post_clip_grad_norm_total",
    "post_clip_grad_norm_h",
    "post_clip_grad_norm_g",
    "parameter_norm_total",
    "parameter_norm_h",
    "parameter_norm_g",
    "update_norm_total",
    "update_norm_h",
    "update_norm_g",
    "relative_update_norm_total",
    "relative_update_norm_h",
    "relative_update_norm_g",
    "base_grad_norm_total",
    "base_grad_norm_h",
    "base_grad_norm_g",
    "consistency_grad_norm_total",
    "consistency_grad_norm_h",
    "consistency_grad_norm_g",
    "weighted_consistency_grad_norm_total",
    "weighted_consistency_grad_norm_h",
    "weighted_consistency_grad_norm_g",
    "total_loss_grad_norm_total",
    "total_loss_grad_norm_h",
    "total_loss_grad_norm_g",
    "weighted_consistency_to_base_grad_ratio_total",
    "weighted_consistency_to_base_grad_ratio_h",
    "weighted_consistency_to_base_grad_ratio_g",
    "h_g_pre_clip_grad_ratio",
    "h_g_post_clip_grad_ratio",
    "g_to_h_relative_update_ratio",
    "train_residual_loss",
    "train_consistency_loss",
    "train_weighted_consistency_loss",
    "train_total_loss",
    "weighted_consistency_to_residual_loss_ratio",
)

SUMMARY_METRICS: Tuple[str, ...] = (
    "pre_clip_grad_norm_total",
    "post_clip_grad_norm_total",
    "clipping_multiplier_total",
    "parameter_norm_total",
    "update_norm_total",
    "relative_update_norm_total",
    "pre_clip_grad_norm_h",
    "pre_clip_grad_norm_g",
    "post_clip_grad_norm_h",
    "post_clip_grad_norm_g",
    "clipping_multiplier_h",
    "clipping_multiplier_g",
    "h_g_pre_clip_grad_ratio",
    "h_g_post_clip_grad_ratio",
    "base_grad_norm_total",
    "base_grad_norm_h",
    "base_grad_norm_g",
    "weighted_consistency_grad_norm_total",
    "weighted_consistency_grad_norm_h",
    "weighted_consistency_grad_norm_g",
    "total_loss_grad_norm_total",
    "total_loss_grad_norm_h",
    "total_loss_grad_norm_g",
    "weighted_consistency_to_base_grad_ratio_total",
    "weighted_consistency_to_base_grad_ratio_h",
    "weighted_consistency_to_base_grad_ratio_g",
    "weighted_consistency_to_residual_loss_ratio",
    "learning_rate",
    "clipped_indicator",
    "relative_update_norm_h",
    "relative_update_norm_g",
    "g_to_h_relative_update_ratio",
)


def parse_interval_subset(value: Any) -> Optional[List[int]]:
    """Parse a comma-separated interval subset.

    ``None``, an empty string, and ``"all"`` all mean no restriction.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return sorted({int(item) for item in value})
    text = str(value).strip()
    if not text or text.lower() == "all":
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return sorted({int(part.strip()) for part in text.split(",") if part.strip()})


def git_commit(repo_root: Optional[os.PathLike[str] | str] = None) -> Optional[str]:
    """Return the current short git commit if available."""
    cwd = str(repo_root) if repo_root is not None else None
    root = Path(cwd or os.getcwd())
    git_dir = root / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            ref_path = git_dir / ref
            if ref_path.is_file():
                commit = ref_path.read_text(encoding="utf-8").strip()
            else:
                commit = ""
                packed = git_dir / "packed-refs"
                if packed.is_file():
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line and not line.startswith("#") and line.endswith(f" {ref}"):
                            commit = line.split(" ", 1)[0]
                            break
        else:
            commit = head
        if commit:
            return commit[:7]
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None
    return float(numerator) / float(denominator)


def tensor_l2_norm(tensors: Iterable[Optional[torch.Tensor]]) -> float:
    """Compute a global L2 norm over tensors, ignoring ``None`` entries."""
    sq_sum = 0.0
    for tensor in tensors:
        if tensor is None:
            continue
        detached = tensor.detach()
        if detached.numel() == 0:
            continue
        value = float(torch.sum(detached.float() * detached.float()).detach().cpu().item())
        sq_sum += value
    return float(math.sqrt(max(sq_sum, 0.0)))


def parameter_l2_norm(params: Sequence[torch.nn.Parameter]) -> float:
    return tensor_l2_norm(param.detach() for param in params)


def gradient_l2_norm(params: Sequence[torch.nn.Parameter]) -> float:
    return tensor_l2_norm(param.grad for param in params)


def split_norms(values: Sequence[Optional[torch.Tensor]], h_count: int) -> Dict[str, float]:
    h_values = values[:h_count]
    g_values = values[h_count:]
    h_norm = tensor_l2_norm(h_values)
    g_norm = tensor_l2_norm(g_values)
    total_norm = tensor_l2_norm(values)
    return {"total": total_norm, "h": h_norm, "g": g_norm}


def group_parameter_norms(
    h_params: Sequence[torch.nn.Parameter],
    g_params: Sequence[torch.nn.Parameter],
) -> Dict[str, float]:
    return {
        "total": parameter_l2_norm([*h_params, *g_params]),
        "h": parameter_l2_norm(h_params),
        "g": parameter_l2_norm(g_params),
    }


def group_gradient_norms(
    h_params: Sequence[torch.nn.Parameter],
    g_params: Sequence[torch.nn.Parameter],
) -> Dict[str, float]:
    return {
        "total": gradient_l2_norm([*h_params, *g_params]),
        "h": gradient_l2_norm(h_params),
        "g": gradient_l2_norm(g_params),
    }


def loss_component_gradient_norms(
    loss_tensor: Optional[torch.Tensor],
    params: Sequence[torch.nn.Parameter],
    h_count: int,
) -> Dict[str, float]:
    """Return H/G/total gradient norms without writing to ``param.grad``."""
    if loss_tensor is None or not isinstance(loss_tensor, torch.Tensor) or not loss_tensor.requires_grad:
        return {"total": 0.0, "h": 0.0, "g": 0.0}
    grads = torch.autograd.grad(
        loss_tensor,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    return split_norms(grads, h_count)


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.12g}"
    return value


class GradientDiagnosticsRecorder:
    """Write sampled optimizer-step diagnostics and produce summary artifacts."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: Optional[os.PathLike[str] | str],
        interval_subset: Optional[Sequence[int]] = None,
        every: int = 100,
        component_every: Optional[int] = None,
        max_records_per_interval: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        append: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir) if output_dir else None
        self.interval_subset = None if interval_subset is None else set(int(item) for item in interval_subset)
        self.every = max(int(every), 0)
        self.component_every = self.every if component_every is None else max(int(component_every), 0)
        self.max_records_per_interval = (
            None if max_records_per_interval is None else max(int(max_records_per_interval), 0)
        )
        self.metadata = dict(metadata or {})
        self.append = bool(append)
        self._counts_by_interval: Dict[int, int] = {}
        self._csv_file = None
        self._writer = None
        self.csv_path: Optional[Path] = None

        if self.enabled:
            if self.output_dir is None:
                raise ValueError("gradient diagnostics require an output directory")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = self.output_dir / "gradient_diagnostics_interval.csv"
            mode = "a" if self.append and self.csv_path.exists() else "w"
            self._csv_file = self.csv_path.open(mode, newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=DIAGNOSTIC_FIELDNAMES)
            if mode == "w":
                self._writer.writeheader()
            self.write_metadata()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None

    def write_metadata(self) -> None:
        if not self.enabled or self.output_dir is None:
            return
        metadata = {
            **self.metadata,
            "diagnostics_enabled": True,
            "diagnostics_every": self.every,
            "diagnostics_component_every": self.component_every,
            "diagnostics_interval_subset": (
                None if self.interval_subset is None else sorted(self.interval_subset)
            ),
            "diagnostics_max_records_per_interval": self.max_records_per_interval,
            "interval_csv": "gradient_diagnostics_interval.csv",
        }
        with (self.output_dir / "gradient_diagnostics_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True, default=str)

    def should_record(self, interval_idx: int, iteration_idx: int) -> bool:
        if not self.enabled or self.every <= 0:
            return False
        interval_idx = int(interval_idx)
        if self.interval_subset is not None and interval_idx not in self.interval_subset:
            return False
        if int(iteration_idx) % self.every != 0:
            return False
        if self.max_records_per_interval is not None:
            if self._counts_by_interval.get(interval_idx, 0) >= self.max_records_per_interval:
                return False
        return True

    def should_record_components(self, iteration_idx: int) -> bool:
        return self.component_every > 0 and int(iteration_idx) % self.component_every == 0

    def prepare_step(
        self,
        *,
        interval_idx: int,
        iteration_idx: int,
        phase: str,
        budget_type: str,
        learning_rate: float,
        optimizer_param_group_count: int,
        h_params: Sequence[torch.nn.Parameter],
        g_params: Sequence[torch.nn.Parameter],
        residual_loss: torch.Tensor,
        consistency_loss: torch.Tensor,
        total_loss: torch.Tensor,
        consistency_coef: float,
    ) -> Optional[Dict[str, Any]]:
        """Collect pre-backward diagnostic state.

        This method may call ``torch.autograd.grad`` but never writes gradients,
        mutates parameters, or steps the optimizer.
        """
        if not self.should_record(interval_idx, iteration_idx):
            return None

        h_params = list(h_params)
        g_params = list(g_params)
        params = [*h_params, *g_params]
        h_count = len(h_params)
        parameter_norms = group_parameter_norms(h_params, g_params)
        snapshot = [param.detach().clone() for param in params]
        component_metrics: Dict[str, float] = {}

        if self.should_record_components(iteration_idx):
            base_norms = loss_component_gradient_norms(residual_loss, params, h_count)
            consistency_norms = loss_component_gradient_norms(consistency_loss, params, h_count)
            weighted_consistency_loss = consistency_loss * float(consistency_coef)
            weighted_consistency_norms = loss_component_gradient_norms(
                weighted_consistency_loss,
                params,
                h_count,
            )
            total_norms = loss_component_gradient_norms(total_loss, params, h_count)
            for key in ("total", "h", "g"):
                component_metrics[f"base_grad_norm_{key}"] = base_norms[key]
                component_metrics[f"consistency_grad_norm_{key}"] = consistency_norms[key]
                component_metrics[f"weighted_consistency_grad_norm_{key}"] = weighted_consistency_norms[key]
                component_metrics[f"total_loss_grad_norm_{key}"] = total_norms[key]
                component_metrics[f"weighted_consistency_to_base_grad_ratio_{key}"] = (
                    safe_ratio(weighted_consistency_norms[key], base_norms[key])
                )

        residual_value = safe_float(residual_loss.detach().cpu().item())
        consistency_value = safe_float(consistency_loss.detach().cpu().item())
        weighted_consistency_value = (
            None if consistency_value is None else float(consistency_coef) * consistency_value
        )
        total_value = safe_float(total_loss.detach().cpu().item())

        return {
            "interval": int(interval_idx),
            "iteration": int(iteration_idx),
            "phase": phase,
            "budget_type": budget_type,
            "learning_rate": float(learning_rate),
            "optimizer_param_group_count": int(optimizer_param_group_count),
            "h_count": h_count,
            "parameter_norms": parameter_norms,
            "param_snapshot": snapshot,
            "component_metrics": component_metrics,
            "loss_values": {
                "train_residual_loss": residual_value,
                "train_consistency_loss": consistency_value,
                "train_weighted_consistency_loss": weighted_consistency_value,
                "train_total_loss": total_value,
                "weighted_consistency_to_residual_loss_ratio": safe_ratio(
                    weighted_consistency_value,
                    residual_value,
                ),
            },
        }

    def record_step(
        self,
        *,
        prepared: Optional[Mapping[str, Any]],
        h_params: Sequence[torch.nn.Parameter],
        g_params: Sequence[torch.nn.Parameter],
        pre_clip_norms: Mapping[str, float],
        post_clip_norms: Mapping[str, float],
        grad_clip_enabled: bool,
        grad_clip_rule: str,
        clip_norm: float,
        clipped_indicator: bool,
        clip_norm_h: Optional[float] = None,
        clip_norm_g: Optional[float] = None,
        clip_norm_total: Optional[float] = None,
    ) -> None:
        if not self.enabled or prepared is None or self._writer is None:
            return

        h_params = list(h_params)
        g_params = list(g_params)
        params = [*h_params, *g_params]
        snapshots = list(prepared["param_snapshot"])
        deltas = [param.detach() - before for param, before in zip(params, snapshots)]
        h_count = int(prepared["h_count"])
        update_norms = split_norms(deltas, h_count)
        parameter_norms = dict(prepared["parameter_norms"])
        interval = int(prepared["interval"])
        sample_index = self._counts_by_interval.get(interval, 0)

        def multiplier(group: str) -> Optional[float]:
            pre = safe_float(pre_clip_norms.get(group))
            if pre is None or pre <= 0.0:
                return 1.0
            if not grad_clip_enabled:
                return 1.0
            group_clip_norm = {
                "total": clip_norm_total,
                "h": clip_norm_h,
                "g": clip_norm_g,
            }.get(group)
            if group_clip_norm is None:
                group_clip_norm = clip_norm
            return min(1.0, float(group_clip_norm) / pre)

        row: Dict[str, Any] = {
            "interval": interval,
            "phase": prepared["phase"],
            "iteration": prepared["iteration"],
            "sample_index": sample_index,
            "budget_type": prepared["budget_type"],
            "learning_rate": prepared["learning_rate"],
            "optimizer_param_group_count": prepared["optimizer_param_group_count"],
            "grad_clip_enabled": bool(grad_clip_enabled),
            "grad_clip_rule": grad_clip_rule,
            "clip_norm": clip_norm,
            "clipped_indicator": bool(clipped_indicator),
            "clipping_multiplier_total": multiplier("total"),
            "clipping_multiplier_h": multiplier("h"),
            "clipping_multiplier_g": multiplier("g"),
            "h_g_pre_clip_grad_ratio": safe_ratio(
                pre_clip_norms.get("h"),
                pre_clip_norms.get("g"),
            ),
            "h_g_post_clip_grad_ratio": safe_ratio(
                post_clip_norms.get("h"),
                post_clip_norms.get("g"),
            ),
        }
        for key in ("total", "h", "g"):
            row[f"pre_clip_grad_norm_{key}"] = pre_clip_norms.get(key)
            row[f"post_clip_grad_norm_{key}"] = post_clip_norms.get(key)
            row[f"parameter_norm_{key}"] = parameter_norms.get(key)
            row[f"update_norm_{key}"] = update_norms.get(key)
            row[f"relative_update_norm_{key}"] = safe_ratio(
                update_norms.get(key),
                parameter_norms.get(key),
            )
        row["g_to_h_relative_update_ratio"] = safe_ratio(
            row.get("relative_update_norm_g"),
            row.get("relative_update_norm_h"),
        )
        row.update(prepared.get("component_metrics", {}))
        row.update(prepared.get("loss_values", {}))

        self._writer.writerow({field: _format_csv_value(row.get(field)) for field in DIAGNOSTIC_FIELDNAMES})
        if self._csv_file is not None:
            self._csv_file.flush()
        self._counts_by_interval[interval] = sample_index + 1

    def finalize(self) -> None:
        if not self.enabled or self.output_dir is None:
            return
        self.close()
        summarize_gradient_diagnostics(self.output_dir, metadata=self.metadata)


def _read_diagnostic_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _numeric_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = safe_float(row.get(key))
        if value is not None:
            values.append(value)
    return np.asarray(values, dtype=float)


def _quantile_summary(values: np.ndarray) -> Dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, **{name: None for name, _ in QUANTILES}, "max": None}
    out: Dict[str, Any] = {"count": int(finite.size)}
    for name, q in QUANTILES:
        out[name] = float(np.percentile(finite, q))
    out["max"] = float(np.max(finite))
    return out


def _write_interval_summary(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in rows:
        interval = int(float(row.get("interval", 0)))
        phase = str(row.get("phase", ""))
        groups.setdefault((phase, interval), []).append(row)

    fieldnames = ["phase", "interval", "num_records"]
    for metric in SUMMARY_METRICS:
        for name, _ in QUANTILES:
            fieldnames.append(f"{metric}_{name}")
        fieldnames.append(f"{metric}_max")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (phase, interval), group_rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
            row: Dict[str, Any] = {
                "phase": phase,
                "interval": interval,
                "num_records": len(group_rows),
            }
            for metric in SUMMARY_METRICS:
                summary = _quantile_summary(_numeric_values(group_rows, metric))
                for name, _ in QUANTILES:
                    row[f"{metric}_{name}"] = summary.get(name)
                row[f"{metric}_max"] = summary.get("max")
            writer.writerow({key: _format_csv_value(value) for key, value in row.items()})


def _plot_outputs(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Dict[str, str]:
    if not rows:
        return {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional plotting stack
        return {"plot_warning": f"matplotlib unavailable: {exc}"}

    paths: Dict[str, str] = {}

    def heatmap(metric: str, filename: str, title: str, log10: bool = False) -> None:
        intervals = sorted({int(float(row["interval"])) for row in rows})
        max_samples = max(int(float(row.get("sample_index", 0))) for row in rows) + 1
        grid = np.full((max_samples, len(intervals)), np.nan, dtype=float)
        interval_to_col = {interval: idx for idx, interval in enumerate(intervals)}
        for row in rows:
            value = safe_float(row.get(metric))
            if value is None:
                continue
            if log10:
                value = math.log10(max(value, 1e-12))
            sample = int(float(row.get("sample_index", 0)))
            interval = int(float(row["interval"]))
            grid[sample, interval_to_col[interval]] = value
        fig, ax = plt.subplots(figsize=(max(8, len(intervals) * 0.18), 4.8))
        image = ax.imshow(grid, aspect="auto", interpolation="nearest", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("interval")
        ax.set_ylabel("diagnostic sample")
        tick_count = min(len(intervals), 12)
        if tick_count > 0:
            tick_positions = np.linspace(0, len(intervals) - 1, tick_count, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(intervals[idx]) for idx in tick_positions], rotation=45, ha="right")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        out = output_dir / filename
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths[filename] = str(out)

    def interval_median(metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        intervals = sorted({int(float(row["interval"])) for row in rows})
        medians = []
        p90s = []
        for interval in intervals:
            values = _numeric_values(
                [row for row in rows if int(float(row["interval"])) == interval],
                metric,
            )
            summary = _quantile_summary(values)
            medians.append(np.nan if summary["p50"] is None else summary["p50"])
            p90s.append(np.nan if summary["p90"] is None else summary["p90"])
        return np.asarray(intervals), np.asarray(medians, dtype=float), np.asarray(p90s, dtype=float)

    heatmap(
        "pre_clip_grad_norm_total",
        "gradient_norm_heatmap.png",
        "Pre-clip Gradient Norm (log10)",
        log10=True,
    )
    heatmap(
        "clipping_multiplier_total",
        "clipping_multiplier_heatmap.png",
        "Clipping Multiplier",
        log10=False,
    )

    for metric, filename, ylabel, title in (
        (
            "h_g_pre_clip_grad_ratio",
            "hg_gradient_ratio.png",
            "H / G pre-clip gradient norm",
            "H/G Gradient Ratio",
        ),
        (
            "weighted_consistency_to_base_grad_ratio_total",
            "loss_component_gradient_ratio.png",
            "weighted consistency / base gradient norm",
            "Loss Component Gradient Ratio",
        ),
    ):
        intervals, medians, p90s = interval_median(metric)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(intervals, medians, marker="o", linewidth=1.2, label="p50")
        ax.plot(intervals, p90s, marker=".", linewidth=1.0, label="p90")
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("interval")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        out = output_dir / filename
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths[filename] = str(out)

    return paths


def summarize_gradient_diagnostics(
    diagnostics_dir: os.PathLike[str] | str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create interval CSV, run summary JSON, and plots for diagnostics."""
    output_dir = Path(diagnostics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        metadata_path = output_dir / "gradient_diagnostics_metadata.json"
        if metadata_path.is_file():
            try:
                with metadata_path.open(encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = None
    rows = _read_diagnostic_rows(output_dir / "gradient_diagnostics_interval.csv")
    interval_summary_path = output_dir / "gradient_diagnostics_interval_summary.csv"
    _write_interval_summary(rows, interval_summary_path)

    overall = {
        metric: _quantile_summary(_numeric_values(rows, metric))
        for metric in SUMMARY_METRICS
    }
    phase_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        phase_rows.setdefault(str(row.get("phase", "")), []).append(row)
    by_phase = {
        phase: {
            metric: _quantile_summary(_numeric_values(group_rows, metric))
            for metric in SUMMARY_METRICS
        }
        for phase, group_rows in phase_rows.items()
    }
    plot_paths = _plot_outputs(rows, output_dir)
    run_summary_path = output_dir / "gradient_diagnostics_run_summary.json"
    summary = {
        "metadata": dict(metadata or {}),
        "num_records": len(rows),
        "interval_summary_csv": str(interval_summary_path),
        "run_summary_json": str(run_summary_path),
        "overall_quantiles": overall,
        "phase_quantiles": by_phase,
        "plots": plot_paths,
    }
    with run_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
    return summary
