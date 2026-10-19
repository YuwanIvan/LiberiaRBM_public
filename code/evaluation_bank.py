"""Hash-pinned common-random-number banks for policy evaluation.

The bank stores the *realized* fine-grid Brownian increments and periodic-
review demand draws, rather than only recording the seed used to generate
them.  This makes evaluation replay independent of NumPy/SciPy/Torch random
number implementation details.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "frozen-evaluation-bank-v1"
FINE_FILENAME = "fine_grid_brownian_increments.npy"
PERIODIC_FILENAME = "periodic_review_demands.npy"
MANIFEST_FILENAME = "evaluation_bank_manifest.json"


class EvaluationBankError(RuntimeError):
    """Raised when a frozen evaluation bank cannot be trusted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationBankError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Hash an array's numerical content together with dtype and shape."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "dtype": array.dtype.str,
                "shape": [int(item) for item in array.shape],
                "order": "C",
            }
        )
    )
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_record(path: Path, values: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "file_sha256": file_sha256(path),
        "array_sha256": array_sha256(values),
        "dtype": np.asarray(values).dtype.str,
        "shape": [int(item) for item in np.asarray(values).shape],
        "bytes": int(path.stat().st_size),
    }


def periodic_moment_summary(demands: np.ndarray) -> dict[str, Any]:
    values = np.asarray(demands, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    if flat.shape[0] > 1 and flat.shape[1] > 1:
        corr = np.corrcoef(flat, rowvar=False)
        off_diag = corr[np.triu_indices(corr.shape[0], 1)]
        finite_corr = off_diag[np.isfinite(off_diag)]
    else:
        finite_corr = np.asarray([], dtype=np.float64)
    return {
        "mean_by_location": np.mean(flat, axis=0).tolist(),
        "variance_by_location": np.var(flat, axis=0, ddof=1).tolist(),
        "pairwise_correlation_mean": float(np.mean(finite_corr)) if finite_corr.size else None,
        "pairwise_correlation_min": float(np.min(finite_corr)) if finite_corr.size else None,
        "pairwise_correlation_max": float(np.max(finite_corr)) if finite_corr.size else None,
    }


def save_frozen_evaluation_bank(
    output_dir: str | Path,
    *,
    bank_id: str,
    scenario: str,
    m: int,
    seed: int,
    periodic_demand_mode: str,
    fine_grid_brownian_increments: np.ndarray,
    periodic_review_demands: np.ndarray,
    inputs: Mapping[str, Any],
) -> tuple[Path, str]:
    """Write a new bank and return its manifest path and SHA-256.

    Existing paths are never replaced.  The caller must choose a new versioned
    directory for every regenerated bank.
    """

    root = Path(output_dir).expanduser().resolve()
    _require(not root.exists(), f"Evaluation-bank output already exists: {root}")
    # Preserve the realized draws exactly as produced by the audited generator.
    # Down-casting here would make the stored bank a rounded reconstruction,
    # rather than the actual CRN sample used by evaluation.
    fine = np.asarray(fine_grid_brownian_increments)
    periodic = np.asarray(periodic_review_demands)
    _require(fine.ndim == 3, "Fine-grid increments must have shape (M, paths, dim).")
    _require(periodic.ndim == 3, "Periodic demands must have shape (periods, paths, dim).")
    _require(fine.shape[0] == int(m), f"Fine-grid bank has {fine.shape[0]} intervals, expected {m}.")
    _require(fine.shape[1] == periodic.shape[1], "Fine and periodic banks must use the same path count.")
    _require(fine.shape[2] == periodic.shape[2], "Fine and periodic banks must use the same dimension.")
    _require(fine.shape[1] > 0 and fine.shape[2] > 0, "Evaluation bank must be nonempty.")
    _require(np.isfinite(fine).all(), "Fine-grid bank contains nonfinite values.")
    _require(np.isfinite(periodic).all(), "Periodic bank contains nonfinite values.")
    _require(np.all(periodic >= 0.0), "Periodic demand bank contains negative demand.")

    root.mkdir(parents=True, exist_ok=False)
    fine_path = root / FINE_FILENAME
    periodic_path = root / PERIODIC_FILENAME
    _atomic_save_npy(fine_path, fine)
    _atomic_save_npy(periodic_path, periodic)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "frozen": True,
        "bank_id": str(bank_id),
        "scenario": str(scenario),
        "m": int(m),
        "review_intervals": int(periodic.shape[0]),
        "dim": int(fine.shape[2]),
        "num_paths": int(fine.shape[1]),
        "seed": int(seed),
        "periodic_demand_mode": str(periodic_demand_mode),
        "inputs": dict(inputs),
        "arrays": {
            "fine_grid_brownian_increments": _array_record(fine_path, fine),
            "periodic_review_demands": _array_record(periodic_path, periodic),
        },
        "periodic_realized_moments": periodic_moment_summary(periodic),
    }
    manifest_path = root / MANIFEST_FILENAME
    _atomic_write_json(manifest_path, payload)
    return manifest_path, file_sha256(manifest_path)


def load_bank_manifest(path: str | Path, expected_sha256: str, *, verify_arrays: bool = True) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    _require(manifest_path.is_file(), f"Evaluation-bank manifest is missing: {manifest_path}")
    actual_manifest_sha = file_sha256(manifest_path)
    _require(actual_manifest_sha == str(expected_sha256), "Evaluation-bank manifest SHA-256 mismatch.")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "Evaluation-bank manifest must be a JSON object.")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "Evaluation-bank schema mismatch.")
    _require(payload.get("frozen") is True, "Evaluation bank is not frozen.")
    arrays = payload.get("arrays")
    _require(isinstance(arrays, dict), "Evaluation-bank array inventory is missing.")
    required = {"fine_grid_brownian_increments", "periodic_review_demands"}
    _require(set(arrays) == required, "Evaluation-bank array inventory is incomplete or contains extras.")
    root = manifest_path.parent
    for name, record in arrays.items():
        _require(isinstance(record, dict), f"Evaluation-bank array record {name} is invalid.")
        relative = Path(str(record.get("path", "")))
        _require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe bank array path: {relative}")
        array_path = (root / relative).resolve()
        _require(array_path.parent == root, f"Bank array must be a direct child of {root}: {array_path}")
        _require(array_path.is_file(), f"Evaluation-bank array is missing: {array_path}")
        _require(file_sha256(array_path) == record.get("file_sha256"), f"Bank file hash mismatch: {array_path}")
        if verify_arrays:
            values = np.load(array_path, allow_pickle=False, mmap_mode="r")
            _require(list(values.shape) == list(record.get("shape", [])), f"Bank shape mismatch: {array_path}")
            _require(values.dtype.str == record.get("dtype"), f"Bank dtype mismatch: {array_path}")
            _require(array_sha256(values) == record.get("array_sha256"), f"Bank content hash mismatch: {array_path}")
    return payload


def load_frozen_evaluation_bank(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_scenario: str,
    expected_m: int,
    expected_num_paths: int,
    expected_seed: int,
    expected_periodic_demand_mode: str,
) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    payload = load_bank_manifest(manifest_path, expected_sha256, verify_arrays=True)
    expected = {
        "scenario": str(expected_scenario),
        "m": int(expected_m),
        "num_paths": int(expected_num_paths),
        "seed": int(expected_seed),
        "periodic_demand_mode": str(expected_periodic_demand_mode),
    }
    for field, wanted in expected.items():
        _require(payload.get(field) == wanted, f"Evaluation-bank {field} mismatch: expected {wanted!r}, got {payload.get(field)!r}.")
    arrays = payload["arrays"]
    fine = np.load(manifest_path.parent / arrays["fine_grid_brownian_increments"]["path"], allow_pickle=False)
    periodic = np.load(manifest_path.parent / arrays["periodic_review_demands"]["path"], allow_pickle=False)
    _require(fine.shape == (int(expected_m), int(expected_num_paths), int(payload["dim"])), "Fine-grid bank shape mismatch.")
    _require(periodic.shape == (int(payload["review_intervals"]), int(expected_num_paths), int(payload["dim"])), "Periodic bank shape mismatch.")
    return {
        "manifest": payload,
        "manifest_path": str(manifest_path),
        "manifest_sha256": str(expected_sha256),
        "fine_grid_brownian_increments": fine,
        "periodic_review_demands": periodic,
    }
