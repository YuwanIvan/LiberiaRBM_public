"""Immutable standard-normal bank for realized-demand rho validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import evaluation_bank


SCHEMA_VERSION = "fangonda-rho-validation-bank-v1"
ARRAY_FILENAME = "standard_normal_draws.npy"
MANIFEST_FILENAME = "rho_validation_bank_manifest.json"
# Storage aliases and legacy serialization tokens are retained for compatibility.
SHARED_LOGIN_ROOT = Path("/project_gpfs")
SHARED_COMPUTE_ROOT = Path("/mmfs1/data/scratch")


class RhoValidationBankError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RhoValidationBankError(message)


def canonical_shared_path(path: str | Path) -> str:
    """Normalize the two supported storage aliases for the shared scratch tree."""

    candidate = Path(str(path)).expanduser()
    require(candidate.is_absolute(), "Shared artifact path must be absolute")
    require(".." not in candidate.parts, "Shared artifact path rejects parent traversal")
    for root in (SHARED_LOGIN_ROOT, SHARED_COMPUTE_ROOT):
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        return str(Path("/__pythia_scratch__") / relative)
    return str(candidate)


def shared_paths_equivalent(left: str | Path, right: str | Path) -> bool:
    return canonical_shared_path(left) == canonical_shared_path(right)


def save_bank(
    output_dir: str | Path,
    *,
    values: np.ndarray,
    seed: int,
    source_config: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> tuple[Path, str]:
    root = Path(output_dir).expanduser().resolve()
    require(not root.exists(), f"Rho validation bank already exists: {root}")
    draws = np.asarray(values)
    require(draws.ndim == 3, "Standard-normal bank must have shape (weeks, paths, dim)")
    require(np.isfinite(draws).all(), "Standard-normal bank contains nonfinite values")
    root.mkdir(parents=True, exist_ok=False)
    array_path = root / ARRAY_FILENAME
    with array_path.open("xb") as handle:
        np.save(handle, draws, allow_pickle=False)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "frozen": True,
        "seed": int(seed),
        "review_intervals": int(draws.shape[0]),
        "num_paths": int(draws.shape[1]),
        "dim": int(draws.shape[2]),
        "source_config": dict(source_config),
        "execution": dict(execution),
        "array": {
            "path": array_path.name,
            "file_sha256": evaluation_bank.file_sha256(array_path),
            "array_sha256": evaluation_bank.array_sha256(draws),
            "shape": list(draws.shape),
            "dtype": draws.dtype.str,
        },
    }
    manifest_path = root / MANIFEST_FILENAME
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path, evaluation_bank.file_sha256(manifest_path)


def load_bank(
    manifest_path: str | Path,
    expected_sha256: str,
    *,
    expected_seed: int,
    expected_review_intervals: int,
    expected_num_paths: int,
    expected_dim: int,
) -> tuple[dict[str, Any], np.ndarray]:
    path = Path(manifest_path).expanduser().resolve()
    require(path.is_file(), f"Rho validation bank manifest is missing: {path}")
    require(
        evaluation_bank.file_sha256(path) == expected_sha256,
        "Rho validation bank manifest SHA-256 mismatch",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(
        payload.get("schema_version") == SCHEMA_VERSION and payload.get("frozen") is True,
        "Rho validation bank is not frozen v1",
    )
    expected = {
        "seed": int(expected_seed),
        "review_intervals": int(expected_review_intervals),
        "num_paths": int(expected_num_paths),
        "dim": int(expected_dim),
    }
    for field, value in expected.items():
        require(payload.get(field) == value, f"Rho validation bank {field} mismatch")
    record = payload.get("array", {})
    relative = Path(str(record.get("path", "")))
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe rho validation bank path")
    array_path = (path.parent / relative).resolve()
    require(array_path.parent == path.parent and array_path.is_file(), "Rho validation bank array is missing")
    require(
        evaluation_bank.file_sha256(array_path) == record.get("file_sha256"),
        "Rho validation bank file SHA-256 mismatch",
    )
    values = np.load(array_path, allow_pickle=False)
    require(list(values.shape) == record.get("shape"), "Rho validation bank shape record mismatch")
    require(values.dtype.str == record.get("dtype"), "Rho validation bank dtype record mismatch")
    require(
        evaluation_bank.array_sha256(values) == record.get("array_sha256"),
        "Rho validation bank content SHA-256 mismatch",
    )
    return payload, values


def validate_slurm_finalization(
    path: str | Path,
    expected_sha256: str,
    *,
    artifact_path: str | Path,
    artifact_sha256: str,
) -> dict[str, Any]:
    final_path = Path(path).expanduser().resolve()
    artifact = Path(artifact_path).expanduser().resolve()
    require(final_path.is_file(), f"Artifact completion record is missing: {final_path}")
    require(
        evaluation_bank.file_sha256(final_path) == expected_sha256,
        "Artifact completion SHA-256 mismatch",
    )
    payload = json.loads(final_path.read_text(encoding="utf-8"))
    require(
        payload.get("schema_version") == "pythia-artifact-slurm-final-v1"
        and payload.get("certified") is True
        and payload.get("state") == "COMPLETED"
        and payload.get("exit_code") == "0:0",
        "Artifact job did not finish COMPLETED/0:0",
    )
    record = payload.get("artifact", {})
    require(
        shared_paths_equivalent(record.get("path", ""), artifact)
        and record.get("sha256") == artifact_sha256,
        "Completion record certifies another artifact",
    )
    return payload
