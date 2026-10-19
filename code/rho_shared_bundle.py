"""Validate portable bundles of calibrated demand-correlation inputs.

Set LIBERIARBM_SHARED_ROOT when bundle storage is outside the current user's
home directory. Validation preserves the recorded numerical and file hashes.
"""

from __future__ import annotations

import json
import os
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import demand_correlation_calibration
import evaluation_bank
import rho_validation_bank
import rho_validation_statistics


SPEC_SCHEMA_VERSION = "fangonda-rho-shared-bundle-spec-v1"
BUNDLE_SCHEMA_VERSION = "fangonda-rho-shared-bundle-v1"
BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"
SHARED_HOME_ROOT = Path(os.environ.get("LIBERIARBM_SHARED_ROOT", str(Path.home()))).expanduser().resolve()


class RhoSharedBundleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RhoSharedBundleError(message)


def file_sha256(path: str | Path) -> str:
    return evaluation_bank.file_sha256(Path(path))


def _hex_sha(value: Any, label: str) -> str:
    text = str(value)
    require(len(text) == 64 and all(char in "0123456789abcdef" for char in text), f"Invalid {label} SHA-256")
    return text


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RhoSharedBundleError(f"Invalid {label}: {path}") from exc
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _absolute_file(record: Mapping[str, Any], path_field: str, sha_field: str, label: str) -> tuple[Path, str]:
    path = Path(str(record.get(path_field, ""))).expanduser()
    require(path.is_absolute(), f"{label} path must be absolute")
    digest = _hex_sha(record.get(sha_field), label)
    require(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}: {path}")
    require(file_sha256(path) == digest, f"{label} SHA-256 mismatch")
    return path.resolve(), digest


def _relative_file(root: Path, record: Mapping[str, Any], label: str) -> tuple[Path, str]:
    relative = Path(str(record.get("relative_path", "")))
    require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe {label} relative path")
    path = (root / relative).resolve()
    require(path != root and root in path.parents, f"{label} escaped the bundle root")
    digest = _hex_sha(record.get("sha256"), label)
    require(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}: {path}")
    require(file_sha256(path) == digest, f"{label} SHA-256 mismatch")
    return path, digest


def validate_source_spec(spec: Mapping[str, Any], *, verify_inputs: bool = True) -> dict[str, Any]:
    """Validate the frozen shared-input bundle specification."""

    require(spec.get("schema_version") == SPEC_SCHEMA_VERSION, "Wrong rho shared-bundle spec schema")
    require(spec.get("frozen") is True, "Rho shared-bundle spec must be frozen")
    repository = spec.get("repository")
    require(isinstance(repository, Mapping), "Rho shared-bundle spec lacks repository identity")
    require(bool(str(repository.get("branch", "")).strip()), "Bundle repository branch is missing")
    commit = str(repository.get("commit", ""))
    require(len(commit) == 40 and all(char in "0123456789abcdef" for char in commit), "Invalid bundle repository commit")
    output_root = Path(str(spec.get("output_root", ""))).expanduser()
    require(output_root.is_absolute(), "Bundle output_root must be absolute")
    require(SHARED_HOME_ROOT == output_root or SHARED_HOME_ROOT in output_root.parents, "Bundle output_root must be in shared home")
    require(not output_root.exists(), f"Bundle output_root already exists: {output_root}")

    bank = spec.get("validation_bank")
    require(isinstance(bank, Mapping), "Bundle spec lacks validation_bank")
    scenarios = spec.get("scenarios")
    require(isinstance(scenarios, list) and len(scenarios) == 2, "Bundle spec must contain exactly two rho scenarios")
    names = [str(item.get("scenario", "")) for item in scenarios if isinstance(item, Mapping)]
    require(names == ["fangonda_rho_realized_0p5", "fangonda_rho_realized_0p7"], "Bundle scenarios must be rho=0.5 then rho=0.70")

    if verify_inputs:
        bank_path, bank_sha = _absolute_file(bank, "manifest_path", "manifest_sha256", "validation bank manifest")
        bank_final_path, bank_final_sha = _absolute_file(
            bank, "finalization_path", "finalization_sha256", "validation bank finalization"
        )
        rho_validation_bank.validate_slurm_finalization(
            bank_final_path,
            bank_final_sha,
            artifact_path=bank_path,
            artifact_sha256=bank_sha,
        )
        for item in scenarios:
            calibration_path, calibration_sha = _absolute_file(
                item, "calibration_report_path", "calibration_report_sha256", "calibration report"
            )
            calibration_final_path, calibration_final_sha = _absolute_file(
                item,
                "calibration_finalization_path",
                "calibration_finalization_sha256",
                "calibration finalization",
            )
            validation_path, validation_sha = _absolute_file(
                item, "validation_report_path", "validation_report_sha256", "validation report"
            )
            validation_final_path, validation_final_sha = _absolute_file(
                item,
                "validation_finalization_path",
                "validation_finalization_sha256",
                "validation finalization",
            )
            rho_validation_bank.validate_slurm_finalization(
                calibration_final_path,
                calibration_final_sha,
                artifact_path=calibration_path,
                artifact_sha256=calibration_sha,
            )
            rho_validation_bank.validate_slurm_finalization(
                validation_final_path,
                validation_final_sha,
                artifact_path=validation_path,
                artifact_sha256=validation_sha,
            )
    return {"output_root": str(output_root), "scenario_count": len(scenarios)}


def _validate_mirrored_finalization(
    root: Path,
    mirror_record: Mapping[str, Any],
    *,
    source_artifact_path: str | Path,
    source_artifact_sha256: str,
    label: str,
) -> tuple[Path, str]:
    path, digest = _relative_file(root, mirror_record, label)
    rho_validation_bank.validate_slurm_finalization(
        path,
        digest,
        artifact_path=source_artifact_path,
        artifact_sha256=source_artifact_sha256,
    )
    return path, digest


def validate_bundle(
    manifest_path: str | Path,
    expected_sha256: str,
    *,
    finalization_path: str | Path,
    finalization_sha256: str,
    scenario: str,
    shared_home_root: str | Path = SHARED_HOME_ROOT,
) -> dict[str, Any]:
    """Validate a local bundle and resolve one scenario's shared files."""

    manifest = Path(manifest_path).expanduser().resolve()
    home = Path(shared_home_root).expanduser().resolve()
    require(home == manifest or home in manifest.parents, "Rho bundle is outside shared home")
    expected = _hex_sha(expected_sha256, "bundle manifest")
    require(manifest.name == BUNDLE_MANIFEST_FILENAME, "Unexpected rho bundle manifest filename")
    require(manifest.is_file() and not manifest.is_symlink(), f"Missing or unsafe rho bundle manifest: {manifest}")
    require(file_sha256(manifest) == expected, "Rho bundle manifest SHA-256 mismatch")
    root = manifest.parent
    final_path = Path(finalization_path).expanduser().resolve()
    require(final_path == root / "provenance" / "slurm_final.json", "Rho bundle finalization path is not canonical")
    rho_validation_bank.validate_slurm_finalization(
        final_path,
        _hex_sha(finalization_sha256, "bundle finalization"),
        artifact_path=manifest,
        artifact_sha256=expected,
    )
    payload = _read_json(manifest, "rho bundle manifest")
    require(payload.get("schema_version") == BUNDLE_SCHEMA_VERSION and payload.get("frozen") is True, "Rho bundle is not frozen v1")
    require(payload.get("full_source_bank_content_verified") is True, "Bundle did not verify the full source bank")

    file_records = payload.get("files")
    require(isinstance(file_records, list) and bool(file_records), "Rho bundle has no file inventory")
    expected_files: set[str] = set()
    for record in file_records:
        require(isinstance(record, Mapping), "Invalid rho bundle file inventory entry")
        path, digest = _relative_file(root, record, "bundle inventory file")
        relative = str(path.relative_to(root))
        require(relative not in expected_files, f"Duplicate bundle file inventory entry: {relative}")
        expected_files.add(relative)
        require(int(record.get("bytes", -1)) == path.stat().st_size, f"Bundle file size mismatch: {relative}")
        require(digest == record.get("sha256"), f"Bundle inventory SHA mismatch: {relative}")
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path not in {manifest, final_path}
    }
    require(actual_files == expected_files, "Rho bundle contains missing or untracked files")

    bank = payload.get("validation_bank")
    require(isinstance(bank, Mapping), "Rho bundle lacks validation bank evidence")
    source_bank = bank.get("source")
    mirror_bank = bank.get("mirror")
    require(isinstance(source_bank, Mapping) and isinstance(mirror_bank, Mapping), "Invalid bundle bank evidence")
    bank_manifest_path, bank_manifest_sha = _relative_file(root, mirror_bank["manifest"], "mirrored bank manifest")
    require(bank_manifest_sha == source_bank.get("manifest_sha256"), "Mirrored bank manifest differs from source hash")
    _validate_mirrored_finalization(
        root,
        mirror_bank["finalization"],
        source_artifact_path=source_bank.get("manifest_path", ""),
        source_artifact_sha256=str(source_bank.get("manifest_sha256", "")),
        label="mirrored bank finalization",
    )
    bank_payload = _read_json(bank_manifest_path, "mirrored bank manifest")
    require(bank_payload.get("schema_version") == rho_validation_bank.SCHEMA_VERSION and bank_payload.get("frozen") is True, "Mirrored bank manifest is invalid")
    contract = bank.get("verified_contract")
    require(isinstance(contract, Mapping), "Bundle lacks verified bank contract")
    for field in ("seed", "review_intervals", "num_paths", "dim"):
        require(bank_payload.get(field) == contract.get(field), f"Mirrored bank {field} mismatch")
    array_record = bank_payload.get("array", {})
    for field in ("file_sha256", "array_sha256", "shape", "dtype"):
        require(array_record.get(field) == contract.get(field), f"Mirrored bank array {field} mismatch")

    scenarios = payload.get("scenarios")
    require(isinstance(scenarios, list), "Bundle scenario list is invalid")
    matches = [item for item in scenarios if isinstance(item, Mapping) and item.get("scenario") == scenario]
    require(len(matches) == 1, f"Bundle contains no unique scenario entry for {scenario}")
    entry = matches[0]
    calibration = entry.get("calibration")
    validation = entry.get("validation")
    require(isinstance(calibration, Mapping) and isinstance(validation, Mapping), "Invalid bundle scenario evidence")
    source_calibration = calibration.get("source")
    source_validation = validation.get("source")
    require(isinstance(source_calibration, Mapping) and isinstance(source_validation, Mapping), "Bundle scenario lacks source records")
    calibration_path, calibration_sha = _relative_file(root, calibration["mirror_report"], "mirrored calibration report")
    validation_path, validation_sha = _relative_file(root, validation["mirror_report"], "mirrored validation report")
    require(calibration_sha == source_calibration.get("report_sha256"), "Mirrored calibration report differs from source")
    require(validation_sha == source_validation.get("report_sha256"), "Mirrored validation report differs from source")
    _validate_mirrored_finalization(
        root,
        calibration["mirror_finalization"],
        source_artifact_path=source_calibration.get("report_path", ""),
        source_artifact_sha256=calibration_sha,
        label="mirrored calibration finalization",
    )
    _validate_mirrored_finalization(
        root,
        validation["mirror_finalization"],
        source_artifact_path=source_validation.get("report_path", ""),
        source_artifact_sha256=validation_sha,
        label="mirrored validation finalization",
    )

    calibration_payload = _read_json(calibration_path, "mirrored calibration report")
    validation_payload = _read_json(validation_path, "mirrored validation report")
    require(
        calibration_payload.get("schema_version") == "fangonda-realized-rho-calibration-v1"
        and calibration_payload.get("frozen") is True
        and calibration_payload.get("numerical_method", {}).get("name") == demand_correlation_calibration.CALIBRATION_METHOD,
        "Mirrored calibration report is not the audited exact method",
    )
    require(
        validation_payload.get("schema_version") == rho_validation_statistics.VALIDATION_SCHEMA_VERSION
        and validation_payload.get("frozen") is True
        and validation_payload.get("validation_passed") is True,
        "Mirrored validation report did not pass validator v2",
    )
    target = float(entry.get("target_realized_rho", float("nan")))
    require(
        calibration_payload.get("scenario") == scenario
        and math.isclose(float(calibration_payload.get("target_realized_rho", float("nan"))), target),
        "Bundle calibration scenario or target mismatch",
    )
    validation_link = validation_payload.get("calibration_report", {})
    require(
        rho_validation_bank.shared_paths_equivalent(
            validation_link.get("path", ""), source_calibration.get("report_path", "")
        )
        and validation_link.get("sha256") == calibration_sha,
        "Bundle validation does not certify its source calibration",
    )
    calibration_final_link = validation_payload.get("calibration_slurm_finalization", {})
    require(
        rho_validation_bank.shared_paths_equivalent(
            calibration_final_link.get("path", ""), source_calibration.get("finalization_path", "")
        )
        and calibration_final_link.get("sha256") == source_calibration.get("finalization_sha256"),
        "Bundle validation points to another calibration finalization",
    )
    validation_bank = validation_payload.get("validation_bank", {})
    require(
        rho_validation_bank.shared_paths_equivalent(
            validation_bank.get("path", ""), source_bank.get("manifest_path", "")
        )
        and validation_bank.get("sha256") == bank_manifest_sha
        and validation_bank.get("array_file_sha256") == contract.get("file_sha256")
        and validation_bank.get("array_sha256") == contract.get("array_sha256"),
        "Bundle validation does not pin the verified source bank",
    )
    require(int(validation_payload.get("num_paths", -1)) == 200000 and int(validation_payload.get("seed", -1)) == 2026082203, "Bundle validation bank contract mismatch")
    bank_final_link = validation_payload.get("validation_bank_slurm_finalization", {})
    require(
        rho_validation_bank.shared_paths_equivalent(
            bank_final_link.get("path", ""), source_bank.get("finalization_path", "")
        )
        and bank_final_link.get("sha256") == source_bank.get("finalization_sha256"),
        "Bundle validation points to another bank finalization",
    )

    matrix_path, matrix_sha = _relative_file(root, calibration["mirror_matrix"], "mirrored latent matrix")
    matrix_record = calibration_payload.get("latent_matrices", {})
    require(matrix_sha == matrix_record.get("file_sha256"), "Mirrored latent matrix file SHA mismatch")
    matrices = np.load(matrix_path, allow_pickle=False)
    require(evaluation_bank.array_sha256(matrices) == matrix_record.get("array_sha256"), "Mirrored latent matrix content SHA mismatch")
    require(list(matrices.shape) == matrix_record.get("shape"), "Mirrored latent matrix shape mismatch")

    return {
        "bundle_manifest_path": str(manifest),
        "bundle_manifest_sha256": expected,
        "scenario": scenario,
        "target_realized_rho": target,
        "calibration_report_path": str(calibration_path),
        "calibration_report_sha256": calibration_sha,
        "validation_report_path": str(validation_path),
        "validation_report_sha256": validation_sha,
        "matrix_path": str(matrix_path),
        "matrix_file_sha256": matrix_sha,
        "matrix_array_sha256": matrix_record.get("array_sha256"),
        "bank_manifest_path": str(bank_manifest_path),
        "bank_manifest_sha256": bank_manifest_sha,
        "bank_array_file_sha256": contract.get("file_sha256"),
        "bank_array_sha256": contract.get("array_sha256"),
    }
