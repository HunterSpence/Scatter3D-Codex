#!/usr/bin/env python3
"""Validate and write-once register the fixed Scatter3D scaling sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SPEC_SCHEMA = "scatter3d.validation.scaling_sweep/v1"
REGISTRATION_SCHEMA = "scatter3d.validation.scaling_sweep_registration/v1"
STATUS_VOCABULARY = ("PASSED", "FAILED", "NOT RUN", "BLOCKED")
_FULL_GIT_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_IDENTITY = re.compile(r"^sha256:[0-9a-fA-F]{64}$")

_EXPECTED_SPEC: dict[str, Any] = {
    "schema": SPEC_SCHEMA,
    "canonical_path": "validation/scaling_sweep_v1.json",
    "status_vocabulary": list(STATUS_VOCABULARY),
    "physical_problem": {
        "fine_degree": 3,
        "coarse_degree": 1,
        "frequency_hz": 100_000_000.0,
        "ports": ["left", "right"],
    },
    "solver": {
        "hierarchy": "p-multigrid",
        "outer_ksp": "fgmres",
        "preconditioning_side": "right",
        "maximum_true_relative_residual": 1.0e-7,
        "maximum_iterations": 1_000,
        "asm_overlap": 1,
        "fine_local_ksp": "preonly",
        "fine_local_pc": "lu",
        "fine_local_factor_solver": "mumps",
        "coarse_ksp": "preonly",
        "coarse_pc": "lu",
        "coarse_factor_solver": "mumps",
    },
    "preconditioner_absorption_shifts": [0.0, 0.25, 0.5, 1.0],
    "cgroup_memory_limit_bytes": 28 * 1024**3,
    "wall_time_limit_seconds": 10_800,
    "rungs": [
        {
            "id": "p3-n9-mpi4",
            "subdivisions": 9,
            "mpi_ranks": 4,
            "gmres_restart": 80,
            "expected_global_complex_dofs": 86_103,
        },
        {
            "id": "p3-n16-mpi8",
            "subdivisions": 16,
            "mpi_ranks": 8,
            "gmres_restart": 100,
            "expected_global_complex_dofs": 470_928,
        },
    ],
}


def validate_scaling_sweep_spec(spec: Mapping[str, Any]) -> None:
    """Fail closed unless the document is the exact registered v1 experiment."""

    if dict(spec) != _EXPECTED_SPEC:
        raise ValueError(
            "scaling sweep specification differs from the immutable v1 contract"
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _validated_source(source: Mapping[str, Any]) -> dict[str, Any]:
    commit = source.get("commit")
    dirty = source.get("dirty")
    if not isinstance(commit, str) or _FULL_GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("source commit must be a full 40-hex Git revision")
    if not isinstance(dirty, bool):
        raise ValueError("source dirty state must be an explicit boolean")
    if dirty:
        raise ValueError("scaling sweep source dirty state must be exactly false")
    return {"commit": commit.lower(), "dirty": False}


def _validated_image_identity(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_IDENTITY.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be sha256 followed by 64 hexadecimal digits")
    return normalized


def _validated_runtime_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    if not normalized:
        raise ValueError("base runtime metadata must be a nonempty JSON object")
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("base runtime metadata must be strict JSON") from exc
    return normalized


def _git_source(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()

    try:
        source = {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(
                run("status", "--porcelain", "--untracked-files=normal")
            ),
        }
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        raise ValueError(f"cannot establish exact Git source identity: {exc}") from exc
    return _validated_source(source)


def _shift_label(shift: float) -> str:
    return f"{shift:.2f}".replace(".", "p")


def _cli_float(value: float) -> str:
    return str(float(value))


def _expected_physical_problem(
    *, subdivisions: int, frequency_hz: float, degree: int
) -> dict[str, Any]:
    """Return the complete physical identity expected from fem_smoke."""

    ports = [
        {
            "name": name,
            "facet_tag": tag,
            "field_wave_impedance_ohm": [200.0, 0.0],
            "outgoing_propagation_index": [1.0, 0.0],
            "circuit_reference_impedance_ohm": 50.0,
            "target_forward_power_w": 1.0,
            "mode_model": "matched-single-tem",
            "time_convention": "exp(-i*omega*t)",
        }
        for name, tag in (("left", 10), ("right", 11))
    ]
    definition = {
        "mesh": {
            "generator": "dolfinx.mesh.create_unit_cube",
            "bounds_m": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "subdivisions_xyz": [subdivisions, subdivisions, subdivisions],
            "geometry_family": "linear-tetrahedral-unit-cube",
        },
        "discretization": {
            "polynomial_degree": degree,
            "geometry_order": 1,
            "quadrature_degree": max(2 * degree + 2, 6),
            "time_convention": "exp(-i*omega*t)",
        },
        "frequencies_hz": [float(frequency_hz)],
        "materials": {
            "default": {
                "name": "lossy",
                "relative_permittivity": [2.0, 0.0],
                "relative_permeability": [1.0, 0.0],
                "conductivity_s_per_m": 0.02,
            },
            "regions": {},
        },
        "mesh_tags_and_boundaries": {
            "volumes": {"volumes": {"domain": 1}, "pml_names": []},
            "boundaries": {
                "ports": {"left": 10, "right": 11},
                "pec_tags": [20],
                "observation_tags": [],
            },
        },
        "ports": ports,
        "operator_convention": {
            "equation": "curl(mu_r^-1 curl(E)) - k0^2 epsilon_r_complex E",
            "time_convention": "exp(-i*omega*t)",
            "conductivity_embedding": (
                "epsilon_r_complex=epsilon_r+i*sigma/(omega*epsilon_0)"
            ),
            "pec": "tangential-electric-essential-boundary",
            "ports": "matched-single-tem-impedance-boundary",
        },
    }
    return {
        "schema": "scatter3d.validation.physical_problem/v1",
        "sha256": _canonical_sha256(definition),
        "definition": definition,
    }


def _solver_command(
    *,
    spec: Mapping[str, Any],
    rung: Mapping[str, Any],
    shift: float,
    output: str,
) -> list[str]:
    physical = spec["physical_problem"]
    solver = spec["solver"]
    return [
        "mpirun",
        "-n",
        str(rung["mpi_ranks"]),
        "python3",
        "validation/fem_smoke.py",
        "--solver",
        "iterative",
        "--iterative-hierarchy",
        "p-multigrid",
        "--degree",
        str(physical["fine_degree"]),
        "--p-multigrid-coarse-degree",
        str(physical["coarse_degree"]),
        "--subdivisions",
        str(rung["subdivisions"]),
        "--frequencies-hz",
        _cli_float(physical["frequency_hz"]),
        "--expected-global-dofs",
        str(rung["expected_global_complex_dofs"]),
        "--maximum-true-relative-residual",
        _cli_float(solver["maximum_true_relative_residual"]),
        "--maximum-iterations",
        str(solver["maximum_iterations"]),
        "--gmres-restart",
        str(rung["gmres_restart"]),
        "--asm-overlap",
        str(solver["asm_overlap"]),
        "--iterative-local-pc",
        "lu",
        "--preconditioner-absorption-shift",
        _cli_float(shift),
        "--output",
        output,
    ]


def build_registration(
    spec: Mapping[str, Any],
    spec_bytes: bytes,
    *,
    source: Mapping[str, Any],
    project_image_kind: str,
    project_image_identity: str,
    base_image_digest: str,
    base_runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand the immutable specification into deterministic initial entries."""

    validate_scaling_sweep_spec(spec)
    if project_image_kind not in {"oci_digest", "local_image_id"}:
        raise ValueError("project image kind must be oci_digest or local_image_id")
    validated_source = _validated_source(source)
    project_identity = _validated_image_identity(
        project_image_identity, "project image identity"
    )
    base_identity = _validated_image_identity(base_image_digest, "base image digest")
    runtime_metadata = _validated_runtime_metadata(base_runtime_metadata)
    runtime_metadata_sha256 = _canonical_sha256(runtime_metadata)
    spec_sha256 = _sha256_bytes(spec_bytes)
    images = {
        "project_image": {
            "kind": project_image_kind,
            "identity": project_identity,
        },
        "base_image": {
            "kind": "oci_digest",
            "identity": base_identity,
            "runtime_metadata": runtime_metadata,
            "runtime_metadata_sha256": runtime_metadata_sha256,
        },
    }
    registration_id = _canonical_sha256(
        {
            "specification_sha256": spec_sha256,
            "source": validated_source,
            "images": images,
        }
    )
    output_root = f"scaling-sweeps/{registration_id}"
    entries: list[dict[str, Any]] = []
    for rung in spec["rungs"]:
        for shift_value in spec["preconditioner_absorption_shifts"]:
            shift = float(shift_value)
            run_id = f"{rung['id']}-shift-{_shift_label(shift)}"
            run_root = f"{output_root}/{run_id}"
            container_run_root = f"/artifacts/{run_id}"
            fem_smoke_json = f"{run_root}/fem-smoke.json"
            container_fem_smoke_json = f"{container_run_root}/fem-smoke.json"
            entries.append(
                {
                    "run_id": run_id,
                    "rung_id": rung["id"],
                    "preconditioner_absorption_shift": shift,
                    "status": "NOT RUN",
                    "passed": None,
                    "resource_contract": {
                        "mpi_ranks": rung["mpi_ranks"],
                        "cgroup_memory_limit_bytes": spec[
                            "cgroup_memory_limit_bytes"
                        ],
                        "wall_time_limit_seconds": spec[
                            "wall_time_limit_seconds"
                        ],
                    },
                    "physical_problem": _expected_physical_problem(
                        subdivisions=rung["subdivisions"],
                        frequency_hz=spec["physical_problem"]["frequency_hz"],
                        degree=spec["physical_problem"]["fine_degree"],
                    ),
                    "command": _solver_command(
                        spec=spec,
                        rung=rung,
                        shift=shift,
                        output=container_fem_smoke_json,
                    ),
                    "outputs": {
                        "directory": run_root,
                        "container_directory": container_run_root,
                        "fem_smoke_json": fem_smoke_json,
                        "container_fem_smoke_json": container_fem_smoke_json,
                        "stdout_log": f"{run_root}/stdout.log",
                        "stderr_log": f"{run_root}/stderr.log",
                        "exit_code_json": f"{run_root}/exit-code.json",
                        "sha256_manifest": f"{run_root}/SHA256SUMS",
                    },
                }
            )
    return {
        "schema": REGISTRATION_SCHEMA,
        "registration_id": registration_id,
        "status_vocabulary": list(STATUS_VOCABULARY),
        "specification": {
            "schema": spec["schema"],
            "canonical_path": spec["canonical_path"],
            "sha256": spec_sha256,
        },
        "source": validated_source,
        "images": images,
        "output_root": output_root,
        "registration_contract": {
            "must_exist_before_first_solve": True,
            "entries_are_initial_and_immutable": True,
            "results_are_written_separately": True,
            "overwrite_permitted": False,
        },
        "entries": entries,
    }


def write_registration(payload: Mapping[str, Any], path: Path) -> Path:
    """Atomically publish a registration without ever replacing an existing file."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        delete=False,
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    if os.name == "posix":
        temporary.chmod(0o644)
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing scaling sweep registration: {destination}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> int:
    repository_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).with_name("scaling_sweep_v1.json"),
    )
    parser.add_argument("--repository", type=Path, default=repository_default)
    parser.add_argument(
        "--project-image-kind",
        choices=("oci_digest", "local_image_id"),
        required=True,
    )
    parser.add_argument("--project-image-identity", required=True)
    parser.add_argument("--base-image-digest", required=True)
    parser.add_argument("--base-runtime-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        spec_bytes = args.spec.read_bytes()
        spec = json.loads(spec_bytes)
        if not isinstance(spec, Mapping):
            raise ValueError("scaling sweep specification must be a JSON object")
        runtime_metadata = json.loads(args.base_runtime_metadata.read_bytes())
        if not isinstance(runtime_metadata, Mapping):
            raise ValueError("base runtime metadata must be a JSON object")
        payload = build_registration(
            spec,
            spec_bytes,
            source=_git_source(args.repository),
            project_image_kind=args.project_image_kind,
            project_image_identity=args.project_image_identity,
            base_image_digest=args.base_image_digest,
            base_runtime_metadata=runtime_metadata,
        )
        destination = write_registration(payload, args.output)
    except (FileExistsError, FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
