#!/usr/bin/env python3
"""End-to-end small smoke and opt-in distributed FEM scaling probe.

No performance claim is embedded in this program.  A 3,000,000-DoF or 0.5x
memory statement is emitted only when the corresponding command-line gates are
actually met by the recorded run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "scatter3d.validation.fem_smoke/v2"
_PROJECT_IMAGE_DIGEST_ENV = "SCATTER3D_PROJECT_IMAGE_DIGEST"
_PROJECT_IMAGE_ID_ENV = "SCATTER3D_PROJECT_IMAGE_ID"
_BASE_IMAGE_DIGEST_ENV = "SCATTER3D_BASE_IMAGE_DIGEST"
_GIT_COMMIT_ENV = "SCATTER3D_GIT_COMMIT"
_GIT_DIRTY_ENV = "SCATTER3D_GIT_DIRTY"
_FULL_GIT_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


def _status(passed: bool) -> str:
    return "PASSED" if passed else "FAILED"


def _gate(passed: bool, **details: Any) -> dict[str, Any]:
    return {"status": _status(passed), "passed": bool(passed), **details}


def _environment_git_metadata(environment: Mapping[str, str]) -> dict[str, Any]:
    commit_value = environment.get(_GIT_COMMIT_ENV, "").strip()
    dirty_value = environment.get(_GIT_DIRTY_ENV, "").strip().lower()
    if not commit_value and not dirty_value:
        return {"commit": None, "dirty": None, "provenance": "unavailable"}
    if not commit_value or not dirty_value:
        raise ValueError(
            f"{_GIT_COMMIT_ENV} and {_GIT_DIRTY_ENV} must be supplied together"
        )
    if _FULL_GIT_COMMIT.fullmatch(commit_value) is None:
        raise ValueError(f"{_GIT_COMMIT_ENV} must be a full 40-hex Git commit")
    if dirty_value not in {"true", "false"}:
        raise ValueError(f"{_GIT_DIRTY_ENV} must be the explicit boolean true or false")
    return {
        "commit": commit_value.lower(),
        "dirty": dirty_value == "true",
        "provenance": "environment",
        "environment_variables": [_GIT_COMMIT_ENV, _GIT_DIRTY_ENV],
    }


def _git_metadata(
    repository: Path, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Return the exact source revision without guessing when Git is unavailable."""

    environment = os.environ if environment is None else environment

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
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=normal"))
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return _environment_git_metadata(environment)
    if _FULL_GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("git rev-parse HEAD did not return a full 40-hex commit")
    return {
        "commit": commit.lower(),
        "dirty": dirty,
        "provenance": "git_worktree",
    }


def _image_digest_metadata(environment: Mapping[str, str]) -> dict[str, Any]:
    """Read externally supplied image identities; never infer or fabricate them."""

    def supplied(variable: str) -> str | None:
        value = environment.get(variable)
        if value is None or not value.strip():
            return None
        normalized = value.strip().lower()
        if _IMAGE_DIGEST.fullmatch(normalized) is None:
            raise ValueError(f"{variable} must be sha256 followed by 64 hexadecimal digits")
        return normalized

    project_digest = supplied(_PROJECT_IMAGE_DIGEST_ENV)
    project_image_id = supplied(_PROJECT_IMAGE_ID_ENV)
    if project_digest is not None and project_image_id is not None:
        raise ValueError(
            f"supply only one of {_PROJECT_IMAGE_DIGEST_ENV} and {_PROJECT_IMAGE_ID_ENV}"
        )
    if project_digest is not None:
        project = {
            "digest": project_digest,
            "local_image_id": None,
            "provenance": "registry_or_oci_digest_environment",
            "environment_variable": _PROJECT_IMAGE_DIGEST_ENV,
        }
    elif project_image_id is not None:
        project = {
            "digest": None,
            "local_image_id": project_image_id,
            "provenance": "docker_local_image_id_environment",
            "environment_variable": _PROJECT_IMAGE_ID_ENV,
        }
    else:
        project = {
            "digest": None,
            "local_image_id": None,
            "provenance": "not_provided",
            "environment_variable": None,
        }
    base_digest = supplied(_BASE_IMAGE_DIGEST_ENV)
    base = {
        "digest": base_digest,
        "local_image_id": None,
        "provenance": (
            "registry_or_oci_digest_environment"
            if base_digest is not None
            else "not_provided"
        ),
        "environment_variable": (
            _BASE_IMAGE_DIGEST_ENV if base_digest is not None else None
        ),
    }

    return {
        "project_image": project,
        "base_image": base,
    }


def _read_cgroup_integer(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _cgroup_memory_metadata(root: Path = Path("/sys/fs/cgroup")) -> dict[str, Any]:
    """Read cgroup v2 (or legacy v1) peak and limit without claiming availability."""

    candidates = (
        ("v2", root / "memory.peak", root / "memory.max"),
        (
            "v1",
            root / "memory" / "memory.max_usage_in_bytes",
            root / "memory" / "memory.limit_in_bytes",
        ),
    )
    for version, peak_path, limit_path in candidates:
        if peak_path.is_file() or limit_path.is_file():
            return {
                "version": version,
                "peak_bytes": _read_cgroup_integer(peak_path),
                "limit_bytes": _read_cgroup_integer(limit_path),
                "peak_source": str(peak_path) if peak_path.is_file() else None,
                "limit_source": str(limit_path) if limit_path.is_file() else None,
                "provenance": "cgroup_files",
            }
    return {
        "version": None,
        "peak_bytes": None,
        "limit_bytes": None,
        "peak_source": None,
        "limit_source": None,
        "provenance": "unavailable",
    }


def _runtime_metadata(dolfinx: Any, PETSc: Any, MPI: Any) -> dict[str, Any]:
    import mpi4py

    vendor = MPI.get_vendor() if hasattr(MPI, "get_vendor") else None
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "dolfinx_version": dolfinx.__version__,
        "petsc_version": list(PETSc.Sys.getVersion()),
        "petsc_scalar_type": str(PETSc.ScalarType),
        "mpi4py_version": mpi4py.__version__,
        "mpi_vendor": list(vendor) if vendor is not None else None,
        "mpi_library_version": " ".join(MPI.Get_library_version().split()),
        "mpi_world_size": MPI.COMM_WORLD.size,
    }


def _command_metadata(argv: Sequence[str]) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "working_directory": str(Path.cwd()),
        "provenance": "process",
    }


def _canonical_identity(definition: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        definition,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema": "scatter3d.validation.physical_problem/v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "definition": dict(definition),
    }


def _physical_problem_metadata(
    *,
    subdivisions: int,
    frequencies_hz: Sequence[float],
    problem_config: Any,
    material_map: Any,
    contract: Any,
    port_definitions: Sequence[Any],
) -> dict[str, Any]:
    """Identify the physical discretization independently of solver settings."""

    definition = {
        "mesh": {
            "generator": "dolfinx.mesh.create_unit_cube",
            "bounds_m": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "subdivisions_xyz": [subdivisions, subdivisions, subdivisions],
            "geometry_family": "linear-tetrahedral-unit-cube",
        },
        "discretization": problem_config.canonical(),
        "frequencies_hz": [float(value) for value in frequencies_hz],
        "materials": material_map.canonical(),
        "mesh_tags_and_boundaries": contract.canonical(),
        "ports": [definition.canonical() for definition in port_definitions],
        "operator_convention": {
            "equation": "curl(mu_r^-1 curl(E)) - k0^2 epsilon_r_complex E",
            "time_convention": "exp(-i*omega*t)",
            "conductivity_embedding": "epsilon_r_complex=epsilon_r+i*sigma/(omega*epsilon_0)",
            "pec": "tangential-electric-essential-boundary",
            "ports": "matched-single-tem-impedance-boundary",
        },
    }
    return _canonical_identity(definition)


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"comparison JSON is missing mapping {key!r}")
    return value


def _validated_source(payload: Mapping[str, Any], label: str) -> str:
    source = _require_mapping(payload, "source")
    commit = source.get("commit")
    if not isinstance(commit, str) or _FULL_GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError(f"{label} source commit must be a full 40-hex revision")
    if source.get("dirty") is not False:
        raise ValueError(f"{label} source must have explicit dirty=false")
    return commit.lower()


def _validated_images(payload: Mapping[str, Any], label: str) -> dict[str, str]:
    images = _require_mapping(payload, "images")
    result: dict[str, str] = {}
    project = images.get("project_image")
    if not isinstance(project, Mapping):
        raise ValueError(f"{label} is missing project_image identity")
    project_digest = project.get("digest")
    project_image_id = project.get("local_image_id")
    if isinstance(project_digest, str) and project_image_id is None:
        project_kind = "digest"
        project_value = project_digest
    elif isinstance(project_image_id, str) and project_digest is None:
        project_kind = "local_image_id"
        project_value = project_image_id
    else:
        raise ValueError(f"{label} project_image must have exactly one explicit identity")
    if _IMAGE_DIGEST.fullmatch(project_value) is None:
        raise ValueError(f"{label} project_image identity must be sha256:64hex")
    result["project_image"] = f"{project_kind}:{project_value.lower()}"

    base = images.get("base_image")
    if not isinstance(base, Mapping):
        raise ValueError(f"{label} is missing base_image identity")
    base_digest = base.get("digest")
    if not isinstance(base_digest, str) or _IMAGE_DIGEST.fullmatch(base_digest) is None:
        raise ValueError(f"{label} base_image digest must be an explicit sha256 digest")
    if base.get("local_image_id") is not None:
        raise ValueError(f"{label} base_image must be identified by its pinned digest")
    result["base_image"] = f"digest:{base_digest.lower()}"
    return result


def _release_provenance_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Report release/comparison provenance without affecting a general solve gate."""

    try:
        _validated_source(payload, "artifact")
        _validated_images(payload, "artifact")
        if not _require_mapping(payload, "runtime"):
            raise ValueError("runtime identity is empty")
        _validated_physical_problem(payload, "artifact")
    except ValueError as exc:
        return _gate(False, scope="release_and_comparison_only", reason=str(exc))
    return _gate(True, scope="release_and_comparison_only")


def _validated_physical_problem(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    identity = _require_mapping(payload, "physical_problem")
    definition = identity.get("definition")
    digest = identity.get("sha256")
    if identity.get("schema") != "scatter3d.validation.physical_problem/v1":
        raise ValueError(f"{label} physical_problem schema is missing or unsupported")
    if not isinstance(definition, Mapping) or not isinstance(digest, str):
        raise ValueError(f"{label} physical_problem identity is incomplete")
    calculated = _canonical_identity(definition)
    if digest.lower() != calculated["sha256"]:
        raise ValueError(f"{label} physical_problem identity hash is invalid")
    return calculated


def _matrix_identity(payload: Mapping[str, Any], label: str) -> tuple[tuple[float, int, int], ...]:
    diagnostics = payload.get("frequency_diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError(f"{label} frequency diagnostics are missing")
    identity: list[tuple[float, int, int]] = []
    for item in diagnostics:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} frequency diagnostic is malformed")
        frequency = item.get("frequency_hz")
        dofs = item.get("global_complex_dofs")
        nonzeros = item.get("matrix_nonzeros")
        if (
            not isinstance(frequency, int | float)
            or isinstance(frequency, bool)
            or not isinstance(dofs, int)
            or isinstance(dofs, bool)
            or not isinstance(nonzeros, int)
            or isinstance(nonzeros, bool)
            or dofs <= 0
            or nonzeros <= 0
        ):
            raise ValueError(f"{label} physical A identity is incomplete")
        identity.append((float(frequency), dofs, nonzeros))
    return tuple(identity)


def _validate_direct_comparison(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> None:
    """Fail closed unless direct and iterative memory runs are truly comparable."""

    if baseline.get("solver") != "direct":
        raise ValueError("comparison JSON must be from a direct run")
    if baseline.get("status") != "PASSED" or baseline.get("passed") is not True:
        raise ValueError("direct comparison overall status must be PASSED")
    convergence = _require_mapping(
        _require_mapping(baseline, "gates"), "convergence_and_true_residual"
    )
    if convergence.get("status") != "PASSED" or convergence.get("passed") is not True:
        raise ValueError("direct comparison convergence gate must be PASSED")
    if current.get("mpi_size") != baseline.get("mpi_size") or not isinstance(
        current.get("mpi_size"), int
    ):
        raise ValueError("direct comparison must use the same explicit MPI size")
    if _validated_source(current, "iterative") != _validated_source(
        baseline, "direct"
    ):
        raise ValueError("direct comparison must use the same source revision")
    if _validated_images(current, "iterative") != _validated_images(baseline, "direct"):
        raise ValueError("direct comparison must use identical project and base images")
    current_runtime = _require_mapping(current, "runtime")
    baseline_runtime = _require_mapping(baseline, "runtime")
    if not current_runtime or current_runtime != baseline_runtime:
        raise ValueError("direct comparison must use identical runtime identity")
    if _validated_physical_problem(current, "iterative") != _validated_physical_problem(
        baseline, "direct"
    ):
        raise ValueError("direct comparison must use the same physical_problem identity")
    if _matrix_identity(current, "iterative") != _matrix_identity(baseline, "direct"):
        raise ValueError(
            "direct comparison must have identical frequencies, global DoFs, and physical A nonzeros"
        )


def _json_safe(value: Any) -> Any:
    """Preserve nonfinite diagnostics explicitly without emitting invalid JSON numbers."""

    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _write_payload(
    payload: Mapping[str, Any], path: Path, *, overwrite: bool = False
) -> Path:
    from scatter3d.pipeline import write_json_report

    destination = write_json_report(_json_safe(payload), path, overwrite=overwrite)
    if os.name == "posix":
        destination.chmod(0o644)
    return destination


def _tagged_cube(comm, subdivisions: int):
    from dolfinx import mesh

    domain = mesh.create_unit_cube(comm, subdivisions, subdivisions, subdivisions)
    tdim = domain.topology.dim
    fdim = tdim - 1
    local_cells = domain.topology.index_map(tdim).size_local
    cells = np.arange(local_cells, dtype=np.int32)
    cell_tags = mesh.meshtags(
        domain, tdim, cells, np.ones(local_cells, dtype=np.int32)
    )
    left = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0))
    right = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1))
    exterior = mesh.exterior_facet_indices(domain.topology)
    walls = np.setdiff1d(exterior, np.concatenate((left, right)), assume_unique=False)
    indices = np.concatenate((left, right, walls)).astype(np.int32)
    values = np.concatenate(
        (
            np.full(left.size, 10, dtype=np.int32),
            np.full(right.size, 11, dtype=np.int32),
            np.full(walls.size, 20, dtype=np.int32),
        )
    )
    order = np.argsort(indices)
    facet_tags = mesh.meshtags(domain, fdim, indices[order], values[order])
    return domain, cell_tags, facet_tags


def _memory_high_water(payload: dict) -> int | None:
    values = [
        item["rank_peak_rss_bytes_sum"]
        for item in payload["frequency_diagnostics"]
        if item["rank_peak_rss_bytes_sum"] is not None
    ]
    return max(values) if values else None


def _validate_hierarchy_configuration(
    *,
    solver: str,
    hierarchy: str,
    fine_degree: int,
    coarse_degree: int | None,
    iterative_local_pc: str | None,
) -> bool:
    """Validate solver hierarchy CLI values before importing the FEM runtime."""

    if solver == "direct":
        if hierarchy != "one-level-asm":
            raise ValueError("--iterative-hierarchy requires --solver iterative")
        if coarse_degree is not None:
            raise ValueError(
                "--p-multigrid-coarse-degree requires "
                "--solver iterative --iterative-hierarchy p-multigrid"
            )
        return False
    if hierarchy == "one-level-asm":
        if coarse_degree is not None:
            raise ValueError(
                "--p-multigrid-coarse-degree requires "
                "--iterative-hierarchy p-multigrid"
            )
        return False
    if coarse_degree is None:
        raise ValueError(
            "--iterative-hierarchy p-multigrid requires "
            "--p-multigrid-coarse-degree"
        )
    if coarse_degree >= fine_degree:
        raise ValueError("p-multigrid coarse degree must be lower than --degree")
    if iterative_local_pc not in (None, "lu"):
        raise ValueError(
            "p-multigrid uses the verified fine ASM local LU/MUMPS path; "
            "--iterative-local-pc must be omitted or set to lu"
        )
    return True


def _expected_solver_counts(
    *,
    frequencies: int,
    ports: int,
    solver: str,
    uses_p_multigrid: bool,
    absorption_shift: float,
) -> dict[str, int]:
    return {
        "matrix_assemblies": frequencies,
        "preconditioner_matrix_assemblies": (
            frequencies
            if solver == "iterative" and absorption_shift != 0.0
            else 0
        ),
        "coarse_preconditioner_matrix_assemblies": (
            frequencies if uses_p_multigrid else 0
        ),
        "transfer_operator_assemblies": 1 if uses_p_multigrid else 0,
        "operator_setups": frequencies,
        "global_numeric_factorizations": frequencies if solver == "direct" else 0,
        "coarse_global_factorizations": frequencies if uses_p_multigrid else 0,
        "rhs_solves": frequencies * ports,
    }


def _observed_solver_counts(result: Any) -> dict[str, int]:
    return {
        key: int(getattr(result, key))
        for key in (
            "matrix_assemblies",
            "preconditioner_matrix_assemblies",
            "coarse_preconditioner_matrix_assemblies",
            "transfer_operator_assemblies",
            "operator_setups",
            "global_numeric_factorizations",
            "coarse_global_factorizations",
            "rhs_solves",
        )
    }


def _p_multigrid_gate(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
    coarse_degree: int | None,
    asm_overlap: int = 1,
) -> dict[str, Any]:
    if not enabled:
        return {
            "status": "NOT RUN",
            "passed": None,
            "reason": "--iterative-hierarchy p-multigrid was not selected",
        }
    def effective_hierarchy_ok(item: Mapping[str, Any]) -> bool:
        hierarchy = item.get("solver_hierarchy")
        if not isinstance(hierarchy, Mapping):
            return False
        effective = hierarchy.get("effective")
        if not isinstance(effective, Mapping):
            return False
        top = effective.get("top_level")
        fine = effective.get("mg_fine_smoother")
        coarse = effective.get("mg_coarse_solver")
        subdomains = effective.get("mg_fine_asm_subdomain_solvers")
        return (
            isinstance(top, Mapping)
            and top.get("ksp_type") == "fgmres"
            and top.get("pc_type") == "mg"
            and effective.get("preconditioning_side") == "right"
            and effective.get("pc_uses_amat") is False
            and effective.get("mg_levels") == 2
            and effective.get("mg_type") == "multiplicative"
            and effective.get("mg_cycle_type") == "v"
            and effective.get("mg_galerkin") == "none"
            and isinstance(fine, Mapping)
            and fine.get("ksp_type") == "richardson"
            and fine.get("pc_type") == "asm"
            and fine.get("maximum_iterations") == 1
            and fine.get("norm_type") == "none"
            and effective.get("mg_fine_asm_type") == "restrict"
            and effective.get("mg_fine_asm_overlap") == asm_overlap
            and isinstance(subdomains, Sequence)
            and bool(subdomains)
            and all(
                isinstance(component, Mapping)
                and component.get("ksp_type") == "preonly"
                and component.get("pc_type") == "lu"
                and component.get("factor_solver_type") == "mumps"
                for component in subdomains
            )
            and isinstance(coarse, Mapping)
            and coarse.get("ksp_type") == "preonly"
            and coarse.get("pc_type") == "lu"
            and coarse.get("factor_solver_type") == "mumps"
        )

    passed = bool(diagnostics) and all(
        item.get("coarse_degree") == coarse_degree
        and isinstance(item.get("coarse_global_complex_dofs"), int)
        and item["coarse_global_complex_dofs"] > 0
        and item["coarse_global_complex_dofs"] < item.get("global_complex_dofs", 0)
        and isinstance(item.get("coarse_preconditioner_matrix_nonzeros"), int)
        and item["coarse_preconditioner_matrix_nonzeros"] > 0
        and item.get("p_multigrid_operator_checks_passed") is True
        and isinstance(item.get("transfer_operator"), Mapping)
        and item["transfer_operator"].get("direction") == "coarse_to_fine"
        and item["transfer_operator"].get("rows") == item.get("global_complex_dofs")
        and item["transfer_operator"].get("columns")
        == item.get("coarse_global_complex_dofs")
        and isinstance(item["transfer_operator"].get("nonzeros"), int)
        and item["transfer_operator"]["nonzeros"] > 0
        and isinstance(
            item["transfer_operator"].get("constrained_fine_rows"), int
        )
        and item["transfer_operator"]["constrained_fine_rows"] > 0
        and isinstance(
            item["transfer_operator"].get("constrained_coarse_columns"), int
        )
        and item["transfer_operator"]["constrained_coarse_columns"] > 0
        and item["transfer_operator"].get("maximum_imaginary_abs") == 0.0
        and effective_hierarchy_ok(item)
        for item in diagnostics
    )
    return _gate(passed, requested_coarse_degree=coarse_degree)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("direct", "iterative"), default="direct")
    parser.add_argument("--degree", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--subdivisions", type=int, default=3)
    parser.add_argument("--frequencies-hz", type=float, nargs="+", default=(1.0e8, 1.2e8))
    parser.add_argument("--minimum-global-dofs", type=int, default=0)
    parser.add_argument("--maximum-true-relative-residual", type=float, default=1.0e-7)
    parser.add_argument("--maximum-iterations", type=int, default=1_000)
    parser.add_argument("--gmres-restart", type=int, default=80)
    parser.add_argument("--asm-overlap", type=int, default=1)
    parser.add_argument(
        "--iterative-hierarchy",
        choices=("one-level-asm", "p-multigrid"),
        default="one-level-asm",
    )
    parser.add_argument(
        "--p-multigrid-coarse-degree",
        type=int,
        choices=(1, 2),
        help="explicit assembled low-order degree for the p-multigrid coarse level",
    )
    parser.add_argument(
        "--preconditioner-absorption-shift",
        type=float,
        default=0.0,
        help="nonnegative absorption shift applied only to the iterative P matrix",
    )
    parser.add_argument("--iterative-local-pc", choices=("ilu", "lu"))
    parser.add_argument("--compare-direct-json", type=Path)
    parser.add_argument("--maximum-memory-ratio", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing output artifact",
    )
    args = parser.parse_args()
    if args.subdivisions < 1:
        parser.error("subdivisions must be positive")
    if any(value <= 0 for value in args.frequencies_hz) or any(
        b <= a for a, b in zip(args.frequencies_hz, args.frequencies_hz[1:], strict=False)
    ):
        parser.error("frequencies must be positive and strictly increasing")
    if args.compare_direct_json and args.solver != "iterative":
        parser.error("--compare-direct-json is meaningful only for the iterative run")
    if args.maximum_iterations < 1 or args.gmres_restart < 1 or args.asm_overlap < 0:
        parser.error("iteration/restart counts must be positive and overlap nonnegative")
    if (
        not math.isfinite(args.preconditioner_absorption_shift)
        or args.preconditioner_absorption_shift < 0
    ):
        parser.error("--preconditioner-absorption-shift must be finite and nonnegative")
    if args.solver == "direct" and args.preconditioner_absorption_shift != 0.0:
        parser.error("--preconditioner-absorption-shift requires --solver iterative")
    try:
        uses_p_multigrid = _validate_hierarchy_configuration(
            solver=args.solver,
            hierarchy=args.iterative_hierarchy,
            fine_degree=args.degree,
            coarse_degree=args.p_multigrid_coarse_degree,
            iterative_local_pc=args.iterative_local_pc,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing artifact: {args.output.resolve()}")

    import dolfinx
    from dolfinx import fem
    from mpi4py import MPI
    from petsc4py import PETSc

    from scatter3d.fem.config import (
        LinearSolverConfig,
        Material,
        MaterialMap,
        MaxwellProblemConfig,
    )
    from scatter3d.fem.ports import (
        MatchedTEMPortExcitation,
        PortDefinition,
        normalize_port_mode,
    )
    from scatter3d.fem.solver import MaxwellSweepSolver
    from scatter3d.fem.tags import (
        BoundaryTagContract,
        MeshTagContract,
        VolumeTagContract,
    )

    domain, cell_tags, facet_tags = _tagged_cube(MPI.COMM_WORLD, args.subdivisions)
    contract = MeshTagContract(
        VolumeTagContract({"domain": 1}),
        BoundaryTagContract(ports={"left": 10, "right": 11}, pec_tags=(20,)),
    )
    if args.solver == "direct":
        solver_config = LinearSolverConfig.direct()
    elif uses_p_multigrid:
        solver_config = LinearSolverConfig.iterative_p_multigrid(
            coarse_degree=args.p_multigrid_coarse_degree,
            maximum_iterations=args.maximum_iterations,
            preconditioner_absorption_shift=args.preconditioner_absorption_shift,
            error_if_not_converged=False,
            petsc_options={
                "ksp_gmres_restart": args.gmres_restart,
                "mg_levels_1_pc_asm_overlap": args.asm_overlap,
            },
        )
    else:
        iterative_local_pc = args.iterative_local_pc or "ilu"
        local_options: dict[str, str | int] = {
            "ksp_gmres_restart": args.gmres_restart,
            "pc_asm_overlap": args.asm_overlap,
            "sub_ksp_type": "preonly",
            "sub_pc_type": iterative_local_pc,
        }
        if iterative_local_pc == "ilu":
            local_options["sub_pc_factor_levels"] = 0
        else:
            local_options["sub_pc_factor_mat_solver_type"] = "mumps"
        solver_config = LinearSolverConfig.iterative_maxwell(
            maximum_iterations=args.maximum_iterations,
            preconditioner_absorption_shift=args.preconditioner_absorption_shift,
            error_if_not_converged=False,
            petsc_options=local_options,
        )
    definitions = tuple(
        PortDefinition(
            name,
            tag,
            field_wave_impedance_ohm=200.0,
            outgoing_propagation_index=1.0,
        )
        for name, tag in (("left", 10), ("right", 11))
    )
    material_map = MaterialMap(
        Material(2.0, conductivity_s_per_m=0.02, name="lossy")
    )
    problem_config = MaxwellProblemConfig(polynomial_degree=args.degree)
    solver = MaxwellSweepSolver.from_mesh(
        domain,
        cell_tags,
        facet_tags,
        contract,
        material_map,
        problem_config,
        matched_ports=definitions,
        solver_config=solver_config,
        initial_frequency_hz=args.frequencies_hz[0],
    )
    excitations = []
    for definition in definitions:
        raw_mode = fem.Function(
            solver.function_space, name=f"mode_raw_{definition.name}"
        )
        raw_mode.interpolate(
            lambda x: np.vstack(
                (
                    np.zeros(x.shape[1], dtype=PETSc.ScalarType),
                    np.ones(x.shape[1], dtype=PETSc.ScalarType),
                    np.zeros(x.shape[1], dtype=PETSc.ScalarType),
                )
            )
        )
        mode = normalize_port_mode(raw_mode, facet_tags, definition)
        excitations.append(MatchedTEMPortExcitation(mode))

    result = solver.solve(
        args.frequencies_hz, excitations, retain_solutions=False
    )
    diagnostics = [asdict(item) for item in result.diagnostics]
    convergence_ok = all(
        port["converged_reason"] > 0
        and port["true_relative_residual"] <= args.maximum_true_relative_residual
        for frequency in diagnostics
        for port in frequency["port_solves"]
    )
    expected_counts = _expected_solver_counts(
        frequencies=len(args.frequencies_hz),
        ports=len(excitations),
        solver=args.solver,
        uses_p_multigrid=uses_p_multigrid,
        absorption_shift=args.preconditioner_absorption_shift,
    )
    observed_counts = _observed_solver_counts(result)
    count_ok = observed_counts == expected_counts
    p_multigrid_gate = _p_multigrid_gate(
        diagnostics,
        enabled=uses_p_multigrid,
        coarse_degree=args.p_multigrid_coarse_degree,
        asm_overlap=args.asm_overlap,
    )
    p_multigrid_ok = not uses_p_multigrid or p_multigrid_gate["passed"] is True
    global_dofs = diagnostics[0]["global_complex_dofs"]
    dof_gate = global_dofs >= args.minimum_global_dofs
    requested_solver = {
        "solver": args.solver,
        "iterative_hierarchy": (
            args.iterative_hierarchy if args.solver == "iterative" else None
        ),
        **solver_config.canonical(),
    }
    effective_solver = [
        {
            "frequency_hz": item["frequency_hz"],
            "hierarchy": item["solver_hierarchy"]["effective"],
        }
        for item in diagnostics
    ]
    preconditioner_metrics = [
        {
            "frequency_hz": item["frequency_hz"],
            "absorption_shift": item["preconditioner_absorption_shift"],
            "operator_is_physical": item["preconditioner_operator_is_physical"],
            "matrix_nonzeros": item["preconditioner_matrix_nonzeros"],
            "matrix_memory_bytes_sum": item[
                "preconditioner_matrix_memory_bytes_sum"
            ],
            "assembly_seconds": item["preconditioner_assembly_seconds"],
            "coarse": {
                "degree": item["coarse_degree"],
                "global_complex_dofs": item["coarse_global_complex_dofs"],
                "matrix_nonzeros": item[
                    "coarse_preconditioner_matrix_nonzeros"
                ],
                "matrix_memory_bytes_sum": item[
                    "coarse_preconditioner_matrix_memory_bytes_sum"
                ],
                "assembly_seconds": item[
                    "coarse_preconditioner_assembly_seconds"
                ],
            },
            "transfer_operator": item["transfer_operator"],
            "p_multigrid_operator_checks_passed": item[
                "p_multigrid_operator_checks_passed"
            ],
        }
        for item in diagnostics
    ]
    repository = Path(__file__).resolve().parents[1]
    payload = {
        "schema": SCHEMA,
        "status": "FAILED",
        "passed": False,
        "source": _git_metadata(repository),
        "command": _command_metadata(sys.argv),
        "images": _image_digest_metadata(os.environ),
        "runtime": _runtime_metadata(dolfinx, PETSc, MPI),
        "cgroup_memory": _cgroup_memory_metadata(),
        "physical_problem": _physical_problem_metadata(
            subdivisions=args.subdivisions,
            frequencies_hz=args.frequencies_hz,
            problem_config=problem_config,
            material_map=material_map,
            contract=contract,
            port_definitions=definitions,
        ),
        "solver": args.solver,
        "iterative_hierarchy": (
            args.iterative_hierarchy if args.solver == "iterative" else None
        ),
        # Retained as an explicit compatibility alias for v1 artifact readers.
        "solver_config": solver_config.canonical(),
        "solver_configuration": {
            "requested": requested_solver,
            "effective_by_frequency": effective_solver,
        },
        "degree": args.degree,
        "geometry_order": 1,
        "subdivisions": args.subdivisions,
        "frequencies_hz": list(args.frequencies_hz),
        "mpi_size": MPI.COMM_WORLD.size,
        "dolfinx_version": dolfinx.__version__,
        "petsc_version": PETSc.Sys.getVersion(),
        "scalar_type": str(PETSc.ScalarType),
        "matrix_assemblies": result.matrix_assemblies,
        "preconditioner_matrix_assemblies": result.preconditioner_matrix_assemblies,
        "coarse_preconditioner_matrix_assemblies": (
            result.coarse_preconditioner_matrix_assemblies
        ),
        "transfer_operator_assemblies": result.transfer_operator_assemblies,
        "operator_setups": result.operator_setups,
        "global_numeric_factorizations": result.global_numeric_factorizations,
        "coarse_global_factorizations": result.coarse_global_factorizations,
        # Backward-compatible v1 alias. This never includes ASM-local factors.
        "numeric_factorizations": result.global_numeric_factorizations,
        "rhs_solves": result.rhs_solves,
        "preconditioner": {
            "requested_absorption_shift": args.preconditioner_absorption_shift,
            "requested_hierarchy": (
                args.iterative_hierarchy if args.solver == "iterative" else None
            ),
            "requested_coarse_degree": args.p_multigrid_coarse_degree,
            "per_frequency": preconditioner_metrics,
        },
        "frequency_diagnostics": diagnostics,
        "gates": {
            "convergence_and_true_residual": _gate(
                convergence_ok,
                maximum_true_relative_residual=args.maximum_true_relative_residual,
            ),
            "assembly_setup_rhs_counts": _gate(
                count_ok,
                expected=expected_counts,
                observed=observed_counts,
            ),
            "p_multigrid_structure": p_multigrid_gate,
            "minimum_global_dofs": _gate(
                dof_gate,
                requested=args.minimum_global_dofs,
                observed=global_dofs,
            ),
        },
    }
    payload["gates"]["release_comparison_provenance"] = _release_provenance_gate(
        payload
    )

    memory_ratio_ok = True
    if args.compare_direct_json:
        baseline = json.loads(args.compare_direct_json.read_text(encoding="utf-8"))
        _validate_direct_comparison(payload, baseline)
        current_rss = _memory_high_water(payload)
        baseline_rss = _memory_high_water(baseline)
        if current_rss is None or baseline_rss is None or baseline_rss <= 0:
            raise ValueError("both runs must contain process peak-RSS instrumentation")
        ratio = current_rss / baseline_rss
        memory_ratio_ok = ratio <= args.maximum_memory_ratio
        payload["memory_comparison"] = {
            "metric": "sum_of_rank_process_high_water_rss_bytes",
            "iterative_bytes": current_rss,
            "direct_bytes": baseline_rss,
            "ratio": ratio,
            "maximum_ratio": args.maximum_memory_ratio,
            "status": _status(memory_ratio_ok),
            "passed": memory_ratio_ok,
            "note": "Scheduler/job MaxRSS is preferred for a publication claim.",
        }
    else:
        payload["memory_comparison"] = {
            "status": "NOT RUN",
            "passed": None,
            "reason": "--compare-direct-json was not supplied",
        }

    passed = (
        convergence_ok
        and count_ok
        and p_multigrid_ok
        and dof_gate
        and memory_ratio_ok
    )
    payload["status"] = _status(passed)
    payload["passed"] = passed
    if MPI.COMM_WORLD.rank == 0:
        safe_payload = _json_safe(payload)
        rendered = json.dumps(
            safe_payload, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        print(rendered, end="")
        _write_payload(safe_payload, args.output, overwrite=args.overwrite)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
