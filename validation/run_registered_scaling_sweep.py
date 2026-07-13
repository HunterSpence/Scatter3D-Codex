#!/usr/bin/env python3
"""Execute an immutable registered scaling sweep through host-side Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from validation import register_scaling_sweep

RESULT_SCHEMA = "scatter3d.validation.scaling_sweep_run_result/v1"
FEM_SMOKE_SCHEMA = "scatter3d.validation.fem_smoke/v2"


def _parse_container_inspection(
    payload: bytes,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    try:
        candidate = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("docker inspect returned malformed JSON") from exc
    if not isinstance(candidate, Mapping):
        raise ValueError("docker inspect did not return an object")
    state = candidate.get("State")
    host_config = candidate.get("HostConfig")
    if not isinstance(state, Mapping) or not isinstance(host_config, Mapping):
        raise ValueError("docker inspect is missing State or HostConfig")
    for key in ("Running", "OOMKilled"):
        if not isinstance(state.get(key), bool):
            raise ValueError(f"docker inspect State.{key} must be boolean")
    exit_code = state.get("ExitCode")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ValueError("docker inspect State.ExitCode must be an integer")
    pid = state.get("Pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 0:
        raise ValueError("docker inspect State.Pid must be a nonnegative integer")
    for key in ("StartedAt", "FinishedAt"):
        if not isinstance(state.get(key), str):
            raise ValueError(f"docker inspect State.{key} must be a string")
    return state, host_config


def _resolve_host_cgroup(pid: int) -> Path:
    """Resolve a running container's unified host cgroup without layout guesses."""

    if pid <= 0:
        raise ValueError("running container PID must be positive")
    lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    relative = None
    for line in lines:
        pieces = line.split(":", 2)
        if len(pieces) == 3 and pieces[0] == "0" and pieces[1] == "":
            relative = pieces[2]
            break
    if relative is None:
        raise ValueError("container has no unified cgroup-v2 path")

    return _resolve_cgroup_relative_path(relative, Path("/sys/fs/cgroup"))


def _resolve_cgroup_relative_path(relative: str, root: Path) -> Path:
    """Resolve one cgroup-v2 relative path beneath a caller-supplied root."""

    parts = Path(relative.lstrip("/")).parts
    if not parts or ".." in parts:
        raise ValueError("container cgroup path is empty or contains traversal")
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    if not candidate.is_relative_to(resolved_root) or not candidate.is_dir():
        raise ValueError("container cgroup path escapes or is absent from host root")
    return candidate


def _read_cgroup_v2_metrics(path: Path) -> dict[str, Any]:
    def read_limit(name: str) -> int | str:
        raw = path.joinpath(name).read_text(encoding="utf-8").strip()
        if raw == "max":
            return raw
        value = int(raw)
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
        return value

    events: dict[str, int] = {}
    for line in path.joinpath("memory.events").read_text(encoding="utf-8").splitlines():
        key, raw = line.split()
        value = int(raw)
        if value < 0:
            raise ValueError("memory.events values must be nonnegative")
        events[key] = value
    required_events = {"oom", "oom_kill", "max"}
    if not required_events.issubset(events):
        raise ValueError("memory.events is missing required counters")
    peak = read_limit("memory.peak")
    if not isinstance(peak, int) or peak <= 0:
        raise ValueError("memory.peak must be a positive finite byte count")
    return {
        "version": "v2-host",
        "path": str(path),
        "peak_bytes": peak,
        "limit_bytes": read_limit("memory.max"),
        "swap_limit_bytes": read_limit("memory.swap.max"),
        "events": events,
    }


def _cgroup_is_populated(path: Path) -> bool:
    values = {}
    for line in path.joinpath("cgroup.events").read_text(encoding="utf-8").splitlines():
        key, raw = line.split()
        values[key] = int(raw)
    if values.get("populated") not in {0, 1}:
        raise ValueError("cgroup.events is missing populated=0|1")
    return values["populated"] == 1


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload), encoded


def _publish_bytes(payload: bytes, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    if os.name == "posix":
        temporary.chmod(0o644)
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite run evidence: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _publish_json(payload: Mapping[str, Any], destination: Path) -> Path:
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    return _publish_bytes(encoded, destination)


def validate_image_inspection(
    registration: Mapping[str, Any], image: Mapping[str, Any]
) -> str:
    """Validate immutable image identities and required provenance labels."""

    images = registration.get("images")
    source = registration.get("source")
    if not isinstance(images, Mapping) or not isinstance(source, Mapping):
        raise ValueError("registration image/source identities are incomplete")
    project = images.get("project_image")
    base = images.get("base_image")
    if not isinstance(project, Mapping) or not isinstance(base, Mapping):
        raise ValueError("registration image identities are incomplete")
    image_id = image.get("Id")
    if not isinstance(image_id, str):
        raise ValueError("Docker inspection is missing image Id")
    image_id = register_scaling_sweep._validated_image_identity(
        image_id, "Docker image Id"
    )
    project_kind = project.get("kind")
    project_identity = project.get("identity")
    if project_kind == "local_image_id":
        if image_id != project_identity:
            raise ValueError("Docker local image Id differs from registration")
    elif project_kind == "oci_digest":
        repo_digests = image.get("RepoDigests")
        if not isinstance(repo_digests, Sequence) or isinstance(repo_digests, str):
            raise ValueError("Docker inspection is missing RepoDigests")
        digests = {
            value.rsplit("@", 1)[-1].lower()
            for value in repo_digests
            if isinstance(value, str) and "@" in value
        }
        if project_identity not in digests:
            raise ValueError("registered OCI project digest is absent from RepoDigests")
    else:
        raise ValueError("registration project image kind is unsupported")
    config = image.get("Config")
    labels = config.get("Labels") if isinstance(config, Mapping) else None
    if not isinstance(labels, Mapping):
        raise ValueError("Docker image provenance labels are missing")
    if labels.get("org.opencontainers.image.revision") != source.get("commit"):
        raise ValueError("Docker image revision label differs from registration source")
    if labels.get("org.opencontainers.image.base.digest") != base.get("identity"):
        raise ValueError("Docker base digest label differs from registration")
    environment = config.get("Env")
    if not isinstance(environment, Sequence) or isinstance(environment, str):
        raise ValueError("Docker image provenance environment is missing")
    environment_map = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in environment
        if isinstance(item, str) and "=" in item
    }
    if (
        environment_map.get("SCATTER3D_GIT_COMMIT") != source.get("commit")
        or environment_map.get("SCATTER3D_GIT_DIRTY") != "false"
    ):
        raise ValueError("Docker image source environment differs from registration")
    return image_id


def validate_execution_context(
    registration: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    spec_bytes: bytes,
    source: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
) -> None:
    """Regenerate the complete registration from current immutable identities."""

    images = registration.get("images")
    if not isinstance(images, Mapping):
        raise ValueError("registration images mapping is missing")
    project = images.get("project_image")
    base = images.get("base_image")
    if not isinstance(project, Mapping) or not isinstance(base, Mapping):
        raise ValueError("registration image identities are incomplete")
    expected = register_scaling_sweep.build_registration(
        spec,
        spec_bytes,
        source=source,
        project_image_kind=str(project.get("kind")),
        project_image_identity=str(project.get("identity")),
        base_image_digest=str(base.get("identity")),
        base_runtime_metadata=runtime_metadata,
    )
    if dict(registration) != expected:
        raise ValueError(
            "registration is incomplete or differs from current spec/source/image/runtime identities"
        )


def build_docker_create_command(
    registration: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    image_reference: str,
    host_run_directory: Path,
    attempt_id: str | None = None,
) -> list[str]:
    resource = entry["resource_contract"]
    memory_bytes = int(resource["cgroup_memory_limit_bytes"])
    project = registration["images"]["project_image"]
    base = registration["images"]["base_image"]
    project_variable = (
        "SCATTER3D_PROJECT_IMAGE_ID"
        if project["kind"] == "local_image_id"
        else "SCATTER3D_PROJECT_IMAGE_DIGEST"
    )
    container_name = (
        f"scatter3d-{str(registration['registration_id'])[:12]}-"
        f"{entry['run_id']}"
    )
    attempt = attempt_id or secrets.token_hex(16)
    return [
        "docker",
        "create",
        "--name",
        container_name,
        "--label",
        f"scatter3d.registration_id={registration['registration_id']}",
        "--label",
        f"scatter3d.run_id={entry['run_id']}",
        "--label",
        f"scatter3d.attempt_id={attempt}",
        "--memory",
        str(memory_bytes),
        "--memory-swap",
        str(memory_bytes),
        "--ipc=host",
        "--env",
        f"{project_variable}={project['identity']}",
        "--env",
        f"SCATTER3D_BASE_IMAGE_DIGEST={base['identity']}",
        "--volume",
        f"{host_run_directory}:{entry['outputs']['container_directory']}",
        "--workdir",
        "/opt/scatter3d",
        image_reference,
        "sh",
        "-c",
        (
            f"run={entry['outputs']['container_directory']}; "
            "touch \"$run/.executor-ready\" && "
            "while [ ! -e \"$run/.executor-go\" ]; do sleep 0.05; done; "
            "\"$@\"; rc=$?; "
            "printf '%s\\n' \"$rc\" > \"$run/.solver-exit-code\"; "
            "touch \"$run/.solver-done\"; "
            "while [ ! -e \"$run/.executor-collected\" ]; do sleep 0.05; done; "
            "exit \"$rc\""
        ),
        "scatter3d-executor",
        *entry["command"],
    ]


def classify_run_result(
    *,
    docker_return_code: int | None,
    fem_payload: Mapping[str, Any] | None,
    launch_prevented: bool,
) -> tuple[str, bool | None, str]:
    if fem_payload is not None:
        status = fem_payload.get("status")
        passed = fem_payload.get("passed")
        if status == "PASSED" and passed is True and docker_return_code == 0:
            return "PASSED", True, "fem_smoke artifact and Docker exit code passed"
        if status == "FAILED" and passed is False and docker_return_code == 1:
            return "FAILED", False, "fem_smoke artifact reported FAILED"
        return "FAILED", False, "fem_smoke artifact status/exit code is inconsistent"
    if launch_prevented:
        return "BLOCKED", None, "Docker could not launch the registered solve"
    return "FAILED", False, "registered solve produced no fem_smoke artifact"


def _registered_argument(entry: Mapping[str, Any], name: str) -> str:
    argv = entry["command"]
    try:
        return str(argv[argv.index(name) + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"registered command is missing {name}") from exc


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"fem_smoke {label} is missing or malformed")
    return value


def _required_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"fem_smoke {label} is missing or malformed")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
        or (not positive and float(value) < 0.0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"fem_smoke {label} must be a finite {qualifier} number")
    return float(value)


def _residual_number(value: Any, label: str) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError(
                f"fem_smoke {label} must be a finite nonnegative number"
            )
        return normalized
    if value in {"NaN", "+Infinity", "-Infinity"}:
        return math.inf
    raise ValueError(f"fem_smoke {label} is missing or malformed")


def _component_has_identity(component: Mapping[str, Any], label: str) -> None:
    for key in ("ksp_type", "pc_type", "options_prefix"):
        if not isinstance(component.get(key), str) or not component[key]:
            raise ValueError(f"fem_smoke {label} is missing {key}")
    instances = component.get("instances")
    ranks = component.get("mpi_ranks")
    if not isinstance(instances, int) or isinstance(instances, bool) or instances <= 0:
        raise ValueError(f"fem_smoke {label} has invalid instances")
    if (
        not isinstance(ranks, Sequence)
        or isinstance(ranks, str | bytes)
        or not ranks
        or any(not isinstance(rank, int) or isinstance(rank, bool) for rank in ranks)
    ):
        raise ValueError(f"fem_smoke {label} has invalid MPI ranks")


def _validate_completed_fem_evidence(
    entry: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, bool]:
    """Recompute completed-run gates from the recorded numerical evidence."""

    expected_frequency = float(_registered_argument(entry, "--frequencies-hz"))
    expected_dofs = int(_registered_argument(entry, "--expected-global-dofs"))
    expected_degree = int(_registered_argument(entry, "--degree"))
    expected_coarse_degree = int(
        _registered_argument(entry, "--p-multigrid-coarse-degree")
    )
    expected_subdivisions = int(_registered_argument(entry, "--subdivisions"))
    maximum_iterations = int(_registered_argument(entry, "--maximum-iterations"))
    maximum_residual = float(
        _registered_argument(entry, "--maximum-true-relative-residual")
    )
    gmres_restart = int(_registered_argument(entry, "--gmres-restart"))
    asm_overlap = int(_registered_argument(entry, "--asm-overlap"))
    shift = float(_registered_argument(entry, "--preconditioner-absorption-shift"))
    mpi_ranks = int(entry["resource_contract"]["mpi_ranks"])
    expected_ranks = list(range(mpi_ranks))

    if payload.get("solver") != "iterative":
        raise ValueError("fem_smoke completed artifact is not iterative")
    if payload.get("iterative_hierarchy") != "p-multigrid":
        raise ValueError("fem_smoke completed artifact is not p-multigrid")
    if payload.get("degree") != expected_degree:
        raise ValueError("fem_smoke fine degree differs from registration")
    if payload.get("subdivisions") != expected_subdivisions:
        raise ValueError("fem_smoke subdivisions differ from registration")
    if payload.get("frequencies_hz") != [expected_frequency]:
        raise ValueError("fem_smoke frequencies differ from registration")
    if "complex128" not in str(payload.get("scalar_type")):
        raise ValueError("fem_smoke scalar type is not complex128")

    requested_options = {
        "ksp_gmres_restart": gmres_restart,
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_factor_mat_solver_type": "mumps",
        "mg_coarse_pc_type": "lu",
        "mg_levels_1_ksp_max_it": 1,
        "mg_levels_1_ksp_type": "richardson",
        "mg_levels_1_pc_asm_overlap": asm_overlap,
        "mg_levels_1_pc_type": "asm",
        "mg_levels_1_sub_ksp_type": "preonly",
        "mg_levels_1_sub_pc_factor_mat_solver_type": "mumps",
        "mg_levels_1_sub_pc_type": "lu",
    }
    solver_configuration = _required_mapping(
        payload.get("solver_configuration"), "solver_configuration"
    )
    requested_solver = _required_mapping(
        solver_configuration.get("requested"),
        "solver_configuration.requested",
    )
    required_requested = {
        "solver": "iterative",
        "iterative_hierarchy": "p-multigrid",
        "ksp_type": "fgmres",
        "pc_type": "mg",
        "preconditioning_side": "right",
        "maximum_iterations": maximum_iterations,
        "p_multigrid_coarse_degree": expected_coarse_degree,
        "preconditioner_absorption_shift": shift,
    }
    if any(requested_solver.get(key) != value for key, value in required_requested.items()):
        raise ValueError("fem_smoke requested solver differs from registration")
    if requested_solver.get("petsc_options") != requested_options:
        raise ValueError("fem_smoke requested PETSc options differ from registration")

    diagnostics = _required_sequence(
        payload.get("frequency_diagnostics"), "frequency_diagnostics"
    )
    if len(diagnostics) != 1:
        raise ValueError("fem_smoke must contain one registered frequency diagnostic")
    item = _required_mapping(diagnostics[0], "frequency diagnostic")
    if item.get("frequency_hz") != expected_frequency:
        raise ValueError("fem_smoke diagnostic frequency differs from registration")
    if item.get("global_complex_dofs") != expected_dofs:
        raise ValueError("fem_smoke diagnostic global DoFs differ from registration")
    if item.get("fine_degree") != expected_degree:
        raise ValueError("fem_smoke diagnostic fine degree differs from registration")
    coarse_dofs = item.get("coarse_global_complex_dofs")
    if (
        item.get("coarse_degree") != expected_coarse_degree
        or not isinstance(coarse_dofs, int)
        or isinstance(coarse_dofs, bool)
        or not 0 < coarse_dofs < expected_dofs
    ):
        raise ValueError("fem_smoke coarse-space identity is missing or malformed")
    for key in (
        "matrix_nonzeros",
        "preconditioner_matrix_nonzeros",
        "coarse_preconditioner_matrix_nonzeros",
        "rank_peak_rss_bytes_max",
        "rank_peak_rss_bytes_sum",
    ):
        value = item.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"fem_smoke diagnostic {key} is missing or malformed")
    for key in (
        "assembly_seconds",
        "preconditioner_assembly_seconds",
        "coarse_preconditioner_assembly_seconds",
        "setup_seconds",
    ):
        _finite_number(item.get(key), f"diagnostic {key}")
    if item.get("preconditioner_absorption_shift") != shift:
        raise ValueError("fem_smoke diagnostic shift differs from registration")
    expected_physical_preconditioner = shift == 0.0
    if item.get("preconditioner_operator_is_physical") is not (
        expected_physical_preconditioner
    ):
        raise ValueError("fem_smoke A/P operator identity differs from registration")
    if not isinstance(item.get("p_multigrid_operator_checks_passed"), bool):
        raise ValueError("fem_smoke p-multigrid operator check is missing")

    transfer = _required_mapping(item.get("transfer_operator"), "transfer operator")
    transfer_well_formed = (
        transfer.get("direction") == "coarse_to_fine"
        and transfer.get("rows") == expected_dofs
        and transfer.get("columns") == coarse_dofs
        and isinstance(transfer.get("nonzeros"), int)
        and not isinstance(transfer.get("nonzeros"), bool)
        and transfer["nonzeros"] > 0
        and isinstance(transfer.get("constrained_fine_rows"), int)
        and transfer["constrained_fine_rows"] > 0
        and isinstance(transfer.get("constrained_coarse_columns"), int)
        and transfer["constrained_coarse_columns"] > 0
        and transfer.get("maximum_imaginary_abs") == 0.0
    )
    _finite_number(transfer.get("assembly_seconds"), "transfer assembly_seconds")

    hierarchy = _required_mapping(item.get("solver_hierarchy"), "solver hierarchy")
    requested_hierarchy = _required_mapping(
        hierarchy.get("requested"), "requested solver hierarchy"
    )
    requested_pairs = _required_sequence(
        requested_hierarchy.get("petsc_options"), "requested hierarchy PETSc options"
    )
    try:
        hierarchy_options = dict(requested_pairs)
    except (TypeError, ValueError) as exc:
        raise ValueError("fem_smoke requested hierarchy options are malformed") from exc
    requested_hierarchy_ok = (
        requested_hierarchy.get("ksp_type") == "fgmres"
        and requested_hierarchy.get("pc_type") == "mg"
        and requested_hierarchy.get("preconditioning_side") == "right"
        and hierarchy_options == requested_options
    )
    effective = _required_mapping(
        hierarchy.get("effective"), "effective solver hierarchy"
    )
    top = _required_mapping(effective.get("top_level"), "effective top-level solver")
    fine = _required_mapping(effective.get("mg_fine_smoother"), "effective fine smoother")
    coarse = _required_mapping(effective.get("mg_coarse_solver"), "effective coarse solver")
    subdomains = _required_sequence(
        effective.get("mg_fine_asm_subdomain_solvers"),
        "effective ASM subdomain solvers",
    )
    if not subdomains:
        raise ValueError("fem_smoke effective ASM subdomain solvers are empty")
    _component_has_identity(top, "effective top-level solver")
    _component_has_identity(fine, "effective fine smoother")
    _component_has_identity(coarse, "effective coarse solver")
    for index, component in enumerate(subdomains):
        _component_has_identity(
            _required_mapping(component, f"effective ASM subdomain solver {index}"),
            f"effective ASM subdomain solver {index}",
        )
    component_contracts = (
        (
            top,
            "top",
            "scatter3d_0_",
            maximum_iterations,
            "unpreconditioned",
        ),
        (
            fine,
            "mg.fine",
            "scatter3d_0_mg_levels_1_",
            1,
            "none",
        ),
        (
            coarse,
            "mg.coarse",
            "scatter3d_0_mg_coarse_",
            10_000,
            "none",
        ),
    )
    if any(
        component.get("path") != path
        or component.get("options_prefix") != prefix
        or component.get("instances") != mpi_ranks
        or component.get("mpi_ranks") != expected_ranks
        or component.get("maximum_iterations") != iterations
        or component.get("norm_type") != norm_type
        for component, path, prefix, iterations, norm_type in component_contracts
    ):
        raise ValueError("fem_smoke effective hierarchy component identity is inconsistent")
    if len(subdomains) != 1:
        raise ValueError("fem_smoke must contain one aggregated ASM subdomain identity")
    subdomain = _required_mapping(subdomains[0], "effective ASM subdomain solver")
    if (
        subdomain.get("path") != "mg.fine.asm.subdomains"
        or subdomain.get("options_prefix") != "scatter3d_0_mg_levels_1_sub_"
        or subdomain.get("instances") != mpi_ranks
        or subdomain.get("mpi_ranks") != expected_ranks
        or subdomain.get("maximum_iterations") != 10_000
        or subdomain.get("norm_type") != "none"
    ):
        raise ValueError("fem_smoke effective ASM subdomain identity is inconsistent")
    raw_view = effective.get("petsc_view_ascii")
    if not isinstance(raw_view, str) or not raw_view.strip():
        raise ValueError("fem_smoke raw PETSc KSP view is missing")
    raw_view_required = (
        "type: fgmres",
        "right preconditioning",
        "type: mg",
        "levels=2",
        "not using galerkin",
        "type: richardson",
        "type: asm",
        "type: preonly",
        "type: lu",
        "package used to perform factorization: mumps",
    )
    if any(fragment not in raw_view.lower() for fragment in raw_view_required):
        raise ValueError("fem_smoke raw PETSc KSP view lacks required hierarchy evidence")
    hierarchy_ok = (
        requested_hierarchy_ok
        and top.get("ksp_type") == "fgmres"
        and top.get("pc_type") == "mg"
        and effective.get("preconditioning_side") == "right"
        and effective.get("pc_uses_amat") is False
        and effective.get("mg_levels") == 2
        and effective.get("mg_type") == "multiplicative"
        and effective.get("mg_cycle_type") == "v"
        and effective.get("mg_galerkin") == "none"
        and effective.get("maximum_iterations") == maximum_iterations
        and effective.get("relative_tolerance") == 1.0e-8
        and effective.get("absolute_tolerance") == 1.0e-12
        and fine.get("ksp_type") == "richardson"
        and fine.get("pc_type") == "asm"
        and fine.get("maximum_iterations") == 1
        and fine.get("norm_type") == "none"
        and effective.get("mg_fine_asm_type") == "restrict"
        and effective.get("mg_fine_asm_overlap") == asm_overlap
        and all(
            isinstance(component, Mapping)
            and component.get("ksp_type") == "preonly"
            and component.get("pc_type") == "lu"
            and component.get("factor_solver_type") == "mumps"
            for component in subdomains
        )
        and coarse.get("ksp_type") == "preonly"
        and coarse.get("pc_type") == "lu"
        and coarse.get("factor_solver_type") == "mumps"
        and transfer_well_formed
        and item.get("p_multigrid_operator_checks_passed") is True
    )

    effective_by_frequency = _required_sequence(
        solver_configuration.get("effective_by_frequency"),
        "solver_configuration.effective_by_frequency",
    )
    if list(effective_by_frequency) != [
        {"frequency_hz": expected_frequency, "hierarchy": dict(effective)}
    ]:
        raise ValueError("fem_smoke effective solver evidence is inconsistent")

    port_solves = _required_sequence(item.get("port_solves"), "port solves")
    if len(port_solves) != 2:
        raise ValueError("fem_smoke must contain exactly two port solves")
    port_names: list[str] = []
    convergence_ok = True
    for index, port_value in enumerate(port_solves):
        port = _required_mapping(port_value, f"port solve {index}")
        port_name = port.get("port_name")
        if not isinstance(port_name, str):
            raise ValueError("fem_smoke port solve name is missing")
        port_names.append(port_name)
        iterations = port.get("iterations")
        reason = port.get("converged_reason")
        if (
            not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or iterations < 0
            or iterations > maximum_iterations
            or not isinstance(reason, int)
            or isinstance(reason, bool)
        ):
            raise ValueError("fem_smoke port iterations/reason are malformed")
        residual = _residual_number(
            port.get("true_relative_residual"),
            f"port {port_name} true relative residual",
        )
        residual_norm = _residual_number(
            port.get("true_residual_norm"), f"port {port_name} true residual norm"
        )
        _finite_number(port.get("solve_seconds"), f"port {port_name} solve_seconds")
        history = _required_sequence(
            port.get("reported_residual_history"),
            f"port {port_name} residual history",
        )
        if not history:
            raise ValueError("fem_smoke reported residual history is empty")
        history_values = [
            _residual_number(history_value, f"port {port_name} residual history")
            for history_value in history
        ]
        convergence_ok = (
            convergence_ok
            and reason > 0
            and residual <= maximum_residual
            and math.isfinite(residual_norm)
            and all(math.isfinite(value) for value in history_values)
        )
    if port_names != ["left", "right"]:
        raise ValueError("fem_smoke port solve order/identity differs from registration")

    expected_counts = {
        "matrix_assemblies": 1,
        "preconditioner_matrix_assemblies": 0 if shift == 0.0 else 1,
        "coarse_preconditioner_matrix_assemblies": 1,
        "transfer_operator_assemblies": 1,
        "operator_setups": 1,
        "global_numeric_factorizations": 0,
        "coarse_global_factorizations": 1,
        "rhs_solves": 2,
    }
    observed_counts: dict[str, int] = {}
    for key in expected_counts:
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"fem_smoke counter {key} is missing or malformed")
        observed_counts[key] = value
    if payload.get("numeric_factorizations") != observed_counts[
        "global_numeric_factorizations"
    ]:
        raise ValueError("fem_smoke numeric factorization counter alias is inconsistent")
    count_ok = observed_counts == expected_counts

    preconditioner = _required_mapping(payload.get("preconditioner"), "preconditioner")
    if (
        preconditioner.get("requested_absorption_shift") != shift
        or preconditioner.get("requested_hierarchy") != "p-multigrid"
        or preconditioner.get("requested_coarse_degree") != expected_coarse_degree
    ):
        raise ValueError("fem_smoke requested preconditioner differs from registration")
    per_frequency = _required_sequence(
        preconditioner.get("per_frequency"), "preconditioner.per_frequency"
    )
    expected_preconditioner = {
        "frequency_hz": expected_frequency,
        "absorption_shift": shift,
        "operator_is_physical": expected_physical_preconditioner,
        "matrix_nonzeros": item["preconditioner_matrix_nonzeros"],
        "matrix_memory_bytes_sum": item.get("preconditioner_matrix_memory_bytes_sum"),
        "assembly_seconds": item["preconditioner_assembly_seconds"],
        "coarse": {
            "degree": expected_coarse_degree,
            "global_complex_dofs": coarse_dofs,
            "matrix_nonzeros": item["coarse_preconditioner_matrix_nonzeros"],
            "matrix_memory_bytes_sum": item.get(
                "coarse_preconditioner_matrix_memory_bytes_sum"
            ),
            "assembly_seconds": item["coarse_preconditioner_assembly_seconds"],
        },
        "transfer_operator": dict(transfer),
        "p_multigrid_operator_checks_passed": item[
            "p_multigrid_operator_checks_passed"
        ],
    }
    if list(per_frequency) != [expected_preconditioner]:
        raise ValueError("fem_smoke preconditioner evidence is inconsistent")

    gates = _required_mapping(payload.get("gates"), "gates")
    convergence_gate = _required_mapping(
        gates.get("convergence_and_true_residual"), "convergence gate"
    )
    count_gate = _required_mapping(
        gates.get("assembly_setup_rhs_counts"), "assembly/setup/RHS gate"
    )
    pmg_gate = _required_mapping(
        gates.get("p_multigrid_structure"), "p-multigrid gate"
    )
    expected_pairs = {
        "convergence_and_true_residual": convergence_ok,
        "assembly_setup_rhs_counts": count_ok,
        "p_multigrid_structure": hierarchy_ok,
    }
    for name, computed in expected_pairs.items():
        gate = _required_mapping(gates.get(name), f"{name} gate")
        expected_pair = ("PASSED", True) if computed else ("FAILED", False)
        if (gate.get("status"), gate.get("passed")) != expected_pair:
            raise ValueError(f"fem_smoke {name} gate differs from recorded evidence")
    if convergence_gate.get("maximum_true_relative_residual") != maximum_residual:
        raise ValueError("fem_smoke convergence threshold differs from registration")
    if (
        count_gate.get("expected") != expected_counts
        or count_gate.get("observed") != observed_counts
    ):
        raise ValueError("fem_smoke assembly/setup/RHS evidence is inconsistent")
    if pmg_gate.get("requested_coarse_degree") != expected_coarse_degree:
        raise ValueError("fem_smoke p-multigrid gate coarse degree is inconsistent")
    return expected_pairs


def validate_fem_smoke_artifact(
    registration: Mapping[str, Any],
    entry: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    """Fail closed unless a completed FEM artifact matches its registration."""

    if payload.get("schema") != FEM_SMOKE_SCHEMA:
        raise ValueError("fem_smoke schema is missing or unsupported")
    observed_source = payload.get("source")
    if not isinstance(observed_source, Mapping) or (
        register_scaling_sweep._validated_source(observed_source)
        != registration.get("source")
    ):
        raise ValueError("fem_smoke source identity differs from registration")
    images = payload.get("images")
    registered_images = registration.get("images")
    if not isinstance(images, Mapping) or not isinstance(registered_images, Mapping):
        raise ValueError("fem_smoke image identity is missing")
    project = registered_images["project_image"]
    observed_project = images.get("project_image")
    observed_base = images.get("base_image")
    if not isinstance(observed_project, Mapping) or not isinstance(
        observed_base, Mapping
    ):
        raise ValueError("fem_smoke image identity is incomplete")
    project_field = (
        "local_image_id" if project["kind"] == "local_image_id" else "digest"
    )
    if observed_project.get(project_field) != project["identity"]:
        raise ValueError("fem_smoke project image differs from registration")
    if observed_base.get("digest") != registered_images["base_image"]["identity"]:
        raise ValueError("fem_smoke base image differs from registration")
    if payload.get("mpi_size") != entry["resource_contract"]["mpi_ranks"]:
        raise ValueError("fem_smoke MPI size differs from registration")
    command = payload.get("command")
    if not isinstance(command, Mapping) or command.get("argv") != entry["command"][4:]:
        raise ValueError("fem_smoke command argv differs from registration")
    physical = payload.get("physical_problem")
    if not isinstance(physical, Mapping) or physical.get("schema") != (
        "scatter3d.validation.physical_problem/v1"
    ):
        raise ValueError("fem_smoke physical problem identity is missing")
    definition = physical.get("definition")
    if not isinstance(definition, Mapping):
        raise ValueError("fem_smoke physical problem definition is missing")
    encoded_definition = json.dumps(
        definition,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if physical.get("sha256") != hashlib.sha256(encoded_definition).hexdigest():
        raise ValueError("fem_smoke physical problem hash is invalid")
    if dict(physical) != entry.get("physical_problem"):
        raise ValueError("fem_smoke physical problem differs from registration")

    preconditioner = payload.get("preconditioner")
    if not isinstance(preconditioner, Mapping):
        raise ValueError("fem_smoke preconditioner identity is missing")
    requested_shift = preconditioner.get("requested_absorption_shift")
    if requested_shift != entry["preconditioner_absorption_shift"]:
        raise ValueError("fem_smoke absorption shift differs from registration")
    cgroup = payload.get("cgroup_memory")
    if not isinstance(cgroup, Mapping) or cgroup.get("limit_bytes") != entry[
        "resource_contract"
    ]["cgroup_memory_limit_bytes"]:
        raise ValueError("fem_smoke cgroup limit differs from registration")
    if (
        cgroup.get("version") != "v2"
        or not isinstance(cgroup.get("peak_bytes"), int)
        or cgroup["peak_bytes"] <= 0
        or cgroup.get("swap_limit_bytes") != 0
    ):
        raise ValueError(
            "fem_smoke cgroup v2 peak or no-swap instrumentation is missing"
        )
    runtime = payload.get("runtime")
    expected_runtime = registered_images["base_image"].get("runtime_metadata")
    if not isinstance(runtime, Mapping) or not isinstance(
        expected_runtime, Mapping
    ):
        raise ValueError("fem_smoke runtime identity is missing")
    observed_petsc = runtime.get("petsc_version")
    if (
        runtime.get("dolfinx_version") != expected_runtime.get("dolfinx")
        or runtime.get("mpi4py_version") != expected_runtime.get("mpi4py")
        or runtime.get("mpi_library_version") != expected_runtime.get("mpi_library")
        or not isinstance(observed_petsc, Sequence)
        or ".".join(str(part) for part in observed_petsc)
        != expected_runtime.get("petsc")
        or "complex128" not in str(runtime.get("petsc_scalar_type"))
    ):
        raise ValueError("fem_smoke runtime differs from registered image metadata")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("fem_smoke gates are missing")
    outcome_gates = (
        "convergence_and_true_residual",
        "assembly_setup_rhs_counts",
        "p_multigrid_structure",
        "expected_global_dofs",
    )
    release_gate = gates.get("release_comparison_provenance")
    if (
        not isinstance(release_gate, Mapping)
        or release_gate.get("status") != "PASSED"
        or release_gate.get("passed") is not True
    ):
        raise ValueError("fem_smoke release provenance gate did not PASSED")
    failed_gates: list[str] = []
    for name in outcome_gates:
        gate = gates.get(name)
        if not isinstance(gate, Mapping):
            raise ValueError(f"fem_smoke registered gate {name} is missing")
        pair = (gate.get("status"), gate.get("passed"))
        if pair not in {("PASSED", True), ("FAILED", False), ("NOT RUN", None)}:
            raise ValueError(f"fem_smoke registered gate {name} is malformed")
        if pair == ("FAILED", False):
            failed_gates.append(name)
    expected_dofs = int(_registered_argument(entry, "--expected-global-dofs"))
    dof_gate = gates["expected_global_dofs"]
    diagnostics = payload.get("frequency_diagnostics")
    if (
        dof_gate.get("expected") != expected_dofs
        or not isinstance(diagnostics, Sequence)
        or not diagnostics
        or any(
            not isinstance(item, Mapping)
            or item.get("global_complex_dofs") != dof_gate.get("observed")
            for item in diagnostics
        )
    ):
        raise ValueError("fem_smoke exact global DoF evidence differs from registration")
    overall = (payload.get("status"), payload.get("passed"))
    preflight_failure = (
        payload.get("execution_phase") == "preflight"
        and overall == ("FAILED", False)
        and failed_gates == ["expected_global_dofs"]
        and all(
            gates[name].get("status") == "NOT RUN"
            for name in outcome_gates
            if name != "expected_global_dofs"
        )
    )
    if preflight_failure:
        expected_frequency = float(
            _registered_argument(entry, "--frequencies-hz")
        )
        observed_dofs = dof_gate.get("observed")
        if (
            dof_gate.get("expected") != expected_dofs
            or observed_dofs == expected_dofs
            or not isinstance(observed_dofs, int)
            or isinstance(observed_dofs, bool)
        ):
            raise ValueError("preflight fem_smoke exact global DoF failure is inconsistent")
        if list(diagnostics) != [
            {
                "frequency_hz": expected_frequency,
                "global_complex_dofs": observed_dofs,
            }
        ]:
            raise ValueError("preflight fem_smoke diagnostic shape is inconsistent")
        completed_fields = {
            "solver_configuration",
            "matrix_assemblies",
            "preconditioner_matrix_assemblies",
            "coarse_preconditioner_matrix_assemblies",
            "transfer_operator_assemblies",
            "operator_setups",
            "global_numeric_factorizations",
            "coarse_global_factorizations",
            "numeric_factorizations",
            "rhs_solves",
        }
        unexpected = sorted(completed_fields.intersection(payload))
        if unexpected:
            raise ValueError(
                "preflight fem_smoke artifact contains completed-run fields: "
                + ", ".join(unexpected)
            )
        if "per_frequency" in preconditioner:
            raise ValueError(
                "preflight fem_smoke artifact contains completed preconditioner evidence"
            )
    else:
        _validate_completed_fem_evidence(entry, payload)
    if any(
        gates[name].get("status") == "NOT RUN" for name in outcome_gates
    ) and not preflight_failure:
        raise ValueError("registered solve outcome gates must not be NOT RUN")
    if overall == ("PASSED", True):
        if failed_gates or any(
            gates[name].get("status") != "PASSED" for name in outcome_gates
        ):
            raise ValueError("PASSED fem_smoke artifact has an unmet registered gate")
        if dof_gate.get("observed") != expected_dofs:
            raise ValueError("PASSED fem_smoke artifact has incorrect global DoFs")
    elif overall == ("FAILED", False):
        if not failed_gates:
            raise ValueError("FAILED fem_smoke artifact has no FAILED registered gate")
    else:
        raise ValueError("fem_smoke overall status is malformed")
    return tuple(failed_gates)


def _entry_host_paths(
    registration: Mapping[str, Any],
    entry: Mapping[str, Any],
    host_output_root: Path,
) -> dict[str, Path]:
    logical_root = Path(str(registration["output_root"]))
    paths: dict[str, Path] = {}
    for key, value in entry["outputs"].items():
        if key.startswith("container_"):
            continue
        logical = Path(str(value))
        try:
            relative = logical.relative_to(logical_root)
        except ValueError as exc:
            raise ValueError(f"registered output {key} escapes output_root") from exc
        paths[key] = host_output_root / relative
    return paths


def _sha256_manifest(run_directory: Path, manifest: Path) -> bytes:
    lines = []
    for path in sorted(run_directory.iterdir(), key=lambda item: item.name):
        if path.is_file() and path != manifest:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.name}")
    return ("\n".join(lines) + "\n").encode("ascii")


def _completed_stdout(completed: subprocess.CompletedProcess[Any]) -> str:
    output = completed.stdout or b""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def _cleanup_container_attempt(
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
    registration: Mapping[str, Any],
    entry: Mapping[str, Any],
    container_name: str,
    attempt_id: str,
    initial_target: str | None,
    ambiguous_create: bool,
    lifecycle_commands: list[list[str]],
    poll_seconds: float,
    stable_checks: int,
    max_checks: int,
) -> tuple[bool, bool, bytes]:
    """Remove only this attempt and prove a stable post-create absence."""

    if stable_checks < 1 or max_checks < stable_checks or poll_seconds < 0.0:
        raise ValueError("container cleanup polling contract is invalid")
    attempted = ambiguous_create
    diagnostics = b""

    def remove(target: str) -> None:
        nonlocal attempted, diagnostics
        attempted = True
        command = ["docker", "rm", "--force", target]
        lifecycle_commands.append(command)
        try:
            removed = runner(
                command,
                check=False,
                capture_output=True,
                timeout=60,
            )
            if int(removed.returncode) != 0:
                diagnostics += removed.stderr or b"docker rm failed\n"
        except (OSError, subprocess.SubprocessError) as exc:
            diagnostics += f"docker rm failed: {exc}\n".encode()

    if initial_target is not None:
        remove(initial_target)

    required_stable = stable_checks if ambiguous_create else 1
    checks = max_checks if ambiguous_create else max(2, stable_checks)
    consecutive_absent = 0
    for check_index in range(checks):
        name_command = [
            "docker",
            "ps",
            "-aq",
            "--no-trunc",
            "--filter",
            f"name=^/{container_name}$",
        ]
        attempt_command = [
            "docker",
            "ps",
            "-aq",
            "--no-trunc",
            "--filter",
            f"label=scatter3d.registration_id={registration['registration_id']}",
            "--filter",
            f"label=scatter3d.run_id={entry['run_id']}",
            "--filter",
            f"label=scatter3d.attempt_id={attempt_id}",
        ]
        observed: list[set[str]] = []
        for command in (name_command, attempt_command):
            lifecycle_commands.append(command)
            try:
                completed = runner(
                    command,
                    check=False,
                    capture_output=True,
                    timeout=60,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                diagnostics += f"container cleanup verification failed: {exc}\n".encode()
                return attempted, False, diagnostics
            if int(completed.returncode) != 0:
                diagnostics += completed.stderr or b"container cleanup verification failed\n"
                return attempted, False, diagnostics
            observed.append(
                {
                    line.strip()
                    for line in _completed_stdout(completed).splitlines()
                    if line.strip()
                }
            )
        name_ids, attempt_ids = observed
        foreign_name_ids = name_ids - attempt_ids
        if foreign_name_ids:
            diagnostics += (
                b"refusing to remove a container name not owned by this executor attempt\n"
            )
            return attempted, False, diagnostics
        if attempt_ids:
            consecutive_absent = 0
            for container_id in sorted(attempt_ids):
                remove(container_id)
        else:
            consecutive_absent += 1
            if not ambiguous_create and consecutive_absent >= required_stable:
                return attempted, True, diagnostics
        if ambiguous_create and check_index + 1 < checks and poll_seconds:
            time.sleep(poll_seconds)
    if ambiguous_create and consecutive_absent >= required_stable:
        return attempted, True, diagnostics
    diagnostics += b"container attempt did not reach the stable absence window\n"
    return attempted, False, diagnostics


def execute_registered_entries(
    registration: Mapping[str, Any],
    *,
    image_reference: str,
    output_parent: Path,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    cgroup_resolver: Callable[[int], Path] = _resolve_host_cgroup,
    cgroup_reader: Callable[[Path], dict[str, Any]] = _read_cgroup_v2_metrics,
    cgroup_populated: Callable[[Path], bool] = _cgroup_is_populated,
    timeout_seconds: float | None = None,
    run_ids: Sequence[str] | None = None,
    cleanup_poll_seconds: float = 1.0,
    cleanup_stable_checks: int = 10,
    cleanup_max_checks: int = 60,
) -> list[dict[str, Any]]:
    """Run every registered entry, preserving evidence and continuing on failure."""
    host_output_root = output_parent.expanduser().resolve() / Path(
        str(registration["output_root"])
    )
    all_entries = registration["entries"]
    available_ids = {entry["run_id"] for entry in all_entries}
    selected_ids = available_ids if run_ids is None else set(run_ids)
    unknown_ids = sorted(selected_ids - available_ids)
    if unknown_ids:
        raise ValueError(f"unregistered run ids requested: {unknown_ids}")
    entries = [entry for entry in all_entries if entry["run_id"] in selected_ids]
    if not entries:
        raise ValueError("at least one registered run id must be selected")
    planned: list[tuple[Mapping[str, Any], dict[str, Path]]] = []
    for entry in entries:
        registered_timeout = entry["resource_contract"].get(
            "wall_time_limit_seconds"
        )
        if (
            not isinstance(registered_timeout, int)
            or isinstance(registered_timeout, bool)
            or registered_timeout <= 0
        ):
            raise ValueError("registered wall-time limit must be a positive integer")
        if timeout_seconds is not None and timeout_seconds != registered_timeout:
            raise ValueError("executor timeout differs from registered wall-time limit")
        paths = _entry_host_paths(registration, entry, host_output_root)
        if paths["directory"].exists():
            raise FileExistsError(
                f"refusing to reuse registered run directory: {paths['directory']}"
            )
        planned.append((entry, paths))
    host_output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for entry, paths in planned:
        started_monotonic = time.monotonic()
        registered_timeout = int(
            entry["resource_contract"]["wall_time_limit_seconds"]
        )
        run_directory = paths["directory"]
        run_directory.mkdir(parents=False, exist_ok=False)
        if os.name == "posix":
            run_directory.chmod(0o777)
        ready_path = run_directory / ".executor-ready"
        go_path = run_directory / ".executor-go"
        solver_done_path = run_directory / ".solver-done"
        solver_exit_path = run_directory / ".solver-exit-code"
        collected_path = run_directory / ".executor-collected"
        attempt_id = secrets.token_hex(16)
        create_command = build_docker_create_command(
            registration,
            entry,
            image_reference=image_reference,
            host_run_directory=run_directory,
            attempt_id=attempt_id,
        )
        stdout = b""
        stderr = b""
        return_code: int | None = None
        solver_exit_code: int | None = None
        launch_prevented = False
        timed_out = False
        lifecycle_error: str | None = None
        container_id: str | None = None
        container_name = create_command[create_command.index("--name") + 1]
        create_ambiguous = False
        lifecycle_stage = "create"
        state: Mapping[str, Any] | None = None
        host_config: Mapping[str, Any] | None = None
        cgroup_path: Path | None = None
        cgroup_metrics: dict[str, Any] | None = None
        cgroup_error: str | None = None
        lifecycle_commands: list[list[str]] = [create_command]
        cleanup_attempted = False
        cleanup_succeeded: bool | None = None
        cleanup_absence_verified: bool | None = None
        try:
            created = runner(
                create_command, check=False, capture_output=True, timeout=60
            )
            if int(created.returncode) != 0:
                launch_prevented = True
                stderr += created.stderr or b"docker create failed\n"
            else:
                raw_container_id = (created.stdout or b"").decode(
                    "utf-8", errors="replace"
                ).strip()
                if not raw_container_id:
                    launch_prevented = True
                    create_ambiguous = True
                    stderr += b"docker create returned no container id\n"
                else:
                    container_id = raw_container_id.splitlines()[-1]
                    lifecycle_stage = "start"
                    start_command = ["docker", "start", container_id]
                    lifecycle_commands.append(start_command)
                    started = runner(
                        start_command, check=False, capture_output=True, timeout=60
                    )
                    if int(started.returncode) != 0:
                        launch_prevented = True
                        stderr += started.stderr or b"docker start failed\n"
                    else:
                        barrier_deadline = time.monotonic() + 60.0
                        while not ready_path.is_file() and time.monotonic() < barrier_deadline:
                            time.sleep(0.05)
                        if not ready_path.is_file():
                            lifecycle_error = "container startup barrier was not reached"
                        inspect_command = [
                            "docker",
                            "inspect",
                            "--format",
                            "{{json .}}",
                            container_id,
                        ]
                        lifecycle_commands.append(inspect_command)
                        inspected = runner(
                            inspect_command,
                            check=False,
                            capture_output=True,
                            timeout=60,
                        )
                        if int(inspected.returncode) != 0:
                            lifecycle_error = "initial docker inspect failed"
                            stderr += inspected.stderr or b"initial docker inspect failed\n"
                        else:
                            try:
                                initial_state, _ = _parse_container_inspection(
                                    inspected.stdout or b""
                                )
                                if initial_state["Running"] is not True:
                                    raise ValueError(
                                        "container exited before executor released barrier"
                                    )
                                cgroup_path = cgroup_resolver(
                                    int(initial_state["Pid"])
                                )
                            except (OSError, ValueError) as exc:
                                lifecycle_error = str(exc)
                        if lifecycle_error is None:
                            go_path.touch(exist_ok=False)
                            if os.name == "posix":
                                go_path.chmod(0o666)
                            solve_deadline = time.monotonic() + registered_timeout
                            while True:
                                try:
                                    cgroup_metrics = cgroup_reader(cgroup_path)
                                    cgroup_error = None
                                except (OSError, ValueError) as exc:
                                    if cgroup_metrics is None:
                                        cgroup_error = str(exc)
                                if solver_done_path.is_file():
                                    break
                                try:
                                    if not cgroup_populated(cgroup_path):
                                        lifecycle_error = (
                                            "container cgroup became empty before solver-done barrier"
                                        )
                                        break
                                except (OSError, ValueError) as exc:
                                    lifecycle_error = str(exc)
                                    break
                                if time.monotonic() >= solve_deadline:
                                    timed_out = True
                                    break
                                time.sleep(0.25)
                            if solver_done_path.is_file():
                                try:
                                    solver_exit_code = int(
                                        solver_exit_path.read_text(
                                            encoding="ascii"
                                        ).strip()
                                    )
                                except (OSError, ValueError) as exc:
                                    lifecycle_error = (
                                        f"solver exit-code barrier is invalid: {exc}"
                                    )
                                try:
                                    cgroup_metrics = cgroup_reader(cgroup_path)
                                    cgroup_error = None
                                except (OSError, ValueError) as exc:
                                    cgroup_error = str(exc)
                                collected_path.touch(exist_ok=False)
                                if os.name == "posix":
                                    collected_path.chmod(0o666)
                                wait_command = ["docker", "wait", container_id]
                                lifecycle_commands.append(wait_command)
                                waited = runner(
                                    wait_command,
                                    check=False,
                                    capture_output=True,
                                    timeout=60,
                                )
                                if int(waited.returncode) != 0:
                                    lifecycle_error = (
                                        lifecycle_error or "docker wait failed"
                                    )
                                    stderr += waited.stderr or b"docker wait failed\n"
                            else:
                                kill_command = ["docker", "kill", container_id]
                                lifecycle_commands.append(kill_command)
                                killed = runner(
                                    kill_command,
                                    check=False,
                                    capture_output=True,
                                    timeout=60,
                                )
                                if int(killed.returncode) != 0:
                                    stderr += killed.stderr or b"docker kill failed\n"
                                reap_command = ["docker", "wait", container_id]
                                lifecycle_commands.append(reap_command)
                                reaped = runner(
                                    reap_command,
                                    check=False,
                                    capture_output=True,
                                    timeout=60,
                                )
                                if int(reaped.returncode) != 0:
                                    stderr += reaped.stderr or b"docker wait failed\n"
                        if cgroup_path is None and not launch_prevented:
                            cgroup_error = "host cgroup-v2 path was not captured"
                        logs_command = ["docker", "logs", container_id]
                        lifecycle_commands.append(logs_command)
                        logs = runner(
                            logs_command,
                            check=False,
                            capture_output=True,
                            timeout=60,
                        )
                        stdout += logs.stdout or b""
                        stderr += logs.stderr or b""
                        final_inspect = [
                            "docker",
                            "inspect",
                            "--format",
                            "{{json .}}",
                            container_id,
                        ]
                        lifecycle_commands.append(final_inspect)
                        inspected = runner(
                            final_inspect,
                            check=False,
                            capture_output=True,
                            timeout=60,
                        )
                        if int(inspected.returncode) != 0:
                            lifecycle_error = lifecycle_error or "final docker inspect failed"
                            stderr += inspected.stderr or b"final docker inspect failed\n"
                        else:
                            try:
                                state, host_config = _parse_container_inspection(
                                    inspected.stdout or b""
                                )
                                return_code = (
                                    None if state["Running"] else int(state["ExitCode"])
                                )
                                if (
                                    solver_exit_code is not None
                                    and return_code != solver_exit_code
                                ):
                                    raise ValueError(
                                        "solver barrier exit code differs from Docker state"
                                    )
                            except ValueError as exc:
                                lifecycle_error = lifecycle_error or str(exc)
        except subprocess.TimeoutExpired as exc:
            if container_id is None:
                launch_prevented = True
                create_ambiguous = lifecycle_stage == "create"
            else:
                timed_out = True
            stdout += exc.stdout or b""
            stderr += exc.stderr or b""
        except (OSError, ValueError) as exc:
            if container_id is None:
                launch_prevented = True
            else:
                lifecycle_error = str(exc)
            stderr += f"{type(exc).__name__}: {exc}\n".encode()
        finally:
            try:
                (
                    cleanup_attempted,
                    cleanup_absence_verified,
                    cleanup_diagnostics,
                ) = _cleanup_container_attempt(
                    runner=runner,
                    registration=registration,
                    entry=entry,
                    container_name=container_name,
                    attempt_id=attempt_id,
                    initial_target=container_id,
                    ambiguous_create=create_ambiguous,
                    lifecycle_commands=lifecycle_commands,
                    poll_seconds=cleanup_poll_seconds,
                    stable_checks=cleanup_stable_checks,
                    max_checks=cleanup_max_checks,
                )
                stderr += cleanup_diagnostics
                cleanup_succeeded = cleanup_absence_verified
            except ValueError as exc:
                cleanup_succeeded = False
                cleanup_absence_verified = False
                stderr += f"container cleanup contract failed: {exc}\n".encode()
            ready_path.unlink(missing_ok=True)
            go_path.unlink(missing_ok=True)
            solver_done_path.unlink(missing_ok=True)
            solver_exit_path.unlink(missing_ok=True)
            collected_path.unlink(missing_ok=True)
        _publish_bytes(stdout, paths["stdout_log"])
        _publish_bytes(stderr, paths["stderr_log"])
        fem_payload = None
        fem_validation_error = None
        fem_failed_gates: tuple[str, ...] = ()
        fem_path = paths["fem_smoke_json"]
        if fem_path.exists():
            mode = fem_path.lstat().st_mode
            if fem_path.is_symlink() or not stat.S_ISREG(mode):
                fem_validation_error = (
                    "fem_smoke artifact must be a regular non-symlink file"
                )
            elif os.name == "posix" and stat.S_IMODE(mode) != 0o644:
                fem_validation_error = "fem_smoke artifact mode must be exactly 0644"
        if fem_validation_error is None and fem_path.is_file():
            try:
                candidate = json.loads(fem_path.read_bytes())
                if isinstance(candidate, Mapping):
                    fem_payload = candidate
                else:
                    fem_validation_error = "fem_smoke artifact must be a JSON object"
            except json.JSONDecodeError:
                fem_validation_error = "fem_smoke artifact is malformed JSON"
        if fem_payload is not None:
            try:
                fem_failed_gates = validate_fem_smoke_artifact(
                    registration, entry, fem_payload
                )
            except ValueError as exc:
                fem_validation_error = str(exc)
        evidence_consistent = True
        if fem_path.exists():
            evidence_consistent = bool(
                fem_validation_error is None
                and fem_payload is not None
                and (
                    (
                        fem_payload.get("status") == "PASSED"
                        and fem_payload.get("passed") is True
                        and return_code == 0
                    )
                    or (
                        fem_payload.get("status") == "FAILED"
                        and fem_payload.get("passed") is False
                        and return_code == 1
                    )
                )
            )
        status, passed, reason = classify_run_result(
            docker_return_code=return_code,
            fem_payload=fem_payload,
            launch_prevented=launch_prevented,
        )
        if fem_validation_error is not None:
            status = "FAILED"
            passed = False
            reason = fem_validation_error
        elif fem_failed_gates:
            reason = "fem_smoke FAILED gates: " + ", ".join(fem_failed_gates)
        if lifecycle_error is not None:
            status = "FAILED"
            passed = False
            reason = lifecycle_error
        if state is not None and (
            state.get("Running") is True
            or state.get("OOMKilled") is True
            or not isinstance(state.get("FinishedAt"), str)
            or not state["FinishedAt"]
            or state["FinishedAt"].startswith("0001-")
        ):
            status = "FAILED"
            passed = False
            reason = "Docker container did not finish cleanly"
        requested_memory = entry["resource_contract"]["cgroup_memory_limit_bytes"]
        resource_contract_passed = bool(
            host_config is not None
            and host_config.get("Memory") == requested_memory
            and host_config.get("MemorySwap") == requested_memory
            and cgroup_metrics is not None
            and cgroup_metrics.get("limit_bytes") == requested_memory
            and cgroup_metrics.get("swap_limit_bytes") == 0
        )
        if state is not None and not resource_contract_passed:
            status = "FAILED"
            passed = False
            reason = "Docker memory/no-swap resource contract was not applied"
        if state is not None and state.get("OOMKilled") is True:
            status = "FAILED"
            passed = False
            reason = "Docker cgroup OOMKilled the registered solve"
        elif return_code == 137:
            status = "FAILED"
            passed = False
            reason = "registered solve exited 137 under the cgroup limit"
        if (
            cgroup_error is not None
            and not launch_prevented
            and lifecycle_error is None
        ):
            status = "FAILED"
            passed = False
            reason = f"cgroup instrumentation FAILED: {cgroup_error}"
        if cgroup_metrics is not None and any(
            cgroup_metrics["events"].get(name, 0) > 0 for name in ("oom", "oom_kill")
        ):
            status = "FAILED"
            passed = False
            reason = "host cgroup memory.events recorded OOM activity"
        if timed_out:
            status = "FAILED"
            passed = False
            reason = "registered solve exceeded the executor timeout"
        if cleanup_succeeded is not True:
            status = "FAILED"
            passed = False
            reason = "container cleanup or post-removal absence verification FAILED"
        result = {
            "schema": RESULT_SCHEMA,
            "registration_id": registration["registration_id"],
            "run_id": entry["run_id"],
            "executor_attempt_id": attempt_id,
            "status": status,
            "passed": passed,
            "docker_return_code": return_code,
            "solver_barrier_exit_code": solver_exit_code,
            "launch_prevented": launch_prevented,
            "timed_out": timed_out,
            "wall_time_limit_seconds": registered_timeout,
            "executor_wall_seconds": time.monotonic() - started_monotonic,
            "container_state": dict(state) if state is not None else None,
            "container_host_config": (
                {
                    "Memory": host_config.get("Memory"),
                    "MemorySwap": host_config.get("MemorySwap"),
                }
                if host_config is not None
                else None
            ),
            "container_resource_contract_passed": resource_contract_passed,
            "host_cgroup": cgroup_metrics,
            "host_cgroup_error": cgroup_error,
            "container_cleanup_attempted": cleanup_attempted,
            "container_cleanup_succeeded": cleanup_succeeded,
            "container_absence_verified": cleanup_absence_verified,
            "fem_smoke_artifact_present": paths["fem_smoke_json"].is_file(),
            "fem_smoke_failed_gates": list(fem_failed_gates),
            "evidence_consistent": evidence_consistent,
            "reason": reason,
            "docker_lifecycle_commands": lifecycle_commands,
        }
        _publish_json(result, paths["exit_code_json"])
        _publish_bytes(
            _sha256_manifest(run_directory, paths["sha256_manifest"]),
            paths["sha256_manifest"],
        )
        if os.name == "posix":
            run_directory.chmod(0o755)
        summaries.append(result)
        if not evidence_consistent:
            raise RuntimeError(
                "inconsistent fem_smoke/Docker evidence was preserved; "
                "refusing to launch another registered run"
            )
    return summaries


def _docker_json(
    command: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Any:
    try:
        completed = runner(
            list(command), check=True, capture_output=True, text=True, timeout=60
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        raise ValueError(f"Docker identity inspection failed: {exc}") from exc
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Docker identity inspection returned malformed JSON") from exc


def inspect_docker_image(image_reference: str) -> dict[str, Any]:
    payload = _docker_json(("docker", "image", "inspect", image_reference))
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], Mapping):
        raise ValueError("Docker image inspection must return exactly one image")
    return dict(payload[0])


def read_image_runtime_metadata(image_reference: str) -> dict[str, Any]:
    payload = _docker_json(
        (
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "cat",
            image_reference,
            "/opt/scatter3d/runtime-metadata.json",
        )
    )
    if not isinstance(payload, Mapping):
        raise ValueError("image runtime metadata must be a JSON object")
    return dict(payload)


def main() -> int:
    repository_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).with_name("scaling_sweep_v1.json"),
    )
    parser.add_argument("--repository", type=Path, default=repository_default)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="execute only this registered run id; repeat for multiple entries",
    )
    args = parser.parse_args()
    try:
        if not args.registration.is_file():
            raise ValueError("complete registration file does not exist")
        if os.name == "posix" and stat.S_IMODE(args.registration.stat().st_mode) != 0o644:
            raise ValueError("registration file mode must be exactly 0644")
        registration, registration_bytes = _read_json_object(
            args.registration, "registration"
        )
        spec, spec_bytes = _read_json_object(args.spec, "scaling sweep specification")
        source = register_scaling_sweep._git_source(args.repository)
        image = inspect_docker_image(args.image)
        immutable_image_id = validate_image_inspection(registration, image)
        runtime_metadata = read_image_runtime_metadata(immutable_image_id)
        validate_execution_context(
            registration,
            spec=spec,
            spec_bytes=spec_bytes,
            source=source,
            runtime_metadata=runtime_metadata,
        )
        summaries = execute_registered_entries(
            registration,
            image_reference=immutable_image_id,
            output_parent=args.output_parent,
            timeout_seconds=args.timeout_seconds,
            run_ids=args.run_ids,
        )
        if args.registration.read_bytes() != registration_bytes:
            raise RuntimeError("registration changed during execution")
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(json.dumps(summaries, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(item["status"] == "PASSED" for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
