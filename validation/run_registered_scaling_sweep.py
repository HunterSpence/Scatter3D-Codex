#!/usr/bin/env python3
"""Execute an immutable registered scaling sweep through host-side Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from validation import register_scaling_sweep

RESULT_SCHEMA = "scatter3d.validation.scaling_sweep_run_result/v1"


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
    return [
        "docker",
        "create",
        "--name",
        container_name,
        "--label",
        f"scatter3d.registration_id={registration['registration_id']}",
        "--label",
        f"scatter3d.run_id={entry['run_id']}",
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
        if status == "FAILED" and passed is False:
            return "FAILED", False, "fem_smoke artifact reported FAILED"
        return "FAILED", False, "fem_smoke artifact status/exit code is inconsistent"
    if launch_prevented:
        return "BLOCKED", None, "Docker could not launch the registered solve"
    return "FAILED", False, "registered solve produced no fem_smoke artifact"


def validate_fem_smoke_artifact(
    registration: Mapping[str, Any],
    entry: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    """Fail closed unless a completed FEM artifact matches its registration."""

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

    def registered_argument(name: str) -> str:
        argv = entry["command"]
        try:
            return str(argv[argv.index(name) + 1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"registered command is missing {name}") from exc

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
    expected_dofs = int(registered_argument("--expected-global-dofs"))
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
        create_command = build_docker_create_command(
            registration,
            entry,
            image_reference=image_reference,
            host_run_directory=run_directory,
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
        cleanup_target: str | None = None
        state: Mapping[str, Any] | None = None
        host_config: Mapping[str, Any] | None = None
        cgroup_path: Path | None = None
        cgroup_metrics: dict[str, Any] | None = None
        cgroup_error: str | None = None
        lifecycle_commands: list[list[str]] = [create_command]
        cleanup_attempted = False
        cleanup_succeeded: bool | None = None
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
                    cleanup_target = container_name
                    stderr += b"docker create returned no container id\n"
                else:
                    container_id = raw_container_id.splitlines()[-1]
                    cleanup_target = container_id
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
            if cleanup_target is not None:
                cleanup_attempted = True
                remove_command = ["docker", "rm", "--force", cleanup_target]
                lifecycle_commands.append(remove_command)
                try:
                    removed = runner(
                        remove_command,
                        check=False,
                        capture_output=True,
                        timeout=60,
                    )
                    cleanup_succeeded = int(removed.returncode) == 0
                    if not cleanup_succeeded:
                        stderr += removed.stderr or b"docker rm failed\n"
                except (OSError, subprocess.SubprocessError) as exc:
                    cleanup_succeeded = False
                    stderr += f"docker rm failed: {exc}\n".encode()
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
            except json.JSONDecodeError:
                fem_payload = None
        if fem_payload is not None:
            try:
                fem_failed_gates = validate_fem_smoke_artifact(
                    registration, entry, fem_payload
                )
            except ValueError as exc:
                fem_validation_error = str(exc)
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
        if cgroup_metrics is not None and cgroup_metrics["events"].get("oom_kill", 0) > 0:
            status = "FAILED"
            passed = False
            reason = "host cgroup memory.events recorded oom_kill"
        if timed_out:
            status = "FAILED"
            passed = False
            reason = "registered solve exceeded the executor timeout"
        if status == "PASSED" and cleanup_succeeded is not True:
            status = "FAILED"
            passed = False
            reason = "registered solve passed but container cleanup FAILED"
        result = {
            "schema": RESULT_SCHEMA,
            "registration_id": registration["registration_id"],
            "run_id": entry["run_id"],
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
            "fem_smoke_artifact_present": paths["fem_smoke_json"].is_file(),
            "fem_smoke_failed_gates": list(fem_failed_gates),
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
