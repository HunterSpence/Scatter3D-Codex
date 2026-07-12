from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from validation import register_scaling_sweep, run_registered_scaling_sweep

SPEC_PATH = Path(__file__).parents[1] / "validation" / "scaling_sweep_v1.json"
RUNTIME = {
    "dolfinx": "0.10.0",
    "mpi4py": "4.1.1",
    "mpi_library": "MPICH 4.3.1",
    "petsc": "3.24.0",
    "petsc_scalar_type": "complex128",
}


def _registration() -> dict:
    encoded = SPEC_PATH.read_bytes()
    return register_scaling_sweep.build_registration(
        json.loads(encoded),
        encoded,
        source={"commit": "a" * 40, "dirty": False},
        project_image_kind="local_image_id",
        project_image_identity="sha256:" + "b" * 64,
        base_image_digest="sha256:" + "c" * 64,
        base_runtime_metadata=RUNTIME,
    )


def _fem_payload(registration: dict, entry: dict) -> dict:
    passed_gate = {"status": "PASSED", "passed": True}
    command = entry["command"][4:]
    expected_dofs = int(command[command.index("--expected-global-dofs") + 1])
    gates = {
        name: dict(passed_gate)
        for name in (
            "convergence_and_true_residual",
            "assembly_setup_rhs_counts",
            "p_multigrid_structure",
            "expected_global_dofs",
            "release_comparison_provenance",
        )
    }
    gates["expected_global_dofs"].update(
        expected=expected_dofs, observed=expected_dofs
    )
    return {
        "status": "PASSED",
        "passed": True,
        "source": {
            **registration["source"],
            "provenance": "environment",
        },
        "command": {"argv": command, "provenance": "process"},
        "physical_problem": deepcopy(entry["physical_problem"]),
        "images": {
            "project_image": {
                "digest": None,
                "local_image_id": registration["images"]["project_image"][
                    "identity"
                ],
            },
            "base_image": {
                "digest": registration["images"]["base_image"]["identity"],
                "local_image_id": None,
            },
        },
        "mpi_size": entry["resource_contract"]["mpi_ranks"],
        "preconditioner": {
            "requested_absorption_shift": entry[
                "preconditioner_absorption_shift"
            ]
        },
        "cgroup_memory": {
            "version": "v2",
            "peak_bytes": 1_000_000,
            "swap_limit_bytes": 0,
            "limit_bytes": entry["resource_contract"][
                "cgroup_memory_limit_bytes"
            ]
        },
        "runtime": {
            "dolfinx_version": "0.10.0",
            "mpi4py_version": "4.1.1",
            "mpi_library_version": "MPICH 4.3.1",
            "petsc_version": [3, 24, 0],
            "petsc_scalar_type": "<class 'numpy.complex128'>",
        },
        "frequency_diagnostics": [{"global_complex_dofs": expected_dofs}],
        "gates": gates,
    }


def _cgroup_kwargs(entry: dict, path: Path) -> dict:
    limit = entry["resource_contract"]["cgroup_memory_limit_bytes"]
    return {
        "cgroup_resolver": lambda pid: path if pid > 0 else None,
        "cgroup_reader": lambda observed_path: {
            "version": "v2-host",
            "path": str(observed_path),
            "peak_bytes": 1_000_000,
            "limit_bytes": limit,
            "swap_limit_bytes": 0,
            "events": {"oom": 0, "oom_kill": 0, "max": 0},
        },
        "cgroup_populated": lambda observed_path: observed_path == path,
    }


def test_image_inspection_requires_registered_ids_and_labels() -> None:
    registration = _registration()
    image = {
        "Id": registration["images"]["project_image"]["identity"],
        "RepoDigests": [],
        "Config": {
            "Labels": {
                "org.opencontainers.image.revision": registration["source"][
                    "commit"
                ],
                "org.opencontainers.image.base.digest": registration["images"][
                    "base_image"
                ]["identity"],
            },
            "Env": [
                f"SCATTER3D_GIT_COMMIT={registration['source']['commit']}",
                "SCATTER3D_GIT_DIRTY=false",
            ],
        },
    }
    assert (
        run_registered_scaling_sweep.validate_image_inspection(
            registration, image
        )
        == image["Id"]
    )
    image["Config"]["Labels"]["org.opencontainers.image.revision"] = "d" * 40
    with pytest.raises(ValueError, match="revision"):
        run_registered_scaling_sweep.validate_image_inspection(
            registration, image
        )
    image["Config"]["Labels"]["org.opencontainers.image.revision"] = registration[
        "source"
    ]["commit"]
    image["Config"]["Labels"]["org.opencontainers.image.base.digest"] = (
        "sha256:" + "d" * 64
    )
    with pytest.raises(ValueError, match="base digest"):
        run_registered_scaling_sweep.validate_image_inspection(registration, image)
    image["Config"]["Labels"]["org.opencontainers.image.base.digest"] = registration[
        "images"
    ]["base_image"]["identity"]
    image["Config"]["Env"] = [
        f"SCATTER3D_GIT_COMMIT={registration['source']['commit']}",
        "SCATTER3D_GIT_DIRTY=true",
    ]
    with pytest.raises(ValueError, match="source environment"):
        run_registered_scaling_sweep.validate_image_inspection(registration, image)


def test_docker_create_enforces_registered_memory_image_and_output_mount(
    tmp_path: Path,
) -> None:
    registration = _registration()
    entry = registration["entries"][0]
    command = run_registered_scaling_sweep.build_docker_create_command(
        registration,
        entry,
        image_reference=registration["images"]["project_image"]["identity"],
        host_run_directory=tmp_path,
    )
    limit = str(entry["resource_contract"]["cgroup_memory_limit_bytes"])
    assert command[command.index("--memory") + 1] == limit
    assert command[command.index("--memory-swap") + 1] == limit
    assert "--ipc=host" in command
    assert f"{tmp_path}:{entry['outputs']['container_directory']}" in command
    assert command[command.index("--name") + 1].startswith("scatter3d-")
    wrapper = command[command.index("sh") + 2]
    assert ".solver-done" in wrapper
    assert ".executor-collected" in wrapper
    assert "exec \"$@\"" not in wrapper
    assert command[-len(entry["command"]) :] == entry["command"]


def test_fem_artifact_validation_is_fail_closed() -> None:
    registration = _registration()
    entry = registration["entries"][0]
    payload = _fem_payload(registration, entry)
    run_registered_scaling_sweep.validate_fem_smoke_artifact(
        registration, entry, payload
    )
    payload["gates"]["expected_global_dofs"] = {
        "status": "FAILED",
        "passed": False,
    }
    with pytest.raises(ValueError, match=r"global DoF|gate"):
        run_registered_scaling_sweep.validate_fem_smoke_artifact(
            registration, entry, payload
        )


def test_failed_fem_artifact_remains_valid_diagnostic_evidence() -> None:
    registration = _registration()
    entry = registration["entries"][0]
    payload = _fem_payload(registration, entry)
    payload["status"] = "FAILED"
    payload["passed"] = False
    payload["gates"]["convergence_and_true_residual"] = {
        "status": "FAILED",
        "passed": False,
        "reason": "iteration cap",
    }
    failed = run_registered_scaling_sweep.validate_fem_smoke_artifact(
        registration, entry, payload
    )
    assert failed == ("convergence_and_true_residual",)


def test_failed_exact_dof_gate_remains_valid_diagnostic_evidence() -> None:
    registration = _registration()
    entry = registration["entries"][0]
    payload = _fem_payload(registration, entry)
    expected = payload["gates"]["expected_global_dofs"]["expected"]
    observed = expected + 1
    payload["status"] = "FAILED"
    payload["passed"] = False
    payload["execution_phase"] = "preflight"
    for name in (
        "convergence_and_true_residual",
        "assembly_setup_rhs_counts",
        "p_multigrid_structure",
    ):
        payload["gates"][name] = {
            "status": "NOT RUN",
            "passed": None,
            "reason": "exact global DoF preflight FAILED before solve",
        }
    payload["gates"]["expected_global_dofs"] = {
        "status": "FAILED",
        "passed": False,
        "expected": expected,
        "observed": observed,
    }
    payload["frequency_diagnostics"] = [{"global_complex_dofs": observed}]
    assert run_registered_scaling_sweep.validate_fem_smoke_artifact(
        registration, entry, payload
    ) == ("expected_global_dofs",)


def test_fem_artifact_rejects_physical_problem_tampering() -> None:
    registration = _registration()
    entry = registration["entries"][0]
    payload = _fem_payload(registration, entry)
    payload["physical_problem"]["definition"]["frequencies_hz"] = [200_000_000.0]
    with pytest.raises(ValueError, match="physical problem"):
        run_registered_scaling_sweep.validate_fem_smoke_artifact(
            registration, entry, payload
        )


def test_non_preflight_registered_artifact_rejects_not_run_outcome_gate() -> None:
    registration = _registration()
    entry = registration["entries"][0]
    payload = _fem_payload(registration, entry)
    payload["status"] = "FAILED"
    payload["passed"] = False
    payload["gates"]["convergence_and_true_residual"] = {
        "status": "FAILED",
        "passed": False,
    }
    payload["gates"]["p_multigrid_structure"] = {
        "status": "NOT RUN",
        "passed": None,
    }
    with pytest.raises(ValueError, match="must not be NOT RUN"):
        run_registered_scaling_sweep.validate_fem_smoke_artifact(
            registration, entry, payload
        )


def test_executor_preserves_result_files_manifest_and_no_clobber(
    tmp_path: Path,
) -> None:
    registration = _registration()
    registration["entries"] = registration["entries"][:1]
    entry = registration["entries"][0]
    host_root = tmp_path / registration["output_root"]
    run_directory = host_root / entry["run_id"]
    inspect_count = 0

    def runner(command, **kwargs):
        nonlocal inspect_count
        del kwargs
        if command[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(command, 0, b"container-id\n", b"")
        if command[:2] == ["docker", "start"]:
            run_directory.joinpath(".executor-ready").touch()
            run_directory.joinpath("fem-smoke.json").write_text(
                json.dumps(_fem_payload(registration, entry)), encoding="utf-8"
            )
            run_directory.joinpath(".solver-exit-code").write_text(
                "0\n", encoding="ascii"
            )
            run_directory.joinpath(".solver-done").touch()
            return subprocess.CompletedProcess(command, 0, b"container-id\n", b"")
        if command[:2] == ["docker", "wait"]:
            return subprocess.CompletedProcess(command, 0, b"0\n", b"")
        if command[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(command, 0, b"solver output\n", b"")
        if command[:2] == ["docker", "inspect"]:
            inspect_count += 1
            running = inspect_count == 1
            inspection = {
                "State": {
                    "StartedAt": "2026-07-12T00:00:00Z",
                    "FinishedAt": (
                        "0001-01-01T00:00:00Z"
                        if running
                        else "2026-07-12T00:01:00Z"
                    ),
                    "ExitCode": 0,
                    "Running": running,
                    "OOMKilled": False,
                    "Pid": 123 if running else 0,
                },
                "HostConfig": {
                    "Memory": entry["resource_contract"][
                        "cgroup_memory_limit_bytes"
                    ],
                    "MemorySwap": entry["resource_contract"][
                        "cgroup_memory_limit_bytes"
                    ],
                },
            }
            return subprocess.CompletedProcess(
                command, 0, json.dumps(inspection).encode(), b""
            )
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, b"container-id\n", b"")
        raise AssertionError(command)

    summaries = run_registered_scaling_sweep.execute_registered_entries(
        registration,
        image_reference=registration["images"]["project_image"]["identity"],
        output_parent=tmp_path,
        runner=runner,
        **_cgroup_kwargs(entry, tmp_path),
    )
    assert summaries[0]["status"] == "PASSED"
    assert json.loads((run_directory / "exit-code.json").read_text())["status"] == "PASSED"
    assert (run_directory / "stdout.log").read_text() == "solver output\n"
    manifest = (run_directory / "SHA256SUMS").read_text()
    assert "fem-smoke.json" in manifest
    assert "exit-code.json" in manifest
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        run_registered_scaling_sweep.execute_registered_entries(
            registration,
            image_reference=registration["images"]["project_image"]["identity"],
            output_parent=tmp_path,
            runner=runner,
        )


def test_classification_distinguishes_blocked_from_failed() -> None:
    assert run_registered_scaling_sweep.classify_run_result(
        docker_return_code=None,
        fem_payload=None,
        launch_prevented=True,
    )[0] == "BLOCKED"
    assert run_registered_scaling_sweep.classify_run_result(
        docker_return_code=137,
        fem_payload=None,
        launch_prevented=False,
    )[0] == "FAILED"


@pytest.mark.parametrize(
    "relative",
    (
        "/system.slice/docker-container.scope",
        "/docker/container-id",
    ),
)
def test_cgroup_path_resolution_supports_systemd_and_cgroupfs(
    tmp_path: Path, relative: str
) -> None:
    expected = tmp_path.joinpath(relative.lstrip("/"))
    expected.mkdir(parents=True)
    assert run_registered_scaling_sweep._resolve_cgroup_relative_path(
        relative, tmp_path
    ) == expected.resolve()
    with pytest.raises(ValueError, match="traversal"):
        run_registered_scaling_sweep._resolve_cgroup_relative_path(
            "/docker/../escape", tmp_path
        )


def test_cgroup_v2_metrics_require_peak_limits_swap_and_events(
    tmp_path: Path,
) -> None:
    values = {
        "memory.peak": "1234\n",
        "memory.max": "5678\n",
        "memory.swap.max": "0\n",
        "memory.events": "low 0\nhigh 0\nmax 2\noom 1\noom_kill 1\n",
    }
    for name, value in values.items():
        tmp_path.joinpath(name).write_text(value, encoding="utf-8")
    metrics = run_registered_scaling_sweep._read_cgroup_v2_metrics(tmp_path)
    assert metrics["peak_bytes"] == 1234
    assert metrics["events"]["oom_kill"] == 1
    tmp_path.joinpath("cgroup.events").write_text(
        "populated 1\nfrozen 0\n", encoding="utf-8"
    )
    assert run_registered_scaling_sweep._cgroup_is_populated(tmp_path) is True
    tmp_path.joinpath("memory.peak").write_text("max\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"memory\.peak"):
        run_registered_scaling_sweep._read_cgroup_v2_metrics(tmp_path)


def test_container_inspection_rejects_malformed_exit_code() -> None:
    payload = {
        "State": {
            "Running": False,
            "OOMKilled": False,
            "ExitCode": None,
            "Pid": 0,
            "StartedAt": "2026-07-12T00:00:00Z",
            "FinishedAt": "2026-07-12T00:01:00Z",
        },
        "HostConfig": {"Memory": 1, "MemorySwap": 1},
    }
    with pytest.raises(ValueError, match="ExitCode"):
        run_registered_scaling_sweep._parse_container_inspection(
            json.dumps(payload).encode()
        )


def test_executor_can_select_remaining_registered_run_after_interruption(
    tmp_path: Path,
) -> None:
    registration = _registration()
    first, second = registration["entries"][:2]
    existing = tmp_path / registration["output_root"] / first["run_id"]
    existing.mkdir(parents=True)

    def blocked_runner(command, **kwargs):
        del kwargs
        if command[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(command, 1, b"", b"capacity")
        raise AssertionError(command)

    summaries = run_registered_scaling_sweep.execute_registered_entries(
        registration,
        image_reference=registration["images"]["project_image"]["identity"],
        output_parent=tmp_path,
        runner=blocked_runner,
        run_ids=[second["run_id"]],
    )
    assert summaries[0]["run_id"] == second["run_id"]
    assert summaries[0]["status"] == "BLOCKED"


def test_executor_persists_failed_evidence_for_malformed_inspect(
    tmp_path: Path,
) -> None:
    registration = _registration()
    registration["entries"] = registration["entries"][:1]
    entry = registration["entries"][0]
    run_directory = tmp_path / registration["output_root"] / entry["run_id"]

    def runner(command, **kwargs):
        del kwargs
        if command[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(command, 0, b"container-id\n", b"")
        if command[:2] == ["docker", "start"]:
            run_directory.joinpath(".executor-ready").touch()
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:2] == ["docker", "inspect"]:
            malformed = {
                "State": {
                    "Running": True,
                    "OOMKilled": False,
                    "ExitCode": None,
                    "Pid": 123,
                    "StartedAt": "2026-07-12T00:00:00Z",
                    "FinishedAt": "0001-01-01T00:00:00Z",
                },
                "HostConfig": {},
            }
            return subprocess.CompletedProcess(
                command, 0, json.dumps(malformed).encode(), b""
            )
        if command[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        raise AssertionError(command)

    result = run_registered_scaling_sweep.execute_registered_entries(
        registration,
        image_reference=registration["images"]["project_image"]["identity"],
        output_parent=tmp_path,
        runner=runner,
    )[0]
    assert result["status"] == "FAILED"
    assert "ExitCode" in result["reason"]
    assert (run_directory / "exit-code.json").is_file()


def test_executor_rejects_unknown_or_empty_run_selection(tmp_path: Path) -> None:
    registration = _registration()
    common = {
        "image_reference": registration["images"]["project_image"]["identity"],
        "output_parent": tmp_path,
    }
    with pytest.raises(ValueError, match="unregistered run ids"):
        run_registered_scaling_sweep.execute_registered_entries(
            registration, run_ids=["not-registered"], **common
        )
    with pytest.raises(ValueError, match="at least one"):
        run_registered_scaling_sweep.execute_registered_entries(
            registration, run_ids=[], **common
        )
    with pytest.raises(ValueError, match="differs from registered"):
        run_registered_scaling_sweep.execute_registered_entries(
            registration,
            timeout_seconds=1,
            run_ids=[registration["entries"][0]["run_id"]],
            **common,
        )


@pytest.mark.parametrize(
    ("state_updates", "host_updates", "expected_reason"),
    [
        ({"Running": True, "FinishedAt": "0001-01-01T00:00:00Z"}, {}, "finish"),
        ({"OOMKilled": True, "ExitCode": 137}, {}, "OOMKilled"),
        ({"ExitCode": 137}, {}, "exited 137"),
        ({}, {"MemorySwap": 0}, "resource contract"),
    ],
)
def test_executor_fails_closed_on_container_state_and_resources(
    tmp_path: Path,
    state_updates: dict,
    host_updates: dict,
    expected_reason: str,
) -> None:
    registration = _registration()
    registration["entries"] = registration["entries"][:1]
    entry = registration["entries"][0]
    run_directory = (
        tmp_path / registration["output_root"] / entry["run_id"]
    )
    limit = entry["resource_contract"]["cgroup_memory_limit_bytes"]
    state = {
        "StartedAt": "2026-07-12T00:00:00Z",
        "FinishedAt": "2026-07-12T00:01:00Z",
        "ExitCode": 0,
        "Running": False,
        "OOMKilled": False,
        **state_updates,
    }
    host_config = {"Memory": limit, "MemorySwap": limit, **host_updates}
    inspect_count = 0

    def runner(command, **kwargs):
        nonlocal inspect_count
        del kwargs
        if command[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(command, 0, b"container-id\n", b"")
        if command[:2] == ["docker", "start"]:
            run_directory.joinpath(".executor-ready").touch()
            run_directory.joinpath("fem-smoke.json").write_text(
                json.dumps(_fem_payload(registration, entry)), encoding="utf-8"
            )
            run_directory.joinpath(".solver-exit-code").write_text(
                f"{state['ExitCode']}\n", encoding="ascii"
            )
            run_directory.joinpath(".solver-done").touch()
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:2] == ["docker", "wait"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:2] == ["docker", "inspect"]:
            inspect_count += 1
            observed_state = (
                {
                    "StartedAt": "2026-07-12T00:00:00Z",
                    "FinishedAt": "0001-01-01T00:00:00Z",
                    "ExitCode": 0,
                    "Running": True,
                    "OOMKilled": False,
                    "Pid": 123,
                }
                if inspect_count == 1
                else {**state, "Pid": 0 if not state["Running"] else 123}
            )
            body = json.dumps(
                {"State": observed_state, "HostConfig": host_config}
            ).encode()
            return subprocess.CompletedProcess(command, 0, body, b"")
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        raise AssertionError(command)

    result = run_registered_scaling_sweep.execute_registered_entries(
        registration,
        image_reference=registration["images"]["project_image"]["identity"],
        output_parent=tmp_path,
        runner=runner,
        **_cgroup_kwargs(entry, tmp_path),
    )[0]
    assert result["status"] == "FAILED"
    assert expected_reason in result["reason"]


def test_executor_timeout_and_cleanup_failure_are_failed(tmp_path: Path) -> None:
    registration = _registration()
    registration["entries"] = registration["entries"][:1]
    entry = registration["entries"][0]
    entry["resource_contract"]["wall_time_limit_seconds"] = 1
    run_directory = tmp_path / registration["output_root"] / entry["run_id"]
    inspect_count = 0

    def runner(command, **kwargs):
        nonlocal inspect_count
        del kwargs
        if command[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(command, 0, b"container-id\n", b"")
        if command[:2] == ["docker", "start"]:
            run_directory.joinpath(".executor-ready").touch()
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:2] == ["docker", "wait"]:
            return subprocess.CompletedProcess(command, 0, b"137\n", b"")
        if command[:2] == ["docker", "kill"]:
            return subprocess.CompletedProcess(command, 0, b"container-id\n", b"")
        if command[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:2] == ["docker", "inspect"]:
            inspect_count += 1
            running = inspect_count == 1
            state = {
                "State": {
                    "StartedAt": "2026-07-12T00:00:00Z",
                    "FinishedAt": (
                        "0001-01-01T00:00:00Z"
                        if running
                        else "2026-07-12T00:01:00Z"
                    ),
                    "ExitCode": 0 if running else 137,
                    "Running": running,
                    "OOMKilled": False,
                    "Pid": 123 if running else 0,
                },
                "HostConfig": {
                    "Memory": registration["entries"][0]["resource_contract"][
                        "cgroup_memory_limit_bytes"
                    ],
                    "MemorySwap": registration["entries"][0]["resource_contract"][
                        "cgroup_memory_limit_bytes"
                    ],
                },
            }
            return subprocess.CompletedProcess(
                command, 0, json.dumps(state).encode(), b""
            )
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 1, b"", b"cleanup failed")
        raise AssertionError(command)

    result = run_registered_scaling_sweep.execute_registered_entries(
        registration,
        image_reference=registration["images"]["project_image"]["identity"],
        output_parent=tmp_path,
        runner=runner,
            timeout_seconds=1,
            **_cgroup_kwargs(entry, tmp_path),
        )[0]
    assert result["status"] == "FAILED"
    assert result["timed_out"] is True
    assert result["container_cleanup_succeeded"] is False
