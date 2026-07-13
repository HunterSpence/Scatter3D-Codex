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
    frequency = float(command[command.index("--frequencies-hz") + 1])
    fine_degree = int(command[command.index("--degree") + 1])
    coarse_degree = int(command[command.index("--p-multigrid-coarse-degree") + 1])
    subdivisions = int(command[command.index("--subdivisions") + 1])
    maximum_iterations = int(command[command.index("--maximum-iterations") + 1])
    maximum_residual = float(
        command[command.index("--maximum-true-relative-residual") + 1]
    )
    restart = int(command[command.index("--gmres-restart") + 1])
    overlap = int(command[command.index("--asm-overlap") + 1])
    shift = float(command[command.index("--preconditioner-absorption-shift") + 1])
    mpi_ranks = entry["resource_contract"]["mpi_ranks"]
    coarse_dofs = max(1, expected_dofs // 10)
    petsc_options = {
        "ksp_gmres_restart": restart,
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_factor_mat_solver_type": "mumps",
        "mg_coarse_pc_type": "lu",
        "mg_levels_1_ksp_max_it": 1,
        "mg_levels_1_ksp_type": "richardson",
        "mg_levels_1_pc_asm_overlap": overlap,
        "mg_levels_1_pc_type": "asm",
        "mg_levels_1_sub_ksp_type": "preonly",
        "mg_levels_1_sub_pc_factor_mat_solver_type": "mumps",
        "mg_levels_1_sub_pc_type": "lu",
    }
    ranks = list(range(mpi_ranks))
    top = {
        "ksp_type": "fgmres",
        "pc_type": "mg",
        "factor_solver_type": None,
        "options_prefix": "scatter3d_0_",
        "instances": mpi_ranks,
        "mpi_ranks": ranks,
        "maximum_iterations": maximum_iterations,
        "norm_type": "unpreconditioned",
        "path": "top",
    }
    fine = {
        "ksp_type": "richardson",
        "pc_type": "asm",
        "factor_solver_type": None,
        "options_prefix": "scatter3d_0_mg_levels_1_",
        "instances": mpi_ranks,
        "mpi_ranks": ranks,
        "maximum_iterations": 1,
        "norm_type": "none",
        "path": "mg.fine",
    }
    subdomain = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "factor_solver_type": "mumps",
        "options_prefix": "scatter3d_0_mg_levels_1_sub_",
        "instances": mpi_ranks,
        "mpi_ranks": ranks,
        "maximum_iterations": 10_000,
        "norm_type": "none",
        "path": "mg.fine.asm.subdomains",
    }
    coarse = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "factor_solver_type": "mumps",
        "options_prefix": "scatter3d_0_mg_coarse_",
        "instances": mpi_ranks,
        "mpi_ranks": ranks,
        "maximum_iterations": 10_000,
        "norm_type": "none",
        "path": "mg.coarse",
    }
    effective = {
        "top_level": top,
        "preconditioning_side": "right",
        "maximum_iterations": maximum_iterations,
        "relative_tolerance": 1.0e-8,
        "absolute_tolerance": 1.0e-12,
        "pc_uses_amat": False,
        "mg_levels": 2,
        "mg_type": "multiplicative",
        "mg_cycle_type": "v",
        "mg_galerkin": "none",
        "mg_fine_smoother": fine,
        "mg_fine_asm_type": "restrict",
        "mg_fine_asm_overlap": overlap,
        "mg_fine_asm_subdomain_solvers": [subdomain],
        "mg_coarse_solver": coarse,
        "petsc_view_ascii": (
            "type: fgmres\nright preconditioning\ntype: mg\nlevels=2\n"
            "Not using Galerkin\ntype: richardson\ntype: asm\n"
            "type: preonly\ntype: lu\n"
            "package used to perform factorization: mumps\n"
        ),
    }
    transfer = {
        "direction": "coarse_to_fine",
        "rows": expected_dofs,
        "columns": coarse_dofs,
        "nonzeros": expected_dofs,
        "constrained_fine_rows": 1,
        "constrained_coarse_columns": 1,
        "maximum_imaginary_abs": 0.0,
        "assembly_seconds": 0.01,
        "memory_bytes_sum": None,
    }
    port_solves = [
        {
            "port_name": name,
            "iterations": 10,
            "converged_reason": 2,
            "true_relative_residual": 1.0e-9,
            "true_residual_norm": 1.0e-10,
            "reported_residual_history": [1.0, 1.0e-9],
            "solve_seconds": 0.1,
        }
        for name in ("left", "right")
    ]
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
    gates["convergence_and_true_residual"].update(
        maximum_true_relative_residual=maximum_residual
    )
    gates["assembly_setup_rhs_counts"].update(
        expected=deepcopy(expected_counts), observed=deepcopy(expected_counts)
    )
    gates["p_multigrid_structure"].update(
        requested_coarse_degree=coarse_degree
    )
    diagnostic = {
        "frequency_hz": frequency,
        "global_complex_dofs": expected_dofs,
        "fine_degree": fine_degree,
        "coarse_degree": coarse_degree,
        "coarse_global_complex_dofs": coarse_dofs,
        "matrix_nonzeros": expected_dofs * 10,
        "preconditioner_matrix_nonzeros": expected_dofs * 10,
        "coarse_preconditioner_matrix_nonzeros": coarse_dofs * 10,
        "matrix_memory_bytes_sum": None,
        "preconditioner_matrix_memory_bytes_sum": None,
        "coarse_preconditioner_matrix_memory_bytes_sum": None,
        "rank_peak_rss_bytes_max": 1_000_000,
        "rank_peak_rss_bytes_sum": 2_000_000,
        "assembly_seconds": 0.1,
        "preconditioner_assembly_seconds": 0.1,
        "coarse_preconditioner_assembly_seconds": 0.01,
        "setup_seconds": 0.2,
        "preconditioner_absorption_shift": shift,
        "preconditioner_operator_is_physical": shift == 0.0,
        "p_multigrid_operator_checks_passed": True,
        "transfer_operator": transfer,
        "solver_hierarchy": {
            "requested": {
                "ksp_type": "fgmres",
                "pc_type": "mg",
                "preconditioning_side": "right",
                "petsc_options": list(petsc_options.items()),
            },
            "effective": effective,
        },
        "port_solves": port_solves,
    }
    requested_solver = {
        "solver": "iterative",
        "iterative_hierarchy": "p-multigrid",
        "ksp_type": "fgmres",
        "pc_type": "mg",
        "preconditioning_side": "right",
        "maximum_iterations": maximum_iterations,
        "p_multigrid_coarse_degree": coarse_degree,
        "preconditioner_absorption_shift": shift,
        "petsc_options": petsc_options,
    }
    preconditioner_metrics = {
        "frequency_hz": frequency,
        "absorption_shift": shift,
        "operator_is_physical": shift == 0.0,
        "matrix_nonzeros": diagnostic["preconditioner_matrix_nonzeros"],
        "matrix_memory_bytes_sum": None,
        "assembly_seconds": diagnostic["preconditioner_assembly_seconds"],
        "coarse": {
            "degree": coarse_degree,
            "global_complex_dofs": coarse_dofs,
            "matrix_nonzeros": diagnostic[
                "coarse_preconditioner_matrix_nonzeros"
            ],
            "matrix_memory_bytes_sum": None,
            "assembly_seconds": diagnostic[
                "coarse_preconditioner_assembly_seconds"
            ],
        },
        "transfer_operator": transfer,
        "p_multigrid_operator_checks_passed": True,
    }
    return {
        "schema": "scatter3d.validation.fem_smoke/v2",
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
        "solver": "iterative",
        "iterative_hierarchy": "p-multigrid",
        "degree": fine_degree,
        "subdivisions": subdivisions,
        "frequencies_hz": [frequency],
        "scalar_type": "<class 'numpy.complex128'>",
        "solver_configuration": {
            "requested": requested_solver,
            "effective_by_frequency": [
                {"frequency_hz": frequency, "hierarchy": effective}
            ],
        },
        "preconditioner": {
            "requested_absorption_shift": entry[
                "preconditioner_absorption_shift"
            ],
            "requested_hierarchy": "p-multigrid",
            "requested_coarse_degree": coarse_degree,
            "per_frequency": [preconditioner_metrics],
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
        **expected_counts,
        "numeric_factorizations": expected_counts["global_numeric_factorizations"],
        "frequency_diagnostics": [diagnostic],
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


@pytest.mark.parametrize(
    "tamper",
    (
        "schema",
        "solver_configuration",
        "port_solves",
        "true_relative_residual",
        "negative_residual",
        "numeric_nonfinite_residual",
        "string_nonfinite_residual_norm",
        "string_nonfinite_history",
        "over_cap_iterations",
        "petsc_view_ascii",
        "hierarchy_ranks",
        "hierarchy_prefix",
        "rhs_solves",
        "operator_identity",
        "setup_seconds",
    ),
)
def test_fem_artifact_rejects_missing_or_tampered_underlying_evidence(
    tamper: str,
) -> None:
    registration = _registration()
    entry = registration["entries"][0]
    payload = _fem_payload(registration, entry)
    diagnostic = payload["frequency_diagnostics"][0]
    if tamper == "schema":
        payload.pop("schema")
    elif tamper == "solver_configuration":
        payload.pop("solver_configuration")
    elif tamper == "port_solves":
        diagnostic.pop("port_solves")
    elif tamper == "true_relative_residual":
        diagnostic["port_solves"][0].pop("true_relative_residual")
    elif tamper == "negative_residual":
        diagnostic["port_solves"][0]["true_relative_residual"] = -1.0
    elif tamper == "numeric_nonfinite_residual":
        diagnostic["port_solves"][0]["true_relative_residual"] = float("-inf")
    elif tamper == "string_nonfinite_residual_norm":
        diagnostic["port_solves"][0]["true_residual_norm"] = "NaN"
    elif tamper == "string_nonfinite_history":
        diagnostic["port_solves"][0]["reported_residual_history"] = [
            1.0,
            "+Infinity",
        ]
    elif tamper == "over_cap_iterations":
        diagnostic["port_solves"][0]["iterations"] = 1_001
    elif tamper == "petsc_view_ascii":
        diagnostic["solver_hierarchy"]["effective"].pop("petsc_view_ascii")
    elif tamper == "hierarchy_ranks":
        diagnostic["solver_hierarchy"]["effective"]["top_level"][
            "mpi_ranks"
        ] = [999]
    elif tamper == "hierarchy_prefix":
        diagnostic["solver_hierarchy"]["effective"]["mg_coarse_solver"][
            "options_prefix"
        ] = "fake_"
    elif tamper == "rhs_solves":
        payload.pop("rhs_solves")
    elif tamper == "operator_identity":
        diagnostic["preconditioner_operator_is_physical"] = not diagnostic[
            "preconditioner_operator_is_physical"
        ]
    elif tamper == "setup_seconds":
        diagnostic.pop("setup_seconds")
    with pytest.raises(ValueError):
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
        "maximum_true_relative_residual": 1.0e-7,
        "reason": "iteration cap",
    }
    payload["frequency_diagnostics"][0]["port_solves"][0][
        "converged_reason"
    ] = -3
    payload["frequency_diagnostics"][0]["port_solves"][0][
        "true_relative_residual"
    ] = 1.0e-3
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
    payload["frequency_diagnostics"] = [
        {"frequency_hz": 100_000_000.0, "global_complex_dofs": observed}
    ]
    for name in (
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
    ):
        payload.pop(name)
    payload["preconditioner"].pop("per_frequency")
    assert run_registered_scaling_sweep.validate_fem_smoke_artifact(
        registration, entry, payload
    ) == ("expected_global_dofs",)

    inconsistent = deepcopy(payload)
    inconsistent["gates"]["expected_global_dofs"]["observed"] = expected
    inconsistent["frequency_diagnostics"][0]["global_complex_dofs"] = expected
    with pytest.raises(ValueError, match="DoF failure"):
        run_registered_scaling_sweep.validate_fem_smoke_artifact(
            registration, entry, inconsistent
        )

    completed = deepcopy(payload)
    completed["rhs_solves"] = 2
    completed["frequency_diagnostics"][0]["port_solves"] = []
    with pytest.raises(ValueError, match="preflight"):
        run_registered_scaling_sweep.validate_fem_smoke_artifact(
            registration, entry, completed
        )


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
    with pytest.raises(ValueError, match="gate"):
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
        if command[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
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


@pytest.mark.parametrize("docker_return_code", (None, 0, 2, 137))
def test_failed_fem_artifact_requires_exact_docker_exit_one(
    docker_return_code: int | None,
) -> None:
    status, passed, reason = run_registered_scaling_sweep.classify_run_result(
        docker_return_code=docker_return_code,
        fem_payload={"status": "FAILED", "passed": False},
        launch_prevented=False,
    )
    assert (status, passed) == ("FAILED", False)
    assert "inconsistent" in reason
    assert run_registered_scaling_sweep.classify_run_result(
        docker_return_code=1,
        fem_payload={"status": "FAILED", "passed": False},
        launch_prevented=False,
    )[:2] == ("FAILED", False)


def test_inconsistent_fem_docker_exit_is_preserved_and_stops_next_run(
    tmp_path: Path,
) -> None:
    registration = _registration()
    first, second = registration["entries"][:2]
    first_directory = tmp_path / registration["output_root"] / first["run_id"]
    second_directory = tmp_path / registration["output_root"] / second["run_id"]
    create_count = 0
    inspect_count = 0

    def runner(command, **kwargs):
        nonlocal create_count, inspect_count
        del kwargs
        if command[:2] == ["docker", "create"]:
            create_count += 1
            return subprocess.CompletedProcess(command, 0, b"container-id\n", b"")
        if command[:2] == ["docker", "start"]:
            payload = _fem_payload(registration, first)
            payload["status"] = "FAILED"
            payload["passed"] = False
            payload["gates"]["convergence_and_true_residual"] = {
                "status": "FAILED",
                "passed": False,
                "maximum_true_relative_residual": 1.0e-7,
            }
            payload["frequency_diagnostics"][0]["port_solves"][0][
                "converged_reason"
            ] = -3
            payload["frequency_diagnostics"][0]["port_solves"][0][
                "true_relative_residual"
            ] = 1.0e-3
            first_directory.joinpath(".executor-ready").touch()
            first_directory.joinpath("fem-smoke.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            first_directory.joinpath(".solver-exit-code").write_text(
                "0\n", encoding="ascii"
            )
            first_directory.joinpath(".solver-done").touch()
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:2] == ["docker", "wait"]:
            return subprocess.CompletedProcess(command, 0, b"0\n", b"")
        if command[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
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
                    "Memory": first["resource_contract"][
                        "cgroup_memory_limit_bytes"
                    ],
                    "MemorySwap": first["resource_contract"][
                        "cgroup_memory_limit_bytes"
                    ],
                },
            }
            return subprocess.CompletedProcess(
                command, 0, json.dumps(inspection).encode(), b""
            )
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="inconsistent"):
        run_registered_scaling_sweep.execute_registered_entries(
            registration,
            image_reference=registration["images"]["project_image"]["identity"],
            output_parent=tmp_path,
            runner=runner,
            **_cgroup_kwargs(first, tmp_path),
        )
    assert create_count == 1
    persisted = json.loads(first_directory.joinpath("exit-code.json").read_text())
    assert persisted["evidence_consistent"] is False
    assert first_directory.joinpath("SHA256SUMS").is_file()
    assert not second_directory.exists()


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
        if command[:3] == ["docker", "rm", "--force"]:
            raise AssertionError("ordinary create failure must not remove by name")
        if command[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
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
    assert summaries[0]["container_cleanup_attempted"] is False


def test_docker_create_timeout_still_cleans_deterministic_container_name(
    tmp_path: Path,
) -> None:
    registration = _registration()
    registration["entries"] = registration["entries"][:1]

    def runner(command, **kwargs):
        del kwargs
        if command[:2] == ["docker", "create"]:
            raise subprocess.TimeoutExpired(command, 60)
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 1, b"", b"not found")
        if command[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        raise AssertionError(command)

    result = run_registered_scaling_sweep.execute_registered_entries(
        registration,
        image_reference=registration["images"]["project_image"]["identity"],
        output_parent=tmp_path,
        runner=runner,
        cleanup_poll_seconds=0.0,
        cleanup_stable_checks=3,
        cleanup_max_checks=8,
    )[0]
    assert result["status"] == "BLOCKED"
    assert result["container_cleanup_attempted"] is True
    assert result["container_cleanup_succeeded"] is True
    assert result["container_absence_verified"] is True


def test_docker_create_timeout_removes_late_attempt_owned_container(
    tmp_path: Path,
) -> None:
    registration = _registration()
    registration["entries"] = registration["entries"][:1]
    ps_queries = 0
    removed_targets: list[str] = []
    late_removed = False

    def runner(command, **kwargs):
        nonlocal ps_queries, late_removed
        del kwargs
        if command[:2] == ["docker", "create"]:
            raise subprocess.TimeoutExpired(command, 60)
        if command[:2] == ["docker", "ps"]:
            cycle = ps_queries // 2
            ps_queries += 1
            body = b"late-container-id\n" if cycle == 3 and not late_removed else b""
            return subprocess.CompletedProcess(command, 0, body, b"")
        if command[:3] == ["docker", "rm", "--force"]:
            removed_targets.append(command[-1])
            late_removed = command[-1] == "late-container-id"
            return subprocess.CompletedProcess(command, 0, b"", b"")
        raise AssertionError(command)

    result = run_registered_scaling_sweep.execute_registered_entries(
        registration,
        image_reference=registration["images"]["project_image"]["identity"],
        output_parent=tmp_path,
        runner=runner,
        cleanup_poll_seconds=0.0,
        cleanup_stable_checks=3,
        cleanup_max_checks=8,
    )[0]
    assert result["status"] == "BLOCKED"
    assert removed_targets == ["late-container-id"]
    assert result["container_cleanup_succeeded"] is True
    assert result["container_absence_verified"] is True


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
        if command[:2] == ["docker", "ps"]:
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
        if command[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        raise AssertionError(command)

    arguments = {
        "image_reference": registration["images"]["project_image"]["identity"],
        "output_parent": tmp_path,
        "runner": runner,
        **_cgroup_kwargs(entry, tmp_path),
    }
    if expected_reason == "resource contract":
        result = run_registered_scaling_sweep.execute_registered_entries(
            registration, **arguments
        )[0]
    else:
        with pytest.raises(RuntimeError, match="inconsistent"):
            run_registered_scaling_sweep.execute_registered_entries(
                registration, **arguments
            )
        result = json.loads(run_directory.joinpath("exit-code.json").read_text())
        assert result["evidence_consistent"] is False
    assert result["status"] == "FAILED"
    assert expected_reason in result["reason"]


def test_executor_rejects_positive_host_cgroup_oom_without_oom_kill(
    tmp_path: Path,
) -> None:
    registration = _registration()
    registration["entries"] = registration["entries"][:1]
    entry = registration["entries"][0]
    run_directory = tmp_path / registration["output_root"] / entry["run_id"]
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
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:2] == ["docker", "wait"]:
            return subprocess.CompletedProcess(command, 0, b"0\n", b"")
        if command[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:2] == ["docker", "inspect"]:
            inspect_count += 1
            running = inspect_count == 1
            body = {
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
                command, 0, json.dumps(body).encode(), b""
            )
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if command[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        raise AssertionError(command)

    cgroup = _cgroup_kwargs(entry, tmp_path)
    cgroup["cgroup_reader"] = lambda observed_path: {
        "version": "v2-host",
        "path": str(observed_path),
        "peak_bytes": 1_000_000,
        "limit_bytes": entry["resource_contract"]["cgroup_memory_limit_bytes"],
        "swap_limit_bytes": 0,
        "events": {"oom": 1, "oom_kill": 0, "max": 1},
    }
    result = run_registered_scaling_sweep.execute_registered_entries(
        registration,
        image_reference=registration["images"]["project_image"]["identity"],
        output_parent=tmp_path,
        runner=runner,
        **cgroup,
    )[0]
    assert result["status"] == "FAILED"
    assert "OOM activity" in result["reason"]


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
        if command[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(command, 0, b"container-id\n", b"")
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
