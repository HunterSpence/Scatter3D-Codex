from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from validation import register_scaling_sweep

SPEC_PATH = Path(__file__).parents[1] / "validation" / "scaling_sweep_v1.json"
RUNTIME_METADATA = {
    "dolfinx": "0.10.0",
    "petsc": "3.24.0",
    "petsc_scalar_type": "complex128",
}


def _spec() -> tuple[dict, bytes]:
    encoded = SPEC_PATH.read_bytes()
    return json.loads(encoded), encoded


def _registration() -> dict:
    spec, encoded = _spec()
    return register_scaling_sweep.build_registration(
        spec,
        encoded,
        source={"commit": "a" * 40, "dirty": False},
        project_image_kind="local_image_id",
        project_image_identity="sha256:" + "b" * 64,
        base_image_digest="sha256:" + "c" * 64,
        base_runtime_metadata=RUNTIME_METADATA,
    )


def test_static_sweep_spec_is_the_exact_immutable_experiment() -> None:
    spec, _ = _spec()
    register_scaling_sweep.validate_scaling_sweep_spec(spec)
    assert spec["status_vocabulary"] == ["PASSED", "FAILED", "NOT RUN", "BLOCKED"]
    assert spec["preconditioner_absorption_shifts"] == [0.0, 0.25, 0.5, 1.0]
    assert [(item["subdivisions"], item["mpi_ranks"], item["gmres_restart"]) for item in spec["rungs"]] == [
        (9, 4, 80),
        (16, 8, 100),
    ]
    assert spec["physical_problem"] == {
        "fine_degree": 3,
        "coarse_degree": 1,
        "frequency_hz": 100_000_000.0,
        "ports": ["left", "right"],
    }
    assert spec["cgroup_memory_limit_bytes"] == 28 * 1024**3


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update(status_vocabulary=["PASSED", "FAILED"]),
        lambda value: value.update(preconditioner_absorption_shifts=[0.5]),
        lambda value: value.update(cgroup_memory_limit_bytes=32 * 1024**3),
        lambda value: value["rungs"][0].update(mpi_ranks=8),
        lambda value: value["solver"].update(coarse_factor_solver="superlu_dist"),
    ),
)
def test_spec_validator_rejects_any_registered_contract_mutation(mutation) -> None:
    spec, _ = _spec()
    mutation(spec)
    with pytest.raises(ValueError, match="immutable v1 contract"):
        register_scaling_sweep.validate_scaling_sweep_spec(spec)


def test_registration_is_deterministic_complete_and_initially_not_run() -> None:
    first = _registration()
    second = _registration()
    assert first == second
    assert first["schema"] == register_scaling_sweep.REGISTRATION_SCHEMA
    assert first["status_vocabulary"] == ["PASSED", "FAILED", "NOT RUN", "BLOCKED"]
    assert first["specification"]["sha256"] == hashlib.sha256(
        SPEC_PATH.read_bytes()
    ).hexdigest()
    assert first["source"] == {"commit": "a" * 40, "dirty": False}
    assert len(first["entries"]) == 8
    assert all(item["status"] == "NOT RUN" for item in first["entries"])
    assert all(item["passed"] is None for item in first["entries"])
    assert len({item["run_id"] for item in first["entries"]}) == 8
    assert len({item["outputs"]["fem_smoke_json"] for item in first["entries"]}) == 8


def test_registered_commands_cover_every_fixed_shift_and_rung_exactly() -> None:
    registration = _registration()
    observed = set()
    expected_restarts = {"p3-n9-mpi4": "80", "p3-n16-mpi8": "100"}
    for entry in registration["entries"]:
        command = entry["command"]
        assert command[:5] == [
            "mpirun",
            "-n",
            str(entry["resource_contract"]["mpi_ranks"]),
            "python3",
            "validation/fem_smoke.py",
        ]
        assert "--iterative-hierarchy" in command
        assert command[command.index("--iterative-hierarchy") + 1] == "p-multigrid"
        assert command[command.index("--degree") + 1] == "3"
        assert command[command.index("--p-multigrid-coarse-degree") + 1] == "1"
        assert command[command.index("--iterative-local-pc") + 1] == "lu"
        assert command[command.index("--maximum-iterations") + 1] == "1000"
        assert command[command.index("--maximum-true-relative-residual") + 1] == "1e-07"
        assert command[command.index("--frequencies-hz") + 1] == "100000000.0"
        assert command[command.index("--gmres-restart") + 1] == expected_restarts[
            entry["rung_id"]
        ]
        assert command[command.index("--asm-overlap") + 1] == "1"
        assert entry["resource_contract"]["cgroup_memory_limit_bytes"] == 28 * 1024**3
        assert command[-2:] == [
            "--output",
            entry["outputs"]["container_fem_smoke_json"],
        ]
        assert command[command.index("--expected-global-dofs") + 1] in {
            "86103",
            "470928",
        }
        observed.add(
            (
                entry["rung_id"],
                entry["preconditioner_absorption_shift"],
            )
        )
    assert observed == {
        (rung, shift)
        for rung in ("p3-n9-mpi4", "p3-n16-mpi8")
        for shift in (0.0, 0.25, 0.5, 1.0)
    }


def test_registration_rejects_malformed_source_and_image_identities() -> None:
    spec, encoded = _spec()
    kwargs = {
        "source": {"commit": "a" * 40, "dirty": False},
        "project_image_kind": "oci_digest",
        "project_image_identity": "sha256:" + "b" * 64,
        "base_image_digest": "sha256:" + "c" * 64,
        "base_runtime_metadata": RUNTIME_METADATA,
    }
    mutations = (
        lambda value: value.update(source={"commit": "short", "dirty": False}),
        lambda value: value.update(source={"commit": "a" * 40, "dirty": "false"}),
        lambda value: value.update(source={"commit": "a" * 40, "dirty": True}),
        lambda value: value.update(project_image_kind="tag"),
        lambda value: value.update(project_image_identity="latest"),
        lambda value: value.update(base_image_digest="sha256:short"),
    )
    for mutation in mutations:
        adversarial = copy.deepcopy(kwargs)
        mutation(adversarial)
        with pytest.raises(ValueError):
            register_scaling_sweep.build_registration(spec, encoded, **adversarial)


def test_registration_writer_is_deterministic_and_never_clobbers(tmp_path: Path) -> None:
    payload = _registration()
    output = tmp_path / "registration.json"
    destination = register_scaling_sweep.write_registration(payload, output)
    assert destination == output.resolve()
    original = output.read_bytes()
    assert json.loads(original) == payload

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        register_scaling_sweep.write_registration(payload, output)
    assert output.read_bytes() == original


def test_registration_identity_changes_all_output_paths() -> None:
    baseline = _registration()
    spec, encoded = _spec()
    changed = register_scaling_sweep.build_registration(
        spec,
        encoded,
        source={"commit": "d" * 40, "dirty": False},
        project_image_kind="local_image_id",
        project_image_identity="sha256:" + "b" * 64,
        base_image_digest="sha256:" + "c" * 64,
        base_runtime_metadata=RUNTIME_METADATA,
    )
    assert changed["registration_id"] != baseline["registration_id"]
    assert changed["output_root"] != baseline["output_root"]
    assert {
        entry["outputs"]["directory"] for entry in changed["entries"]
    }.isdisjoint(
        entry["outputs"]["directory"] for entry in baseline["entries"]
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX registration mode contract")
def test_registration_is_published_world_readable(tmp_path: Path) -> None:
    output = tmp_path / "registration.json"
    register_scaling_sweep.write_registration(_registration(), output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
