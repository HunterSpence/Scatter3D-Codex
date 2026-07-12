from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path

import pytest

from validation import fem_smoke


def test_v2_status_and_gate_use_exact_public_vocabulary() -> None:
    assert fem_smoke.SCHEMA == "scatter3d.validation.fem_smoke/v2"
    assert fem_smoke._gate(True) == {"status": "PASSED", "passed": True}
    assert fem_smoke._gate(False) == {"status": "FAILED", "passed": False}


def test_image_digests_are_explicitly_sourced_or_null() -> None:
    missing = fem_smoke._image_digest_metadata({})
    assert missing["project_image"] == {
        "digest": None,
        "local_image_id": None,
        "provenance": "not_provided",
        "environment_variable": None,
    }
    project_image_id = "sha256:" + "a" * 64
    base_digest = "sha256:" + "b" * 64
    supplied = fem_smoke._image_digest_metadata(
        {
            "SCATTER3D_PROJECT_IMAGE_ID": f" {project_image_id.upper()} ",
            "SCATTER3D_BASE_IMAGE_DIGEST": base_digest,
        }
    )
    assert supplied["project_image"]["digest"] is None
    assert supplied["project_image"]["local_image_id"] == project_image_id
    assert (
        supplied["project_image"]["provenance"]
        == "docker_local_image_id_environment"
    )
    assert supplied["base_image"]["digest"] == base_digest


@pytest.mark.parametrize(
    "variable,value",
    (
        ("SCATTER3D_PROJECT_IMAGE_DIGEST", "sha256:project"),
        ("SCATTER3D_PROJECT_IMAGE_ID", "sha256:" + "g" * 64),
        ("SCATTER3D_BASE_IMAGE_DIGEST", "not-a-digest"),
    ),
)
def test_image_metadata_rejects_malformed_supplied_values(
    variable: str, value: str
) -> None:
    with pytest.raises(ValueError, match="sha256"):
        fem_smoke._image_digest_metadata({variable: value})


def test_image_metadata_rejects_ambiguous_project_identity() -> None:
    identity = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="only one"):
        fem_smoke._image_digest_metadata(
            {
                "SCATTER3D_PROJECT_IMAGE_DIGEST": identity,
                "SCATTER3D_PROJECT_IMAGE_ID": identity,
            }
        )


def test_cgroup_v2_peak_and_unlimited_limit_are_recorded(tmp_path: Path) -> None:
    (tmp_path / "memory.peak").write_text("12345\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")

    metadata = fem_smoke._cgroup_memory_metadata(tmp_path)

    assert metadata["version"] == "v2"
    assert metadata["peak_bytes"] == 12345
    assert metadata["limit_bytes"] is None
    assert metadata["provenance"] == "cgroup_files"


def test_cgroup_metadata_does_not_invent_unavailable_values(tmp_path: Path) -> None:
    assert fem_smoke._cgroup_memory_metadata(tmp_path) == {
        "version": None,
        "peak_bytes": None,
        "limit_bytes": None,
        "peak_source": None,
        "limit_source": None,
        "provenance": "unavailable",
    }


def test_command_metadata_preserves_argv_order(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    metadata = fem_smoke._command_metadata(
        ["validation/fem_smoke.py", "--solver", "iterative"]
    )
    assert metadata == {
        "argv": ["validation/fem_smoke.py", "--solver", "iterative"],
        "working_directory": str(tmp_path),
        "provenance": "process",
    }


def test_git_metadata_uses_only_validated_complete_environment_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def git_unavailable(*args, **kwargs):
        raise FileNotFoundError("git unavailable in image")

    monkeypatch.setattr(fem_smoke.subprocess, "run", git_unavailable)
    commit = "a" * 40
    metadata = fem_smoke._git_metadata(
        tmp_path,
        {
            "SCATTER3D_GIT_COMMIT": commit.upper(),
            "SCATTER3D_GIT_DIRTY": "false",
        },
    )
    assert metadata["commit"] == commit
    assert metadata["dirty"] is False
    assert metadata["provenance"] == "environment"

    invalid_environments = (
        {"SCATTER3D_GIT_COMMIT": commit},
        {"SCATTER3D_GIT_DIRTY": "false"},
        {
            "SCATTER3D_GIT_COMMIT": "short",
            "SCATTER3D_GIT_DIRTY": "false",
        },
        {
            "SCATTER3D_GIT_COMMIT": commit,
            "SCATTER3D_GIT_DIRTY": "0",
        },
    )
    for environment in invalid_environments:
        with pytest.raises(ValueError):
            fem_smoke._git_metadata(tmp_path, environment)


class _Canonical:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def canonical(self) -> dict:
        return self.payload


def test_physical_problem_identity_is_deterministic_and_covers_mutations() -> None:
    kwargs = {
        "subdivisions": 9,
        "frequencies_hz": [1.0e8],
        "problem_config": _Canonical(
            {
                "polynomial_degree": 3,
                "geometry_order": 1,
                "quadrature_degree": 6,
                "time_convention": "exp(-i*omega*t)",
            }
        ),
        "material_map": _Canonical({"default": {"relative_permittivity": [2, 0]}}),
        "contract": _Canonical(
            {"volumes": {"domain": 1}, "boundaries": {"ports": {"left": 10}}}
        ),
        "port_definitions": [_Canonical({"name": "left", "facet_tag": 10})],
    }
    first = fem_smoke._physical_problem_metadata(**kwargs)
    second = fem_smoke._physical_problem_metadata(**kwargs)
    assert first == second
    assert len(first["sha256"]) == 64

    changed = dict(kwargs)
    changed["subdivisions"] = 10
    assert fem_smoke._physical_problem_metadata(**changed)["sha256"] != first["sha256"]


def _comparison_pair() -> tuple[dict, dict]:
    physical_problem = fem_smoke._canonical_identity(
        {
            "mesh": {"subdivisions_xyz": [9, 9, 9]},
            "discretization": {"polynomial_degree": 3, "geometry_order": 1},
            "frequencies_hz": [1.0e8],
            "materials": {"relative_permittivity": [2.0, 0.0]},
            "mesh_tags_and_boundaries": {"ports": {"left": 10, "right": 11}},
            "ports": [{"name": "left"}, {"name": "right"}],
            "operator_convention": {"time_convention": "exp(-i*omega*t)"},
        }
    )
    common = {
        "source": {"commit": "a" * 40, "dirty": False},
        "images": {
            "project_image": {"digest": "sha256:" + "b" * 64},
            "base_image": {"digest": "sha256:" + "c" * 64},
        },
        "runtime": {
            "dolfinx_version": "0.10.0",
            "petsc_version": [3, 24, 0],
            "petsc_scalar_type": "complex128",
            "mpi_library_version": "MPICH 4.3.1",
        },
        "physical_problem": physical_problem,
        "mpi_size": 4,
        "frequency_diagnostics": [
            {
                "frequency_hz": 1.0e8,
                "global_complex_dofs": 86_103,
                "matrix_nonzeros": 7_038_009,
                "rank_peak_rss_bytes_sum": 1_000,
            }
        ],
    }
    iterative = {"solver": "iterative", **copy.deepcopy(common)}
    direct = {
        "solver": "direct",
        "status": "PASSED",
        "passed": True,
        "gates": {
            "convergence_and_true_residual": {"status": "PASSED", "passed": True}
        },
        **copy.deepcopy(common),
    }
    return iterative, direct


def test_direct_comparison_accepts_only_complete_same_problem_evidence() -> None:
    iterative, direct = _comparison_pair()
    fem_smoke._validate_direct_comparison(iterative, direct)

    mutations = (
        lambda value: value.update(status="FAILED", passed=False),
        lambda value: value["gates"]["convergence_and_true_residual"].update(
            status="FAILED", passed=False
        ),
        lambda value: value.update(mpi_size=8),
        lambda value: value["source"].update(commit="d" * 40),
        lambda value: value["source"].update(dirty=True),
        lambda value: value["images"]["project_image"].update(
            digest="sha256:" + "d" * 64
        ),
        lambda value: value["runtime"].update(petsc_version=[3, 25, 0]),
        lambda value: value["physical_problem"].update(sha256="0" * 64),
        lambda value: value["frequency_diagnostics"][0].update(
            global_complex_dofs=86_104
        ),
        lambda value: value["frequency_diagnostics"][0].update(
            matrix_nonzeros=7_038_010
        ),
    )
    for mutation in mutations:
        adversarial = copy.deepcopy(direct)
        mutation(adversarial)
        with pytest.raises(ValueError):
            fem_smoke._validate_direct_comparison(iterative, adversarial)


def test_direct_comparison_rejects_missing_required_identity() -> None:
    iterative, direct = _comparison_pair()
    for key in ("source", "images", "runtime", "physical_problem", "frequency_diagnostics"):
        adversarial = copy.deepcopy(direct)
        del adversarial[key]
        with pytest.raises(ValueError):
            fem_smoke._validate_direct_comparison(iterative, adversarial)


def test_release_provenance_gate_is_explicit_and_not_a_general_solve_gate() -> None:
    iterative, _ = _comparison_pair()
    passed = fem_smoke._release_provenance_gate(iterative)
    assert passed == {
        "status": "PASSED",
        "passed": True,
        "scope": "release_and_comparison_only",
    }

    iterative["images"]["base_image"]["digest"] = None
    failed = fem_smoke._release_provenance_gate(iterative)
    assert failed["status"] == "FAILED"
    assert failed["passed"] is False
    assert failed["scope"] == "release_and_comparison_only"


def test_nonfinite_diagnostics_are_explicit_strict_json() -> None:
    safe = fem_smoke._json_safe(
        {
            "residual": float("nan"),
            "history": [float("inf"), float("-inf"), 1.0],
        }
    )
    assert safe == {
        "residual": "NaN",
        "history": ["+Infinity", "-Infinity", 1.0],
    }
    assert json.loads(json.dumps(safe, allow_nan=False)) == safe


def test_artifact_write_is_atomic_no_clobber_by_default(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    fem_smoke._write_payload({"status": "FAILED"}, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fem_smoke._write_payload({"status": "PASSED"}, output)
    assert output.read_bytes() == original

    fem_smoke._write_payload({"status": "PASSED"}, output, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASSED"


@pytest.mark.skipif(os.name != "posix", reason="POSIX artifact mode contract")
def test_completed_validation_artifact_is_publicly_readable(tmp_path: Path) -> None:
    output = tmp_path / "public-evidence.json"
    fem_smoke._write_payload({"status": "PASSED"}, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
