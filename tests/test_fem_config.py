from __future__ import annotations

from math import pi

import pytest

from scatter3d.fem.config import (
    EPSILON_0,
    ExperimentMaterials,
    LinearSolverConfig,
    Material,
    MaterialMap,
    MaxwellProblemConfig,
)


def test_conductivity_is_frequency_live_and_uses_declared_time_convention() -> None:
    material = Material(2.5, conductivity_s_per_m=0.04)
    frequency = 2.0e9
    effective = material.effective_relative_permittivity(frequency)
    assert effective.real == pytest.approx(2.5)
    assert effective.imag == pytest.approx(0.04 / (2 * pi * frequency * EPSILON_0))


def test_field_degree_three_is_independent_of_geometry_order() -> None:
    config = MaxwellProblemConfig(polynomial_degree=3, geometry_order=1)
    assert config.polynomial_degree == 3
    assert config.geometry_order == 1
    assert config.quadrature_degree >= 6
    with pytest.raises(ValueError, match="only geometry_order=1"):
        MaxwellProblemConfig(polynomial_degree=3, geometry_order=2)


def test_iterative_preset_cannot_fall_back_to_global_lu() -> None:
    config = LinearSolverConfig.iterative_maxwell()
    assert config.is_iterative
    assert config.pc_type == "asm"
    assert config.factor_solver_type is None
    assert config.preconditioning_side == "right"
    with pytest.raises(ValueError, match="forbids LU"):
        LinearSolverConfig(solver_path="iterative", ksp_type="gmres", pc_type="lu")
    with pytest.raises(ValueError, match="preconditioning_side"):
        LinearSolverConfig(preconditioning_side="diagonal")


def test_absorption_shift_is_finite_nonnegative_and_iterative_only() -> None:
    shifted = LinearSolverConfig.iterative_maxwell(
        preconditioner_absorption_shift=0.5
    )
    assert shifted.preconditioner_absorption_shift == 0.5
    assert shifted.canonical()["preconditioner_absorption_shift"] == 0.5
    for invalid in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            LinearSolverConfig.iterative_maxwell(
                preconditioner_absorption_shift=invalid
            )
    with pytest.raises(ValueError, match="only for iterative"):
        LinearSolverConfig(
            solver_path="direct",
            preconditioner_absorption_shift=0.25,
        )


@pytest.mark.parametrize(
    "reserved",
    (
        "pc_type",
        "-PC_TYPE",
        "-- pc_type",
        "ksp_type",
        "ksp_pc_side",
        "ksp_rtol",
        "ksp_atol",
        "ksp_max_it",
        "pc_factor_mat_solver_type",
    ),
)
def test_petsc_options_cannot_override_typed_top_level_fields(
    reserved: str,
) -> None:
    with pytest.raises(ValueError, match="typed top-level"):
        LinearSolverConfig.iterative_maxwell(petsc_options={reserved: "lu"})


def test_petsc_option_keys_are_normalized_without_blocking_nested_tuning() -> None:
    config = LinearSolverConfig.iterative_maxwell(
        petsc_options={
            "-SUB_PC_TYPE": "lu",
            "sub_pc_factor_mat_solver_type": "mumps",
            "KSP_GMRES_RESTART": 60,
            "pc_asm_overlap": 2,
        }
    )
    assert dict(config.petsc_options) == {
        "sub_pc_type": "lu",
        "sub_pc_factor_mat_solver_type": "mumps",
        "ksp_gmres_restart": 60,
        "pc_asm_overlap": 2,
    }
    with pytest.raises(ValueError, match="duplicate normalized"):
        LinearSolverConfig.iterative_maxwell(
            petsc_options={"sub_pc_type": "ilu", "-SUB_PC_TYPE": "lu"}
        )


def test_reference_and_dut_are_distinct_model_states() -> None:
    shared = MaterialMap(Material(1.0))
    with pytest.raises(ValueError, match="distinct MaterialMap"):
        ExperimentMaterials(shared, shared)
    pair = ExperimentMaterials(
        MaterialMap(Material(1.0)),
        MaterialMap(Material(1.0), {7: Material(2.2, name="target")}),
    )
    assert pair.reference is not pair.dut
