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
    with pytest.raises(ValueError, match="forbids LU"):
        LinearSolverConfig(solver_path="iterative", ksp_type="gmres", pc_type="lu")


def test_reference_and_dut_are_distinct_model_states() -> None:
    shared = MaterialMap(Material(1.0))
    with pytest.raises(ValueError, match="distinct MaterialMap"):
        ExperimentMaterials(shared, shared)
    pair = ExperimentMaterials(
        MaterialMap(Material(1.0)),
        MaterialMap(Material(1.0), {7: Material(2.2, name="target")}),
    )
    assert pair.reference is not pair.dut
