from __future__ import annotations

import numpy as np
import pytest


def _tiny_problem(comm):
    pytest.importorskip("dolfinx")
    from dolfinx import mesh

    domain = mesh.create_unit_cube(comm, 2, 2, 2)
    tdim = domain.topology.dim
    fdim = tdim - 1
    local_cells = domain.topology.index_map(tdim).size_local
    cell_indices = np.arange(local_cells, dtype=np.int32)
    cell_tags = mesh.meshtags(
        domain, tdim, cell_indices, np.ones(local_cells, dtype=np.int32)
    )
    left = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], 0.0)
    )
    right = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], 1.0)
    )
    exterior = mesh.exterior_facet_indices(domain.topology)
    walls = np.setdiff1d(exterior, np.concatenate((left, right)), assume_unique=False)
    raw_indices = np.concatenate((left, right, walls)).astype(np.int32)
    raw_values = np.concatenate(
        (
            np.full(left.size, 10, dtype=np.int32),
            np.full(right.size, 11, dtype=np.int32),
            np.full(walls.size, 20, dtype=np.int32),
        )
    )
    order = np.argsort(raw_indices)
    facet_tags = mesh.meshtags(
        domain, fdim, raw_indices[order], raw_values[order]
    )
    return domain, cell_tags, facet_tags


def _solver_and_ports(comm, iterative: bool = False):
    from dolfinx import fem
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

    domain, cell_tags, facet_tags = _tiny_problem(comm)
    contract = MeshTagContract(
        VolumeTagContract({"domain": 1}),
        BoundaryTagContract(ports={"left": 10, "right": 11}, pec_tags=(20,)),
    )
    solver_config = (
        LinearSolverConfig.iterative_maxwell(maximum_iterations=500)
        if iterative
        else LinearSolverConfig.direct()
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
    solver = MaxwellSweepSolver.from_mesh(
        domain,
        cell_tags,
        facet_tags,
        contract,
        MaterialMap(Material(2.0, conductivity_s_per_m=0.02, name="lossy")),
        MaxwellProblemConfig(polynomial_degree=1),
        matched_ports=definitions,
        solver_config=solver_config,
        initial_frequency_hz=1.0e8,
    )
    ports = []
    for definition in definitions:
        raw_mode = fem.Function(solver.function_space)
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
        ports.append(MatchedTEMPortExcitation(mode))
    return solver, ports


@pytest.mark.heavy
def test_direct_solver_assembles_and_factorizes_once_per_frequency() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI

    solver, ports = _solver_and_ports(MPI.COMM_SELF)
    result = solver.solve((1.0e8, 1.2e8), ports)
    assert result.matrix_assemblies == 2
    assert result.operator_setups == 2
    assert result.numeric_factorizations == 2
    assert result.rhs_solves == 4
    assert all(
        port.converged_reason > 0 and port.true_relative_residual < 1.0e-9
        for frequency in result.diagnostics
        for port in frequency.port_solves
    )


@pytest.mark.heavy
def test_degree_three_edge_space_builds() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI

    from scatter3d.fem.config import Material, MaterialMap, MaxwellProblemConfig
    from scatter3d.fem.forms import build_maxwell_forms
    from scatter3d.fem.ports import PortDefinition
    from scatter3d.fem.tags import (
        BoundaryTagContract,
        MeshTagContract,
        VolumeTagContract,
    )

    domain, cell_tags, facet_tags = _tiny_problem(MPI.COMM_SELF)
    forms = build_maxwell_forms(
        domain,
        cell_tags,
        facet_tags,
        MeshTagContract(
            VolumeTagContract({"domain": 1}),
            BoundaryTagContract(ports={"left": 10, "right": 11}, pec_tags=(20,)),
        ),
        MaterialMap(Material(1.0)),
        MaxwellProblemConfig(polynomial_degree=3),
        matched_ports=(
            PortDefinition(
                "left",
                10,
                field_wave_impedance_ohm=377.0,
                outgoing_propagation_index=1.0,
            ),
            PortDefinition(
                "right",
                11,
                field_wave_impedance_ohm=377.0,
                outgoing_propagation_index=1.0,
            ),
        ),
    )
    assert forms.function_space.dofmap.index_map.size_global > 0


@pytest.mark.heavy
def test_nested_asm_options_remain_available_through_setup() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI
    from petsc4py import PETSc

    from scatter3d.fem.config import LinearSolverConfig

    solver, ports = _solver_and_ports(MPI.COMM_SELF, iterative=True)
    solver.solver_config = LinearSolverConfig.iterative_maxwell(
        maximum_iterations=1,
        petsc_options={
            "sub_ksp_type": "preonly",
            "sub_pc_type": "scatter3d_deliberately_invalid_pc",
        },
    )
    with pytest.raises(PETSc.Error):
        solver.solve((1.0e8,), ports, retain_solutions=False)


@pytest.mark.heavy
@pytest.mark.mpi
def test_iterative_path_is_distributed_and_never_counts_a_factorization() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI

    if MPI.COMM_WORLD.size < 2:
        pytest.skip("run under mpirun -n 2 or more")
    solver, ports = _solver_and_ports(MPI.COMM_WORLD, iterative=True)
    result = solver.solve((1.0e8,), ports, retain_solutions=False)
    assert result.solver_path == "iterative"
    assert result.numeric_factorizations == 0
    assert result.diagnostics[0].global_complex_dofs > 0
