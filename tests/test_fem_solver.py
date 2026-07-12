from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.heavy
def test_wrong_expected_dof_fails_before_any_rhs_solve(tmp_path: Path) -> None:
    pytest.importorskip("dolfinx")
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "wrong-dof-preflight.json"
    command = [
        sys.executable,
        "validation/fem_smoke.py",
        "--solver",
        "iterative",
        "--iterative-hierarchy",
        "p-multigrid",
        "--p-multigrid-coarse-degree",
        "1",
        "--iterative-local-pc",
        "lu",
        "--degree",
        "3",
        "--subdivisions",
        "2",
        "--frequencies-hz",
        "1.0e8",
        "--expected-global-dofs",
        "1",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 1, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["passed"] is False
    assert payload["execution_phase"] == "preflight"
    assert payload["gates"]["expected_global_dofs"]["status"] == "FAILED"
    assert payload["gates"]["convergence_and_true_residual"]["status"] == "NOT RUN"
    assert "rhs_solves" not in payload
    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o644


def test_petsc_asm_view_parser_requires_effective_type_and_overlap() -> None:
    from scatter3d.fem.diagnostics import parse_petsc_asm_view

    view = """
    PC Object: 1 MPI process
      type: asm
      total subdomain blocks = 1, amount of overlap = 2
      restriction/interpolation type - RESTRICT
    """
    assert parse_petsc_asm_view(view) == ("restrict", 2)
    with pytest.raises(ValueError, match="amount of overlap"):
        parse_petsc_asm_view("restriction/interpolation type - BASIC")
    with pytest.raises(ValueError, match="restriction/interpolation type"):
        parse_petsc_asm_view("amount of overlap = 1")


def test_petsc_mg_view_parser_requires_non_galerkin_hierarchy_fields() -> None:
    from scatter3d.fem.diagnostics import parse_petsc_mg_view

    view = """
      type is MULTIPLICATIVE, levels=2 cycles=v
        Not using Galerkin computed coarse grid matrices
    """
    assert parse_petsc_mg_view(view) == ("multiplicative", 2, "v", "none")
    with pytest.raises(ValueError, match="type/levels/cycles"):
        parse_petsc_mg_view("Not using Galerkin computed coarse grid matrices")
    with pytest.raises(ValueError, match="Galerkin mode"):
        parse_petsc_mg_view("type is MULTIPLICATIVE, levels=2 cycles=v")


def test_collective_view_path_broadcasts_rank_zero_creation_failure(
    monkeypatch,
) -> None:
    import scatter3d.fem.diagnostics as diagnostics

    class RootComm:
        rank = 0

        def __init__(self) -> None:
            self.envelopes = []

        def bcast(self, value, root=0):
            assert root == 0
            self.envelopes.append(value)
            return value

    def fail_mkstemp(*args, **kwargs):
        del args, kwargs
        raise OSError("injected mkstemp failure")

    comm = RootComm()
    monkeypatch.setattr(diagnostics.tempfile, "mkstemp", fail_mkstemp)
    with pytest.raises(RuntimeError, match="injected mkstemp failure"):
        diagnostics._collective_temporary_view_path(comm)
    assert comm.envelopes == [(None, "OSError: injected mkstemp failure")]


def test_collective_view_path_peer_receives_rank_zero_failure() -> None:
    from scatter3d.fem.diagnostics import _collective_temporary_view_path

    class PeerComm:
        rank = 1

        def bcast(self, value, root=0):
            assert value is None
            assert root == 0
            return None, "OSError: injected rank-zero failure"

    with pytest.raises(RuntimeError, match="injected rank-zero failure"):
        _collective_temporary_view_path(PeerComm())


def test_collective_view_cleanup_is_nonthrowing(monkeypatch) -> None:
    import scatter3d.fem.diagnostics as diagnostics

    def fail_unlink(self):
        del self
        raise PermissionError("injected unlink failure")

    monkeypatch.setattr(diagnostics.Path, "unlink", fail_unlink)
    diagnostics._remove_temporary_view("unused")


def test_effective_component_aggregation_deduplicates_all_mpi_ranks() -> None:
    from scatter3d.fem.diagnostics import (
        SolverComponentDiagnostics,
        _aggregate_solver_components,
    )

    common = ("preonly", "ilu", None, "scatter3d_0_sub_", 1, "none")
    local_lu = (
        "preonly",
        "lu",
        "mumps",
        "scatter3d_0_sub_",
        1,
        "none",
    )

    class FakeComm:
        def allgather(self, value):
            del value
            return ((common, common), (common, local_lu))

    local = (
        SolverComponentDiagnostics(
            path="local",
            ksp_type=common[0],
            pc_type=common[1],
            factor_solver_type=common[2],
            options_prefix=common[3],
            maximum_iterations=common[4],
            norm_type=common[5],
            mpi_ranks=(),
            instances=1,
        ),
    )
    aggregated = _aggregate_solver_components(FakeComm(), local, "asm.subdomains")
    by_pc = {item.pc_type: item for item in aggregated}
    assert by_pc["ilu"].mpi_ranks == (0, 1)
    assert by_pc["ilu"].instances == 3
    assert by_pc["lu"].mpi_ranks == (1,)
    assert by_pc["lu"].instances == 1


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


def _solver_and_ports(
    comm,
    iterative: bool = False,
    absorption_shift: float = 0.0,
    degree: int = 1,
    p_multigrid: bool = False,
):
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
    if p_multigrid:
        solver_config = LinearSolverConfig.iterative_p_multigrid(
            maximum_iterations=500,
            preconditioner_absorption_shift=absorption_shift,
        )
    elif iterative:
        solver_config = LinearSolverConfig.iterative_maxwell(
            maximum_iterations=500,
            preconditioner_absorption_shift=absorption_shift,
        )
    else:
        solver_config = LinearSolverConfig.direct()
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
        MaxwellProblemConfig(polynomial_degree=degree),
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
    assert result.preconditioner_matrix_assemblies == 0
    assert result.operator_setups == 2
    assert result.global_numeric_factorizations == 2
    assert result.numeric_factorizations == 2
    assert result.rhs_solves == 4
    assert all(
        item.preconditioner_operator_is_physical
        and item.preconditioner_absorption_shift == 0.0
        and item.solver_hierarchy.requested.pc_type == "lu"
        and item.solver_hierarchy.effective.top_level.pc_type == "lu"
        and item.solver_hierarchy.effective.top_level.factor_solver_type == "mumps"
        and item.solver_hierarchy.effective.top_level.mpi_ranks == (0,)
        and item.solver_hierarchy.effective.top_level.instances == 1
        for item in result.diagnostics
    )
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
def test_external_prefixed_option_cannot_change_iterative_top_pc_to_lu() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI
    from petsc4py import PETSc

    solver, ports = _solver_and_ports(MPI.COMM_SELF, iterative=True)
    key = "scatter3d_0_pc_type"
    options = PETSc.Options()
    options[key] = "lu"
    try:
        with pytest.raises(RuntimeError, match="global factorizing PC"):
            solver.solve((1.0e8,), ports, retain_solutions=False)
    finally:
        del options[key]


@pytest.mark.heavy
def test_effective_asm_local_lu_uses_mumps() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI

    from scatter3d.fem.config import LinearSolverConfig

    solver, ports = _solver_and_ports(MPI.COMM_SELF, iterative=True)
    solver.solver_config = LinearSolverConfig.iterative_maxwell(
        maximum_iterations=500,
        petsc_options={
            "sub_ksp_type": "preonly",
            "sub_pc_type": "lu",
            "sub_pc_factor_mat_solver_type": "mumps",
        },
    )
    result = solver.solve((1.0e8,), ports, retain_solutions=False)
    subdomains = result.diagnostics[0].solver_hierarchy.effective.asm_subdomain_solvers
    assert subdomains
    assert all(
        item.ksp_type == "preonly"
        and item.pc_type == "lu"
        and item.factor_solver_type == "mumps"
        for item in subdomains
    )


@pytest.mark.heavy
def test_shifted_preconditioner_preserves_physical_operator() -> None:
    pytest.importorskip("dolfinx")
    from dolfinx.fem import petsc as fem_petsc
    from mpi4py import MPI

    solver, _ = _solver_and_ports(MPI.COMM_SELF, iterative=True)
    forms = solver.forms
    matrices = []
    try:
        forms.set_preconditioner_absorption_shift(0.0)
        physical_zero = fem_petsc.assemble_matrix(
            forms.bilinear_form, bcs=forms.boundary_conditions
        )
        physical_zero.assemble()
        matrices.append(physical_zero)
        preconditioner_zero = fem_petsc.assemble_matrix(
            forms.preconditioner_bilinear_form,
            bcs=forms.boundary_conditions,
        )
        preconditioner_zero.assemble()
        matrices.append(preconditioner_zero)

        forms.set_preconditioner_absorption_shift(0.5)
        physical_shifted = fem_petsc.assemble_matrix(
            forms.bilinear_form, bcs=forms.boundary_conditions
        )
        physical_shifted.assemble()
        matrices.append(physical_shifted)
        preconditioner_shifted = fem_petsc.assemble_matrix(
            forms.preconditioner_bilinear_form,
            bcs=forms.boundary_conditions,
        )
        preconditioner_shifted.assemble()
        matrices.append(preconditioner_shifted)

        assert physical_zero.equal(preconditioner_zero)
        assert physical_zero.equal(physical_shifted)
        assert not preconditioner_zero.equal(preconditioner_shifted)
    finally:
        for matrix in reversed(matrices):
            matrix.destroy()


@pytest.mark.heavy
def test_shifted_iterative_path_reports_effective_asm_hierarchy() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI

    solver, ports = _solver_and_ports(
        MPI.COMM_SELF, iterative=True, absorption_shift=0.5
    )
    result = solver.solve((1.0e8,), ports, retain_solutions=False)
    assert result.preconditioner_matrix_assemblies == 1
    diagnostics = result.diagnostics[0]
    assert diagnostics.preconditioner_absorption_shift == 0.5
    assert not diagnostics.preconditioner_operator_is_physical
    assert diagnostics.preconditioner_matrix_nonzeros > 0
    assert diagnostics.preconditioner_assembly_seconds >= 0.0
    hierarchy = diagnostics.solver_hierarchy
    assert hierarchy.requested.pc_type == "asm"
    assert hierarchy.requested.preconditioning_side == "right"
    assert hierarchy.effective.top_level.ksp_type == "fgmres"
    assert hierarchy.effective.top_level.pc_type == "asm"
    assert hierarchy.effective.preconditioning_side == "right"
    assert hierarchy.effective.asm_type == "restrict"
    assert hierarchy.effective.asm_overlap == 1
    assert "amount of overlap = 1" in hierarchy.effective.petsc_view_ascii
    assert (
        "restriction/interpolation type - RESTRICT"
        in hierarchy.effective.petsc_view_ascii
    )
    assert hierarchy.effective.asm_subdomain_solvers
    assert all(
        item.ksp_type == "preonly"
        and item.pc_type == "ilu"
        and item.mpi_ranks == (0,)
        and item.instances >= 1
        for item in hierarchy.effective.asm_subdomain_solvers
    )


@pytest.mark.heavy
def test_two_level_p_multigrid_reuses_transfer_and_reports_live_hierarchy() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI

    solver, ports = _solver_and_ports(
        MPI.COMM_SELF,
        absorption_shift=0.5,
        degree=3,
        p_multigrid=True,
    )
    result = solver.solve((1.0e8, 1.2e8), ports, retain_solutions=False)
    assert result.matrix_assemblies == 2
    assert result.preconditioner_matrix_assemblies == 2
    assert result.coarse_preconditioner_matrix_assemblies == 2
    assert result.transfer_operator_assemblies == 1
    assert result.operator_setups == 2
    assert result.global_numeric_factorizations == 0
    assert result.coarse_global_factorizations == 2
    assert result.rhs_solves == 4
    for item in result.diagnostics:
        assert item.fine_degree == 3
        assert item.coarse_degree == 1
        assert item.coarse_global_complex_dofs < item.global_complex_dofs
        assert item.p_multigrid_operator_checks_passed is True
        transfer = item.transfer_operator
        assert transfer is not None
        assert transfer.direction == "coarse_to_fine"
        assert transfer.rows == item.global_complex_dofs
        assert transfer.columns == item.coarse_global_complex_dofs
        assert transfer.nonzeros > 0
        assert transfer.constrained_fine_rows > 0
        assert transfer.constrained_coarse_columns > 0
        assert transfer.maximum_imaginary_abs == 0.0
        effective = item.solver_hierarchy.effective
        assert effective.mg_levels == 2
        assert effective.mg_type == "multiplicative"
        assert effective.mg_cycle_type == "v"
        assert effective.mg_galerkin == "none"
        assert effective.mg_fine_smoother is not None
        assert effective.mg_fine_smoother.ksp_type == "richardson"
        assert effective.mg_fine_smoother.pc_type == "asm"
        assert effective.mg_fine_smoother.maximum_iterations == 1
        assert effective.mg_fine_smoother.norm_type == "none"
        assert effective.mg_fine_asm_type == "restrict"
        assert effective.mg_fine_asm_overlap == 1
        assert effective.mg_coarse_solver is not None
        assert effective.mg_coarse_solver.pc_type == "lu"
        assert effective.mg_coarse_solver.factor_solver_type == "mumps"
        assert all(
            component.pc_type == "lu"
            and component.factor_solver_type == "mumps"
            for component in effective.mg_fine_asm_subdomain_solvers
        )


@pytest.mark.heavy
def test_p_multigrid_shift_changes_both_p_levels_but_not_physical_a() -> None:
    pytest.importorskip("dolfinx")
    from dolfinx.fem import petsc as fem_petsc
    from mpi4py import MPI

    solver, _ = _solver_and_ports(
        MPI.COMM_SELF, degree=3, p_multigrid=True
    )
    coarse = solver.coarse_forms
    assert coarse is not None
    matrices = []

    def assemble(form, bcs):
        matrix = fem_petsc.assemble_matrix(form, bcs=bcs)
        matrix.assemble()
        matrices.append(matrix)
        return matrix

    try:
        solver.forms.set_preconditioner_absorption_shift(0.0)
        coarse.set_preconditioner_absorption_shift(0.0)
        physical_zero = assemble(
            solver.forms.bilinear_form, solver.forms.boundary_conditions
        )
        fine_zero = assemble(
            solver.forms.preconditioner_bilinear_form,
            solver.forms.boundary_conditions,
        )
        coarse_zero = assemble(
            coarse.preconditioner_bilinear_form, coarse.boundary_conditions
        )
        solver.forms.set_preconditioner_absorption_shift(0.5)
        coarse.set_preconditioner_absorption_shift(0.5)
        physical_shifted = assemble(
            solver.forms.bilinear_form, solver.forms.boundary_conditions
        )
        fine_shifted = assemble(
            solver.forms.preconditioner_bilinear_form,
            solver.forms.boundary_conditions,
        )
        coarse_shifted = assemble(
            coarse.preconditioner_bilinear_form, coarse.boundary_conditions
        )
        assert physical_zero.equal(physical_shifted)
        assert not fine_zero.equal(fine_shifted)
        assert not coarse_zero.equal(coarse_shifted)
    finally:
        for matrix in reversed(matrices):
            matrix.destroy()


@pytest.mark.heavy
def test_p_multigrid_synchronizes_material_changes_to_coarse_forms() -> None:
    pytest.importorskip("dolfinx")
    from dolfinx.fem import petsc as fem_petsc
    from mpi4py import MPI

    from scatter3d.fem.config import Material, MaterialMap

    solver, ports = _solver_and_ports(
        MPI.COMM_SELF,
        absorption_shift=0.5,
        degree=3,
        p_multigrid=True,
    )
    coarse = solver.coarse_forms
    assert coarse is not None
    matrices = []

    def assemble(form, bcs):
        matrix = fem_petsc.assemble_matrix(form, bcs=bcs)
        matrix.assemble()
        matrices.append(matrix)
        return matrix

    try:
        fine_before = assemble(
            solver.forms.bilinear_form, solver.forms.boundary_conditions
        )
        coarse_before = assemble(
            coarse.preconditioner_bilinear_form, coarse.boundary_conditions
        )
        replacement = MaterialMap(
            Material(3.0, conductivity_s_per_m=0.05, name="replacement")
        )
        solver.forms.set_materials(replacement)
        result = solver.solve((1.0e8,), ports, retain_solutions=False)
        assert result.diagnostics[0].p_multigrid_operator_checks_passed is True
        assert coarse.materials is replacement
        fine_after = assemble(
            solver.forms.bilinear_form, solver.forms.boundary_conditions
        )
        coarse_after = assemble(
            coarse.preconditioner_bilinear_form, coarse.boundary_conditions
        )
        assert not fine_before.equal(fine_after)
        assert not coarse_before.equal(coarse_after)
    finally:
        for matrix in reversed(matrices):
            matrix.destroy()


@pytest.mark.heavy
def test_p_multigrid_typed_structure_wins_over_prefixed_external_options() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI
    from petsc4py import PETSc

    solver, ports = _solver_and_ports(
        MPI.COMM_SELF,
        absorption_shift=0.5,
        degree=3,
        p_multigrid=True,
    )
    options = PETSc.Options()
    keys = ("scatter3d_0_pc_type", "scatter3d_0_pc_mg_levels")
    options[keys[0]] = "asm"
    options[keys[1]] = 3
    try:
        result = solver.solve((1.0e8,), ports, retain_solutions=False)
    finally:
        for key in keys:
            if options.hasName(key):
                del options[key]
    effective = result.diagnostics[0].solver_hierarchy.effective
    assert effective.top_level.pc_type == "mg"
    assert effective.mg_levels == 2


@pytest.mark.heavy
def test_p_multigrid_invalid_coarse_pc_option_is_consumed() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI
    from petsc4py import PETSc

    from scatter3d.fem.config import LinearSolverConfig

    solver, ports = _solver_and_ports(
        MPI.COMM_SELF, degree=3, p_multigrid=True
    )
    solver.solver_config = LinearSolverConfig.iterative_p_multigrid(
        maximum_iterations=1,
        petsc_options={
            "mg_coarse_pc_type": "scatter3d_deliberately_invalid_pc",
        },
    )
    with pytest.raises(PETSc.Error):
        solver.solve((1.0e8,), ports, retain_solutions=False)


@pytest.mark.heavy
@pytest.mark.mpi
def test_p_multigrid_effective_hierarchy_is_aggregated_across_ranks() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI

    if MPI.COMM_WORLD.size < 2:
        pytest.skip("run under mpirun -n 2 or more")
    solver, ports = _solver_and_ports(
        MPI.COMM_WORLD,
        absorption_shift=0.5,
        degree=3,
        p_multigrid=True,
    )
    result = solver.solve((1.0e8,), ports, retain_solutions=False)
    effective = result.diagnostics[0].solver_hierarchy.effective
    assert effective.mg_fine_smoother is not None
    assert effective.mg_fine_smoother.mpi_ranks == tuple(
        range(MPI.COMM_WORLD.size)
    )
    assert effective.mg_coarse_solver is not None
    assert effective.mg_coarse_solver.mpi_ranks == tuple(
        range(MPI.COMM_WORLD.size)
    )
    covered_ranks = {
        rank
        for component in effective.mg_fine_asm_subdomain_solvers
        for rank in component.mpi_ranks
    }
    assert covered_ranks == set(range(MPI.COMM_WORLD.size))


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
    assert result.global_numeric_factorizations == 0
    assert result.numeric_factorizations == 0
    assert result.diagnostics[0].global_complex_dofs > 0
    hierarchy = result.diagnostics[0].solver_hierarchy.effective
    assert hierarchy.top_level.mpi_ranks == tuple(range(MPI.COMM_WORLD.size))
    covered_ranks = {
        rank
        for component in hierarchy.asm_subdomain_solvers
        for rank in component.mpi_ranks
    }
    assert covered_ranks == set(range(MPI.COMM_WORLD.size))
