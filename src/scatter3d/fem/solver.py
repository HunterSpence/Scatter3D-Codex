"""Frequency sweeps with one operator setup and repeated port right-hand sides."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter
from typing import Any

import numpy as np

from .config import (
    ExperimentMaterials,
    LinearSolverConfig,
    MaterialMap,
    MaxwellProblemConfig,
    PMLConfig,
)
from .diagnostics import (
    MaterialChange,
    SolverHierarchyDiagnostics,
    compare_material_models,
    inspect_petsc_solver_hierarchy,
    petsc_true_relative_residual,
    process_peak_rss_bytes,
    validate_effective_solver_hierarchy,
)
from .forms import MaxwellForms, build_maxwell_forms
from .ports import MatchedTEMPortExcitation, PortDefinition
from .tags import MeshTagContract, validate_mesh_tags


@dataclass(frozen=True, slots=True)
class PortSolveDiagnostics:
    port_name: str
    converged_reason: int
    iterations: int
    true_residual_norm: float
    true_relative_residual: float
    reported_residual_history: tuple[float, ...]
    solve_seconds: float


@dataclass(frozen=True, slots=True)
class TransferOperatorDiagnostics:
    direction: str
    rows: int
    columns: int
    nonzeros: int
    memory_bytes_sum: int | None
    assembly_seconds: float
    constrained_fine_rows: int
    constrained_coarse_columns: int
    maximum_imaginary_abs: float


@dataclass(frozen=True, slots=True)
class FrequencyDiagnostics:
    frequency_hz: float
    fine_degree: int
    global_complex_dofs: int
    local_owned_dofs: int
    matrix_nonzeros: int
    matrix_memory_bytes_sum: int | None
    preconditioner_absorption_shift: float
    preconditioner_operator_is_physical: bool
    preconditioner_matrix_nonzeros: int
    preconditioner_matrix_memory_bytes_sum: int | None
    preconditioner_assembly_seconds: float
    coarse_degree: int | None
    coarse_global_complex_dofs: int | None
    coarse_local_owned_dofs: int | None
    coarse_preconditioner_matrix_nonzeros: int | None
    coarse_preconditioner_matrix_memory_bytes_sum: int | None
    coarse_preconditioner_assembly_seconds: float | None
    transfer_operator: TransferOperatorDiagnostics | None
    p_multigrid_operator_checks_passed: bool | None
    rank_peak_rss_bytes_max: int | None
    rank_peak_rss_bytes_sum: int | None
    assembly_seconds: float
    setup_seconds: float
    solver_path: str
    solver_hierarchy: SolverHierarchyDiagnostics
    port_solves: tuple[PortSolveDiagnostics, ...]


@dataclass(frozen=True, slots=True)
class SweepResult:
    solutions: dict[float, dict[str, Any]]
    diagnostics: tuple[FrequencyDiagnostics, ...]
    matrix_assemblies: int
    preconditioner_matrix_assemblies: int
    coarse_preconditioner_matrix_assemblies: int
    transfer_operator_assemblies: int
    operator_setups: int
    global_numeric_factorizations: int
    coarse_global_factorizations: int
    rhs_solves: int
    solver_path: str

    @property
    def numeric_factorizations(self) -> int:
        """Backward-compatible alias for global top-level factorizations.

        Local factorizations inside ASM or future multilevel PCs are not
        included in this counter.
        """

        return self.global_numeric_factorizations


@dataclass(frozen=True, slots=True)
class ExperimentSweepResult:
    reference: SweepResult
    dut: SweepResult
    material_changes: tuple[MaterialChange, ...]


def _strict_frequencies(values: Iterable[float]) -> tuple[float, ...]:
    frequencies = tuple(float(value) for value in values)
    if not frequencies or any(not np.isfinite(value) or value <= 0 for value in frequencies):
        raise ValueError("frequencies_hz must contain finite positive values")
    if any(right <= left for left, right in pairwise(frequencies)):
        raise ValueError("frequencies_hz must be strictly increasing")
    return frequencies


def _matrix_metrics(matrix: Any, comm: Any) -> tuple[int, int | None]:
    from mpi4py import MPI
    from petsc4py import PETSc

    try:
        info = matrix.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
        nonzeros = int(info.get("nz_used", 0))
        memory = int(info.get("memory", 0)) or None
        return nonzeros, memory
    except (AttributeError, KeyError, TypeError):
        local = matrix.getInfo()
        nonzeros = int(comm.allreduce(int(local.get("nz_used", 0)), op=MPI.SUM))
        local_memory = int(local.get("memory", 0))
        memory = int(comm.allreduce(local_memory, op=MPI.SUM)) if local_memory else None
        return nonzeros, memory


def _rss_metrics(comm: Any) -> tuple[int | None, int | None]:
    from mpi4py import MPI

    local = process_peak_rss_bytes()
    available = int(comm.allreduce(int(local is not None), op=MPI.SUM))
    if available != comm.size:
        return None, None
    assert local is not None
    return (
        int(comm.allreduce(local, op=MPI.MAX)),
        int(comm.allreduce(local, op=MPI.SUM)),
    )


def _owned_bc_local_dofs(forms: MaxwellForms) -> np.ndarray:
    """Return unique owned constrained dofs in local vector numbering."""

    owned: list[np.ndarray] = []
    for bc in forms.boundary_conditions:
        local_dofs, first_ghost = bc.dof_indices()
        local = np.asarray(local_dofs[:first_ghost], dtype=np.int64)
        if local.size:
            owned.append(local)
    if not owned:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate(owned)).astype(np.int32, copy=False)


def _build_masked_p_interpolation(
    coarse_forms: MaxwellForms,
    fine_forms: MaxwellForms,
) -> tuple[Any, TransferOperatorDiagnostics]:
    """Build coarse-to-fine interpolation and remove PEC rows and columns."""

    from dolfinx.fem import petsc as fem_petsc
    from mpi4py import MPI
    from petsc4py import PETSc

    comm = fine_forms.mesh.comm
    start = perf_counter()
    interpolation = fem_petsc.interpolation_matrix(
        coarse_forms.function_space, fine_forms.function_space
    )
    try:
        interpolation.assemble()
        coarse_mask, fine_mask = interpolation.createVecs()
        try:
            coarse_mask.set(1.0)
            fine_mask.set(1.0)
            fine_rows = _owned_bc_local_dofs(fine_forms)
            coarse_columns = _owned_bc_local_dofs(coarse_forms)
            local_error = None
            if fine_rows.size and (
                int(fine_rows.min()) < 0
                or int(fine_rows.max()) >= fine_mask.array.size
            ):
                local_error = "fine PEC dof is outside the owned transfer row mask"
            elif coarse_columns.size and (
                int(coarse_columns.min()) < 0
                or int(coarse_columns.max()) >= coarse_mask.array.size
            ):
                local_error = (
                    "coarse PEC dof is outside the owned transfer column mask"
                )
            rank_errors = tuple(comm.allgather(local_error))
            if any(error is not None for error in rank_errors):
                raise RuntimeError(f"invalid distributed transfer mask: {rank_errors}")
            fine_mask.array[fine_rows] = 0.0
            coarse_mask.array[coarse_columns] = 0.0
            interpolation.diagonalScale(fine_mask, coarse_mask)
        finally:
            fine_mask.destroy()
            coarse_mask.destroy()
        interpolation.assemble()

        fine_map = fine_forms.function_space.dofmap.index_map
        fine_bs = fine_forms.function_space.dofmap.index_map_bs
        coarse_map = coarse_forms.function_space.dofmap.index_map
        coarse_bs = coarse_forms.function_space.dofmap.index_map_bs
        expected = (
            int(fine_map.size_global * fine_bs),
            int(coarse_map.size_global * coarse_bs),
        )
        actual_shape = tuple(interpolation.getSize())
        if actual_shape != expected:
            raise RuntimeError(
                f"coarse-to-fine interpolation shape {actual_shape} != {expected}"
            )
        imaginary = interpolation.duplicate(copy=True)
        try:
            imaginary.imagPart()
            maximum_imaginary = float(imaginary.norm(PETSc.NormType.INFINITY))
        finally:
            imaginary.destroy()
        if maximum_imaginary != 0.0:
            raise RuntimeError(
                "p-multigrid interpolation must be real-valued in the complex build"
            )
        nonzeros, memory = _matrix_metrics(interpolation, comm)
        fine_count = int(comm.allreduce(fine_rows.size, op=MPI.SUM))
        coarse_count = int(comm.allreduce(coarse_columns.size, op=MPI.SUM))
        diagnostics = TransferOperatorDiagnostics(
            direction="coarse_to_fine",
            rows=expected[0],
            columns=expected[1],
            nonzeros=nonzeros,
            memory_bytes_sum=memory,
            assembly_seconds=perf_counter() - start,
            constrained_fine_rows=fine_count,
            constrained_coarse_columns=coarse_count,
            maximum_imaginary_abs=maximum_imaginary,
        )
        return interpolation, diagnostics
    except Exception:
        interpolation.destroy()
        raise


class MaxwellSweepSolver:
    """Own a compiled weak form and solve independent RHS vectors per frequency."""

    def __init__(
        self,
        forms: MaxwellForms,
        solver_config: LinearSolverConfig | None = None,
    ) -> None:
        self.forms = forms
        self.solver_config = solver_config or LinearSolverConfig.direct()
        self.coarse_forms: MaxwellForms | None = None
        self._options_counter = 0
        if self.solver_config.uses_p_multigrid:
            self._ensure_coarse_forms()

    @classmethod
    def from_mesh(
        cls,
        mesh: Any,
        cell_tags: Any,
        facet_tags: Any,
        tag_contract: MeshTagContract,
        materials: MaterialMap,
        problem_config: MaxwellProblemConfig,
        *,
        pml_config: PMLConfig | None = None,
        matched_ports: Sequence[PortDefinition] | None = None,
        solver_config: LinearSolverConfig | None = None,
        initial_frequency_hz: float = 1.0e9,
    ) -> MaxwellSweepSolver:
        validate_mesh_tags(mesh, cell_tags, facet_tags, tag_contract)
        forms = build_maxwell_forms(
            mesh,
            cell_tags,
            facet_tags,
            tag_contract,
            materials,
            problem_config,
            pml_config,
            matched_ports=matched_ports,
            initial_frequency_hz=initial_frequency_hz,
        )
        return cls(forms, solver_config)

    @property
    def function_space(self) -> Any:
        return self.forms.function_space

    def _ensure_coarse_forms(self) -> MaxwellForms:
        coarse_degree = self.solver_config.p_multigrid_coarse_degree
        if coarse_degree is None:
            raise RuntimeError("p-multigrid coarse forms requested without a coarse degree")
        fine_degree = self.forms.config.polynomial_degree
        if coarse_degree >= fine_degree:
            raise ValueError(
                "p_multigrid_coarse_degree must be below the fine polynomial degree"
            )
        if (
            self.coarse_forms is not None
            and self.coarse_forms.config.polynomial_degree == coarse_degree
        ):
            return self.coarse_forms
        coarse_config = MaxwellProblemConfig(
            polynomial_degree=coarse_degree,
            geometry_order=self.forms.config.geometry_order,
            quadrature_degree=self.forms.config.quadrature_degree,
        )
        self.coarse_forms = build_maxwell_forms(
            self.forms.mesh,
            self.forms.cell_tags,
            self.forms.facet_tags,
            self.forms.tag_contract,
            self.forms.materials,
            coarse_config,
            self.forms.pml_config,
            matched_ports=tuple(self.forms.matched_ports.values()),
            initial_frequency_hz=self.forms.frequency_hz,
        )
        return self.coarse_forms

    def _configure_ksp(
        self,
        matrix: Any,
        preconditioner_matrix: Any,
        *,
        coarse_preconditioner_matrix: Any | None = None,
        interpolation: Any | None = None,
    ) -> tuple[Any, str, list[str]]:
        from petsc4py import PETSc

        config = self.solver_config
        ksp = PETSc.KSP().create(self.forms.mesh.comm)
        ksp.setOperators(matrix, preconditioner_matrix)
        ksp.setType(config.ksp_type)
        side = {
            "left": PETSc.PC.Side.LEFT,
            "right": PETSc.PC.Side.RIGHT,
            "symmetric": PETSc.PC.Side.SYMMETRIC,
        }[config.preconditioning_side]
        ksp.setPCSide(side)
        ksp.setTolerances(
            rtol=config.relative_tolerance,
            atol=config.absolute_tolerance,
            max_it=config.maximum_iterations,
        )
        pc = ksp.getPC()
        pc.setType(config.pc_type)
        if config.is_direct and config.factor_solver_type:
            pc.setFactorSolverType(config.factor_solver_type)
        elif config.is_iterative and config.pc_type.lower() in {"lu", "cholesky"}:
            raise ValueError("iterative path cannot configure a factorizing PC")

        prefix = f"scatter3d_{self._options_counter}_"
        self._options_counter += 1
        ksp.setOptionsPrefix(prefix)
        options = PETSc.Options()
        installed: list[str] = []
        try:
            fine_smoother = None
            coarse_solver = None
            if config.uses_p_multigrid:
                if coarse_preconditioner_matrix is None or interpolation is None:
                    raise ValueError(
                        "p-multigrid requires coarse matrix and interpolation"
                    )
                # PCMG otherwise defaults to the outer Amat and can silently
                # replace the shifted fine hierarchy with the physical A.
                pc.setUseAmat(False)
                pc.setMGLevels(2)
                pc.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)
                pc.setMGInterpolation(1, interpolation)
                fine_smoother = pc.getMGSmoother(1)
                fine_smoother.setOperators(
                    preconditioner_matrix, preconditioner_matrix
                )
                # PCMG applies level smoothers with a nonzero initial guess;
                # PETSc rejects KSPPREONLY in that mode. One Richardson step
                # gives an explicit single application of the ASM smoother.
                fine_smoother.setType("richardson")
                fine_smoother.setTolerances(max_it=1)
                fine_smoother.getPC().setType("asm")
                coarse_solver = pc.getMGCoarseSolve()
                coarse_solver.setOperators(
                    coarse_preconditioner_matrix,
                    coarse_preconditioner_matrix,
                )
                coarse_solver.setType("preonly")
                coarse_pc = coarse_solver.getPC()
                coarse_pc.setType("lu")
                coarse_pc.setFactorSolverType("mumps")
                structural = {
                    "pc_mg_levels": 2,
                    "pc_mg_galerkin": "none",
                    "pc_mg_type": "multiplicative",
                    "pc_type": "mg",
                    "pc_use_amat": False,
                }
                for raw_key, value in structural.items():
                    key = prefix + raw_key
                    options[key] = value
                    installed.append(key)
            for raw_key, value in config.petsc_options.items():
                key = prefix + str(raw_key).lstrip("-")
                options[key] = value
                installed.append(key)
            ksp.setFromOptions()
            effective_pc_type = str(pc.getType()).lower()
            if effective_pc_type != config.pc_type.lower():
                raise RuntimeError(
                    "effective PETSc PC changed before nested hierarchy setup: "
                    f"{effective_pc_type!r} != {config.pc_type.lower()!r}"
                )
            if fine_smoother is not None and coarse_solver is not None:
                if int(pc.getMGLevels()) != 2:
                    raise RuntimeError("effective p-multigrid level count is not two")
                # Reacquire after outer options in case PCMG rebuilt its levels.
                fine_smoother = pc.getMGSmoother(1)
                coarse_solver = pc.getMGCoarseSolve()
                # These objects exist before options are parsed; invoke their
                # supported option paths explicitly and retain all options
                # until the full outer KSP setup creates ASM subdomain PCs.
                fine_smoother.setFromOptions()
                coarse_solver.setFromOptions()
            if config.is_iterative and effective_pc_type in {"lu", "cholesky"}:
                raise RuntimeError(
                    "iterative PETSc configuration resolved to a global "
                    f"factorizing PC {effective_pc_type!r} before setup"
                )
        except Exception:
            for key in installed:
                del options[key]
            ksp.destroy()
            raise
        if hasattr(ksp, "setErrorIfNotConverged"):
            # Always preserve PETSc's divergence reason, iteration count, and
            # true residual. The public flag controls the explicit checked
            # error below instead of asking PETSc to abort before diagnostics.
            ksp.setErrorIfNotConverged(False)
        pc.setReusePreconditioner(True)
        ksp.setConvergenceHistory(config.maximum_iterations + 1, reset=True)
        return ksp, prefix, installed

    def solve(
        self,
        frequencies_hz: Iterable[float],
        ports: Sequence[MatchedTEMPortExcitation],
        *,
        retain_solutions: bool = True,
    ) -> SweepResult:
        """Solve all ports, assembling and setting up the matrix once per frequency."""

        from dolfinx import fem
        from dolfinx.fem import petsc as fem_petsc
        from petsc4py import PETSc

        frequencies = _strict_frequencies(frequencies_hz)
        if not ports:
            raise ValueError("at least one port excitation is required")
        port_names = [port.definition.name for port in ports]
        if len(set(port_names)) != len(port_names):
            raise ValueError("port excitation names must be unique")
        for port in ports:
            if port.mode.field.function_space is not self.function_space:
                raise ValueError(
                    f"port {port.definition.name!r} mode must use solver.function_space"
                )

        comm = self.forms.mesh.comm
        index_map = self.function_space.dofmap.index_map
        block_size = self.function_space.dofmap.index_map_bs
        global_dofs = int(index_map.size_global * block_size)
        local_dofs = int(index_map.size_local * block_size)
        coarse_forms = (
            self._ensure_coarse_forms()
            if self.solver_config.uses_p_multigrid
            else None
        )
        if coarse_forms is not None:
            # Public callers may update the fine material state directly
            # between solves; the low-order hierarchy must follow it exactly.
            coarse_forms.set_materials(self.forms.materials)
            coarse_map = coarse_forms.function_space.dofmap.index_map
            coarse_bs = coarse_forms.function_space.dofmap.index_map_bs
            coarse_global_dofs = int(coarse_map.size_global * coarse_bs)
            coarse_local_dofs = int(coarse_map.size_local * coarse_bs)
        else:
            coarse_global_dofs = None
            coarse_local_dofs = None
        all_solutions: dict[float, dict[str, Any]] = {}
        frequency_diagnostics: list[FrequencyDiagnostics] = []
        matrix_assemblies = 0
        preconditioner_matrix_assemblies = 0
        coarse_preconditioner_matrix_assemblies = 0
        transfer_operator_assemblies = 0
        operator_setups = 0
        numeric_factorizations = 0
        coarse_global_factorizations = 0
        rhs_solves = 0

        interpolation = None
        transfer_diagnostics = None
        if coarse_forms is not None:
            interpolation, transfer_diagnostics = _build_masked_p_interpolation(
                coarse_forms, self.forms
            )
            transfer_operator_assemblies = 1

        for frequency_index, frequency in enumerate(frequencies):
            completed_frequency = False
            p_multigrid_operator_checks_passed = None
            matrix = None
            preconditioner_matrix = None
            coarse_preconditioner_matrix = None
            ksp = None
            try:
                self.forms.update_frequency(frequency)
                shift = self.solver_config.preconditioner_absorption_shift
                self.forms.set_preconditioner_absorption_shift(shift)
                if coarse_forms is not None:
                    coarse_forms.update_frequency(frequency)
                    coarse_forms.set_preconditioner_absorption_shift(shift)

                start = perf_counter()
                matrix = fem_petsc.assemble_matrix(
                    self.forms.bilinear_form, bcs=self.forms.boundary_conditions
                )
                matrix.assemble()
                assembly_seconds = perf_counter() - start
                matrix_assemblies += 1
                nonzeros, matrix_memory = _matrix_metrics(matrix, comm)

                if shift == 0.0:
                    # Zero shift has explicit identity semantics and incurs no
                    # duplicate sparse matrix or assembly.
                    preconditioner_matrix = matrix
                    preconditioner_assembly_seconds = 0.0
                    preconditioner_nonzeros = nonzeros
                    preconditioner_memory = matrix_memory
                else:
                    start = perf_counter()
                    preconditioner_matrix = fem_petsc.assemble_matrix(
                        self.forms.preconditioner_bilinear_form,
                        bcs=self.forms.boundary_conditions,
                    )
                    preconditioner_matrix.assemble()
                    preconditioner_assembly_seconds = perf_counter() - start
                    preconditioner_matrix_assemblies += 1
                    preconditioner_nonzeros, preconditioner_memory = _matrix_metrics(
                        preconditioner_matrix, comm
                    )

                coarse_nonzeros = None
                coarse_memory = None
                coarse_assembly_seconds = None
                if coarse_forms is not None:
                    start = perf_counter()
                    coarse_preconditioner_matrix = fem_petsc.assemble_matrix(
                        coarse_forms.preconditioner_bilinear_form,
                        bcs=coarse_forms.boundary_conditions,
                    )
                    coarse_preconditioner_matrix.assemble()
                    coarse_assembly_seconds = perf_counter() - start
                    coarse_preconditioner_matrix_assemblies += 1
                    coarse_nonzeros, coarse_memory = _matrix_metrics(
                        coarse_preconditioner_matrix, comm
                    )

                ksp, prefix, installed_options = self._configure_ksp(
                    matrix,
                    preconditioner_matrix,
                    coarse_preconditioner_matrix=coarse_preconditioner_matrix,
                    interpolation=interpolation,
                )
                start = perf_counter()
                try:
                    # ASM creates nested KSP/PC objects during setup. Keep the
                    # prefixed options live until those objects consume them.
                    ksp.setUp()
                finally:
                    options = PETSc.Options()
                    for key in installed_options:
                        del options[key]
                setup_seconds = perf_counter() - start
                operator_setups += 1
                if self.solver_config.is_direct:
                    numeric_factorizations += 1
                if coarse_preconditioner_matrix is not None:
                    live_pc = ksp.getPC()
                    if live_pc.getUseAmat():
                        raise RuntimeError(
                            "PCMG resolved to Amat instead of the shifted Pmat"
                        )
                    fine_ksp = live_pc.getMGSmoother(1)
                    live_outer_amat, live_outer_pmat = ksp.getOperators()
                    if (
                        live_outer_amat.handle != matrix.handle
                        or live_outer_pmat.handle != preconditioner_matrix.handle
                    ):
                        raise RuntimeError(
                            "outer KSP did not retain the physical A and shifted P"
                        )
                    live_fine_amat, live_fine_pmat = fine_ksp.getOperators()
                    if (
                        live_fine_amat.handle != preconditioner_matrix.handle
                        or live_fine_pmat.handle != preconditioner_matrix.handle
                    ):
                        raise RuntimeError(
                            "PCMG fine smoother did not retain the shifted fine P matrix"
                        )
                    coarse_ksp = live_pc.getMGCoarseSolve()
                    live_coarse_amat, live_coarse_pmat = coarse_ksp.getOperators()
                    if (
                        live_coarse_amat.handle
                        != coarse_preconditioner_matrix.handle
                        or live_coarse_pmat.handle
                        != coarse_preconditioner_matrix.handle
                    ):
                        raise RuntimeError(
                            "PCMG coarse solver did not retain the explicit coarse P"
                        )
                    live_interpolation = live_pc.getMGInterpolation(1)
                    if live_interpolation.handle != interpolation.handle:
                        raise RuntimeError(
                            "PCMG did not retain the supplied p-transfer operator"
                        )
                    p_multigrid_operator_checks_passed = True
                    coarse_global_factorizations += 1
                hierarchy = inspect_petsc_solver_hierarchy(
                    ksp, self.solver_config, prefix, comm
                )
                validate_effective_solver_hierarchy(
                    hierarchy, self.solver_config
                )

                solutions_at_frequency: dict[str, Any] = {}
                port_diagnostics: list[PortSolveDiagnostics] = []
                for excitation in ports:
                    linear_form = self.forms.rhs_form(excitation)
                    rhs = fem_petsc.assemble_vector(linear_form)
                    try:
                        fem_petsc.apply_lifting(
                            rhs,
                            [self.forms.bilinear_form],
                            bcs=[self.forms.boundary_conditions],
                        )
                        rhs.ghostUpdate(
                            addv=PETSc.InsertMode.ADD,
                            mode=PETSc.ScatterMode.REVERSE,
                        )
                        fem_petsc.set_bc(rhs, self.forms.boundary_conditions)
                        solution = fem.Function(
                            self.function_space,
                            name=f"E_{excitation.definition.name}_{frequency:g}Hz",
                        )
                        ksp.setConvergenceHistory(reset=True)
                        start = perf_counter()
                        ksp.solve(rhs, solution.x.petsc_vec)
                        solve_seconds = perf_counter() - start
                        solution.x.scatter_forward()
                        rhs_solves += 1
                        reason = int(ksp.getConvergedReason())
                        iterations = int(ksp.getIterationNumber())
                        absolute, relative = petsc_true_relative_residual(
                            matrix, solution.x.petsc_vec, rhs
                        )
                        history = tuple(
                            float(value) for value in ksp.getConvergenceHistory()
                        )
                        if reason <= 0 and self.solver_config.error_if_not_converged:
                            raise RuntimeError(
                                f"PETSc failed for {excitation.definition.name} "
                                f"at {frequency:g} Hz: reason={reason}, "
                                f"iterations={iterations}, "
                                f"true_relative_residual={relative:.3e}"
                            )
                        port_diagnostics.append(
                            PortSolveDiagnostics(
                                port_name=excitation.definition.name,
                                converged_reason=reason,
                                iterations=iterations,
                                true_residual_norm=absolute,
                                true_relative_residual=relative,
                                reported_residual_history=history,
                                solve_seconds=solve_seconds,
                            )
                        )
                        if retain_solutions:
                            solutions_at_frequency[excitation.definition.name] = solution
                    finally:
                        rhs.destroy()

                rss_max, rss_sum = _rss_metrics(comm)
                frequency_diagnostics.append(
                    FrequencyDiagnostics(
                        frequency_hz=frequency,
                        fine_degree=self.forms.config.polynomial_degree,
                        global_complex_dofs=global_dofs,
                        local_owned_dofs=local_dofs,
                        matrix_nonzeros=nonzeros,
                        matrix_memory_bytes_sum=matrix_memory,
                        preconditioner_absorption_shift=shift,
                        preconditioner_operator_is_physical=(
                            preconditioner_matrix is matrix
                        ),
                        preconditioner_matrix_nonzeros=preconditioner_nonzeros,
                        preconditioner_matrix_memory_bytes_sum=preconditioner_memory,
                        preconditioner_assembly_seconds=preconditioner_assembly_seconds,
                        coarse_degree=(
                            None
                            if coarse_forms is None
                            else coarse_forms.config.polynomial_degree
                        ),
                        coarse_global_complex_dofs=coarse_global_dofs,
                        coarse_local_owned_dofs=coarse_local_dofs,
                        coarse_preconditioner_matrix_nonzeros=coarse_nonzeros,
                        coarse_preconditioner_matrix_memory_bytes_sum=coarse_memory,
                        coarse_preconditioner_assembly_seconds=coarse_assembly_seconds,
                        transfer_operator=transfer_diagnostics,
                        p_multigrid_operator_checks_passed=(
                            p_multigrid_operator_checks_passed
                        ),
                        rank_peak_rss_bytes_max=rss_max,
                        rank_peak_rss_bytes_sum=rss_sum,
                        assembly_seconds=assembly_seconds,
                        setup_seconds=setup_seconds,
                        solver_path=self.solver_config.solver_path,
                        solver_hierarchy=hierarchy,
                        port_solves=tuple(port_diagnostics),
                    )
                )
                if retain_solutions:
                    all_solutions[frequency] = solutions_at_frequency
                completed_frequency = True
            finally:
                if ksp is not None:
                    ksp.destroy()
                if (
                    preconditioner_matrix is not None
                    and preconditioner_matrix is not matrix
                ):
                    preconditioner_matrix.destroy()
                if coarse_preconditioner_matrix is not None:
                    coarse_preconditioner_matrix.destroy()
                if matrix is not None:
                    matrix.destroy()
                if interpolation is not None and (
                    not completed_frequency
                    or frequency_index == len(frequencies) - 1
                ):
                    interpolation.destroy()
                    interpolation = None

        return SweepResult(
            solutions=all_solutions,
            diagnostics=tuple(frequency_diagnostics),
            matrix_assemblies=matrix_assemblies,
            preconditioner_matrix_assemblies=preconditioner_matrix_assemblies,
            coarse_preconditioner_matrix_assemblies=(
                coarse_preconditioner_matrix_assemblies
            ),
            transfer_operator_assemblies=transfer_operator_assemblies,
            operator_setups=operator_setups,
            global_numeric_factorizations=numeric_factorizations,
            coarse_global_factorizations=coarse_global_factorizations,
            rhs_solves=rhs_solves,
            solver_path=self.solver_config.solver_path,
        )

    def solve_material_pair(
        self,
        frequencies_hz: Iterable[float],
        ports: Sequence[MatchedTEMPortExcitation],
        materials: ExperimentMaterials,
        *,
        retain_solutions: bool = True,
    ) -> ExperimentSweepResult:
        """Run separate reference and DUT operators and keep diagnostics separate."""

        frequencies = _strict_frequencies(frequencies_hz)
        self.forms.set_materials(materials.reference)
        if self.coarse_forms is not None:
            self.coarse_forms.set_materials(materials.reference)
        reference = self.solve(frequencies, ports, retain_solutions=retain_solutions)
        self.forms.set_materials(materials.dut)
        if self.coarse_forms is not None:
            self.coarse_forms.set_materials(materials.dut)
        dut = self.solve(frequencies, ports, retain_solutions=retain_solutions)
        return ExperimentSweepResult(reference, dut, compare_material_models(materials))
