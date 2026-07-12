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
    compare_material_models,
    petsc_true_relative_residual,
    process_peak_rss_bytes,
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
class FrequencyDiagnostics:
    frequency_hz: float
    global_complex_dofs: int
    local_owned_dofs: int
    matrix_nonzeros: int
    matrix_memory_bytes_sum: int | None
    rank_peak_rss_bytes_max: int | None
    rank_peak_rss_bytes_sum: int | None
    assembly_seconds: float
    setup_seconds: float
    solver_path: str
    port_solves: tuple[PortSolveDiagnostics, ...]


@dataclass(frozen=True, slots=True)
class SweepResult:
    solutions: dict[float, dict[str, Any]]
    diagnostics: tuple[FrequencyDiagnostics, ...]
    matrix_assemblies: int
    operator_setups: int
    numeric_factorizations: int
    rhs_solves: int
    solver_path: str


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


class MaxwellSweepSolver:
    """Own a compiled weak form and solve independent RHS vectors per frequency."""

    def __init__(
        self,
        forms: MaxwellForms,
        solver_config: LinearSolverConfig | None = None,
    ) -> None:
        self.forms = forms
        self.solver_config = solver_config or LinearSolverConfig.direct()
        self._options_counter = 0

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

    def _configure_ksp(self, matrix: Any) -> tuple[Any, str, list[str]]:
        from petsc4py import PETSc

        config = self.solver_config
        ksp = PETSc.KSP().create(self.forms.mesh.comm)
        ksp.setOperators(matrix)
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
        for raw_key, value in config.petsc_options.items():
            key = prefix + str(raw_key).lstrip("-")
            options[key] = value
            installed.append(key)
        ksp.setFromOptions()
        for key in installed:
            del options[key]
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
        all_solutions: dict[float, dict[str, Any]] = {}
        frequency_diagnostics: list[FrequencyDiagnostics] = []
        matrix_assemblies = 0
        operator_setups = 0
        numeric_factorizations = 0
        rhs_solves = 0

        for frequency in frequencies:
            self.forms.update_frequency(frequency)
            start = perf_counter()
            matrix = fem_petsc.assemble_matrix(
                self.forms.bilinear_form, bcs=self.forms.boundary_conditions
            )
            matrix.assemble()
            assembly_seconds = perf_counter() - start
            matrix_assemblies += 1
            nonzeros, matrix_memory = _matrix_metrics(matrix, comm)

            ksp, _, _ = self._configure_ksp(matrix)
            start = perf_counter()
            ksp.setUp()
            setup_seconds = perf_counter() - start
            operator_setups += 1
            if self.solver_config.is_direct:
                numeric_factorizations += 1

            solutions_at_frequency: dict[str, Any] = {}
            port_diagnostics: list[PortSolveDiagnostics] = []
            for excitation in ports:
                linear_form = self.forms.rhs_form(excitation)
                rhs = fem_petsc.assemble_vector(linear_form)
                fem_petsc.apply_lifting(
                    rhs,
                    [self.forms.bilinear_form],
                    bcs=[self.forms.boundary_conditions],
                )
                rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
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
                history = tuple(float(value) for value in ksp.getConvergenceHistory())
                if reason <= 0 and self.solver_config.error_if_not_converged:
                    rhs.destroy()
                    ksp.destroy()
                    matrix.destroy()
                    raise RuntimeError(
                        f"PETSc failed for {excitation.definition.name} at {frequency:g} Hz: "
                        f"reason={reason}, iterations={iterations}, "
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
                rhs.destroy()

            rss_max, rss_sum = _rss_metrics(comm)
            frequency_diagnostics.append(
                FrequencyDiagnostics(
                    frequency_hz=frequency,
                    global_complex_dofs=global_dofs,
                    local_owned_dofs=local_dofs,
                    matrix_nonzeros=nonzeros,
                    matrix_memory_bytes_sum=matrix_memory,
                    rank_peak_rss_bytes_max=rss_max,
                    rank_peak_rss_bytes_sum=rss_sum,
                    assembly_seconds=assembly_seconds,
                    setup_seconds=setup_seconds,
                    solver_path=self.solver_config.solver_path,
                    port_solves=tuple(port_diagnostics),
                )
            )
            if retain_solutions:
                all_solutions[frequency] = solutions_at_frequency
            ksp.destroy()
            matrix.destroy()

        return SweepResult(
            solutions=all_solutions,
            diagnostics=tuple(frequency_diagnostics),
            matrix_assemblies=matrix_assemblies,
            operator_setups=operator_setups,
            numeric_factorizations=numeric_factorizations,
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
        reference = self.solve(frequencies, ports, retain_solutions=retain_solutions)
        self.forms.set_materials(materials.dut)
        dut = self.solve(frequencies, ports, retain_solutions=retain_solutions)
        return ExperimentSweepResult(reference, dut, compare_material_models(materials))
