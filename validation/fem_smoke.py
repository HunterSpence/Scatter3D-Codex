#!/usr/bin/env python3
"""End-to-end small smoke and opt-in distributed FEM scaling probe.

No performance claim is embedded in this program.  A 3,000,000-DoF or 0.5x
memory statement is emitted only when the corresponding command-line gates are
actually met by the recorded run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _tagged_cube(comm, subdivisions: int):
    from dolfinx import mesh

    domain = mesh.create_unit_cube(comm, subdivisions, subdivisions, subdivisions)
    tdim = domain.topology.dim
    fdim = tdim - 1
    local_cells = domain.topology.index_map(tdim).size_local
    cells = np.arange(local_cells, dtype=np.int32)
    cell_tags = mesh.meshtags(
        domain, tdim, cells, np.ones(local_cells, dtype=np.int32)
    )
    left = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0))
    right = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1))
    exterior = mesh.exterior_facet_indices(domain.topology)
    walls = np.setdiff1d(exterior, np.concatenate((left, right)), assume_unique=False)
    indices = np.concatenate((left, right, walls)).astype(np.int32)
    values = np.concatenate(
        (
            np.full(left.size, 10, dtype=np.int32),
            np.full(right.size, 11, dtype=np.int32),
            np.full(walls.size, 20, dtype=np.int32),
        )
    )
    order = np.argsort(indices)
    facet_tags = mesh.meshtags(domain, fdim, indices[order], values[order])
    return domain, cell_tags, facet_tags


def _memory_high_water(payload: dict) -> int | None:
    values = [
        item["rank_peak_rss_bytes_sum"]
        for item in payload["frequency_diagnostics"]
        if item["rank_peak_rss_bytes_sum"] is not None
    ]
    return max(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("direct", "iterative"), default="direct")
    parser.add_argument("--degree", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--subdivisions", type=int, default=3)
    parser.add_argument("--frequencies-hz", type=float, nargs="+", default=(1.0e8, 1.2e8))
    parser.add_argument("--minimum-global-dofs", type=int, default=0)
    parser.add_argument("--maximum-true-relative-residual", type=float, default=1.0e-7)
    parser.add_argument("--compare-direct-json", type=Path)
    parser.add_argument("--maximum-memory-ratio", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.subdivisions < 1:
        parser.error("subdivisions must be positive")
    if any(value <= 0 for value in args.frequencies_hz) or any(
        b <= a for a, b in zip(args.frequencies_hz, args.frequencies_hz[1:], strict=False)
    ):
        parser.error("frequencies must be positive and strictly increasing")
    if args.compare_direct_json and args.solver != "iterative":
        parser.error("--compare-direct-json is meaningful only for the iterative run")

    import dolfinx
    from dolfinx import fem
    from mpi4py import MPI
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

    domain, cell_tags, facet_tags = _tagged_cube(MPI.COMM_WORLD, args.subdivisions)
    contract = MeshTagContract(
        VolumeTagContract({"domain": 1}),
        BoundaryTagContract(ports={"left": 10, "right": 11}, pec_tags=(20,)),
    )
    solver_config = (
        LinearSolverConfig.direct()
        if args.solver == "direct"
        else LinearSolverConfig.iterative_maxwell()
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
        MaxwellProblemConfig(polynomial_degree=args.degree),
        matched_ports=definitions,
        solver_config=solver_config,
        initial_frequency_hz=args.frequencies_hz[0],
    )
    excitations = []
    for definition in definitions:
        raw_mode = fem.Function(
            solver.function_space, name=f"mode_raw_{definition.name}"
        )
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
        excitations.append(MatchedTEMPortExcitation(mode))

    result = solver.solve(
        args.frequencies_hz, excitations, retain_solutions=False
    )
    diagnostics = [asdict(item) for item in result.diagnostics]
    convergence_ok = all(
        port["converged_reason"] > 0
        and port["true_relative_residual"] <= args.maximum_true_relative_residual
        for frequency in diagnostics
        for port in frequency["port_solves"]
    )
    count_ok = (
        result.matrix_assemblies == len(args.frequencies_hz)
        and result.operator_setups == len(args.frequencies_hz)
        and result.rhs_solves == len(args.frequencies_hz) * len(excitations)
        and result.numeric_factorizations
        == (len(args.frequencies_hz) if args.solver == "direct" else 0)
    )
    global_dofs = diagnostics[0]["global_complex_dofs"]
    dof_gate = global_dofs >= args.minimum_global_dofs
    payload = {
        "schema": "scatter3d.validation.fem_smoke/v1",
        "solver": args.solver,
        "degree": args.degree,
        "geometry_order": 1,
        "subdivisions": args.subdivisions,
        "frequencies_hz": list(args.frequencies_hz),
        "mpi_size": MPI.COMM_WORLD.size,
        "dolfinx_version": dolfinx.__version__,
        "petsc_version": PETSc.Sys.getVersion(),
        "scalar_type": str(PETSc.ScalarType),
        "matrix_assemblies": result.matrix_assemblies,
        "operator_setups": result.operator_setups,
        "numeric_factorizations": result.numeric_factorizations,
        "rhs_solves": result.rhs_solves,
        "frequency_diagnostics": diagnostics,
        "gates": {
            "convergence_and_true_residual": convergence_ok,
            "assembly_setup_rhs_counts": count_ok,
            "minimum_global_dofs_requested": args.minimum_global_dofs,
            "minimum_global_dofs_passed": dof_gate,
        },
    }

    memory_ratio_ok = True
    if args.compare_direct_json:
        baseline = json.loads(args.compare_direct_json.read_text(encoding="utf-8"))
        if baseline.get("solver") != "direct":
            raise ValueError("comparison JSON must be from a direct run")
        if (
            baseline.get("degree") != args.degree
            or baseline.get("subdivisions") != args.subdivisions
            or baseline.get("frequencies_hz") != list(args.frequencies_hz)
        ):
            raise ValueError("direct comparison must use the identical discrete problem")
        current_rss = _memory_high_water(payload)
        baseline_rss = _memory_high_water(baseline)
        if current_rss is None or baseline_rss is None or baseline_rss <= 0:
            raise ValueError("both runs must contain process peak-RSS instrumentation")
        ratio = current_rss / baseline_rss
        memory_ratio_ok = ratio <= args.maximum_memory_ratio
        payload["memory_comparison"] = {
            "metric": "sum_of_rank_process_high_water_rss_bytes",
            "iterative_bytes": current_rss,
            "direct_bytes": baseline_rss,
            "ratio": ratio,
            "maximum_ratio": args.maximum_memory_ratio,
            "passed": memory_ratio_ok,
            "note": "Scheduler/job MaxRSS is preferred for a publication claim.",
        }

    passed = convergence_ok and count_ok and dof_gate and memory_ratio_ok
    payload["passed"] = passed
    if MPI.COMM_WORLD.rank == 0:
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        print(rendered, end="")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
