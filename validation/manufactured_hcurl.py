#!/usr/bin/env python3
"""Manufactured-solution convergence check for 3-D Nedelec Maxwell forms.

This is an executable validation, not a unit-test surrogate.  It solves a
smooth PEC-compatible field on successively refined tetrahedral meshes and
fails when H(curl) error is not monotone or the observed order misses its gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import log
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LevelResult:
    subdivisions: int
    h: float
    global_complex_dofs: int
    l2_error: float
    hcurl_error: float
    converged_reason: int
    iterations: int
    true_relative_residual: float


def _tag_cube(domain):
    from dolfinx import mesh

    tdim = domain.topology.dim
    fdim = tdim - 1
    cell_count = domain.topology.index_map(tdim).size_local
    cell_indices = np.arange(cell_count, dtype=np.int32)
    cell_tags = mesh.meshtags(
        domain, tdim, cell_indices, np.ones(cell_count, dtype=np.int32)
    )
    exterior = mesh.exterior_facet_indices(domain.topology).astype(np.int32)
    facet_tags = mesh.meshtags(
        domain,
        fdim,
        np.sort(exterior),
        np.full(exterior.size, 20, dtype=np.int32),
    )
    return cell_tags, facet_tags


def solve_level(degree: int, subdivisions: int, frequency_hz: float) -> LevelResult:
    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem import petsc as fem_petsc
    from mpi4py import MPI
    from petsc4py import PETSc

    from scatter3d.fem.config import Material, MaterialMap, MaxwellProblemConfig
    from scatter3d.fem.diagnostics import petsc_true_relative_residual
    from scatter3d.fem.forms import build_maxwell_forms
    from scatter3d.fem.tags import (
        BoundaryTagContract,
        MeshTagContract,
        VolumeTagContract,
    )

    domain = mesh.create_unit_cube(
        MPI.COMM_WORLD, subdivisions, subdivisions, subdivisions
    )
    cell_tags, facet_tags = _tag_cube(domain)
    contract = MeshTagContract(
        VolumeTagContract({"domain": 1}),
        BoundaryTagContract(ports={}, pec_tags=(20,)),
    )
    forms = build_maxwell_forms(
        domain,
        cell_tags,
        facet_tags,
        contract,
        MaterialMap(Material(1.0, name="vacuum")),
        MaxwellProblemConfig(polynomial_degree=degree),
        initial_frequency_hz=frequency_hz,
    )

    x = ufl.SpatialCoordinate(domain)
    exact = ufl.as_vector(
        (
            ufl.sin(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[2]),
            ufl.sin(ufl.pi * x[2]) * ufl.sin(ufl.pi * x[0]),
            ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        )
    )
    source = ufl.curl(ufl.curl(exact)) - forms.k0**2 * exact
    dx = ufl.Measure("dx", domain=domain)
    linear_form = fem.form(ufl.inner(source, forms.test_function) * dx)

    matrix = fem_petsc.assemble_matrix(
        forms.bilinear_form, bcs=forms.boundary_conditions
    )
    matrix.assemble()
    rhs = fem_petsc.assemble_vector(linear_form)
    fem_petsc.apply_lifting(
        rhs, [forms.bilinear_form], bcs=[forms.boundary_conditions]
    )
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(rhs, forms.boundary_conditions)

    solution = fem.Function(forms.function_space, name=f"manufactured_p{degree}")
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    try:
        pc.setFactorSolverType("mumps")
    except PETSc.Error:
        if domain.comm.size > 1:
            raise RuntimeError("parallel manufactured validation requires MUMPS")
    ksp.setErrorIfNotConverged(True)
    ksp.solve(rhs, solution.x.petsc_vec)
    solution.x.scatter_forward()
    absolute, relative = petsc_true_relative_residual(
        matrix, solution.x.petsc_vec, rhs
    )
    del absolute

    error = solution - exact
    local_l2 = fem.assemble_scalar(fem.form(ufl.inner(error, error) * dx)).real
    local_curl = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.curl(error), ufl.curl(error)) * dx)
    ).real
    global_l2 = domain.comm.allreduce(local_l2, op=MPI.SUM)
    global_curl = domain.comm.allreduce(local_curl, op=MPI.SUM)
    index_map = forms.function_space.dofmap.index_map
    global_dofs = index_map.size_global * forms.function_space.dofmap.index_map_bs
    result = LevelResult(
        subdivisions=subdivisions,
        h=1.0 / subdivisions,
        global_complex_dofs=int(global_dofs),
        l2_error=float(global_l2**0.5),
        hcurl_error=float((global_l2 + global_curl) ** 0.5),
        converged_reason=int(ksp.getConvergedReason()),
        iterations=int(ksp.getIterationNumber()),
        true_relative_residual=relative,
    )
    ksp.destroy()
    rhs.destroy()
    matrix.destroy()
    return result


def observed_orders(levels: Sequence[LevelResult]) -> list[float]:
    return [
        log(coarse.hcurl_error / fine.hcurl_error) / log(coarse.h / fine.h)
        for coarse, fine in zip(levels, levels[1:])
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degrees", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--subdivisions", type=int, nargs="+", default=(3, 4, 6))
    parser.add_argument("--frequency-hz", type=float, default=1.0e8)
    parser.add_argument("--minimum-order-slack", type=float, default=0.75)
    parser.add_argument("--maximum-true-relative-residual", type=float, default=1.0e-9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(degree not in (1, 2, 3) for degree in args.degrees):
        parser.error("degrees must be selected from 1, 2, and 3")
    if len(args.subdivisions) < 3 or any(value < 1 for value in args.subdivisions):
        parser.error("provide at least three positive subdivisions")
    if any(b <= a for a, b in zip(args.subdivisions, args.subdivisions[1:])):
        parser.error("subdivisions must be strictly increasing")

    import dolfinx
    from mpi4py import MPI
    from petsc4py import PETSc

    runs: dict[str, dict[str, object]] = {}
    passed = True
    for degree in args.degrees:
        levels = [
            solve_level(degree, subdivisions, args.frequency_hz)
            for subdivisions in args.subdivisions
        ]
        orders = observed_orders(levels)
        monotone = all(
            fine.hcurl_error < coarse.hcurl_error
            for coarse, fine in zip(levels, levels[1:])
        )
        residual_ok = all(
            level.converged_reason > 0
            and level.true_relative_residual <= args.maximum_true_relative_residual
            for level in levels
        )
        order_ok = min(orders) >= degree - args.minimum_order_slack
        passed = passed and monotone and residual_ok and order_ok
        runs[str(degree)] = {
            "levels": [asdict(level) for level in levels],
            "observed_hcurl_orders": orders,
            "gates": {
                "monotone_error": monotone,
                "positive_petsc_reason_and_true_residual": residual_ok,
                "minimum_order": degree - args.minimum_order_slack,
                "minimum_order_passed": order_ok,
            },
        }

    payload = {
        "schema": "scatter3d.validation.manufactured_hcurl/v1",
        "passed": passed,
        "frequency_hz": args.frequency_hz,
        "mpi_size": MPI.COMM_WORLD.size,
        "dolfinx_version": dolfinx.__version__,
        "petsc_version": PETSc.Sys.getVersion(),
        "scalar_type": str(PETSc.ScalarType),
        "runs": runs,
    }
    if MPI.COMM_WORLD.rank == 0:
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        print(rendered, end="")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
