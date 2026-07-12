from __future__ import annotations

import pytest

pytestmark = pytest.mark.heavy


def _contains_identity(root, target) -> bool:
    stack = [root]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node is target:
            return True
        identity = id(node)
        if identity in seen:
            continue
        seen.add(identity)
        stack.extend(getattr(node, "ufl_operands", ()))
    return False


def test_pml_tensor_retains_live_frequency_constant() -> None:
    pytest.importorskip("dolfinx")
    pytest.importorskip("ufl")
    from dolfinx import fem, mesh
    from mpi4py import MPI
    from petsc4py import PETSc
    import ufl

    from scatter3d.fem.config import PMLConfig
    from scatter3d.fem.pml import cartesian_pml_tensor

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    k0 = fem.Constant(domain, PETSc.ScalarType(4.0))
    config = PMLConfig(
        physical_min_m=(0.2, 0.2, 0.2),
        physical_max_m=(0.8, 0.8, 0.8),
        thickness_m=(0.2, 0.2, 0.2),
    )
    tensor = cartesian_pml_tensor(ufl.SpatialCoordinate(domain), k0, config)
    assert _contains_identity(tensor, k0)
