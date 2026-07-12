"""Frequency-live Cartesian PML tensors for the Maxwell weak form."""

from __future__ import annotations

from math import log
from typing import Any

import numpy as np

from .config import PMLConfig


def _axis_stretch(
    coordinate: Any,
    k0: Any,
    lower: float,
    upper: float,
    thickness: float,
    polynomial_order: int,
    target_power_reflection: float,
) -> Any:
    import ufl

    if thickness == 0.0:
        return ufl.as_ufl(1.0 + 0.0j)
    lower_depth = ufl.conditional(
        ufl.lt(coordinate, lower), (lower - coordinate) / thickness, 0.0
    )
    upper_depth = ufl.conditional(
        ufl.gt(coordinate, upper), (coordinate - upper) / thickness, 0.0
    )
    normalized_depth = lower_depth + upper_depth
    # A round trip crosses the layer twice.  Choosing integral(sigma)=-log(R)/4
    # makes exp(-4*integral(sigma))=R for the round-trip power attenuation.
    sigma_max = -0.25 * (polynomial_order + 1) * log(target_power_reflection) / thickness
    sigma = sigma_max * normalized_depth**polynomial_order
    # exp(-i*omega*t), outgoing exp(+i*k*x): Im(s)>0 gives spatial decay.
    # Crucially, k0 remains a live fem.Constant in this expression.
    return 1.0 + 1j * sigma / k0


def cartesian_pml_stretches(x: Any, k0: Any, config: PMLConfig) -> tuple[Any, Any, Any]:
    """Return the three symbolic coordinate stretches containing live ``k0``."""

    return tuple(
        _axis_stretch(
            x[axis],
            k0,
            config.physical_min_m[axis],
            config.physical_max_m[axis],
            config.thickness_m[axis],
            config.polynomial_order,
            config.target_power_reflection,
        )
        for axis in range(3)
    )  # type: ignore[return-value]


def cartesian_pml_tensor(x: Any, k0: Any, config: PMLConfig) -> Any:
    """Transformation tensor T used in mass and curl terms.

    For diagonal stretch ``S=diag(sx,sy,sz)``, this returns
    ``det(S) S^-1 S^-T``.  The Maxwell form uses ``T`` in the mass term and
    ``T^-1`` in the curl term.
    """

    import ufl

    sx, sy, sz = cartesian_pml_stretches(x, k0, config)
    return ufl.diag(ufl.as_vector((sy * sz / sx, sx * sz / sy, sx * sy / sz)))


def cartesian_pml_inverse_tensor(x: Any, k0: Any, config: PMLConfig) -> Any:
    import ufl

    sx, sy, sz = cartesian_pml_stretches(x, k0, config)
    return ufl.diag(ufl.as_vector((sx / (sy * sz), sy / (sx * sz), sz / (sx * sy))))


def validate_pml_mesh_extent(mesh: Any, config: PMLConfig, tolerance_m: float = 1.0e-10) -> None:
    """Verify collectively that the mesh reaches every configured PML exterior."""

    coordinates = np.asarray(mesh.geometry.x)
    local_min = coordinates.min(axis=0)[:3]
    local_max = coordinates.max(axis=0)[:3]
    from mpi4py import MPI

    global_min = np.array([mesh.comm.allreduce(v, op=MPI.MIN) for v in local_min])
    global_max = np.array([mesh.comm.allreduce(v, op=MPI.MAX) for v in local_max])
    expected_min = np.asarray(config.physical_min_m) - np.asarray(config.thickness_m)
    expected_max = np.asarray(config.physical_max_m) + np.asarray(config.thickness_m)
    if np.any(global_min > expected_min + tolerance_m) or np.any(
        global_max < expected_max - tolerance_m
    ):
        raise ValueError(
            "mesh does not span the configured PML exterior: "
            f"mesh=[{global_min.tolist()}, {global_max.tolist()}], "
            f"required=[{expected_min.tolist()}, {expected_max.tolist()}]"
        )
