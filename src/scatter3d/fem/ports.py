"""Port definitions, independent mode normalization, and modal observables."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class PortDefinition:
    name: str
    facet_tag: int
    reference_impedance_ohm: float = 50.0
    target_forward_power_w: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("port name must not be empty")
        if int(self.facet_tag) <= 0:
            raise ValueError("facet_tag must be a positive integer")
        impedance = float(self.reference_impedance_ohm)
        power = float(self.target_forward_power_w)
        if not isfinite(impedance) or impedance <= 0:
            raise ValueError("reference_impedance_ohm must be finite and positive")
        if not isfinite(power) or power <= 0:
            raise ValueError("target_forward_power_w must be finite and positive")
        object.__setattr__(self, "facet_tag", int(self.facet_tag))
        object.__setattr__(self, "reference_impedance_ohm", impedance)
        object.__setattr__(self, "target_forward_power_w", power)

    def canonical(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "facet_tag": self.facet_tag,
            "reference_impedance_ohm": self.reference_impedance_ohm,
            "target_forward_power_w": self.target_forward_power_w,
        }


@dataclass(frozen=True, slots=True)
class NormalizedPortMode:
    definition: PortDefinition
    field: Any
    raw_power_w: float
    scale: float


@dataclass(frozen=True, slots=True)
class PortExcitation:
    """Equivalent surface current used as one Maxwell right-hand side.

    ``surface_current`` is a DOLFINx coefficient in A/m.  Its weak-form units
    and sign are explicit: the solver assembles ``integral J_s dot v ds`` and
    does not hide an antenna calibration factor.
    """

    definition: PortDefinition
    surface_current: Any
    amplitude: complex = 1.0 + 0.0j


def _tangential(field: Any, normal: Any) -> Any:
    import ufl

    return field - normal * ufl.dot(field, normal)


def normalize_port_mode(
    mode: Any,
    facet_tags: Any,
    definition: PortDefinition,
) -> NormalizedPortMode:
    """Normalize one tangential E mode to its own requested forward power.

    The calculation is repeated independently for each port.  This prevents a
    common error where every antenna is scaled with port zero's mode norm.
    """

    import numpy as np
    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    mesh = mode.function_space.mesh
    normal = ufl.FacetNormal(mesh)
    tangential = _tangential(mode, normal)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    local_energy = fem.assemble_scalar(
        fem.form(ufl.inner(tangential, tangential) * ds(definition.facet_tag))
    )
    energy = float(mesh.comm.allreduce(local_energy.real, op=MPI.SUM))
    raw_power = energy / (2.0 * definition.reference_impedance_ohm)
    if not np.isfinite(raw_power) or raw_power <= 0:
        raise ValueError(
            f"port {definition.name!r} has zero or invalid tangential modal power"
        )
    scale = float((definition.target_forward_power_w / raw_power) ** 0.5)
    normalized = fem.Function(mode.function_space, name=f"mode_{definition.name}")
    normalized.x.array[:] = scale * mode.x.array
    normalized.x.scatter_forward()
    return NormalizedPortMode(definition, normalized, raw_power, scale)


def modal_overlap(solution: Any, normalized_mode: NormalizedPortMode, facet_tags: Any) -> complex:
    """Return a deterministic surface overlap; calibration converts it to S."""

    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    mesh = solution.function_space.mesh
    normal = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    integrand = ufl.inner(
        _tangential(solution, normal),
        _tangential(normalized_mode.field, normal),
    )
    local = fem.assemble_scalar(
        fem.form(integrand * ds(normalized_mode.definition.facet_tag))
    )
    return complex(mesh.comm.allreduce(local, op=MPI.SUM))
