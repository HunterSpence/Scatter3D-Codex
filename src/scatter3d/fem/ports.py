"""Power-normalized single-mode TEM ports and explicitly uncalibrated loads.

The FEM field convention is ``exp(-i omega t)``.  ``PortDefinition`` keeps the
VNA/circuit reference impedance separate from the field-wave impedance used in
Maxwell's boundary condition.  They have different physical roles and are not
interchangeable.

The matched condition implemented by :mod:`scatter3d.fem.forms` assumes one
known TEM mode and a scalar, frequency-independent field-wave impedance
``Z_f``.  With outward normal ``n`` and ``Y_f = 1 / Z_f``, it is

``n x (mu_r^-1 curl(E)) + i k0 Z_vac Y_f E_t = 2 i k0 Z_vac Y_f E_inc``.

It is a first-order, single-mode termination.  It is not a waveguide eigenmode
solver, a multimode DtN map, or an automatic 50-ohm S-parameter calibration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from math import isfinite, sqrt
from operator import index
from typing import Any

from .config import EPSILON_0, MU_0

VACUUM_IMPEDANCE_OHM = sqrt(MU_0 / EPSILON_0)


def _finite_complex(value: complex, label: str) -> complex:
    result = complex(value)
    if not (isfinite(result.real) and isfinite(result.imag)):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_tag(value: int, label: str = "facet_tag") -> int:
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


@dataclass(frozen=True, slots=True)
class PortDefinition:
    """Physical contract for one matched single-mode TEM boundary.

    ``field_wave_impedance_ohm`` is the local tangential ``|E| / |H|`` ratio
    used by the Maxwell boundary and Poynting normalization.  It is required
    and keyword-only so a 50-ohm VNA reference cannot silently take its place.

    ``circuit_reference_impedance_ohm`` defines the external circuit power-wave
    reference.  It is provenance for later S-parameter renormalization and does
    not enter the field boundary condition.

    ``outgoing_propagation_index`` defines the spatial branch by
    ``exp(i*k0*n_eff*s)`` in the outward direction.  Positive real part is
    required for a forward phase-propagating mode and nonnegative imaginary
    part for passive attenuation under the package time convention.

    A mode is accepted only when ``Re(1 / Z_f) > 0``.  This rejects zero-power
    evanescent impedances and non-passive branches.  A scalar impedance is an
    explicit single-mode approximation; frequency-dependent or multimode ports
    require a new definition/operator at each frequency.
    """

    name: str
    facet_tag: int
    _: KW_ONLY
    field_wave_impedance_ohm: complex
    outgoing_propagation_index: complex
    circuit_reference_impedance_ohm: float = 50.0
    target_forward_power_w: float = 1.0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("port name must not be empty")
        tag = _positive_tag(self.facet_tag)
        field_impedance = _finite_complex(
            self.field_wave_impedance_ohm, "field_wave_impedance_ohm"
        )
        if field_impedance == 0:
            raise ValueError("field_wave_impedance_ohm must be nonzero")
        try:
            field_admittance = 1.0 / field_impedance
        except ZeroDivisionError as exc:
            raise ValueError(
                "field_wave_impedance_ohm is too small to define finite admittance"
            ) from exc
        if not (
            isfinite(field_admittance.real)
            and isfinite(field_admittance.imag)
            and field_admittance.real > 0.0
        ):
            raise ValueError(
                "field_wave_impedance_ohm must select a passive propagating mode "
                "with Re(1/Z_f) > 0; evanescent and non-passive modes are unsupported"
            )
        propagation_index = _finite_complex(
            self.outgoing_propagation_index, "outgoing_propagation_index"
        )
        if propagation_index.real <= 0.0:
            raise ValueError(
                "outgoing_propagation_index must have positive real part; "
                "cutoff/evanescent and backward-wave modes are unsupported"
            )
        if propagation_index.imag < 0.0:
            raise ValueError(
                "outgoing_propagation_index must have nonnegative imaginary part "
                "for the exp(i*k0*n_eff*s) passive outgoing convention"
            )
        circuit_impedance = float(self.circuit_reference_impedance_ohm)
        target_power = float(self.target_forward_power_w)
        if not isfinite(circuit_impedance) or circuit_impedance <= 0.0:
            raise ValueError(
                "circuit_reference_impedance_ohm must be finite and positive"
            )
        if not isfinite(target_power) or target_power <= 0.0:
            raise ValueError("target_forward_power_w must be finite and positive")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "facet_tag", tag)
        object.__setattr__(self, "field_wave_impedance_ohm", field_impedance)
        object.__setattr__(self, "outgoing_propagation_index", propagation_index)
        object.__setattr__(
            self, "circuit_reference_impedance_ohm", circuit_impedance
        )
        object.__setattr__(self, "target_forward_power_w", target_power)

    @property
    def field_wave_admittance_siemens(self) -> complex:
        return 1.0 / self.field_wave_impedance_ohm

    @property
    def forward_power_admittance_siemens(self) -> float:
        """Real admittance multiplying ``|E_t|^2 / 2`` in forward power."""

        return self.field_wave_admittance_siemens.real

    def canonical(self) -> dict[str, Any]:
        field_impedance = self.field_wave_impedance_ohm
        return {
            "name": self.name,
            "facet_tag": self.facet_tag,
            "field_wave_impedance_ohm": [
                field_impedance.real,
                field_impedance.imag,
            ],
            "outgoing_propagation_index": [
                self.outgoing_propagation_index.real,
                self.outgoing_propagation_index.imag,
            ],
            "circuit_reference_impedance_ohm": self.circuit_reference_impedance_ohm,
            "target_forward_power_w": self.target_forward_power_w,
            "mode_model": "matched-single-tem",
            "time_convention": "exp(-i*omega*t)",
        }


@dataclass(frozen=True, slots=True)
class NormalizedPortMode:
    """A tangential electric mode scaled to known real forward power."""

    definition: PortDefinition
    field: Any
    raw_forward_power_w: float
    scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.definition, PortDefinition):
            raise TypeError("definition must be a PortDefinition")
        raw_power = float(self.raw_forward_power_w)
        scale = float(self.scale)
        if not isfinite(raw_power) or raw_power <= 0.0:
            raise ValueError("raw_forward_power_w must be finite and positive")
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("mode scale must be finite and positive")
        object.__setattr__(self, "raw_forward_power_w", raw_power)
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True, slots=True)
class MatchedTEMPortExcitation:
    """An inward-travelling incident TEM mode on its matched boundary.

    ``amplitude`` multiplies a mode whose power is
    ``definition.target_forward_power_w``.  Thus the incident power is
    ``abs(amplitude)**2 * target_forward_power_w``.
    """

    mode: NormalizedPortMode
    amplitude: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        if not isinstance(self.mode, NormalizedPortMode):
            raise TypeError("mode must be a NormalizedPortMode")
        amplitude = _finite_complex(self.amplitude, "incident amplitude")
        if amplitude == 0:
            raise ValueError("incident amplitude must be nonzero")
        try:
            incident_power = (
                abs(amplitude) ** 2
                * self.mode.definition.target_forward_power_w
            )
        except OverflowError as exc:
            raise ValueError(
                "incident amplitude produces nonfinite incident power"
            ) from exc
        if not isfinite(incident_power):
            raise ValueError("incident amplitude produces nonfinite incident power")
        object.__setattr__(self, "amplitude", amplitude)

    @property
    def definition(self) -> PortDefinition:
        return self.mode.definition

    @property
    def incident_power_w(self) -> float:
        return abs(self.amplitude) ** 2 * self.definition.target_forward_power_w


@dataclass(frozen=True, slots=True)
class UncalibratedSurfaceCurrentExcitation:
    """Generic pre-scaled weak surface load, explicitly not an S-parameter port.

    ``surface_current`` retains the old generic ``inner(load, test) * ds`` path.
    It has no field-wave impedance, power normalization, matched termination, or
    circuit calibration.  Its amplitude therefore has application-defined weak
    form units and its solution must not be reported as a calibrated port result.
    """

    boundary_name: str
    facet_tag: int
    surface_current: Any
    amplitude: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        name = str(self.boundary_name).strip()
        if not name:
            raise ValueError("boundary_name must not be empty")
        tag = _positive_tag(self.facet_tag)
        amplitude = _finite_complex(self.amplitude, "surface-current amplitude")
        if amplitude == 0:
            raise ValueError("surface-current amplitude must be nonzero")
        object.__setattr__(self, "boundary_name", name)
        object.__setattr__(self, "facet_tag", tag)
        object.__setattr__(self, "amplitude", amplitude)


def validate_port_definitions(
    definitions: Sequence[PortDefinition],
    declared_ports: Mapping[str, int],
) -> tuple[PortDefinition, ...]:
    """Require a one-to-one match between physical definitions and mesh contract."""

    ports = tuple(definitions)
    if any(not isinstance(port, PortDefinition) for port in ports):
        raise TypeError("matched_ports must contain only PortDefinition objects")
    names = [port.name for port in ports]
    tags = [port.facet_tag for port in ports]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    duplicate_tags = sorted({tag for tag in tags if tags.count(tag) > 1})
    if duplicate_names:
        raise ValueError(f"matched port names are duplicated: {duplicate_names}")
    if duplicate_tags:
        raise ValueError(f"matched port facet tags are duplicated: {duplicate_tags}")

    declared = {str(name): int(tag) for name, tag in declared_ports.items()}
    supplied = {port.name: port.facet_tag for port in ports}
    missing = sorted(set(declared) - set(supplied))
    unknown = sorted(set(supplied) - set(declared))
    mismatched = sorted(
        name
        for name in set(declared) & set(supplied)
        if declared[name] != supplied[name]
    )
    if missing or unknown or mismatched:
        pieces: list[str] = []
        if missing:
            pieces.append(f"missing matched definitions for {missing}")
        if unknown:
            pieces.append(f"definitions for undeclared ports {unknown}")
        if mismatched:
            detail = {
                name: {"declared": declared[name], "supplied": supplied[name]}
                for name in mismatched
            }
            pieces.append(f"name/tag mismatches {detail}")
        raise ValueError("; ".join(pieces))
    return ports


def matched_tem_operator_coefficient(k0_per_m: Any, definition: PortDefinition) -> Any:
    """Return ``-i k0 Z_vac / Z_f`` for the matched bilinear form."""

    return (
        -1.0j
        * k0_per_m
        * VACUUM_IMPEDANCE_OHM
        * definition.field_wave_admittance_siemens
    )


def matched_tem_incident_coefficient(
    k0_per_m: Any, excitation: MatchedTEMPortExcitation
) -> Any:
    """Return ``-2 i k0 Z_vac amplitude / Z_f`` for the incident RHS."""

    return (
        2.0
        * excitation.amplitude
        * matched_tem_operator_coefficient(k0_per_m, excitation.definition)
    )


def tangential_trace(field: Any, normal: Any) -> Any:
    """Return the rotated H(curl) tangential trace ``field x normal``.

    Its norm and pairwise inner products equal those of the tangential field,
    while avoiding an unsupported normal trace of an H(curl) function.  This is
    also the representation used by the official DOLFINx Maxwell demo.
    """

    import ufl

    return ufl.cross(field, normal)


def _global_tag_count(mesh: Any, facet_tags: Any, tag: int) -> int:
    from mpi4py import MPI

    local_count = int(facet_tags.find(tag).size)
    return int(mesh.comm.allreduce(local_count, op=MPI.SUM))


def normalize_port_mode(
    mode: Any,
    facet_tags: Any,
    definition: PortDefinition,
) -> NormalizedPortMode:
    """Normalize tangential ``E`` by real forward Poynting power.

    For the declared outgoing TEM branch, ``H = Y_f (n x E_t)`` and therefore

    ``P_forward = 1/2 Re integral(E x conj(H)) . n ds``
    ``          = 1/2 Re(Y_f) integral(|E_t|^2) ds``.

    Each port is integrated and scaled independently.  The circuit reference
    impedance is deliberately absent from this calculation.
    """

    import numpy as np
    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    mesh = mode.function_space.mesh
    if int(facet_tags.dim) != int(mesh.topology.dim) - 1:
        raise ValueError("facet_tags must have mesh-topology dimension minus one")
    if _global_tag_count(mesh, facet_tags, definition.facet_tag) == 0:
        raise ValueError(
            f"port {definition.name!r} facet tag {definition.facet_tag} is absent"
        )
    normal = ufl.FacetNormal(mesh)
    tangential = tangential_trace(mode, normal)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    local_energy = fem.assemble_scalar(
        fem.form(ufl.inner(tangential, tangential) * ds(definition.facet_tag))
    )
    energy = complex(mesh.comm.allreduce(local_energy, op=MPI.SUM))
    imaginary_tolerance = 1.0e-11 * max(1.0, abs(energy.real))
    if abs(energy.imag) > imaginary_tolerance:
        raise ValueError(
            f"port {definition.name!r} modal energy is unexpectedly complex"
        )
    raw_power = 0.5 * definition.forward_power_admittance_siemens * energy.real
    if not np.isfinite(raw_power) or raw_power <= 0.0:
        raise ValueError(
            f"port {definition.name!r} has zero or invalid real forward modal power"
        )
    scale = float(sqrt(definition.target_forward_power_w / raw_power))
    normalized = fem.Function(mode.function_space, name=f"mode_{definition.name}")
    normalized.x.array[:] = scale * mode.x.array
    normalized.x.scatter_forward()
    return NormalizedPortMode(definition, normalized, raw_power, scale)


def total_electric_modal_coefficient(
    solution: Any, normalized_mode: NormalizedPortMode, facet_tags: Any
) -> complex:
    """Project total tangential ``E`` onto one mode; this is not an S-parameter.

    Separating incident and outgoing power waves additionally requires the
    magnetic/curl trace and a validated modal extraction convention.  This
    routine intentionally returns only the electric expansion coefficient so a
    surface overlap cannot be mistaken for calibrated ``S_ij``.
    """

    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    mesh = solution.function_space.mesh
    if normalized_mode.field.function_space is not solution.function_space:
        raise ValueError("solution and normalized mode must use the same function space")
    definition = normalized_mode.definition
    if _global_tag_count(mesh, facet_tags, definition.facet_tag) == 0:
        raise ValueError(
            f"port {definition.name!r} facet tag {definition.facet_tag} is absent"
        )
    normal = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    solution_t = tangential_trace(solution, normal)
    mode_t = tangential_trace(normalized_mode.field, normal)
    numerator_local = fem.assemble_scalar(
        fem.form(ufl.inner(solution_t, mode_t) * ds(definition.facet_tag))
    )
    denominator_local = fem.assemble_scalar(
        fem.form(ufl.inner(mode_t, mode_t) * ds(definition.facet_tag))
    )
    numerator = complex(mesh.comm.allreduce(numerator_local, op=MPI.SUM))
    denominator = complex(mesh.comm.allreduce(denominator_local, op=MPI.SUM))
    if not (
        isfinite(numerator.real)
        and isfinite(numerator.imag)
        and isfinite(denominator.real)
        and isfinite(denominator.imag)
    ):
        raise ValueError(f"port {definition.name!r} modal projection is nonfinite")
    if abs(denominator) == 0.0:
        raise ValueError(f"port {definition.name!r} has zero modal projection norm")
    return numerator / denominator
