"""Analytical TEM coaxial-line oracle for inexpensive port/cable checks.

The routines in this module use standard telegrapher-equation formulas and do
not depend on DOLFINx or PETSc.  They are intended as a fast, independently
checkable gate before running a three-dimensional Maxwell solve.

The model assumes a uniform, homogeneous, isotropic dielectric fill and
frequency-local per-unit-length parameters.  It does not model connectors,
higher-order modes, radiation, conductor dispersion, or the three-dimensional
port-normalization integral; passing this oracle is therefore necessary but
not sufficient evidence for a finite-element port implementation.

Conventions
-----------
Time dependence is ``exp(-j*omega*t)`` and a forward wave varies as
``exp(-gamma*z)``.  The ABCD convention is
``[V_in, I_in] = [[A, B], [C, D]] [V_out, I_out]`` with ``I_out`` directed
toward the load.  Consequently, a lossless short has
``Gamma_in = -exp(+2j*beta*length)``.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

# CODATA 2018 values.  Keeping them local avoids a SciPy dependency in this
# lightweight analytical gate.
VACUUM_PERMITTIVITY_F_PER_M = 8.854_187_812_8e-12
VACUUM_PERMEABILITY_H_PER_M = 1.256_637_062_12e-6


def _finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _real(name: str, value: Any, *, minimum: float, strict: bool) -> float:
    result = _finite_real(name, value)
    invalid = result <= minimum if strict else result < minimum
    if invalid:
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be {relation} {minimum}")
    return result


def _positive_real(name: str, value: Any) -> float:
    return _real(name, value, minimum=0.0, strict=True)


def _nonnegative_real(name: str, value: Any) -> float:
    return _real(name, value, minimum=0.0, strict=False)


def _finite_complex(name: str, value: Any) -> complex:
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a complex scalar")
    try:
        result = complex(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must have finite real and imaginary parts") from exc
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a complex scalar") from exc
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError(f"{name} must have finite real and imaginary parts")
    real = 0.0 if result.real == 0.0 else result.real
    imag = 0.0 if result.imag == 0.0 else result.imag
    return complex(real, imag)


@dataclass(frozen=True, slots=True)
class ComplexValue:
    """JSON-safe representation of a complex scalar."""

    real: float
    imag: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "real", _finite_real("real", self.real))
        object.__setattr__(self, "imag", _finite_real("imag", self.imag))

    @classmethod
    def from_complex(cls, value: Any) -> ComplexValue:
        finite = _finite_complex("complex value", value)
        return cls(real=finite.real, imag=finite.imag)

    @property
    def value(self) -> complex:
        return complex(self.real, self.imag)

    def __complex__(self) -> complex:
        return self.value

    def to_dict(self) -> dict[str, float]:
        return {"real": self.real, "imag": self.imag}


@dataclass(frozen=True, slots=True)
class CoaxialCable:
    """Uniform coax geometry and passive material data.

    ``loss_tangent`` and ``dielectric_conductivity_s_per_m`` are additive loss
    mechanisms.  ``series_resistance_ohm_per_m`` permits a measured or
    independently estimated conductor-loss term without embedding a particular
    skin-effect model in this oracle.
    """

    inner_radius_m: float
    outer_radius_m: float
    relative_permittivity: float = 1.0
    relative_permeability: float = 1.0
    loss_tangent: float = 0.0
    dielectric_conductivity_s_per_m: float = 0.0
    series_resistance_ohm_per_m: float = 0.0

    def __post_init__(self) -> None:
        normalized = {
            "inner_radius_m": _positive_real("inner_radius_m", self.inner_radius_m),
            "outer_radius_m": _positive_real("outer_radius_m", self.outer_radius_m),
            "relative_permittivity": _positive_real(
                "relative_permittivity", self.relative_permittivity
            ),
            "relative_permeability": _positive_real(
                "relative_permeability", self.relative_permeability
            ),
            "loss_tangent": _nonnegative_real("loss_tangent", self.loss_tangent),
            "dielectric_conductivity_s_per_m": _nonnegative_real(
                "dielectric_conductivity_s_per_m", self.dielectric_conductivity_s_per_m
            ),
            "series_resistance_ohm_per_m": _nonnegative_real(
                "series_resistance_ohm_per_m", self.series_resistance_ohm_per_m
            ),
        }
        if normalized["outer_radius_m"] <= normalized["inner_radius_m"]:
            raise ValueError("outer_radius_m must be greater than inner_radius_m")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CoaxLineParameters:
    """Per-unit-length quantities at one frequency, encoded for JSON output."""

    frequency_hz: float
    angular_frequency_rad_s: float
    inductance_h_per_m: float
    capacitance_f_per_m: float
    series_resistance_ohm_per_m: float
    shunt_conductance_s_per_m: float
    propagation_constant_per_m: ComplexValue
    characteristic_impedance_ohm: ComplexValue

    def __post_init__(self) -> None:
        positive_fields = (
            "frequency_hz",
            "angular_frequency_rad_s",
            "inductance_h_per_m",
            "capacitance_f_per_m",
        )
        nonnegative_fields = (
            "series_resistance_ohm_per_m",
            "shunt_conductance_s_per_m",
        )
        for name in positive_fields:
            object.__setattr__(self, name, _positive_real(name, getattr(self, name)))
        for name in nonnegative_fields:
            object.__setattr__(self, name, _nonnegative_real(name, getattr(self, name)))
        if not math.isclose(
            self.angular_frequency_rad_s,
            math.tau * self.frequency_hz,
            rel_tol=2.0e-15,
        ):
            raise ValueError("angular_frequency_rad_s is inconsistent with frequency_hz")
        if not isinstance(self.propagation_constant_per_m, ComplexValue):
            raise TypeError("propagation_constant_per_m must be a ComplexValue")
        if not isinstance(self.characteristic_impedance_ohm, ComplexValue):
            raise TypeError("characteristic_impedance_ohm must be a ComplexValue")
        if self.gamma.real < 0.0 or self.gamma.imag >= 0.0:
            raise ValueError(
                "propagation constant must have nonnegative loss and negative "
                "imaginary part for exp(-j*omega*t)"
            )
        if self.z0.real <= 0.0:
            raise ValueError("characteristic impedance must have positive real part")

    @property
    def gamma(self) -> complex:
        return self.propagation_constant_per_m.value

    @property
    def z0(self) -> complex:
        return self.characteristic_impedance_ohm.value

    @property
    def attenuation_np_per_m(self) -> float:
        return self.gamma.real

    @property
    def phase_constant_rad_per_m(self) -> float:
        return -self.gamma.imag

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coax_line_parameters(cable: CoaxialCable, frequency_hz: float) -> CoaxLineParameters:
    """Return TEM telegrapher parameters for a uniform coaxial cable.

    The per-unit-length model is

    ``L = mu*ln(b/a)/(2*pi)``, ``C = 2*pi*epsilon/ln(b/a)``,
    ``G = 2*pi*sigma/ln(b/a) + omega*C*tan(delta)``,
    ``gamma = sqrt((R-j*omega*L)*(G-j*omega*C))``, and
    ``Z0 = sqrt((R-j*omega*L)/(G-j*omega*C))``.
    """

    if not isinstance(cable, CoaxialCable):
        raise TypeError("cable must be a CoaxialCable")
    frequency = _positive_real("frequency_hz", frequency_hz)
    omega = _positive_real("angular_frequency_rad_s", math.tau * frequency)
    # log1p preserves adjacent representable radii; the difference-of-logs
    # fallback avoids overflow when the finite radius ratio is enormous.
    relative_gap = (cable.outer_radius_m - cable.inner_radius_m) / cable.inner_radius_m
    if math.isfinite(relative_gap):
        log_radius_ratio = math.log1p(relative_gap)
    else:
        log_radius_ratio = math.log(cable.outer_radius_m) - math.log(cable.inner_radius_m)
    log_radius_ratio = _positive_real("log radius ratio", log_radius_ratio)
    permeability = VACUUM_PERMEABILITY_H_PER_M * cable.relative_permeability
    permittivity = VACUUM_PERMITTIVITY_F_PER_M * cable.relative_permittivity
    inductance = _positive_real(
        "inductance_h_per_m", permeability * log_radius_ratio / math.tau
    )
    capacitance = _positive_real(
        "capacitance_f_per_m", math.tau * permittivity / log_radius_ratio
    )
    conductance = _nonnegative_real(
        "shunt_conductance_s_per_m",
        math.tau * cable.dielectric_conductivity_s_per_m / log_radius_ratio
        + omega * capacitance * cable.loss_tangent,
    )

    series_reactance = _positive_real("series reactance per metre", omega * inductance)
    shunt_susceptance = _positive_real("shunt susceptance per metre", omega * capacitance)
    series_impedance = complex(cable.series_resistance_ohm_per_m, -series_reactance)
    shunt_admittance = complex(conductance, -shunt_susceptance)
    if cable.series_resistance_ohm_per_m == 0.0 and conductance == 0.0:
        # Preserve the exact lossless signs and zeros instead of relying on
        # cancellation between the real parts of two complex square roots.
        inductance_root = math.sqrt(inductance)
        capacitance_root = math.sqrt(capacitance)
        gamma = _finite_complex(
            "propagation constant",
            complex(0.0, -omega * (inductance_root * capacitance_root)),
        )
        z0 = _finite_complex(
            "characteristic impedance", inductance_root / capacitance_root
        )
    else:
        # Taking the roots before multiplying/dividing avoids avoidable
        # overflow in ZY and Z/Y while retaining the passive branches.
        series_root = cmath.sqrt(series_impedance)
        shunt_root = cmath.sqrt(shunt_admittance)
        gamma = _finite_complex("propagation constant", series_root * shunt_root)
        z0 = _finite_complex("characteristic impedance", series_root / shunt_root)

    # Choose the passive forward-wave branches.  The principal square root
    # already has these signs for passive R/L/G/C, but making the choice explicit
    # prevents platform-dependent signed-zero surprises at the lossless limit.
    if gamma.imag > 0.0:
        gamma = -gamma
    if gamma.real < 0.0:
        # For passive R/L/G/C this can only be cancellation roundoff: both
        # square-root factors have arguments in [-pi/4, 0].
        gamma = complex(0.0, gamma.imag)
    if z0.real < 0.0:
        z0 = -z0

    return CoaxLineParameters(
        frequency_hz=frequency,
        angular_frequency_rad_s=omega,
        inductance_h_per_m=inductance,
        capacitance_f_per_m=capacitance,
        series_resistance_ohm_per_m=cable.series_resistance_ohm_per_m,
        shunt_conductance_s_per_m=conductance,
        propagation_constant_per_m=ComplexValue.from_complex(gamma),
        characteristic_impedance_ohm=ComplexValue.from_complex(z0),
    )


@dataclass(frozen=True, slots=True)
class ABCDMatrix:
    """JSON-ready two-port transmission matrix."""

    a: ComplexValue
    b: ComplexValue
    c: ComplexValue
    d: ComplexValue

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "d"):
            if not isinstance(getattr(self, name), ComplexValue):
                raise TypeError(f"{name} must be a ComplexValue")

    @classmethod
    def from_complex(cls, a: Any, b: Any, c: Any, d: Any) -> ABCDMatrix:
        return cls(
            a=ComplexValue.from_complex(a),
            b=ComplexValue.from_complex(b),
            c=ComplexValue.from_complex(c),
            d=ComplexValue.from_complex(d),
        )

    @classmethod
    def identity(cls) -> ABCDMatrix:
        return cls.from_complex(1.0, 0.0, 0.0, 1.0)

    @property
    def A(self) -> complex:
        return self.a.value

    @property
    def B(self) -> complex:
        return self.b.value

    @property
    def C(self) -> complex:
        return self.c.value

    @property
    def D(self) -> complex:
        return self.d.value

    @property
    def determinant(self) -> complex:
        return self.A * self.D - self.B * self.C

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def uniform_line_abcd(parameters: CoaxLineParameters, length_m: float) -> ABCDMatrix:
    """Return the ABCD matrix for one uniform line section."""

    if not isinstance(parameters, CoaxLineParameters):
        raise TypeError("parameters must be CoaxLineParameters")
    length = _nonnegative_real("length_m", length_m)
    try:
        electrical_length = _finite_complex("electrical length", parameters.gamma * length)
        hyperbolic_cosine = _finite_complex(
            "ABCD hyperbolic cosine", cmath.cosh(electrical_length)
        )
        hyperbolic_sine = _finite_complex(
            "ABCD hyperbolic sine", cmath.sinh(electrical_length)
        )
    except (OverflowError, ValueError) as exc:
        raise ValueError("line length and loss produce a non-finite ABCD matrix") from exc
    return ABCDMatrix.from_complex(
        hyperbolic_cosine,
        parameters.z0 * hyperbolic_sine,
        hyperbolic_sine / parameters.z0,
        hyperbolic_cosine,
    )


def cascade_abcd(*sections: ABCDMatrix) -> ABCDMatrix:
    """Cascade one or more sections in physical source-to-load order."""

    if not sections:
        raise ValueError("at least one ABCD section is required")
    result = ABCDMatrix.identity()
    for index, section in enumerate(sections):
        if not isinstance(section, ABCDMatrix):
            raise TypeError(f"sections[{index}] must be an ABCDMatrix")
        result = ABCDMatrix.from_complex(
            result.A * section.A + result.B * section.C,
            result.A * section.B + result.B * section.D,
            result.C * section.A + result.D * section.C,
            result.C * section.B + result.D * section.D,
        )
    return result


@dataclass(frozen=True, slots=True)
class Termination:
    """Passive short, open, matched, or finite-impedance termination."""

    kind: str
    impedance_ohm: ComplexValue | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise TypeError("termination kind must be a string")
        allowed = {"short", "open", "matched", "impedance"}
        if self.kind not in allowed:
            raise ValueError(f"termination kind must be one of {sorted(allowed)}")
        if self.kind in {"short", "open"}:
            if self.impedance_ohm is not None:
                raise ValueError(f"{self.kind} termination must not include impedance_ohm")
            return
        if not isinstance(self.impedance_ohm, ComplexValue):
            raise TypeError(f"{self.kind} termination requires ComplexValue impedance_ohm")
        impedance = self.impedance_ohm.value
        minimum_real = 0.0
        if impedance.real < minimum_real:
            raise ValueError("termination impedance must be passive (real part >= 0)")
        if self.kind == "matched" and impedance.real == 0.0:
            raise ValueError("matched termination impedance must have positive real part")

    @classmethod
    def short(cls) -> Termination:
        return cls(kind="short")

    @classmethod
    def open(cls) -> Termination:
        return cls(kind="open")

    @classmethod
    def matched(cls, impedance_ohm: Any = 50.0) -> Termination:
        return cls(kind="matched", impedance_ohm=ComplexValue.from_complex(impedance_ohm))

    @classmethod
    def impedance(cls, impedance_ohm: Any) -> Termination:
        return cls(kind="impedance", impedance_ohm=ComplexValue.from_complex(impedance_ohm))

    @property
    def resolved_impedance_ohm(self) -> complex | None:
        if self.kind == "open":
            return None
        if self.kind == "short":
            return 0.0j
        assert self.impedance_ohm is not None
        return self.impedance_ohm.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def terminated_input_impedance(
    network: ABCDMatrix, termination: Termination
) -> complex | None:
    """Return input impedance, with ``None`` representing an ideal open circuit."""

    if not isinstance(network, ABCDMatrix):
        raise TypeError("network must be an ABCDMatrix")
    if not isinstance(termination, Termination):
        raise TypeError("termination must be a Termination")

    load = termination.resolved_impedance_ohm
    if load is None:
        numerator, denominator = network.A, network.C
    else:
        direct_numerator = network.A * load + network.B
        direct_denominator = network.C * load + network.D
        direct_is_finite = all(
            math.isfinite(component)
            for value in (direct_numerator, direct_denominator)
            for component in (value.real, value.imag)
        )
        if direct_is_finite:
            numerator, denominator = direct_numerator, direct_denominator
        else:
            # Divide the fractional-linear transform by Z_load only when the
            # direct form overflows.  Keeping the direct form when possible
            # also preserves exact identities at the largest finite float.
            inverse_load = 1.0 / load
            numerator = network.A + network.B * inverse_load
            denominator = network.C + network.D * inverse_load
    if denominator == 0.0j:
        if numerator == 0.0j:
            raise ValueError("network and termination produce an indeterminate input impedance")
        return None
    return _finite_complex("input impedance", numerator / denominator)


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    """Input reflection plus finite, deterministic serialization fields."""

    reference_impedance_ohm: float
    termination: Termination
    input_impedance_ohm: ComplexValue | None
    input_is_open: bool
    reflection_coefficient: ComplexValue
    magnitude: float
    phase_deg: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_impedance_ohm",
            _positive_real("reference_impedance_ohm", self.reference_impedance_ohm),
        )
        if not isinstance(self.termination, Termination):
            raise TypeError("termination must be a Termination")
        if self.input_impedance_ohm is not None and not isinstance(
            self.input_impedance_ohm, ComplexValue
        ):
            raise TypeError("input_impedance_ohm must be a ComplexValue or None")
        if not isinstance(self.input_is_open, bool):
            raise TypeError("input_is_open must be a bool")
        if self.input_is_open != (self.input_impedance_ohm is None):
            raise ValueError("input_is_open is inconsistent with input_impedance_ohm")
        if not isinstance(self.reflection_coefficient, ComplexValue):
            raise TypeError("reflection_coefficient must be a ComplexValue")
        magnitude = _nonnegative_real("magnitude", self.magnitude)
        if not math.isclose(magnitude, abs(self.gamma), rel_tol=2.0e-15, abs_tol=0.0):
            raise ValueError("magnitude is inconsistent with reflection_coefficient")
        object.__setattr__(self, "magnitude", magnitude)
        if self.gamma == 0.0j:
            if self.phase_deg is not None:
                raise ValueError("phase_deg must be None for zero reflection")
        else:
            if self.phase_deg is None:
                raise ValueError("phase_deg is required for nonzero reflection")
            phase = _finite_real("phase_deg", self.phase_deg)
            if phase < -180.0 or phase > 180.0:
                raise ValueError("phase_deg must be in [-180, 180]")
            object.__setattr__(self, "phase_deg", phase)

    @property
    def gamma(self) -> complex:
        return self.reflection_coefficient.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def input_reflection(
    network: ABCDMatrix,
    termination: Termination,
    reference_impedance_ohm: float = 50.0,
) -> ReflectionResult:
    """Compute the voltage-wave input reflection for a positive real reference."""

    reference = _positive_real("reference_impedance_ohm", reference_impedance_ohm)
    input_impedance = terminated_input_impedance(network, termination)
    if input_impedance is None:
        reflection = 1.0 + 0.0j
        encoded_input = None
    else:
        impedance_scale = max(
            abs(input_impedance.real), abs(input_impedance.imag), reference
        )
        scaled_impedance = input_impedance / impedance_scale
        scaled_reference = reference / impedance_scale
        denominator = scaled_impedance + scaled_reference
        if denominator == 0.0j:
            raise ValueError("input impedance makes reflection coefficient singular")
        reflection = _finite_complex(
            "reflection coefficient",
            (scaled_impedance - scaled_reference) / denominator,
        )
        encoded_input = ComplexValue.from_complex(input_impedance)
    magnitude = _nonnegative_real("reflection magnitude", abs(reflection))
    phase = None if reflection == 0.0j else math.degrees(cmath.phase(reflection))
    return ReflectionResult(
        reference_impedance_ohm=reference,
        termination=termination,
        input_impedance_ohm=encoded_input,
        input_is_open=input_impedance is None,
        reflection_coefficient=ComplexValue.from_complex(reflection),
        magnitude=magnitude,
        phase_deg=phase,
    )


def circular_phase_error_deg(actual: Any, expected: Any) -> float:
    """Return ``phase(actual)-phase(expected)`` wrapped to [-180, 180] degrees."""

    actual_value = _finite_complex("actual", actual)
    expected_value = _finite_complex("expected", expected)
    if actual_value == 0.0j or expected_value == 0.0j:
        raise ValueError("phase error is undefined for a zero-magnitude value")
    difference = math.degrees(cmath.phase(actual_value) - cmath.phase(expected_value))
    wrapped = math.remainder(difference, 360.0)
    if wrapped == -180.0:
        wrapped = 180.0
    return 0.0 if wrapped == 0.0 else wrapped


@dataclass(frozen=True, slots=True)
class SParameterErrorMetrics:
    """Aggregate complex-S errors with phase differences computed circularly."""

    sample_count: int
    phase_sample_count: int
    complex_mae: float
    complex_rmse: float
    normalized_complex_rmse: float | None
    max_abs_complex_error: float
    magnitude_rmse: float
    phase_mae_deg: float | None
    phase_rmse_deg: float | None
    max_abs_phase_error_deg: float | None

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, Integral):
            raise TypeError("sample_count must be an integer")
        if isinstance(self.phase_sample_count, bool) or not isinstance(
            self.phase_sample_count, Integral
        ):
            raise TypeError("phase_sample_count must be an integer")
        sample_count = int(self.sample_count)
        phase_sample_count = int(self.phase_sample_count)
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if not 0 <= phase_sample_count <= sample_count:
            raise ValueError("phase_sample_count must be between zero and sample_count")
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "phase_sample_count", phase_sample_count)

        required_metrics = (
            "complex_mae",
            "complex_rmse",
            "max_abs_complex_error",
            "magnitude_rmse",
        )
        for name in required_metrics:
            object.__setattr__(self, name, _nonnegative_real(name, getattr(self, name)))
        if self.normalized_complex_rmse is not None:
            object.__setattr__(
                self,
                "normalized_complex_rmse",
                _nonnegative_real(
                    "normalized_complex_rmse", self.normalized_complex_rmse
                ),
            )

        phase_names = ("phase_mae_deg", "phase_rmse_deg", "max_abs_phase_error_deg")
        if phase_sample_count == 0:
            if any(getattr(self, name) is not None for name in phase_names):
                raise ValueError("phase metrics must be None when phase_sample_count is zero")
        else:
            for name in phase_names:
                value = getattr(self, name)
                if value is None:
                    raise ValueError(
                        "phase metrics are required when phase_sample_count is positive"
                    )
                normalized = _nonnegative_real(name, value)
                if normalized > 180.0:
                    raise ValueError(f"{name} must not exceed 180 degrees")
                object.__setattr__(self, name, normalized)

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _finite_complex_array(name: str, values: Sequence[complex] | np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.complex128)
    except OverflowError as exc:
        raise ValueError(f"{name} must contain only finite complex128 values") from exc
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an array of complex-compatible values") from exc
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array.real) & np.isfinite(array.imag)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _scaled_mean(nonnegative_values: np.ndarray) -> float:
    """Return a finite mean without overflowing the intermediate sum."""

    scale = float(np.max(nonnegative_values))
    if scale == 0.0:
        return 0.0
    return float(scale * np.mean(nonnegative_values / scale))


def _scaled_root_mean_square(nonnegative_values: np.ndarray) -> float:
    """Return a finite RMS without squaring values at their original scale."""

    scale = float(np.max(nonnegative_values))
    if scale == 0.0:
        return 0.0
    scaled = nonnegative_values / scale
    return float(scale * np.sqrt(np.mean(scaled * scaled)))


def _circular_phase_differences_deg(
    actual: np.ndarray, expected: np.ndarray
) -> np.ndarray:
    """Return wrapped phase differences without magnitude products."""

    differences = np.degrees(np.angle(actual) - np.angle(expected))
    wrapped = np.remainder(differences + 180.0, 360.0) - 180.0
    wrapped = np.where(wrapped == -180.0, 180.0, wrapped)
    return np.where(wrapped == 0.0, 0.0, wrapped)


def complex_s_error_metrics(
    actual: Sequence[complex] | np.ndarray,
    expected: Sequence[complex] | np.ndarray,
    *,
    phase_magnitude_floor: float = 0.0,
) -> SParameterErrorMetrics:
    """Compare equal-shaped complex S arrays without phase-wrap artifacts.

    Samples for which either magnitude is at or below
    ``phase_magnitude_floor`` are excluded only from the phase statistics; they
    remain in all complex and magnitude-error statistics.
    """

    actual_array = _finite_complex_array("actual", actual)
    expected_array = _finite_complex_array("expected", expected)
    if actual_array.shape != expected_array.shape:
        raise ValueError(
            f"actual and expected shapes differ: {actual_array.shape} != {expected_array.shape}"
        )
    floor = _nonnegative_real("phase_magnitude_floor", phase_magnitude_floor)
    actual_flat = actual_array.reshape(-1)
    expected_flat = expected_array.reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        errors = actual_flat - expected_flat
        absolute_errors = np.abs(errors)
        actual_magnitudes = np.abs(actual_flat)
        expected_magnitudes = np.abs(expected_flat)
    derived_arrays = (absolute_errors, actual_magnitudes, expected_magnitudes)
    if any(not np.all(np.isfinite(values)) for values in derived_arrays):
        raise ValueError("actual and expected produce errors outside the finite float range")

    complex_mae = _scaled_mean(absolute_errors)
    complex_rmse = _scaled_root_mean_square(absolute_errors)
    expected_rms = _scaled_root_mean_square(expected_magnitudes)
    if expected_rms > 0.0:
        normalized_rmse = complex_rmse / expected_rms
        if not math.isfinite(normalized_rmse):
            raise ValueError("normalized complex RMSE exceeds the finite float range")
    else:
        normalized_rmse = None
    magnitude_rmse = _scaled_root_mean_square(
        np.abs(actual_magnitudes - expected_magnitudes)
    )

    phase_mask = (actual_magnitudes > floor) & (expected_magnitudes > floor)
    phase_errors = _circular_phase_differences_deg(
        actual_flat[phase_mask], expected_flat[phase_mask]
    )
    if phase_errors.size:
        phase_mae = float(np.mean(np.abs(phase_errors)))
        phase_rmse = float(np.sqrt(np.mean(phase_errors**2)))
        max_phase = float(np.max(np.abs(phase_errors)))
    else:
        phase_mae = None
        phase_rmse = None
        max_phase = None

    return SParameterErrorMetrics(
        sample_count=int(actual_flat.size),
        phase_sample_count=int(phase_errors.size),
        complex_mae=complex_mae,
        complex_rmse=complex_rmse,
        normalized_complex_rmse=normalized_rmse,
        max_abs_complex_error=float(np.max(absolute_errors)),
        magnitude_rmse=magnitude_rmse,
        phase_mae_deg=phase_mae,
        phase_rmse_deg=phase_rmse,
        max_abs_phase_error_deg=max_phase,
    )
