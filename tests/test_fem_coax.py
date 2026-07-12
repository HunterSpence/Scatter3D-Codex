from __future__ import annotations

import cmath
import json
import math

import numpy as np
import pytest

from scatter3d.fem.coax import (
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMITTIVITY_F_PER_M,
    ABCDMatrix,
    CoaxialCable,
    ComplexValue,
    Termination,
    cascade_abcd,
    circular_phase_error_deg,
    coax_line_parameters,
    complex_s_error_metrics,
    input_reflection,
    terminated_input_impedance,
    uniform_line_abcd,
)


def _lossless_cable(
    *, relative_permittivity: float = 1.0, radius_ratio: float = 3.0
) -> CoaxialCable:
    return CoaxialCable(
        inner_radius_m=1.0e-3,
        outer_radius_m=radius_ratio * 1.0e-3,
        relative_permittivity=relative_permittivity,
    )


def test_lossless_matched_line_has_zero_input_reflection() -> None:
    parameters = coax_line_parameters(_lossless_cable(), frequency_hz=1.0e9)
    network = uniform_line_abcd(parameters, length_m=0.137)
    result = input_reflection(
        network,
        Termination.matched(parameters.z0),
        reference_impedance_ohm=parameters.z0.real,
    )

    assert parameters.attenuation_np_per_m == pytest.approx(0.0, abs=1.0e-15)
    assert parameters.z0.imag == pytest.approx(0.0, abs=1.0e-15)
    assert result.gamma == pytest.approx(0.0j, abs=2.0e-15)
    assert result.input_impedance_ohm is not None
    assert result.input_impedance_ohm.value == pytest.approx(parameters.z0, rel=1.0e-14)


def test_lossy_dielectric_parameters_match_closed_form_tem_limit() -> None:
    loss_tangent = 0.04
    parameters = coax_line_parameters(
        CoaxialCable(
            inner_radius_m=1.0e-3,
            outer_radius_m=3.0e-3,
            relative_permittivity=2.5,
            loss_tangent=loss_tangent,
        ),
        frequency_hz=2.3e9,
    )
    loss_factor = cmath.sqrt(1.0 + 1.0j * loss_tangent)
    lossless_beta = parameters.angular_frequency_rad_s * math.sqrt(
        parameters.inductance_h_per_m * parameters.capacitance_f_per_m
    )
    expected_gamma = -1.0j * lossless_beta * loss_factor
    expected_z0 = (
        math.sqrt(parameters.inductance_h_per_m / parameters.capacitance_f_per_m)
        / loss_factor
    )

    assert parameters.gamma == pytest.approx(expected_gamma, rel=2.0e-14)
    assert parameters.z0 == pytest.approx(expected_z0, rel=2.0e-14)
    assert parameters.attenuation_np_per_m > 0.0
    assert parameters.z0.imag < 0.0


def test_geometry_and_material_parameters_match_independent_closed_forms() -> None:
    frequency_hz = 1.7e9
    radius_ratio = 4.25
    cable = CoaxialCable(
        inner_radius_m=0.8e-3,
        outer_radius_m=0.8e-3 * radius_ratio,
        relative_permittivity=2.7,
        relative_permeability=1.3,
        loss_tangent=0.018,
        dielectric_conductivity_s_per_m=2.1e-4,
        series_resistance_ohm_per_m=0.37,
    )
    parameters = coax_line_parameters(cable, frequency_hz)
    log_ratio = math.log(radius_ratio)
    expected_inductance = (
        VACUUM_PERMEABILITY_H_PER_M * cable.relative_permeability * log_ratio / math.tau
    )
    expected_capacitance = (
        math.tau
        * VACUUM_PERMITTIVITY_F_PER_M
        * cable.relative_permittivity
        / log_ratio
    )
    expected_conductance = (
        math.tau * cable.dielectric_conductivity_s_per_m / log_ratio
        + math.tau * frequency_hz * expected_capacitance * cable.loss_tangent
    )
    series_impedance = complex(
        cable.series_resistance_ohm_per_m,
        -parameters.angular_frequency_rad_s * expected_inductance,
    )
    shunt_admittance = complex(
        expected_conductance,
        -parameters.angular_frequency_rad_s * expected_capacitance,
    )
    expected_gamma = cmath.sqrt(series_impedance * shunt_admittance)
    expected_z0 = cmath.sqrt(series_impedance / shunt_admittance)

    assert parameters.inductance_h_per_m == pytest.approx(expected_inductance, rel=2.0e-15)
    assert parameters.capacitance_f_per_m == pytest.approx(expected_capacitance, rel=2.0e-15)
    assert parameters.shunt_conductance_s_per_m == pytest.approx(
        expected_conductance, rel=2.0e-15
    )
    assert parameters.gamma == pytest.approx(expected_gamma, rel=3.0e-15)
    assert parameters.z0 == pytest.approx(expected_z0, rel=3.0e-15)


def test_conductor_loss_has_passive_gamma_and_expected_impedance_sign() -> None:
    parameters = coax_line_parameters(
        CoaxialCable(
            inner_radius_m=1.0e-3,
            outer_radius_m=3.0e-3,
            series_resistance_ohm_per_m=4.0,
        ),
        50.0e6,
    )

    assert parameters.gamma.real > 0.0
    assert parameters.gamma.imag < 0.0
    assert parameters.z0.real > 0.0
    assert parameters.z0.imag > 0.0


def test_lossless_short_has_unit_magnitude_and_analytic_phase() -> None:
    parameters = coax_line_parameters(_lossless_cable(relative_permittivity=2.25), 2.0e9)
    length_m = 0.031
    network = uniform_line_abcd(parameters, length_m)
    result = input_reflection(
        network,
        Termination.short(),
        reference_impedance_ohm=parameters.z0.real,
    )
    expected = -cmath.exp(2.0j * parameters.phase_constant_rad_per_m * length_m)

    assert result.magnitude == pytest.approx(1.0, abs=2.0e-14)
    assert result.gamma == pytest.approx(expected, abs=2.0e-14)
    assert circular_phase_error_deg(result.gamma, expected) == pytest.approx(0.0, abs=1.0e-12)


def test_quarter_wave_line_transforms_load_impedance() -> None:
    parameters = coax_line_parameters(_lossless_cable(radius_ratio=4.0), 900.0e6)
    quarter_wave_length = math.pi / (2.0 * parameters.phase_constant_rad_per_m)
    network = uniform_line_abcd(parameters, quarter_wave_length)
    load_ohm = 80.0
    input_impedance = terminated_input_impedance(network, Termination.impedance(load_ohm))

    assert input_impedance is not None
    assert input_impedance == pytest.approx(parameters.z0**2 / load_ohm, rel=2.0e-14)
    assert network.determinant == pytest.approx(1.0 + 0.0j, abs=2.0e-14)


def test_lossy_line_open_short_and_matched_inputs_match_closed_forms() -> None:
    parameters = coax_line_parameters(
        CoaxialCable(
            inner_radius_m=0.6e-3,
            outer_radius_m=2.8e-3,
            relative_permittivity=2.2,
            loss_tangent=0.025,
            series_resistance_ohm_per_m=0.8,
        ),
        1.4e9,
    )
    length_m = 0.083
    network = uniform_line_abcd(parameters, length_m)
    tanh_electrical_length = cmath.tanh(parameters.gamma * length_m)

    short_input = terminated_input_impedance(network, Termination.short())
    open_input = terminated_input_impedance(network, Termination.open())
    matched_input = terminated_input_impedance(network, Termination.matched(parameters.z0))

    assert short_input == pytest.approx(
        parameters.z0 * tanh_electrical_length, rel=3.0e-14
    )
    assert open_input == pytest.approx(
        parameters.z0 / tanh_electrical_length, rel=3.0e-14
    )
    assert matched_input == pytest.approx(parameters.z0, rel=3.0e-14)
    assert network.determinant == pytest.approx(1.0 + 0.0j, abs=3.0e-14)


def test_asymmetric_two_section_cascade_changes_when_direction_reverses() -> None:
    frequency_hz = 1.2e9
    first = uniform_line_abcd(
        coax_line_parameters(
            _lossless_cable(relative_permittivity=4.0, radius_ratio=2.2), frequency_hz
        ),
        0.019,
    )
    second = uniform_line_abcd(
        coax_line_parameters(
            _lossless_cable(relative_permittivity=1.4, radius_ratio=5.5), frequency_hz
        ),
        0.043,
    )
    forward = cascade_abcd(first, second)
    reverse = cascade_abcd(second, first)
    forward_reflection = input_reflection(forward, Termination.impedance(73.0), 50.0).gamma
    reverse_reflection = input_reflection(reverse, Termination.impedance(73.0), 50.0).gamma

    assert pytest.approx(forward.A) == first.A * second.A + first.B * second.C
    assert pytest.approx(forward.B) == first.A * second.B + first.B * second.D
    assert abs(forward_reflection - reverse_reflection) > 0.05


def test_phase_errors_wrap_across_plus_minus_180_degrees() -> None:
    expected = np.exp(1.0j * np.deg2rad(np.array([179.0, -179.0])))
    actual = np.exp(1.0j * np.deg2rad(np.array([-179.0, 179.0])))
    metrics = complex_s_error_metrics(actual, expected)

    assert circular_phase_error_deg(actual[0], expected[0]) == pytest.approx(2.0)
    assert metrics.phase_sample_count == 2
    assert metrics.phase_mae_deg == pytest.approx(2.0)
    assert metrics.phase_rmse_deg == pytest.approx(2.0)
    assert metrics.max_abs_phase_error_deg == pytest.approx(2.0)


def test_zero_magnitude_samples_are_excluded_only_from_phase_metrics() -> None:
    metrics = complex_s_error_metrics(
        np.array([0.0j, 1.0 + 0.0j]),
        np.array([0.0j, 1.0j]),
    )

    assert metrics.sample_count == 2
    assert metrics.phase_sample_count == 1
    assert metrics.phase_mae_deg == pytest.approx(90.0)
    assert metrics.complex_rmse == pytest.approx(1.0)


def test_tiny_nonzero_samples_preserve_phase_and_normalized_error() -> None:
    actual = 2.0e-300j
    expected = 1.0e-300 + 0.0j
    metrics = complex_s_error_metrics([actual], [expected])

    assert circular_phase_error_deg(actual, expected) == pytest.approx(90.0)
    assert metrics.phase_mae_deg == pytest.approx(90.0)
    assert metrics.complex_rmse == pytest.approx(math.sqrt(5.0) * 1.0e-300)
    assert metrics.normalized_complex_rmse == pytest.approx(math.sqrt(5.0))


def test_antipodal_phase_uses_a_canonical_positive_180_degrees() -> None:
    assert circular_phase_error_deg(-1.0 + 0.0j, 1.0 + 0.0j) == 180.0
    assert circular_phase_error_deg(1.0 + 0.0j, -1.0 + 0.0j) == 180.0
    metrics = complex_s_error_metrics([-1.0 + 0.0j], [1.0 + 0.0j])
    assert metrics.phase_mae_deg == 180.0


def test_results_are_strict_json_serializable_and_deterministic() -> None:
    def serialized_result() -> str:
        parameters = coax_line_parameters(_lossless_cable(), 1.0e9)
        network = uniform_line_abcd(parameters, 0.01)
        result = input_reflection(network, Termination.open(), parameters.z0.real)
        metrics = complex_s_error_metrics([result.gamma], [result.gamma])
        return json.dumps(
            {
                "parameters": parameters.to_dict(),
                "network": network.to_dict(),
                "reflection": result.to_dict(),
                "metrics": metrics.to_dict(),
            },
            allow_nan=False,
            separators=(",", ":"),
        )

    first = serialized_result()
    second = serialized_result()
    assert first == second


def test_zero_reflection_reports_undefined_phase_as_json_null() -> None:
    result = input_reflection(ABCDMatrix.identity(), Termination.matched(50.0), 50.0)

    assert result.gamma == 0.0j
    assert result.magnitude == 0.0
    assert result.phase_deg is None
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["phase_deg"] is None


def test_adjacent_representable_radii_do_not_cancel_the_log_ratio() -> None:
    inner_radius = 1.0e-3
    outer_radius = float(np.nextafter(inner_radius, math.inf))
    parameters = coax_line_parameters(CoaxialCable(inner_radius, outer_radius), 1.0e9)

    expected_log_ratio = math.log1p((outer_radius - inner_radius) / inner_radius)
    assert parameters.inductance_h_per_m == pytest.approx(
        VACUUM_PERMEABILITY_H_PER_M * expected_log_ratio / math.tau,
        rel=2.0e-15,
    )
    assert math.isfinite(parameters.capacitance_f_per_m)


def test_large_finite_metrics_are_stable_and_unrepresentable_errors_fail() -> None:
    metrics = complex_s_error_metrics([1.0e200 + 0.0j], [0.0j])
    assert metrics.complex_mae == pytest.approx(1.0e200)
    assert metrics.complex_rmse == pytest.approx(1.0e200)
    assert metrics.max_abs_complex_error == pytest.approx(1.0e200)
    assert metrics.normalized_complex_rmse is None

    with pytest.raises(ValueError, match="finite float range"):
        complex_s_error_metrics([1.0e308 + 0.0j], [-1.0e308 + 0.0j])


def test_large_finite_load_and_reference_do_not_overflow_fractional_transforms() -> None:
    largest_float = np.finfo(np.float64).max
    network = ABCDMatrix.from_complex(2.0, 3.0, 4.0, 5.0)
    input_impedance = terminated_input_impedance(
        network, Termination.impedance(largest_float)
    )
    matched_result = input_reflection(
        ABCDMatrix.identity(),
        Termination.impedance(largest_float),
        largest_float,
    )

    assert input_impedance == pytest.approx(0.5)
    assert matched_result.gamma == 0.0j
    assert matched_result.phase_deg is None


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"inner_radius_m": 0.0, "outer_radius_m": 2.0e-3}, ValueError),
        ({"inner_radius_m": 2.0e-3, "outer_radius_m": 2.0e-3}, ValueError),
        (
            {
                "inner_radius_m": 1.0e-3,
                "outer_radius_m": 2.0e-3,
                "relative_permittivity": float("inf"),
            },
            ValueError,
        ),
        (
            {
                "inner_radius_m": 1.0e-3,
                "outer_radius_m": 2.0e-3,
                "loss_tangent": -0.01,
            },
            ValueError,
        ),
    ],
)
def test_invalid_cable_inputs_are_rejected(kwargs: dict[str, float], error: type[Exception]) -> None:
    with pytest.raises(error):
        CoaxialCable(**kwargs)


def test_invalid_network_metric_and_termination_inputs_are_rejected() -> None:
    parameters = coax_line_parameters(_lossless_cable(), 1.0e9)
    with pytest.raises(ValueError, match="frequency_hz"):
        coax_line_parameters(_lossless_cable(), 0.0)
    with pytest.raises(ValueError, match="length_m"):
        uniform_line_abcd(parameters, -0.1)
    with pytest.raises(ValueError, match="non-finite ABCD"):
        uniform_line_abcd(parameters, np.finfo(np.float64).max)
    with pytest.raises(ValueError, match="at least one"):
        cascade_abcd()
    with pytest.raises(ValueError, match="passive"):
        Termination.impedance(-1.0)
    with pytest.raises(TypeError, match="string"):
        Termination(kind=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        ABCDMatrix.from_complex(complex(float("nan"), 0.0), 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="shapes differ"):
        complex_s_error_metrics([1.0j], [1.0j, 2.0j])
    with pytest.raises(ValueError, match="finite"):
        complex_s_error_metrics([complex(float("nan"), 0.0)], [1.0j])
    with pytest.raises(ValueError, match="must not be empty"):
        complex_s_error_metrics([], [])
    with pytest.raises(TypeError, match="real scalar"):
        complex_s_error_metrics([1.0j], [1.0j], phase_magnitude_floor=True)
    with pytest.raises(ValueError, match="undefined"):
        circular_phase_error_deg(0.0j, 1.0j)


def test_complex_value_rejects_nonfinite_parts() -> None:
    with pytest.raises(ValueError, match="finite"):
        ComplexValue(real=1.0, imag=float("inf"))
    with pytest.raises(TypeError, match="real scalar"):
        ComplexValue(real=True, imag=0.0)


def test_out_of_range_numeric_scalars_raise_domain_errors() -> None:
    with pytest.raises(ValueError, match="finite"):
        CoaxialCable(inner_radius_m=10**400, outer_radius_m=2 * 10**400)
    with pytest.raises(ValueError, match="finite"):
        ComplexValue.from_complex(10**400)
