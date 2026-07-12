from __future__ import annotations

import json
import math

import pytest

from scatter3d.fem.ports import (
    VACUUM_IMPEDANCE_OHM,
    MatchedTEMPortExcitation,
    NormalizedPortMode,
    PortDefinition,
    UncalibratedSurfaceCurrentExcitation,
    matched_tem_incident_coefficient,
    matched_tem_operator_coefficient,
    validate_port_definitions,
)


def _port(**overrides: object) -> PortDefinition:
    values: dict[str, object] = {
        "name": "tx",
        "facet_tag": 10,
        "field_wave_impedance_ohm": 240.0 + 30.0j,
        "outgoing_propagation_index": 1.4 + 0.02j,
        "circuit_reference_impedance_ohm": 50.0,
        "target_forward_power_w": 1.0,
    }
    values.update(overrides)
    return PortDefinition(**values)  # type: ignore[arg-type]


def test_field_wave_impedance_is_required_and_keyword_only() -> None:
    with pytest.raises(TypeError, match="field_wave_impedance_ohm"):
        PortDefinition("tx", 10)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        PortDefinition("tx", 10, 377.0)  # type: ignore[misc]


def test_circuit_reference_never_changes_field_boundary_coefficients() -> None:
    first = _port(circuit_reference_impedance_ohm=50.0)
    second = _port(circuit_reference_impedance_ohm=75.0)
    different_field = _port(field_wave_impedance_ohm=480.0 + 60.0j)

    assert matched_tem_operator_coefficient(17.0, first) == pytest.approx(
        matched_tem_operator_coefficient(17.0, second)
    )
    assert matched_tem_operator_coefficient(17.0, different_field) == pytest.approx(
        0.5 * matched_tem_operator_coefficient(17.0, first)
    )


@pytest.mark.parametrize(
    "impedance",
    [
        0.0,
        50.0j,
        -50.0,
        -50.0 + 20.0j,
        complex(math.inf, 0.0),
        complex(math.nan, 0.0),
        complex(1.0e-320, 0.0),
    ],
)
def test_evanescent_nonpassive_zero_and_nonfinite_modes_are_rejected(
    impedance: complex,
) -> None:
    with pytest.raises(ValueError):
        _port(field_wave_impedance_ohm=impedance)


def test_passive_lossy_mode_uses_real_admittance_for_forward_power() -> None:
    port = _port(field_wave_impedance_ohm=40.0 + 30.0j)

    assert port.field_wave_admittance_siemens == pytest.approx(0.016 - 0.012j)
    assert port.forward_power_admittance_siemens == pytest.approx(0.016)
    assert port.forward_power_admittance_siemens != pytest.approx(
        1.0 / port.circuit_reference_impedance_ohm
    )


@pytest.mark.parametrize(
    "propagation_index",
    [0.0, 1.0j, -1.0 + 0.0j, 1.0 - 0.01j, complex(math.inf, 0.0)],
)
def test_evanescent_backward_active_and_nonfinite_propagation_is_rejected(
    propagation_index: complex,
) -> None:
    with pytest.raises(ValueError, match="outgoing_propagation_index"):
        _port(outgoing_propagation_index=propagation_index)


def test_operator_and_incident_rhs_have_exp_minus_iwt_sign_and_factor_two() -> None:
    port = _port(field_wave_impedance_ohm=VACUUM_IMPEDANCE_OHM)
    mode = NormalizedPortMode(port, object(), raw_forward_power_w=0.25, scale=2.0)
    excitation = MatchedTEMPortExcitation(mode, amplitude=0.5 - 0.25j)
    k0 = 9.0

    operator = matched_tem_operator_coefficient(k0, port)
    incident = matched_tem_incident_coefficient(k0, excitation)

    assert operator == pytest.approx(-9.0j)
    assert incident == pytest.approx(-18.0j * excitation.amplitude)
    assert operator.real == pytest.approx(0.0, abs=1.0e-15)
    assert operator.imag < 0.0


def test_incident_amplitude_has_explicit_power_meaning() -> None:
    port = _port(target_forward_power_w=2.5)
    mode = NormalizedPortMode(port, object(), raw_forward_power_w=0.4, scale=2.5)
    excitation = MatchedTEMPortExcitation(mode, amplitude=1.0 + 2.0j)

    assert excitation.definition is port
    assert excitation.incident_power_w == pytest.approx(12.5)
    with pytest.raises(ValueError, match="nonzero"):
        MatchedTEMPortExcitation(mode, amplitude=0.0j)
    with pytest.raises(ValueError, match="nonfinite incident power"):
        MatchedTEMPortExcitation(mode, amplitude=1.0e308 + 0.0j)


def test_port_canonical_record_separates_impedances_and_is_strict_json() -> None:
    port = _port(
        field_wave_impedance_ohm=240.0 + 30.0j,
        circuit_reference_impedance_ohm=50.0,
    )
    payload = port.canonical()

    assert payload["field_wave_impedance_ohm"] == [240.0, 30.0]
    assert payload["outgoing_propagation_index"] == [1.4, 0.02]
    assert payload["circuit_reference_impedance_ohm"] == 50.0
    assert payload["mode_model"] == "matched-single-tem"
    assert "reference_impedance_ohm" not in payload
    json.dumps(payload, allow_nan=False)


def test_mesh_contract_and_physical_definitions_must_match_one_to_one() -> None:
    tx = _port(name="tx", facet_tag=10)
    rx = _port(name="rx", facet_tag=11)

    assert validate_port_definitions((tx, rx), {"tx": 10, "rx": 11}) == (tx, rx)
    with pytest.raises(ValueError, match="missing matched definitions"):
        validate_port_definitions((tx,), {"tx": 10, "rx": 11})
    with pytest.raises(ValueError, match="undeclared ports"):
        validate_port_definitions((tx, rx), {"tx": 10})
    with pytest.raises(ValueError, match="name/tag mismatches"):
        validate_port_definitions((tx,), {"tx": 99})
    with pytest.raises(ValueError, match="names are duplicated"):
        validate_port_definitions((tx, _port(name="tx", facet_tag=12)), {"tx": 10})
    with pytest.raises(ValueError, match="facet tags are duplicated"):
        validate_port_definitions((tx, _port(name="rx", facet_tag=10)), {"tx": 10})


def test_generic_surface_current_is_explicitly_uncalibrated() -> None:
    sentinel = object()
    load = UncalibratedSurfaceCurrentExcitation(
        boundary_name="diagnostic-load",
        facet_tag=90,
        surface_current=sentinel,
        amplitude=2.0j,
    )

    assert load.surface_current is sentinel
    assert load.boundary_name == "diagnostic-load"
    assert not hasattr(load, "definition")
    assert not hasattr(load, "field_wave_impedance_ohm")


def test_names_tags_powers_and_amplitudes_fail_closed() -> None:
    with pytest.raises(ValueError, match="name"):
        _port(name="  ")
    with pytest.raises((TypeError, ValueError), match="facet_tag"):
        _port(facet_tag=2.5)
    with pytest.raises(ValueError, match="target_forward_power_w"):
        _port(target_forward_power_w=0.0)
    with pytest.raises(ValueError, match="circuit_reference_impedance_ohm"):
        _port(circuit_reference_impedance_ohm=-50.0)
    with pytest.raises(ValueError, match="finite"):
        UncalibratedSurfaceCurrentExcitation("load", 90, object(), complex(math.nan, 0))


@pytest.mark.heavy
def test_mode_normalization_uses_real_poynting_power_not_circuit_impedance() -> None:
    pytest.importorskip("dolfinx")
    import numpy as np
    from dolfinx import fem, mesh
    from mpi4py import MPI
    from petsc4py import PETSc

    from scatter3d.fem.ports import (
        normalize_port_mode,
        total_electric_modal_coefficient,
    )

    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    fdim = domain.topology.dim - 1
    left = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    tags = mesh.meshtags(
        domain,
        fdim,
        np.asarray(left, dtype=np.int32),
        np.full(left.size, 10, dtype=np.int32),
    )
    space = fem.functionspace(domain, ("N1curl", 1))
    raw_mode = fem.Function(space)
    raw_mode.interpolate(
        lambda x: np.vstack(
            (
                np.zeros(x.shape[1], dtype=PETSc.ScalarType),
                np.full(x.shape[1], 2.0, dtype=PETSc.ScalarType),
                np.zeros(x.shape[1], dtype=PETSc.ScalarType),
            )
        )
    )
    port = _port(
        field_wave_impedance_ohm=200.0,
        circuit_reference_impedance_ohm=50.0,
        target_forward_power_w=1.0,
    )

    normalized = normalize_port_mode(raw_mode, tags, port)

    # Unit-area face, |E_t|=2: P = 1/2 * (1/200) * 4 = 0.01 W.
    assert normalized.raw_forward_power_w == pytest.approx(0.01, rel=1.0e-11)
    assert normalized.scale == pytest.approx(10.0, rel=1.0e-11)
    expected_coefficient = 0.3 - 0.7j
    solution = fem.Function(space)
    solution.x.array[:] = expected_coefficient * normalized.field.x.array
    solution.x.scatter_forward()
    assert total_electric_modal_coefficient(
        solution, normalized, tags
    ) == pytest.approx(expected_coefficient, rel=1.0e-11, abs=1.0e-11)


@pytest.mark.heavy
def test_mode_normalization_rejects_absent_port_tag() -> None:
    pytest.importorskip("dolfinx")
    import numpy as np
    from dolfinx import fem, mesh
    from mpi4py import MPI

    from scatter3d.fem.ports import normalize_port_mode

    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    fdim = domain.topology.dim - 1
    left = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    tags = mesh.meshtags(
        domain,
        fdim,
        np.asarray(left, dtype=np.int32),
        np.full(left.size, 10, dtype=np.int32),
    )
    mode = fem.Function(fem.functionspace(domain, ("N1curl", 1)))

    with pytest.raises(ValueError, match="absent"):
        normalize_port_mode(mode, tags, _port(facet_tag=11))
