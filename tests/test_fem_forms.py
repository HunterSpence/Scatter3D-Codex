from __future__ import annotations

import numpy as np
import pytest


def _tagged_cube(comm):
    from dolfinx import mesh

    domain = mesh.create_unit_cube(comm, 1, 1, 1)
    tdim = domain.topology.dim
    fdim = tdim - 1
    local_cells = domain.topology.index_map(tdim).size_local
    cell_indices = np.arange(local_cells, dtype=np.int32)
    cell_tags = mesh.meshtags(
        domain,
        tdim,
        cell_indices,
        np.ones(local_cells, dtype=np.int32),
    )
    left = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    facet_tags = mesh.meshtags(
        domain,
        fdim,
        np.asarray(left, dtype=np.int32),
        np.full(left.size, 10, dtype=np.int32),
    )
    return domain, cell_tags, facet_tags


def _contracts():
    from scatter3d.fem.tags import (
        BoundaryTagContract,
        MeshTagContract,
        VolumeTagContract,
    )

    volume = VolumeTagContract({"domain": 1})
    matched = MeshTagContract(volume, BoundaryTagContract(ports={"tx": 10}))
    natural = MeshTagContract(volume, BoundaryTagContract(observation_tags=(10,)))
    return matched, natural


def _port(*, tag: int = 10, impedance: complex = 200.0):
    from scatter3d.fem.ports import PortDefinition

    return PortDefinition(
        "tx",
        tag,
        field_wave_impedance_ohm=impedance,
        outgoing_propagation_index=1.0,
        circuit_reference_impedance_ohm=50.0,
        target_forward_power_w=1.0,
    )


def _material_and_config():
    from scatter3d.fem.config import Material, MaterialMap, MaxwellProblemConfig

    return MaterialMap(Material(1.0, name="vacuum")), MaxwellProblemConfig(1)


@pytest.mark.heavy
def test_declared_ports_fail_closed_without_exact_matched_definitions() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI

    from scatter3d.fem.forms import build_maxwell_forms

    domain, cell_tags, facet_tags = _tagged_cube(MPI.COMM_SELF)
    matched, _ = _contracts()
    materials, config = _material_and_config()

    with pytest.raises(ValueError, match="missing matched definitions"):
        build_maxwell_forms(
            domain,
            cell_tags,
            facet_tags,
            matched,
            materials,
            config,
        )
    with pytest.raises(ValueError, match="name/tag mismatches"):
        build_maxwell_forms(
            domain,
            cell_tags,
            facet_tags,
            matched,
            materials,
            config,
            matched_ports=(_port(tag=11),),
        )


@pytest.mark.heavy
def test_missing_matched_facet_tag_is_rejected_before_compilation() -> None:
    pytest.importorskip("dolfinx")
    from mpi4py import MPI

    from scatter3d.fem.forms import build_maxwell_forms
    from scatter3d.fem.tags import (
        BoundaryTagContract,
        MeshTagContract,
        VolumeTagContract,
    )

    domain, cell_tags, facet_tags = _tagged_cube(MPI.COMM_SELF)
    materials, config = _material_and_config()
    contract = MeshTagContract(
        VolumeTagContract({"domain": 1}),
        BoundaryTagContract(ports={"tx": 11}),
    )

    with pytest.raises(ValueError, match="absent from the mesh"):
        build_maxwell_forms(
            domain,
            cell_tags,
            facet_tags,
            contract,
            materials,
            config,
            matched_ports=(_port(tag=11),),
        )


def _dense_matrix(form, boundary_conditions):
    from dolfinx.fem import petsc as fem_petsc

    matrix = fem_petsc.assemble_matrix(form, bcs=boundary_conditions)
    matrix.assemble()
    dense = matrix.convert("dense")
    values = dense.getDenseArray().copy()
    dense.destroy()
    matrix.destroy()
    return values


@pytest.mark.heavy
def test_matched_operator_is_live_in_frequency_with_negative_i_sign() -> None:
    pytest.importorskip("dolfinx")
    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    from scatter3d.fem.forms import build_maxwell_forms
    from scatter3d.fem.ports import VACUUM_IMPEDANCE_OHM

    domain, cell_tags, facet_tags = _tagged_cube(MPI.COMM_SELF)
    matched_contract, natural_contract = _contracts()
    materials, config = _material_and_config()
    port = _port(impedance=VACUUM_IMPEDANCE_OHM)
    matched = build_maxwell_forms(
        domain,
        cell_tags,
        facet_tags,
        matched_contract,
        materials,
        config,
        matched_ports=(port,),
        initial_frequency_hz=1.0e8,
    )
    natural = build_maxwell_forms(
        domain,
        cell_tags,
        facet_tags,
        natural_contract,
        materials,
        config,
        initial_frequency_hz=1.0e8,
    )

    def boundary_matrix(frequency_hz: float) -> np.ndarray:
        matched.update_frequency(frequency_hz)
        natural.update_frequency(frequency_hz)
        return _dense_matrix(matched.bilinear_form, []) - _dense_matrix(
            natural.bilinear_form, []
        )

    first = boundary_matrix(1.0e8)
    second = boundary_matrix(2.0e8)
    assert np.linalg.norm(first) > 0.0
    assert second == pytest.approx(2.0 * first, rel=2.0e-11, abs=2.0e-11)

    trial = ufl.TrialFunction(matched.function_space)
    test = ufl.TestFunction(matched.function_space)
    normal = ufl.FacetNormal(domain)
    trial_t = ufl.cross(trial, normal)
    test_t = ufl.cross(test, normal)
    expected = fem.form(
        (-1.0j * matched.k0)
        * ufl.inner(trial_t, test_t)
        * matched.ds(port.facet_tag)
    )
    expected_values = _dense_matrix(expected, [])
    assert second == pytest.approx(expected_values, rel=2.0e-11, abs=2.0e-11)


@pytest.mark.heavy
def test_incident_mode_rhs_has_minus_two_i_and_conjugate_test_convention() -> None:
    pytest.importorskip("dolfinx")
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc as fem_petsc
    from mpi4py import MPI
    from petsc4py import PETSc

    from scatter3d.fem.forms import build_maxwell_forms
    from scatter3d.fem.ports import (
        VACUUM_IMPEDANCE_OHM,
        MatchedTEMPortExcitation,
        normalize_port_mode,
    )

    domain, cell_tags, facet_tags = _tagged_cube(MPI.COMM_SELF)
    matched_contract, _ = _contracts()
    materials, config = _material_and_config()
    port = _port(impedance=VACUUM_IMPEDANCE_OHM)
    forms = build_maxwell_forms(
        domain,
        cell_tags,
        facet_tags,
        matched_contract,
        materials,
        config,
        matched_ports=(port,),
        initial_frequency_hz=1.0e8,
    )
    raw_mode = fem.Function(forms.function_space)
    raw_mode.interpolate(
        lambda x: np.vstack(
            (
                np.zeros(x.shape[1], dtype=PETSc.ScalarType),
                np.ones(x.shape[1], dtype=PETSc.ScalarType),
                np.zeros(x.shape[1], dtype=PETSc.ScalarType),
            )
        )
    )
    mode = normalize_port_mode(raw_mode, facet_tags, port)
    excitation = MatchedTEMPortExcitation(mode, amplitude=0.25 + 0.5j)

    actual = fem_petsc.assemble_vector(forms.matched_port_rhs_form(excitation))
    actual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    actual_values = actual.getArray(readonly=True).copy()

    normal = ufl.FacetNormal(domain)
    mode_t = ufl.cross(mode.field, normal)
    test_t = ufl.cross(forms.test_function, normal)
    expected_form = fem.form(
        (-2.0j * forms.k0 * excitation.amplitude)
        * ufl.inner(mode_t, test_t)
        * forms.ds(port.facet_tag)
    )
    expected = fem_petsc.assemble_vector(expected_form)
    expected.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    expected_values = expected.getArray(readonly=True).copy()

    assert np.linalg.norm(actual_values) > 0.0
    assert actual_values == pytest.approx(expected_values, rel=2.0e-12, abs=2.0e-12)
    actual.destroy()
    expected.destroy()


@pytest.mark.heavy
def test_uncalibrated_surface_current_is_forbidden_on_matched_port_tag() -> None:
    pytest.importorskip("dolfinx")
    from dolfinx import fem
    from mpi4py import MPI

    from scatter3d.fem.forms import build_maxwell_forms
    from scatter3d.fem.ports import UncalibratedSurfaceCurrentExcitation

    domain, cell_tags, facet_tags = _tagged_cube(MPI.COMM_SELF)
    matched_contract, _ = _contracts()
    materials, config = _material_and_config()
    port = _port()
    forms = build_maxwell_forms(
        domain,
        cell_tags,
        facet_tags,
        matched_contract,
        materials,
        config,
        matched_ports=(port,),
    )
    current = fem.Function(forms.function_space)
    load = UncalibratedSurfaceCurrentExcitation("not-a-port", 10, current)

    with pytest.raises(ValueError, match="never on matched-port"):
        forms.uncalibrated_surface_current_rhs_form(load)
