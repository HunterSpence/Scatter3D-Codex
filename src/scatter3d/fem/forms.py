"""Nedelec Maxwell weak form with mutable frequency/material coefficients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import pi
from types import MappingProxyType
from typing import Any

import numpy as np

from .config import SPEED_OF_LIGHT, MaterialMap, MaxwellProblemConfig, PMLConfig
from .pml import cartesian_pml_inverse_tensor, cartesian_pml_tensor
from .ports import (
    MatchedTEMPortExcitation,
    PortDefinition,
    UncalibratedSurfaceCurrentExcitation,
    matched_tem_incident_coefficient,
    matched_tem_operator_coefficient,
    tangential_trace,
    validate_port_definitions,
)
from .tags import MeshTagContract


@dataclass(slots=True)
class MaxwellForms:
    mesh: Any
    cell_tags: Any
    facet_tags: Any
    tag_contract: MeshTagContract
    config: MaxwellProblemConfig
    pml_config: PMLConfig | None
    function_space: Any
    test_function: Any
    k0: Any
    epsilon_r: Any
    inverse_mu_r: Any
    preconditioner_absorption_shift: Any
    bilinear_form: Any
    preconditioner_bilinear_form: Any
    boundary_conditions: list[Any]
    ds: Any
    matched_ports: Mapping[str, PortDefinition]
    _materials: MaterialMap
    _frequency_hz: float

    @property
    def materials(self) -> MaterialMap:
        return self._materials

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    def set_materials(self, materials: MaterialMap) -> None:
        unknown = sorted(set(materials.regions) - set(self.tag_contract.volumes.all_tags))
        if unknown:
            raise ValueError(f"material map refers to undeclared volume tags {unknown}")
        self._materials = materials
        self.update_frequency(self._frequency_hz)

    def update_frequency(self, frequency_hz: float) -> None:
        from petsc4py import PETSc

        frequency = float(frequency_hz)
        if frequency <= 0 or not np.isfinite(frequency):
            raise ValueError("frequency_hz must be finite and positive")
        self.k0.value = PETSc.ScalarType(2.0 * pi * frequency / SPEED_OF_LIGHT)
        eps_array = self.epsilon_r.x.array
        inv_mu_array = self.inverse_mu_r.x.array
        default = self._materials.default
        eps_array[:] = PETSc.ScalarType(default.effective_relative_permittivity(frequency))
        inv_mu_array[:] = PETSc.ScalarType(1.0 / default.relative_permeability)
        dofmap = self.epsilon_r.function_space.dofmap
        for tag, material in self._materials.regions.items():
            for cell in self.cell_tags.find(tag):
                dof = dofmap.cell_dofs(int(cell))[0]
                eps_array[dof] = PETSc.ScalarType(
                    material.effective_relative_permittivity(frequency)
                )
                inv_mu_array[dof] = PETSc.ScalarType(1.0 / material.relative_permeability)
        self.epsilon_r.x.scatter_forward()
        self.inverse_mu_r.x.scatter_forward()
        self._frequency_hz = frequency

    def set_preconditioner_absorption_shift(self, value: float) -> None:
        """Set the dimensionless artificial loss used only by the P operator.

        With the declared ``exp(-i*omega*t)`` convention, a positive value
        adds ``+i*value`` to relative permittivity in the preconditioning
        form.  The physical form, material coefficients, and right-hand sides
        are not mutated.
        """

        from math import isfinite

        from petsc4py import PETSc

        shift = float(value)
        if not isfinite(shift) or shift < 0:
            raise ValueError(
                "preconditioner_absorption_shift must be finite and nonnegative"
            )
        self.preconditioner_absorption_shift.value = PETSc.ScalarType(shift)

    def matched_port_rhs_form(self, excitation: MatchedTEMPortExcitation) -> Any:
        """Compile the inward incident-mode RHS for one configured TEM port.

        For ``exp(-i omega t)``, integration by parts places
        ``-2 i k0 Z_vac Y_f <E_inc, v_t>`` on the right-hand side.  UFL
        ``inner`` conjugates the test field (its second operand).
        """

        import ufl
        from dolfinx import fem

        configured = self.matched_ports.get(excitation.definition.name)
        if configured is None:
            raise ValueError(
                f"port {excitation.definition.name!r} is not configured in this operator"
            )
        if configured != excitation.definition:
            raise ValueError(
                f"port {excitation.definition.name!r} excitation definition does not "
                "match the operator definition"
            )
        if excitation.mode.field.function_space is not self.function_space:
            raise ValueError(
                f"port {excitation.definition.name!r} mode must use forms.function_space"
            )
        normal = ufl.FacetNormal(self.mesh)
        incident_t = tangential_trace(excitation.mode.field, normal)
        test_t = tangential_trace(self.test_function, normal)
        linear = (
            matched_tem_incident_coefficient(self.k0, excitation)
            * ufl.inner(incident_t, test_t)
            * self.ds(excitation.definition.facet_tag)
        )
        return fem.form(linear)

    def uncalibrated_surface_current_rhs_form(
        self, excitation: UncalibratedSurfaceCurrentExcitation
    ) -> Any:
        """Compile an uncalibrated weak surface load on an observation boundary.

        Port and PEC tags are rejected: this generic load has no matched term or
        power-wave meaning and must remain impossible to confuse with a port.
        """

        import ufl
        from dolfinx import fem

        if excitation.facet_tag not in self.tag_contract.boundaries.observation_tags:
            raise ValueError(
                "uncalibrated surface-current loads are allowed only on declared "
                "observation tags, never on matched-port or PEC tags"
            )
        if excitation.surface_current.function_space is not self.function_space:
            raise ValueError("surface current must use forms.function_space")
        normal = ufl.FacetNormal(self.mesh)
        current_t = tangential_trace(excitation.surface_current, normal)
        test_t = tangential_trace(self.test_function, normal)
        return fem.form(
            excitation.amplitude
            * ufl.inner(current_t, test_t)
            * self.ds(excitation.facet_tag)
        )

    def rhs_form(
        self,
        excitation: MatchedTEMPortExcitation | UncalibratedSurfaceCurrentExcitation,
    ) -> Any:
        """Dispatch only between explicit matched and explicit uncalibrated loads."""

        if isinstance(excitation, MatchedTEMPortExcitation):
            return self.matched_port_rhs_form(excitation)
        if isinstance(excitation, UncalibratedSurfaceCurrentExcitation):
            return self.uncalibrated_surface_current_rhs_form(excitation)
        raise TypeError(
            "excitation must be MatchedTEMPortExcitation or "
            "UncalibratedSurfaceCurrentExcitation"
        )


def _piecewise_integral(integrand: Any, measure: Any, tags: tuple[int, ...]) -> Any:
    if not tags:
        return None
    result = integrand * measure(tags[0])
    for tag in tags[1:]:
        result += integrand * measure(tag)
    return result


def _validate_port_facets_present(
    mesh: Any, facet_tags: Any, ports: Sequence[PortDefinition]
) -> None:
    local_tags = set(int(value) for value in np.asarray(facet_tags.values).tolist())
    global_tags = set().union(*mesh.comm.allgather(local_tags))
    missing = sorted(port.facet_tag for port in ports if port.facet_tag not in global_tags)
    if missing:
        raise ValueError(f"matched port facet tags are absent from the mesh: {missing}")


def build_maxwell_forms(
    mesh: Any,
    cell_tags: Any,
    facet_tags: Any,
    tag_contract: MeshTagContract,
    materials: MaterialMap,
    config: MaxwellProblemConfig,
    pml_config: PMLConfig | None = None,
    *,
    matched_ports: Sequence[PortDefinition] | None = None,
    initial_frequency_hz: float = 1.0e9,
) -> MaxwellForms:
    """Compile one frequency-live operator for a tagged three-dimensional mesh.

    Every boundary declared as a port must have exactly one matched TEM
    definition.  Omitting definitions fails closed rather than leaving a
    port-labelled surface on the natural/PMC boundary.
    """

    import basix.ufl
    import ufl
    from dolfinx import fem
    from petsc4py import PETSc

    if mesh.topology.dim != 3:
        raise ValueError("the Maxwell FEM core currently requires a three-dimensional mesh")
    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        raise RuntimeError("Scatter3D Maxwell/PML solves require a complex PETSc build")
    if bool(tag_contract.volumes.pml_tags) != bool(pml_config):
        raise ValueError("PML volume tags and PMLConfig must either both be present or both absent")
    unknown = sorted(set(materials.regions) - set(tag_contract.volumes.all_tags))
    if unknown:
        raise ValueError(f"material map refers to undeclared volume tags {unknown}")
    port_definitions = validate_port_definitions(
        () if matched_ports is None else matched_ports,
        tag_contract.boundaries.ports,
    )
    _validate_port_facets_present(mesh, facet_tags, port_definitions)

    element = basix.ufl.element(
        "N1curl", mesh.basix_cell(), config.polynomial_degree
    )
    function_space = fem.functionspace(mesh, element)
    scalar_element = basix.ufl.element("DG", mesh.basix_cell(), 0)
    coefficient_space = fem.functionspace(mesh, scalar_element)
    epsilon_r = fem.Function(coefficient_space, name="effective_epsilon_r")
    inverse_mu_r = fem.Function(coefficient_space, name="inverse_mu_r")
    k0 = fem.Constant(mesh, PETSc.ScalarType(1.0))
    absorption_shift = fem.Constant(mesh, PETSc.ScalarType(0.0))

    trial = ufl.TrialFunction(function_space)
    test = ufl.TestFunction(function_space)
    curl_trial = ufl.curl(trial)
    curl_test = ufl.curl(test)
    metadata = {"quadrature_degree": config.quadrature_degree}
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags, metadata=metadata)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, metadata=metadata)

    physical_integrand = ufl.inner(inverse_mu_r * curl_trial, curl_test) - k0**2 * ufl.inner(
        epsilon_r * trial, test
    )
    shifted_epsilon_r = epsilon_r + PETSc.ScalarType(1j) * absorption_shift
    preconditioner_integrand = ufl.inner(
        inverse_mu_r * curl_trial, curl_test
    ) - k0**2 * ufl.inner(shifted_epsilon_r * trial, test)
    bilinear = _piecewise_integral(
        physical_integrand, dx, tag_contract.volumes.physical_tags
    )
    preconditioner_bilinear = _piecewise_integral(
        preconditioner_integrand, dx, tag_contract.volumes.physical_tags
    )
    if pml_config is not None:
        x = ufl.SpatialCoordinate(mesh)
        tensor = cartesian_pml_tensor(x, k0, pml_config)
        inverse_tensor = cartesian_pml_inverse_tensor(x, k0, pml_config)
        pml_integrand = ufl.inner(
            inverse_mu_r * ufl.dot(inverse_tensor, curl_trial), curl_test
        ) - k0**2 * ufl.inner(epsilon_r * ufl.dot(tensor, trial), test)
        preconditioner_pml_integrand = ufl.inner(
            inverse_mu_r * ufl.dot(inverse_tensor, curl_trial), curl_test
        ) - k0**2 * ufl.inner(
            shifted_epsilon_r * ufl.dot(tensor, trial), test
        )
        pml_part = _piecewise_integral(pml_integrand, dx, tag_contract.volumes.pml_tags)
        preconditioner_pml_part = _piecewise_integral(
            preconditioner_pml_integrand, dx, tag_contract.volumes.pml_tags
        )
        bilinear = pml_part if bilinear is None else bilinear + pml_part
        preconditioner_bilinear = (
            preconditioner_pml_part
            if preconditioner_bilinear is None
            else preconditioner_bilinear + preconditioner_pml_part
        )
    if bilinear is None:
        raise ValueError("tag contract does not contain any integrable volume tags")

    normal = ufl.FacetNormal(mesh)
    trial_t = tangential_trace(trial, normal)
    test_t = tangential_trace(test, normal)
    for port in port_definitions:
        # exp(-i omega t): the outgoing TEM branch gives
        # n x mu_r^-1 curl(E) = -i k0 Z_vac Y_f E_t.  Since the curl-curl
        # Green identity contributes +<n x mu_r^-1 curl(E), v_t>, the Robin
        # contribution to this sign convention is negative imaginary.
        bilinear += (
            matched_tem_operator_coefficient(k0, port)
            * ufl.inner(trial_t, test_t)
            * ds(port.facet_tag)
        )
        preconditioner_bilinear += (
            matched_tem_operator_coefficient(k0, port)
            * ufl.inner(trial_t, test_t)
            * ds(port.facet_tag)
        )

    boundary_conditions: list[Any] = []
    if tag_contract.boundaries.pec_tags:
        facets = np.unique(
            np.concatenate(
                [facet_tags.find(tag) for tag in tag_contract.boundaries.pec_tags]
            )
        )
        dofs = fem.locate_dofs_topological(function_space, mesh.topology.dim - 1, facets)
        zero = fem.Function(function_space, name="pec_zero")
        zero.x.array[:] = 0
        boundary_conditions.append(fem.dirichletbc(zero, dofs))

    forms = MaxwellForms(
        mesh=mesh,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        tag_contract=tag_contract,
        config=config,
        pml_config=pml_config,
        function_space=function_space,
        test_function=test,
        k0=k0,
        epsilon_r=epsilon_r,
        inverse_mu_r=inverse_mu_r,
        preconditioner_absorption_shift=absorption_shift,
        bilinear_form=fem.form(bilinear),
        preconditioner_bilinear_form=fem.form(preconditioner_bilinear),
        boundary_conditions=boundary_conditions,
        ds=ds,
        matched_ports=MappingProxyType(
            {port.name: port for port in port_definitions}
        ),
        _materials=materials,
        _frequency_hz=float(initial_frequency_hz),
    )
    forms.update_frequency(initial_frequency_hz)
    return forms
