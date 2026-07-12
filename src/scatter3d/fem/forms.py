"""Nedelec Maxwell weak form with mutable frequency/material coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np

from .config import MaterialMap, MaxwellProblemConfig, PMLConfig, SPEED_OF_LIGHT
from .pml import cartesian_pml_inverse_tensor, cartesian_pml_tensor
from .ports import PortExcitation
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
    bilinear_form: Any
    boundary_conditions: list[Any]
    ds: Any
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

    def rhs_form(self, excitation: PortExcitation) -> Any:
        import ufl
        from dolfinx import fem
        from petsc4py import PETSc

        expected = self.tag_contract.boundaries.ports.get(excitation.definition.name)
        if expected != excitation.definition.facet_tag:
            raise ValueError(
                f"port {excitation.definition.name!r} tag does not match mesh contract"
            )
        linear = (
            PETSc.ScalarType(excitation.amplitude)
            * ufl.inner(excitation.surface_current, self.test_function)
            * self.ds(excitation.definition.facet_tag)
        )
        return fem.form(linear)


def _piecewise_integral(integrand: Any, measure: Any, tags: tuple[int, ...]) -> Any:
    if not tags:
        return None
    result = integrand * measure(tags[0])
    for tag in tags[1:]:
        result += integrand * measure(tag)
    return result


def build_maxwell_forms(
    mesh: Any,
    cell_tags: Any,
    facet_tags: Any,
    tag_contract: MeshTagContract,
    materials: MaterialMap,
    config: MaxwellProblemConfig,
    pml_config: PMLConfig | None = None,
    *,
    initial_frequency_hz: float = 1.0e9,
) -> MaxwellForms:
    """Compile one frequency-live operator form for a tagged three-dimensional mesh."""

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

    element = basix.ufl.element(
        "N1curl", mesh.basix_cell(), config.polynomial_degree
    )
    function_space = fem.functionspace(mesh, element)
    scalar_element = basix.ufl.element("DG", mesh.basix_cell(), 0)
    coefficient_space = fem.functionspace(mesh, scalar_element)
    epsilon_r = fem.Function(coefficient_space, name="effective_epsilon_r")
    inverse_mu_r = fem.Function(coefficient_space, name="inverse_mu_r")
    k0 = fem.Constant(mesh, PETSc.ScalarType(1.0))

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
    bilinear = _piecewise_integral(
        physical_integrand, dx, tag_contract.volumes.physical_tags
    )
    if pml_config is not None:
        x = ufl.SpatialCoordinate(mesh)
        tensor = cartesian_pml_tensor(x, k0, pml_config)
        inverse_tensor = cartesian_pml_inverse_tensor(x, k0, pml_config)
        pml_integrand = ufl.inner(
            inverse_mu_r * ufl.dot(inverse_tensor, curl_trial), curl_test
        ) - k0**2 * ufl.inner(epsilon_r * ufl.dot(tensor, trial), test)
        pml_part = _piecewise_integral(pml_integrand, dx, tag_contract.volumes.pml_tags)
        bilinear = pml_part if bilinear is None else bilinear + pml_part
    if bilinear is None:
        raise ValueError("tag contract does not contain any integrable volume tags")

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
        bilinear_form=fem.form(bilinear),
        boundary_conditions=boundary_conditions,
        ds=ds,
        _materials=materials,
        _frequency_hz=float(initial_frequency_hz),
    )
    forms.update_frequency(initial_frequency_hz)
    return forms
