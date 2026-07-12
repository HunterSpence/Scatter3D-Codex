"""Validated physical and numerical configuration for the Maxwell solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite, pi
from types import MappingProxyType
from typing import Any

EPSILON_0 = 8.854_187_812_8e-12
MU_0 = 1.256_637_062_12e-6
SPEED_OF_LIGHT = 1.0 / (EPSILON_0 * MU_0) ** 0.5


def _finite_complex(value: complex, name: str) -> complex:
    result = complex(value)
    if not (isfinite(result.real) and isfinite(result.imag)):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Material:
    """Linear, isotropic material using the ``exp(-i omega t)`` convention.

    Conductivity is kept separate from relative permittivity so that its
    frequency dependence cannot accidentally be frozen during a sweep.
    """

    relative_permittivity: complex
    relative_permeability: complex = 1.0 + 0.0j
    conductivity_s_per_m: float = 0.0
    name: str = "material"

    def __post_init__(self) -> None:
        epsr = _finite_complex(self.relative_permittivity, "relative_permittivity")
        mur = _finite_complex(self.relative_permeability, "relative_permeability")
        sigma = float(self.conductivity_s_per_m)
        if mur == 0:
            raise ValueError("relative_permeability must be nonzero")
        if not isfinite(sigma) or sigma < 0:
            raise ValueError("conductivity_s_per_m must be finite and nonnegative")
        if not self.name.strip():
            raise ValueError("material name must not be empty")
        object.__setattr__(self, "relative_permittivity", epsr)
        object.__setattr__(self, "relative_permeability", mur)
        object.__setattr__(self, "conductivity_s_per_m", sigma)

    def effective_relative_permittivity(self, frequency_hz: float) -> complex:
        """Return epsilon_r including conduction loss at ``frequency_hz``."""

        frequency = float(frequency_hz)
        if not isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        omega = 2.0 * pi * frequency
        return self.relative_permittivity + 1j * self.conductivity_s_per_m / (
            omega * EPSILON_0
        )

    def canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "relative_permittivity": [
                self.relative_permittivity.real,
                self.relative_permittivity.imag,
            ],
            "relative_permeability": [
                self.relative_permeability.real,
                self.relative_permeability.imag,
            ],
            "conductivity_s_per_m": self.conductivity_s_per_m,
        }


@dataclass(frozen=True, slots=True)
class MaterialMap:
    """Default material plus explicit overrides keyed by Gmsh volume tag."""

    default: Material
    regions: Mapping[int, Material] = field(default_factory=dict)

    def __post_init__(self) -> None:
        copied: dict[int, Material] = {}
        for raw_tag, material in self.regions.items():
            tag = int(raw_tag)
            if tag <= 0:
                raise ValueError("material volume tags must be positive integers")
            if not isinstance(material, Material):
                raise TypeError(f"material for tag {tag} is not a Material")
            copied[tag] = material
        object.__setattr__(self, "regions", MappingProxyType(copied))

    def for_tag(self, tag: int) -> Material:
        return self.regions.get(int(tag), self.default)

    def canonical(self) -> dict[str, Any]:
        return {
            "default": self.default.canonical(),
            "regions": {
                str(tag): material.canonical()
                for tag, material in sorted(self.regions.items())
            },
        }


@dataclass(frozen=True, slots=True)
class ExperimentMaterials:
    """Independent constitutive models for reference and DUT solves."""

    reference: MaterialMap
    dut: MaterialMap

    def __post_init__(self) -> None:
        if self.reference is self.dut:
            raise ValueError(
                "reference and DUT must be distinct MaterialMap objects; construct "
                "two maps even when intentionally identical"
            )

    def canonical(self) -> dict[str, Any]:
        return {
            "reference": self.reference.canonical(),
            "dut": self.dut.canonical(),
        }


@dataclass(frozen=True, slots=True)
class PMLConfig:
    """Axis-aligned Cartesian PML surrounding a physical bounding box.

    A zero thickness disables stretching on that axis.  The mesh itself must
    contain the requested layer; this object does not silently extend geometry.
    """

    physical_min_m: tuple[float, float, float]
    physical_max_m: tuple[float, float, float]
    thickness_m: tuple[float, float, float]
    polynomial_order: int = 3
    target_power_reflection: float = 1.0e-8

    def __post_init__(self) -> None:
        lo = tuple(float(v) for v in self.physical_min_m)
        hi = tuple(float(v) for v in self.physical_max_m)
        thickness = tuple(float(v) for v in self.thickness_m)
        if not (len(lo) == len(hi) == len(thickness) == 3):
            raise ValueError("PML bounds and thickness must contain three values")
        if not all(isfinite(v) for v in (*lo, *hi, *thickness)):
            raise ValueError("PML bounds and thickness must be finite")
        if any(a >= b for a, b in zip(lo, hi, strict=False)):
            raise ValueError("each physical_min_m coordinate must be below physical_max_m")
        if any(v < 0 for v in thickness) or not any(v > 0 for v in thickness):
            raise ValueError("PML thickness must be nonnegative and nonzero on at least one axis")
        if int(self.polynomial_order) < 1:
            raise ValueError("PML polynomial_order must be at least one")
        reflection = float(self.target_power_reflection)
        if not (0.0 < reflection < 1.0):
            raise ValueError("target_power_reflection must lie strictly between zero and one")
        object.__setattr__(self, "physical_min_m", lo)
        object.__setattr__(self, "physical_max_m", hi)
        object.__setattr__(self, "thickness_m", thickness)
        object.__setattr__(self, "polynomial_order", int(self.polynomial_order))
        object.__setattr__(self, "target_power_reflection", reflection)

    def canonical(self) -> dict[str, Any]:
        return {
            "physical_min_m": list(self.physical_min_m),
            "physical_max_m": list(self.physical_max_m),
            "thickness_m": list(self.thickness_m),
            "polynomial_order": self.polynomial_order,
            "target_power_reflection": self.target_power_reflection,
        }


@dataclass(frozen=True, slots=True)
class MaxwellProblemConfig:
    """Discretization settings whose defaults favor correctness over speed."""

    polynomial_degree: int = 1
    geometry_order: int = 1
    quadrature_degree: int | None = None

    def __post_init__(self) -> None:
        degree = int(self.polynomial_degree)
        if degree not in (1, 2, 3):
            raise ValueError("polynomial_degree must be 1, 2, or 3")
        # Curved Gmsh geometry is intentionally unavailable until its complete
        # import, interpolation, and convergence path is covered by validation.
        if int(self.geometry_order) != 1:
            raise ValueError(
                "only geometry_order=1 is currently validated; curved geometry "
                "must not be enabled by configuration alone"
            )
        quadrature = (
            max(2 * degree + 2, 6)
            if self.quadrature_degree is None
            else int(self.quadrature_degree)
        )
        if quadrature < 2 * degree:
            raise ValueError("quadrature_degree must be at least twice the field degree")
        object.__setattr__(self, "polynomial_degree", degree)
        object.__setattr__(self, "geometry_order", 1)
        object.__setattr__(self, "quadrature_degree", quadrature)

    def canonical(self) -> dict[str, Any]:
        return {
            "polynomial_degree": self.polynomial_degree,
            "geometry_order": self.geometry_order,
            "quadrature_degree": self.quadrature_degree,
            "time_convention": "exp(-i*omega*t)",
        }


@dataclass(frozen=True, slots=True)
class LinearSolverConfig:
    """PETSc KSP configuration for one operator and repeated port RHS vectors."""

    ksp_type: str = "preonly"
    pc_type: str = "lu"
    factor_solver_type: str | None = "mumps"
    solver_path: str = "direct"
    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 1.0e-12
    maximum_iterations: int = 2_000
    error_if_not_converged: bool = True
    petsc_options: Mapping[str, str | int | float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ksp_type.strip() or not self.pc_type.strip():
            raise ValueError("ksp_type and pc_type must not be empty")
        path = self.solver_path.lower()
        if path not in ("direct", "iterative"):
            raise ValueError("solver_path must be 'direct' or 'iterative'")
        factor_types = {"lu", "cholesky"}
        if path == "direct" and self.pc_type.lower() not in factor_types:
            raise ValueError("direct solver_path requires an LU or Cholesky preconditioner")
        if path == "iterative" and self.pc_type.lower() in factor_types:
            raise ValueError("iterative solver_path forbids LU and Cholesky factorization")
        rtol = float(self.relative_tolerance)
        atol = float(self.absolute_tolerance)
        maximum = int(self.maximum_iterations)
        if not (isfinite(rtol) and isfinite(atol) and rtol > 0 and atol >= 0):
            raise ValueError("solver tolerances must be finite with rtol > 0 and atol >= 0")
        if maximum < 1:
            raise ValueError("maximum_iterations must be positive")
        object.__setattr__(self, "relative_tolerance", rtol)
        object.__setattr__(self, "absolute_tolerance", atol)
        object.__setattr__(self, "maximum_iterations", maximum)
        object.__setattr__(self, "solver_path", path)
        object.__setattr__(self, "petsc_options", MappingProxyType(dict(self.petsc_options)))

    @property
    def is_direct(self) -> bool:
        return self.solver_path == "direct"

    @property
    def is_iterative(self) -> bool:
        return self.solver_path == "iterative"

    @classmethod
    def direct(cls, *, factor_solver_type: str = "mumps") -> LinearSolverConfig:
        """Small-problem reference path: one sparse factorization per frequency."""

        return cls(
            solver_path="direct",
            ksp_type="preonly",
            pc_type="lu",
            factor_solver_type=factor_solver_type,
        )

    @classmethod
    def iterative_maxwell(cls, **overrides: Any) -> LinearSolverConfig:
        """Distributed, non-factorizing baseline for large edge-element systems.

        FGMRES with overlapping additive Schwarz and local ILU(0) is a
        conservative portable baseline for complex, indefinite PML operators.
        It is not claimed to be mesh-independent.  Production runs should use
        the exposed PETSc options to evaluate platform-specific Maxwell
        preconditioners and must retain the true-residual gate.
        """

        values: dict[str, Any] = {
            "solver_path": "iterative",
            "ksp_type": "fgmres",
            "pc_type": "asm",
            "factor_solver_type": None,
            "relative_tolerance": 1.0e-8,
            "maximum_iterations": 1_000,
            "petsc_options": {
                "ksp_gmres_restart": 80,
                "pc_asm_overlap": 1,
                "sub_ksp_type": "preonly",
                "sub_pc_type": "ilu",
                "sub_pc_factor_levels": 0,
            },
        }
        values.update(overrides)
        return cls(**values)

    def canonical(self) -> dict[str, Any]:
        return {
            "ksp_type": self.ksp_type,
            "pc_type": self.pc_type,
            "factor_solver_type": self.factor_solver_type,
            "solver_path": self.solver_path,
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "maximum_iterations": self.maximum_iterations,
            "error_if_not_converged": self.error_if_not_converged,
            "petsc_options": dict(sorted(self.petsc_options.items())),
        }
