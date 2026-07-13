"""Explicit contracts between Gmsh physical groups and solver semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


class TagContractError(ValueError):
    """Raised when imported mesh tags do not satisfy the declared contract."""


def _positive_unique(values: list[int], label: str) -> None:
    if any(value <= 0 for value in values):
        raise TagContractError(f"{label} tags must be positive integers")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise TagContractError(f"{label} tags are reused: {duplicates}")


@dataclass(frozen=True, slots=True)
class VolumeTagContract:
    """Names every volume and identifies the subset transformed as PML."""

    volumes: Mapping[str, int]
    pml_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        copied = {str(name): int(tag) for name, tag in self.volumes.items()}
        if not copied or any(not name.strip() for name in copied):
            raise TagContractError("at least one nonempty volume name is required")
        _positive_unique(list(copied.values()), "volume")
        names = tuple(str(name) for name in self.pml_names)
        missing = sorted(set(names) - set(copied))
        if missing:
            raise TagContractError(f"unknown PML volume names: {missing}")
        if len(set(names)) != len(names):
            raise TagContractError("PML volume names must be unique")
        object.__setattr__(self, "volumes", MappingProxyType(copied))
        object.__setattr__(self, "pml_names", names)

    @property
    def all_tags(self) -> tuple[int, ...]:
        return tuple(self.volumes.values())

    @property
    def pml_tags(self) -> tuple[int, ...]:
        return tuple(self.volumes[name] for name in self.pml_names)

    @property
    def physical_tags(self) -> tuple[int, ...]:
        pml = set(self.pml_tags)
        return tuple(tag for tag in self.all_tags if tag not in pml)

    def canonical(self) -> dict[str, Any]:
        return {
            "volumes": dict(sorted(self.volumes.items())),
            "pml_names": list(self.pml_names),
        }


@dataclass(frozen=True, slots=True)
class BoundaryTagContract:
    """Disjoint boundary groups for PEC walls, ports, and diagnostics."""

    ports: Mapping[str, int] = field(default_factory=dict)
    pec_tags: tuple[int, ...] = ()
    observation_tags: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        ports = {str(name): int(tag) for name, tag in self.ports.items()}
        if any(not name.strip() for name in ports):
            raise TagContractError("port names must not be empty")
        pec = tuple(int(tag) for tag in self.pec_tags)
        observation = tuple(int(tag) for tag in self.observation_tags)
        all_tags = list(ports.values()) + list(pec) + list(observation)
        _positive_unique(all_tags, "boundary")
        object.__setattr__(self, "ports", MappingProxyType(ports))
        object.__setattr__(self, "pec_tags", pec)
        object.__setattr__(self, "observation_tags", observation)

    @property
    def all_tags(self) -> tuple[int, ...]:
        return tuple(self.ports.values()) + self.pec_tags + self.observation_tags

    def canonical(self) -> dict[str, Any]:
        return {
            "ports": dict(sorted(self.ports.items())),
            "pec_tags": list(self.pec_tags),
            "observation_tags": list(self.observation_tags),
        }


@dataclass(frozen=True, slots=True)
class MeshTagContract:
    volumes: VolumeTagContract
    boundaries: BoundaryTagContract = field(
        default_factory=lambda: BoundaryTagContract(ports={})
    )

    def canonical(self) -> dict[str, Any]:
        return {
            "volumes": self.volumes.canonical(),
            "boundaries": self.boundaries.canonical(),
        }


def validate_mesh_tags(mesh: Any, cell_tags: Any, facet_tags: Any, contract: MeshTagContract) -> None:
    """Fail early when dimensions or required physical groups are incorrect."""

    tdim = int(mesh.topology.dim)
    if int(cell_tags.dim) != tdim:
        raise TagContractError(
            f"cell tags have dimension {cell_tags.dim}; expected mesh dimension {tdim}"
        )
    if int(facet_tags.dim) != tdim - 1:
        raise TagContractError(
            f"facet tags have dimension {facet_tags.dim}; expected {tdim - 1}"
        )
    local_cells = set(int(v) for v in np.asarray(cell_tags.values).tolist())
    local_facets = set(int(v) for v in np.asarray(facet_tags.values).tolist())
    comm = mesh.comm
    global_cells = set().union(*comm.allgather(local_cells))
    global_facets = set().union(*comm.allgather(local_facets))
    missing_cells = sorted(set(contract.volumes.all_tags) - global_cells)
    missing_facets = sorted(set(contract.boundaries.all_tags) - global_facets)
    if missing_cells or missing_facets:
        pieces: list[str] = []
        if missing_cells:
            pieces.append(f"missing volume tags {missing_cells}")
        if missing_facets:
            pieces.append(f"missing boundary tags {missing_facets}")
        raise TagContractError("; ".join(pieces))

    unexpected_cells = sorted(global_cells - set(contract.volumes.all_tags))
    if unexpected_cells:
        raise TagContractError(
            "mesh contains undeclared volume tags "
            f"{unexpected_cells}; declare their material/PML semantics explicitly"
        )
