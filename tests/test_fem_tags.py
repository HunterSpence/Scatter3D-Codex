from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scatter3d.fem.tags import (
    BoundaryTagContract,
    MeshTagContract,
    TagContractError,
    VolumeTagContract,
    validate_mesh_tags,
)


class _SerialComm:
    def allgather(self, value):
        return [value]


def test_contract_rejects_reused_boundary_semantics() -> None:
    with pytest.raises(TagContractError, match="reused"):
        BoundaryTagContract(ports={"p1": 10}, pec_tags=(10,))


def test_mesh_tag_validation_is_strict_about_undeclared_volumes() -> None:
    mesh = SimpleNamespace(topology=SimpleNamespace(dim=3), comm=_SerialComm())
    cells = SimpleNamespace(dim=3, values=np.array([1, 99], dtype=np.int32))
    facets = SimpleNamespace(dim=2, values=np.array([10, 20], dtype=np.int32))
    contract = MeshTagContract(
        VolumeTagContract({"background": 1}),
        BoundaryTagContract(ports={"input": 10}, pec_tags=(20,)),
    )
    with pytest.raises(TagContractError, match=r"undeclared volume tags \[99\]"):
        validate_mesh_tags(mesh, cells, facets, contract)


def test_pml_names_resolve_to_declared_numeric_tags() -> None:
    volumes = VolumeTagContract(
        {"object": 1, "air": 2, "pml": 3}, pml_names=("pml",)
    )
    assert volumes.physical_tags == (1, 2)
    assert volumes.pml_tags == (3,)
