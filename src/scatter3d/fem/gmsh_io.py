"""Narrow, validated Gmsh import boundary for the FEM package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoints import sha256_file
from .config import MaxwellProblemConfig
from .tags import MeshTagContract, validate_mesh_tags


@dataclass(frozen=True, slots=True)
class LoadedMesh:
    mesh: Any
    cell_tags: Any
    facet_tags: Any
    source_sha256: str


def load_gmsh_mesh(
    path: str | Path,
    comm: Any,
    contract: MeshTagContract,
    config: MaxwellProblemConfig,
    *,
    model_rank: int = 0,
    geometric_dimension: int = 3,
) -> LoadedMesh:
    """Collectively import a linear Gmsh ``.msh`` file and validate all tags.

    Curved coordinate elements are rejected rather than silently accepted.  The
    public configuration exposes only geometry order one until a curved import
    and convergence validation is added.
    """

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".msh" or not source.is_file():
        raise FileNotFoundError(f"expected an existing Gmsh .msh file: {source}")
    if config.geometry_order != 1:
        raise ValueError("load_gmsh_mesh currently supports only geometry_order=1")

    from dolfinx.io import gmsh as gmsh_io

    imported = gmsh_io.read_from_msh(
        str(source),
        comm,
        rank=int(model_rank),
        gdim=int(geometric_dimension),
    )
    if hasattr(imported, "mesh"):
        mesh = imported.mesh
        cell_tags = imported.cell_tags
        facet_tags = imported.facet_tags
    else:  # DOLFINx 0.9 compatibility; retained for readable error handling.
        mesh, cell_tags, facet_tags = imported
    if cell_tags is None or facet_tags is None:
        raise ValueError("Gmsh file must contain physical groups for cells and facets")

    coordinate_degree = getattr(getattr(mesh.geometry, "cmap", None), "degree", 1)
    if int(coordinate_degree) != 1:
        raise ValueError(
            f"imported coordinate element has degree {coordinate_degree}; only degree 1 "
            "geometry has a validated I/O path"
        )
    validate_mesh_tags(mesh, cell_tags, facet_tags, contract)
    digest = sha256_file(source) if comm.rank == model_rank else None
    digest = comm.bcast(digest, root=model_rank)
    return LoadedMesh(mesh, cell_tags, facet_tags, digest)
