"""Content-addressed checkpoint identity and manifest helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Every input that can change a FEM solution or its parallel layout."""

    mesh_sha256: str
    frequencies_hz: tuple[float, ...]
    tag_contract: Mapping[str, Any]
    problem_config: Mapping[str, Any]
    material_model: Mapping[str, Any]
    solver_config: Mapping[str, Any]
    port_model: Sequence[Mapping[str, Any]]
    mpi_size: int
    implementation_version: str
    schema: str = "scatter3d.fem.checkpoint/v1"

    def __post_init__(self) -> None:
        digest = self.mesh_sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("mesh_sha256 must be a 64-character hexadecimal digest")
        frequencies = tuple(float(v) for v in self.frequencies_hz)
        if not frequencies or any(v <= 0 for v in frequencies):
            raise ValueError("frequencies_hz must contain positive values")
        if any(b <= a for a, b in pairwise(frequencies)):
            raise ValueError("frequencies_hz must be strictly increasing")
        if int(self.mpi_size) < 1:
            raise ValueError("mpi_size must be positive")
        if not self.implementation_version.strip():
            raise ValueError("implementation_version must not be empty")
        object.__setattr__(self, "mesh_sha256", digest)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "mpi_size", int(self.mpi_size))

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mesh_sha256": self.mesh_sha256,
            "frequencies_hz": list(self.frequencies_hz),
            "tag_contract": self.tag_contract,
            "problem_config": self.problem_config,
            "material_model": self.material_model,
            "solver_config": self.solver_config,
            "port_model": list(self.port_model),
            "mpi_size": self.mpi_size,
            "implementation_version": self.implementation_version,
        }


def checkpoint_fingerprint(identity: CheckpointIdentity) -> str:
    return sha256(_canonical_json(identity.canonical()).encode("utf-8")).hexdigest()


def write_manifest(path: str | Path, identity: CheckpointIdentity) -> None:
    """Atomically write a human-readable checkpoint manifest."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = identity.canonical() | {"fingerprint": checkpoint_fingerprint(identity)}
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_manifest(path: str | Path, identity: CheckpointIdentity) -> bool:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = identity.canonical() | {"fingerprint": checkpoint_fingerprint(identity)}
    return payload == expected
