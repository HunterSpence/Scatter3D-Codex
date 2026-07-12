"""Deterministic provenance helpers.

The project intentionally keeps provenance independent of wall-clock time.  A
caller may add an acquisition timestamp as an input parameter, but this module
will never add a changing value implicitly.  Consequently, identical inputs
and parameters produce byte-for-byte identical manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_SCHEMA = "scatter3d.provenance/v1"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_array(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    if result.dtype.hasobject:
        raise TypeError("object arrays cannot be hashed deterministically")
    if result.dtype.fields is not None:
        raise TypeError("structured arrays are not supported")

    # Canonicalize byte order so a manifest is portable between architectures.
    dtype = result.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        result = result.byteswap().view(dtype.newbyteorder("<"))
    elif dtype.byteorder == "=":
        result = result.view(dtype.newbyteorder("<"))
    return np.ascontiguousarray(result)


def sha256_array(array: np.ndarray) -> str:
    """Hash an array including its canonical dtype and shape.

    Including the header prevents equal byte streams with different shapes or
    element types from being treated as the same input.
    """

    canonical = _canonical_array(np.asarray(array))
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def array_record(array: np.ndarray) -> dict[str, Any]:
    """Return a compact, deterministic manifest record for an array."""

    canonical = _canonical_array(np.asarray(array))
    return {
        "kind": "ndarray",
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "sha256": sha256_array(canonical),
    }


def _normalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        # Avoid dataclasses.asdict(), which deep-copies large numerical arrays
        # before they can be reduced to compact content-hash records.
        return _normalize({item.name: getattr(value, item.name) for item in fields(value)})
    if isinstance(value, np.ndarray):
        return array_record(value)
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("manifest mapping keys must be strings")
            normalized[key] = _normalize(item)
        return normalized
    if isinstance(value, list | tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, bytes | bytearray | memoryview):
        data = bytes(value)
        return {
            "kind": "bytes",
            "length": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    if isinstance(value, complex):
        return {"real": _normalize(value.real), "imag": _normalize(value.imag)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest values must not contain NaN or infinity")
        # Convert negative zero, whose distinction is not useful in manifests.
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, str | int | bool):
        return value
    raise TypeError(f"unsupported manifest value type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a supported value to canonical UTF-8 JSON."""

    return (
        json.dumps(
            _normalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest, excluding a pre-existing self-hash field."""

    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def deterministic_manifest(
    *,
    inputs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
    software: Mapping[str, Any] | None = None,
    schema: str = MANIFEST_SCHEMA,
) -> dict[str, Any]:
    """Build a deterministic, self-hashing run manifest.

    Arrays are represented by shape, dtype, and content digest rather than
    expanded into JSON.  No timestamp or machine-specific path is inserted.
    """

    if not schema:
        raise ValueError("schema must be non-empty")
    manifest: dict[str, Any] = {
        "schema": schema,
        "inputs": _normalize(inputs),
        "parameters": _normalize(parameters),
        "artifacts": _normalize(artifacts or {}),
        "software": _normalize(software or {}),
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    """Atomically write a normalized manifest and return its digest.

    If ``manifest_sha256`` is present, it must already match the contents.
    Otherwise the field is added before writing.
    """

    normalized = _normalize(manifest)
    assert isinstance(normalized, dict)  # Mapping normalization guarantees this.
    expected = manifest_sha256(normalized)
    supplied = normalized.get("manifest_sha256")
    if supplied is not None and supplied != expected:
        raise ValueError("manifest_sha256 does not match manifest contents")
    normalized["manifest_sha256"] = expected

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(canonical_json_bytes(normalized))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return expected


__all__ = [
    "MANIFEST_SCHEMA",
    "array_record",
    "canonical_json_bytes",
    "deterministic_manifest",
    "manifest_sha256",
    "sha256_array",
    "sha256_file",
    "write_manifest",
]
