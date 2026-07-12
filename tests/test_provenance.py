from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scatter3d.provenance import (
    canonical_json_bytes,
    deterministic_manifest,
    manifest_sha256,
    sha256_array,
    sha256_file,
    write_manifest,
)


def test_array_hash_includes_shape_dtype_and_portable_byte_order() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype="<f8")
    assert sha256_array(values) == sha256_array(values.astype(">f8"))
    assert sha256_array(values) != sha256_array(values.reshape(2, 2))
    assert sha256_array(values) != sha256_array(values.astype(np.float32))
    with pytest.raises(TypeError, match="object arrays"):
        sha256_array(np.asarray([object()], dtype=object))


def test_manifest_is_deterministic_and_self_hashing() -> None:
    array = np.asarray([1 + 2j, 3 - 4j])
    first = deterministic_manifest(
        inputs={"data": array, "name": "synthetic"},
        parameters={"rank": 2, "nested": {"b": 2, "a": 1}},
        software={"scatter3d": "0.1.0"},
    )
    second = deterministic_manifest(
        inputs={"name": "synthetic", "data": array.copy()},
        parameters={"nested": {"a": 1, "b": 2}, "rank": 2},
        software={"scatter3d": "0.1.0"},
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["manifest_sha256"] == manifest_sha256(first)
    assert first["inputs"]["data"]["kind"] == "ndarray"
    assert "sha256" in first["inputs"]["data"]


def test_write_manifest_is_byte_stable_and_detects_tampering(tmp_path: Path) -> None:
    manifest = deterministic_manifest(
        inputs={"measurement_sha256": "a" * 64},
        parameters={"method": "fixed", "rank": 3},
    )
    path = tmp_path / "manifest.json"
    digest = write_manifest(path, manifest)
    first_bytes = path.read_bytes()
    assert sha256_file(path)
    assert digest == manifest["manifest_sha256"]
    write_manifest(path, dict(reversed(list(manifest.items()))))
    assert path.read_bytes() == first_bytes
    assert json.loads(path.read_text(encoding="utf-8"))["manifest_sha256"] == digest

    tampered = dict(manifest)
    tampered["parameters"] = {"method": "fixed", "rank": 2}
    with pytest.raises(ValueError, match="does not match"):
        write_manifest(tmp_path / "tampered.json", tampered)


def test_manifest_rejects_nondeterministic_numeric_values() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        deterministic_manifest(inputs={"bad": float("nan")}, parameters={})
    with pytest.raises(TypeError, match="mapping keys"):
        deterministic_manifest(inputs={1: "not allowed"}, parameters={})  # type: ignore[dict-item]
