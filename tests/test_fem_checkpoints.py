from __future__ import annotations

import json
from dataclasses import replace

from scatter3d.fem.checkpoints import (
    CheckpointIdentity,
    checkpoint_fingerprint,
    verify_manifest,
    write_manifest,
)


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        mesh_sha256="a" * 64,
        frequencies_hz=(5.0e9, 5.1e9),
        tag_contract={"volumes": {"air": 1}},
        problem_config={"polynomial_degree": 3},
        material_model={"reference": {}, "dut": {}},
        solver_config={"solver_path": "iterative"},
        port_model=({"name": "p1", "facet_tag": 10},),
        mpi_size=8,
        implementation_version="test",
    )


def test_checkpoint_identity_changes_with_parallel_layout() -> None:
    identity = _identity()
    assert checkpoint_fingerprint(identity) != checkpoint_fingerprint(
        replace(identity, mpi_size=16)
    )


def test_checkpoint_identity_changes_with_dut_material() -> None:
    identity = _identity()
    changed = replace(identity, material_model={"reference": {}, "dut": {"epsr": 2.2}})
    assert checkpoint_fingerprint(identity) != checkpoint_fingerprint(changed)


def test_checkpoint_manifest_rejects_payload_tamper(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    identity = _identity()
    write_manifest(path, identity)
    assert verify_manifest(path, identity)
    assert not list(tmp_path.glob("*.tmp"))

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mpi_size"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not verify_manifest(path, identity)
