from __future__ import annotations

import json

import numpy as np
import pytest

from scatter3d.cli import main
from scatter3d.pipeline import (
    SCHEMA_VERSION,
    SENSITIVITY_SCHEMA_VERSION,
    diagnose_measurement_bundle,
    reconstruct_from_bundle,
    validate_measurement_bundle,
)


def _write_bundle(path, *, reference_s, dut_s, frequencies_hz, angles_deg, labels):
    np.savez_compressed(
        path,
        schema_version=np.array(SCHEMA_VERSION),
        reference_s=reference_s,
        dut_s=dut_s,
        frequencies_hz=frequencies_hz,
        angles_deg=angles_deg,
        port_labels=np.asarray(labels),
    )


def test_validate_and_diagnose_repeat_bundle(tmp_path):
    rng = np.random.default_rng(12)
    shape = (3, 2, 3, 2, 2)
    reference_mean = rng.normal(size=shape[1:]) + 1j * rng.normal(size=shape[1:])
    target_shift = np.full(shape[1:], 0.2 + 0.1j)
    noise = 1e-3 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    reference = reference_mean[np.newaxis, ...] + noise
    dut = reference_mean[np.newaxis, ...] + target_shift[np.newaxis, ...] + 0.8 * noise
    bundle = tmp_path / "measurement.npz"
    _write_bundle(
        bundle,
        reference_s=reference,
        dut_s=dut,
        frequencies_hz=np.array([5.0e9, 5.5e9, 6.0e9]),
        angles_deg=np.array([0.0, 15.0]),
        labels=("P1", "P2"),
    )

    validation = validate_measurement_bundle(bundle)
    assert validation.reference_repeats == 3
    assert validation.rows == 2 * 3 * 2 * 2

    report = diagnose_measurement_bundle(bundle)
    assert report.repeat_noise_rms is not None
    assert report.signal_to_repeat_noise is not None
    assert report.signal_to_repeat_noise > 10.0
    assert report.unique_reference_angle_traces == 2
    assert report.unique_dut_angle_traces == 2


def test_reconstruct_bundle_checks_coordinates_and_recovers_complex_truth(tmp_path):
    rng = np.random.default_rng(5)
    angles = np.array([0.0])
    frequencies = np.array([5.0e9, 5.5e9])
    labels = np.array(["P1", "P2"])
    rows, voxels = 8, 3
    A = rng.normal(size=(rows, voxels)) + 1j * rng.normal(size=(rows, voxels))
    truth = np.array([0.4 + 0.1j, -0.2 + 0.05j, 0.7 - 0.3j])
    b = A @ truth
    reference = np.zeros((1, 1, 2, 2, 2), dtype=np.complex128)
    dut = b.reshape(1, 1, 2, 2, 2)

    bundle = tmp_path / "measurement.npz"
    sensitivity = tmp_path / "sensitivity.npz"
    output = tmp_path / "reconstruction.npz"
    _write_bundle(
        bundle,
        reference_s=reference,
        dut_s=dut,
        frequencies_hz=frequencies,
        angles_deg=angles,
        labels=labels,
    )
    np.savez_compressed(
        sensitivity,
        schema_version=np.array(SENSITIVITY_SCHEMA_VERSION),
        A=A,
        frequencies_hz=frequencies,
        angles_deg=angles,
        port_labels=labels,
    )

    report = reconstruct_from_bundle(
        bundle,
        sensitivity,
        output,
        method="fixed",
        rank=voxels,
    )
    assert report.selected_rank == voxels
    assert report.relative_residual < 1e-12
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["estimate"], truth, rtol=1e-12, atol=1e-12)
        assert str(archive["bundle_sha256"].item()) == report.bundle_sha256


def test_reconstruct_rejects_coordinate_mismatch(tmp_path):
    bundle = tmp_path / "measurement.npz"
    sensitivity = tmp_path / "sensitivity.npz"
    reference = np.zeros((1, 1, 1, 2, 2), dtype=np.complex128)
    _write_bundle(
        bundle,
        reference_s=reference,
        dut_s=reference,
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0]),
        labels=("P1", "P2"),
    )
    np.savez_compressed(
        sensitivity,
        schema_version=np.array(SENSITIVITY_SCHEMA_VERSION),
        A=np.eye(4, dtype=np.complex128),
        frequencies_hz=np.array([5.1e9]),
        angles_deg=np.array([0.0]),
        port_labels=np.array(["P1", "P2"]),
    )
    with pytest.raises(ValueError, match="frequency grids differ"):
        reconstruct_from_bundle(bundle, sensitivity, tmp_path / "out.npz", method="fixed", rank=4)


def test_reconstruct_auto_whitens_paired_repeat_mean(tmp_path):
    bundle = tmp_path / "measurement.npz"
    sensitivity = tmp_path / "sensitivity.npz"
    output = tmp_path / "out.npz"
    truth = np.array([0.2 + 0.1j, -0.3 + 0.2j, 0.4 - 0.1j, 0.1 + 0.3j])
    offsets = np.array([-0.01 - 0.02j, 0.01 + 0.02j, 0.0 + 0.0j])[:, None]
    reference = np.zeros((3, 1, 1, 2, 2), dtype=np.complex128)
    dut = (truth[None, :] + offsets).reshape(3, 1, 1, 2, 2)
    _write_bundle(
        bundle,
        reference_s=reference,
        dut_s=dut,
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0]),
        labels=("P1", "P2"),
    )
    np.savez_compressed(
        sensitivity,
        schema_version=np.array(SENSITIVITY_SCHEMA_VERSION),
        A=np.eye(4, dtype=np.complex128),
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0]),
        port_labels=np.array(["P1", "P2"]),
    )

    report = reconstruct_from_bundle(bundle, sensitivity, output, method="fixed", rank=4)
    assert report.whitening_used
    assert report.paired_repeats == 3
    assert report.noise_model_sha256
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["estimate"], truth, atol=1e-12)
        assert bool(archive["whitening_used"])
        assert archive["noise_standard_deviation"].shape == (4,)


def test_discrepancy_whitening_preserves_a_null_as_rank_zero(tmp_path):
    bundle = tmp_path / "measurement.npz"
    sensitivity = tmp_path / "sensitivity.npz"
    output = tmp_path / "out.npz"
    offsets = np.asarray([-0.02 - 0.01j, 0.0 + 0.0j, 0.02 + 0.01j])[:, None]
    reference = np.zeros((3, 1, 1, 2, 2), dtype=np.complex128)
    dut = np.broadcast_to(offsets, (3, 4)).reshape(3, 1, 1, 2, 2).copy()
    _write_bundle(
        bundle,
        reference_s=reference,
        dut_s=dut,
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0]),
        labels=("P1", "P2"),
    )
    np.savez_compressed(
        sensitivity,
        schema_version=np.array(SENSITIVITY_SCHEMA_VERSION),
        A=np.eye(4, dtype=np.complex128),
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0]),
        port_labels=np.array(["P1", "P2"]),
    )

    report = reconstruct_from_bundle(
        bundle,
        sensitivity,
        output,
        method="discrepancy",
    )

    assert report.whitening_used
    assert report.selected_rank == 0
    assert report.noise_norm_used == pytest.approx(2.0)
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["estimate"], np.zeros(4, dtype=np.complex128)
        )


def test_cli_validate_writes_machine_readable_report(tmp_path, capsys):
    bundle = tmp_path / "measurement.npz"
    report_path = tmp_path / "validation.json"
    scattering = np.zeros((1, 1, 1, 1, 1), dtype=np.complex128)
    _write_bundle(
        bundle,
        reference_s=scattering,
        dut_s=scattering,
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0]),
        labels=("P1",),
    )
    assert main(["validate", str(bundle), "--json", str(report_path)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout == on_disk
    assert stdout["schema_version"] == SCHEMA_VERSION


def test_cli_returns_two_for_invalid_input(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "missing.npz")]) == 2
    assert "error" in capsys.readouterr().err
