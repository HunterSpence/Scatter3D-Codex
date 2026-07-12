from __future__ import annotations

import json

import numpy as np
import pytest

import scatter3d.pipeline as pipeline_module
from scatter3d.cli import main
from scatter3d.pipeline import (
    SCHEMA_VERSION,
    SENSITIVITY_SCHEMA_VERSION,
    diagnose_measurement_bundle,
    load_measurement_bundle,
    reconstruct_from_bundle,
    schema_description,
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
    assert report.status == "PASSED"
    assert report.available_rank == voxels
    assert report.singular_values_sha256
    assert report.criterion_sha256
    assert report.relative_residual < 1e-12
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["estimate"], truth, rtol=1e-12, atol=1e-12)
        assert str(archive["bundle_sha256"].item()) == report.bundle_sha256
        assert archive["singular_values"].shape == (voxels,)
        assert archive["criterion_ranks"].shape == archive["criterion_values"].shape
        assert str(archive["status"].item()) == "PASSED"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reconstruct_from_bundle(
            bundle,
            sensitivity,
            output,
            method="fixed",
            rank=voxels,
        )


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
    assert report.noise_norm_basis == "whitened_expected_rms_sqrt_rows"
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


@pytest.mark.parametrize(
    ("key", "replacement", "message"),
    [
        ("reference_s", np.zeros((1, 1, 1, 1, 1), dtype=np.complex64), "complex128"),
        ("frequencies_hz", np.array([5.0e9], dtype=np.float32), "float64"),
        ("angles_deg", np.array([0], dtype=np.int64), "float64"),
        ("port_labels", np.array([b"P1"]), "Unicode"),
    ],
)
def test_measurement_npz_rejects_schema_dtype_coercion(
    tmp_path, key, replacement, message
):
    path = tmp_path / "measurement.npz"
    values = {
        "schema_version": np.array(SCHEMA_VERSION),
        "reference_s": np.zeros((1, 1, 1, 1, 1), dtype=np.complex128),
        "dut_s": np.zeros((1, 1, 1, 1, 1), dtype=np.complex128),
        "frequencies_hz": np.array([5.0e9], dtype=np.float64),
        "angles_deg": np.array([0.0], dtype=np.float64),
        "port_labels": np.array(["P1"]),
    }
    values[key] = replacement
    np.savez_compressed(path, **values)

    with pytest.raises(ValueError, match=message):
        load_measurement_bundle(path)


@pytest.mark.parametrize(
    ("matrix", "row_indices", "message"),
    [
        (np.eye(4), None, "complex128"),
        (np.eye(4, dtype=np.complex128), np.array([0.9, 1.9, 2.9, 3.9]), "int64"),
        (np.eye(4, dtype=np.complex128), np.arange(4, dtype=np.int32), "int64"),
        (np.eye(4, dtype=np.complex128), np.array([True, False, True, False]), "int64"),
    ],
)
def test_sensitivity_npz_rejects_matrix_and_row_identity_coercion(
    tmp_path, matrix, row_indices, message
):
    bundle = tmp_path / "measurement.npz"
    sensitivity = tmp_path / "sensitivity.npz"
    scattering = np.zeros((1, 1, 1, 2, 2), dtype=np.complex128)
    _write_bundle(
        bundle,
        reference_s=scattering,
        dut_s=scattering,
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0]),
        labels=("P1", "P2"),
    )
    payload = {
        "schema_version": np.array(SENSITIVITY_SCHEMA_VERSION),
        "A": matrix,
        "frequencies_hz": np.array([5.0e9]),
        "angles_deg": np.array([0.0]),
        "port_labels": np.array(["P1", "P2"]),
    }
    if row_indices is not None:
        payload["row_indices"] = row_indices
    np.savez_compressed(sensitivity, **payload)

    with pytest.raises(ValueError, match=message):
        reconstruct_from_bundle(
            bundle, sensitivity, tmp_path / "out.npz", method="fixed", rank=1
        )


def test_unmet_discrepancy_is_forensic_failed_artifact_and_nonzero_cli(tmp_path, capsys):
    bundle = tmp_path / "measurement.npz"
    sensitivity = tmp_path / "sensitivity.npz"
    reference = np.zeros((1, 1, 1, 2, 2), dtype=np.complex128)
    dut = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128).reshape(
        1, 1, 1, 2, 2
    )
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
        A=np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.complex128),
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0]),
        port_labels=np.array(["P1", "P2"]),
    )

    report = reconstruct_from_bundle(
        bundle,
        sensitivity,
        tmp_path / "library-out.npz",
        method="discrepancy",
        noise_norm=0.0,
        whitening="off",
    )
    assert report.status == "FAILED"
    assert report.selection_target_met is False
    with np.load(report.output_path, allow_pickle=False) as archive:
        assert str(archive["status"].item()) == "FAILED"
        assert int(archive["selection_target_met"]) == 0

    base_args = [
        "invert",
        str(bundle),
        str(sensitivity),
        str(tmp_path / "cli-out.npz"),
        "--method",
        "discrepancy",
        "--noise-norm",
        "0",
        "--whitening",
        "off",
    ]
    assert main(base_args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAILED"
    allowed = [*base_args]
    allowed[3] = str(tmp_path / "cli-allowed.npz")
    allowed.append("--allow-unmet-discrepancy")
    assert main(allowed) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "FAILED"


def test_diagnostics_expose_bad_frequency_channel_hidden_by_global_rms(tmp_path):
    bundle = tmp_path / "measurement.npz"
    repeat_offsets = np.array([-1.0, -0.25, 0.25, 1.0])[:, None]
    reference = np.zeros((4, 2, 1, 2, 2), dtype=np.complex128)
    mean = np.array([10.0, 0.1, 1.0, 1.0], dtype=np.complex128)
    noise_scale = np.array([0.01, 2.0, 0.1, 0.1], dtype=np.complex128)
    dut_flat = mean[None, :] + repeat_offsets * noise_scale[None, :]
    dut = np.broadcast_to(dut_flat[:, None, :], (4, 2, 4)).reshape(4, 2, 1, 2, 2)
    _write_bundle(
        bundle,
        reference_s=reference,
        dut_s=dut,
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0, 15.0]),
        labels=("P1", "P2"),
    )

    report = diagnose_measurement_bundle(bundle)
    by_pair = {
        (item.receiver_index, item.source_index): item
        for item in report.frequency_channels
    }
    assert report.signal_to_repeat_noise is not None
    assert by_pair[(0, 0)].signal_to_repeat_noise > 100.0
    assert by_pair[(0, 1)].signal_to_repeat_noise < 0.1


def test_bundle_hash_detects_mutation_and_schema_lists_reconstruction(tmp_path, monkeypatch):
    bundle = tmp_path / "measurement.npz"
    scattering = np.zeros((1, 1, 1, 1, 1), dtype=np.complex128)
    _write_bundle(
        bundle,
        reference_s=scattering,
        dut_s=scattering,
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0]),
        labels=("P1",),
    )
    digests = iter(("0" * 64, "1" * 64))
    monkeypatch.setattr(pipeline_module, "sha256_file", lambda _path: next(digests))
    with pytest.raises(OSError, match="changed while"):
        load_measurement_bundle(bundle)

    assert "reconstruction" in schema_description()


def test_sensitivity_hash_detects_mutation_during_parse(tmp_path, monkeypatch):
    bundle = tmp_path / "measurement.npz"
    sensitivity = tmp_path / "sensitivity.npz"
    scattering = np.zeros((1, 1, 1, 1, 1), dtype=np.complex128)
    _write_bundle(
        bundle,
        reference_s=scattering,
        dut_s=scattering,
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0]),
        labels=("P1",),
    )
    np.savez_compressed(
        sensitivity,
        schema_version=np.array(SENSITIVITY_SCHEMA_VERSION),
        A=np.ones((1, 1), dtype=np.complex128),
        frequencies_hz=np.array([5.0e9]),
        angles_deg=np.array([0.0]),
        port_labels=np.array(["P1"]),
    )
    digests = iter(("0" * 64, "0" * 64, "1" * 64, "2" * 64))
    monkeypatch.setattr(pipeline_module, "sha256_file", lambda _path: next(digests))

    with pytest.raises(OSError, match="sensitivity archive changed"):
        reconstruct_from_bundle(
            bundle,
            sensitivity,
            tmp_path / "out.npz",
            method="fixed",
            rank=1,
        )
