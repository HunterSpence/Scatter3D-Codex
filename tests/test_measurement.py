from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scatter3d.measurement as measurement_module
from scatter3d.measurement import (
    CSV_COLUMNS,
    ScatteringDataset,
    apply_reference_alignment,
    differential_scattering,
    estimate_reference_alignment,
    read_scattering_csv,
    write_scattering_csv,
)


def make_dataset(*, angles: int = 2, frequencies: int = 3) -> ScatteringDataset:
    shape = (angles, frequencies, 2, 2)
    values = np.empty(shape, dtype=np.complex128)
    for ai, fi, receiver, source in np.ndindex(shape):
        token = 1000 * ai + 100 * fi + 10 * receiver + source
        values[ai, fi, receiver, source] = token + 1j * (token + 0.25)
    return ScatteringDataset(
        s=values,
        frequencies_hz=np.linspace(5.0e9, 5.2e9, frequencies),
        angles_deg=np.arange(angles, dtype=float) * 15.0,
        port_labels=("P1", "P2"),
    )


def test_dataset_contract_and_vector_order_are_explicit() -> None:
    dataset = make_dataset()
    assert dataset.s.shape == (2, 3, 2, 2)
    assert dataset.vector()[1] == dataset.s[0, 0, 0, 1]
    assert dataset.vector()[2] == dataset.s[0, 0, 1, 0]
    assert not dataset.s.flags.writeable

    with pytest.raises(ValueError, match="strictly increasing"):
        ScatteringDataset(
            dataset.s,
            frequencies_hz=dataset.frequencies_hz[::-1],
            angles_deg=dataset.angles_deg,
            port_labels=dataset.port_labels,
        )
    with pytest.raises(ValueError, match="whitespace"):
        ScatteringDataset(
            dataset.s,
            frequencies_hz=dataset.frequencies_hz,
            angles_deg=dataset.angles_deg,
            port_labels=("P1", " P2"),
        )


def test_csv_roundtrip_is_lossless_and_hashed(tmp_path: Path) -> None:
    original = make_dataset()
    path = tmp_path / "measurement.csv"
    written_hashes = write_scattering_csv(path, original)
    restored = read_scattering_csv(path, expected_ports=("P1", "P2"))

    assert np.array_equal(restored.s, original.s)
    assert np.array_equal(restored.frequencies_hz, original.frequencies_hz)
    assert np.array_equal(restored.angles_deg, original.angles_deg)
    assert restored.port_labels == original.port_labels
    assert restored.fingerprint == original.fingerprint
    assert restored.metadata["csv_sha256"] == written_hashes["csv_sha256"]
    assert set(written_hashes) == {
        "csv_sha256",
        "csv_schema_sha256",
        "angle_axis_sha256",
        "frequency_axis_sha256",
        "port_axis_sha256",
        "row_order_sha256",
    }
    read_scattering_csv(path, expected_hashes=written_hashes)
    wrong_hashes = dict(written_hashes)
    wrong_hashes["csv_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="csv_sha256"):
        read_scattering_csv(path, expected_hashes=wrong_hashes)


def test_csv_rejects_schema_and_row_order_ambiguity(tmp_path: Path) -> None:
    dataset = make_dataset(angles=1, frequencies=1)
    path = tmp_path / "measurement.csv"
    write_scattering_csv(path, dataset)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out of order"):
        read_scattering_csv(path)

    bad_header = tmp_path / "bad-header.csv"
    bad_header.write_text(
        ",".join(reversed(CSV_COLUMNS)) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unexpected CSV schema"):
        read_scattering_csv(bad_header)


def test_csv_rejects_noncanonical_labels_and_huge_sparse_indices(tmp_path: Path) -> None:
    path = tmp_path / "measurement.csv"
    write_scattering_csv(path, make_dataset(angles=1, frequencies=1))
    lines = path.read_text(encoding="utf-8").splitlines()

    whitespace = list(lines)
    fields = whitespace[1].split(",")
    fields[5] = " P1"
    whitespace[1] = ",".join(fields)
    path.write_text("\n".join(whitespace) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="whitespace"):
        read_scattering_csv(path)

    huge_index = list(lines)
    fields = huge_index[1].split(",")
    fields[0] = "1000000000"
    huge_index[1] = ",".join(fields)
    path.write_text("\n".join(huge_index) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        read_scattering_csv(path)


def test_csv_rejects_a_file_that_changes_during_parse(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "measurement.csv"
    write_scattering_csv(path, make_dataset(angles=1, frequencies=1))
    digests = iter(("0" * 64, "1" * 64))
    monkeypatch.setattr(measurement_module, "sha256_file", lambda _path: next(digests))

    with pytest.raises(OSError, match="changed while"):
        read_scattering_csv(path)


def test_same_index_differential_is_default_and_alignment_is_off() -> None:
    reference = make_dataset()
    contrast = np.full(reference.shape, 0.125 - 0.25j)
    dut = reference.with_s(reference.s + contrast)

    result = differential_scattering(dut, reference)
    assert np.allclose(result.s, contrast)
    assert result.metadata["differential_mode"] == "same_index"
    assert result.metadata["reference_alignment_enabled"] is False
    with pytest.raises(ValueError, match="same_index"):
        differential_scattering(dut, reference, mode="reverse_reference")


def test_reference_alignment_is_fit_once_and_frozen_for_both_inputs() -> None:
    measured_reference = make_dataset()
    # Avoid a zero first sample in the synthetic reference.
    measured_reference = measured_reference.with_s(measured_reference.s + (2.0 + 0.5j))
    frequency_coordinate = np.linspace(-0.5, 0.5, measured_reference.shape[1])
    true_factor = np.ones(measured_reference.s.shape[1:], dtype=np.complex128)
    true_factor[:, 0, 1] = np.exp(1j * (0.3 + 0.7 * frequency_coordinate))
    true_factor[:, 1, 0] = np.exp(1j * (-0.2 + 0.4 * frequency_coordinate))
    model_reference = measured_reference.with_s(
        measured_reference.s * true_factor[None, :, :, :]
    )

    alignment = estimate_reference_alignment(measured_reference, model_reference)
    assert np.allclose(alignment.factors, true_factor, atol=1e-12)
    corrected_reference = apply_reference_alignment(measured_reference, alignment)
    assert np.allclose(corrected_reference.s, model_reference.s, atol=1e-12)

    contrast = np.full(measured_reference.shape, 0.01 + 0.02j)
    dut = measured_reference.with_s(measured_reference.s + contrast)
    differential = differential_scattering(dut, measured_reference, alignment=alignment)
    assert np.allclose(
        differential.s, contrast * true_factor[None, :, :, :], atol=1e-12
    )
    assert differential.metadata["reference_alignment_enabled"] is True


def test_alignment_and_differential_reject_axis_mismatch() -> None:
    reference = make_dataset()
    shifted = ScatteringDataset(
        reference.s,
        frequencies_hz=reference.frequencies_hz + 1.0,
        angles_deg=reference.angles_deg,
        port_labels=reference.port_labels,
    )
    with pytest.raises(ValueError, match="frequency axes differ"):
        differential_scattering(reference, shifted)
