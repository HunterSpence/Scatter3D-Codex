"""Strict measurement data contracts for microwave scattering data.

There is exactly one in-memory convention in Scatter3D-Codex:

``S[angle, frequency, receiver, source]``.

The long-form CSV format deliberately contains explicit integer axis indices.
Readers therefore validate ordering instead of guessing whether an external
file is receiver-major, source-major, or ordered like a particular instrument.
"""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .provenance import canonical_json_bytes, sha256_array, sha256_file

CSV_COLUMNS = (
    "angle_index",
    "angle_deg",
    "frequency_index",
    "frequency_hz",
    "receiver_index",
    "receiver_port",
    "source_index",
    "source_port",
    "s_real",
    "s_imag",
)
CSV_SCHEMA = "scatter3d.scattering-long/v1"


def _read_only(array: np.ndarray, *, dtype: np.dtype[Any] | type) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    result.setflags(write=False)
    return result


def _metadata_copy(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    # A shallow immutable wrapper is intentional: metadata is descriptive and
    # never participates in the numerical data fingerprint.
    return MappingProxyType(dict(metadata))


@dataclass(frozen=True, slots=True)
class ScatteringDataset:
    """A validated scattering measurement in canonical axis order."""

    s: np.ndarray
    frequencies_hz: np.ndarray
    angles_deg: np.ndarray
    port_labels: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        s = _read_only(self.s, dtype=np.complex128)
        frequencies = _read_only(self.frequencies_hz, dtype=np.float64)
        angles = _read_only(self.angles_deg, dtype=np.float64)
        labels = tuple(str(label) for label in self.port_labels)

        if s.ndim != 4:
            raise ValueError(
                "s must have shape [angle, frequency, receiver, source]"
            )
        if not all(size > 0 for size in s.shape):
            raise ValueError("all scattering axes must be non-empty")
        if s.shape[2] != s.shape[3]:
            raise ValueError("receiver and source axes must have equal port counts")
        if frequencies.ndim != 1 or frequencies.size != s.shape[1]:
            raise ValueError("frequencies_hz must match the frequency axis")
        if angles.ndim != 1 or angles.size != s.shape[0]:
            raise ValueError("angles_deg must match the angle axis")
        if len(labels) != s.shape[2]:
            raise ValueError("port_labels must match the receiver/source axes")
        if any(label != label.strip() for label in labels):
            raise ValueError("port labels must not contain leading or trailing whitespace")
        if any(not label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("port labels must be non-empty and unique")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
            raise ValueError("frequencies_hz must be finite and positive")
        if not np.all(np.diff(frequencies) > 0.0):
            raise ValueError("frequencies_hz must be strictly increasing")
        if not np.all(np.isfinite(angles)):
            raise ValueError("angles_deg must be finite")
        if angles.size > 1 and not np.all(np.diff(angles) > 0.0):
            raise ValueError("angles_deg must be strictly increasing")
        if not np.all(np.isfinite(s.real)) or not np.all(np.isfinite(s.imag)):
            raise ValueError("scattering values must be finite")

        object.__setattr__(self, "s", s)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "angles_deg", angles)
        object.__setattr__(self, "port_labels", labels)
        object.__setattr__(self, "metadata", _metadata_copy(self.metadata))

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.s.shape

    @property
    def number_of_ports(self) -> int:
        return self.s.shape[2]

    @property
    def number_of_observations(self) -> int:
        return self.s.size

    def vector(self) -> np.ndarray:
        """Return a copy flattened in angle/frequency/receiver/source order."""

        return np.array(self.s.reshape(-1, order="C"), copy=True)

    @property
    def coordinate_hashes(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "angles_sha256": sha256_array(self.angles_deg),
                "frequencies_sha256": sha256_array(self.frequencies_hz),
                "ports_sha256": hashlib.sha256(
                    canonical_json_bytes(list(self.port_labels))
                ).hexdigest(),
            }
        )

    @property
    def fingerprint(self) -> str:
        """Hash numerical data and axes, excluding descriptive metadata."""

        payload = {
            "schema": "scatter3d.dataset/v1",
            "s_sha256": sha256_array(self.s),
            **self.coordinate_hashes,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def with_s(
        self, s: np.ndarray, *, metadata_updates: Mapping[str, Any] | None = None
    ) -> ScatteringDataset:
        metadata = dict(self.metadata)
        metadata.update(metadata_updates or {})
        return ScatteringDataset(
            s=s,
            frequencies_hz=self.frequencies_hz,
            angles_deg=self.angles_deg,
            port_labels=self.port_labels,
            metadata=metadata,
        )


def assert_compatible(
    first: ScatteringDataset,
    second: ScatteringDataset,
    *,
    first_name: str = "first",
    second_name: str = "second",
) -> None:
    """Require exact axes and port labels for elementwise operations."""

    if first.shape != second.shape:
        raise ValueError(f"{first_name} and {second_name} shapes differ")
    if not np.array_equal(first.frequencies_hz, second.frequencies_hz):
        raise ValueError(f"{first_name} and {second_name} frequency axes differ")
    if not np.array_equal(first.angles_deg, second.angles_deg):
        raise ValueError(f"{first_name} and {second_name} angle axes differ")
    if first.port_labels != second.port_labels:
        raise ValueError(f"{first_name} and {second_name} port labels differ")


def _strict_integer(text: str, *, field_name: str, line_number: int) -> int:
    stripped = text.strip()
    try:
        value = int(stripped)
    except ValueError as error:
        raise ValueError(
            f"line {line_number}: {field_name} must be an integer"
        ) from error
    if str(value) != stripped or value < 0:
        raise ValueError(
            f"line {line_number}: {field_name} must be a canonical non-negative integer"
        )
    return value


def _strict_float(text: str, *, field_name: str, line_number: int) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"line {line_number}: invalid {field_name}") from error
    if not np.isfinite(value):
        raise ValueError(f"line {line_number}: {field_name} must be finite")
    return value


def _contiguous_count(indices: set[int], *, axis_name: str) -> int:
    if not indices:
        raise ValueError(f"CSV contains no {axis_name} indices")
    # Do not materialize ``range(max_index + 1)``: one corrupt sparse index
    # must not turn a validation error into an unbounded allocation.
    count = len(indices)
    if min(indices) != 0 or max(indices) != count - 1:
        raise ValueError(f"{axis_name} indices must be contiguous and start at zero")
    return count


def _axis_record(
    mapping: dict[int, Any], index: int, value: Any, *, axis_name: str, line: int
) -> None:
    previous = mapping.setdefault(index, value)
    if previous != value:
        raise ValueError(
            f"line {line}: {axis_name} index {index} maps to inconsistent values"
        )


def _csv_hashes(
    path: Path,
    angles: np.ndarray,
    frequencies: np.ndarray,
    port_labels: Sequence[str],
    order_indices: np.ndarray,
    *,
    csv_sha256: str | None = None,
) -> dict[str, str]:
    return {
        "csv_sha256": sha256_file(path) if csv_sha256 is None else csv_sha256,
        "csv_schema_sha256": hashlib.sha256(
            canonical_json_bytes({"schema": CSV_SCHEMA, "columns": CSV_COLUMNS})
        ).hexdigest(),
        "angle_axis_sha256": sha256_array(angles),
        "frequency_axis_sha256": sha256_array(frequencies),
        "port_axis_sha256": hashlib.sha256(
            canonical_json_bytes(list(port_labels))
        ).hexdigest(),
        "row_order_sha256": sha256_array(order_indices),
    }


def read_scattering_csv(
    path: str | Path,
    *,
    expected_ports: int | Sequence[str] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
) -> ScatteringDataset:
    """Read and strictly validate Scatter3D-Codex long-form CSV data.

    The header must exactly equal :data:`CSV_COLUMNS`, and rows must be in
    canonical C order: angle, then frequency, receiver, and source.  No sorting
    or port-order inference is performed silently.  ``expected_hashes`` may be
    any subset of the hashes returned by :func:`write_scattering_csv`; a
    mismatch aborts loading before the dataset is returned.
    """

    source = Path(path)
    records: list[tuple[int, float, int, float, int, str, int, str, complex]] = []
    angle_values: dict[int, float] = {}
    frequency_values: dict[int, float] = {}
    port_values: dict[int, str] = {}
    angle_indices: set[int] = set()
    frequency_indices: set[int] = set()
    receiver_indices: set[int] = set()
    source_indices: set[int] = set()

    source_sha256_before = sha256_file(source)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ValueError("scattering CSV is empty") from error
        if header != CSV_COLUMNS:
            raise ValueError(
                "unexpected CSV schema; expected columns in this exact order: "
                + ", ".join(CSV_COLUMNS)
            )

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(CSV_COLUMNS):
                raise ValueError(
                    f"line {line_number}: expected {len(CSV_COLUMNS)} columns, "
                    f"found {len(row)}"
                )
            ai = _strict_integer(row[0], field_name="angle_index", line_number=line_number)
            angle = _strict_float(row[1], field_name="angle_deg", line_number=line_number)
            fi = _strict_integer(
                row[2], field_name="frequency_index", line_number=line_number
            )
            frequency = _strict_float(
                row[3], field_name="frequency_hz", line_number=line_number
            )
            ri = _strict_integer(
                row[4], field_name="receiver_index", line_number=line_number
            )
            receiver_label = row[5]
            si = _strict_integer(
                row[6], field_name="source_index", line_number=line_number
            )
            source_label = row[7]
            if receiver_label != receiver_label.strip() or source_label != source_label.strip():
                raise ValueError(
                    f"line {line_number}: port labels must not contain leading or trailing whitespace"
                )
            if not receiver_label or not source_label:
                raise ValueError(f"line {line_number}: port labels must be non-empty")
            real = _strict_float(row[8], field_name="s_real", line_number=line_number)
            imag = _strict_float(row[9], field_name="s_imag", line_number=line_number)

            _axis_record(angle_values, ai, angle, axis_name="angle", line=line_number)
            _axis_record(
                frequency_values, fi, frequency, axis_name="frequency", line=line_number
            )
            _axis_record(port_values, ri, receiver_label, axis_name="port", line=line_number)
            _axis_record(port_values, si, source_label, axis_name="port", line=line_number)
            angle_indices.add(ai)
            frequency_indices.add(fi)
            receiver_indices.add(ri)
            source_indices.add(si)
            records.append(
                (ai, angle, fi, frequency, ri, receiver_label, si, source_label, real + 1j * imag)
            )

    na = _contiguous_count(angle_indices, axis_name="angle")
    nf = _contiguous_count(frequency_indices, axis_name="frequency")
    nr = _contiguous_count(receiver_indices, axis_name="receiver")
    ns = _contiguous_count(source_indices, axis_name="source")
    if nr != ns or receiver_indices != source_indices:
        raise ValueError("receiver and source port index sets must be identical")
    if set(port_values) != receiver_indices:
        raise ValueError("every port index must have exactly one shared label")

    labels = tuple(port_values[index] for index in range(nr))
    if len(set(labels)) != len(labels):
        raise ValueError("port labels must be unique")
    if isinstance(expected_ports, int):
        if nr != expected_ports:
            raise ValueError(f"expected {expected_ports} ports, found {nr}")
    elif expected_ports is not None and labels != tuple(str(p) for p in expected_ports):
        raise ValueError("CSV port labels/order do not match expected_ports")

    expected_rows = na * nf * nr * ns
    if len(records) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} canonical rows, found {len(records)}; "
            "data may contain duplicates or omissions"
        )

    s = np.empty((na, nf, nr, ns), dtype=np.complex128)
    order_indices = np.empty((expected_rows, 4), dtype=np.int64)
    for row_number, (expected, record) in enumerate(
        zip(np.ndindex(na, nf, nr, ns), records, strict=True)
    ):
        ai, _angle, fi, _frequency, ri, _rl, si, _sl, value = record
        actual = (ai, fi, ri, si)
        if actual != expected:
            raise ValueError(
                f"CSV row {row_number + 2} is out of order: expected indices "
                f"{expected}, found {actual}"
            )
        s[expected] = value
        order_indices[row_number] = actual

    angles = np.asarray([angle_values[i] for i in range(na)], dtype=np.float64)
    frequencies = np.asarray(
        [frequency_values[i] for i in range(nf)], dtype=np.float64
    )
    source_sha256_after = sha256_file(source)
    if source_sha256_after != source_sha256_before:
        raise OSError("scattering CSV changed while it was being parsed")
    hashes = _csv_hashes(
        source,
        angles,
        frequencies,
        labels,
        order_indices,
        csv_sha256=source_sha256_before,
    )
    if expected_hashes is not None:
        unknown = set(expected_hashes).difference(hashes)
        if unknown:
            raise ValueError(f"unknown expected CSV hash fields: {sorted(unknown)}")
        for name, expected in expected_hashes.items():
            if not isinstance(expected, str) or hashes[name] != expected.lower():
                raise ValueError(f"CSV provenance mismatch for {name}")
    return ScatteringDataset(
        s=s,
        frequencies_hz=frequencies,
        angles_deg=angles,
        port_labels=labels,
        metadata={"csv_schema": CSV_SCHEMA, "source_name": source.name, **hashes},
    )


def write_scattering_csv(
    path: str | Path, dataset: ScatteringDataset
) -> Mapping[str, str]:
    """Write deterministic canonical long-form CSV and return its hashes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(CSV_COLUMNS)
            for ai, fi, ri, si in np.ndindex(dataset.shape):
                value = dataset.s[ai, fi, ri, si]
                writer.writerow(
                    (
                        ai,
                        format(dataset.angles_deg[ai], ".17g"),
                        fi,
                        format(dataset.frequencies_hz[fi], ".17g"),
                        ri,
                        dataset.port_labels[ri],
                        si,
                        dataset.port_labels[si],
                        format(value.real, ".17g"),
                        format(value.imag, ".17g"),
                    )
                )
            stream.flush()
            os.fsync(stream.fileno())
        assert temporary is not None
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    order = np.asarray(list(np.ndindex(dataset.shape)), dtype=np.int64)
    return MappingProxyType(
        _csv_hashes(
            destination,
            dataset.angles_deg,
            dataset.frequencies_hz,
            dataset.port_labels,
            order,
        )
    )


@dataclass(frozen=True, slots=True)
class ReferenceAlignment:
    """A frozen reference-derived correction applied to every acquisition.

    ``factors[frequency, receiver, source]`` maps measured data toward a model
    reference.  The factors are estimated from the reference only; callers must
    apply this same object to the reference and DUT.  Alignment is opt-in and is
    never performed by :func:`differential_scattering` unless supplied.
    """

    factors: np.ndarray
    frequencies_sha256: str
    port_labels: tuple[str, ...]
    fitted_channels: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        factors = _read_only(self.factors, dtype=np.complex128)
        fitted = _read_only(self.fitted_channels, dtype=np.bool_)
        labels = tuple(str(label) for label in self.port_labels)
        if factors.ndim != 3 or factors.shape[1] != factors.shape[2]:
            raise ValueError("alignment factors must have shape [frequency, receiver, source]")
        if fitted.shape != factors.shape[1:]:
            raise ValueError("fitted_channels shape does not match alignment ports")
        if len(labels) != factors.shape[1]:
            raise ValueError("alignment port_labels do not match factors")
        if not np.all(np.isfinite(factors.real)) or not np.all(np.isfinite(factors.imag)):
            raise ValueError("alignment factors must be finite")
        if np.any(np.abs(factors) == 0.0):
            raise ValueError("alignment factors must be nonzero")
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "fitted_channels", fitted)
        object.__setattr__(self, "port_labels", labels)
        object.__setattr__(self, "diagnostics", _metadata_copy(self.diagnostics))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": "scatter3d.reference-alignment/v1",
                    "factors_sha256": sha256_array(self.factors),
                    "frequencies_sha256": self.frequencies_sha256,
                    "port_labels": self.port_labels,
                    "fitted_channels_sha256": sha256_array(self.fitted_channels),
                }
            )
        ).hexdigest()


def estimate_reference_alignment(
    measured_reference: ScatteringDataset,
    model_reference: ScatteringDataset,
    *,
    transmission_only: bool = True,
    fit_log_gain: bool = False,
    min_relative_power: float = 1.0e-12,
) -> ReferenceAlignment:
    """Estimate an affine phase/delay correction from reference data only.

    For each selected channel, the complex least-squares model/measured ratio is
    aggregated over angles at each frequency.  An affine unwrapped phase is fit
    versus frequency; log-magnitude is either held at unity (default) or fit
    affinely when explicitly requested.  No DUT samples enter this function.
    """

    assert_compatible(
        measured_reference,
        model_reference,
        first_name="measured_reference",
        second_name="model_reference",
    )
    if not 0.0 < min_relative_power < 1.0:
        raise ValueError("min_relative_power must lie strictly between zero and one")

    nf = measured_reference.shape[1]
    ports = measured_reference.number_of_ports
    factors = np.ones((nf, ports, ports), dtype=np.complex128)
    fitted = np.zeros((ports, ports), dtype=bool)
    fitted_points: dict[str, int] = {}
    frequency = measured_reference.frequencies_hz
    span = float(np.ptp(frequency))
    scaled_frequency = (
        (frequency - float(np.mean(frequency))) / span
        if span > 0.0
        else np.zeros_like(frequency)
    )

    for receiver in range(ports):
        for source in range(ports):
            if transmission_only and receiver == source:
                continue
            measured = measured_reference.s[:, :, receiver, source]
            model = model_reference.s[:, :, receiver, source]
            denominator = np.sum(np.abs(measured) ** 2, axis=0)
            numerator = np.sum(model * measured.conj(), axis=0)
            maximum_power = float(np.max(denominator, initial=0.0))
            valid = denominator > min_relative_power * maximum_power
            valid &= np.abs(numerator) > 0.0
            if np.count_nonzero(valid) < 1:
                raise ValueError(
                    f"reference channel receiver={receiver}, source={source} "
                    "has no usable samples"
                )

            ratio = numerator[valid] / denominator[valid]
            x = scaled_frequency[valid]
            design = np.column_stack((np.ones_like(x), x))
            if x.size == 1:
                design = design[:, :1]
            weights = np.sqrt(denominator[valid] / np.max(denominator[valid]))
            weighted_design = design * weights[:, None]
            phase = np.unwrap(np.angle(ratio))
            phase_coefficients = np.linalg.lstsq(
                weighted_design, phase * weights, rcond=None
            )[0]
            phase_fit = np.column_stack(
                (np.ones(nf), scaled_frequency)
            )[:, : phase_coefficients.size] @ phase_coefficients

            if fit_log_gain:
                gain_coefficients = np.linalg.lstsq(
                    weighted_design, np.log(np.abs(ratio)) * weights, rcond=None
                )[0]
                log_gain_fit = np.column_stack(
                    (np.ones(nf), scaled_frequency)
                )[:, : gain_coefficients.size] @ gain_coefficients
            else:
                log_gain_fit = np.zeros(nf)

            factors[:, receiver, source] = np.exp(log_gain_fit + 1j * phase_fit)
            fitted[receiver, source] = True
            fitted_points[f"r{receiver}_s{source}"] = int(np.count_nonzero(valid))

    if not np.any(fitted):
        raise ValueError(
            "reference alignment selected no channels; include reflections or provide multiple ports"
        )
    return ReferenceAlignment(
        factors=factors,
        frequencies_sha256=sha256_array(frequency),
        port_labels=measured_reference.port_labels,
        fitted_channels=fitted,
        diagnostics={
            "source": "reference_only",
            "transmission_only": transmission_only,
            "fit_log_gain": fit_log_gain,
            "min_relative_power": min_relative_power,
            "fitted_frequency_points": fitted_points,
        },
    )


def apply_reference_alignment(
    dataset: ScatteringDataset, alignment: ReferenceAlignment
) -> ScatteringDataset:
    """Apply a previously frozen correction without re-estimating it."""

    if sha256_array(dataset.frequencies_hz) != alignment.frequencies_sha256:
        raise ValueError("alignment frequency axis does not match dataset")
    if dataset.port_labels != alignment.port_labels:
        raise ValueError("alignment port labels do not match dataset")
    if alignment.factors.shape != dataset.s.shape[1:]:
        raise ValueError("alignment shape does not match dataset")
    return dataset.with_s(
        dataset.s * alignment.factors[None, :, :, :],
        metadata_updates={"reference_alignment_sha256": alignment.fingerprint},
    )


def differential_scattering(
    dut: ScatteringDataset,
    reference: ScatteringDataset,
    *,
    mode: str = "same_index",
    alignment: ReferenceAlignment | None = None,
) -> ScatteringDataset:
    """Form DUT minus reference using identical indices.

    Alignment is disabled by default.  If supplied, the same frozen,
    reference-derived correction is applied to both inputs before subtraction.
    """

    if mode != "same_index":
        raise ValueError("only the explicit 'same_index' differential is supported")
    assert_compatible(dut, reference, first_name="dut", second_name="reference")
    corrected_dut = apply_reference_alignment(dut, alignment) if alignment else dut
    corrected_reference = (
        apply_reference_alignment(reference, alignment) if alignment else reference
    )
    metadata: dict[str, Any] = {
        "operation": "dut_minus_reference",
        "differential_mode": "same_index",
        "dut_fingerprint": dut.fingerprint,
        "reference_fingerprint": reference.fingerprint,
        "reference_alignment_enabled": alignment is not None,
    }
    if alignment is not None:
        metadata["reference_alignment_sha256"] = alignment.fingerprint
    return ScatteringDataset(
        s=corrected_dut.s - corrected_reference.s,
        frequencies_hz=reference.frequencies_hz,
        angles_deg=reference.angles_deg,
        port_labels=reference.port_labels,
        metadata=metadata,
    )


__all__ = [
    "CSV_COLUMNS",
    "CSV_SCHEMA",
    "ReferenceAlignment",
    "ScatteringDataset",
    "apply_reference_alignment",
    "assert_compatible",
    "differential_scattering",
    "estimate_reference_alignment",
    "read_scattering_csv",
    "write_scattering_csv",
]
