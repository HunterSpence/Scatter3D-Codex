"""File-level, verification-first measurement and inversion workflows.

The numerical modules intentionally operate on in-memory arrays.  This module
adds one small, documented NPZ interchange contract and refuses ambiguous axis
orders rather than guessing them.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .measurement import ScatteringDataset, differential_scattering
from .provenance import sha256_array, sha256_file

SCHEMA_VERSION = "scatter3d.measurement.v1"
SENSITIVITY_SCHEMA_VERSION = "scatter3d.sensitivity.v1"
RECONSTRUCTION_SCHEMA_VERSION = "scatter3d.reconstruction.v1"
ChannelMode = Literal["all", "transmission", "reflection"]
WhiteningMode = Literal["auto", "off", "required"]


def _finite_complex(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value.real)) or not np.all(np.isfinite(value.imag)):
        raise ValueError(f"{name} contains non-finite values")


def _exact_dtype(name: str, value: Any, dtype: np.dtype[Any] | type) -> np.ndarray:
    array = np.asarray(value)
    expected = np.dtype(dtype)
    if array.dtype != expected:
        raise ValueError(f"{name} must use dtype {expected.name}; found {array.dtype}")
    return array


def _unicode_vector(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "U":
        raise ValueError(f"{name} must use a Unicode dtype")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _scalar_text(archive: Any, key: str) -> str:
    if key not in archive:
        raise ValueError(f"missing required key {key!r}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar string")
    return str(value.item())


def _repeat_axis(name: str, value: np.ndarray) -> np.ndarray:
    array = _exact_dtype(name, value, np.complex128)
    if array.ndim == 4:
        array = array[np.newaxis, ...]
    if array.ndim != 5:
        raise ValueError(
            f"{name} must have shape [repeat, angle, frequency, receiver, source] "
            "or omit only the repeat axis"
        )
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one repeat")
    result = np.asarray(array, dtype=np.complex128)
    _finite_complex(name, result)
    return result


@dataclass(frozen=True)
class MeasurementBundle:
    """Validated reference and DUT repeats sharing one coordinate system."""

    reference_s: np.ndarray
    dut_s: np.ndarray
    frequencies_hz: np.ndarray
    angles_deg: np.ndarray
    port_labels: tuple[str, ...]
    source_path: Path
    source_sha256: str

    @property
    def measurement_shape(self) -> tuple[int, int, int, int]:
        return tuple(int(item) for item in self.reference_s.shape[1:])  # type: ignore[return-value]

    @property
    def full_row_count(self) -> int:
        return int(np.prod(self.measurement_shape, dtype=np.int64))

    def mean_dataset(self, which: Literal["reference", "dut"]) -> ScatteringDataset:
        values = self.reference_s if which == "reference" else self.dut_s
        return ScatteringDataset(
            s=np.mean(values, axis=0),
            frequencies_hz=self.frequencies_hz,
            angles_deg=self.angles_deg,
            port_labels=self.port_labels,
            metadata={
                "source_sha256": self.source_sha256,
                "repeat_count": int(values.shape[0]),
                "role": which,
            },
        )


@dataclass(frozen=True)
class BundleValidation:
    schema_version: str
    source_sha256: str
    reference_repeats: int
    dut_repeats: int
    angles: int
    frequencies: int
    ports: int
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrequencyChannelDiagnostic:
    frequency_index: int
    frequency_hz: float
    receiver_index: int
    receiver_port: str
    source_index: int
    source_port: str
    differential_rms: float
    repeat_noise_rms: float | None
    signal_to_repeat_noise: float | None
    signal_to_repeat_noise_db: float | None


@dataclass(frozen=True)
class DiagnosticReport:
    source_sha256: str
    reference_repeats: int
    dut_repeats: int
    differential_rms: float
    repeat_noise_rms: float | None
    signal_to_repeat_noise: float | None
    signal_to_repeat_noise_db: float | None
    reference_reciprocity_rms: float
    dut_reciprocity_rms: float
    unique_reference_angle_traces: int
    unique_dut_angle_traces: int
    complex_variance_convention: str
    frequency_channels: tuple[FrequencyChannelDiagnostic, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionReport:
    bundle_sha256: str
    sensitivity_sha256: str
    output_path: str
    method: str
    status: str
    channel_mode: str
    row_order: str
    requested_rank: int | None
    selected_rank: int
    available_rank: int
    rows_used: int
    voxels: int
    residual_norm: float
    relative_residual: float
    solve_residual_norm: float
    solution_norm: float
    selected_condition_number: float | None
    singular_value_threshold: float
    singular_values_sha256: str
    criterion_sha256: str
    selection_target_met: bool | None
    whitening_used: bool
    whitening_reason: str
    paired_repeats: int | None
    noise_model_sha256: str | None
    noise_norm_used: float | None
    noise_norm_basis: str | None
    energy_fraction_used: float | None
    noise_relative_floor: float
    noise_absolute_floor: float
    overwrite_requested: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_measurement_bundle(path: str | Path) -> MeasurementBundle:
    """Load the documented NPZ contract with pickle explicitly disabled."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha256 = sha256_file(source)
    with np.load(source, allow_pickle=False) as archive:
        version = _scalar_text(archive, "schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}")
        required = {"reference_s", "dut_s", "frequencies_hz", "angles_deg", "port_labels"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"measurement bundle is missing keys: {sorted(missing)}")
        reference_s = _repeat_axis("reference_s", archive["reference_s"])
        dut_s = _repeat_axis("dut_s", archive["dut_s"])
        frequencies_hz = _exact_dtype(
            "frequencies_hz", archive["frequencies_hz"], np.float64
        )
        angles_deg = _exact_dtype("angles_deg", archive["angles_deg"], np.float64)
        labels_array = _unicode_vector("port_labels", archive["port_labels"])

    if sha256_file(source) != source_sha256:
        raise OSError("measurement bundle changed while it was being parsed")

    if reference_s.shape[1:] != dut_s.shape[1:]:
        raise ValueError("reference_s and dut_s measurement axes do not match")
    port_labels = tuple(str(item) for item in labels_array.tolist())

    # ScatteringDataset is the single source of truth for coordinate validation.
    ScatteringDataset(
        s=reference_s[0],
        frequencies_hz=frequencies_hz,
        angles_deg=angles_deg,
        port_labels=port_labels,
    )
    ScatteringDataset(
        s=dut_s[0],
        frequencies_hz=frequencies_hz,
        angles_deg=angles_deg,
        port_labels=port_labels,
    )
    return MeasurementBundle(
        reference_s=reference_s,
        dut_s=dut_s,
        frequencies_hz=frequencies_hz,
        angles_deg=angles_deg,
        port_labels=port_labels,
        source_path=source,
        source_sha256=source_sha256,
    )


def validate_measurement_bundle(path: str | Path) -> BundleValidation:
    bundle = load_measurement_bundle(path)
    angles, frequencies, receivers, sources = bundle.measurement_shape
    if receivers != sources:  # defensive; ScatteringDataset also checks this
        raise ValueError("receiver/source port counts differ")
    return BundleValidation(
        schema_version=SCHEMA_VERSION,
        source_sha256=bundle.source_sha256,
        reference_repeats=int(bundle.reference_s.shape[0]),
        dut_repeats=int(bundle.dut_s.shape[0]),
        angles=angles,
        frequencies=frequencies,
        ports=receivers,
        rows=bundle.full_row_count,
    )


def _repeat_noise_rms(value: np.ndarray) -> float | None:
    if value.shape[0] < 2:
        return None
    residual = value - np.mean(value, axis=0, keepdims=True)
    # Unbiased per-complex-sample variance, then return its RMS amplitude.
    variance = np.sum(np.abs(residual) ** 2) / ((value.shape[0] - 1) * np.prod(value.shape[1:]))
    return float(np.sqrt(variance))


def _frequency_channel_noise_rms(value: np.ndarray) -> np.ndarray | None:
    """Return repeat RMS by [frequency, receiver, source], averaging angles."""

    if value.shape[0] < 2:
        return None
    residual = value - np.mean(value, axis=0, keepdims=True)
    sample_variance = np.sum(np.abs(residual) ** 2, axis=0) / (value.shape[0] - 1)
    return np.sqrt(np.mean(sample_variance, axis=0))


def _ratio_metrics(signal: float, noise: float | None) -> tuple[float | None, float | None]:
    if noise is None or noise <= 0.0:
        return None, None
    ratio = float(signal / noise)
    ratio_db = float(20.0 * np.log10(ratio)) if ratio > 0.0 else None
    return ratio, ratio_db


def _reciprocity_rms(value: np.ndarray) -> float:
    difference = value - np.swapaxes(value, -2, -1)
    return float(np.sqrt(np.mean(np.abs(difference) ** 2)))


def _unique_angle_traces(value: np.ndarray) -> int:
    # Exact equality is intentional: this catches the classic aliasing bug where
    # every list entry references the final mutable angle array.
    flattened = np.ascontiguousarray(value).reshape(value.shape[0], -1)
    return len({row.tobytes() for row in flattened})


def diagnose_measurement_bundle(path: str | Path) -> DiagnosticReport:
    bundle = load_measurement_bundle(path)
    reference = np.mean(bundle.reference_s, axis=0)
    dut = np.mean(bundle.dut_s, axis=0)
    differential = dut - reference
    differential_rms = float(np.sqrt(np.mean(np.abs(differential) ** 2)))
    ref_noise = _repeat_noise_rms(bundle.reference_s)
    dut_noise = _repeat_noise_rms(bundle.dut_s)
    if ref_noise is None or dut_noise is None:
        noise = ratio = ratio_db = None
    else:
        noise = float(np.hypot(ref_noise, dut_noise))
        ratio, ratio_db = _ratio_metrics(differential_rms, noise)

    differential_by_channel = np.sqrt(np.mean(np.abs(differential) ** 2, axis=0))
    reference_noise_by_channel = _frequency_channel_noise_rms(bundle.reference_s)
    dut_noise_by_channel = _frequency_channel_noise_rms(bundle.dut_s)
    channel_reports: list[FrequencyChannelDiagnostic] = []
    for frequency_index, frequency_hz in enumerate(bundle.frequencies_hz):
        for receiver_index, receiver_port in enumerate(bundle.port_labels):
            for source_index, source_port in enumerate(bundle.port_labels):
                channel_signal = float(
                    differential_by_channel[frequency_index, receiver_index, source_index]
                )
                if reference_noise_by_channel is None or dut_noise_by_channel is None:
                    channel_noise = None
                else:
                    channel_noise = float(
                        np.hypot(
                            reference_noise_by_channel[
                                frequency_index, receiver_index, source_index
                            ],
                            dut_noise_by_channel[
                                frequency_index, receiver_index, source_index
                            ],
                        )
                    )
                channel_ratio, channel_ratio_db = _ratio_metrics(
                    channel_signal, channel_noise
                )
                channel_reports.append(
                    FrequencyChannelDiagnostic(
                        frequency_index=frequency_index,
                        frequency_hz=float(frequency_hz),
                        receiver_index=receiver_index,
                        receiver_port=receiver_port,
                        source_index=source_index,
                        source_port=source_port,
                        differential_rms=channel_signal,
                        repeat_noise_rms=channel_noise,
                        signal_to_repeat_noise=channel_ratio,
                        signal_to_repeat_noise_db=channel_ratio_db,
                    )
                )
    return DiagnosticReport(
        source_sha256=bundle.source_sha256,
        reference_repeats=int(bundle.reference_s.shape[0]),
        dut_repeats=int(bundle.dut_s.shape[0]),
        differential_rms=differential_rms,
        repeat_noise_rms=noise,
        signal_to_repeat_noise=ratio,
        signal_to_repeat_noise_db=ratio_db,
        reference_reciprocity_rms=_reciprocity_rms(reference),
        dut_reciprocity_rms=_reciprocity_rms(dut),
        unique_reference_angle_traces=_unique_angle_traces(reference),
        unique_dut_angle_traces=_unique_angle_traces(dut),
        complex_variance_convention=(
            "unbiased E[abs(z-mean)^2] per complex sample; RMS=sqrt(variance)"
        ),
        frequency_channels=tuple(channel_reports),
    )


def _channel_mask(shape: tuple[int, int, int, int], mode: ChannelMode) -> np.ndarray:
    angles, frequencies, receivers, sources = shape
    port_mask = np.ones((receivers, sources), dtype=bool)
    if mode == "transmission":
        port_mask = ~np.eye(receivers, dtype=bool)
    elif mode == "reflection":
        port_mask = np.eye(receivers, dtype=bool)
    elif mode != "all":
        raise ValueError(f"unsupported channel mode {mode!r}")
    return np.broadcast_to(port_mask, (angles, frequencies, receivers, sources)).reshape(-1)


def _check_sensitivity_coordinates(archive: Any, bundle: MeasurementBundle) -> None:
    for key in ("frequencies_hz", "angles_deg", "port_labels"):
        if key not in archive:
            raise ValueError(f"sensitivity archive is missing coordinate key {key!r}")
    frequencies = _exact_dtype(
        "sensitivity frequencies_hz", archive["frequencies_hz"], np.float64
    )
    angles = _exact_dtype("sensitivity angles_deg", archive["angles_deg"], np.float64)
    labels_array = _unicode_vector("sensitivity port_labels", archive["port_labels"])
    labels = tuple(str(item) for item in labels_array.tolist())
    if not np.array_equal(frequencies, bundle.frequencies_hz):
        raise ValueError("sensitivity and measurement frequency grids differ")
    if not np.array_equal(angles, bundle.angles_deg):
        raise ValueError("sensitivity and measurement angle grids differ")
    if labels != bundle.port_labels:
        raise ValueError("sensitivity and measurement port labels differ")


def _solution_vector(solution: Any) -> np.ndarray:
    # TSVDSolution exposes `x`; the explicit check keeps a future API change from
    # silently writing the wrong object to disk.
    if not hasattr(solution, "x"):
        raise TypeError("tsvd_solve returned an object without the documented 'x' field")
    result = np.asarray(solution.x, dtype=np.complex128)
    if result.ndim != 1:
        raise ValueError("TSVD result must be one-dimensional")
    _finite_complex("TSVD result", result)
    return result


@dataclass(frozen=True)
class _WhiteningDecision:
    standard_deviation: np.ndarray | None
    reason: str
    paired_repeats: int | None


def _repeat_whitening(
    bundle: MeasurementBundle,
    *,
    mode: WhiteningMode,
    relative_floor: float,
    absolute_floor: float,
) -> _WhiteningDecision:
    if mode not in {"auto", "off", "required"}:
        raise ValueError(f"unsupported whitening mode {mode!r}")
    if mode == "off":
        return _WhiteningDecision(None, "disabled by request", None)

    reference_count = int(bundle.reference_s.shape[0])
    dut_count = int(bundle.dut_s.shape[0])
    if reference_count != dut_count or reference_count < 2:
        reason = (
            "paired whitening requires equal reference/DUT repeat counts of at least two; "
            f"found reference={reference_count}, DUT={dut_count}"
        )
        if mode == "required":
            raise ValueError(reason)
        return _WhiteningDecision(None, reason, None)

    from .inverse import DiagonalNoiseModel, estimate_repeat_differential_noise

    estimate = estimate_repeat_differential_noise(bundle.reference_s, bundle.dut_s)
    # The inverted observation is the mean paired differential, so use its
    # explicitly named variance rather than the individual-sample variance.
    mean_variance = estimate.mean_variance
    if not np.any(mean_variance > 0.0) and absolute_floor <= 0.0:
        reason = "all paired-repeat variances are zero and no absolute floor was supplied"
        if mode == "required":
            raise ValueError(reason)
        return _WhiteningDecision(None, reason, estimate.repeat_count)
    model = DiagonalNoiseModel.from_variance(
        mean_variance,
        relative_floor=relative_floor,
        absolute_floor=absolute_floor,
    )
    return _WhiteningDecision(
        model.standard_deviation,
        "paired same-index differential variance scaled by 1/repeat_count",
        estimate.repeat_count,
    )


def reconstruct_from_bundle(
    bundle_path: str | Path,
    sensitivity_path: str | Path,
    output_path: str | Path,
    *,
    channel_mode: ChannelMode = "all",
    method: Literal["fixed", "gcv", "discrepancy", "energy"] = "gcv",
    rank: int | None = None,
    noise_norm: float | None = None,
    energy_fraction: float | None = None,
    whitening: WhiteningMode = "auto",
    noise_relative_floor: float = 1.0e-12,
    noise_absolute_floor: float = 0.0,
    overwrite: bool = False,
) -> ReconstructionReport:
    """Form a same-index differential and solve a coordinate-checked TSVD."""

    from .inverse import DiagonalNoiseModel, tsvd_solve, whiten_system

    if not np.isfinite(noise_relative_floor) or noise_relative_floor < 0.0:
        raise ValueError("noise_relative_floor must be finite and non-negative")
    if not np.isfinite(noise_absolute_floor) or noise_absolute_floor < 0.0:
        raise ValueError("noise_absolute_floor must be finite and non-negative")
    bundle = load_measurement_bundle(bundle_path)
    sensitivity_source = Path(sensitivity_path).expanduser().resolve()
    if not sensitivity_source.is_file():
        raise FileNotFoundError(sensitivity_source)
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing reconstruction: {output}")
    sensitivity_sha256 = sha256_file(sensitivity_source)

    reference = bundle.mean_dataset("reference")
    dut = bundle.mean_dataset("dut")
    b_full = differential_scattering(dut, reference, mode="same_index").vector()
    mask = _channel_mask(bundle.measurement_shape, channel_mode)

    with np.load(sensitivity_source, allow_pickle=False) as archive:
        version = _scalar_text(archive, "schema_version")
        if version != SENSITIVITY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported sensitivity schema {version!r}; "
                f"expected {SENSITIVITY_SCHEMA_VERSION!r}"
            )
        if "A" not in archive:
            raise ValueError("sensitivity archive is missing matrix key 'A'")
        _check_sensitivity_coordinates(archive, bundle)
        A = _exact_dtype("A", archive["A"], np.complex128)
        row_indices = (
            _exact_dtype("row_indices", archive["row_indices"], np.int64)
            if "row_indices" in archive
            else np.arange(bundle.full_row_count, dtype=np.int64)
        )

    if sha256_file(sensitivity_source) != sensitivity_sha256:
        raise OSError("sensitivity archive changed while it was being parsed")

    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional complex matrix")
    _finite_complex("A", A)
    if row_indices.ndim != 1 or row_indices.size != A.shape[0]:
        raise ValueError("row_indices must be one-dimensional and match A rows")
    if np.unique(row_indices).size != row_indices.size:
        raise ValueError("row_indices contains duplicates")
    if np.any(row_indices < 0) or np.any(row_indices >= bundle.full_row_count):
        raise ValueError("row_indices contains an out-of-range measurement row")

    keep = mask[row_indices]
    if not np.any(keep):
        raise ValueError(f"channel_mode={channel_mode!r} selects no sensitivity rows")
    A_used = A[keep]
    b_used = b_full[row_indices[keep]]
    whitening_decision = _repeat_whitening(
        bundle,
        mode=whitening,
        relative_floor=noise_relative_floor,
        absolute_floor=noise_absolute_floor,
    )
    selected_noise: np.ndarray | None = None
    if whitening_decision.standard_deviation is not None:
        selected_noise = whitening_decision.standard_deviation[row_indices[keep]]
        selected_model = DiagonalNoiseModel(
            selected_noise,
            diagnostics={
                "source": "paired same-index DUT-reference repeats",
                "variance_of": "mean differential",
                "repeat_count": whitening_decision.paired_repeats,
            },
        )
        A_solve, b_solve = whiten_system(A_used, b_used, selected_model)
    else:
        A_solve, b_solve = A_used, b_used

    effective_noise_norm = noise_norm
    noise_norm_basis = "user_supplied" if noise_norm is not None else None
    if method == "discrepancy" and effective_noise_norm is None and selected_noise is not None:
        # With E|z_i|^2 = 1 after whitening, E||z||^2 equals the row count.
        # This is an expected RMS scale, not a high-confidence statistical bound.
        effective_noise_norm = float(np.sqrt(b_solve.size))
        noise_norm_basis = "whitened_expected_rms_sqrt_rows"
    effective_energy_fraction = (
        0.999 if method == "energy" and energy_fraction is None else energy_fraction
    )
    solution = tsvd_solve(
        A_solve,
        b_solve,
        method=method,
        rank=rank,
        noise_norm=effective_noise_norm,
        energy_fraction=effective_energy_fraction,
    )
    estimate = _solution_vector(solution)
    selected_rank = int(solution.selected_rank)
    residual_norm = float(np.linalg.norm(A_used @ estimate - b_used))
    b_norm = float(np.linalg.norm(b_used))
    relative_residual = residual_norm / b_norm if b_norm > 0.0 else residual_norm
    solve_residual_norm = float(np.linalg.norm(A_solve @ estimate - b_solve))
    noise_model_sha256 = sha256_array(selected_noise) if selected_noise is not None else None
    singular_values_sha256 = sha256_array(solution.singular_values)
    criterion_sha256 = hashlib.sha256(
        (
            sha256_array(solution.criterion_ranks)
            + ":"
            + sha256_array(solution.criterion_values)
        ).encode("ascii")
    ).hexdigest()
    status = "FAILED" if solution.target_met is False else "PASSED"

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            schema_version=np.array(RECONSTRUCTION_SCHEMA_VERSION),
            estimate=estimate,
            status=np.array(status),
            requested_rank=np.array(-1 if rank is None else rank, dtype=np.int64),
            selected_rank=np.array(selected_rank, dtype=np.int64),
            available_rank=np.array(solution.available_rank, dtype=np.int64),
            method=np.array(method),
            channel_mode=np.array(channel_mode),
            row_order=np.array("C:[angle,frequency,receiver,source]"),
            residual_norm=np.array(residual_norm, dtype=np.float64),
            relative_residual=np.array(relative_residual, dtype=np.float64),
            solve_residual_norm=np.array(solve_residual_norm, dtype=np.float64),
            solution_norm=np.array(solution.solution_norm, dtype=np.float64),
            selected_condition_number=np.array(
                np.nan
                if solution.selected_condition_number is None
                else solution.selected_condition_number,
                dtype=np.float64,
            ),
            singular_value_threshold=np.array(
                solution.singular_value_threshold, dtype=np.float64
            ),
            singular_values=solution.singular_values,
            singular_values_sha256=np.array(singular_values_sha256),
            criterion_ranks=solution.criterion_ranks,
            criterion_values=solution.criterion_values,
            criterion_sha256=np.array(criterion_sha256),
            selection_target_met=np.array(
                -1 if solution.target_met is None else int(solution.target_met),
                dtype=np.int8,
            ),
            whitening_used=np.array(selected_noise is not None),
            whitening_reason=np.array(whitening_decision.reason),
            paired_repeats=np.array(
                -1 if whitening_decision.paired_repeats is None else whitening_decision.paired_repeats,
                dtype=np.int64,
            ),
            noise_standard_deviation=(
                np.array([], dtype=np.float64) if selected_noise is None else selected_noise
            ),
            noise_model_sha256=np.array(noise_model_sha256 or ""),
            noise_norm_used=np.array(
                np.nan if effective_noise_norm is None else effective_noise_norm, dtype=np.float64
            ),
            noise_norm_basis=np.array(noise_norm_basis or ""),
            energy_fraction_used=np.array(
                np.nan
                if effective_energy_fraction is None
                else effective_energy_fraction,
                dtype=np.float64,
            ),
            noise_relative_floor=np.array(noise_relative_floor, dtype=np.float64),
            noise_absolute_floor=np.array(noise_absolute_floor, dtype=np.float64),
            overwrite_requested=np.array(overwrite),
            bundle_sha256=np.array(bundle.source_sha256),
            sensitivity_sha256=np.array(sensitivity_sha256),
            row_indices=row_indices[keep],
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return ReconstructionReport(
        bundle_sha256=bundle.source_sha256,
        sensitivity_sha256=sensitivity_sha256,
        output_path=str(output),
        method=method,
        status=status,
        channel_mode=channel_mode,
        row_order="C:[angle,frequency,receiver,source]",
        requested_rank=rank,
        selected_rank=selected_rank,
        available_rank=int(solution.available_rank),
        rows_used=int(A_used.shape[0]),
        voxels=int(A_used.shape[1]),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        solve_residual_norm=solve_residual_norm,
        solution_norm=float(solution.solution_norm),
        selected_condition_number=solution.selected_condition_number,
        singular_value_threshold=float(solution.singular_value_threshold),
        singular_values_sha256=singular_values_sha256,
        criterion_sha256=criterion_sha256,
        selection_target_met=solution.target_met,
        whitening_used=selected_noise is not None,
        whitening_reason=whitening_decision.reason,
        paired_repeats=whitening_decision.paired_repeats,
        noise_model_sha256=noise_model_sha256,
        noise_norm_used=effective_noise_norm,
        noise_norm_basis=noise_norm_basis,
        energy_fraction_used=effective_energy_fraction,
        noise_relative_floor=float(noise_relative_floor),
        noise_absolute_floor=float(noise_absolute_floor),
        overwrite_requested=overwrite,
    )


def write_json_report(
    report: Any, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Atomically write an object exposing ``to_dict`` as UTF-8 JSON."""

    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict() if hasattr(report, "to_dict") else report
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def schema_description() -> dict[str, Any]:
    return {
        "measurement": {
            "schema_version": SCHEMA_VERSION,
            "required_keys": {
                "reference_s": "complex128 [repeat, angle, frequency, receiver, source]",
                "dut_s": "complex128 [repeat, angle, frequency, receiver, source]",
                "frequencies_hz": "float64 [frequency], finite and strictly increasing",
                "angles_deg": "float64 [angle], finite and strictly increasing",
                "port_labels": "unicode [port], unique",
            },
            "repeat_axis": "may be omitted only when exactly one repeat is present",
        },
        "sensitivity": {
            "schema_version": SENSITIVITY_SCHEMA_VERSION,
            "required_keys": {
                "A": "complex128 [measurement_row, voxel]",
                "frequencies_hz": "float64; must exactly equal the measurement grid",
                "angles_deg": "float64; must exactly equal the measurement grid",
                "port_labels": "Unicode; must exactly equal the measurement labels",
            },
            "optional_keys": {
                "row_indices": "int64 rows into C-order [angle, frequency, receiver, source]",
            },
        },
        "reconstruction": {
            "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
            "required_keys": {
                "estimate": "complex128 [voxel]",
                "status": "PASSED or FAILED",
                "method": "fixed, gcv, discrepancy, or energy",
                "requested_rank": "int64; -1 when not requested",
                "selected_rank": "int64 retained TSVD rank",
                "available_rank": "int64 numerical rank above threshold",
                "singular_values": "float64 full singular spectrum",
                "criterion_ranks": "int64 selector candidate ranks",
                "criterion_values": "float64 selector curve",
                "selection_target_met": "int8: -1 not applicable, 0 failed, 1 passed",
                "noise_standard_deviation": "float64 selected-row mean-noise scale",
                "row_indices": "int64 canonical measurement rows used",
                "bundle_sha256": "measurement archive content hash",
                "sensitivity_sha256": "sensitivity archive content hash",
            },
            "overwrite_default": False,
        },
    }
