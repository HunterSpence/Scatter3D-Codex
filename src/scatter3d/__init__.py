"""Scatter3D-Codex: auditable microwave-imaging building blocks."""

from .inverse import (
    DiagonalNoiseModel,
    RepeatNoiseEstimate,
    TSVDSolution,
    estimate_repeat_differential_noise,
    tsvd_solve,
    whiten_system,
)
from .measurement import (
    CSV_COLUMNS,
    CSV_SCHEMA,
    ReferenceAlignment,
    ScatteringDataset,
    apply_reference_alignment,
    assert_compatible,
    differential_scattering,
    estimate_reference_alignment,
    read_scattering_csv,
    write_scattering_csv,
)
from .metrics import ImageMetrics, volume_weighted_metrics
from .provenance import (
    MANIFEST_SCHEMA,
    array_record,
    canonical_json_bytes,
    deterministic_manifest,
    manifest_sha256,
    sha256_array,
    sha256_file,
    write_manifest,
)

__version__ = "0.1.0"

__all__ = [
    "CSV_COLUMNS",
    "CSV_SCHEMA",
    "MANIFEST_SCHEMA",
    "DiagonalNoiseModel",
    "ImageMetrics",
    "ReferenceAlignment",
    "RepeatNoiseEstimate",
    "ScatteringDataset",
    "TSVDSolution",
    "apply_reference_alignment",
    "array_record",
    "assert_compatible",
    "canonical_json_bytes",
    "deterministic_manifest",
    "differential_scattering",
    "estimate_reference_alignment",
    "estimate_repeat_differential_noise",
    "manifest_sha256",
    "read_scattering_csv",
    "sha256_array",
    "sha256_file",
    "tsvd_solve",
    "volume_weighted_metrics",
    "whiten_system",
    "write_manifest",
    "write_scattering_csv",
]
