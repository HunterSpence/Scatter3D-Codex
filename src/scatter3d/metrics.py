"""Volume-weighted quantitative metrics for reconstructed images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ImageMetrics:
    """Scalar image-quality metrics with explicit physical volume weighting."""

    weighted_relative_l2: float
    weighted_rmse: float
    weighted_mae: float
    weighted_correlation: float
    support_iou: float
    support_dice: float
    truth_support_volume: float
    estimate_support_volume: float
    support_threshold: float
    centroid_error: float | None = None
    peak_location_error: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_image(name: str, image: np.ndarray) -> np.ndarray:
    result = np.asarray(image)
    if result.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.all(np.isfinite(result.real)) or not np.all(np.isfinite(result.imag)):
        raise ValueError(f"{name} must be finite")
    return result.reshape(-1)


def volume_weighted_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    voxel_volumes: np.ndarray,
    *,
    voxel_centers: np.ndarray | None = None,
    support_threshold_fraction: float = 0.5,
) -> ImageMetrics:
    """Compare estimate and truth without treating unequal cells as equal.

    Errors use the complex values directly.  Correlation and support metrics use
    magnitudes; the support threshold is one absolute level derived from the
    truth maximum and applied to both images.  This avoids independently
    rescaling the estimate merely to improve overlap.
    """

    if np.asarray(estimate).shape != np.asarray(truth).shape:
        raise ValueError("estimate and truth must have the same image shape")
    estimate_flat = _finite_image("estimate", estimate)
    truth_flat = _finite_image("truth", truth)
    volumes = np.asarray(voxel_volumes, dtype=np.float64).reshape(-1)
    if volumes.size != truth_flat.size:
        raise ValueError("voxel_volumes must match the image cell count")
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("voxel_volumes must be finite and strictly positive")
    if (
        not np.isfinite(support_threshold_fraction)
        or not 0.0 < support_threshold_fraction <= 1.0
    ):
        raise ValueError("support_threshold_fraction must lie in (0, 1]")

    error = estimate_flat - truth_flat
    total_volume = float(np.sum(volumes))
    error_energy = float(np.sum(volumes * np.abs(error) ** 2))
    truth_energy = float(np.sum(volumes * np.abs(truth_flat) ** 2))
    estimate_energy = float(np.sum(volumes * np.abs(estimate_flat) ** 2))
    weighted_rmse = float(np.sqrt(error_energy / total_volume))
    weighted_mae = float(np.sum(volumes * np.abs(error)) / total_volume)
    if truth_energy > 0.0:
        relative_l2 = float(np.sqrt(error_energy / truth_energy))
    else:
        relative_l2 = 0.0 if error_energy == 0.0 else float("inf")

    correlation_denominator = np.sqrt(truth_energy * estimate_energy)
    correlation = (
        float(
            np.abs(np.sum(volumes * np.conj(truth_flat) * estimate_flat))
            / correlation_denominator
        )
        if correlation_denominator > 0.0
        else (1.0 if truth_energy == 0.0 and estimate_energy == 0.0 else 0.0)
    )
    # Clip only floating-point roundoff beyond the mathematical [0, 1] range.
    correlation = float(np.clip(correlation, 0.0, 1.0))

    truth_magnitude = np.abs(truth_flat)
    estimate_magnitude = np.abs(estimate_flat)
    truth_peak = float(np.max(truth_magnitude))
    threshold = support_threshold_fraction * truth_peak
    if truth_peak > 0.0:
        truth_support = truth_magnitude >= threshold
        estimate_support = estimate_magnitude >= threshold
    else:
        truth_support = truth_magnitude > 0.0
        estimate_support = estimate_magnitude > 0.0

    intersection_volume = float(np.sum(volumes[truth_support & estimate_support]))
    union_volume = float(np.sum(volumes[truth_support | estimate_support]))
    truth_support_volume = float(np.sum(volumes[truth_support]))
    estimate_support_volume = float(np.sum(volumes[estimate_support]))
    iou = intersection_volume / union_volume if union_volume > 0.0 else 1.0
    support_sum = truth_support_volume + estimate_support_volume
    dice = 2.0 * intersection_volume / support_sum if support_sum > 0.0 else 1.0

    centroid_error: float | None = None
    peak_location_error: float | None = None
    if voxel_centers is not None:
        centers = np.asarray(voxel_centers, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[0] != truth_flat.size:
            raise ValueError("voxel_centers must have shape [cell, spatial_dimension]")
        if centers.shape[1] < 1 or not np.all(np.isfinite(centers)):
            raise ValueError("voxel_centers must be finite and have at least one coordinate")
        truth_mass = volumes * truth_magnitude
        estimate_mass = volumes * estimate_magnitude
        if np.sum(truth_mass) > 0.0 and np.sum(estimate_mass) > 0.0:
            truth_centroid = np.sum(centers * truth_mass[:, None], axis=0) / np.sum(
                truth_mass
            )
            estimate_centroid = np.sum(
                centers * estimate_mass[:, None], axis=0
            ) / np.sum(estimate_mass)
            centroid_error = float(np.linalg.norm(estimate_centroid - truth_centroid))
            peak_location_error = float(
                np.linalg.norm(
                    centers[int(np.argmax(estimate_magnitude))]
                    - centers[int(np.argmax(truth_magnitude))]
                )
            )

    return ImageMetrics(
        weighted_relative_l2=relative_l2,
        weighted_rmse=weighted_rmse,
        weighted_mae=weighted_mae,
        weighted_correlation=correlation,
        support_iou=float(iou),
        support_dice=float(dice),
        truth_support_volume=truth_support_volume,
        estimate_support_volume=estimate_support_volume,
        support_threshold=threshold,
        centroid_error=centroid_error,
        peak_location_error=peak_location_error,
    )


__all__ = ["ImageMetrics", "volume_weighted_metrics"]
