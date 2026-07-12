from __future__ import annotations

import numpy as np
import pytest

from scatter3d.metrics import volume_weighted_metrics


def test_exact_image_has_zero_error_and_perfect_overlap() -> None:
    truth = np.asarray([0.0, 1.0, 2.0, 0.0])
    volumes = np.asarray([1.0, 1.0, 2.0, 3.0])
    metrics = volume_weighted_metrics(truth, truth, volumes)
    assert metrics.weighted_relative_l2 == pytest.approx(0.0)
    assert metrics.weighted_rmse == pytest.approx(0.0)
    assert metrics.weighted_mae == pytest.approx(0.0)
    assert metrics.weighted_correlation == pytest.approx(1.0)
    assert metrics.support_iou == pytest.approx(1.0)
    assert metrics.support_dice == pytest.approx(1.0)


def test_errors_are_physically_volume_weighted() -> None:
    truth = np.asarray([1.0, 1.0])
    small_cell_error = volume_weighted_metrics(
        np.asarray([0.0, 1.0]), truth, np.asarray([1.0, 9.0])
    )
    large_cell_error = volume_weighted_metrics(
        np.asarray([1.0, 0.0]), truth, np.asarray([1.0, 9.0])
    )
    assert large_cell_error.weighted_rmse == pytest.approx(
        3.0 * small_cell_error.weighted_rmse
    )


def test_support_and_localization_use_one_truth_derived_threshold() -> None:
    truth = np.asarray([1.0, 0.0])
    estimate = np.asarray([0.0, 1.0])
    volumes = np.asarray([1.0, 3.0])
    centers = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    metrics = volume_weighted_metrics(
        estimate,
        truth,
        volumes,
        voxel_centers=centers,
        support_threshold_fraction=0.5,
    )
    assert metrics.support_threshold == pytest.approx(0.5)
    assert metrics.truth_support_volume == pytest.approx(1.0)
    assert metrics.estimate_support_volume == pytest.approx(3.0)
    assert metrics.support_iou == pytest.approx(0.0)
    assert metrics.support_dice == pytest.approx(0.0)
    assert metrics.centroid_error == pytest.approx(10.0)
    assert metrics.peak_location_error == pytest.approx(10.0)


def test_complex_global_phase_has_unit_magnitude_correlation() -> None:
    truth = np.asarray([1 + 1j, 2 - 0.5j])
    estimate = truth * np.exp(0.4j)
    metrics = volume_weighted_metrics(estimate, truth, np.ones(2))
    assert metrics.weighted_correlation == pytest.approx(1.0)
    assert metrics.weighted_relative_l2 > 0.0


def test_zero_truth_behavior_is_explicit() -> None:
    both_zero = volume_weighted_metrics(np.zeros(2), np.zeros(2), np.ones(2))
    assert both_zero.weighted_relative_l2 == 0.0
    assert both_zero.weighted_correlation == 1.0
    assert both_zero.support_iou == 1.0

    false_positive = volume_weighted_metrics(
        np.asarray([0.0, 1.0]), np.zeros(2), np.ones(2)
    )
    assert np.isinf(false_positive.weighted_relative_l2)
    assert false_positive.weighted_correlation == 0.0
    assert false_positive.support_iou == 0.0


def test_metric_inputs_are_strictly_validated() -> None:
    with pytest.raises(ValueError, match="same image shape"):
        volume_weighted_metrics(np.ones((1, 2)), np.ones((2, 1)), np.ones(2))
    with pytest.raises(ValueError, match="strictly positive"):
        volume_weighted_metrics(np.ones(2), np.ones(2), np.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="voxel_centers"):
        volume_weighted_metrics(
            np.ones(2), np.ones(2), np.ones(2), voxel_centers=np.zeros((3, 3))
        )
