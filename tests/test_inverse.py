from __future__ import annotations

import numpy as np
import pytest

from scatter3d.inverse import (
    DiagonalNoiseModel,
    estimate_repeat_differential_noise,
    tsvd_solve,
    whiten_system,
)
from scatter3d.measurement import ScatteringDataset


def _single_observation_dataset(
    value: complex,
    *,
    frequency_hz: float = 5.0e9,
    angle_deg: float = 0.0,
    port_label: str = "P1",
) -> ScatteringDataset:
    return ScatteringDataset(
        np.asarray([[[[value]]]], dtype=np.complex128),
        frequencies_hz=np.asarray([frequency_hz], dtype=np.float64),
        angles_deg=np.asarray([angle_deg], dtype=np.float64),
        port_labels=(port_label,),
    )


def test_paired_repeat_covariance_uses_complex_x_xh_orientation() -> None:
    # Four observations correspond to one angle, one frequency, and two ports.
    samples = np.asarray(
        [
            [1 + 2j, 2 - 1j, -1 + 0.5j, 0.25 - 2j],
            [2 + 0j, -1 + 3j, 0.5 + 1j, 2 + 0.5j],
            [-1 + 1j, 0.5 - 2j, 3 - 0.25j, -0.5 + 1j],
            [0.5 - 0.5j, 1 + 1.5j, -2 + 0.25j, 1.5 - 0.75j],
            [3 - 1j, -0.25 + 0.5j, 1.25 - 1.5j, -1 + 2j],
        ],
        dtype=np.complex128,
    )
    reference = np.zeros((5, 1, 1, 2, 2), dtype=np.complex128)
    dut = samples.reshape(5, 1, 1, 2, 2)
    estimate = estimate_repeat_differential_noise(
        reference, dut, full_covariance=True
    )

    centered = samples - samples.mean(axis=0)
    expected = centered.T @ centered.conj() / 4
    assert np.allclose(estimate.sample_covariance, expected)
    assert np.allclose(
        estimate.sample_covariance, estimate.sample_covariance.conj().T
    )
    assert not np.allclose(expected, expected.conj())
    assert np.allclose(estimate.sample_variance, np.diag(expected).real)
    assert np.allclose(estimate.mean_variance, np.diag(expected).real / 5)
    assert np.allclose(estimate.mean_covariance, expected / 5)
    assert estimate.diagnostics["pairing"] == "same_index"


def test_dense_covariance_has_an_explicit_size_guard() -> None:
    repeats = np.zeros((6, 1, 1, 2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="dense covariance"):
        estimate_repeat_differential_noise(
            repeats,
            repeats,
            full_covariance=True,
            max_full_covariance_observations=3,
        )


def test_dense_covariance_rejects_insufficient_repeat_rank() -> None:
    repeats = np.zeros((4, 1, 1, 2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="rank-deficient"):
        estimate_repeat_differential_noise(repeats, repeats, full_covariance=True)

    diagonal = estimate_repeat_differential_noise(repeats, repeats)
    assert diagonal.sample_covariance is None


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"frequency_hz": 5.1e9}, "frequency axes differ"),
        ({"angle_deg": 1.0}, "angle axes differ"),
        ({"port_label": "P2"}, "port labels differ"),
    ],
)
def test_paired_dataset_repeats_reject_cross_stack_coordinate_mismatch(
    changed: dict[str, float | str], message: str
) -> None:
    reference = [_single_observation_dataset(0.0j) for _ in range(2)]
    dut = [_single_observation_dataset(1.0 + 0.0j, **changed) for _ in range(2)]

    with pytest.raises(ValueError, match=message):
        estimate_repeat_differential_noise(reference, dut)


def test_raw_repeat_arrays_require_complex_nonempty_axes_and_matching_representation() -> None:
    real = np.zeros((2, 1, 1, 1, 1), dtype=np.float64)
    strings = np.full((2, 1, 1, 1, 1), "1+2j")
    empty = np.zeros((2, 0, 1, 1, 1), dtype=np.complex128)
    complex_stack = np.zeros((2, 1, 1, 1, 1), dtype=np.complex128)
    datasets = [_single_observation_dataset(0.0j) for _ in range(2)]

    for malformed in (real, strings):
        with pytest.raises(ValueError, match="complex dtype"):
            estimate_repeat_differential_noise(malformed, malformed)
    with pytest.raises(ValueError, match="non-empty"):
        estimate_repeat_differential_noise(empty, empty)
    with pytest.raises(TypeError, match="both be raw arrays"):
        estimate_repeat_differential_noise(complex_stack, datasets)


def test_diagonal_whitening_scales_a_and_b_together() -> None:
    variance = np.asarray([4.0, 9.0])
    noise = DiagonalNoiseModel.from_variance(variance)
    matrix = np.asarray([[2 + 2j, 4], [3j, 6 - 3j]])
    observations = np.asarray([8 + 2j, 9 - 6j])
    whitened_matrix, whitened_observations = whiten_system(
        matrix, observations, noise
    )
    assert np.allclose(whitened_matrix, matrix / np.asarray([2.0, 3.0])[:, None])
    assert np.allclose(whitened_observations, observations / np.asarray([2.0, 3.0]))
    assert np.iscomplexobj(whitened_matrix)
    assert np.iscomplexobj(whitened_observations)


def test_zero_repeat_variance_requires_physical_absolute_floor() -> None:
    with pytest.raises(ValueError, match="absolute_floor"):
        DiagonalNoiseModel.from_variance(np.zeros(3))
    model = DiagonalNoiseModel.from_variance(
        np.zeros(3), absolute_floor=1.0e-12
    )
    assert np.allclose(model.standard_deviation, 1.0e-6)


def test_repeat_estimate_mean_model_uses_variance_of_the_mean() -> None:
    reference = np.zeros((3, 1, 1, 1, 1), dtype=np.complex128)
    dut = np.asarray([-1.0, 0.0, 1.0], dtype=np.complex128).reshape(3, 1, 1, 1, 1)
    estimate = estimate_repeat_differential_noise(reference, dut)

    assert estimate.sample_variance.item() == pytest.approx(1.0)
    assert estimate.mean_variance.item() == pytest.approx(1.0 / 3.0)
    assert estimate.mean_diagonal_model().standard_deviation.item() == pytest.approx(
        np.sqrt(1.0 / 3.0)
    )


def test_fixed_rank_complex_tsvd_recovers_exact_solution() -> None:
    matrix = np.asarray(
        [
            [3 + 1j, 0.5 - 0.25j, 0.0],
            [0.0, 2 - 0.5j, 0.25j],
            [0.2, 0.0, 1 + 0.1j],
            [1j, 0.5, 0.25 - 0.5j],
        ],
        dtype=np.complex128,
    )
    truth = np.asarray([1 + 2j, -0.5 + 0.25j, 2 - 1j])
    observations = matrix @ truth
    solution = tsvd_solve(matrix, observations, method="fixed", rank=3)

    assert solution.selected_rank == 3
    assert np.allclose(solution.x, truth, atol=1e-12)
    assert solution.residual_norm < 1e-12
    assert np.iscomplexobj(solution.x)
    assert solution.diagnostics["method"] == "fixed"


def test_energy_and_discrepancy_rank_rules_are_auditable() -> None:
    energy_matrix = np.diag(np.asarray([5.0, 2.0, 0.1]))
    energy_solution = tsvd_solve(
        energy_matrix,
        np.ones(3),
        method="energy",
        energy_fraction=0.9,
    )
    assert energy_solution.selected_rank == 2
    assert np.all(np.diff(energy_solution.criterion_values) >= 0.0)

    matrix = np.diag(np.asarray([4.0, 2.0, 1.0]))
    observations = np.asarray([4.0, 2.0, 1.0])
    discrepancy = tsvd_solve(
        matrix, observations, method="discrepancy", noise_norm=1.1
    )
    assert discrepancy.selected_rank == 2
    assert discrepancy.target_met is True
    assert discrepancy.residual_norm <= 1.1


def test_discrepancy_selects_rank_zero_for_a_registered_null() -> None:
    matrix = np.diag(np.asarray([4.0, 2.0, 1.0]))
    observations = np.asarray([0.2 + 0.1j, -0.1j, 0.05])
    noise_norm = float(np.linalg.norm(observations))

    solution = tsvd_solve(
        matrix,
        observations,
        method="discrepancy",
        noise_norm=noise_norm,
    )

    assert solution.selected_rank == 0
    assert solution.selected_condition_number is None
    assert solution.target_met is True
    np.testing.assert_array_equal(solution.criterion_ranks, np.arange(4))
    np.testing.assert_array_equal(solution.x, np.zeros(3, dtype=np.complex128))
    np.testing.assert_array_equal(solution.predicted, np.zeros(3, dtype=np.complex128))


def test_gcv_returns_finite_selection_and_curve() -> None:
    matrix = np.asarray(
        [
            [4.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.2, -0.1, 0.05],
            [0.0, 0.1, 0.0],
        ]
    )
    observations = np.asarray([4.0, 2.0, 0.2, 0.15, 0.11])
    solution = tsvd_solve(matrix, observations, method="gcv")
    assert 0 <= solution.selected_rank <= solution.available_rank
    assert np.all(np.isfinite(solution.criterion_values))
    assert solution.criterion_ranks.shape == solution.criterion_values.shape


def test_gcv_can_select_rank_zero_for_orthogonal_null_noise() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    observations = np.asarray([0.0, 0.0, 1.0, -1.0])
    solution = tsvd_solve(matrix, observations, method="gcv")

    assert solution.selected_rank == 0
    assert solution.selected_condition_number is None
    np.testing.assert_array_equal(solution.x, np.zeros(2, dtype=np.complex128))


def test_tsvd_rejects_ambiguous_or_unusable_settings() -> None:
    matrix = np.eye(2)
    observations = np.ones(2)
    with pytest.raises(ValueError, match="only valid"):
        tsvd_solve(matrix, observations, method="gcv", rank=1)
    with pytest.raises(ValueError, match="noise_norm"):
        tsvd_solve(matrix, observations, method="discrepancy")
    with pytest.raises(ValueError, match="only valid"):
        tsvd_solve(matrix, observations, method="gcv", noise_norm=1.0)
    with pytest.raises(ValueError, match="only valid"):
        tsvd_solve(matrix, observations, method="gcv", energy_fraction=0.9)
    with pytest.raises(np.linalg.LinAlgError, match="nonzero"):
        tsvd_solve(np.zeros((2, 2)), observations)


def test_seeded_complex_discrepancy_curve_matches_direct_residuals() -> None:
    rng = np.random.default_rng(20260712)
    for rows, columns in ((3, 2), (5, 3), (3, 5), (8, 4)):
        for _ in range(10):
            matrix = rng.normal(size=(rows, columns)) + 1j * rng.normal(
                size=(rows, columns)
            )
            observations = rng.normal(size=rows) + 1j * rng.normal(size=rows)
            full = tsvd_solve(
                matrix,
                observations,
                method="discrepancy",
                noise_norm=0.0,
            )
            u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
            direct = []
            for rank in full.criterion_ranks:
                if rank == 0:
                    estimate = np.zeros(columns, dtype=np.complex128)
                else:
                    estimate = (
                        vh[:rank].conj().T
                        @ ((u[:, :rank].conj().T @ observations) / singular_values[:rank])
                    )
                direct.append(np.linalg.norm(observations - matrix @ estimate))
            np.testing.assert_allclose(full.criterion_values, direct, rtol=1e-11, atol=1e-12)
