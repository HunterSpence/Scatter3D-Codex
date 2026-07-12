from __future__ import annotations

import numpy as np
import pytest

from scatter3d.inverse import (
    DiagonalNoiseModel,
    estimate_repeat_differential_noise,
    tsvd_solve,
    whiten_system,
)


def test_paired_repeat_covariance_uses_complex_x_xh_orientation() -> None:
    # Four observations correspond to one angle, one frequency, and two ports.
    samples = np.asarray(
        [
            [1 + 2j, 2 - 1j, -1 + 0.5j, 0.25 - 2j],
            [2 + 0j, -1 + 3j, 0.5 + 1j, 2 + 0.5j],
            [-1 + 1j, 0.5 - 2j, 3 - 0.25j, -0.5 + 1j],
        ],
        dtype=np.complex128,
    )
    reference = np.zeros((3, 1, 1, 2, 2), dtype=np.complex128)
    dut = samples.reshape(3, 1, 1, 2, 2)
    estimate = estimate_repeat_differential_noise(
        reference, dut, full_covariance=True
    )

    centered = samples - samples.mean(axis=0)
    expected = centered.T @ centered.conj() / 2
    assert np.allclose(estimate.covariance, expected)
    assert np.allclose(estimate.covariance, estimate.covariance.conj().T)
    assert not np.allclose(expected, expected.conj())
    assert np.allclose(estimate.variance, np.diag(expected).real)
    assert estimate.diagnostics["pairing"] == "same_index"


def test_dense_covariance_has_an_explicit_size_guard() -> None:
    repeats = np.zeros((2, 1, 1, 2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="dense covariance"):
        estimate_repeat_differential_noise(
            repeats,
            repeats,
            full_covariance=True,
            max_full_covariance_observations=3,
        )


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
    assert 1 <= solution.selected_rank <= solution.available_rank
    assert np.all(np.isfinite(solution.criterion_values))
    assert solution.criterion_ranks.shape == solution.criterion_values.shape


def test_tsvd_rejects_ambiguous_or_unusable_settings() -> None:
    matrix = np.eye(2)
    observations = np.ones(2)
    with pytest.raises(ValueError, match="only valid"):
        tsvd_solve(matrix, observations, method="gcv", rank=1)
    with pytest.raises(ValueError, match="noise_norm"):
        tsvd_solve(matrix, observations, method="discrepancy")
    with pytest.raises(np.linalg.LinAlgError, match="nonzero"):
        tsvd_solve(np.zeros((2, 2)), observations)
