"""Noise estimation, diagonal whitening, and complex TSVD inversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .measurement import ScatteringDataset, assert_compatible


def _readonly(array: np.ndarray, dtype: np.dtype[Any] | type | None = None) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class DiagonalNoiseModel:
    """Per-observation complex-noise standard deviations.

    The model stores only an O(N) vector.  Whitening scales rows directly and
    never constructs a dense diagonal matrix.
    """

    standard_deviation: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sigma = _readonly(self.standard_deviation, np.float64)
        if sigma.ndim != 1 or sigma.size == 0:
            raise ValueError("standard_deviation must be a non-empty vector")
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("standard_deviation must be finite and strictly positive")
        object.__setattr__(self, "standard_deviation", sigma)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @classmethod
    def from_variance(
        cls,
        variance: np.ndarray,
        *,
        relative_floor: float = 1.0e-12,
        absolute_floor: float = 0.0,
    ) -> DiagonalNoiseModel:
        """Create a model with a scale-aware variance floor.

        If every observed variance is zero, ``absolute_floor`` must be supplied;
        silently using a floor at unit scale would overwhelm small S-parameters.
        """

        values = np.asarray(variance, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("variance must be a non-empty vector")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("variance must be finite and non-negative")
        if relative_floor < 0.0 or not np.isfinite(relative_floor):
            raise ValueError("relative_floor must be finite and non-negative")
        if absolute_floor < 0.0 or not np.isfinite(absolute_floor):
            raise ValueError("absolute_floor must be finite and non-negative")

        positive = values[values > 0.0]
        data_scale = float(np.median(positive)) if positive.size else 0.0
        floor = max(float(absolute_floor), float(relative_floor) * data_scale)
        if floor <= 0.0 and np.any(values == 0.0):
            raise ValueError(
                "zero variances require a positive absolute_floor when no positive "
                "repeat variance establishes a data scale"
            )
        floored = np.maximum(values, floor)
        return cls(
            standard_deviation=np.sqrt(floored),
            diagnostics={
                "variance_floor": floor,
                "relative_floor": relative_floor,
                "absolute_floor": absolute_floor,
                "floored_observations": int(np.count_nonzero(values < floor)),
            },
        )

    def whiten_vector(self, observations: np.ndarray) -> np.ndarray:
        vector = np.asarray(observations)
        if vector.ndim != 1 or vector.size != self.standard_deviation.size:
            raise ValueError("observation vector does not match noise model")
        return vector / self.standard_deviation

    def whiten_matrix(self, matrix: np.ndarray) -> np.ndarray:
        operator = np.asarray(matrix)
        if operator.ndim != 2 or operator.shape[0] != self.standard_deviation.size:
            raise ValueError("matrix rows do not match noise model")
        return operator / self.standard_deviation[:, None]


@dataclass(frozen=True, slots=True)
class RepeatNoiseEstimate:
    """Noise statistics from paired DUT-minus-reference repeats."""

    mean_differential: np.ndarray
    sample_variance: np.ndarray
    repeat_count: int
    observation_shape: tuple[int, int, int, int]
    sample_covariance: np.ndarray | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mean = _readonly(self.mean_differential, np.complex128)
        variance = _readonly(self.sample_variance, np.float64)
        expected = int(np.prod(self.observation_shape, dtype=np.int64))
        if self.repeat_count < 2:
            raise ValueError("at least two paired repeats are required")
        if len(self.observation_shape) != 4 or any(n <= 0 for n in self.observation_shape):
            raise ValueError("observation_shape must be [angle, frequency, receiver, source]")
        if mean.shape != (expected,) or variance.shape != (expected,):
            raise ValueError("repeat statistics do not match observation_shape")
        if not np.all(np.isfinite(mean.real)) or not np.all(np.isfinite(mean.imag)):
            raise ValueError("mean differential must be finite")
        if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
            raise ValueError("variance must be finite and non-negative")

        covariance = self.sample_covariance
        if covariance is not None:
            covariance = _readonly(covariance, np.complex128)
            if covariance.shape != (expected, expected):
                raise ValueError("covariance shape does not match observations")
            if not np.allclose(covariance, covariance.conj().T, rtol=1e-11, atol=1e-14):
                raise ValueError("complex covariance must be Hermitian")
            if not np.allclose(
                np.diag(covariance).real, variance, rtol=1e-11, atol=1e-14
            ):
                raise ValueError("covariance diagonal does not match variance")
        object.__setattr__(self, "mean_differential", mean)
        object.__setattr__(self, "sample_variance", variance)
        object.__setattr__(self, "sample_covariance", covariance)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def mean_variance(self) -> np.ndarray:
        """Variance of ``mean_differential`` under independent paired repeats."""

        return _readonly(self.sample_variance / self.repeat_count, np.float64)

    @property
    def mean_covariance(self) -> np.ndarray | None:
        """Covariance of ``mean_differential`` when dense covariance was requested."""

        if self.sample_covariance is None:
            return None
        return _readonly(self.sample_covariance / self.repeat_count, np.complex128)

    def mean_diagonal_model(
        self,
        *,
        relative_floor: float = 1.0e-12,
        absolute_floor: float = 0.0,
    ) -> DiagonalNoiseModel:
        return DiagonalNoiseModel.from_variance(
            self.mean_variance,
            relative_floor=relative_floor,
            absolute_floor=absolute_floor,
        )


def _repeat_stack(
    repeats: np.ndarray | Sequence[ScatteringDataset], *, name: str
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if isinstance(repeats, np.ndarray):
        stack = np.asarray(repeats, dtype=np.complex128)
        if stack.ndim != 5:
            raise ValueError(
                f"{name} must have shape [repeat, angle, frequency, receiver, source]"
            )
        if stack.shape[3] != stack.shape[4]:
            raise ValueError(f"{name} receiver/source port counts differ")
        return stack, tuple(int(n) for n in stack.shape[1:])  # type: ignore[return-value]

    datasets = tuple(repeats)
    if not datasets:
        raise ValueError(f"{name} must contain at least two repeats")
    if not all(isinstance(item, ScatteringDataset) for item in datasets):
        raise TypeError(f"{name} must contain ScatteringDataset objects")
    first = datasets[0]
    for index, item in enumerate(datasets[1:], start=1):
        assert_compatible(first, item, first_name=f"{name}[0]", second_name=f"{name}[{index}]")
    return np.stack([item.s for item in datasets]), first.shape


def estimate_repeat_differential_noise(
    reference_repeats: np.ndarray | Sequence[ScatteringDataset],
    dut_repeats: np.ndarray | Sequence[ScatteringDataset],
    *,
    full_covariance: bool = False,
    max_full_covariance_observations: int = 4096,
) -> RepeatNoiseEstimate:
    """Estimate covariance from paired, same-index repeat differentials.

    Repeat ``i`` of the reference is subtracted from repeat ``i`` of the DUT.
    Pairing preserves drift common to the acquisition schedule.  The diagonal
    variance is always computed; dense covariance is opt-in and size-guarded.
    """

    reference, reference_shape = _repeat_stack(reference_repeats, name="reference_repeats")
    dut, dut_shape = _repeat_stack(dut_repeats, name="dut_repeats")
    if reference.shape != dut.shape or reference_shape != dut_shape:
        raise ValueError("reference and DUT repeat stacks must have identical shapes")
    repeat_count = reference.shape[0]
    if repeat_count < 2:
        raise ValueError("at least two paired repeats are required")
    if not np.all(np.isfinite(reference.real)) or not np.all(np.isfinite(reference.imag)):
        raise ValueError("reference repeats must be finite")
    if not np.all(np.isfinite(dut.real)) or not np.all(np.isfinite(dut.imag)):
        raise ValueError("DUT repeats must be finite")

    samples = (dut - reference).reshape(repeat_count, -1)
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    variance = np.sum(np.abs(centered) ** 2, axis=0).real / (repeat_count - 1)

    covariance: np.ndarray | None = None
    observations = samples.shape[1]
    if full_covariance:
        if max_full_covariance_observations <= 0:
            raise ValueError("max_full_covariance_observations must be positive")
        if observations > max_full_covariance_observations:
            raise ValueError(
                f"dense covariance for {observations} observations is disabled; "
                "use diagonal whitening or raise the explicit safety limit"
            )
        if repeat_count <= observations:
            raise ValueError(
                "dense covariance is rank-deficient when paired repeat_count is not "
                "greater than observation_count; use diagonal whitening or collect "
                "more independent paired repeats"
            )
        # Each row is a repeat. This orientation produces E[x x^H], not its
        # elementwise conjugate, for a complex random column vector x.
        covariance = centered.T @ centered.conj() / (repeat_count - 1)

    positive = variance[variance > 0.0]
    return RepeatNoiseEstimate(
        mean_differential=mean,
        sample_variance=variance,
        repeat_count=repeat_count,
        observation_shape=reference_shape,
        sample_covariance=covariance,
        diagnostics={
            "pairing": "same_index",
            "observation_count": observations,
            "minimum_variance": float(np.min(variance)),
            "median_positive_variance": float(np.median(positive)) if positive.size else 0.0,
            "maximum_variance": float(np.max(variance)),
            "full_covariance_computed": full_covariance,
        },
    )


def whiten_system(
    matrix: np.ndarray,
    observations: np.ndarray,
    noise: DiagonalNoiseModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten A and b together using O(N) diagonal-noise storage."""

    return noise.whiten_matrix(matrix), noise.whiten_vector(observations)


@dataclass(frozen=True, slots=True)
class TSVDSolution:
    """A TSVD result with enough diagnostics to audit rank selection."""

    x: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    singular_values: np.ndarray
    selected_rank: int
    available_rank: int
    method: str
    matrix_shape: tuple[int, int]
    residual_norm: float
    relative_residual: float
    solution_norm: float
    selected_condition_number: float | None
    criterion_ranks: np.ndarray
    criterion_values: np.ndarray
    singular_value_threshold: float
    target_met: bool | None = None
    noise_norm: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _readonly(self.x, np.complex128))
        object.__setattr__(self, "predicted", _readonly(self.predicted, np.complex128))
        object.__setattr__(self, "residual", _readonly(self.residual, np.complex128))
        object.__setattr__(self, "singular_values", _readonly(self.singular_values, np.float64))
        object.__setattr__(self, "criterion_ranks", _readonly(self.criterion_ranks, np.int64))
        object.__setattr__(self, "criterion_values", _readonly(self.criterion_values, np.float64))

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "method": self.method,
                "matrix_shape": self.matrix_shape,
                "selected_rank": self.selected_rank,
                "available_rank": self.available_rank,
                "residual_norm": self.residual_norm,
                "relative_residual": self.relative_residual,
                "solution_norm": self.solution_norm,
                "selected_condition_number": self.selected_condition_number,
                "singular_value_threshold": self.singular_value_threshold,
                "target_met": self.target_met,
                "noise_norm": self.noise_norm,
            }
        )


def tsvd_solve(
    matrix: np.ndarray,
    observations: np.ndarray,
    *,
    method: str = "gcv",
    rank: int | None = None,
    noise_norm: float | None = None,
    energy_fraction: float = 0.999,
    rcond: float | None = None,
) -> TSVDSolution:
    """Solve a complex linear inverse problem with transparent TSVD selection.

    Parameters
    ----------
    method:
        ``"fixed"`` uses ``rank``; ``"gcv"`` minimizes the generalized
        cross-validation curve; ``"discrepancy"`` chooses the smallest rank
        whose residual is at most ``noise_norm``; ``"energy"`` retains the
        requested fraction of squared singular-value energy.
    """

    a = np.asarray(matrix)
    b = np.asarray(observations)
    if a.ndim != 2 or min(a.shape) == 0:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    if b.ndim != 1 or b.size != a.shape[0]:
        raise ValueError("observations must be a vector matching matrix rows")
    if not np.all(np.isfinite(a.real)) or not np.all(np.isfinite(a.imag)):
        raise ValueError("matrix must be finite")
    if not np.all(np.isfinite(b.real)) or not np.all(np.isfinite(b.imag)):
        raise ValueError("observations must be finite")
    if method not in {"fixed", "gcv", "discrepancy", "energy"}:
        raise ValueError("method must be fixed, gcv, discrepancy, or energy")
    if rcond is not None and (not np.isfinite(rcond) or rcond < 0.0):
        raise ValueError("rcond must be finite and non-negative")
    if method != "fixed" and rank is not None:
        raise ValueError("rank is only valid when method='fixed'")

    dtype = np.result_type(a.dtype, b.dtype, np.complex128)
    a = np.asarray(a, dtype=dtype)
    b = np.asarray(b, dtype=dtype)
    u, singular_values, vh = np.linalg.svd(a, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        raise np.linalg.LinAlgError("matrix has no nonzero singular values")
    relative_cutoff = (
        float(rcond)
        if rcond is not None
        else np.finfo(singular_values.dtype).eps * max(a.shape)
    )
    threshold = relative_cutoff * float(singular_values[0])
    available = int(np.count_nonzero(singular_values > threshold))
    if available == 0:
        raise np.linalg.LinAlgError("no singular values exceed the requested threshold")

    beta = u.conj().T @ b
    projected_energy = float(np.sum(np.abs(beta) ** 2))
    outside_energy = max(0.0, float(np.vdot(b, b).real) - projected_energy)
    ranks = np.arange(1, available + 1, dtype=np.int64)
    residual_squared = np.asarray(
        [outside_energy + float(np.sum(np.abs(beta[k:]) ** 2)) for k in ranks],
        dtype=np.float64,
    )
    residual_curve = np.sqrt(np.maximum(residual_squared, 0.0))
    criterion_ranks = ranks
    target_met: bool | None = None

    if method == "fixed":
        if rank is None or isinstance(rank, bool) or int(rank) != rank:
            raise ValueError("method='fixed' requires an integer rank")
        selected = int(rank)
        if not 1 <= selected <= available:
            raise ValueError(f"rank must lie between 1 and {available}")
        criterion_values = residual_curve
    elif method == "energy":
        if not np.isfinite(energy_fraction) or not 0.0 < energy_fraction <= 1.0:
            raise ValueError("energy_fraction must lie in (0, 1]")
        energy_curve = np.cumsum(singular_values[:available] ** 2)
        energy_curve /= energy_curve[-1]
        selected = int(np.searchsorted(energy_curve, energy_fraction) + 1)
        criterion_values = energy_curve
    elif method == "discrepancy":
        if noise_norm is None or not np.isfinite(noise_norm) or noise_norm < 0.0:
            raise ValueError("method='discrepancy' requires a finite non-negative noise_norm")
        # Rank zero is a meaningful discrepancy solution: if the unmodelled
        # observation already lies inside the registered noise ball, fitting a
        # singular vector would manufacture structure from a null experiment.
        criterion_ranks = np.arange(0, available + 1, dtype=np.int64)
        criterion_values = np.concatenate(
            (np.asarray([float(np.linalg.norm(b))]), residual_curve)
        )
        meeting = np.flatnonzero(criterion_values <= noise_norm)
        target_met = meeting.size > 0
        selected = int(criterion_ranks[meeting[0]]) if target_met else available
    else:
        # GCV also needs the no-fit candidate. Otherwise a null experiment is
        # forced to retain at least one singular direction.
        criterion_ranks = np.arange(0, available + 1, dtype=np.int64)
        gcv_residual_squared = np.concatenate(
            (np.asarray([float(np.vdot(b, b).real)]), residual_squared)
        )
        denominator = (a.shape[0] - criterion_ranks).astype(np.float64) ** 2
        criterion_values = np.divide(
            gcv_residual_squared,
            denominator,
            out=np.full_like(gcv_residual_squared, np.inf),
            where=denominator > 0.0,
        )
        finite = np.flatnonzero(np.isfinite(criterion_values))
        selected = (
            int(criterion_ranks[finite[np.argmin(criterion_values[finite])]])
            if finite.size
            else available
        )

    if selected == 0:
        x = np.zeros(a.shape[1], dtype=dtype)
    else:
        coefficients = beta[:selected] / singular_values[:selected]
        x = vh[:selected, :].conj().T @ coefficients
    predicted = a @ x
    residual = b - predicted
    residual_norm = float(np.linalg.norm(residual))
    observation_norm = float(np.linalg.norm(b))
    relative_residual = (
        residual_norm / observation_norm
        if observation_norm > 0.0
        else (0.0 if residual_norm == 0.0 else float("inf"))
    )
    condition = (
        None if selected == 0 else float(singular_values[0] / singular_values[selected - 1])
    )
    return TSVDSolution(
        x=x,
        predicted=predicted,
        residual=residual,
        singular_values=singular_values,
        selected_rank=selected,
        available_rank=available,
        method=method,
        matrix_shape=(int(a.shape[0]), int(a.shape[1])),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        solution_norm=float(np.linalg.norm(x)),
        selected_condition_number=condition,
        criterion_ranks=criterion_ranks,
        criterion_values=criterion_values,
        singular_value_threshold=threshold,
        target_met=target_met,
        noise_norm=float(noise_norm) if noise_norm is not None else None,
    )


__all__ = [
    "DiagonalNoiseModel",
    "RepeatNoiseEstimate",
    "TSVDSolution",
    "estimate_repeat_differential_noise",
    "tsvd_solve",
    "whiten_system",
]
