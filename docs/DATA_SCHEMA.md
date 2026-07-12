# Data schema

## Scattering convention

All in-memory scattering data use:

```text
S[angle, frequency, receiver, source]
```

`S[..., m, n]` is the wave received at port `m` when port `n` is the source.
The corresponding full-vector row is

```text
row = (((angle * F) + frequency) * P + receiver) * P + source
```

where `F` is the frequency count and `P` is the port count. This is NumPy C
order. A nonsymmetric sentinel test is mandatory because reciprocal or diagonal
test data cannot reveal a receiver/source transpose.

Do not describe this ordering as “Touchstone order.” The official Touchstone
format has its own matrix and special two-port conventions. Convert explicitly
and preserve the original export hash.

## Measurement NPZ: `scatter3d.measurement.v1`

NPZ files are loaded with `allow_pickle=False`.

| Key | dtype and shape | Requirement |
|---|---|---|
| `schema_version` | scalar Unicode | exactly `scatter3d.measurement.v1` |
| `reference_s` | complex128, `[Rr,A,F,P,P]` | reference repeats |
| `dut_s` | complex128, `[Rd,A,F,P,P]` | DUT repeats |
| `frequencies_hz` | float64, `[F]` | finite, positive, strictly increasing |
| `angles_deg` | float64, `[A]` | finite and strictly increasing |
| `port_labels` | Unicode, `[P]` | nonempty and unique |

The repeat axis may be omitted only for a single measurement. Reference and DUT
may have different repeat counts but must share all measurement axes exactly.
When their repeat counts are equal and at least two, equal repeat indices declare
an acquisition pair for the default mean-differential noise estimate.
No frequency interpolation or port relabelling occurs during load.
The listed dtypes are exact. In particular, complex64 measurement tensors,
integer coordinate arrays, byte-string labels, and object arrays are rejected
rather than promoted or decoded implicitly.

Example:

```python
import numpy as np
from scatter3d.pipeline import SCHEMA_VERSION

np.savez_compressed(
    "measurement.npz",
    schema_version=np.array(SCHEMA_VERSION),
    reference_s=reference_repeats,  # [repeat, angle, frequency, rx, tx]
    dut_s=dut_repeats,
    frequencies_hz=frequencies_hz,
    angles_deg=angles_deg,
    port_labels=np.array(["P1", "P2", "P3", "P4"]),
)
```

## Sensitivity NPZ: `scatter3d.sensitivity.v1`

| Key | dtype and shape | Requirement |
|---|---|---|
| `schema_version` | scalar Unicode | exactly `scatter3d.sensitivity.v1` |
| `A` | complex128, `[M,V]` | sensitivity matrix |
| `frequencies_hz` | float64, `[F]` | exact measurement-grid match |
| `angles_deg` | float64, `[A]` | exact measurement-grid match |
| `port_labels` | Unicode, `[P]` | exact measurement-label match |
| `row_indices` | int64, `[M]` | optional full-vector rows represented by `A` |

Without `row_indices`, `A` must contain all `A*F*P*P` rows in canonical C order.
With `row_indices`, indices must be unique and in range. Channel filtering is
applied to both `A` and `b` through those indices.

The current schema assumes one sensitivity value per reconstruction voxel. A
production file should be accompanied by a manifest recording voxel coordinates,
volumes, mesh hash, material background, field normalization, quadrature, and
the precise sensitivity equation. These fields are not guessed by the CLI.

## Reconstruction NPZ: `scatter3d.reconstruction.v1`

The CLI writes the complete audit artifact below. NumPy scalar strings use a
Unicode dtype; hashes are lowercase SHA-256 strings.

| Key | dtype and shape | Meaning |
|---|---|---|
| `schema_version` | scalar Unicode | exactly `scatter3d.reconstruction.v1` |
| `estimate` | complex128, `[V]` | reconstructed complex contrast vector |
| `status` | scalar Unicode | `PASSED`, or `FAILED` when a discrepancy target was unmet |
| `requested_rank` | scalar int64 | fixed rank, or `-1` when not requested |
| `selected_rank` | scalar int64 | retained TSVD rank; discrepancy/GCV may select rank zero |
| `available_rank` | scalar int64 | singular values strictly above the recorded numerical threshold |
| `method` | scalar Unicode | `fixed`, `gcv`, `discrepancy`, or `energy` |
| `channel_mode` | scalar Unicode | `all`, `transmission`, or `reflection` |
| `row_order` | scalar Unicode | `C:[angle,frequency,receiver,source]` |
| `residual_norm` | scalar float64 | `||A x - b||_2` on the selected unwhitened rows |
| `relative_residual` | scalar float64 | unwhitened residual divided by `||b||_2` |
| `solve_residual_norm` | scalar float64 | residual in whitened solve space, or raw space when unwhitened |
| `solution_norm` | scalar float64 | `||x||_2` |
| `selected_condition_number` | scalar float64 | retained-spectrum condition estimate; NaN when undefined |
| `singular_value_threshold` | scalar float64 | numerical-rank cutoff |
| `singular_values` | float64, `[min(M,V)]` | full compact singular-value spectrum returned by the SVD |
| `singular_values_sha256` | scalar Unicode | deterministic array hash of `singular_values` |
| `criterion_ranks` | int64, `[K]` | selector candidate ranks; GCV/discrepancy include rank zero |
| `criterion_values` | float64, `[K]` | residual, GCV, discrepancy-residual, or cumulative-energy curve |
| `criterion_sha256` | scalar Unicode | combined deterministic hash of ranks and criterion values |
| `selection_target_met` | scalar int8 | `-1` not applicable, `0` unmet, `1` met |
| `whitening_used` | scalar bool | whether paired-repeat diagonal whitening was applied |
| `whitening_reason` | scalar Unicode | explicit use, disablement, or unavailable-evidence reason |
| `paired_repeats` | scalar int64 | paired repeat count, or `-1` when unavailable |
| `noise_standard_deviation` | float64, `[M_used]` or `[0]` | selected-row standard deviation of the mean differential |
| `noise_model_sha256` | scalar Unicode | hash of that vector, or an empty string when unwhitened |
| `noise_norm_used` | scalar float64 | discrepancy target in solve space; NaN when not applicable |
| `noise_norm_basis` | scalar Unicode | `user_supplied`, `whitened_expected_rms_sqrt_rows`, or empty |
| `energy_fraction_used` | scalar float64 | requested/default fraction; NaN outside energy mode |
| `noise_relative_floor` | scalar float64 | registered relative variance floor |
| `noise_absolute_floor` | scalar float64 | registered absolute variance floor |
| `overwrite_requested` | scalar bool | whether the run explicitly authorized clobbering via `--force` |
| `bundle_sha256` | scalar Unicode | exact measurement NPZ content hash |
| `sensitivity_sha256` | scalar Unicode | exact sensitivity NPZ content hash |
| `row_indices` | int64, `[M_used]` | canonical measurement rows actually used |

The automatic discrepancy target `sqrt(M_used)` is emitted only after paired
repeat whitening and is an expected RMS heuristic under `E|z_i|^2=1`, not a
confidence bound. If no candidate rank meets the target, the NPZ still records
the best available full-rank result but sets `status=FAILED` and
`selection_target_met=0`. The CLI exits `1` unless the caller explicitly uses
`--allow-unmet-discrepancy`; that override does not rewrite the status.

Reconstruction and JSON outputs are created atomically and are no-clobber by
default. `--force` is required to replace an existing path, and the NPZ records
that request in `overwrite_requested`.

Complex estimates are not silently converted to real. Interpret the imaginary
part according to the time convention and material model used to build `A`.

## CSV API

`read_scattering_csv` and `write_scattering_csv` use explicit receiver and source
columns and return `ScatteringDataset`. Convert every vendor export with a
documented map and verify one intentionally nonsymmetric sample before using a
large sweep. Never infer whether `S12` or `S21` comes first from a rectangular
numeric matrix alone.

## Coordinate equality is exact by design

The pipeline rejects even a small frequency mismatch between `A` and `b`.
Resampling complex VNA data can alter phase and covariance and therefore requires
a separate, reviewed preprocessing step that creates a new file and provenance
record. A tolerance-based implicit match is not safe at the inversion boundary.
