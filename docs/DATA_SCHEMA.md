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
| `reference_s` | complex, `[Rr,A,F,P,P]` | reference repeats |
| `dut_s` | complex, `[Rd,A,F,P,P]` | DUT repeats |
| `frequencies_hz` | float64, `[F]` | finite, positive, strictly increasing |
| `angles_deg` | float64, `[A]` | finite and strictly increasing |
| `port_labels` | Unicode, `[P]` | nonempty and unique |

The repeat axis may be omitted only for a single measurement. Reference and DUT
may have different repeat counts but must share all measurement axes exactly.
When their repeat counts are equal and at least two, equal repeat indices declare
an acquisition pair for the default mean-differential noise estimate.
No frequency interpolation or port relabelling occurs during load.

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

The CLI writes:

| Key | Meaning |
|---|---|
| `estimate` | complex permittivity-contrast vector |
| `selected_rank` | retained TSVD rank |
| `method` | fixed, GCV, discrepancy, or energy |
| `residual_norm` | `||A x - b||_2` on used rows |
| `relative_residual` | residual divided by `||b||_2` |
| `solve_residual_norm` | residual in the whitened solve space, or raw space when unwhitened |
| `channel_mode`, `row_order` | explicit row selection and canonical ordering |
| `whitening_used`, `whitening_reason` | whether/why paired-repeat whitening ran |
| `paired_repeats` | paired repeat count, or `-1` when unavailable |
| `noise_standard_deviation` | selected-row standard deviation of the mean differential |
| `noise_model_sha256` | hash of that standard-deviation vector |
| `noise_norm_used` | discrepancy target in solve space; NaN when not applicable |
| `bundle_sha256` | exact measurement NPZ hash |
| `sensitivity_sha256` | exact sensitivity NPZ hash |
| `row_indices` | canonical rows used |

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
