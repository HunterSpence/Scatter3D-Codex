# CLI workflows

The CLI operates on offline files only. It does not configure or trigger a VNA.

## Inspect the contract

```bash
scatter3d schema
```

This is the quickest way to confirm required keys, units, axes, and schema
versions from the installed revision.

## Validate an input bundle

```bash
scatter3d validate data/processed/measurement.npz \
  --json results/measurement-validation.json
```

Validation checks schema version, complex dtype, finite values, array shapes,
frequency and angle coordinates, port labels, and reference/DUT compatibility.
The report contains the exact file SHA-256 and row count.

The NPZ boundary is intentionally strict: measurement tensors must be
`complex128`, coordinate arrays must be `float64`, port labels must be a NumPy
Unicode vector, sensitivity rows must be `complex128`, and optional row indices
must be `int64`. Inputs are rejected rather than silently cast.

Exit code `0` means this contract passed. Exit code `2` means the input or
requested workflow was invalid. It does not mean the measurement is physically
adequate. Existing JSON reports are protected by default; pass `--force` only
when replacing a registered artifact is intentional.

## Diagnose repeatability before imaging

```bash
scatter3d diagnose data/processed/measurement.npz \
  --json results/measurement-diagnostics.json
```

The report includes:

- reference and DUT repeat counts;
- RMS mean differential;
- combined repeat-noise RMS when both sides have at least two repeats;
- signal-to-repeat-noise ratio and dB value;
- reciprocity residuals as diagnostics, not automatic corrections;
- number of exact unique angle traces, which catches array-alias failures; and
- one diagnostic record for every frequency/receiver/source channel, including
  indices, labels, differential RMS, repeat-noise RMS, and signal-to-noise
  ratio when repeat evidence exists.

A bundle with one repeat validates structurally but cannot estimate a repeat
floor. Do not promote that file to a real-image result.

## Reconstruct

```bash
scatter3d invert \
  data/processed/measurement.npz \
  results/sensitivity.npz \
  results/reconstruction.npz \
  --channel-mode transmission \
  --method discrepancy \
  --whitening required \
  --json results/reconstruction.json
```

Rank-selection modes:

- `fixed --rank K`: deterministic canaries and registered comparisons;
- `gcv`: data-driven selection when no defensible noise norm exists;
- `discrepancy --noise-norm ETA`: smallest rank meeting the supplied residual
  target, preferred when repeat data justify `ETA`;
- `energy --energy-fraction Q`: retains a registered singular-value energy
  fraction; this is a matrix heuristic, not a noise model.

`--rank` is accepted only with `--method fixed`; `--noise-norm` only with
`--method discrepancy`; and `--energy-fraction` only with `--method energy`.
Energy selection defaults to `0.999` only when energy mode is selected and no
fraction is supplied.

The pipeline always forms the same-index differential and checks frequency,
angle, port, and sensitivity-row identity before solving. `--channel-mode`
selects both `A` and `b`; it cannot reorder either.

`--whitening auto` is the default. With equal reference/DUT repeat counts of at
least two, repeat `i` is paired with repeat `i`, the paired-difference variance
is divided by the repeat count to obtain variance of the mean, and `A` and `b`
are scaled together. Use `--whitening required` to fail if that evidence is
unavailable, or `--whitening off` for a registered unweighted control. After
whitening, discrepancy mode defaults to `sqrt(rows)` unless `--noise-norm`
explicitly supplies a solve-space target. This default is only the expected RMS
scale when each whitened complex row obeys `E|z_i|^2 = 1`; it is not a confidence
bound or proof that the noise model is correct.

`--noise-relative-floor` and `--noise-absolute-floor` control the variance floor
before diagonal whitening. Both must be finite and non-negative. The relative
floor is multiplied by the median positive variance; if all estimated variances
are zero, a positive absolute floor is required for whitening instead of
inventing a unit-scale floor.

Every reconstruction NPZ contains the full compact singular-value spectrum,
the selector's candidate ranks and criterion curve, numerical-rank threshold,
selected and available ranks, residuals in raw and solve space, input hashes,
and `PASSED`/`FAILED` status. For discrepancy selection, failure to meet the
target writes a `FAILED` artifact and returns exit code `1`. The explicit
`--allow-unmet-discrepancy` option changes that exit code to `0`; it does not
change the artifact status or make the target met.

The reconstruction NPZ and optional JSON report are no-clobber by default.
`--force` authorizes replacement of both and is recorded as
`overwrite_requested=true` in the reconstruction artifact. Validation and
diagnosis JSON reports follow the same no-clobber default.

## End-to-end software canary

```bash
python examples/make_synthetic_bundle.py example-output
scatter3d validate example-output/measurement.npz
scatter3d diagnose example-output/measurement.npz
scatter3d invert example-output/measurement.npz \
  example-output/sensitivity.npz example-output/reconstruction.npz \
  --method fixed --rank 4
```

This should recover the generated linear truth to floating-point accuracy. It
tests packaging, schemas, row order, subtraction, and TSVD. It does not exercise
FEM field accuracy, VNA calibration, model mismatch, or Born-approximation error.

## Reproducible batch pattern

```bash
set -euo pipefail
scatter3d validate "$BUNDLE" --json "$RUN/validation.json"
scatter3d diagnose "$BUNDLE" --json "$RUN/diagnostics.json"
scatter3d invert "$BUNDLE" "$SENSITIVITY" "$RUN/reconstruction.npz" \
  --method discrepancy --noise-norm "$NOISE_NORM" \
  --json "$RUN/reconstruction.json"
sha256sum "$BUNDLE" "$SENSITIVITY" "$RUN/reconstruction.npz" > "$RUN/SHA256SUMS"
```

Archive the Git revision, container digest, FEM manifest, raw-export hashes, and
experiment log with this directory. Never overwrite an earlier run in place.

## Exit codes

- `0`: command completed, and any discrepancy target was met or an explicit
  `--allow-unmet-discrepancy` override was supplied;
- `1`: discrepancy reconstruction completed but no available TSVD rank met the
  target;
- `2`: invalid arguments, schema/input error, missing file, or refused overwrite.
