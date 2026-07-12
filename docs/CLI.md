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

Exit code `0` means this contract passed. Exit code `2` means the input or
requested workflow was invalid. It does not mean the measurement is physically
adequate.

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
- number of exact unique angle traces, which catches array-alias failures.

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

The pipeline always forms the same-index differential and checks frequency,
angle, port, and sensitivity-row identity before solving. `--channel-mode`
selects both `A` and `b`; it cannot reorder either.

`--whitening auto` is the default. With equal reference/DUT repeat counts of at
least two, repeat `i` is paired with repeat `i`, the paired-difference variance
is divided by the repeat count to obtain variance of the mean, and `A` and `b`
are scaled together. Use `--whitening required` to fail if that evidence is
unavailable, or `--whitening off` for a registered unweighted control. After
whitening, discrepancy mode defaults to `sqrt(rows)` unless `--noise-norm`
explicitly supplies a solve-space target.

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
