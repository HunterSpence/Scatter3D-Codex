# Scatter3D-Codex

[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Scatter3D-Codex is a clean-room, verification-first framework for differential
microwave imaging from multiport scattering parameters. It combines explicit
measurement contracts, repeat-noise diagnostics, complex-valued regularized
inversion, and modular Maxwell FEM components for DOLFINx/PETSc.

This is an independent implementation, **not a fork or relicensing of the
original Scatt3D code**. The motivating method is cited, but no implementation
from that repository was copied.

## The honest status

| Capability | Evidence required by this repository | Claim |
|---|---|---|
| Data order, hashes, reference/DUT subtraction | Pure unit and end-to-end synthetic tests on the exact revision | **PASSED** on CPython 3.11–3.14 at `ad9a43b` |
| Repeat-floor diagnostics and complex TSVD | Pure deterministic tests on the exact revision | **PASSED** at `ad9a43b` |
| Manufactured H(curl), Nedelec p=1/2/3 | Three mesh levels per degree in the pinned complex runtime | **PASSED** at `ad9a43b`; archived JSON records orders and residuals |
| Matched TEM boundary and electric-mode power normalization | Digest-pinned complex DOLFINx tests | Software/runtime checks **PASSED**; calibrated incident/outgoing S-parameter extraction and an independent port benchmark are **NOT RUN** |
| Two-rank operation | Dedicated MPI test and iterative repeated-RHS smoke solve with zero permitted skips | **PASSED** at `ad9a43b` for the small 98-DoF correctness case; this is not scaling evidence |
| Real POM/PLA object imaging | Archived VNA repeats, nulls, known target, materials, and acceptance report | **NOT RUN** |
| At least 3,000,000 global complex DoFs | Archived distributed convergence artifact | **NOT RUN** |
| Peak memory at most 50% of direct | Instrumented identical-problem direct/iterative comparison | **NOT RUN** |

Passing software tests proves the software checks they exercise. It does not
prove that a particular fixture, calibration, material model, or linearized
inverse problem contains enough information to image a real object.

The identified evidence is retained by [GitHub Actions run
29206335149](https://github.com/HunterSpence/Scatter3D-Codex/actions/runs/29206335149)
as `scatter3d-heavy-verification` and `scatter3d-mpi-verification` artifacts.

## Why this design

Synthetic-to-synthetic reconstruction can succeed while measured data fail
because the same model generates both the sensitivity matrix and observations.
The hardware path additionally contains calibration residuals, cable and thermal
drift, port/angle ordering risk, positioning error, uncertain materials,
unmodelled fixture scattering, limited dynamic range, and approximation error.
Scatter3D-Codex therefore treats the **null and repeatability floor as first-class
inputs**, not as an afterthought applied after a noisy image appears.

## Quick start: pure Python canary

Python 3.11 through 3.14 are the configured CI targets for the measurement and
inverse layers. A version is supported only when the exact-revision CI job for
that version passes.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python examples/make_synthetic_bundle.py example-output
scatter3d validate example-output/measurement.npz
scatter3d diagnose example-output/measurement.npz
scatter3d invert \
  example-output/measurement.npz \
  example-output/sensitivity.npz \
  example-output/reconstruction.npz \
  --method fixed --rank 4
python -m pytest -m "not heavy and not mpi"
```

The example is a deterministic software canary with an exactly linear synthetic
problem. It is intentionally not presented as experimental validation.

Print the complete file contract at any time:

```bash
scatter3d schema
```

## Complex DOLFINx/PETSc runtime

DOLFINx is not installed from PyPI. The repository pins the official DOLFINx
0.10.0 image by digest and explicitly selects its complex PETSc build:

```bash
docker build -f docker/Dockerfile -t scatter3d-codex:local .
docker run --rm scatter3d-codex:local \
  python3 -m pytest -m "heavy and not mpi" -ra
docker run --rm --ipc=host scatter3d-codex:local \
  mpirun -n 2 python3 -m pytest -m mpi -ra
```

CI rejects a heavy or MPI job that collects no tests or reports any skip. A green
pure-Python job cannot conceal a missing FEM runtime.

## Core contracts

- A scattering sample is `S[angle, frequency, receiver, source]` with
  `S[m, n]` meaning source port `n`, receiver port `m`.
- Repeated measurements add one leading `repeat` axis.
- Flattening is C order: angle, then frequency, then receiver, then source.
- Reference alignment is estimated from reference information only and then
  frozen. The DUT is never independently fitted to the model.
- Differential data use the explicit same-index quantity `S_dut[m,n] -
  S_ref[m,n]`; silent transposition is forbidden.
- Every frequency grid uses hertz; every angle grid uses degrees; every port has
  an explicit label.
- Complex values remain complex through noise estimation, whitening, and
  inversion.
- Measurement and sensitivity archives use exact `complex128`, `float64`,
  Unicode, and `int64` schema dtypes; the loader does not silently cast them.
- Diagnosis reports both aggregate and per-frequency/per-receiver/per-source
  repeat-floor metrics.
- Reconstruction artifacts retain the complete compact SVD spectrum, selector
  ranks and criterion values, status, thresholds, and input/artifact hashes.
- The automatic whitened discrepancy target `sqrt(rows)` is only the expected
  RMS scale under the documented complex-noise convention. If no rank meets it,
  the artifact is `FAILED` and the CLI exits nonzero unless the user explicitly
  requests `--allow-unmet-discrepancy`.
- Existing JSON and reconstruction outputs are not overwritten by default;
  `--force` is explicit and recorded in reconstruction provenance.
- A FEM linear solve is numerically accepted only with a positive PETSc
  convergence reason and a reported true relative residual. This is not, by
  itself, validation of a physical port or an S-parameter.

See [Data schema](docs/DATA_SCHEMA.md) and
[Architecture](docs/ARCHITECTURE.md) for the full contracts.

## Before attempting a real image

Follow the [measurement runbook](docs/MEASUREMENT_RUNBOOK.md). Its minimum ladder
is:

1. pass a nonsymmetric source/receiver software canary;
2. collect at least 20 stationary repeats without touching the fixture;
3. repeat after the intended rotation/motion cycle with no target change;
4. measure a twin-reference null and a deliberately strong known target;
5. demonstrate that the known-target differential is materially above the
   repeat and motion floors;
6. converge mesh, polynomial order, quadrature, PML, port normalization, and the
   S-parameter quantity of interest;
7. reconstruct whitened data with a recorded regularization criterion;
8. attempt the subtle PLA inclusion only after all earlier gates pass.

If the target differential is below the repeat/motion floor, changing the TSVD
rank cannot recover information that was not measured.

## Repository map

```text
src/scatter3d/
  measurement.py       coordinate-explicit S-parameter datasets and alignment
  inverse.py           repeat noise, whitening, and transparent complex TSVD
  metrics.py           volume-weighted image metrics
  provenance.py        stable hashing and manifests
  pipeline.py          checked NPZ-to-reconstruction workflow
  fem/                  DOLFINx/PETSc mesh, forms, PML, port research, solver, checkpoints
tests/                  pure, heavy, and MPI verification
examples/               deterministic software canaries
docs/                   experiment, convergence, schema, and evidence guides
docker/                 digest-pinned complex numerical runtime
```

## Documentation

- [Measurement runbook](docs/MEASUREMENT_RUNBOOK.md)
- [Convergence protocol](docs/CONVERGENCE_PROTOCOL.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Data schema](docs/DATA_SCHEMA.md)
- [CLI workflows](docs/CLI.md)
- [Verification status and release gate](docs/VERIFICATION.md)
- [Primary references](docs/REFERENCES.md)

## Scope boundaries

Scatter3D-Codex does not automate RF emission or control a VNA. It does not
currently implement a trusted nonlinear distorted-Born iterative reconstruction.
The initial inverse layer addresses the documented small-perturbation linear
model; use its residual and null controls to decide when that approximation is
not credible.

The historical FEM surface-current load is explicitly uncalibrated: it has no
matched termination, power-wave reference, or S-parameter meaning. A matched
single-mode TEM boundary and power normalization are being integrated, but the
heavy tests and incident/outgoing modal extraction required for calibrated
S-parameters must pass before that path is described as physical validation.

## License and citation

Code and original documentation are licensed under Apache-2.0. See [LICENSE](LICENSE)
and [CITATION.cff](CITATION.cff). Third-party software in the container retains
its own license. Cite the physical method and calibration literature relevant to
your experiment in addition to this software; a starting bibliography is in
[docs/REFERENCES.md](docs/REFERENCES.md).
