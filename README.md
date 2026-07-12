# Scatter3D-Codex

[![quality](https://github.com/HunterSpence/Scatter3D-Codex/actions/workflows/quality.yml/badge.svg)](https://github.com/HunterSpence/Scatter3D-Codex/actions/workflows/quality.yml)
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
| Data order, hashes, reference/DUT subtraction | Pure unit and end-to-end synthetic tests | Implemented and testable |
| Repeat-floor diagnostics and complex TSVD | Pure deterministic tests | Implemented and testable |
| Maxwell forms, PML, port normalization, checkpoints | Digest-pinned complex DOLFINx heavy tests | Must pass the heavy CI job |
| Two-rank operation | Dedicated MPI tests with zero permitted skips | Must pass the MPI CI job |
| Real POM/PLA object imaging | Archived VNA repeats, nulls, known target, materials, and acceptance report | **Not yet demonstrated** |
| Production-scale convergence | Registered mesh/PML/quadrature sweep on the actual geometry | **Not yet demonstrated** |

Passing software tests proves the software checks they exercise. It does not
prove that a particular fixture, calibration, material model, or linearized
inverse problem contains enough information to image a real object.

## Why this design

Synthetic-to-synthetic reconstruction can succeed while measured data fail
because the same model generates both the sensitivity matrix and observations.
The hardware path additionally contains calibration residuals, cable and thermal
drift, port/angle ordering risk, positioning error, uncertain materials,
unmodelled fixture scattering, limited dynamic range, and approximation error.
Scatter3D-Codex therefore treats the **null and repeatability floor as first-class
inputs**, not as an afterthought applied after a noisy image appears.

## Quick start: pure Python canary

Python 3.11 or 3.12 is supported for the measurement and inverse layers.

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
- A FEM solve is accepted only with a positive PETSc convergence reason and a
  reported true relative residual.

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
  fem/                  DOLFINx/PETSc mesh, forms, PML, ports, solver, checkpoints
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

## License and citation

Code and original documentation are licensed under Apache-2.0. See [LICENSE](LICENSE)
and [CITATION.cff](CITATION.cff). Third-party software in the container retains
its own license. Cite the physical method and calibration literature relevant to
your experiment in addition to this software; a starting bibliography is in
[docs/REFERENCES.md](docs/REFERENCES.md).
