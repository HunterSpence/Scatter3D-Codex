# Changelog

All notable changes will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This development version has not been released. The entries below describe the
current repository state, not completed hardware or scaling acceptance.

### Added

- Clean-room Apache-2.0 project identity.
- Canonical scattering-data contract with explicit receiver/source ordering.
- Reference-only alignment, differential measurement, repeat-noise estimation,
  whitening, and transparent TSVD selection.
- DOLFINx/PETSc Maxwell building blocks with PML, per-port normalization,
  same-matrix multi-right-hand-side reuse, diagnostics, and checkpoint identity.
- Verification-first CLI, measurement runbook, convergence protocol, pinned
  complex DOLFINx container, and separate pure/heavy/MPI CI gates.
- Distributed iterative validation controls for preconditioning side, GMRES
  restart, ASM overlap, local ILU/LU choice, true-residual gates, and
  same-problem memory comparison.
- A separately assembled absorption-shifted iterative preconditioning operator
  `P`, while preserving the physical Maxwell operator `A` and right-hand sides.
- A genuine two-level p=3-to-p=1 Nedelec p-multigrid path with a PEC-masked
  interpolation operator, one-step ASM/MUMPS fine smoothing, and a global
  p=1 MUMPS coarse correction.
- Validation schema `scatter3d.validation.fem_smoke/v2` with source, command,
  image, runtime, cgroup, physical-problem, requested/effective solver, and
  preconditioner provenance plus atomic no-clobber output.
- Immutable eight-run scaling-sweep registration and an independent-per-run Docker
  executor with exact physical/DoF/image/runtime binding, registered wall time,
  no-swap enforcement, host cgroup-v2 peak capture, and write-once manifests.

### Changed

- Expanded the declared pure-Python CI matrix to Python 3.11 through 3.14 while
  retaining an open-ended `>=3.11` package requirement.
- Added reproducibility material to the source distribution and archived built
  wheel/source artifacts in CI.
- Added static workflow, Compose, Bash, citation, local-link, and secret checks
  ahead of numerical CI gates.
- Kept prefixed PETSc options installed through `KSPSetUp` so nested ASM
  subdomain KSP/PC objects consume their requested configuration before the
  temporary options are removed.
- Iterative solves now call `KSPSetOperators(A, P)`, record fine/preconditioner
  matrix metrics separately, and reuse `A` as `P` when the shift is zero.
- Effective PETSc diagnostics now fail closed on typed top-level mismatches and
  retain live nested ASM components, parsed ASM type/overlap, and the raw ASCII
  KSP view.

### Fixed

- Corrected a PETSc option-lifecycle bug in which a requested ASM subdomain LU
  remained the default ILU because nested options were removed before setup.
  A heavy regression now proves nested setup consumes the requested options.

### Security

- Excluded dotenv files, credential/key formats, private measurements, and
  generated numerical artifacts from Git and Docker contexts while retaining
  explicitly placed example fixtures.

### Validation boundary

- Pure numerical and schema paths are covered by automated tests.
- Heavy FEM and MPI paths are required to execute in their dedicated CI jobs.
- Repository/static/package, CPython 3.11–3.14, heavy complex DOLFINx, and
  two-rank MPI CI gates **PASSED** at `c3c1ded`.
- A p=3, 86,103-global-complex-DoF iterative solve with corrected local MUMPS LU
  **PASSED** for two right-hand sides.
- The identical-problem peak-memory-at-most-50%-of-direct gate **FAILED** with a
  summed rank peak-RSS ratio of `0.8263214111`.
- The p=3, 470,928-global-complex-DoF iterative rung **FAILED** its registered
  true-residual gate after both right-hand sides reached 1,000 iterations.
- Convergence at 3,000,000 or more global complex DoFs is **NOT RUN**.
- Small serial and two-rank p=3-to-p=1 p-multigrid correctness artifacts
  **PASSED** at `bee9e9d`; the registered 86k/471k absorption-shift sweep is
  **NOT RUN**. This does not alter the historical `c3c1ded` scaling evidence.
- Real POM/PLA VNA reconstruction is **BLOCKED** because no accepted raw
  measurement/control bundle has been supplied; no successful real-object image
  is claimed.

[Unreleased]: https://github.com/HunterSpence/Scatter3D-Codex/commits/main
