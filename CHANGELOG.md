# Changelog

All notable changes will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Clean-room Apache-2.0 project identity.
- Canonical scattering-data contract with explicit receiver/source ordering.
- Reference-only alignment, differential measurement, repeat-noise estimation,
  whitening, and transparent TSVD selection.
- DOLFINx/PETSc Maxwell building blocks with PML, per-port normalization,
  same-matrix multi-right-hand-side reuse, diagnostics, and checkpoint identity.
- Verification-first CLI, measurement runbook, convergence protocol, pinned
  complex DOLFINx container, and separate pure/heavy/MPI CI gates.

### Validation boundary

- Pure numerical and schema paths are covered by automated tests.
- Heavy FEM and MPI paths are required to execute in their dedicated CI jobs.
- No real POM/PLA VNA dataset is distributed, so a successful real-object image
  is **not** claimed.

[Unreleased]: https://github.com/HunterSpence/Scatter3D-Codex/commits/main
