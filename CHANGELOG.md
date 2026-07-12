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

### Changed

- Expanded the declared pure-Python CI matrix to Python 3.11 through 3.14 while
  retaining an open-ended `>=3.11` package requirement.
- Added reproducibility material to the source distribution and archived built
  wheel/source artifacts in CI.
- Added static workflow, Compose, Bash, citation, local-link, and secret checks
  ahead of numerical CI gates.

### Security

- Excluded dotenv files, credential/key formats, private measurements, and
  generated numerical artifacts from Git and Docker contexts while retaining
  explicitly placed example fixtures.

### Validation boundary

- Pure numerical and schema paths are covered by automated tests.
- Heavy FEM and MPI paths are required to execute in their dedicated CI jobs.
- The expanded release, container, and CI gates remain unverified until their
  corresponding jobs execute successfully on the remote runner and GitHub.
- No real POM/PLA VNA dataset is distributed, so a successful real-object image
  is **not** claimed.

[Unreleased]: https://github.com/HunterSpence/Scatter3D-Codex/commits/main
