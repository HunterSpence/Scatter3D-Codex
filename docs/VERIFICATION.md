# Verification status and release gate

## Source of truth

The current GitHub Actions run for the exact commit is the source of truth for
automated tests. This file describes what each class of evidence means; it does
not replace CI logs or experiment artifacts.

## CI separation

| Job | Environment | What it may establish |
|---|---|---|
| `pure-python` | CPython 3.11, 3.12, 3.13, and 3.14 | array contracts, CSV round trips, hashes, reference-only alignment, repeat noise, whitening, TSVD, metrics, NPZ pipeline, and pure FEM configuration/tag/checkpoint contracts |
| `heavy-dolfinx` | digest-pinned complex DOLFINx/PETSc | only the heavy tests actually collected on that commit, including any matched-TEM/form tests present there |
| `mpi` | same image, exactly two ranks | a small distributed iterative solve and its recorded ownership-independent global DoF count |

Heavy and MPI jobs fail if zero tests are collected or any selected test skips.
This prevents an unavailable DOLFINx dependency from producing a misleading
green check.

These smoke jobs do not establish PML reflection performance, a calibrated
physical/S-parameter port, manufactured H(curl) convergence, analytical
scattering accuracy, production scaling, or checkpoint/restart behavior. The
executable programs under `validation/` are present in the pinned image, but
their results become evidence only when the exact command and resulting
artifacts are archived.

## Current truth boundary

- The old FEM surface-current RHS is an explicitly uncalibrated load, not a
  matched physical port and not an S-parameter source or receiver.
- Matched single-mode TEM forms and electric-mode power normalization are being
  integrated. Until their exact-commit heavy tests and incident/outgoing modal
  extraction pass, their status is **NOT RUN**, not verified port calibration.
- Real POM/PLA VNA reconstruction is **NOT RUN** because no accepted raw repeat
  bundle and physical-validation artifact have passed this protocol.
- Convergence at 3,000,000 or more global complex DoFs is **NOT RUN**.
- Peak memory at or below 50% of direct on the identical problem is **NOT RUN**.
  If direct factorization cannot run at 3M on available RAM, the 3M ratio remains
  **NOT PROVEN** even if a ratio is measured on a smaller common problem.

Do not infer a pass from the presence of source or test files. A status becomes
`PASSED` only when the exact revision, command, environment, and artifact are
identified.

## What automated tests do not establish

Even when all jobs are green, the following remain unvalidated until their own
artifacts are committed or archived with a release:

- the actual four-antenna CAD/mesh and 5–7 GHz production problem;
- a converged PML reflection, h/p/quadrature, sphere/waveguide, and sensitivity
  finite-difference campaign on that geometry;
- a production-scale distributed benchmark and memory claim;
- VNA calibration quality and cable/thermal stability;
- measured POM and printed-PLA complex material properties;
- stationary, motion, reseat, twin-POM, and known-target controls;
- a successful real POM/PLA reconstruction;
- a nonlinear distorted-Born iterative reconstruction.

The absence of a bundled real dataset is deliberate: private or unreviewed data
must not be turned into a public success claim.

## Release evidence checklist

Before tagging an experimental release, attach or archive:

- [ ] green pure, heavy, and MPI jobs on the release commit;
- [ ] `git diff --check` and a clean signed/tagged revision;
- [ ] source distribution and wheel built from that revision;
- [ ] container digest and complex-PETSc assertion;
- [ ] manufactured/analytic/PML/discretization convergence report;
- [ ] matched-port accepted-power, incident/outgoing modal extraction,
      reciprocity, and independent transmission-line comparison;
- [ ] sensitivity finite-difference and linearization-range report;
- [ ] production scaling/memory report;
- [ ] raw-input hashes and coordinate/manifests;
- [ ] complete VNA calibration and independent verification record;
- [ ] repeat/null/motion/known-target report;
- [ ] frozen reconstruction configuration and negative controls;
- [ ] stated failures, exclusions, and unexecuted checks.

## Claim vocabulary

Use these labels consistently:

- **PASSED:** the registered gate completed successfully on an identified exact
  revision and environment, with its artifact retained;
- **FAILED:** the gate ran and did not meet its registered criterion. For TSVD
  discrepancy selection, the reconstruction artifact remains `FAILED` even if
  `--allow-unmet-discrepancy` suppresses the nonzero exit;
- **NOT RUN:** no executed evidence exists for that exact gate and revision;
- **BLOCKED:** the gate could not run because a named prerequisite or resource
  was unavailable; this is not a pass or a failure;
- **implemented:** code exists; this label alone says nothing about execution;
- **verified:** a `PASSED` result was compared with analytic or independent
  numerical truth;
- **experimentally validated:** the frozen pipeline passed registered physical
  controls and truth metrics.

Never upgrade one label because a visually plausible image was produced.
