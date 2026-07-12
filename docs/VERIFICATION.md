# Verification status and release gate

## Source of truth

The current GitHub Actions run for the exact commit is the source of truth for
automated tests. This file describes what each class of evidence means; it does
not replace CI logs or experiment artifacts.

## CI separation

| Job | Environment | What it may establish |
|---|---|---|
| `pure-python` | CPython 3.11 and 3.12 | array contracts, CSV round trips, hashes, reference-only alignment, repeat noise, whitening, TSVD, metrics, NPZ pipeline, and pure FEM configuration/tag/checkpoint contracts |
| `heavy-dolfinx` | digest-pinned complex DOLFINx/PETSc | complex runtime import, live-frequency PML-expression construction, p=3 edge-space construction, and a small direct repeated-RHS solve |
| `mpi` | same image, exactly two ranks | a small distributed iterative solve and its recorded ownership-independent global DoF count |

Heavy and MPI jobs fail if zero tests are collected or any selected test skips.
This prevents an unavailable DOLFINx dependency from producing a misleading
green check.

These smoke jobs do not establish PML reflection performance, absolute port
power normalization, manufactured H(curl) convergence, analytical scattering
accuracy, production scaling, or checkpoint/restart behavior. The executable
programs under `validation/` are present in the pinned image, but their results
become evidence only when the exact command and resulting artifacts are archived.

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
- [ ] sensitivity finite-difference and linearization-range report;
- [ ] production scaling/memory report;
- [ ] raw-input hashes and coordinate/manifests;
- [ ] complete VNA calibration and independent verification record;
- [ ] repeat/null/motion/known-target report;
- [ ] frozen reconstruction configuration and negative controls;
- [ ] stated failures, exclusions, and unexecuted checks.

## Claim vocabulary

Use these labels consistently:

- **implemented:** code exists and passes its unit contract;
- **tested:** the exact path executed in an identified environment;
- **verified:** compared with an analytic/independent numerical truth;
- **experimentally validated:** frozen pipeline passed registered physical
  controls and truth metrics;
- **not run:** no executed evidence exists.

Never upgrade one label because a visually plausible image was produced.
