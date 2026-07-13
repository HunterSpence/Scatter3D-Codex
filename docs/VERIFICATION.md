# Verification status and release gate

## Source of truth

The current GitHub Actions run for the exact commit is the source of truth for
automated tests. This file describes what each class of evidence means; it does
not replace CI logs or experiment artifacts.

## CI separation

| Job | Environment | What it may establish |
|---|---|---|
| `pure-python` | CPython 3.11, 3.12, 3.13, and 3.14 | array contracts, CSV round trips, hashes, reference-only alignment, repeat noise, whitening, TSVD, metrics, NPZ pipeline, and pure FEM configuration/tag/checkpoint contracts |
| `heavy-dolfinx` | digest-pinned complex DOLFINx/PETSc | heavy tests, manufactured H(curl) p=1/2/3, and a direct p=3 repeated-RHS smoke solve |
| `mpi` | same image, exactly two ranks | MPI tests plus a small distributed iterative repeated-RHS solve and its ownership-independent global DoF count |

Heavy and MPI jobs fail if zero tests are collected or any selected test skips.
This prevents an unavailable DOLFINx dependency from producing a misleading
green check.

These jobs archive JSON evidence, but do not establish PML reflection
performance, a calibrated physical/S-parameter port, analytical scattering
accuracy, production scaling, or checkpoint/restart behavior.

## Identified automated evidence

[GitHub Actions run
29207223783](https://github.com/HunterSpence/Scatter3D-Codex/actions/runs/29207223783)
executed revision `c3c1ded041fc6f8bcf768db8a0acefc65647bb7d` and all configured
repository/static/package, CPython 3.11–3.14, complex DOLFINx, and two-rank MPI
jobs **PASSED**. Remote exact-revision verification also recorded 109 non-heavy
tests and 11 heavy non-MPI tests with warnings treated as errors and zero heavy
skips.

[GitHub Actions run
29206335149](https://github.com/HunterSpence/Scatter3D-Codex/actions/runs/29206335149)
retains the earlier numeric manufactured-solution and solver artifacts from
`ad9a43b`. Those results remain applicable where the exercised code was
unchanged by the later PETSc option-lifecycle correction.

- **PASSED:** 109 pure tests on each of CPython 3.11, 3.12, 3.13, and 3.14;
  wheel/sdist build, clean installs, CLI checks, static checks, link validation,
  CFF validation, and redacted secret scans.
- **PASSED:** 11 DOLFINx-heavy tests with zero skips at `c3c1ded` in DOLFINx
  0.10.0 and complex128 PETSc 3.24.0. The added regression proves nested ASM
  options remain installed until setup consumes them.
- **PASSED:** manufactured H(curl) convergence over subdivisions 3, 4, and 6.
  Observed orders were 0.913/0.958 for p=1, 1.902/1.950 for p=2, and
  2.948/2.983 for p=3. The finest p=3 system had 26,298 global complex DoFs;
  all recorded true relative residuals were below `3.1e-13`.
- **PASSED:** a direct p=3, 1,158-DoF repeated-RHS smoke solve. It recorded one
  matrix assembly, one operator setup/factorization, two RHS solves, and true
  relative residuals below `9.1e-15`.
- **PASSED:** two-rank MPI test with zero skips and a 98-DoF iterative
  repeated-RHS smoke solve. It recorded zero global factorizations, 13 and 14
  FGMRES iterations, and true relative residuals `6.76e-9` and `1.97e-9`.

The 98-DoF MPI result is a correctness canary only. It is not evidence of
parallel efficiency, million-DoF capacity, or a memory advantage.

GitHub Actions run
[29212215039](https://github.com/HunterSpence/Scatter3D-Codex/actions/runs/29212215039)
at `bee9e9d3628cc72ef0de2bda69f902629a057e24` added exact-revision two-level
p=3-to-p=1 p-multigrid correctness evidence:

- **PASSED:** repository/static/package gates, clean wheel/sdist installs,
  CPython 3.11–3.14 pure jobs, 20 complex DOLFINx-heavy tests with zero skips,
  and the selected two-rank MPI tests with zero skips.

- **PASSED:** serial, 1,158 fine and 98 coarse complex DoFs, shift `0.5`, two
  RHS in 6 and 6 iterations, with true relative residuals `1.615e-10` and
  `1.417e-10`.
- **PASSED:** two MPI ranks on the identical discrete size, two RHS in 10 and
  10 iterations, with true relative residuals `1.246e-9` and `1.298e-9`.
- **PASSED:** both artifacts record `PCUseAmat=false`, one-step
  Richardson/ASM fine smoothing, local MUMPS LU, a global p=1 MUMPS coarse
  solve, PEC-masked transfer, and live outer/fine/coarse operator identities.
- **PASSED:** both artifact SHA-256 manifests verified after download. The
  serial p-multigrid JSON hash is
  `b364be215ad0a42a5788aa471305a119f3eb3717afafef0709e37eb88f07d284`
  and the MPI JSON hash is
  `f0a93d119de6b45f9a6093998c9216a556dfad64d8515e211657d1d5d44b3c61`.

| Artifact | Exact command inside the pinned image | Actions artifact path | SHA-256 |
|---|---|---|---|
| Serial pMG | `python3 validation/fem_smoke.py --solver iterative --iterative-hierarchy p-multigrid --p-multigrid-coarse-degree 1 --preconditioner-absorption-shift 0.5 --iterative-local-pc lu --degree 3 --subdivisions 2 --frequencies-hz 1.0e8 --output /artifacts/fem-smoke-pmg-serial-p3-p1.json` | `scatter3d-heavy-verification-bee9e9d3628cc72ef0de2bda69f902629a057e24/fem-smoke-pmg-serial-p3-p1.json` | `b364be215ad0a42a5788aa471305a119f3eb3717afafef0709e37eb88f07d284` |
| MPI2 pMG | `mpirun -n 2 python3 validation/fem_smoke.py --solver iterative --iterative-hierarchy p-multigrid --p-multigrid-coarse-degree 1 --preconditioner-absorption-shift 0.5 --iterative-local-pc lu --degree 3 --subdivisions 2 --frequencies-hz 1.0e8 --output /artifacts/fem-smoke-pmg-mpi2-p3-p1.json` | `scatter3d-mpi-verification-bee9e9d3628cc72ef0de2bda69f902629a057e24/fem-smoke-pmg-mpi2-p3-p1.json` | `f0a93d119de6b45f9a6093998c9216a556dfad64d8515e211657d1d5d44b3c61` |

Both used project image ID
`sha256:2793b3d41da9ddba0d3f2838c6d3d22f7fdf2e66b1f8090548301d49fecdf66e`,
base digest
`sha256:f7cce2a2271bf838c080751348c471064acb41fef0330e2c08178a688f71890d`,
DOLFINx 0.10.0, PETSc/petsc4py 3.24.0 complex128, and MPICH 4.3.1. Actions
artifacts expire; any release citing these results must attach their JSON and
`SHA256SUMS` files.

These are small correctness cases, not p-multigrid scaling evidence.

Separate remote scaling runs at `c3c1ded` established the following:

- **PASSED:** p=3 at 86,103 global complex DoFs with right FGMRES, ASM overlap
  1, and local MUMPS LU. Two RHS converged in 233 and 230 iterations with true
  relative residuals `9.3958e-9` and `9.9821e-9`.
- **PASSED:** the identical direct MUMPS reference at 86,103 DoFs, with true
  relative residuals `9.59e-14` and `9.83e-14`.
- **FAILED:** peak memory at most 50% of direct on that identical problem. The
  summed rank process high-water RSS values were `2,544,521,216` iterative and
  `3,079,335,936` direct bytes, a ratio of `0.8263214111`.
- **FAILED:** p=3 at 470,928 global complex DoFs. Both RHS reached 1,000
  iterations; true relative residuals were `2.4996e-7` and `5.5071e-6`.

The exact environment, artifact hashes, option-lifecycle correction, and command
provenance limitation are recorded in [Distributed solver scaling
evidence](SCALING_EVIDENCE.md).

## Current development capability boundary

Source changes after `c3c1ded` now implement a separate optional
absorption-shifted preconditioning matrix `P` and call
`KSPSetOperators(A, P)` while preserving the physical matrix `A`, right-hand
sides, and true-residual calculation. Zero shift reuses `A` as `P` without a
second matrix assembly.

The solver now captures requested and effective PETSc configuration after
setup. It validates top-level types, factor backend, side, tolerances, and
iteration cap; aggregates live ASM subdomain solvers across ranks; parses ASM
type and overlap from the PETSc ASCII view; and retains that raw view. FEM
validation schema `scatter3d.validation.fem_smoke/v2` adds source/command/image,
runtime, cgroup, physical-problem, and requested/effective solver provenance,
with atomic no-clobber output unless `--overwrite` is explicit.

The exact-revision serial and MPI correctness artifacts above validate the
two-level software hierarchy at 1,158 fine DoFs. The registered shift sweep at
86,103 and 470,928 DoFs is **NOT RUN**. All historical `c3c1ded`
**PASSED**/**FAILED** scaling and memory results above are unchanged.

Campaign-03 reached a disposable runner with attestation and bootstrap
**PASSED**, but remote preparation **FAILED** before source checkout because a
disk-capacity check used mutually incompatible GNU `df` options. The image
build, lifecycle canary, wrong-DoF canary, immutable registration, and all eight
solver runs are **NOT RUN**. Attempt-owned provider resources and the ephemeral
key were deleted and independently verified absent. See the
[sanitized failure record](evidence/campaign-03-preparation-failure.json).

## Current truth boundary

- The old FEM surface-current RHS is an explicitly uncalibrated load, not a
  matched physical port and not an S-parameter source or receiver.
- Matched single-mode TEM forms and electric-mode power normalization **PASSED**
  their exact-commit software/runtime tests. Incident/outgoing magnetic modal
  extraction, calibrated S-parameters, reciprocity, and an independent thru
  benchmark are **NOT RUN**, so physical port calibration remains unverified.
- Real POM/PLA VNA reconstruction is **BLOCKED** because no accepted raw repeat,
  null, known-target, calibration, coordinate, material, geometry, and protocol
  bundle has been supplied.
- Convergence at 3,000,000 or more global complex DoFs is **NOT RUN**.
- Peak memory at or below 50% of direct on the identical 86,103-DoF problem is
  **FAILED**. A same-problem comparison at 3,000,000 DoFs is **NOT RUN**; a ratio
  measured only on a smaller common problem cannot establish the 3M target.

Do not infer a pass from the presence of source or test files. A status becomes
`PASSED` only when the exact revision, command, environment, and artifact are
identified.

## What automated tests do not establish

Even when all jobs are green, the following remain unvalidated until their own
artifacts are committed or archived with a release:

- the actual four-antenna CAD/mesh and 5–7 GHz production problem;
- a converged PML reflection, h/p/quadrature, sphere/waveguide, and sensitivity
  finite-difference campaign on that geometry;
- a distributed convergence result at 3,000,000 or more global complex DoFs;
- a same-problem memory ratio at 3,000,000 DoFs;
- VNA calibration quality and cable/thermal stability;
- measured POM and printed-PLA complex material properties;
- stationary, motion, reseat, twin-POM, and known-target controls;
- a successful real POM/PLA reconstruction;
- a nonlinear distorted-Born iterative reconstruction.

The absence of a bundled real dataset is deliberate: private or unreviewed data
must not be turned into a public success claim.

## Release evidence checklist

Before tagging an experimental release, attach or archive:

- [x] green pure, heavy, MPI, package, and static development CI at `bee9e9d`;
- [ ] green pure, heavy, MPI, package, and static CI on the final release
      revision;
- [ ] `git diff --check` and a clean signed/tagged revision;
- [ ] source distribution and wheel built from the final release revision;
- [ ] final release container identity and complex-PETSc assertion;
- [ ] manufactured/analytic/PML/discretization convergence report (manufactured
      p=1/2/3 **PASSED**; analytic scattering, PML reflection, and production
      discretization remain **NOT RUN**);
- [ ] matched-port accepted-power, incident/outgoing modal extraction,
      reciprocity, and independent transmission-line comparison;
- [ ] sensitivity finite-difference and linearization-range report;
- [x] 86,103- and 470,928-DoF scaling attempts and 86,103-DoF same-problem
      memory comparison archived with honest **PASSED**/**FAILED** statuses;
- [ ] convergence at 3,000,000 or more global complex DoFs;
- [ ] same-problem memory ratio at 3,000,000 DoFs;
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
