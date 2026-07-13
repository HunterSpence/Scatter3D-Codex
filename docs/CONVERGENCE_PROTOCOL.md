# Convergence and verification protocol

## What “converged” means here

A solver has not converged merely because it returned a field, PETSc reported a
positive reason, or the mesh uses elements smaller than `lambda/4`. Three
different questions must be answered:

1. **Algebraic convergence:** was the discrete linear system solved to the
   registered true residual?
2. **Discretization convergence:** are fields, port quantities, S-parameters,
   and sensitivities stable under mesh/order/quadrature/PML refinement?
3. **Model adequacy:** does the converged numerical reference agree with a
   calibration-quality physical reference within the uncertainty budget?

These gates are sequential. A smaller KSP residual cannot repair geometry or
material mismatch.

## Reproducibility header

Record before every registered run:

```text
git revision and dirty state:
container image digest:
DOLFINx / UFL / Basix versions:
PETSc / petsc4py version and scalar type:
MPI implementation and ranks:
host CPU and memory limit:
OMP/MKL/OpenBLAS/NUMEXPR threads:
mesh/tag SHA-256:
geometry order / element family / degree:
quadrature degree:
frequency and port selection:
material values, loss convention, and source:
PML thickness/profile/target reflection:
requested and effective KSP/PC hierarchy, options prefix, and tolerances:
physical operator and preconditioning-operator identities:
```

The pinned image selects complex PETSc and caps threaded math libraries at one.
Changing any header item creates a new comparison series.

## Verification ladder

### Gate 1 — configuration and tag contracts

- Every required volume and boundary tag exists and is nonempty.
- Tags are disjoint where the physics requires disjoint regions.
- Material arrays cover every physical cell exactly once.
- Frequencies are finite, positive, and strictly increasing.
- Every port excitation lives on the intended tagged boundary.
- Geometry order other than the explicitly supported order is rejected.
- Mesh/config/checkpoint fingerprints change when any numerical input changes.

Acceptance: all pure tests pass; invalid cases fail with specific errors.

### Gate 2 — algebraic solver checks

For every `(frequency, source_port)` report:

```text
PETSc converged reason
iteration count
reported residual history
true relative residual = ||A x - b||_2 / ||b||_2
matrix size and nonzeros
assembly/setup/solve time
rank RSS high-water mark (max and sum)
```

Acceptance defaults:

- PETSc reason is positive;
- true relative residual `<= 1e-10` for the reference direct solve and `<= 1e-8`
  for the distributed iterative profile;
- no NaN/Inf in matrix, right-hand side, solution, or derived S-parameters;
- tightening algebraic tolerance by 100× changes the S-parameter quantity of
  interest by less than 10% of the final discretization tolerance.

These are project defaults and may be made stricter. Any relaxation must be
registered before viewing an image and justified against the error budget.

At one frequency, assemble the operator once and solve successive port right-
hand sides with the same KSP. PETSc documents that repeated `KSPSolve` calls for
an unchanged operator reuse setup/preconditioner work
([PETSc KSP manual](https://petsc.org/main/manual/ksp/)). Reassemble and reset the
operator whenever frequency or material coefficients change.

### Gate 3 — manufactured H(curl) solution

Use a smooth vector field with analytically derived curl-curl forcing and
boundary data on a simple 3-D domain. Run at least three uniform `h` refinements
for each supported polynomial degree.

Report:

- global degrees of freedom;
- `L2(E)` and `H(curl)` errors;
- observed order between each pair;
- algebraic true residual and convergence reason;
- quadrature degree.

Acceptance:

- both norms decrease monotonically over the final three levels;
- the final two observed slopes meet the pre-registered theoretical expectation
  for the exact element family, norm, and solution regularity (default warning
  margin: expected order minus 0.5);
- changing quadrature does not materially change the observed slope.

Do not quote one universal order for all Nédélec families and norms. State the
Basix element variant and the theorem/benchmark that supplies the expectation.

### Gate 4 — port normalization and same-material null

For every port:

- calculate and record the discrete mode norm independently;
- scale the mode to the declared accepted-power or wave-amplitude convention;
- verify a known thru/waveguide case, including phase;
- permute port labels and prove the corresponding S-matrix permutation is exact;
- solve reference and DUT with identical mesh/materials and require their
  differential to be at roundoff/algebraic-tolerance scale.

Never normalize all sources using port 1's mode integral.

### Gate 5 — analytic or independently solved scattering benchmark

Use at least one geometry with an independent solution in the same frequency
and material regime:

- dielectric/PEC sphere compared with a Mie series;
- waveguide propagation/reflection compared with modal theory; or
- a published, controlled benchmark such as the Institut Fresnel dataset.

Compare complex amplitude and phase—not only dB magnitude. Register observation
points/ports, reference impedance, and time convention. Acceptance is a
monotone approach to the independent result and a final error below the
application's S-parameter error budget.

### Gate 6 — PML convergence

The DOLFINx PML demo provides a reference implementation pattern for complex
Maxwell/PML forms
([DOLFINx documentation](https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_pml.html)).
For the actual frequency band, sweep independently:

- PML thickness (at least three values);
- absorption/profile strength (at least three values);
- mesh resolution across the PML;
- quadrature degree;
- distance from object/antennas to PML.

Measure reflection in an empty/known outgoing-wave case and its effect on every
S-parameter and sensitivity row. Acceptance: the final two refinements change
the target `ΔS` by less than 20% of the allocated numerical-error budget, and no
late PML resonance appears anywhere in 5–7 GHz.

A symbolic form that contains the current frequency is a unit test; it is not a
PML reflection validation.

### Gate 7 — h, p, geometry, and quadrature convergence

Use the same physical configuration and compute at least:

- three systematically refined meshes `h0`, `h0/r`, `h0/r^2`;
- supported degrees `p=1,2,3` where memory permits;
- quadrature sequence `2p`, `2p+2`, `2p+4` (for `p=3`: 6, 8, 10);
- extra interface refinement around antennas, the cylindrical defect, and high
  material gradients.

For every run compare:

1. absolute reference `S`;
2. full-wave DUT-reference `ΔS`;
3. linearized sensitivity prediction of `ΔS`;
4. selected singular spectrum and reconstruction on an unchanged voxel grid.

Use relative complex differences with an absolute floor so channels near zero
do not create meaningless percentages. Pre-register an application tolerance;
a useful starting allocation is to keep total numerical `ΔS` error below 20% of
the measured repeat RMS, then tighten if the known-target margin permits.

The initial release intentionally uses linear mesh geometry. Curved cylinders
therefore require geometric refinement; increasing field degree alone cannot
remove faceting error.

### Gate 8 — sensitivity finite difference

For representative voxels/regions and ports, evaluate full-wave central
differences:

```text
dS/dε ≈ [S(ε + δ) - S(ε - δ)] / (2δ)
```

Sweep at least three `δ` values. Truncation should decrease before roundoff/
solver noise dominates. Compare complex sensitivity row-by-row, including the
frequency prefactor, excitation normalization, and sign/time convention.

Acceptance: derivative error is below the allocated sensitivity budget for
strong, medium, and weak rows. A single scalar scale-factor fit is not a pass if
phase or port dependence remains.

### Gate 9 — linearization range

With converged discretization, simulate inclusion contrast/volume fractions
from zero through the intended PLA case. Compare:

```text
full-wave ΔS(alpha)
linear prediction A x(alpha)
relative model error ||full - linear|| / ||full||
```

The linear model is accepted only where its error is below the combined
measurement/numerical budget. Otherwise stop or validate a nonlinear method in a
separate protocol; do not merely retain more TSVD singular values.

### Gate 10 — inverse and experimental controls

- exact synthetic `b=A x` recovers registered truth at fixed rank;
- noisy synthetic data meet pre-registered localization/amplitude metrics;
- same-material, motion, and twin-reference nulls do not create target-like
  structure;
- known-target reconstruction passes location/contrast metrics;
- rank choice remains stable under registered noise/mesh perturbations;
- the intended target is evaluated last with frozen settings.

## Large-path solver profiles

Use `LinearSolverConfig.direct()` as the small reference profile. For distributed
large cases, use `LinearSolverConfig.iterative_maxwell(...)`, which configures a
globally iterative FGMRES path with additive Schwarz and local ILU rather than a
global factorization. Compare both on the largest problem the direct profile can
solve and require agreement of fields/S-parameters within the registered
algebraic tolerance.

The one-level profile is a portability and correctness baseline, not a scalable
endpoint. Executed p=3 evidence at `c3c1ded` is mixed and must be reported as
such:

- 86,103 global complex DoFs with ASM overlap 1 and corrected local MUMPS LU:
  **PASSED** for both right-hand sides;
- peak memory at most 50% of direct on the identical 86,103-DoF problem:
  **FAILED**, with summed rank peak-RSS ratio `0.8263214111`;
- 470,928 global complex DoFs with the same one-level method: **FAILED**, with
  both right-hand sides reaching the 1,000-iteration cap;
- at least 3,000,000 global complex DoFs: **NOT RUN**.

See [Distributed solver scaling evidence](SCALING_EVIDENCE.md) for the exact
revision, runtime digests, residuals, and artifact hashes.

The current development source keeps the physical Maxwell matrix as `A`,
assembles a separate absorption-shifted/lossy Maxwell matrix as `P`, and uses
right-preconditioned FGMRES through `KSPSetOperators(A, P)`. The shift does not
change `A` or any right-hand side, and zero shift explicitly aliases `P` to `A`.
Validation schema `scatter3d.validation.fem_smoke/v2` records matrix metrics,
the shift, requested and effective PETSc hierarchy, raw ASCII KSP view,
provenance, physical-problem identity, and cgroup metadata. It is no-clobber by
default. Exact-revision serial and two-rank p=3-to-p=1 correctness artifacts
**PASSED** at `bee9e9d` on 1,158 fine and 98 coarse DoFs. They do not change the
executed `c3c1ded` scaling statuses above.

An absorption shift alone is not a global correction. The 470k failure requires
a genuine coarse level. The next candidate is p-multigrid from the p=3 Nedelec
space to an assembled p=1 Nedelec coarse operator on the same mesh and physical
model. That two-level candidate has executed small correctness evidence, but
its registered 86k/471k shift sweep remains **NOT RUN**. HPDDM is **BLOCKED** in
the pinned image because PETSc lacks its HPDDM backend; it is not an equivalent
fallback. Do not infer scalability from configuration or the small canaries.

The capability probe is part of `/opt/scatter3d/runtime-metadata.json` in every
new project image. For the pinned base digest, this exact probe reports both
values as `false`:

```bash
python3 -c 'from petsc4py import PETSc; print({name: bool(PETSc.Sys.hasExternalPackage(name)) for name in ("hpddm", "slepc")})'
# {'hpddm': False, 'slepc': False}
```

The image build records `petsc_has_hpddm` and `petsc_has_slepc`, and the scaling
registration hashes that runtime metadata. This makes **BLOCKED** an auditable
capability result rather than an assumption.

Record setup versus solve time separately. Report global complex degrees of
freedom, matrix nonzeros/estimated memory, and per-rank RSS high-water max and
sum, plus cgroup `memory.peak` where available. Do not claim a memory reduction
or million-DoF capacity from configuration alone; publish the executed benchmark
artifact. Retain every registered shift/coarse-space attempt, including
failures, rather than publishing only the best parameter choice.

## Parallel scaling protocol

Run the exact same mesh/configuration at 1, 2, 4, and then available larger rank
counts. Perform one warm-up outside timing. Report median of at least three
measured runs and include:

- mesh distribution and global/local DoFs;
- assembly, setup, solve-per-RHS, and total wall time;
- iterations and true residual for each RHS;
- maximum and sum of rank RSS;
- speedup, parallel efficiency, and solver-option database.

Stop increasing ranks once communication increases total time or per-rank work
becomes too small. Never compare a cold JIT run with a warm cached run.

## Production release gate

A real-image result may be labelled validated only if all applicable gates above
have machine-readable evidence and the measurement runbook's null/known-target
controls pass. At minimum, archive:

| Gate | Artifact |
|---|---|
| Runtime | container digest and version dump |
| Algebra | per-RHS convergence JSON |
| Manufactured | errors/orders CSV or JSON |
| Benchmark | complex S comparison |
| PML | thickness/profile sweep |
| Discretization | h/p/quadrature table |
| Sensitivity | finite-difference comparison |
| Linearization | contrast/volume ladder |
| Scaling | ranks/time/memory table |
| Experiment | repeat/null/known-target reports |

If a gate has no executed evidence, write **NOT RUN**. “Expected to work” is not evidence.
