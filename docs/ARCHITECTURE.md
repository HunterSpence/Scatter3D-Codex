# Architecture

## Design objective

Scatter3D-Codex is organized around a single question: **what evidence would
distinguish a physical result from a software, calibration, or discretization
artifact?** Each boundary therefore carries coordinates, units, hashes, and
diagnostics instead of passing anonymous arrays between scripts.

```text
VNA export / simulation
        |
        v
ScatteringDataset -- validate axes, labels, finiteness, file/array hashes
        |
        +--> repeat/null diagnostics -----------+
        |                                       |
reference-only alignment                       |
        |                                       |
same-index DUT - reference                      |
        |                                       |
        v                                       v
measurement vector b <--- shared row map ---> sensitivity matrix A
        |                                       |
        +--------- optional joint whitening ----+
                            |
                            v
                    transparent complex TSVD
                            |
                            v
                  reconstruction + provenance
```

## Package layers

### Measurement

`scatter3d.measurement` owns the canonical
`[angle, frequency, receiver, source]` contract. `ScatteringDataset` validates
coordinates and exposes a deterministic C-order vector. CSV import/export uses
explicit columns; it does not infer port order from filenames.

`ReferenceAlignment` is estimated once from measured and modelled reference
data. The same frozen correction can then be applied to reference and DUT. This
prevents the calibration step from fitting away the defect signal.

### Inverse

`scatter3d.inverse` keeps complex arithmetic native. It estimates noise from
paired same-index repeat differentials, divides sample variance by the paired
repeat count for the variance of the mean, whitens `A` and `b` together, and
returns TSVD diagnostics including available and selected rank, the full compact
singular spectrum, residuals, solution norm, condition estimate, numerical-rank
threshold, and the complete rank-selection criterion curve.

The default GCV choice is useful when a noise norm is unavailable; discrepancy
selection is preferable when repeat measurements supply a defensible noise
norm. The automatic whitened `sqrt(rows)` target is only an expected RMS
heuristic. An unmet discrepancy target produces a `FAILED` artifact and nonzero
exit unless explicitly allowed; the override never converts the result to
`PASSED`. A fixed rank is intended for reproducibility tests, not visual tuning.

### FEM

`scatter3d.fem` is split into small pieces rather than one stateful solver:

- `config`: immutable materials, PML, Maxwell, and linear-solver settings;
- `tags` and `gmsh_io`: validate mesh/physical-tag contracts;
- `pml`: Cartesian complex-stretch tensors;
- `ports`: explicit separation between uncalibrated surface loads and the
  in-progress matched single-mode TEM boundary/mode-normalization path;
- `forms`: frequency-dependent Maxwell forms;
- `solver`: one matrix/preconditioner setup per frequency with successive port
  right-hand sides, true-residual checks, and requested/setup-observed solver
  diagnostics;
- `checkpoints`: identity hashes that reject stale mesh/config/frequency reuse;
- `diagnostics`: true residual and material-model comparisons.

The historical surface-current RHS is a generic weak load only. It does not
define a matched termination, power wave, circuit reference, or calibrated
S-parameter. The newer matched TEM forms and electric-mode normalization are
still integration work: heavy DOLFINx tests and incident/outgoing modal
extraction must land and pass before `MaxwellSweepSolver` output can be described
as a physical port result. DOLFINx imports remain inside the FEM layer so
measurement and inverse tests are usable on ordinary Python installations.

Geometry order is intentionally limited to one in the initial release. Curved
geometry support must not be enabled until a curved-boundary convergence test is
added; refine linear geometry instead of pretending a higher coordinate order
was validated.

### Pipeline and CLI

`scatter3d.pipeline` defines a compact NPZ interchange format. It rejects
unknown schema versions, pickle-backed arrays, non-finite values, coordinate
mismatches, noncanonical dtypes, duplicate sensitivity rows, and missing row
maps. Measurement and sensitivity arrays use exact complex128/float64/Unicode/
int64 contracts. It writes the reconstruction atomically, refuses to clobber an
existing artifact unless explicitly forced, and records input SHA-256 hashes,
overwrite intent, selector status, spectrum hashes, and noise-model provenance.

`scatter3d.cli` deliberately exposes only offline validation, diagnosis, and
inversion. Instrument control and billable or safety-relevant RF actions are not
part of the CLI.

## Frequency and right-hand-side lifecycle

At a fixed frequency, all antenna excitations share the same Maxwell operator.
The intended lifecycle is:

1. update the live frequency and PML coefficients;
2. assemble the operator once;
3. set the PETSc operator and create its preconditioner/factorization once;
4. assemble and solve one right-hand side per source port;
5. record PETSc reason and true residual for every source;
6. advance frequency and rebuild because the operator changed.

PETSc explicitly supports repeated `KSPSolve` calls for the same operator and
different right-hand sides. Reusing a factorization across a changed frequency
or material model is forbidden.

PETSc options are installed under a unique prefix for each KSP. Some
preconditioners, including ASM, do not create and configure nested subdomain KSP
and PC objects until `KSPSetUp`. The options must therefore remain installed
through setup and may be removed from PETSc's global options database only
afterward. Revision `c3c1ded` corrected an earlier lifecycle error that removed
them after `KSPSetFromOptions` but before setup, causing a requested subdomain LU
to remain PETSc's default ILU. The regression test proves nested options are
consumed during setup, and the corrected remote run was independently inspected
with `-ksp_view`.

The current development solver retains both requested and setup-observed PETSc
configuration. After `KSPSetUp`, it checks the effective top-level KSP/PC,
factor backend, preconditioning side, tolerances, and iteration cap against the
typed configuration. For ASM it aggregates the live subdomain KSP/PC hierarchy
across MPI ranks, parses restriction/interpolation type and overlap from PETSc's
official ASCII view, and retains the raw view in diagnostics. A mismatch fails
closed; requested options alone are never treated as effective-runtime evidence.

The current development forms distinguish the unchanged physical Maxwell
operator `A` from a separately assembled absorption-shifted preconditioning
operator `P`. A positive dimensionless shift adds artificial loss only to the
mass term in `P`; the physical form and all right-hand sides remain unchanged.
The solver calls `KSPSetOperators(A, P)` and always recomputes the acceptance
residual with `A`. Zero shift has explicit identity semantics and reuses `A` as
`P` without a duplicate matrix assembly.

The solver also provides a genuine assembled p=3-to-p=1 two-level correction.
Small serial and two-rank 1,158-fine/98-coarse-DoF correctness artifacts
**PASSED** at `bee9e9d`, including shifted fine/coarse operators and live
hierarchy checks. The registered 86,103- and 470,928-DoF p-multigrid sweep is
**NOT RUN**; the small canaries are not convergence-at-scale evidence and do not
alter the historical `c3c1ded` scaling results.

## Provenance boundary

A defensible result should identify at least:

- Git revision and dirty state;
- input file hashes;
- schema versions and coordinate hashes;
- mesh and physical-tag hash;
- material values and units;
- frequency, port, angle, and row selections;
- DOLFINx, PETSc, MPI, scalar type, and container digest;
- MPI ranks and thread limits;
- solver options, convergence reasons, and true residuals;
- regularization method, rank, singular values, and whitening model.

The FEM validation artifact schema
`scatter3d.validation.fem_smoke/v2` records source identity, process command,
container/base-image identities, runtime versions, cgroup memory metadata, a
canonical physical-problem identity, requested/effective solver configuration,
and per-frequency preconditioner metrics. Writes are atomic and no-clobber by
default; replacing an artifact requires explicit `--overwrite`.

The reconstruction NPZ is itself the selector evidence artifact: it retains the
full compact singular spectrum, criterion ranks and values, status, numerical
threshold, selected rows, floors, and hashes. A JSON summary is useful for
inspection but does not replace that NPZ.

Descriptive metadata is not allowed to change a numerical fingerprint. Secrets
and private dataset contents never belong in a manifest.

## Deliberate non-goals for 0.1

- automatic VNA control or calibration-kit programming;
- an implicit Touchstone convention guesser;
- independently fitting reference and DUT calibration;
- silently symmetrizing reciprocal channels;
- a claimed nonlinear DBIM implementation;
- describing an uncalibrated surface-current load as a VNA/S-parameter port;
- production-scale or real-object validation without published evidence.
