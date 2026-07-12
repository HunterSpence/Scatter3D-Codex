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
repeat differentials, whitens `A` and `b` together, and returns TSVD diagnostics
including available and selected rank, singular values, residuals, solution
norm, condition estimate, and the rank-selection criterion.

The default GCV choice is useful when a noise norm is unavailable; discrepancy
selection is preferable when repeat measurements supply a defensible noise
norm. A fixed rank is intended for reproducibility tests, not visual tuning.

### FEM

`scatter3d.fem` is split into small pieces rather than one stateful solver:

- `config`: immutable materials, PML, Maxwell, and linear-solver settings;
- `tags` and `gmsh_io`: validate mesh/physical-tag contracts;
- `pml`: Cartesian complex-stretch tensors;
- `ports`: discrete per-port mode normalization and excitations;
- `forms`: frequency-dependent Maxwell forms;
- `solver`: one matrix/preconditioner setup per frequency with successive port
  right-hand sides, plus convergence diagnostics;
- `checkpoints`: identity hashes that reject stale mesh/config/frequency reuse;
- `diagnostics`: true residual and material-model comparisons.

The public high-level entry is `MaxwellSweepSolver.solve(frequencies_hz,
ports)`. DOLFINx imports are kept inside the FEM layer so measurement and inverse
tests remain usable on ordinary Python installations.

Geometry order is intentionally limited to one in the initial release. Curved
geometry support must not be enabled until a curved-boundary convergence test is
added; refine linear geometry instead of pretending a higher coordinate order
was validated.

### Pipeline and CLI

`scatter3d.pipeline` defines a compact NPZ interchange format. It rejects
unknown schema versions, pickle-backed arrays, non-finite values, coordinate
mismatches, duplicate sensitivity rows, and missing row maps. It writes the
reconstruction atomically and records input SHA-256 hashes.

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

Descriptive metadata is not allowed to change a numerical fingerprint. Secrets
and private dataset contents never belong in a manifest.

## Deliberate non-goals for 0.1

- automatic VNA control or calibration-kit programming;
- an implicit Touchstone convention guesser;
- independently fitting reference and DUT calibration;
- silently symmetrizing reciprocal channels;
- a claimed nonlinear DBIM implementation;
- production-scale or real-object validation without published evidence.
