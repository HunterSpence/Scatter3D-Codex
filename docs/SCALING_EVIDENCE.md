# Distributed solver scaling evidence

This page records executed solver results without upgrading a near miss into a
pass. The registered algebraic acceptance condition is a positive PETSc
convergence reason and a recomputed true relative residual no greater than
`1e-7` for both independent port right-hand sides.

## Historical one-level evidence identity

| Item | Value |
|---|---|
| Post-fix Git revision | `c3c1ded041fc6f8bcf768db8a0acefc65647bb7d` |
| Earlier baseline revision | `c4e1c3a80952175822cf19ff341fa99a2ff6e244` |
| Base image | `dolfinx/dolfinx:v0.10.0@sha256:f7cce2a2271bf838c080751348c471064acb41fef0330e2c08178a688f71890d` |
| Project image digest | `sha256:22cdddd5fd74259c8ffe27543d831a575787272af31a0ca5c6a1867ace88435d` |
| DOLFINx | `0.10.0` |
| PETSc / petsc4py | `3.24.0` |
| PETSc scalar | `complex128` |
| MPI | MPICH `4.3.1` |
| Exact historical CI | [GitHub Actions run 29207223783](https://github.com/HunterSpence/Scatter3D-Codex/actions/runs/29207223783) |

The later small two-level correctness evidence has a separate identity and does
not replace the historical scaling artifacts above:

| Item | Value |
|---|---|
| Git revision | `bee9e9d3628cc72ef0de2bda69f902629a057e24` |
| Project image ID | `sha256:2793b3d41da9ddba0d3f2838c6d3d22f7fdf2e66b1f8090548301d49fecdf66e` |
| Base image digest | `sha256:f7cce2a2271bf838c080751348c471064acb41fef0330e2c08178a688f71890d` |
| Exact CI | [GitHub Actions run 29212215039](https://github.com/HunterSpence/Scatter3D-Codex/actions/runs/29212215039) |
| Serial pMG JSON | `fem-smoke-pmg-serial-p3-p1.json`; SHA-256 `b364be215ad0a42a5788aa471305a119f3eb3717afafef0709e37eb88f07d284` |
| MPI2 pMG JSON | `fem-smoke-pmg-mpi2-p3-p1.json`; SHA-256 `f0a93d119de6b45f9a6093998c9216a556dfad64d8515e211657d1d5d44b3c61` |

## Registered p-multigrid campaign preparation

Campaign-03 attempted to prepare one disposable runner from exact green source
`5393e4494a5f63bdca24defba18eb7af7ffa5bf4` (Actions run
[`29222225396`](https://github.com/HunterSpence/Scatter3D-Codex/actions/runs/29222225396),
**PASSED**). Cryptographic runner attestation and bootstrap **PASSED**. The
preparation script then **FAILED** before source checkout because GNU `df`
rejects combining `-P` with `--output=avail`.

No project image was built, neither lifecycle nor wrong-DoF canary ran, no
immutable registration was created, and no FEM process launched. Therefore all
eight registered p-multigrid entries remain **NOT RUN**. This preparation
failure is not a numerical result and supplies no scaling or memory evidence.

The exact attempt-owned server, Primary IP, firewall, and ephemeral SSH key were
deleted; controller state recorded repeated exact absence over the deletion
window and an independent live inventory check **PASSED**. The public-safe
evidence is
[`campaign-03-preparation-failure.json`](evidence/campaign-03-preparation-failure.json).
The public JSON has SHA-256
`64119bcff1a784bb26e319fe47194087a274bde123d187ae8c2cd288824937d6`.
Its canonical private archive manifest has SHA-256
`77d6a44e09e0876500849c507122d3b7f926b285c235affc52f79013b6383e42`.
Infrastructure identities and raw provider records remain private.

The portable capacity helper added for the next campaign has no shell or `df`
dependency. After cloning and detach-checking out the exact green commit, a
future identity-bound control pack must supply the resolved campaign and
Docker-root paths and preserve its JSON before image build or other
disk-intensive work. The example assumes `$docker_root` was resolved from the
live Docker runtime and `$PRIVATE_DIR` is a pre-created private evidence
directory:

```bash
python3 validation/remote_capacity_preflight.py \
  --path campaign=/opt \
  --path docker="$docker_root" \
  --minimum-free-bytes 107374182400 \
  > "$PRIVATE_DIR/disk-capacity-contract.json"
```

That example is not campaign-03 evidence and has not yet passed the mandatory
live runner canary.

## Direct-solver memory candidates — NOT RUN

Two upstream developments are credible candidates for a separately registered
memory campaign, but neither is evidence for the current release:

- [PETSc 3.25](https://petsc.org/release/changes/325/) adds MUMPS
  `-pc_precision <single,double>` and an out-of-core temporary-directory
  control. The [official PETSc 3.25.3 package
  recipe](https://gitlab.com/petsc/petsc/-/raw/v3.25.3/config/BuildSystem/config/packages/MUMPS.py)
  still pins MUMPS 5.8.2 and enables mixed precision only when the required
  single- and double-precision MUMPS libraries are actually present.
- [MUMPS 5.9.0](https://mumps-solver.org/index.php?page=dwnld) adds the
  experimental adaptive-precision BLR control `ICNTL(40)` and a
  single-precision factorization in a double-precision instance via
  `ICNTL(47)`. Its manual describes up to seven custom formats; this project
  does not infer a blanket IEEE-FP16 storage claim from that description.

The pinned release runtime remains DOLFINx 0.10.0 with PETSc/petsc4py 3.24.0,
so both candidates are **NOT RUN** here. Testing either path requires a new
digest-pinned image, explicit build-feature and effective-option evidence,
double-precision residual/output comparisons, and a new immutable campaign
identity. Results from that runtime must not be spliced into the current pMG
sweep.

MUMPS `INFOG(21)` and `INFOG(22)` remain useful secondary diagnostics: the
[MUMPS 5.9.0 manual](https://mumps-solver.org/doc/userguide_5.9.0.pdf) defines
them as effective memory used during factorization, respectively the maximum
and sum over processors. They are not total process or host high-water marks.
The project's at-most-50% target therefore continues to require identical-
problem cgroup-v2 `memory.peak` (or equivalent scheduler high-water) evidence
for both converged solvers.

The historical c3 scaling JSON files do not retain the command line, Git
revision, dirty state, or image digest. The final server log archive retains `/usr/bin/time`'s verbatim
`Command being timed` records. Those commands are reproduced below. Future JSON
schemas should record the same provenance directly.

## Exact benchmark commands

The 470,928-DoF post-fix run, from
`p3-n16-mpi8-right-asm1-lu.log`:

```bash
docker run --rm --ipc=host --memory=28g --memory-swap=28g --volume /opt/scatter3d-codex/artifacts/c4e1c3a:/artifacts scatter3d-codex:bench-c3 mpirun -n 8 python3 validation/fem_smoke.py --solver iterative --degree 3 --subdivisions 16 --frequencies-hz 1.0e8 --maximum-iterations 1000 --gmres-restart 100 --asm-overlap 1 --iterative-local-pc lu --minimum-global-dofs 450000 --output /artifacts/p3-n16-mpi8-right-asm1-lu.json
```

The identical direct post-fix reference, from `p3-n9-mpi4-direct.log`:

```bash
docker run --rm --ipc=host --volume /opt/scatter3d-codex/artifacts/c4e1c3a:/artifacts scatter3d-codex:bench-c3 mpirun -n 4 python3 validation/fem_smoke.py --solver direct --degree 3 --subdivisions 9 --frequencies-hz 1.0e8 --output /artifacts/p3-n9-mpi4-direct.json
```

The earlier one-level ILU baseline, from
`p3-n9-mpi4-right-asm1-ilu.log`:

```bash
docker run --rm --ipc=host --volume /opt/scatter3d-codex/artifacts/c4e1c3a:/artifacts scatter3d-codex:bench-c4 mpirun -n 4 python3 validation/fem_smoke.py --solver iterative --degree 3 --subdivisions 9 --frequencies-hz 1.0e8 --maximum-iterations 1000 --gmres-restart 80 --asm-overlap 1 --iterative-local-pc ilu --output /artifacts/p3-n9-mpi4-right-asm1-ilu.json
```

The corrected post-fix local-MUMPS run, from
`p3-n9-mpi4-right-asm1-lu-fixed.log`:

```bash
docker run --rm --ipc=host --volume /opt/scatter3d-codex/artifacts/c4e1c3a:/artifacts scatter3d-codex:bench-c3 mpirun -n 4 python3 validation/fem_smoke.py --solver iterative --degree 3 --subdivisions 9 --frequencies-hz 1.0e8 --maximum-iterations 1000 --gmres-restart 80 --asm-overlap 1 --iterative-local-pc lu --output /artifacts/p3-n9-mpi4-right-asm1-lu-fixed.json
```

The pre-fix supposed-LU canary, from
`p3-n9-mpi4-right-asm1-lu.log`:

```bash
docker run --rm --ipc=host --volume /opt/scatter3d-codex/artifacts/c4e1c3a:/artifacts scatter3d-codex:bench-c4 mpirun -n 4 python3 validation/fem_smoke.py --solver iterative --degree 3 --subdivisions 9 --frequencies-hz 1.0e8 --maximum-iterations 1000 --gmres-restart 80 --asm-overlap 1 --iterative-local-pc lu --output /artifacts/p3-n9-mpi4-right-asm1-lu.json
```

That final command requested LU, but the pre-fix lifecycle bug prevented the
nested PC from consuming the request; the effective local PC was ILU. The
command therefore documents intent, not effective LU evidence. The exact
standalone command that rendered the memory-comparison JSON was not present in
the retained `Command being timed` records; its two input runs and comparison
values are retained in the JSON.

## Executed scaling ladder

| Gate | Problem and solver | Result | Evidence |
|---|---|---|---|
| One-level ASM/ILU baseline | p=3, subdivisions 9, 4 MPI ranks, 100 MHz, 86,103 global complex DoFs, 7,038,009 nonzeros, right FGMRES restart 80, ASM overlap 1, local ILU(0) | **FAILED** | Both RHS reached 1,000 iterations; best recorded true relative residual was approximately `4.93e-3` |
| Corrected one-level ASM/local-MUMPS | Same p=3 problem, right FGMRES restart 80, ASM overlap 1, local preonly/LU/MUMPS | **PASSED** | 233 and 230 iterations; true relative residuals `9.3958e-9` and `9.9821e-9` |
| Identical direct MUMPS reference | Same p=3 problem and 4 MPI ranks, direct preonly/LU/MUMPS with two RHS | **PASSED** | True relative residuals `9.59e-14` and `9.83e-14` |
| Same-problem memory gate | Iterative versus direct at 86,103 DoFs; metric is sum of rank process high-water RSS bytes | **FAILED** | Iterative `2,544,521,216`; direct `3,079,335,936`; ratio `0.8263214111`, above the required `0.5` |
| Next scaling rung | p=3, subdivisions 16, 8 MPI ranks, 100 MHz, 470,928 global complex DoFs, 39,290,544 nonzeros, right FGMRES restart 100, ASM overlap 1, local LU/MUMPS | **FAILED** | Both RHS reached 1,000 iterations; true relative residuals `2.4996e-7` and `5.5071e-6` |
| At least 3,000,000 global complex DoFs | Two RHS with the registered true-residual gate | **NOT RUN** | No executed artifact exists |
| Real POM/PLA VNA reconstruction | Accepted raw reference, DUT, null, known-target, calibration, coordinates, geometry, materials, and protocol bundle | **BLOCKED** | No accepted raw measurement bundle has been supplied |

The 470,928-DoF run assembled in about `17.89` seconds and did not exhaust its
28 GiB cgroup limit. Its summed rank peak RSS was `13,988,892,672` bytes and its
maximum single-rank peak RSS was `1,946,419,200` bytes. Available memory was not
the acceptance criterion: the algebraic gate **FAILED**.

Process high-water RSS is useful diagnostic evidence, but scheduler or cgroup
`memory.peak` is preferred for a publication-grade memory claim. No memory ratio
at 3,000,000 DoFs has been measured.

## PETSc nested-options correction

Before `c3c1ded`, temporary prefixed PETSc options were removed immediately after
`KSPSetFromOptions` and before `KSPSetUp`. ASM creates and configures its nested
subdomain KSP and PC objects during setup, so the requested `sub_pc_type=lu` was
not consumed; PETSc retained its default ILU. Early ILU and supposed-LU results
were therefore identical because both used ILU.

Revision `c3c1ded` keeps the prefixed options installed through `KSPSetUp`, then
removes them in a `finally` block. A heavy regression requests an invalid nested
PC and proves that setup consumes the nested option. A live `-ksp_view` in the
pinned image then confirmed ASM with subdomain `preonly`, LU, and MUMPS. The
post-fix `p3-n9-mpi4-right-asm1-lu-fixed.json` is the admissible local-LU
artifact. The pre-fix `p3-n9-mpi4-right-asm1-lu.json` is **FAILED** as local-LU
evidence and must not be used for that claim.

## Artifact SHA-256 manifest

These hashes were recomputed from the local evidence mirror rather than copied
from filenames or terminal prose.

| Artifact | SHA-256 | Interpretation |
|---|---|---|
| `p3-n16-mpi8-right-asm1-lu.json` | `e0f0e40bd1d1b100645087da3bbe22821682afc939455d0e16fa51957bdbddb7` | 470,928-DoF **FAILED** rung |
| `p3-n9-mpi4-direct.json` | `00be6c3560c39bb4f23b32409c5ff9a7c94298fa7b73c5e9b3fe9f5072fe4196` | Identical direct **PASSED** reference |
| `p3-n9-mpi4-memory-gate.json` | `5caf2f39e05106307d1c11aab4ac55eb80fb347d6b9c97f25cf8e72010d8f3ab` | Same-problem memory **FAILED** gate |
| `p3-n9-mpi4-right-asm1-ilu.json` | `da385b3cc910970f97b1f81dd7cb9d4f1fb1cb3f38a256f5305bccda319ae2a2` | One-level ILU **FAILED** baseline |
| `p3-n9-mpi4-right-asm1-lu-fixed.json` | `f7e7b1ffbb2830a0b80089ad58699b93db101e24cbbfb445a615c7e6ca2b0a89` | Corrected local-MUMPS **PASSED** run |
| `p3-n9-mpi4-right-asm1-lu.json` | `bf437d5ac34ce1418a654c454628b5657b3866f0282b15882f1f988f74ceb362` | Pre-fix non-LU canary; **FAILED** as LU evidence |

The JSON files are numerical release evidence, not source code. They remain
outside the Git tree under the repository's artifact-ignore policy and must be
attached with a SHA-256 manifest to any release that cites them.

## Required next solver step

The 86k pass followed by the 471k failure demonstrates that one-level ASM with
exact local solves is not mesh-scalable. Raising the iteration cap alone is not
the next acceptance step. The current development source now keeps the physical
Maxwell matrix as `A`, can assemble a separate absorption-shifted Maxwell matrix
as `P`, and calls `KSPSetOperators(A, P)`. It also preserves requested and
setup-observed PETSc hierarchy, the raw ASCII KSP view, and validation-v2
provenance. Small serial and two-rank p=3-to-p=1 correctness artifacts
**PASSED** at `bee9e9d` with 1,158 fine and 98 coarse DoFs; they do not modify
the historical scaling results in this file.

The p=3-to-p=1 Nedelec p-multigrid implementation is now present and its small
correctness cases **PASSED**. Its registered 86k/471k shift sweep remains
**NOT RUN**. Every registered shift, including failures, must be retained; no
scaling claim is made from the small CI cases.

The immutable experiment is
[`validation/scaling_sweep_v1.json`](../validation/scaling_sweep_v1.json).
`validation/register_scaling_sweep.py` must write its complete registration
outside the clean source tree before the first solve.
`validation/run_registered_scaling_sweep.py --run-id <registered-id>` executes
one entry without clobbering prior evidence. A non-passing entry stops automatic
execution; continuing with an unstarted ID requires an explicit reviewed finding
that the completed failure was numerical and the evidence system remains safe.
Each run records separate stdout/stderr, `fem-smoke.json`, `exit-code.json`, and
`SHA256SUMS`; the executor binds the complete physical problem, exact DoFs,
source, images, runtime, 28 GiB no-swap cgroup, host `memory.peak`, and the
registered 10,800-second wall-time cap. The exact public command sequence is in
the [README](../README.md#immutable-remote-scaling-sweep).
