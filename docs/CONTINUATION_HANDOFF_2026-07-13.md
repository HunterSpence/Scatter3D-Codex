# Continuation handoff — 2026-07-13

## Purpose

This document is the restart point for completing the strongest honest public
release supported by evidence. It is intentionally public-safe: it contains no
credentials, control-host details, SSH identities, server identifiers, or private
measurement/source material.

The project is an independently authored Apache-2.0 clean-room implementation.
It is not a fork or relicensing of third-party/private Scatt3D source for which
no compatible public license was verified. No code or private data from that
source may be copied, merged, cherry-picked, or published here.

## Exact checkpoint before this handoff

| Item | Exact value |
|---|---|
| Code checkpoint | `d5fe81486d0d8bf2611d4b4f8df8fa4b9aaf6ad4` |
| Branch | `codex/verification-release-20260712` |
| Preserved original checkpoint | `codex/checkpoint-20260712` at `85c40392e711cc2749f85330409951c4bf7f4824` |
| Pull request | [PR #1](https://github.com/HunterSpence/Scatter3D-Codex/pull/1), OPEN and MERGEABLE at the checkpoint |
| Exact CI | [GitHub Actions run 29214242189](https://github.com/HunterSpence/Scatter3D-Codex/actions/runs/29214242189), **PASSED** |
| Dedicated benchmark runner | None active; no Scatter3D benchmark runner was billing at handoff |

The commit containing this document is documentation-only relative to the code
checkpoint above. At restart, treat the current pushed PR head and its own exact
CI result as authoritative; do not assume a stale local `HEAD` is current.

## First commands at restart

Run these read-only checks before editing or provisioning compute:

```powershell
$repo = (Get-Location).Path  # run from the Scatter3D-Codex checkout
git -C $repo fetch origin
git -C $repo status --short --branch
git -C $repo log -5 --oneline --decorate
git -C $repo rev-parse HEAD
git -C $repo rev-parse origin/codex/verification-release-20260712
gh pr view 1 --repo HunterSpence/Scatter3D-Codex `
  --json url,state,mergeable,headRefOid,statusCheckRollup
gh run view 29214242189 --repo HunterSpence/Scatter3D-Codex --exit-status
```

Required result before continuing:

- local worktree clean;
- local and remote branch synchronized;
- PR head exact and mergeable;
- all checks on the current PR head **PASSED**;
- no pre-existing dedicated benchmark runner that could be double-billed.

If the PR head moved because this handoff was committed, use the newer exact-head
CI run. Run 29214242189 remains the immutable evidence for `d5fe814`.

## What is already complete — do not redo it

### Measurement and inverse contracts

- Canonical `S[angle, frequency, receiver, source]` ordering and explicit
  receiver/source meaning.
- Strict indexed long-form CSV import with coordinate, order, dtype, uniqueness,
  and hash validation.
- Same-index DUT-reference subtraction and reference-only optional alignment,
  disabled by default.
- Paired repeat-differential covariance, variance-of-the-mean handling,
  diagonal O(N) whitening, a guarded dense-covariance path, and explicit noise
  floors.
- Native-complex TSVD with fixed, GCV, discrepancy, and singular-energy
  selection plus complete diagnostics and no-clobber artifacts.
- Volume-weighted reconstruction metrics, deterministic provenance, CLI,
  examples, schemas, packaging, Docker, and CI.

### Maxwell solver

- DOLFINx 0.10/PETSc 3.24 complex128 3-D Maxwell forms for p=1/2/3 Nedelec
  fields with linear geometry.
- Independent REF/DUT material maps, Cartesian PML support, validated mesh/tag
  contracts, matched single-mode TEM software forms, repeated-RHS reuse, direct
  MUMPS reference, and iterative FGMRES paths.
- Physical Maxwell matrix `A` remains unshifted. A nonzero absorption shift is
  applied only to a separate preconditioning matrix `P` through
  `KSPSetOperators(A, P)`.
- Genuine two-level p=3-to-p=1 p-multigrid: PEC-masked interpolation,
  one-step Richardson/ASM fine smoothing with local MUMPS LU, and a global p=1
  MUMPS coarse correction.
- Requested and setup-observed PETSc hierarchy, raw KSP view, operator identity,
  true residual, DoF, nonzero, timing, RSS, factorization, and repeated-RHS
  counters.

### Fail-closed scaling tooling

`validation/scaling_sweep_v1.json` and the registration/executor added at
`d5fe814` establish:

- eight immutable runs: n=9/MPI4 and n=16/MPI8, each at absorption shifts
  `0.0`, `0.25`, `0.5`, and `1.0`;
- p=3 fine and p=1 coarse fields, 100 MHz, and two independent port RHS solves;
- exact DoF gates of 86,103 and 470,928;
- recomputed true relative residual at most `1e-7` and 1,000-iteration cap;
- 28 GiB cgroup memory limit, no swap, and 10,800-second wall-time cap per run;
- clean source, immutable image/base/runtime, complete physical-problem identity,
  and per-run output binding;
- write-once registration and run directories, separate stdout/stderr,
  `fem-smoke.json`, `exit-code.json`, and `SHA256SUMS`;
- exact-DoF mismatch preflight before any RHS solve;
- a Docker supervisor handshake that keeps PID 1 and the cgroup alive until the
  host captures positive `memory.peak`, `memory.max`, `memory.swap.max`, and
  `memory.events` before container cleanup;
- attempt-specific Docker ownership labels plus a stable absence window after
  ambiguous create timeouts, without deleting a pre-existing name conflict;
- honest preservation of `FAILED` solver results instead of converting them to
  skips.

Do not replace the supervisor with `docker start --attach` or read cgroup metrics
only after `docker wait`; the kernel cgroup may already have disappeared.

## Exact automated evidence at `d5fe814`

Run 29214242189 recorded all eight jobs **PASSED**:

| Gate | Result |
|---|---|
| Repository/static/security/actionlint/Compose/Bash/CFF/links | **PASSED** |
| Wheel and sdist build plus clean installs | **PASSED** |
| CPython 3.11 | **PASSED**: 182 passed, 23 deselected |
| CPython 3.12 | **PASSED**: 182 passed, 23 deselected |
| CPython 3.13 | **PASSED**: 182 passed, 23 deselected |
| CPython 3.14 | **PASSED**: 182 passed, 23 deselected |
| Complex DOLFINx heavy | **PASSED**: 21 passed, 184 deselected, zero skips |
| MPI selection on two ranks | **PASSED**: 2 logical tests passed, 203 deselected, zero skips |

Each MPI rank's JUnit file records the same two selected tests; do not describe
that as four independent tests.

### Runtime identity

| Item | Value |
|---|---|
| Base digest | `sha256:f7cce2a2271bf838c080751348c471064acb41fef0330e2c08178a688f71890d` |
| Heavy project image ID | `sha256:516786a15a4f17e5f31641d6ae78a25849c12f3d2b6a4012e72ad349bc6c12c8` |
| MPI project image ID | `sha256:1b67bc8ae057c62c242619cb133ae9ae02782e7a13b0521226718ea8f7471539` |
| DOLFINx | 0.10.0 |
| PETSc / petsc4py | 3.24.0 / 3.24.0, complex128 |
| PETSc arch | `linux-gnu-complex128-32` |
| mpi4py / MPI | 4.1.1 / MPICH 4.3.1 |
| HPDDM external package | false |
| SLEPc external package | false |

The two project image IDs differ because the heavy and MPI jobs built separate
local images. Both bind the same source revision and base digest.

### Current CI artifact hashes

Both downloaded `SHA256SUMS` files were independently rechecked: 8/8 targets
**PASSED** in each artifact set.

| Artifact | SHA-256 |
|---|---|
| Heavy `SHA256SUMS` | `e54040571663e7f1f5d90eb47e6eef419df4f309de274300255175c75eaba6d1` |
| MPI `SHA256SUMS` | `bb82fca8be7021d65fefbdfe1990714b55df2cbb59dcfdca100b06563075594c` |
| `fem-smoke-direct-p3.json` | `0a672f8a39a626144f7763bb5e7cbd81d0cfd604abb299fe23be61a9576e602c` |
| `fem-smoke-pmg-serial-p3-p1.json` | `007482af744c96c92beda86b107a2bc5204d84335270093e14d1e251d91d7253` |
| `fem-smoke-iterative-mpi2.json` | `7f40ae679b50c17d205866d56dc4069438c184db4a7a195751c731bbc88fea6a` |
| `fem-smoke-pmg-mpi2-p3-p1.json` | `98515035ccf623a1ec111889b5c4cc2da4cdc5ccd413e76860bd0ab623df2c15` |
| `manufactured-hcurl.json` | `70197800b299b65ba1f05a55f376f9fd3187611ab655788916e81ab62d72081c` |
| Wheel | `bf61f83f357eed5cb3847c598604e20c7b1dc29a1e71645a7efa438ba45aa7c9` |
| Sdist | `9f84a0edbb6355062fc1289a7b0203d113b27513491d07653079a953cc8b82cc` |

GitHub Actions artifacts expire. Attach the needed JSON/manifests to any release
that cites them.

### Current small solver results

| Case | Result |
|---|---|
| Direct p=3 serial | **PASSED**: 1,158 DoFs, two RHS, residuals `9.35e-15` and `7.48e-15` |
| p=3-to-p=1 pMG serial | **PASSED**: 1,158 fine / 98 coarse DoFs, 6/6 iterations, residuals `1.615e-10` and `1.417e-10` |
| One-level MPI2 correctness | **PASSED**: 98 DoFs, 13/14 iterations, residuals `6.75e-9` and `1.97e-9` |
| p=3-to-p=1 pMG MPI2 | **PASSED**: 1,158 fine / 98 coarse DoFs, 10/10 iterations, residuals `1.246e-9` and `1.298e-9` |

These are correctness cases, not mesh-scaling or memory evidence.

## Historical large-run evidence — unchanged

All historical results below use the one-level path at `c3c1ded`, not the new
two-level pMG sweep.

| Experiment | Status | Evidence |
|---|---|---|
| p=3, n=9, MPI4, 86,103 DoFs, ASM/ILU(0) | **FAILED** | Both RHS hit 1,000 iterations; best true residual about `4.93e-3` |
| p=3, n=9, MPI4, ASM/local MUMPS LU | **PASSED** | 233/230 iterations; residuals `9.3958e-9`, `9.9821e-9` |
| Identical direct MUMPS problem | **PASSED** | Residuals `9.59e-14`, `9.83e-14` |
| Same-problem peak-memory ratio at 86,103 DoFs | **FAILED** | Iterative `2,544,521,216`; direct `3,079,335,936`; ratio `0.8263214111` > `0.5` |
| p=3, n=16, MPI8, 470,928 DoFs, one-level local MUMPS LU | **FAILED** | Both RHS hit 1,000 iterations; residuals `2.4996e-7`, `5.5071e-6` |

Historical hashes:

- corrected local-LU n=9: `f7e7b1ffbb2830a0b80089ad58699b93db101e24cbbfb445a615c7e6ca2b0a89`;
- n=9 direct: `00be6c3560c39bb4f23b32409c5ff9a7c94298fa7b73c5e9b3fe9f5072fe4196`;
- n=9 memory gate: `5caf2f39e05106307d1c11aab4ac55eb80fb347d6b9c97f25cf8e72010d8f3ab`;
- n=9 ILU: `da385b3cc910970f97b1f81dd7cb9d4f1fb1cb3f38a256f5305bccda319ae2a2`;
- n=16 one-level LU: `e0f0e40bd1d1b100645087da3bbe22821682afc939455d0e16fa51957bdbddb7`.

Important: `p3-n9-mpi4-right-asm1-lu.json` with SHA-256
`bf437d5ac34ce1418a654c454628b5657b3866f0282b15882f1f988f74ceb362`
was generated before the PETSc option-lifecycle fix. It actually used ILU and
must never be cited as local-LU evidence.

These hashes were recomputed from the retained local evidence mirror. Reverify
the canonical files again before attaching them to a release.

## PETSc root cause already fixed

Prefixed options were once removed after `KSPSetFromOptions` but before
`KSPSetUp`. ASM creates/configures nested subdomain solvers during setup, so a
requested subdomain LU silently remained default ILU. `c3c1ded` retains the
options through setup and removes them in a `finally` block afterward. Heavy
regressions deliberately request invalid nested configuration and prove setup
consumes it. Effective hierarchy is now recorded, not inferred from requested
options.

Do not undo this lifecycle or trust requested PETSc options without setup-observed
evidence.

## Required next execution — in order

### 1. Select ephemeral Linux capacity

Use a disposable Linux Docker runner. Keep credential lookup, account details,
server IDs, IPs, SSH identities, and deletion API output outside this public
repository. Before provisioning, privately verify that no prior Scatter3D
benchmark runner remains active.

Preferred minimum for the registered sweep:

- 16 CPU threads;
- approximately 32 GiB RAM;
- at least 100 GiB free disk;
- Ubuntu 24.04 or equivalent;
- Docker with cgroup v2;
- no host swap;
- inbound firewall allowing SSH only.

Recheck larger-memory capacity before later 1M/3M work. Do not begin a charge
for an unavailable dedicated type, and do not leave a paid runner unattended.

### 2. Clone the exact pushed source and build an identity-bound image

```bash
set -euo pipefail
: "${VERIFIED_COMMIT:?set VERIFIED_COMMIT to the exact green PR headRefOid}"
git clone https://github.com/HunterSpence/Scatter3D-Codex.git repo
cd repo
git fetch origin "$VERIFIED_COMMIT"
git checkout --detach "$VERIFIED_COMMIT"
test -z "$(git status --porcelain --untracked-files=normal)"
SOURCE_COMMIT=$(git rev-parse HEAD)
test "$SOURCE_COMMIT" = "$VERIFIED_COMMIT"

docker build --file docker/Dockerfile \
  --build-arg SCATTER3D_GIT_COMMIT="$SOURCE_COMMIT" \
  --build-arg SCATTER3D_GIT_DIRTY=false \
  --tag scatter3d-codex:bench .

IMAGE_ID=$(docker image inspect --format '{{.Id}}' scatter3d-codex:bench)
mkdir -p ../evidence
docker image inspect "$IMAGE_ID" > ../evidence/image-inspect.json
docker run --rm --entrypoint cat "$IMAGE_ID" \
  /opt/scatter3d/runtime-metadata.json > ../evidence/runtime-metadata.json
docker info > ../evidence/docker-info.txt
uname -a > ../evidence/uname.txt
test "$(stat -fc %T /sys/fs/cgroup)" = cgroup2fs
```

Require the image labels and environment to contain the exact commit and
`dirty=false`. Require the base digest and complex runtime to match the pinned
contract.

Keep `image-inspect.json`, `docker-info.txt`, and `uname.txt` private until they
are sanitized. Remove hostnames, proxy/registry configuration, host labels, and
other account/infrastructure identifiers before attaching any excerpt publicly.

### 3. Linux lifecycle canary before the first paid solve

Exercise the real supervisor sequence on Linux before releasing an expensive
registered solve:

```text
container .executor-ready
host      .executor-go
child     exits and writes .solver-exit-code + .solver-done
PID 1     remains alive
host      resolves /proc/<Pid>/cgroup and reads cgroup-v2 memory files
host      writes .executor-collected
container exits with the child code
host      waits, inspects, logs, removes, and verifies cleanup
```

Acceptance:

- positive host `memory.peak`;
- `memory.max` equals the requested limit;
- `memory.swap.max` equals zero;
- `memory.events` contains required counters;
- child and Docker exit codes agree;
- supervisor control files are removed;
- cleanup attempted and succeeded;
- no leftover container;
- output files mode 0644 and manifests verify.

Run this cheap live-Docker canary. Its overall status is deliberately **FAILED**
because the tiny child does not publish a FEM artifact; all lifecycle/resource
assertions must nevertheless pass:

```bash
CANARY_PARENT=../evidence/lifecycle-canary
mkdir -p "$CANARY_PARENT"
python3 - "$IMAGE_ID" "$CANARY_PARENT" <<'PY'
import json
import sys
from pathlib import Path

from validation.run_registered_scaling_sweep import execute_registered_entries

image_id = sys.argv[1]
parent = Path(sys.argv[2]).resolve()
memory = 256 * 1024**2
output_root = "lifecycle-smoke-output"
run_id = "lifecycle-smoke"
run_root = f"{output_root}/{run_id}"
entry = {
    "run_id": run_id,
    "resource_contract": {
        "mpi_ranks": 1,
        "cgroup_memory_limit_bytes": memory,
        "wall_time_limit_seconds": 30,
    },
    "command": [
        "python3",
        "-c",
        "import time; x=bytearray(8*1024*1024); time.sleep(0.5); print(len(x))",
    ],
    "outputs": {
        "directory": run_root,
        "container_directory": "/artifacts/lifecycle-smoke",
        "fem_smoke_json": f"{run_root}/fem-smoke.json",
        "container_fem_smoke_json": "/artifacts/lifecycle-smoke/fem-smoke.json",
        "stdout_log": f"{run_root}/stdout.log",
        "stderr_log": f"{run_root}/stderr.log",
        "exit_code_json": f"{run_root}/exit-code.json",
        "sha256_manifest": f"{run_root}/SHA256SUMS",
    },
}
registration = {
    "registration_id": "lifecycle-smoke",
    "output_root": output_root,
    "images": {
        "project_image": {"kind": "local_image_id", "identity": image_id},
        "base_image": {
            "kind": "oci_digest",
            "identity": "sha256:f7cce2a2271bf838c080751348c471064acb41fef0330e2c08178a688f71890d",
        },
    },
    "entries": [entry],
}
result = execute_registered_entries(
    registration,
    image_reference=image_id,
    output_parent=parent,
)[0]
assert result["status"] == "FAILED"
assert result["reason"] == "registered solve produced no fem_smoke artifact"
assert result["docker_return_code"] == 0
assert result["solver_barrier_exit_code"] == 0
assert result["launch_prevented"] is False
assert result["timed_out"] is False
assert result["container_resource_contract_passed"] is True
assert result["container_cleanup_attempted"] is True
assert result["container_cleanup_succeeded"] is True
assert result["container_absence_verified"] is True
assert result["host_cgroup"]["peak_bytes"] > 0
assert result["host_cgroup"]["limit_bytes"] == memory
assert result["host_cgroup"]["swap_limit_bytes"] == 0
assert {"oom", "oom_kill", "max"} <= result["host_cgroup"]["events"].keys()
print(json.dumps(result, indent=2, sort_keys=True))
PY
CANARY_RUN_DIR="$CANARY_PARENT/lifecycle-smoke-output/lifecycle-smoke"
test -z "$(docker ps -aq --filter label=scatter3d.registration_id=lifecycle-smoke)"
test -z "$(find "$CANARY_RUN_DIR" -maxdepth 1 \
  \( -name '.executor-*' -o -name '.solver-*' \) -print -quit)"
for file in stdout.log stderr.log exit-code.json SHA256SUMS; do
  test "$(stat -c %a "$CANARY_RUN_DIR/$file")" = 644
done
(cd "$CANARY_RUN_DIR" && sha256sum -c SHA256SUMS)
```

Then run the small wrong-DoF heavy canary already covered in CI:

```bash
WRONG_DOF_DIR=../evidence/wrong-dof-canary
mkdir -p "$WRONG_DOF_DIR"
chmod 0777 "$WRONG_DOF_DIR"
set +e
docker run --rm --memory=536870912 --memory-swap=536870912 \
  --env SCATTER3D_PROJECT_IMAGE_ID="$IMAGE_ID" \
  --env SCATTER3D_BASE_IMAGE_DIGEST=sha256:f7cce2a2271bf838c080751348c471064acb41fef0330e2c08178a688f71890d \
  --volume "$(realpath "$WRONG_DOF_DIR"):/artifacts" \
  "$IMAGE_ID" python3 validation/fem_smoke.py \
    --solver iterative \
    --iterative-hierarchy p-multigrid \
    --p-multigrid-coarse-degree 1 \
    --iterative-local-pc lu \
    --degree 3 \
    --subdivisions 2 \
    --frequencies-hz 1.0e8 \
    --expected-global-dofs 1 \
    --output /artifacts/wrong-dof.json \
  > "$WRONG_DOF_DIR/stdout.log" 2> "$WRONG_DOF_DIR/stderr.log"
wrong_dof_rc=$?
set -e
test "$wrong_dof_rc" -eq 1
test "$(stat -c %a "$WRONG_DOF_DIR/wrong-dof.json")" = 644
python3 - "$WRONG_DOF_DIR/wrong-dof.json" "$VERIFIED_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["status"] == "FAILED"
assert payload["passed"] is False
assert payload["execution_phase"] == "preflight"
assert payload["source"]["commit"] == sys.argv[2]
assert payload["gates"]["expected_global_dofs"]["status"] == "FAILED"
for name in (
    "convergence_and_true_residual",
    "assembly_setup_rhs_counts",
    "p_multigrid_structure",
):
    assert payload["gates"][name]["status"] == "NOT RUN"
assert "rhs_solves" not in payload
PY
```

Any lifecycle/cgroup failure is **FAILED**. Fix it in a normal commit and obtain
new exact-head CI before registering or running the scaling sweep.

### 4. Create the registration once, outside the clean source tree

```bash
python3 validation/register_scaling_sweep.py \
  --repository . \
  --project-image-kind local_image_id \
  --project-image-identity "$IMAGE_ID" \
  --base-image-digest sha256:f7cce2a2271bf838c080751348c471064acb41fef0330e2c08178a688f71890d \
  --base-runtime-metadata ../evidence/runtime-metadata.json \
  --output ../evidence/registration.json

(cd ../evidence && sha256sum registration.json > registration.SHA256)
(cd ../evidence && sha256sum -c registration.SHA256)
test -z "$(git status --porcelain --untracked-files=normal)"
```

Never edit or replace this registration after the first solve begins.

### 5. Execute all eight registered runs sequentially

Run IDs, in order:

1. `p3-n9-mpi4-shift-0p00`
2. `p3-n9-mpi4-shift-0p25`
3. `p3-n9-mpi4-shift-0p50`
4. `p3-n9-mpi4-shift-1p00`
5. `p3-n16-mpi8-shift-0p00`
6. `p3-n16-mpi8-shift-0p25`
7. `p3-n16-mpi8-shift-0p50`
8. `p3-n16-mpi8-shift-1p00`

For each ID:

```bash
RUN_ID=p3-n9-mpi4-shift-0p00  # replace with the next registered ID
set +e
python3 validation/run_registered_scaling_sweep.py \
  --registration ../evidence/registration.json \
  --repository . \
  --image "$IMAGE_ID" \
  --output-parent ../evidence \
  --run-id "$RUN_ID"
executor_rc=$?
set -e

RESULT_JSON=$(python3 - "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

registration = json.loads(Path("../evidence/registration.json").read_text())
entry = next(item for item in registration["entries"] if item["run_id"] == sys.argv[1])
print(Path("../evidence") / entry["outputs"]["exit_code_json"])
PY
)
RESULT_STATUS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RESULT_JSON")

case "$RESULT_STATUS:$executor_rc" in
  PASSED:0) ;;
  FAILED:1) echo "$RUN_ID preserved as FAILED; inspect and archive before continuing" ;;
  BLOCKED:1) echo "$RUN_ID is BLOCKED; archive and stop for the external blocker"; exit 1 ;;
  *) echo "inconsistent executor result for $RUN_ID"; exit 1 ;;
esac
```

The executor exits nonzero for an honest `FAILED` run. Preserve the directory
and inspect it before continuing. The executor stops a multi-ID invocation after
any `FAILED` or `BLOCKED` entry; after independently confirming a numerical-only
failure and a safe runner/evidence state, launch the next unstarted registered ID
explicitly with `--run-id`. Do not rerun or overwrite the same run ID. An
interrupted entry remains preserved; only remaining unstarted IDs can proceed.

Stop immediately for corrupted evidence, source/image mismatch, missing cgroup
instrumentation, Docker instability, disk pressure, unsafe resource state, or
inability to archive/delete the runner. Do not parameter-fish or publish only
the best shift.

### 6. Inspect and mirror after every run

For each completed run require:

- `fem-smoke.json` when fem_smoke reached normal or preflight artifact
  publication. Timeout, OOM, interpreter failure, or rank crash may instead
  produce only `exit-code.json`, cgroup/state evidence, and logs; that absence is
  preserved **FAILED** evidence, not a reason to relabel it **BLOCKED**;
- `exit-code.json` with the exact registration/run IDs;
- stdout and stderr retained separately;
- child/Docker exit parity or explicit failure reason;
- host cgroup peak/limit/swap/events;
- complete requested/effective pMG hierarchy;
- two RHS diagnostics and recomputed true residuals when a solve ran;
- successful container cleanup for any **PASSED** result;
- `SHA256SUMS` verifies.

Mirror each completed directory to durable storage before proceeding when
practical. A numerical failure is evidence, not trash.

### 7. Archive and deprovision on every stop path

Before deleting the runner, archive:

- registration plus its hash;
- all eight completed/uncompleted status directories that exist;
- every JSON, stdout/stderr log, and per-run manifest;
- image inspection and runtime metadata;
- exact Git revision and clean-status evidence;
- Docker/OS/cgroup identity;
- secret-free provisioning metadata.

Verify all per-run manifests after the copy, then make and verify a top-level
manifest. Only after successful archival, delete the paid runner and privately
verify it no longer exists. Never leave it billing overnight unattended.

## Acceptance and status rules

Use only `PASSED`, `FAILED`, `NOT RUN`, and `BLOCKED` for gates.

### PASSED

A registered scaling run is **PASSED** only if all are true:

- correct registration, source, image, base, runtime, command, and full physical
  problem identities;
- exact global DoFs;
- both independent port RHS solves have positive PETSc convergence reason;
- both recomputed true relative residuals are at most `1e-7`;
- requested and effective pMG hierarchy/counts/operator identities agree;
- physical `A` remains unshifted and shift applies only to `P`;
- host cgroup peak is positive, 28 GiB limit is applied, swap is zero;
- no OOM, timeout, malformed state, corrupt artifact, or missing instrumentation;
- cleanup succeeds and all manifests verify.

### FAILED

Mark **FAILED** when a launched solve or required instrumentation misses a gate,
including iteration cap, residual, exact DoF, hierarchy, OOM, timeout, malformed
evidence, invalid cgroup data, or cleanup failure that prevents a pass claim.
Preserve it.

### BLOCKED

Mark **BLOCKED** only when the registered solve could not launch or an external
capacity/input prerequisite prevented execution. `BLOCKED` is not a pass and
must not replace a numerical failure.

### NOT RUN

Mark **NOT RUN** only when no attempt occurred for that gate.

## Decision after the eight-run sweep

- If all n=9 and n=16 runs finish, document every result and shift.
- If one or more n=16 runs **PASSED**, do not jump ad hoc to 1M/3M. Add the next
  rung through a new reviewed registration/specification and exact-head CI.
- If all pMG runs **FAILED**, retain them, analyze iteration/residual/effective
  hierarchy, and design the next coarse strategy without raising the iteration
  cap cosmetically.
- The current approximately 32 GiB runner is unlikely to support an assembled
  3M-DoF problem or a same-problem direct reference. Obtain adequate high-memory
  capacity before attempting that claim.
- If direct cannot run on the same 3M problem, write `3M MEMORY RATIO NOT PROVEN`.
  A smaller common problem cannot establish the 3M ratio.

## Truth boundary that must survive release work

| Claim | Current status |
|---|---|
| Measurement ordering/schema/hash/noise/TSVD software | **PASSED** on identified automated tests |
| Manufactured H(curl) p=1/2/3 | **PASSED** on archived evidence |
| Small direct and MPI solver correctness | **PASSED** |
| Small p=3-to-p=1 pMG serial/MPI correctness | **PASSED** |
| Historical one-level 86,103-DoF local-LU solve | **PASSED** |
| Historical one-level 470,928-DoF solve | **FAILED** |
| Historical same-problem <=50% memory gate | **FAILED**, ratio `0.8263214111` |
| New pMG 86,103/470,928 sweep | **NOT RUN** until the eight artifacts exist |
| Convergence at >=3,000,000 complex DoFs | **NOT RUN** |
| 3M same-problem <=50% memory ratio | **NOT RUN** / not proven |
| Matched-TEM boundary and electric-mode normalization software checks | **PASSED** |
| Calibrated incident/outgoing E/H extraction and physical S-parameters | **NOT RUN** |
| Independent analytical scattering and transmission-line/thru benchmarks | **NOT RUN** |
| PML reflection campaign, sensitivity, and linearization range | **NOT RUN** |
| HPDDM path in the pinned image | **BLOCKED**: HPDDM and SLEPc packages unavailable |
| Real POM/PLA reconstruction, hardware ordering, and real-data diagnosis | **BLOCKED** pending an accepted raw measurement bundle |

Never claim stable, production-ready, first, validated VNA imaging,
mesh-scalable, a memory win, million/3M capability, or validated physical
S-parameters without the corresponding evidence.

## Required real-data bundle

Real POM/PLA acceptance remains **BLOCKED** until an approved bundle contains:

- indexed raw complex reference and DUT repeats;
- stationary, motion/reseat, null, and known-target controls;
- calibration plane/state and complete frequency/angle/source/receiver/port
  coordinates;
- material properties with uncertainty;
- geometry/CAD/mesh identity and protocol metadata;
- acquisition settings and time/temperature/cable-motion records.

Private or unreviewed inputs must not be published or converted into a success
claim.

## Final merge and experimental release sequence

After remote evidence is archived and the paid runner is deleted:

1. Update README, verification, scaling evidence, changelog, and release notes
   with every run, command, hash, identity, and honest status.
2. Run local static/pure/package checks and push normally.
3. Require all checks **PASSED** on the final PR head.
4. Download and verify final CI artifacts/manifests.
5. Merge PR #1 normally; never force-push `main`.
6. Require CI **PASSED** on the actual `main` merge commit.
7. Tag that exact green `main` commit `v0.1.0`.
8. Publish an Experimental/Pre-release, not a stable release.
9. Attach wheel, sdist, runtime metadata, public/sanitized JSON evidence, every
   cited failure, and verified manifests.

Acceptable headline:

> Experimental v0.1.0 verification-first clean-room software infrastructure
> with tested measurement/inverse contracts and small FEM correctness evidence.

## Public/private boundary

The public release may contain source, synthetic fixtures, public CI artifacts,
sanitized benchmark evidence, and generic ephemeral-runner specifications.

It must not contain credentials, vault instructions, control-host details,
account/quota data, server IDs/IPs, SSH identities, local-user paths, unsanitized
remote logs, third-party private source/data/CAD, or raw VNA measurements
without explicit publication approval and acceptance-protocol success.
