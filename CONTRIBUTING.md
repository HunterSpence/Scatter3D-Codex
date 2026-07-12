# Contributing

Scatter3D-Codex uses a clean-room, evidence-first development process. Submit
only code you wrote or code whose license is compatible with Apache-2.0. Do not
copy implementation text from the Scatt3D repository that motivated this work.

## Development setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -m "not heavy and not mpi"
python -m ruff check .
```

The DOLFINx/PETSc stack is tested in the pinned container:

```bash
docker build --pull=false -f docker/Dockerfile -t scatter3d-codex:test .
docker run --rm scatter3d-codex:test \
  python3 -m pytest -p no:cacheprovider -W error -m "heavy and not mpi" -ra
docker run --rm --ipc=host scatter3d-codex:test \
  mpirun -n 2 python3 -m pytest -p no:cacheprovider -W error -m mpi -ra
```

Run `examples/run_container_verification.sh` for the full helper that also
rejects zero-test and skipped-test heavy/MPI reports.

## Pull-request requirements

1. Explain the physical or numerical contract being changed.
2. Add a failing regression test before or with the repair.
3. Report commands actually executed; label unexecuted checks explicitly.
4. Preserve complex values and declare every array axis and unit.
5. For solver changes, report PETSc reason and true relative residual.
6. For performance changes, compare identical meshes, MPI ranks, tolerances, and
   cold/warm compilation state.
7. Do not claim measured-image validation without an archived, checksummed input
   bundle and a reproducible acceptance report.

Heavy tests must not silently skip in CI. If a test cannot run in the pinned
runtime, fix the runtime or mark the pull request blocked.

## Commit style

Use focused, imperative commits such as `Validate receiver/source row order`.
Generated solver output and private measurement data do not belong in Git.
