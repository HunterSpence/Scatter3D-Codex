# Reproducible numerical runtime

The image is based on the complex-scalar DOLFINx 0.10.0 image pinned by digest.
It does not install replacement NumPy, SciPy, MPI, PETSc, petsc4py, or DOLFINx
wheels. That avoids accidentally mixing real and complex PETSc builds.

```bash
docker build -f docker/Dockerfile -t scatter3d-codex:local .
docker run --rm scatter3d-codex:local python3 - <<'PY'
from petsc4py import PETSc
import dolfinx
import numpy as np
print(dolfinx.__version__, PETSc.ScalarType)
assert np.issubdtype(PETSc.ScalarType, np.complexfloating)
PY
```

Run one MPI process per allocated core and leave the thread caps at one unless a
controlled scaling experiment proves a different placement is faster:

```bash
docker run --rm --ipc=host scatter3d-codex:local \
  mpirun -n 2 python3 -m pytest -m mpi -ra
```

The base digest pins software, not CPU behavior. Record host CPU, rank count,
memory limit, and solver options alongside every production result.
