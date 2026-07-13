# The digest makes the numerical base reproducible even if the v0.10.0 tag moves.
FROM dolfinx/dolfinx:v0.10.0@sha256:f7cce2a2271bf838c080751348c471064acb41fef0330e2c08178a688f71890d

ARG SCATTER3D_GIT_COMMIT=""
ARG SCATTER3D_GIT_DIRTY=""

LABEL org.opencontainers.image.title="Scatter3D-Codex" \
      org.opencontainers.image.description="Complex DOLFINx/PETSc verification runtime" \
      org.opencontainers.image.source="https://github.com/HunterSpence/Scatter3D-Codex" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${SCATTER3D_GIT_COMMIT}" \
      org.opencontainers.image.base.name="dolfinx/dolfinx:v0.10.0" \
      org.opencontainers.image.base.digest="sha256:f7cce2a2271bf838c080751348c471064acb41fef0330e2c08178a688f71890d"

# Select the complex-scalar installation for non-interactive commands and cap
# threaded BLAS so MPI ranks do not oversubscribe a workstation or runner.
ENV PETSC_ARCH=linux-gnu-complex128-32 \
    PKG_CONFIG_PATH=/usr/local/dolfinx-complex/lib/pkgconfig \
    CMAKE_PREFIX_PATH=/usr/local/dolfinx-complex/lib/cmake \
    PYTHONPATH=/usr/local/dolfinx-complex/lib/python3.12/dist-packages:/dolfinx-env/lib/python3.12/site-packages:/usr/local/lib \
    LD_LIBRARY_PATH=/usr/local/dolfinx-complex/lib \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SCATTER3D_GIT_COMMIT="${SCATTER3D_GIT_COMMIT}" \
    SCATTER3D_GIT_DIRTY="${SCATTER3D_GIT_DIRTY}"

WORKDIR /opt/scatter3d

COPY docker/requirements-test.txt /tmp/requirements-test.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-test.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir --no-deps . \
    && python3 -m pip check

# Fail during the image build if the digest no longer exposes the expected
# DOLFINx release or a complex PETSc scalar. Emit exact PETSc/MPI build metadata
# into the build log so every verification run can identify its numerical ABI.
RUN python3 - <<'PY'
import json
import os
import re
from pathlib import Path

import dolfinx
import mpi4py
import numpy as np
import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

if re.match(r"^0\.10(?:\.|$)", dolfinx.__version__) is None:
    raise SystemExit(f"expected DOLFINx 0.10.x, found {dolfinx.__version__}")
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    raise SystemExit(f"complex PETSc required, found {np.dtype(PETSc.ScalarType)}")

metadata = {
    "dolfinx": dolfinx.__version__,
    "mpi_library": " ".join(MPI.Get_library_version().split()),
    "mpi_standard": ".".join(str(part) for part in MPI.Get_version()),
    "mpi4py": mpi4py.__version__,
    "petsc": ".".join(str(part) for part in PETSc.Sys.getVersion()),
    "petsc_arch": os.environ.get("PETSC_ARCH", ""),
    "petsc_scalar_type": str(np.dtype(PETSc.ScalarType)),
    "petsc_version_info": PETSc.Sys.getVersionInfo(),
    "petsc4py": petsc4py.__version__,
    "petsc_has_hpddm": bool(PETSc.Sys.hasExternalPackage("hpddm")),
    "petsc_has_slepc": bool(PETSc.Sys.hasExternalPackage("slepc")),
}
payload = json.dumps(metadata, indent=2, sort_keys=True)
Path("/opt/scatter3d/runtime-metadata.json").write_text(payload + "\n", encoding="utf-8")
print("scatter3d-runtime-metadata=" + json.dumps(metadata, sort_keys=True))
PY

RUN python3 -c "import scatter3d"

COPY tests ./tests
COPY examples ./examples
COPY docs ./docs
COPY validation ./validation

# The pinned DOLFINx base already provides an unprivileged UID/GID 1000.
# Reuse it instead of assuming the corresponding account name is absent.
RUN chown -R 1000:1000 /opt/scatter3d
USER 1000:1000

CMD ["python3", "-m", "scatter3d.cli", "--help"]
