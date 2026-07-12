# The digest makes the numerical base reproducible even if the v0.10.0 tag moves.
FROM dolfinx/dolfinx:v0.10.0@sha256:f7cce2a2271bf838c080751348c471064acb41fef0330e2c08178a688f71890d

LABEL org.opencontainers.image.title="Scatter3D-Codex" \
      org.opencontainers.image.description="Complex DOLFINx/PETSc verification runtime" \
      org.opencontainers.image.source="https://github.com/HunterSpence/Scatter3D-Codex" \
      org.opencontainers.image.licenses="Apache-2.0"

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
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/scatter3d

COPY docker/requirements-test.txt /tmp/requirements-test.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-test.txt \
    && python3 -c "import numpy as np; from petsc4py import PETSc; assert np.issubdtype(PETSc.ScalarType, np.complexfloating)"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir --no-deps . \
    && python3 -c "import scatter3d"

COPY tests ./tests
COPY examples ./examples
COPY docs ./docs
COPY validation ./validation

RUN useradd --create-home --uid 1000 scatter3d \
    && chown -R scatter3d:scatter3d /opt/scatter3d
USER scatter3d

CMD ["python3", "-m", "scatter3d.cli", "--help"]
