#!/usr/bin/env bash
set -euo pipefail

image="${1:-scatter3d-codex:verify}"

docker build --pull=false -f docker/Dockerfile -t "${image}" .
docker run --rm "${image}" python3 -m pytest -m "heavy and not mpi" -ra
docker run --rm --ipc=host "${image}" \
  mpirun -n 2 python3 -m pytest -m mpi -ra

echo "Container heavy and two-rank commands completed. Review test counts and CI skip gate."
