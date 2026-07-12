#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

image="${1:-scatter3d-codex:verify}"

docker build --pull=false -f docker/Dockerfile -t "${image}" .

heavy_command=$(cat <<'BASH'
python3 -m pytest -p no:cacheprovider -W error -m "heavy and not mpi" -ra --junitxml=/tmp/heavy.xml
python3 - <<'PY'
import xml.etree.ElementTree as ET

root = ET.parse("/tmp/heavy.xml").getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
tests = sum(int(suite.get("tests", 0)) for suite in suites)
skipped = sum(int(suite.get("skipped", 0)) for suite in suites)
if tests == 0 or skipped:
    raise SystemExit(f"heavy verification invalid: tests={tests}, skipped={skipped}")
print(f"heavy verification complete: tests={tests}, skipped={skipped}")
PY
BASH
)
docker run --rm "${image}" bash -euc "${heavy_command}"

mpi_command=$(cat <<'BASH'
mpirun -n 2 sh -euc '
  python3 -m pytest -p no:cacheprovider -W error -m mpi -ra --junitxml=/tmp/mpi-${OMPI_COMM_WORLD_RANK}.xml
'
python3 - <<'PY'
import glob
import xml.etree.ElementTree as ET

files = glob.glob("/tmp/mpi-*.xml")
if len(files) != 2:
    raise SystemExit(f"expected two MPI reports, found {files}")
for path in files:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.get("tests", 0)) for suite in suites)
    skipped = sum(int(suite.get("skipped", 0)) for suite in suites)
    if tests == 0 or skipped:
        raise SystemExit(f"MPI verification invalid in {path}: tests={tests}, skipped={skipped}")
print("MPI verification complete on two ranks with no skips")
PY
BASH
)
docker run --rm --ipc=host "${image}" bash -euc "${mpi_command}"
