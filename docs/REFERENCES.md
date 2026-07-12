# Primary references

This bibliography supports the measurement runbook, numerical formulation, and
verification gates. A citation here is not evidence that Scatter3D-Codex has
reproduced a paper's hardware result; reproduced results are listed only in
`VERIFICATION.md` with commands and artifacts.

## Imaging method and experiment context

- EuCAP 2025, *Three-Dimensional Microwave Imaging Using a Scattering-Parameter
  Data Equation*: <https://doi.org/10.23919/EuCAP63536.2025.10999660>
- URSI EMTS 2025 follow-up paper: <https://www.ursi.org/proceedings/commission/ComB/EMTS/2025/papers/144.pdf>
- Fresnel Institute 3-D electromagnetic inverse-scattering database:
  <https://www.fresnel.fr/3Ddatabase/>
- Oblique-illumination microwave tomography study:
  <https://doi.org/10.3390/s16071046>

The supplied EuCAP experiment is a synthetic demonstration. It does not, by
itself, validate transfer of a simulated sensitivity operator to a real VNA,
antennas, cables, fixtures, and rotating target.

## VNA calibration and data interchange

- Keysight full four-port calibration:
  <https://helpfiles.keysight.com/csg/e5071c/measurement/calibration/basic_calibrations/full_4_port_calibration.htm>
- Keysight systematic, random, and drift errors:
  <https://helpfiles.keysight.com/csg/N1930xB/VNACalAndMeas/Errors.htm>
- Keysight VNA troubleshooting:
  <https://helpfiles.keysight.com/csg/pxivna/Support/Tshoot.htm>
- NIST multiport VNA calibration and uncertainty:
  <https://www.nist.gov/publications/three-port-vector-network-analyzer-calibrations-using-nist-microwave-uncertainty>
- Touchstone 2.0 specification: <https://ibis.org/touchstone_ver2.0/touchstone_ver2_0.pdf>

Scatter3D-Codex's indexed long-form CSV is an explicit project schema, not a
Touchstone file. Conversion must preserve the receiver/source convention and
must never infer a port order silently.

## DOLFINx and PETSc

- DOLFINx 0.10 PML demo:
  <https://docs.fenicsproject.org/dolfinx/v0.10.0.post3/python/demos/demo_pml.html>
- DOLFINx scattering-boundary-condition demo:
  <https://docs.fenicsproject.org/dolfinx/v0.10.0/python/demos/demo_scattering_boundary_conditions.html>
- PETSc KSP manual and repeated right-hand sides:
  <https://petsc.org/main/manual/ksp/>
- PETSc MUMPS interface: <https://petsc.org/main/manualpages/Mat/MATSOLVERMUMPS/>
- PETSc HPDDM preconditioner: <https://petsc.org/main/manualpages/PC/PCHPDDM.html>
- PETSc residual norm selection: <https://petsc.org/main/manualpages/KSP/KSPSetNormType/>
- PETSc convergence semantics: <https://petsc.org/main/manualpages/KSP/KSPConvergedDefault/>
- PETSc matrix memory metadata: <https://petsc.org/release/manualpages/Mat/MatGetInfo/>

## Scalable Maxwell solvers

- Two-level domain decomposition for high-frequency Maxwell systems:
  <https://doi.org/10.1090/mcom/3447>
- hypre AMS formulation and high-order requirements:
  <https://hypre.readthedocs.io/en/latest/solvers-ams.html>
- hypre scalar-type limitations: <https://hypre.readthedocs.io/en/stable/ch-intro.html>
- DOLFINx matrix-free PETSc demonstration:
  <https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_matrix-free-petsc.html>
- PETSc shell-matrix limitations: <https://petsc.org/main/manualpages/Mat/MATSHELL/>

One-level ASM/ILU is included only as a portable baseline. The intended
large-problem research path is right-preconditioned FGMRES with a shifted
Maxwell surrogate and a verified two-level coarse correction. Neither the
3,000,000-DoF target nor a 50% memory reduction may be claimed until the
same-problem benchmark in `CONVERGENCE_PROTOCOL.md` passes.
