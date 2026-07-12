"""Diagnostics that are independent of a particular mesh or linear backend."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any

from .config import ExperimentMaterials, LinearSolverConfig, Material, MaterialMap


@dataclass(frozen=True, slots=True)
class MaterialChange:
    volume_tag: int | None
    reference: Material
    dut: Material


@dataclass(frozen=True, slots=True)
class SolverComponentDiagnostics:
    """Effective PETSc types for one KSP/PC component after setup."""

    path: str
    ksp_type: str
    pc_type: str
    factor_solver_type: str | None
    options_prefix: str
    mpi_ranks: tuple[int, ...]
    instances: int


@dataclass(frozen=True, slots=True)
class RequestedSolverHierarchy:
    """Requested top-level hierarchy, including the exact PETSc options."""

    ksp_type: str
    preconditioning_side: str
    pc_type: str
    factor_solver_type: str | None
    options_prefix: str
    petsc_options: tuple[tuple[str, str | int | float | None], ...]


@dataclass(frozen=True, slots=True)
class EffectiveSolverHierarchy:
    """Observed PETSc hierarchy after nested objects have been created."""

    preconditioning_side: str
    top_level: SolverComponentDiagnostics
    relative_tolerance: float
    absolute_tolerance: float
    maximum_iterations: int
    asm_type: str | None
    asm_overlap: int | None
    asm_subdomain_solvers: tuple[SolverComponentDiagnostics, ...]
    petsc_view_ascii: str


@dataclass(frozen=True, slots=True)
class SolverHierarchyDiagnostics:
    """Side-by-side requested and setup-observed PETSc configuration."""

    requested: RequestedSolverHierarchy
    effective: EffectiveSolverHierarchy


def _factor_solver_type(pc: Any) -> str | None:
    if str(pc.getType()).lower() not in {"lu", "cholesky"}:
        return None
    getter = getattr(pc, "getFactorSolverType", None)
    if getter is None:
        return None
    value = getter()
    return None if value is None else str(value)


def _local_solver_component(ksp: Any, path: str) -> SolverComponentDiagnostics:
    pc = ksp.getPC()
    return SolverComponentDiagnostics(
        path=path,
        ksp_type=str(ksp.getType()),
        pc_type=str(pc.getType()),
        factor_solver_type=_factor_solver_type(pc),
        options_prefix=str(ksp.getOptionsPrefix() or ""),
        mpi_ranks=(),
        instances=1,
    )


def _aggregate_solver_components(
    comm: Any,
    local_components: tuple[SolverComponentDiagnostics, ...],
    path: str,
) -> tuple[SolverComponentDiagnostics, ...]:
    """Allgather and deduplicate live solver components across MPI ranks."""

    local_payload = tuple(
        (
            item.ksp_type,
            item.pc_type,
            item.factor_solver_type,
            item.options_prefix,
        )
        for item in local_components
    )
    gathered = comm.allgather(local_payload)
    groups: dict[tuple[str, str, str | None, str], dict[str, Any]] = {}
    for rank, components in enumerate(gathered):
        for component in components:
            group = groups.setdefault(component, {"ranks": set(), "instances": 0})
            group["ranks"].add(rank)
            group["instances"] += 1
    return tuple(
        SolverComponentDiagnostics(
            path=path if len(groups) == 1 else f"{path}.variant[{index}]",
            ksp_type=key[0],
            pc_type=key[1],
            factor_solver_type=key[2],
            options_prefix=key[3],
            mpi_ranks=tuple(sorted(group["ranks"])),
            instances=int(group["instances"]),
        )
        for index, (key, group) in enumerate(
            sorted(groups.items(), key=lambda item: repr(item[0]))
        )
    )


def _consistent_rank_value(comm: Any, value: Any, name: str) -> Any:
    values = comm.allgather(value)
    if any(item != values[0] for item in values[1:]):
        raise RuntimeError(f"effective PETSc {name} differs across MPI ranks: {values}")
    return values[0]


_ASM_OVERLAP_PATTERN = re.compile(
    r"\bamount of overlap\s*=\s*(\d+)\b", re.IGNORECASE
)
_ASM_TYPE_PATTERN = re.compile(
    r"\brestriction/interpolation type\s*-\s*([A-Za-z_]+)\b",
    re.IGNORECASE,
)


def parse_petsc_asm_view(view_text: str) -> tuple[str, int]:
    """Parse PETSc 3.24's official ASCII ``PCView_ASM`` fields."""

    type_match = _ASM_TYPE_PATTERN.search(view_text)
    overlap_match = _ASM_OVERLAP_PATTERN.search(view_text)
    missing = []
    if type_match is None:
        missing.append("restriction/interpolation type")
    if overlap_match is None:
        missing.append("amount of overlap")
    if missing:
        raise ValueError(
            "PETSc ASM view is missing required field(s): " + ", ".join(missing)
        )
    assert type_match is not None and overlap_match is not None
    return type_match.group(1).lower(), int(overlap_match.group(1))


def _remove_temporary_view(path: str) -> None:
    """Best-effort cleanup that must not strand peers at a later barrier."""

    with suppress(OSError):
        Path(path).unlink()


def _collective_temporary_view_path(comm: Any) -> str:
    """Create a rank-0 temporary path and broadcast success or failure."""

    envelope: tuple[str | None, str | None] | None = None
    if comm.rank == 0:
        descriptor = None
        path = None
        try:
            descriptor, path = tempfile.mkstemp(
                prefix="scatter3d-ksp-view-", suffix=".txt"
            )
            os.close(descriptor)
            descriptor = None
            envelope = (path, None)
        except OSError as exc:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            if path is not None:
                _remove_temporary_view(path)
            envelope = (None, f"{type(exc).__name__}: {exc}")
    path, error = comm.bcast(envelope, root=0)
    if error is not None:
        raise RuntimeError(f"failed to create PETSc ASCII KSP view: {error}")
    if path is None:
        raise RuntimeError("failed to create PETSc ASCII KSP view: missing path")
    return path


def _capture_petsc_ascii_view(ksp: Any, comm: Any) -> str:
    """Collectively capture ``KSPView`` through PETSc's supported ASCII viewer."""

    from petsc4py import PETSc

    path = _collective_temporary_view_path(comm)
    viewer = None
    try:
        try:
            viewer = PETSc.Viewer().createASCII(
                path, mode=PETSc.Viewer.FileMode.WRITE, comm=comm
            )
            ksp.view(viewer)
        finally:
            if viewer is not None:
                viewer.destroy()
        comm.barrier()
        payload: tuple[str | None, str | None] | None = None
        if comm.rank == 0:
            try:
                payload = (Path(path).read_text(encoding="utf-8"), None)
            except OSError as exc:
                payload = (None, f"{type(exc).__name__}: {exc}")
        text, error = comm.bcast(payload, root=0)
        if error is not None:
            raise RuntimeError(f"failed to read PETSc ASCII KSP view: {error}")
        assert text is not None
        return text
    finally:
        comm.barrier()
        if comm.rank == 0:
            _remove_temporary_view(path)
        comm.barrier()


def validate_effective_solver_hierarchy(
    hierarchy: SolverHierarchyDiagnostics, config: LinearSolverConfig
) -> None:
    """Fail closed if PETSc changed any typed top-level solver setting."""

    effective = hierarchy.effective
    top = effective.top_level
    expected_ksp = config.ksp_type.lower()
    expected_pc = config.pc_type.lower()
    actual_ksp = top.ksp_type.lower()
    actual_pc = top.pc_type.lower()
    if actual_ksp != expected_ksp:
        raise RuntimeError(
            f"effective top-level KSP {actual_ksp!r} != requested {expected_ksp!r}"
        )
    if actual_pc != expected_pc:
        raise RuntimeError(
            f"effective top-level PC {actual_pc!r} != requested {expected_pc!r}"
        )
    factor_types = {"lu", "cholesky"}
    if config.is_iterative and actual_pc in factor_types:
        raise RuntimeError("iterative solver resolved to a global factorizing PC")
    if config.is_direct and actual_pc not in factor_types:
        raise RuntimeError("direct solver did not resolve to a factorizing PC")
    if (
        config.factor_solver_type is not None
        and (top.factor_solver_type or "").lower()
        != config.factor_solver_type.lower()
    ):
        raise RuntimeError(
            "effective factor solver "
            f"{top.factor_solver_type!r} != requested {config.factor_solver_type!r}"
        )
    if effective.preconditioning_side != config.preconditioning_side:
        raise RuntimeError(
            "effective preconditioning side "
            f"{effective.preconditioning_side!r} != requested "
            f"{config.preconditioning_side!r}"
        )
    if not isclose(
        effective.relative_tolerance,
        config.relative_tolerance,
        rel_tol=1.0e-15,
        abs_tol=0.0,
    ) or not isclose(
        effective.absolute_tolerance,
        config.absolute_tolerance,
        rel_tol=1.0e-15,
        abs_tol=0.0,
    ):
        raise RuntimeError("effective PETSc tolerances differ from typed configuration")
    if effective.maximum_iterations != config.maximum_iterations:
        raise RuntimeError(
            "effective PETSc maximum iterations differs from typed configuration"
        )


def inspect_petsc_solver_hierarchy(
    ksp: Any,
    config: LinearSolverConfig,
    option_prefix: str,
    comm: Any,
) -> SolverHierarchyDiagnostics:
    """Inspect actual top-level and ASM-nested solver types after ``KSPSetUp``.

    The requested options are retained separately because PETSc does not expose
    getters for every ASM option (notably overlap) through petsc4py.  Nested KSP
    and PC types, including local factor backends, are queried from the live
    objects and are therefore not inferred from the request.
    """

    from petsc4py import PETSc

    side_value = ksp.getPCSide()
    side = {
        PETSc.PC.Side.LEFT: "left",
        PETSc.PC.Side.RIGHT: "right",
        PETSc.PC.Side.SYMMETRIC: "symmetric",
    }.get(side_value, str(side_value))
    side = _consistent_rank_value(comm, side, "preconditioning side")
    tolerances = _consistent_rank_value(
        comm,
        tuple(ksp.getTolerances()),
        "top-level tolerances",
    )
    pc = ksp.getPC()
    top_levels = _aggregate_solver_components(
        comm, (_local_solver_component(ksp, "top"),), "top"
    )
    if len(top_levels) != 1:
        raise RuntimeError(
            "effective top-level PETSc hierarchy differs across MPI ranks"
        )
    asm_type = None
    asm_overlap = None
    view_text = _capture_petsc_ascii_view(ksp, comm)
    subdomains: tuple[SolverComponentDiagnostics, ...] = ()
    if str(pc.getType()).lower() == "asm":
        asm_type, asm_overlap = parse_petsc_asm_view(view_text)
        local_subdomains = tuple(
            _local_solver_component(sub_ksp, f"asm.subdomain[{index}]")
            for index, sub_ksp in enumerate(pc.getASMSubKSP())
        )
        subdomains = _aggregate_solver_components(
            comm, local_subdomains, "asm.subdomains"
        )
    return SolverHierarchyDiagnostics(
        requested=RequestedSolverHierarchy(
            ksp_type=config.ksp_type,
            preconditioning_side=config.preconditioning_side,
            pc_type=config.pc_type,
            factor_solver_type=config.factor_solver_type,
            options_prefix=option_prefix,
            petsc_options=tuple(
                (str(key), value)
                for key, value in sorted(config.petsc_options.items())
            ),
        ),
        effective=EffectiveSolverHierarchy(
            preconditioning_side=side,
            top_level=top_levels[0],
            relative_tolerance=float(tolerances[0]),
            absolute_tolerance=float(tolerances[1]),
            maximum_iterations=int(tolerances[3]),
            asm_type=asm_type,
            asm_overlap=asm_overlap,
            asm_subdomain_solvers=subdomains,
            petsc_view_ascii=view_text,
        ),
    )


def compare_material_models(materials: ExperimentMaterials) -> tuple[MaterialChange, ...]:
    """Report reference/DUT changes without conflating their two model states."""

    reference = materials.reference
    dut = materials.dut
    changes: list[MaterialChange] = []
    if reference.default != dut.default:
        changes.append(MaterialChange(None, reference.default, dut.default))
    for tag in sorted(set(reference.regions) | set(dut.regions)):
        ref_material = reference.for_tag(tag)
        dut_material = dut.for_tag(tag)
        if ref_material != dut_material:
            changes.append(MaterialChange(tag, ref_material, dut_material))
    return tuple(changes)


def petsc_true_relative_residual(matrix: Any, solution: Any, rhs: Any) -> tuple[float, float]:
    """Compute ||b-Ax|| and ||b-Ax||/max(||b||, tiny) with PETSc vectors."""

    residual = rhs.duplicate()
    matrix.mult(solution, residual)
    residual.aypx(-1.0, rhs)  # residual = rhs - matrix * solution
    absolute = float(residual.norm())
    rhs_norm = float(rhs.norm())
    relative = absolute / max(rhs_norm, 2.225_073_858_507_201_4e-308)
    residual.destroy()
    return absolute, relative


def process_peak_rss_bytes() -> int | None:
    """Best-effort process high-water RSS without adding a runtime dependency."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        try:
            import psutil

            return int(psutil.Process().memory_info().peak_wset)
        except (ImportError, AttributeError, OSError):
            return None


def material_map_for_volume_tags(
    material_map: MaterialMap, volume_tags: tuple[int, ...]
) -> dict[int, Material]:
    """Resolve default material semantics for checkpointing and reporting."""

    return {tag: material_map.for_tag(tag) for tag in volume_tags}
