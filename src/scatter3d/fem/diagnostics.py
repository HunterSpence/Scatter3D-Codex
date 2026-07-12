"""Diagnostics that are independent of a particular mesh or linear backend."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from .config import ExperimentMaterials, Material, MaterialMap


@dataclass(frozen=True, slots=True)
class MaterialChange:
    volume_tag: int | None
    reference: Material
    dut: Material


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
