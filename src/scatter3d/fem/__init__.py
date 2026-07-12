"""Finite-element building blocks for frequency-domain Maxwell problems.

The package deliberately keeps mesh/tag contracts, constitutive data, weak
forms, excitation modes, linear solves, and checkpoint identity separate.  A
caller can therefore validate every interface before spending time on a sweep.
"""

from .checkpoints import CheckpointIdentity, checkpoint_fingerprint, sha256_file
from .config import (
    ExperimentMaterials,
    LinearSolverConfig,
    Material,
    MaterialMap,
    MaxwellProblemConfig,
    PMLConfig,
)
from .diagnostics import MaterialChange, compare_material_models
from .gmsh_io import LoadedMesh, load_gmsh_mesh
from .ports import (
    MatchedTEMPortExcitation,
    NormalizedPortMode,
    PortDefinition,
    UncalibratedSurfaceCurrentExcitation,
    normalize_port_mode,
    total_electric_modal_coefficient,
)
from .solver import (
    ExperimentSweepResult,
    FrequencyDiagnostics,
    MaxwellSweepSolver,
    SweepResult,
)
from .tags import (
    BoundaryTagContract,
    MeshTagContract,
    TagContractError,
    VolumeTagContract,
    validate_mesh_tags,
)

__all__ = [
    "BoundaryTagContract",
    "CheckpointIdentity",
    "ExperimentMaterials",
    "ExperimentSweepResult",
    "FrequencyDiagnostics",
    "LinearSolverConfig",
    "LoadedMesh",
    "MatchedTEMPortExcitation",
    "Material",
    "MaterialChange",
    "MaterialMap",
    "MaxwellProblemConfig",
    "MaxwellSweepSolver",
    "MeshTagContract",
    "NormalizedPortMode",
    "PMLConfig",
    "PortDefinition",
    "SweepResult",
    "TagContractError",
    "UncalibratedSurfaceCurrentExcitation",
    "VolumeTagContract",
    "checkpoint_fingerprint",
    "compare_material_models",
    "load_gmsh_mesh",
    "normalize_port_mode",
    "sha256_file",
    "total_electric_modal_coefficient",
    "validate_mesh_tags",
]
