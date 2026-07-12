"""Create a deterministic, tiny end-to-end canary dataset.

This is a software-contract example, not evidence that a physical object can be
imaged.  The generated inverse problem is deliberately linear and noise-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scatter3d.pipeline import SCHEMA_VERSION, SENSITIVITY_SCHEMA_VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("example-output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260712)
    frequencies_hz = np.array([5.0e9, 5.5e9])
    angles_deg = np.array([0.0, 15.0])
    port_labels = np.array(["P1", "P2"])
    measurement_shape = (angles_deg.size, frequencies_hz.size, 2, 2)
    rows, voxels = int(np.prod(measurement_shape)), 4
    A = rng.normal(size=(rows, voxels)) + 1j * rng.normal(size=(rows, voxels))
    truth = np.array([0.6 + 0.05j, -0.2 + 0.1j, 0.35 - 0.15j, 0.1 + 0.0j])
    b = A @ truth
    reference = np.zeros((1, *measurement_shape), dtype=np.complex128)
    dut = b.reshape((1, *measurement_shape))

    np.savez_compressed(
        args.output_dir / "measurement.npz",
        schema_version=np.array(SCHEMA_VERSION),
        reference_s=reference,
        dut_s=dut,
        frequencies_hz=frequencies_hz,
        angles_deg=angles_deg,
        port_labels=port_labels,
    )
    np.savez_compressed(
        args.output_dir / "sensitivity.npz",
        schema_version=np.array(SENSITIVITY_SCHEMA_VERSION),
        A=A,
        frequencies_hz=frequencies_hz,
        angles_deg=angles_deg,
        port_labels=port_labels,
    )
    np.savez_compressed(args.output_dir / "truth.npz", delta_epsilon_r=truth)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
