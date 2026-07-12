"""Command-line entry point for auditable file-level workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .pipeline import (
    diagnose_measurement_bundle,
    reconstruct_from_bundle,
    schema_description,
    validate_measurement_bundle,
    write_json_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scatter3d",
        description="Validate, diagnose, and invert coordinate-explicit microwave data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("schema", help="print the NPZ data contracts as JSON")

    validate = subparsers.add_parser("validate", help="validate a measurement NPZ bundle")
    validate.add_argument("bundle", type=Path)
    validate.add_argument("--json", dest="json_path", type=Path, help="also write a JSON report")

    diagnose = subparsers.add_parser("diagnose", help="measure repeat floor and basic consistency")
    diagnose.add_argument("bundle", type=Path)
    diagnose.add_argument("--json", dest="json_path", type=Path, help="also write a JSON report")

    invert = subparsers.add_parser("invert", help="run a coordinate-checked TSVD reconstruction")
    invert.add_argument("bundle", type=Path)
    invert.add_argument("sensitivity", type=Path)
    invert.add_argument("output", type=Path)
    invert.add_argument("--channel-mode", choices=("all", "transmission", "reflection"), default="all")
    invert.add_argument("--method", choices=("fixed", "gcv", "discrepancy", "energy"), default="gcv")
    invert.add_argument("--rank", type=int, help="required for --method fixed")
    invert.add_argument(
        "--noise-norm",
        type=float,
        help="optional solve-space target; whitened discrepancy defaults to sqrt(rows)",
    )
    invert.add_argument("--energy-fraction", type=float, default=0.999)
    invert.add_argument(
        "--whitening",
        choices=("auto", "off", "required"),
        default="auto",
        help="use paired-repeat diagonal whitening when available",
    )
    invert.add_argument("--noise-relative-floor", type=float, default=1.0e-12)
    invert.add_argument("--noise-absolute-floor", type=float, default=0.0)
    invert.add_argument("--json", dest="json_path", type=Path, help="also write a JSON report")
    return parser


def _emit(report: object, json_path: Path | None) -> None:
    payload = report.to_dict() if hasattr(report, "to_dict") else report
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if json_path is not None:
        write_json_report(report, json_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "schema":
            _emit(schema_description(), None)
        elif args.command == "validate":
            _emit(validate_measurement_bundle(args.bundle), args.json_path)
        elif args.command == "diagnose":
            _emit(diagnose_measurement_bundle(args.bundle), args.json_path)
        elif args.command == "invert":
            report = reconstruct_from_bundle(
                args.bundle,
                args.sensitivity,
                args.output,
                channel_mode=args.channel_mode,
                method=args.method,
                rank=args.rank,
                noise_norm=args.noise_norm,
                energy_fraction=args.energy_fraction,
                whitening=args.whitening,
                noise_relative_floor=args.noise_relative_floor,
                noise_absolute_floor=args.noise_absolute_floor,
            )
            _emit(report, args.json_path)
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(args.command)
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"scatter3d: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
