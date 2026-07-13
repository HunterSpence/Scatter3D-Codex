#!/usr/bin/env python3
"""Fail-closed filesystem-capacity preflight for disposable Linux runners.

The report deliberately contains labels and filesystem statistics, but not paths,
hostnames, environment variables, or raw operating-system errors.  This keeps the
machine-independent JSON suitable for evidence archives without leaking runner
identity or credential-bearing path components.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

SCHEMA = "scatter3d.validation.remote_capacity_preflight/v1"
PASSED = "PASSED"
FAILED = "FAILED"
MAXIMUM_FREE_BYTES = (1 << 63) - 1

_SAFE_LABEL = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_POSITIVE_INTEGER = re.compile(r"^[1-9][0-9]*$")
_MAXIMUM_FREE_BYTES_TEXT = str(MAXIMUM_FREE_BYTES)


class CapacityPreflightInputError(ValueError):
    """An input error with a stable, non-sensitive public error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _SanitizedArgumentParser(argparse.ArgumentParser):
    """Convert argparse failures into stable evidence without echoing argv."""

    def error(self, message: str) -> NoReturn:
        del message
        raise CapacityPreflightInputError(
            "INVALID_COMMAND_LINE",
            "command-line arguments are invalid",
        )


@dataclass(frozen=True, slots=True)
class LabeledPath:
    """A validated capacity target; its path is intentionally not serialized."""

    label: str
    path: Path


def _resolve_existing_absolute_path(candidate: Path) -> Path:
    try:
        absolute = candidate.is_absolute()
    except (OSError, RuntimeError, ValueError):
        raise CapacityPreflightInputError(
            "PATH_NOT_ACCESSIBLE",
            "capacity paths must exist and be accessible",
        ) from None
    if not absolute:
        raise CapacityPreflightInputError(
            "PATH_NOT_ABSOLUTE",
            "capacity paths must be absolute",
        )
    try:
        return candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise CapacityPreflightInputError(
            "PATH_NOT_ACCESSIBLE",
            "capacity paths must exist and be accessible",
        ) from None


def parse_minimum_free_bytes(value: int | str) -> int:
    """Return a strictly positive byte threshold or raise a stable input error."""

    if isinstance(value, bool):
        raise CapacityPreflightInputError(
            "INVALID_MINIMUM_FREE_BYTES",
            "minimum free bytes must be a positive base-10 integer",
        )
    if isinstance(value, int):
        threshold = value
    elif isinstance(value, str) and _POSITIVE_INTEGER.fullmatch(value) is not None:
        if len(value) > len(_MAXIMUM_FREE_BYTES_TEXT) or (
            len(value) == len(_MAXIMUM_FREE_BYTES_TEXT) and value > _MAXIMUM_FREE_BYTES_TEXT
        ):
            raise CapacityPreflightInputError(
                "INVALID_MINIMUM_FREE_BYTES",
                "minimum free bytes must be a positive base-10 integer",
            )
        threshold = int(value)
    else:
        raise CapacityPreflightInputError(
            "INVALID_MINIMUM_FREE_BYTES",
            "minimum free bytes must be a positive base-10 integer",
        )
    if threshold <= 0 or threshold > MAXIMUM_FREE_BYTES:
        raise CapacityPreflightInputError(
            "INVALID_MINIMUM_FREE_BYTES",
            "minimum free bytes must be a positive base-10 integer",
        )
    return threshold


def parse_labeled_paths(specifications: Iterable[str]) -> tuple[LabeledPath, ...]:
    """Validate ``LABEL=ABSOLUTE_PATH`` specifications and return label order.

    Labels are intentionally limited to a small ASCII vocabulary so they cannot
    inject terminal control characters, path fragments, or structured-output
    delimiters into evidence.  Paths must already exist and resolve strictly.
    """

    targets: list[LabeledPath] = []
    labels: set[str] = set()
    for specification in specifications:
        if not isinstance(specification, str) or "=" not in specification:
            raise CapacityPreflightInputError(
                "MALFORMED_PATH_SPECIFICATION",
                "each path must use LABEL=ABSOLUTE_PATH",
            )
        label, raw_path = specification.split("=", 1)
        if _SAFE_LABEL.fullmatch(label) is None:
            raise CapacityPreflightInputError(
                "UNSAFE_PATH_LABEL",
                "path labels must match [a-z][a-z0-9_-]{0,63}",
            )
        if label in labels:
            raise CapacityPreflightInputError(
                "DUPLICATE_PATH_LABEL",
                "path labels must be unique",
            )
        try:
            candidate = Path(raw_path)
        except (TypeError, ValueError):
            raise CapacityPreflightInputError(
                "PATH_NOT_ACCESSIBLE",
                "capacity paths must exist and be accessible",
            ) from None
        resolved = _resolve_existing_absolute_path(candidate)
        labels.add(label)
        targets.append(LabeledPath(label=label, path=resolved))
    if not targets:
        raise CapacityPreflightInputError(
            "NO_PATHS",
            "at least one labeled path is required",
        )
    return tuple(sorted(targets, key=lambda item: item.label))


def _system_statvfs(path: Path) -> Any:
    """Call the POSIX filesystem API without importing or invoking shell tools."""

    implementation = getattr(os, "statvfs", None)
    if implementation is None:
        raise OSError("statvfs is unavailable")
    return implementation(path)


def _nonnegative_integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError
    return value


def _filesystem_bytes(statistics: Any) -> tuple[int, int]:
    """Return total and unprivileged-available bytes from a statvfs result."""

    try:
        fragment_size = _nonnegative_integer(statistics.f_frsize)
        total_blocks = _nonnegative_integer(statistics.f_blocks)
        available_blocks = _nonnegative_integer(statistics.f_bavail)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("invalid statvfs result") from exc
    if fragment_size == 0 or available_blocks > total_blocks:
        raise ValueError("invalid statvfs result")
    return total_blocks * fragment_size, available_blocks * fragment_size


def _validated_targets(targets: Iterable[LabeledPath]) -> tuple[LabeledPath, ...]:
    """Independently validate objects supplied to the public evaluator."""

    try:
        supplied = tuple(targets)
    except TypeError:
        raise CapacityPreflightInputError(
            "INVALID_CAPACITY_TARGET",
            "capacity targets must be validated labeled paths",
        ) from None
    if not supplied:
        raise CapacityPreflightInputError(
            "NO_PATHS",
            "at least one labeled path is required",
        )

    labels: set[str] = set()
    validated: list[LabeledPath] = []
    for target in supplied:
        if (
            not isinstance(target, LabeledPath)
            or not isinstance(target.label, str)
            or not isinstance(target.path, Path)
        ):
            raise CapacityPreflightInputError(
                "INVALID_CAPACITY_TARGET",
                "capacity targets must be validated labeled paths",
            )
        if _SAFE_LABEL.fullmatch(target.label) is None:
            raise CapacityPreflightInputError(
                "UNSAFE_PATH_LABEL",
                "path labels must match [a-z][a-z0-9_-]{0,63}",
            )
        if target.label in labels:
            raise CapacityPreflightInputError(
                "DUPLICATE_PATH_LABEL",
                "path labels must be unique",
            )
        labels.add(target.label)
        validated.append(
            LabeledPath(
                label=target.label,
                path=_resolve_existing_absolute_path(target.path),
            )
        )
    return tuple(sorted(validated, key=lambda item: item.label))


def evaluate_capacity(
    targets: Iterable[LabeledPath],
    minimum_free_bytes: int | str,
    *,
    statvfs: Callable[[Path], Any] = _system_statvfs,
) -> dict[str, Any]:
    """Evaluate validated targets and return a deterministic evidence object."""

    threshold = parse_minimum_free_bytes(minimum_free_bytes)
    ordered = _validated_targets(targets)

    checks: list[dict[str, Any]] = []
    for target in ordered:
        try:
            total_bytes, available_bytes = _filesystem_bytes(statvfs(target.path))
        except (OSError, ValueError):
            checks.append(
                {
                    "available_bytes": None,
                    "label": target.label,
                    "minimum_free_bytes": threshold,
                    "passed": False,
                    "reason": "filesystem statistics unavailable or invalid",
                    "reason_code": "STATVFS_FAILED",
                    "shortfall_bytes": None,
                    "status": FAILED,
                    "total_bytes": None,
                }
            )
            continue

        passed = available_bytes >= threshold
        checks.append(
            {
                "available_bytes": available_bytes,
                "label": target.label,
                "minimum_free_bytes": threshold,
                "passed": passed,
                "reason": (
                    "available bytes meet the required minimum"
                    if passed
                    else "available bytes are below the required minimum"
                ),
                "reason_code": "ENOUGH_FREE_BYTES" if passed else "INSUFFICIENT_FREE_BYTES",
                "shortfall_bytes": max(threshold - available_bytes, 0),
                "status": PASSED if passed else FAILED,
                "total_bytes": total_bytes,
            }
        )

    passed = all(check["status"] == PASSED for check in checks)
    return {
        "checks": checks,
        "error": None,
        "minimum_free_bytes": threshold,
        "passed": passed,
        "schema": SCHEMA,
        "status": PASSED if passed else FAILED,
    }


def failed_input_report(error: CapacityPreflightInputError) -> dict[str, Any]:
    """Return deterministic, sanitized JSON for rejected input."""

    return {
        "checks": [],
        "error": {"code": error.code, "message": str(error)},
        "minimum_free_bytes": None,
        "passed": False,
        "schema": SCHEMA,
        "status": FAILED,
    }


def run_preflight(
    path_specifications: Iterable[str],
    minimum_free_bytes: int | str,
    *,
    statvfs: Callable[[Path], Any] = _system_statvfs,
) -> dict[str, Any]:
    """Parse and evaluate a preflight, preserving input failures as JSON."""

    try:
        threshold = parse_minimum_free_bytes(minimum_free_bytes)
        targets = parse_labeled_paths(path_specifications)
        return evaluate_capacity(targets, threshold, statvfs=statvfs)
    except CapacityPreflightInputError as exc:
        return failed_input_report(exc)


def encode_report(report: dict[str, Any]) -> str:
    """Serialize a report with stable ordering and strict JSON values."""

    return json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _SanitizedArgumentParser(
        allow_abbrev=False,
        description="Fail closed unless every labeled filesystem has enough free bytes.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        dest="paths",
        metavar="LABEL=ABSOLUTE_PATH",
        help="filesystem path to check; repeat for each required filesystem",
    )
    parser.add_argument("--minimum-free-bytes", required=True)
    try:
        args = parser.parse_args(argv)
    except CapacityPreflightInputError as exc:
        print(encode_report(failed_input_report(exc)))
        return 1
    report = run_preflight(
        args.paths,
        args.minimum_free_bytes,
        statvfs=_system_statvfs,
    )
    print(encode_report(report))
    return 0 if report["status"] == PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
