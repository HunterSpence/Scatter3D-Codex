from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from validation import remote_capacity_preflight


def _statistics(*, total_blocks: int = 1_000, available_blocks: int = 500, size: int = 4_096):
    return SimpleNamespace(
        f_frsize=size,
        f_blocks=total_blocks,
        f_bavail=available_blocks,
    )


def _spec(label: str, path: Path) -> str:
    return f"{label}={path.resolve()}"


def test_all_paths_pass_at_or_above_the_exact_threshold(tmp_path: Path) -> None:
    docker = tmp_path / "docker"
    evidence = tmp_path / "evidence"
    docker.mkdir()
    evidence.mkdir()
    calls: list[Path] = []

    def statvfs(path: Path):
        calls.append(path)
        return _statistics(available_blocks=10, size=100)

    report = remote_capacity_preflight.run_preflight(
        [_spec("evidence", evidence), _spec("docker", docker)],
        "1000",
        statvfs=statvfs,
    )

    assert report["status"] == "PASSED"
    assert report["passed"] is True
    assert [check["label"] for check in report["checks"]] == ["docker", "evidence"]
    assert all(check["available_bytes"] == 1_000 for check in report["checks"])
    assert all(check["shortfall_bytes"] == 0 for check in report["checks"])
    assert calls == [docker.resolve(), evidence.resolve()]


def test_one_low_space_filesystem_makes_the_whole_report_failed(tmp_path: Path) -> None:
    enough = tmp_path / "enough"
    low = tmp_path / "low"
    enough.mkdir()
    low.mkdir()

    def statvfs(path: Path):
        available = 200 if path.name == "enough" else 99
        return _statistics(available_blocks=available, size=10)

    report = remote_capacity_preflight.run_preflight(
        [_spec("scratch", enough), _spec("artifacts", low)],
        1_000,
        statvfs=statvfs,
    )

    assert report["status"] == "FAILED"
    assert report["passed"] is False
    checks = {check["label"]: check for check in report["checks"]}
    assert checks["scratch"]["status"] == "PASSED"
    assert checks["artifacts"] == {
        "available_bytes": 990,
        "label": "artifacts",
        "minimum_free_bytes": 1_000,
        "passed": False,
        "reason": "available bytes are below the required minimum",
        "reason_code": "INSUFFICIENT_FREE_BYTES",
        "shortfall_bytes": 10,
        "status": "FAILED",
        "total_bytes": 10_000,
    }


@pytest.mark.parametrize("failure", (OSError("denied"), ValueError("bad stat")))
def test_stat_errors_are_sanitized_failed_evidence(tmp_path: Path, failure: Exception) -> None:
    target = tmp_path / "capacity"
    target.mkdir()

    def statvfs(_path: Path):
        raise failure

    report = remote_capacity_preflight.run_preflight(
        [_spec("capacity", target)],
        1,
        statvfs=statvfs,
    )

    assert report["status"] == "FAILED"
    assert report["checks"][0]["reason_code"] == "STATVFS_FAILED"
    assert report["checks"][0]["available_bytes"] is None
    encoded = remote_capacity_preflight.encode_report(report)
    assert "denied" not in encoded
    assert "bad stat" not in encoded
    assert str(target) not in encoded


@pytest.mark.parametrize(
    ("statistics", "reason_code"),
    (
        (_statistics(size=0), "STATVFS_FAILED"),
        (_statistics(total_blocks=10, available_blocks=11), "STATVFS_FAILED"),
        (SimpleNamespace(f_frsize=1, f_blocks=10), "STATVFS_FAILED"),
        (SimpleNamespace(f_frsize=1.5, f_blocks=10, f_bavail=5), "STATVFS_FAILED"),
    ),
)
def test_malformed_statvfs_results_fail_closed(
    tmp_path: Path, statistics, reason_code: str
) -> None:
    target = tmp_path / "capacity"
    target.mkdir()
    report = remote_capacity_preflight.run_preflight(
        [_spec("capacity", target)],
        1,
        statvfs=lambda _path: statistics,
    )
    assert report["status"] == "FAILED"
    assert report["checks"][0]["reason_code"] == reason_code


@pytest.mark.parametrize(
    ("specifications", "error_code"),
    (
        ([], "NO_PATHS"),
        (["missing-separator"], "MALFORMED_PATH_SPECIFICATION"),
        (["../unsafe=/tmp"], "UNSAFE_PATH_LABEL"),
        (["UPPER=/tmp"], "UNSAFE_PATH_LABEL"),
        (["with space=/tmp"], "UNSAFE_PATH_LABEL"),
        (["label=relative/path"], "PATH_NOT_ABSOLUTE"),
    ),
)
def test_malformed_inputs_are_rejected_as_formal_failed_json(
    specifications: list[str], error_code: str
) -> None:
    report = remote_capacity_preflight.run_preflight(specifications, 1)
    assert report["status"] == "FAILED"
    assert report["passed"] is False
    assert report["checks"] == []
    assert report["error"]["code"] == error_code
    json.loads(remote_capacity_preflight.encode_report(report))


def test_duplicate_label_is_rejected_without_exposing_paths(tmp_path: Path) -> None:
    first = tmp_path / "first-secret-name"
    second = tmp_path / "second-secret-name"
    first.mkdir()
    second.mkdir()
    report = remote_capacity_preflight.run_preflight(
        [_spec("disk", first), _spec("disk", second)],
        1,
    )
    encoded = remote_capacity_preflight.encode_report(report)
    assert report["status"] == "FAILED"
    assert report["error"]["code"] == "DUPLICATE_PATH_LABEL"
    assert str(first) not in encoded
    assert str(second) not in encoded


def test_absolute_path_containing_equals_is_supported(tmp_path: Path) -> None:
    target = tmp_path / "docker=data"
    target.mkdir()
    report = remote_capacity_preflight.run_preflight(
        [_spec("disk", target)],
        1,
        statvfs=lambda _path: _statistics(),
    )
    assert report["status"] == "PASSED"


def test_nonexistent_absolute_path_is_rejected(tmp_path: Path) -> None:
    missing = (tmp_path / "credential-like-secret-name").resolve()
    report = remote_capacity_preflight.run_preflight([f"disk={missing}"], 1)
    encoded = remote_capacity_preflight.encode_report(report)
    assert report["status"] == "FAILED"
    assert report["error"]["code"] == "PATH_NOT_ACCESSIBLE"
    assert str(missing) not in encoded


def test_nul_path_is_rejected_as_inaccessible_without_echo(tmp_path: Path) -> None:
    malformed = f"{tmp_path.resolve()}\0credential-like-secret"
    report = remote_capacity_preflight.run_preflight([f"disk={malformed}"], 1)
    encoded = remote_capacity_preflight.encode_report(report)
    assert report["status"] == "FAILED"
    assert report["error"]["code"] == "PATH_NOT_ACCESSIBLE"
    assert "credential-like-secret" not in encoded


@pytest.mark.parametrize(
    ("targets", "error_code"),
    (
        ([object()], "INVALID_CAPACITY_TARGET"),
        (
            [remote_capacity_preflight.LabeledPath(label="../unsafe", path=Path.cwd())],
            "UNSAFE_PATH_LABEL",
        ),
        (
            [remote_capacity_preflight.LabeledPath(label="disk", path="not-a-path")],
            "INVALID_CAPACITY_TARGET",
        ),
        (
            [remote_capacity_preflight.LabeledPath(label="disk", path=Path("relative"))],
            "PATH_NOT_ABSOLUTE",
        ),
    ),
)
def test_evaluator_rejects_untrusted_target_objects(targets, error_code: str) -> None:
    with pytest.raises(remote_capacity_preflight.CapacityPreflightInputError) as captured:
        remote_capacity_preflight.evaluate_capacity(
            targets,
            1,
            statvfs=lambda _path: _statistics(),
        )
    assert captured.value.code == error_code


@pytest.mark.parametrize("threshold", (0, -1, True, "0", "-1", "+1", " 1", "1.0"))
def test_invalid_threshold_is_rejected(threshold) -> None:
    report = remote_capacity_preflight.run_preflight([], threshold)
    assert report["status"] == "FAILED"
    assert report["error"]["code"] == "INVALID_MINIMUM_FREE_BYTES"


@pytest.mark.parametrize(
    "threshold",
    (
        "9" * 5_000,
        10**5_000,
        remote_capacity_preflight.MAXIMUM_FREE_BYTES + 1,
    ),
    ids=("oversized-string", "oversized-int", "above-signed-64-bit"),
)
def test_threshold_has_an_explicit_module_owned_upper_bound(threshold) -> None:
    report = remote_capacity_preflight.run_preflight([], threshold)
    assert report["status"] == "FAILED"
    assert report["error"]["code"] == "INVALID_MINIMUM_FREE_BYTES"
    json.loads(remote_capacity_preflight.encode_report(report))


@pytest.mark.parametrize(
    "threshold",
    (
        remote_capacity_preflight.MAXIMUM_FREE_BYTES,
        str(remote_capacity_preflight.MAXIMUM_FREE_BYTES),
    ),
)
def test_exact_signed_64_bit_maximum_is_accepted(tmp_path: Path, threshold) -> None:
    assert remote_capacity_preflight.MAXIMUM_FREE_BYTES == (1 << 63) - 1
    target = tmp_path / "capacity"
    target.mkdir()
    report = remote_capacity_preflight.run_preflight(
        [_spec("disk", target)],
        threshold,
        statvfs=lambda _path: _statistics(
            total_blocks=remote_capacity_preflight.MAXIMUM_FREE_BYTES,
            available_blocks=remote_capacity_preflight.MAXIMUM_FREE_BYTES,
            size=1,
        ),
    )
    assert report["status"] == "PASSED"
    assert report["minimum_free_bytes"] == remote_capacity_preflight.MAXIMUM_FREE_BYTES


def test_report_is_deterministic_and_contains_no_machine_identity(tmp_path: Path) -> None:
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()

    def statvfs(_path: Path):
        return _statistics(available_blocks=8, size=128)

    first = remote_capacity_preflight.run_preflight(
        [_spec("beta", beta), _spec("alpha", alpha)], 512, statvfs=statvfs
    )
    second = remote_capacity_preflight.run_preflight(
        [_spec("alpha", alpha), _spec("beta", beta)], 512, statvfs=statvfs
    )
    first_json = remote_capacity_preflight.encode_report(first)
    second_json = remote_capacity_preflight.encode_report(second)
    assert first_json == second_json
    assert str(alpha) not in first_json
    assert str(beta) not in first_json
    assert "hostname" not in first_json.lower()


def test_cli_returns_zero_only_for_all_passed(monkeypatch, capsys, tmp_path: Path) -> None:
    target = tmp_path / "capacity"
    target.mkdir()

    monkeypatch.setattr(
        remote_capacity_preflight,
        "_system_statvfs",
        lambda _path: _statistics(available_blocks=2, size=100),
    )
    passed_code = remote_capacity_preflight.main(
        ["--path", _spec("disk", target), "--minimum-free-bytes", "200"]
    )
    passed = json.loads(capsys.readouterr().out)
    assert passed_code == 0
    assert passed["status"] == "PASSED"

    failed_code = remote_capacity_preflight.main(
        ["--path", _spec("disk", target), "--minimum-free-bytes", "201"]
    )
    failed = json.loads(capsys.readouterr().out)
    assert failed_code == 1
    assert failed["status"] == "FAILED"


def test_cli_oversized_threshold_is_sanitized_failed_json(capsys, tmp_path: Path) -> None:
    target = tmp_path / "capacity"
    target.mkdir()
    oversized = "9" * 5_000

    return_code = remote_capacity_preflight.main(
        ["--path", _spec("disk", target), "--minimum-free-bytes", oversized]
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert return_code == 1
    assert captured.err == ""
    assert report["status"] == "FAILED"
    assert report["error"] == {
        "code": "INVALID_MINIMUM_FREE_BYTES",
        "message": "minimum free bytes must be a positive base-10 integer",
    }
    assert oversized not in captured.out


@pytest.mark.parametrize(
    "arguments",
    (
        ["--unknown-option", "credential-like-secret", "--minimum-free-bytes", "1"],
        ["--path"],
        ["credential-like-secret", "--minimum-free-bytes", "1"],
        ["--minimum-free-bytes"],
        ["--minimum-free", "1"],
    ),
)
def test_cli_parse_failures_are_sanitized_failed_json(arguments, capsys) -> None:
    return_code = remote_capacity_preflight.main(arguments)
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 1
    assert captured.err == ""
    assert report["status"] == "FAILED"
    assert report["passed"] is False
    assert report["error"] == {
        "code": "INVALID_COMMAND_LINE",
        "message": "command-line arguments are invalid",
    }
    assert "credential-like-secret" not in captured.out


def test_implementation_has_no_shell_or_df_dependency(tmp_path: Path) -> None:
    source_path = Path(remote_capacity_preflight.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden_os_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr in {"popen", "system"}
    }
    assert "subprocess" not in imported_roots
    assert forbidden_os_calls == set()

    target = tmp_path / "capacity"
    target.mkdir()
    report = remote_capacity_preflight.run_preflight(
        [_spec("disk", target)],
        1,
        statvfs=lambda _path: _statistics(),
    )
    assert report["status"] == "PASSED"
