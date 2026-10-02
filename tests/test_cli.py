"""CLI shell: argument handling, files written, readable errors, exit codes."""

from __future__ import annotations

from pathlib import Path

import pytest

from scout_planner import cli, generate
from scout_planner.config import DEFAULT_CONFIG_PATH, load_params


def test_data_writes_the_raw_tables_and_params(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    assert cli.main(["data", "--params", str(DEFAULT_CONFIG_PATH), "--out", str(out)]) == 0
    for name in generate.TABLE_SCHEMAS:
        assert (out / "raw" / f"{name}.parquet").is_file()
    assert load_params(out / "params.yaml") == load_params(DEFAULT_CONFIG_PATH)
    printed = capsys.readouterr().out
    assert "requests.parquet" in printed and "start load" in printed
    world = generate.read_world(out)
    assert len(world.scouts) == 40


def test_data_rejects_invalid_params_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("team:\n  full_time_count: -3\n", encoding="utf-8")
    assert cli.main(["data", "--params", str(bad), "--out", str(tmp_path / "run")]) == 2
    err = capsys.readouterr().err
    assert "invalid parameters" in err and "full_time_count" in err
    assert not (tmp_path / "run").exists()


def test_data_reports_a_missing_params_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["data", "--params", str(tmp_path / "nope.yaml"), "--out", str(tmp_path)]) == 2
    assert "not found" in capsys.readouterr().err


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
