"""Sweeps: expansion (cross-product, fixed overrides), sweep_id tagging, published flattening."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scout_planner import cli, runs, sweep
from scout_planner.config import load_default_params
from scout_planner.sweep import SWEEPS_DIR


def _write(tmp_path: Path, data: dict, name: str = "s.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


SPEC = {
    "name": "mini",
    "base": "config/default.yaml",
    "vary": {"assignment.policy": ["fcfs", "edf", "optimiser"], "demand.actual_growth": [1.0, 4.0]},
    "fixed": {"sim.seeds": 2, "cost.late_penalty": 500.0},
}


def test_expansion_is_the_full_cross_product_in_file_order(tmp_path) -> None:
    runs_ = sweep.expand(sweep.load_sweep(_write(tmp_path, SPEC)))
    assert len(runs_) == 6
    combos = [(r.params.assignment.policy, r.params.demand.actual_growth) for r in runs_]
    assert combos == [
        ("fcfs", 1.0), ("fcfs", 4.0), ("edf", 1.0), ("edf", 4.0),
        ("optimiser", 1.0), ("optimiser", 4.0),
    ]  # fmt: skip
    assert runs_[0].name == "mini: policy=fcfs, actual_growth=1.0"
    assert runs_[0].values == {"assignment.policy": "fcfs", "demand.actual_growth": 1.0}


def test_fixed_overrides_apply_to_every_run_and_nothing_else_changes(tmp_path) -> None:
    default = load_default_params()
    for r in sweep.expand(sweep.load_sweep(_write(tmp_path, SPEC))):
        assert r.params.sim.seeds == 2 and r.params.cost.late_penalty == 500.0
        assert r.params.team == default.team and r.params.automation == default.automation


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"vary": {}}, "vary"),
        ({"vary": {"assignment.policy": []}}, "non-empty list"),
        ({"vary": {"assignment.policy": ["edf", "edf"]}}, "duplicate"),
        ({"fixed": {"assignment.policy": "edf"}}, "both varied and fixed"),
        ({"surprise": 1}, "unknown sweep keys"),
        ({"name": ""}, "name"),
    ],
)
def test_malformed_sweeps_are_rejected(change, message) -> None:
    with pytest.raises(ValueError, match=message):
        sweep.parse_sweep({**SPEC, **change})


def test_unknown_or_invalid_parameters_fail_at_expansion(tmp_path) -> None:
    bad_key = {**SPEC, "vary": {"assignment.polcy": ["edf"]}}
    with pytest.raises(ValueError, match="unknown parameter"):
        sweep.expand(sweep.load_sweep(_write(tmp_path, bad_key)))
    bad_value = {**SPEC, "vary": {"demand.actual_growth": [1.0, 9.0]}}
    with pytest.raises(ValueError):
        sweep.expand(sweep.load_sweep(_write(tmp_path, bad_value)))


@pytest.mark.parametrize("name", ["quick", "cadence", "headline"])
def test_shipped_sweep_files_expand(name) -> None:
    spec = sweep.load_sweep(SWEEPS_DIR / f"{name}.yaml")
    expanded = sweep.expand(spec)
    assert spec.name == name and len(expanded) >= 4
    assert len({r.name for r in expanded}) == len(expanded)


def test_queue_sweep_tags_every_run_with_the_sweep_id(tmp_path) -> None:
    root = tmp_path / "runs"
    spec = sweep.load_sweep(_write(tmp_path, SPEC))
    sweep_id, run_ids = sweep.queue_sweep(spec, root)
    assert sweep_id.startswith("mini-") and len(run_ids) == 6
    statuses = {s.run_id: s for s in runs.list_runs(root)}
    assert set(statuses) == set(run_ids)
    assert {s.sweep_id for s in statuses.values()} == {sweep_id}
    assert {s.state for s in statuses.values()} == {"queued"}
    assert runs.read_params(run_ids[-1], root).assignment.policy == "optimiser"


def _finish(run_id: str, root: Path, on_time: float) -> None:
    """Pretend the worker ran it: mark done and drop a summary.json."""
    runs.update_status(run_id, root=root, state="running")
    runs.update_status(run_id, root=root, state="done")
    summary = {
        "on_time_rate": {"mean": on_time, "min": on_time - 0.01, "max": on_time + 0.01},
        "cost_total": {"mean": 1e6, "min": 9e5, "max": 1.1e6},
        "n_late": {"mean": 10.0, "min": 8.0, "max": 12.0},
        "meets_target": on_time >= 0.95,
        "n_seeds": 2,
        "diagnostics": {"optimiser_solves": 0, "per_seed": []},
    }
    (runs.run_dir(run_id, root) / "summary.json").write_text(json.dumps(summary))


def test_publish_flattens_one_row_per_finished_run(tmp_path) -> None:
    root = tmp_path / "runs"
    spec = sweep.load_sweep(_write(tmp_path, SPEC))
    sweep_id, run_ids = sweep.queue_sweep(spec, root)
    for k, run_id in enumerate(run_ids[:-1]):
        _finish(run_id, root, 0.90 + 0.01 * k)
    # the last one stays queued: not published
    path = sweep.publish(spec, sweep_id, root, out_dir=tmp_path / "published")
    assert path.name == "mini_summary.parquet"
    df = pd.read_parquet(path)
    assert len(df) == 5
    for col in (
        "name", "run_id", "assignment.policy", "demand.actual_growth", "sim.seeds",
        "cost.late_penalty", "automation.enabled", "sim.target_on_time",
        "on_time_rate_mean", "on_time_rate_min", "on_time_rate_max",
        "cost_total_mean", "n_late_mean", "meets_target",
    ):  # fmt: skip
        assert col in df.columns, col
    assert not any(c.startswith("diagnostics") or c.startswith("n_seeds") for c in df.columns)
    assert set(df["sim.seeds"]) == {2} and set(df["cost.late_penalty"]) == {500.0}
    assert set(df["automation.enabled"]) == {False}
    row = df.set_index("run_id").loc[run_ids[0]]
    assert row["assignment.policy"] == "fcfs" and row["on_time_rate_mean"] == pytest.approx(0.90)
    assert not row["meets_target"]
    assert sweep.latest_sweep_id("mini", root) == sweep_id


def test_publish_without_finished_runs_is_an_error(tmp_path) -> None:
    root = tmp_path / "runs"
    spec = sweep.load_sweep(_write(tmp_path, SPEC))
    sweep_id, _ = sweep.queue_sweep(spec, root)
    with pytest.raises(ValueError, match="no finished runs"):
        sweep.publish(spec, sweep_id, root, out_dir=tmp_path / "published")


def test_cli_sweep_queue_only_queues(tmp_path, capsys) -> None:
    root = tmp_path / "runs"
    path = _write(tmp_path, SPEC)
    assert cli.main(["sweep", "--sweep", str(path), "--root", str(root), "--queue"]) == 0
    assert {s.state for s in runs.list_runs(root)} == {"queued"}
    assert "6 runs" in capsys.readouterr().out


def test_cli_sweep_rejects_a_bad_file(tmp_path, capsys) -> None:
    path = _write(tmp_path, {**SPEC, "vary": {"assignment.polcy": ["edf"]}})
    assert cli.main(["sweep", "--sweep", str(path), "--root", str(tmp_path / "r")]) == 2
    assert "invalid sweep" in capsys.readouterr().err


@pytest.mark.slow
def test_cli_sweep_executes_and_publishes(tmp_path) -> None:
    tiny = {
        "name": "tiny",
        "base": "config/default.yaml",
        "vary": {"assignment.policy": ["fcfs", "edf"]},
        "fixed": {
            "sim.months": 1, "sim.seeds": 1, "team.full_time_count": 8,
            "team.freelance_count": 4, "demand.start_load": 0.3,
            "team.follow_hiring_plan": False,
        },
    }  # fmt: skip
    root, published = tmp_path / "runs", tmp_path / "published"
    args = ["sweep", "--sweep", str(_write(tmp_path, tiny)), "--root", str(root)]
    assert cli.main([*args, "--publish", "--published-dir", str(published)]) == 0
    assert {s.state for s in runs.list_runs(root)} == {"done"}
    df = pd.read_parquet(published / "tiny_summary.parquet")
    assert sorted(df["assignment.policy"]) == ["edf", "fcfs"]
    assert set(df["sim.months"]) == {1}
    # --publish-only republishes the latest execution
    assert cli.main([*args, "--publish-only", "--published-dir", str(tmp_path / "p2")]) == 0
    assert (tmp_path / "p2" / "tiny_summary.parquet").is_file()
