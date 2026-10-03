"""run_pipeline: complete run folder, progress, cancellation, determinism, CLI and worker."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from scout_planner import cli, pipeline, runs
from scout_planner.config import apply_overrides, dump_params, load_default_params
from scout_planner.errors import RunCancelled
from sim_fixtures import tiny_params

STAGES = {"generate", "forecast", "capacity_plan", "simulate"}
CONTRACT_FILES = [
    "backtest.parquet",
    "capacity_plan.parquet",
    "forecast.parquet",
    "hiring_plan.parquet",
    "raw/fixtures.parquet",
    "raw/requests.parquet",
    "raw/scout_unavailability.parquet",
    "raw/scout_weekly_hours.parquet",
    "raw/scouts.parquet",
    "results/requests.parquet",
    "results/seeds.parquet",
    "results/weekly.parquet",
    "summary.json",
]
TIMING = ["sim_seconds", "mean_solve_seconds", "max_solve_seconds"]


def _run(tmp_path: Path, name: str, params=None, workers: int | None = 2):
    calls: list[tuple[float, str, str]] = []
    folder = tmp_path / name
    pipeline.run_pipeline(
        params or tiny_params(),
        folder,
        lambda f, s, m: calls.append((f, s, m)),
        lambda: False,
        max_workers=workers,
    )
    return folder, calls


@pytest.fixture(scope="module")
def pooled(tmp_path_factory):
    """One tiny run through the process pool (2 seeds, 2 workers)."""
    return _run(tmp_path_factory.mktemp("pipeline"), "run")


def test_pipeline_writes_every_file_in_the_contract(pooled) -> None:
    folder, _ = pooled
    found = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    assert found == CONTRACT_FILES


def test_result_tables_match_the_contract(pooled) -> None:
    folder, _ = pooled
    seeds = pd.read_parquet(folder / "results/seeds.parquet")
    assert list(seeds["seed"]) == [0, 1]
    for col in ("on_time_rate", "cost_total", "n_late", "util_full_time", "n_hires_freelance"):
        assert col in seeds.columns
    weekly = pd.read_parquet(folder / "results/weekly.parquet")
    assert list(weekly.columns) == [
        "seed", "week_start", "open_requests", "open_hours", "late_requests",
        "team_full_time", "team_freelance",
    ]  # fmt: skip
    req = pd.read_parquet(folder / "results/requests.parquet")
    assert list(req.columns[:7]) == [
        "seed", "request_id", "completed_date", "turnaround_days", "on_time",
        "at_risk_day_one", "scouts_involved",
    ]  # fmt: skip
    summary = json.loads((folder / "summary.json").read_text())
    assert set(summary["on_time_rate"]) == {"mean", "min", "max"}
    assert isinstance(summary["meets_target"], bool)
    assert summary["n_seeds"] == 2 and "diagnostics" in summary
    # cost_total = salaried + freelance + automation + late penalty (G2)
    parts = seeds[["cost_salaried", "cost_freelance", "cost_automation", "cost_late_penalty"]]
    assert (parts.sum(axis=1) - seeds["cost_total"]).abs().max() < 1e-6


def test_progress_is_monotonic_uses_contract_stages_and_ends_at_one(pooled) -> None:
    _, calls = pooled
    fractions = [f for f, _, _ in calls]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0 and fractions[0] == 0.0
    assert {s for _, s, _ in calls} == STAGES
    assert any("simulating seed 2/2, week" in m for _, _, m in calls)


def test_same_params_give_identical_results(tmp_path, pooled) -> None:
    folder_a, _ = pooled
    folder_b, _ = _run(tmp_path, "again", workers=1)  # in-process path, same answer
    for table in ("requests", "weekly"):
        a = pd.read_parquet(folder_a / f"results/{table}.parquet")
        b = pd.read_parquet(folder_b / f"results/{table}.parquet")
        pdt.assert_frame_equal(a, b)
    a = pd.read_parquet(folder_a / "results/seeds.parquet").drop(columns=TIMING)
    b = pd.read_parquet(folder_b / "results/seeds.parquet").drop(columns=TIMING)
    pdt.assert_frame_equal(a, b)
    assert (folder_a / "raw/requests.parquet").read_bytes() == (
        folder_b / "raw/requests.parquet"
    ).read_bytes()


def _cancel_after_first_week(tmp_path: Path, workers: int) -> list[tuple[float, str, str]]:
    calls: list[tuple[float, str, str]] = []
    params = apply_overrides(
        load_default_params(),
        {"assignment.policy": "edf", "sim.seeds": 2, "team.follow_hiring_plan": False},
    )

    def should_cancel() -> bool:
        return any("week" in m for _, _, m in calls)

    with pytest.raises(RunCancelled):
        pipeline.run_pipeline(
            params,
            tmp_path / "cancelled",
            lambda f, s, m: calls.append((f, s, m)),
            should_cancel,
            max_workers=workers,
        )
    return calls


def test_cancellation_stops_the_pool_and_leaves_no_child_processes(tmp_path) -> None:
    calls = _cancel_after_first_week(tmp_path, workers=2)
    assert any("week" in m for _, _, m in calls)
    assert calls[-1][0] < 0.9
    assert multiprocessing.active_children() == []
    assert not (tmp_path / "cancelled" / "summary.json").exists()


def test_cancellation_in_process(tmp_path) -> None:
    calls = _cancel_after_first_week(tmp_path, workers=1)
    assert calls[-1][1] == "simulate" and calls[-1][0] < 0.9


def test_cancel_before_simulation_raises_at_the_next_stage(tmp_path) -> None:
    calls: list[str] = []
    with pytest.raises(RunCancelled):
        pipeline.run_pipeline(
            tiny_params(), tmp_path / "r", lambda f, s, m: calls.append(s), lambda: bool(calls)
        )
    assert calls == ["generate"]


def test_default_max_workers_leaves_a_core_free(monkeypatch) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    assert pipeline.default_max_workers(3) == 3
    assert pipeline.default_max_workers(12) == 7
    monkeypatch.setattr("os.cpu_count", lambda: 1)
    assert pipeline.default_max_workers(3) == 1


# --- CLI ----------------------------------------------------------------------------


def test_cli_stage_by_stage_then_simulate(tmp_path, capsys) -> None:
    params_file = tmp_path / "tiny.yaml"
    dump_params(tiny_params(sim__seeds=1), params_file)
    run = tmp_path / "dev"
    assert cli.main(["data", "--params", str(params_file), "--out", str(run)]) == 0
    assert cli.main(["simulate", "--run", str(run)]) == 2  # no plan yet: clear error
    assert "run `forecast` first" in capsys.readouterr().err
    assert cli.main(["forecast", "--run", str(run)]) == 0
    assert cli.main(["plan", "--run", str(run)]) == 0
    assert cli.main(["simulate", "--run", str(run)]) == 0
    out = capsys.readouterr().out
    assert "on time" in out and "cost" in out
    assert (run / "results/seeds.parquet").is_file() and (run / "summary.json").is_file()


def test_cli_run_creates_and_executes_a_run(tmp_path, capsys) -> None:
    params_file = tmp_path / "tiny.yaml"
    dump_params(tiny_params(sim__seeds=1), params_file)
    root = tmp_path / "runs"
    code = cli.main(["run", "--params", str(params_file), "--name", "tiny", "--root", str(root)])
    assert code == 0
    (status,) = runs.list_runs(root)
    assert status.state == "done" and status.name == "tiny" and status.progress == 1.0
    assert (runs.run_dir(status.run_id, root) / "summary.json").is_file()
    out = capsys.readouterr().out
    assert "on time" in out and "done" in out
    assert runs.read_heartbeat(root) is None  # the lease cleaned up after itself


def test_cli_run_refuses_while_a_worker_holds_the_folder(tmp_path, capsys) -> None:
    params_file = tmp_path / "tiny.yaml"
    dump_params(tiny_params(sim__seeds=1), params_file)
    root = tmp_path / "runs"
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_lock, args=(str(root), ready, release))
    holder.start()
    try:
        assert ready.wait(30)
        code = cli.main(["run", "--params", str(params_file), "--root", str(root)])
        assert code == 3
        assert "background worker" in capsys.readouterr().err
        assert runs.list_runs(root) == []  # nothing created
    finally:
        release.set()
        holder.join(30)


def _hold_lock(root: str, ready, release) -> None:
    """Stand-in for the app's worker: hold the runs lock until released."""
    from scout_planner.worker import runs_lock

    with runs_lock(root):
        ready.set()
        release.wait(60)


# --- the real worker + the real pipeline ------------------------------------------------


@pytest.mark.slow
def test_worker_executes_a_real_run_end_to_end(tmp_path) -> None:
    from scout_planner.worker import Worker

    root = tmp_path / "runs"
    run_id = runs.create_run(tiny_params(), "integration", root=root)
    executed = Worker(root=root, poll_s=0.1).drain()
    assert executed == [run_id]
    status = runs.read_status(run_id, root)
    assert status.state == "done", runs.read_log_tail(run_id, root=root)
    folder = runs.run_dir(run_id, root)
    for rel in CONTRACT_FILES:
        assert (folder / rel).is_file(), rel
    summary = json.loads((folder / "summary.json").read_text())
    assert 0.0 <= summary["on_time_rate"]["mean"] <= 1.0
