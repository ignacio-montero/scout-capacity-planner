"""M4 acceptance: run pipeline, CLI, determinism, actual vs assumed growth, CRN, sweeps.

These are *integration* and *end-to-end* tests: they treat the run folder as
the product and check PRD section 5 (M4) at that level, without reaching into
any stage's internals (in particular, nothing here depends on how the
optimiser works, only on what a run folder promises).

* Fast (default suite): one-month, tiny-team runs, in-process.
* ``slow``: the CLI as real child processes, the optimiser, the sweep round trip.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

from acceptance_helpers import (
    check_run_folder,
    month_index,
    read_summary_strict,
    without_timing,
)
from scout_planner import cli, forecast, generate, pipeline, plan, runs, sweep
from scout_planner.config import dump_params, load_default_params
from scout_planner.results import load_published, normalise_sweep_frame, params_for_sweep_row
from sim_fixtures import tiny_params

REPO = Path(__file__).resolve().parents[1]
TIMING_COLUMNS = ["sim_seconds", "mean_solve_seconds", "max_solve_seconds"]


def _pipeline(params, folder: Path, workers: int = 1) -> Path:
    pipeline.run_pipeline(params, folder, lambda *_: None, lambda: False, max_workers=workers)
    return folder


def _plan_only(params, folder: Path) -> Path:
    """Stages [1]-[3] only (no simulation), written like the pipeline writes them."""
    world = generate.generate_world(params)
    generate.write_world(world, folder)
    fc, bt = forecast.build_forecast(world, params)
    forecast.write_forecast(fc, bt, folder)
    capacity, hiring = plan.build_capacity_plan(
        world.scouts, world.scout_unavailability, fc, params
    )
    plan.write_plan(capacity, hiring, folder)
    return folder


def _bytes(folder: Path, rel: str) -> bytes:
    return (folder / rel).read_bytes()


# --- determinism: same params.yaml -> identical results ------------------------------


def _assert_same_run(a: Path, b: Path) -> None:
    """Two run folders hold the same results, except wall-clock timings."""
    for rel in sorted(p.relative_to(a).as_posix() for p in a.rglob("*.parquet")):
        if rel == "results/seeds.parquet":
            sa = pd.read_parquet(a / rel).drop(columns=TIMING_COLUMNS)
            sb = pd.read_parquet(b / rel).drop(columns=TIMING_COLUMNS)
            pdt.assert_frame_equal(sa, sb)
        else:
            assert _bytes(a, rel) == _bytes(b, rel), f"{rel} differs between identical runs"
    assert without_timing(read_summary_strict(a)) == without_timing(read_summary_strict(b))


def test_same_params_yaml_gives_an_identical_run_folder_via_cli_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """PRD M4: same ``params.yaml`` -> identical results, through the real ``run`` command.

    One execution in-process (1 worker), one through the process pool (2
    workers): the parallel path must not change the answer.
    """
    params_file = tmp_path / "tiny.yaml"
    dump_params(tiny_params(sim__months=1), params_file)
    folders = []
    for k, workers in enumerate(("1", "2")):
        root = tmp_path / f"root{k}"
        argv = ["run", "--params", str(params_file), "--root", str(root), "--workers", workers]
        assert cli.main(argv) == 0
        (status,) = runs.list_runs(root)
        assert status.state == "done" and status.progress == 1.0
        folder = runs.run_dir(status.run_id, root)
        check_run_folder(folder, tiny_params(sim__months=1))
        folders.append(folder)
    capsys.readouterr()
    a, b = folders
    assert _bytes(a, "params.yaml") == _bytes(b, "params.yaml") == params_file.read_bytes()
    _assert_same_run(a, b)


@pytest.mark.slow
def test_optimiser_full_run_is_deterministic(tmp_path: Path) -> None:
    """The default policy (optimiser) under a time limit must still be reproducible.

    A solver stopped by *wall-clock* time can return different answers on a
    busy machine; this guards the run-level promise, whatever the optimiser does inside.
    """
    params = tiny_params(sim__months=1, assignment__policy="optimiser")
    a = _pipeline(params, tmp_path / "a", workers=2)
    b = _pipeline(params, tmp_path / "b", workers=2)
    check_run_folder(a, params)
    _assert_same_run(a, b)


# --- actual vs assumed growth (D-014), hires join when due ----------------------------


@pytest.fixture(scope="module")
def growth_folders(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("growth")
    variants = {
        "base": tiny_params(),
        "assumed": tiny_params(capacity_plan__assumed_growth=6.0),
        "actual": tiny_params(demand__actual_growth=6.0),
    }
    return {name: _plan_only(p, base / name) for name, p in variants.items()}


RAW = [f"raw/{t}.parquet" for t in generate.TABLE_SCHEMAS]


def test_assumed_growth_drives_the_plan_but_never_the_arrivals(growth_folders) -> None:
    base, assumed = growth_folders["base"], growth_folders["assumed"]
    for rel in RAW:
        assert _bytes(base, rel) == _bytes(assumed, rel), f"assumed growth changed {rel}"
    assert _bytes(base, "backtest.parquet") == _bytes(assumed, "backtest.parquet")
    fa, fb = (pd.read_parquet(f / "forecast.parquet") for f in (base, assumed))
    # the growth ramp is gentle in the first months: higher, month by month, but not by much
    assert (fb["requests_p50"] >= fa["requests_p50"]).all()
    assert fb["requests_p50"].sum() > fa["requests_p50"].sum()
    ha, hb = (pd.read_parquet(f / "capacity_plan.parquet") for f in (base, assumed))
    assert hb["required_hours"].sum() > ha["required_hours"].sum()


def test_actual_growth_drives_the_arrivals_but_never_the_plan(growth_folders) -> None:
    base, actual = growth_folders["base"], growth_folders["actual"]
    for rel in ("forecast.parquet", "backtest.parquet", "capacity_plan.parquet"):
        assert _bytes(base, rel) == _bytes(actual, rel), f"actual growth changed {rel}"
    pdt.assert_frame_equal(
        pd.read_parquet(base / "hiring_plan.parquet"),
        pd.read_parquet(actual / "hiring_plan.parquet"),
    )
    ra, rb = (pd.read_parquet(f / "raw/requests.parquet") for f in (base, actual))
    pdt.assert_frame_equal(
        ra[ra["period"] == "history"].reset_index(drop=True),
        rb[rb["period"] == "history"].reset_index(drop=True),
    )
    assert (rb["period"] == "future").sum() > (ra["period"] == "future").sum()


HIRING = {
    "sim__months": 3,
    "sim__seeds": 1,
    "team__follow_hiring_plan": True,
    "demand__actual_growth": 6.0,
    "capacity_plan__assumed_growth": 6.0,
    "capacity_plan__lead_time_full_time_months": 1,
    "capacity_plan__lead_time_freelance_months": 1,
}


def _expected_team(hiring: pd.DataFrame, kind: str, month: int, months: int, initial: int) -> int:
    rows = hiring[
        (hiring["hire_type"] == kind)
        & (hiring["joins_month"] <= month)
        & (hiring["joins_month"] < months)
    ]
    return initial + int(rows["count"].sum())


def test_hires_join_the_simulated_team_in_their_joins_month(tmp_path: Path) -> None:
    params = tiny_params(**HIRING)
    folder = _pipeline(params, tmp_path / "run")
    summary = check_run_folder(folder, params)
    hiring = pd.read_parquet(folder / "hiring_plan.parquet")
    joining = hiring[hiring["joins_month"].between(1, params.sim.months - 1)]
    assert not joining.empty, "this world was built to need hires; adjust HIRING"
    weekly = pd.read_parquet(folder / "results/weekly.parquet")
    for row in weekly.itertuples(index=False):
        m = month_index(row.week_start)
        assert row.team_full_time == _expected_team(hiring, "full_time", m, 3, 8), row
        assert row.team_freelance == _expected_team(hiring, "freelance", m, 3, 4), row
    by_type = joining.groupby("hire_type")["count"].sum()
    assert summary["n_hires_full_time"]["mean"] == by_type.get("full_time", 0)
    assert summary["n_hires_freelance"]["mean"] == by_type.get("freelance", 0)


def test_hires_stay_out_when_the_plan_is_not_followed(tmp_path: Path) -> None:
    params = tiny_params(**{**HIRING, "team__follow_hiring_plan": False})
    folder = _pipeline(params, tmp_path / "run")
    check_run_folder(folder, params)
    assert not pd.read_parquet(folder / "hiring_plan.parquet").empty  # the plan exists...
    weekly = pd.read_parquet(folder / "results/weekly.parquet")
    assert set(weekly["team_full_time"]) == {8} and set(weekly["team_freelance"]) == {4}


def test_hires_joining_in_the_first_month_are_counted_as_hires(tmp_path: Path) -> None:
    overrides = {
        **HIRING,
        "sim__months": 2,
        "capacity_plan__lead_time_full_time_months": 0,
        "capacity_plan__lead_time_freelance_months": 0,
    }
    params = tiny_params(**overrides)
    folder = _pipeline(params, tmp_path / "run")
    summary = check_run_folder(folder, params)
    hiring = pd.read_parquet(folder / "hiring_plan.parquet")
    assert (hiring["joins_month"] == 0).any(), "precondition: some hires join in month 0"
    joined = hiring[hiring["joins_month"] < params.sim.months].groupby("hire_type")["count"].sum()
    weekly = pd.read_parquet(folder / "results/weekly.parquet")
    last = weekly.iloc[-1]
    # the weekly team size already counts every hire (consistent with the plan)...
    assert last["team_full_time"] == 8 + joined.get("full_time", 0)
    # ...but the summary's hire counts must agree with it
    assert summary["n_hires_full_time"]["mean"] == joined.get("full_time", 0)
    assert summary["n_hires_freelance"]["mean"] == joined.get("freelance", 0)


# --- changing only the policy leaves the request stream identical (D-015) -------------


def test_changing_only_the_policy_keeps_every_replications_requests(tmp_path: Path) -> None:
    """Run-folder level CRN, including replication 1 (regenerated inside a pool process)."""
    edf = _pipeline(tiny_params(sim__months=1), tmp_path / "edf", workers=1)
    fcfs = _pipeline(
        tiny_params(sim__months=1, assignment__policy="fcfs", assignment__unit="bundle"),
        tmp_path / "fcfs",
        workers=2,
    )
    for rel in RAW:
        assert _bytes(edf, rel) == _bytes(fcfs, rel), rel
    cols = ["seed", "request_id", "received_date", "due_date", "skill_type", "needs_live_view"]
    a, b = (pd.read_parquet(f / "results/requests.parquet")[cols] for f in (edf, fcfs))
    assert set(a["seed"]) == {0, 1}
    pdt.assert_frame_equal(a, b)
    outcome = ["completed_date", "on_time"]
    ra, rb = (pd.read_parquet(f / "results/requests.parquet") for f in (edf, fcfs))
    assert not ra[outcome].equals(rb[outcome]), "policies differ, so outcomes should too"


# --- metrics present (PRD M4 list) ----------------------------------------------------


def test_every_prd_metric_is_in_the_summary(tmp_path: Path) -> None:
    params = tiny_params(sim__months=1, sim__seeds=1)
    summary = check_run_folder(_pipeline(params, tmp_path / "run"), params)
    prd_metrics = [
        "on_time_rate",
        "mean_turnaround_days",
        "p90_turnaround_days",
        "util_full_time",
        "util_freelance",
        "cost_salaried",
        "cost_freelance",
        "cost_late_penalty",
    ]
    for name in prd_metrics:
        assert set(summary[name]) == {"mean", "min", "max"}, name
    weekly = pd.read_parquet(tmp_path / "run/results/weekly.parquet")
    assert {"open_requests", "open_hours", "late_requests"} <= set(weekly.columns)  # backlog


# --- the CLI as real processes (what `make data forecast plan simulate` runs) ---------


def _cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scout_planner.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.slow
def test_cli_stage_chain_as_child_processes(tmp_path: Path) -> None:
    """data -> forecast -> plan -> simulate, each a fresh interpreter, like `make` runs them."""
    params_file = tmp_path / "tiny.yaml"
    params = tiny_params(sim__months=1, sim__seeds=2)
    dump_params(params, params_file)
    run = tmp_path / "dev"

    out = _cli("simulate", "--run", str(run), cwd=tmp_path)
    assert out.returncode == 2 and "run `data` first" in out.stderr

    out = _cli("data", "--params", str(params_file), "--out", str(run), cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "start load" in out.stdout

    out = _cli("forecast", "--run", str(run), cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    # PRD M2: backtest summary printed, model vs seasonal naive, per skill and aggregate
    assert "WAPE" in out.stdout and "ALL" in out.stdout
    assert "seasonal naive on aggregate" in out.stdout
    skill_rows = [s for s in generate.SKILL_TYPES if s in out.stdout]
    assert len(skill_rows) == len(generate.SKILL_TYPES)

    out = _cli("plan", "--run", str(run), cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "hiring plan" in out.stdout

    out = _cli("simulate", "--run", str(run), "--workers", "2", cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "on time" in out.stdout and ("MISSES" in out.stdout or "meets" in out.stdout)
    check_run_folder(run, params)

    bad = tmp_path / "bad.yaml"
    bad.write_text("sim:\n  seeds: 9\n", encoding="utf-8")
    out = _cli("data", "--params", str(bad), "--out", str(tmp_path / "x"), cwd=tmp_path)
    assert out.returncode == 2 and "seeds" in out.stderr and "Traceback" not in out.stderr


# --- sweep: run + publish + rebuild each row's parameters ------------------------------


TINY_SWEEP = {
    "name": "tiny-roundtrip",
    "base": "config/default.yaml",
    "vary": {"assignment.policy": ["fcfs", "edf"], "demand.actual_growth": [1.0, 4.0]},
    "fixed": {
        "sim.months": 1,
        "sim.seeds": 1,
        "team.full_time_count": 8,
        "team.freelance_count": 4,
        "team.follow_hiring_plan": False,
        "demand.start_load": 0.3,
        "assignment.weights": {"continuity": 2.0},
    },
}


def _write_sweep(tmp_path: Path, spec: dict, name: str = "sweep.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.slow
def test_sweep_publish_round_trip_rebuilds_every_runs_parameters(tmp_path: Path) -> None:
    """`sweep --publish` -> published parquet -> "Clone settings" rebuilds each params.yaml."""
    root, published = tmp_path / "runs", tmp_path / "published"
    path = _write_sweep(tmp_path, TINY_SWEEP)
    argv = ["sweep", "--sweep", str(path), "--root", str(root), "--workers", "1"]
    assert cli.main([*argv, "--publish", "--published-dir", str(published)]) == 0

    statuses = runs.list_runs(root)
    assert len(statuses) == 4 and {s.state for s in statuses} == {"done"}
    assert len({s.sweep_id for s in statuses}) == 1
    for s in statuses:
        folder = runs.run_dir(s.run_id, root)
        check_run_folder(folder, runs.read_params(s.run_id, root))

    loaded = load_published(published / "tiny-roundtrip_summary.parquet")
    assert loaded.value is not None, loaded.problem
    frame = normalise_sweep_frame(loaded.value, load_default_params())
    assert len(frame) == 4
    for row in frame.to_dict("records"):
        rebuilt = params_for_sweep_row(row, load_default_params(), path)
        assert rebuilt == runs.read_params(row["run_id"], root), row["name"]
        summary = read_summary_strict(runs.run_dir(row["run_id"], root))
        assert row["on_time_rate_mean"] == pytest.approx(summary["on_time_rate"]["mean"])
        assert row["meets_target"] == summary["meets_target"]


def _finish_with_fake_summary(run_id: str, root: Path) -> None:
    runs.update_status(run_id, root=root, state="running")
    runs.update_status(run_id, root=root, state="done")
    summary = {
        "on_time_rate": {"mean": 0.9, "min": 0.9, "max": 0.9},
        "cost_total": {"mean": 1.0, "min": 1.0, "max": 1.0},
        "meets_target": False,
        "n_seeds": 1,
        "diagnostics": {},
    }
    (runs.run_dir(run_id, root) / "summary.json").write_text(json.dumps(summary))


def test_published_rows_rebuild_queued_parameters_including_nested_fixed_values(
    tmp_path: Path,
) -> None:
    """Fast twin of the round trip above: no simulation, fake summaries (a *stub*)."""
    root = tmp_path / "runs"
    path = _write_sweep(tmp_path, TINY_SWEEP)
    spec = sweep.load_sweep(path)
    sweep_id, run_ids = sweep.queue_sweep(spec, root)
    for run_id in run_ids:
        _finish_with_fake_summary(run_id, root)
    out = sweep.publish(spec, sweep_id, root, out_dir=tmp_path / "published")
    frame = normalise_sweep_frame(pd.read_parquet(out), load_default_params())
    for row in frame.to_dict("records"):
        assert params_for_sweep_row(row, load_default_params(), path) == runs.read_params(
            row["run_id"], root
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "GAP sweep.publish / results.params_for_sweep_row: a sweep whose `base` is not "
        "config/default.yaml publishes only varied/fixed/always columns, and the rebuild "
        "starts from the defaults, so 'Clone settings' silently drops the base's changes "
        "(DATA_CONTRACTS section 6 promises an exact rebuild)"
    ),
)
def test_published_rows_rebuild_parameters_of_a_sweep_with_a_custom_base(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    base_file = tmp_path / "small_team.yaml"
    dump_params(tiny_params(), base_file)  # 8 + 4 scouts, two months, EDF...
    spec_dict = {
        "name": "custom-base",
        "base": str(base_file),
        "vary": {"demand.actual_growth": [1.0, 2.0]},
    }
    path = _write_sweep(tmp_path, spec_dict)
    spec = sweep.load_sweep(path)
    sweep_id, run_ids = sweep.queue_sweep(spec, root)
    for run_id in run_ids:
        _finish_with_fake_summary(run_id, root)
    out = sweep.publish(spec, sweep_id, root, out_dir=tmp_path / "published")
    frame = normalise_sweep_frame(pd.read_parquet(out), load_default_params())
    for row in frame.to_dict("records"):
        rebuilt = params_for_sweep_row(row, load_default_params(), path)
        assert rebuilt.team.full_time_count == 8
        assert rebuilt == runs.read_params(row["run_id"], root)


# --- make sweep SWEEP=quick ----------------------------------------------------------


def test_quick_sweep_is_small_enough_for_make_all() -> None:
    """PRD M4: `make sweep SWEEP=quick` takes minutes: few runs, few repeats, nothing exotic."""
    spec = sweep.load_sweep(REPO / "config/sweeps/quick.yaml")
    expanded = sweep.expand(spec)
    assert len(expanded) <= 6
    assert all(r.params.sim.seeds <= 2 for r in expanded)
    assert all(r.params.assignment.time_limit_s <= 1.0 for r in expanded)
    assert all(r.params.sim.months <= 12 for r in expanded)
