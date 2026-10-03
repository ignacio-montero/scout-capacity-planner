"""Read-only loaders for run outputs and sweep summaries, including missing/broken files.

Doubles as a contract check on the demo fixture: every table it writes must
load through the same schema checks the app uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import ui_fixtures
from scout_planner import runs
from scout_planner.config import apply_overrides, load_default_params
from scout_planner.results import (
    SUMMARY_METRICS,
    TABLES,
    flatten_params,
    list_published,
    load_published,
    load_summary,
    load_table,
    local_sweep_frame,
    local_sweeps,
    monthly_demand,
    monthly_on_time,
    normalise_sweep_frame,
    params_for_sweep_row,
    parse_summary,
    seeds_late_reports,
    summary_from_seeds,
)

DEFAULTS = load_default_params()


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    base = tmp_path_factory.mktemp("results")
    root = base / "runs"
    ids = ui_fixtures.write_demo_runs(root, include_broken=True)
    published = ui_fixtures.write_demo_published(base / "published")
    return {"root": root, "ids": ids, "published": published, "base": base}


def run_path(demo, name: str) -> Path:
    return runs.run_dir(demo["ids"][name], demo["root"])


@pytest.mark.parametrize("table", sorted(TABLES))
def test_done_run_has_every_table(demo, table: str) -> None:
    loaded = load_table(run_path(demo, "Demo: Defaults"), table)
    assert loaded.ok, loaded.problem
    assert set(TABLES[table].columns) <= set(loaded.value.columns)
    for column in TABLES[table].date_columns:
        assert pd.api.types.is_datetime64_any_dtype(loaded.value[column])


def test_summary_matches_seeds(demo) -> None:
    path = run_path(demo, "Demo: Defaults")
    summary = load_summary(path).value
    seeds = load_table(path, "seeds").value
    assert set(summary.metrics) == set(SUMMARY_METRICS)
    assert summary.mean("on_time_rate") == pytest.approx(seeds["on_time_rate"].mean())
    assert summary.meets_target == (summary.mean("on_time_rate") >= 0.95)
    # cost_total = salaried + freelance + automation + late penalty (contract G2)
    parts = seeds[["cost_salaried", "cost_freelance", "cost_automation", "cost_late_penalty"]].sum(
        axis=1
    )
    assert parts.to_numpy() == pytest.approx(seeds["cost_total"].to_numpy())


def test_demo_outputs_are_consistent(demo) -> None:
    path = run_path(demo, "Demo: Defaults")
    seeds = load_table(path, "seeds").value
    outcomes = load_table(path, "outcomes").value
    per_seed = outcomes.groupby("seed")["on_time"].mean()
    assert per_seed.to_numpy() == pytest.approx(seeds.set_index("seed")["on_time_rate"].to_numpy())
    late = seeds_late_reports(seeds)
    assert late.mean == pytest.approx((~outcomes["on_time"]).sum() / seeds["seed"].nunique())


def test_running_run_has_no_results_yet(demo) -> None:
    path = run_path(demo, "Demo: Very cautious plan")
    assert load_table(path, "forecast").ok
    missing = load_summary(path)
    assert not missing.ok and missing.problem == "summary.json is missing"
    assert load_table(path, "weekly").problem == "results/weekly.parquet is missing"


def test_corrupt_and_partial_files(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text("{nope", encoding="utf-8")
    assert "could not be read" in load_summary(tmp_path).problem
    (tmp_path / "summary.json").write_text(
        json.dumps({"cost_total": {"mean": 1}}), encoding="utf-8"
    )
    assert load_summary(tmp_path).problem == "summary.json has no on_time_rate"
    (tmp_path / "forecast.parquet").write_bytes(b"not parquet")
    assert "could not be read" in load_table(tmp_path, "forecast").problem
    pd.DataFrame({"month": [1]}).to_parquet(tmp_path / "backtest.parquet")
    assert "lacks columns" in load_table(tmp_path, "backtest").problem
    with pytest.raises(KeyError):
        load_table(tmp_path, "nonsense")


def test_parse_summary_tolerates_plain_numbers() -> None:
    loaded = parse_summary(
        {"on_time_rate": 0.9, "cost_total": {"mean": 1, "min": 0, "max": 2}, "meets_target": False}
    )
    assert loaded.value.stat("on_time_rate").has_range is False
    assert loaded.value.mean("cost_total") == 1.0
    assert loaded.value.meets_target is False
    assert not parse_summary([1, 2]).ok


def test_summary_from_seeds_and_flat() -> None:
    seeds = pd.DataFrame({"seed": [1, 2], "on_time_rate": [0.94, 0.96], "cost_total": [10.0, 20.0]})
    summary = summary_from_seeds(seeds, 0.95)
    assert summary.meets_target is True
    flat = summary.to_flat()
    assert flat["on_time_rate_min"] == 0.94 and flat["cost_total_max"] == 20.0


def test_published_listing_and_loading(demo) -> None:
    (demo["base"] / "published" / "other_summary.parquet").write_bytes(b"broken")
    listed = list_published(demo["base"] / "published")
    assert [s.name for s in listed] == ["headline", "other"]  # headline first
    assert not load_published(listed[1].path).ok
    df = normalise_sweep_frame(load_published(listed[0].path).value, DEFAULTS)
    assert {"policy", "actual_growth", "automation_enabled", "row_id", "meets_target"} <= set(
        df.columns
    )
    assert len(df) == 18
    assert list_published(demo["base"] / "missing") == []


def test_normalise_fills_unvaried_parameters_with_defaults() -> None:
    df = pd.DataFrame({"on_time_rate_mean": [0.9, 0.97], "cost_total_mean": [1.0, 2.0]})
    out = normalise_sweep_frame(df, DEFAULTS)
    assert out["policy"].tolist() == ["optimiser", "optimiser"]
    assert out["meets_target"].tolist() == [False, True]
    assert out["on_time_rate_min"].tolist() == [0.9, 0.97]


def test_local_sweep_frame(demo) -> None:
    groups = local_sweeps(demo["root"])
    assert list(groups) == [ui_fixtures.DEMO_SWEEP_ID]
    frame = local_sweep_frame(groups[ui_fixtures.DEMO_SWEEP_ID], demo["root"])
    assert len(frame) == 4
    # only what varies (plus the scatter's key parameters) becomes a column
    assert {"assignment.policy", "automation.enabled", "demand.actual_growth"} <= set(frame.columns)
    assert "team.full_time_count" not in frame.columns
    assert frame["run_id"].notna().all()


def test_params_for_sweep_row_round_trip(demo, tmp_path: Path) -> None:
    df = normalise_sweep_frame(load_published(demo["published"]).value, DEFAULTS)
    row = df[
        (df["policy"] == "edf") & (df["actual_growth"] == 2.0) & df["automation_enabled"]
    ].iloc[0]
    params = params_for_sweep_row(row.to_dict(), DEFAULTS)
    assert params.assignment.policy == "edf"
    assert params.demand.actual_growth == 2.0 and params.capacity_plan.assumed_growth == 2.0
    assert params.automation.enabled is True and params.sim.seeds == 3
    sweep_file = tmp_path / "headline.yaml"
    sweep_file.write_text("fixed:\n  cost.late_penalty: 2000.0\n", encoding="utf-8")
    with_fixed = params_for_sweep_row(row.to_dict(), DEFAULTS, sweep_file)
    assert with_fixed.cost.late_penalty == 2000.0


def test_flatten_params() -> None:
    flat = flatten_params(apply_overrides(DEFAULTS, {"sim.seeds": 5}))
    assert flat["sim.seeds"] == 5
    assert flat["team.freelance_weekly_hours"] == (8.0, 25.0)
    assert "assignment.weights.lateness" in flat


def test_monthly_joins(demo) -> None:
    path = run_path(demo, "Demo: Defaults")
    outcomes, raw = load_table(path, "outcomes").value, load_table(path, "raw_requests").value
    monthly = monthly_on_time(outcomes, raw)
    assert set(monthly.columns) == {"seed", "month", "on_time_rate", "n"}
    assert monthly["n"].sum() == len(outcomes)
    demand = monthly_demand(raw)
    assert (demand["period"] == "history").sum() == 12
    assert (demand["period"] == "future").sum() == 12


def test_unreadable_folder_is_listed_not_fatal(demo) -> None:
    states = {s.run_id: s.state for s in runs.list_runs(demo["root"])}
    assert states[demo["ids"]["(broken)"]] == "unreadable"
