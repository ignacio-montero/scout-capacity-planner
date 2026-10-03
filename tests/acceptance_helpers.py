"""Shared checks for the acceptance and edge-case tests (PRD section 5, M0-M5).

``check_run_folder`` is an *invariant oracle*: it does not know what the right
on-time rate is for a given parameter set, but it knows rules that must hold
for **every** run folder (costs add up, nobody is on time without finishing,
the summary agrees with the per-seed table, JSON has no NaN). Edge-case tests
feed it unusual parameters and let it look for broken rules.

``check_trace`` does the same for one replication's full work log.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scout_planner.config import Params
from scout_planner.generate import SIM_START, sim_end, sim_week_starts

CONTRACT_FILES = (  # run_pipeline's outputs; params.yaml is the run store's job
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
)
COST_PARTS = ("cost_salaried", "cost_freelance", "cost_automation", "cost_late_penalty")
EPS = 1e-6


def _no_nan(token: str) -> Any:
    raise AssertionError(f"summary.json contains a non-JSON number: {token}")


def read_summary_strict(folder: Path) -> dict[str, Any]:
    """``summary.json``, failing on NaN / Infinity (which ``json`` would accept silently)."""
    return json.loads((folder / "summary.json").read_text(encoding="utf-8"), parse_constant=_no_nan)


def without_timing(value: Any) -> Any:
    """Drop wall-clock fields (``*_seconds``): the only part of a run allowed to differ."""
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if "seconds" not in k}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def month_index(day: dt.date) -> int:
    """Month index of ``day`` relative to the simulated start (0 = first simulated month)."""
    day = max(day, SIM_START)
    return (day.year - SIM_START.year) * 12 + day.month - SIM_START.month


def _in_unit_interval(x: float) -> bool:
    return bool(np.isnan(x) or -EPS <= x <= 1 + EPS)


def check_run_folder(folder: Path, params: Params) -> dict[str, Any]:
    """Assert every run-level invariant on a finished run folder; return the summary."""
    for rel in CONTRACT_FILES:
        assert (folder / rel).is_file(), f"missing {rel}"
    summary = read_summary_strict(folder)
    seeds = pd.read_parquet(folder / "results/seeds.parquet")
    weekly = pd.read_parquet(folder / "results/weekly.parquet")
    req = pd.read_parquet(folder / "results/requests.parquet")
    raw_scouts = pd.read_parquet(folder / "raw/scouts.parquet")

    # --- seeds.parquet: one row per replication, internally consistent
    n = params.sim.seeds
    assert list(seeds["seed"]) == list(range(n))
    assert summary["n_seeds"] == n
    assert (seeds["n_late"] >= 0).all() and (seeds["n_late"] <= seeds["n_requests"]).all()
    assert (seeds["n_at_risk_day_one"] <= seeds["n_requests"]).all()
    for col in ("on_time_rate", "util_full_time", "util_freelance"):
        assert seeds[col].map(_in_unit_interval).all(), (col, seeds[col].tolist())
    for col in (*COST_PARTS, "cost_total"):
        assert (seeds[col] >= -EPS).all(), col
    parts = seeds[list(COST_PARTS)].sum(axis=1)
    assert np.allclose(parts, seeds["cost_total"]), "cost_total != sum of its parts (G2)"
    assert np.allclose(seeds["cost_late_penalty"], seeds["n_late"] * params.cost.late_penalty)
    auto = params.automation.monthly_cost * params.sim.months if params.automation.enabled else 0
    assert np.allclose(seeds["cost_automation"], auto)
    initial_ft = int((raw_scouts["employment"] == "full_time").sum())
    initial_fl = int((raw_scouts["employment"] == "freelance").sum())
    floor = initial_ft * params.cost.full_time_monthly_salary * params.sim.months
    assert (seeds["cost_salaried"] >= floor - EPS).all(), "the initial team is always paid"
    if not params.team.follow_hiring_plan:
        assert np.allclose(seeds["cost_salaried"], floor)
        assert (seeds[["n_hires_full_time", "n_hires_freelance"]] == 0).all().all()
    if initial_ft == 0 and not params.team.follow_hiring_plan:
        assert seeds["util_full_time"].isna().all()
    if initial_fl == 0 and not params.team.follow_hiring_plan:
        assert seeds["util_freelance"].isna().all()
        assert np.allclose(seeds["cost_freelance"], 0.0)
    rated = seeds["on_time_rate"].notna()
    expected_rate = 1 - seeds.loc[rated, "n_late"] / seeds.loc[rated, "n_requests"]
    assert np.allclose(seeds.loc[rated, "on_time_rate"], expected_rate)

    # --- summary.json agrees with seeds.parquet
    for metric in ("on_time_rate", "cost_total", "n_late", "n_requests"):
        stat = summary[metric]
        col = seeds[metric].astype(float)
        if col.notna().any():
            assert stat["min"] - EPS <= stat["mean"] <= stat["max"] + EPS, metric
            assert math.isclose(stat["mean"], col.mean(), rel_tol=1e-9, abs_tol=1e-9), metric
        else:
            assert stat == {"mean": None, "min": None, "max": None}, metric
    mean_on_time = summary["on_time_rate"]["mean"]
    expected_meets = mean_on_time is not None and mean_on_time >= params.sim.target_on_time
    assert summary["meets_target"] is expected_meets

    # --- weekly.parquet: every week of every seed, backlog never negative
    weeks = sim_week_starts(params)
    assert len(weekly) == n * len(weeks)
    for _, w in weekly.groupby("seed"):
        assert list(w["week_start"]) == weeks
    assert (weekly["open_requests"] >= 0).all() and (weekly["open_hours"] >= -EPS).all()
    assert (weekly["late_requests"] <= weekly["open_requests"]).all()
    assert (weekly["team_full_time"] >= initial_ft).all(), "the team only ever grows"
    assert (weekly["team_freelance"] >= initial_fl).all()

    # --- results/requests.parquet: outcomes are physically possible
    assert not req.duplicated(["seed", "request_id"]).any()
    end = sim_end(params)
    done = req["completed_date"].notna()
    assert (req.loc[done, "completed_date"] >= req.loc[done, "received_date"]).all()
    assert (req.loc[done, "completed_date"] <= end).all()
    turnaround = (
        pd.to_datetime(req.loc[done, "completed_date"])
        - pd.to_datetime(req.loc[done, "received_date"])
    ).dt.days
    assert np.allclose(turnaround, req.loc[done, "turnaround_days"])
    on_time = req["on_time"]
    assert done[on_time].all(), "a request cannot be on time without being finished"
    assert (req.loc[on_time, "completed_date"] <= req.loc[on_time, "due_date"]).all()
    late_done = done & ~on_time
    assert (req.loc[late_done, "completed_date"] > req.loc[late_done, "due_date"]).all()
    assert (req.loc[req["scored"], "due_date"] <= end).all(), "scored = due inside the horizon"
    assert (req.loc[~req["scored"], "due_date"] > end).all()
    assert (req.loc[done, "scouts_involved"] >= 1).all()
    scored = req[req["scored"]]
    per_seed = scored.groupby("seed").agg(n=("request_id", "size"), ok=("on_time", "sum"))
    per_seed = per_seed.reindex(range(n), fill_value=0)
    assert list(per_seed["n"]) == list(seeds["n_requests"])
    assert list(per_seed["n"] - per_seed["ok"]) == list(seeds["n_late"])
    return summary


def check_trace(result: Any, params: Params) -> None:
    """Invariants on one replication's full trace (``trace=True``)."""
    day, log = result.day_log, result.work_log
    assert (day["used"] <= day["offered"] + EPS).all(), "a scout worked more than they offered"
    assert (day["lieu_owed"] >= -EPS).all()
    assert (log["hours"] > 0).all()
    assert (log.groupby("item_id")["scout_id"].nunique() <= 1).all(), "started work moved scout"
    live = log[log["kind"] == "live"]
    assert (live["date"] == live["fixture_date"]).all(), "live view off its fixture date"
    assert not live.duplicated(["scout_id", "date"]).any(), "two live views on one day"
    first = log.groupby(["request_id", "kind"])["date"].min().unstack()
    last = log.groupby(["request_id", "kind"])["date"].max().unstack()
    if "writeup" in first.columns:
        for rid, start in first["writeup"].dropna().items():
            assert pd.notna(last.at[rid, "desk"]) and start >= last.at[rid, "desk"], rid
            if "live" in last.columns and pd.notna(last.at[rid, "live"]):
                assert start > last.at[rid, "live"], rid
