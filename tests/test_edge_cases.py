"""Edge cases through the whole pipeline: odd but legal parameters must not crash or break rules.

Technique: *boundary-value analysis*. For each parameter we pick the extreme
legal values (0 freelancers, a single scout, rework at its 0.6 ceiling, flat
seasonality, every request urgent, a one-month horizon...) and run the real
pipeline. We don't know the "right" on-time rate for these worlds, so the
oracle is a set of *invariants* (``acceptance_helpers.check_run_folder`` and
``check_trace``): rules that hold for every run, whatever the parameters.

The default suite runs the ``QUICK`` subset over one month, one repeat (about
0.6 s per case); ``slow`` runs every case, then every case again over two
months with two repeats through the process pool, plus the optimiser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from acceptance_helpers import check_run_folder, check_trace
from scout_planner import cli, generate, pipeline
from scout_planner.config import apply_overrides, load_default_params
from sim_fixtures import run_world, tiny_params

EDGES: dict[str, dict[str, Any]] = {
    "zero_freelancers": {"team__freelance_count": 0},
    "zero_full_timers": {"team__full_time_count": 0},
    "single_scout": {"team__full_time_count": 1, "team__freelance_count": 0},
    "single_freelancer": {"team__full_time_count": 0, "team__freelance_count": 1},
    "freelancers_offer_no_hours": {
        "team__full_time_count": 2,
        "team__freelance_count": 3,
        "team__freelance_weekly_hours": [0.0, 0.0],
    },
    "no_leave": {"team__leave_days_per_year": 0},
    "automation_worst_case": {
        "automation__enabled": True,
        "automation__rework_rate": 0.6,  # the ceiling: "100% rework" is not a legal input
        "automation__desk_reduction": 0.9,
        "automation__rework_overhead_hours": 4.0,
    },
    "automation_perfect": {"automation__enabled": True, "automation__rework_rate": 0.0},
    "flat_seasonality": {"demand__seasonality_strength": 0.0},
    "max_seasonality": {"demand__seasonality_strength": 2.0},
    "every_request_urgent": {"demand__urgent_share": 1.0},
    "every_request_live": {"demand__live_view_share": 1.0, "demand__urgent_share": 0.0},
    "no_live_views": {"demand__live_view_share": 0.0},
    "shortest_promises": {"demand__turnaround_days": 7, "demand__urgent_turnaround_days": 3},
    "longest_promise": {"demand__turnaround_days": 28, "demand__urgent_turnaround_days": 28},
    "overloaded": {
        "demand__start_load": 1.2,
        "team__full_time_count": 3,
        "team__freelance_count": 0,
    },
    "shrinking_demand": {"demand__actual_growth": 0.5, "capacity_plan__assumed_growth": 0.5},
    "one_day_horizon": {
        "assignment__horizon_days": 1,  # item-size rule: every item must fit one day
        "demand__desk_hours": [1.0, 2.0],
        "demand__writeup_hours": [1.0, 2.0],
        "demand__live_view_hours": 4.0,
    },
    "fcfs_bundle_weekly": {
        "assignment__policy": "fcfs",
        "assignment__unit": "bundle",
        "assignment__cadence": "weekly",
    },
    "hires_join_immediately": {
        "team__follow_hiring_plan": True,
        "demand__actual_growth": 6.0,
        "capacity_plan__assumed_growth": 6.0,
        "capacity_plan__lead_time_full_time_months": 0,
        "capacity_plan__lead_time_freelance_months": 0,
    },
    "free_lateness_perfect_target": {"cost__late_penalty": 0.0, "sim__target_on_time": 1.0},
}


def _run(tmp_path: Path, overrides: dict[str, Any], workers: int = 1):
    params = tiny_params(**overrides)
    folder = tmp_path / "run"
    pipeline.run_pipeline(params, folder, lambda *_: None, lambda: False, max_workers=workers)
    return params, folder


# The cases the default suite runs (about 0.6 s each); ``slow`` runs them all.
QUICK = [
    "zero_freelancers",
    "single_scout",
    "automation_worst_case",
    "flat_seasonality",
    "every_request_urgent",
    "every_request_live",
    "one_day_horizon",
    "hires_join_immediately",
]


@pytest.mark.parametrize(
    "case", [c if c in QUICK else pytest.param(c, marks=pytest.mark.slow) for c in EDGES]
)
def test_edge_case_runs_and_keeps_every_invariant(tmp_path: Path, case: str) -> None:
    params, folder = _run(tmp_path, {"sim__months": 1, "sim__seeds": 1, **EDGES[case]})
    check_run_folder(folder, params)
    check_trace(run_world(params, 0, trace=True), params)


@pytest.mark.slow
@pytest.mark.parametrize("case", list(EDGES))
def test_edge_case_two_months_two_repeats_in_the_pool(tmp_path: Path, case: str) -> None:
    params, folder = _run(tmp_path, EDGES[case], workers=2)
    check_run_folder(folder, params)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["zero_freelancers", "single_scout", "every_request_live"])
def test_edge_case_with_the_optimiser(tmp_path: Path, case: str) -> None:
    overrides = {
        **EDGES[case],
        "sim__months": 1,
        "sim__seeds": 1,
        "assignment__policy": "optimiser",
        "assignment__time_limit_s": 0.2,
    }
    params, folder = _run(tmp_path, overrides)
    check_run_folder(folder, params)
    check_trace(run_world(params, 0, trace=True), params)


# --- edge cases with a specific expected outcome ----------------------------------------


@pytest.mark.parametrize("rate", [0.61, 1.0])
def test_rework_beyond_the_ceiling_is_rejected_loudly(rate: float) -> None:
    """ "100% rework" is outside the documented 0-0.6 range: refused at load time."""
    with pytest.raises(ValidationError, match="rework_rate"):
        apply_overrides(load_default_params(), {"automation.rework_rate": rate})


def test_every_request_urgent_means_no_live_views_and_no_day_one_risk(tmp_path: Path) -> None:
    params, folder = _run(
        tmp_path, {"sim__months": 1, "sim__seeds": 1, **EDGES["every_request_urgent"]}
    )
    raw = pd.read_parquet(folder / "raw/requests.parquet")
    future = raw[raw["period"] == "future"]
    assert future["urgent"].all()
    assert not future["needs_live_view"].any() and not future["at_risk_day_one"].any()
    days = (pd.to_datetime(future["due_date"]) - pd.to_datetime(future["received_date"])).dt.days
    assert (days == params.demand.urgent_turnaround_days).all()
    out = pd.read_parquet(folder / "results/requests.parquet")
    assert not out["needs_live_view"].any() and not out["at_risk_day_one"].any()


def test_single_scout_holds_every_skill_so_no_request_is_unassignable() -> None:
    """With one scout, coverage wins over ``team.skills_per_scout``: they hold all skills."""
    world = generate.generate_world(tiny_params(**EDGES["single_scout"]))
    assert len(world.scouts) == 1
    held = set(world.scouts.iloc[0]["skills"])
    assert held == set(generate.SKILL_TYPES)


def test_a_team_without_freelancers_prints_na_not_a_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    params, folder = _run(
        tmp_path, {"sim__months": 1, "sim__seeds": 1, **EDGES["zero_freelancers"]}
    )
    summary = check_run_folder(folder, params)
    assert summary["util_freelance"] == {"mean": None, "min": None, "max": None}
    cli.print_summary(summary, params)
    assert "freelance n/a" in capsys.readouterr().out


def test_overload_grows_the_backlog_and_misses_the_target(tmp_path: Path) -> None:
    params, folder = _run(tmp_path, {**EDGES["overloaded"], "sim__seeds": 1})
    summary = check_run_folder(folder, params)
    weekly = pd.read_parquet(folder / "results/weekly.parquet")
    first, last = weekly.groupby("seed").first(), weekly.groupby("seed").last()
    assert (last["open_requests"] > first["open_requests"]).all()
    assert summary["meets_target"] is False
    assert summary["util_full_time"]["min"] > 0.9  # everybody is flat out


def test_a_perfect_target_is_met_only_by_a_perfect_run(tmp_path: Path) -> None:
    overrides = {"sim__months": 1, "sim__seeds": 1, **EDGES["free_lateness_perfect_target"]}
    params, folder = _run(tmp_path, overrides)
    summary = check_run_folder(folder, params)
    assert summary["cost_late_penalty"]["max"] == 0.0
    assert summary["meets_target"] is (summary["on_time_rate"]["mean"] == 1.0)
