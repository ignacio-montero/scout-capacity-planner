"""Metrics: the scoring rule, censoring, costs and the summary.json shape (DATA_CONTRACTS §5)."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from scout_planner import metrics
from scout_planner.simulate import OUTCOME_COLUMNS, SCOUT_COLUMNS, WEEKLY_COLUMNS, ReplicationResult
from sim_fixtures import D, hand_params


def _result(
    outcomes: list[dict], scouts: list[dict], replication: int = 0, **diag
) -> ReplicationResult:
    base = {
        "skill_type": "MID-Iberia-ES",
        "needs_live_view": False,
        "at_risk_day_one": False,
        "scouts_involved": 1,
    }
    rows = []
    for o in outcomes:
        row = {**base, **o}
        done = row.get("completed_date")
        row["turnaround_days"] = (
            float((done - row["received_date"]).days) if done is not None else math.nan
        )
        row["on_time"] = done is not None and done <= row["due_date"]
        rows.append(row)
    return ReplicationResult(
        replication,
        pd.DataFrame(rows, columns=OUTCOME_COLUMNS),
        pd.DataFrame([(D(2026, 12, 28), 0, 0.0, 0, 1, 1)], columns=WEEKLY_COLUMNS),
        pd.DataFrame(scouts, columns=SCOUT_COLUMNS),
        {"assignment_runs": 31, "sim_seconds": 0.5, **diag},
    )


FT = {
    "scout_id": "S1", "employment": "full_time", "joined_month": 0, "monthly_salary": 4500.0,
    "hourly_rate": None, "hours_offered": 160.0, "hours_worked": 120.0, "lieu_owed_at_end": 0.0,
}  # fmt: skip
HIRE = {**FT, "scout_id": "H1", "joined_month": 2, "hours_offered": 100.0, "hours_worked": 50.0}
FL = {
    "scout_id": "S2", "employment": "freelance", "joined_month": 0, "monthly_salary": None,
    "hourly_rate": 40.0, "hours_offered": 80.0, "hours_worked": 20.0, "lieu_owed_at_end": 0.0,
}  # fmt: skip


def _req(rid, received, due, done, scored=True, **kw):
    return {"request_id": rid, "received_date": received, "due_date": due,
            "completed_date": done, "scored": scored, **kw}  # fmt: skip


def test_scoring_censoring_and_turnaround() -> None:
    params = hand_params(sim__months=3)
    r = _result(
        [
            _req("R1", D(2027, 1, 4), D(2027, 1, 18), D(2027, 1, 10)),  # on time, 6 d
            _req("R2", D(2027, 1, 4), D(2027, 1, 18), D(2027, 1, 20)),  # late, 16 d
            _req("R3", D(2027, 3, 1), D(2027, 3, 15), None, at_risk_day_one=True),  # open: late
            _req("R4", D(2027, 3, 25), D(2027, 4, 8), D(2027, 3, 27), scored=False),  # censored
            _req("R5", D(2027, 3, 28), D(2027, 4, 11), None, scored=False),  # censored, open
        ],
        [FT, FL],
    )
    row = metrics.seed_row(r, params)
    assert row["n_requests"] == 3 and row["n_censored"] == 2 and row["n_arrived"] == 5
    assert row["n_late"] == 2 and row["n_unfinished_scored"] == 1
    assert row["on_time_rate"] == pytest.approx(1 / 3)
    # Turnaround over scored, completed requests only (R1, R2): censored R4 is excluded.
    assert row["mean_turnaround_days"] == pytest.approx(11.0)
    assert row["p90_turnaround_days"] == pytest.approx(6 + 0.9 * 10)
    assert row["n_at_risk_day_one"] == 1


def test_costs_salaried_pro_rata_from_join_month() -> None:
    params = hand_params(
        sim__months=12, automation__enabled=True, automation__monthly_cost=1000.0,
        cost__late_penalty=700.0,
    )  # fmt: skip
    c = metrics.costs(pd.DataFrame([FT, HIRE, FL]), n_late=3, params=params)
    assert c["cost_salaried"] == 4500 * 12 + 4500 * 10
    assert c["cost_freelance"] == 20 * 40.0
    assert c["cost_automation"] == 12_000.0
    assert c["cost_late_penalty"] == 2100.0
    assert c["cost_total"] == pytest.approx(sum(v for k, v in c.items() if k != "cost_total"))


def test_automation_cost_is_zero_when_off() -> None:
    c = metrics.costs(pd.DataFrame([FT]), 0, hand_params(automation__enabled=False))
    assert c["cost_automation"] == 0.0


def test_utilisation_and_hire_counts() -> None:
    params = hand_params(sim__months=12)
    r = _result([_req("R1", D(2027, 1, 4), D(2027, 1, 18), D(2027, 1, 6))], [FT, HIRE, FL])
    row = metrics.seed_row(r, params)
    assert row["util_full_time"] == pytest.approx(170 / 260)
    assert row["util_freelance"] == pytest.approx(20 / 80)
    assert row["n_hires_full_time"] == 1 and row["n_hires_freelance"] == 0


def test_seeds_frame_has_contract_columns_first_and_stable_dtypes() -> None:
    params = hand_params()
    rs = [
        _result([_req("R1", D(2027, 1, 4), D(2027, 1, 18), D(2027, 1, 6))], [FT], replication=k)
        for k in (1, 0)
    ]
    seeds = metrics.seeds_frame(rs, params)
    contract = ["seed", "n_requests", "n_at_risk_day_one", "on_time_rate", "mean_turnaround_days",
                "p90_turnaround_days", "util_full_time", "util_freelance", "cost_salaried",
                "cost_freelance", "cost_automation", "cost_late_penalty", "cost_total",
                "n_hires_full_time", "n_hires_freelance"]  # fmt: skip
    assert list(seeds.columns[: len(contract)]) == contract
    assert "n_late" in seeds.columns
    assert list(seeds["seed"]) == [0, 1]
    assert seeds["n_late"].dtype == "int64" and seeds["on_time_rate"].dtype == "float64"


def test_summary_json_shape_and_target() -> None:
    params = hand_params(sim__target_on_time=0.9)
    rs = [
        _result(
            [
                _req("R1", D(2027, 1, 4), D(2027, 1, 18), D(2027, 1, 6)),
                _req("R2", D(2027, 1, 4), D(2027, 1, 18), D(2027, 1, 6 + 14 * k)),
            ],
            [FT],
            replication=k,
            optimiser_solves=10,
            solve_seconds_total=2.0 * (k + 1),
            solve_seconds_max=0.5,
            wall_clock_hits=k,
            fallbacks=0,
        )  # fmt: skip
        for k in (0, 1)
    ]
    summary = metrics.summarise(metrics.seeds_frame(rs, params), params)
    json.dumps(summary, allow_nan=False)  # valid JSON, no NaN
    assert summary["on_time_rate"] == {"mean": 0.75, "min": 0.5, "max": 1.0}
    assert summary["n_late"] == {"mean": 0.5, "min": 0.0, "max": 1.0}
    assert summary["meets_target"] is False  # 0.75 < 0.9
    d = summary["diagnostics"]
    assert d["optimiser_solves"] == 20 and d["wall_clock_hits"] == 1
    assert d["mean_solve_seconds"] == pytest.approx(6.0 / 20)
    assert [s["seed"] for s in d["per_seed"]] == [0, 1]
    flat = metrics.flatten_summary(summary)
    assert flat["on_time_rate_mean"] == 0.75 and flat["meets_target"] is False
    assert "diagnostics_mean" not in flat


def test_undefined_metrics_become_null_in_summary() -> None:
    params = hand_params()
    r = _result([_req("R1", D(2027, 1, 25), D(2027, 2, 8), None, scored=False)], [FT])
    summary = metrics.summarise(metrics.seeds_frame([r], params), params)
    assert summary["on_time_rate"]["mean"] is None
    assert summary["meets_target"] is False
    json.dumps(summary, allow_nan=False)
