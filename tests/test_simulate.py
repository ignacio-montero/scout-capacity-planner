"""Stage [5] simulation: invariants on small worlds, hand-built cases, CRN, sanity checks.

The invariant tests read the full trace (``trace=True``): every hour of work
(``work_log``) and every scout-day (``day_log``), and check that the simulated
year never breaks a rule a reviewer could question.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from scout_planner import generate, simulate
from scout_planner.errors import RunCancelled
from scout_planner.generate import sim_end
from sim_fixtures import (
    SKILL,
    D,
    fixture,
    hand_inputs,
    hand_params,
    request,
    run_world,
    scout,
    tiny_params,
)

EPS = 1e-6

VARIANTS = {
    "edf-task-daily": {},
    "fcfs": {"assignment__policy": "fcfs"},
    "bundle": {"assignment__unit": "bundle"},
    "weekly": {"assignment__cadence": "weekly"},
    "automation": {"automation__enabled": True, "automation__rework_rate": 0.3},
}


@pytest.fixture(scope="module", params=list(VARIANTS))
def traced(request):
    params = tiny_params(**VARIANTS[request.param])
    world = simulate.replication_world(params, 0)
    result = run_world(params, 0)
    return params, world, result


# --- invariants on small generated worlds ------------------------------------------------


def test_every_arrived_request_is_completed_or_open(traced) -> None:
    params, world, result = traced
    fut = world.requests[world.requests["period"] == "future"]
    o = result.outcomes
    assert sorted(o["request_id"]) == sorted(fut["request_id"])
    assert o["request_id"].is_unique
    done = o["completed_date"].notna()
    assert done.sum() > 0.5 * len(o)
    assert (o.loc[done, "completed_date"] >= o.loc[done, "received_date"]).all()
    assert (o.loc[done, "completed_date"] <= sim_end(params)).all()
    assert o.loc[~done, "turnaround_days"].isna().all()
    assert not o.loc[~done, "on_time"].any()


def test_no_scout_works_more_than_their_hours_on_any_day(traced) -> None:
    _, _, result = traced
    day = result.day_log
    assert (day["used"] <= day["offered"] + EPS).all()
    assert (day["lieu_owed"] >= -EPS).all()
    # Every hour of work is paid for by a day's hours, except time in lieu still owed.
    worked = result.scouts.set_index("scout_id")
    used = day.groupby("scout_id")["used"].sum()
    gap = worked["hours_worked"] - used.reindex(worked.index, fill_value=0.0)
    assert np.allclose(gap, worked["lieu_owed_at_end"], atol=1e-6)


def test_precedence_holds_in_execution(traced) -> None:
    params, _, result = traced
    log = result.work_log
    first = log.groupby(["request_id", "kind"])["date"].min().unstack()
    last = log.groupby(["request_id", "kind"])["date"].max().unstack()
    wu = first["writeup"].dropna()
    for rid, start in wu.items():
        desk_end = last.at[rid, "desk"]
        assert pd.notna(desk_end)
        if params.assignment.unit == "task":
            assert start > desk_end  # the write-up is assigned at a later run
        else:
            assert start >= desk_end  # bundle: same scout, back to back
        if "live" in last.columns and pd.notna(last.at[rid, "live"]):
            assert start > last.at[rid, "live"]


def test_live_views_happen_on_fixture_dates_by_same_region_scouts(traced) -> None:
    params, world, result = traced
    live = result.work_log[result.work_log["kind"] == "live"]
    assert len(live) > 5
    fixtures = {f.fixture_id: f for f in generate.fixtures_from_frame(world.fixtures)}
    scouts = {s.scout_id: s for s in generate.scouts_from_frame(world.scouts)}
    requests = world.requests.set_index("request_id")
    off = world.scout_unavailability.groupby("scout_id")["date"].apply(set).to_dict()
    for row in live.itertuples(index=False):
        f = fixtures[row.fixture_id]
        assert row.date == row.fixture_date == f.date
        assert scouts[row.scout_id].home_region == f.region
        assert f.involves(requests.at[row.request_id, "player_club"])
        assert row.date not in off.get(row.scout_id, set())
        assert row.hours == pytest.approx(params.demand.live_view_hours)
    assert not live.duplicated(["scout_id", "date"]).any()  # one live view per scout per day


def test_work_respects_skill_and_conflict_of_interest(traced) -> None:
    _, world, result = traced
    scouts = {s.scout_id: s for s in generate.scouts_from_frame(world.scouts)}
    req = world.requests.set_index("request_id")
    for row in result.work_log.drop_duplicates(["scout_id", "request_id"]).itertuples():
        s = scouts[row.scout_id]
        r = req.loc[row.request_id]
        assert s.has_skill(r["skill_type"])
        assert not s.has_conflict({r["client_club"], r["player_club"]})


def test_started_work_never_changes_scout(traced) -> None:
    _, _, result = traced
    per_item = result.work_log.groupby("item_id")["scout_id"].nunique()
    assert (per_item == 1).all()


def test_task_hours_are_conserved(traced) -> None:
    params, world, result = traced
    req = world.requests.set_index("request_id")
    log = result.work_log
    done = log.groupby(["request_id", "kind"])["hours"].sum()
    auto = params.automation
    factor = 1 - auto.desk_reduction if auto.enabled else 1.0
    finished = set(result.outcomes.loc[result.outcomes["completed_date"].notna(), "request_id"])
    for rid in sorted(finished)[:200]:
        r = req.loc[rid]
        assert done[rid, "writeup"] == pytest.approx(r["writeup_hours"])
        desk = r["desk_hours"] * factor
        if auto.enabled and r["rework_draw"] < auto.rework_rate:
            desk = r["desk_hours"] + auto.rework_overhead_hours  # full desk + overhead in total
        assert done[rid, "desk"] == pytest.approx(desk)


def test_weekly_snapshots_cover_every_week(traced) -> None:
    params, _, result = traced
    w = result.weekly
    assert list(w["week_start"]) == generate.sim_week_starts(params)
    assert w.iloc[0]["open_requests"] == 0  # cold start
    assert (w["late_requests"] <= w["open_requests"]).all()
    assert (w["team_full_time"] == params.team.full_time_count).all()


# --- determinism and common random numbers ------------------------------------------------


def test_same_inputs_give_identical_results() -> None:
    params = tiny_params()
    a, b = run_world(params, 1), run_world(params, 1)
    pdt.assert_frame_equal(a.outcomes, b.outcomes)
    pdt.assert_frame_equal(a.weekly, b.weekly)
    pdt.assert_frame_equal(a.scouts, b.scouts)
    pdt.assert_frame_equal(a.work_log, b.work_log)


def test_optimiser_runs_are_deterministic_and_valid() -> None:
    params = tiny_params(
        sim__months=1, assignment__policy="optimiser", assignment__time_limit_s=0.1
    )
    a, b = run_world(params, 0), run_world(params, 0)
    pdt.assert_frame_equal(a.outcomes, b.outcomes)
    pdt.assert_frame_equal(a.work_log, b.work_log)
    assert a.diagnostics["optimiser_solves"] > 0
    assert a.diagnostics["wall_clock_hits"] == 0
    assert (a.day_log["used"] <= a.day_log["offered"] + EPS).all()


def test_changing_only_the_policy_leaves_arrivals_identical() -> None:
    edf = run_world(tiny_params(), 0, trace=False).outcomes
    fcfs = run_world(tiny_params(assignment__policy="fcfs", assignment__unit="bundle"), 0).outcomes
    cols = ["request_id", "received_date", "due_date", "skill_type", "needs_live_view"]
    pdt.assert_frame_equal(edf[cols], fcfs[cols])


def test_replications_differ_in_luck_but_share_the_team() -> None:
    params = tiny_params()
    w0, w1 = simulate.replication_world(params, 0), simulate.replication_world(params, 1)
    pdt.assert_frame_equal(w0.scouts, w1.scouts)
    pdt.assert_frame_equal(w0.fixtures, w1.fixtures)
    assert not w0.requests.equals(w1.requests)
    assert not w0.scout_weekly_hours.equals(w1.scout_weekly_hours)
    # replication 0 is exactly the world written to raw/
    pdt.assert_frame_equal(w0.requests, generate.generate_world(params).requests)


# --- hand-built cases ------------------------------------------------------------------


def test_cost_split_and_censoring_by_hand() -> None:
    """One full-timer (conflicted out), one freelancer, three requests.

    * R1: received Mon 4 Jan, desk 4 h (2 h with automation at 50%), write-up
      2 h. The full-timer used to work for the client, so the freelancer
      (4 h/day) does the desk on Jan 4 and the write-up on Jan 5.
    * R2: a skill nobody has: never done, due Jan 18 -> scored and late.
    * R3: received Jan 25, due Feb 8, after the horizon -> censored, but the
      freelancer still does its 4 h (and is paid for them).
    """
    from scout_planner import metrics

    params = hand_params(
        automation__enabled=True,
        automation__desk_reduction=0.5,
        automation__rework_rate=0.0,
        automation__monthly_cost=1500.0,
        cost__late_penalty=1000.0,
    )
    team = [scout("S1", former=["Club Client"]), scout("S2", "freelance", rate=40.0)]
    reqs = [
        request("R1", D(2027, 1, 4)),
        request("R2", D(2027, 1, 4), skill="GK-Iberia-ES"),
        request("R3", D(2027, 1, 25)),
    ]
    result = simulate.simulate_replication(hand_inputs(params, team, reqs), trace=True)
    o = result.outcomes.set_index("request_id")
    assert o.at["R1", "completed_date"] == D(2027, 1, 5)
    assert o.at["R1", "turnaround_days"] == 1.0
    assert bool(o.at["R1", "on_time"]) and o.at["R1", "scouts_involved"] == 1
    assert pd.isna(o.at["R2", "completed_date"])
    assert not o.at["R3", "scored"] and o.at["R3", "completed_date"] == D(2027, 1, 26)

    row = metrics.seed_row(result, params)
    assert row["n_requests"] == 2 and row["n_censored"] == 1 and row["n_late"] == 1
    assert row["on_time_rate"] == 0.5
    assert row["cost_salaried"] == 4500.0  # one full-timer x one month, busy or not
    assert row["cost_freelance"] == pytest.approx(8 * 40.0)  # (2 h desk + 2 h write-up) x 2
    assert row["cost_automation"] == 1500.0
    assert row["cost_late_penalty"] == 1000.0
    assert row["cost_total"] == pytest.approx(4500 + 320 + 1500 + 1000)
    assert row["util_full_time"] == 0.0
    # Freelancer: 20 h/week over 5 weekdays; Jan 1 (Fri) + 4 full weeks in January.
    assert row["util_freelance"] == pytest.approx(8 / (4 + 4 * 20))


def test_weekend_live_view_is_repaid_as_time_in_lieu() -> None:
    """A Saturday fixture: 8 h owed, repaid Monday (7.5 h) and Tuesday (0.5 h)."""
    params = hand_params()
    team = [scout("S1"), scout("S2", region="NorthernEurope")]
    reqs = [request("R1", D(2027, 1, 4), live=True)]
    fixtures = [fixture("F1", D(2027, 1, 9))]  # Saturday, in [received+2, due-1]
    result = simulate.simulate_replication(hand_inputs(params, team, reqs, fixtures), trace=True)
    log = result.work_log
    live = log[log["kind"] == "live"].iloc[0]
    assert live["scout_id"] == "S1" and live["date"] == D(2027, 1, 9)
    day = result.day_log.set_index(["scout_id", "date"])
    assert day.at[("S1", D(2027, 1, 9)), "lieu_owed"] == pytest.approx(8.0)
    assert day.at[("S1", D(2027, 1, 11)), "used"] == pytest.approx(7.5)
    assert day.at[("S1", D(2027, 1, 11)), "lieu_owed"] == pytest.approx(0.5)
    assert day.at[("S1", D(2027, 1, 12)), "lieu_owed"] == pytest.approx(0.0)
    wu_start = log.loc[log["kind"] == "writeup", "date"].min()
    assert wu_start > D(2027, 1, 9)
    assert result.outcomes.iloc[0]["on_time"]


def test_missed_fixture_is_retargeted_to_the_next_one() -> None:
    """The only Iberia scout is on leave on the first fixture: the live view moves."""
    params = hand_params()
    team = [scout("S1")]
    reqs = [request("R1", D(2027, 1, 4), live=True)]
    fixtures = [fixture("F1", D(2027, 1, 9)), fixture("F2", D(2027, 1, 16))]
    inputs = hand_inputs(params, team, reqs, fixtures, unavailability={"S1": [D(2027, 1, 9)]})
    result = simulate.simulate_replication(inputs, trace=True)
    live = result.work_log[result.work_log["kind"] == "live"].iloc[0]
    assert live["fixture_id"] == "F2" and live["date"] == D(2027, 1, 16)
    assert result.diagnostics["live_view_retargets"] == 1


def test_at_risk_request_targets_the_first_fixture_after_receipt() -> None:
    params = hand_params()
    reqs = [request("R1", D(2027, 1, 4), live=True)]
    fixtures = [fixture("F1", D(2027, 1, 23))]  # after due - 1 = Jan 17: at risk
    inputs = hand_inputs(params, [scout("S1")], reqs, fixtures, at_risk={"R1": True})
    result = simulate.simulate_replication(inputs, trace=True)
    o = result.outcomes.iloc[0]
    assert o["at_risk_day_one"] and not o["on_time"]
    assert o["completed_date"] > D(2027, 1, 23)


def test_rework_continues_with_the_same_scout() -> None:
    params = hand_params(
        automation__enabled=True,
        automation__desk_reduction=0.5,
        automation__rework_rate=0.5,
        automation__rework_overhead_hours=1.0,
    )
    reqs = [request("R1", D(2027, 1, 4), desk=6.0, rework_draw=0.1)]
    result = simulate.simulate_replication(hand_inputs(params, [scout("S1")], reqs), trace=True)
    desk = result.work_log[result.work_log["kind"] == "desk"]
    # 3 h with the tool, then topped up to the full 6 h + 1 h overhead in total.
    assert desk["hours"].sum() == pytest.approx(6.0 + 1.0)
    assert desk["scout_id"].nunique() == 1
    assert result.diagnostics["reworks"] == 1
    wu_start = result.work_log.loc[result.work_log["kind"] == "writeup", "date"].min()
    assert wu_start > desk["date"].max()


def test_hires_work_only_from_their_join_month() -> None:
    params = hand_params(sim__months=2)
    team = [scout("S1"), scout("H1", joined_month=1)]
    reqs = [request(f"R{i}", D(2027, 1, 4) + dt.timedelta(days=i)) for i in range(40)]
    result = simulate.simulate_replication(hand_inputs(params, team, reqs), trace=True)
    hire_work = result.work_log[result.work_log["scout_id"] == "H1"]
    assert len(hire_work) > 0
    assert hire_work["date"].min() >= D(2027, 2, 1)
    w = result.weekly.set_index("week_start")["team_full_time"]
    assert w[D(2027, 1, 25)] == 1 and w[D(2027, 2, 1)] == 2


def test_hires_follow_the_plan_switch_and_have_their_own_streams() -> None:
    params = tiny_params(team__follow_hiring_plan=True, sim__months=3)
    world = generate.generate_world(params)
    hiring = pd.DataFrame(
        {
            "month_to_act": [0, 0, 0],
            "joins_month": [1, 2, 5],  # the last one joins after the horizon
            "skill_type": [SKILL, SKILL, SKILL],
            "hire_type": ["full_time", "freelance", "full_time"],
            "count": [1, 2, 1],
            "reason": ["", "", ""],
        }
    )
    inputs = simulate.prepare_replication(params, world, hiring, 1)
    hires = [s for s in inputs.scouts if s.joined_month > 0]
    assert [h.joined_month for h in hires] == [1, 2, 2]
    for h in hires:
        weekly = [v for (sid, _), v in inputs.weekly_hours.items() if sid == h.scout_id]
        assert len(weekly) == len(generate.sim_week_starts(params))
    off = params.model_copy(
        update={"team": params.team.model_copy(update={"follow_hiring_plan": False})}
    )
    assert simulate.materialise_hires(off, world, hiring) == []
    # One hire's luck does not depend on how many others are hired.
    fl = hires[1]
    alone = simulate.hire_availability(params, fl, 1)
    assert alone == simulate.hire_availability(params, fl, 1)
    assert alone != simulate.hire_availability(params, fl, 2)


def test_cancellation_is_checked_weekly() -> None:
    params = tiny_params()
    world = simulate.replication_world(params, 0)
    inputs = simulate.prepare_replication(params, world, pd.DataFrame(), 0)
    calls = []

    def cancel() -> bool:
        calls.append(1)
        return len(calls) >= 3

    with pytest.raises(RunCancelled):
        simulate.simulate_replication(inputs, should_cancel=cancel)
    assert len(calls) == 3


def test_progress_is_reported_every_week_and_ends_complete() -> None:
    params = tiny_params()
    world = simulate.replication_world(params, 0)
    inputs = simulate.prepare_replication(params, world, pd.DataFrame(), 0)
    seen: list[tuple[int, int]] = []
    simulate.simulate_replication(inputs, on_week=lambda d, t: seen.append((d, t)))
    total = len(generate.sim_week_starts(params))
    assert [d for d, _ in seen] == list(range(1, total + 1))
    assert all(t == total for _, t in seen)


# --- sanity checks from the PRD (metamorphic tests) ----------------------------------------


def _mean_on_time(params, seeds=(0, 1)) -> float:
    rates = []
    for k in seeds:
        o = run_world(params, k, trace=False).outcomes
        rates.append(o.loc[o["scored"], "on_time"].mean())
    return float(np.mean(rates))


def test_more_capacity_never_lowers_mean_on_time_rate() -> None:
    """Bigger teams, same demand (D-018): the on-time rate must not go down."""
    teams = [(6, 4), (9, 6), (12, 8), (18, 12)]
    rates = [
        _mean_on_time(tiny_params(team__full_time_count=ft, team__freelance_count=fl))
        for ft, fl in teams
    ]
    assert all(b >= a - 1e-9 for a, b in zip(rates, rates[1:], strict=False)), rates
    assert rates[-1] > rates[0]


def test_automation_without_rework_beats_automation_off() -> None:
    """Same overloaded world; the pre-screen tool with 0% rework only removes work."""
    base = {"team__full_time_count": 8, "team__freelance_count": 4}
    off = _mean_on_time(tiny_params(**base))
    on = _mean_on_time(tiny_params(**base, automation__enabled=True, automation__rework_rate=0.0))
    assert on > off


def test_simulated_desk_effort_matches_the_plans_automation_model() -> None:
    """Cross-module consistency (B1): the simulation and the capacity plan must agree
    on what the pre-screen tool costs, or the plan hires for a different world.

    Plan (``forecast.automation_factor``): expected desk hours = E[D] x factor, where a
    failed pre-screen costs the full desk review + overhead in total. Checked per
    request (exact) and on average over a simulated default year (noise tolerance).
    """
    from scout_planner import forecast
    from scout_planner.config import apply_overrides, load_default_params

    params = apply_overrides(
        load_default_params(),
        {"assignment.policy": "edf", "sim.seeds": 1, "automation.enabled": True,
         "automation.rework_rate": 0.3, "automation.rework_overhead_hours": 2.0,
         "team.follow_hiring_plan": False, "demand.actual_growth": 1.0},
    )  # fmt: skip
    auto = params.automation
    result = run_world(params, 0)
    world = simulate.replication_world(params, 0)
    req = world.requests.set_index("request_id")
    log = result.work_log
    desk_done = set(log.loc[(log["kind"] == "writeup"), "request_id"])  # desk finished
    spent = log[log["kind"] == "desk"].groupby("request_id")["hours"].sum().loc[sorted(desk_done)]
    assert len(spent) > 2000

    d = req.loc[spent.index, "desk_hours"]
    failed = req.loc[spent.index, "rework_draw"] < auto.rework_rate
    exact = np.where(failed, d + auto.rework_overhead_hours, d * (1 - auto.desk_reduction))
    np.testing.assert_allclose(spent.to_numpy(), exact, atol=1e-6)

    planned = float(np.mean(params.demand.desk_hours)) * forecast.automation_factor(params)
    assert spent.mean() == pytest.approx(planned, rel=0.03)


def test_salaried_cost_is_pro_rata_for_a_mid_year_hire_in_a_simulated_run() -> None:
    from scout_planner import metrics

    params = hand_params(sim__months=3)
    team = [scout("S1"), scout("H1", joined_month=2), scout("F1", "freelance")]
    reqs = [request("R1", D(2027, 1, 4))]
    result = simulate.simulate_replication(hand_inputs(params, team, reqs))
    row = metrics.seed_row(result, params)
    assert row["cost_salaried"] == 4500.0 * 3 + 4500.0 * 1  # hire paid from month 2 only
    assert row["n_hires_full_time"] == 1
    offered = result.scouts.set_index("scout_id")["hours_offered"]
    assert offered["H1"] == pytest.approx(7.5 * 23)  # March 2027 has 23 weekdays


def test_large_pool_fallbacks_are_counted_and_reported(monkeypatch) -> None:
    """Optimiser falling back to EDF on big pools must be visible per run, not hidden."""
    from scout_planner import metrics
    from scout_planner.assign import AssignOutcome, SolveReport

    real = simulate.assign_with_report
    big = 40

    def fake(pool, scouts, window, cfg, rng, *, cost):
        outcome = real(pool, scouts, window, cfg, rng, cost=cost)
        fell_back = len(pool) >= big
        report = SolveReport(
            outcome.assignments, "UNKNOWN" if fell_back else "OPTIMAL", None, None, None,
            0.01, 0.01, fell_back_to_edf=fell_back,
        )  # fmt: skip
        return AssignOutcome(outcome.assignments, report)

    pool_sizes = []
    monkeypatch.setattr(
        simulate, "assign_with_report",
        lambda pool, *a, **k: (pool_sizes.append(len(pool)), fake(pool, *a, **k))[1],
    )  # fmt: skip
    params = tiny_params(sim__months=1)
    result = run_world(params, 0, trace=False)
    n_big = sum(n >= big for n in pool_sizes)
    assert 0 < n_big < len(pool_sizes)
    d = result.diagnostics
    assert d["fallbacks"] == n_big and d["status_UNKNOWN"] == n_big
    seeds = metrics.seeds_frame([result], params)
    assert seeds.loc[0, "fallback_share"] == pytest.approx(n_big / len(pool_sizes))
    summary = metrics.summarise(seeds, params)
    assert summary["diagnostics"]["status_UNKNOWN"] == n_big
    flat = metrics.flatten_summary(summary)
    assert flat["diag_fallback_share"] == pytest.approx(n_big / len(pool_sizes))
    assert flat["diag_status_OPTIMAL"] == len(pool_sizes) - n_big
