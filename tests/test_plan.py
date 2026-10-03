"""Stage [3] capacity plan: planned hours, the coverage allocation, every hiring rule."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scout_planner import cli
from scout_planner import forecast as f
from scout_planner import generate as g
from scout_planner import plan as p
from scout_planner.config import DEFAULT_CONFIG_PATH, Params, apply_overrides
from scout_planner.domain import Scout
from scout_planner.rng import make_stream

P = Params()
FT_CAP = p.hire_monthly_hours("full_time", P)  # one full-time hire, per plan month
FL_CAP = p.hire_monthly_hours("freelance", P)


@pytest.fixture(scope="module")
def world() -> g.World:
    return g.generate_world(P)


@pytest.fixture(scope="module")
def built(world: g.World) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fc, _ = f.build_forecast(world, P)
    capacity, hiring = p.build_capacity_plan(world.scouts, world.scout_unavailability, fc, P)
    return fc, capacity, hiring


def scout(scout_id: str, skills: set[str], employment: str = "full_time", **kw) -> Scout:
    hours = (37.5, 37.5) if employment == "full_time" else (8.0, 24.0)
    return Scout(
        scout_id, f"Name {scout_id}", employment, frozenset(skills), "Iberia", *hours, **kw
    )


def no_leave() -> pd.DataFrame:
    return pd.DataFrame({"scout_id": [], "date": [], "reason": []})


def hires_for(gap_in_caps: list[float], cap: np.ndarray = FT_CAP, params: Params = P):
    """Run the hiring rule for one skill whose gap is ``gap_in_caps x cap`` per month."""
    gap = np.array(gap_in_caps) * cap[: len(gap_in_caps)]
    available = np.full(len(gap), 1000.0)
    return p.plan_skill_hires("MID-Iberia-ES", available + gap, available, params)


# --- available hours (by hand) -------------------------------------------------------------


def test_weeks_in_month() -> None:
    assert p.weeks_in_month(dt.date(2027, 2, 1)) == 4.0
    assert p.weeks_in_month(dt.date(2027, 1, 1)) == pytest.approx(31 / 7)


def test_scout_monthly_hours_by_hand() -> None:
    ft = scout("S1", {"MID-Iberia-ES"})
    fl = scout("S2", {"MID-Iberia-ES"}, employment="freelance")
    late = scout("S3", {"MID-Iberia-ES"}, joined_month=2)
    # S1: leave Fri 2027-01-08 .. Tue 01-12 = 3 weekdays (weekend inside the block).
    # S2: a freelancer's days off are not deducted from their weekly hours.
    leave_days = [dt.date(2027, 1, d) for d in range(8, 13)]
    unavailability = pd.DataFrame(
        {
            "scout_id": ["S1"] * 5 + ["S2"],
            "date": [*leave_days, dt.date(2027, 1, 20)],
            "reason": ["leave"] * 5 + ["other"],
        }
    )
    hours = p.scout_monthly_hours([ft, fl, late], unavailability, P)
    weeks_jan = 31 / 7
    assert hours[0, 0] == pytest.approx((37.5 * weeks_jan - 3 * 7.5) * 0.8)
    assert hours[0, 1] == pytest.approx(37.5 * 4.0 * 0.8)
    assert hours[1, 0] == pytest.approx(16.0 * weeks_jan * 0.8)
    assert (hours[2, :2] == 0).all() and hours[2, 2] == pytest.approx(37.5 * 31 / 7 * 0.8)


def test_leave_never_makes_hours_negative() -> None:
    days = [dt.date(2027, 2, 1) + dt.timedelta(days=k) for k in range(28)]
    leave = pd.DataFrame({"scout_id": ["S1"] * 28, "date": days, "reason": ["leave"] * 28})
    hours = p.scout_monthly_hours([scout("S1", {"MID-Iberia-ES"})], leave, P)
    assert hours[0, 1] == 0.0


# --- coverage: the min-cost max-flow allocation ---------------------------------------------


def test_allocate_moves_spare_hours_to_where_they_are_needed() -> None:
    # S1 holds A and B (100 h), S2 only B (50 h). A needs 80, B 60: all coverable
    # (S1 -> A 80, S1 -> B 10, S2 -> B 50). A fixed 50/50 split of S1 would leave
    # A 30 h "short" while S1 idles on B: a phantom gap.
    alloc = p.allocate(np.array([100.0, 50.0]), [np.array([0, 1]), np.array([1])],
                       np.array([80.0, 60.0]))  # fmt: skip
    np.testing.assert_allclose(alloc.sum(axis=0), [80.0, 60.0])
    assert (alloc.sum(axis=1) <= [100.0, 50.0]).all()
    assert alloc[1, 0] == 0.0  # S2 does not hold A


def test_allocate_puts_an_unavoidable_shortfall_on_the_larger_skill() -> None:
    holds = [np.array([0, 1])]
    alloc = p.allocate(np.array([100.0]), holds, np.array([40.0, 90.0]))
    np.testing.assert_allclose(alloc.sum(axis=0), [40.0, 60.0])  # B (larger) is short 30
    alloc = p.allocate(np.array([100.0]), holds, np.array([70.0, 70.0]))
    np.testing.assert_allclose(alloc.sum(axis=0), [70.0, 30.0])  # tie: skill order
    assert p.allocate(np.array([0.0]), holds, np.array([5.0, 5.0])).sum() == 0.0


def test_team_coverage_available_includes_spread_spare_hours() -> None:
    hours = np.array([[100.0], [50.0]])
    holds = [np.array([0, 1]), np.array([1])]
    required = np.array([[20.0], [30.0]])
    covered, available = p.team_coverage(hours, holds, required, weights=required)
    np.testing.assert_allclose(covered[:, 0], [20.0, 30.0])
    np.testing.assert_allclose(available.sum(), 150.0)  # every planned hour is somewhere
    assert (available[:, 0] >= required[:, 0]).all()


def forecast_frame(hours: dict[str, float], season: list[float] | None = None) -> pd.DataFrame:
    """A flat forecast table: ``hours[skill]`` required every plan month."""
    months = f.plan_months(P)
    season = season or [1.0] * len(months)
    rows = [
        (m, s, h / 13.2, h / 13.2, h, h, season[k], "ets")
        for k, m in enumerate(months)
        for s, h in hours.items()
    ]
    return pd.DataFrame(rows, columns=f.FORECAST_SCHEMAS["forecast"].names)


def test_team_that_can_cover_every_skill_hires_nobody() -> None:
    # Two full-timers (~120-133 planned h/month): S1 holds MID and FWD, S2 only FWD.
    # MID 100 + FWD 100 fits only if S1 moves to MID: a demand-share split would
    # put 50% of S1 on FWD and invent a MID shortfall every month.
    team = [scout("S1", {"MID-Iberia-ES", "FWD-Iberia-ES"}), scout("S2", {"FWD-Iberia-ES"})]
    fc = forecast_frame({"MID-Iberia-ES": 100.0, "FWD-Iberia-ES": 100.0})
    capacity, hiring = p.build_capacity_plan(team, no_leave(), fc, P)
    assert hiring.empty
    assert (capacity["gap_hours"] <= 1e-6).all()


def test_single_holder_shortfall_still_hires() -> None:
    # Only S1 holds GK and GK needs 200 h a month: a real, lasting shortfall.
    team = [scout("S1", {"GK-Iberia-ES"}), scout("S2", {"FWD-Iberia-ES"})]
    fc = forecast_frame({"GK-Iberia-ES": 200.0, "FWD-Iberia-ES": 60.0})
    capacity, hiring = p.build_capacity_plan(team, no_leave(), fc, P)
    assert set(hiring["skill_type"]) == {"GK-Iberia-ES"}
    assert "full_time" in set(hiring["hire_type"])
    fwd = capacity[capacity["skill_type"] == "FWD-Iberia-ES"]
    assert (fwd["gap_hours"] < 0).all()  # FWD has slack, and nobody is hired for it


def test_hire_for_one_skill_can_close_another_skills_gap() -> None:
    # S1 holds MID and FWD; FWD has nobody else. A MID hire frees S1 for FWD.
    team = [scout("S1", {"MID-Iberia-ES", "FWD-Iberia-ES"})]
    fc = forecast_frame({"MID-Iberia-ES": 120.0, "FWD-Iberia-ES": 100.0})
    capacity, hiring = p.build_capacity_plan(team, no_leave(), fc, P)
    # The shortfall lands on the larger skill (MID); the FWD gap then closes too.
    assert set(hiring["skill_type"]) == {"MID-Iberia-ES"}
    reachable = capacity["month"] >= f.plan_months(P)[3]
    assert (capacity.loc[reachable, "gap_after_hires_hours"] <= p.hiring_tolerance(P)).all()


# --- the hiring rules ---------------------------------------------------------------------

TOL = p.hiring_tolerance(P)


def test_hiring_tolerance_is_a_quarter_of_a_full_timer() -> None:
    assert p.hiring_tolerance(P) == pytest.approx(0.25 * FT_CAP.mean())


def test_no_gap_or_a_gap_below_tolerance_hires_nobody() -> None:
    for gaps in ([0.0] * 12, [-2.0] * 12, [0.2] * 12):  # 0.2 FT < 0.25 FT tolerance
        hires, hired = hires_for(gaps)
        assert hires == [] and (hired == 0).all()


def test_persistent_gap_hires_full_time_lead_time_ahead() -> None:
    # Half a full-timer short in months 4..7 (4 months >= 3): one full-time hire,
    # acting at 4 - 3 = 1, joining at 4.
    hires, hired = hires_for([0.0] * 4 + [0.5] * 4 + [0.0] * 4)
    assert len(hires) == 1
    h = hires[0]
    assert (h.month_to_act, h.joins_month, h.hire_type, h.count) == (1, 4, "full_time", 1)
    assert h.skill_type == "MID-Iberia-ES"
    assert "persistent" in h.reason and "late" not in h.reason
    np.testing.assert_allclose(hired, np.r_[np.zeros(4), FT_CAP[4:]])


def test_open_ended_gap_at_the_end_of_the_plan_is_persistent() -> None:
    # Only 2 short months, but they are the last two and growing: the forecast
    # stops, demand does not. Full-time, not "peak-only" freelancers.
    hires, _ = hires_for([0.0] * 10 + [0.6, 1.2])
    assert [(h.month_to_act, h.joins_month, h.hire_type) for h in hires] == [(7, 10, "full_time")]
    assert "open-ended" in hires[0].reason


def test_peak_only_gap_hires_freelance_one_month_ahead() -> None:
    # Short in months 6 and 7 only (2 < 3): freelancers sized to the larger month.
    gaps = [0.0] * 6 + [1.2, 2.5] + [0.0] * 4
    hires, hired = hires_for(gaps, cap=FL_CAP)
    assert len(hires) == 1
    h = hires[0]
    expected = int(np.ceil((2.5 * FL_CAP[7] - TOL) / FL_CAP[7]))
    assert (h.month_to_act, h.joins_month, h.hire_type, h.count) == (5, 6, "freelance", expected)
    assert "peak-only" in h.reason
    assert ((np.array(gaps) * FL_CAP - hired) <= TOL + 1e-6).all()


def test_seasonal_peak_at_flat_demand_is_freelance_not_full_time() -> None:
    # Deseasonalised demand equals capacity; June-August are short only because
    # of the season (3 consecutive months, = persistent_gap_months).
    season = np.array(g.SEASONALITY_PROFILE)
    available = np.full(12, 1000.0)
    hires, _ = p.plan_skill_hires("MID-Iberia-ES", 1000.0 * season, available, P, season=season)
    assert hires and {h.hire_type for h in hires} == {"freelance"}
    assert all("peak-only" in h.reason for h in hires)
    # Judged on raw gaps (no seasonal index), the same peak looks persistent.
    naive, _ = p.plan_skill_hires("MID-Iberia-ES", 1000.0 * season, available, P)
    assert "full_time" in {h.hire_type for h in naive}


def test_full_time_sized_to_the_floor_of_the_run_peaks_go_freelance() -> None:
    # Run of 4 (persistent). Base load 1 FT; what is left above it is peak-only.
    gaps = [0.0] * 4 + [1.5, 1.0, 2.6, 1.0] + [0.0] * 4
    hires, hired = hires_for(gaps)
    ft = [h for h in hires if h.hire_type == "full_time"]
    fl = [h for h in hires if h.hire_type == "freelance"]
    assert [(h.month_to_act, h.joins_month, h.count) for h in ft] == [(1, 4, 1)]
    assert sorted((h.joins_month, h.month_to_act) for h in fl) == [(4, 3), (6, 5)]
    assert ((np.array(gaps) * FT_CAP - hired) <= TOL + 1e-6).all()


def test_late_act_month_is_clipped_to_zero_and_early_months_stay_short() -> None:
    # Persistent gap from month 0: full-time can join at 3 at the earliest; they act
    # now (0) and the reason says late. Freelancers bridge months 1..2 (they can join
    # at 1, also acting at 0); month 0 itself cannot be helped.
    gaps = [1.0] * 12
    hires, hired = hires_for(gaps)
    ft = [h for h in hires if h.hire_type == "full_time"]
    fl = [h for h in hires if h.hire_type == "freelance"]
    assert ft[0].month_to_act == 0 and ft[0].joins_month == 3 and "late" in ft[0].reason
    assert [(h.month_to_act, h.joins_month) for h in fl] == [(0, 1)]
    assert "bridge" in fl[0].reason and "late" in fl[0].reason
    assert hired[0] == 0.0  # month 0 stays short
    gap_after = np.array(gaps) * FT_CAP - hired
    assert gap_after[0] > TOL and (gap_after[1:] <= TOL + 1e-6).all()
    assert all(h.month_to_act >= 0 for h in hires)


def test_late_peak_nobody_can_reach_hires_nobody() -> None:
    # A one-month peak in month 0: no freelancer can join in time, and the run is
    # over before anyone could, so nothing is hired.
    hires, hired = hires_for([1.0] + [0.0] * 11, cap=FL_CAP)
    assert hires == [] and (hired == 0).all()


def test_hires_reduce_later_gaps() -> None:
    # Gap grows: 0.6, 1.2, ..., months 3..11. Each persistent decision is recomputed
    # after the previous hires, so earlier hires shrink what later months need.
    gaps = [0.0] * 3 + [0.6 * k for k in range(1, 10)]
    hires, hired = hires_for(gaps)
    assert {h.hire_type for h in hires} == {"full_time"}
    joins = [h.joins_month for h in hires]
    assert joins == sorted(joins) and joins[0] == 3
    # Never more full-timers than the December gap needs (rounded up).
    assert sum(h.count for h in hires) <= int(np.ceil(gaps[-1]))
    assert ((np.array(gaps) * FT_CAP - hired) <= TOL + 1e-6).all()


def test_lead_times_and_persistence_are_parameters() -> None:
    params = apply_overrides(
        P,
        {
            "capacity_plan.lead_time_full_time_months": 5,
            "capacity_plan.lead_time_freelance_months": 2,
            "capacity_plan.persistent_gap_months": 6,
        },
    )
    hires, _ = hires_for([0.0] * 6 + [0.5] * 6, params=params)
    assert [(h.month_to_act, h.joins_month, h.hire_type) for h in hires] == [(1, 6, "full_time")]
    hires, _ = hires_for([0.0] * 6 + [0.5] * 5 + [0.0], params=params)  # 5 < 6: a peak
    assert [(h.month_to_act, h.joins_month, h.hire_type) for h in hires] == [(4, 6, "freelance")]


# --- the stage on the default world ---------------------------------------------------------


def test_capacity_plan_contract_and_arithmetic(built) -> None:
    _, capacity, hiring = built
    assert list(capacity.columns) == p.PLAN_SCHEMAS["capacity_plan"].names
    assert list(hiring.columns) == p.PLAN_SCHEMAS["hiring_plan"].names
    assert len(capacity) == 12 * len(g.SKILL_TYPES)
    np.testing.assert_allclose(
        capacity["gap_hours"], capacity["required_hours"] - capacity["available_hours"]
    )
    short_before = capacity["gap_hours"].clip(lower=0).groupby(capacity["month"]).sum()
    short_after = capacity["gap_after_hires_hours"].clip(lower=0).groupby(capacity["month"]).sum()
    assert (short_after <= short_before + 1e-6).all()
    assert set(hiring["hire_type"]) <= {"full_time", "freelance"}
    assert (hiring["count"] > 0).all()
    assert (hiring["joins_month"] >= hiring["month_to_act"]).all()
    assert (hiring["month_to_act"] >= 0).all()


def test_required_hours_are_the_planning_quantile(built) -> None:
    fc, capacity, _ = built
    merged = capacity.merge(fc, on=["month", "skill_type"])
    np.testing.assert_allclose(merged["required_hours"], merged["hours_pq"])


def test_available_hours_account_for_every_planned_hour(world: g.World, built) -> None:
    _, capacity, _ = built
    team = g.scouts_from_frame(world.scouts)
    total = p.scout_monthly_hours(team, world.scout_unavailability, P).sum(axis=0)
    by_month = capacity.groupby("month", sort=True)["available_hours"].sum().to_numpy()
    np.testing.assert_allclose(by_month, total)


def test_every_month_a_freelancer_can_reach_is_covered(built) -> None:
    _, capacity, _ = built
    reachable = capacity["month"] >= f.plan_months(P)[P.capacity_plan.lead_time_freelance_months]
    assert (capacity.loc[reachable, "gap_after_hires_hours"] <= TOL + 1e-6).all()


def plan_for(growth: float) -> pd.DataFrame:
    params = apply_overrides(
        P, {"demand.actual_growth": growth, "capacity_plan.assumed_growth": growth}
    )
    w = g.generate_world(params)
    fc, _ = f.build_forecast(w, params)
    return p.build_capacity_plan(w.scouts, w.scout_unavailability, fc, params)[1]


def test_no_phantom_hiring_when_demand_shrinks() -> None:
    hiring = plan_for(0.5)
    assert "full_time" not in set(hiring["hire_type"])
    assert hiring["count"].sum() <= 3


def test_summer_peak_at_flat_growth_is_covered_by_freelancers() -> None:
    hiring = plan_for(1.0)
    summer = hiring[hiring["joins_month"].between(5, 7)]
    assert not summer.empty and set(summer["hire_type"]) == {"freelance"}


def test_growth_at_the_end_of_the_year_hires_full_time(built) -> None:
    _, _, hiring = built  # 4x
    late_peak = hiring[(hiring["joins_month"] >= 9) & hiring["reason"].str.startswith("peak-only")]
    assert late_peak.empty
    assert (hiring[hiring["joins_month"] >= 9]["hire_type"] == "full_time").all()


def test_hired_hours_match_the_hiring_table(built) -> None:
    _, capacity, hiring = built
    for skill, rows in hiring.groupby("skill_type"):
        expected = np.zeros(12)
        for r in rows.itertuples():
            expected[r.joins_month :] += (
                r.count * p.hire_monthly_hours(r.hire_type, P)[r.joins_month :]
            )
        got = capacity[capacity["skill_type"] == skill]["hired_hours"].to_numpy()
        np.testing.assert_allclose(got, expected)


def test_assumed_growth_changes_the_plan(world: g.World, built) -> None:
    low = apply_overrides(P, {"capacity_plan.assumed_growth": 2.0})
    fc_low, _ = f.build_forecast(world, low)
    _, hiring_low = p.build_capacity_plan(world.scouts, world.scout_unavailability, fc_low, low)
    _, _, hiring = built
    assert hiring_low["count"].sum() < hiring["count"].sum()


def test_missing_season_index_means_flat(world: g.World, built) -> None:
    fc, _, _ = built
    old = fc.drop(columns="season_index")  # e.g. a forecast written before the column
    capacity, hiring = p.build_capacity_plan(world.scouts, world.scout_unavailability, old, P)
    assert len(capacity) == 12 * len(g.SKILL_TYPES) and not hiring.empty


def test_plan_is_deterministic_and_round_trips(world: g.World, built, tmp_path: Path) -> None:
    fc, capacity, hiring = built
    again = p.build_capacity_plan(world.scouts, world.scout_unavailability, fc, P)
    pd.testing.assert_frame_equal(capacity, again[0])
    pd.testing.assert_frame_equal(hiring, again[1])
    p.write_plan(capacity, hiring, tmp_path)
    back = p.read_plan(tmp_path)
    pd.testing.assert_frame_equal(back[0], capacity)
    pd.testing.assert_frame_equal(back[1], hiring)


# --- hires as scouts -------------------------------------------------------------------------


def test_hires_to_scouts(built) -> None:
    _, _, hiring = built
    taken = {"Ana Velmar"}
    hires = p.hires_to_scouts(hiring, P, make_stream(42, p.HIRES_STREAM), taken_names=taken)
    assert len(hires) == hiring["count"].sum()
    assert [h.scout_id for h in hires[:2]] == ["H001", "H002"]
    assert len({h.scout_id for h in hires}) == len(hires)
    assert len({h.name for h in hires}) == len(hires) and not taken & {h.name for h in hires}
    expanded = hiring.loc[hiring.index.repeat(hiring["count"])].reset_index(drop=True)
    for h, row in zip(hires, expanded.itertuples(), strict=True):
        assert h.skills == {row.skill_type}
        assert h.home_region == g.skill_region(row.skill_type)
        assert h.employment == row.hire_type and h.joined_month == row.joins_month
        if h.employment == "full_time":
            assert h.monthly_salary == P.cost.full_time_monthly_salary and h.hourly_rate is None
            assert h.min_weekly_hours == h.max_weekly_hours == 37.5
        else:
            assert h.hourly_rate == pytest.approx(P.cost.freelance_hourly_rate)
            assert (h.min_weekly_hours, h.max_weekly_hours) == P.team.freelance_weekly_hours
    again = p.hires_to_scouts(hiring, P, make_stream(42, p.HIRES_STREAM), taken_names=taken)
    assert again == hires
    assert p.hires_to_scouts(hiring.iloc[:0], P, make_stream(42, p.HIRES_STREAM)) == []


def test_changing_one_hire_count_leaves_the_other_hires_alone(built) -> None:
    """Per-hire random streams: common random numbers across hiring plans (D-015)."""
    _, _, hiring = built
    more = hiring.copy()
    more.loc[0, "count"] += 2
    a = p.hires_to_scouts(hiring, P, make_stream(42, p.HIRES_STREAM))
    b = p.hires_to_scouts(more, P, make_stream(42, p.HIRES_STREAM))

    def keyed(hires: list[Scout], table: pd.DataFrame) -> dict:
        rows = table.loc[table.index.repeat(table["count"])]
        ks = rows.groupby(level=0).cumcount()
        keys = [(r.skill_type, r.hire_type, r.joins_month, k)
                for r, k in zip(rows.itertuples(), ks, strict=True)]  # fmt: skip
        return {key: (h.former_clubs, h.name) for key, h in zip(keys, hires, strict=True)}

    ka, kb = keyed(a, hiring), keyed(b, more)
    first = hiring.iloc[0]
    for key, value in ka.items():
        if key[:3] != (first.skill_type, first.hire_type, first.joins_month):
            assert kb[key][0] == value[0]  # former clubs: always identical
    same_name = sum(kb[k][1] == v[1] for k, v in ka.items())
    assert same_name >= len(ka) - 2  # names: identical unless a collision forced a redraw


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_forecast_then_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "run"
    assert cli.main(["data", "--params", str(DEFAULT_CONFIG_PATH), "--out", str(run)]) == 0
    assert cli.main(["forecast", "--run", str(run)]) == 0
    out = capsys.readouterr().out
    assert "backtest" in out and "ALL" in out and "seasonal naive on aggregate" in out
    assert "realised" in out
    assert cli.main(["plan", "--run", str(run)]) == 0
    out = capsys.readouterr().out
    assert "hiring plan" in out and "full_time" in out
    for name in ("forecast", "backtest", "capacity_plan", "hiring_plan"):
        assert (run / f"{name}.parquet").is_file()


def test_cli_reports_missing_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["forecast", "--run", str(tmp_path)]) == 2
    assert "run `data` first" in capsys.readouterr().err
    run = tmp_path / "run"
    assert cli.main(["data", "--params", str(DEFAULT_CONFIG_PATH), "--out", str(run)]) == 0
    capsys.readouterr()
    assert cli.main(["plan", "--run", str(run)]) == 2
    assert "run `forecast` first" in capsys.readouterr().err


# --- capacity_plan.hire_mix -----------------------------------------------------------------

FL_ONLY = apply_overrides(P, {"capacity_plan.hire_mix": "freelance_only"})
FT_ONLY = apply_overrides(P, {"capacity_plan.hire_mix": "full_time_only"})


def test_freelance_only_covers_a_persistent_gap_with_freelancers() -> None:
    # The rule would hire one full-timer for this 4-month run (see above).
    gaps = [0.0] * 4 + [0.5] * 4 + [0.0] * 4
    hires, hired = hires_for(gaps, params=FL_ONLY)
    assert [(h.month_to_act, h.joins_month, h.hire_type) for h in hires] == [(3, 4, "freelance")]
    assert hires[0].reason.startswith("freelance_only")
    assert ((np.array(gaps) * FT_CAP - hired) <= TOL + 1e-6).all()


def test_full_time_only_covers_a_peak_with_full_timers_sized_to_the_peak() -> None:
    # A 2-month peak: the rule would use freelancers. Full-time only sizes to the
    # peak month and keeps them for the rest of the year.
    gaps = [0.0] * 6 + [1.2, 2.5] + [0.0] * 4
    hires, hired = hires_for(gaps, params=FT_ONLY)
    expected = int(np.ceil((2.5 * FT_CAP[7] - TOL) / FT_CAP[7]))
    assert [(h.month_to_act, h.joins_month, h.hire_type, h.count) for h in hires] == [
        (3, 6, "full_time", expected)
    ]
    np.testing.assert_allclose(hired, np.r_[np.zeros(6), expected * FT_CAP[6:]])


def test_full_time_only_has_no_bridge_so_early_months_stay_short() -> None:
    gaps = [1.0] * 12
    hires, hired = hires_for(gaps, params=FT_ONLY)
    assert {h.hire_type for h in hires} == {"full_time"}
    assert hires[0].joins_month == 3 and "late" in hires[0].reason
    assert (hired[:3] == 0).all()
    assert ((np.array(gaps) * FT_CAP - hired)[3:] <= TOL + 1e-6).all()


def test_rule_mix_is_the_default_behaviour(world: g.World, built) -> None:
    fc, capacity, hiring = built
    explicit = apply_overrides(P, {"capacity_plan.hire_mix": "rule"})
    cap2, hire2 = p.build_capacity_plan(world.scouts, world.scout_unavailability, fc, explicit)
    pd.testing.assert_frame_equal(capacity, cap2)
    pd.testing.assert_frame_equal(hiring, hire2)
    assert set(hiring["hire_type"]) == {"full_time", "freelance"}


@pytest.mark.parametrize(
    ("mix", "kind"), [("freelance_only", "freelance"), ("full_time_only", "full_time")]
)
def test_one_type_mixes_on_the_default_world(world: g.World, built, mix: str, kind: str) -> None:
    fc, _, rule_hiring = built
    params = apply_overrides(P, {"capacity_plan.hire_mix": mix})
    capacity, hiring = p.build_capacity_plan(world.scouts, world.scout_unavailability, fc, params)
    assert len(hiring) and set(hiring["hire_type"]) == {kind}
    lead = (
        P.capacity_plan.lead_time_freelance_months
        if kind == "freelance"
        else P.capacity_plan.lead_time_full_time_months
    )
    assert ((hiring["joins_month"] - hiring["month_to_act"]) == lead).all()
    # Every month a hire of this type could reach ends covered (within tolerance).
    reachable = capacity[capacity["month"] >= f.plan_months(P)[lead]]
    assert (reachable["gap_after_hires_hours"] <= p.hiring_tolerance(P) + 1e-6).all()
