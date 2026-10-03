"""Stage [3] capacity plan: available hours, the multi-skill split, every hiring rule."""

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


def test_multi_skill_split_sums_to_the_scouts_hours_and_follows_demand() -> None:
    months = f.plan_months(P)[:3]
    demand = pd.DataFrame(
        {"A": [30.0, 0.0, 0.0], "B": [10.0, 0.0, 5.0], "C": [60.0, 50.0, 0.0]}, index=months
    )
    s1, s2 = scout("S1", {"A", "B"}), scout("S2", {"C"})
    hours = np.array([[100.0, 80.0, 60.0], [50.0, 50.0, 50.0]])
    split = p.split_by_demand([s1, s2], hours, demand)
    np.testing.assert_allclose(split[["A", "B"]].sum(axis=1), hours[0])
    np.testing.assert_allclose(split["C"], hours[1])
    assert split.loc[months[0], "A"] == pytest.approx(75.0)  # 30 / (30 + 10)
    assert split.loc[months[1], "A"] == pytest.approx(40.0)  # no demand at all: equal split
    assert split.loc[months[2], "B"] == pytest.approx(60.0)


def test_multi_skill_split_on_the_default_team(world: g.World, built) -> None:
    fc, capacity, _ = built
    team = g.scouts_from_frame(world.scouts)
    total = p.scout_monthly_hours(team, world.scout_unavailability, P).sum(axis=0)
    by_month = capacity.groupby("month", sort=True)["available_hours"].sum().to_numpy()
    np.testing.assert_allclose(by_month, total)


# --- the hiring rules ---------------------------------------------------------------------


def test_no_gap_no_hires() -> None:
    hires, hired = hires_for([0.0] * 12)
    assert hires == [] and (hired == 0).all()
    hires, _ = hires_for([-2.0] * 12)
    assert hires == []


def test_persistent_gap_hires_full_time_lead_time_ahead() -> None:
    # Half a full-timer short in months 5..11 (7 months >= 3): one full-time hire,
    # acting at 5 - 3 = 2, joining at 5.
    hires, hired = hires_for([0.0] * 5 + [0.5] * 7)
    assert len(hires) == 1
    h = hires[0]
    assert (h.month_to_act, h.joins_month, h.hire_type, h.count) == (2, 5, "full_time", 1)
    assert h.skill_type == "MID-Iberia-ES"
    assert "persistent" in h.reason and "late" not in h.reason
    np.testing.assert_allclose(hired, np.r_[np.zeros(5), FT_CAP[5:]])


def test_peak_only_gap_hires_freelance_one_month_ahead() -> None:
    # Short in months 6 and 7 only (2 < 3): freelancers sized to the larger month.
    gaps = [0.0] * 6 + [1.2, 2.5] + [0.0] * 4
    hires, hired = hires_for(gaps, cap=FL_CAP)
    assert len(hires) == 1
    h = hires[0]
    assert (h.month_to_act, h.joins_month, h.hire_type, h.count) == (5, 6, "freelance", 3)
    assert "peak-only" in h.reason
    assert (hired[6:] == 3 * FL_CAP[6:]).all()


def test_full_time_sized_to_the_smallest_gap_of_the_run_peaks_go_freelance() -> None:
    # Run of 4 (persistent). Base load 1 FT; what is left above it is peak-only.
    gaps = [0.0] * 4 + [1.5, 1.0, 2.6, 1.0] + [0.0] * 4
    hires, hired = hires_for(gaps)
    ft = [h for h in hires if h.hire_type == "full_time"]
    fl = [h for h in hires if h.hire_type == "freelance"]
    assert [(h.month_to_act, h.joins_month, h.count) for h in ft] == [(1, 4, 1)]
    assert sorted((h.joins_month, h.month_to_act) for h in fl) == [(4, 3), (6, 5)]
    gap_after = np.array(gaps) * FT_CAP - hired
    assert (gap_after <= 1e-6).all()


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
    assert gap_after[0] > 0 and (gap_after[1:] <= 1e-6).all()
    assert all(h.month_to_act >= 0 for h in hires)


def test_late_peak_with_zero_lead_options_left_hires_nobody() -> None:
    # A one-month peak in month 0: no freelancer can join in time, and the run is
    # over before anyone could, so nothing is hired.
    hires, hired = hires_for([1.0] + [0.0] * 11, cap=FL_CAP)
    assert hires == [] and (hired == 0).all()


def test_hires_reduce_later_gaps() -> None:
    # Gap grows: 0.6, 1.2, ..., months 3..11. Each persistent decision is recomputed
    # after the previous hires, so earlier hires shrink what later months need.
    gaps = [0.0] * 3 + [0.6 * k for k in range(1, 10)]
    hires, hired = hires_for(gaps)
    joins = [h.joins_month for h in hires if h.hire_type == "full_time"]
    total_ft = sum(h.count for h in hires if h.hire_type == "full_time")
    assert joins == sorted(joins) and joins[0] == 3
    # Never more full-timers than the December gap needs (rounded up), however
    # many months triggered a decision.
    assert total_ft <= int(np.ceil(gaps[-1]))
    assert ((np.array(gaps) * FT_CAP - hired) <= 1e-6).all()


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
    np.testing.assert_allclose(
        capacity["gap_after_hires_hours"], capacity["gap_hours"] - capacity["hired_hours"]
    )
    assert set(hiring["hire_type"]) <= {"full_time", "freelance"}
    assert (hiring["count"] > 0).all()
    assert (hiring["joins_month"] >= hiring["month_to_act"]).all()
    assert (hiring["month_to_act"] >= 0).all()


def test_required_hours_are_the_planning_quantile(built) -> None:
    fc, capacity, _ = built
    merged = capacity.merge(fc, on=["month", "skill_type"])
    np.testing.assert_allclose(merged["required_hours"], merged["hours_pq"])


def test_every_month_a_freelancer_can_reach_is_covered(built) -> None:
    _, capacity, _ = built
    reachable = capacity["month"] >= f.plan_months(P)[P.capacity_plan.lead_time_freelance_months]
    assert (capacity.loc[reachable, "gap_after_hires_hours"] <= 1e-6).all()


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
