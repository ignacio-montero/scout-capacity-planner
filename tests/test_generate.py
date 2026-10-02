"""Stage [1] synthetic world: PRD M1 acceptance criteria and the generator's rules.

Statistical checks use the default seed, so they are deterministic; their
tolerances are still set at roughly 3 standard deviations of the noise, so a
change of seed or of an unrelated draw should not make them flaky.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from scout_planner import generate as g
from scout_planner.config import FULL_TIME_WEEKLY_HOURS, WEEKS_PER_YEAR, Params, apply_overrides
from scout_planner.domain import Fixture, Request

P = Params()


@pytest.fixture(scope="module")
def world() -> g.World:
    return g.generate_world(P)


def assert_world_equal(a: g.World, b: g.World, skip: tuple[str, ...] = ()) -> None:
    for name, df in a.tables().items():
        if name not in skip:
            pd.testing.assert_frame_equal(df, b.tables()[name], obj=name)


def future(world: g.World) -> pd.DataFrame:
    return world.requests[world.requests["period"] == "future"]


def history(world: g.World) -> pd.DataFrame:
    return world.requests[world.requests["period"] == "history"]


def monthly_counts(requests: pd.DataFrame) -> pd.Series:
    """Requests per calendar month, indexed by the month's first day."""
    months = pd.to_datetime(requests["received_date"]).dt.to_period("M").dt.to_timestamp()
    return requests.groupby(months).size()


def deseasonalised_daily(counts: pd.Series, strength: float = 1.0) -> np.ndarray:
    """Monthly counts -> requests per day with the seasonal multiplier divided out."""
    season = g.seasonal_index(strength)[counts.index.month - 1]
    return counts.to_numpy() / (season * counts.index.days_in_month.to_numpy())


# --- schema and contract -------------------------------------------------------------

CONTRACT_COLUMNS = {  # DATA_CONTRACTS.md section 1, plus the M1 additions
    "scouts": [
        "scout_id", "name", "employment", "skills", "home_region", "min_weekly_hours",
        "max_weekly_hours", "former_clubs", "monthly_salary", "hourly_rate", "joined_month",
    ],
    "scout_unavailability": ["scout_id", "date", "reason"],
    "scout_weekly_hours": ["scout_id", "week_start", "hours"],
    "fixtures": ["fixture_id", "date", "league", "region", "home_club", "away_club"],
    "requests": [
        "request_id", "client_club", "player_club", "skill_type", "received_date", "due_date",
        "needs_live_view", "desk_hours", "writeup_hours", "period", "rework_draw",
        "at_risk_day_one",
    ],
}  # fmt: skip


def test_tables_have_the_contract_columns(world: g.World) -> None:
    assert {k: list(v.columns) for k, v in world.tables().items()} == CONTRACT_COLUMNS
    assert {k: s.names for k, s in g.TABLE_SCHEMAS.items()} == CONTRACT_COLUMNS


def test_in_memory_dtypes(world: g.World) -> None:
    req, scouts = world.requests, world.scouts
    assert req["request_id"].dtype == "str"
    assert req["needs_live_view"].dtype == bool and req["at_risk_day_one"].dtype == bool
    assert req["desk_hours"].dtype == "float64" and scouts["joined_month"].dtype == "int64"
    assert all(type(d) is dt.date for d in req["received_date"])
    assert all(type(d) is dt.date for d in world.fixtures["date"])
    assert all(isinstance(s, list) for s in scouts["skills"])


def test_written_parquet_schema_matches_contract(world: g.World, tmp_path: Path) -> None:
    paths = g.write_world(world, tmp_path)
    assert set(paths) == set(g.TABLE_SCHEMAS)
    for name, path in paths.items():
        assert path == tmp_path / "raw" / f"{name}.parquet"
        written = pq.read_schema(path).remove_metadata()
        assert written.equals(g.TABLE_SCHEMAS[name]), name


def test_write_then_read_round_trips(world: g.World, tmp_path: Path) -> None:
    g.write_world(world, tmp_path)
    assert_world_equal(world, g.read_world(tmp_path))


def test_no_duplicate_ids(world: g.World) -> None:
    for table, col in [
        ("scouts", "scout_id"),
        ("fixtures", "fixture_id"),
        ("requests", "request_id"),
    ]:
        ids = world.tables()[table][col]
        assert ids.is_unique, table
    assert world.scouts["name"].is_unique
    assert not world.scout_unavailability.duplicated(["scout_id", "date"]).any()
    assert not world.scout_weekly_hours.duplicated(["scout_id", "week_start"]).any()


def test_id_formats(world: g.World) -> None:
    assert world.scouts["scout_id"].str.fullmatch(r"S\d{3}").all()
    assert world.fixtures["fixture_id"].str.fullmatch(r"F\d{5}").all()
    assert world.requests["request_id"].str.fullmatch(r"R\d{5}").all()


# --- determinism ---------------------------------------------------------------------


def test_same_params_give_equal_frames(world: g.World) -> None:
    assert_world_equal(world, g.generate_world(P))


def test_same_params_give_byte_identical_parquet(tmp_path: Path) -> None:
    a = g.write_world(g.generate_world(P), tmp_path / "a")
    b = g.write_world(g.generate_world(P), tmp_path / "b")
    for name in a:
        assert a[name].read_bytes() == b[name].read_bytes(), name


def test_different_seed_gives_a_different_world(world: g.World) -> None:
    other = g.generate_world(P, seed=P.sim.seed + 1)
    assert not other.requests["desk_hours"].equals(world.requests["desk_hours"])
    assert not other.fixtures.equals(world.fixtures)


def test_generation_is_fast() -> None:
    started = time.perf_counter()
    g.generate_world(P)
    assert time.perf_counter() - started < 10.0


# --- common random numbers (D-015) ---------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"assignment.policy": "edf", "assignment.cadence": "weekly", "assignment.unit": "bundle"},
        {"assignment.weights.lateness": 1.0, "assignment.time_limit_s": 5.0},
        {"automation.enabled": True, "automation.rework_rate": 0.4},
        {"automation.desk_reduction": 0.1, "automation.monthly_cost": 0.0},
        {"capacity_plan.assumed_growth": 2.0, "capacity_plan.quantile": 0.5},
        {"cost.late_penalty": 0.0, "sim.seeds": 5, "sim.target_on_time": 0.9},
        {"team.follow_hiring_plan": False},
    ],
)
def test_params_outside_the_world_do_not_change_it(world: g.World, overrides: dict) -> None:
    assert_world_equal(world, g.generate_world(apply_overrides(P, overrides)))


def test_cost_params_only_change_the_echoed_money_columns(world: g.World) -> None:
    params = apply_overrides(
        P, {"cost.full_time_monthly_salary": 6000.0, "cost.freelance_premium": 2.0}
    )
    other = g.generate_world(params)
    assert_world_equal(world, other, skip=("scouts",))
    money = ["monthly_salary", "hourly_rate"]
    pd.testing.assert_frame_equal(
        world.scouts.drop(columns=money), other.scouts.drop(columns=money)
    )
    ft = other.scouts["employment"] == "full_time"
    assert (other.scouts.loc[ft, "monthly_salary"] == 6000.0).all()
    rates = other.scouts.loc[~ft, "hourly_rate"].to_numpy()
    assert rates == pytest.approx(np.full(len(rates), params.cost.freelance_hourly_rate))


def test_actual_growth_leaves_scouts_fixtures_and_history_unchanged(world: g.World) -> None:
    other = g.generate_world(apply_overrides(P, {"demand.actual_growth": 2.0}))
    assert_world_equal(world, other, skip=("requests",))
    pd.testing.assert_frame_equal(history(world), history(other))
    assert len(future(other)) < len(future(world))


def _content(requests: pd.DataFrame) -> set[tuple]:
    """Requests as comparable tuples, ignoring the id (ids are numbered after thinning)."""
    return set(requests.drop(columns="request_id").itertuples(index=False, name=None))


def test_lower_growth_requests_are_a_subset_of_higher_growth_ones(world: g.World) -> None:
    """Thinning: a 2x world's requests all appear, identical, in the 4x world."""
    low = g.generate_world(apply_overrides(P, {"demand.actual_growth": 2.0}))
    assert _content(future(low)) <= _content(future(world))


SHAPE_OVERRIDES = [
    {"team.freelance_count": 26},
    {"team.full_time_count": 30, "team.leave_days_per_year": 10},
    {"team.freelance_weekly_hours": [4, 12]},
    {"demand.live_view_share": 0.2},
    {"demand.desk_hours": [3, 12], "demand.writeup_hours": [1, 5], "demand.live_view_hours": 6},
]
ARRIVAL_COLUMNS = [
    "request_id", "client_club", "player_club", "skill_type", "received_date", "due_date",
    "period", "rework_draw",
]  # fmt: skip


@pytest.mark.parametrize("overrides", SHAPE_OVERRIDES)
def test_base_volume_ignores_the_runs_team_and_demand_shape(overrides: dict) -> None:
    """Reference-world calibration: only start_load moves the base volume."""
    assert g.base_monthly_volume(apply_overrides(P, overrides)) == g.base_monthly_volume(P)
    doubled = apply_overrides(P, {"demand.start_load": 1.0})
    assert g.base_monthly_volume(doubled) == pytest.approx(g.base_monthly_volume(P) / 0.7)


@pytest.mark.parametrize("overrides", SHAPE_OVERRIDES)
def test_team_and_demand_shape_keep_the_arrival_stream(world: g.World, overrides: dict) -> None:
    """Same requests on the same days; only attributes drawn through the params change.

    Durations are one shared uniform mapped onto the run's range, and live-view
    flags are nested (live at a lower share => live at a higher one).
    """
    params = apply_overrides(P, overrides)
    other = g.generate_world(params).requests
    base = world.requests
    pd.testing.assert_frame_equal(base[ARRIVAL_COLUMNS], other[ARRIVAL_COLUMNS])
    for col, rng_now, rng_then in [
        ("desk_hours", P.demand.desk_hours, params.demand.desk_hours),
        ("writeup_hours", P.demand.writeup_hours, params.demand.writeup_hours),
    ]:
        u_now = (base[col] - rng_now[0]) / (rng_now[1] - rng_now[0])
        u_then = (other[col] - rng_then[0]) / (rng_then[1] - rng_then[0])
        np.testing.assert_allclose(u_now, u_then, atol=1e-9)
    low, high = sorted([base, other], key=lambda r: r["needs_live_view"].sum())
    assert (low["needs_live_view"] <= high["needs_live_view"]).all()


def test_a_bigger_team_has_a_lower_realised_load(world: g.World) -> None:
    bigger = apply_overrides(P, {"team.freelance_count": P.team.freelance_count + 10})
    load = g.realised_start_load(g.generate_world(bigger), bigger, months=3)
    default_load = g.realised_start_load(world, P, months=3)
    expected_ratio = g.team_available_hours_per_month(P) / g.team_available_hours_per_month(bigger)
    assert load < default_load
    assert load == pytest.approx(default_load * expected_ratio)  # same work, more hours
    assert load == pytest.approx(0.70 * expected_ratio, abs=0.05)


def test_horizon_length_does_not_change_the_overlap(world: g.World) -> None:
    longer = g.generate_world(apply_overrides(P, {"sim.months": 18}))
    pd.testing.assert_frame_equal(world.scouts, longer.scouts)
    first_year = longer.requests[longer.requests["received_date"] <= g.sim_end(P)]
    pd.testing.assert_frame_equal(world.requests, first_year)


def test_history_length_does_not_change_the_future_or_fixture_content(world: g.World) -> None:
    shorter = g.generate_world(apply_overrides(P, {"demand.history_months": 36}))
    cols = ["client_club", "player_club", "received_date", "desk_hours", "rework_draw"]
    pd.testing.assert_frame_equal(
        future(world)[cols].reset_index(drop=True), future(shorter)[cols].reset_index(drop=True)
    )
    no_id = world.fixtures.drop(columns="fixture_id")
    start = g.history_start(apply_overrides(P, {"demand.history_months": 36}))
    tail = no_id[no_id["date"] >= start].reset_index(drop=True)
    pd.testing.assert_frame_equal(tail, shorter.fixtures.drop(columns="fixture_id"))


def test_adding_a_freelancer_keeps_the_others_weekly_hours(world: g.World) -> None:
    bigger = g.generate_world(apply_overrides(P, {"team.freelance_count": 17}))
    old = world.scout_weekly_hours
    new = bigger.scout_weekly_hours
    pd.testing.assert_frame_equal(
        old, new[new["scout_id"].isin(old["scout_id"])].reset_index(drop=True)
    )


# --- scouts ----------------------------------------------------------------------------


def test_scout_mix_matches_the_team_counts(world: g.World) -> None:
    counts = world.scouts["employment"].value_counts()
    assert counts["full_time"] == P.team.full_time_count
    assert counts["freelance"] == P.team.freelance_count
    assert counts["full_time"] / len(world.scouts) == pytest.approx(0.6)  # PRD: ~60/40


def holders(scouts: pd.DataFrame, skill: str, home_only: bool) -> int:
    has = scouts["skills"].map(lambda s: skill in s)
    if home_only:
        has &= scouts["home_region"] == g.skill_region(skill)
    return int(has.sum())


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"team.full_time_count": 8, "team.freelance_count": 0, "team.skills_per_scout": [1, 1]},
        {"team.full_time_count": 3, "team.freelance_count": 60, "team.skills_per_scout": [2, 2]},
        {"team.full_time_count": 100, "team.freelance_count": 100, "team.skills_per_scout": [1, 6]},
    ],
)
def test_every_skill_has_two_holders_living_in_its_region(overrides: dict) -> None:
    scouts = g.generate_world(apply_overrides(P, overrides)).scouts
    for skill in g.SKILL_TYPES:
        assert holders(scouts, skill, home_only=True) >= 2, skill
        assert holders(scouts, skill, home_only=False) >= 2, skill


def test_tiny_team_still_covers_every_skill_twice() -> None:
    params = apply_overrides(P, {"team.full_time_count": 2, "team.freelance_count": 1})
    scouts = g.generate_world(params).scouts
    assert all(holders(scouts, s, home_only=False) >= 2 for s in g.SKILL_TYPES)


def test_default_skills_respect_the_range_and_are_mostly_home_region(world: g.World) -> None:
    lo, hi = P.team.skills_per_scout
    sizes = world.scouts["skills"].map(len)
    assert sizes.between(lo, hi).all()
    assert all(s in g.SKILL_TYPES for skills in world.scouts["skills"] for s in skills)
    pairs = world.scouts.explode("skills")
    home_share = (pairs["skills"].map(g.skill_region) == pairs["home_region"]).mean()
    assert home_share > 0.7


def test_every_region_has_scouts_in_proportion_to_demand(world: g.World) -> None:
    counts = world.scouts["home_region"].value_counts()
    for region, share in g.REGION_DEMAND_SHARE.items():
        assert counts[region] >= 2
        assert counts[region] == pytest.approx(share * len(world.scouts), abs=1)


def test_former_clubs_are_known_clubs(world: g.World) -> None:
    former = world.scouts["former_clubs"]
    assert former.map(len).between(0, 3).all()
    assert former.map(len).sum() > 0
    assert all(c in g.CLUB_NAMES for clubs in former for c in clubs)
    assert all(len(set(clubs)) == len(clubs) for clubs in former)


def test_pay_and_hours_columns_follow_employment(world: g.World) -> None:
    s = world.scouts
    ft, fl = s[s["employment"] == "full_time"], s[s["employment"] == "freelance"]
    assert (ft["monthly_salary"] == P.cost.full_time_monthly_salary).all()
    assert ft["hourly_rate"].isna().all() and fl["monthly_salary"].isna().all()
    assert (fl["hourly_rate"] == P.cost.freelance_hourly_rate).all()
    assert (ft["min_weekly_hours"] == FULL_TIME_WEEKLY_HOURS).all()
    assert (fl["min_weekly_hours"] == P.team.freelance_weekly_hours[0]).all()
    assert (fl["max_weekly_hours"] == P.team.freelance_weekly_hours[1]).all()
    assert (s["joined_month"] == 0).all()


def test_scouts_convert_to_domain_objects(world: g.World) -> None:
    scouts = g.scouts_from_frame(world.scouts)
    assert len(scouts) == len(world.scouts)
    assert sum(sc.is_freelance for sc in scouts) == P.team.freelance_count
    assert all(sc.monthly_salary is None for sc in scouts if sc.is_freelance)


# --- availability ------------------------------------------------------------------------


def test_weekly_hours_cover_every_scout_and_week(world: g.World) -> None:
    wh = world.scout_weekly_hours
    weeks = g.sim_week_starts(P)
    assert weeks[0] <= g.SIM_START < weeks[0] + dt.timedelta(days=7)
    assert all(w.weekday() == 0 for w in weeks)
    assert len(wh) == len(world.scouts) * len(weeks)
    kind = wh["scout_id"].map(world.scouts.set_index("scout_id")["employment"])
    assert (wh.loc[kind == "full_time", "hours"] == FULL_TIME_WEEKLY_HOURS).all()
    fl = wh.loc[kind == "freelance", "hours"]
    lo, hi = P.team.freelance_weekly_hours
    assert fl.between(lo, hi).all()
    assert ((fl * 2) % 1 == 0).all()  # half-hour steps
    assert fl.mean() == pytest.approx((lo + hi) / 2, abs=0.5)
    assert fl.nunique() > 10  # really drawn per week, not constant


def test_full_time_leave_is_the_allowance_in_blocks(world: g.World) -> None:
    un = world.scout_unavailability
    assert un["date"].between(g.SIM_START, g.sim_end(P)).all()
    leave = un[un["reason"] == "leave"]
    ft_ids = set(world.scouts.loc[world.scouts["employment"] == "full_time", "scout_id"])
    assert set(leave["scout_id"]) == ft_ids
    weekdays = leave[leave["date"].map(lambda d: d.weekday() < 5)]
    assert (weekdays.groupby("scout_id").size() == P.team.leave_days_per_year).all()
    # Blocks: most leave days sit next to another leave day of the same scout.
    days = leave.groupby("scout_id")["date"].apply(set)
    adjacent = sum(
        (d + dt.timedelta(days=1)) in s or (d - dt.timedelta(days=1)) in s for s in days for d in s
    )
    assert adjacent / len(leave) > 0.8


def test_freelancers_have_a_few_days_off(world: g.World) -> None:
    other = world.scout_unavailability[world.scout_unavailability["reason"] == "other"]
    fl_ids = set(world.scouts.loc[world.scouts["employment"] == "freelance", "scout_id"])
    assert set(other["scout_id"]) <= fl_ids
    assert len(other) / len(fl_ids) == pytest.approx(g.FREELANCE_DAYS_OFF_PER_YEAR, rel=0.5)


# --- fixtures and live views ----------------------------------------------------------------


def club_games(fixtures: pd.DataFrame) -> pd.DataFrame:
    home = fixtures[["date", "region", "home_club"]].rename(columns={"home_club": "club"})
    away = fixtures[["date", "region", "away_club"]].rename(columns={"away_club": "club"})
    games = pd.concat([home, away]).sort_values(["club", "date"])
    games["gap"] = pd.to_datetime(games["date"]).groupby(games["club"]).diff().dt.days
    return games


def test_fixtures_are_consistent_with_the_world_shape(world: g.World) -> None:
    f = world.fixtures
    assert (f["home_club"] != f["away_club"]).all()
    league_region = {lg.name: lg.region for lg in g.LEAGUES}
    club_league = {c.name: c.league for c in g.CLUBS}
    assert (f["league"].map(league_region) == f["region"]).all()
    assert (f["home_club"].map(club_league) == f["league"]).all()
    assert (f["away_club"].map(club_league) == f["league"]).all()
    assert set(f["home_club"]) == set(g.CLUB_NAMES)
    assert f["date"].min() >= g.history_start(P)
    assert f["date"].max() <= g.sim_end(P) + dt.timedelta(days=g.FIXTURE_TAIL_DAYS)
    assert f["date"].max() >= g.sim_end(P) + dt.timedelta(days=P.demand.turnaround_days)


def test_world_shape_is_as_documented() -> None:
    assert len(g.REGIONS) == 4 and len(g.LEAGUES) == 8
    assert all(sum(lg.region == r for lg in g.LEAGUES) == 2 for r in g.REGIONS)
    assert all(12 <= lg.size <= 16 and lg.size % 2 == 0 for lg in g.LEAGUES)
    assert len(set(g.CLUB_NAMES)) == len(g.CLUB_NAMES) == sum(lg.size for lg in g.LEAGUES)
    assert 12 <= len(g.SKILL_TYPES) <= 18
    for skill in g.SKILL_TYPES:
        group, region, language = skill.split("-")
        assert group in {"GK", "DEF", "MID", "FWD"}
        assert language in g.REGION_LANGUAGES[region]


def test_each_club_plays_about_once_a_week_with_rest_days(world: g.World) -> None:
    games = club_games(world.fixtures)
    assert not games.duplicated(["club", "date"]).any()
    assert games["gap"].min() >= 2  # never on consecutive days
    span = games["date"].max() - games["date"].min()
    per_week = games.groupby("club").size() / (span.days / 7)
    assert per_week.between(0.9, 1.3).all()
    # Weekend-heavy: most games on Fri-Mon.
    weekday = games["date"].map(lambda d: d.weekday())
    assert weekday.isin([4, 5, 6, 0]).mean() > 0.8


def test_every_region_has_a_summer_style_break_every_year(world: g.World) -> None:
    games = club_games(world.fixtures)
    games["year"] = games["date"].map(lambda d: d.year)
    full_years = games[games["year"].between(2023, 2027)]
    longest = full_years.groupby(["region", "year"])["gap"].max()
    assert longest.between(14, 35).all(), longest


def fx(fid: str, day: int, home: str = "A", away: str = "B") -> Fixture:
    return Fixture(fid, dt.date(2027, 3, day), "L", "Iberia", home, away)


def test_next_fixture_finds_the_first_game_in_the_window() -> None:
    by_club = g.fixtures_by_club([fx("F3", 20), fx("F1", 5), fx("F2", 12, home="C", away="A")])
    assert [f.fixture_id for f in by_club["A"]] == ["F1", "F2", "F3"]
    d = lambda day: dt.date(2027, 3, day)  # noqa: E731
    assert g.next_fixture(by_club, "A", d(6), d(30)).fixture_id == "F2"
    assert g.next_fixture(by_club, "A", d(12), d(12)).fixture_id == "F2"  # bounds inclusive
    assert g.next_fixture(by_club, "A", d(1), d(5)).fixture_id == "F1"
    assert g.next_fixture(by_club, "A", d(13), d(19)) is None
    assert g.next_fixture(by_club, "B", d(6), d(19)) is None  # B is not in F2
    assert g.next_fixture(by_club, "Unknown", d(1), d(30)) is None
    assert g.next_fixture(by_club, "A", d(20), d(10)) is None  # empty window


def test_live_view_window() -> None:
    received = dt.date(2027, 3, 1)
    assert g.live_view_window(received, received + dt.timedelta(days=14)) == (
        dt.date(2027, 3, 3),
        dt.date(2027, 3, 14),
    )


def test_at_risk_share_is_small_but_not_zero(world: g.World) -> None:
    assert 0.02 <= g.at_risk_share(world, "future") <= 0.08
    assert 0.01 <= g.at_risk_share(world, "history") <= 0.10
    req = world.requests
    assert not req.loc[~req["needs_live_view"], "at_risk_day_one"].any()


def test_at_risk_flags_agree_with_a_brute_force_search(world: g.World) -> None:
    games = club_games(world.fixtures)
    dates_by_club = games.groupby("club")["date"].apply(list).to_dict()
    live = world.requests[world.requests["needs_live_view"]]
    sample = pd.concat([live[live["at_risk_day_one"]], live[~live["at_risk_day_one"]].iloc[::20]])
    for row in sample.itertuples():
        lo, hi = g.live_view_window(row.received_date, row.due_date)
        has_game = any(lo <= d <= hi for d in dates_by_club[row.player_club])
        assert has_game != row.at_risk_day_one, row.request_id


# --- requests: attributes -----------------------------------------------------------------


def test_request_attributes_follow_the_parameters(world: g.World) -> None:
    req = world.requests
    d = P.demand
    assert ((req["due_date"] - req["received_date"]).map(lambda x: x.days) == 14).all()
    assert req["desk_hours"].between(*d.desk_hours).all()
    assert req["writeup_hours"].between(*d.writeup_hours).all()
    assert req["desk_hours"].mean() == pytest.approx(np.mean(d.desk_hours), abs=0.1)
    assert req["needs_live_view"].mean() == pytest.approx(d.live_view_share, abs=0.02)
    assert ((req["rework_draw"] >= 0) & (req["rework_draw"] < 1)).all()
    assert req["rework_draw"].mean() == pytest.approx(0.5, abs=0.02)
    club_region = {c.name: c.region for c in g.CLUBS}
    assert (req["skill_type"].map(g.skill_region) == req["player_club"].map(club_region)).all()
    assert (req["client_club"] != req["player_club"]).all()
    assert set(req["client_club"]) <= set(g.CLUB_NAMES)


def test_requests_are_ordered_and_split_into_periods(world: g.World) -> None:
    req = world.requests
    assert req["received_date"].is_monotonic_increasing
    last_history_day = g.SIM_START - dt.timedelta(days=1)
    assert history(world)["received_date"].between(g.history_start(P), last_history_day).all()
    assert future(world)["received_date"].between(g.SIM_START, g.sim_end(P)).all()
    assert history(world)["received_date"].min() < g.add_months(g.history_start(P), 1)


def test_skill_mix_follows_the_weights(world: g.World) -> None:
    share = world.requests["skill_type"].value_counts(normalize=True)
    total = sum(g.SKILL_WEIGHTS.values())
    for skill, weight in g.SKILL_WEIGHTS.items():
        assert share[skill] == pytest.approx(weight / total, abs=0.01)


def test_requests_convert_to_domain_objects(world: g.World) -> None:
    requests = g.requests_from_frame(world.requests)
    assert len(requests) == len(world.requests)
    first = requests[0]
    assert first.request_id == world.requests["request_id"].iloc[0]
    assert first.period == "history" and 0 <= first.rework_draw < 1


def test_request_rework_is_decided_by_the_pre_drawn_number() -> None:
    day = dt.date(2027, 1, 4)
    base = dict(
        request_id="R1", client_club="A", player_club="B", skill_type="MID-Iberia-ES",
        received_date=day, due_date=day, needs_live_view=False, desk_hours=5.0,
        writeup_hours=2.0,
    )  # fmt: skip
    assert not Request(**base).needs_rework(0.6)  # default draw 1.0: never
    r = Request(**base, rework_draw=0.1)
    assert r.needs_rework(0.15) and not r.needs_rework(0.1) and not r.needs_rework(0.05)
    with pytest.raises(ValueError, match="rework_draw"):
        Request(**base, rework_draw=1.5)


# --- demand: seasonality, calibration, growth ------------------------------------------------


def test_seasonal_index_has_mean_one_and_scales_with_strength() -> None:
    for k in (0.0, 0.5, 1.0, 2.0):
        idx = g.seasonal_index(k)
        assert idx.mean() == pytest.approx(1.0)
        assert (idx > 0).all()
    assert (g.seasonal_index(0.0) == 1.0).all()


def test_history_volume_peaks_in_the_transfer_windows(world: g.World) -> None:
    counts = monthly_counts(history(world))
    per_day = counts / counts.index.days_in_month
    by_month = per_day.groupby(per_day.index.month).mean()
    assert set(by_month.nlargest(4).index) == {1, 6, 7, 8}
    # January is a peak of its own (above December and February). It is not
    # necessarily the yearly maximum: organic growth lifts July above it.
    assert by_month[1] > 1.3 * max(by_month[12], by_month[2])


def test_calibration_identity() -> None:
    expected_work = g.base_monthly_volume(P) * g.expected_hours_per_request(P)
    assert expected_work / g.team_available_hours_per_month(P) == pytest.approx(0.70)
    # Default numbers, for orientation: 7 + 0.4 x 8 + 3 = 13.2 h per request.
    assert g.expected_hours_per_request(P) == pytest.approx(13.2)


def test_rate_at_start_is_the_calibrated_base() -> None:
    rate = g.daily_rate(P, np.array([np.datetime64(g.SIM_START, "D")]))[0]
    january = g.seasonal_index(1.0)[0]
    assert rate / january * g.DAYS_PER_YEAR / 12 == pytest.approx(g.base_monthly_volume(P))


@pytest.mark.parametrize("start_load", [0.5, 0.7, 1.0])
def test_generated_team_and_rates_start_at_the_target_load(start_load: float) -> None:
    """D-003 on expectation, using the team as actually generated (leave, freelance draws)."""
    params = apply_overrides(P, {"demand.start_load": start_load})
    w = g.generate_world(params)
    kind = w.scout_weekly_hours["scout_id"].map(w.scouts.set_index("scout_id")["employment"])
    fl_weekly = w.scout_weekly_hours.loc[kind == "freelance", "hours"].mean()
    un = w.scout_unavailability
    leave_days = (un["reason"] == "leave").sum() - un["date"].map(lambda d: d.weekday() >= 5)[
        un["reason"] == "leave"
    ].sum()
    full_time_year = (
        params.team.full_time_count * FULL_TIME_WEEKLY_HOURS * WEEKS_PER_YEAR
        - leave_days * g.FULL_TIME_DAILY_HOURS
    )
    available = (full_time_year + params.team.freelance_count * fl_weekly * WEEKS_PER_YEAR) / 12
    work = g.base_monthly_volume(params) * g.expected_hours_per_request(params)
    assert work / available == pytest.approx(start_load, rel=0.05)


def test_realised_start_load_is_near_the_target(world: g.World) -> None:
    assert g.realised_start_load(world, P) == pytest.approx(0.70, abs=0.07)
    assert g.realised_start_load(world, P, months=3) == pytest.approx(0.70, abs=0.05)


def test_rate_ramps_to_actual_growth_after_one_year() -> None:
    start = np.datetime64(g.SIM_START, "D")
    rates = g.daily_rate(P, np.array([start, start + np.timedelta64(365, "D")]))  # 2 Januaries
    assert rates[1] / rates[0] == pytest.approx(P.demand.actual_growth)


@pytest.mark.parametrize("growth", [1.0, 2.0, 4.0])
def test_realised_growth_matches_the_target(growth: float) -> None:
    w = g.generate_world(apply_overrides(P, {"demand.actual_growth": growth}))
    per_day = deseasonalised_daily(monthly_counts(future(w)))
    ratio = per_day[10:12].mean() / per_day[0:2].mean()  # Nov-Dec vs Jan-Feb
    expected = growth ** (305 / 365)  # the two blocks' midpoints are ~305 days apart
    assert ratio == pytest.approx(expected, rel=0.15)


def test_history_grows_organically(world: g.World) -> None:
    per_day = deseasonalised_daily(monthly_counts(history(world)))
    ratio = per_day[-12:].mean() / per_day[:12].mean()
    assert ratio == pytest.approx((1 + P.demand.history_growth_per_year) ** 3, rel=0.1)


def test_history_and_future_meet_without_a_jump(world: g.World) -> None:
    per_day = deseasonalised_daily(monthly_counts(world.requests))
    last_hist, first_future = per_day[47], per_day[48]
    assert first_future / last_hist == pytest.approx(1.0, abs=0.2)


def test_growth_ceiling_covers_every_allowed_actual_growth() -> None:
    """Thinning needs rate <= ceiling: the ceiling must be the parameter's upper bound."""
    from scout_planner.config import DemandParams

    metadata = DemandParams.model_fields["actual_growth"].metadata
    upper = next(m.le for m in metadata if hasattr(m, "le"))
    assert upper == g.GROWTH_CEILING
