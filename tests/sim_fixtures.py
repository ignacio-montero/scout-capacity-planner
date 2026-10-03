"""Small worlds for the simulation, metrics and pipeline tests (M4).

* :func:`tiny_params`: a generated world small enough to simulate in well
  under a second per replication, but with every mechanism active (live
  views, weekends, leave, freelancers, at-risk requests).
* :func:`hand_inputs`: a fully hand-built world (scouts, fixtures, requests)
  for tests whose expected result is worked out by hand.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from scout_planner import simulate
from scout_planner.config import Params, apply_overrides, load_default_params
from scout_planner.domain import Fixture, Request, Scout
from scout_planner.generate import sim_week_starts

TINY: dict[str, Any] = {
    "sim.months": 2,
    "sim.seeds": 2,
    "team.full_time_count": 8,
    "team.freelance_count": 4,
    "team.follow_hiring_plan": False,
    "demand.start_load": 0.3,  # of the 40-scout reference team: ~12 scouts' worth of work
    "demand.actual_growth": 2.0,
    "assignment.policy": "edf",
}


def tiny_params(**overrides: Any) -> Params:
    """``TINY`` on top of the defaults, then ``overrides`` (dotted keys with ``__`` for dots)."""
    extra = {k.replace("__", "."): v for k, v in overrides.items()}
    return apply_overrides(load_default_params(), {**TINY, **extra})


def run_world(params: Params, replication: int = 0, *, trace: bool = True):
    """One replication of a generated world without hires (``follow_hiring_plan`` is off)."""
    world = simulate.replication_world(params, replication)
    inputs = simulate.prepare_replication(params, world, pd.DataFrame(), replication)
    return simulate.simulate_replication(inputs, trace=trace)


# --- hand-built worlds --------------------------------------------------------------

SKILL = "MID-Iberia-ES"
D = dt.date


def scout(
    sid: str,
    employment: str = "full_time",
    *,
    region: str = "Iberia",
    skills: Iterable[str] = (SKILL,),
    former: Iterable[str] = (),
    joined_month: int = 0,
    rate: float = 40.0,
) -> Scout:
    ft = employment == "full_time"
    return Scout(
        scout_id=sid,
        name=f"Scout {sid}",
        employment=employment,  # type: ignore[arg-type]
        skills=frozenset(skills),
        home_region=region,
        min_weekly_hours=37.5 if ft else 8.0,
        max_weekly_hours=37.5 if ft else 25.0,
        former_clubs=frozenset(former),
        monthly_salary=4500.0 if ft else None,
        hourly_rate=None if ft else rate,
        joined_month=joined_month,
    )


def request(
    rid: str,
    received: dt.date,
    *,
    client: str = "Club Client",
    player: str = "Club Player",
    skill: str = SKILL,
    live: bool = False,
    desk: float = 4.0,
    writeup: float = 2.0,
    turnaround: int = 14,
    rework_draw: float = 1.0,
) -> Request:
    return Request(
        request_id=rid,
        client_club=client,
        player_club=player,
        skill_type=skill,
        received_date=received,
        due_date=received + dt.timedelta(days=turnaround),
        needs_live_view=live,
        desk_hours=desk,
        writeup_hours=writeup,
        rework_draw=rework_draw,
    )


def fixture(fid: str, date: dt.date, home: str = "Club Player", region: str = "Iberia") -> Fixture:
    return Fixture(fid, date, "Liga Costera", region, home, "Club Away")


def hand_inputs(
    params: Params,
    scouts: list[Scout],
    requests: list[Request],
    fixtures: list[Fixture] = (),  # type: ignore[assignment]
    *,
    freelance_weekly_hours: float = 20.0,
    unavailability: Mapping[str, Iterable[dt.date]] | None = None,
    at_risk: Mapping[str, bool] | None = None,
    replication: int = 0,
) -> simulate.SimInputs:
    weeks = sim_week_starts(params)
    weekly = {
        (s.scout_id, w): (freelance_weekly_hours if s.is_freelance else 37.5)
        for s in scouts
        for w in weeks
    }
    return simulate.SimInputs(
        params=params,
        replication=replication,
        scouts=tuple(scouts),
        unavailability={k: frozenset(v) for k, v in (unavailability or {}).items()},
        weekly_hours=weekly,
        fixtures=tuple(fixtures),
        requests=tuple(requests),
        at_risk=dict(at_risk or {}),
    )


def hand_params(**overrides: Any) -> Params:
    """One simulated month (January 2027), EDF, defaults otherwise."""
    extra = {k.replace("__", "."): v for k, v in overrides.items()}
    base = {"sim.months": 1, "sim.seeds": 1, "assignment.policy": "edf"}
    return apply_overrides(load_default_params(), {**base, **extra})
