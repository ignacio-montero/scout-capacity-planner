"""Stage [1]: the synthetic world (scouts, fixtures, requests). Spec: DATA_CONTRACTS section 1.

``generate_world(params)`` is pure: same parameters in, equal tables out, no
file access. ``write_world`` / ``read_world`` are the thin I/O edge (parquet
under ``<run>/raw/``), in the same spirit as ``config.load_params``.

What is a parameter and what is fixed
-------------------------------------
Numbers that express a business judgement are parameters (``Params``). The
world's *shape* is fixed here as module constants on purpose (PARAMETERS.md,
"Not parameters"): 4 invented regions, 2 invented leagues per region, the
skill-type list and its demand mix, the seasonality profile, the league
calendars and the name lists.

Modelling choices
-----------------
* **Calendar.** The simulated period starts on ``SIM_START`` (2027-01-01) and
  lasts ``sim.months`` months (default 12). The history is the
  ``demand.history_months`` months before it (default 48: 2023-01..2026-12).
* **Demand rate.** Requests arrive as a Poisson process, one draw per day:
  ``rate(day) = base_daily x season[month] x growth(day)``. ``season`` is a
  fixed monthly profile with mean 1 (peaks in January and June-August, the
  transfer windows), stretched by ``demand.seasonality_strength``. ``growth``
  is ``(1 + history_growth_per_year) ** t`` in the history (``t`` in years,
  negative) and ``actual_growth ** t`` in the simulated period (D-014), so
  the run-rate is continuous at ``SIM_START`` and is exactly
  ``actual_growth`` times higher one year later (an exponential ramp).
* **Calibration against a fixed reference world (D-003).**
  ``base_monthly_volume`` is the deseasonalised monthly run-rate at
  ``SIM_START``. It is chosen so that the *reference* team, the code defaults
  in ``REFERENCE`` (24 full-time + 16 freelancers, default leave, hours and
  live-view share), would be ``demand.start_load`` busy: expected work =
  ``start_load`` x reference available hours, where available = full-time
  37.5 h/week net of leave + freelancers' mean weekly hours, and work per
  request = mean desk + live share x live hours + mean write-up (automation
  is ignored: demand does not depend on tools). Only ``start_load`` comes
  from the run. Why not the run's own team: then "+10 freelancers" or "fewer
  live views" would rescale demand to keep the load fixed, and those what-ifs
  would answer nothing. With a fixed reference they change the *load*, the
  arrival stream stays identical, and ``realised_start_load`` reports the
  run's own team load (below ``start_load`` for a bigger team).
* **Request attributes.** Skill type from a fixed demand mix; player club
  uniform among the clubs of the skill's region; client club uniform among
  all other clubs; desk and write-up hours uniform on their ranges;
  ``needs_live_view`` with probability ``live_view_share``, except that
  urgent requests never need one (an express report is video-based), so the
  effective share is ``live_view_share x (1 - urgent_share)``
  (``effective_live_view_share``); a uniform
  ``rework_draw`` pre-drawn per request (rework iff ``rework_draw <
  rework_rate``); ``urgent`` with probability ``urgent_share``.
  ``due_date = received + urgent_turnaround_days`` for urgent requests and
  ``received + turnaround_days`` otherwise. Mixed deadlines are what make
  earliest-due-first differ from first-come-first-served.
* **At risk from day one.** A live view must happen in
  ``[received + 2, due - 1]`` (``live_view_window``, on the request's own due
  date): two days to arrange it, one day to write it up. A live-view request
  whose player club has no fixture in that window is flagged
  ``at_risk_day_one``. Urgent requests have no live view, so are never at risk.
* **Fixtures.** Each league plays a repeating double round-robin: one round
  per weekend (Fri-Mon, mostly Sat/Sun) plus a midweek round every few weeks,
  except during its region's breaks (a ~3-week summer break and a short winter
  break, staggered by region; the South American calendar has its long break
  over the southern summer). Breaks are what create at-risk requests.
* **Scouts are stable entities.** Ids are ``FT001``.. and ``FL001``.., and
  each scout draws everything from its own streams keyed by id
  (``scout_stream``): home region (from the demand mix), skills (mostly
  home-region), name, 0-3 former clubs (biased to the home region), leave and
  weekly hours. So adding scouts never changes an existing one. Two
  deterministic team-level repairs run last and only act when needed: every
  region gets at least 2 residents (teams of 8+), and every skill type gets at
  least 2 holders living in its region, which live views need (D-008). Skill
  repair may push a scout above ``skills_per_scout``'s max only if no other
  home-region scout has room.
* **Availability.** Full-time: 37.5 h every week; leave of
  ``leave_days_per_year`` weekdays (pro rata for the horizon) in blocks of
  10/5/5/3/2 days placed at random. Freelancers: weekly hours pre-drawn per
  scout per week from ``freelance_weekly_hours`` (D-015, table
  ``scout_weekly_hours``) and a few random days off. Days off constrain *which*
  days someone can work (notably live views); a freelancer's weekly hours are
  what they offer for the week.

Random streams and common random numbers (D-015)
------------------------------------------------
Each purpose draws from its own named stream (``rng.make_streams``), and each
request-level stream is split by period (``arrivals.history``,
``arrivals.future``...). Per-entity draws use a fixed number of uniforms per
entity, so a longer array only *appends* numbers. Consequences, all tested:

* Scouts and fixtures depend on no demand parameter; the history does not
  depend on ``actual_growth`` or ``sim.months``.
* ``assignment.*``, ``automation.*``, ``capacity_plan.*`` and ``cost.*`` do not
  change any random draw. (``cost.*`` only changes the salary/rate columns of
  ``scouts``, which echo it.)
* **Thinning (D-019).** Arrivals are drawn at a ceiling rate
  (``ceiling_daily_rate``: the maximum start load, the maximum seasonal
  multiplier of each month, and from ``SIM_START`` on the maximum growth)
  and each candidate is kept with probability ``rate / ceiling``, using its
  own pre-drawn uniform (Lewis-Shedler thinning). So a 2x world's requests
  are a subset of a 4x world's, a 0.70 start load's of a 0.75 one's, and a
  stronger seasonality adds requests in peak months and removes some in
  quiet ones: runs share their common requests instead of drawing unrelated
  ones.
* Team size, hours and live-view share do not change the arrival stream at
  all (reference calibration); urgency and live-view flags are nested across
  their shares; per-scout streams make every existing scout identical when
  the team grows.
"""

from __future__ import annotations

import bisect
import dataclasses
import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scout_planner.config import FULL_TIME_WEEKLY_HOURS, WEEKS_PER_YEAR, Params
from scout_planner.domain import Fixture, Request, Scout
from scout_planner.rng import make_stream, make_streams

# --- calendar ----------------------------------------------------------------------

SIM_START = dt.date(2027, 1, 1)
DAYS_PER_YEAR = 365.0  # growth exponent unit: t = days since SIM_START / 365
MAX_HISTORY_MONTHS = 72  # upper bound of demand.history_months
MAX_SIM_MONTHS = 24  # upper bound of sim.months
# Fixtures run past the end of the simulated period so that requests received
# on its last days still have fixtures before their due date (turnaround <= 28).
FIXTURE_TAIL_DAYS = 35
# A live view must happen in [received + LEAD, due - TAIL] (see live_view_window).
LIVE_VIEW_MIN_LEAD_DAYS = 2
LIVE_VIEW_MIN_TAIL_DAYS = 1


def add_months(day: dt.date, months: int) -> dt.date:
    """First of the month ``months`` after (or before, if negative) ``day``'s month."""
    index = day.year * 12 + (day.month - 1) + months
    return dt.date(index // 12, index % 12 + 1, 1)


def sim_end(params: Params) -> dt.date:
    """Last day of the simulated period (2027-12-31 by default)."""
    return add_months(SIM_START, params.sim.months) - dt.timedelta(days=1)


def history_start(params: Params) -> dt.date:
    """First day of the history (2023-01-01 by default)."""
    return add_months(SIM_START, -params.demand.history_months)


def live_view_window(received: dt.date, due: dt.date) -> tuple[dt.date, dt.date]:
    """Dates on which a live view for this request is useful, both ends included."""
    return (
        received + dt.timedelta(days=LIVE_VIEW_MIN_LEAD_DAYS),
        due - dt.timedelta(days=LIVE_VIEW_MIN_TAIL_DAYS),
    )


# --- world shape (fixed on purpose: not parameters) ---------------------------------

REGION_LANGUAGES: dict[str, tuple[str, ...]] = {
    "Iberia": ("ES", "PT"),
    "NorthernEurope": ("EN",),
    "CentralEurope": ("DE",),
    "SouthAmerica": ("ES", "PT"),
}
REGIONS: tuple[str, ...] = tuple(REGION_LANGUAGES)

# Skill types actually used (position group x region x language), with their
# relative share of demand. Insertion order is the canonical skill order.
SKILL_WEIGHTS: dict[str, float] = {
    "GK-Iberia-ES": 3,
    "DEF-Iberia-ES": 7,
    "MID-Iberia-ES": 9,
    "FWD-Iberia-ES": 9,
    "MID-Iberia-PT": 4,
    "GK-NorthernEurope-EN": 3,
    "DEF-NorthernEurope-EN": 7,
    "MID-NorthernEurope-EN": 9,
    "FWD-NorthernEurope-EN": 8,
    "DEF-CentralEurope-DE": 6,
    "MID-CentralEurope-DE": 7,
    "FWD-CentralEurope-DE": 6,
    "DEF-SouthAmerica-ES": 5,
    "MID-SouthAmerica-PT": 5,
    "FWD-SouthAmerica-ES": 8,
}
SKILL_TYPES: tuple[str, ...] = tuple(SKILL_WEIGHTS)


def skill_region(skill_type: str) -> str:
    """``"MID-Iberia-ES" -> "Iberia"``."""
    return skill_type.split("-")[1]


REGION_SKILLS: dict[str, tuple[str, ...]] = {
    r: tuple(s for s in SKILL_TYPES if skill_region(s) == r) for r in REGIONS
}
REGION_DEMAND_SHARE: dict[str, float] = {
    r: sum(SKILL_WEIGHTS[s] for s in REGION_SKILLS[r]) / sum(SKILL_WEIGHTS.values())
    for r in REGIONS
}

# Monthly demand profile, Jan..Dec, mean exactly 1: transfer windows in January
# and June-August. seasonality_strength k gives 1 + k * (profile - 1).
SEASONALITY_PROFILE: tuple[float, ...] = (
    1.35, 0.95, 0.85, 0.85, 0.95, 1.20, 1.30, 1.25, 0.90, 0.80, 0.80, 0.80,
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class LeagueSpec:
    """An invented league: name, region, number of clubs (even, for round-robin)."""

    name: str
    region: str
    size: int


LEAGUES: tuple[LeagueSpec, ...] = (
    LeagueSpec("Liga Costera", "Iberia", 16),
    LeagueSpec("Liga Mesetaria", "Iberia", 12),
    LeagueSpec("Fjordline League", "NorthernEurope", 14),
    LeagueSpec("Moorland League", "NorthernEurope", 14),
    LeagueSpec("Alpenrand Liga", "CentralEurope", 16),
    LeagueSpec("Waldmark Liga", "CentralEurope", 12),
    LeagueSpec("Liga Solaris", "SouthAmerica", 14),
    LeagueSpec("Liga Pampera", "SouthAmerica", 14),
)

# Invented town names; each gives two clubs (two name templates), so 14 towns
# make the 28 clubs of a region's two leagues.
_TOWNS: dict[str, tuple[str, ...]] = {
    "Iberia": (
        "Valdoro", "Puerto Azul", "Castrelo", "Montesal", "Sierra Clara", "Alcaren",
        "Villaserena", "Portomar", "Costaluz", "Robledal", "Penaclara", "Arenosa",
        "Lagomar", "Ribaverde",
    ),
    "NorthernEurope": (
        "Ashmere", "Kestrel Bay", "Highmoor", "Thornwick", "Greyhaven", "Stormholt",
        "Fenwick Vale", "Bramblecombe", "Larkspur", "Hollin Sound", "Oakhollow",
        "Saltmarsh", "Wyndford", "Elderby",
    ),
    "CentralEurope": (
        "Rabenstadt", "Auenfeld", "Felsburg", "Lindenhain", "Silberbach", "Sternau",
        "Hagenwalde", "Moosbrunn", "Kaltenhof", "Tannenried", "Wolkenau", "Ebersfeld",
        "Grauweiler", "Birkenstein",
    ),
    "SouthAmerica": (
        "Rio Celeste", "San Albor", "Puerto Sereno", "Villa Ocaso", "Santa Brisa",
        "Campo Dorado", "Itaverde", "Serra Alta", "Bela Aurora", "Lagoa Clara",
        "Cerro Ventura", "Quebrada Sol", "Pampa Verde", "Vale do Sino",
    ),
}  # fmt: skip
_CLUB_TEMPLATES: dict[str, tuple[str, ...]] = {
    "Iberia": ("{} CF", "Atletico {}", "{} SC", "Union {}", "Deportivo {}", "Racing {}"),
    "NorthernEurope": ("{} FC", "{} United", "{} Town", "{} Rovers", "{} Athletic", "{} City"),
    "CentralEurope": ("FC {}", "SV {}", "TSV {}", "SC {}", "SpVgg {}", "Union {}"),
    "SouthAmerica": ("Club {}", "{} FC", "Deportivo {}", "Sporting {}", "{} EC", "Atletico {}"),
}

# Breaks with no weekend round, per region: (month, day, length in days). Any
# round whose anchor day falls in [start, start + length) is skipped.
SEASON_BREAKS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "Iberia": ((6, 14, 14), (12, 23, 10)),
    "NorthernEurope": ((6, 28, 14), (1, 8, 7)),
    "CentralEurope": ((6, 7, 14), (12, 20, 14)),
    "SouthAmerica": ((12, 13, 21), (7, 4, 14)),  # long break in the southern summer
}
# Weekend round anchored on Saturday; fixtures spread Fri..Mon.
WEEKEND_DAY_OFFSETS = (-1, 0, 1, 2)
WEEKEND_DAY_WEIGHTS = (0.10, 0.45, 0.40, 0.05)
# Midweek round: all on Wednesday, so a club always has 2+ days between games
# (a Monday weekend game and a Tuesday midweek game would be back to back).
MIDWEEK_DAY_OFFSETS = (0,)
MIDWEEK_DAY_WEIGHTS = (1.0,)
MIDWEEK_ROUND_EVERY_WEEKS = 5

# Scouts.
HOME_SKILL_SHARE = 0.85  # chance each skill pick comes from the home region
MIN_HOME_REGION_HOLDERS = 2  # per skill type: holders living in its region (D-008)
FORMER_CLUB_COUNT_WEIGHTS = (0.35, 0.35, 0.20, 0.10)  # P(0), P(1), P(2), P(3)
FORMER_CLUB_HOME_SHARE = 0.8
FULL_TIME_DAILY_HOURS = FULL_TIME_WEEKLY_HOURS / 5  # one leave day costs this
LEAVE_BLOCK_PATTERN = (10, 5, 5, 3, 2)  # leave is taken in blocks of these sizes
FREELANCE_DAYS_OFF_PER_YEAR = 6.0  # Poisson mean, single days
FREELANCE_HOURS_STEP = 0.5  # weekly offers are rounded to half hours

# Thinning ceilings (D-019): arrivals are drawn at the highest rate any
# parameter value allows and thinned down to the run's rate. Each equals the
# upper bound of its parameter (tests pin them together).
GROWTH_CEILING = 6.0  # demand.actual_growth
START_LOAD_CEILING = 1.2  # demand.start_load
SEASONALITY_STRENGTH_CEILING = 2.0  # demand.seasonality_strength

_FIRST_NAMES: dict[str, tuple[str, ...]] = {
    "Iberia": (
        "Ana", "Lucia", "Marta", "Ines", "Carmen", "Sofia", "Javier", "Diego",
        "Pablo", "Alvaro", "Tiago", "Rui", "Beatriz", "Joana", "Nuno", "Hugo",
    ),
    "NorthernEurope": (
        "Oliver", "Freya", "Harriet", "James", "Callum", "Isla", "Niamh", "Rhys",
        "Erik", "Ingrid", "Lars", "Maja", "Tom", "Ellie", "Sven", "Astrid",
    ),
    "CentralEurope": (
        "Lukas", "Jonas", "Leonie", "Hanna", "Felix", "Mia", "Tobias", "Katrin",
        "Matthias", "Lena", "Stefan", "Julia", "Paul", "Clara", "Moritz", "Anja",
    ),
    "SouthAmerica": (
        "Mateo", "Valentina", "Santiago", "Camila", "Joaquin", "Martina", "Thiago",
        "Gabriela", "Facundo", "Larissa", "Rafael", "Bruna", "Emiliano", "Florencia",
        "Caio", "Renata",
    ),
}  # fmt: skip
_SURNAMES: dict[str, tuple[str, ...]] = {
    "Iberia": (
        "Velmar", "Ortaza", "Brinedo", "Calvanes", "Mirantes", "Soterra", "Quiroval",
        "Albadin", "Teixal", "Corvelo", "Paredal", "Lunares",
    ),
    "NorthernEurope": (
        "Brackwell", "Dunmoor", "Whitlowe", "Kestridge", "Selwyth", "Marlowby",
        "Stanmere", "Hollingrave", "Pennick", "Thursby", "Ravensholt", "Calderby",
    ),
    "CentralEurope": (
        "Kellerhof", "Wendtner", "Falkenrath", "Steinwald", "Grubenbach", "Riedmann",
        "Tannberg", "Wolkenauer", "Ebersberger", "Lindtauer", "Haselgruber", "Morsbacher",
    ),
    "SouthAmerica": (
        "Arandil", "Quintebar", "Moraviel", "Cardonel", "Itabera", "Villaturo",
        "Coroval", "Iturrame", "Sanzaro", "Pirapema", "Tabarel", "Ucheda",
    ),
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class Club:
    """An invented club and where it plays."""

    name: str
    league: str
    region: str


def _build_clubs() -> tuple[Club, ...]:
    clubs: list[Club] = []
    for region in REGIONS:
        towns, templates = _TOWNS[region], _CLUB_TEMPLATES[region]
        # First every town's "A" club, then every town's "B" club, so the first
        # league holds one club per town plus a few city rivals.
        names = [templates[(2 * i) % len(templates)].format(t) for i, t in enumerate(towns)]
        names += [templates[(2 * i + 1) % len(templates)].format(t) for i, t in enumerate(towns)]
        leagues = [lg for lg in LEAGUES if lg.region == region]
        if sum(lg.size for lg in leagues) != len(names):
            raise AssertionError(f"club count mismatch in {region}")  # pragma: no cover
        start = 0
        for lg in leagues:
            clubs += [Club(n, lg.name, region) for n in names[start : start + lg.size]]
            start += lg.size
    return tuple(clubs)


CLUBS: tuple[Club, ...] = _build_clubs()
CLUB_NAMES: tuple[str, ...] = tuple(c.name for c in CLUBS)
REGION_CLUBS: dict[str, tuple[str, ...]] = {
    r: tuple(c.name for c in CLUBS if c.region == r) for r in REGIONS
}

# --- table schemas (the contract, as code) -----------------------------------------

TABLE_SCHEMAS: dict[str, pa.Schema] = {
    "scouts": pa.schema(
        [
            ("scout_id", pa.string()),
            ("name", pa.string()),
            ("employment", pa.string()),
            ("skills", pa.list_(pa.string())),
            ("home_region", pa.string()),
            ("min_weekly_hours", pa.float64()),
            ("max_weekly_hours", pa.float64()),
            ("former_clubs", pa.list_(pa.string())),
            ("monthly_salary", pa.float64()),
            ("hourly_rate", pa.float64()),
            ("joined_month", pa.int64()),
        ]
    ),
    "scout_unavailability": pa.schema(
        [("scout_id", pa.string()), ("date", pa.date32()), ("reason", pa.string())]
    ),
    "scout_weekly_hours": pa.schema(
        [("scout_id", pa.string()), ("week_start", pa.date32()), ("hours", pa.float64())]
    ),
    "fixtures": pa.schema(
        [
            ("fixture_id", pa.string()),
            ("date", pa.date32()),
            ("league", pa.string()),
            ("region", pa.string()),
            ("home_club", pa.string()),
            ("away_club", pa.string()),
        ]
    ),
    "requests": pa.schema(
        [
            ("request_id", pa.string()),
            ("client_club", pa.string()),
            ("player_club", pa.string()),
            ("skill_type", pa.string()),
            ("received_date", pa.date32()),
            ("due_date", pa.date32()),
            ("urgent", pa.bool_()),
            ("needs_live_view", pa.bool_()),
            ("desk_hours", pa.float64()),
            ("writeup_hours", pa.float64()),
            ("period", pa.string()),
            ("rework_draw", pa.float64()),
            ("at_risk_day_one", pa.bool_()),
        ]
    ),
}


def _pandas_dtype(arrow_type: pa.DataType) -> Any:
    """In-memory dtype for an arrow column: what ``read_parquet`` gives back."""
    if pa.types.is_string(arrow_type):
        return "str"
    if pa.types.is_float64(arrow_type):
        return "float64"
    if pa.types.is_int64(arrow_type):
        return "int64"
    if pa.types.is_boolean(arrow_type):
        return "bool"
    return object  # dates (datetime.date) and lists (list[str])


def _frame(table: str, columns: Mapping[str, Sequence[Any] | np.ndarray]) -> pd.DataFrame:
    """Build a table with exactly the contract's columns, in order, with stable dtypes."""
    schema = TABLE_SCHEMAS[table]
    if list(columns) != schema.names:
        raise ValueError(f"{table}: columns {list(columns)} != contract {schema.names}")
    data = {}
    for f in schema:
        values = columns[f.name]
        dtype = _pandas_dtype(f.type)
        if dtype is object:
            series = pd.Series(list(values), dtype=object)
        else:
            series = pd.Series(values, dtype=dtype)
        data[f.name] = series
    return pd.DataFrame(data)


@dataclass(frozen=True, eq=False)
class World:
    """Every table of stage [1]. Frozen (no attribute is reassigned); the frames
    themselves are mutable pandas objects, so treat them as read-only."""

    scouts: pd.DataFrame
    scout_unavailability: pd.DataFrame
    scout_weekly_hours: pd.DataFrame
    fixtures: pd.DataFrame
    requests: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        """``{table name: frame}`` in contract order."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


# --- demand: rates and calibration (pure, no randomness) --------------------------


def seasonal_index(strength: float) -> np.ndarray:
    """Monthly multipliers Jan..Dec with mean 1; ``strength=0`` is a flat year."""
    profile = np.asarray(SEASONALITY_PROFILE)
    return 1.0 + strength * (profile - 1.0)


def team_available_hours_per_month(params: Params) -> float:
    """Expected hours the initial team offers per average month (D-003).

    Full-time: 37.5 h x 52 weeks minus leave days x 7.5 h. Freelance: the mean
    of the weekly range x 52 weeks.
    """
    team = params.team
    full_time_year = (
        FULL_TIME_WEEKLY_HOURS * WEEKS_PER_YEAR - team.leave_days_per_year * FULL_TIME_DAILY_HOURS
    )
    freelance_year = float(np.mean(team.freelance_weekly_hours)) * WEEKS_PER_YEAR
    return (team.full_time_count * full_time_year + team.freelance_count * freelance_year) / 12


def effective_live_view_share(params: Params) -> float:
    """Share of requests that really need a live view: urgent ones never do.

    The live-view and urgency draws are independent uniforms, so this is
    exactly ``live_view_share x (1 - urgent_share)`` (0.34 by default).
    """
    d = params.demand
    return d.live_view_share * (1.0 - d.urgent_share)


def expected_hours_per_request(params: Params) -> float:
    """Mean work per request without automation: desk + effective live share x live + write-up.

    Used on ``REFERENCE`` for the calibration, so the reference expectation
    includes the default urgent share and ``start_load`` keeps meaning the
    default team's average load.
    """
    d = params.demand
    return (
        float(np.mean(d.desk_hours))
        + effective_live_view_share(params) * d.live_view_hours
        + float(np.mean(d.writeup_hours))
    )


# The fixed reference world the demand volume is calibrated against: the code
# defaults. Only ``demand.start_load`` is taken from the run.
REFERENCE = Params()


def base_monthly_volume(params: Params) -> float:
    """Deseasonalised requests per month at ``SIM_START`` (D-003 calibration).

    Chosen so that ``base x expected_hours_per_request(REFERENCE) = start_load x
    team_available_hours_per_month(REFERENCE)``: the *reference* (default) team
    starts the year ``start_load`` busy. Team size, hours and live-view share
    of the run deliberately do not enter, so changing them changes the load,
    not the demand.
    """
    return (
        params.demand.start_load
        * team_available_hours_per_month(REFERENCE)
        / expected_hours_per_request(REFERENCE)
    )


def _years_since_start(days: np.ndarray) -> np.ndarray:
    return (days - np.datetime64(SIM_START, "D")).astype(np.int64) / DAYS_PER_YEAR


def daily_rate(params: Params, days: np.ndarray, *, growth: float | None = None) -> np.ndarray:
    """Expected requests per day for each ``datetime64[D]`` day.

    Days before ``SIM_START`` follow the history's organic growth; days from it
    on follow ``growth`` (default ``demand.actual_growth``) per year.
    """
    base_daily = base_monthly_volume(params) * 12 / DAYS_PER_YEAR
    return base_daily * _demand_multiplier(params, days, growth=growth)


def _demand_multiplier(
    params: Params, days: np.ndarray, *, growth: float | None = None
) -> np.ndarray:
    """Seasonality x trend per day: the rate relative to the run-rate at ``SIM_START``."""
    d = params.demand
    g = d.actual_growth if growth is None else growth
    days = np.asarray(days, dtype="datetime64[D]")
    t = _years_since_start(days)
    month = days.astype("datetime64[M]").astype(np.int64) % 12
    trend = np.where(t < 0, (1.0 + d.history_growth_per_year) ** t, g**t)
    return seasonal_index(d.seasonality_strength)[month] * trend


def ceiling_daily_rate(params: Params, days: np.ndarray) -> np.ndarray:
    """The thinning envelope: an upper bound of ``daily_rate`` over every value of
    ``start_load``, ``seasonality_strength`` and (from ``SIM_START`` on)
    ``actual_growth``, with the run's other parameters.

    The seasonal multiplier ``1 + k (profile - 1)`` is linear in the strength
    ``k``, so its maximum over ``[0, 2]`` sits at an end: ``max(1, 2 profile - 1)``.
    """
    d = params.demand
    days = np.asarray(days, dtype="datetime64[D]")
    t = _years_since_start(days)
    month = days.astype("datetime64[M]").astype(np.int64) % 12
    season = np.maximum(seasonal_index(0.0), seasonal_index(SEASONALITY_STRENGTH_CEILING))
    trend = np.where(t < 0, (1.0 + d.history_growth_per_year) ** t, GROWTH_CEILING**t)
    base_daily = base_monthly_volume(params) / d.start_load * START_LOAD_CEILING
    return base_daily * 12 / DAYS_PER_YEAR * season[month] * trend


def _day_range(start: dt.date, end: dt.date) -> np.ndarray:
    """``datetime64[D]`` days from ``start`` to ``end``, both included."""
    return np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + np.timedelta64(1, "D"))


# --- fixtures ----------------------------------------------------------------------


def _in_break(region: str, day: dt.date) -> bool:
    for month, start_day, length in SEASON_BREAKS[region]:
        for year in (day.year - 1, day.year):  # a break may straddle New Year
            start = dt.date(year, month, start_day)
            if start <= day < start + dt.timedelta(days=length):
                return True
    return False


def _double_round_robin(n: int) -> list[list[tuple[int, int]]]:
    """Rounds of (home, away) index pairs: everyone meets everyone home and away.

    The *circle method*: fix team 0, rotate the others one place per round.
    """
    order = list(range(n))
    first_half: list[list[tuple[int, int]]] = []
    for r in range(n - 1):
        pairs = [(order[i], order[n - 1 - i]) for i in range(n // 2)]
        # Alternate home advantage by round so nobody is always at home.
        first_half.append([(a, b) if r % 2 == 0 else (b, a) for a, b in pairs])
        order = [order[0], order[-1], *order[1:-1]]
    return first_half + [[(b, a) for a, b in rnd] for rnd in first_half]


FixtureRow = tuple[dt.date, str, str, str, str]  # date, league, region, home, away


def _round_anchors(region: str, first_monday: dt.date, n_weeks: int) -> list[tuple[dt.date, bool]]:
    """``(anchor day, is_midweek)`` of every round a league in ``region`` plays, in order.

    A weekend round every week whose Saturday is not in a break; a Wednesday
    round every ``MIDWEEK_ROUND_EVERY_WEEKS`` weeks when the weekends either
    side of it are played too (no midweek games right before or after a break).
    """
    anchors: list[tuple[dt.date, bool]] = []
    for week in range(n_weeks):
        monday = first_monday + dt.timedelta(weeks=week)
        saturday, wednesday = monday + dt.timedelta(days=5), monday + dt.timedelta(days=2)
        if week % MIDWEEK_ROUND_EVERY_WEEKS == 2 and not any(
            _in_break(region, d) for d in (wednesday, saturday, saturday - dt.timedelta(days=7))
        ):
            anchors.append((wednesday, True))
        if not _in_break(region, saturday):
            anchors.append((saturday, False))
    return anchors


def _fixture_calendar(rng: np.random.Generator) -> list[FixtureRow]:
    """Every fixture from a fixed epoch to the longest possible horizon.

    The span does not depend on any parameter, so a fixture's date and clubs
    never change with ``history_months`` or ``sim.months``; callers filter it.
    """
    epoch = add_months(SIM_START, -MAX_HISTORY_MONTHS)
    first_monday = epoch - dt.timedelta(days=epoch.weekday())
    last_day = add_months(SIM_START, MAX_SIM_MONTHS) + dt.timedelta(days=FIXTURE_TAIL_DAYS)
    n_weeks = (last_day - first_monday).days // 7 + 1
    styles = {
        False: (WEEKEND_DAY_OFFSETS, np.cumsum(WEEKEND_DAY_WEIGHTS)),
        True: (MIDWEEK_DAY_OFFSETS, np.cumsum(MIDWEEK_DAY_WEIGHTS)),
    }
    out: list[FixtureRow] = []
    for league in LEAGUES:
        clubs = [c.name for c in CLUBS if c.league == league.name]
        clubs = [clubs[i] for i in rng.permutation(len(clubs))]
        rounds = _double_round_robin(len(clubs))
        anchors = _round_anchors(league.region, first_monday, n_weeks)
        for k, (anchor, midweek) in enumerate(anchors):
            pairs = rounds[k % len(rounds)]  # the round-robin simply repeats
            offsets, cdf = styles[midweek]
            picks = np.searchsorted(cdf, rng.random(len(pairs)) * cdf[-1], side="right")
            for (home, away), pick in zip(pairs, picks, strict=True):
                day = anchor + dt.timedelta(days=offsets[int(pick)])
                out.append((day, league.name, league.region, clubs[home], clubs[away]))
    return out


def _id_width(n: int, minimum: int) -> int:
    return max(minimum, len(str(n)))


def _generate_fixtures(params: Params, rng: np.random.Generator) -> pd.DataFrame:
    first = history_start(params)
    last = sim_end(params) + dt.timedelta(days=FIXTURE_TAIL_DAYS)
    rows = sorted(r for r in _fixture_calendar(rng) if first <= r[0] <= last)
    width = _id_width(len(rows), 5)
    return _frame(
        "fixtures",
        {
            "fixture_id": [f"F{i:0{width}d}" for i in range(1, len(rows) + 1)],
            "date": [r[0] for r in rows],
            "league": [r[1] for r in rows],
            "region": [r[2] for r in rows],
            "home_club": [r[3] for r in rows],
            "away_club": [r[4] for r in rows],
        },
    )


def fixtures_by_club(fixtures: Iterable[Fixture]) -> dict[str, tuple[Fixture, ...]]:
    """Index fixtures by club (home or away), each club's tuple sorted by date."""
    index: dict[str, list[Fixture]] = {}
    for f in fixtures:
        index.setdefault(f.home_club, []).append(f)
        index.setdefault(f.away_club, []).append(f)
    return {
        club: tuple(sorted(fs, key=lambda f: (f.date, f.fixture_id))) for club, fs in index.items()
    }


def next_fixture(
    by_club: Mapping[str, Sequence[Fixture]], club: str, earliest: dt.date, latest: dt.date
) -> Fixture | None:
    """The first fixture of ``club`` dated in ``[earliest, latest]``, or ``None``.

    ``by_club`` comes from :func:`fixtures_by_club` (sorted by date), so this is
    a binary search, cheap enough to call for every request every day.
    """
    if latest < earliest:
        return None
    games = by_club.get(club, ())
    i = bisect.bisect_left(games, earliest, key=lambda f: f.date)
    if i < len(games) and games[i].date <= latest:
        return games[i]
    return None


# --- scouts ------------------------------------------------------------------------


def _pick_weighted(items: Sequence[str], weights: Sequence[float], u: float) -> str:
    """Inverse-CDF pick of one item from a uniform ``u`` in [0, 1)."""
    cdf = np.cumsum(weights)
    return items[int(np.searchsorted(cdf, u * cdf[-1], side="right"))]


def _repair_home_regions(homes: list[str], priority: Sequence[float]) -> None:
    """Give every region at least 2 resident scouts when the team has 8 or more.

    Home regions are drawn per scout, so a small region can come up short by
    chance (rare for 40 scouts). The fix moves the scout with the highest
    ``priority`` out of the most populated region; deterministic, and it
    touches nobody when no region is short. Mutates ``homes`` in place.
    """
    if len(homes) < MIN_HOME_REGION_HOLDERS * len(REGIONS):
        return
    for region in REGIONS:
        while homes.count(region) < MIN_HOME_REGION_HOLDERS:
            donor = max(REGIONS, key=lambda r: (homes.count(r), -REGIONS.index(r)))
            movers = [i for i, h in enumerate(homes) if h == donor]
            homes[max(movers, key=lambda i: priority[i])] = region


def _draw_skills(home: str, k: int, rng: np.random.Generator) -> set[str]:
    chosen: list[str] = []
    for _ in range(k):
        home_pool = [s for s in REGION_SKILLS[home] if s not in chosen]
        away_pool = [s for s in SKILL_TYPES if skill_region(s) != home and s not in chosen]
        use_home = rng.random() < HOME_SKILL_SHARE
        pool = home_pool if (use_home and home_pool) or not away_pool else away_pool
        chosen.append(_pick_weighted(pool, [SKILL_WEIGHTS[s] for s in pool], rng.random()))
    return set(chosen)


def _repair_skill_coverage(
    skills: list[set[str]], homes: list[str], max_skills: int, priority: Sequence[float]
) -> None:
    """Ensure each skill type has ``MIN_HOME_REGION_HOLDERS`` holders living in its region.

    Adds the skill to home-region scouts who lack it, preferring those with
    room under ``max_skills``, then those with the fewest skills, then the
    lowest per-scout ``priority`` (a stable random tie-break: it belongs to the
    scout, so it does not change when others join). Falls back to any scout if
    the region has too few scouts (only possible for teams smaller than 8).
    Mutates ``skills`` in place.
    """
    tiebreak = priority
    for skill in SKILL_TYPES:
        region = skill_region(skill)
        for pool in ([i for i, h in enumerate(homes) if h == region], range(len(skills))):
            holders = sum(1 for i in pool if skill in skills[i])
            candidates = sorted(
                (i for i in pool if skill not in skills[i]),
                key=lambda i: (len(skills[i]) >= max_skills, len(skills[i]), tiebreak[i]),
            )
            for i in candidates[: max(0, MIN_HOME_REGION_HOLDERS - holders)]:
                skills[i].add(skill)


def _draw_former_clubs(home: str, rng: np.random.Generator) -> list[str]:
    cdf = np.cumsum(FORMER_CLUB_COUNT_WEIGHTS)
    n = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
    clubs: set[str] = set()
    while len(clubs) < min(n, len(CLUB_NAMES)):
        pool = REGION_CLUBS[home] if rng.random() < FORMER_CLUB_HOME_SHARE else CLUB_NAMES
        clubs.add(pool[int(rng.random() * len(pool))])
    return sorted(clubs)


def scout_ids(params: Params) -> list[str]:
    """Initial-team ids: ``FT001``.. for full-timers, then ``FL001``.. for freelancers.

    Prefixing by employment keeps ids stable: one more full-timer is
    ``FT025`` and every freelancer keeps its id.
    """
    team = params.team
    ft_width = _id_width(team.full_time_count, 3)
    fl_width = _id_width(team.freelance_count, 3)
    return [f"FT{i:0{ft_width}d}" for i in range(1, team.full_time_count + 1)] + [
        f"FL{i:0{fl_width}d}" for i in range(1, team.freelance_count + 1)
    ]


def scout_stream(
    seed: int, purpose: str, scout_id: str, replication: int = 0
) -> np.random.Generator:
    """One scout's own stream for one purpose (per-entity streams, D-015).

    Profile purposes (``SCOUT_PROFILE_PURPOSES``: home, skills, name, former
    clubs) always come from replication 0, the team being the same in every
    replication; luck purposes (``scout_leave``, ``scout_hours``) come from
    ``replication``. Keyed by scout id, so adding scouts never changes anyone
    else's draws.
    """
    rep = 0 if purpose in SCOUT_PROFILE_PURPOSES else replication
    return make_stream(seed, f"{purpose}.{scout_id}", rep)


SCOUT_PROFILE_PURPOSES = ("scout", "scout_name", "scout_clubs")
_REGION_WEIGHTS = [REGION_DEMAND_SHARE[r] for r in REGIONS]


@dataclass(frozen=True, slots=True)
class _TeamDraw:
    """The initial team before and after the deterministic repair steps."""

    ids: list[str]
    employment: list[str]
    homes: list[str]
    drawn_skills: list[frozenset[str]]  # each scout's own draw, before coverage repair
    skills: list[set[str]]  # after coverage repair
    names: list[str]
    former: list[list[str]]


def _unique_name(home: str, taken: set[str], rng: np.random.Generator) -> str:
    firsts, lasts = _FIRST_NAMES[home], _SURNAMES[home]
    for _ in range(1000):
        name = f"{firsts[int(rng.random() * len(firsts))]} {lasts[int(rng.random() * len(lasts))]}"
        if name not in taken:
            return name
    k = 2  # the region's name space is exhausted: disambiguate with a counter
    while f"{name} {k}" in taken:
        k += 1
    return f"{name} {k}"


def _draw_team(params: Params, seed: int) -> _TeamDraw:
    """Every initial scout from its own streams, then the two team-level repairs.

    Per scout, stream ``scout.<id>`` gives, in order: a home-region uniform, a
    repair priority, the skill count, then exactly 2 uniforms per skill pick.
    Names (``scout_name.<id>``) and former clubs (``scout_clubs.<id>``) have
    their own streams, so a name clash that forces a redraw shifts nothing else.
    Only the repairs look at the whole team, and they only act when needed.
    """
    team = params.team
    ids = scout_ids(params)
    employment = ["full_time"] * team.full_time_count + ["freelance"] * team.freelance_count
    profiles = [scout_stream(seed, "scout", sid) for sid in ids]
    homes = [_pick_weighted(REGIONS, _REGION_WEIGHTS, rng.random()) for rng in profiles]
    priority = [float(rng.random()) for rng in profiles]
    _repair_home_regions(homes, priority)

    lo, hi = team.skills_per_scout
    drawn = [
        frozenset(_draw_skills(home, lo + int(rng.random() * (hi - lo + 1)), rng))
        for home, rng in zip(homes, profiles, strict=True)
    ]
    skills = [set(d) for d in drawn]
    _repair_skill_coverage(skills, homes, hi, priority)

    names: list[str] = []
    taken: set[str] = set()
    for sid, home in zip(ids, homes, strict=True):  # canonical order: FT, then FL
        names.append(_unique_name(home, taken, scout_stream(seed, "scout_name", sid)))
        taken.add(names[-1])
    former = [
        _draw_former_clubs(home, scout_stream(seed, "scout_clubs", sid))
        for sid, home in zip(ids, homes, strict=True)
    ]
    return _TeamDraw(ids, employment, homes, drawn, skills, names, former)


def _generate_scouts(params: Params, seed: int) -> pd.DataFrame:
    draw = _draw_team(params, seed)
    team, cost = params.team, params.cost
    fl_lo, fl_hi = team.freelance_weekly_hours
    is_ft = [e == "full_time" for e in draw.employment]
    return _frame(
        "scouts",
        {
            "scout_id": draw.ids,
            "name": draw.names,
            "employment": draw.employment,
            "skills": [sorted(s, key=SKILL_TYPES.index) for s in draw.skills],
            "home_region": draw.homes,
            "min_weekly_hours": [FULL_TIME_WEEKLY_HOURS if ft else fl_lo for ft in is_ft],
            "max_weekly_hours": [FULL_TIME_WEEKLY_HOURS if ft else fl_hi for ft in is_ft],
            "former_clubs": draw.former,
            "monthly_salary": [cost.full_time_monthly_salary if ft else np.nan for ft in is_ft],
            "hourly_rate": [np.nan if ft else cost.freelance_hourly_rate for ft in is_ft],
            "joined_month": [0] * len(draw.ids),
        },
    )


def _leave_blocks(total: int) -> list[int]:
    """Split ``total`` leave days into blocks following ``LEAVE_BLOCK_PATTERN``."""
    blocks: list[int] = []
    i = 0
    while total > 0:
        size = min(total, LEAVE_BLOCK_PATTERN[i % len(LEAVE_BLOCK_PATTERN)])
        blocks.append(size)
        total -= size
        i += 1
    return blocks


def _full_time_leave(
    weekdays: list[dt.date], total: int, rng: np.random.Generator
) -> list[dt.date]:
    """Calendar days of leave: ``total`` weekdays in blocks, weekends inside a block included."""
    free = np.ones(len(weekdays), dtype=bool)
    days: list[dt.date] = []
    for size in _leave_blocks(min(total, len(weekdays))):
        for _attempt in range(100):
            start = int(rng.random() * (len(weekdays) - size + 1))
            if free[start : start + size].all():
                break
        else:  # crowded calendar: take the first free run that fits, else single days
            runs = [s for s in range(len(weekdays) - size + 1) if free[s : s + size].all()]
            if not runs:
                singles = np.flatnonzero(free)[:size]
                free[singles] = False
                days += [weekdays[i] for i in singles]
                continue
            start = runs[0]
        free[start : start + size] = False
        first, last = weekdays[start], weekdays[start + size - 1]
        days += [first + dt.timedelta(days=k) for k in range((last - first).days + 1)]
    return sorted(days)


def _generate_unavailability(
    params: Params, scouts: pd.DataFrame, seed: int, replication: int
) -> pd.DataFrame:
    """Leave (full-time) and days off (freelance), each scout from ``scout_leave.<id>``."""
    start, end = SIM_START, sim_end(params)
    all_days = [start + dt.timedelta(days=k) for k in range((end - start).days + 1)]
    weekdays = [d for d in all_days if d.weekday() < 5]
    share_of_year = params.sim.months / 12
    leave_total = round(params.team.leave_days_per_year * share_of_year)
    ids, dates, reasons = [], [], []
    for scout_id, employment in zip(scouts["scout_id"], scouts["employment"], strict=True):
        rng = scout_stream(seed, "scout_leave", scout_id, replication)
        if employment == "full_time":
            days, reason = _full_time_leave(weekdays, leave_total, rng), "leave"
        else:
            n = min(int(rng.poisson(FREELANCE_DAYS_OFF_PER_YEAR * share_of_year)), len(all_days))
            picks = rng.permutation(len(all_days))[:n]
            days, reason = sorted(all_days[i] for i in picks), "other"
        ids += [scout_id] * len(days)
        dates += days
        reasons += [reason] * len(days)
    return _frame("scout_unavailability", {"scout_id": ids, "date": dates, "reason": reasons})


def sim_week_starts(params: Params) -> list[dt.date]:
    """Mondays of every week overlapping the simulated period."""
    first = SIM_START - dt.timedelta(days=SIM_START.weekday())
    end = sim_end(params)
    return [first + dt.timedelta(weeks=k) for k in range((end - first).days // 7 + 1)]


def _generate_weekly_hours(
    params: Params, scouts: pd.DataFrame, seed: int, replication: int
) -> pd.DataFrame:
    """Hours each scout offers each week; freelancers' are pre-drawn here (D-015).

    Each freelancer draws one uniform per week from ``scout_hours.<id>``, so
    their hours never depend on who else is on the team.
    """
    weeks = sim_week_starts(params)
    lo, hi = params.team.freelance_weekly_hours
    hours = np.full((len(scouts), len(weeks)), FULL_TIME_WEEKLY_HOURS)
    for row, (sid, employment) in enumerate(
        zip(scouts["scout_id"], scouts["employment"], strict=True)
    ):
        if employment == "freelance":
            draws = lo + scout_stream(seed, "scout_hours", sid, replication).random(len(weeks)) * (
                hi - lo
            )
            step = FREELANCE_HOURS_STEP
            hours[row] = np.clip(np.round(draws / step) * step, lo, hi)
    return _frame(
        "scout_weekly_hours",
        {
            "scout_id": np.repeat(scouts["scout_id"].to_numpy(dtype=object), len(weeks)),
            "week_start": weeks * len(scouts),
            "hours": hours.ravel(),
        },
    )


# --- requests ----------------------------------------------------------------------


def _requests_for_period(
    params: Params,
    period: str,
    days: np.ndarray,
    rate: np.ndarray,
    ceiling: np.ndarray,
    streams: Mapping[str, np.random.Generator],
) -> dict[str, np.ndarray]:
    """Arrivals of one period by thinning: candidates at ``ceiling``, kept at ``rate``.

    Every candidate gets all its attributes from fixed-size uniform draws on
    per-purpose streams, kept or not, so the kept set and its attributes are
    nested across rates that share the same ceiling.
    """
    d = params.demand
    counts = streams[f"arrivals.{period}"].poisson(ceiling)
    n = int(counts.sum())
    received = np.repeat(days, counts)
    keep_prob = np.repeat(
        np.divide(rate, ceiling, out=np.zeros_like(rate), where=ceiling > 0), counts
    )
    keep = streams[f"thinning.{period}"].random(n) < keep_prob
    skill_u = streams[f"request_skill.{period}"].random(n)
    club_u = streams[f"request_clubs.{period}"].random((n, 2))
    dur_u = streams[f"durations.{period}"].random((n, 2))
    live_u = streams[f"live_view.{period}"].random(n)
    rework = streams[f"rework.{period}"].random(n)
    urgent_u = streams[f"urgency.{period}"].random(n)

    weights = np.array([SKILL_WEIGHTS[s] for s in SKILL_TYPES], dtype=float)
    cdf = np.cumsum(weights) / weights.sum()
    skill_idx = np.minimum(np.searchsorted(cdf, skill_u, side="right"), len(SKILL_TYPES) - 1)
    skills = np.array(SKILL_TYPES, dtype=object)[skill_idx]

    club_index = {c: i for i, c in enumerate(CLUB_NAMES)}
    player = np.empty(n, dtype=object)
    player_global = np.empty(n, dtype=np.int64)
    for k, skill in enumerate(SKILL_TYPES):
        mask = skill_idx == k
        pool = REGION_CLUBS[skill_region(skill)]
        picks = (club_u[mask, 0] * len(pool)).astype(np.int64)
        player[mask] = np.array(pool, dtype=object)[picks]
        player_global[mask] = [club_index[pool[p]] for p in picks]
    # Client: uniform over every club except the player's own.
    client_idx = (club_u[:, 1] * (len(CLUB_NAMES) - 1)).astype(np.int64)
    client_idx += client_idx >= player_global
    client = np.array(CLUB_NAMES, dtype=object)[client_idx]

    (desk_lo, desk_hi), (wu_lo, wu_hi) = d.desk_hours, d.writeup_hours
    return {
        "received": received[keep],
        "client_club": client[keep],
        "player_club": player[keep],
        "skill_type": skills[keep],
        # Urgent (express) reports are video-based: never a live view. The
        # live draw itself is unchanged, so flags stay nested across shares.
        "needs_live_view": ((live_u < d.live_view_share) & (urgent_u >= d.urgent_share))[keep],
        "desk_hours": (desk_lo + dur_u[:, 0] * (desk_hi - desk_lo))[keep],
        "writeup_hours": (wu_lo + dur_u[:, 1] * (wu_hi - wu_lo))[keep],
        "rework_draw": rework[keep],
        "urgent": (urgent_u < d.urgent_share)[keep],
    }


def _generate_requests(
    params: Params,
    streams: Mapping[str, np.random.Generator],
    by_club: Mapping[str, Sequence[Fixture]],
) -> pd.DataFrame:
    hist_days = _day_range(history_start(params), SIM_START - dt.timedelta(days=1))
    fut_days = _day_range(SIM_START, sim_end(params))
    parts = {
        period: _requests_for_period(
            params,
            period,
            days,
            daily_rate(params, days),
            ceiling_daily_rate(params, days),
            streams,
        )
        for period, days in (("history", hist_days), ("future", fut_days))
    }
    cols = {k: np.concatenate([p[k] for p in parts.values()]) for k in parts["history"]}
    period = np.repeat(list(parts), [len(p["received"]) for p in parts.values()])
    n = len(period)

    received = cols["received"].astype(object).tolist()
    normal = dt.timedelta(days=params.demand.turnaround_days)
    urgent = dt.timedelta(days=params.demand.urgent_turnaround_days)
    due = [r + (urgent if u else normal) for r, u in zip(received, cols["urgent"], strict=True)]
    at_risk = [
        bool(live) and next_fixture(by_club, club, *live_view_window(r, dd)) is None
        for live, club, r, dd in zip(
            cols["needs_live_view"], cols["player_club"], received, due, strict=True
        )
    ]
    width = _id_width(n, 5)
    return _frame(
        "requests",
        {
            "request_id": [f"R{i:0{width}d}" for i in range(1, n + 1)],
            "client_club": cols["client_club"],
            "player_club": cols["player_club"],
            "skill_type": cols["skill_type"],
            "received_date": received,
            "due_date": due,
            "urgent": cols["urgent"],
            "needs_live_view": cols["needs_live_view"],
            "desk_hours": cols["desk_hours"],
            "writeup_hours": cols["writeup_hours"],
            "period": period,
            "rework_draw": cols["rework_draw"],
            "at_risk_day_one": at_risk,
        },
    )


# --- the whole world ----------------------------------------------------------------

_REQUEST_STREAMS = (
    "arrivals", "thinning", "request_skill", "request_clubs", "durations", "live_view", "rework",
    "urgency",
)  # fmt: skip
STREAM_NAMES: tuple[str, ...] = ("fixtures",) + tuple(
    f"{name}.{period}" for name in _REQUEST_STREAMS for period in ("history", "future")
)


# Shared streams that define the *shape* of the world: the fixture calendar.
# They always come from replication 0. (The team's shape lives in per-scout
# streams, see ``scout_stream``.) Everything else is luck.
FIXED_STREAMS: tuple[str, ...] = ("fixtures",)


def world_streams(
    seed: int, names: Sequence[str] = STREAM_NAMES, replication: int = 0
) -> dict[str, np.random.Generator]:
    """The shared named streams of one world: fixed ones from replication 0, the rest
    from ``replication``.

    A replication (``sim.seeds``) varies *luck*, not *decisions* or the team:
    the scouts and the fixture calendar are the same in every replication,
    while arrivals, durations, live-view flags, urgency, rework draws,
    freelancer hours and leave are drawn again from the replication's own
    streams (``rng.make_streams(seed, names, replication=k)``; per-scout luck
    via ``scout_stream``). Replication 0 is the world written to ``raw/``.
    """
    fixed = [n for n in names if n in FIXED_STREAMS]
    luck = [n for n in names if n not in FIXED_STREAMS]
    return {**make_streams(seed, fixed), **make_streams(seed, luck, replication=replication)}


def generate_world(params: Params, *, seed: int | None = None, replication: int = 0) -> World:
    """Build every stage-[1] table from ``params`` (pure; no file access).

    ``seed`` defaults to ``params.sim.seed``. ``replication`` k > 0 keeps the
    team and fixtures of replication 0 and redraws the luck (see
    :func:`world_streams`).
    """
    seed = params.sim.seed if seed is None else seed
    streams = world_streams(seed, STREAM_NAMES, replication=replication)
    scouts = _generate_scouts(params, seed)
    fixtures = _generate_fixtures(params, streams["fixtures"])
    return World(
        scouts=scouts,
        scout_unavailability=_generate_unavailability(params, scouts, seed, replication),
        scout_weekly_hours=_generate_weekly_hours(params, scouts, seed, replication),
        fixtures=fixtures,
        requests=_generate_requests(
            params, streams, fixtures_by_club(fixtures_from_frame(fixtures))
        ),
    )


# --- frames -> domain objects ---------------------------------------------------------


def scouts_from_frame(df: pd.DataFrame) -> list[Scout]:
    """``Scout`` objects from a ``scouts`` table (nulls become ``None``)."""
    out = []
    for row in df.to_dict("records"):
        for key in ("monthly_salary", "hourly_rate"):
            if pd.isna(row[key]):
                row[key] = None
        out.append(Scout(**row))
    return out


def fixtures_from_frame(df: pd.DataFrame) -> list[Fixture]:
    """``Fixture`` objects from a ``fixtures`` table."""
    return [Fixture(*row) for row in df.itertuples(index=False, name=None)]


def requests_from_frame(df: pd.DataFrame) -> list[Request]:
    """``Request`` objects from a ``requests`` table (extra columns are ignored)."""
    names = [f.name for f in dataclasses.fields(Request)]
    return [Request(*row) for row in df[names].itertuples(index=False, name=None)]


# --- summaries used by the CLI and tests (pure) ---------------------------------------


def realised_start_load(
    world: World, params: Params, months: int = 1, *, deseasonalised: bool = True
) -> float:
    """Realised load of the run's own initial team over the first ``months`` months.

    Work hours of the requests received in that span over the run's team
    available hours. With ``deseasonalised`` (default) the work is first
    brought back to the ``SIM_START`` run-rate (divided by the mean seasonality
    x growth multiplier of the span): for the reference team that is
    ``start_load`` in expectation, and a bigger team gives a lower load. With
    ``deseasonalised=False`` it is the raw calendar load, e.g. January's peak
    (compare :func:`month0_peak_load`). The rest is the Poisson and duration
    noise of this particular draw.
    """
    req = world.requests
    end = add_months(SIM_START, months)
    span = req[(req["received_date"] >= SIM_START) & (req["received_date"] < end)]
    realised = (
        span["desk_hours"].sum()
        + span["writeup_hours"].sum()
        + span["needs_live_view"].sum() * params.demand.live_view_hours
    )
    days = _day_range(SIM_START, end - dt.timedelta(days=1))
    # Deseasonalised: the multiplier sum = "days at the SIM_START run-rate" the
    # span is worth. Raw: plain calendar days.
    day_weight = _demand_multiplier(params, days).sum() if deseasonalised else len(days)
    capacity = team_available_hours_per_month(params) * day_weight * 12 / DAYS_PER_YEAR
    return float(realised / capacity) if capacity > 0 else float("nan")


def month0_peak_load(params: Params) -> float:
    """Expected load of the run's initial team in month 0 (January), seasonality included.

    ``start_load`` is defined on the deseasonalised, average-month run-rate;
    January is a transfer-window peak, so the team is busier than that in
    practice (about 0.70 x 1.35 x the January growth, ~1.0, by default).
    Expected work = the month's arrivals x the run's hours per request;
    capacity = the run's team hours for that many calendar days.
    """
    days = _day_range(SIM_START, add_months(SIM_START, 1) - dt.timedelta(days=1))
    work = daily_rate(params, days).sum() * expected_hours_per_request(params)
    capacity = team_available_hours_per_month(params) * len(days) * 12 / DAYS_PER_YEAR
    return float(work / capacity) if capacity > 0 else float("nan")


def at_risk_share(world: World, period: str = "future") -> float:
    """Share of live-view requests in ``period`` flagged at risk from day one."""
    req = world.requests
    live = req[(req["period"] == period) & req["needs_live_view"]]
    return float(live["at_risk_day_one"].mean()) if len(live) else 0.0


# --- I/O edge ----------------------------------------------------------------------


def write_world(world: World, out_dir: str | Path) -> dict[str, Path]:
    """Write each table to ``<out_dir>/raw/<table>.parquet``; returns the paths.

    The explicit arrow schema pins every column's type to the contract
    (``date32`` dates, ``list<string>`` lists, nullable floats), whatever pandas
    would have inferred. Same world -> byte-identical files.
    """
    raw = Path(out_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in world.tables().items():
        table = pa.Table.from_pandas(df, schema=TABLE_SCHEMAS[name], preserve_index=False)
        paths[name] = raw / f"{name}.parquet"
        pq.write_table(table, paths[name])
    return paths


def read_world(out_dir: str | Path) -> World:
    """Read the tables written by :func:`write_world` back into an equal ``World``."""
    raw = Path(out_dir) / "raw"
    frames = {}
    for name, schema in TABLE_SCHEMAS.items():
        df = pq.read_table(raw / f"{name}.parquet", schema=schema).to_pandas()
        for f in schema:
            if pa.types.is_list(f.type):  # arrow lists arrive as numpy arrays
                df[f.name] = pd.Series([list(v) for v in df[f.name]], dtype=object)
        frames[name] = _frame(name, {c: df[c].tolist() for c in schema.names})
    return World(**frames)
