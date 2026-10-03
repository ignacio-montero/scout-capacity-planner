"""Stage [3]: capacity plan and hiring table. Spec: DATA_CONTRACTS section 3.

``build_capacity_plan(scouts, unavailability, forecast, params)`` is pure:
team + forecast in, ``(capacity_plan, hiring_plan)`` out. ``hires_to_scouts``
turns the hiring table into ``Scout`` objects for the simulation (M4).
``write_plan`` / ``read_plan`` are the I/O edge.

Planned hours per scout
-----------------------
Per scout and plan month: ``(weekly hours x weeks in month - leave hours) x
target_utilisation``, never below 0, from the month the scout joined.

* Weekly hours: full-time 37.5; a freelancer the *mean* of their weekly range.
  The planner plans on expected hours; it does not know the pre-drawn future
  weekly offers (those belong to the simulated world).
* Leave: weekdays a full-time scout is unavailable x 7.5 h. A freelancer's
  days off move *which* days they work, not their weekly total, so they are
  not deducted (DATA_CONTRACTS section 1).
* Utilisation is applied to the hours a scout is actually present, i.e. after
  leave: a scout on leave all month offers 0, not a negative number.

Coverage: who covers which skill (a transportation problem)
------------------------------------------------------------
Required hours per skill are the forecast's planning quantile (``hours_pq``).
Each month, the team's hours are allocated to skills by a **min-cost max-flow**
(OR-Tools): source -> scout (capacity = planned hours) -> each skill the scout
holds -> sink (capacity = required hours). The maximum flow is the most
required work the current team can cover; whatever is left is the true
shortfall. A fixed split of each scout's hours (e.g. by demand share) would
leave spare hours stranded on one skill while another skill nobody else can
cover looks short: *phantom gaps* that make the plan hire people the team
does not need (D-021).

Ties: when several skills compete for the same scouts and not all can be
covered, arc costs make the flow cover the skills with the smaller requirement
first, so the shortfall lands on the larger skill, where a single-skilled hire
has the most work. Hours are scaled to integer centi-hours for the solver.

Columns of ``capacity_plan`` (hours):

* ``available_hours``: what the current team puts on the skill under that
  allocation (hours covering it), plus each scout's spare hours spread over
  their skills by demand. Below ``required`` only where the team truly cannot
  cover the skill.
* ``gap_hours = required - available``: positive = shortfall, negative = slack.
* ``hired_hours``: planned hours the hires add (each hire is single-skilled).
* ``gap_after_hires_hours``: the same gap after re-allocating with the hires.

Hiring rule (greedy, month by month)
------------------------------------
Hires are single-skilled (the gap's skill), home region = the skill's region.
A hire's hours serve only their skill, so they are netted off that skill's
requirement before the team is re-allocated (a dedicated resource is always
best used first). Walking the plan months in order, while some skill is short
by more than the **hiring tolerance** in month ``m`` (``HIRING_TOLERANCE_SHARE``
of one full-timer's planned month, ~31 h: smaller gaps are absorbed by
utilisation slack, not hires), take the skill with the largest shortfall:

* **Persistent or peak?** The shortfall is also computed on *deseasonalised*
  requirements (divided by the forecast's ``season_index``). It is persistent
  when the deseasonalised shortfall lasts ``>= persistent_gap_months``
  consecutive months from ``m``, **or reaches the end of the plan** (an
  open-ended run: the forecast stops, the demand does not). A shortfall that
  exists only because of the season (a summer peak at flat demand) is a peak.
* Persistent → **full-time** hires, joining at ``m``, acting at
  ``m - lead_time_full_time_months``. Sized to the smallest deseasonalised
  shortfall of the run: full-time staff are a standing cost, so they cover the
  base load.
* Peak → **freelance** hires, joining at ``m``, acting at
  ``m - lead_time_freelance_months``. Sized to the largest shortfall of the
  peak (the run, up to where a persistent shortfall begins): freelancers are
  paid only for hours used, so covering the whole peak costs nothing idle.
* Counts are rounded up after allowing the tolerance (``ceil((gap - tol) /
  hours per hire)``, at least 1).
* If the act month would be before month 0, the hire acts at 0, joins at the
  lead time and the reason says ``late``: the months before it joins stay
  short (no hire can arrive in time; that is what lead times mean). Sizing
  then uses the run's months from the join month on; if the run ends before
  anyone can join, nothing is hired.
* When a persistent shortfall starts before full-time hires can join,
  freelancers (shorter lead time) **bridge** the months in between (reason
  ``bridge``).
* Every hire adds hours from its join month to the end of the plan, and the
  allocation is re-solved before the next decision, so one hire can also
  close another skill's gap (it frees a multi-skilled scout).

This is a *greedy* heuristic (each decision is the locally sensible one,
never revisited), not an optimisation: the simulation is where its cost is
judged.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ortools.graph.python import min_cost_flow

from scout_planner import generate
from scout_planner.config import FULL_TIME_WEEKLY_HOURS, Params
from scout_planner.domain import Scout
from scout_planner.forecast import plan_months
from scout_planner.generate import FULL_TIME_DAILY_HOURS, skill_region
from scout_planner.rng import make_stream

HIRES_STREAM = "hires"  # random stream passed to hires_to_scouts (D-015)
HIRING_TOLERANCE_SHARE = 0.25  # ignore gaps below this share of a full-timer's month
FLOW_SCALE = 100  # hours -> integer centi-hours for the flow solver
DAYS_PER_WEEK = 7.0
HIRE_TYPES = ("full_time", "freelance")

PLAN_SCHEMAS: dict[str, pa.Schema] = {
    # ``hired_hours`` and ``gap_after_hires_hours`` extend DATA_CONTRACTS section 3:
    # available/gap are the current team's; the extra two show the plan's effect.
    "capacity_plan": pa.schema(
        [
            ("month", pa.date32()),
            ("skill_type", pa.string()),
            ("required_hours", pa.float64()),
            ("available_hours", pa.float64()),
            ("gap_hours", pa.float64()),
            ("hired_hours", pa.float64()),
            ("gap_after_hires_hours", pa.float64()),
        ]
    ),
    "hiring_plan": pa.schema(
        [
            ("month_to_act", pa.int64()),
            ("joins_month", pa.int64()),
            ("skill_type", pa.string()),
            ("hire_type", pa.string()),
            ("count", pa.int64()),
            ("reason", pa.string()),
        ]
    ),
}


# --- hours per scout and month (pure) ------------------------------------------------


def weeks_in_month(month: dt.date) -> float:
    """Days in the month / 7 (4.0 to 4.43)."""
    return (generate.add_months(month, 1) - month).days / DAYS_PER_WEEK


def leave_hours(
    unavailability: pd.DataFrame, scouts: Sequence[Scout], months: list[dt.date]
) -> np.ndarray:
    """Leave hours per scout (rows, ``scouts`` order) and plan month (columns).

    Weekdays a full-time scout is unavailable x 7.5 h. Freelancers: 0.
    """
    out = np.zeros((len(scouts), len(months)))
    full_time = {s.scout_id: i for i, s in enumerate(scouts) if s.employment == "full_time"}
    month_index = {(m.year, m.month): k for k, m in enumerate(months)}
    for scout_id, day in zip(unavailability["scout_id"], unavailability["date"], strict=True):
        i = full_time.get(scout_id)
        k = month_index.get((day.year, day.month))
        if i is not None and k is not None and day.weekday() < 5:
            out[i, k] += FULL_TIME_DAILY_HOURS
    return out


def scout_monthly_hours(
    scouts: Sequence[Scout], unavailability: pd.DataFrame, params: Params
) -> np.ndarray:
    """Planned hours per scout (rows) and plan month (columns), see the module docstring."""
    months = plan_months(params)
    weeks = np.array([weeks_in_month(m) for m in months])
    weekly = np.array([(s.min_weekly_hours + s.max_weekly_hours) / 2 for s in scouts])
    gross = weekly[:, None] * weeks[None, :]
    usable = (gross - leave_hours(unavailability, scouts, months)).clip(min=0.0)
    usable *= params.capacity_plan.target_utilisation
    joined = np.array([s.joined_month for s in scouts])
    usable[np.arange(len(months))[None, :] < joined[:, None]] = 0.0
    return usable


def hire_monthly_hours(hire_type: str, params: Params) -> np.ndarray:
    """Planned hours one new hire adds in each plan month (if on the team all month).

    Full-time: 37.5 h/week minus an average month's leave; freelance: mean
    weekly range. Both x ``target_utilisation``.
    """
    months = plan_months(params)
    weeks = np.array([weeks_in_month(m) for m in months])
    util = params.capacity_plan.target_utilisation
    if hire_type == "full_time":
        leave = params.team.leave_days_per_year * FULL_TIME_DAILY_HOURS / 12
        return ((FULL_TIME_WEEKLY_HOURS * weeks - leave) * util).clip(min=0.0)
    return float(np.mean(params.team.freelance_weekly_hours)) * weeks * util


def hiring_tolerance(params: Params) -> float:
    """Shortfall (hours/month) below which nobody is hired: a quarter of a full-timer."""
    return HIRING_TOLERANCE_SHARE * float(hire_monthly_hours("full_time", params).mean())


# --- coverage: min-cost max-flow (pure) ----------------------------------------------


def _skill_index(scouts: Sequence[Scout], skills: Sequence[str]) -> list[np.ndarray]:
    col = {s: j for j, s in enumerate(skills)}
    return [
        np.array(sorted(col[s] for s in sc.skills if s in col), dtype=np.int64) for sc in scouts
    ]


def allocate(capacity: np.ndarray, holds: Sequence[np.ndarray], required: np.ndarray) -> np.ndarray:
    """Hours each scout (rows) puts on each skill (columns) to cover the most required work.

    ``capacity[i]``: scout ``i``'s hours; ``holds[i]``: indices of the skills
    they hold; ``required[j]``: hours skill ``j`` needs. Solves a maximum flow
    source -> scouts -> skills -> sink; among maximum flows, the one that
    covers smaller requirements first (the shortfall falls on larger skills).
    """
    n_scouts, n_skills = len(capacity), len(required)
    alloc = np.zeros((n_scouts, n_skills))
    # Floor: never allocate a hundredth of an hour more than a scout has.
    cap = np.floor(np.asarray(capacity) * FLOW_SCALE + 1e-9).astype(np.int64)
    req = np.ceil(np.asarray(required) * FLOW_SCALE - 1e-9).astype(np.int64)
    if cap.sum() <= 0 or req.sum() <= 0:
        return alloc
    source, sink = 0, 1 + n_scouts + n_skills
    pairs = [(i, j) for i in range(n_scouts) for j in holds[i] if cap[i] > 0 and req[j] > 0]
    if not pairs:
        return alloc
    # Smallest requirement -> cheapest arc to the sink -> covered first (ties: skill order).
    rank = np.empty(n_skills, dtype=np.int64)
    rank[np.lexsort((np.arange(n_skills), req))] = np.arange(n_skills)
    tails = np.r_[np.zeros(n_scouts), [1 + i for i, _ in pairs], 1 + n_scouts + np.arange(n_skills)]
    heads = np.r_[
        1 + np.arange(n_scouts), [1 + n_scouts + j for _, j in pairs], np.full(n_skills, sink)
    ]
    caps = np.r_[cap, [cap[i] for i, _ in pairs], req]
    costs = np.r_[np.zeros(n_scouts + len(pairs)), rank]
    flow = min_cost_flow.SimpleMinCostFlow()
    flow.add_arcs_with_capacity_and_unit_cost(
        tails.astype(np.int64),
        heads.astype(np.int64),
        caps.astype(np.int64),
        costs.astype(np.int64),
    )
    total = int(cap.sum())
    flow.set_node_supply(source, total)
    flow.set_node_supply(sink, -total)
    status = flow.solve_max_flow_with_min_cost()
    if status != flow.OPTIMAL:  # pragma: no cover - the network is always well formed
        raise RuntimeError(f"capacity allocation failed: {status}")
    flows = flow.flows(np.arange(n_scouts, n_scouts + len(pairs)))
    for (i, j), value in zip(pairs, flows, strict=True):
        alloc[i, j] = value / FLOW_SCALE
    return alloc


def team_coverage(
    hours: np.ndarray, holds: Sequence[np.ndarray], required: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(covered, available)`` per skill (rows) and month (columns) for the team.

    ``hours``: scouts x months; ``required``: skills x months. ``covered`` is
    the flow allocation; ``available`` adds each scout's spare hours, spread
    over their skills in proportion to ``weights`` (skills x months; equal
    split where all are 0).
    """
    n_skills, n_months = required.shape
    covered = np.zeros((n_skills, n_months))
    available = np.zeros((n_skills, n_months))
    for k in range(n_months):
        alloc = allocate(hours[:, k], holds, required[:, k])
        covered[:, k] = alloc.sum(axis=0)
        available[:, k] = covered[:, k]
        spare = hours[:, k] - alloc.sum(axis=1)
        for i, js in enumerate(holds):
            if len(js) == 0 or spare[i] <= 1e-9:
                continue
            w = weights[js, k]
            share = w / w.sum() if w.sum() > 0 else np.full(len(js), 1 / len(js))
            available[js, k] += share * spare[i]
    return covered, available


# --- the hiring rule (pure) -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hire:
    """One row of the hiring table."""

    month_to_act: int
    joins_month: int
    skill_type: str
    hire_type: str
    count: int
    reason: str


# unmet(requirements skills x months, months to compute) -> shortfall skills x those months
UnmetFn = Callable[[np.ndarray, range], np.ndarray]


def _run_length(gap: np.ndarray, start: int, tol: float) -> int:
    n = 0
    while start + n < len(gap) and gap[start + n] > tol:
        n += 1
    return n


def greedy_hiring(
    skills: Sequence[str],
    required: np.ndarray,
    season: np.ndarray,
    unmet: UnmetFn,
    params: Params,
    months: list[dt.date] | None = None,
) -> tuple[list[Hire], np.ndarray]:
    """The hiring rule over all skills. Returns ``(hires, hired hours skills x months)``.

    ``required``: skills x months (planning quantile); ``season``: seasonal
    index per month; ``unmet(R, months)``: the team's shortfall for
    requirements ``R`` (hires are netted off ``R`` before calling it).

    Three shortfalls are tracked: ``gap`` (every hire counts: is the month
    covered?), ``ft_gap`` (only full-time hires count: what full-timers still
    have to cover; freelancers are the flexible layer, so one hired for a peak
    does not hide a lasting shortfall) and ``base_gap`` (as ``ft_gap`` but on
    deseasonalised requirements: used only to *classify* persistence).
    """
    cp = params.capacity_plan
    n_skills, n_months = required.shape
    months = months or plan_months(params)
    tol = hiring_tolerance(params)
    cap = {t: hire_monthly_hours(t, params)[:n_months] for t in HIRE_TYPES}
    lead = {"full_time": cp.lead_time_full_time_months, "freelance": cp.lead_time_freelance_months}
    base_required = required / np.asarray(season, dtype=float)[None, :n_months]
    hired = {t: np.zeros((n_skills, n_months)) for t in HIRE_TYPES}
    hires: dict[tuple[int, int, int, str], Hire] = {}  # merge repeats of one decision
    gap = unmet(required, range(n_months))
    ft_gap = gap.copy()
    base_gap = unmet(base_required, range(n_months))

    def refresh(start: int) -> None:
        later = range(start, n_months)
        total = hired["full_time"] + hired["freelance"]
        gap[:, start:] = unmet((required - total).clip(min=0.0), later)
        ft_gap[:, start:] = unmet((required - hired["full_time"]).clip(min=0.0), later)
        base_gap[:, start:] = unmet((base_required - hired["full_time"]).clip(min=0.0), later)

    def persistent_from(j: int, k: int) -> int:
        """Length of the persistent base shortfall from month ``k`` (0 if not persistent)."""
        run = _run_length(base_gap[j], k, tol)
        return run if run >= cp.persistent_gap_months or k + run == n_months else 0

    def hire(j: int, hire_type: str, m: int, window: Iterable[int], g: np.ndarray, size_on: str,
             why: str) -> int | None:  # fmt: skip
        """Hire for skill ``j``'s months in ``window``; returns the join month or ``None``."""
        join = max(m, lead[hire_type])
        sizing = [k for k in window if k >= join and cap[hire_type][k] > 0]
        if not sizing:
            return None
        ratios = [(g[k] - tol) / cap[hire_type][k] for k in sizing]
        needed = min(ratios) if size_on == "min" else max(ratios)
        count = max(1, math.ceil(needed - 1e-9))
        hired[hire_type][j, join:] += count * cap[hire_type][join:]
        act = join - lead[hire_type]
        if join > m:
            why += f"; late: earliest join {months[join]:%Y-%m}"
        key = (act, join, j, hire_type)
        old = hires.get(key)
        if old is None:
            hires[key] = Hire(act, join, skills[j], hire_type, count, why)
        else:  # another pass of the same decision: one row, first reason
            hires[key] = dataclasses.replace(old, count=old.count + count)
        refresh(join)
        return join

    for m in range(n_months):
        stuck: set[int] = set()  # skills whose month-m shortfall no hire can reach in time
        for _ in range(100 * n_skills):  # every pass hires or marks a skill stuck
            urgency = {
                j: max(gap[j, m], ft_gap[j, m] if persistent_from(j, m) else 0.0)
                for j in range(n_skills)
                if j not in stuck
            }
            short = [j for j, u in urgency.items() if u > tol]
            if not short:
                break
            j = max(short, key=lambda s: (urgency[s], -s))
            run = _run_length(gap[j], m, tol)
            base_run = persistent_from(j, m)
            ft_join = max(m, lead["full_time"])
            floor_months = [k for k in range(m, m + base_run) if k >= ft_join]
            floor = min((ft_gap[j, k] for k in floor_months), default=0.0)
            if not base_run or (floor <= tol and ft_join == m):
                # Peak (or a lasting shortfall whose floor is already covered):
                # freelancers for the run, up to where a persistent shortfall begins.
                end = m + run
                if not base_run:
                    end = next((k for k in range(m + 1, m + run) if persistent_from(j, k)), end)
                why = (
                    f"peak-only gap: {run} month(s) from {months[m]:%Y-%m} "
                    f"(seasonal, or < {cp.persistent_gap_months} months)"
                )
                join = hire(j, "freelance", m, range(m, end), gap[j], "max", why) if run else None
                if join is None or join > m:
                    stuck.add(j)
                continue
            # Persistent: full-time sized to the floor of the run's shortfall.
            open_ended = m + base_run == n_months
            why = f"persistent gap: {base_run} month(s) from {months[m]:%Y-%m} " + (
                "(open-ended)" if open_ended else f"(>= {cp.persistent_gap_months})"
            )
            join = None
            if floor > tol:
                join = hire(j, "full_time", m, floor_months, ft_gap[j], "min", why)
            if join == m:
                continue
            # Full-time cannot arrive by month m: freelancers bridge the actual
            # shortfall until they join.
            until = n_months if join is None else join
            what = (
                "no full-time hire fits"
                if join is None
                else f"full-time joins {months[join]:%Y-%m}"
            )
            why = f"bridge: {run} month(s) from {months[m]:%Y-%m}, {what}"
            if run:
                hire(j, "freelance", m, range(m, min(m + run, until)), gap[j], "max", why)
            stuck.add(j)
    return list(hires.values()), hired["full_time"] + hired["freelance"]


def plan_skill_hires(
    skill: str,
    required: np.ndarray,
    available: np.ndarray,
    params: Params,
    months: list[dt.date] | None = None,
    season: np.ndarray | None = None,
) -> tuple[list[Hire], np.ndarray]:
    """The hiring rule for one skill with dedicated capacity ``available`` (no sharing).

    Returns ``(hires, hired hours per month)``. ``season`` defaults to flat.
    Handy to reason about (and test) the rule in isolation.
    """
    req = np.asarray(required, dtype=float)[None, :]
    avail = np.asarray(available, dtype=float)

    def unmet(r: np.ndarray, ks: range) -> np.ndarray:
        idx = list(ks)
        return (r[:, idx] - avail[idx]).clip(min=0.0)

    flat = np.ones(req.shape[1]) if season is None else np.asarray(season, dtype=float)
    hires, hired = greedy_hiring([skill], req, flat, unmet, params, months)
    return hires, hired[0]


# --- the stage --------------------------------------------------------------------------


def _as_scouts(scouts: pd.DataFrame | Iterable[Scout]) -> list[Scout]:
    if isinstance(scouts, pd.DataFrame):
        return generate.scouts_from_frame(scouts)
    return list(scouts)


def _pivot(forecast: pd.DataFrame, column: str, months: list[dt.date]) -> pd.DataFrame:
    """``forecast[column]`` as months (rows) x skills (columns, forecast order)."""
    skills = list(dict.fromkeys(forecast["skill_type"]))
    table = forecast.pivot_table(index="month", columns="skill_type", values=column, aggfunc="sum")
    return table.reindex(index=months, columns=skills).fillna(0.0)


def _season(forecast: pd.DataFrame, months: list[dt.date]) -> np.ndarray:
    """The forecast's seasonal index per plan month (flat if the column is absent)."""
    if "season_index" not in forecast.columns:
        return np.ones(len(months))
    by_month = forecast.groupby("month")["season_index"].first()
    values = by_month.reindex(months).to_numpy(dtype=float)
    return np.where(np.isfinite(values) & (values > 0), values, 1.0)


def build_capacity_plan(
    scouts: pd.DataFrame | Iterable[Scout],
    unavailability: pd.DataFrame,
    forecast: pd.DataFrame,
    params: Params,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage [3]: ``(capacity_plan, hiring_plan)`` from the team and the forecast (pure)."""
    team = _as_scouts(scouts)
    months = plan_months(params)
    required_df = _pivot(forecast, "hours_pq", months)
    skills = list(required_df.columns)
    required = required_df.to_numpy().T  # skills x months
    weights = _pivot(forecast, "hours_p50", months).to_numpy().T
    holds = _skill_index(team, skills)
    hours = scout_monthly_hours(team, unavailability, params)

    def unmet(r: np.ndarray, ks: range) -> np.ndarray:
        idx = list(ks)
        covered, _ = team_coverage(hours[:, idx], holds, r[:, idx], weights[:, idx])
        return (r[:, idx] - covered).clip(min=0.0)

    _, available = team_coverage(hours, holds, required, weights)
    hire_rows, hired = greedy_hiring(
        skills, required, _season(forecast, months), unmet, params, months
    )
    _, available_after = team_coverage(hours, holds, (required - hired).clip(min=0.0), weights)

    cap_rows: dict[str, list] = {c: [] for c in PLAN_SCHEMAS["capacity_plan"].names}
    for j, skill in enumerate(skills):
        for k, month in enumerate(months):
            cap_rows["month"].append(month)
            cap_rows["skill_type"].append(skill)
            cap_rows["required_hours"].append(float(required[j, k]))
            cap_rows["available_hours"].append(float(available[j, k]))
            cap_rows["gap_hours"].append(float(required[j, k] - available[j, k]))
            cap_rows["hired_hours"].append(float(hired[j, k]))
            cap_rows["gap_after_hires_hours"].append(
                float(required[j, k] - available_after[j, k] - hired[j, k])
            )

    order = {s: i for i, s in enumerate(skills)}
    hire_rows.sort(key=lambda h: (h.month_to_act, h.joins_month, order[h.skill_type], h.hire_type))
    hiring = {c: [getattr(h, c) for h in hire_rows] for c in PLAN_SCHEMAS["hiring_plan"].names}
    return _frame("capacity_plan", cap_rows), _frame("hiring_plan", hiring)


# --- hires as scouts (for the simulation) ------------------------------------------------


def hires_to_scouts(
    hiring_plan: pd.DataFrame,
    params: Params,
    rng: np.random.Generator,
    *,
    taken_names: Iterable[str] = (),
) -> list[Scout]:
    """One ``Scout`` per hire in the table, in table order (pure given ``rng``).

    Ids ``H001``…, invented names from the skill's region (unique, also vs
    ``taken_names``), the gap's single skill, home region = the skill's
    region (live views need it, D-008), former clubs drawn like the initial
    team's, salary or rate from ``cost.*``, ``joined_month = joins_month``.

    Randomness: ``rng`` (pass ``make_stream(seed, HIRES_STREAM)``) is used for
    one draw only, a base seed; each hire then draws from its own stream named
    ``hires.<skill>.<hire_type>.<joins_month>.<k>`` (k-th person of that
    row). So changing one row's count never changes another hire's name or
    former clubs (common random numbers, D-015); only a name collision makes
    one hire redraw.

    What the simulation still has to supply: the hires' weekly hours and leave
    (full-time 37.5 h with ``team.leave_days_per_year`` pro rata, freelancers
    drawn weekly in ``team.freelance_weekly_hours``), per replication, e.g.
    with ``generate.generate_availability`` on ``generate.scouts_to_frame(hires)``.
    """
    base = int(rng.integers(0, 2**62))
    n = int(hiring_plan["count"].sum()) if len(hiring_plan) else 0
    width = max(3, len(str(n)))
    taken = set(taken_names)
    cost, team = params.cost, params.team
    out: list[Scout] = []
    for row in hiring_plan.itertuples(index=False):
        region = skill_region(row.skill_type)
        full_time = row.hire_type == "full_time"
        lo, hi = (
            (FULL_TIME_WEEKLY_HOURS, FULL_TIME_WEEKLY_HOURS)
            if full_time
            else team.freelance_weekly_hours
        )
        for k in range(int(row.count)):
            name = f"hires.{row.skill_type}.{row.hire_type}.{int(row.joins_month)}.{k}"
            person = make_stream(base, name)
            # Former clubs first: their draws never depend on name collisions.
            former = frozenset(generate._draw_former_clubs(region, person))
            full_name = _unique_name(region, taken, person)
            taken.add(full_name)
            out.append(
                Scout(
                    scout_id=f"H{len(out) + 1:0{width}d}",
                    name=full_name,
                    employment=row.hire_type,
                    skills=frozenset({row.skill_type}),
                    home_region=region,
                    min_weekly_hours=float(lo),
                    max_weekly_hours=float(hi),
                    former_clubs=former,
                    monthly_salary=cost.full_time_monthly_salary if full_time else None,
                    hourly_rate=None if full_time else cost.freelance_hourly_rate,
                    joined_month=int(row.joins_month),
                )
            )
    return out


def _unique_name(region: str, taken: set[str], rng: np.random.Generator) -> str:
    firsts, lasts = generate._FIRST_NAMES[region], generate._SURNAMES[region]
    for _ in range(1000):
        name = f"{firsts[int(rng.random() * len(firsts))]} {lasts[int(rng.random() * len(lasts))]}"
        if name not in taken:
            return name
    # The region's name space is exhausted: disambiguate with a counter.
    k = 2
    while f"{name} {k}" in taken:
        k += 1
    return f"{name} {k}"


# --- I/O edge ------------------------------------------------------------------------------


def _frame(table: str, columns: dict[str, list]) -> pd.DataFrame:
    """A table with exactly the contract's columns, in order, with stable dtypes."""
    schema = PLAN_SCHEMAS[table]
    if list(columns) != schema.names:
        raise ValueError(f"{table}: columns {list(columns)} != contract {schema.names}")
    data = {}
    for f in schema:
        if pa.types.is_date32(f.type):
            data[f.name] = pd.Series(list(columns[f.name]), dtype=object)
        elif pa.types.is_string(f.type):
            data[f.name] = pd.Series(columns[f.name], dtype="str")
        elif pa.types.is_int64(f.type):
            data[f.name] = pd.Series(columns[f.name], dtype="int64")
        else:
            data[f.name] = pd.Series(columns[f.name], dtype="float64")
    return pd.DataFrame(data)


def write_plan(capacity: pd.DataFrame, hiring: pd.DataFrame, run_dir: str | Path) -> dict:
    """Write ``capacity_plan.parquet`` and ``hiring_plan.parquet`` into the run folder."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in (("capacity_plan", capacity), ("hiring_plan", hiring)):
        table = pa.Table.from_pandas(df, schema=PLAN_SCHEMAS[name], preserve_index=False)
        paths[name] = run_dir / f"{name}.parquet"
        pq.write_table(table, paths[name])
    return paths


def read_plan(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read back what :func:`write_plan` wrote: ``(capacity_plan, hiring_plan)``."""
    out = []
    for name in ("capacity_plan", "hiring_plan"):
        schema = PLAN_SCHEMAS[name]
        df = pq.read_table(Path(run_dir) / f"{name}.parquet", schema=schema).to_pandas()
        out.append(_frame(name, {c: df[c].tolist() for c in schema.names}))
    return out[0], out[1]
