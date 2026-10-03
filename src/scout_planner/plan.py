"""Stage [3]: capacity plan and hiring table. Spec: DATA_CONTRACTS section 3.

``build_capacity_plan(scouts, unavailability, forecast, params)`` is pure:
team + forecast in, ``(capacity_plan, hiring_plan)`` out. ``hires_to_scouts``
turns the hiring table into ``Scout`` objects for the simulation (M4).
``write_plan`` / ``read_plan`` are the I/O edge.

Available hours
---------------
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
* A scout with several skills splits their hours across them in proportion to
  each skill's forecast demand that month (a simplification: a real team
  would steer multi-skilled scouts towards the scarce skill).

Required hours are the forecast's planning quantile (``hours_pq``).
``gap = required - available``; positive means short.

Hiring rule (greedy, month by month, per skill)
-----------------------------------------------
Hires are single-skilled (the gap's skill), so skills are planned
independently. Walking the plan months in order, while month ``m`` is short:

* Find the *run*: consecutive short months starting at ``m``.
* Run of ``>= persistent_gap_months`` → **full-time** hires, joining at
  ``m``, acting at ``m - lead_time_full_time_months``. Sized to the run's
  *smallest* gap (the part that persists through the whole run): full-time
  staff are a standing cost, so they cover the base load.
* Shorter run (a peak) → **freelance** hires, joining at ``m``, acting at
  ``m - lead_time_freelance_months``. Sized to the run's *largest* gap:
  freelancers are paid only for hours used, so covering the whole peak costs
  nothing when they are idle.
* If the act month would be before month 0, the hire acts at 0, joins at the
  lead time and the reason says ``late``: the months before it joins stay short
  (no hire can arrive in time; that is what lead times mean). Sizing then uses
  the run's months from the join month on; if the run ends before anyone can
  join, nothing is hired.
* When a persistent gap starts before full-time hires can join, freelancers
  (shorter lead time) **bridge** the months in between, sized to the largest
  gap of those months (reason ``bridge``).
* Each hire adds capacity from its join month to the end of the plan, and the
  gaps are recomputed before the next decision. So staged growth produces
  staged hiring, and a full-time hire for a long run reduces every later gap.

This is a *greedy* heuristic (each decision is the locally sensible one,
never revisited), not an optimisation: the simulation is where its cost is
judged.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scout_planner import generate
from scout_planner.config import FULL_TIME_WEEKLY_HOURS, Params
from scout_planner.domain import Scout
from scout_planner.forecast import plan_months
from scout_planner.generate import FULL_TIME_DAILY_HOURS, skill_region

HIRES_STREAM = "hires"  # random stream for hires_to_scouts (D-015)
EPS_HOURS = 1e-6  # a gap below this is no gap (float noise)
DAYS_PER_WEEK = 7.0

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
    """Usable hours per scout (rows) and plan month (columns), see the module docstring."""
    months = plan_months(params)
    weeks = np.array([weeks_in_month(m) for m in months])
    weekly = np.array([(s.min_weekly_hours + s.max_weekly_hours) / 2 for s in scouts])
    gross = weekly[:, None] * weeks[None, :]
    usable = (gross - leave_hours(unavailability, scouts, months)).clip(min=0.0)
    usable *= params.capacity_plan.target_utilisation
    joined = np.array([s.joined_month for s in scouts])
    usable[np.arange(len(months))[None, :] < joined[:, None]] = 0.0
    return usable


def split_by_demand(
    scouts: Sequence[Scout], hours: np.ndarray, demand: pd.DataFrame
) -> pd.DataFrame:
    """Available hours per month (rows) and skill (columns).

    Each scout's hours in a month go to their skills in proportion to that
    month's ``demand`` (months x skills); equal split if all of them are 0.
    Skills with no demand column are ignored for the split.
    """
    out = pd.DataFrame(0.0, index=demand.index, columns=demand.columns)
    values = demand.to_numpy()
    col = {s: j for j, s in enumerate(demand.columns)}
    for i, scout in enumerate(scouts):
        js = sorted(col[s] for s in scout.skills if s in col)
        if not js:
            continue
        weights = values[:, js]
        totals = weights.sum(axis=1, keepdims=True)
        shares = np.where(totals > 0, weights / np.where(totals > 0, totals, 1), 1 / len(js))
        out.iloc[:, js] += shares * hours[i][:, None]
    return out


def hire_monthly_hours(hire_type: str, params: Params) -> np.ndarray:
    """Usable hours one new hire adds in each plan month (if on the team all month).

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


def _run_length(gap: np.ndarray, start: int) -> int:
    n = 0
    while start + n < len(gap) and gap[start + n] > EPS_HOURS:
        n += 1
    return n


def plan_skill_hires(
    skill: str,
    required: np.ndarray,
    available: np.ndarray,
    params: Params,
    months: list[dt.date] | None = None,
) -> tuple[list[Hire], np.ndarray]:
    """Greedy hiring for one skill. Returns ``(hires, hired hours per month)``.

    See the module docstring for the rule. ``months`` (plan month dates) are
    only used to word the ``reason``.
    """
    cp = params.capacity_plan
    n_months = len(required)
    months = months or plan_months(params)
    cap = {t: hire_monthly_hours(t, params)[:n_months] for t in ("full_time", "freelance")}
    lead = {"full_time": cp.lead_time_full_time_months, "freelance": cp.lead_time_freelance_months}
    hired = np.zeros(n_months)
    hires: dict[tuple[int, int, str], Hire] = {}  # merge repeats of one decision

    def hire(hire_type: str, m: int, window: range, size_on: str, why: str) -> int | None:
        """Hire ``hire_type`` for the short months in ``window`` (gap seen at month m).

        Returns the join month, or ``None`` if nobody can join within the window.
        """
        gap = required - available - hired
        join = max(m, lead[hire_type])
        sizing = [k for k in window if k >= join and cap[hire_type][k] > 0]
        if not sizing:
            return None
        ratios = [gap[k] / cap[hire_type][k] for k in sizing]
        count = max(1, math.ceil((min(ratios) if size_on == "min" else max(ratios)) - 1e-9))
        hired[join:] += count * cap[hire_type][join:]
        act = join - lead[hire_type]
        if join > m:
            why += f"; late: earliest join {months[join]:%Y-%m}"
        key = (act, join, hire_type)
        old = hires.get(key)
        if old is None:
            hires[key] = Hire(act, join, skill, hire_type, count, why)
        else:  # another pass of the same decision: one row, first reason
            hires[key] = dataclasses.replace(old, count=old.count + count)
        return join

    for m in range(n_months):
        for _ in range(10 * n_months + 10):  # each pass hires; a guard, never reached
            gap = required - available - hired
            if gap[m] <= EPS_HOURS:
                break
            run = _run_length(gap, m)
            span = f"{run} month(s) from {months[m]:%Y-%m}"
            if run < cp.persistent_gap_months:
                # Peak only: freelancers sized to the peak of the run.
                why = f"peak-only gap: {span} (< {cp.persistent_gap_months})"
                join = hire("freelance", m, range(m, m + run), "max", why)
                if join is None or join > m:
                    break  # this month stays short: nobody can arrive in time
                continue
            # Persistent: full-time sized to the smallest gap of the run (base load).
            why = f"persistent gap: {span} (>= {cp.persistent_gap_months})"
            join = hire("full_time", m, range(m, m + run), "min", why)
            if join == m:
                continue
            # Full-time cannot arrive by month m: freelancers bridge the months
            # until they join (or the whole run if no full-time hire fits in it).
            ft_join = n_months if join is None else join
            bridge = range(m, min(m + run, ft_join))
            until = (
                "no full-time hire fits"
                if join is None
                else f"full-time joins {months[join]:%Y-%m}"
            )
            why = f"bridge: persistent gap {span}, {until}"
            hire("freelance", m, bridge, "max", why)
            break
    return list(hires.values()), hired


# --- the stage --------------------------------------------------------------------------


def _as_scouts(scouts: pd.DataFrame | Iterable[Scout]) -> list[Scout]:
    if isinstance(scouts, pd.DataFrame):
        return generate.scouts_from_frame(scouts)
    return list(scouts)


def build_capacity_plan(
    scouts: pd.DataFrame | Iterable[Scout],
    unavailability: pd.DataFrame,
    forecast: pd.DataFrame,
    params: Params,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage [3]: ``(capacity_plan, hiring_plan)`` from the team and the forecast (pure)."""
    team = _as_scouts(scouts)
    months = plan_months(params)
    required = _pivot(forecast, "hours_pq", months)
    demand = _pivot(forecast, "hours_p50", months)
    available = split_by_demand(team, scout_monthly_hours(team, unavailability, params), demand)

    cap_rows: dict[str, list] = {c: [] for c in PLAN_SCHEMAS["capacity_plan"].names}
    hire_rows: list[Hire] = []
    for skill in required.columns:
        req = required[skill].to_numpy()
        avail = available[skill].to_numpy()
        hires, hired = plan_skill_hires(skill, req, avail, params, months)
        hire_rows += hires
        for k, month in enumerate(months):
            cap_rows["month"].append(month)
            cap_rows["skill_type"].append(skill)
            cap_rows["required_hours"].append(float(req[k]))
            cap_rows["available_hours"].append(float(avail[k]))
            cap_rows["gap_hours"].append(float(req[k] - avail[k]))
            cap_rows["hired_hours"].append(float(hired[k]))
            cap_rows["gap_after_hires_hours"].append(float(req[k] - avail[k] - hired[k]))

    order = {s: i for i, s in enumerate(required.columns)}
    hire_rows.sort(key=lambda h: (h.month_to_act, h.joins_month, order[h.skill_type], h.hire_type))
    hiring = {c: [getattr(h, c) for h in hire_rows] for c in PLAN_SCHEMAS["hiring_plan"].names}
    return _frame("capacity_plan", cap_rows), _frame("hiring_plan", hiring)


def _pivot(forecast: pd.DataFrame, column: str, months: list[dt.date]) -> pd.DataFrame:
    """``forecast[column]`` as months (rows) x skills (columns, forecast order)."""
    skills = list(dict.fromkeys(forecast["skill_type"]))
    table = forecast.pivot_table(index="month", columns="skill_type", values=column, aggfunc="sum")
    return table.reindex(index=months, columns=skills).fillna(0.0)


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
    Pass ``rng.make_stream(seed, HIRES_STREAM, replication)`` (D-015).
    """
    n = int(hiring_plan["count"].sum()) if len(hiring_plan) else 0
    width = max(3, len(str(n)))
    taken = set(taken_names)
    cost, team = params.cost, params.team
    out: list[Scout] = []
    for row in hiring_plan.itertuples(index=False):
        region = skill_region(row.skill_type)
        full_time = row.hire_type == "full_time"
        for _ in range(int(row.count)):
            name = _unique_name(region, taken, rng)
            taken.add(name)
            lo, hi = (
                (FULL_TIME_WEEKLY_HOURS, FULL_TIME_WEEKLY_HOURS)
                if full_time
                else team.freelance_weekly_hours
            )
            out.append(
                Scout(
                    scout_id=f"H{len(out) + 1:0{width}d}",
                    name=name,
                    employment=row.hire_type,
                    skills=frozenset({row.skill_type}),
                    home_region=region,
                    min_weekly_hours=float(lo),
                    max_weekly_hours=float(hi),
                    former_clubs=frozenset(generate._draw_former_clubs(region, rng)),
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


# --- summaries and I/O edge ----------------------------------------------------------------


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
