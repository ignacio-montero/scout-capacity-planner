"""Stage [5]: the day-by-day simulation of one replication. Spec: DATA_CONTRACTS section 5.

A **time-stepped simulation** (as opposed to a discrete-event one such as
SimPy): a plain loop over the calendar days of the simulated period. Every day
the same steps run in the same order, which makes the model easy to read and
its invariants easy to test. ``simulate_replication(inputs)`` is pure: same
inputs, same result; no file access. The process pool, progress relay and
file writing live in ``pipeline.py`` (the imperative shell).

What one replication is (orchestrator decision, D-023 draft)
------------------------------------------------------------
``sim.seeds = N`` replications vary *luck*, not *decisions*:

* the team (initial scouts) and the fixture calendar come from replication 0;
* requests (arrivals, durations, live-view flags, rework draws), freelancer
  weekly hours and leave are redrawn per replication
  (``generate.generate_world(params, replication=k)``);
* the forecast and capacity plan are computed **once**, from replication 0's
  history: the plan is a decision taken before the year starts, so it must not
  change with the luck of the year. Hires are materialised once
  (``plan.hires_to_scouts`` on stream ``hires``, replication 0); only their
  weekly hours and leave are drawn per replication, on per-hire streams
  (:func:`hire_availability`).

The calendar
------------
The simulation covers ``sim.months`` months from ``SIM_START`` (2027-01-01).

* Full-time scouts offer their weekly hours / 5 (7.5 h) every weekday, 0 on
  weekends and leave days (leave is not made up).
* Freelancers offer their pre-drawn weekly hours spread evenly over the
  weekdays of that week they are available. (The first week starts on the
  Monday before ``SIM_START``; its days before the start are simply not
  simulated, so a partial week offers a pro-rata share.)
* Hires are active from the first day of their ``joined_month`` (only if
  ``team.follow_hiring_plan``; hires joining after the horizon never appear).

One day, in order
-----------------
1. **Weekly bookkeeping** (on the first day and every Monday): backlog
   snapshot for ``weekly.parquet``, progress callback, cancellation check.
2. **Arrivals**: requests received today become tasks (desk review, optional
   live view, write-up). With automation on, desk hours are x (1 - reduction).
   A live view targets the player club's first fixture in
   ``[received + 2, due - 1]``; if there is none (*at risk from day one*), the
   first fixture on or after ``received + 2``, which will make it late.
3. **Re-targeting**: a live view whose fixture has passed without it being
   done moves to the club's next fixture from today on (D-021), unassigned.
4. **Assignment run** on ``assignment.cadence`` days (``daily``: every
   calendar day, weekends included, see ``_is_assignment_day``; ``weekly``:
   Mondays), in the morning, over the window
   ``today .. today + horizon_days - 1``. The pool (D-021) holds every
   unfinished, not-started work item: tasks or desk+write-up bundles per
   ``assignment.unit``, ``depends_on`` = unfinished prerequisites,
   ``current_scout_id`` = the scout it is committed to (cleared when a run
   leaves it unassigned), ``request_scout_ids`` = scouts holding started or
   finished work on the request. Each scout's ``ScoutState`` offers the raw
   calendar hours of the window; started work and owed time in lieu are its
   ``frozen_hours``. One stream, ``assignment``, per replication.
5. **Execution**, per scout:

   a. a live view assigned for today happens: it costs ``live_view_hours``,
      taken from today's hours, the rest (all of it on a weekend) is owed as
      *time in lieu* and repaid from the scout's next working hours;
   b. owed time in lieu is repaid;
   c. the remaining hours go into the scout's queue: started work first (it is
      frozen to them), then committed items earliest-due-first. An item is
      *started* the first time hours go into it and never changes scout again.

   When a desk review finishes with automation on and
   ``request.needs_rework(rework_rate)``, the pre-screen output was unusable:
   the same scout continues with the full original desk hours + the overhead
   (still the desk step, so the write-up stays blocked). A finished task
   unlocks its dependants for the next assignment run.

Assignment happens before execution, so work committed this morning can be
done today, and a request received today can start today. A live view is
done within its fixture day, so it is never "started but unfinished" at an
assignment run; the hours it still costs are owed time in lieu, which is
frozen. (M3 suggested marking a frozen live view's date unavailable; with
this execution model that case cannot occur.)

Known simplifications: the year starts with an empty backlog (a *cold start*:
the first fortnight is slightly easier than steady state); scouts never leave
the team; hours are fluid within a day (no switching cost).
"""

from __future__ import annotations

import datetime as dt
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scout_planner import generate, plan
from scout_planner.assign import assign_with_report
from scout_planner.config import FULL_TIME_WEEKLY_HOURS, Params
from scout_planner.domain import (
    AssignmentWindow,
    Fixture,
    Request,
    Scout,
    ScoutState,
    WorkItem,
    task_id,
)
from scout_planner.errors import RunCancelled
from scout_planner.generate import SIM_START, World, add_months, sim_end, sim_week_starts
from scout_planner.rng import make_stream

EPS = 1e-9  # hours below this are zero (float noise)
ASSIGNMENT_STREAM = "assignment"
# How far ahead to look for a re-targeted live view (fixtures run past the horizon).
RETARGET_LOOKAHEAD_DAYS = 400

OnWeek = Callable[[int, int], None]  # (weeks done, total weeks)
ShouldCancel = Callable[[], bool]


# --- inputs ------------------------------------------------------------------------


@dataclass(frozen=True)
class SimInputs:
    """Everything one replication needs, as domain objects. Built by :func:`prepare_replication`.

    ``scouts`` holds the initial team plus any hires (``joined_month > 0``);
    ``weekly_hours`` maps ``(scout_id, week_start)`` to offered hours;
    ``requests`` are the simulated period's requests only.
    """

    params: Params
    replication: int
    scouts: tuple[Scout, ...]
    unavailability: Mapping[str, frozenset[dt.date]]
    weekly_hours: Mapping[tuple[str, dt.date], float]
    fixtures: tuple[Fixture, ...]
    requests: tuple[Request, ...]
    at_risk: Mapping[str, bool] = field(default_factory=dict)


def _unavailability_map(*frames: pd.DataFrame) -> dict[str, frozenset[dt.date]]:
    days: defaultdict[str, set[dt.date]] = defaultdict(set)
    for df in frames:
        for sid, day in zip(df["scout_id"], df["date"], strict=True):
            days[sid].add(day)
    return {sid: frozenset(d) for sid, d in days.items()}


def _weekly_map(*frames: pd.DataFrame) -> dict[tuple[str, dt.date], float]:
    out: dict[tuple[str, dt.date], float] = {}
    for df in frames:
        for sid, week, hours in zip(df["scout_id"], df["week_start"], df["hours"], strict=True):
            out[sid, week] = float(hours)
    return out


def materialise_hires(params: Params, world: World, hiring_plan: pd.DataFrame) -> list[Scout]:
    """The hiring table as ``Scout`` objects (once per run: the same hires in every replication).

    Empty when ``team.follow_hiring_plan`` is off. Hires that would join after
    the simulated horizon are dropped: they never work in this run.
    """
    if not params.team.follow_hiring_plan or hiring_plan.empty:
        return []
    hires = plan.hires_to_scouts(
        hiring_plan,
        params,
        make_stream(params.sim.seed, plan.HIRES_STREAM, 0),
        taken_names=world.scouts["name"],
    )
    return [h for h in hires if h.joined_month < params.sim.months]


def hire_availability(
    params: Params, hire: Scout, replication: int
) -> tuple[frozenset[dt.date], dict[tuple[str, dt.date], float]]:
    """One hire's days off and weekly hours for one replication.

    Same rules as the initial team's (``generate``): full-timers get
    ``leave_days_per_year`` weekdays of leave in blocks (pro rata for the
    horizon) and offer 37.5 h a week; freelancers get a few random days off
    and pre-drawn weekly hours on ``team.freelance_weekly_hours``, rounded to
    half hours. Leave is placed over the whole period; days before the hire
    joins go unused, so their leave is pro rata in expectation.

    Each hire draws from its **own** streams, ``hire_leave.<id>`` and
    ``hire_hours.<id>`` (replication ``k``), so hiring one more person, or one
    fewer, never reshuffles anyone else's luck (common random numbers, D-015).
    """
    seed, sid = params.sim.seed, hire.scout_id
    start, end = SIM_START, sim_end(params)
    weeks = sim_week_starts(params)
    all_days = [start + dt.timedelta(days=k) for k in range((end - start).days + 1)]
    share_of_year = params.sim.months / 12
    leave_rng = make_stream(seed, f"hire_leave.{sid}", replication)
    hours_rng = make_stream(seed, f"hire_hours.{sid}", replication)
    if hire.is_freelance:
        n_off = int(leave_rng.poisson(generate.FREELANCE_DAYS_OFF_PER_YEAR * share_of_year))
        picks = leave_rng.permutation(len(all_days))[: min(n_off, len(all_days))]
        days_off = frozenset(all_days[i] for i in picks)
        lo, hi = params.team.freelance_weekly_hours
        step = generate.FREELANCE_HOURS_STEP
        draws = lo + hours_rng.random(len(weeks)) * (hi - lo)
        hours = np.clip(np.round(draws / step) * step, lo, hi)
    else:
        weekdays = [d for d in all_days if d.weekday() < 5]
        total = round(params.team.leave_days_per_year * share_of_year)
        days_off = frozenset(generate._full_time_leave(weekdays, total, leave_rng))
        hours = np.full(len(weeks), FULL_TIME_WEEKLY_HOURS)
    return days_off, {(sid, w): float(h) for w, h in zip(weeks, hours, strict=True)}


def prepare_replication(
    params: Params, world: World, hiring_plan: pd.DataFrame, replication: int
) -> SimInputs:
    """Inputs of replication ``replication`` from its world and the run's hiring table.

    ``world`` must be this replication's world (``generate_world(params,
    replication=k)``; replication 0 is the one written to ``raw/``). Its
    scouts are the same in every replication, so the hires are too.
    """
    hires = materialise_hires(params, world, hiring_plan)
    unavailability = _unavailability_map(world.scout_unavailability)
    weekly = _weekly_map(world.scout_weekly_hours)
    for hire in hires:
        days_off, hours = hire_availability(params, hire, replication)
        unavailability[hire.scout_id] = days_off
        weekly.update(hours)
    future = world.requests[world.requests["period"] == "future"]
    return SimInputs(
        params=params,
        replication=replication,
        scouts=(*generate.scouts_from_frame(world.scouts), *hires),
        unavailability=unavailability,
        weekly_hours=weekly,
        fixtures=tuple(generate.fixtures_from_frame(world.fixtures)),
        requests=tuple(generate.requests_from_frame(future)),
        at_risk=dict(zip(future["request_id"], future["at_risk_day_one"], strict=True)),
    )


def scout_calendar(
    scout: Scout,
    weekly_hours: Mapping[tuple[str, dt.date], float],
    unavailable: frozenset[dt.date],
    weeks: Iterable[dt.date],
    first: dt.date,
    last: dt.date,
) -> dict[dt.date, float]:
    """Hours the scout offers on each working day in ``[first, last]`` (days off omitted = 0).

    Full-time: weekly hours / 5 on every available weekday (leave is not made
    up). Freelance: weekly hours spread evenly over the week's available
    weekdays. Weekends offer nothing.
    """
    out: dict[dt.date, float] = {}
    for week in weeks:
        hours = weekly_hours.get((scout.scout_id, week), 0.0)
        weekdays = [week + dt.timedelta(days=k) for k in range(5)]
        available = [d for d in weekdays if d not in unavailable]
        if not available or hours <= 0:
            continue
        per_day = hours / len(available) if scout.is_freelance else hours / 5
        for d in available:
            if first <= d <= last:
                out[d] = per_day
    return out


# --- mutable bookkeeping (private: the policies only ever see frozen snapshots) -----


@dataclass(slots=True)
class _Task:
    task_id: str
    request_id: str
    kind: str  # desk | live | writeup
    remaining: float
    prerequisites: tuple[str, ...] = ()
    fixture: Fixture | None = None  # live view only
    base_desk_hours: float = 0.0  # desk only: original hours, for rework
    rework_done: bool = False
    done: bool = False
    completed: dt.date | None = None


@dataclass(slots=True)
class _Item:
    item_id: str
    request_id: str
    kind: str  # desk | live | writeup | bundle
    task_ids: tuple[str, ...]
    depends_on_tasks: tuple[str, ...]
    scout_id: str | None = None
    started: bool = False
    done: bool = False


@dataclass(slots=True)
class _Req:
    request: Request
    at_risk: bool
    item_ids: list[str]
    scouts: set[str] = field(default_factory=set)
    completed: dt.date | None = None


# --- result ------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicationResult:
    """What one replication produced; ``metrics.py`` turns it into the result tables.

    * ``outcomes``: one row per arrived request (completed or still open).
    * ``weekly``: backlog snapshot at the start of every week.
    * ``scouts``: per scout that was ever active: hours offered and worked.
    * ``diagnostics``: assignment-run and solver counters.
    * ``work_log`` / ``day_log``: full trace, only with ``trace=True`` (tests).
    """

    replication: int
    outcomes: pd.DataFrame
    weekly: pd.DataFrame
    scouts: pd.DataFrame
    diagnostics: dict[str, float]
    work_log: pd.DataFrame | None = None
    day_log: pd.DataFrame | None = None


# --- the simulation ----------------------------------------------------------------


class _Simulation:
    """One replication's state and daily steps. Use :func:`simulate_replication`."""

    def __init__(self, inputs: SimInputs, trace: bool) -> None:
        p = self.params = inputs.params
        self.inputs = inputs
        self.trace = trace
        self.start, self.end = SIM_START, sim_end(p)
        self.days = [
            self.start + dt.timedelta(days=k) for k in range((self.end - self.start).days + 1)
        ]
        self.week_starts = sim_week_starts(p)
        self.horizon = p.assignment.horizon_days
        self.live_hours = p.demand.live_view_hours
        auto = p.automation
        self.desk_factor = (1.0 - auto.desk_reduction) if auto.enabled else 1.0
        self.rework_rate = auto.rework_rate if auto.enabled else 0.0
        self.rng = make_stream(p.sim.seed, ASSIGNMENT_STREAM, inputs.replication)

        self.scouts: dict[str, Scout] = {}
        self.active_from: dict[str, dt.date] = {}
        for s in inputs.scouts:
            if s.joined_month >= p.sim.months:
                continue
            self.scouts[s.scout_id] = s
            self.active_from[s.scout_id] = add_months(SIM_START, s.joined_month)
        self.unavailable = {sid: inputs.unavailability.get(sid, frozenset()) for sid in self.scouts}
        self.calendar = {
            sid: scout_calendar(
                s,
                inputs.weekly_hours,
                self.unavailable[sid],
                self.week_starts,
                max(self.start, self.active_from[sid]),
                self.end,
            )
            for sid, s in self.scouts.items()
        }
        self.by_club = generate.fixtures_by_club(inputs.fixtures)
        self.arrivals: defaultdict[dt.date, list[Request]] = defaultdict(list)
        for r in sorted(inputs.requests, key=lambda r: (r.received_date, r.request_id)):
            if self.start <= r.received_date <= self.end:
                self.arrivals[r.received_date].append(r)

        self.tasks: dict[str, _Task] = {}
        self.items: dict[str, _Item] = {}
        self.item_of_task: dict[str, str] = {}
        self.requests: dict[str, _Req] = {}
        self.open_items: dict[str, _Item] = {}  # not done
        self.open_requests: set[str] = set()
        self.queue: dict[str, set[str]] = {sid: set() for sid in self.scouts}
        self.lieu: dict[str, float] = dict.fromkeys(self.scouts, 0.0)
        self.offered: dict[str, float] = dict.fromkeys(self.scouts, 0.0)
        self.worked: dict[str, float] = dict.fromkeys(self.scouts, 0.0)

        self.weekly_rows: list[tuple] = []
        self.work_rows: list[tuple] = []
        self.day_rows: list[tuple] = []
        self.diag: dict[str, float] = {
            "assignment_runs": 0,
            "optimiser_solves": 0,
            "wall_clock_hits": 0,
            "fallbacks": 0,
            "solve_seconds_total": 0.0,
            "solve_seconds_max": 0.0,
            "assign_seconds_total": 0.0,
            "max_pool_size": 0,
            "live_view_retargets": 0,
            "reworks": 0,
        }

    # -- the loop --

    def run(self, on_week: OnWeek | None, should_cancel: ShouldCancel | None) -> ReplicationResult:
        started = time.perf_counter()
        n_weeks = len(self.week_starts)
        week_index = 0
        for day in self.days:
            # One "week tick" on the first day and on every Monday: week_starts[0]
            # is the Monday on or before SIM_START, so the ticks match it 1:1.
            if day == self.start or day.weekday() == 0:
                if should_cancel is not None and should_cancel():
                    raise RunCancelled
                if on_week is not None and week_index > 0:
                    on_week(week_index, n_weeks)
                self._snapshot(self.week_starts[week_index], day)
                week_index += 1
            self._arrive(day)
            self._retarget(day)
            if self._is_assignment_day(day):
                self._assign(day)
            self._execute(day)
        if on_week is not None:
            on_week(n_weeks, n_weeks)
        self.diag["sim_seconds"] = time.perf_counter() - started
        return self._result()

    def _is_assignment_day(self, day: dt.date) -> bool:
        """``daily``: every calendar day; ``weekly``: Mondays.

        Daily includes weekends on purpose. The optimiser prices a live view as
        *perishable* when its fixture is before the next run, and for ``daily``
        it takes the next run to be tomorrow (``optimiser.next_run_date``). With
        weekday-only runs, Friday's run saw Saturday/Sunday fixtures as
        non-perishable, left them for "tomorrow", and no run came before the
        match: in a default-parameter year 934 live views missed their fixture
        (vs 27 with weekend runs). Scouts offer no desk hours at weekends, so a
        weekend run only changes who attends weekend fixtures and the plan.
        """
        return self.params.assignment.cadence == "daily" or day.weekday() == 0

    def _active(self, day: dt.date) -> list[str]:
        return sorted(sid for sid, first in self.active_from.items() if first <= day)

    # -- step 1: weekly snapshot --

    def _snapshot(self, week_start: dt.date, day: dt.date) -> None:
        open_hours = sum(
            t.remaining
            for rid in self.open_requests
            for iid in self.requests[rid].item_ids
            for t in (self.tasks[x] for x in self.items[iid].task_ids)
            if not t.done
        )
        late = sum(1 for rid in self.open_requests if self.requests[rid].request.due_date < day)
        active = self._active(day)
        n_ft = sum(1 for sid in active if not self.scouts[sid].is_freelance)
        self.weekly_rows.append(
            (week_start, len(self.open_requests), float(open_hours), late, n_ft, len(active) - n_ft)
        )

    # -- step 2: arrivals --

    def _first_fixture(self, club: str, earliest: dt.date) -> Fixture | None:
        return generate.next_fixture(
            self.by_club, club, earliest, earliest + dt.timedelta(days=RETARGET_LOOKAHEAD_DAYS)
        )

    def _arrive(self, day: dt.date) -> None:
        for r in self.arrivals.get(day, ()):
            rid = r.request_id
            desk_id, live_id, wu_id = (task_id(rid, k) for k in ("desk", "live", "writeup"))
            desk = _Task(
                desk_id, rid, "desk", r.desk_hours * self.desk_factor, base_desk_hours=r.desk_hours
            )
            tasks = [desk]
            if r.needs_live_view:
                window = generate.live_view_window(r.received_date, r.due_date)
                fixture = generate.next_fixture(self.by_club, r.player_club, *window)
                if fixture is None:  # at risk from day one: it will be late
                    fixture = self._first_fixture(r.player_club, window[0])
                tasks.append(_Task(live_id, rid, "live", self.live_hours, fixture=fixture))
            prereqs = tuple(t.task_id for t in tasks)
            tasks.append(_Task(wu_id, rid, "writeup", r.writeup_hours, prerequisites=prereqs))
            for t in tasks:
                self.tasks[t.task_id] = t

            if self.params.assignment.unit == "bundle":
                groups = [("bundle", (desk_id, wu_id))]
                if r.needs_live_view:
                    groups.append(("live", (live_id,)))
            else:
                groups = [(t.kind, (t.task_id,)) for t in tasks]
            item_ids = []
            for kind, tids in groups:
                iid = task_id(rid, "bundle") if kind == "bundle" else tids[0]
                deps = tuple(
                    p for tid in tids for p in self.tasks[tid].prerequisites if p not in tids
                )
                item = _Item(iid, rid, kind, tids, deps)
                self.items[iid] = item
                self.open_items[iid] = item
                for tid in tids:
                    self.item_of_task[tid] = iid
                item_ids.append(iid)
            self.requests[rid] = _Req(r, bool(self.inputs.at_risk.get(rid, False)), item_ids)
            self.open_requests.add(rid)

    # -- step 3: re-target live views whose fixture passed --

    def _retarget(self, day: dt.date) -> None:
        for item in self.open_items.values():
            if item.kind != "live":
                continue
            t = self.tasks[item.task_ids[0]]
            if t.fixture is None or t.fixture.date >= day:
                continue
            club = self.requests[item.request_id].request.player_club
            t.fixture = self._first_fixture(club, day)
            self._unassign(item)
            self.diag["live_view_retargets"] += 1

    # -- step 4: the assignment run --

    def _unassign(self, item: _Item) -> None:
        if item.scout_id is not None:
            self.queue[item.scout_id].discard(item.item_id)
            item.scout_id = None

    def _item_hours(self, item: _Item) -> float:
        return sum(self.tasks[t].remaining for t in item.task_ids if not self.tasks[t].done)

    def _work_item(self, item: _Item) -> WorkItem | None:
        req = self.requests[item.request_id]
        r = req.request
        fixed_date = region = None
        if item.kind == "live":
            fixture = self.tasks[item.task_ids[0]].fixture
            if fixture is None:  # no fixture left on the calendar: can never be done
                return None
            fixed_date, region = fixture.date, fixture.region
        depends = sorted(
            {self.item_of_task[t] for t in item.depends_on_tasks if not self.tasks[t].done}
        )
        return WorkItem(
            item_id=item.item_id,
            request_id=item.request_id,
            kind=item.kind,  # type: ignore[arg-type]
            task_ids=item.task_ids,
            hours=self._item_hours(item),
            skill_type=r.skill_type,
            received_date=r.received_date,
            due_date=r.due_date,
            conflict_clubs=r.conflict_clubs,
            current_scout_id=item.scout_id,
            fixed_date=fixed_date,
            region=region,
            depends_on=tuple(depends),
            request_scout_ids=frozenset(req.scouts),
        )

    def _scout_state(self, sid: str, window: AssignmentWindow) -> ScoutState:
        cal = self.calendar[sid]
        frozen = self.lieu[sid] + sum(
            self._item_hours(self.items[iid]) for iid in self.queue[sid] if self.items[iid].started
        )
        off = self.unavailable[sid]
        return ScoutState(
            scout=self.scouts[sid],
            hours_by_day={d: cal.get(d, 0.0) for d in window.days},
            frozen_hours=frozen,
            unavailable_dates=frozenset(d for d in window.days if d in off),
        )

    def _assign(self, day: dt.date) -> None:
        window = AssignmentWindow.starting(day, self.horizon)
        states = [self._scout_state(sid, window) for sid in self._active(day)]
        pool = [
            w
            for item in self.open_items.values()
            if not item.started and (w := self._work_item(item)) is not None
        ]
        t0 = time.perf_counter()
        outcome = assign_with_report(
            pool, states, window, self.params.assignment, self.rng, cost=self.params.cost
        )
        self.diag["assign_seconds_total"] += time.perf_counter() - t0
        self.diag["assignment_runs"] += 1
        self.diag["max_pool_size"] = max(self.diag["max_pool_size"], len(pool))
        solve = outcome.solve
        if solve is not None and solve.status != "EMPTY":
            self.diag["optimiser_solves"] += 1
            self.diag["wall_clock_hits"] += int(solve.hit_wall_clock)
            self.diag["fallbacks"] += int(solve.fell_back_to_edf)
            self.diag["solve_seconds_total"] += solve.wall_time_s
            self.diag["solve_seconds_max"] = max(self.diag["solve_seconds_max"], solve.wall_time_s)

        chosen = {a.item_id: a.scout_id for a in outcome.assignments}
        for w in pool:
            item = self.items[w.item_id]
            new = chosen.get(w.item_id)
            if new == item.scout_id:
                continue
            self._unassign(item)  # left unassigned -> commitment cleared (D-021)
            if new is not None:
                item.scout_id = new
                self.queue[new].add(item.item_id)

    # -- step 5: execution --

    def _log_work(self, day: dt.date, sid: str, task: _Task, hours: float) -> None:
        self.worked[sid] += hours
        self.requests[task.request_id].scouts.add(sid)
        if self.trace:
            fixture = task.fixture
            self.work_rows.append(
                (
                    day,
                    sid,
                    self.item_of_task[task.task_id],
                    task.task_id,
                    task.request_id,
                    task.kind,
                    hours,
                    fixture.fixture_id if fixture is not None else None,
                    fixture.date if fixture is not None else None,
                )
            )

    def _finish_task(self, task: _Task, day: dt.date) -> bool:
        """Mark a task whose hours ran out as done; False if it turned into rework instead."""
        if task.kind == "desk" and not task.rework_done:
            request = self.requests[task.request_id].request
            if request.needs_rework(self.rework_rate):
                task.rework_done = True
                task.remaining = task.base_desk_hours + self.params.automation.rework_overhead_hours
                self.diag["reworks"] += 1
                if task.remaining > EPS:
                    return False
        task.done, task.remaining, task.completed = True, 0.0, day
        item = self.items[self.item_of_task[task.task_id]]
        if all(self.tasks[t].done for t in item.task_ids):
            item.done = True
            self.open_items.pop(item.item_id, None)
            if item.scout_id is not None:
                self.queue[item.scout_id].discard(item.item_id)
        if task.kind == "writeup":
            req = self.requests[task.request_id]
            req.completed = day
            self.open_requests.discard(task.request_id)
        return True

    def _queue_order(self, item: _Item) -> tuple:
        r = self.requests[item.request_id].request
        return (not item.started, r.due_date, r.received_date, item.item_id)

    def _execute(self, day: dt.date) -> None:
        for sid in self._active(day):
            offered = self.calendar[sid].get(day, 0.0)
            self.offered[sid] += offered
            avail = offered
            queue = [self.items[iid] for iid in self.queue[sid]]

            # a. live views on today's fixtures (a live view takes the day, D-008)
            for item in sorted((i for i in queue if i.kind == "live"), key=lambda i: i.item_id):
                task = self.tasks[item.task_ids[0]]
                if task.fixture is None or task.fixture.date != day:
                    continue
                from_today = min(avail, task.remaining)
                avail -= from_today
                self.lieu[sid] += task.remaining - from_today  # time in lieu, repaid later
                item.started = True
                self._log_work(day, sid, task, task.remaining)
                self._finish_task(task, day)

            # b. repay time in lieu
            repay = min(avail, self.lieu[sid])
            avail -= repay
            self.lieu[sid] -= repay

            # c. the queue: started work first, then earliest due first
            for item in sorted((i for i in queue if i.kind != "live"), key=self._queue_order):
                for tid in item.task_ids:
                    task = self.tasks[tid]
                    while not task.done and avail > EPS:
                        hours = min(avail, task.remaining)
                        avail -= hours
                        task.remaining -= hours
                        item.started = True
                        self._log_work(day, sid, task, hours)
                        if task.remaining <= EPS:
                            self._finish_task(task, day)
                    if not task.done:
                        break
                if avail <= EPS:
                    break
            if self.trace:
                self.day_rows.append((day, sid, offered, offered - avail, self.lieu[sid]))

    # -- result tables --

    def _result(self) -> ReplicationResult:
        rows = []
        for rid in sorted(self.requests):
            req = self.requests[rid]
            r = req.request
            done = req.completed
            rows.append(
                {
                    "request_id": rid,
                    "received_date": r.received_date,
                    "due_date": r.due_date,
                    "skill_type": r.skill_type,
                    "needs_live_view": r.needs_live_view,
                    "completed_date": done,
                    "turnaround_days": float((done - r.received_date).days)
                    if done is not None
                    else np.nan,
                    "on_time": done is not None and r.is_on_time(done),
                    "at_risk_day_one": req.at_risk,
                    "scouts_involved": len(req.scouts),
                    "scored": r.due_date <= self.end,
                }
            )
        outcomes = pd.DataFrame(rows, columns=OUTCOME_COLUMNS)
        weekly = pd.DataFrame(self.weekly_rows, columns=WEEKLY_COLUMNS)
        scouts = pd.DataFrame(
            [
                {
                    "scout_id": sid,
                    "employment": s.employment,
                    "joined_month": s.joined_month,
                    "monthly_salary": s.monthly_salary,
                    "hourly_rate": s.hourly_rate,
                    "hours_offered": self.offered[sid],
                    "hours_worked": self.worked[sid],
                    "lieu_owed_at_end": self.lieu[sid],
                }
                for sid, s in sorted(self.scouts.items())
            ],
            columns=SCOUT_COLUMNS,
        )
        work_log = day_log = None
        if self.trace:
            work_log = pd.DataFrame(self.work_rows, columns=WORK_LOG_COLUMNS)
            day_log = pd.DataFrame(self.day_rows, columns=DAY_LOG_COLUMNS)
        return ReplicationResult(
            self.inputs.replication, outcomes, weekly, scouts, dict(self.diag), work_log, day_log
        )


OUTCOME_COLUMNS = (
    "request_id",
    "received_date",
    "due_date",
    "skill_type",
    "needs_live_view",
    "completed_date",
    "turnaround_days",
    "on_time",
    "at_risk_day_one",
    "scouts_involved",
    "scored",
)
WEEKLY_COLUMNS = (
    "week_start",
    "open_requests",
    "open_hours",
    "late_requests",
    "team_full_time",
    "team_freelance",
)
SCOUT_COLUMNS = (
    "scout_id",
    "employment",
    "joined_month",
    "monthly_salary",
    "hourly_rate",
    "hours_offered",
    "hours_worked",
    "lieu_owed_at_end",
)
WORK_LOG_COLUMNS = (
    "date",
    "scout_id",
    "item_id",
    "task_id",
    "request_id",
    "kind",
    "hours",
    "fixture_id",
    "fixture_date",
)
DAY_LOG_COLUMNS = ("date", "scout_id", "offered", "used", "lieu_owed")


def simulate_replication(
    inputs: SimInputs,
    *,
    on_week: OnWeek | None = None,
    should_cancel: ShouldCancel | None = None,
    trace: bool = False,
) -> ReplicationResult:
    """Simulate one replication day by day (pure given ``inputs``).

    ``on_week(done, total)`` is called once per simulated week and at the end;
    ``should_cancel()`` is polled at the start of every week and raises
    :class:`RunCancelled` when it returns True. ``trace=True`` also returns
    the per-day and per-hour logs used by the invariant tests.
    """
    return _Simulation(inputs, trace).run(on_week, should_cancel)


def simulate_world(
    params: Params,
    world: World,
    hiring_plan: pd.DataFrame,
    replication: int = 0,
    **kwargs: object,
) -> ReplicationResult:
    """Convenience: :func:`prepare_replication` + :func:`simulate_replication`."""
    return simulate_replication(
        prepare_replication(params, world, hiring_plan, replication),
        **kwargs,  # type: ignore[arg-type]
    )


def replication_world(params: Params, replication: int, world0: World | None = None) -> World:
    """Replication ``k``'s world: ``world0`` (as written to ``raw/``) for 0, else regenerated."""
    if replication == 0 and world0 is not None:
        return world0
    return generate.generate_world(params, replication=replication)


__all__: Sequence[str] = (
    "ReplicationResult",
    "SimInputs",
    "materialise_hires",
    "prepare_replication",
    "replication_world",
    "scout_calendar",
    "simulate_replication",
    "simulate_world",
)
