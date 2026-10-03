"""CP-SAT assignment policy: decide all items of one assignment run at once.

Where the greedy policies decide item by item, this one states the whole
problem as a *constraint program* and lets OR-Tools' CP-SAT solver search for
the best answer. The model, per assignment run:

Decision variables
    ``x[i, s] in {0, 1}``: item ``i`` goes to scout ``s``. Created **only for
    eligible pairs** (``eligibility.py``: skill, conflict of interest,
    precedence, live-view date / region / availability, and the item fits the
    scout's free hours on its own). Pruning in Python keeps the model small:
    ~400 items x 60 scouts is 24,000 pairs, but only a few thousand are allowed.

Hard constraints
    * each item goes to at most one scout (unassigned items roll over);
    * **due-date capacity** per scout and window day ``d``: frozen hours +
      hours of assigned items due by ``d`` <= hours offered up to ``d``. This
      is the condition under which one scout working their queue
      earliest-due-first finishes everything on time (see
      ``eligibility.py``). Its last day is the plain window-total cap. Without
      the per-day part, the model believed 10 h due tomorrow fit in a 37.5 h
      week and counted it as on time;
    * at most one live view per scout per date (a live view takes the day, D-008).

Objective (minimise, in cost units)
    * **lateness**: ``cost.late_penalty x weights.lateness x urgency`` per
      *request* left (partly) unassigned, shared across the request's pool
      items by hours. One late request costs about one late penalty however
      many tasks it has. See :func:`urgency` and :func:`lateness_penalties`.
    * **cost**: ``weights.cost x freelance hours x the freelancer's hourly
      rate``. Salaried hours cost 0 here: salaries are paid whether the scout
      is busy or not (a *sunk cost*), so only freelance hours change the bill.
    * **continuity**: ``weights.continuity`` per extra scout on one request (a
      desk review and a write-up by different scouts means a hand-over).
      Scouts already on the request outside the pool come in through
      ``WorkItem.request_scout_ids``.
    * **churn**: ``weights.churn`` per item taken away from its
      ``current_scout_id`` (assigned earlier, not started), whether moved to
      another scout or dropped back to the pool. Without the "dropped" half,
      un-assigning would be a free way round the penalty. Changing the plan is
      allowed, but must pay for itself (*plan nervousness* control, D-010).
      A move *forced* by the current scout no longer being able to take the
      item pays nothing extra (every option costs the same).

Hours are integers in hundredths of an hour (``eligibility.HOURS_SCALE``),
work rounded up and capacity down, so every solution is also valid in exact
float arithmetic and passes ``validate_assignments(check_due_dates=True)``.

Determinism vs wall-clock time
    Same inputs + same ``rng`` state must give the same assignments, or two
    runs of one ``params.yaml`` would differ. A wall-clock time limit breaks
    that: how far the search gets in 1 s depends on the machine and on what
    else is running. So the solver runs with

    * ``num_workers = 1``: a multi-threaded search is a race between threads
      and is not reproducible;
    * ``random_seed`` drawn from ``rng`` (the run's assignment stream, D-015);
    * ``max_deterministic_time = time_limit_s``: CP-SAT's internal work
      counter, which is the same on every machine. One unit is roughly one
      second on a laptop, not exactly (measured in M3: 1.0 unit = ~0.8 s on
      the dev machine for 400 items x 60 scouts);
    * ``max_time_in_seconds = max(10 x limit, limit + 5 s)`` as a backstop for
      a machine that is very slow or overloaded. It is set far above the
      deterministic limit so it practically never fires; if it does, that
      solve is no longer reproducible and ``SolveReport.hit_wall_clock`` says
      so, so a run can count and report it.

    Inputs are also sorted by id on entry, so the model (and the answer) does
    not depend on the order of the lists passed in.

    Rejected: wall-clock limit only (simple, but not reproducible); several
    workers (faster on big models, not reproducible); no limit at all (proves
    optimality on small days but can stall a whole simulated year on a busy one).

Warm start: the EDF answer, trimmed to what the model allows, is passed as a
complete, feasible *solution hint* (continuity helpers included), so the
solver has a first solution almost at once and only searches for
improvements. Measured: with a partial hint, a 0.1 s budget on 400 items ended
with no solution at all. If the solver is still empty-handed when stopped
(status ``UNKNOWN``), the policy falls back to EDF and logs a warning.
``INFEASIBLE`` or ``MODEL_INVALID`` raise: assigning nothing is always
feasible, so either one means a bug in the model.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

from scout_planner.assign.eligibility import (
    check_inputs,
    deadline_index,
    eligible_scouts,
    free_by_day,
    planned_date,
    scaled_hours,
)
from scout_planner.assign.greedy import edf
from scout_planner.config import FULL_TIME_WEEKLY_HOURS, AssignmentParams, CostParams
from scout_planner.domain import Assignment, AssignmentWindow, Scout, ScoutState, WorkItem

log = logging.getLogger(__name__)

OBJECTIVE_SCALE = 100  # objective terms -> integers, keeping two decimals
WALL_CLOCK_FACTOR = 10.0  # backstop: max(factor x limit, limit + extra)
WALL_CLOCK_MIN_EXTRA_S = 5.0

DAY_HOURS = FULL_TIME_WEEKLY_HOURS / 5  # 7.5 h: one working day, for slack in days
URGENCY_ZERO_SLACK = 1.0  # last chance to be on time: one full late penalty
URGENCY_OVERDUE = 1.25  # already late: ranked above any still-savable request
URGENCY_LOST_FIXTURE = URGENCY_OVERDUE  # a live view whose match is gone after this run
RUN_INTERVAL_DAYS = {"daily": 1, "weekly": 7}  # days until the next assignment run


def wall_clock_cap(time_limit_s: float) -> float:
    """The wall-clock backstop for one solve, in seconds."""
    return max(WALL_CLOCK_FACTOR * time_limit_s, time_limit_s + WALL_CLOCK_MIN_EXTRA_S)


# --- lateness -------------------------------------------------------------------------


def remaining_work_days(items: Sequence[WorkItem], today: dt.date) -> int:
    """Lower bound on the days one request still needs, from its items in the pool.

    Two bounds, take the larger:

    * all remaining hours at one full-time day (7.5 h) per day;
    * if a live view is pending: days until the fixture, plus the work that
      waits for it (the write-up), which cannot start before.

    Work started outside the pool and calendar gaps (weekends) are ignored:
    this is a cheap urgency signal, not a schedule.
    """
    days = math.ceil(sum(i.hours for i in items) / DAY_HOURS)
    for live in items:
        if live.fixed_date is not None and live.fixed_date >= today:
            after = sum(i.hours for i in items if live.item_id in i.depends_on)
            until_fixture = (live.fixed_date - today).days + 1
            days = max(days, until_fixture + math.ceil(after / DAY_HOURS))
    return days


def slack_days(due_date: dt.date, today: dt.date, work_days: int) -> int:
    """Calendar days to spare: days left including today, minus days of work left.

    Due today with one day of work left -> slack 0 (finish today, still on time).
    """
    return (due_date - today).days + 1 - work_days


def urgency(slack: int, *, overdue: bool) -> float:
    """Share of a late penalty at stake if the request waits: 0 < urgency <= 1.25.

    ``1 / (1 + slack)`` for slack >= 0: 1.0 at zero slack (the last chance to
    be on time is now), 0.5 with one day to spare, 0.1 with nine. A hyperbola
    rather than a straight line because a day of slack matters much more when
    there are two left than when there are twelve. Negative slack (can no
    longer make it) counts as zero slack.

    Overdue requests get 1.25: above any request that can still be saved, so
    old work is not pushed back indefinitely by a stream of new urgent work.
    This lowers the risk of starvation; it does not rule it out (a large
    overdue item can still lose to several small urgent ones competing for
    the same hours, or to a freelancer's cost).
    """
    if overdue:
        return URGENCY_OVERDUE
    return URGENCY_ZERO_SLACK / (1 + max(slack, 0))


def next_run_date(window: AssignmentWindow, cfg: AssignmentParams) -> dt.date:
    """When the next assignment run happens, derived from ``assignment.cadence``."""
    return window.start + dt.timedelta(days=RUN_INTERVAL_DAYS[cfg.cadence])


def is_perishable(item: WorkItem, next_run: dt.date) -> bool:
    """A live view whose fixture is before the next run: assign it now or lose the match."""
    return item.fixed_date is not None and item.fixed_date < next_run


def lateness_penalties(
    pool: Sequence[WorkItem],
    today: dt.date,
    next_run: dt.date,
    late_penalty: float,
    weight: float,
) -> dict[str, float]:
    """Cost units at stake per item if it is left unassigned this run.

    Per request: ``late_penalty x weight x urgency(request)``, shared across
    the request's pool items in proportion to their hours. Sharing (rather
    than charging each item in full) keeps one request worth about one late
    penalty, so a request with a live view (three tasks) does not count more
    than one without (two tasks).

    Exception, *perishable* live views (:func:`is_perishable`): leaving one
    unassigned loses the match, which makes the request late almost surely,
    so it carries a full ``URGENCY_LOST_FIXTURE`` penalty of its own, whatever
    the request's slack.
    """
    by_request: defaultdict[str, list[WorkItem]] = defaultdict(list)
    for item in pool:
        by_request[item.request_id].append(item)
    result: dict[str, float] = {}
    for items in by_request.values():
        due = min(i.due_date for i in items)
        slack = slack_days(due, today, remaining_work_days(items, today))
        at_stake = late_penalty * weight * urgency(slack, overdue=due < today)
        total_hours = sum(i.hours for i in items)
        for i in items:
            if is_perishable(i, next_run):
                result[i.item_id] = late_penalty * weight * URGENCY_LOST_FIXTURE
            else:
                result[i.item_id] = at_stake * i.hours / total_hours
    return result


def hourly_cost(scout: Scout) -> float:
    """Marginal cost of one more hour: 0 for salaried scouts, the rate for freelancers."""
    if not scout.is_freelance:
        return 0.0
    if scout.hourly_rate is None:
        raise ValueError(f"freelance scout {scout.scout_id} has no hourly_rate")
    return scout.hourly_rate


# --- the solve ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolveReport:
    """What one optimiser call did. ``assignments`` is the policy's answer.

    ``objective`` and ``best_bound`` are in cost units; ``gap`` is
    ``(objective - best_bound) / max(1, |objective|)``: 0 means proven optimal.
    A simulation should count ``hit_wall_clock`` (that solve was not
    reproducible) and ``fell_back_to_edf`` per run.
    """

    assignments: list[Assignment]
    status: str  # CP-SAT status name, or "EMPTY" when there was nothing to decide
    objective: float | None
    best_bound: float | None
    gap: float | None
    wall_time_s: float
    deterministic_time: float
    fell_back_to_edf: bool = False
    hit_wall_clock: bool = False


def solve_assignment(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
    *,
    cost: CostParams,
) -> SolveReport:
    """Build and solve the CP-SAT model; see the module docstring for the model."""
    # Always exactly one draw per call, so the stream advances the same way
    # whether or not there is anything to solve today.
    seed = int(rng.integers(0, 2**31 - 1))
    check_inputs(pool, scouts, window)
    scouts = sorted(scouts, key=lambda s: s.scout_id)
    pool = sorted(pool, key=lambda i: i.item_id)

    items = {i.item_id: i for i in pool}
    states = {s.scout_id: s for s in scouts}
    free = {s.scout_id: free_by_day(s, window) for s in scouts}
    units = {i.item_id: scaled_hours(i.hours) for i in pool}
    deadline: dict[tuple[str, str], int] = {}
    for item_id, allowed in eligible_scouts(pool, scouts, window).items():
        for state in allowed:
            j = deadline_index(items[item_id], free[state.scout_id], window)
            if j is not None:
                deadline[item_id, state.scout_id] = j
    if not deadline:
        return SolveReport([], "EMPTY", None, None, None, 0.0, 0.0)

    w = cfg.weights
    model = cp_model.CpModel()
    x = {key: model.new_bool_var(f"x[{key[0]},{key[1]}]") for key in deadline}
    by_item: defaultdict[str, list[str]] = defaultdict(list)
    by_scout: defaultdict[str, list[str]] = defaultdict(list)
    for item_id, scout_id in x:
        by_item[item_id].append(scout_id)
        by_scout[scout_id].append(item_id)

    # Hard constraints.
    for item_id, scout_ids in by_item.items():
        model.add_at_most_one(x[item_id, s] for s in scout_ids)
    for scout_id, item_ids in by_scout.items():
        # Due-date capacity: only days that are some item's deadline can bind
        # (free hours never decrease), so one constraint per distinct deadline.
        for j in sorted({deadline[i, scout_id] for i in item_ids}):
            due_by_j = [i for i in item_ids if deadline[i, scout_id] <= j]
            model.add(sum(units[i] * x[i, scout_id] for i in due_by_j) <= free[scout_id][j])
        live_by_date: defaultdict[dt.date, list[str]] = defaultdict(list)
        for i in item_ids:
            if items[i].fixed_date is not None:
                live_by_date[items[i].fixed_date].append(i)
        for same_day in live_by_date.values():
            if len(same_day) > 1:
                model.add_at_most_one(x[i, scout_id] for i in same_day)

    # Objective.
    late = lateness_penalties(
        pool, window.start, next_run_date(window, cfg), cost.late_penalty, w.lateness
    )
    churn = round(OBJECTIVE_SCALE * w.churn)
    terms: list[cp_model.LinearExprT] = []
    for item_id, scout_ids in by_item.items():
        item = items[item_id]
        assigned = sum(x[item_id, s] for s in scout_ids)
        terms.append(round(OBJECTIVE_SCALE * late[item_id]) * (1 - assigned))
        for s in scout_ids:
            pay = round(OBJECTIVE_SCALE * w.cost * hourly_cost(states[s].scout) * item.hours)
            if pay:
                terms.append(pay * x[item_id, s])
        current = (item_id, item.current_scout_id)
        if churn and current in x:
            terms.append(churn * (1 - x[current]))
    hint = _edf_hint(pool, scouts, window, cfg, rng, cost, deadline, units, free)
    for key, var in x.items():
        model.add_hint(var, key in hint)
    terms.extend(_continuity_terms(model, x, by_item, items, w.continuity, hint))
    model.minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = seed
    solver.parameters.max_deterministic_time = cfg.time_limit_s
    solver.parameters.max_time_in_seconds = wall_clock_cap(cfg.time_limit_s)
    status = solver.solve(model)
    status_name = solver.status_name(status)
    # Stopped without a proof (FEASIBLE / UNKNOWN) before the deterministic budget
    # was used up: only the wall-clock backstop stops a search there.
    hit_wall_clock = (
        status in (cp_model.FEASIBLE, cp_model.UNKNOWN)
        and solver.deterministic_time < 0.999 * cfg.time_limit_s
    )
    if hit_wall_clock:
        log.warning("optimiser stopped by the wall-clock backstop: this solve is not reproducible")

    if status in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID):
        raise RuntimeError(f"optimiser model is {status_name}: assigning nothing is feasible, bug")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        log.warning(
            "optimiser found no feasible solution (%s) for %d items; falling back to EDF",
            status_name,
            len(pool),
        )
        return SolveReport(
            edf(pool, scouts, window, cfg, rng, cost=cost),
            status_name,
            None,
            None,
            None,
            solver.wall_time,
            solver.deterministic_time,
            fell_back_to_edf=True,
            hit_wall_clock=hit_wall_clock,
        )

    chosen = [
        Assignment(item_id, scout_id, planned_date(items[item_id]))
        for (item_id, scout_id), var in x.items()
        if solver.boolean_value(var)
    ]
    objective = solver.objective_value / OBJECTIVE_SCALE
    bound = solver.best_objective_bound / OBJECTIVE_SCALE
    report = SolveReport(
        sorted(chosen, key=lambda a: a.item_id),
        status_name,
        objective,
        bound,
        (objective - bound) / max(1.0, abs(objective)),
        solver.wall_time,
        solver.deterministic_time,
        hit_wall_clock=hit_wall_clock,
    )
    log.debug(
        "optimiser: %s, %d/%d items assigned, %d vars, objective %.2f (gap %.3f), %.3f s wall",
        status_name,
        len(chosen),
        len(pool),
        len(x),
        objective,
        report.gap,
        report.wall_time_s,
    )
    return report


def _edf_hint(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
    cost: CostParams,
    deadline: dict[tuple[str, str], int],
    units: dict[str, int],
    free: dict[str, list[int]],
) -> set[tuple[str, str]]:
    """EDF's answer as ``(item_id, scout_id)`` pairs, trimmed to what the model allows.

    EDF only respects each scout's window total, in floats. The model also
    asks for due-date capacity, in rounded integers. Pairs that would break it
    are dropped, in EDF order: a hint must be feasible for CP-SAT to start
    from it, and a *complete* feasible hint gives a first solution at once.
    """
    n_days = window.n_days
    due_units = {sid: [0] * n_days for sid in free}  # hinted units per deadline day
    hint: set[tuple[str, str]] = set()
    edf_answer = {a.item_id: a.scout_id for a in edf(pool, scouts, window, cfg, rng, cost=cost)}
    for item in sorted(pool, key=lambda i: (i.due_date, i.received_date, i.item_id)):
        sid = edf_answer.get(item.item_id)
        key = (item.item_id, sid or "")
        if sid is None or key not in deadline:
            continue
        j = deadline[key]
        trial = due_units[sid].copy()
        trial[j] += units[item.item_id]
        running = 0
        ok = True
        for k in range(n_days):
            running += trial[k]
            if running > free[sid][k]:
                ok = False
                break
        if ok:
            due_units[sid] = trial
            hint.add(key)
    return hint


def _continuity_terms(
    model: cp_model.CpModel,
    x: dict[tuple[str, str], cp_model.IntVar],
    by_item: dict[str, list[str]],
    items: dict[str, WorkItem],
    weight: float,
    hint: set[tuple[str, str]],
) -> list[cp_model.LinearExprT]:
    """Penalty per extra scout on a request (and the hint for its helper variables).

    For each request and each candidate scout not already on it, a boolean
    ``y[s]`` that must be 1 if the scout gets any of the request's items
    (``x[i, s] => y[s]``; minimising keeps ``y`` at 0 otherwise). If someone is
    already on the request, every new scout is an extra one; if nobody is,
    the first scout is free and only the ones after it count.
    """
    coef = round(OBJECTIVE_SCALE * weight)
    if coef == 0:
        return []
    by_request: defaultdict[str, list[str]] = defaultdict(list)
    for item_id in sorted(by_item):
        by_request[items[item_id].request_id].append(item_id)

    terms: list[cp_model.LinearExprT] = []
    for request_id, item_ids in sorted(by_request.items()):
        prior = frozenset().union(*(items[i].request_scout_ids for i in item_ids))
        if len(item_ids) < 2 and not prior:
            continue
        new_scouts = sorted({s for i in item_ids for s in by_item[i]} - prior)
        if not new_scouts:
            continue
        y = {s: model.new_bool_var(f"y[{request_id},{s}]") for s in new_scouts}
        hinted_on = {s for i in item_ids for s in by_item[i] if (i, s) in hint}
        for s, var in y.items():
            model.add_hint(var, s in hinted_on)
        for i in item_ids:
            for s in by_item[i]:
                if s in y:
                    model.add_implication(x[i, s], y[s])
        if prior:
            terms.append(coef * sum(y.values()))
        else:
            extra = model.new_int_var(0, len(new_scouts), f"extra[{request_id}]")
            model.add(extra >= sum(y.values()) - 1)
            model.add_hint(extra, max(0, len(hinted_on & set(y)) - 1))
            terms.append(coef * extra)
    return terms


def optimiser(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
    *,
    cost: CostParams,
) -> list[Assignment]:
    """CP-SAT policy (the registry entry): the assignments of :func:`solve_assignment`."""
    return solve_assignment(pool, scouts, window, cfg, rng, cost=cost).assignments
