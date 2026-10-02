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
    * each scout's assigned hours <= free hours in the window (window hours
      minus frozen work: started work is counted first, D-010);
    * at most one live view per scout per date (a live view takes the day, D-008).

Objective (minimise; weights are ``assignment.weights``)
    * **lateness**: for each item left unassigned, ``lateness x urgency``.
      See :func:`urgency`: it grows as the request's slack shrinks, and
      overdue requests are the most urgent of all.
    * **cost**: ``cost x freelance hours x the freelancer's hourly rate``.
      Salaried hours cost 0 here: salaries are paid whether the scout is busy
      or not (a *sunk cost*), so only freelance hours change the bill.
    * **continuity**: ``continuity`` per extra scout on one request (a desk
      review and a write-up by different scouts means a hand-over). Scouts
      already on the request outside the pool come in through
      ``WorkItem.request_scout_ids``.
    * **churn**: ``churn`` per item taken away from its ``current_scout_id``
      (assigned earlier, not started), whether moved to another scout or
      dropped back to the pool. Without the "dropped" half, un-assigning would
      be a free way round the penalty. Changing the plan is allowed, but must
      pay for itself (*plan nervousness* control, D-010).

Hours are scaled to integers (CP-SAT only handles integers): item hours are
rounded *up* and capacities *down* to 1/100 h, so every solution is also valid
in exact float arithmetic.

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
    * ``max_time_in_seconds = WALL_CLOCK_SAFETY x time_limit_s`` as a backstop
      for very slow machines. If that cap ever fires, that solve stops being
      reproducible; the trade-off is a guaranteed bound on run time.

    Rejected: wall-clock limit only (simple, but not reproducible); several
    workers (faster on big models, not reproducible); no limit at all (proves
    optimality on small days but can stall a whole simulated year on a busy one).

Warm start: the EDF answer is passed as a complete, feasible *solution hint*
(continuity helpers included), so the solver has a first solution almost at
once and only searches for improvements. Measured: with a partial hint, a
0.1 s budget on 400 items ended with no solution at all. If the solver is
still empty-handed when stopped (status ``UNKNOWN``), the policy falls back to
EDF and logs a warning.
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

from scout_planner.assign.eligibility import check_unique_ids, eligible_scouts, planned_date
from scout_planner.assign.greedy import edf
from scout_planner.config import FULL_TIME_WEEKLY_HOURS, AssignmentParams
from scout_planner.domain import Assignment, AssignmentWindow, Scout, ScoutState, WorkItem

log = logging.getLogger(__name__)

HOURS_SCALE = 100  # hours -> integer hundredths of an hour
OBJECTIVE_SCALE = 100  # objective terms -> integers, keeping two decimals
WALL_CLOCK_SAFETY = 2.0  # wall-clock cap = this x time_limit_s

DAY_HOURS = FULL_TIME_WEEKLY_HOURS / 5  # 7.5 h: one working day, for slack in days
URGENCY_MAX = 10.0  # zero slack: lateness weight 100 x 10 = 1000 ~ cost.late_penalty
URGENCY_OVERDUE = 2 * URGENCY_MAX  # already past due: the most urgent of all


# --- urgency -----------------------------------------------------------------------


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
    """How bad it is to leave an item unassigned today, on a 0..20 scale.

    ``URGENCY_MAX / (1 + slack)`` for slack >= 0: 10 at zero slack, 5 with one
    day to spare, 1 with nine. A hyperbola rather than a straight line because
    a day of slack matters much more when there are two left than when there
    are twelve. Negative slack (can no longer make it) counts as zero slack.
    Overdue requests get twice the maximum, so they are never starved by a
    stream of new zero-slack work (that would make turnaround tails explode).

    Calibration: at the default lateness weight (100), a zero-slack item is
    worth 1000, the default late penalty per report. So with default weights
    a freelancer at ~41.5/h is hired for a 6 h desk review (~250) once slack
    is down to about 3 days, and for urgent work always.
    """
    if overdue:
        return URGENCY_OVERDUE
    return URGENCY_MAX / (1 + max(slack, 0))


def item_urgencies(pool: Sequence[WorkItem], today: dt.date) -> dict[str, float]:
    """Urgency per item id; every item of a request shares the request's urgency."""
    by_request: defaultdict[str, list[WorkItem]] = defaultdict(list)
    for item in pool:
        by_request[item.request_id].append(item)
    result: dict[str, float] = {}
    for items in by_request.values():
        due = min(i.due_date for i in items)
        slack = slack_days(due, today, remaining_work_days(items, today))
        u = urgency(slack, overdue=due < today)
        result.update((i.item_id, u) for i in items)
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
    """What one optimiser call did. ``assignments`` is the policy's answer."""

    assignments: list[Assignment]
    status: str  # CP-SAT status name, or "EMPTY" when there was nothing to decide
    objective: float | None  # in objective units (already divided by OBJECTIVE_SCALE)
    wall_time_s: float
    deterministic_time: float
    fell_back_to_edf: bool = False


def solve_assignment(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
) -> SolveReport:
    """Build and solve the CP-SAT model; see the module docstring for the model."""
    # Always exactly one draw per call, so the stream advances the same way
    # whether or not there is anything to solve today.
    seed = int(rng.integers(0, 2**31 - 1))
    check_unique_ids(pool, scouts)
    candidates = eligible_scouts(pool, scouts, window)
    if not any(candidates.values()):
        return SolveReport([], "EMPTY", None, 0.0, 0.0)

    w = cfg.weights
    states = {s.scout_id: s for s in scouts}
    items = {i.item_id: i for i in pool}
    urg = item_urgencies(pool, window.start)
    model = cp_model.CpModel()

    # Decision variables, eligible pairs only. Sorted ids -> same model every time.
    x: dict[tuple[str, str], cp_model.IntVar] = {}
    for item_id in sorted(candidates):
        for state in candidates[item_id]:
            x[item_id, state.scout_id] = model.new_bool_var(f"x[{item_id},{state.scout_id}]")

    by_item: defaultdict[str, list[str]] = defaultdict(list)
    by_scout: defaultdict[str, list[str]] = defaultdict(list)
    for item_id, scout_id in x:
        by_item[item_id].append(scout_id)
        by_scout[scout_id].append(item_id)

    # Hard constraints.
    for item_id, scout_ids in by_item.items():
        model.add_at_most_one(x[item_id, s] for s in scout_ids)
    capacity = {s: _scaled_capacity(states[s].free_hours) for s in by_scout}
    for scout_id, item_ids in by_scout.items():
        model.add(
            sum(_scaled_hours(items[i].hours) * x[i, scout_id] for i in item_ids)
            <= capacity[scout_id]
        )
        live_by_date: defaultdict[dt.date, list[str]] = defaultdict(list)
        for i in item_ids:
            if items[i].fixed_date is not None:
                live_by_date[items[i].fixed_date].append(i)
        for same_day in live_by_date.values():
            if len(same_day) > 1:
                model.add_at_most_one(x[i, scout_id] for i in same_day)

    # Objective.
    terms: list[cp_model.LinearExprT] = []
    for item_id, scout_ids in by_item.items():
        item = items[item_id]
        assigned = sum(x[item_id, s] for s in scout_ids)
        late = round(OBJECTIVE_SCALE * w.lateness * urg[item_id])
        terms.append(late * (1 - assigned))
        for s in scout_ids:
            cost = round(OBJECTIVE_SCALE * w.cost * hourly_cost(states[s].scout) * item.hours)
            if cost:
                terms.append(cost * x[item_id, s])
        # Churn: paid unless the item stays with its current scout (moved *or* dropped).
        # If the current scout can no longer take it at all, every option pays the
        # same churn, so it is a constant and left out.
        churn = round(OBJECTIVE_SCALE * w.churn)
        current = (item_id, item.current_scout_id)
        if churn and current in x:
            terms.append(churn * (1 - x[current]))
    hint = _edf_hint(pool, scouts, window, cfg, rng, items, capacity)
    for key, var in x.items():
        model.add_hint(var, key in hint)
    terms.extend(_continuity_terms(model, x, by_item, items, w.continuity, hint))
    model.minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = seed
    solver.parameters.max_deterministic_time = cfg.time_limit_s
    solver.parameters.max_time_in_seconds = WALL_CLOCK_SAFETY * cfg.time_limit_s
    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        log.warning(
            "optimiser found no feasible solution (%s) for %d items; falling back to EDF",
            status_name,
            len(pool),
        )
        return SolveReport(
            edf(pool, scouts, window, cfg, rng),
            status_name,
            None,
            solver.wall_time,
            solver.deterministic_time,
            fell_back_to_edf=True,
        )

    chosen = [
        Assignment(item_id, scout_id, planned_date(items[item_id]))
        for (item_id, scout_id), var in x.items()
        if solver.boolean_value(var)
    ]
    report = SolveReport(
        sorted(chosen, key=lambda a: a.item_id),
        status_name,
        solver.objective_value / OBJECTIVE_SCALE,
        solver.wall_time,
        solver.deterministic_time,
    )
    log.debug(
        "optimiser: %s, %d/%d items assigned, %d vars, objective %.2f, %.3f s wall",
        status_name,
        len(chosen),
        len(pool),
        len(x),
        report.objective,
        report.wall_time_s,
    )
    return report


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
    (``x[i, s] <= y[s]``; minimising keeps ``y`` at 0 otherwise). If someone is
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


def _scaled_hours(hours: float) -> int:
    """Item hours in hundredths, rounded up (never under-count work)."""
    return math.ceil(hours * HOURS_SCALE - 1e-6)


def _scaled_capacity(hours: float) -> int:
    """Free hours in hundredths, rounded down (never over-count capacity)."""
    return math.floor(hours * HOURS_SCALE + 1e-6)


def _edf_hint(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
    items: dict[str, WorkItem],
    capacity: dict[str, int],
) -> set[tuple[str, str]]:
    """EDF's answer as ``(item_id, scout_id)`` pairs, trimmed to fit the integer model.

    EDF checks hours in floats; the model rounds item hours up and capacity
    down, so a scout EDF filled to the last minute may be a few hundredths
    over in the model. Such pairs are dropped: a hint must be feasible for
    CP-SAT to start from it, and a *complete* feasible hint lets the solver
    have a first solution almost immediately.
    """
    used: defaultdict[str, int] = defaultdict(int)
    hint: set[tuple[str, str]] = set()
    for a in edf(pool, scouts, window, cfg, rng):
        need = _scaled_hours(items[a.item_id].hours)
        if used[a.scout_id] + need <= capacity.get(a.scout_id, 0):
            used[a.scout_id] += need
            hint.add((a.item_id, a.scout_id))
    return hint


def optimiser(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
) -> list[Assignment]:
    """CP-SAT policy (the registry entry): the assignments of :func:`solve_assignment`."""
    return solve_assignment(pool, scouts, window, cfg, rng).assignments
