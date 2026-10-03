"""Hard constraints of an assignment run, in one place, shared by every policy.

A *hard* constraint can never be broken, whatever the cost (the opposite of a
*soft* constraint, which the optimiser may break at a penalty). There are two
kinds here:

* **Pairwise** (one item, one scout): skill match, conflict of interest,
  precedence, and for live views the fixture date, region and the scout's
  availability that day. :func:`ineligibility_reasons` checks these. Every
  policy filters on them before deciding anything, so the optimiser only
  creates decision variables for pairs that are allowed at all.
* **Aggregate** (several items on one scout): the scout's hours in the window
  (frozen work counted first) and at most one live view per scout per date.
  Each policy enforces these its own way (greedy: running totals; CP-SAT:
  linear constraints). :func:`validate_assignments` re-checks everything on a
  finished answer; the tests run it on the output of every policy.

Hours rule (D-008): a live view counts its ``hours`` (normally 8) against the
scout's *window total*, even when the fixture falls on a day on which the scout
offers no desk hours (a weekend match). Only ``unavailable_dates`` (leave)
forbid a live view on a date.

Due-date feasibility (the stricter, optional check)
---------------------------------------------------
"Fits in the window total" does not mean "finishes on time": 10 h of work due
tomorrow does not fit in one 7.5 h day, even if the week has 37.5 h. For one
scout working their queue earliest-due-first, every item is on time **iff**
for every day ``d``::

    frozen hours + hours of items due on or before d  <=  hours offered up to d

(the classic feasibility condition for *EDF on a single machine*). The
optimiser enforces this as a hard constraint; :func:`due_date_violations`
checks it. The greedy baselines do **not** (they only respect the window
total), so for them this check is a measurement, not a rule; that gap is part
of what the optimiser buys.

Two details keep the condition usable when work is already late:

* An item's day ``d`` is its due date, or its fixture date for a live view
  (it happens that day). Items due after the window count at the last day.
* If a scout could not finish the item on time even with nothing else queued
  (overdue work, a tight deadline after a lot of frozen work), its day moves
  to the first day the scout *could* finish it. It is then "as soon as
  possible" work: it still uses up the early hours, so it cannot hide
  capacity, but it does not make every assignment impossible.

Hours are compared in integer hundredths of an hour (:data:`HOURS_SCALE`),
work rounded up and capacity rounded down, exactly as in the CP-SAT model, so
the check and the optimiser always agree.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from scout_planner.domain import Assignment, AssignmentWindow, ScoutState, WorkItem

# One float tolerance for every hour comparison in this package, in hours
# (0.1 + 0.2 != 0.3 in binary floating point). The optimiser imports it too.
EPS = 1e-6
# Hours -> integer hundredths of an hour (CP-SAT works on integers only).
HOURS_SCALE = 100


def scaled_hours(hours: float) -> int:
    """Work in hundredths of an hour, rounded up (never under-count work)."""
    return math.ceil(hours * HOURS_SCALE - EPS * HOURS_SCALE)


def scaled_capacity(hours: float) -> int:
    """Capacity in hundredths of an hour, rounded down (never over-count hours)."""
    return math.floor(hours * HOURS_SCALE + EPS * HOURS_SCALE)


# --- input checks ------------------------------------------------------------------


def check_inputs(
    pool: Sequence[WorkItem], scouts: Sequence[ScoutState], window: AssignmentWindow
) -> None:
    """Fail loudly on a malformed assignment input: the pool builder has a bug.

    * item and scout ids are unique;
    * a scout's ``hours_by_day`` only has days inside the window;
    * a leave day inside the window offers 0 hours.
    """
    for what, ids in (
        ("work item", [i.item_id for i in pool]),
        ("scout", [s.scout_id for s in scouts]),
    ):
        dupes = sorted(k for k, n in Counter(ids).items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate {what} ids in assignment input: {dupes}")
    for s in scouts:
        outside = sorted(d for d in s.hours_by_day if d not in window)
        if outside:
            raise ValueError(f"scout {s.scout_id}: hours_by_day has days outside the window")
        on_leave = sorted(d for d in s.unavailable_dates if s.hours_by_day.get(d, 0.0) > EPS)
        if on_leave:
            raise ValueError(f"scout {s.scout_id}: offers hours on leave days {on_leave}")


# --- pairwise rules --------------------------------------------------------------------


def ineligibility_reasons(item: WorkItem, state: ScoutState, window: AssignmentWindow) -> list[str]:
    """Every pairwise hard constraint that forbids giving ``item`` to this scout.

    Empty list = the pair is allowed (capacity is checked separately, because
    it depends on what else the scout gets).
    """
    scout = state.scout
    reasons: list[str] = []
    if not scout.has_skill(item.skill_type):
        reasons.append(f"skill: scout lacks {item.skill_type}")
    if scout.has_conflict(item.conflict_clubs):
        reasons.append("conflict of interest: scout used to work for a club on the request")
    if item.depends_on:
        reasons.append(f"precedence: waits for {', '.join(item.depends_on)}")
    if item.fixed_date is not None:  # a live view (WorkItem invariant)
        if item.fixed_date not in window:
            reasons.append(f"live view: fixture date {item.fixed_date} outside the window")
        if scout.home_region != item.region:
            reasons.append(
                f"live view: fixture in {item.region}, scout based in {scout.home_region}"
            )
        if not state.is_available(item.fixed_date):
            reasons.append(f"live view: scout unavailable on {item.fixed_date}")
    return reasons


def is_eligible(item: WorkItem, state: ScoutState, window: AssignmentWindow) -> bool:
    """True when no pairwise hard constraint forbids the pair."""
    return not ineligibility_reasons(item, state, window)


def fits(hours: float, remaining_hours: float) -> bool:
    """Whether ``hours`` of work fit in ``remaining_hours`` (with float tolerance)."""
    return hours <= remaining_hours + EPS


def eligible_scouts(
    pool: Iterable[WorkItem], scouts: Sequence[ScoutState], window: AssignmentWindow
) -> dict[str, list[ScoutState]]:
    """For each item id, the scouts that may take it *alone*: pairwise rules + free hours.

    Scouts keep the order of ``scouts``. An item no scout can take maps to ``[]``.
    """
    return {
        item.item_id: [
            s for s in scouts if fits(item.hours, s.free_hours) and is_eligible(item, s, window)
        ]
        for item in pool
    }


def planned_date(item: WorkItem) -> dt.date | None:
    """The date to put on an ``Assignment``: the fixture date for a live view, else ``None``."""
    return item.fixed_date if item.kind == "live" else None


# --- due-date feasibility ------------------------------------------------------------------


def free_by_day(state: ScoutState, window: AssignmentWindow) -> list[int]:
    """Scaled hours the scout has free *up to and including* each window day.

    Cumulative offered hours minus frozen work (done first), never negative.
    Non-decreasing by construction.
    """
    total, out = 0.0, []
    for d in window.days:
        total += state.hours_by_day.get(d, 0.0)
        out.append(scaled_capacity(max(0.0, total - state.frozen_hours)))
    return out


def deadline_index(
    item: WorkItem,
    free: Sequence[int],
    window: AssignmentWindow,
    *,
    commit_by: int | None = None,
) -> int | None:
    """Window-day index the item must be done by, for this scout; ``None`` if it never fits.

    The due date (or fixture date) clamped into the window, moved later to the
    first day the scout could finish the item alone if it cannot be on time.
    ``commit_by`` (optimiser, see ``optimiser.commit_index``) pulls the
    deadline of non-live work forward to that day index: only work the scout
    can finish by then is committed now (a live view keeps its fixture date).
    """
    key = item.fixed_date or item.due_date
    index = min(max((key - window.start).days, 0), window.n_days - 1)
    if commit_by is not None and item.fixed_date is None:
        index = min(index, commit_by)
    need = scaled_hours(item.hours)
    for j in range(index, window.n_days):
        if free[j] >= need:
            return j
    return None


def due_date_violations(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    assignments: Sequence[Assignment],
) -> list[str]:
    """Days on which a scout is promised more work than they can do by then.

    The EDF single-scout condition from the module docstring; ``[]`` = every
    scout can finish every assigned item by its (possibly moved) deadline.
    Unknown ids are skipped here (``validate_assignments`` reports them).
    """
    items = {i.item_id: i for i in pool}
    states = {s.scout_id: s for s in scouts}
    by_scout: defaultdict[str, list[WorkItem]] = defaultdict(list)
    for a in assignments:
        if a.item_id in items and a.scout_id in states:
            by_scout[a.scout_id].append(items[a.item_id])

    problems: list[str] = []
    for scout_id, assigned in sorted(by_scout.items()):
        free = free_by_day(states[scout_id], window)
        due_units = [0] * window.n_days
        for item in assigned:
            j = deadline_index(item, free, window)
            if j is None:
                problems.append(f"{scout_id}: {item.item_id} cannot be finished in the window")
                continue
            due_units[j] += scaled_hours(item.hours)
        running = 0
        for j, day in enumerate(window.days):
            running += due_units[j]
            if running > free[j]:
                problems.append(
                    f"{scout_id}: due-date capacity: {running / HOURS_SCALE:.2f} h due by {day}, "
                    f"{free[j] / HOURS_SCALE:.2f} h free by then"
                )
    return problems


# --- the validator ---------------------------------------------------------------------------


def validate_assignments(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    assignments: Sequence[Assignment],
    *,
    check_due_dates: bool = False,
) -> list[str]:
    """Every hard constraint the answer of a policy breaks; ``[]`` means it is valid.

    Checks, in order: known ids and one assignment per item; the pairwise
    rules; the live view's ``planned_date``; per scout, hours within the window
    after frozen work, and at most one live view per date. With
    ``check_due_dates`` also :func:`due_date_violations` (the optimiser
    guarantees it; the greedy baselines don't).
    """
    items = {i.item_id: i for i in pool}
    states = {s.scout_id: s for s in scouts}
    problems: list[str] = []

    seen: set[str] = set()
    hours_used: defaultdict[str, float] = defaultdict(float)
    live_dates: defaultdict[str, list[dt.date]] = defaultdict(list)

    for a in assignments:
        where = f"{a.item_id} -> {a.scout_id}"
        item, state = items.get(a.item_id), states.get(a.scout_id)
        if item is None:
            problems.append(f"{where}: item not in the pool")
            continue
        if state is None:
            problems.append(f"{where}: unknown scout")
            continue
        if a.item_id in seen:
            problems.append(f"{where}: item assigned more than once")
        seen.add(a.item_id)

        problems.extend(f"{where}: {r}" for r in ineligibility_reasons(item, state, window))
        expected_date = planned_date(item)
        if a.planned_date != expected_date:
            problems.append(f"{where}: planned_date {a.planned_date}, expected {expected_date}")

        hours_used[a.scout_id] += item.hours
        if item.fixed_date is not None:
            live_dates[a.scout_id].append(item.fixed_date)

    for scout_id, used in sorted(hours_used.items()):
        state = states[scout_id]
        if not fits(used, state.free_hours):
            problems.append(
                f"{scout_id}: hours cap: {used:.4f} h assigned, {state.free_hours:.4f} h free "
                f"({state.window_hours:.4f} h in window - {state.frozen_hours:.4f} h frozen; "
                f"over by {used - state.free_hours:.2e} h)"
            )
    for scout_id, dates in sorted(live_dates.items()):
        for day, n in sorted(Counter(dates).items()):
            if n > 1:
                problems.append(f"{scout_id}: {n} live views on {day} (max 1 per day)")
    if check_due_dates:
        problems.extend(due_date_violations(pool, scouts, window, assignments))
    return problems
