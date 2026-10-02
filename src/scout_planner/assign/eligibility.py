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
"""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from scout_planner.domain import Assignment, AssignmentWindow, ScoutState, WorkItem

# Float tolerance for hour sums (0.1 + 0.2 != 0.3 in binary floating point).
EPS = 1e-9


def check_unique_ids(pool: Sequence[WorkItem], scouts: Sequence[ScoutState]) -> None:
    """Fail loudly on duplicate item or scout ids: the caller built a broken pool."""
    for what, ids in (
        ("work item", [i.item_id for i in pool]),
        ("scout", [s.scout_id for s in scouts]),
    ):
        dupes = sorted(k for k, n in Counter(ids).items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate {what} ids in assignment input: {dupes}")


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
    if item.kind == "live":
        # fixed_date and region are guaranteed set for live items (WorkItem invariant).
        assert item.fixed_date is not None
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


def validate_assignments(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    assignments: Sequence[Assignment],
) -> list[str]:
    """Every hard constraint the answer of a policy breaks; ``[]`` means it is valid.

    Checks, in order: known ids and one assignment per item; the pairwise
    rules; the live view's ``planned_date``; per scout, hours within the window
    after frozen work, and at most one live view per date.
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
                f"{scout_id}: hours cap: {used:g} h assigned, {state.free_hours:g} h free "
                f"({state.window_hours:g} h in window - {state.frozen_hours:g} h frozen)"
            )
    for scout_id, dates in sorted(live_dates.items()):
        for day, n in sorted(Counter(dates).items()):
            if n > 1:
                problems.append(f"{scout_id}: {n} live views on {day} (max 1 per day)")
    return problems
