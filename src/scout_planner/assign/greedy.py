"""Greedy baselines: FCFS (first come, first served) and EDF (earliest due date first).

Both are *list-scheduling* heuristics: sort the items once by a priority rule,
then walk the list and give each item to a scout that can still take it. Each
item is decided once, looking only at what is left at that moment, with no
look-ahead. That is what makes them fast, easy to explain, and beatable by
the optimiser (see ``optimiser.py``).

Rules shared by both (D-010):

1. **Commitments first, never moved.** Items that already have a
   ``current_scout_id`` (assigned in an earlier run, not started) are placed
   first, in policy order, with that same scout. If the scout can no longer
   take one (e.g. their window shrank), the item is left unassigned this run
   rather than moved; it returns to the open pool for the next run.
2. **Then the open items**, in policy order. Among the scouts that may take
   the item (``eligibility.py``) and still have the hours, pick by:

   a. salaried before freelance: salaried hours are already paid for, a
      freelancer's are paid per hour used;
   b. then the scout with the most hours left (*least-loaded first*), to keep
      everyone's queue short;
   c. then scout id, so ties are broken the same way on every run.

   Rejected: "first eligible scout in list order" (the brief's wording). It is
   even simpler, but it makes results depend on how the team list happens to
   be sorted and piles work on low-numbered scouts. The rule above is still
   naive on purpose: it ignores that a multi-skilled scout is scarce, which is
   exactly the mistake the optimiser exists to avoid.

Greedy policies use no randomness; ``rng`` is accepted only to satisfy the
common interface.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence

import numpy as np

from scout_planner.assign.eligibility import check_unique_ids, fits, is_eligible, planned_date
from scout_planner.config import AssignmentParams
from scout_planner.domain import Assignment, AssignmentWindow, ScoutState, WorkItem

OrderKey = Callable[[WorkItem], tuple]


def fcfs_key(item: WorkItem) -> tuple[dt.date, str]:
    """FCFS order: oldest request first; item id breaks ties (groups a request's items)."""
    return (item.received_date, item.item_id)


def edf_key(item: WorkItem) -> tuple[dt.date, dt.date, str]:
    """EDF order: earliest due date first, then oldest request, then item id."""
    return (item.due_date, item.received_date, item.item_id)


def greedy_assign(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    order_key: OrderKey,
) -> list[Assignment]:
    """List scheduling with the rules in the module docstring, for any priority order."""
    check_unique_ids(pool, scouts)
    states = {s.scout_id: s for s in scouts}
    remaining = {s.scout_id: s.free_hours for s in scouts}
    live_days: dict[str, set[dt.date]] = {s.scout_id: set() for s in scouts}
    chosen: list[Assignment] = []

    def can_take(item: WorkItem, state: ScoutState) -> bool:
        sid = state.scout_id
        return (
            fits(item.hours, remaining[sid])
            and (item.fixed_date is None or item.fixed_date not in live_days[sid])
            and is_eligible(item, state, window)
        )

    def take(item: WorkItem, state: ScoutState) -> None:
        sid = state.scout_id
        remaining[sid] -= item.hours
        if item.fixed_date is not None:
            live_days[sid].add(item.fixed_date)
        chosen.append(Assignment(item.item_id, sid, planned_date(item)))

    def preference(state: ScoutState) -> tuple[bool, float, str]:
        return (state.scout.is_freelance, -remaining[state.scout_id], state.scout_id)

    ordered = sorted(pool, key=order_key)

    # 1. Commitments: keep the current scout or leave unassigned. Never move.
    for item in ordered:
        if item.current_scout_id is None:
            continue
        state = states.get(item.current_scout_id)
        if state is not None and can_take(item, state):
            take(item, state)

    # 2. Open items: best scout by the documented preference.
    for item in ordered:
        if item.current_scout_id is not None:
            continue
        candidates = [s for s in scouts if can_take(item, s)]
        if candidates:
            take(item, min(candidates, key=preference))

    return sorted(chosen, key=lambda a: a.item_id)


def fcfs(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
) -> list[Assignment]:
    """First come, first served: work through items by ``received_date``."""
    return greedy_assign(pool, scouts, window, fcfs_key)


def edf(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
) -> list[Assignment]:
    """Earliest due date first: work through items by ``due_date``."""
    return greedy_assign(pool, scouts, window, edf_key)
