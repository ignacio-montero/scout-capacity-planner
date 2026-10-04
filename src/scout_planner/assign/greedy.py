"""Greedy baselines: FCFS, EDF, and EDF with a due-date check (``edf_feasible``).

Both are *list-scheduling* heuristics: sort the items once by a priority rule,
then walk the list and give each item to a scout that can still take it. Each
item is decided once, looking only at what is left at that moment, with no
look-ahead. That is what makes them fast, easy to explain, and beatable by
the optimiser (see ``optimiser.py``).

Rules shared by both (D-010):

1. **Commitments first, never moved by choice.** Items that already have a
   ``current_scout_id`` (assigned in an earlier run, not started) are placed
   first, in policy order, with that same scout.
2. **Forced moves are not churn.** If the current scout can no longer take a
   committed item (away on the fixture day, no longer eligible, hours gone, or
   no longer on the team), the commitment is void and the item is re-placed
   in step 3 like a new item. Leaving it stranded would only make it late.
3. **Then the open items**, in policy order. Among the scouts that may take
   the item (``eligibility.py``) and still have the hours, pick by:

   a. salaried before freelance: salaried hours are already paid for, a
      freelancer's are paid per hour used;
   b. then the scout with the most hours left (*least-loaded first*), to keep
      everyone's queue short;
   c. then scout id, so ties are broken the same way on every run.

   Rejected: "first eligible scout in list order" (the brief's wording). It is
   even simpler, but it makes results depend on how the team list happens to
   be sorted and piles work on low-numbered scouts. The rule above is still
   naive on purpose: it ignores that a multi-skilled scout is scarce, and it
   only checks the scout's *window total*, not whether each item can be
   finished by its due date (``eligibility.due_date_violations``). Those are
   exactly the mistakes the optimiser exists to avoid.

``edf_feasible`` (the *strong baseline*): EDF order and the same scout
preference, but a scout is only a candidate if they can still finish every
item they hold by its due date with this one added (``DueDateLedger``: the
same per-day cumulative capacity rule the optimiser enforces). Otherwise the
next eligible scout is tried, freelancers included; else the item stays
pooled. A commitment is kept only if it passes that check too (failing it is
a forced move, rule 2). Why it exists: a red-team showed that this check alone
lets EDF beat the optimiser, so the optimiser must be judged against it.

Decision: it checks due dates over the **whole window** and ignores
``assignment.commit_buffer_days``. The commitment horizon is the optimiser's
remedy for being indifferent to *when* work starts; the greedy
least-loaded rule does not have that problem, and with the horizon it spilled
work onto freelancers (measured at 4x growth, 3 seeds: 95.1% on time / 3,599k
with the horizon vs 96.1% / 3,232k without). :func:`edf_feasible_assign` with
a ``commit_by`` is the variant that obeys the optimiser's exact rules; the
optimiser uses it for its warm start.

Greedy policies use no randomness and no cost figures; ``rng`` and ``cost``
are accepted only to satisfy the common interface. Inputs are sorted by id on
entry, so the answer does not depend on the order of the lists passed in.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence

import numpy as np

from scout_planner.assign.eligibility import (
    DueDateLedger,
    check_inputs,
    fits,
    is_eligible,
    planned_date,
)
from scout_planner.config import AssignmentParams, CostParams
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
    ledger: DueDateLedger | None = None,
) -> list[Assignment]:
    """List scheduling with the rules in the module docstring, for any priority order.

    With a ``ledger``, a scout must also pass its due-date capacity check.
    """
    check_inputs(pool, scouts, window)
    scouts = sorted(scouts, key=lambda s: s.scout_id)
    states = {s.scout_id: s for s in scouts}
    remaining = {s.scout_id: s.free_hours for s in scouts}
    live_days: dict[str, set[dt.date]] = {s.scout_id: set() for s in scouts}
    chosen: list[Assignment] = []
    placed: set[str] = set()

    def can_take(item: WorkItem, state: ScoutState) -> bool:
        sid = state.scout_id
        return (
            fits(item.hours, remaining[sid])
            and (item.fixed_date is None or item.fixed_date not in live_days[sid])
            and is_eligible(item, state, window)
            and (ledger is None or ledger.fits(item, sid))
        )

    def take(item: WorkItem, state: ScoutState) -> None:
        sid = state.scout_id
        remaining[sid] -= item.hours
        if item.fixed_date is not None:
            live_days[sid].add(item.fixed_date)
        if ledger is not None:
            ledger.add(item, sid)
        chosen.append(Assignment(item.item_id, sid, planned_date(item)))
        placed.add(item.item_id)

    def preference(state: ScoutState) -> tuple[bool, float, str]:
        return (state.scout.is_freelance, -remaining[state.scout_id], state.scout_id)

    ordered = sorted(pool, key=order_key)

    # 1. Commitments: keep the current scout when that is still possible.
    for item in ordered:
        state = states.get(item.current_scout_id) if item.current_scout_id else None
        if state is not None and can_take(item, state):
            take(item, state)

    # 2-3. Open items and voided commitments: best scout by the documented preference.
    for item in ordered:
        if item.item_id in placed:
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
    *,
    cost: CostParams,
) -> list[Assignment]:
    """First come, first served: work through items by ``received_date``."""
    return greedy_assign(pool, scouts, window, fcfs_key)


def edf(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
    *,
    cost: CostParams,
) -> list[Assignment]:
    """Earliest due date first: work through items by ``due_date``."""
    return greedy_assign(pool, scouts, window, edf_key)


def edf_feasible_assign(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    commit_by: int | None = None,
) -> list[Assignment]:
    """EDF with the due-date check; ``commit_by`` adds the optimiser's commitment horizon."""
    return greedy_assign(pool, scouts, window, edf_key, DueDateLedger(scouts, window, commit_by))


def edf_feasible(
    pool: Sequence[WorkItem],
    scouts: Sequence[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
    *,
    cost: CostParams,
) -> list[Assignment]:
    """EDF that never promises work a scout cannot finish by its due date (strong baseline)."""
    return edf_feasible_assign(pool, scouts, window)
