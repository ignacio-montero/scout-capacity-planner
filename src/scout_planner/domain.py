"""Domain objects shared by every stage. Spec: ``docs/DATA_CONTRACTS.md`` section 4.

Design rules:

* **Immutable values** (``frozen=True``): a ``Request`` or ``WorkItem`` is a fact
  or a snapshot, never edited in place. The simulation keeps its own mutable
  bookkeeping (remaining hours, who started what) and builds fresh snapshots
  for each assignment run. Immutable inputs are what let the assignment
  policies be pure functions.
* ``slots=True``: fixed attribute set (a typo like ``item.huors = 3`` fails) and
  lower memory, which matters for ~10k requests x seeds.
* Collections are coerced to ``frozenset``/``tuple``/read-only mappings in
  ``__post_init__``, so callers may pass lists (e.g. straight from a parquet
  row) and the object still cannot be mutated through a shared reference.
* ``Literal`` annotations are not enforced by Python at runtime, so each class
  checks its own invariants and raises ``ValueError`` early.

Units: dates are ``datetime.date`` (the simulation is daily), hours are floats.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, get_args

Employment = Literal["full_time", "freelance"]
TaskKind = Literal["desk", "live", "writeup"]
ItemKind = Literal["desk", "live", "writeup", "bundle"]
Period = Literal["history", "future"]


def _set(obj: object, name: str, value: Any) -> None:
    """Assign inside ``__post_init__`` of a frozen dataclass (normalisation only)."""
    object.__setattr__(obj, name, value)


def _check_literal(value: str, literal: Any, what: str) -> None:
    allowed = get_args(literal)
    if value not in allowed:
        raise ValueError(f"{what} must be one of {allowed}, got {value!r}")


def task_id(request_id: str, kind: str) -> str:
    """Task / work-item id from a request id: ``("R00001", "desk") -> "T00001-desk"``."""
    return f"T{request_id.removeprefix('R')}-{kind}"


# --- the synthetic world ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scout:
    """A scout on the team. Rows of ``raw/scouts.parquet``."""

    scout_id: str
    name: str
    employment: Employment
    skills: frozenset[str]
    home_region: str
    min_weekly_hours: float
    max_weekly_hours: float
    former_clubs: frozenset[str] = frozenset()
    monthly_salary: float | None = None  # full-time only
    hourly_rate: float | None = None  # freelance only
    joined_month: int = 0  # 0 = initial team; >0 = hired during the year

    def __post_init__(self) -> None:
        _check_literal(self.employment, Employment, "employment")
        _set(self, "skills", frozenset(self.skills))
        _set(self, "former_clubs", frozenset(self.former_clubs))
        if not self.skills:
            raise ValueError(f"scout {self.scout_id} needs at least one skill")
        if not 0 <= self.min_weekly_hours <= self.max_weekly_hours:
            raise ValueError(
                f"scout {self.scout_id}: need 0 <= min_weekly_hours <= max_weekly_hours"
            )

    @property
    def is_freelance(self) -> bool:
        """True for freelancers (paid per hour used), False for salaried scouts."""
        return self.employment == "freelance"

    def has_skill(self, skill_type: str) -> bool:
        """Whether this scout may do work of ``skill_type``."""
        return skill_type in self.skills

    def has_conflict(self, clubs: Iterable[str]) -> bool:
        """Conflict of interest: the scout used to work for any of ``clubs``."""
        return not self.former_clubs.isdisjoint(clubs)


@dataclass(frozen=True, slots=True)
class Fixture:
    """A scheduled match. Rows of ``raw/fixtures.parquet``."""

    fixture_id: str
    date: dt.date
    league: str
    region: str
    home_club: str
    away_club: str

    def involves(self, club: str) -> bool:
        """Whether ``club`` plays in this fixture (so its players can be seen live)."""
        return club in (self.home_club, self.away_club)


@dataclass(frozen=True, slots=True)
class Request:
    """A club's request for a report on one player. Rows of ``raw/requests.parquet``.

    ``desk_hours`` / ``writeup_hours`` are the true durations, sampled at
    generation (D-015: pre-drawn per entity).

    ``rework_draw`` is a uniform number in [0, 1), also pre-drawn at
    generation. When the pre-screen tool is on, its output for this request is
    unusable iff ``rework_draw < rework_rate`` (see :meth:`needs_rework`). Two
    runs with different rework rates therefore fail on nested sets of requests
    instead of on unrelated ones (common random numbers, D-015). The default
    1.0 means "never fails", convenient for hand-built requests in tests.

    ``urgent`` marks a request promised on the shorter, urgent turnaround; its
    ``due_date`` already reflects that, so nothing else needs to read the flag
    to be correct (it is there for metrics and display).
    """

    request_id: str
    client_club: str
    player_club: str
    skill_type: str
    received_date: dt.date
    due_date: dt.date
    needs_live_view: bool
    desk_hours: float
    writeup_hours: float
    period: Period = "future"
    rework_draw: float = 1.0
    urgent: bool = False

    def __post_init__(self) -> None:
        _check_literal(self.period, Period, "period")
        if self.due_date < self.received_date:
            raise ValueError(f"request {self.request_id}: due_date before received_date")
        if self.desk_hours <= 0 or self.writeup_hours <= 0:
            raise ValueError(f"request {self.request_id}: task hours must be positive")
        if not 0.0 <= self.rework_draw <= 1.0:
            raise ValueError(f"request {self.request_id}: rework_draw must be in [0, 1]")

    def needs_rework(self, rework_rate: float) -> bool:
        """Whether the pre-screen output for this request is unusable at ``rework_rate``."""
        return self.rework_draw < rework_rate

    @property
    def conflict_clubs(self) -> frozenset[str]:
        """Clubs a scout must not have worked for to take this request.

        Modelling choice (M0, open for review): both the player's club and the
        client club.
        """
        return frozenset({self.client_club, self.player_club})

    def is_on_time(self, completed_date: dt.date) -> bool:
        """On time = the write-up completes on or before the due date."""
        return completed_date <= self.due_date


# --- units of work ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Task:
    """One step of a request: desk review, optional live view, write-up.

    A live view is pinned to one fixture: its date and region are fixed and it
    occupies the scout's whole day (D-008). ``prerequisites`` lists the task ids
    that must be finished before this one may start (the write-up waits for the
    desk review and, if any, the live view).
    """

    task_id: str
    request_id: str
    kind: TaskKind
    hours: float
    skill_type: str
    due_date: dt.date
    fixture_id: str | None = None  # live view only
    fixture_date: dt.date | None = None  # live view only
    region: str | None = None  # live view only: the fixture's region
    prerequisites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_literal(self.kind, TaskKind, "task kind")
        _set(self, "prerequisites", tuple(self.prerequisites))
        if self.hours <= 0:
            raise ValueError(f"task {self.task_id}: hours must be positive")
        live_fields = (self.fixture_id, self.fixture_date, self.region)
        if self.kind == "live" and None in live_fields:
            raise ValueError(f"live task {self.task_id} needs fixture_id, fixture_date, region")
        if self.kind != "live" and any(f is not None for f in live_fields):
            raise ValueError(f"{self.kind} task {self.task_id} must not reference a fixture")


def tasks_for_request(
    request: Request,
    *,
    live_view_hours: float,
    fixture: Fixture | None = None,
    desk_hours: float | None = None,
) -> tuple[Task, ...]:
    """Split a request into its tasks with the precedence rule built in.

    ``fixture`` is required when the request needs a live view (choosing which
    fixture is the caller's job). ``desk_hours`` overrides the request's true
    desk hours, e.g. when the pre-screen tool shortens the review.
    """
    desk = Task(
        task_id=task_id(request.request_id, "desk"),
        request_id=request.request_id,
        kind="desk",
        hours=request.desk_hours if desk_hours is None else desk_hours,
        skill_type=request.skill_type,
        due_date=request.due_date,
    )
    tasks = [desk]
    if request.needs_live_view:
        if fixture is None:
            raise ValueError(f"request {request.request_id} needs a fixture for its live view")
        tasks.append(
            Task(
                task_id=task_id(request.request_id, "live"),
                request_id=request.request_id,
                kind="live",
                hours=live_view_hours,
                skill_type=request.skill_type,
                due_date=request.due_date,
                fixture_id=fixture.fixture_id,
                fixture_date=fixture.date,
                region=fixture.region,
            )
        )
    writeup = Task(
        task_id=task_id(request.request_id, "writeup"),
        request_id=request.request_id,
        kind="writeup",
        hours=request.writeup_hours,
        skill_type=request.skill_type,
        due_date=request.due_date,
        prerequisites=tuple(t.task_id for t in tasks),
    )
    return (*tasks, writeup)


@dataclass(frozen=True, slots=True)
class WorkItem:
    """What an assignment policy sees: one task, or a desk review + write-up bundle.

    Policies only see work items, so they don't care whether
    ``assignment.unit`` is ``task`` or ``bundle`` (D-010). A work item is a
    snapshot taken at the start of an assignment run:

    * ``hours`` is the work still to do.
    * ``current_scout_id`` is set when the item is assigned but not started
      (greedy policies keep it; the optimiser may move it at a churn penalty).
      Started work is never in the pool; it is a scout's ``frozen_hours``.
    * ``depends_on`` lists ids of unfinished prerequisite items, whether they
      are in the same pool or frozen in a scout's queue. Empty = ready now.
    * ``fixed_date`` and ``region`` are set only for live views.
    * ``request_scout_ids``: scouts already on this item's request *outside the
      pool* (they hold started work on it or finished part of it). Items of the
      same request that are still in the pool are not listed: the policy sees
      them directly. Read only by the optimiser's continuity term (M3).
    """

    item_id: str
    request_id: str
    kind: ItemKind
    task_ids: tuple[str, ...]
    hours: float
    skill_type: str
    received_date: dt.date  # FCFS order
    due_date: dt.date  # EDF order, lateness
    conflict_clubs: frozenset[str] = frozenset()
    current_scout_id: str | None = None
    fixed_date: dt.date | None = None
    region: str | None = None
    depends_on: tuple[str, ...] = ()
    request_scout_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _check_literal(self.kind, ItemKind, "work item kind")
        _set(self, "task_ids", tuple(self.task_ids))
        _set(self, "conflict_clubs", frozenset(self.conflict_clubs))
        _set(self, "depends_on", tuple(self.depends_on))
        _set(self, "request_scout_ids", frozenset(self.request_scout_ids))
        if not self.task_ids:
            raise ValueError(f"work item {self.item_id} has no tasks")
        if self.hours <= 0:
            raise ValueError(f"work item {self.item_id}: hours must be positive")
        is_live = self.kind == "live"
        if is_live != (self.fixed_date is not None) or is_live != (self.region is not None):
            raise ValueError(
                f"work item {self.item_id}: fixed_date and region are set iff it is a live view"
            )

    @property
    def is_ready(self) -> bool:
        """No unfinished prerequisites."""
        return not self.depends_on

    @classmethod
    def from_task(
        cls,
        task: Task,
        request: Request,
        *,
        remaining_hours: float | None = None,
        current_scout_id: str | None = None,
        done_task_ids: frozenset[str] = frozenset(),
        request_scout_ids: frozenset[str] = frozenset(),
    ) -> WorkItem:
        """Wrap one task (``assignment.unit = task``); item id = task id."""
        return cls(
            item_id=task.task_id,
            request_id=task.request_id,
            kind=task.kind,
            task_ids=(task.task_id,),
            hours=task.hours if remaining_hours is None else remaining_hours,
            skill_type=task.skill_type,
            received_date=request.received_date,
            due_date=task.due_date,
            conflict_clubs=request.conflict_clubs,
            current_scout_id=current_scout_id,
            fixed_date=task.fixture_date,
            region=task.region,
            depends_on=tuple(p for p in task.prerequisites if p not in done_task_ids),
            request_scout_ids=request_scout_ids,
        )

    @classmethod
    def bundle(
        cls,
        desk: Task,
        writeup: Task,
        request: Request,
        *,
        current_scout_id: str | None = None,
        done_task_ids: frozenset[str] = frozenset(),
        request_scout_ids: frozenset[str] = frozenset(),
    ) -> WorkItem:
        """Desk review + write-up as one item for one scout (``assignment.unit = bundle``).

        The item still depends on the live view (if any) via the write-up's
        prerequisites.
        """
        if desk.kind != "desk" or writeup.kind != "writeup":
            raise ValueError("bundle needs a desk task and a write-up task")
        if desk.request_id != writeup.request_id:
            raise ValueError("bundle tasks must belong to the same request")
        return cls(
            item_id=task_id(desk.request_id, "bundle"),
            request_id=desk.request_id,
            kind="bundle",
            task_ids=(desk.task_id, writeup.task_id),
            hours=desk.hours + writeup.hours,
            skill_type=desk.skill_type,
            received_date=request.received_date,
            due_date=writeup.due_date,
            conflict_clubs=request.conflict_clubs,
            current_scout_id=current_scout_id,
            depends_on=tuple(
                p for p in writeup.prerequisites if p != desk.task_id and p not in done_task_ids
            ),
            request_scout_ids=request_scout_ids,
        )


# --- assignment run inputs and outputs ---------------------------------------------


@dataclass(frozen=True, slots=True)
class AssignmentWindow:
    """The rolling horizon of one assignment run: ``start`` .. ``end`` inclusive."""

    start: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("assignment window end is before its start")

    @classmethod
    def starting(cls, today: dt.date, horizon_days: int) -> AssignmentWindow:
        """Window of ``horizon_days`` days beginning today (7 days -> today .. today + 6)."""
        if horizon_days < 1:
            raise ValueError("horizon_days must be at least 1")
        return cls(today, today + dt.timedelta(days=horizon_days - 1))

    @property
    def n_days(self) -> int:
        """Number of days in the window, both ends included."""
        return (self.end - self.start).days + 1

    @property
    def days(self) -> tuple[dt.date, ...]:
        """Every date in the window, in order."""
        return tuple(self.start + dt.timedelta(days=i) for i in range(self.n_days))

    def __contains__(self, day: object) -> bool:
        return isinstance(day, dt.date) and self.start <= day <= self.end


@dataclass(frozen=True, slots=True)
class ScoutState:
    """A scout's capacity as seen by one assignment run.

    * ``hours_by_day``: working hours the scout offers on each day of the window
      (0 on days off). Freelancers' weekly draw is already spread over days.
    * ``frozen_hours``: started work still to finish; it is counted against the
      window first, before any new assignment (D-010).
    * ``unavailable_dates``: leave and other absences. Kept separate from
      zero-hour days because a live view may fall on a day with no desk hours
      (e.g. a weekend fixture) but never on a day the scout is away.
    """

    scout: Scout
    hours_by_day: Mapping[dt.date, float] = field(default_factory=dict)
    frozen_hours: float = 0.0
    unavailable_dates: frozenset[dt.date] = frozenset()

    def __post_init__(self) -> None:
        _set(self, "hours_by_day", MappingProxyType(dict(self.hours_by_day)))
        _set(self, "unavailable_dates", frozenset(self.unavailable_dates))
        if self.frozen_hours < 0 or any(h < 0 for h in self.hours_by_day.values()):
            raise ValueError(f"scout {self.scout.scout_id}: hours must be non-negative")

    @property
    def scout_id(self) -> str:
        """Shortcut for ``scout.scout_id``."""
        return self.scout.scout_id

    @property
    def window_hours(self) -> float:
        """Total hours offered across the window, before frozen work."""
        return float(sum(self.hours_by_day.values()))

    @property
    def free_hours(self) -> float:
        """Hours left for new assignments once frozen work is counted (never negative)."""
        return max(0.0, self.window_hours - self.frozen_hours)

    def is_available(self, day: dt.date) -> bool:
        """Whether the scout is not away on ``day`` (live-view date check)."""
        return day not in self.unavailable_dates


@dataclass(frozen=True, slots=True)
class Assignment:
    """One decision of an assignment run: this item goes to this scout.

    ``planned_date`` is the day the work is expected to start (the fixture date
    for a live view); ``None`` means "as soon as the scout's queue allows".
    """

    item_id: str
    scout_id: str
    planned_date: dt.date | None = None
