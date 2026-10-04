"""Hard-constraint checks shared by every policy (``assign/eligibility.py``).

These tests prove the *checker* works: each broken rule is reported. The
policy tests (``test_assign.py``) then run the checker on every policy's
output, so a policy can only pass if it respects every rule.
"""

from __future__ import annotations

import datetime as dt

import pytest

from scout_planner.assign.eligibility import (
    DueDateLedger,
    check_inputs,
    deadline_index,
    due_date_violations,
    eligible_scouts,
    free_by_day,
    ineligibility_reasons,
    is_eligible,
    validate_assignments,
)
from scout_planner.domain import (
    Assignment,
    AssignmentWindow,
    Request,
    Scout,
    ScoutState,
    WorkItem,
    tasks_for_request,
)

D0 = dt.date(2027, 3, 1)  # a Monday
WINDOW = AssignmentWindow.starting(D0, 7)
SAT = D0 + dt.timedelta(days=5)


def day(n: int) -> dt.date:
    return D0 + dt.timedelta(days=n)


def make_state(
    sid: str = "S001",
    skills: tuple[str, ...] = ("MID-North",),
    *,
    region: str = "North",
    former: tuple[str, ...] = (),
    weekday_hours: float = 7.5,
    frozen: float = 0.0,
    unavailable: tuple[dt.date, ...] = (),
) -> ScoutState:
    scout = Scout(sid, f"Scout {sid}", "full_time", skills, region, 37.5, 37.5, former, 4500.0)
    hours = {
        d: (weekday_hours if d.weekday() < 5 and d not in unavailable else 0.0) for d in WINDOW.days
    }
    return ScoutState(scout, hours, frozen, unavailable)


def make_item(
    iid: str = "T00001-desk",
    *,
    skill: str = "MID-North",
    hours: float = 4.0,
    live_on: dt.date | None = None,
    region: str = "North",
    clubs: tuple[str, ...] = ("Rio Claro FC",),
    depends_on: tuple[str, ...] = (),
    due: dt.date | None = None,
) -> WorkItem:
    live = live_on is not None
    return WorkItem(
        item_id=iid,
        request_id="R" + iid[1:6],
        kind="live" if live else "desk",
        task_ids=(iid,),
        hours=8.0 if live else hours,
        skill_type=skill,
        received_date=D0,
        due_date=due or day(10),
        conflict_clubs=clubs,
        fixed_date=live_on,
        region=region if live else None,
        depends_on=depends_on,
    )


# --- pairwise rules -------------------------------------------------------------------


def test_clean_pair_is_eligible() -> None:
    assert is_eligible(make_item(), make_state(), WINDOW)
    assert is_eligible(make_item("T00001-live", live_on=SAT), make_state(), WINDOW)


@pytest.mark.parametrize(
    ("item", "state", "reason"),
    [
        (make_item(skill="GK-South"), make_state(), "skill"),
        (make_item(), make_state(former=("Rio Claro FC",)), "conflict of interest"),
        (make_item(depends_on=("T00001-live",)), make_state(), "precedence"),
        (make_item("T00001-live", live_on=day(7)), make_state(), "outside the window"),
        (make_item("T00001-live", live_on=SAT, region="South"), make_state(), "fixture in South"),
        (
            make_item("T00001-live", live_on=SAT),
            make_state(unavailable=(SAT,)),
            "unavailable",
        ),
    ],
    ids=["skill", "conflict", "precedence", "live-date", "live-region", "live-leave"],
)
def test_each_pairwise_rule_is_reported(item: WorkItem, state: ScoutState, reason: str) -> None:
    reasons = ineligibility_reasons(item, state, WINDOW)
    assert len(reasons) == 1 and reason in reasons[0]
    assert not is_eligible(item, state, WINDOW)


def test_live_view_allowed_on_a_zero_hour_day_but_not_on_leave() -> None:
    """D-008: a weekend fixture is fine (no desk hours that day); leave is not."""
    state = make_state()
    assert state.hours_by_day[SAT] == 0.0
    assert is_eligible(make_item("T00001-live", live_on=SAT), state, WINDOW)


def test_eligible_scouts_also_requires_the_item_to_fit_alone() -> None:
    small, big = make_state("S001", frozen=35.0), make_state("S002")
    result = eligible_scouts([make_item(hours=4.0)], [small, big], WINDOW)
    assert [s.scout_id for s in result["T00001-desk"]] == ["S002"]


def test_malformed_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="duplicate work item"):
        check_inputs([make_item(), make_item()], [make_state()], WINDOW)
    with pytest.raises(ValueError, match="duplicate scout"):
        check_inputs([make_item()], [make_state(), make_state()], WINDOW)
    state = make_state()
    beyond = ScoutState(state.scout, {**state.hours_by_day, day(7): 7.5})
    with pytest.raises(ValueError, match="outside the window"):
        check_inputs([], [beyond], WINDOW)
    on_leave = ScoutState(state.scout, state.hours_by_day, unavailable_dates={D0})
    with pytest.raises(ValueError, match="leave days"):
        check_inputs([], [on_leave], WINDOW)
    check_inputs([make_item()], [make_state(unavailable=(D0,))], WINDOW)  # 0 h on leave: fine


# --- the validator ----------------------------------------------------------------------


def test_valid_answer_has_no_problems() -> None:
    pool = [make_item(), make_item("T00002-live", live_on=SAT)]
    answer = [Assignment("T00001-desk", "S001"), Assignment("T00002-live", "S001", SAT)]
    assert validate_assignments(pool, [make_state()], WINDOW, answer) == []


@pytest.mark.parametrize(
    ("pool", "state", "answer", "problem"),
    [
        ([make_item()], make_state(), [Assignment("T09999-desk", "S001")], "not in the pool"),
        ([make_item()], make_state(), [Assignment("T00001-desk", "S999")], "unknown scout"),
        (
            [make_item()],
            make_state(),
            [Assignment("T00001-desk", "S001")] * 2,
            "more than once",
        ),
        ([make_item(skill="GK-South")], make_state(), [Assignment("T00001-desk", "S001")], "skill"),
        (
            [make_item()],
            make_state(former=("Rio Claro FC",)),
            [Assignment("T00001-desk", "S001")],
            "conflict",
        ),
        (
            [make_item(depends_on=("T00001-live",))],
            make_state(),
            [Assignment("T00001-desk", "S001")],
            "precedence",
        ),
        (
            [make_item("T00001-live", live_on=SAT)],
            make_state(),
            [Assignment("T00001-live", "S001")],  # planned_date missing
            "planned_date",
        ),
        (
            [make_item(hours=8.0)],
            make_state(frozen=30.0),  # 37.5 - 30 = 7.5 h free
            [Assignment("T00001-desk", "S001")],
            "hours cap",
        ),
        (
            [make_item("T00001-live", live_on=SAT), make_item("T00002-live", live_on=SAT)],
            make_state(),
            [Assignment("T00001-live", "S001", SAT), Assignment("T00002-live", "S001", SAT)],
            "2 live views",
        ),
    ],
    ids=[
        "unknown-item",
        "unknown-scout",
        "duplicate",
        "skill",
        "conflict",
        "precedence",
        "planned-date",
        "hours-incl-frozen",
        "two-live-same-day",
    ],
)
def test_validator_reports_each_violation(
    pool: list[WorkItem], state: ScoutState, answer: list[Assignment], problem: str
) -> None:
    problems = validate_assignments(pool, [state], WINDOW, answer)
    assert any(problem in p for p in problems), problems


def test_weekend_live_view_still_counts_its_hours() -> None:
    """A live view on a 0 h Saturday uses 8 h of the window total (D-008)."""
    pool = [make_item("T00001-live", live_on=SAT)]
    answer = [Assignment("T00001-live", "S001", SAT)]
    assert validate_assignments(pool, [make_state(frozen=29.0)], WINDOW, answer) == []
    problems = validate_assignments(pool, [make_state(frozen=30.0)], WINDOW, answer)
    assert any("hours cap" in p for p in problems)


# --- WorkItem.request_scout_ids (continuity input, added in M3) ----------------------


def test_work_item_carries_scouts_already_on_the_request() -> None:
    request = Request(
        "R00001", "Rio Claro FC", "Monte Verde", "MID-North", D0, day(14), False, 6.0, 3.0
    )
    desk, writeup = tasks_for_request(request, live_view_hours=8.0)
    item = WorkItem.from_task(
        writeup, request, done_task_ids=frozenset({desk.task_id}), request_scout_ids={"S001"}
    )
    assert item.request_scout_ids == frozenset({"S001"})
    assert WorkItem.bundle(desk, writeup, request).request_scout_ids == frozenset()


# --- due-date feasibility (EDF condition for one scout) --------------------------------


def test_free_by_day_is_cumulative_after_frozen_work() -> None:
    free = free_by_day(make_state(frozen=10.0), WINDOW)
    assert free == [0, 500, 1250, 2000, 2750, 2750, 2750]  # hundredths of an hour


def test_deadline_is_the_due_day_or_the_first_day_the_item_could_be_done() -> None:
    free = free_by_day(make_state(), WINDOW)  # 7.5 h a day, nothing frozen
    assert deadline_index(make_item(hours=4.0, due=day(2)), free, WINDOW) == 2
    assert deadline_index(make_item(hours=4.0, due=day(30)), free, WINDOW) == 6  # clamped
    assert deadline_index(make_item(hours=10.0, due=D0), free, WINDOW) == 1  # can't be on time
    assert deadline_index(make_item(hours=4.0, due=day(-3)), free, WINDOW) == 0  # overdue
    assert deadline_index(make_item("T00001-live", live_on=day(3)), free, WINDOW) == 3
    assert deadline_index(make_item(hours=40.0), free, WINDOW) is None


def test_due_date_violations_catch_work_that_fits_the_week_but_not_the_day() -> None:
    pool = [
        make_item("T00001-desk", hours=5.0, due=D0),
        make_item("T00002-desk", hours=5.0, due=D0),
    ]
    both = [Assignment("T00001-desk", "S001"), Assignment("T00002-desk", "S001")]
    assert validate_assignments(pool, [make_state()], WINDOW, both) == []  # window total is fine
    problems = due_date_violations(pool, [make_state()], WINDOW, both)
    assert len(problems) == 1 and "due-date capacity" in problems[0]
    assert validate_assignments(pool, [make_state()], WINDOW, both, check_due_dates=True)
    assert due_date_violations(pool, [make_state()], WINDOW, both[:1]) == []


def test_due_date_check_counts_frozen_work_first() -> None:
    pool = [make_item(hours=4.0, due=day(1))]
    answer = [Assignment("T00001-desk", "S001")]
    assert due_date_violations(pool, [make_state(frozen=11.0)], WINDOW, answer) == []
    # 12 h frozen + 4 h due by Tuesday > 15 h by Tuesday: the deadline moves to
    # Wednesday (first day it could be done), so the item is late but still legal.
    assert due_date_violations(pool, [make_state(frozen=12.0)], WINDOW, answer) == []


def test_due_date_ledger_matches_the_violation_check() -> None:
    """Adding items one by one through the ledger never builds a violating answer."""
    state = make_state()
    ledger = DueDateLedger([state], WINDOW)
    first, second = (
        make_item("T00001-desk", hours=5.0, due=D0),
        make_item("T00002-desk", hours=5.0, due=D0),
    )
    assert ledger.fits(first, "S001")
    ledger.add(first, "S001")
    assert not ledger.fits(second, "S001")  # 10 h due today > 7.5 h today
    later = make_item("T00003-desk", hours=5.0, due=day(1))
    assert ledger.fits(later, "S001")  # 10 h by tomorrow <= 15 h
    ledger.add(later, "S001")
    answer = [Assignment("T00001-desk", "S001"), Assignment("T00003-desk", "S001")]
    assert due_date_violations([first, later], [state], WINDOW, answer) == []
    short = DueDateLedger([state], WINDOW, commit_by=0)
    assert short.deadline(make_item(hours=4.0, due=day(9)), "S001") == 0
