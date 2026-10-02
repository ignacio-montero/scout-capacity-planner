"""Domain dataclasses: construction, invariants, and the precedence structure."""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

from scout_planner.domain import (
    Assignment,
    AssignmentWindow,
    Fixture,
    Request,
    Scout,
    ScoutState,
    Task,
    WorkItem,
    task_id,
    tasks_for_request,
)

D = dt.date(2027, 1, 4)  # a Monday


def make_scout(**kw: object) -> Scout:
    base: dict[str, object] = {
        "scout_id": "S001",
        "name": "Ana Velmar",
        "employment": "full_time",
        "skills": ["MID-Iberia-ES", "DEF-Iberia-ES"],
        "home_region": "Iberia",
        "min_weekly_hours": 37.5,
        "max_weekly_hours": 37.5,
        "former_clubs": ["Puerto Azul FC"],
        "monthly_salary": 4500.0,
    }
    return Scout(**(base | kw))  # type: ignore[arg-type]


def make_request(*, live: bool = False) -> Request:
    return Request(
        request_id="R00001",
        client_club="Valdoro CF",
        player_club="Puerto Azul FC",
        skill_type="MID-Iberia-ES",
        received_date=D,
        due_date=D + dt.timedelta(days=14),
        needs_live_view=live,
        desk_hours=6.0,
        writeup_hours=3.0,
    )


FIXTURE = Fixture(
    fixture_id="F0001",
    date=D + dt.timedelta(days=5),
    league="Liga Costera",
    region="Iberia",
    home_club="Puerto Azul FC",
    away_club="Atletico Sierra",
)


# --- world objects ----------------------------------------------------------------


def test_scout_normalises_collections_and_answers_questions() -> None:
    scout = make_scout()
    assert scout.skills == frozenset({"MID-Iberia-ES", "DEF-Iberia-ES"})
    assert scout.has_skill("MID-Iberia-ES") and not scout.has_skill("FWD-Nordic-EN")
    assert scout.has_conflict(make_request().conflict_clubs)
    assert not scout.has_conflict({"Atletico Sierra"})
    assert not scout.is_freelance


@pytest.mark.parametrize(
    "bad",
    [
        {"employment": "intern"},
        {"skills": []},
        {"min_weekly_hours": 30, "max_weekly_hours": 20},
    ],
)
def test_scout_rejects_invalid_values(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_scout(**bad)


def test_objects_are_frozen_and_slotted() -> None:
    scout = make_scout()
    with pytest.raises(dataclasses.FrozenInstanceError):
        scout.name = "Other"  # type: ignore[misc]
    assert not hasattr(scout, "__dict__")


def test_fixture_involves_its_clubs() -> None:
    assert FIXTURE.involves("Puerto Azul FC") and not FIXTURE.involves("Valdoro CF")


def test_request_on_time_boundary_and_conflict_clubs() -> None:
    req = make_request()
    assert req.is_on_time(req.due_date)
    assert not req.is_on_time(req.due_date + dt.timedelta(days=1))
    assert req.conflict_clubs == {"Valdoro CF", "Puerto Azul FC"}


def test_request_rejects_due_before_received() -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(make_request(), due_date=D - dt.timedelta(days=1))


# --- tasks and precedence ---------------------------------------------------------


def test_task_id_format() -> None:
    assert task_id("R00001", "desk") == "T00001-desk"


def test_tasks_without_live_view() -> None:
    desk, writeup = tasks_for_request(make_request(), live_view_hours=8.0)
    assert (desk.kind, desk.hours, desk.prerequisites) == ("desk", 6.0, ())
    assert (writeup.kind, writeup.hours) == ("writeup", 3.0)
    assert writeup.prerequisites == (desk.task_id,)


def test_tasks_with_live_view_and_automation_override() -> None:
    desk, live, writeup = tasks_for_request(
        make_request(live=True), live_view_hours=8.0, fixture=FIXTURE, desk_hours=3.6
    )
    assert desk.hours == 3.6
    assert (live.kind, live.hours, live.fixture_date, live.region) == (
        "live",
        8.0,
        FIXTURE.date,
        "Iberia",
    )
    assert set(writeup.prerequisites) == {desk.task_id, live.task_id}


def test_live_view_request_needs_a_fixture() -> None:
    with pytest.raises(ValueError, match="fixture"):
        tasks_for_request(make_request(live=True), live_view_hours=8.0)


@pytest.mark.parametrize(
    "kw",
    [
        {"kind": "live"},  # live without fixture fields
        {"kind": "desk", "region": "Iberia"},  # non-live with a fixture field
        {"kind": "desk", "hours": 0.0},
        {"kind": "review"},
    ],
)
def test_task_rejects_inconsistent_fields(kw: dict[str, object]) -> None:
    base: dict[str, object] = {
        "task_id": "T1",
        "request_id": "R1",
        "hours": 2.0,
        "skill_type": "MID-Iberia-ES",
        "due_date": D,
    }
    with pytest.raises(ValueError):
        Task(**(base | kw))  # type: ignore[arg-type]


# --- work items ---------------------------------------------------------------------


def test_work_item_from_task_tracks_unfinished_prerequisites() -> None:
    req = make_request(live=True)
    desk, live, writeup = tasks_for_request(req, live_view_hours=8.0, fixture=FIXTURE)

    item = WorkItem.from_task(writeup, req)
    assert item.item_id == writeup.task_id
    assert set(item.depends_on) == {desk.task_id, live.task_id}
    assert not item.is_ready

    item = WorkItem.from_task(writeup, req, done_task_ids=frozenset({desk.task_id, live.task_id}))
    assert item.is_ready
    assert item.received_date == req.received_date and item.due_date == req.due_date
    assert item.conflict_clubs == req.conflict_clubs


def test_live_work_item_carries_date_and_region() -> None:
    req = make_request(live=True)
    _, live, _ = tasks_for_request(req, live_view_hours=8.0, fixture=FIXTURE)
    item = WorkItem.from_task(live, req, current_scout_id="S001")
    assert (item.fixed_date, item.region, item.current_scout_id) == (
        FIXTURE.date,
        "Iberia",
        "S001",
    )


def test_work_item_remaining_hours_override() -> None:
    req = make_request()
    desk, _ = tasks_for_request(req, live_view_hours=8.0)
    assert WorkItem.from_task(desk, req, remaining_hours=1.5).hours == 1.5


def test_bundle_combines_desk_and_writeup_but_still_waits_for_live_view() -> None:
    req = make_request(live=True)
    desk, live, writeup = tasks_for_request(req, live_view_hours=8.0, fixture=FIXTURE)
    item = WorkItem.bundle(desk, writeup, req)
    assert item.kind == "bundle"
    assert item.task_ids == (desk.task_id, writeup.task_id)
    assert item.hours == 9.0
    assert item.depends_on == (live.task_id,)
    assert item.fixed_date is None


def test_bundle_rejects_wrong_task_kinds() -> None:
    req = make_request()
    desk, writeup = tasks_for_request(req, live_view_hours=8.0)
    with pytest.raises(ValueError):
        WorkItem.bundle(writeup, desk, req)


def test_work_item_live_fields_must_be_consistent() -> None:
    with pytest.raises(ValueError, match="live view"):
        WorkItem(
            item_id="T1-desk",
            request_id="R1",
            kind="desk",
            task_ids=["T1-desk"],
            hours=4.0,
            skill_type="MID-Iberia-ES",
            received_date=D,
            due_date=D,
            fixed_date=D,
            region="Iberia",
        )


# --- assignment run inputs and outputs --------------------------------------------


def test_assignment_window_rolling_seven_days() -> None:
    window = AssignmentWindow.starting(D, 7)
    assert window.end == D + dt.timedelta(days=6)
    assert window.n_days == 7 == len(window.days)
    assert D in window and window.end in window
    assert window.end + dt.timedelta(days=1) not in window


def test_assignment_window_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        AssignmentWindow(D, D - dt.timedelta(days=1))


def test_scout_state_counts_frozen_work_first() -> None:
    days = AssignmentWindow.starting(D, 7).days
    hours = {d: (7.5 if d.weekday() < 5 else 0.0) for d in days}
    state = ScoutState(
        scout=make_scout(),
        hours_by_day=hours,
        frozen_hours=10.0,
        unavailable_dates={days[2]},
    )
    assert state.scout_id == "S001"
    assert state.window_hours == 37.5
    assert state.free_hours == 27.5
    assert not state.is_available(days[2]) and state.is_available(days[5])

    overloaded = dataclasses.replace(state, frozen_hours=50.0)
    assert overloaded.free_hours == 0.0


def test_scout_state_keeps_its_own_read_only_copy_of_hours() -> None:
    hours = {D: 7.5}
    state = ScoutState(scout=make_scout(), hours_by_day=hours)
    hours[D] = 0.0  # caller mutates its dict afterwards
    assert state.hours_by_day[D] == 7.5
    with pytest.raises(TypeError):
        state.hours_by_day[D] = 1.0  # type: ignore[index]


def test_assignment_constructs() -> None:
    a = Assignment("T00001-live", "S001", planned_date=FIXTURE.date)
    assert (a.item_id, a.scout_id, a.planned_date) == ("T00001-live", "S001", FIXTURE.date)
    assert Assignment("T00001-desk", "S001").planned_date is None
