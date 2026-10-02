"""Assignment policies (PRD M3): hard constraints for every policy, greedy
semantics, what the optimiser buys over EDF, determinism and time limits.

Every policy answer in this file goes through ``validate_assignments``
(see ``run``), so each test is also a hard-constraint test for that policy.
"""

from __future__ import annotations

import datetime as dt
import inspect
import logging
import time

import numpy as np
import pytest

from scout_planner.assign import POLICIES, assign, get_policy, validate_assignments
from scout_planner.assign import optimiser as opt
from scout_planner.config import AssignmentParams, AssignmentWeights
from scout_planner.domain import Assignment, AssignmentWindow, Scout, ScoutState, WorkItem

D0 = dt.date(2027, 3, 1)  # a Monday
WINDOW = AssignmentWindow.starting(D0, 7)
SAT = D0 + dt.timedelta(days=5)
CFG = AssignmentParams()
ALL = sorted(POLICIES)
GREEDY = ["edf", "fcfs"]


def day(n: int) -> dt.date:
    return D0 + dt.timedelta(days=n)


def make_state(
    sid: str,
    skills: tuple[str, ...] = ("MID-North",),
    *,
    region: str = "North",
    freelance: bool = False,
    rate: float = 40.0,
    former: tuple[str, ...] = (),
    weekday_hours: float = 7.5,
    frozen: float = 0.0,
    unavailable: tuple[dt.date, ...] = (),
) -> ScoutState:
    scout = Scout(
        scout_id=sid,
        name=f"Scout {sid}",
        employment="freelance" if freelance else "full_time",
        skills=skills,
        home_region=region,
        min_weekly_hours=8.0 if freelance else 37.5,
        max_weekly_hours=25.0 if freelance else 37.5,
        former_clubs=former,
        monthly_salary=None if freelance else 4500.0,
        hourly_rate=rate if freelance else None,
    )
    hours = {d: (weekday_hours if d.weekday() < 5 else 0.0) for d in WINDOW.days}
    return ScoutState(scout, hours, frozen, unavailable)


def make_item(
    iid: str,
    *,
    request: str | None = None,
    skill: str = "MID-North",
    hours: float = 4.0,
    received: dt.date = D0,
    due: dt.date | None = None,
    live_on: dt.date | None = None,
    region: str = "North",
    clubs: tuple[str, ...] = ("Rio Claro FC",),
    current: str | None = None,
    depends_on: tuple[str, ...] = (),
    request_scouts: tuple[str, ...] = (),
) -> WorkItem:
    live = live_on is not None
    return WorkItem(
        item_id=iid,
        request_id=request or "R" + iid[1:6],
        kind="live" if live else "desk",
        task_ids=(iid,),
        hours=8.0 if live else hours,
        skill_type=skill,
        received_date=received,
        due_date=due or day(10),
        conflict_clubs=clubs,
        current_scout_id=current,
        fixed_date=live_on,
        region=region if live else None,
        depends_on=depends_on,
        request_scout_ids=request_scouts,
    )


def run(
    name: str,
    pool: list[WorkItem],
    scouts: list[ScoutState],
    cfg: AssignmentParams = CFG,
    seed: int = 0,
) -> dict[str, str]:
    """Run one policy, check every hard constraint, return ``{item_id: scout_id}``."""
    answer = get_policy(name)(pool, scouts, WINDOW, cfg, np.random.default_rng(seed))
    assert validate_assignments(pool, scouts, WINDOW, answer) == []
    return {a.item_id: a.scout_id for a in answer}


# --- the common interface ---------------------------------------------------------


def test_registry_has_the_three_policies_with_one_signature() -> None:
    assert set(POLICIES) == {"fcfs", "edf", "optimiser"}
    for policy in POLICIES.values():
        params = list(inspect.signature(policy).parameters)
        assert params == ["pool", "scouts", "window", "cfg", "rng"]


def test_unknown_policy_name_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown assignment policy"):
        get_policy("random")


def test_assign_dispatches_on_cfg_policy() -> None:
    pool = [make_item("T00001-desk")]
    scouts = [make_state("S001")]
    for name in ALL:
        cfg = AssignmentParams(policy=name)
        answer = assign(pool, scouts, WINDOW, cfg, np.random.default_rng(0))
        assert answer == [Assignment("T00001-desk", "S001")]


@pytest.mark.parametrize("name", ALL)
def test_empty_pool_or_no_scouts_gives_no_assignments(name: str) -> None:
    assert run(name, [], [make_state("S001")]) == {}
    assert run(name, [make_item("T00001-desk")], []) == {}


# --- hard constraints, every policy ---------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_skill_match(name: str) -> None:
    pool = [make_item("T00001-desk", skill="GK-South")]
    keeper = make_state("S002", ("GK-South",), weekday_hours=1.0)  # fewer hours: not preferred
    assert run(name, pool, [make_state("S001"), keeper]) == {"T00001-desk": "S002"}
    assert run(name, pool, [make_state("S001")]) == {}


@pytest.mark.parametrize("name", ALL)
def test_hours_cap_counts_frozen_work_first(name: str) -> None:
    pool = [make_item(f"T0000{k}-desk", hours=4.0) for k in (1, 2, 3)]
    # 37.5 h in the window - 30 h of started work = 7.5 h free: room for one 4 h item.
    assert len(run(name, pool, [make_state("S001", frozen=30.0)])) == 1
    assert run(name, pool, [make_state("S001", frozen=40.0)]) == {}
    assert len(run(name, pool, [make_state("S001")])) == 3


@pytest.mark.parametrize("name", ALL)
def test_conflict_of_interest(name: str) -> None:
    pool = [make_item("T00001-desk", clubs=("Rio Claro FC", "Monte Verde"))]
    ex_player = make_state("S001", former=("Monte Verde",))  # more hours, but conflicted
    clean = make_state("S002", weekday_hours=2.0)
    assert run(name, pool, [ex_player, clean]) == {"T00001-desk": "S002"}
    assert run(name, pool, [ex_player]) == {}


@pytest.mark.parametrize("name", ALL)
def test_precedence_write_up_waits(name: str) -> None:
    pool = [
        make_item("T00001-desk"),
        make_item("T00001-writeup", hours=2.0, depends_on=("T00001-desk",)),
    ]
    assert run(name, pool, [make_state("S001")]) == {"T00001-desk": "S001"}


@pytest.mark.parametrize("name", ALL)
def test_live_view_date_must_be_inside_the_window(name: str) -> None:
    pool = [make_item("T00001-live", live_on=day(7)), make_item("T00002-live", live_on=day(6))]
    assert run(name, pool, [make_state("S001")]) == {"T00002-live": "S001"}


@pytest.mark.parametrize("name", ALL)
def test_live_view_needs_a_scout_from_the_fixture_region(name: str) -> None:
    pool = [make_item("T00001-live", live_on=SAT, region="South")]
    north = make_state("S001", region="North")
    south = make_state("S002", region="South", weekday_hours=2.0)
    assert run(name, pool, [north, south]) == {"T00001-live": "S002"}
    assert run(name, pool, [north]) == {}


@pytest.mark.parametrize("name", ALL)
def test_live_view_planned_on_the_fixture_date(name: str) -> None:
    pool = [make_item("T00001-live", live_on=SAT)]
    answer = get_policy(name)(pool, [make_state("S001")], WINDOW, CFG, np.random.default_rng(0))
    assert answer == [Assignment("T00001-live", "S001", planned_date=SAT)]


@pytest.mark.parametrize("name", ALL)
def test_live_view_never_on_a_day_off(name: str) -> None:
    pool = [make_item("T00001-live", live_on=SAT)]
    away = make_state("S001", unavailable=(SAT,))
    here = make_state("S002", weekday_hours=2.0)
    assert run(name, pool, [away, here]) == {"T00001-live": "S002"}
    assert run(name, pool, [away]) == {}


@pytest.mark.parametrize("name", ALL)
def test_one_live_view_per_scout_per_date(name: str) -> None:
    same_day = [make_item("T00001-live", live_on=SAT), make_item("T00002-live", live_on=SAT)]
    assert len(run(name, same_day, [make_state("S001")])) == 1
    assert len(run(name, same_day, [make_state("S001"), make_state("S002")])) == 2
    two_days = [make_item("T00001-live", live_on=SAT), make_item("T00002-live", live_on=day(6))]
    assert len(run(name, two_days, [make_state("S001")])) == 2


@pytest.mark.parametrize("name", ALL)
def test_weekend_live_view_counts_eight_hours_against_the_window(name: str) -> None:
    pool = [make_item("T00001-live", live_on=SAT)]
    assert run(name, pool, [make_state("S001", frozen=29.0)]) == {"T00001-live": "S001"}
    assert run(name, pool, [make_state("S001", frozen=30.0)]) == {}  # 7.5 h free < 8 h


@pytest.mark.parametrize("name", ALL)
def test_started_work_is_frozen_not_in_the_pool(name: str) -> None:
    """Frozen hours shrink the scout; they are never handed out again."""
    scouts = [make_state("S001", frozen=37.5), make_state("S002", weekday_hours=1.0)]
    assert run(name, [make_item("T00001-desk")], scouts) == {"T00001-desk": "S002"}


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_instances_never_break_a_hard_constraint(name: str, seed: int) -> None:
    pool, scouts, _ = random_instance(120, 25, seed)
    run(name, pool, scouts, AssignmentParams(time_limit_s=0.2))


# --- greedy semantics ---------------------------------------------------------------


def test_fcfs_and_edf_order_differently() -> None:
    """One slot, two items: the older one is due later, the newer one sooner."""
    pool = [
        make_item("T00001-desk", hours=6.0, received=day(-10), due=day(5)),
        make_item("T00002-desk", hours=6.0, received=day(-2), due=day(2)),
    ]
    scouts = [make_state("S001", weekday_hours=2.0)]  # 10 h: room for one item
    assert run("fcfs", pool, scouts) == {"T00001-desk": "S001"}
    assert run("edf", pool, scouts) == {"T00002-desk": "S001"}


@pytest.mark.parametrize("name", GREEDY)
def test_greedy_prefers_salaried_then_least_loaded(name: str) -> None:
    pool = [make_item("T00001-desk")]
    freelancer = make_state("S001", freelance=True)
    busy = make_state("S002", weekday_hours=3.0)
    idle = make_state("S003", weekday_hours=5.0)
    assert run(name, pool, [freelancer, busy, idle]) == {"T00001-desk": "S003"}
    assert run(name, pool, [freelancer]) == {"T00001-desk": "S001"}


@pytest.mark.parametrize("name", GREEDY)
def test_greedy_keeps_the_current_scout(name: str) -> None:
    """Never reassigns (D-010): even when another scout would be preferred now."""
    pool = [make_item("T00001-desk", current="S001")]
    scouts = [make_state("S001", freelance=True), make_state("S002")]
    assert run(name, pool, scouts) == {"T00001-desk": "S001"}


@pytest.mark.parametrize("name", GREEDY)
def test_greedy_does_not_move_an_item_whose_scout_can_no_longer_take_it(name: str) -> None:
    pool = [make_item("T00001-desk", hours=6.0, current="S001")]
    scouts = [make_state("S001", frozen=35.0), make_state("S002")]
    assert run(name, pool, scouts) == {}


@pytest.mark.parametrize("name", GREEDY)
def test_greedy_commitments_claim_hours_before_new_items(name: str) -> None:
    pool = [
        make_item("T00001-desk", hours=6.0, due=day(1)),  # more urgent, but new
        make_item("T00002-desk", hours=6.0, due=day(9), current="S001"),
    ]
    scouts = [make_state("S001", weekday_hours=2.0)]
    assert run(name, pool, scouts) == {"T00002-desk": "S001"}


# --- what the optimiser buys --------------------------------------------------------


def tight_deadline_case() -> tuple[list[WorkItem], list[ScoutState]]:
    """A scarce two-skill scout that EDF spends on work a one-skill scout could do.

    A (MID + FWD, 10 h) and B (MID only, 8 h). Item 1 is MID and due first,
    item 2 is FWD. EDF takes item 1 first and gives it to A (most hours left),
    leaving no FWD scout for item 2. Both items are due inside the window, so
    unassigned here means late.
    """
    pool = [
        make_item("T00001-desk", skill="MID-North", hours=6.0, due=day(2)),
        make_item("T00002-desk", skill="FWD-North", hours=6.0, due=day(4)),
        make_item("T00003-desk", skill="MID-North", hours=2.0, due=day(5)),
    ]
    a = make_state("S001", ("MID-North", "FWD-North"), weekday_hours=2.0)
    b = make_state("S002", ("MID-North",), weekday_hours=1.6)
    return pool, [a, b]


def test_optimiser_beats_edf_on_a_tight_deadline_case() -> None:
    pool, scouts = tight_deadline_case()
    on_time = {name: len(run(name, pool, scouts)) for name in ALL}
    assert on_time == {"edf": 2, "fcfs": 2, "optimiser": 3}
    answer = run("optimiser", pool, scouts)
    assert (answer["T00001-desk"], answer["T00002-desk"]) == ("S002", "S001")


def test_optimiser_prefers_salaried_over_freelance() -> None:
    pool = [make_item("T00001-desk")]
    freelancer = make_state("S001", freelance=True)  # more hours, but paid per hour
    salaried = make_state("S002", weekday_hours=1.0)
    assert run("optimiser", pool, [freelancer, salaried]) == {"T00001-desk": "S002"}


def test_optimiser_uses_a_freelancer_for_urgent_work_only() -> None:
    """Default weights: lateness x urgency vs freelance cost (see ``urgency``)."""
    freelancer = [make_state("S001", freelance=True, rate=40.0)]
    urgent = [make_item("T00001-desk", hours=6.0, due=day(0))]  # due today: slack 0
    relaxed = [make_item("T00001-desk", hours=6.0, due=day(13))]  # slack 13
    assert run("optimiser", urgent, freelancer) == {"T00001-desk": "S001"}
    assert run("optimiser", relaxed, freelancer) == {}
    # With cost weighted at zero, any work is worth doing now.
    free = AssignmentParams(weights=AssignmentWeights(cost=0.0))
    assert run("optimiser", relaxed, freelancer, free) == {"T00001-desk": "S001"}


def test_optimiser_moves_a_not_started_item_when_it_pays() -> None:
    """Churn (3) is cheap next to leaving an urgent item for the scarce scout undone."""
    pool = [
        make_item("T00001-desk", skill="MID-North", hours=6.0, due=day(6), current="S001"),
        make_item("T00002-desk", skill="FWD-North", hours=6.0, due=day(2)),
    ]
    a = make_state("S001", ("MID-North", "FWD-North"), weekday_hours=2.0)
    b = make_state("S002", ("MID-North",), weekday_hours=1.6)
    assert run("edf", pool, [a, b]) == {"T00001-desk": "S001"}
    assert run("optimiser", pool, [a, b]) == {"T00001-desk": "S002", "T00002-desk": "S001"}


def test_optimiser_keeps_the_current_scout_when_moving_gains_nothing() -> None:
    pool = [make_item("T00001-desk", current="S001")]
    scouts = [make_state("S001", weekday_hours=1.0), make_state("S002")]
    assert run("optimiser", pool, scouts) == {"T00001-desk": "S001"}


def test_churn_weight_decides_whether_a_cost_saving_move_happens() -> None:
    """Moving 4 h from a freelancer (40/h) to a salaried scout saves 160."""
    pool = [make_item("T00001-desk", hours=4.0, current="S001")]
    scouts = [make_state("S001", freelance=True), make_state("S002")]
    assert run("optimiser", pool, scouts) == {"T00001-desk": "S002"}
    sticky = AssignmentParams(weights=AssignmentWeights(churn=500.0))
    assert run("optimiser", pool, scouts, sticky) == {"T00001-desk": "S001"}


def test_dropping_a_committed_item_also_pays_churn() -> None:
    """Relaxed item on a freelancer: keeping costs 160, dropping saves that but
    leaves ~91 of lateness. Churn on a drop is what keeps the plan stable."""
    pool = [make_item("T00001-desk", hours=4.0, current="S001")]
    freelancer = [make_state("S001", freelance=True)]
    no_churn = AssignmentParams(weights=AssignmentWeights(churn=0.0))
    assert run("optimiser", pool, freelancer, no_churn) == {}
    sticky = AssignmentParams(weights=AssignmentWeights(churn=100.0))
    assert run("optimiser", pool, freelancer, sticky) == {"T00001-desk": "S001"}


def test_optimiser_keeps_a_request_with_the_scout_already_on_it() -> None:
    """Continuity: S002 did the desk review, so the write-up goes to S002 too."""
    pool = [make_item("T00001-writeup", hours=2.0, request_scouts=("S002",))]
    scouts = [make_state("S001"), make_state("S002", weekday_hours=1.0)]
    assert run("edf", pool, scouts) == {"T00001-writeup": "S001"}  # least loaded
    assert run("optimiser", pool, scouts) == {"T00001-writeup": "S002"}


def test_optimiser_gives_one_request_to_one_scout_when_it_can() -> None:
    pool = [
        make_item("T00001-desk", request="R00001", hours=4.0),
        make_item("T00001-live", request="R00001", live_on=day(3)),
    ]
    scouts = [make_state("S001"), make_state("S002")]
    answer = run("optimiser", pool, scouts)
    assert len(answer) == 2 and len(set(answer.values())) == 1


def test_optimiser_serves_overdue_work_first() -> None:
    pool = [
        make_item("T00001-desk", hours=6.0, due=day(1)),
        make_item("T00002-desk", hours=6.0, received=day(-20), due=day(-3)),
    ]
    assert run("optimiser", pool, [make_state("S001", weekday_hours=2.0)]) == {
        "T00002-desk": "S001"
    }


# --- urgency --------------------------------------------------------------------------


def test_slack_counts_today_as_a_working_day() -> None:
    assert opt.slack_days(D0, D0, 1) == 0  # due today, one day of work: just in time
    assert opt.slack_days(day(13), D0, 2) == 12


def test_urgency_grows_as_slack_shrinks_and_overdue_is_highest() -> None:
    values = [opt.urgency(s, overdue=False) for s in (13, 6, 2, 1, 0)]
    assert values == sorted(values) and values[-1] == opt.URGENCY_MAX
    assert opt.urgency(-3, overdue=False) == opt.URGENCY_MAX
    assert opt.urgency(0, overdue=True) > opt.URGENCY_MAX


def test_remaining_work_days_respects_a_late_fixture() -> None:
    desk = make_item("T00001-desk", hours=4.0)
    live = make_item("T00001-live", live_on=day(5))
    writeup = make_item("T00001-writeup", hours=3.0, depends_on=("T00001-live",))
    assert opt.remaining_work_days([desk, writeup], D0) == 1  # 7 h of work
    assert opt.remaining_work_days([desk, live, writeup], D0) == 7  # fixture day 6 + write-up


def test_freelancer_without_a_rate_fails_loudly() -> None:
    state = make_state("S001", freelance=True)
    broken = ScoutState(
        Scout("S001", "Scout S001", "freelance", ("MID-North",), "North", 8.0, 25.0),
        state.hours_by_day,
    )
    with pytest.raises(ValueError, match="hourly_rate"):
        run("optimiser", [make_item("T00001-desk")], [broken])


# --- determinism, time limit, fallback --------------------------------------------------


def random_instance(
    n_items: int, n_scouts: int, seed: int
) -> tuple[list[WorkItem], list[ScoutState], np.random.Generator]:
    """A busy, messy day: every kind of item and constraint, at a realistic ratio."""
    r = np.random.default_rng(seed)
    regions = ["North", "South", "East", "West"]
    skills = [f"{p}-{reg}" for p in ("GK", "DEF", "MID", "FWD") for reg in regions][:15]
    clubs = [f"Club {k}" for k in range(40)]
    scouts = []
    for k in range(n_scouts):
        freelance = k % 5 >= 3  # ~60/40 salaried/freelance
        state = make_state(
            f"S{k:03d}",
            tuple(r.choice(skills, size=int(r.integers(1, 5)), replace=False)),
            region=regions[k % 4],
            freelance=freelance,
            rate=41.5,
            former=tuple(r.choice(clubs, size=2, replace=False)),
            weekday_hours=float(r.uniform(1.6, 5.0)) if freelance else 7.5,
            frozen=float(r.uniform(0, 12)),
            unavailable=(WINDOW.days[int(r.integers(7))],) if r.random() < 0.2 else (),
        )
        scouts.append(state)
    pool = []
    for j in range(n_items):
        received = day(-int(r.integers(0, 16)))
        skill = str(skills[int(r.integers(len(skills)))])
        live = r.random() < 0.2
        pool.append(
            make_item(
                f"T{j:05d}-{'live' if live else 'desk'}",
                request=f"R{j // 2:05d}",
                skill=skill,
                hours=float(r.uniform(2.0, 10.0)),
                received=received,
                due=received + dt.timedelta(days=14),
                live_on=WINDOW.days[int(r.integers(7))] if live else None,
                region=skill.split("-")[1],
                clubs=tuple(r.choice(clubs, size=2, replace=False)),
                current=f"S{int(r.integers(n_scouts)):03d}" if r.random() < 0.15 else None,
                depends_on=("T99999-desk",) if r.random() < 0.1 else (),
                request_scouts=(f"S{int(r.integers(n_scouts)):03d}",) if r.random() < 0.2 else (),
            )
        )
    return pool, scouts, r


def test_optimiser_is_deterministic() -> None:
    pool, scouts, _ = random_instance(200, 40, seed=7)
    cfg = AssignmentParams(time_limit_s=0.3)
    first = opt.solve_assignment(pool, scouts, WINDOW, cfg, np.random.default_rng(11))
    second = opt.solve_assignment(pool, scouts, WINDOW, cfg, np.random.default_rng(11))
    assert first.assignments == second.assignments
    assert first.objective == second.objective


def test_optimiser_draws_exactly_one_number_per_call() -> None:
    """The assignment stream advances the same way whatever the pool holds (D-015)."""
    used, fresh = np.random.default_rng(5), np.random.default_rng(5)
    opt.optimiser([], [], WINDOW, CFG, used)
    fresh.integers(0, 2**31 - 1)
    assert used.random() == fresh.random()


def test_optimiser_respects_its_time_limit() -> None:
    pool, scouts, _ = random_instance(400, 60, seed=3)
    cfg = AssignmentParams(time_limit_s=0.1)  # the smallest allowed
    report = opt.solve_assignment(pool, scouts, WINDOW, cfg, np.random.default_rng(0))
    assert report.status in ("OPTIMAL", "FEASIBLE") and not report.fell_back_to_edf
    assert report.deterministic_time <= cfg.time_limit_s * 1.05
    assert report.wall_time_s <= opt.WALL_CLOCK_SAFETY * cfg.time_limit_s * 1.05
    assert validate_assignments(pool, scouts, WINDOW, report.assignments) == []


@pytest.mark.slow
def test_benchmark_400_items_60_scouts_within_twice_the_time_limit() -> None:
    """Default limit (1 s): model build + solve must stay under 2 s end to end."""
    pool, scouts, _ = random_instance(400, 60, seed=1)
    started = time.perf_counter()
    report = opt.solve_assignment(pool, scouts, WINDOW, CFG, np.random.default_rng(0))
    elapsed = time.perf_counter() - started
    assert elapsed <= 2 * CFG.time_limit_s, f"{elapsed:.2f} s"
    assert report.status in ("OPTIMAL", "FEASIBLE")
    assert validate_assignments(pool, scouts, WINDOW, report.assignments) == []


def test_optimiser_falls_back_to_edf_when_the_solver_finds_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A zero wall-clock cap stops the solver before its first solution.
    monkeypatch.setattr(opt, "WALL_CLOCK_SAFETY", 0.0)
    pool, scouts = tight_deadline_case()
    with caplog.at_level(logging.WARNING, logger=opt.__name__):
        report = opt.solve_assignment(pool, scouts, WINDOW, CFG, np.random.default_rng(0))
    assert report.fell_back_to_edf and report.status == "UNKNOWN"
    assert report.assignments == get_policy("edf")(
        pool, scouts, WINDOW, CFG, np.random.default_rng(0)
    )
    assert "falling back to EDF" in caplog.text
