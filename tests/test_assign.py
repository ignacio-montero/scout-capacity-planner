"""Assignment policies (PRD M3): hard constraints for every policy, greedy
semantics, what the optimiser buys over EDF, determinism and time limits.

Every policy answer in this file goes through ``validate_assignments``
(see ``run``), so each test is also a hard-constraint test for that policy.
The optimiser's answers are also checked for due-date feasibility.
"""

from __future__ import annotations

import datetime as dt
import inspect
import logging
import time

import numpy as np
import pytest

from scout_planner.assign import (
    POLICIES,
    assign,
    assign_with_report,
    get_policy,
    validate_assignments,
)
from scout_planner.assign import optimiser as opt
from scout_planner.config import AssignmentParams, AssignmentWeights, CostParams
from scout_planner.domain import Assignment, AssignmentWindow, Scout, ScoutState, WorkItem

D0 = dt.date(2027, 3, 1)  # a Monday
WINDOW = AssignmentWindow.starting(D0, 7)
SAT = D0 + dt.timedelta(days=5)
COST = CostParams()  # late penalty 1000 per report
# The *plain* model, pinned so each mechanism is tested on its own arithmetic and
# the tests don't move with config defaults: lateness weight 1.0 (one late penalty
# at zero slack), freelance hours at the full rate, commit anything that fits the
# window, urgency from own slack only. The calibrated defaults (premium, commit
# horizon, load-aware) have their own tests below and are used by the random,
# order, time-limit and benchmark tests via DEFAULTS.
CFG = AssignmentParams(
    weights=AssignmentWeights(lateness=1.0),
    cost_basis="full",
    commit_buffer_days=None,
    load_aware=False,
)
DEFAULTS = AssignmentParams()
ALL = sorted(POLICIES)
GREEDY = ["edf", "fcfs"]


def day(n: int) -> dt.date:
    return D0 + dt.timedelta(days=n)


def weights(**kw: float) -> AssignmentParams:
    return CFG.model_copy(update={"weights": CFG.weights.model_copy(update=kw)})


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
    hours = {
        d: (weekday_hours if d.weekday() < 5 and d not in unavailable else 0.0) for d in WINDOW.days
    }
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


def answer_of(
    name: str,
    pool: list[WorkItem],
    scouts: list[ScoutState],
    cfg: AssignmentParams = CFG,
    seed: int = 0,
) -> list[Assignment]:
    """Run one policy and check every hard constraint (plus due dates for the optimiser)."""
    answer = get_policy(name)(pool, scouts, WINDOW, cfg, np.random.default_rng(seed), cost=COST)
    strict = name == "optimiser"
    assert validate_assignments(pool, scouts, WINDOW, answer, check_due_dates=strict) == []
    return answer


def run(
    name: str,
    pool: list[WorkItem],
    scouts: list[ScoutState],
    cfg: AssignmentParams = CFG,
    seed: int = 0,
) -> dict[str, str]:
    """``{item_id: scout_id}`` of a checked policy answer."""
    return {a.item_id: a.scout_id for a in answer_of(name, pool, scouts, cfg, seed)}


def replay_on_time(pool: list[WorkItem], scouts: list[ScoutState], answer: list[Assignment]) -> int:
    """Items finished on time when each scout works its queue day by day.

    Independent of the policies' own bookkeeping: each scout first finishes
    its frozen work, then its assigned items earliest-due-first, using the
    hours it offers each day. An item is on time if the day its last hour is
    worked is on or before its due date. (Desk-type items only.)
    """
    items = {i.item_id: i for i in pool}
    on_time = 0
    for state in scouts:
        queue = sorted(
            (items[a.item_id] for a in answer if a.scout_id == state.scout_id),
            key=lambda i: (i.due_date, i.item_id),
        )
        assert all(i.fixed_date is None for i in queue), "replay handles desk work only"
        need = state.frozen_hours
        for item in queue:
            need += item.hours
            worked = 0.0
            for d in WINDOW.days:
                worked += state.hours_by_day.get(d, 0.0)
                if worked >= need - 1e-9:
                    on_time += d <= item.due_date
                    break
    return on_time


# --- the common interface ---------------------------------------------------------


def test_registry_has_the_three_policies_with_one_signature() -> None:
    assert set(POLICIES) == {"fcfs", "edf", "optimiser"}
    for policy in POLICIES.values():
        params = inspect.signature(policy).parameters
        assert list(params) == ["pool", "scouts", "window", "cfg", "rng", "cost"]
        assert params["cost"].kind is inspect.Parameter.KEYWORD_ONLY


def test_unknown_policy_name_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown assignment policy"):
        get_policy("random")


def test_assign_dispatches_on_cfg_policy() -> None:
    pool = [make_item("T00001-desk")]
    scouts = [make_state("S001")]
    for name in ALL:
        cfg = CFG.model_copy(update={"policy": name})
        answer = assign(pool, scouts, WINDOW, cfg, np.random.default_rng(0), cost=COST)
        assert answer == [Assignment("T00001-desk", "S001")]


def test_assign_with_report_exposes_the_solver_report_for_the_optimiser_only() -> None:
    pool, scouts = [make_item("T00001-desk")], [make_state("S001")]
    for name in ALL:
        cfg = CFG.model_copy(update={"policy": name})
        out = assign_with_report(pool, scouts, WINDOW, cfg, np.random.default_rng(0), cost=COST)
        assert out.assignments == [Assignment("T00001-desk", "S001")]
        if name == "optimiser":
            assert out.solve is not None and out.solve.status == "OPTIMAL"
            assert not out.solve.fell_back_to_edf and not out.solve.hit_wall_clock
            assert out.solve.gap == 0.0
        else:
            assert out.solve is None


@pytest.mark.parametrize("name", ALL)
def test_empty_pool_or_no_scouts_gives_no_assignments(name: str) -> None:
    assert run(name, [], [make_state("S001")]) == {}
    assert run(name, [make_item("T00001-desk")], []) == {}


@pytest.mark.parametrize("name", ALL)
def test_malformed_scout_state_fails_loudly(name: str) -> None:
    pool = [make_item("T00001-desk")]
    outside = make_state("S001")
    outside = ScoutState(outside.scout, {**outside.hours_by_day, day(7): 7.5})
    leave_with_hours = ScoutState(outside.scout, {D0: 7.5}, unavailable_dates={D0})
    for bad in (outside, leave_with_hours):
        with pytest.raises(ValueError):
            run(name, pool, [bad])


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
    answer = answer_of(name, pool, [make_state("S001")])
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
    pool, scouts = random_instance(120, 25, seed)
    run(name, pool, scouts, CFG.model_copy(update={"time_limit_s": 0.2}))
    run(name, pool, scouts, DEFAULTS.model_copy(update={"time_limit_s": 0.2}))


@pytest.mark.parametrize("name", ALL)
def test_answer_does_not_depend_on_input_order(name: str) -> None:
    pool, scouts = random_instance(120, 25, seed=4)
    cfg = DEFAULTS.model_copy(update={"time_limit_s": 0.2})
    forward = answer_of(name, pool, scouts, cfg)
    backward = answer_of(name, pool[::-1], scouts[::-1], cfg)
    assert forward == backward


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
    """Never reassigns by choice (D-010): even when another scout would be preferred now."""
    pool = [make_item("T00001-desk", current="S001")]
    scouts = [make_state("S001", freelance=True), make_state("S002")]
    assert run(name, pool, scouts) == {"T00001-desk": "S001"}


@pytest.mark.parametrize("name", GREEDY)
def test_greedy_replaces_a_commitment_whose_scout_ran_out_of_hours(name: str) -> None:
    pool = [make_item("T00001-desk", hours=6.0, current="S001")]
    scouts = [make_state("S001", frozen=35.0), make_state("S002")]
    assert run(name, pool, scouts) == {"T00001-desk": "S002"}


@pytest.mark.parametrize("name", ALL)
def test_forced_move_live_view_today_scout_on_leave(name: str) -> None:
    """S001 holds today's live view but is on leave today: S002 takes it (not churn)."""
    pool = [make_item("T00001-live", live_on=D0, current="S001")]
    scouts = [make_state("S001", unavailable=(D0,)), make_state("S002")]
    assert run(name, pool, scouts) == {"T00001-live": "S002"}


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

    A (MID + FWD, 8 h/day) and B (MID only, 7.5 h/day). Two 6 h items are due
    today, one MID and one FWD; a third MID item is due tomorrow. EDF gives
    the MID item due today to A (most hours left) and then the FWD one to A
    as well: 12 h of work due today for a scout with 8 h today.
    """
    pool = [
        make_item("T00001-desk", skill="MID-North", hours=6.0, due=D0),
        make_item("T00002-desk", skill="FWD-North", hours=6.0, due=D0),
        make_item("T00003-desk", skill="MID-North", hours=6.0, due=day(1)),
    ]
    a = make_state("S001", ("MID-North", "FWD-North"), weekday_hours=8.0)
    b = make_state("S002", ("MID-North",), weekday_hours=7.5)
    return pool, [a, b]


def test_optimiser_beats_edf_on_a_tight_deadline_case() -> None:
    """Counted by a day-by-day replay of each scout's queue, not by the policies' own view."""
    pool, scouts = tight_deadline_case()
    on_time = {name: replay_on_time(pool, scouts, answer_of(name, pool, scouts)) for name in ALL}
    assert on_time == {"edf": 2, "fcfs": 2, "optimiser": 3}

    # EDF promised more than a scout can do by the due date; the optimiser never does.
    edf_answer = answer_of("edf", pool, scouts)
    assert validate_assignments(pool, scouts, WINDOW, edf_answer, check_due_dates=True)
    answer = run("optimiser", pool, scouts)
    assert (answer["T00001-desk"], answer["T00002-desk"]) == ("S002", "S001")


def test_optimiser_never_promises_more_than_a_day_holds() -> None:
    """10 h due today with 7.5 h today: only one of two 5 h items is taken now."""
    pool = [make_item(f"T0000{k}-desk", hours=5.0, due=D0) for k in (1, 2)]
    scouts = [make_state("S001")]
    assert len(run("edf", pool, scouts)) == 2  # fits the week, so EDF takes both
    assert len(run("optimiser", pool, scouts)) == 1


def test_optimiser_prefers_salaried_over_freelance() -> None:
    pool = [make_item("T00001-desk")]
    freelancer = make_state("S001", freelance=True)  # more hours, but paid per hour
    salaried = make_state("S002", weekday_hours=1.0)
    assert run("optimiser", pool, [freelancer, salaried]) == {"T00001-desk": "S002"}


def test_optimiser_uses_a_freelancer_for_urgent_work_only() -> None:
    """Lateness (late penalty x urgency) vs freelance cost (6 h x 40 = 240)."""
    freelancer = [make_state("S001", freelance=True, rate=40.0)]
    urgent = [make_item("T00001-desk", hours=6.0, due=day(0))]  # slack 0: 1000 at stake
    relaxed = [make_item("T00001-desk", hours=6.0, due=day(13))]  # slack 13: ~71
    assert run("optimiser", urgent, freelancer) == {"T00001-desk": "S001"}
    assert run("optimiser", relaxed, freelancer) == {}
    # With cost weighted at zero, any work is worth doing now.
    assert run("optimiser", relaxed, freelancer, weights(cost=0.0)) == {"T00001-desk": "S001"}


def test_premium_cost_basis_hires_a_freelancer_earlier() -> None:
    """Slack 10: ~91 at stake. Full rate 6 h x 40 = 240 > 91: wait. The premium
    over a salaried hour (40 - 27.7 = 12.3/h, ~74) < 91: do it now."""
    pool = [make_item("T00001-desk", hours=6.0, due=day(10))]
    freelancer = [make_state("S001", freelance=True, rate=40.0)]
    assert run("optimiser", pool, freelancer) == {}
    premium = CFG.model_copy(update={"cost_basis": "premium"})
    assert run("optimiser", pool, freelancer, premium) == {"T00001-desk": "S001"}


def test_hourly_cost_by_basis() -> None:
    freelancer = make_state("S001", freelance=True, rate=40.0).scout
    salaried = make_state("S002").scout
    tie = opt.SALARIED_TIE_BREAK
    assert opt.hourly_cost(salaried, "premium", 27.7) == 0.0
    assert opt.hourly_cost(freelancer, "full", 27.7) == pytest.approx(40.0 + tie)
    assert opt.hourly_cost(freelancer, "premium", 27.7) == pytest.approx(12.3 + tie)
    cheap = make_state("S003", freelance=True, rate=20.0).scout
    assert opt.hourly_cost(cheap, "premium", 27.7) == pytest.approx(tie)  # never below 0


def test_salaried_first_even_when_the_premium_is_zero() -> None:
    """A freelancer as cheap as a salaried hour: the tie-break still picks salaried."""
    pool = [make_item("T00001-desk")]
    freelancer = make_state("S001", freelance=True, rate=20.0)  # below 27.7: premium 0
    salaried = make_state("S002", weekday_hours=1.0)
    premium = CFG.model_copy(update={"cost_basis": "premium"})
    assert run("optimiser", pool, [freelancer, salaried], premium) == {"T00001-desk": "S002"}


def test_commit_horizon_limits_how_far_ahead_work_is_committed() -> None:
    """Four 6 h items due in 10 days, one scout with 7.5 h a day (daily cadence):
    commit what can be finished by tomorrow + buffer days, keep the rest pooled."""
    pool = [make_item(f"T0000{k}-desk", hours=6.0) for k in (1, 2, 3, 4)]
    scouts = [make_state("S001")]
    by_buffer = {
        b: len(run("optimiser", pool, scouts, CFG.model_copy(update={"commit_buffer_days": b})))
        for b in (0, 1, 2, None)
    }
    assert by_buffer == {0: 1, 1: 2, 2: 3, None: 4}  # 7.5 h, 15 h, 22.5 h, 37.5 h


def test_commit_horizon_follows_cadence_and_spares_live_views() -> None:
    assert opt.commit_index(WINDOW, CFG.model_copy(update={"commit_buffer_days": 2})) == 2
    weekly = CFG.model_copy(update={"cadence": "weekly", "commit_buffer_days": 0})
    assert opt.commit_index(WINDOW, weekly) == 6  # next run in 7 days: whole window
    assert opt.commit_index(WINDOW, CFG) is None
    # A live view later in the window is still taken: it happens on its date.
    pool = [make_item("T00001-live", live_on=day(5))]
    tight = CFG.model_copy(update={"commit_buffer_days": 0})
    assert run("optimiser", pool, [make_state("S001")], tight) == {"T00001-live": "S001"}


def test_queue_wait_days_counts_earlier_due_work_on_the_salaried_team() -> None:
    salaried = make_state("S001")  # 37.5 h / 7 days = 5.36 h per calendar day
    pool = [
        make_item("T00001-desk", hours=10.0, due=day(3)),
        make_item("T00002-desk", hours=6.0, due=day(5)),
        make_item("T00003-desk", hours=4.0, due=day(5)),  # same due date: ties count
    ]
    per_day = 37.5 / 7
    waits = opt.queue_wait_days(pool, [salaried], WINDOW)
    assert waits["R00001"] == pytest.approx(0.0)
    assert waits["R00002"] == pytest.approx((10 + 4) / per_day)
    assert waits["R00003"] == pytest.approx((10 + 6) / per_day)
    # Frozen work is ahead of everything; a second skill halves the capacity.
    busy = make_state("S001", ("MID-North", "FWD-North"), frozen=5.0)
    waits = opt.queue_wait_days(pool[:1], [busy], WINDOW)
    assert waits["R00001"] == pytest.approx(2.5 / (per_day / 2))


def test_queue_wait_days_ignores_freelancers_unless_no_salaried_scout_has_the_skill() -> None:
    pool = [make_item("T00001-desk", due=day(3)), make_item("T00002-desk", due=day(5))]
    freelancer = make_state("S002", freelance=True, weekday_hours=7.5)
    with_salaried = opt.queue_wait_days(pool, [make_state("S001"), freelancer], WINDOW)
    assert with_salaried["R00002"] == pytest.approx(4.0 / (37.5 / 7))  # salaried capacity only
    alone = opt.queue_wait_days(pool, [freelancer], WINDOW)
    assert alone["R00002"] == pytest.approx(4.0 / (37.5 / 7))  # falls back to all scouts
    assert opt.queue_wait_days(pool, [], WINDOW) == {}


def test_load_aware_urgency_raises_the_stake_behind_a_long_queue() -> None:
    item = make_item("T00001-desk", hours=6.0, due=day(12))  # own slack 12
    plain = opt.lateness_penalties([item], D0, day(1), 1000.0, 1.0)
    queued = opt.lateness_penalties([item], D0, day(1), 1000.0, 1.0, {"R00001": 10.0})
    assert plain["T00001-desk"] == pytest.approx(1000 / 13)
    assert queued["T00001-desk"] == pytest.approx(1000 / 3)  # slack 12 - 10 = 2


def test_load_aware_optimiser_calls_in_a_freelancer_when_the_salaried_queue_is_long() -> None:
    """The only salaried scout is booked (33 h frozen); a relaxed item could wait for them.

    Plain urgency (slack 14, ~67 at stake) is below the freelancer's premium
    (6 h x 12.3 = ~74): wait. Counting the ~6.2 days of salaried work ahead,
    the stake (~113) beats the premium: the idle freelancer takes it now.
    """
    pool = [make_item("T00001-desk", hours=6.0, due=day(14))]
    scouts = [make_state("S001", frozen=33.0), make_state("S002", freelance=True)]
    plain = CFG.model_copy(update={"cost_basis": "premium"})
    assert run("optimiser", pool, scouts, plain) == {}
    aware = plain.model_copy(update={"load_aware": True})
    assert run("optimiser", pool, scouts, aware) == {"T00001-desk": "S002"}


def test_perishable_live_view_is_taken_even_with_lots_of_slack() -> None:
    """Fixture today, report due in 12 days, only a freelancer in the region.

    By slack alone the request is relaxed (~77 at stake vs 8 h x 40 = 320),
    but if the live view is not assigned now the match is gone.
    """
    pool = [make_item("T00001-live", live_on=D0, due=day(12))]
    freelancer = [make_state("S001", freelance=True, rate=40.0)]
    assert run("optimiser", pool, freelancer) == {"T00001-live": "S001"}


def test_perishability_follows_the_cadence() -> None:
    """A fixture in 3 days survives until tomorrow's run, but not until next week's."""
    pool = [make_item("T00001-live", live_on=day(3), due=day(12))]
    freelancer = [make_state("S001", freelance=True, rate=40.0)]
    assert run("optimiser", pool, freelancer) == {}
    weekly = CFG.model_copy(update={"cadence": "weekly"})
    assert run("optimiser", pool, freelancer, weekly) == {"T00001-live": "S001"}


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
    assert run("optimiser", pool, scouts, weights(churn=500.0)) == {"T00001-desk": "S001"}


def test_dropping_a_committed_item_also_pays_churn() -> None:
    """Relaxed item on a freelancer: keeping costs 160, dropping saves that but
    leaves ~91 of lateness. Churn on a drop is what keeps the plan stable."""
    pool = [make_item("T00001-desk", hours=4.0, current="S001")]
    freelancer = [make_state("S001", freelance=True)]
    assert run("optimiser", pool, freelancer, weights(churn=0.0)) == {}
    assert run("optimiser", pool, freelancer, weights(churn=100.0)) == {"T00001-desk": "S001"}


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


# --- lateness pricing ---------------------------------------------------------------------


def test_slack_counts_today_as_a_working_day() -> None:
    assert opt.slack_days(D0, D0, 1) == 0  # due today, one day of work: just in time
    assert opt.slack_days(day(13), D0, 2) == 12


def test_urgency_grows_as_slack_shrinks_and_overdue_is_highest() -> None:
    values = [opt.urgency(s, overdue=False) for s in (13, 6, 2, 1, 0)]
    assert values == sorted(values) and values[-1] == 1.0
    assert opt.urgency(-3, overdue=False) == 1.0
    assert opt.urgency(0, overdue=True) == opt.URGENCY_OVERDUE > 1.0


def test_remaining_work_days_respects_a_late_fixture() -> None:
    desk = make_item("T00001-desk", hours=4.0)
    live = make_item("T00001-live", live_on=day(5))
    writeup = make_item("T00001-writeup", hours=3.0, depends_on=("T00001-live",))
    assert opt.remaining_work_days([desk, writeup], D0) == 1  # 7 h of work
    assert opt.remaining_work_days([desk, live, writeup], D0) == 7  # fixture day 6 + write-up


def test_one_request_is_worth_one_late_penalty_however_many_items() -> None:
    """Penalties are per request, shared by hours (zero slack: 1000 in total)."""
    two_tasks = [
        make_item("T00001-desk", hours=6.0, due=D0),
        make_item("T00001-writeup", hours=3.0, due=D0, depends_on=("T00001-desk",)),
    ]
    pen = opt.lateness_penalties(two_tasks, D0, day(1), 1000.0, 1.0)
    assert pen == pytest.approx({"T00001-desk": 1000 * 6 / 9, "T00001-writeup": 1000 * 3 / 9})
    three_tasks = [
        make_item("T00002-desk", hours=6.0, due=day(1)),
        make_item("T00002-live", live_on=day(1), due=day(1)),
        make_item("T00002-writeup", hours=3.0, due=day(1), depends_on=("T00002-live",)),
    ]
    pen = opt.lateness_penalties(three_tasks, D0, day(1), 1000.0, 1.0)
    assert sum(pen.values()) == pytest.approx(1000.0)


def test_perishable_live_view_carries_a_full_penalty_of_its_own() -> None:
    items = [make_item("T00001-desk", due=day(12)), make_item("T00001-live", live_on=D0)]
    pen = opt.lateness_penalties(items, D0, day(1), 1000.0, 1.0)
    assert pen["T00001-live"] == pytest.approx(1000.0 * opt.URGENCY_LOST_FIXTURE)
    assert pen["T00001-desk"] < 100


def test_freelancer_without_a_rate_fails_loudly() -> None:
    state = make_state("S001", freelance=True)
    broken = ScoutState(
        Scout("S001", "Scout S001", "freelance", ("MID-North",), "North", 8.0, 25.0),
        state.hours_by_day,
    )
    with pytest.raises(ValueError, match="hourly_rate"):
        run("optimiser", [make_item("T00001-desk")], [broken])


# --- determinism, randomness, time limit, fallback ----------------------------------------


def random_instance(
    n_items: int, n_scouts: int, seed: int
) -> tuple[list[WorkItem], list[ScoutState]]:
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
            tuple(str(s) for s in r.choice(skills, size=int(r.integers(1, 5)), replace=False)),
            region=regions[k % 4],
            freelance=freelance,
            rate=41.5,
            former=tuple(str(c) for c in r.choice(clubs, size=2, replace=False)),
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
                clubs=tuple(str(c) for c in r.choice(clubs, size=2, replace=False)),
                current=f"S{int(r.integers(n_scouts)):03d}" if r.random() < 0.15 else None,
                depends_on=("T99999-desk",) if r.random() < 0.1 else (),
                request_scouts=(f"S{int(r.integers(n_scouts)):03d}",) if r.random() < 0.2 else (),
            )
        )
    return pool, scouts


def test_optimiser_is_deterministic() -> None:
    pool, scouts = random_instance(200, 40, seed=7)
    cfg = CFG.model_copy(update={"time_limit_s": 0.3})
    first = opt.solve_assignment(pool, scouts, WINDOW, cfg, np.random.default_rng(11), cost=COST)
    second = opt.solve_assignment(pool, scouts, WINDOW, cfg, np.random.default_rng(11), cost=COST)
    assert first.assignments == second.assignments
    assert first.objective == second.objective


def rng_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def one_draw_later(seed: int) -> dict:
    fresh = np.random.default_rng(seed)
    fresh.integers(0, 2**31 - 1)
    return rng_state(fresh)


def test_optimiser_draws_exactly_one_number_on_every_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The assignment stream advances the same way whatever the day holds (D-015)."""
    pool, scouts = tight_deadline_case()
    cases = {
        "empty pool": ([], scouts),
        "nothing eligible": ([make_item("T00001-desk", skill="GK-South")], scouts),
        "solved": (pool, scouts),
    }
    for label, (p, s) in cases.items():
        rng = np.random.default_rng(5)
        opt.optimiser(p, s, WINDOW, CFG, rng, cost=COST)
        assert rng_state(rng) == one_draw_later(5), label

    monkeypatch.setattr(opt, "wall_clock_cap", lambda limit: 0.0)
    pool, scouts = random_instance(120, 25, seed=1)  # too big to be solved in presolve
    rng = np.random.default_rng(5)
    assert opt.solve_assignment(pool, scouts, WINDOW, CFG, rng, cost=COST).fell_back_to_edf
    assert rng_state(rng) == one_draw_later(5), "fallback"


@pytest.mark.parametrize("name", GREEDY)
def test_greedy_policies_draw_no_random_numbers(name: str) -> None:
    pool, scouts = tight_deadline_case()
    rng = np.random.default_rng(5)
    get_policy(name)(pool, scouts, WINDOW, CFG, rng, cost=COST)
    assert rng_state(rng) == rng_state(np.random.default_rng(5))


def test_optimiser_respects_its_time_limit() -> None:
    """Checked on the deterministic clock; wall time only against the generous backstop."""
    pool, scouts = random_instance(400, 60, seed=3)
    cfg = DEFAULTS.model_copy(update={"time_limit_s": 0.1})  # the smallest allowed
    report = opt.solve_assignment(pool, scouts, WINDOW, cfg, np.random.default_rng(0), cost=COST)
    assert report.status in ("OPTIMAL", "FEASIBLE") and not report.fell_back_to_edf
    assert not report.hit_wall_clock
    assert report.deterministic_time <= cfg.time_limit_s * 1.05
    assert report.wall_time_s <= opt.wall_clock_cap(cfg.time_limit_s)
    assert report.best_bound is not None and report.gap is not None and report.gap >= 0
    problems = validate_assignments(pool, scouts, WINDOW, report.assignments, check_due_dates=True)
    assert problems == []


def test_wall_clock_backstop_is_far_above_the_limit() -> None:
    assert opt.wall_clock_cap(0.1) == pytest.approx(5.1)
    assert opt.wall_clock_cap(1.0) == pytest.approx(10.0)


@pytest.mark.slow
def test_benchmark_400_items_60_scouts() -> None:
    """Default limit (1 s): model build + solve, with room for a loaded machine."""
    pool, scouts = random_instance(400, 60, seed=1)
    cfg = DEFAULTS.model_copy(update={"time_limit_s": 1.0})
    started = time.perf_counter()
    report = opt.solve_assignment(pool, scouts, WINDOW, cfg, np.random.default_rng(0), cost=COST)
    elapsed = time.perf_counter() - started
    assert report.status in ("OPTIMAL", "FEASIBLE") and not report.hit_wall_clock
    assert report.deterministic_time <= cfg.time_limit_s * 1.05
    assert elapsed <= 5 * cfg.time_limit_s, f"{elapsed:.2f} s"
    problems = validate_assignments(pool, scouts, WINDOW, report.assignments, check_due_dates=True)
    assert problems == []


def test_optimiser_falls_back_to_edf_when_the_solver_finds_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A zero wall-clock cap stops the solver before its first solution (on an
    # instance big enough not to be solved outright by presolve).
    monkeypatch.setattr(opt, "wall_clock_cap", lambda limit: 0.0)
    pool, scouts = random_instance(120, 25, seed=1)
    with caplog.at_level(logging.WARNING, logger=opt.__name__):
        report = opt.solve_assignment(
            pool, scouts, WINDOW, CFG, np.random.default_rng(0), cost=COST
        )
    assert report.fell_back_to_edf and report.status == "UNKNOWN" and report.hit_wall_clock
    assert report.assignments == answer_of("edf", pool, scouts)
    assert "falling back to EDF" in caplog.text


def test_an_infeasible_model_is_a_bug_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assigning nothing is always feasible, so INFEASIBLE can only mean a broken model."""
    monkeypatch.setattr(opt, "free_by_day", lambda state, window: [-1] * window.n_days)
    monkeypatch.setattr(opt, "deadline_index", lambda item, free, window, **kw: 0)
    pool, scouts = tight_deadline_case()
    with pytest.raises(RuntimeError, match="INFEASIBLE"):
        opt.solve_assignment(pool, scouts, WINDOW, CFG, np.random.default_rng(0), cost=COST)
