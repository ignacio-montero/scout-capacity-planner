"""Assignment policies (FCFS, EDF, CP-SAT optimiser) behind a common interface.

This is the *strategy pattern*: every policy is a function with the same
signature (:class:`Policy`, spec in ``docs/DATA_CONTRACTS.md`` section 4), and
callers look one up by name instead of importing it. The simulation only ever
calls :func:`assign` / :func:`assign_with_report` (or :func:`get_policy`), so
adding a fourth policy means adding one entry to :data:`POLICIES`.

All policies share the hard constraints in :mod:`.eligibility`;
:func:`~.eligibility.validate_assignments` checks any policy's answer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np

from scout_planner.assign.eligibility import validate_assignments
from scout_planner.assign.greedy import edf, fcfs
from scout_planner.assign.optimiser import SolveReport, solve_assignment
from scout_planner.assign.optimiser import optimiser as cp_sat_optimiser
from scout_planner.config import AssignmentParams, CostParams
from scout_planner.domain import Assignment, AssignmentWindow, ScoutState, WorkItem


class Policy(Protocol):
    """One assignment run: decide which scout takes which work item.

    * ``pool``: open items (unassigned + assigned-but-not-started). Started
      work is not here; it is each scout's ``frozen_hours``.
    * ``scouts``: each scout's offered hours per window day, frozen work, leave.
    * ``window``: today .. today + horizon - 1.
    * ``rng``: the run's assignment stream (only the optimiser draws from it,
      exactly one number per call).
    * ``cost``: the run's money parameters (the optimiser prices lateness with
      ``cost.late_penalty``). Keyword-only and required, so a caller cannot
      silently run with default economics.

    Returns at most one ``Assignment`` per item; items without one stay open.
    Must be a pure function: no I/O, no mutation of its inputs, and the answer
    must not depend on the order of ``pool`` or ``scouts``.
    """

    def __call__(
        self,
        pool: list[WorkItem],
        scouts: list[ScoutState],
        window: AssignmentWindow,
        cfg: AssignmentParams,
        rng: np.random.Generator,
        *,
        cost: CostParams,
    ) -> list[Assignment]: ...


# The CP-SAT policy is imported under an alias on purpose: importing the function
# as ``optimiser`` would shadow the submodule ``scout_planner.assign.optimiser`` on
# this package, so ``from scout_planner.assign import optimiser`` would return the
# function instead of the module.
POLICIES: Mapping[str, Policy] = MappingProxyType(
    {"fcfs": fcfs, "edf": edf, "optimiser": cp_sat_optimiser}
)

# Policies that can also say *how* they decided (solver status, time, fallback).
_REPORTING: Mapping[str, Callable[..., SolveReport]] = MappingProxyType(
    {"optimiser": solve_assignment}
)


def get_policy(name: str) -> Policy:
    """Look a policy up by its ``assignment.policy`` name."""
    try:
        return POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown assignment policy {name!r}; known: {sorted(POLICIES)}") from None


def assign(
    pool: list[WorkItem],
    scouts: list[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
    *,
    cost: CostParams,
) -> list[Assignment]:
    """Run the policy named by ``cfg.policy``: the one call the simulation makes."""
    return get_policy(cfg.policy)(pool, scouts, window, cfg, rng, cost=cost)


@dataclass(frozen=True, slots=True)
class AssignOutcome:
    """An assignment run's answer plus, for the optimiser, how the solve went.

    ``solve`` is ``None`` for the greedy policies (nothing to report). A
    simulation can count ``solve.fell_back_to_edf`` and ``solve.hit_wall_clock``
    per run and keep ``solve.gap`` / ``solve.wall_time_s`` for diagnostics.
    """

    assignments: list[Assignment]
    solve: SolveReport | None = None


def assign_with_report(
    pool: list[WorkItem],
    scouts: list[ScoutState],
    window: AssignmentWindow,
    cfg: AssignmentParams,
    rng: np.random.Generator,
    *,
    cost: CostParams,
) -> AssignOutcome:
    """Like :func:`assign`, also returning the solver report when there is one."""
    reporting = _REPORTING.get(cfg.policy)
    if reporting is None:
        return AssignOutcome(assign(pool, scouts, window, cfg, rng, cost=cost))
    report = reporting(pool, scouts, window, cfg, rng, cost=cost)
    return AssignOutcome(report.assignments, report)


__all__ = [
    "POLICIES",
    "AssignOutcome",
    "Policy",
    "SolveReport",
    "assign",
    "assign_with_report",
    "get_policy",
    "validate_assignments",
]
