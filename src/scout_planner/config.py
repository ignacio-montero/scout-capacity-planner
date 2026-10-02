"""Typed simulator parameters: the single source of truth for every tunable input.

Each class mirrors one group in ``docs/PARAMETERS.md``; ``config/default.yaml``
holds the same defaults (a test pins the two together). Validation is strict on
purpose so that a bad value fails at load time with a precise message, never
halfway through a simulated year:

* ``extra="forbid"`` everywhere: an unknown or misspelled key is an error.
* ``strict=True``: no silent type coercion (``"0.7"`` is not a float, ``1`` is
  not ``True``, ``24.0`` is not an int). An int is still accepted for a float.
* ``frozen=True``: a parameter set is an immutable value. Changing it means
  building a new one (see :func:`apply_overrides`), which re-runs validation.

Pure helpers (:func:`params_from_yaml`, :func:`params_to_yaml`) do the parsing;
the thin file wrappers (:func:`load_params`, :func:`dump_params`) only add I/O.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

# Fixed in code on purpose (see PARAMETERS.md, "Not parameters").
FULL_TIME_WEEKLY_HOURS = 37.5
WEEKS_PER_YEAR = 52

# Repo-root default file. The project is installed in editable mode by `uv sync`,
# so this path resolves to the checked-out repo.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _ordered_pair(pair: tuple[float, float]) -> tuple[float, float]:
    """Reject a ``[min, max]`` pair whose min exceeds its max."""
    low, high = pair
    if low > high:
        raise ValueError(f"[min, max] pair needs min <= max, got [{low}, {high}]")
    return pair


# `strict=False` on the tuple itself lets YAML lists become tuples; the element
# types stay strict, so ["8", 25] is still rejected.
WeeklyHours = Annotated[float, Field(ge=0, le=40)]
TaskHours = Annotated[float, Field(gt=0, le=40)]
SkillCount = Annotated[int, Field(ge=1, le=6)]
WeeklyHoursRange = Annotated[
    tuple[WeeklyHours, WeeklyHours], Field(strict=False), AfterValidator(_ordered_pair)
]
TaskHoursRange = Annotated[
    tuple[TaskHours, TaskHours], Field(strict=False), AfterValidator(_ordered_pair)
]
SkillCountRange = Annotated[
    tuple[SkillCount, SkillCount], Field(strict=False), AfterValidator(_ordered_pair)
]


class _StrictModel(BaseModel):
    """Base for every parameter group: immutable, no unknown keys, no coercion."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TeamParams(_StrictModel):
    """Initial team: who is on the books when the simulated year starts."""

    full_time_count: int = Field(24, ge=0, le=100)
    freelance_count: int = Field(16, ge=0, le=100)
    freelance_weekly_hours: WeeklyHoursRange = (8.0, 25.0)
    skills_per_scout: SkillCountRange = (1, 4)
    leave_days_per_year: int = Field(25, ge=0, le=60)
    follow_hiring_plan: bool = True

    @model_validator(mode="after")
    def _at_least_one_scout(self) -> TeamParams:
        # Base demand is calibrated as a share of the initial team's hours (D-003);
        # an empty team would make that calibration meaningless.
        if self.full_time_count + self.freelance_count == 0:
            raise ValueError("the initial team needs at least one scout")
        return self


class DemandParams(_StrictModel):
    """What actually arrives during the simulated year (D-014: actual growth)."""

    start_load: float = Field(0.70, ge=0.3, le=1.2)
    actual_growth: float = Field(4.0, ge=0.5, le=6.0)
    live_view_share: float = Field(0.40, ge=0.0, le=1.0)
    seasonality_strength: float = Field(1.0, ge=0.0, le=2.0)
    history_months: int = Field(48, ge=36, le=72)
    turnaround_days: int = Field(14, ge=7, le=28)
    desk_hours: TaskHoursRange = (4.0, 10.0)
    writeup_hours: TaskHoursRange = (2.0, 4.0)
    live_view_hours: float = Field(8.0, gt=0, le=24)


class CapacityPlanParams(_StrictModel):
    """What the agency believes is coming and hires for (D-014: assumed growth)."""

    assumed_growth: float = Field(4.0, ge=0.5, le=6.0)
    quantile: float = Field(0.80, ge=0.5, le=0.95)
    target_utilisation: float = Field(0.80, ge=0.5, le=1.0)
    lead_time_full_time_months: int = Field(3, ge=0, le=6)
    lead_time_freelance_months: int = Field(1, ge=0, le=3)
    persistent_gap_months: int = Field(3, ge=1, le=12)


class AutomationParams(_StrictModel):
    """The video pre-screen tool that shortens desk reviews when it works."""

    enabled: bool = False
    desk_reduction: float = Field(0.40, ge=0.0, le=0.9)
    rework_rate: float = Field(0.15, ge=0.0, le=0.6)
    rework_overhead_hours: float = Field(1.0, ge=0.0, le=4.0)
    monthly_cost: float = Field(1500.0, ge=0.0)


class AssignmentWeights(_StrictModel):
    """Optimiser objective weights (only the CP-SAT policy reads these)."""

    lateness: float = Field(100.0, ge=0.0)
    cost: float = Field(1.0, ge=0.0)
    continuity: float = Field(5.0, ge=0.0)
    churn: float = Field(3.0, ge=0.0)


class AssignmentParams(_StrictModel):
    """How the daily (or weekly) assignment run decides who does which work item."""

    policy: Literal["fcfs", "edf", "optimiser"] = "optimiser"
    cadence: Literal["daily", "weekly"] = "daily"
    unit: Literal["task", "bundle"] = "task"
    horizon_days: int = Field(7, ge=1, le=14)
    weights: AssignmentWeights = AssignmentWeights()
    time_limit_s: float = Field(1.0, ge=0.1, le=10.0)


class CostParams(_StrictModel):
    """Money: salaries, the freelance premium and the price of a late report."""

    full_time_monthly_salary: float = Field(4500.0, gt=0)
    freelance_premium: float = Field(1.5, ge=1.0, le=3.0)
    late_penalty: float = Field(1000.0, ge=0.0)

    @property
    def salaried_hourly_equivalent(self) -> float:
        """Monthly salary spread over a full-time year of 37.5 h weeks (~27.7 by default)."""
        return self.full_time_monthly_salary * 12 / (FULL_TIME_WEEKLY_HOURS * WEEKS_PER_YEAR)

    @property
    def freelance_hourly_rate(self) -> float:
        """Freelance rate = salaried hourly equivalent x premium (~41.5 by default)."""
        return self.salaried_hourly_equivalent * self.freelance_premium


class SimParams(_StrictModel):
    """Simulation horizon, replications and the service target."""

    seeds: int = Field(3, ge=1, le=5)
    seed: int = Field(42, ge=0)
    months: int = Field(12, ge=1, le=24)
    target_on_time: float = Field(0.95, ge=0.5, le=1.0)


class Params(_StrictModel):
    """A complete, validated parameter set. One run = one ``Params`` (D-012)."""

    team: TeamParams = TeamParams()
    demand: DemandParams = DemandParams()
    capacity_plan: CapacityPlanParams = CapacityPlanParams()
    automation: AutomationParams = AutomationParams()
    assignment: AssignmentParams = AssignmentParams()
    cost: CostParams = CostParams()
    sim: SimParams = SimParams()


# --- parsing and serialisation (pure) --------------------------------------------


def params_from_yaml(text: str) -> Params:
    """Parse and validate a YAML document. Missing keys take their defaults.

    Raises ``pydantic.ValidationError`` (a ``ValueError``) on bad values or
    unknown keys, and ``ValueError`` if the document is not a mapping.
    """
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ValueError(f"parameter YAML must be a mapping, got {type(data).__name__}")
    return Params.model_validate(data)


def params_to_yaml(params: Params) -> str:
    """Serialise to YAML that :func:`params_from_yaml` reads back into an equal object."""
    # mode="json" turns tuples into plain lists, so the YAML has no Python tags.
    return yaml.safe_dump(params.model_dump(mode="json"), sort_keys=False)


def apply_overrides(params: Params, overrides: Mapping[str, Any]) -> Params:
    """Return a new ``Params`` with dotted-key overrides applied and re-validated.

    Example: ``apply_overrides(p, {"assignment.policy": "edf", "sim.seeds": 5})``.
    Used by sweeps and the app. Every key must already exist (a typo raises
    ``ValueError`` naming the key); the result goes through full validation.
    """
    data = params.model_dump(mode="python")
    for dotted_key, value in overrides.items():
        *parents, leaf = dotted_key.split(".")
        node: Any = data
        for part in parents:
            if not isinstance(node, dict) or part not in node:
                raise ValueError(f"unknown parameter {dotted_key!r}")
            node = node[part]
        if not isinstance(node, dict) or leaf not in node:
            raise ValueError(f"unknown parameter {dotted_key!r}")
        # deepcopy so a caller's list/dict is never shared with our data.
        node[leaf] = copy.deepcopy(value)
    return Params.model_validate(data)


# --- file I/O wrappers -----------------------------------------------------------


def load_params(path: str | Path) -> Params:
    """Load and validate a parameter YAML file."""
    return params_from_yaml(Path(path).read_text(encoding="utf-8"))


def load_default_params() -> Params:
    """Load ``config/default.yaml``."""
    return load_params(DEFAULT_CONFIG_PATH)


def dump_params(params: Params, path: str | Path) -> None:
    """Write a parameter set as YAML (round-trips through :func:`load_params`)."""
    Path(path).write_text(params_to_yaml(params), encoding="utf-8")
