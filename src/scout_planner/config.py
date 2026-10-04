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
* Duplicate YAML keys are an error (plain YAML silently keeps the last one).
* ``schema_version`` + :func:`migrate`: a saved ``params.yaml`` records the
  layout it was written in, so old run folders stay loadable after a rename.

Pure helpers (:func:`params_from_yaml`, :func:`params_to_yaml`) do the parsing;
the thin file wrappers (:func:`load_params`, :func:`dump_params`) only add I/O.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

# Fixed in code on purpose (see PARAMETERS.md, "Not parameters").
FULL_TIME_WEEKLY_HOURS = 37.5
WEEKS_PER_YEAR = 52

# Layout version of the parameter file. Bump it when a key is renamed, moved or
# removed, and register a step in MIGRATIONS that upgrades the old layout.
SCHEMA_VERSION = 1

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
    # Organic growth of the history before the simulated year (D-004); the
    # simulated year itself follows ``actual_growth`` instead (D-014).
    history_growth_per_year: float = Field(0.15, ge=-0.5, le=1.0)
    turnaround_days: int = Field(14, ge=7, le=28)
    # Urgent requests get a shorter promise; mixed deadlines are what make
    # earliest-due-first differ from first-come-first-served.
    urgent_share: float = Field(0.15, ge=0.0, le=1.0)
    urgent_turnaround_days: int = Field(7, ge=3, le=28)
    desk_hours: TaskHoursRange = (4.0, 10.0)
    writeup_hours: TaskHoursRange = (2.0, 4.0)
    live_view_hours: float = Field(8.0, gt=0, le=24)

    @model_validator(mode="after")
    def _urgent_is_not_slower(self) -> DemandParams:
        if self.urgent_turnaround_days > self.turnaround_days:
            raise ValueError(
                f"urgent_turnaround_days ({self.urgent_turnaround_days}) must be <= "
                f"turnaround_days ({self.turnaround_days})"
            )
        return self


class CapacityPlanParams(_StrictModel):
    """What the agency believes is coming and hires for (D-014: assumed growth)."""

    assumed_growth: float = Field(4.0, ge=0.5, le=6.0)
    quantile: float = Field(0.80, ge=0.5, le=0.95)
    target_utilisation: float = Field(0.80, ge=0.5, le=1.0)
    lead_time_full_time_months: int = Field(3, ge=0, le=6)
    lead_time_freelance_months: int = Field(1, ge=0, le=3)
    persistent_gap_months: int = Field(3, ge=1, le=12)
    # Which hire types the plan may use: ``rule`` = full-time for persistent
    # gaps + freelance for peaks (plan.greedy_hiring); the other two cover every
    # gap with one type only (what-if comparisons for the headline sweep).
    hire_mix: Literal["rule", "freelance_only", "full_time_only"] = "rule"


class AutomationParams(_StrictModel):
    """The video pre-screen tool that shortens desk reviews when it works."""

    enabled: bool = False
    desk_reduction: float = Field(0.40, ge=0.0, le=0.9)
    rework_rate: float = Field(0.15, ge=0.0, le=0.6)
    rework_overhead_hours: float = Field(1.0, ge=0.0, le=4.0)
    monthly_cost: float = Field(1500.0, ge=0.0)


class AssignmentWeights(_StrictModel):
    """Optimiser objective weights (only the CP-SAT policy reads these)."""

    # Multiplier on ``cost.late_penalty`` (S4): leaving urgent work unassigned
    # costs late_penalty x lateness x urgency, so it stays in cost units.
    lateness: float = Field(1.0, gt=0.0, le=10.0)
    cost: float = Field(1.0, ge=0.0)
    continuity: float = Field(5.0, ge=0.0)
    churn: float = Field(3.0, ge=0.0)


class AssignmentParams(_StrictModel):
    """How the daily (or weekly) assignment run decides who does which work item."""

    policy: Literal["fcfs", "edf", "edf_feasible", "optimiser"] = "edf_feasible"
    cadence: Literal["daily", "weekly"] = "daily"
    unit: Literal["task", "bundle"] = "task"
    horizon_days: int = Field(7, ge=1, le=14)
    weights: AssignmentWeights = AssignmentWeights()
    time_limit_s: float = Field(1.0, ge=0.1, le=10.0)
    # Optimiser only (advanced). Defaults chosen by the M3 calibration sweep
    # (config/sweeps/optimiser_eval*.yaml); the old behaviour is full / None / False.
    # What a freelance hour costs in the objective: "full" = the hourly rate;
    # "premium" = rate - salaried hourly equivalent (the work must be done by
    # someone; deferring it only saves the premium).
    cost_basis: Literal["full", "premium"] = "premium"
    # Commit only non-live work a scout can finish within this many days after
    # the next assignment run; the rest stays in the pool for the next run.
    # None = commit anything that fits the whole window.
    commit_buffer_days: int | None = Field(2, ge=0, le=14)
    # Urgency uses slack minus the wait behind earlier-due work of the same
    # skill on the salaried team (so deferral looks expensive when queues are long).
    load_aware: bool = True

    @model_validator(mode="after")
    def _weekly_needs_a_week_of_horizon(self) -> AssignmentParams:
        # With weekly runs, a fixture more than horizon_days after a Monday run is
        # outside that run's window and has passed before the next one: live views
        # on those days could never be assigned (critic S5).
        if self.cadence == "weekly" and self.horizon_days < 7:
            raise ValueError(
                f"assignment.cadence 'weekly' needs assignment.horizon_days >= 7 "
                f"(got {self.horizon_days}): fixtures between runs would be unreachable"
            )
        return self


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

    schema_version: int = SCHEMA_VERSION
    team: TeamParams = TeamParams()
    demand: DemandParams = DemandParams()
    capacity_plan: CapacityPlanParams = CapacityPlanParams()
    automation: AutomationParams = AutomationParams()
    assignment: AssignmentParams = AssignmentParams()
    cost: CostParams = CostParams()
    sim: SimParams = SimParams()

    @field_validator("schema_version")
    @classmethod
    def _current_schema(cls, version: int) -> int:
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {version} is not the current layout ({SCHEMA_VERSION}); "
                "load the file through params_from_yaml / load_params, which migrate it"
            )
        return version

    @model_validator(mode="after")
    def _items_fit_in_a_window(self) -> Params:
        # A work item is never split across scouts, so one item must fit in a
        # single full-timer's hours over the assignment window. Otherwise no
        # policy can ever assign it and it silently sits in the pool forever.
        horizon = self.assignment.horizon_days
        cap = FULL_TIME_WEEKLY_HOURS * horizon / 7
        d = self.demand
        largest = {
            "demand.desk_hours (max)": d.desk_hours[1],
            "demand.writeup_hours (max)": d.writeup_hours[1],
            "demand.live_view_hours": d.live_view_hours,
        }
        if self.assignment.unit == "bundle":
            largest["demand.desk_hours (max) + demand.writeup_hours (max), unit=bundle"] = (
                d.desk_hours[1] + d.writeup_hours[1]
            )
        too_big = [f"{key} = {hours:g}" for key, hours in largest.items() if hours > cap]
        if too_big:
            raise ValueError(
                f"work items must fit in one full-time scout's hours over the window "
                f"({FULL_TIME_WEEKLY_HOURS:g} h/week x assignment.horizon_days={horizon} / 7 "
                f"= {cap:.2f} h); too large: " + "; ".join(too_big)
            )
        return self


# --- schema versioning ------------------------------------------------------------

# MIGRATIONS[v] upgrades a raw v-layout dict to layout v + 1. Empty while
# SCHEMA_VERSION is 1; e.g. a future rename would register
# ``MIGRATIONS[1] = lambda raw: rename(raw, "old.key", "new.key")``.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def migrate(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a raw parameter dict to the current ``SCHEMA_VERSION`` (a copy).

    A file without ``schema_version`` predates versioning and is read as v1.
    No-op for v1 today; this is the hook that keeps old run folders loadable
    after the parameter layout changes.
    """
    data = copy.deepcopy(dict(raw))
    version = data.get("schema_version", 1)
    if type(version) is not int:
        raise ValueError(f"schema_version must be an integer, got {version!r}")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {version} is newer than this code understands ({SCHEMA_VERSION})"
        )
    while version < SCHEMA_VERSION:
        data = MIGRATIONS[version](data)
        version += 1
    data["schema_version"] = version
    return data


# --- parsing and serialisation (pure) --------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys instead of keeping the last one.

    Plain YAML loaders accept ``seeds: 3`` followed later by ``seeds: 5`` and
    silently use 5: a hand-edited file could then run with a value nobody
    meant. The ``<<`` merge key is left to YAML's normal handling.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def params_from_yaml(text: str) -> Params:
    """Parse, migrate and validate a YAML document. Missing keys take their defaults.

    Every failure is a ``ValueError``: malformed YAML or duplicate keys,
    a non-mapping document, an unsupported ``schema_version``, and bad values
    or unknown keys (``pydantic.ValidationError`` subclasses ``ValueError``).
    """
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)  # a SafeLoader: builds no Python objects
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid parameter YAML: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ValueError(f"parameter YAML must be a mapping, got {type(data).__name__}")
    return Params.model_validate(migrate(data))


def params_to_yaml(params: Params) -> str:
    """Serialise to YAML that :func:`params_from_yaml` reads back into an equal object."""
    # mode="json" turns tuples into plain lists, so the YAML has no Python tags.
    return yaml.safe_dump(params.model_dump(mode="json"), sort_keys=False)


def _to_builtin(value: Any) -> Any:
    """Turn numpy scalars/arrays (and tuples) into plain Python, recursively.

    Sweep values often come out of pandas/numpy (``np.int64(3)``,
    ``np.float64(0.8)``, ``np.bool_(True)``). Strict validation would reject
    them as "not an int/float/bool", so they are converted at the boundary.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list | tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _to_builtin(v) for k, v in value.items()}
    return value


def _deep_merge(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    """Merge ``updates`` into ``target`` in place, descending into nested dicts."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def apply_overrides(params: Params, overrides: Mapping[str, Any]) -> Params:
    """Return a new ``Params`` with dotted-key overrides applied and re-validated.

    Example: ``apply_overrides(p, {"assignment.policy": "edf", "sim.seeds": 5})``.
    Used by sweeps and the app.

    * Every dotted key must already exist (a typo raises ``ValueError``).
    * A dict value is **deep-merged** into the group it names:
      ``{"assignment.weights": {"lateness": 2}}`` changes only ``lateness`` and
      keeps the other weights. Unknown keys inside the dict fail validation.
    * numpy scalars/arrays are converted to plain Python first.
    * The result goes through full validation, cross-field rules included.
    """
    data = params.model_dump(mode="python")
    for dotted_key, raw_value in overrides.items():
        value = _to_builtin(copy.deepcopy(raw_value))
        *parents, leaf = dotted_key.split(".")
        node: Any = data
        for part in parents:
            if not isinstance(node, dict) or part not in node:
                raise ValueError(f"unknown parameter {dotted_key!r}")
            node = node[part]
        if not isinstance(node, dict) or leaf not in node:
            raise ValueError(f"unknown parameter {dotted_key!r}")
        _deep_merge(node, {leaf: value})
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
