"""Parameter models: defaults match the spec, invalid input fails loudly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from scout_planner.config import (
    DEFAULT_CONFIG_PATH,
    Params,
    apply_overrides,
    dump_params,
    load_default_params,
    load_params,
    params_from_yaml,
    params_to_yaml,
)

# Every key and default from docs/PARAMETERS.md. If this table and the models
# disagree, one of them drifted from the spec.
SPEC_DEFAULTS: dict[str, Any] = {
    "team.full_time_count": 24,
    "team.freelance_count": 16,
    "team.freelance_weekly_hours": (8, 25),
    "team.skills_per_scout": (1, 4),
    "team.leave_days_per_year": 25,
    "team.follow_hiring_plan": True,
    "demand.start_load": 0.70,
    "demand.actual_growth": 4.0,
    "demand.live_view_share": 0.40,
    "demand.seasonality_strength": 1.0,
    "demand.history_months": 48,
    "demand.turnaround_days": 14,
    "demand.desk_hours": (4, 10),
    "demand.writeup_hours": (2, 4),
    "demand.live_view_hours": 8,
    "capacity_plan.assumed_growth": 4.0,
    "capacity_plan.quantile": 0.80,
    "capacity_plan.target_utilisation": 0.80,
    "capacity_plan.lead_time_full_time_months": 3,
    "capacity_plan.lead_time_freelance_months": 1,
    "capacity_plan.persistent_gap_months": 3,
    "automation.enabled": False,
    "automation.desk_reduction": 0.40,
    "automation.rework_rate": 0.15,
    "automation.rework_overhead_hours": 1.0,
    "automation.monthly_cost": 1500,
    "assignment.policy": "optimiser",
    "assignment.cadence": "daily",
    "assignment.unit": "task",
    "assignment.horizon_days": 7,
    "assignment.weights.lateness": 100,
    "assignment.weights.cost": 1,
    "assignment.weights.continuity": 5,
    "assignment.weights.churn": 3,
    "assignment.time_limit_s": 1.0,
    "cost.full_time_monthly_salary": 4500,
    "cost.freelance_premium": 1.5,
    "cost.late_penalty": 1000,
    "sim.seeds": 3,
    "sim.seed": 42,
    "sim.months": 12,
    "sim.target_on_time": 0.95,
}


def leaf_values(model: BaseModel, prefix: str = "") -> dict[str, Any]:
    """Flatten nested models to ``{"group.key": value}``."""
    out: dict[str, Any] = {}
    for name in type(model).model_fields:
        value = getattr(model, name)
        key = f"{prefix}{name}"
        if isinstance(value, BaseModel):
            out |= leaf_values(value, prefix=f"{key}.")
        else:
            out[key] = value
    return out


# --- defaults ----------------------------------------------------------------


def test_models_mirror_every_documented_parameter() -> None:
    assert leaf_values(Params()) == SPEC_DEFAULTS


def test_default_yaml_equals_code_defaults() -> None:
    assert DEFAULT_CONFIG_PATH.is_file()
    assert load_default_params() == Params()


def test_derived_hourly_rates() -> None:
    cost = Params().cost
    assert cost.salaried_hourly_equivalent == pytest.approx(4500 * 12 / (37.5 * 52))
    assert cost.salaried_hourly_equivalent == pytest.approx(27.69, abs=0.01)
    assert cost.freelance_hourly_rate == pytest.approx(41.54, abs=0.01)


# --- round trip --------------------------------------------------------------


def test_yaml_round_trip_defaults(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    dump_params(Params(), path)
    assert load_params(path) == Params()


def test_yaml_round_trip_non_defaults(tmp_path: Path) -> None:
    params = apply_overrides(
        Params(),
        {
            "assignment.policy": "edf",
            "team.freelance_weekly_hours": [10, 30],
            "demand.start_load": 0.65,
            "assignment.weights.churn": 0.5,
        },
    )
    path = tmp_path / "params.yaml"
    dump_params(params, path)
    assert load_params(path) == params


def test_dumped_yaml_has_no_python_specific_tags() -> None:
    assert "!!python" not in params_to_yaml(Params())


def test_partial_and_empty_yaml_fall_back_to_defaults() -> None:
    assert params_from_yaml("") == Params()
    assert params_from_yaml("sim:\n  seeds: 5\n").sim.seeds == 5


def test_non_mapping_yaml_is_rejected() -> None:
    with pytest.raises(ValueError, match="mapping"):
        params_from_yaml("- just\n- a list\n")


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("team.full_time_count", 101),
        ("team.freelance_count", -1),
        ("team.leave_days_per_year", 61),
        ("team.freelance_weekly_hours", [0, 41]),
        ("team.freelance_weekly_hours", [30, 10]),  # min > max
        ("team.freelance_weekly_hours", [8, 25, 30]),  # not a pair
        ("team.skills_per_scout", [0, 2]),
        ("team.skills_per_scout", [1, 7]),
        ("demand.start_load", 0.2),
        ("demand.actual_growth", 6.5),
        ("demand.live_view_share", 1.1),
        ("demand.history_months", 24),
        ("demand.turnaround_days", 6),
        ("demand.desk_hours", [0, 5]),
        ("demand.writeup_hours", [4, 2]),
        ("demand.live_view_hours", 0),
        ("capacity_plan.quantile", 0.99),
        ("capacity_plan.target_utilisation", 0.4),
        ("capacity_plan.lead_time_full_time_months", 7),
        ("capacity_plan.persistent_gap_months", 0),
        ("automation.desk_reduction", 0.95),
        ("automation.rework_rate", 0.7),
        ("automation.monthly_cost", -1),
        ("assignment.policy", "random"),
        ("assignment.cadence", "hourly"),
        ("assignment.unit", "request"),
        ("assignment.horizon_days", 15),
        ("assignment.time_limit_s", 0.05),
        ("assignment.weights.lateness", -1),
        ("cost.full_time_monthly_salary", 0),
        ("cost.freelance_premium", 0.9),
        ("cost.late_penalty", -10),
        ("sim.seeds", 6),
        ("sim.seed", -1),
        ("sim.months", 25),
        ("sim.target_on_time", 0.4),
    ],
)
def test_out_of_range_values_are_rejected(key: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        apply_overrides(Params(), {key: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("team.follow_hiring_plan", "yes"),  # str is not bool
        ("team.follow_hiring_plan", 1),  # int is not bool
        ("team.full_time_count", 24.5),  # float is not int
        ("demand.start_load", "0.7"),  # str is not float
        ("team.freelance_weekly_hours", ["8", 25]),  # str element in a pair
    ],
)
def test_wrong_types_are_not_coerced(key: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        apply_overrides(Params(), {key: value})


def test_empty_team_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one scout"):
        apply_overrides(Params(), {"team.full_time_count": 0, "team.freelance_count": 0})


@pytest.mark.parametrize(
    "text",
    [
        "teams:\n  full_time_count: 3\n",  # unknown group
        "team:\n  full_time_cout: 3\n",  # typo in a key
        "assignment:\n  weights:\n    speed: 2\n",  # unknown nested key
    ],
)
def test_unknown_keys_are_rejected(text: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        params_from_yaml(text)


def test_params_are_immutable() -> None:
    params = Params()
    with pytest.raises(ValidationError):
        params.sim.seeds = 5  # type: ignore[misc]


# --- overrides ---------------------------------------------------------------


def test_apply_overrides_returns_new_validated_params() -> None:
    base = Params()
    new = apply_overrides(
        base,
        {"assignment.policy": "fcfs", "sim.seeds": 5, "assignment.weights.continuity": 0},
    )
    assert (new.assignment.policy, new.sim.seeds, new.assignment.weights.continuity) == (
        "fcfs",
        5,
        0,
    )
    assert base == Params(), "the input must not be modified"
    assert new.team == base.team


def test_apply_overrides_accepts_list_for_pair() -> None:
    new = apply_overrides(Params(), {"demand.desk_hours": [3, 12]})
    assert new.demand.desk_hours == (3.0, 12.0)


@pytest.mark.parametrize("key", ["sim.seedz", "simulation.seeds", "assignment.weights.speed", ""])
def test_apply_overrides_rejects_unknown_keys(key: str) -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        apply_overrides(Params(), {key: 1})


def test_apply_overrides_rejects_descending_into_a_leaf() -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        apply_overrides(Params(), {"sim.seeds.value": 1})
