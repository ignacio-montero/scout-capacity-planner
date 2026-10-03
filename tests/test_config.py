"""Parameter models: defaults match the spec, invalid input fails loudly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from scout_planner.config import (
    DEFAULT_CONFIG_PATH,
    SCHEMA_VERSION,
    Params,
    apply_overrides,
    dump_params,
    load_default_params,
    load_params,
    migrate,
    params_from_yaml,
    params_to_yaml,
)

# Every key and default from docs/PARAMETERS.md. If this table and the models
# disagree, one of them drifted from the spec.
SPEC_DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
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
    "demand.history_growth_per_year": 0.15,
    "demand.turnaround_days": 14,
    "demand.urgent_share": 0.15,
    "demand.urgent_turnaround_days": 7,
    "demand.desk_hours": (4, 10),
    "demand.writeup_hours": (2, 4),
    "demand.live_view_hours": 8,
    "capacity_plan.assumed_growth": 4.0,
    "capacity_plan.quantile": 0.80,
    "capacity_plan.target_utilisation": 0.80,
    "capacity_plan.lead_time_full_time_months": 3,
    "capacity_plan.lead_time_freelance_months": 1,
    "capacity_plan.persistent_gap_months": 3,
    "capacity_plan.hire_mix": "rule",
    "automation.enabled": False,
    "automation.desk_reduction": 0.40,
    "automation.rework_rate": 0.15,
    "automation.rework_overhead_hours": 1.0,
    "automation.monthly_cost": 1500,
    "assignment.policy": "optimiser",
    "assignment.cadence": "daily",
    "assignment.unit": "task",
    "assignment.horizon_days": 7,
    "assignment.weights.lateness": 1.0,  # multiplier on cost.late_penalty (S4)
    "assignment.weights.cost": 1,
    "assignment.weights.continuity": 5,
    "assignment.weights.churn": 3,
    "assignment.time_limit_s": 1.0,
    "assignment.cost_basis": "premium",
    "assignment.commit_buffer_days": 2,
    "assignment.load_aware": True,
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
        ("demand.history_growth_per_year", -0.6),
        ("demand.history_growth_per_year", 1.1),
        ("demand.turnaround_days", 6),
        ("demand.urgent_share", 1.1),
        ("demand.urgent_turnaround_days", 2),
        ("demand.urgent_turnaround_days", 15),  # > turnaround_days (14)
        ("demand.desk_hours", [0, 5]),
        ("demand.writeup_hours", [4, 2]),
        ("demand.live_view_hours", 0),
        ("capacity_plan.quantile", 0.99),
        ("capacity_plan.target_utilisation", 0.4),
        ("capacity_plan.lead_time_full_time_months", 7),
        ("capacity_plan.persistent_gap_months", 0),
        ("capacity_plan.hire_mix", "contractors"),
        ("automation.desk_reduction", 0.95),
        ("automation.rework_rate", 0.7),
        ("automation.monthly_cost", -1),
        ("assignment.policy", "random"),
        ("assignment.cadence", "hourly"),
        ("assignment.unit", "request"),
        ("assignment.horizon_days", 15),
        ("assignment.time_limit_s", 0.05),
        ("assignment.cost_basis", "half"),
        ("assignment.commit_buffer_days", -1),
        ("assignment.commit_buffer_days", 2.0),  # strict: not an int
        ("assignment.load_aware", "yes"),
        ("assignment.weights.lateness", 0),  # must be > 0
        ("assignment.weights.lateness", 10.5),
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


def test_apply_overrides_deep_merges_dict_values() -> None:
    new = apply_overrides(Params(), {"assignment.weights": {"lateness": 2.0}})
    assert new.assignment.weights.lateness == 2.0
    assert new.assignment.weights.churn == Params().assignment.weights.churn  # kept
    new = apply_overrides(Params(), {"team": {"full_time_count": 30}})
    assert new.team.full_time_count == 30
    assert new.team.freelance_count == Params().team.freelance_count


def test_apply_overrides_dict_value_with_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        apply_overrides(Params(), {"assignment.weights": {"speed": 2.0}})


def test_apply_overrides_does_not_keep_references_to_caller_values() -> None:
    weights = {"lateness": 2.0}
    new = apply_overrides(Params(), {"assignment.weights": weights})
    weights["lateness"] = 9.0
    assert new.assignment.weights.lateness == 2.0


def test_apply_overrides_accepts_numpy_values() -> None:
    new = apply_overrides(
        Params(),
        {
            "sim.seeds": np.int64(5),
            "demand.start_load": np.float64(0.8),
            "automation.enabled": np.bool_(True),
            "demand.desk_hours": np.array([3.0, 9.0]),
            "team.skills_per_scout": (np.int64(2), np.int64(3)),
        },
    )
    assert new.sim.seeds == 5 and type(new.sim.seeds) is int
    assert new.demand.start_load == 0.8
    assert new.automation.enabled is True
    assert new.demand.desk_hours == (3.0, 9.0)
    assert new.team.skills_per_scout == (2, 3)


# --- cross-field rule: one work item fits in one full-timer's window (S5) --------


def test_defaults_satisfy_the_item_size_rule_for_both_units() -> None:
    apply_overrides(Params(), {"assignment.unit": "bundle"})  # 14 h <= 37.5 h


@pytest.mark.parametrize(
    ("overrides", "named_key"),
    [
        ({"assignment.horizon_days": 1}, "demand.desk_hours"),  # 10 h > 5.36 h
        ({"assignment.horizon_days": 1, "demand.desk_hours": [1, 2]}, "demand.live_view_hours"),
        (
            {
                "assignment.horizon_days": 2,
                "demand.desk_hours": [1, 2],
                "demand.live_view_hours": 4,
            },
            None,  # 2 days -> 10.7 h: every single item fits
        ),
        (
            {
                "assignment.horizon_days": 2,
                "assignment.unit": "bundle",
                "demand.desk_hours": [1, 8],
                "demand.writeup_hours": [1, 4],
                "demand.live_view_hours": 4,
            },
            "unit=bundle",  # 8 + 4 = 12 h > 10.7 h, though each task alone fits
        ),
    ],
)
def test_items_larger_than_a_window_are_rejected(
    overrides: dict[str, Any], named_key: str | None
) -> None:
    if named_key is None:
        apply_overrides(Params(), overrides)
        return
    with pytest.raises(ValidationError, match="assignment.horizon_days") as excinfo:
        apply_overrides(Params(), overrides)
    assert named_key in str(excinfo.value)


# --- YAML edge cases ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "sim:\n  seeds: 3\n  seeds: 5\n",  # duplicate leaf
        "sim:\n  seeds: 3\nsim:\n  months: 6\n",  # duplicate group
    ],
)
def test_duplicate_yaml_keys_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        params_from_yaml(text)


def test_malformed_yaml_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="invalid parameter YAML"):
        params_from_yaml("sim: [1, 2\n")


# --- schema versioning -------------------------------------------------------------


def test_schema_version_is_written_first_in_yaml() -> None:
    assert params_to_yaml(Params()).startswith(f"schema_version: {SCHEMA_VERSION}\n")


def test_file_without_schema_version_is_read_as_v1() -> None:
    old_layout = "sim:\n  seeds: 2\n"  # e.g. a run folder written before versioning
    assert params_from_yaml(old_layout).schema_version == SCHEMA_VERSION


def test_migrate_is_a_pure_no_op_for_current_version() -> None:
    raw = {"schema_version": 1, "sim": {"seeds": 2}}
    out = migrate(raw)
    assert out == raw and out is not raw
    out["sim"]["seeds"] = 4
    assert raw["sim"]["seeds"] == 2, "migrate must not mutate its input"


@pytest.mark.parametrize("version", [SCHEMA_VERSION + 1, "1", True])
def test_unsupported_schema_versions_are_rejected(version: object) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        migrate({"schema_version": version})


def test_params_reject_a_non_current_schema_version_directly() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        Params.model_validate({"schema_version": 0})
