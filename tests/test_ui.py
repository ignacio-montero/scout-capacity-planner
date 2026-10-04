"""Pure UI helpers: formatters, labels, validation, review text, diff and metric tables.

These are the app's *functional core* (``app/viewmodel.py`` and
``scout_planner.formatting``): no Streamlit needed to test them.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import viewmodel as vm  # noqa: E402

from scout_planner.config import Params, apply_overrides, load_default_params  # noqa: E402
from scout_planner.formatting import (  # noqa: E402
    fmt_cost_k,
    fmt_count,
    fmt_days,
    fmt_days_delta,
    fmt_duration,
    fmt_growth,
    fmt_money,
    fmt_month,
    fmt_pct,
    fmt_pts,
    fmt_when,
)
from scout_planner.results import MetricStat, Summary  # noqa: E402

DEFAULTS = load_default_params()


def summary(
    on_time: float, lo: float | None = None, hi: float | None = None, **extra: float
) -> Summary:
    metrics = {
        "on_time_rate": MetricStat(
            on_time, lo if lo is not None else on_time, hi if hi is not None else on_time
        )
    }
    for name, value in extra.items():
        metrics[name] = MetricStat(value, value * 0.98, value * 1.02)
    return Summary(metrics, on_time >= 0.95)


# --- formatters ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "value", "expected"),
    [
        (fmt_pct, 0.943, "94.3%"),
        (fmt_pct, None, "—"),
        (fmt_pts, -0.016, "-1.6 pts"),
        (fmt_pts, 0.004, "+0.4 pts"),
        (fmt_pts, 0.0001, "0.0 pts"),
        (fmt_cost_k, 1_240_000, "1,240k"),
        (fmt_cost_k, 4_500, "4.5k"),
        (fmt_cost_k, 0, "0k"),
        (fmt_money, 4500, "4,500"),
        (fmt_days, 17.0, "17.0 days"),
        (fmt_growth, 4.0, "4x"),
        (fmt_growth, 2.5, "2.5x"),
        (fmt_growth, 0.5, "0.5x"),
        (fmt_count, 3120, "3,120"),
        (fmt_duration, 134, "2m 14s"),
        (fmt_duration, 18, "18s"),
        (fmt_duration, 3720, "1h 2m"),
        (fmt_duration, None, "—"),
    ],
)
def test_formatters(fn, value, expected) -> None:
    assert fn(value) == expected


def test_signed_formatters_use_ascii_hyphen() -> None:
    # st.metric reads the arrow direction from a leading ASCII "-".
    for text in (fmt_pts(-0.02), fmt_cost_k(-70_000, signed=True), fmt_days_delta(-1.0)):
        assert text.startswith("-") and "−" not in text
    assert fmt_cost_k(70_000, signed=True) == "+70k"
    assert fmt_days(9.8, short=True) == "9.8 d"
    assert fmt_count(0, signed=True) == "0"
    assert fmt_count(12, signed=True) == "+12"


def test_date_formatters() -> None:
    assert fmt_when(dt.datetime(2026, 10, 3, 10, 15)) == "03 Oct 10:15"
    assert fmt_month(dt.date(2027, 3, 1)) == "Mar 2027"
    assert fmt_when(None) == "—"


# --- PARAM_UI --------------------------------------------------------------------------


def test_param_ui_covers_exactly_the_basic_parameters() -> None:
    basic = {
        "team.full_time_count", "team.freelance_count", "team.follow_hiring_plan",
        "demand.start_load", "demand.actual_growth", "demand.live_view_share",
        "demand.urgent_share",
        "capacity_plan.assumed_growth", "capacity_plan.quantile",
        "capacity_plan.target_utilisation", "automation.enabled", "automation.desk_reduction",
        "automation.rework_rate", "automation.monthly_cost", "assignment.policy",
        "assignment.cadence", "assignment.unit", "cost.full_time_monthly_salary",
        "cost.freelance_premium", "cost.late_penalty", "sim.seeds", "sim.target_on_time",
    }  # fmt: skip
    assert basic == vm.BASIC_KEYS
    for spec in vm.PARAM_UI.values():
        assert spec.group in vm.GROUP_TITLES
        assert f"`{spec.key}`" in spec.help  # YAML key on the tooltip's last line


@pytest.mark.parametrize("key", sorted(vm.PARAM_UI))
def test_display_stored_round_trip(key: str) -> None:
    spec = vm.PARAM_UI[key]
    stored = vm.draft_from_params(DEFAULTS)[key]
    assert spec.to_stored(spec.to_display(stored)) == stored


def test_percent_widgets_show_integers_and_store_fractions() -> None:
    spec = vm.PARAM_UI["demand.start_load"]
    assert spec.to_display(0.7) == 70
    assert spec.to_stored(70) == 0.7


# --- advanced YAML and validation --------------------------------------------------------


def draft(**changes: object) -> dict[str, object]:
    d = vm.draft_from_params(DEFAULTS)
    d.update({k.replace("__", "."): v for k, v in changes.items()})
    return d


def test_defaults_validate_cleanly() -> None:
    check = vm.validate_draft(draft(), vm.advanced_yaml_text(DEFAULTS), DEFAULTS)
    assert check.ok and check.params == DEFAULTS
    assert check.warnings == [] and check.notes == []


def test_advanced_yaml_holds_only_non_basic_keys() -> None:
    text = vm.advanced_yaml_text(DEFAULTS)
    overrides, errors = vm.parse_advanced_yaml(text)
    assert errors == []
    assert "assignment.time_limit_s" in overrides
    assert not set(overrides) & vm.BASIC_KEYS
    assert "schema_version" not in overrides


def test_basic_key_in_advanced_yaml_is_rejected() -> None:
    check = vm.validate_draft(draft(), "sim:\n  seeds: 5\n", DEFAULTS)
    assert not check.ok
    assert any("`sim.seeds` is set in the form above" in e for e in check.advanced_errors)


def test_schema_version_in_advanced_yaml_is_rejected() -> None:
    check = vm.validate_draft(draft(), "schema_version: 1\n", DEFAULTS)
    assert not check.ok


def test_out_of_range_advanced_value_names_key_range_and_input() -> None:
    check = vm.validate_draft(draft(), "assignment:\n  time_limit_s: 30\n", DEFAULTS)
    assert not check.ok
    (message,) = check.errors
    assert message.startswith("Advanced › `assignment.time_limit_s`")
    assert "between 0.1 and 10" in message and "30" in message


def test_unknown_advanced_key_is_rejected() -> None:
    check = vm.validate_draft(draft(), "assignment:\n  horizn_days: 3\n", DEFAULTS)
    assert not check.ok
    assert "horizn_days" in check.errors[0]


def test_yaml_syntax_error_reports_a_line() -> None:
    check = vm.validate_draft(draft(), "assignment:\n  time_limit_s: [1\n", DEFAULTS)
    assert not check.ok
    assert check.errors[0].startswith("Line ")


def test_wrong_type_is_rejected_in_plain_words() -> None:
    check = vm.validate_draft(draft(), "assignment:\n  horizon_days: soon\n", DEFAULTS)
    assert not check.ok
    assert "whole number" in check.errors[0]


def test_invalid_basic_value_uses_the_plain_label() -> None:
    check = vm.validate_draft(
        draft(**{"team.full_time_count": 0, "team.freelance_count": 0}), "", DEFAULTS
    )
    assert not check.ok
    assert "at least one scout" in check.errors[0]


def test_same_growth_link_copies_actual_growth() -> None:
    d = draft(**{"demand.actual_growth": 2.5, "capacity_plan.assumed_growth": 6.0})
    d[vm.SAME_GROWTH] = True
    check = vm.validate_draft(d, "", DEFAULTS)
    assert check.params is not None
    assert check.params.capacity_plan.assumed_growth == 2.5


def test_warnings_and_notes() -> None:
    d = draft(**{"cost.late_penalty": 0.0, "sim.seeds": 1, "capacity_plan.assumed_growth": 2.0})
    d[vm.SAME_GROWTH] = False
    check = vm.validate_draft(d, "", DEFAULTS)
    assert check.ok
    assert any("cost 0" in w for w in check.warnings)
    assert any("One repeat" in w for w in check.warnings)
    assert check.notes == ["Forecast error experiment: hiring for 2x while 4x arrives."]


# --- review card, names, run time -------------------------------------------------------


def test_describe_params_four_sentences() -> None:
    sentences = vm.describe_params(DEFAULTS)
    assert len(sentences) == 4
    assert "**4x**" in sentences[0] and "**70%**" in sentences[0]
    assert "**15%** of requests are urgent (7-day promise)" in sentences[0]
    assert "January" not in sentences[0]  # only with the peak figure
    with_peak = vm.describe_params(DEFAULTS, january_load=1.0)
    assert "about **100%** in January, the transfer window" in with_peak[0]
    assert "cautiously (P80)" in sentences[1]
    assert "**24 full-time scouts and 16 freelancers**" in sentences[2]
    assert "**1,000**" in sentences[3]


def test_changed_settings() -> None:
    params = apply_overrides(
        DEFAULTS, {"capacity_plan.assumed_growth": 2.0, "assignment.horizon_days": 5}
    )
    changes = vm.changed_settings(params, DEFAULTS)
    assert [str(c) for c in changes] == [
        "Assumed growth (what we hire for): 4x → 2x",
        "assignment.horizon_days: 7 → 5",
    ]
    assert vm.changed_settings(DEFAULTS, DEFAULTS) == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "Defaults"),
        ({"capacity_plan.assumed_growth": 2.0}, "EDF + due-date check · 4x actual · 2x assumed"),
        ({"assignment.policy": "edf"}, "earliest deadline"),
        ({"automation.enabled": True}, "EDF + due-date check · pre-screen on"),
        ({"cost.late_penalty": 2000.0}, "EDF + due-date check · cost of one late report 2,000"),
    ],
)
def test_auto_run_name(overrides, expected) -> None:
    assert vm.auto_run_name(apply_overrides(DEFAULTS, overrides), DEFAULTS) == expected


def test_auto_run_name_is_bounded() -> None:
    params = apply_overrides(
        DEFAULTS,
        {"capacity_plan.assumed_growth": 2.0, "automation.enabled": True,
         "team.full_time_count": 30,
         "cost.late_penalty": 50.0, "sim.seeds": 1},
    )  # fmt: skip
    assert len(vm.auto_run_name(params, DEFAULTS)) <= vm.NAME_MAX


def test_runtime_estimate_greedy_is_fast() -> None:
    edf = apply_overrides(DEFAULTS, {"assignment.policy": "edf"})
    assert "under 30 s" in vm.runtime_caption(edf)
    optimiser = apply_overrides(DEFAULTS, {"assignment.policy": "optimiser"})
    assert "min" in vm.runtime_caption(optimiser)


# --- takeaway, KPIs -----------------------------------------------------------------------


def test_run_takeaway_pass_fail_and_close_call() -> None:
    passing = summary(0.961, cost_total=1_310_000)
    assert vm.run_takeaway(passing, DEFAULTS) == (
        "On time 96.1%: 1.1 pts above the 95% target, at 1,310k a year."
    )
    failing = summary(0.934, cost_total=1_240_000)
    assert vm.run_takeaway(failing, DEFAULTS) == (
        "On time 93.4%: 1.6 pts below the 95% target, at 1,240k a year."
    )
    close = summary(0.948, lo=0.94, hi=0.955, cost_total=1_000_000)
    assert vm.run_takeaway(close, DEFAULTS).endswith("Close call: some repeats pass, some don't.")


def test_run_takeaway_reads_target_from_params() -> None:
    params = apply_overrides(DEFAULTS, {"sim.target_on_time": 0.9})
    assert "above the 90% target" in vm.run_takeaway(summary(0.93), params)


def test_kpi_cards_deltas_and_colours() -> None:
    s = summary(0.934, p90_turnaround_days=17.0, cost_total=1_240_000, n_requests=3120.0,
                mean_turnaround_days=9.8, n_at_risk_day_one=41.0)  # fmt: skip
    cards = vm.kpi_cards(s, DEFAULTS)
    assert [c.label for c in cards] == [
        "On-time rate",
        "Turnaround P90",
        "Total cost",
        "Late reports",
    ]
    assert cards[0].delta == "-1.6 pts vs 95% target" and cards[0].delta_color == "normal"
    assert cards[1].delta == "+3.0 days vs 14-day promise" and cards[1].delta_color == "inverse"
    assert cards[2].delta is None
    assert cards[3].value == "206"


# --- compare tables ----------------------------------------------------------------------


def test_param_diff_only_differences_and_code_version() -> None:
    a = DEFAULTS
    b = apply_overrides(DEFAULTS, {"capacity_plan.assumed_growth": 2.0})
    diff = vm.param_diff([a, b])
    assert list(diff.columns) == ["Setting", "Group", "A", "B"]
    assert diff.to_dict("records") == [
        {
            "Setting": "Assumed growth (what we hire for)",
            "Group": vm.GROUP_TITLES["capacity_plan"],
            "A": "4x",
            "B": "2x",
        }
    ]
    with_code = vm.param_diff([a, b], code_versions=["abc", "def"])
    assert with_code.iloc[-1]["Setting"] == "Code version"
    everything = vm.param_diff([a, b], show_all=True)
    assert len(everything) > 20


def test_compare_note() -> None:
    edf = apply_overrides(DEFAULTS, {"assignment.policy": "edf"})
    assert "how work is assigned" in vm.compare_note([DEFAULTS, edf], ["x", "x"])
    assert "deterministic" in vm.compare_note([DEFAULTS, DEFAULTS], ["x", "x"])
    other = apply_overrides(DEFAULTS, {"demand.actual_growth": 2.0})
    assert vm.compare_note([DEFAULTS, other], ["x", "x"]) is None


def test_metric_table_deltas_against_baseline() -> None:
    a = summary(0.934, cost_total=1_240_000, n_requests=3000.0)
    b = summary(0.910, cost_total=1_310_000, n_requests=3000.0)
    table = vm.metric_table([a, b])
    rows = {r["Metric"]: r for r in table.to_dict("records")}
    assert rows["On-time rate"]["A"] == "93.4%"
    assert rows["On-time rate"]["B"] == "91.0% (-2.4 pts)"
    assert rows["On-time rate"]["Better is"] == "higher"
    assert rows["Total cost"]["B"] == "1,310k (+70k)"
    assert rows["Total cost"]["Better is"] == "lower"
    assert rows["Meets target"]["A"] == "No"


def test_param_table_flags_changes() -> None:
    params = apply_overrides(DEFAULTS, {"sim.seeds": 5})
    table = vm.param_table(params, DEFAULTS)
    changed = table[table["Changed"]]
    assert changed["Setting"].tolist() == ["Repeats"]
    assert changed.iloc[0]["Value"] == "5" and changed.iloc[0]["Default"] == "3"


# --- status ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "color", "label"),
    [
        ("queued", "gray", "Queued"),
        ("running", "blue", "Running"),
        ("cancelling", "orange", "Cancelling"),
        ("done", "green", "Done"),
        ("failed", "red", "Failed"),
        ("cancelled", "gray", "Cancelled"),
        ("unreadable", "gray", "Unreadable"),
        ("something-new", "gray", "Unreadable"),
    ],
)
def test_badges(state: str, color: str, label: str) -> None:
    got_color, icon, got_label = vm.badge_spec(state)
    assert (got_color, got_label) == (color, label)
    assert icon.startswith(":material/")  # icon + word: colour is never the only cue


def test_outcome_badge_is_never_red() -> None:
    assert vm.outcome_spec(True)[0] == "green"
    assert vm.outcome_spec(False)[0] == "orange"
    assert vm.outcome_spec(None) is None
    assert (
        vm.badge_markdown(vm.outcome_spec(False))
        == ":orange-badge[:material/trending_down: Misses target]"
    )


def test_stage_line_and_time_left() -> None:
    assert vm.stage_line("simulate") == "Generate › Forecast › Capacity plan › **Simulate**"
    assert vm.time_left(0.05, 30) is None  # too early for a sensible guess
    assert vm.time_left(0.5, 120) == "about 2 min left"
    assert vm.time_left(0.9, 90) == "less than a minute left"


def test_queue_sentence() -> None:
    assert vm.queue_sentence(2) == "It is #2 in line."
    assert vm.queue_sentence(None) == "It is running now."


def test_params_type_is_unchanged() -> None:
    assert isinstance(vm.draft_from_params(Params()), dict)


def test_diagnostics_table_and_warning() -> None:
    quiet = {"assignment_runs": 365, "optimiser_solves": 365, "wall_clock_hits": 0,
             "fallbacks": 0, "mean_solve_seconds": 0.2, "max_solve_seconds": 1.0,
             "per_seed": [{"seed": 1}], "something_new": 5}  # fmt: skip
    table = vm.diagnostics_table(quiet)
    assert table["What"].tolist()[:2] == ["Assignment rounds", "Optimiser solves"]
    assert "something_new" not in table.to_string()  # unknown keys skipped, never fatal
    assert vm.diagnostics_warning(quiet) is None
    noisy = {**quiet, "wall_clock_hits": 3, "fallbacks": 1}
    warning = vm.diagnostics_warning(noisy)
    assert "time limit 3 times" in warning and "earliest deadline first 1 time" in warning
    assert vm.diagnostics_warning({}) is None


def test_start_load_help_mentions_the_january_peak() -> None:
    text = vm.PARAM_UI["demand.start_load"].help
    assert "average month" in text and "January" in text and "1.35x" in text


def test_unknown_advanced_keys_flow_through(monkeypatch: pytest.MonkeyPatch) -> None:
    # A parameter added to config.py later and not in PARAM_UI is just "advanced":
    # it gets a dotted label, its own group heading, and shows up in diffs.
    assert vm.setting_label("demand.some_new_setting") == "demand.some_new_setting"
    assert vm.setting_group("demand.some_new_setting").startswith("Advanced › Demand")
    assert vm.fmt_value("demand.some_new_setting", 0.25) == "0.25"
    edited = apply_overrides(DEFAULTS, {"assignment.horizon_days": 5})
    diff = vm.param_diff([DEFAULTS, edited])
    assert diff["Setting"].tolist() == ["assignment.horizon_days"]


def test_near_target_formatting_never_rounds_a_miss_into_a_pass() -> None:
    from scout_planner.formatting import fmt_pct_near, fmt_pts_near

    assert fmt_pct_near(0.9497, 0.95) == "94.97%"
    assert fmt_pct_near(0.9013, 0.95) == "90.1%"
    assert fmt_pts_near(-0.0003) == "0.03 pts"
    assert fmt_pts_near(0.016) == "1.6 pts"
    s = summary(0.9497, lo=0.94, hi=0.955, cost_total=1_000_000)
    sentence = vm.run_takeaway(s, DEFAULTS)
    assert sentence.startswith("On time 94.97%: 0.03 pts below the 95% target")


def test_policy_options_follow_the_config() -> None:
    from typing import get_args

    from scout_planner.config import AssignmentParams

    allowed = set(get_args(AssignmentParams.model_fields["policy"].annotation))
    spec = vm.PARAM_UI["assignment.policy"]
    assert set(spec.options) == allowed
    assert len(spec.captions) == len(spec.options) and all(spec.captions)
    assert "edf_feasible" in vm.POLICY_CAPTIONS  # ready for when the config allows it
