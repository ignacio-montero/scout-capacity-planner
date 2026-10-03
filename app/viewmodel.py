"""Pure view-model helpers for the simulator app: no Streamlit import here.

Everything the pages *decide* (labels, formats, validation messages, review
sentences, diff and metric tables, the takeaway sentence, badge specs) lives
here as plain functions over plain data, so it is unit-tested without a
browser (tests/test_ui.py). ``ui.py`` holds the Streamlit components and
re-exports these names, so pages can use ``ui.fmt_pct`` etc. as the design
doc's section 8 names them.

This split is the *functional core, imperative shell* idea applied to a UI:
the shell (Streamlit) renders, the core decides.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yaml
from pydantic import ValidationError

from scout_planner.charts import POLICY_LABELS, POLICY_ORDER, SLOT_LETTERS
from scout_planner.config import Params, apply_overrides
from scout_planner.formatting import (
    DASH,
    fmt_cost_k,
    fmt_count,
    fmt_days,
    fmt_days_delta,
    fmt_duration,
    fmt_growth,
    fmt_money,
    fmt_pct,
    fmt_pts,
    fmt_when,
)
from scout_planner.results import MetricStat, Summary, flatten_params

# No __all__: ui.py re-exports every public name here with a star import.
HIDDEN_KEYS = frozenset({"schema_version"})  # managed by config.py, never by the user
SAME_GROWTH = "_same_growth"  # draft flag: "Same as actual growth" checkbox

# --- parameter UI spec (DESIGN_SYSTEM.md section 5.1) ------------------------------

GROUPS: tuple[tuple[str, str], ...] = (
    ("demand", "Demand (what actually arrives)"),
    ("capacity_plan", "Capacity plan (what the agency hires for)"),
    ("team", "Team (on day one)"),
    ("assignment", "Assignment"),
    ("automation", "Video pre-screen"),
    ("cost", "Costs"),
    ("sim", "Simulation"),
)
GROUP_TITLES = dict(GROUPS)
# Two groups per row; Demand and Capacity plan side by side (actual vs assumed).
FORM_ROWS: tuple[tuple[str, str | None], ...] = (
    ("demand", "capacity_plan"),
    ("team", "assignment"),
    ("automation", "cost"),
    ("sim", None),
)
GROWTH_OPTIONS: tuple[float, ...] = tuple(x / 2 for x in range(1, 13))  # 0.5x .. 6x
QUANTILE_LABELS: dict[float, str] = {
    0.5: "Average (P50)",
    0.8: "Cautious (P80)",
    0.9: "Very cautious (P90)",
}


def quantile_label(q: float) -> str:
    return QUANTILE_LABELS.get(round(q, 4), f"Custom (P{round(q * 100)})")


def _on_off(value: Any) -> str:
    return "On" if value else "Off"


@dataclass(frozen=True)
class ParamUI:
    """How one basic setting appears in the form, tables and diffs.

    ``percent=True``: the widget shows 70 and the parameter stores 0.70; the
    conversion happens only in :meth:`to_display` / :meth:`to_stored`.
    """

    key: str
    label: str
    group: str
    widget: str  # growth | pct | int | money | toggle | radio | premium | seeds | quantile
    help: str
    fmt: Callable[[Any], str]
    min: float | None = None
    max: float | None = None
    step: float | None = None
    percent: bool = False
    options: tuple[Any, ...] = ()
    option_labels: Mapping[Any, str] = field(default_factory=dict)
    captions: tuple[str, ...] = ()
    horizontal: bool = True
    depends_on: str | None = None  # disabled while this boolean setting is off

    def to_display(self, value: Any) -> Any:
        if self.percent:
            return round(float(value) * 100, 2)
        if self.widget in ("int", "seeds"):
            return int(value)
        if self.widget in ("money", "premium", "growth"):
            return float(value)
        return value

    def to_stored(self, value: Any) -> Any:
        if self.percent:
            return round(float(value) / 100, 4)
        if self.widget in ("int", "seeds"):
            return int(value)
        if self.widget in ("money", "premium", "growth", "quantile"):
            return round(float(value), 4)
        return value


def _help(text: str, key: str) -> str:
    # The YAML key goes last, in code font, for people who use the YAML editor.
    return f"{text}\n\n`{key}`"


_PARAMS: tuple[ParamUI, ...] = (
    ParamUI(
        "demand.actual_growth", "Actual growth (what arrives)", "demand", "growth",
        _help("How much request volume grows over the year: volume in month 12 ÷ volume "
              "in month 1.", "demand.actual_growth"),
        fmt_growth, options=GROWTH_OPTIONS,
    ),
    ParamUI(
        "demand.start_load", "Starting workload", "demand", "pct",
        _help("How busy the starting team is in month one. 70% means arriving work needs 70% "
              "of the team's usable hours. Sets the base volume.", "demand.start_load"),
        lambda v: fmt_pct(v, 0), 30, 120, 5, percent=True,
    ),
    ParamUI(
        "demand.live_view_share", "Live-view share", "demand", "pct",
        _help("Share of requests that need a scout at a match. A live view takes the "
              "scout's whole day, in the match's region.", "demand.live_view_share"),
        lambda v: fmt_pct(v, 0), 0, 100, 5, percent=True,
    ),
    ParamUI(
        "capacity_plan.assumed_growth", "Assumed growth (what we hire for)", "capacity_plan",
        "growth", _help("The growth the hiring plan is built on.", "capacity_plan.assumed_growth"),
        fmt_growth, options=GROWTH_OPTIONS,
    ),
    ParamUI(
        "capacity_plan.quantile", "Planning caution", "capacity_plan", "quantile",
        _help("How much buffer the plan keeps. Cautious plans for a month busier than 80% of "
              "likely outcomes: more hires, fewer late reports.", "capacity_plan.quantile"),
        quantile_label, options=(0.5, 0.8, 0.9), option_labels=QUANTILE_LABELS,
    ),
    ParamUI(
        "capacity_plan.target_utilisation", "Target workload per scout", "capacity_plan", "pct",
        _help("The plan hires once scouts would be busier than this. Lower means more slack "
              "and more cost.", "capacity_plan.target_utilisation"),
        lambda v: fmt_pct(v, 0), 50, 100, 5, percent=True,
    ),
    ParamUI(
        "team.full_time_count", "Full-time scouts", "team", "int",
        _help("Salaried scouts on day one. Fixed monthly cost, busy or not.",
              "team.full_time_count"),
        fmt_count, 0, 100, 1,
    ),
    ParamUI(
        "team.freelance_count", "Freelancers", "team", "int",
        _help("Freelancers on day one. They offer 8–25 hours a week and are paid only for "
              "hours used, at a premium.", "team.freelance_count"),
        fmt_count, 0, 100, 1,
    ),
    ParamUI(
        "team.follow_hiring_plan", "Planned hires actually join", "team", "toggle",
        _help("On: the scouts the hiring plan asks for join when recruitment finishes. "
              "Off: the team never grows.", "team.follow_hiring_plan"),
        _on_off,
    ),
    ParamUI(
        "assignment.policy", "Who does what", "assignment", "radio",
        _help("The rule that decides which scout does which task.", "assignment.policy"),
        lambda v: POLICY_LABELS.get(v, v), options=POLICY_ORDER, option_labels=POLICY_LABELS,
        captions=("Oldest request first", "Most urgent first",
                  "Plans the best fit each round; slowest"),
        horizontal=False,
    ),
    ParamUI(
        "assignment.cadence", "How often work is assigned", "assignment", "radio",
        _help("Every day re-plans the coming week each morning; once a week plans on "
              "Mondays only.", "assignment.cadence"),
        lambda v: {"daily": "Every day", "weekly": "Once a week"}.get(v, v),
        options=("daily", "weekly"), option_labels={"daily": "Every day", "weekly": "Once a week"},
    ),
    ParamUI(
        "assignment.unit", "How work is handed out", "assignment", "radio",
        _help("Together keeps one scout on a report's desk work and write-up.",
              "assignment.unit"),
        lambda v: {"task": "Task by task", "bundle": "Desk review and write-up together"}.get(v, v),
        options=("task", "bundle"),
        option_labels={"task": "Task by task", "bundle": "Desk review and write-up together"},
    ),
    ParamUI(
        "automation.enabled", "Use the video pre-screen tool", "automation", "toggle",
        _help("Software that does part of the desk review.", "automation.enabled"), _on_off,
    ),
    ParamUI(
        "automation.desk_reduction", "Desk time saved when it works", "automation", "pct",
        _help("Share of the desk review the tool does when its output is usable.",
              "automation.desk_reduction"),
        lambda v: fmt_pct(v, 0), 0, 90, 5, percent=True, depends_on="automation.enabled",
    ),
    ParamUI(
        "automation.rework_rate", "Chance its output is unusable", "automation", "pct",
        _help("When it fails, the scout does the full desk review anyway, plus about an hour "
              "lost.", "automation.rework_rate"),
        lambda v: fmt_pct(v, 0), 0, 60, 5, percent=True, depends_on="automation.enabled",
    ),
    ParamUI(
        "automation.monthly_cost", "Tool licence per month", "automation", "money",
        _help("What the tool costs each month it is switched on. Without it the tool would "
              "look free.", "automation.monthly_cost"),
        fmt_money, 0, 20_000, 100, depends_on="automation.enabled",
    ),
    ParamUI(
        "cost.full_time_monthly_salary", "Monthly cost per full-time scout", "cost", "money",
        _help("Fully loaded salary.", "cost.full_time_monthly_salary"), fmt_money, 100, 20_000, 100,
    ),
    ParamUI(
        "cost.freelance_premium", "Freelance premium", "cost", "premium",
        _help("Freelance hourly rate ÷ salaried hourly cost.", "cost.freelance_premium"),
        lambda v: f"{v:.1f}x", 1.0, 3.0, 0.1,
    ),
    ParamUI(
        "cost.late_penalty", "Cost of one late report", "cost", "money",
        _help("Refunds and lost clients per late report.", "cost.late_penalty"),
        fmt_money, 0, 20_000, 100,
    ),
    ParamUI(
        "sim.seeds", "Repeats", "sim", "seeds",
        _help("The year is simulated this many times with different luck; results show the "
              "average and the range. More repeats: slower, more reliable.", "sim.seeds"),
        lambda v: str(v), 1, 5, 1,
    ),
    ParamUI(
        "sim.target_on_time", "On-time target", "sim", "pct",
        _help("The service level a run must reach to pass.", "sim.target_on_time"),
        lambda v: fmt_pct(v, 0), 50, 100, 1, percent=True,
    ),
)  # fmt: skip
PARAM_UI: dict[str, ParamUI] = {p.key: p for p in _PARAMS}
BASIC_KEYS: frozenset[str] = frozenset(PARAM_UI)


def fmt_value(key: str, value: Any) -> str:
    """Any setting's value in plain words (basic keys use their own formatter)."""
    spec = PARAM_UI.get(key)
    if spec is not None:
        return spec.fmt(value)
    if isinstance(value, bool):
        return _on_off(value)
    if isinstance(value, tuple | list):
        return "[" + ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def setting_label(key: str) -> str:
    spec = PARAM_UI.get(key)
    return spec.label if spec else key


def setting_group(key: str) -> str:
    spec = PARAM_UI.get(key)
    if spec:
        return GROUP_TITLES[spec.group]
    return f"Advanced › {GROUP_TITLES.get(key.split('.')[0], key.split('.')[0])}"


# --- the draft: form values between reruns -------------------------------------------


def draft_from_params(params: Params) -> dict[str, Any]:
    """Stored values of the basic settings, plus the "same growth" link flag."""
    flat = flatten_params(params)
    draft = {key: flat[key] for key in PARAM_UI}
    draft[SAME_GROWTH] = math.isclose(
        flat["demand.actual_growth"], flat["capacity_plan.assumed_growth"]
    )
    return draft


def draft_overrides(draft: Mapping[str, Any]) -> dict[str, Any]:
    overrides = {key: draft[key] for key in PARAM_UI if key in draft}
    if draft.get(SAME_GROWTH):
        overrides["capacity_plan.assumed_growth"] = overrides.get("demand.actual_growth")
    return overrides


def advanced_yaml_text(params: Params) -> str:
    """The non-basic settings as nested YAML (what the advanced editor starts with)."""

    def strip(node: dict[str, Any], prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in node.items():
            dotted = f"{prefix}{key}"
            if dotted in BASIC_KEYS or dotted in HIDDEN_KEYS:
                continue
            if isinstance(value, dict):
                inner = strip(value, f"{dotted}.")
                if inner:
                    out[key] = inner
            else:
                out[key] = value
        return out

    return yaml.safe_dump(strip(params.model_dump(mode="json"), ""), sort_keys=False)


def _flatten_mapping(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def parse_advanced_yaml(text: str) -> tuple[dict[str, Any], list[str]]:
    """The advanced editor's text -> dotted overrides, or plain-words errors."""
    try:
        data = yaml.safe_load(text or "")
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        problem = getattr(exc, "problem", None) or "the text is not valid YAML"
        where = f"Line {mark.line + 1}: " if mark is not None else ""
        return {}, [f"{where}{problem}. Check the indentation and the `key: value` format."]
    if data is None:
        return {}, []
    if not isinstance(data, Mapping):
        return {}, ["The advanced settings must be `key: value` lines, grouped as shown."]
    overrides = _flatten_mapping(data)
    errors = []
    for key in overrides:
        if key in HIDDEN_KEYS:
            errors.append(f"`{key}` is set automatically; remove it here.")
        elif key in BASIC_KEYS:
            errors.append(f"`{key}` is set in the form above; remove it here.")
    if errors:
        return {}, errors
    return overrides, []


def _field_bounds(dotted: str) -> tuple[float | None, float | None]:
    """``(lower, upper)`` declared on a parameter field, for friendlier messages."""
    model: Any = Params
    parts = dotted.split(".")
    try:
        for part in parts[:-1]:
            model = model.model_fields[part].annotation
        info = model.model_fields[parts[-1]]
    except (KeyError, AttributeError):
        return None, None
    lo = hi = None
    for meta in info.metadata:
        lo = getattr(meta, "ge", getattr(meta, "gt", lo)) if lo is None else lo
        hi = getattr(meta, "le", getattr(meta, "lt", hi)) if hi is None else hi
    return lo, hi


def _where(key: str) -> str:
    if not key:
        return "Settings"
    if key in PARAM_UI:
        return PARAM_UI[key].label
    return f"Advanced › `{key}`"


def _shown(key: str, value: Any) -> str:
    spec = PARAM_UI.get(key)
    if spec is not None and isinstance(value, int | float) and not isinstance(value, bool):
        try:
            return spec.fmt(value)
        except (TypeError, ValueError):
            pass
    return repr(value) if isinstance(value, str) else str(value)


def validation_messages(exc: ValidationError) -> list[str]:
    """Pydantic errors -> "what is wrong, and with which value" in plain words."""
    messages = []
    for err in exc.errors():
        key = ".".join(str(p) for p in err["loc"] if not isinstance(p, int))
        kind, value = err["type"], err.get("input")
        if kind in ("greater_than_equal", "less_than_equal", "greater_than", "less_than"):
            lo, hi = _field_bounds(key)
            if lo is not None and hi is not None:
                text = f"must be between {lo:g} and {hi:g}"
            elif lo is not None:
                text = f"must be at least {lo:g}"
            else:
                text = f"must be at most {hi:g}"
            text += f" (you entered {value})"
        elif kind == "literal_error":
            text = f"must be one of {err.get('ctx', {}).get('expected')} (you entered {value!r})"
        elif kind == "extra_forbidden":
            text = "is not a setting; check the spelling"
        elif kind.endswith("_type") or kind.endswith("_parsing"):
            wanted = (
                kind.split("_")[0].replace("bool", "true or false").replace("int", "whole number")
            )
            wanted = wanted.replace("float", "number")
            text = f"must be a {wanted} (you entered {value!r})"
        else:
            text = str(err.get("msg", "invalid value")).removeprefix("Value error, ")
        messages.append(f"{_where(key)}: {text}.")
    return messages


@dataclass(frozen=True)
class DraftCheck:
    """Result of validating the form + advanced YAML. ``params`` is None if invalid."""

    params: Params | None
    errors: list[str]
    advanced_errors: list[str]
    warnings: list[str]
    notes: list[str]

    @property
    def ok(self) -> bool:
        return self.params is not None and not self.errors


def validate_draft(draft: Mapping[str, Any], advanced_yaml: str, defaults: Params) -> DraftCheck:
    """Merge defaults + form + advanced YAML and validate with the pydantic models.

    Errors block the Run button; warnings (valid but suspicious) and notes
    (a deliberate experiment) don't.
    """
    overrides, advanced_errors = parse_advanced_yaml(advanced_yaml)
    errors = list(advanced_errors)
    params: Params | None = None
    try:
        params = apply_overrides(defaults, {**draft_overrides(draft), **overrides})
    except ValidationError as exc:
        messages = validation_messages(exc)
        errors += messages
        advanced_errors += [m for m in messages if m.startswith("Advanced")]
    except ValueError as exc:
        message = str(exc)
        if message.startswith("unknown parameter"):
            key = message.split("'")[1] if "'" in message else "?"
            message = f"Advanced › `{key}`: is not a setting; check the spelling."
        errors.append(message)
        advanced_errors.append(message)
    if errors:
        return DraftCheck(None, errors, advanced_errors, [], [])
    assert params is not None
    return DraftCheck(params, [], [], draft_warnings(params), draft_notes(params))


def draft_warnings(params: Params) -> list[str]:
    warnings = []
    if params.cost.late_penalty == 0:
        warnings.append("Late reports cost 0: understaffing will look cheapest.")
    if params.sim.seeds == 1:
        warnings.append("One repeat: results will show no range.")
    if params.automation.enabled and params.automation.monthly_cost == 0:
        warnings.append("The pre-screen tool costs nothing in this run, so it will look free.")
    if not params.team.follow_hiring_plan and params.demand.actual_growth >= 1.5:
        warnings.append(
            "Planned hires don't join while demand grows "
            f"{fmt_growth(params.demand.actual_growth)}:"
            " expect a growing backlog."
        )
    return warnings


def draft_notes(params: Params) -> list[str]:
    actual, assumed = params.demand.actual_growth, params.capacity_plan.assumed_growth
    if not math.isclose(actual, assumed):
        return [
            f"Forecast error experiment: hiring for {fmt_growth(assumed)} while "
            f"{fmt_growth(actual)} arrives."
        ]
    return []


# --- review card, names, run time ----------------------------------------------------


def _caution_phrase(q: float) -> str:
    return {
        0.5: "on the average forecast (P50)",
        0.8: "cautiously (P80)",
        0.9: "very cautiously (P90)",
    }.get(round(q, 4), f"with a P{round(q * 100)} buffer")


def describe_params(params: Params) -> list[str]:
    """The four review-card sentences, values in bold (markdown)."""
    d, c, t, a, s = params.demand, params.capacity_plan, params.team, params.assignment, params.sim
    cadence = "every day" if a.cadence == "daily" else "once a week"
    unit = "task by task" if a.unit == "task" else "desk review and write-up together"
    policy = POLICY_LABELS.get(a.policy, a.policy).lower()
    auto = params.automation
    if auto.enabled:
        tool = (
            f"**on** (saves {fmt_pct(auto.desk_reduction, 0)} of desk time, unusable "
            f"{fmt_pct(auto.rework_rate, 0)} of the time, {fmt_money(auto.monthly_cost)} a month)"
        )
    else:
        tool = "**off**"
    repeats = "1 repeat" if s.seeds == 1 else f"{s.seeds} repeats"
    return [
        f"Demand grows **{fmt_growth(d.actual_growth)}** over the year, starting at "
        f"**{fmt_pct(d.start_load, 0)}** of the team's capacity.",
        f"The agency **hires for {fmt_growth(c.assumed_growth)}**, "
        f"**{_caution_phrase(c.quantile)}**,"
        f" and planned hires **{'join' if t.follow_hiring_plan else 'never join'}**.",
        f"**{t.full_time_count} full-time scouts and {t.freelance_count} freelancers** to start; "
        f"work is assigned **{cadence}, {unit}**, by **{policy}**; the pre-screen tool is {tool}.",
        f"A late report costs **{fmt_money(params.cost.late_penalty)}**; **{repeats}**; "
        f"target **{fmt_pct(s.target_on_time, 0)}** on time.",
    ]


@dataclass(frozen=True)
class Change:
    key: str
    label: str
    old: str
    new: str

    def __str__(self) -> str:
        return f"{self.label}: {self.old} → {self.new}"


def changed_settings(params: Params, reference: Params) -> list[Change]:
    """Every setting that differs from ``reference``: basic ones first, in form order."""
    new, old = flatten_params(params), flatten_params(reference)
    keys = [k for k in PARAM_UI if new[k] != old[k]]
    keys += [
        k for k in new if k not in BASIC_KEYS and k not in HIDDEN_KEYS and new[k] != old.get(k)
    ]
    return [
        Change(k, setting_label(k), fmt_value(k, old.get(k)), fmt_value(k, new[k])) for k in keys
    ]


POLICY_SHORT = {"fcfs": "first come", "edf": "earliest deadline", "optimiser": "optimiser"}
NAME_MAX = 60


def auto_run_name(params: Params, defaults: Params) -> str:
    """A readable default run name: the policy, then what changed that matters most.

    ``"Defaults"`` if nothing changed; e.g. ``"optimiser · 4x actual · 2x assumed"``.
    """
    changes = changed_settings(params, defaults)
    if not changes:
        return "Defaults"
    keys = {c.key for c in changes}
    parts = [POLICY_SHORT.get(params.assignment.policy, params.assignment.policy)]
    d, c = params.demand, params.capacity_plan
    if keys & {"demand.actual_growth", "capacity_plan.assumed_growth"}:
        parts.append(f"{fmt_growth(d.actual_growth)} actual")
        if not math.isclose(d.actual_growth, c.assumed_growth):
            parts.append(f"{fmt_growth(c.assumed_growth)} assumed")
    if "automation.enabled" in keys:
        parts.append(f"pre-screen {'on' if params.automation.enabled else 'off'}")
    if keys & {"team.full_time_count", "team.freelance_count"}:
        parts.append(f"{params.team.full_time_count} FT + {params.team.freelance_count} freelance")
    shown = {
        "assignment.policy", "demand.actual_growth", "capacity_plan.assumed_growth",
        "automation.enabled", "team.full_time_count", "team.freelance_count",
    }  # fmt: skip
    parts = parts[:4]
    rest = [ch for ch in changes if ch.key not in shown]
    if rest and len(parts) < 4:
        first = rest[0]
        parts.append(f"{first.label.lower()} {first.new}")
        if len(rest) > 1:
            parts.append(f"+{len(rest) - 1} more")
    return " · ".join(parts)[:NAME_MAX]


def estimate_runtime(params: Params) -> tuple[float, float]:
    """Rough (low, high) seconds, from ARCHITECTURE.md section 7's budget."""
    lo, hi = (60.0, 360.0) if params.assignment.policy == "optimiser" else (5.0, 20.0)
    scale = params.sim.seeds / 3 * params.sim.months / 12
    if params.assignment.cadence == "weekly" and params.assignment.policy == "optimiser":
        scale /= 3
    return lo * max(scale, 0.3), hi * max(scale, 0.3)


def runtime_caption(params: Params) -> str:
    lo, hi = estimate_runtime(params)
    if hi <= 30:
        rough = "under 30 s"
    elif hi < 90:
        rough = f"{fmt_duration(lo)}–{fmt_duration(hi)}"
    else:
        rough = f"{max(1, round(lo / 60))}–{max(1, round(hi / 60))} min"
    return (
        f"Rough run time: {rough}. You can leave this page or close the tab; the run keeps going."
    )


# --- run detail ----------------------------------------------------------------------


def run_takeaway(summary: Summary, params: Params) -> str:
    """The one-sentence answer at the top of Run detail (DESIGN_SYSTEM.md section 5.3)."""
    stat = summary.stat("on_time_rate")
    if stat is None:
        return "No results for this run."
    target = params.sim.target_on_time
    gap = stat.mean - target
    cost = summary.mean("cost_total")
    cost_part = f", at {fmt_cost_k(cost)} a year" if cost is not None else ""
    points = abs(round(gap * 100, 1))
    if points == 0:
        where = f"right at the {fmt_pct(target, 0)} target"
    else:
        side = "above" if gap > 0 else "below"
        where = f"{points:.1f} pts {side} the {fmt_pct(target, 0)} target"
    sentence = f"On time {fmt_pct(stat.mean)}: {where}{cost_part}."
    if stat.min < target <= stat.max:
        sentence += " Close call: some repeats pass, some don't."
    return sentence


@dataclass(frozen=True)
class Kpi:
    label: str
    value: str
    delta: str | None
    delta_color: str
    caption: str
    help: str


def _range(stat: MetricStat | None, fmt: Callable[[float], str]) -> str:
    if stat is None:
        return DASH
    return f"{fmt(stat.min)}–{fmt(stat.max)}"


def kpi_cards(summary: Summary, params: Params, late: MetricStat | None = None) -> list[Kpi]:
    """The four KPI cards. ``late`` (exact, from seeds.parquet) overrides the approximation."""
    target = params.sim.target_on_time
    promise = params.demand.turnaround_days
    seeds = params.sim.seeds
    on_time = summary.stat("on_time_rate")
    p90 = summary.stat("p90_turnaround_days")
    cost = summary.stat("cost_total")
    at_risk = summary.mean("n_at_risk_day_one")
    n_req = summary.mean("n_requests")
    late_mean = late.mean if late is not None else summary.late_reports()
    repeats = f"over {seeds} repeat{'s' if seeds != 1 else ''}"
    risk = f" · {fmt_count(at_risk)} requests at risk from day one" if at_risk is not None else ""
    return [
        Kpi(
            "On-time rate",
            fmt_pct(on_time.mean if on_time else None),
            f"{fmt_pts(on_time.mean - target)} vs {fmt_pct(target, 0)} target" if on_time else None,
            "normal",
            f"Range {_range(on_time, fmt_pct)} {repeats}{risk}",
            "Share of reports delivered within the promised days.",
        ),
        Kpi(
            "Turnaround P90",
            fmt_days(p90.mean if p90 else None),
            f"{fmt_days_delta(p90.mean - promise)} vs {promise}-day promise" if p90 else None,
            "inverse",
            f"Average {fmt_days(summary.mean('mean_turnaround_days'))} · range "
            f"{_range(p90, lambda v: f'{v:.1f}')}",
            "9 in 10 reports were delivered within this many days.",
        ),
        Kpi(
            "Total cost",
            fmt_cost_k(cost.mean if cost else None),
            None,
            "off",
            f"Range {_range(cost, fmt_cost_k)}",
            "Salaries + freelance hours + late penalties + tool licence, for the year.",
        ),
        Kpi(
            "Late reports",
            fmt_count(late_mean),
            None,
            "off",
            (
                f"of {fmt_count(n_req)} requests ({fmt_pct(late_mean / n_req)})"
                if late_mean is not None and n_req
                else DASH
            ),
            "Reports delivered after the promised date (mean per repeat).",
        ),
    ]


def param_table(params: Params, reference: Params) -> pd.DataFrame:
    """Group, Setting, Value, Default, Changed: every setting, basic ones first."""
    new, old = flatten_params(params), flatten_params(reference)
    keys = list(PARAM_UI) + [k for k in new if k not in BASIC_KEYS and k not in HIDDEN_KEYS]
    return pd.DataFrame(
        [
            {
                "Group": setting_group(k),
                "Setting": setting_label(k),
                "Value": fmt_value(k, new[k]),
                "Default": fmt_value(k, old.get(k)),
                "Changed": new[k] != old.get(k),
            }
            for k in keys
        ]
    )


# --- compare ---------------------------------------------------------------------------


def slot_letters(n: int) -> list[str]:
    return list(SLOT_LETTERS[:n])


def param_diff(
    params_list: Sequence[Params],
    *,
    show_all: bool = False,
    code_versions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Setting, Group, A, B, ...: only the settings whose values differ (unless ``show_all``)."""
    flats = [flatten_params(p) for p in params_list]
    letters = slot_letters(len(flats))
    keys = list(PARAM_UI) + [k for k in flats[0] if k not in BASIC_KEYS and k not in HIDDEN_KEYS]
    rows = []
    for key in keys:
        values = [f.get(key) for f in flats]
        if not show_all and all(v == values[0] for v in values):
            continue
        row = {"Setting": setting_label(key), "Group": setting_group(key)}
        row.update({letter: fmt_value(key, v) for letter, v in zip(letters, values, strict=True)})
        rows.append(row)
    if code_versions is not None and len(set(code_versions)) > 1:
        row = {"Setting": "Code version", "Group": "Run"}
        row.update(dict(zip(letters, code_versions, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows, columns=["Setting", "Group", *letters])


def differing_keys(params_list: Sequence[Params]) -> list[str]:
    flats = [flatten_params(p) for p in params_list]
    return [k for k in flats[0] if k not in HIDDEN_KEYS and len({repr(f[k]) for f in flats}) > 1]


def compare_note(params_list: Sequence[Params], code_versions: Sequence[str]) -> str | None:
    """The one-line explanation under "What's different", if a rule applies."""
    keys = differing_keys(params_list)
    same_code = len(set(code_versions)) <= 1
    if not keys and same_code:
        return (
            "These runs have identical settings, so their results are identical: "
            "the simulator is deterministic."
        )
    if keys and all(k.startswith("assignment.") for k in keys):
        return (
            "Same requests arrive in all these runs, so differences come from how work is assigned."
        )
    return None


@dataclass(frozen=True)
class MetricUI:
    key: str
    label: str
    better: str  # "higher" | "lower" | "—"
    fmt: Callable[[float], str]
    delta: Callable[[float], str]
    help: str = ""


METRIC_UI: tuple[MetricUI, ...] = (
    MetricUI("on_time_rate", "On-time rate", "higher", fmt_pct, fmt_pts,
             "Share of reports delivered within the promised days"),
    MetricUI("meets_target", "Meets target", "—", lambda v: "Yes" if v else "No", lambda v: ""),
    MetricUI("p90_turnaround_days", "Turnaround P90", "lower",
             lambda v: fmt_days(v, short=True), lambda v: fmt_days_delta(v, short=True)),
    MetricUI("mean_turnaround_days", "Average turnaround", "lower",
             lambda v: fmt_days(v, short=True), lambda v: fmt_days_delta(v, short=True)),
    MetricUI("late_reports", "Late reports", "lower", fmt_count,
             lambda v: fmt_count(v, signed=True)),
    MetricUI("n_at_risk_day_one", "At risk from day one", "—", fmt_count,
             lambda v: fmt_count(v, signed=True)),
    MetricUI("cost_total", "Total cost", "lower", fmt_cost_k, lambda v: fmt_cost_k(v, signed=True)),
    MetricUI("cost_salaried", "Salaried cost", "—", fmt_cost_k,
             lambda v: fmt_cost_k(v, signed=True)),
    MetricUI("cost_freelance", "Freelance cost", "—", fmt_cost_k,
             lambda v: fmt_cost_k(v, signed=True)),
    MetricUI("cost_automation", "Pre-screen licence", "—", fmt_cost_k,
             lambda v: fmt_cost_k(v, signed=True)),
    MetricUI("cost_late_penalty", "Late penalties", "lower", fmt_cost_k,
             lambda v: fmt_cost_k(v, signed=True)),
    MetricUI("util_full_time", "Utilisation full-time", "—", fmt_pct, fmt_pts),
    MetricUI("util_freelance", "Utilisation freelance", "—", fmt_pct, fmt_pts),
    MetricUI("n_hires_full_time", "Full-time hires", "—", fmt_count,
             lambda v: fmt_count(v, signed=True)),
    MetricUI("n_hires_freelance", "Freelance hires", "—", fmt_count,
             lambda v: fmt_count(v, signed=True)),
)  # fmt: skip


def _metric_value(summary: Summary, key: str) -> Any:
    if key == "meets_target":
        return summary.meets_target
    if key == "late_reports":
        return summary.late_reports()
    return summary.mean(key)


def metric_table(summaries: Sequence[Summary], baseline: int = 0) -> pd.DataFrame:
    """Metric, Better is, A, B, ...: values as text, non-baseline cells with "(delta vs A)"."""
    letters = slot_letters(len(summaries))
    rows = []
    for spec in METRIC_UI:
        values = [_metric_value(s, spec.key) for s in summaries]
        if all(v is None for v in values):
            continue
        base = values[baseline]
        row: dict[str, str] = {"Metric": spec.label, "Better is": spec.better}
        for i, (letter, value) in enumerate(zip(letters, values, strict=True)):
            if value is None:
                row[letter] = DASH
                continue
            text = spec.fmt(value)
            if i != baseline and base is not None and spec.key != "meets_target":
                text += f" ({spec.delta(value - base)})"
            row[letter] = text
        rows.append(row)
    return pd.DataFrame(rows, columns=["Metric", "Better is", *letters])


# --- status and progress ---------------------------------------------------------------

# display state -> (st.badge colour, material icon, label). Icon + word, never colour alone.
BADGES: dict[str, tuple[str, str, str]] = {
    "queued": ("gray", ":material/schedule:", "Queued"),
    "running": ("blue", ":material/progress_activity:", "Running"),
    "cancelling": ("orange", ":material/hourglass_top:", "Cancelling"),
    "done": ("green", ":material/check_circle:", "Done"),
    "failed": ("red", ":material/error:", "Failed"),
    "cancelled": ("gray", ":material/block:", "Cancelled"),
    "unreadable": ("gray", ":material/help:", "Unreadable"),
}


def badge_spec(state: str) -> tuple[str, str, str]:
    return BADGES.get(state, BADGES["unreadable"])


def outcome_spec(meets_target: bool | None) -> tuple[str, str, str] | None:
    """Outcome is not lifecycle: missing the target is orange (a result), never red (an error)."""
    if meets_target is None:
        return None
    if meets_target:
        return ("green", ":material/verified:", "Meets target")
    return ("orange", ":material/trending_down:", "Misses target")


STAGE_LABELS = {
    "generate": "Generate",
    "forecast": "Forecast",
    "capacity_plan": "Capacity plan",
    "simulate": "Simulate",
}


def stage_line(stage: str | None) -> str:
    """``"Generate › Forecast › **Capacity plan** › Simulate"``."""
    return " › ".join(
        f"**{label}**" if key == stage else label for key, label in STAGE_LABELS.items()
    )


def time_left(progress: float, elapsed_s: float | None) -> str | None:
    """ "about 2 min left", extrapolated from elapsed / progress; only after 10%."""
    if elapsed_s is None or progress < 0.1 or progress >= 1:
        return None
    remaining = elapsed_s / progress * (1 - progress)
    if remaining < 60:
        return "less than a minute left"
    return f"about {round(remaining / 60)} min left"


def run_option_label(name: str, created_at: dt.datetime) -> str:
    """Picker option: ``"Hire for 2x, get 4x · 03 Oct 10:15"``."""
    return f"{name} · {fmt_when(created_at)}"


def queue_sentence(position: int | None, ahead_name: str | None = None) -> str:
    """ "It is #2 in line." (position 1 = next)."""
    if position is None:
        return "It is running now."
    return f"It is #{position} in line."


def badge_markdown(spec: tuple[str, str, str] | None) -> str:
    """Inline badge for ``st.markdown``: ``":green-badge[:material/check_circle: Done]"``.

    Lets two badges (status + outcome) sit on one line, which ``st.badge`` can't.
    """
    if spec is None:
        return ""
    color, icon, label = spec
    return f":{color}-badge[{icon} {label}]"
