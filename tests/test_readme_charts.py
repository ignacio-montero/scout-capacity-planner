"""README charts (M6): selection logic, computed titles, figure structure, the script shell.

Synthetic published frames with hand-picked numbers, so every expected
winner and title is known in advance. PNG export is marked ``slow`` and
skipped when no Chrome is available.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from scout_planner import charts
from scout_planner import readme_charts as rc
from scout_planner.config import load_default_params

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_readme_charts as script  # noqa: E402

DEFAULTS = load_default_params()
TARGET = 0.95


def row(on_time: float, cost_k: float, spread: float = 0.01, **params: object) -> dict:
    """One published row: dotted parameters + flattened metrics (G5 shape)."""
    cost = cost_k * 1000
    late = max(0.0, (0.97 - on_time)) * 4_000_000
    return {
        "name": " · ".join(str(v) for v in params.values()),
        "run_id": None,
        **params,
        "sim.seeds": 3,
        "sim.target_on_time": TARGET,
        "on_time_rate_mean": on_time,
        "on_time_rate_min": on_time - spread,
        "on_time_rate_max": on_time + spread,
        "cost_total_mean": cost,
        "cost_total_min": cost * 0.98,
        "cost_total_max": cost * 1.02,
        "cost_salaried_mean": cost * 0.7,
        "cost_freelance_mean": cost * 0.3 - late,
        "cost_automation_mean": 18_000.0 if params.get("automation.enabled") else 0.0,
        "cost_late_penalty_mean": late,
        "p90_turnaround_days_mean": 12.0,
        "meets_target": on_time >= TARGET,
    }


def headline_frame() -> pd.DataFrame:
    """policy x hire mix x quantile x tool (54 rows) with a known answer:

    * cheapest passing overall: optimiser, rule, P80, tool on (95.7%, 2,871k);
    * freelance_only never passes;
    * full_time_only best: optimiser, P80, tool on (96.1%, 3,196k).
    """
    rows = []
    for policy, mix, q, tool in itertools.product(
        ("fcfs", "edf", "optimiser"),
        ("rule", "freelance_only", "full_time_only"),
        (0.5, 0.8, 0.9),
        (False, True),
    ):
        on_time, cost = 0.90, 3500.0
        if mix == "freelance_only":
            on_time, cost = 0.85, 3400.0
        if (policy, mix, q, tool) == ("optimiser", "rule", 0.8, True):
            on_time, cost = 0.957, 2871.0
        elif (policy, mix, q) == ("edf", "rule", 0.8) and not tool:
            on_time, cost = 0.951, 3266.0
        elif (policy, mix, q, tool) == ("optimiser", "full_time_only", 0.8, True):
            on_time, cost = 0.961, 3196.0
        elif mix == "rule" and q == 0.9 and policy == "optimiser" and tool:
            on_time, cost = 0.963, 2919.0
        params = {
            "assignment.policy": policy,
            "capacity_plan.hire_mix": mix,
            "capacity_plan.quantile": q,
            "automation.enabled": tool,
            "demand.actual_growth": 4.0,
        }
        rows.append(row(on_time, cost, **params))
    return rc.prepare(pd.DataFrame(rows), DEFAULTS)


def small_frame(var: str, values: tuple, results: dict, **fixed: object) -> pd.DataFrame:
    rows = [
        row(*results[(policy, value)], **{"assignment.policy": policy, var: value, **fixed})
        for policy, value in results
    ]
    return rc.prepare(pd.DataFrame(rows), DEFAULTS)


def growth_frame() -> pd.DataFrame:
    res = {
        ("edf", 1.0): (0.945, 1604), ("optimiser", 1.0): (0.952, 1615),
        ("edf", 2.0): (0.945, 2218), ("optimiser", 2.0): (0.9495, 2241),
        ("edf", 4.0): (0.951, 3266), ("optimiser", 4.0): (0.956, 3299),
    }  # fmt: skip
    return small_frame("demand.actual_growth", (1.0, 2.0, 4.0), res)


def forecast_frame() -> pd.DataFrame:
    res = {
        ("edf", 2.0): (0.53, 5496), ("optimiser", 2.0): (0.52, 5574),
        ("edf", 4.0): (0.951, 3266), ("optimiser", 4.0): (0.956, 3299),
        ("edf", 6.0): (0.966, 3921), ("optimiser", 6.0): (0.967, 3955),
    }  # fmt: skip
    return small_frame(
        "capacity_plan.assumed_growth", (2.0, 4.0, 6.0), res, **{"demand.actual_growth": 4.0}
    )


def baseline_frame() -> pd.DataFrame:
    res = {
        (p, t): (0.24 + 0.08 * (p == "optimiser" and t), 6300 - 400 * t)
        for p in ("fcfs", "edf", "optimiser")
        for t in (False, True)
    }
    return small_frame(
        "automation.enabled",
        (False, True),
        res,
        **{"demand.actual_growth": 4.0, "team.follow_hiring_plan": False},
    )


def cadence_frame() -> pd.DataFrame:
    res = {}
    for p, (daily, weekly) in {
        "fcfs": (0.951, 0.667),
        "edf": (0.951, 0.662),
        "optimiser": (0.955, 0.668),
    }.items():
        res[(p, "daily")] = (daily, 3270)
        res[(p, "weekly")] = (weekly, 5270)
    return small_frame("assignment.cadence", ("daily", "weekly"), res)


def title(fig: go.Figure) -> str:
    return script.title_text(fig)


# --- selection logic --------------------------------------------------------------------


def test_cheapest_passing_overall_and_without_tool() -> None:
    df = headline_frame()
    best = rc.cheapest_passing_row(df)
    assert (best["policy"], best["capacity_plan.hire_mix"], best["capacity_plan.quantile"]) == (
        "optimiser",
        "rule",
        0.8,
    )
    assert bool(best["automation_enabled"]) and best["cost_total_mean"] == 2_871_000
    no_tool = rc.cheapest_passing_row(df[~df["automation_enabled"]])
    assert no_tool["policy"] == "edf" and no_tool["cost_total_mean"] == 3_266_000


def test_cheapest_passing_ignores_cheaper_failing_rows_and_breaks_ties_by_row() -> None:
    df = rc.prepare(
        pd.DataFrame(
            [
                row(0.90, 1000, **{"assignment.policy": "fcfs"}),  # cheaper but fails
                row(0.96, 2000, **{"assignment.policy": "edf"}),
                row(0.97, 2000, **{"assignment.policy": "optimiser"}),  # tie on cost
            ]
        ),
        DEFAULTS,
    )
    assert rc.cheapest_passing_row(df)["policy"] == "edf"


def test_nothing_passes_falls_back_to_best_on_time() -> None:
    df = rc.prepare(
        pd.DataFrame(
            [row(0.90, 1000, **{"assignment.policy": "fcfs"}),
             row(0.93, 2000, **{"assignment.policy": "edf"})]
        ),
        DEFAULTS,
    )  # fmt: skip
    assert rc.cheapest_passing_row(df) is None
    assert rc.best_row(df)["policy"] == "edf"


def test_cheapest_per_mix() -> None:
    winners = rc.cheapest_per_mix(headline_frame())
    assert list(winners) == ["rule", "freelance_only", "full_time_only"]
    assert winners["rule"]["cost_total_mean"] == 2_871_000
    assert not bool(winners["freelance_only"]["meets_target"])  # best on-time shown
    assert winners["full_time_only"]["cost_total_mean"] == 3_196_000


def test_default_plan_uses_config_defaults() -> None:
    plan = rc.default_plan(headline_frame(), DEFAULTS)
    assert len(plan) == 6  # 3 policies x tool on/off
    assert set(plan["capacity_plan.hire_mix"]) == {DEFAULTS.capacity_plan.hire_mix}
    assert set(plan["capacity_plan.quantile"]) == {DEFAULTS.capacity_plan.quantile}


def test_prepare_fills_unpublished_parameters_with_defaults() -> None:
    df = rc.prepare(pd.DataFrame([row(0.9, 1000, **{"assignment.policy": "edf"})]), DEFAULTS)
    assert df["capacity_plan.hire_mix"].iloc[0] == DEFAULTS.capacity_plan.hire_mix
    assert df["assignment.cadence"].iloc[0] == DEFAULTS.assignment.cadence


# --- computed titles --------------------------------------------------------------------


def test_titles_state_the_takeaway() -> None:
    df = headline_frame()
    # the winner's worst repeat (95.7% - 1 pt) fell below 95%: the title says so
    assert title(rc.fig_headline(df, TARGET)) == (
        "Cheapest way to stay on time at 4x: Optimiser, planned mix, P80, pre-screen on, "
        "2,871k a year; the worst of 3 simulated years reached 94.70%"
    )
    assert title(rc.fig_mix(df, TARGET)) == (
        "Planned mix is the cheapest way to meet 95% (2,871k); full-timers only costs 325k "
        "more; freelancers only never reaches it"
    )
    assert title(rc.fig_policies(df, TARGET, DEFAULTS)) == (
        "At the default plan, Optimiser with the pre-screen is the cheapest way to meet 95% "
        "(2,871k); without it, Earliest deadline first (3,266k)"
    )


def test_growth_title_says_narrowly_when_repeats_straddle_the_target() -> None:
    assert title(rc.fig_growth(growth_frame(), TARGET)) == (
        "Earliest deadline first narrowly misses 95% at 1x and 2x; Optimiser narrowly misses "
        "95% at 2x; cost rises 2.0x from 1x to 4x"
    )
    clear_miss = growth_frame()
    clear_miss["on_time_rate_max"] = clear_miss["on_time_rate_mean"]
    assert "Earliest deadline first misses 95% at 1x and 2x" in rc.title_growth(clear_miss, TARGET)


def test_forecast_error_effects_and_title() -> None:
    df = forecast_frame()
    fx = rc.forecast_error_effects(df)
    assert fx["under"] == 2.0 and fx["over"] == 6.0
    assert fx["under_pts"] == pytest.approx(((0.53 + 0.52) - (0.951 + 0.956)) / 2)
    assert fx["over_cost"] == pytest.approx(((3921 + 3955) - (3266 + 3299)) / 2 * 1000)
    assert title(rc.fig_forecast_error(df, TARGET)) == (
        "Hiring for 2x when 4x arrives loses 42.9 pts of on-time and adds 2,252k; hiring for "
        "6x gains 1.3 pts of on-time and adds 656k"
    )


def test_baseline_and_cadence_titles() -> None:
    assert title(rc.fig_baseline(baseline_frame(), TARGET)) == (
        "Without hiring, 4x growth overwhelms the team: the best setting (Optimiser, "
        "pre-screen on) delivers only 32% on time"
    )
    assert title(rc.fig_cadence(cadence_frame(), TARGET)) == (
        "Assigning work daily instead of weekly adds 28.7 pts on time on average "
        "(28.4–28.9 pts across rules)"
    )


def test_cadence_title_when_it_does_not_matter() -> None:
    df = cadence_frame()
    df.loc[df["assignment.cadence"] == "weekly", "on_time_rate_mean"] = (
        df.loc[df["assignment.cadence"] == "daily", "on_time_rate_mean"].to_numpy() - 0.001
    )
    assert rc.title_cadence(df).startswith("Assigning work daily or weekly changes on-time by less")


def test_mix_title_when_no_mix_passes() -> None:
    df = headline_frame()
    df["meets_target"] = False
    assert rc.title_mix(rc.cheapest_per_mix(df), TARGET) == "No hiring mix reaches 95% on time"


# --- figure structure ---------------------------------------------------------------------


def test_headline_figure_facets_by_hire_mix_and_highlights_the_answer() -> None:
    fig = rc.fig_headline(headline_frame(), TARGET)
    panels = [a.text for a in fig.layout.annotations if a.text in charts.HIRE_MIX_LABELS.values()]
    assert panels == ["Planned mix", "Freelancers only", "Full-timers only"]
    highlighted = [t for t in fig.data if t.name == "Highlighted"]
    assert len(highlighted) == 1 and highlighted[0].x[0] == pytest.approx(2871)
    assert highlighted[0].xaxis == "x"  # in the planned-mix panel
    dots = [t for t in fig.data if t.name != "Highlighted"]
    assert sum(len(t.x) for t in dots) == 54
    assert {t.marker.color for t in dots} == {
        charts.POLICY_COLORS[p] for p in ("fcfs", "edf", "optimiser")
    }
    assert len([s for s in fig.layout.shapes if s.y0 == TARGET]) == 3  # target in each panel
    assert (fig.layout.width, fig.layout.height) == charts.README_SIZE
    assert fig.layout.paper_bgcolor == "white"


def test_mix_figure_stacks_cost_parts_with_on_time_in_labels() -> None:
    fig = rc.fig_mix(headline_frame(), TARGET)
    names = [t.name for t in fig.data]
    assert names[:2] == ["Salaried", "Freelance"] and "Late penalties" in names
    labels = list(fig.data[0].y)
    assert len(labels) == 3 and "95.7% on time" in labels[0]
    assert "misses target" in labels[1]
    assert fig.layout.barmode == "stack"


def test_policy_dot_figures_encode_tool_by_fill() -> None:
    fig = rc.fig_policies(headline_frame(), TARGET, DEFAULTS)
    symbols = {t.marker.symbol for t in fig.data}
    assert {"diamond", "diamond-open", "square", "square-open"} <= symbols
    assert all(t.error_y.array is not None for t in fig.data)
    assert fig.layout.scattermode == "group"
    cadence = rc.fig_cadence(cadence_frame(), TARGET)
    assert len(cadence.data) == 12  # 3 policies x 2 cadences x 2 panels


def test_line_figures_one_trace_per_policy_per_panel() -> None:
    fig = rc.fig_growth(growth_frame(), TARGET)
    assert [t.name for t in fig.data] == ["Earliest deadline first", "Optimiser"] * 2
    assert list(fig.data[0].x) == ["1x", "2x", "4x"]
    assert fig.layout.yaxis2.range[0] == 0  # cost growth read from zero


def test_long_titles_are_wrapped_for_the_png() -> None:
    fig = rc.fig_mix(headline_frame(), TARGET)
    head = fig.layout.title.text.split("<br><span")[0]
    assert all(len(line) <= charts.README_TITLE_WRAP for line in head.split("<br>"))


# --- key numbers ----------------------------------------------------------------------------


def test_key_numbers_markdown_sections_and_skips() -> None:
    frames = {"headline": headline_frame(), "growth": None, "cadence": cadence_frame()}
    text = rc.key_numbers_markdown(frames, DEFAULTS)
    assert "Cheapest setting that meets 95%: **Optimiser, planned mix, P80, pre-screen on**" in text
    assert "| Freelancers only |" in text and "best on-time shown" in text
    assert "### Assignment rules at the default plan (planned mix, P80)" in text
    assert "## Assignment cadence" in text
    assert "## Growth" not in text  # missing input: no section


# --- the script shell -------------------------------------------------------------------------


def test_published_path_accepts_slugified_names(tmp_path: Path) -> None:
    (tmp_path / "forecast-error_summary.parquet").write_bytes(b"x")
    assert (
        script.published_path("forecast_error", tmp_path).name == "forecast-error_summary.parquet"
    )
    assert script.published_path("growth", tmp_path) is None


def test_build_figures_skips_missing_inputs() -> None:
    frames = {name: None for name in rc.SWEEPS}
    frames["cadence"] = cadence_frame()
    figures, skipped = script.build_figures(frames, DEFAULTS)
    assert list(figures) == ["cadence.png"]
    assert any("skip headline.png" in m for m in skipped)
    assert len(skipped) == len(script.chart_specs()) - 1


def test_find_browser_respects_the_environment(tmp_path: Path) -> None:
    fake = tmp_path / "chrome"
    fake.write_text("", encoding="utf-8")
    assert script.find_browser({"BROWSER_PATH": str(fake)}) == fake
    assert script.find_browser({"BROWSER_PATH": str(tmp_path / "missing")}) is None


def test_main_without_published_files(tmp_path: Path) -> None:
    assert script.main(["--published", str(tmp_path), "--out", str(tmp_path / "img")]) == 2


def test_main_key_numbers_only(tmp_path: Path) -> None:
    published = tmp_path / "published"
    published.mkdir()
    cadence_frame().drop(columns=["row_id"]).to_parquet(published / "cadence_summary.parquet")
    out = tmp_path / "img"
    assert script.main(["--published", str(published), "--out", str(out), "--no-images"]) == 0
    text = (out / "key_numbers.md").read_text(encoding="utf-8")
    assert "## Assignment cadence" in text and "`cadence.png`: Assigning work daily" in text
    assert not list(out.glob("*.png"))


@pytest.mark.slow
def test_png_export(tmp_path: Path) -> None:
    browser = script.find_browser()
    if browser is None:
        pytest.skip("no Chrome available for kaleido")
    figures = {"cadence.png": rc.fig_cadence(cadence_frame(), TARGET)}
    (path,) = script.export(figures, tmp_path, browser)
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
    assert (width, height) == (script.WIDTH * script.SCALE, script.HEIGHT * script.SCALE)


# --- fourth policy, honesty near the target, fallbacks, late penalty ---------------------


def late_penalty_frame(switch: bool = True) -> pd.DataFrame:
    """penalty x rule. With ``switch``, EDF is cheapest at 250 and the optimiser at 4,000."""
    rows = []
    for penalty in (250.0, 1000.0, 4000.0):
        for policy, base, late_reports in (
            ("edf", 3000, 300),
            ("edf_feasible", 3050 if switch else 3400, 220),
            ("optimiser", 3150 if switch else 3700, 150),
        ):
            late_cost = late_reports * penalty
            r = row(
                0.95,
                base + late_cost / 1000,
                **{"assignment.policy": policy, "cost.late_penalty": penalty},
            )
            r["cost_late_penalty_mean"] = late_cost
            rows.append(r)
    return rc.prepare(pd.DataFrame(rows), DEFAULTS)


def test_four_policies_have_distinct_tokens_and_draw_everywhere() -> None:
    df = late_penalty_frame()
    fig = rc.fig_late_penalty(df, TARGET)
    names = [t.name for t in fig.data]
    assert names[:3] == ["Earliest deadline first", "EDF + due-date check", "Optimiser"]
    olive = [t for t in fig.data if t.name == "EDF + due-date check"]
    assert {t.line.color for t in olive} == {"#999933"}
    assert {t.marker.symbol for t in olive} == {"triangle-up"}
    scatter = charts.fig_sweep_scatter(df.assign(automation_enabled=False), TARGET)
    assert "EDF + due-date check" in {t.name for t in scatter.data}


def test_late_penalty_title_and_cost_excluding_penalty() -> None:
    df = late_penalty_frame()
    assert (
        df["cost_excl_penalty_mean"] == df["cost_total_mean"] - df["cost_late_penalty_mean"]
    ).all()
    assert rc.title_late_penalty(df) == (
        "The cheapest rule depends on the late penalty: Earliest deadline first at 250, "
        "Optimiser at 4,000; Earliest deadline first spends least excluding penalties"
    )
    flat = late_penalty_frame(switch=False)
    assert rc.title_late_penalty(flat).startswith(
        "Earliest deadline first is the cheapest rule at every late penalty (250–4,000)"
    )
    fig = rc.fig_late_penalty(df, TARGET)
    assert list(fig.data[0].x) == ["250", "1,000", "4,000"]
    assert fig.layout.yaxis.range == fig.layout.yaxis2.range  # one scale for both views


def test_key_numbers_include_cost_excluding_penalty() -> None:
    text = rc.key_numbers_markdown(
        {"headline": headline_frame(), "late_penalty": late_penalty_frame()}, DEFAULTS
    )
    assert text.count("Cost excl. late penalty") == 2
    assert "## Late penalty" in text and "The cheapest rule depends on the late penalty" in text


def test_near_target_values_get_two_decimals_and_worst_repeat_is_named() -> None:
    df = headline_frame()
    text = rc.key_numbers_markdown({"headline": df}, DEFAULTS)
    # winner: mean 95.7%, worst repeat 94.70% -> never shown as a rounded "94.7%" pass-looking 95
    assert "the worst of 3 simulated years reached 94.70%" in text
    near = df.copy()
    near.loc[near["on_time_rate_mean"] == 0.957, "on_time_rate_min"] = 0.9497
    best = rc.cheapest_passing_row(near)
    assert (
        charts.worst_repeat_note(best, TARGET) == "; the worst of 3 simulated years reached 94.97%"
    )
    assert charts.worst_repeat_note(best.replace(0.9497, 0.951), TARGET) == ""


def test_fallback_note_in_footnote() -> None:
    df = headline_frame()
    assert rc.fallback_note(df) is None  # no diagnostics columns: no note
    df["fallbacks"] = 0
    assert rc.fallback_note(df) is None
    opt = df["policy"] == "optimiser"
    df.loc[opt & df["automation_enabled"], "fallbacks"] = 12
    assert rc.fallback_note(df) == "optimiser fell back to EDF 108 times (9 of 18 optimiser runs)"
    df["fallback_share_mean"] = 0.0
    df.loc[opt & df["automation_enabled"], "fallback_share_mean"] = 0.032
    note = rc.fallback_note(df)
    assert note == "optimiser fell back to EDF in up to 3.2% of rounds (9 of 18 optimiser runs)"
    fig = rc.fig_headline(df, TARGET)
    assert any(note in (a.text or "") for a in fig.layout.annotations)
