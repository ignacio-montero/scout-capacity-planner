"""Chart functions: each returns a figure with the expected traces, colours and title."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

import ui_fixtures
from scout_planner import charts
from scout_planner.config import load_default_params
from scout_planner.results import (
    forecast_totals,
    load_published,
    load_table,
    monthly_demand,
    monthly_on_time,
    normalise_sweep_frame,
)

DEFAULTS = load_default_params()


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    base = tmp_path_factory.mktemp("charts")
    root = base / "runs"
    ids = {}
    ids["defaults"] = ui_fixtures.write_run(root, "Demo: Defaults")
    ids["edf"] = ui_fixtures.write_run(root, "Demo: EDF", {"assignment.policy": "edf"})
    published = ui_fixtures.write_demo_published(base / "published")
    tables = {
        key: {
            name: load_table(root / run_id, name).value
            for name in (
                "weekly",
                "outcomes",
                "raw_requests",
                "forecast",
                "capacity_plan",
                "hiring_plan",
            )
        }
        for key, run_id in ids.items()
    }
    sweep = normalise_sweep_frame(load_published(published).value, DEFAULTS)
    return {"tables": tables, "sweep": sweep}


def colors(fig: go.Figure) -> set[str]:
    return {t.line.color for t in fig.data if getattr(t, "line", None) is not None and t.line.color}


def title(fig: go.Figure) -> str:
    return fig.layout.title.text.split("<br>")[0]


def test_tokens_are_the_okabe_ito_values() -> None:
    assert charts.POLICY_COLORS == {"fcfs": "#CC79A7", "edf": "#009E73", "optimiser": "#0072B2"}
    assert charts.POLICY_ORDER == ("fcfs", "edf", "optimiser")
    assert set(charts.POLICY_SYMBOLS) == set(charts.POLICY_ORDER)
    assert charts.COST_COLORS["cost_late_penalty"] == "#D55E00"
    assert charts.SLOT_DASHES == {"A": "solid", "B": "dash", "C": "dot", "D": "dashdot"}
    assert charts.hex_to_rgba("#0072B2", 0.2) == "rgba(0,114,178,0.2)"


def test_seed_band() -> None:
    df = pd.DataFrame({"seed": [1, 2, 1, 2], "x": [1, 1, 2, 2], "y": [10, 20, 30, 50]})
    band = charts.seed_band(df, "x", "y")
    assert band["mean"].tolist() == [15, 40]
    assert band["min"].tolist() == [10, 30] and band["max"].tolist() == [20, 50]


def test_backlog_single_run(demo) -> None:
    weekly = demo["tables"]["defaults"]["weekly"]
    fig = charts.fig_backlog({"run": weekly}, {"run": "optimiser"})
    means = [t for t in fig.data if t.name == "Optimiser"]
    assert len(means) == 1 and means[0].line.color == "#0072B2" and means[0].line.dash == "solid"
    assert any(t.fill == "toself" for t in fig.data)  # min-max band
    assert any(t.name == "Of which already late" for t in fig.data)
    assert title(fig).startswith(("Backlog peaks", "Backlog stays"))


def test_backlog_overlay_uses_policy_colour_and_slot_dash(demo) -> None:
    weekly = {"A": demo["tables"]["defaults"]["weekly"], "B": demo["tables"]["edf"]["weekly"]}
    fig = charts.fig_backlog(weekly, {"A": "optimiser", "B": "edf"}, show_range=False)
    lines = {t.name: t for t in fig.data}
    assert lines["A"].line.color == "#0072B2" and lines["A"].line.dash == "solid"
    assert lines["B"].line.color == "#009E73" and lines["B"].line.dash == "dash"
    assert not any(t.fill == "toself" for t in fig.data)  # bands off by default on Compare
    assert {a.text for a in fig.layout.annotations} >= {"<b>A</b>", "<b>B</b>"}  # end labels


def test_on_time_by_month_has_target_line(demo) -> None:
    t = demo["tables"]["defaults"]
    monthly = monthly_on_time(t["outcomes"], t["raw_requests"])
    fig = charts.fig_on_time_by_month({"run": monthly}, {"run": "optimiser"}, 0.95)
    assert any(s.y0 == 0.95 for s in fig.layout.shapes)
    assert any(a.text == "95% target" for a in fig.layout.annotations)
    assert fig.layout.yaxis.range[1] == 1.0
    assert title(fig).startswith("On time")


def test_takeaway_on_time_rules() -> None:
    months = pd.to_datetime(["2027-01-01", "2027-02-01", "2027-03-01"])
    ok = pd.DataFrame({"month": months, "mean": [0.97, 0.96, 0.98]})
    assert charts.takeaway_on_time_by_month(ok, 0.95) == "On time stays above 95% every month"
    bad = pd.DataFrame({"month": months, "mean": [0.97, 0.93, 0.98]})
    assert charts.takeaway_on_time_by_month(bad, 0.95) == "On time drops below 95% from Feb 2027"


def test_turnaround_hist(demo) -> None:
    outcomes = demo["tables"]["defaults"]["outcomes"]
    fig = charts.fig_turnaround_hist(outcomes, "edf", promise_days=14, p90=12.0)
    assert isinstance(fig.data[0], go.Histogram) and fig.data[0].marker.color == "#009E73"
    assert any(s.x0 == 14 and s.line.dash == "dot" for s in fig.layout.shapes)
    assert "of reports arrive within 14 days" in title(fig)


def test_cost_split_share_and_absolute() -> None:
    parts = {
        "cost_salaried": 780_000,
        "cost_freelance": 200_000,
        "cost_automation": 0,
        "cost_late_penalty": 260_000,
    }
    fig = charts.fig_cost_split({"Run": parts}, mode="share")
    assert [t.name for t in fig.data] == [
        "Salaried",
        "Freelance",
        "Late penalties",
    ]  # zero part dropped
    assert [t.marker.color for t in fig.data] == ["#E69F00", "#56B4E9", "#D55E00"]
    assert fig.data[0].text[0] == "Salaried 780k (63%)"
    assert title(fig) == "Late penalties are 21% of the cost"
    two = charts.fig_cost_split(
        {"A · x": parts, "B · y": {**parts, "cost_salaried": 900_000}}, mode="absolute"
    )
    assert title(two) == "B costs 120k more, mostly salaried"
    with pytest.raises(ValueError):
        charts.fig_cost_split({"Run": parts}, mode="pie")


def test_team_size_stacks_employment_types(demo) -> None:
    t = demo["tables"]["defaults"]
    fig = charts.fig_team_size(t["weekly"], hiring_plan=t["hiring_plan"])
    stacked = [tr for tr in fig.data if tr.stackgroup == "team"]
    assert [tr.name for tr in stacked] == ["Full-time", "Freelance"]
    assert [tr.line.color for tr in stacked] == ["#E69F00", "#56B4E9"]
    assert title(fig).startswith("The team")


def test_utilisation_bars_with_error_bars() -> None:
    fig = charts.fig_utilisation(
        {"full_time": (0.82, 0.8, 0.85), "freelance": (0.6, 0.55, 0.62)}, target_utilisation=0.8
    )
    bar = fig.data[0]
    assert list(bar.x) == ["Full-time", "Freelance"]
    assert list(bar.marker.color) == ["#E69F00", "#56B4E9"]
    assert bar.error_y.array[0] == pytest.approx(0.03)
    assert title(fig) == "Full-time scouts are 82% busy, freelancers 60%"


def test_demand_vs_forecast(demo) -> None:
    t = demo["tables"]["defaults"]
    fig = charts.fig_demand_vs_forecast(
        monthly_demand(t["raw_requests"]), forecast_totals(t["forecast"]), quantile=0.8
    )
    names = [tr.name for tr in fig.data]
    assert names == ["Actual", "Forecast (P50)", "Planning forecast (P80)"]
    assert fig.data[1].line.dash == "dash" and fig.data[2].line.dash == "dot"
    assert any(a.text == "Plan year starts" for a in fig.layout.annotations)


def test_demand_takeaway_ratio() -> None:
    months = pd.to_datetime(["2027-01-01", "2027-02-01"])
    actual = pd.DataFrame({"month": months, "requests": [300, 300], "period": ["future", "future"]})
    forecast = pd.DataFrame(
        {"month": months, "requests_p50": [100.0, 100.0], "requests_pq": [110.0, 110.0]}
    )
    assert charts.takeaway_demand(actual, forecast) == "Demand came in 3.0x above the plan"
    forecast["requests_p50"] = [300.0, 300.0]
    assert charts.takeaway_demand(actual, forecast) == "The forecast tracked demand within 10%"


def test_capacity_heatmap_is_diverging_and_sorted(demo) -> None:
    plan = demo["tables"]["defaults"]["capacity_plan"]
    fig = charts.fig_capacity_heatmap(plan)
    heat = fig.data[0]
    assert heat.zmid == 0 and heat.zmin == -heat.zmax
    assert heat.colorscale[0][1] == "#0072B2" and heat.colorscale[-1][1] == "#D55E00"
    short = plan.assign(s=plan["gap_hours"].clip(lower=0)).groupby("skill_type")["s"].sum()
    assert heat.y[-1] == short.idxmax()  # worst skill drawn on top
    assert title(fig).startswith(("Shortfalls concentrate", "No skill"))


def test_sweep_scatter_single_growth(demo) -> None:
    sweep = demo["sweep"]
    fig = charts.fig_sweep_scatter(sweep, 0.95, growth=4.0)
    points = [t for t in fig.data if t.name != "Highlighted"]
    assert sum(len(t.x) for t in points) == len(sweep[sweep["actual_growth"] == 4.0])
    for t in points:
        policy = next(p for p, label in charts.POLICY_LABELS.items() if label == t.name)
        assert t.marker.color == charts.POLICY_COLORS[policy]
        assert t.marker.symbol.startswith(charts.POLICY_SYMBOLS[policy])
    assert any(t.marker.symbol.endswith("-open") for t in points)  # hollow = pre-screen off
    assert any(t.name == "Highlighted" and t.marker.size == 18 for t in fig.data)
    assert any(s.y0 == 0.95 for s in fig.layout.shapes)
    assert title(fig).startswith(("Cheapest way to stay on time at 4x", "No setting reaches"))
    assert all(t.error_x.array is None for t in points)  # error bars off by default
    with_range = charts.fig_sweep_scatter(sweep, 0.95, growth=4.0, show_range=True)
    assert all(t.error_y.array is not None for t in with_range.data if t.name != "Highlighted")


def test_sweep_scatter_small_multiples(demo) -> None:
    fig = charts.fig_sweep_scatter(demo["sweep"], 0.95, growth="all")
    assert {a.text for a in fig.layout.annotations} >= {"4x growth", "2x growth", "1x growth"}
    assert fig.layout.xaxis3 is not None


def test_sweep_takeaway_when_nothing_passes() -> None:
    df = pd.DataFrame(
        {"row_id": [0, 1], "policy": ["fcfs", "optimiser"], "automation_enabled": [False, True],
         "on_time_rate_mean": [0.80, 0.90], "cost_total_mean": [1e6, 2e6],
         "meets_target": [False, False]}
    )  # fmt: skip
    assert charts.cheapest_passing(df) is None
    assert (
        charts.takeaway_sweep(df, 0.95, 4.0)
        == "No setting reaches 95% at 4x; best is Optimiser at 90.0%"
    )


def test_readme_mode_is_white_and_sized(demo) -> None:
    fig = charts.fig_sweep_scatter(demo["sweep"], 0.95, growth=4.0, for_readme=True)
    assert fig.layout.paper_bgcolor == "white"
    assert (fig.layout.width, fig.layout.height) == charts.README_SIZE
    assert any(charts.README_FOOTNOTE in (a.text or "") for a in fig.layout.annotations)


def test_charts_module_has_no_streamlit_import() -> None:
    source = Path(charts.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source
