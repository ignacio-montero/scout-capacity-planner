"""Design tokens and chart functions shared by the app and the README script.

Every figure in the project is built here, by a **pure function** that takes
DataFrames and returns a ``plotly.graph_objects.Figure``. Nothing here imports
Streamlit: the app renders these figures with ``st.plotly_chart(fig,
theme="streamlit")`` and the README script (M6) exports the very same output to
PNG. One function, two consumers, so the README can never drift from the app.

Conventions (DESIGN_SYSTEM.md sections 2-3):

* **Design tokens** at the top: every colour, marker and dash is named once
  and used by name. Blue always means "optimiser", orange always "full-time".
* **Redundant encoding**: colour is always paired with a symbol, a dash, a
  fill or a direct text label, so no information depends on colour alone.
* **Action titles**: each figure's title states the takeaway, produced by a
  small ``takeaway_*`` rule; when no rule fires it falls back to a
  descriptive title. The subtitle says what is plotted.
* Uncertainty over repeats: a time series is the **mean** line plus a min-max
  **band**; a point estimate gets min-max **error bars**.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scout_planner.formatting import (
    fmt_cost_k,
    fmt_days,
    fmt_growth,
    fmt_month,
    fmt_pct,
)

# --- design tokens (DESIGN_SYSTEM.md section 2.2) --------------------------------

POLICY_ORDER: tuple[str, ...] = ("fcfs", "edf", "optimiser")  # simplest to smartest
POLICY_COLORS: dict[str, str] = {"fcfs": "#CC79A7", "edf": "#009E73", "optimiser": "#0072B2"}
POLICY_SYMBOLS: dict[str, str] = {"fcfs": "circle", "edf": "square", "optimiser": "diamond"}
POLICY_LABELS: dict[str, str] = {
    "fcfs": "First come, first served",
    "edf": "Earliest deadline first",
    "optimiser": "Optimiser",
}
EMPLOYMENT_COLORS: dict[str, str] = {"full_time": "#E69F00", "freelance": "#56B4E9"}
EMPLOYMENT_LABELS: dict[str, str] = {"full_time": "Full-time", "freelance": "Freelance"}
COST_ORDER: tuple[str, ...] = (
    "cost_salaried",
    "cost_freelance",
    "cost_automation",
    "cost_late_penalty",
)
COST_COLORS: dict[str, str] = {
    "cost_salaried": "#E69F00",
    "cost_freelance": "#56B4E9",
    "cost_automation": "#999999",
    "cost_late_penalty": "#D55E00",  # the only "bad" colour
}
COST_LABELS: dict[str, str] = {
    "cost_salaried": "Salaried",
    "cost_freelance": "Freelance",
    "cost_automation": "Pre-screen licence",
    "cost_late_penalty": "Late penalties",
}
SLOT_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")
SLOT_DASHES: dict[str, str] = {"A": "solid", "B": "dash", "C": "dot", "D": "dashdot"}
NEUTRAL = "#808080"  # readable on both Streamlit backgrounds; never pure black/white
TARGET_LINE: dict[str, Any] = {"color": NEUTRAL, "dash": "dash", "width": 1.5}
PROMISE_LINE: dict[str, Any] = {"color": NEUTRAL, "dash": "dot", "width": 1.5}
BAND_OPACITY = 0.2
NEUTRAL_MID = "#E8E8E8"
GAP_COLORSCALE: list[list[Any]] = [[0.0, "#0072B2"], [0.5, NEUTRAL_MID], [1.0, "#D55E00"]]
FALLBACK_COLOR = NEUTRAL  # a policy name the tokens don't know

TITLE_SIZE = 18
# Wrap widths in characters: titles stay readable in a ~800 px wide app window,
# where a one-line 90-character title would be clipped.
TITLE_WRAP = 58
SUBTITLE_WRAP = 85
SUBTITLE_SIZE = 13
README_SIZE = (1200, 675)
README_FOOTNOTE = "Synthetic data · fictional scouting agency"


def policy_color(policy: str) -> str:
    return POLICY_COLORS.get(policy, FALLBACK_COLOR)


def policy_label(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy)


def hex_to_rgba(color: str, alpha: float) -> str:
    """``"#0072B2", 0.2 -> "rgba(0,114,178,0.2)"`` (plotly fills need an alpha colour)."""
    value = color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def ordered_policies(policies: Sequence[str]) -> list[str]:
    """Policies in the canonical fcfs -> edf -> optimiser order (unknown ones last)."""
    known = [p for p in POLICY_ORDER if p in policies]
    return known + sorted({p for p in policies if p not in POLICY_ORDER})


# --- styling helpers ---------------------------------------------------------------


def style_figure(
    fig: go.Figure,
    title: str,
    subtitle: str | None = None,
    *,
    for_readme: bool = False,
    show_legend: bool | None = None,
    height: int | None = None,
) -> go.Figure:
    """Title (takeaway) + grey subtitle, legend above the plot, margins.

    In the app the Streamlit theme supplies background, grid and font colour.
    ``for_readme=True`` makes a self-contained white figure at 1200x675 with a
    source footnote (a transparent PNG would put dark text on GitHub's dark mode).
    """
    title_lines = wrap_text(title, TITLE_WRAP)
    sub_lines = wrap_text(subtitle, SUBTITLE_WRAP) if subtitle else []
    text = "<br>".join(title_lines)
    span = f"<span style='font-size:{SUBTITLE_SIZE}px;color:{NEUTRAL}'>"
    for line in sub_lines:
        text += f"<br>{span}{line}</span>"
    top = 40 + 24 * len(title_lines) + 18 * len(sub_lines)
    fig.update_layout(
        title={"text": text, "font": {"size": TITLE_SIZE}, "x": 0, "xanchor": "left"},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "xanchor": "left"},
        margin={"t": top, "l": 10, "r": 30, "b": 10},
        hovermode="closest",
    )
    if show_legend is not None:
        fig.update_layout(showlegend=show_legend)
    if height is not None:
        fig.update_layout(height=height)
    if for_readme:
        width, readme_height = README_SIZE
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            width=width,
            height=readme_height,
            margin={"b": 60},
        )
        fig.add_annotation(
            text=README_FOOTNOTE,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.12,
            showarrow=False,
            xanchor="left",
            font={"size": 11, "color": NEUTRAL},
        )
    return fig


def wrap_text(text: str, width: int) -> list[str]:
    """Split a title into lines of at most ``width`` characters (plotly needs ``<br>``)."""
    return textwrap.wrap(text, width=width, break_long_words=False) or [text]


def add_target_line(
    fig: go.Figure, target: float, label: str | None = None, **subplot: Any
) -> go.Figure:
    """The on-time target as a grey dashed line below the data, labelled at the right."""
    fig.add_hline(
        y=target,
        line=TARGET_LINE,
        layer="below",
        annotation_text=label or f"{fmt_pct(target, 0)} target",
        annotation_position="top right",
        annotation_font={"size": 12, "color": NEUTRAL},
        **subplot,
    )
    return fig


def add_range_band(
    fig: go.Figure,
    x: Sequence[Any],
    lo: Sequence[float],
    hi: Sequence[float],
    color: str,
    name: str,
    **subplot: Any,
) -> go.Figure:
    """A filled min-max band (no outline, no hover) behind a mean line."""
    xs = list(x)
    fig.add_trace(
        go.Scatter(
            x=xs + xs[::-1],
            y=list(hi) + list(lo)[::-1],
            fill="toself",
            fillcolor=hex_to_rgba(color, BAND_OPACITY),
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            name=f"{name} range",
        ),
        **subplot,
    )
    return fig


def _end_label(fig: go.Figure, x: Any, y: float, text: str, color: str) -> None:
    """Direct label at a line's right end (instead of a legend)."""
    fig.add_annotation(
        x=x,
        y=y,
        text=f"<b>{text}</b>",
        showarrow=False,
        xanchor="left",
        xshift=6,
        font={"color": color, "size": 13},
    )


def seed_band(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Mean, min and max of ``y`` over repeats (seeds) at each ``x``."""
    if df.empty:
        return pd.DataFrame(columns=[x, "mean", "min", "max", "n"])
    grouped = df.groupby(x)[y].agg(["mean", "min", "max", "count"]).reset_index()
    return grouped.rename(columns={"count": "n"}).sort_values(x).reset_index(drop=True)


def _n_repeats(frames: Mapping[str, pd.DataFrame]) -> int:
    counts = [df["seed"].nunique() for df in frames.values() if "seed" in df.columns]
    return max(counts) if counts else 0


def _repeat_phrase(n: int) -> str:
    return "single repeat, no range" if n <= 1 else f"mean of {n} repeats · band = range"


def _slot_dash(slot: str, index: int) -> str:
    return SLOT_DASHES.get(slot, SLOT_DASHES[SLOT_LETTERS[index % len(SLOT_LETTERS)]])


# --- takeaway rules (DESIGN_SYSTEM.md section 5.3) -----------------------------------


def takeaway_backlog(band: pd.DataFrame) -> str | None:
    if band.empty:
        return None
    first, last = float(band["mean"].iloc[0]), float(band["mean"].iloc[-1])
    if last <= 1.2 * max(first, 1.0):
        return "Backlog stays under control all year"
    peak_idx = band["mean"].idxmax()
    peak = band.loc[peak_idx]
    return (
        f"Backlog peaks at {round(peak['mean']):,} open requests in "
        f"{fmt_month(peak.iloc[0], with_year=False)}"
    )


def takeaway_on_time_share(on_time: pd.Series) -> str | None:
    """With two promises (standard and urgent) "within N days" is ambiguous; count
    deliveries by each request's own promised date instead."""
    if on_time.empty:
        return None
    return f"{fmt_pct(float(on_time.mean()))} of reports arrive by their promised date"


def takeaway_on_time_by_month(band: pd.DataFrame, target: float) -> str | None:
    if band.empty:
        return None
    failing = band[band["mean"] < target]
    if failing.empty:
        return f"On time stays above {fmt_pct(target, 0)} every month"
    return f"On time drops below {fmt_pct(target, 0)} from {fmt_month(failing.iloc[0, 0])}"


def takeaway_turnaround(turnaround: pd.Series, promise_days: float) -> str | None:
    """Share delivered within the promise. Unfinished requests (NaN turnaround) stay
    in the denominator as "not within", so the figure agrees with the on-time rate."""
    if turnaround.empty or turnaround.isna().all():
        return None
    share = float((turnaround <= promise_days).sum() / len(turnaround))
    return f"{fmt_pct(share)} of reports arrive within {promise_days:g} days"


def takeaway_team(band_total: pd.DataFrame) -> str | None:
    if band_total.empty:
        return None
    first = round(float(band_total["mean"].iloc[0]))
    last = round(float(band_total["mean"].iloc[-1]))
    if last > first:
        return f"The team grows from {first} to {last} scouts"
    return f"The team stays at {first} scouts"


def takeaway_utilisation(full_time: float | None, freelance: float | None) -> str | None:
    if full_time is None or freelance is None:
        return None
    return f"Full-time scouts are {fmt_pct(full_time, 0)} busy, freelancers {fmt_pct(freelance, 0)}"


def takeaway_demand(actual: pd.DataFrame, forecast: pd.DataFrame) -> str | None:
    future = actual[actual["period"] != "history"]
    joined = future.merge(forecast, on="month", how="inner")
    if joined.empty or joined["requests_p50"].sum() <= 0:
        return None
    ratio = float(joined["requests"].sum() / joined["requests_p50"].sum())
    if ratio > 1.1:
        return f"Demand came in {ratio:.1f}x above the plan"
    if ratio < 0.9:
        return f"Demand came in at {ratio:.1f}x of the plan, below it"
    return "The forecast tracked demand within 10%"


def takeaway_capacity(plan: pd.DataFrame) -> str | None:
    if plan.empty:
        return None
    short = plan[plan["gap_hours"] > 0]
    if short.empty:
        return "No skill runs short of capacity"
    by_skill = short.groupby("skill_type")["gap_hours"].sum().sort_values(ascending=False)
    top = str(by_skill.index[0])
    first = short[short["skill_type"] == top]["month"].min()
    return f"Before hiring, shortfalls concentrate in {top} from {fmt_month(first)}"


def takeaway_cost_split(parts: Mapping[str, float]) -> str | None:
    total = sum(parts.values())
    if total <= 0:
        return None
    late = parts.get("cost_late_penalty", 0.0) / total
    if late >= 0.05:
        return f"Late penalties are {fmt_pct(late, 0)} of the cost"
    biggest = max(parts, key=lambda k: parts[k])
    return f"{COST_LABELS[biggest]} cost is {fmt_pct(parts[biggest] / total, 0)} of the total"


def cheapest_passing(df: pd.DataFrame, target: float | pd.Series | None = None) -> int | None:
    """``row_id`` of the cheapest row that meets the target (None if no row does)."""
    if df.empty:
        return None
    passing = df[df["meets_target"]] if target is None else df[df["on_time_rate_mean"] >= target]
    if passing.empty:
        return None
    return int(passing.sort_values(["cost_total_mean", "row_id"]).iloc[0]["row_id"])


def best_on_time(df: pd.DataFrame) -> int | None:
    if df.empty:
        return None
    return int(
        df.sort_values(["on_time_rate_mean", "row_id"], ascending=[False, True]).iloc[0]["row_id"]
    )


def describe_sweep_row(row: Mapping[str, Any]) -> str:
    """``"Optimiser, pre-screen on"``: the short identity used in titles and annotations."""
    screen = "on" if row.get("automation_enabled") else "off"
    return f"{policy_label(str(row.get('policy')))}, pre-screen {screen}"


def takeaway_sweep(df: pd.DataFrame, target: float, growth: float | None) -> str:
    at = f" at {fmt_growth(growth)}" if growth is not None else ""
    best = cheapest_passing(df)
    if best is not None:
        row = df[df["row_id"] == best].iloc[0]
        return (
            f"Cheapest way to stay on time{at}: {describe_sweep_row(row)}, "
            f"{fmt_cost_k(row['cost_total_mean'])} a year"
        )
    top = best_on_time(df)
    if top is None:
        return "Sweep results"
    row = df[df["row_id"] == top].iloc[0]
    return (
        f"No setting reaches {fmt_pct(target, 0)}{at}; best is "
        f"{policy_label(str(row['policy']))} at {fmt_pct(row['on_time_rate_mean'])}"
    )


# --- time series: backlog, on-time by month ----------------------------------------


def _overlay(
    frames: Mapping[str, pd.DataFrame],
    policies: Mapping[str, str],
    x: str,
    y: str,
    *,
    show_range: bool,
    hover_fmt: str,
    markers: bool = False,
) -> tuple[go.Figure, dict[str, pd.DataFrame]]:
    """Mean line (+ band) per slot. One slot: solid line in the policy colour.
    Several: colour = policy, dash = slot, direct slot label at each line's end."""
    fig = go.Figure()
    bands: dict[str, pd.DataFrame] = {}
    single = len(frames) == 1
    for index, (slot, df) in enumerate(frames.items()):
        band = seed_band(df, x, y)
        bands[slot] = band
        if band.empty:
            continue
        color = policy_color(policies.get(slot, ""))
        dash = "solid" if single else _slot_dash(slot, index)
        if show_range and (band["min"] != band["max"]).any():
            add_range_band(fig, band[x], band["min"], band["max"], color, slot)
        name = policy_label(policies.get(slot, "")) if single else slot
        fig.add_trace(
            go.Scatter(
                x=band[x],
                y=band["mean"],
                mode="lines+markers" if markers else "lines",
                name=name,
                line={"color": color, "dash": dash, "width": 2.5},
                marker={"symbol": POLICY_SYMBOLS.get(policies.get(slot, ""), "circle"), "size": 7},
                customdata=band[["min", "max"]].to_numpy(),
                hovertemplate=(f"{name}<br>%{{x|%d %b %Y}}<br>{hover_fmt}<extra></extra>"),
            )
        )
        if not single:
            _end_label(fig, band[x].iloc[-1], float(band["mean"].iloc[-1]), slot, color)
    return fig, bands


def fig_backlog(
    weekly: Mapping[str, pd.DataFrame],
    policies: Mapping[str, str],
    *,
    show_range: bool = True,
    for_readme: bool = False,
) -> go.Figure:
    """Open requests at each week start: one run (mean + band + late line) or an overlay.

    ``weekly`` maps a slot (``"A"``, or any key for a single run) to that run's
    ``weekly.parquet``; ``policies`` maps the same keys to the run's policy.
    """
    hover = "Open requests %{y:,.0f} (range %{customdata[0]:,.0f}–%{customdata[1]:,.0f})"
    fig, bands = _overlay(
        weekly, policies, "week_start", "open_requests", show_range=show_range, hover_fmt=hover
    )
    n = _n_repeats(weekly)
    if len(weekly) == 1:
        slot, df = next(iter(weekly.items()))
        late = seed_band(df, "week_start", "late_requests")
        if not late.empty:
            fig.add_trace(
                go.Scatter(
                    x=late["week_start"],
                    y=late["mean"],
                    mode="lines",
                    name="Of which already late",
                    line={"color": NEUTRAL, "width": 1},
                    hovertemplate="Already late %{y:,.0f}<extra></extra>",
                )
            )
        title = takeaway_backlog(bands[slot]) or "Backlog over time"
        legend = True
    else:
        title = takeaway_backlog_compare(bands) or "Backlog over time"
        legend = False
    fig.update_yaxes(title_text="Open requests", rangemode="tozero", tickformat=",d")
    fig.update_xaxes(title_text="Week", tickformat="%b %Y")
    return style_figure(
        fig,
        title,
        "Open requests at each week start · "
        + (_repeat_phrase(n) if show_range else f"mean of {n} repeats"),
        for_readme=for_readme,
        show_legend=legend,
    )


def takeaway_backlog_compare(bands: Mapping[str, pd.DataFrame]) -> str | None:
    if len(bands) != 2 or any(b.empty for b in bands.values()):
        return None
    (a, band_a), (b, band_b) = bands.items()
    peak_a, peak_b = float(band_a["mean"].max()), float(band_b["mean"].max())
    if peak_a <= 0:
        return None
    change = (peak_b - peak_a) / peak_a
    if abs(change) < 0.05:
        return f"{a} and {b} peak at about the same backlog"
    direction = "lower" if change < 0 else "higher"
    return f"{b}'s backlog peaks {abs(change):.0%} {direction} than {a}'s"


def fig_on_time_by_month(
    monthly: Mapping[str, pd.DataFrame],
    policies: Mapping[str, str],
    targets: float | Sequence[float],
    *,
    show_range: bool = True,
    for_readme: bool = False,
) -> go.Figure:
    """On-time rate by due month (``results.monthly_on_time`` per slot), with the target."""
    hover = "On time %{y:.1%} (range %{customdata[0]:.1%}–%{customdata[1]:.1%})"
    fig, bands = _overlay(
        monthly,
        policies,
        "month",
        "on_time_rate",
        show_range=show_range,
        hover_fmt=hover,
        markers=True,
    )
    target_list = sorted({float(targets)} if isinstance(targets, int | float) else set(targets))
    for target in target_list:
        add_target_line(fig, target)
    lows = [float(b["min" if show_range else "mean"].min()) for b in bands.values() if not b.empty]
    low = min([0.6, *[v - 0.02 for v in lows]]) if lows else 0.6
    fig.update_yaxes(title_text="On-time rate", tickformat=".0%", range=[max(0.0, low), 1.0])
    fig.update_xaxes(title_text="Month the report was due", tickformat="%b %Y")
    primary = target_list[0] if target_list else 0.95
    if len(monthly) == 1:
        band = next(iter(bands.values()))
        title = takeaway_on_time_by_month(band, primary) or "On-time rate by month"
    else:
        title = takeaway_on_time_compare(bands, primary) or "On-time rate by month"
    n = _n_repeats(monthly)
    return style_figure(
        fig,
        title,
        f"Share of reports delivered on time, by due month · {_repeat_phrase(n)}",
        for_readme=for_readme,
        show_legend=len(monthly) == 1,
    )


def takeaway_on_time_compare(bands: Mapping[str, pd.DataFrame], target: float) -> str | None:
    parts = []
    for slot, band in bands.items():
        if band.empty:
            return None
        failing = band[band["mean"] < target]
        if failing.empty:
            parts.append(f"{slot} never does" if parts else f"{slot} never drops below target")
        else:
            month = fmt_month(failing.iloc[0, 0], with_year=False)
            parts.append(f"{slot} drops below {fmt_pct(target, 0)} from {month}")
    return "; ".join(parts)


# --- distributions and splits ------------------------------------------------------


def fig_turnaround_hist(
    outcomes: pd.DataFrame,
    policy: str,
    *,
    promise_days: float = 14,
    p90: float | None = None,
    urgent_days: float | None = None,
    for_readme: bool = False,
) -> go.Figure:
    """Turnaround of every delivered, scored report (all repeats pooled), 1-day bins.

    Censored requests (due after the year ends) are left out, like in the
    metrics; unfinished ones have no turnaround and drop out by themselves.
    ``urgent_days`` (when some requests carry the shorter urgent promise) adds
    a second promise line, and the title then counts "by the promised date".
    """
    if "scored" in outcomes.columns:
        outcomes = outcomes[outcomes["scored"].astype(bool)]
    all_values = outcomes["turnaround_days"]
    values = all_values.dropna()
    color = policy_color(policy)
    fig = go.Figure(
        go.Histogram(
            x=values,
            xbins={"size": 1},
            marker={"color": color},
            name="Reports",
            hovertemplate="%{x} days: %{y:,} reports<extra></extra>",
        )
    )
    fig.add_vline(
        x=promise_days,
        line=PROMISE_LINE,
        annotation_text=f"{promise_days:g}-day promise",
        annotation_position="top right",
        annotation_font={"size": 12, "color": NEUTRAL},
    )
    if urgent_days is not None and urgent_days != promise_days:
        fig.add_vline(
            x=urgent_days,
            line=PROMISE_LINE,
            annotation_text=f"{urgent_days:g}-day urgent",
            annotation_position="bottom right",
            annotation_font={"size": 12, "color": NEUTRAL},
        )
    if p90 is not None and not math.isnan(p90):
        fig.add_vline(
            x=p90,
            line={"color": color, "dash": "dash", "width": 1.5},
            annotation_text=f"P90 {fmt_days(p90, short=True)}",
            annotation_position="top left",
            annotation_font={"size": 12, "color": color},
        )
    fig.update_xaxes(title_text="Turnaround (days)")
    fig.update_yaxes(title_text="Reports", rangemode="tozero")
    n = int(outcomes["seed"].nunique()) if "seed" in outcomes.columns else 1
    return style_figure(
        fig,
        (
            takeaway_on_time_share(outcomes["on_time"])
            if urgent_days is not None and "on_time" in outcomes.columns
            else takeaway_turnaround(all_values, promise_days)
        )
        or "Turnaround",
        f"Days from request to delivered report · all {n} repeats pooled",
        for_readme=for_readme,
        show_legend=False,
    )


def fig_cost_split(
    parts_by_run: Mapping[str, Mapping[str, float]],
    *,
    mode: str = "share",
    for_readme: bool = False,
) -> go.Figure:
    """Horizontal stacked bar(s) of the cost components.

    ``mode="share"``: one run, a 100% bar with direct labels ("Salaried 780k
    (63%)"), labels hidden under 8%. ``mode="absolute"``: one bar per run in
    thousands (Compare). Bars, not pies: lengths compare more accurately than
    angles, and the same style stacks for several runs.
    """
    if mode not in ("share", "absolute"):
        raise ValueError(f"mode must be 'share' or 'absolute', got {mode!r}")
    runs_ = list(parts_by_run)
    fig = go.Figure()
    for component in COST_ORDER:
        values = [float(parts_by_run[r].get(component, 0.0)) for r in runs_]
        if not any(values):
            continue
        totals = [sum(parts_by_run[r].values()) or 1.0 for r in runs_]
        shares = [v / t for v, t in zip(values, totals, strict=True)]
        label = COST_LABELS[component]
        texts = [
            f"{label} {fmt_cost_k(v)} ({fmt_pct(s, 0)})" if s >= 0.08 else ""
            for v, s in zip(values, shares, strict=True)
        ]
        x = shares if mode == "share" else [v / 1000 for v in values]
        fig.add_trace(
            go.Bar(
                y=runs_,
                x=x,
                orientation="h",
                name=label,
                marker={"color": COST_COLORS[component]},
                text=texts,
                textposition="inside",
                insidetextanchor="middle",
                customdata=[
                    [fmt_cost_k(v), fmt_pct(s)] for v, s in zip(values, shares, strict=True)
                ],
                hovertemplate=f"{label}: %{{customdata[0]}} (%{{customdata[1]}})<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack", bargap=0.35)
    if mode == "share":
        part = parts_by_run[runs_[0]] if runs_ else {}
        title = takeaway_cost_split(part) or "Cost split"
        fig.update_xaxes(visible=False, range=[0, 1])
        fig.update_yaxes(visible=False)
        return style_figure(
            fig,
            title,
            "Share of the year's total cost · mean over repeats",
            for_readme=for_readme,
            show_legend=False,
            height=170,
        )
    title = takeaway_cost_compare(parts_by_run) or "Cost for the year by component"
    fig.update_xaxes(title_text="Total cost for the year (k)", ticksuffix="k", tickformat=",.0f")
    fig.update_yaxes(autorange="reversed")
    return style_figure(
        fig,
        title,
        "Mean over repeats",
        for_readme=for_readme,
        show_legend=True,
        height=140 + 60 * len(runs_),
    )


def takeaway_cost_compare(parts_by_run: Mapping[str, Mapping[str, float]]) -> str | None:
    if len(parts_by_run) != 2:
        return None
    (a, pa), (b, pb) = parts_by_run.items()
    a_slot, b_slot = a.split(" ")[0], b.split(" ")[0]
    delta = sum(pb.values()) - sum(pa.values())
    if abs(delta) < 1000:
        return f"{a_slot} and {b_slot} cost about the same"
    deltas = {c: pb.get(c, 0.0) - pa.get(c, 0.0) for c in COST_ORDER}
    sign = 1 if delta > 0 else -1
    driver = max(deltas, key=lambda c: sign * deltas[c])
    more = "more" if delta > 0 else "less"
    return f"{b_slot} costs {fmt_cost_k(abs(delta))} {more}, mostly {COST_LABELS[driver].lower()}"


# --- team ------------------------------------------------------------------------


def fig_team_size(
    weekly: pd.DataFrame,
    *,
    hiring_plan: pd.DataFrame | None = None,
    for_readme: bool = False,
) -> go.Figure:
    """Scouts on the team each week, stacked by employment type (mean over repeats)."""
    fig = go.Figure()
    ft = seed_band(weekly, "week_start", "team_full_time")
    fl = seed_band(weekly, "week_start", "team_freelance")
    for band, kind in ((ft, "full_time"), (fl, "freelance")):
        if band.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=band["week_start"],
                y=band["mean"],
                name=EMPLOYMENT_LABELS[kind],
                mode="lines",
                stackgroup="team",
                line={"color": EMPLOYMENT_COLORS[kind], "width": 1, "shape": "hv"},
                fillcolor=hex_to_rgba(EMPLOYMENT_COLORS[kind], 0.6),
                hovertemplate=f"{EMPLOYMENT_LABELS[kind]} %{{y:,.0f}}<extra></extra>",
            )
        )
    total = ft.copy()
    if not ft.empty and not fl.empty:
        total["mean"] = ft["mean"].to_numpy() + fl["mean"].to_numpy()
    if hiring_plan is not None and not hiring_plan.empty and not total.empty:
        joins = (
            hiring_plan.groupby(["joins_month", "hire_type"], as_index=False)
            .agg(count=("count", "sum"), acted=("month_to_act", "min"))
            .sort_values("joins_month")
        )
        xs, ys, texts = [], [], []
        for month, group in joins.groupby("joins_month"):
            later = total[total["week_start"] >= month]
            if later.empty:
                continue
            xs.append(later["week_start"].iloc[0])
            ys.append(float(later["mean"].iloc[0]))
            texts.append(
                "<br>".join(
                    f"{int(r['count'])} "
                    f"{EMPLOYMENT_LABELS.get(r['hire_type'], r['hire_type']).lower()}"
                    f" joined (planned in {fmt_month(r['acted'], with_year=False)})"
                    for _, r in group.iterrows()
                )
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name="Hires join",
                marker={"symbol": "triangle-up", "size": 9, "color": NEUTRAL},
                text=texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Scouts", rangemode="tozero")
    fig.update_xaxes(title_text="Week", tickformat="%b %Y")
    return style_figure(
        fig,
        takeaway_team(total) or "Team size over time",
        "Scouts on the team at each week start · mean over repeats",
        for_readme=for_readme,
    )


def fig_utilisation(
    stats: Mapping[str, tuple[float, float, float] | None],
    *,
    target_utilisation: float | None = None,
    for_readme: bool = False,
) -> go.Figure:
    """Utilisation per employment type: bars with min-max error bars, 0-100%.

    ``stats`` maps ``"full_time"`` / ``"freelance"`` to ``(mean, min, max)``.
    """
    kinds = [k for k in ("full_time", "freelance") if stats.get(k) is not None]
    means = [stats[k][0] for k in kinds]  # type: ignore[index]
    fig = go.Figure(
        go.Bar(
            x=[EMPLOYMENT_LABELS[k] for k in kinds],
            y=means,
            marker={"color": [EMPLOYMENT_COLORS[k] for k in kinds]},
            text=[fmt_pct(m) for m in means],
            textposition="outside",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [stats[k][2] - stats[k][0] for k in kinds],  # type: ignore[index]
                "arrayminus": [stats[k][0] - stats[k][1] for k in kinds],  # type: ignore[index]
                "color": NEUTRAL,
                "thickness": 1.2,
            },
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    if target_utilisation is not None:
        fig.add_hline(
            y=target_utilisation,
            line={"color": NEUTRAL, "dash": "dash", "width": 1.5},
            layer="below",
            annotation_text=f"Target workload {fmt_pct(target_utilisation, 0)}",
            annotation_position="top right",
            annotation_font={"size": 12, "color": NEUTRAL},
        )
    fig.update_yaxes(title_text="Utilisation", tickformat=".0%", range=[0, 1.05])
    ft = stats.get("full_time")
    fl = stats.get("freelance")
    title = takeaway_utilisation(ft[0] if ft else None, fl[0] if fl else None)
    return style_figure(
        fig,
        title or "Utilisation by employment type",
        "Share of available hours spent on work · bars = mean, whiskers = range over repeats",
        for_readme=for_readme,
        show_legend=False,
    )


# --- demand and capacity -----------------------------------------------------------


def fig_demand_vs_forecast(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    policy: str = "optimiser",
    quantile: float = 0.8,
    for_readme: bool = False,
) -> go.Figure:
    """Requests per month: actual (last history year + plan year) vs the forecast.

    ``actual`` is ``results.monthly_demand``; ``forecast`` is
    ``results.forecast_totals`` (summed over skills).
    """
    fig = go.Figure()
    color = policy_color(policy)
    if {"requests_min", "requests_max"} <= set(actual.columns) and (
        actual["requests_min"] != actual["requests_max"]
    ).any():
        future = actual[actual["period"] != "history"]
        add_range_band(
            fig, future["month"], future["requests_min"], future["requests_max"], color, "Actual"
        )
    fig.add_trace(
        go.Scatter(
            x=actual["month"],
            y=actual["requests"],
            mode="lines+markers",
            name="Actual",
            line={"color": color, "width": 2.5},
            marker={"symbol": POLICY_SYMBOLS.get(policy, "circle"), "size": 6},
            hovertemplate="Actual %{y:,.0f} requests<br>%{x|%b %Y}<extra></extra>",
        )
    )
    q_label = f"P{round(quantile * 100)}"
    if not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast["month"],
                y=forecast["requests_p50"],
                mode="lines",
                name="Forecast (P50)",
                line={"color": NEUTRAL, "dash": "dash", "width": 2},
                hovertemplate="Forecast P50 %{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["month"],
                y=forecast["requests_pq"],
                mode="lines",
                name=f"Planning forecast ({q_label})",
                line={"color": NEUTRAL, "dash": "dot", "width": 2},
                hovertemplate=f"Planning forecast {q_label} %{{y:,.0f}}<extra></extra>",
            )
        )
    future = actual[actual["period"] != "history"]
    if not future.empty and (actual["period"] == "history").any():
        start = future["month"].min()
        fig.add_vline(
            x=start,
            line={"color": NEUTRAL, "dash": "dot", "width": 1},
            annotation_text="Plan year starts",
            annotation_position="top left",
            annotation_font={"size": 12, "color": NEUTRAL},
        )
    fig.update_yaxes(title_text="Requests per month", rangemode="tozero", tickformat=",d")
    fig.update_xaxes(title_text="Month", tickformat="%b %Y")
    return style_figure(
        fig,
        takeaway_demand(actual, forecast) or "Demand: actual vs forecast",
        "Requests received per month · last history year, then the plan year"
        " (mean of repeats; band = range)",
        for_readme=for_readme,
    )


def fig_capacity_heatmap(plan: pd.DataFrame, *, for_readme: bool = False) -> go.Figure:
    """Capacity gap per skill type and month, diverging around zero.

    Blue = spare hours, vermillion = shortfall (stays distinct under red-green
    colour blindness). Skills sorted by total shortfall, worst on top.
    """
    if plan.empty:
        fig = go.Figure()
        return style_figure(fig, "No capacity plan", None, for_readme=for_readme)
    shortfall = plan.assign(short=plan["gap_hours"].clip(lower=0))
    order = (
        shortfall.groupby("skill_type")["short"].sum().sort_values(ascending=True).index.tolist()
    )  # ascending: plotly draws the first row at the bottom, so the worst ends on top
    months = sorted(plan["month"].unique())

    def grid(column: str) -> pd.DataFrame:
        return plan.pivot_table(index="skill_type", columns="month", values=column).reindex(
            index=order, columns=months
        )

    gap, req, avail = grid("gap_hours"), grid("required_hours"), grid("available_hours")
    limit = float(abs(plan["gap_hours"]).max()) or 1.0
    # "after planned hires" (an M4 extra column) goes in the hover when present.
    after = (
        grid("gap_after_hires_hours")
        if "gap_after_hires_hours" in plan.columns
        else gap * float("nan")
    )
    custom = [
        [[r, a, g] for r, a, g in zip(req_row, avail_row, after_row, strict=True)]
        for req_row, avail_row, after_row in zip(
            req.to_numpy(), avail.to_numpy(), after.to_numpy(), strict=True
        )
    ]
    after_line = (
        "<br>After planned hires %{customdata[2]:+,.0f} h"
        if "gap_after_hires_hours" in plan.columns
        else ""
    )
    fig = go.Figure(
        go.Heatmap(
            z=gap.to_numpy(),
            x=[pd.Timestamp(m) for m in months],
            y=order,
            colorscale=GAP_COLORSCALE,
            zmid=0,
            zmin=-limit,
            zmax=limit,
            customdata=custom,
            colorbar={"title": {"text": "Hours short (+) / spare (−)"}},
            hovertemplate=(
                "%{y} · %{x|%b %Y}<br>Required %{customdata[0]:,.0f} h"
                "<br>Available %{customdata[1]:,.0f} h<br>Gap %{z:+,.0f} h"
                + after_line
                + "<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text="Month", tickformat="%b %Y")
    fig.update_yaxes(title_text="Skill type")
    return style_figure(
        fig,
        takeaway_capacity(plan) or "Capacity gap",
        "Hours of work needed minus hours the starting team can give, per skill and month"
        " (before planned hires)",
        for_readme=for_readme,
        height=140 + 26 * len(order),
    )


# --- the headline: sweep scatter ---------------------------------------------------


def _sweep_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    show_range: bool,
    legend: bool,
    **subplot: Any,
) -> None:
    shown: set[str] = set()
    for policy in ordered_policies(df["policy"].unique().tolist()):
        for enabled in (False, True):
            part = df[(df["policy"] == policy) & (df["automation_enabled"] == enabled)]
            if part.empty:
                continue
            symbol = POLICY_SYMBOLS.get(policy, "circle") + ("" if enabled else "-open")
            color = policy_color(policy)
            extra: dict[str, Any] = {}
            if show_range:
                extra["error_x"] = {
                    "type": "data",
                    "symmetric": False,
                    "array": (part["cost_total_max"] - part["cost_total_mean"]) / 1000,
                    "arrayminus": (part["cost_total_mean"] - part["cost_total_min"]) / 1000,
                    "color": color,
                    "thickness": 1,
                    "width": 0,
                }
                extra["error_y"] = {
                    "type": "data",
                    "symmetric": False,
                    "array": part["on_time_rate_max"] - part["on_time_rate_mean"],
                    "arrayminus": part["on_time_rate_mean"] - part["on_time_rate_min"],
                    "color": color,
                    "thickness": 1,
                    "width": 0,
                }
            hover = [
                f"<b>{r['name']}</b><br>{describe_sweep_row(r)} · "
                f"{fmt_growth(r['actual_growth'])} growth"
                f"<br>On time {fmt_pct(r['on_time_rate_mean'])} (range "
                f"{fmt_pct(r['on_time_rate_min'])}–{fmt_pct(r['on_time_rate_max'])})"
                f"<br>Total cost {fmt_cost_k(r['cost_total_mean'])} (range "
                f"{fmt_cost_k(r['cost_total_min'])}–{fmt_cost_k(r['cost_total_max'])})"
                + (
                    f"<br>Turnaround P90 {fmt_days(r['p90_turnaround_days_mean'], short=True)}"
                    if "p90_turnaround_days_mean" in r
                    else ""
                )
                for _, r in part.iterrows()
            ]
            fig.add_trace(
                go.Scatter(
                    x=part["cost_total_mean"] / 1000,
                    y=part["on_time_rate_mean"],
                    mode="markers",
                    name=policy_label(policy),
                    legendgroup=policy,
                    showlegend=legend and policy not in shown,
                    marker={
                        "symbol": symbol,
                        "size": 10,
                        "color": color,
                        "line": {"color": color, "width": 2},
                    },
                    customdata=part["row_id"].to_numpy(),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                    **extra,
                ),
                **subplot,
            )
            shown.add(policy)


def _highlight(fig: go.Figure, point: pd.Series, text: str, **subplot: Any) -> None:
    color = policy_color(str(point["policy"]))
    symbol = POLICY_SYMBOLS.get(str(point["policy"]), "circle") + (
        "" if point["automation_enabled"] else "-open"
    )
    fig.add_trace(
        go.Scatter(
            x=[point["cost_total_mean"] / 1000],
            y=[point["on_time_rate_mean"]],
            mode="markers",
            name="Highlighted",
            showlegend=False,
            hoverinfo="skip",
            marker={
                "symbol": symbol,
                "size": 18,
                "color": color,
                "line": {"color": NEUTRAL, "width": 2},
            },
        ),
        **subplot,
    )
    fig.add_annotation(
        x=point["cost_total_mean"] / 1000,
        y=point["on_time_rate_mean"],
        text=text,
        showarrow=True,
        arrowhead=2,
        arrowcolor=NEUTRAL,
        ax=40,
        ay=40,
        font={"size": 12},
        align="left",
        xanchor="left",
        **subplot,
    )


def fig_sweep_scatter(
    summary: pd.DataFrame,
    target: float = 0.95,
    *,
    growth: float | str | None = None,
    show_range: bool = False,
    highlight: bool = True,
    for_readme: bool = False,
) -> go.Figure:
    """The headline: on-time rate vs total cost, one dot per run.

    ``summary`` is a normalised sweep table (``results.normalise_sweep_frame``).
    ``growth``: a value filters to that actual growth (one panel); ``"all"``
    draws **small multiples**, one panel per growth level on a shared y axis;
    ``None`` puts every row in one panel. Colour + symbol = policy, filled =
    pre-screen on, hollow = off. The cheapest run that meets the target is
    enlarged and annotated; if none does, the best on-time run is.
    """
    df = summary
    if growth not in (None, "all"):
        df = summary[summary["actual_growth"] == float(growth)]
    n_repeats = int(df["sim.seeds"].max()) if "sim.seeds" in df.columns and not df.empty else None
    repeats = f" · mean of {n_repeats} repeats" if n_repeats else ""
    subtitle = f"One dot per run · {len(df)} runs{repeats} · synthetic data"
    if df.empty:
        fig = go.Figure()
        return style_figure(fig, "No runs to show", subtitle, for_readme=for_readme)

    levels = sorted(df["actual_growth"].unique(), reverse=True)
    small_multiples = growth == "all" and len(levels) > 1
    y_low = max(0.0, float(df["on_time_rate_mean"].min()) - 0.02)
    if show_range:
        y_low = max(0.0, min(y_low, float(df["on_time_rate_min"].min()) - 0.01))
    if small_multiples:
        fig = make_subplots(
            rows=1,
            cols=len(levels),
            shared_yaxes=True,
            horizontal_spacing=0.03,
            subplot_titles=[f"{fmt_growth(g)} growth" for g in levels],
        )
        for col, level in enumerate(levels, start=1):
            panel = df[df["actual_growth"] == level]
            _sweep_traces(fig, panel, show_range=show_range, legend=col == 1, row=1, col=col)
            add_target_line(fig, target, row=1, col=col)
            if highlight:
                _annotate_best(fig, panel, target, row=1, col=col)
            fig.update_xaxes(title_text="Total cost (k)", ticksuffix="k", row=1, col=col)
        fig.update_yaxes(tickformat=".0%", range=[y_low, 1.0])
        fig.update_yaxes(title_text="On-time rate", row=1, col=1)
        title = f"Cheapest way to stay on time, by demand growth ({fmt_pct(target, 0)} target)"
    else:
        fig = go.Figure()
        _sweep_traces(fig, df, show_range=show_range, legend=True)
        add_target_line(fig, target)
        if highlight:
            _annotate_best(fig, df, target)
        lo, hi = float(df["cost_total_mean"].min()), float(df["cost_total_mean"].max())
        if show_range:
            lo, hi = float(df["cost_total_min"].min()), float(df["cost_total_max"].max())
        pad = max((hi - lo) * 0.05, hi * 0.02)
        fig.update_xaxes(
            title_text="Total cost for the year (k)",
            ticksuffix="k",
            tickformat=",.0f",
            range=[(lo - pad) / 1000, (hi + pad) / 1000],
        )
        fig.update_yaxes(title_text="On-time rate", tickformat=".0%", range=[y_low, 1.0])
        single_growth = levels[0] if len(levels) == 1 else None
        title = takeaway_sweep(df, target, single_growth)
    return style_figure(fig, title, subtitle, for_readme=for_readme, show_legend=True)


def _annotate_best(fig: go.Figure, df: pd.DataFrame, target: float, **subplot: Any) -> None:
    best = cheapest_passing(df)
    if best is not None:
        row = df[df["row_id"] == best].iloc[0]
        text = (
            f"Cheapest that meets {fmt_pct(target, 0)}:<br>{describe_sweep_row(row)}"
            f"<br>{fmt_cost_k(row['cost_total_mean'])} · {fmt_pct(row['on_time_rate_mean'])}"
        )
    else:
        top = best_on_time(df)
        if top is None:
            return
        row = df[df["row_id"] == top].iloc[0]
        text = (
            f"Best on time (misses target):<br>{describe_sweep_row(row)}"
            f"<br>{fmt_cost_k(row['cost_total_mean'])} · {fmt_pct(row['on_time_rate_mean'])}"
        )
    _highlight(fig, row, text, **subplot)
