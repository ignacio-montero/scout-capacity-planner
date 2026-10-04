"""README figures and key numbers, built from the published sweep summaries (M6).

Pure functions: each takes a normalised sweep table (``prepare``) and returns a
plotly figure whose title states the takeaway computed from the data (an
*action title*, DESIGN_SYSTEM.md section 3.4). ``scripts/make_readme_charts.py``
is the thin shell that reads ``data/published/``, calls these, and exports PNGs.

The same tokens as the app (``charts.py``): colour + symbol = assignment
policy; a filled marker = the condition named in the subtitle (pre-screen on,
or daily assignment), hollow = without it. Rates are drawn as dots (not bars)
so their axis may start above zero, where the differences around 95% are.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scout_planner import charts
from scout_planner.config import Params
from scout_planner.formatting import fmt_cost_k, fmt_growth, fmt_money, fmt_pct, fmt_pct_near
from scout_planner.results import flatten_params, normalise_sweep_frame

# Parameters the README charts group by; a published file only has a column
# for a parameter that was varied, fixed or non-default, so the rest is filled
# from the defaults.
USED_PARAMS: tuple[str, ...] = (
    "assignment.policy",
    "assignment.cadence",
    "automation.enabled",
    "capacity_plan.hire_mix",
    "capacity_plan.quantile",
    "capacity_plan.assumed_growth",
    "demand.actual_growth",
    "team.follow_hiring_plan",
    "cost.late_penalty",
    "sim.seeds",
    "sim.target_on_time",
)
SWEEPS: tuple[str, ...] = (
    "headline",
    "growth",
    "forecast_error",
    "baseline",
    "cadence",
    "late_penalty",
)


# --- data preparation ---------------------------------------------------------------


def prepare(df: pd.DataFrame, defaults: Params) -> pd.DataFrame:
    """Normalise a published table and fill the parameter columns the charts need."""
    out = normalise_sweep_frame(df, defaults)
    flat = flatten_params(defaults)
    for key in USED_PARAMS:
        if key not in out.columns:
            out[key] = flat[key]
    # Cost without the late penalty: what the agency actually spends on capacity.
    # Only the mean is exact (min and max of a difference aren't differences of
    # min and max), so the range columns repeat the mean and draw no whiskers.
    if {"cost_total_mean", "cost_late_penalty_mean"} <= set(out.columns):
        excl = out["cost_total_mean"] - out["cost_late_penalty_mean"].fillna(0.0)
        out["cost_excl_penalty_mean"] = excl
        out["cost_excl_penalty_min"] = excl
        out["cost_excl_penalty_max"] = excl
    return out


def target_of(df: pd.DataFrame, defaults: Params) -> float:
    if "sim.target_on_time" in df.columns and not df.empty:
        return float(df["sim.target_on_time"].max())
    return defaults.sim.target_on_time


def repeats_of(df: pd.DataFrame) -> int | None:
    return int(df["sim.seeds"].max()) if "sim.seeds" in df.columns and not df.empty else None


def footnote(sweep: str, df: pd.DataFrame) -> str:
    n = repeats_of(df)
    per = f" · {n} repeats per setting" if n else ""
    note = fallback_note(df)
    return f"Synthetic data · {sweep.replace('_', ' ')} sweep · {len(df)} runs{per}" + (
        f" · {note}" if note else ""
    )


def _column(df: pd.DataFrame, name: str) -> pd.Series | None:
    """A diagnostics column, as published (``diag_fallbacks``), plain or flattened."""
    for candidate in (f"diag_{name}", name, f"diag_{name}_mean", f"{name}_mean"):
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce")
    return None


def fallback_note(df: pd.DataFrame) -> str | None:
    """When optimiser rounds fell back to EDF, say so (the point isn't purely the optimiser).

    ``"optimiser fell back to EDF in up to 3.2% of rounds (4 of 18 optimiser runs)"``;
    None when there are no optimiser runs, no diagnostics columns or no fallbacks.
    """
    fallbacks = _column(df, "fallbacks")
    if fallbacks is None or "policy" not in df.columns:
        return None
    opt = df["policy"] == "optimiser"
    hit = opt & (fallbacks.fillna(0) > 0)
    if not hit.any():
        return None
    runs_text = f"{int(hit.sum())} of {int(opt.sum())} optimiser runs"
    share = _column(df, "fallback_share")
    if share is not None and share[hit].notna().any():
        return (
            f"optimiser fell back to EDF in up to {fmt_pct(float(share[hit].max()))} of rounds "
            f"({runs_text})"
        )
    return f"optimiser fell back to EDF {int(fallbacks[hit].sum()):,} times ({runs_text})"


def cheapest_passing_row(df: pd.DataFrame) -> pd.Series | None:
    """The lowest-cost row that meets its target (ties: lowest row_id), or None."""
    best = charts.cheapest_passing(df)
    return None if best is None else df[df["row_id"] == best].iloc[0]


def best_row(df: pd.DataFrame) -> pd.Series | None:
    """Cheapest passing row, else the best on-time row (flagged by ``meets_target``)."""
    row = cheapest_passing_row(df)
    if row is not None:
        return row
    top = charts.best_on_time(df)
    return None if top is None else df[df["row_id"] == top].iloc[0]


def cheapest_per_mix(df: pd.DataFrame) -> dict[str, pd.Series | None]:
    """Hire mix -> its cheapest passing row (or its best on-time row if none passes)."""
    mixes = [m for m in charts.HIRE_MIX_ORDER if m in set(df["capacity_plan.hire_mix"])]
    mixes += sorted(set(df["capacity_plan.hire_mix"]) - set(mixes))
    return {m: best_row(df[df["capacity_plan.hire_mix"] == m]) for m in mixes}


def default_plan(df: pd.DataFrame, defaults: Params) -> pd.DataFrame:
    """Headline rows at the default plan (the default hire mix and planning quantile)."""
    cp = defaults.capacity_plan
    return df[
        (df["capacity_plan.hire_mix"] == cp.hire_mix)
        & ((df["capacity_plan.quantile"] - cp.quantile).abs() < 1e-9)
    ]


def mix_label(mix: str) -> str:
    return charts.HIRE_MIX_LABELS.get(mix, mix)


def _short(row: pd.Series) -> str:
    """Identity without the hire mix: ``"Optimiser, P80, pre-screen on"``."""
    return charts.describe_sweep_row(row.drop(labels=["capacity_plan.hire_mix"], errors="ignore"))


# --- shared drawing helpers -------------------------------------------------------------


def _err(df: pd.DataFrame, metric: str, scale: float = 1.0) -> dict[str, Any]:
    mean, lo, hi = (df[f"{metric}_{b}"] / scale for b in ("mean", "min", "max"))
    return {
        "type": "data",
        "symmetric": False,
        "array": (hi - mean).tolist(),
        "arrayminus": (mean - lo).tolist(),
        "thickness": 1.2,
        "width": 4,
    }


def _policy_dots(
    fig: go.Figure,
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    filled_value: Any,
    group_names: Mapping[Any, str],
    *,
    scale: float = 1.0,
    legend: bool = True,
    row: int = 1,
    col: int = 1,
) -> None:
    """One dot per policy x group with min-max error bars; filled = ``filled_value``."""
    hover_fmt = "%{y:.1%}" if metric == "on_time_rate" else "%{y:,.0f}k"
    for policy in charts.ordered_policies(df["policy"].unique().tolist()):
        for value, name in group_names.items():
            part = df[(df["policy"] == policy) & (df[group_col] == value)]
            if part.empty:
                continue
            color = charts.policy_color(policy)
            symbol = charts.POLICY_SYMBOLS.get(policy, "circle")
            fig.add_trace(
                go.Scatter(
                    x=[charts.policy_label(policy)] * len(part),
                    y=part[f"{metric}_mean"] / scale,
                    mode="markers",
                    name=charts.policy_label(policy),
                    legendgroup=policy,
                    showlegend=legend and value == filled_value,
                    offsetgroup=str(value),
                    marker={
                        "symbol": symbol if value == filled_value else f"{symbol}-open",
                        "size": 13,
                        "color": color,
                        "line": {"color": color, "width": 2},
                    },
                    error_y={**_err(part, metric, scale), "color": color},
                    hovertemplate=f"{charts.policy_label(policy)} · {name}<br>{hover_fmt}"
                    "<extra></extra>",
                ),
                row=row,
                col=col,
            )


def _policy_lines(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    metric: str,
    *,
    scale: float = 1.0,
    legend: bool = True,
    x_fmt: Any = fmt_growth,
    row: int = 1,
    col: int = 1,
) -> None:
    """One line per policy over an ordered x (growth "1x", "2x"... by default)."""
    for policy in charts.ordered_policies(df["policy"].unique().tolist()):
        part = df[df["policy"] == policy].sort_values(x_col)
        color = charts.policy_color(policy)
        fig.add_trace(
            go.Scatter(
                x=[x_fmt(v) for v in part[x_col]],
                y=part[f"{metric}_mean"] / scale,
                mode="lines+markers",
                name=charts.policy_label(policy),
                legendgroup=policy,
                showlegend=legend,
                line={"color": color, "width": 2.5},
                marker={"symbol": charts.POLICY_SYMBOLS.get(policy, "circle"), "size": 11},
                error_y={**_err(part, metric, scale), "color": color},
            ),
            row=row,
            col=col,
        )


def _two_panels(left: str, right: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=[left, right])
    fig.update_layout(scattermode="group", scattergap=0.6)
    return fig


def _rate_axis(fig: go.Figure, df: pd.DataFrame, target: float, col: int = 1) -> None:
    low = min(float(df["on_time_rate_min"].min()), target) - 0.03
    fig.update_yaxes(
        title_text="On-time rate", tickformat=".0%", range=[max(0.0, low), 1.0], row=1, col=col
    )
    charts.add_target_line(fig, target, row=1, col=col)


def _cost_axis(fig: go.Figure, df: pd.DataFrame, col: int = 2, *, from_zero: bool = False) -> None:
    """Cost in thousands. Dots may start above zero (only bars must start at zero),
    so the range is set from the data; growth lines start at zero so the eye
    reads the proportional increase."""
    lo, hi = float(df["cost_total_min"].min()) / 1000, float(df["cost_total_max"].max()) / 1000
    pad = max((hi - lo) * 0.12, hi * 0.01)
    fig.update_yaxes(
        title_text="Total cost for the year (k)",
        ticksuffix="k",
        tickformat=",.0f",
        range=[0 if from_zero else lo - pad, hi + pad],
        row=1,
        col=col,
    )


def _room_for_target_label(fig: go.Figure, n_categories: int) -> None:
    """Extend the category axis to the right so "95% target" doesn't sit on a dot."""
    fig.update_xaxes(range=[-0.5, n_categories - 0.5 + 0.45], row=1, col=1)


def _finish(fig: go.Figure, title: str, subtitle: str, note: str) -> go.Figure:
    return charts.style_figure(
        fig, title, subtitle, for_readme=True, show_legend=True, footnote=note
    )


# --- 1. headline ------------------------------------------------------------------------


def fig_headline(df: pd.DataFrame, target: float) -> go.Figure:
    """On-time vs cost, one dot per run, one panel per hire mix; the answer annotated."""
    fig = charts.fig_sweep_scatter(
        df,
        target,
        facet="capacity_plan.hire_mix",
        facet_labels=charts.HIRE_MIX_LABELS,
    )
    n = repeats_of(df)
    subtitle = (
        f"One dot per setting ({len(df)}) · colour and shape = assignment rule · "
        f"filled = pre-screen on, hollow = off · mean of {n} repeats"
    )
    growth = df["actual_growth"].iloc[0] if df["actual_growth"].nunique() == 1 else None
    return _finish(
        fig, charts.takeaway_sweep(df, target, growth), subtitle, footnote("headline", df)
    )


# --- 2. mix -------------------------------------------------------------------------------


def title_mix(winners: Mapping[str, pd.Series | None], target: float) -> str:
    passing = {m: r for m, r in winners.items() if r is not None and bool(r["meets_target"])}
    failing = [m for m, r in winners.items() if r is not None and not bool(r["meets_target"])]
    if not passing:
        return f"No hiring mix reaches {fmt_pct(target, 0)} on time"
    ranked = sorted(passing, key=lambda m: float(passing[m]["cost_total_mean"]))
    best = ranked[0]
    title = (
        f"{mix_label(best)} is the cheapest way to meet {fmt_pct(target, 0)} "
        f"({fmt_cost_k(passing[best]['cost_total_mean'])})"
    )
    if len(ranked) > 1:
        other = ranked[-1]
        delta = float(passing[other]["cost_total_mean"] - passing[best]["cost_total_mean"])
        title += f"; {mix_label(other).lower()} costs {fmt_cost_k(delta)} more"
    if failing:
        names = " and ".join(mix_label(m).lower() for m in failing)
        title += f"; {names} never {'reach' if len(failing) > 1 else 'reaches'} it"
    return title


def fig_mix(df: pd.DataFrame, target: float) -> go.Figure:
    """Cost split of the cheapest passing setting per hire mix, on-time in the label."""
    winners = cheapest_per_mix(df)
    parts: dict[str, dict[str, float]] = {}
    for mix, row in winners.items():
        if row is None:
            continue
        verdict = "" if bool(row["meets_target"]) else ", misses target"
        label = (
            f"<b>{mix_label(mix)}</b><br>{_short(row)}<br>"
            f"{fmt_pct_near(row['on_time_rate_mean'], target)} on time{verdict}"
        )
        parts[label] = {c: float(row.get(f"{c}_mean") or 0.0) for c in charts.COST_ORDER}
    fig = charts.fig_cost_split(parts, mode="absolute")
    fig.update_yaxes(automargin=True)
    subtitle = (
        "Cheapest setting that meets the target, per hiring mix (best on-time one if none "
        "does) · cost split, mean over repeats"
    )
    return _finish(fig, title_mix(winners, target), subtitle, footnote("headline", df))


# --- 3. policies --------------------------------------------------------------------------


def title_policies(plan: pd.DataFrame, target: float) -> str:
    best = cheapest_passing_row(plan)
    if best is None:
        top = best_row(plan)
        if top is None:
            return "Assignment rules at the default plan"
        return (
            f"At the default plan no rule meets {fmt_pct(target, 0)}; best is "
            f"{charts.describe_sweep_row(top)} at {fmt_pct(top['on_time_rate_mean'])}"
        )
    tool = "with" if bool(best["automation_enabled"]) else "without"
    title = (
        f"At the default plan, {charts.policy_label(str(best['policy']))} {tool} the "
        f"pre-screen is the cheapest way to meet {fmt_pct(target, 0)} "
        f"({fmt_cost_k(best['cost_total_mean'])})"
    )
    # The same question with the tool switched the other way.
    other = cheapest_passing_row(plan[plan["automation_enabled"] != best["automation_enabled"]])
    flip = "without it" if tool == "with" else "with it"
    if other is None:
        title += f"; {flip}, no rule meets it"
    else:
        title += (
            f"; {flip}, {charts.policy_label(str(other['policy']))} "
            f"({fmt_cost_k(other['cost_total_mean'])})"
        )
    return title


def fig_policies(df: pd.DataFrame, target: float, defaults: Params) -> go.Figure:
    plan = default_plan(df, defaults)
    fig = _two_panels("On-time rate", "Total cost for the year")
    names = {False: "pre-screen off", True: "pre-screen on"}
    _policy_dots(fig, plan, "on_time_rate", "automation_enabled", True, names, col=1)
    _policy_dots(
        fig, plan, "cost_total", "automation_enabled", True, names, scale=1000, legend=False, col=2
    )
    _rate_axis(fig, plan, target)
    _cost_axis(fig, plan)
    _room_for_target_label(fig, plan["policy"].nunique())
    cp = defaults.capacity_plan
    subtitle = (
        f"Default plan: {mix_label(cp.hire_mix).lower()}, P{round(cp.quantile * 100)} · "
        "filled = pre-screen on, hollow = off · whiskers = range over repeats"
    )
    return _finish(fig, title_policies(plan, target), subtitle, footnote("headline", df))


# --- 4. growth ----------------------------------------------------------------------------


def _first_failing(part: pd.DataFrame, x_col: str) -> Any:
    failing = part[~part["meets_target"]].sort_values(x_col)
    return None if failing.empty else failing[x_col].iloc[0]


def title_growth(df: pd.DataFrame, target: float) -> str:
    parts = []
    for policy in charts.ordered_policies(df["policy"].unique().tolist()):
        part = df[df["policy"] == policy]
        fails = part[~part["meets_target"]]
        label = charts.policy_label(policy)
        if fails.empty:
            parts.append(f"{label} meets {fmt_pct(target, 0)} at every growth level")
        else:
            levels = " and ".join(fmt_growth(g) for g in sorted(fails["actual_growth"]))
            # "narrowly" when some repeat of every failing setting still passed:
            # the miss is within the luck of the draw, and the title should say so.
            narrow = bool((fails["on_time_rate_max"] >= target).all())
            verb = "narrowly misses" if narrow else "misses"
            parts.append(f"{label} {verb} {fmt_pct(target, 0)} at {levels}")
    lo, hi = df["actual_growth"].min(), df["actual_growth"].max()
    cost = df.groupby("actual_growth")["cost_total_mean"].mean()
    if lo != hi and cost[lo] > 0:
        parts.append(
            f"cost rises {cost[hi] / cost[lo]:.1f}x from {fmt_growth(lo)} to {fmt_growth(hi)}"
        )
    return "; ".join(parts)


def fig_growth(df: pd.DataFrame, target: float) -> go.Figure:
    fig = _two_panels("On-time rate", "Total cost for the year")
    _policy_lines(fig, df, "actual_growth", "on_time_rate", col=1)
    _policy_lines(fig, df, "actual_growth", "cost_total", scale=1000, legend=False, col=2)
    _rate_axis(fig, df, target)
    _cost_axis(fig, df, from_zero=True)
    _room_for_target_label(fig, df["actual_growth"].nunique())
    fig.update_xaxes(title_text="Demand growth over the year (planned for exactly)")
    subtitle = "The agency plans for the growth that arrives · whiskers = range over repeats"
    return _finish(fig, title_growth(df, target), subtitle, footnote("growth", df))


# --- 5. forecast error ----------------------------------------------------------------------


def forecast_error_effects(df: pd.DataFrame, policy: str | None = None) -> dict[str, Any] | None:
    """On-time and cost change of under- and over-planning vs planning right.

    Averaged over the policies in the sweep unless ``policy`` is given.
    Returns ``{"actual", "under", "over", "under_pts", "under_cost", "over_pts", "over_cost"}``.
    """
    part = df if policy is None else df[df["policy"] == policy]
    if part.empty or part["actual_growth"].nunique() != 1:
        return None
    actual = float(part["actual_growth"].iloc[0])
    by = part.groupby("capacity_plan.assumed_growth")[
        ["on_time_rate_mean", "cost_total_mean"]
    ].mean()
    if actual not in by.index:
        return None
    base = by.loc[actual]
    under = [a for a in by.index if a < actual]
    over = [a for a in by.index if a > actual]
    out: dict[str, Any] = {"actual": actual, "under": None, "over": None}
    if under:
        u = min(under)
        out.update(
            under=u,
            under_pts=float(by.loc[u, "on_time_rate_mean"] - base["on_time_rate_mean"]),
            under_cost=float(by.loc[u, "cost_total_mean"] - base["cost_total_mean"]),
        )
    if over:
        o = max(over)
        out.update(
            over=o,
            over_pts=float(by.loc[o, "on_time_rate_mean"] - base["on_time_rate_mean"]),
            over_cost=float(by.loc[o, "cost_total_mean"] - base["cost_total_mean"]),
        )
    return out


def _effect(pts: float, cost: float) -> str:
    pts_text = f"{'loses' if pts < 0 else 'gains'} {abs(round(pts * 100, 1)):.1f} pts of on-time"
    cost_text = f"{'adds' if cost > 0 else 'saves'} {fmt_cost_k(abs(cost))}"
    return f"{pts_text} and {cost_text}"


def title_forecast_error(df: pd.DataFrame) -> str:
    fx = forecast_error_effects(df)
    if fx is None:
        return "The cost of forecast error"
    parts = []
    if fx["under"] is not None:
        parts.append(
            f"Hiring for {fmt_growth(fx['under'])} when {fmt_growth(fx['actual'])} arrives "
            + _effect(fx["under_pts"], fx["under_cost"])
        )
    if fx["over"] is not None:
        parts.append(
            f"hiring for {fmt_growth(fx['over'])} " + _effect(fx["over_pts"], fx["over_cost"])
        )
    text = "; ".join(parts)
    return text[:1].upper() + text[1:]


def fig_forecast_error(df: pd.DataFrame, target: float) -> go.Figure:
    fig = _two_panels("On-time rate", "Total cost for the year")
    x = "capacity_plan.assumed_growth"
    _policy_lines(fig, df, x, "on_time_rate", col=1)
    _policy_lines(fig, df, x, "cost_total", scale=1000, legend=False, col=2)
    _rate_axis(fig, df, target)
    _cost_axis(fig, df, from_zero=True)
    _room_for_target_label(fig, df[x].nunique())
    actual = df["actual_growth"].iloc[0]
    fig.update_xaxes(title_text=f"Growth the agency hired for ({fmt_growth(actual)} arrives)")
    subtitle = (
        "Changes are averaged over the assignment rules, against hiring for what "
        "arrives · whiskers = range over repeats"
    )
    return _finish(fig, title_forecast_error(df), subtitle, footnote("forecast_error", df))


# --- 6. baseline and cadence ----------------------------------------------------------------


def rule_and_tool(row: pd.Series) -> str:
    """``"Optimiser, pre-screen on"``: for sweeps where the hiring plan doesn't apply."""
    screen = "on" if bool(row["automation_enabled"]) else "off"
    return f"{charts.policy_label(str(row['policy']))}, pre-screen {screen}"


def title_baseline(df: pd.DataFrame, target: float) -> str:
    top = df.sort_values(["on_time_rate_mean", "row_id"], ascending=[False, True]).iloc[0]
    growth = fmt_growth(df["actual_growth"].max())
    if bool(top["meets_target"]):
        return f"Even without hiring, {rule_and_tool(top)} meets {fmt_pct(target, 0)} at {growth}"
    return (
        f"Without hiring, {growth} growth overwhelms the team: the best setting "
        f"({rule_and_tool(top)}) delivers only {fmt_pct(top['on_time_rate_mean'], 0)} on time"
    )


def fig_baseline(df: pd.DataFrame, target: float) -> go.Figure:
    fig = _two_panels("On-time rate", "Total cost for the year")
    names = {False: "pre-screen off", True: "pre-screen on"}
    _policy_dots(fig, df, "on_time_rate", "automation_enabled", True, names, col=1)
    _policy_dots(
        fig, df, "cost_total", "automation_enabled", True, names, scale=1000, legend=False, col=2
    )
    _rate_axis(fig, df, target)
    _cost_axis(fig, df)
    _room_for_target_label(fig, df["policy"].nunique())
    subtitle = (
        "Planned hires never join: the starting team alone · filled = pre-screen on, "
        "hollow = off · whiskers = range over repeats"
    )
    return _finish(fig, title_baseline(df, target), subtitle, footnote("baseline", df))


def cadence_effect(df: pd.DataFrame) -> pd.Series:
    """Per policy: on-time rate of daily minus weekly assignment (fraction)."""
    wide = df.pivot_table(index="policy", columns="assignment.cadence", values="on_time_rate_mean")
    if not {"daily", "weekly"} <= set(wide.columns):
        return pd.Series(dtype=float)
    return (wide["daily"] - wide["weekly"]).dropna()


def title_cadence(df: pd.DataFrame) -> str:
    effect = cadence_effect(df)
    if effect.empty:
        return "Daily vs weekly assignment"
    mean = float(effect.mean())
    if abs(mean) < 0.005:
        return "Assigning work daily or weekly changes on-time by less than 0.5 pts"
    better = "daily" if mean > 0 else "weekly"
    worse = "weekly" if mean > 0 else "daily"
    lo, hi = abs(effect).min() * 100, abs(effect).max() * 100
    spread = f"{lo:.1f}–{hi:.1f} pts"
    return (
        f"Assigning work {better} instead of {worse} adds {abs(round(mean * 100, 1)):.1f} pts "
        f"on time on average ({spread} across rules)"
    )


def fig_cadence(df: pd.DataFrame, target: float) -> go.Figure:
    fig = _two_panels("On-time rate", "Total cost for the year")
    names = {"weekly": "weekly", "daily": "daily"}
    _policy_dots(fig, df, "on_time_rate", "assignment.cadence", "daily", names, col=1)
    _policy_dots(
        fig, df, "cost_total", "assignment.cadence", "daily", names, scale=1000, legend=False, col=2
    )
    _rate_axis(fig, df, target)
    _cost_axis(fig, df)
    _room_for_target_label(fig, df["policy"].nunique())
    subtitle = (
        "How often work is assigned · filled = every day, hollow = once a week · "
        "whiskers = range over repeats"
    )
    return _finish(fig, title_cadence(df), subtitle, footnote("cadence", df))


# --- 7. late penalty -------------------------------------------------------------------------


def title_late_penalty(df: pd.DataFrame, target: float) -> str:
    """The cheapest rule **that meets the target** at each penalty level.

    A rule that misses the target is never called "cheapest" (it is cheap
    because it is late); rules that never pass are named. The "spends least
    excluding penalties" clause also only considers passing rows.
    """
    col = "cost.late_penalty"
    levels = sorted(df[col].unique())
    goal = fmt_pct(target, 0)
    passing = df[df["meets_target"]]
    cheapest = {}
    for lvl in levels:
        rows = passing[passing[col] == lvl].sort_values(["cost_total_mean", "row_id"])
        cheapest[lvl] = None if rows.empty else str(rows.iloc[0]["policy"])
    never = [
        p
        for p in charts.ordered_policies(df["policy"].unique().tolist())
        if p not in set(passing["policy"])
    ]
    span = f"({fmt_money(levels[0])}–{fmt_money(levels[-1])})"
    winners = set(cheapest.values())
    if winners == {None}:
        return f"No rule reaches {goal} at any late penalty {span}"
    if len(winners) == 1:
        title = (
            f"{charts.policy_label(next(iter(winners)))} is the cheapest rule that meets {goal} "
            f"at every late penalty {span}"
        )
    else:

        def at(lvl: float) -> str:
            name = cheapest[lvl]
            return f"{charts.policy_label(name) if name else 'none'} at {fmt_money(lvl)}"

        title = (
            f"The cheapest rule that meets {goal} depends on the late penalty: "
            f"{at(levels[0])}, {at(levels[-1])}"
        )
    if "cost_excl_penalty_mean" in passing.columns and passing["policy"].nunique() > 1:
        staff = passing.groupby("policy")["cost_excl_penalty_mean"].mean().sort_values()
        title += (
            f"; among those, {charts.policy_label(str(staff.index[0]))} spends least "
            "excluding penalties"
        )
    if never:
        names = " and ".join(charts.policy_label(p) for p in never)
        title += f"; {names} never {'reach' if len(never) > 1 else 'reaches'} {goal}"
    return title


def fig_late_penalty(df: pd.DataFrame, target: float) -> go.Figure:
    """Total cost and cost excluding the penalty, per rule, across late-penalty levels."""
    fig = _two_panels("Total cost for the year", "Cost excluding late penalties")
    x = "cost.late_penalty"
    _policy_lines(fig, df, x, "cost_total", scale=1000, x_fmt=fmt_money, col=1)
    _policy_lines(fig, df, x, "cost_excl_penalty", scale=1000, x_fmt=fmt_money, legend=False, col=2)
    lo = float(df["cost_excl_penalty_mean"].min()) / 1000
    hi = float(df["cost_total_max"].max()) / 1000
    pad = (hi - lo) * 0.08
    fig.update_yaxes(
        title_text="Cost for the year (k)",
        ticksuffix="k",
        tickformat=",.0f",
        range=[lo - pad, hi + pad],  # both panels on one scale, so they compare directly
    )
    fig.update_xaxes(title_text="Cost of one late report")
    subtitle = (
        "Same runs, two views: what the year costs with late penalties, and what is spent on "
        "capacity · whiskers on total cost = range over repeats"
    )
    return _finish(fig, title_late_penalty(df, target), subtitle, footnote("late_penalty", df))


# --- key numbers (markdown) -----------------------------------------------------------------


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _on_time(row: pd.Series, target: float) -> str:
    """``"95.7% (94.97%–96.3%)"``: 2 decimals for any value within 0.5 pts of the target."""
    return _rng(row, "on_time_rate", lambda v: fmt_pct_near(v, target))


def _rng(row: pd.Series, metric: str, fmt: Any) -> str:
    return f"{fmt(row[f'{metric}_mean'])} ({fmt(row[f'{metric}_min'])}–{fmt(row[f'{metric}_max'])})"


def _yes(row: pd.Series) -> str:
    return "yes" if bool(row["meets_target"]) else "no"


def key_numbers_markdown(frames: Mapping[str, pd.DataFrame | None], defaults: Params) -> str:
    """The numbers the README quotes, as markdown tables (one section per available sweep)."""
    out = ["# Key numbers (generated by scripts/make_readme_charts.py)", ""]
    head = frames.get("headline")
    if head is not None and not head.empty:
        target = target_of(head, defaults)
        best = cheapest_passing_row(head)
        out += ["## Headline", ""]
        if best is not None:
            out += [
                f"Cheapest setting that meets {fmt_pct(target, 0)}: "
                f"**{charts.describe_sweep_row(best)}**, "
                f"{fmt_pct_near(best['on_time_rate_mean'], target)} on time, "
                f"{fmt_cost_k(best['cost_total_mean'])} a year"
                + charts.worst_repeat_note(best, target)
                + ".",
                "",
            ]
            no_tool = cheapest_passing_row(head[~head["automation_enabled"]])
            if no_tool is not None:
                out += [
                    f"Cheapest without the pre-screen: **{charts.describe_sweep_row(no_tool)}**, "
                    f"{fmt_pct_near(no_tool['on_time_rate_mean'], target)}, "
                    f"{fmt_cost_k(no_tool['cost_total_mean'])}"
                    + charts.worst_repeat_note(no_tool, target)
                    + ".",
                    "",
                ]
        n_pass = int(head["meets_target"].sum())
        out += [f"{n_pass} of {len(head)} settings meet the target.", ""]
        out += ["### Cheapest passing setting per hiring mix", ""]
        rows = []
        for mix, row in cheapest_per_mix(head).items():
            if row is None:
                continue
            rows.append(
                [
                    mix_label(mix),
                    _short(row),
                    _on_time(row, target),
                    _rng(row, "cost_total", fmt_cost_k),
                    _yes(row) + ("" if bool(row["meets_target"]) else " (best on-time shown)"),
                ]
            )
        out += [_md_table(["Hiring mix", "Setting", "On time", "Total cost", "Meets target"], rows)]
        plan = default_plan(head, defaults)
        cp = defaults.capacity_plan
        out += ["", f"### Assignment rules at the default plan ({mix_label(cp.hire_mix).lower()}, "
                f"P{round(cp.quantile * 100)})", ""]  # fmt: skip
        rows = [
            [
                charts.policy_label(str(r["policy"])),
                "on" if bool(r["automation_enabled"]) else "off",
                _on_time(r, target),
                _rng(r, "cost_total", fmt_cost_k),
                fmt_cost_k(r.get("cost_late_penalty_mean")),
                fmt_cost_k(r.get("cost_excl_penalty_mean")),
                _yes(r),
            ]
            for _, r in plan.sort_values(["automation_enabled", "policy"]).iterrows()
        ]
        out += [
            _md_table(
                [
                    "Rule",
                    "Pre-screen",
                    "On time",
                    "Total cost",
                    "Late penalties",
                    "Cost excl. late penalty",
                    "Meets",
                ],
                rows,
            ),
            "",
        ]
    simple = {
        "growth": ("Growth (planned for exactly)", "actual_growth", fmt_growth, "Growth"),
        "forecast_error": (
            "Forecast error (4x arrives)",
            "capacity_plan.assumed_growth",
            fmt_growth,
            "Hired for",
        ),
        "baseline": (
            "No-hiring baseline",
            "automation_enabled",
            lambda v: "on" if v else "off",
            "Pre-screen",
        ),  # fmt: skip
        "cadence": ("Assignment cadence", "assignment.cadence", str, "Cadence"),
        "late_penalty": ("Late penalty", "cost.late_penalty", fmt_money, "Late penalty"),
    }
    for name, (heading, col, fmt, label) in simple.items():
        df = frames.get(name)
        if df is None or df.empty:
            continue
        target = target_of(df, defaults)
        rows = [
            [
                fmt(r[col]),
                charts.policy_label(str(r["policy"])),
                _on_time(r, target),
                _rng(r, "cost_total", fmt_cost_k),
                fmt_cost_k(r.get("cost_excl_penalty_mean")),
                _yes(r),
            ]
            for _, r in df.sort_values([col, "policy"]).iterrows()
        ]
        header = [label, "Rule", "On time", "Total cost", "Cost excl. late penalty", "Meets"]
        out += [f"## {heading}", "", _md_table(header, rows), ""]
        if note := fallback_note(df):
            out += [f"Note: {note}.", ""]
        if name == "forecast_error" and forecast_error_effects(df) is not None:
            out += [title_forecast_error(df) + " (averaged over rules).", ""]
        if name == "cadence":
            out += [title_cadence(df) + ".", ""]
        if name == "late_penalty":
            out += [title_late_penalty(df, target) + ".", ""]
    return "\n".join(out).rstrip() + "\n"
