"""Sweep results (landing page): the headline answer, one dot per run (DESIGN_SYSTEM.md 5.5).

Landing on the answer, not on a blank form, shows a first-time user what the
tool is for; every dot is a door to "clone these settings and change one thing".
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
import ui

from scout_planner import charts, runs
from scout_planner.formatting import fmt_cost_k, fmt_days, fmt_growth, fmt_pct
from scout_planner.results import (
    list_published,
    load_published,
    local_sweep_frame,
    local_sweeps,
    normalise_sweep_frame,
)

PICKER = "sweep-picker"
SELECTED = "_sweep_selected"


def sources() -> dict[str, str]:
    """Option key -> label: published summaries first (headline on top), then local sweeps."""
    options = {
        f"published:{s.name}": f"{s.name.replace('_', ' ').capitalize()} (published)"
        for s in list_published(ui.published_dir())
    }
    for sweep_id, statuses in sorted(local_sweeps(ui.runs_root()).items()):
        newest = max(s.created_at for s in statuses)
        options[f"local:{sweep_id}"] = f"{sweep_id} (local, {newest:%d %b})"
    return options


def _on_pick() -> None:
    st.query_params["sweep"] = st.session_state[PICKER]
    st.session_state.pop(SELECTED, None)


def load_source(key: str) -> tuple[pd.DataFrame | None, tuple[int, int] | None]:
    """The normalised table, plus (finished, total) for a local sweep still running."""
    kind, _, name = key.partition(":")
    if kind == "published":
        match = [s for s in list_published(ui.published_dir()) if s.name == name]
        loaded = load_published(match[0].path) if match else None
        if loaded is None or loaded.value is None:
            path = match[0].path if match else name
            st.error(
                f"Couldn't read the published results file {path}. Re-create it with "
                "`make sweep SWEEP=headline`."
            )
            return None, None
        return normalise_sweep_frame(loaded.value, ui.defaults()), None
    statuses = local_sweeps(ui.runs_root()).get(name, [])
    frame = local_sweep_frame(statuses, ui.runs_root())
    finished = sum(1 for s in statuses if s.is_terminal)
    progress = (finished, len(statuses)) if finished < len(statuses) else None
    if frame.empty:
        return frame, progress
    return normalise_sweep_frame(frame, ui.defaults()), progress


def selected_from_chart(event: Any) -> int | None:
    try:
        points = event.selection.points
    except AttributeError:
        return None
    for point in points or []:
        data = point.get("customdata") if isinstance(point, dict) else None
        if isinstance(data, list):
            data = data[0] if data else None
        if data is not None:
            return int(data)
    return None


def selected_panel(row: pd.Series, source_key: str) -> None:
    with st.container(border=True):
        st.markdown(
            f"**Selected:** {row['name']} · {charts.describe_sweep_row(row)} · "
            f"{fmt_growth(row['actual_growth'])} growth"
        )
        cols = st.columns(3)
        cols[0].metric("On-time rate", fmt_pct(row["on_time_rate_mean"]), border=True)
        cols[1].metric("Total cost", fmt_cost_k(row["cost_total_mean"]), border=True)
        p90 = row.get("p90_turnaround_days_mean")
        cols[2].metric("Turnaround P90", fmt_days(p90) if p90 is not None else ui.DASH, border=True)
        run_id = row.get("run_id")
        exists = False
        if isinstance(run_id, str) and run_id:
            try:
                runs.read_status(run_id, ui.runs_root())
                exists = True
            except runs.RunNotFound:
                exists = False
        left, right, _ = st.columns([1, 1, 3])
        if exists and left.button("Open run", key="sweep-open"):
            ui.go_to("run", run=run_id)
        if right.button("Clone settings", key="sweep-clone"):
            if exists:
                ui.go_to("new", clone=run_id)
            else:
                ui.go_to("new", sweep=source_key, row=str(int(row["row_id"])))
        if not exists:
            left.caption("Only the summary of this run is stored.")


def sweep_table(df: pd.DataFrame) -> int | None:
    shown = df.sort_values(["cost_total_mean", "row_id"]).reset_index(drop=True)
    table = pd.DataFrame(
        {
            "Who does what": shown["policy"].map(charts.policy_label),
            "Actual growth": shown["actual_growth"].map(fmt_growth),
            "Pre-screen": shown["automation_enabled"].map({True: "On", False: "Off"}),
            "Meets target": shown["meets_target"],
            "On-time rate": shown["on_time_rate_mean"] * 100,
            "Turnaround P90": shown.get("p90_turnaround_days_mean"),
            "Total cost": shown["cost_total_mean"] / 1000,
            "Late penalties": shown.get("cost_late_penalty_mean", pd.Series(dtype=float)) / 1000,
            "Name": shown["name"],
        }
    )
    event = st.dataframe(
        table,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"sweep-table-{st.session_state.get(PICKER)}",
        column_config={
            "Meets target": st.column_config.CheckboxColumn(disabled=True),
            "On-time rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Turnaround P90": st.column_config.NumberColumn(format="%.1f d"),
            "Total cost": st.column_config.NumberColumn(format="%.0fk"),
            "Late penalties": st.column_config.NumberColumn(format="%.0fk"),
        },
    )
    rows = event.selection.rows
    return int(shown.iloc[rows[0]]["row_id"]) if rows else None


def results_view(df: pd.DataFrame, source_key: str, progress: tuple[int, int] | None) -> None:
    if progress is not None:
        done, total = progress
        st.progress(done / max(total, 1), text=f"{done} of {total} runs finished")
    if df.empty:
        st.info("No finished runs in this sweep yet.")
        return
    levels = sorted(df["actual_growth"].unique(), reverse=True)
    options: list[Any] = [float(g) for g in levels] + (["all"] if len(levels) > 1 else [])
    controls = st.columns([3, 2], vertical_alignment="bottom")
    growth = controls[0].radio(
        "Actual growth",
        options,
        horizontal=True,
        format_func=lambda g: "All" if g == "all" else fmt_growth(g),
        key=f"sweep-growth-{source_key}",
        help="Compare settings at one demand growth; All shows one panel per growth level.",
    )
    show_range = controls[1].toggle("Show range over repeats", key="sweep-range")
    target = (
        float(df["sim.target_on_time"].max())
        if "sim.target_on_time" in df.columns
        else ui.defaults().sim.target_on_time
    )
    fig = charts.fig_sweep_scatter(df, target, growth=growth, show_range=show_range)
    if progress is not None:
        fig.update_layout(title_text="So far: " + str(fig.layout.title.text))
    event = st.plotly_chart(
        fig,
        theme="streamlit",
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=f"sweep-chart-{source_key}-{growth}",
    )
    st.caption(
        "Colour and shape = who does what. Filled = pre-screen on, hollow = off. "
        "Click a dot (or a table row) for details."
    )
    visible = df if growth == "all" else df[df["actual_growth"] == growth]
    st.subheader("All runs in this sweep")
    table_pick = sweep_table(visible)

    # Chart and table both select a run; the one that changed last wins.
    chart_pick = selected_from_chart(event)
    previous = st.session_state.get("_sweep_prev", (None, None))
    if chart_pick != previous[0] and chart_pick is not None:
        st.session_state[SELECTED] = chart_pick
    elif table_pick != previous[1] and table_pick is not None:
        st.session_state[SELECTED] = table_pick
    st.session_state["_sweep_prev"] = (chart_pick, table_pick)
    pick = st.session_state.get(SELECTED)
    match = df[df["row_id"] == pick] if pick is not None else df.iloc[0:0]
    if not match.empty:
        selected_panel(match.iloc[0], source_key)


def live_results(key: str) -> None:
    """A local sweep that is still running: reload its table on every refresh."""
    df, progress = load_source(key)
    results_view(df if df is not None else pd.DataFrame(), key, progress)


# --- page ----------------------------------------------------------------------------------

st.title("Sweep results")
st.caption("A sweep is a batch of runs that vary a few settings.")
options = sources()
if not options:
    ui.empty_state(
        "No sweep results yet",
        "A sweep runs many settings in one go. From a terminal, run `make sweep SWEEP=quick` "
        "(a few minutes) or `make all`.",
        [("Start a single run instead", "new", True)],
    )
    st.stop()

wanted = ui.query_value("sweep")
if wanted not in options:
    wanted = next(iter(options))
if st.session_state.get(PICKER) != wanted:
    st.session_state[PICKER] = wanted
source_key = st.selectbox(
    "Sweep", list(options), key=PICKER, format_func=options.get, on_change=_on_pick
)
frame, sweep_progress = load_source(source_key)
if frame is not None:
    if sweep_progress is not None:
        # A local sweep still running: refresh this part every 5 s.
        st.fragment(run_every="5s")(live_results)(source_key)
    else:
        results_view(frame, source_key, sweep_progress)
