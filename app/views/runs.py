"""Runs: what is queued, running and finished, with actions (DESIGN_SYSTEM.md section 5.2).

Only the "In progress" part refreshes itself (``st.fragment(run_every="2s")``):
if the whole page refreshed, the user's row selection in the Finished table
would jump every two seconds. When a run leaves "In progress", the fragment
asks for one full rerun so the Finished table picks it up.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import ui

from scout_planner import runs
from scout_planner.charts import POLICY_LABELS
from scout_planner.formatting import fmt_duration, fmt_growth, fmt_when

ACTIVE_KEY = "_runs_active_ids"
FILTER_KEYS = ("runs-search", "runs-status", "runs-sweeps")
STATUS_CHOICES = ["Done", "Failed", "Cancelled", "Unreadable"]


def _policy(run_id: str) -> str:
    params = ui.run_params(run_id)
    return POLICY_LABELS.get(params.assignment.policy, "?") if params else ui.DASH


def just_queued_banner() -> None:
    new_id = ui.query_value("new")
    if not new_id:
        return
    try:
        status = runs.read_status(new_id, ui.runs_root())
    except runs.RunNotFound:
        return
    queued = [s.run_id for s in runs.queued_runs(ui.runs_root())]
    if status.state == "queued" and new_id in queued:
        where = ui.queue_sentence(queued.index(new_id) + 1)
    elif status.state == "running":
        where = ui.queue_sentence(None)
    else:
        where = f"It is {status.state} already."
    left, right = st.columns([5, 1], vertical_alignment="center")
    left.success(f"Queued '{status.name}'. {where}", icon=":material/check:")
    if right.button("Open run", key="banner-open"):
        ui.go_to("run", run=new_id)


def running_card(status: runs.RunStatus) -> None:
    state = ui.run_display_state(status)
    with st.container(border=True):
        head, open_col, cancel_col = st.columns([5, 1, 1], vertical_alignment="center")
        head.markdown(f"**{status.name}** {ui.badge_markdown(ui.badge_spec(state))}")
        if open_col.button("Open", key=f"open-{status.run_id}"):
            ui.go_to("run", run=status.run_id)
        with cancel_col:
            if state == "cancelling":
                st.caption("Cancelling…")
            else:
                ui.confirm_popover(
                    "Cancel",
                    "Stop this run? Work done so far is discarded.",
                    "Stop run",
                    lambda: runs.request_cancel(status.run_id, ui.runs_root()),
                    key=f"cancel-{status.run_id}",
                )
        ui.run_progress(status)
        if state == "cancelling":
            st.caption("Stops at the end of the current simulated week.")


def queued_table(queued: list[runs.RunStatus]) -> None:
    root = ui.runs_root()
    rows, position = [], 0
    for status in queued:
        cancelling = runs.cancel_requested(status.run_id, root)
        if not cancelling:
            position += 1
        params = ui.run_params(status.run_id)
        rows.append(
            {
                "#": "Cancelling" if cancelling else str(position),
                "Name": status.name,
                "Policy": _policy(status.run_id),
                "Actual growth": fmt_growth(params.demand.actual_growth) if params else ui.DASH,
                "Queued at": fmt_when(status.created_at),
                "Sweep": status.sweep_id or "",
            }
        )
    st.markdown(f"**Queued ({position})**")
    event = st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="queued-table",
    )
    chosen = [queued[i] for i in event.selection.rows if i < len(queued)]
    if st.button("Cancel selected", disabled=not chosen, help="Select queued runs first."):
        for status in chosen:
            runs.request_cancel(status.run_id, root)
        st.rerun(scope="fragment")


@st.fragment(run_every="2s")
def in_progress() -> None:
    root = ui.runs_root()
    statuses = ui.list_runs()
    running = [s for s in statuses if s.state == "running"]
    queued = runs.queued_runs(root, include_cancel_requested=True)
    active = {s.run_id for s in running + queued}
    before = st.session_state.get(ACTIVE_KEY)
    st.session_state[ACTIVE_KEY] = active
    if before is not None and before - active:
        st.rerun(scope="app")  # a run finished: refresh the Finished table too
    if not running and not queued:
        st.caption("Nothing running.")
        st.page_link(ui.PAGES["new"], label="Start a new run", icon=":material/tune:")
        return
    for status in running:
        running_card(status)
    if queued:
        queued_table(queued)
    if not runs.worker_alive(root):
        ui.worker_down_warning()


def finished_frame(statuses: list[runs.RunStatus]) -> pd.DataFrame:
    rows = []
    for status in statuses:
        params = ui.run_params(status.run_id) if status.state != "unreadable" else None
        summary = ui.run_summary(status.run_id).value if status.state == "done" else None
        on_time = summary.mean("on_time_rate") if summary else None
        p90 = summary.mean("p90_turnaround_days") if summary else None
        cost = summary.mean("cost_total") if summary else None
        rows.append(
            {
                "run_id": status.run_id,
                "Name": status.name,
                "Status": ui.badge_spec(status.state)[2],
                "Meets target": summary.meets_target if summary else None,
                "On-time rate": on_time * 100 if on_time is not None else None,
                "Turnaround P90": p90,
                "Total cost": cost / 1000 if cost is not None else None,
                "Policy": POLICY_LABELS.get(params.assignment.policy, "") if params else "",
                "Actual growth": fmt_growth(params.demand.actual_growth) if params else "",
                "Assumed growth": fmt_growth(params.capacity_plan.assumed_growth) if params else "",
                "Pre-screen": ("On" if params.automation.enabled else "Off") if params else "",
                "Repeats": params.sim.seeds if params else None,
                "Finished": status.finished_at,
                "Duration": fmt_duration(status.duration_s) if status.started_at else "",
                "Sweep": status.sweep_id or "",
            }
        )
    return pd.DataFrame(rows)


COLUMN_CONFIG = {
    "Name": st.column_config.TextColumn(pinned=True),
    "Meets target": st.column_config.CheckboxColumn(disabled=True),
    "On-time rate": st.column_config.NumberColumn(format="%.1f%%"),
    "Turnaround P90": st.column_config.NumberColumn(format="%.1f d"),
    "Total cost": st.column_config.NumberColumn(format="%.0fk"),
    "Repeats": st.column_config.NumberColumn(format="%d"),
    "Finished": st.column_config.DatetimeColumn(format="DD MMM HH:mm"),
}


def clear_filters() -> None:
    st.session_state["runs-search"] = ""
    st.session_state["runs-status"] = STATUS_CHOICES
    st.session_state["runs-sweeps"] = False


def finished_section(statuses: list[runs.RunStatus]) -> None:
    finished = [s for s in statuses if s.is_terminal or s.state == "unreadable"]
    if not finished:
        st.caption("No finished runs yet.")
        return
    sweep_count = sum(1 for s in finished if s.sweep_id)
    search_col, status_col, sweep_col = st.columns([2, 2, 1], vertical_alignment="bottom")
    search = search_col.text_input("Search by name", key="runs-search")
    st.session_state.setdefault("runs-status", STATUS_CHOICES)
    chosen = status_col.multiselect("Status", STATUS_CHOICES, key="runs-status")
    show_sweeps = sweep_col.toggle(
        f"Show sweep runs ({sweep_count} hidden)" if sweep_count else "Show sweep runs",
        key="runs-sweeps",
    )
    visible = [
        s
        for s in finished
        if (show_sweeps or not s.sweep_id)
        and ui.badge_spec(s.state)[2] in chosen
        and search.lower() in s.name.lower()
    ]
    if not visible:
        st.caption("No runs match these filters.")
        st.button("Clear filters", on_click=clear_filters)
        return
    df = finished_frame(visible)
    order = [c for c in df.columns if c != "run_id" and (c != "Sweep" or show_sweeps)]
    event = st.dataframe(
        df,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        column_order=order,
        column_config=COLUMN_CONFIG,
        key="finished-table",
    )
    selected = [visible[i] for i in event.selection.rows if i < len(visible)]
    n = len(selected)
    all_done = n > 0 and all(s.state == "done" for s in selected)
    cols = st.columns([1, 1, 1, 1, 3], vertical_alignment="center")
    if cols[0].button("Open", disabled=n != 1, help="Select exactly one run."):
        ui.go_to("run", run=selected[0].run_id)
    if cols[1].button(
        "Compare", disabled=not (2 <= n <= 4 and all_done), help="Select 2–4 finished (Done) runs."
    ):
        ui.go_to("compare", runs=",".join(s.run_id for s in selected))
    if cols[2].button(
        "Clone",
        disabled=n != 1 or selected[0].state == "unreadable",
        help="Select exactly one run.",
    ):
        ui.go_to("new", clone=selected[0].run_id)
    with cols[3]:
        names = "\n".join(f"- {s.name}" for s in selected)

        def delete_selected() -> None:
            for status in selected:
                runs.delete_run(status.run_id, ui.runs_root())

        ui.confirm_popover(
            "Delete",
            f"Delete {n} run{'s' if n != 1 else ''}? Their folders are removed for good."
            f"\n\n{names}",
            "Delete",
            delete_selected,
            disabled=n == 0,
            help="Select runs to delete.",
            key="delete-runs",
        )
    cols[4].caption(f"{n} selected" if n else "Select rows to act on them.")
    broken = [s for s in finished if s.state == "unreadable"]
    if broken:
        ids = ", ".join(f"`{s.run_id}`" for s in broken)
        st.caption(
            f"{len(broken)} run folder{'s' if len(broken) > 1 else ''} couldn't be read: {ids}. "
            "Delete it or check `log.txt`."
        )


# --- page ----------------------------------------------------------------------------------

st.title("Runs")
st.caption("Runs execute one at a time, oldest first. Closing the tab does not stop them.")
all_runs = ui.list_runs()
if not all_runs:
    ui.no_runs_state()
    st.stop()
just_queued_banner()
st.subheader("In progress")
in_progress()
st.divider()
st.subheader("Finished")
finished_section(all_runs)
