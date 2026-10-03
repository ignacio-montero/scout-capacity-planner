"""Run detail: one run's results, or its live progress (DESIGN_SYSTEM.md section 5.3).

Reading order: takeaway sentence -> KPI cards -> cost split -> tabs of charts
-> raw parameters. The selected run lives in the URL (``?run=<id>``), so a
reload or a shared link lands on the same run.
"""

from __future__ import annotations

import streamlit as st
import ui

from scout_planner import charts, runs
from scout_planner.config import Params, params_to_yaml
from scout_planner.formatting import fmt_count, fmt_duration, fmt_pct, fmt_when
from scout_planner.results import (
    forecast_totals,
    monthly_demand,
    monthly_on_time,
    seeds_late_reports,
)

PICKER = "run-picker"
FILE_NAMES = {
    "seeds": "results/seeds.parquet",
    "weekly": "results/weekly.parquet",
    "outcomes": "results/requests.parquet",
    "forecast": "forecast.parquet",
    "capacity_plan": "capacity_plan.parquet",
    "hiring_plan": "hiring_plan.parquet",
    "raw_requests": "raw/requests.parquet",
}


def _on_pick() -> None:
    st.query_params["run"] = st.session_state[PICKER]


def pick_run(statuses: list[runs.RunStatus]) -> runs.RunStatus | None:
    ids = [s.run_id for s in statuses]
    by_id = {s.run_id: s for s in statuses}
    wanted = ui.query_value("run")
    if wanted and wanted not in by_id:
        st.error(f"Run '{wanted}' doesn't exist; it may have been deleted.")
        st.page_link(ui.PAGES["runs"], label="Go to Runs", icon=":material/list:")
        return None
    if not wanted:
        done = [s for s in statuses if s.state == "done"]
        wanted = (done or statuses)[0].run_id
    if st.session_state.get(PICKER) != wanted:
        st.session_state[PICKER] = wanted
    st.selectbox(
        "Run",
        ids,
        key=PICKER,
        format_func=lambda i: ui.run_option_label(by_id[i].name, by_id[i].created_at),
        on_change=_on_pick,
    )
    return by_id[wanted]


def run_again(status: runs.RunStatus, params: Params | None) -> None:
    if params is None:
        return
    if st.button("Run again", type="primary", key="run-again"):
        try:
            new_id = runs.create_run(params, f"{status.name} (again)"[:200], root=ui.runs_root())
        except Exception as exc:
            st.error(f"Couldn't queue the run: {exc}. Nothing was created.")
        else:
            ui.go_to("runs", new=new_id)


def header(status: runs.RunStatus, state: str, params: Params | None) -> None:
    left, right = st.columns([3, 1])
    with left:
        st.title(status.name)
        summary = ui.run_summary(status.run_id).value if state == "done" else None
        badges = ui.badge_markdown(ui.badge_spec(state))
        if summary is not None:
            badges += " " + ui.badge_markdown(ui.outcome_spec(summary.meets_target))
        st.markdown(badges)
        bits = [f"`{status.run_id}`"]
        if status.finished_at:
            bits.append(f"finished {fmt_when(status.finished_at)}")
        if status.duration_s is not None:
            bits.append(f"took {fmt_duration(status.duration_s)}")
        bits.append(f"code {status.code_version}")
        st.caption(" · ".join(bits))
    with right:
        if params is not None and st.button("Clone to new run", width="stretch"):
            ui.go_to("new", clone=status.run_id)
        if state == "done" and st.button("Compare with…", width="stretch"):
            ui.go_to("compare", runs=status.run_id)
        if params is not None:
            st.download_button(
                "Download parameters",
                params_to_yaml(params),
                file_name=f"{status.run_id}-params.yaml",
                mime="text/yaml",
                width="stretch",
            )
        if state in ("queued", "running"):
            ui.confirm_popover(
                "Cancel",
                "Stop this run? Work done so far is discarded.",
                "Stop run",
                lambda: runs.request_cancel(status.run_id, ui.runs_root()),
                key=f"detail-cancel-{status.run_id}",
            )


def parameters_section(params: Params | None, run_id: str, seeds_table: bool = False) -> None:
    if params is None:
        st.warning("Couldn't load params.yaml for this run.")
        return
    only_changed = st.toggle("Only show changed settings", value=True, key="only-changed")
    table = ui.param_table(params, ui.defaults())
    if only_changed:
        table = table[table["Changed"]]
    if table.empty:
        st.caption("Every setting is at its default.")
    else:
        st.dataframe(
            table,
            hide_index=True,
            column_config={"Changed": st.column_config.CheckboxColumn(disabled=True)},
        )
    with st.expander("Full params.yaml"):
        st.code(params_to_yaml(params), language="yaml")
    if seeds_table:
        summary = ui.run_summary(run_id).value
        with st.expander("Solver diagnostics"):
            diag = ui.diagnostics_table(summary.diagnostics if summary else {})
            if diag.empty:
                st.caption("This run recorded no solver diagnostics.")
            else:
                st.caption(
                    "How the assignment rounds went, totalled over repeats. Time-limit stops "
                    "and fallbacks mean the optimiser didn't plan those rounds itself."
                )
                st.dataframe(diag, hide_index=True)
        seeds = ui.run_table(run_id, "seeds")
        with st.expander("Per-repeat results"):
            if seeds.value is None:
                st.warning(f"Couldn't load {FILE_NAMES['seeds']} for this run.")
            else:
                st.dataframe(seeds.value, hide_index=True)


def _missing(table: str) -> None:
    st.warning(f"Couldn't load {FILE_NAMES[table]} for this run.")


def chart(fig: object) -> None:
    st.plotly_chart(fig, theme="streamlit", width="stretch")


def done_body(status: runs.RunStatus, params: Params | None) -> None:
    run_id = status.run_id
    summary_l = ui.run_summary(run_id)
    if summary_l.value is None or params is None:
        st.warning(
            f"Couldn't load the results of this run: {summary_l.problem or 'no parameters'}."
        )
        parameters_section(params, run_id)
        return
    summary = summary_l.value
    policy = params.assignment.policy
    with st.spinner("Loading results…"):
        tables = {name: ui.run_table(run_id, name) for name in FILE_NAMES}
    st.subheader(ui.run_takeaway(summary, params))
    seeds = tables["seeds"].value
    ui.kpi_row(summary, params, seeds_late_reports(seeds) if seeds is not None else None)
    if warning := ui.diagnostics_warning(summary.diagnostics):
        st.warning(warning, icon=":material/warning:")
    chart(charts.fig_cost_split({status.name: summary.cost_parts()}, mode="share"))

    service, team, demand, parameters = st.tabs(
        ["Service", "Team & cost", "Demand & hiring plan", "Parameters"]
    )
    with service:
        weekly, outcomes = tables["weekly"].value, tables["outcomes"].value
        if weekly is not None:
            chart(charts.fig_backlog({"run": weekly}, {"run": policy}))
        else:
            _missing("weekly")
        if outcomes is not None:
            monthly = monthly_on_time(outcomes)
            chart(
                charts.fig_on_time_by_month(
                    {"run": monthly}, {"run": policy}, params.sim.target_on_time
                )
            )
        else:
            _missing("outcomes")
        if outcomes is not None:
            chart(
                charts.fig_turnaround_hist(
                    outcomes,
                    policy,
                    promise_days=params.demand.turnaround_days,
                    p90=summary.mean("p90_turnaround_days"),
                    urgent_days=(
                        params.demand.urgent_turnaround_days if params.demand.urgent_share else None
                    ),
                )
            )
        at_risk, n_req = summary.mean("n_at_risk_day_one"), summary.mean("n_requests")
        if at_risk is not None and n_req:
            st.caption(
                f"{fmt_count(at_risk)} requests ({fmt_pct(at_risk / n_req)}) were at risk from day "
                "one: no suitable match before their due date. They count in the on-time rate."
            )
        censored = summary.mean("n_censored")
        if censored:
            st.caption(
                f"{fmt_count(censored)} requests arrived too late in the year to be due before "
                "it ends, so they are not scored either way (their outcome is unknown)."
            )
    with team:
        weekly, hiring = tables["weekly"].value, tables["hiring_plan"].value
        if weekly is not None:
            chart(
                charts.fig_team_size(
                    weekly, hiring_plan=hiring if params.team.follow_hiring_plan else None
                )
            )
        else:
            _missing("weekly")
        stats = {
            kind: (s.mean, s.min, s.max) if (s := summary.stat(f"util_{kind}")) else None
            for kind in ("full_time", "freelance")
        }
        chart(
            charts.fig_utilisation(
                stats, target_utilisation=params.capacity_plan.target_utilisation
            )
        )
        ft_col, fl_col = st.columns(2)
        ft_col.metric("Full-time hires", fmt_count(summary.mean("n_hires_full_time")), border=True)
        fl_col.metric("Freelance hires", fmt_count(summary.mean("n_hires_freelance")), border=True)
    with demand:
        raw, forecast = tables["raw_requests"].value, tables["forecast"].value
        if raw is not None and forecast is not None:
            chart(
                charts.fig_demand_vs_forecast(
                    monthly_demand(raw, tables["outcomes"].value),
                    forecast_totals(forecast),
                    policy=policy,
                    quantile=params.capacity_plan.quantile,
                )
            )
        else:
            _missing("raw_requests" if raw is None else "forecast")
        plan = tables["capacity_plan"].value
        if plan is not None:
            chart(charts.fig_capacity_heatmap(plan))
        else:
            _missing("capacity_plan")
        st.markdown("**Hiring plan**")
        hiring = tables["hiring_plan"].value
        if not params.team.follow_hiring_plan:
            st.info(
                "Planned hires did not join in this run (setting 'Planned hires actually join' "
                "is off)."
            )
        if hiring is None:
            _missing("hiring_plan")
        elif hiring.empty:
            st.caption("The plan made no hires: the starting team covers the assumed demand.")
        else:
            st.dataframe(
                hiring.assign(
                    month_to_act=hiring["month_to_act"].dt.strftime("%b %Y"),
                    joins_month=hiring["joins_month"].dt.strftime("%b %Y"),
                    hire_type=hiring["hire_type"].map(charts.EMPLOYMENT_LABELS),
                ).rename(
                    columns={
                        "month_to_act": "Start recruiting",
                        "joins_month": "Joins",
                        "skill_type": "Skill",
                        "hire_type": "Hire type",
                        "count": "Count",
                        "reason": "Reason",
                    }
                ),
                hide_index=True,
            )
    with parameters:
        parameters_section(params, run_id, seeds_table=True)


def live_body(run_id: str) -> None:
    """Running / cancelling: refresh every 2 s; rerun the page when the run ends."""

    @st.fragment(run_every="2s")
    def live() -> None:
        try:
            status = runs.read_status(run_id, ui.runs_root())
        except runs.RunNotFound:
            st.rerun(scope="app")
            return
        if status.is_terminal:
            st.rerun(scope="app")
        state = ui.run_display_state(status)
        with st.status(f"Running: {status.message or 'working'}", state="running", expanded=True):
            ui.run_progress(status)
        if state == "cancelling":
            st.caption("Stops at the end of the current simulated week.")
        if not runs.worker_alive(ui.runs_root()):
            ui.worker_down_warning()

    live()


def body(status: runs.RunStatus, state: str, params: Params | None) -> None:
    root = ui.runs_root()
    if state in ("queued", "cancelling") and status.state == "queued":
        queued = [s.run_id for s in runs.queued_runs(root)]
        running = [s for s in ui.list_runs() if s.state == "running"]
        if status.run_id in queued:
            ahead = f" It starts when '{running[0].name}' finishes." if running else ""
            st.info(f"Waiting in line: #{queued.index(status.run_id) + 1}.{ahead}")
        else:
            st.info("Cancel requested: this run will not start.")
        if not runs.worker_alive(root):
            ui.worker_down_warning()
        parameters_section(params, status.run_id)
    elif status.state == "running":
        live_body(status.run_id)
        parameters_section(params, status.run_id)
    elif state == "failed":
        if status.error == "interrupted":
            st.error(
                "The background worker stopped while this run was in progress, for example "
                "because the app was closed. Its settings are fine; run it again."
            )
        else:
            st.error(f"This run failed: {status.error or 'unknown error'}.")
        with st.expander("Log (last 50 lines)"):
            st.code(runs.read_log_tail(status.run_id, 50, root) or "(empty log)")
        run_again(status, params)
        parameters_section(params, status.run_id)
    elif state == "cancelled":
        detail = status.message.capitalize() if status.message else "Cancelled"
        st.info(f"{detail}. No results were kept.")
        run_again(status, params)
        parameters_section(params, status.run_id)
    elif state == "unreadable":
        st.error(
            f"This run's status file couldn't be read ({status.error}). Delete it on Runs or "
            "check its log.txt."
        )
    else:
        done_body(status, params)


# --- page ----------------------------------------------------------------------------------

all_runs = ui.list_runs()
if not all_runs:
    st.title("Run detail")
    ui.no_runs_state()
    st.stop()
selected = pick_run(all_runs)
if selected is None:
    st.stop()
display = ui.run_display_state(selected)
run_params = ui.run_params(selected.run_id) if selected.state != "unreadable" else None
header(selected, display, run_params)
st.divider()
body(selected, display, run_params)
