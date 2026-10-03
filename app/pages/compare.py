"""Compare: 2-4 finished runs side by side (DESIGN_SYSTEM.md section 5.4).

The first run picked is the baseline (A); every difference is measured
against it (*anchoring on a baseline*). Charts keep "colour = policy" true on
every page: runs are told apart by line dash and a letter at each line's end.
"""

from __future__ import annotations

from collections import Counter

import streamlit as st
import ui

from scout_planner import charts, runs
from scout_planner.formatting import fmt_cost_k, fmt_pct
from scout_planner.results import monthly_on_time

PICKER = "compare-picker"


def _on_pick() -> None:
    chosen = st.session_state[PICKER]
    if chosen:
        st.query_params["runs"] = ",".join(chosen)
    else:
        st.query_params.pop("runs", None)


def picker(done: list[runs.RunStatus]) -> list[str]:
    by_id = {s.run_id: s for s in done}
    raw = ui.query_value("runs")
    wanted = [r for r in (raw.split(",") if raw else []) if r]
    for missing in [r for r in wanted if r not in by_id]:
        st.warning(f"Run '{missing}' no longer exists and was removed from the comparison.")
    wanted = [r for r in wanted if r in by_id][:4]
    if st.session_state.get(PICKER) != wanted:
        st.session_state[PICKER] = wanted
    chosen = st.multiselect(
        "Runs to compare (2–4)",
        list(by_id),
        key=PICKER,
        max_selections=4,
        format_func=lambda i: ui.run_option_label(by_id[i].name, by_id[i].created_at),
        on_change=_on_pick,
    )
    st.caption("The first run is the baseline (A); others show differences.")
    return chosen


def chart(fig: object) -> None:
    st.plotly_chart(fig, theme="streamlit", width="stretch")


st.title("Compare")
done_runs = [s for s in ui.list_runs() if s.state == "done"]
if len(done_runs) < 2:
    st.info("You need at least two finished runs.")
    if st.button("Start a new run", type="primary"):
        ui.go_to("new")
    st.stop()

selected_ids = picker(done_runs)
if len(selected_ids) < 2:
    st.info("Pick at least two finished runs to compare.")
    st.caption("Tip: on Runs, select rows and press Compare.")
    st.stop()

statuses = {s.run_id: s for s in done_runs}
letters = ui.slot_letters(len(selected_ids))
params_list = [ui.run_params(r) for r in selected_ids]
summaries = [ui.run_summary(r).value for r in selected_ids]
if any(p is None for p in params_list) or any(s is None for s in summaries):
    broken = [
        statuses[r].name
        for r, p, s in zip(selected_ids, params_list, summaries, strict=True)
        if p is None or s is None
    ]
    st.error(f"Couldn't load the results of: {', '.join(broken)}. Pick other runs.")
    st.stop()

# Header cards ------------------------------------------------------------------------------
for col, letter, run_id, params, summary in zip(
    st.columns(len(selected_ids), gap="medium"),
    letters,
    selected_ids,
    params_list,
    summaries,
    strict=True,
):
    with col, st.container(border=True):
        st.page_link(
            ui.PAGES["run"],
            label=f"{letter} · {statuses[run_id].name}",
            query_params={"run": run_id},
        )
        st.caption(charts.policy_label(params.assignment.policy))
        st.markdown(ui.badge_markdown(ui.outcome_spec(summary.meets_target)))
        st.markdown(
            f"On time **{fmt_pct(summary.mean('on_time_rate'))}** · "
            f"total cost **{fmt_cost_k(summary.mean('cost_total'))}**"
        )

# What's different ---------------------------------------------------------------------------
st.subheader("What's different")
show_all = st.toggle("Show all settings", key="compare-show-all")
versions = [statuses[r].code_version for r in selected_ids]
diff = ui.param_diff(params_list, show_all=show_all, code_versions=versions)
if diff.empty:
    st.caption("No setting differs.")
else:
    st.dataframe(diff, hide_index=True)
note = ui.compare_note(params_list, versions)
if note:
    st.info(note)
if len(set(versions)) > 1:
    st.caption("Different code versions: results may differ for reasons other than the settings.")

# Results ------------------------------------------------------------------------------------
st.subheader("Results")
st.dataframe(ui.metric_table(summaries, baseline=0), hide_index=True)

# Charts -------------------------------------------------------------------------------------
st.subheader("Charts")
show_range = st.toggle("Show range over repeats", key="compare-range")
policies = {letter: p.assignment.policy for letter, p in zip(letters, params_list, strict=True)}
weekly = {
    letter: ui.run_table(r, "weekly").value for letter, r in zip(letters, selected_ids, strict=True)
}
if all(df is not None for df in weekly.values()):
    chart(charts.fig_backlog(weekly, policies, show_range=show_range))  # type: ignore[arg-type]
else:
    st.warning(
        "Couldn't load results/weekly.parquet for every run, so the backlog chart is hidden."
    )
monthly = {}
for letter, run_id in zip(letters, selected_ids, strict=True):
    outcomes, raw = (
        ui.run_table(run_id, "outcomes").value,
        ui.run_table(run_id, "raw_requests").value,
    )
    if outcomes is not None and raw is not None:
        monthly[letter] = monthly_on_time(outcomes, raw)
if len(monthly) == len(letters):
    targets = [p.sim.target_on_time for p in params_list]
    chart(charts.fig_on_time_by_month(monthly, policies, targets, show_range=show_range))
else:
    st.warning(
        "Couldn't load the per-request results for every run, so the on-time chart is hidden."
    )
parts = {
    f"{letter} · {statuses[r].name}": s.cost_parts()
    for letter, r, s in zip(letters, selected_ids, summaries, strict=True)
}
chart(charts.fig_cost_split(parts, mode="absolute"))
if max(Counter(policies.values()).values()) >= 3:
    st.caption(
        "Several runs share a policy colour; tell them apart by line style and the letter at "
        "each line's end."
    )
