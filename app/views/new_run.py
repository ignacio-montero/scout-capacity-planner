"""New run: choose settings, review them, queue a run (DESIGN_SYSTEM.md section 5.1).

State pattern: the form's values live in ``st.session_state["draft"]`` (a plain
dict that survives page switches), not only in the widgets, because Streamlit
forgets a widget's state once it isn't rendered. Each widget is initialised
from the draft and writes back to it in ``on_change``.

No ``st.form`` on purpose: every change reruns the page so the review card and
the validation messages are always current (this page does no computation).
The Run button only queues a folder through the run store (D-013).
"""

from __future__ import annotations

from typing import Any

import streamlit as st
import ui

from scout_planner import runs
from scout_planner.config import Params, params_to_yaml
from scout_planner.formatting import fmt_growth, fmt_when
from scout_planner.generate import month0_peak_load  # allow-listed in test_app_pages
from scout_planner.results import (
    list_published,
    load_published,
    normalise_sweep_frame,
    params_for_sweep_row,
)

DRAFT = "draft"
ADVANCED = "draft_advanced"
NAME = "draft_name"
ORIGIN = "draft_origin"
SOURCE = "draft_source"  # (label, Params) of the run or sweep row we started from
W_ADVANCED = "w:advanced"
W_NAME = "w:name"


def wkey(key: str) -> str:
    return f"w:{key}"


# --- loading a draft ------------------------------------------------------------------


def _push_widgets(draft: dict[str, Any]) -> None:
    """Copy draft values into widget state (allowed before the widgets render)."""
    for key, spec in ui.PARAM_UI.items():
        st.session_state[wkey(key)] = spec.to_display(draft[key])
    st.session_state[wkey(ui.SAME_GROWTH)] = draft[ui.SAME_GROWTH]
    st.session_state[W_ADVANCED] = st.session_state[ADVANCED]
    st.session_state[W_NAME] = st.session_state.get(NAME, "")


def load_draft(params: Params, origin: str, source: tuple[str, Params] | None) -> None:
    st.session_state[DRAFT] = ui.draft_from_params(params)
    st.session_state[ADVANCED] = ui.advanced_yaml_text(params)
    st.session_state[NAME] = ""
    st.session_state[ORIGIN] = origin
    st.session_state[SOURCE] = source
    _push_widgets(st.session_state[DRAFT])


def reset_to_defaults() -> None:
    load_draft(ui.defaults(), "defaults", None)


def _from_sweep_row(sweep_key: str, row_id: str) -> tuple[str, Params] | None:
    kind, _, name = sweep_key.partition(":")
    if kind != "published":
        return None
    match = [s for s in list_published(ui.published_dir()) if s.name == name]
    if not match:
        return None
    loaded = load_published(match[0].path)
    if loaded.value is None:
        return None
    df = normalise_sweep_frame(loaded.value, ui.defaults())
    rows = df[df["row_id"].astype(str) == row_id]
    if rows.empty:
        return None
    row = rows.iloc[0]
    params = params_for_sweep_row(row.to_dict(), ui.defaults(), ui.sweeps_dir() / f"{name}.yaml")
    return f"{row['name']}' from sweep '{name}", params


def ensure_draft() -> None:
    """Seed the draft on first visit, or when the URL asks for a different source."""
    clone = ui.query_value("clone")
    sweep, row = ui.query_value("sweep"), ui.query_value("row")
    wanted = f"clone:{clone}" if clone else f"sweep:{sweep}:{row}" if sweep and row else None
    if wanted is not None and wanted != st.session_state.get(ORIGIN):
        if clone:
            params = ui.run_params(clone) if _exists(clone) else None
            if params is None:
                st.warning(f"Run '{clone}' no longer exists; starting from defaults.")
                load_draft(ui.defaults(), wanted, None)
            else:
                name = runs.read_status(clone, ui.runs_root()).name
                load_draft(params, wanted, (name, params))
        else:
            found = _from_sweep_row(sweep or "", row or "")
            if found is None:
                st.warning("That sweep row couldn't be found; starting from defaults.")
                load_draft(ui.defaults(), wanted, None)
            else:
                load_draft(found[1], wanted, found)
    elif DRAFT not in st.session_state:
        reset_to_defaults()
    # Coming back from another page: widget state was dropped, the draft wasn't.
    draft = st.session_state[DRAFT]
    for key, spec in ui.PARAM_UI.items():
        if wkey(key) not in st.session_state:
            st.session_state[wkey(key)] = spec.to_display(draft[key])
    st.session_state.setdefault(wkey(ui.SAME_GROWTH), draft[ui.SAME_GROWTH])
    st.session_state.setdefault(W_ADVANCED, st.session_state[ADVANCED])
    st.session_state.setdefault(W_NAME, st.session_state.get(NAME, ""))


def _exists(run_id: str) -> bool:
    try:
        runs.read_status(run_id, ui.runs_root())
    except runs.RunNotFound:
        return False
    return True


# --- widgets ----------------------------------------------------------------------------


def _sync(key: str) -> None:
    spec = ui.PARAM_UI[key]
    st.session_state[DRAFT][key] = spec.to_stored(st.session_state[wkey(key)])


def _sync_same_growth() -> None:
    draft = st.session_state[DRAFT]
    draft[ui.SAME_GROWTH] = st.session_state[wkey(ui.SAME_GROWTH)]
    if draft[ui.SAME_GROWTH]:
        draft["capacity_plan.assumed_growth"] = draft["demand.actual_growth"]


def _sync_text(widget: str, target: str) -> None:
    st.session_state[target] = st.session_state[widget]


def _with_current(options: tuple[Any, ...], current: Any) -> list[Any]:
    """Grid options plus an off-grid value from a clone (shown, never silently snapped)."""
    values = list(options)
    if current not in values:
        values = sorted([*values, current])
    return values


def render_param(key: str) -> None:
    spec = ui.PARAM_UI[key]
    draft = st.session_state[DRAFT]
    k = wkey(key)
    disabled = bool(spec.depends_on) and not draft[spec.depends_on]
    if key == "capacity_plan.assumed_growth" and draft[ui.SAME_GROWTH]:
        disabled = True
        st.session_state[k] = spec.to_display(draft["demand.actual_growth"])
    common: dict[str, Any] = {
        "key": k,
        "help": spec.help,
        "on_change": _sync,
        "args": (key,),
        "disabled": disabled,
    }
    current = st.session_state[k]
    if spec.widget == "growth":
        st.select_slider(
            spec.label,
            options=_with_current(spec.options, current),
            format_func=fmt_growth,
            **common,
        )
    elif spec.widget == "pct":
        st.slider(
            spec.label,
            min_value=float(min(spec.min or 0, current)),
            max_value=float(max(spec.max or 100, current)),
            step=float(spec.step or 1),
            format="%.0f%%",
            **common,
        )
    elif spec.widget in ("int", "seeds"):
        lo, hi = int(min(spec.min or 0, current)), int(max(spec.max or 100, current))
        if spec.widget == "seeds":
            st.slider(spec.label, min_value=lo, max_value=hi, step=1, **common)
        else:
            st.number_input(spec.label, min_value=lo, max_value=hi, step=1, **common)
    elif spec.widget == "money":
        st.number_input(
            spec.label,
            min_value=float(min(spec.min or 0, current)),
            max_value=float(max(spec.max or 20_000, current)),
            step=float(spec.step or 100),
            format="%.0f",
            **common,
        )
    elif spec.widget == "premium":
        st.slider(
            spec.label,
            min_value=float(min(spec.min or 1, current)),
            max_value=float(max(spec.max or 3, current)),
            step=float(spec.step or 0.1),
            format="%.1fx",
            **common,
        )
    elif spec.widget == "toggle":
        st.toggle(spec.label, **common)
    elif spec.widget == "quantile":
        st.radio(
            spec.label,
            options=_with_current(spec.options, current),
            format_func=ui.quantile_label,
            horizontal=True,
            **common,
        )
    elif spec.widget == "radio":
        st.radio(
            spec.label,
            options=list(spec.options),
            format_func=lambda v, s=spec: s.option_labels.get(v, v),
            captions=list(spec.captions) or None,
            horizontal=spec.horizontal,
            **common,
        )
    else:  # pragma: no cover - a new widget kind must be added here
        raise ValueError(f"unknown widget kind {spec.widget!r}")


def render_group(group: str) -> None:
    draft = st.session_state[DRAFT]
    with st.container(border=True):
        st.markdown(f"**{ui.GROUP_TITLES[group]}**")
        for key, spec in ui.PARAM_UI.items():
            if spec.group != group:
                continue
            if key == "capacity_plan.assumed_growth":
                st.checkbox(
                    "Same as actual growth",
                    key=wkey(ui.SAME_GROWTH),
                    on_change=_sync_same_growth,
                    help="On: the agency predicts growth perfectly. Off: set a different "
                    "figure to test a wrong forecast.",
                )
            render_param(key)
            _derived_caption(key, draft)


def _derived_caption(key: str, draft: dict[str, Any]) -> None:
    salary = draft["cost.full_time_monthly_salary"]
    hourly = salary * 12 / (37.5 * 52)
    if key == "cost.full_time_monthly_salary":
        st.caption(f"≈ {hourly:,.0f} per hour")
    elif key == "cost.freelance_premium":
        st.caption(f"Freelance rate ≈ {hourly * draft['cost.freelance_premium']:,.0f} per hour")
    elif key == "cost.late_penalty":
        st.caption(
            "This number shapes the answer most: it decides whether paying for capacity "
            "beats paying for lateness."
        )


# --- review card --------------------------------------------------------------------------


def duplicate_of(params: Params) -> runs.RunStatus | None:
    """A finished run with identical settings (runs are deterministic: same results)."""
    target = params_to_yaml(params)
    for status in ui.list_runs():
        if status.state != "done":
            continue
        other = ui.run_params(status.run_id)
        if other is not None and params_to_yaml(other) == target:
            return status
    return None


def january_load(params: Params) -> float | None:
    """Expected January load (closed-form calibration formula, microseconds; no simulation)."""
    try:
        return month0_peak_load(params)
    except Exception:  # an odd parameter set: just leave the clause out
        return None


def render_review(check: ui.DraftCheck) -> None:
    source = st.session_state.get(SOURCE)
    defaults = ui.defaults()
    with st.container(border=True):
        st.subheader("What you are about to simulate")
        if check.params is not None:
            for sentence in ui.describe_params(check.params, january_load(check.params)):
                st.markdown(sentence)
            reference, ref_label = (
                (source[1], f"from '{source[0]}'") if source else (defaults, "from defaults")
            )
            changes = ui.changed_settings(check.params, reference)
            lines = "  \n".join(str(c) for c in changes) if changes else "Nothing changed"
            st.caption(f"**Changed {ref_label}**  \n{lines}")
        if check.errors:
            n = len(check.errors)
            st.error(
                f"Can't queue this run: {n} setting{'s are' if n > 1 else ' is'} invalid. "
                "Fix the lines below.\n\n" + "\n".join(f"- {e}" for e in check.errors)
            )
        for warning in check.warnings:
            st.warning(warning)
        for note in check.notes:
            st.info(note)
        if check.params is not None:
            twin = duplicate_of(check.params)
            if twin is not None:
                st.info(
                    f"A finished run with identical settings exists: '{twin.name}' "
                    f"({fmt_when(twin.created_at)}). Results would be identical."
                )
                if st.button("Open it", key="open-duplicate"):
                    ui.go_to("run", run=twin.run_id)

        placeholder = (
            ui.auto_run_name(check.params, defaults) if check.params is not None else "Run name"
        )
        st.text_input(
            "Run name",
            key=W_NAME,
            placeholder=placeholder,
            max_chars=ui.NAME_MAX,
            on_change=_sync_text,
            args=(W_NAME, NAME),
            help="Leave empty to use the suggested name.",
        )
        clicked = st.button("Run", type="primary", width="stretch", disabled=not check.ok)
        if check.errors:
            n = len(check.errors)
            st.caption(f"Fix {n} problem{'s' if n > 1 else ''} above to run.")
        elif check.params is not None:
            st.caption(ui.runtime_caption(check.params))
        if clicked and check.params is not None:
            name = (st.session_state.get(W_NAME) or "").strip() or placeholder
            try:
                run_id = runs.create_run(check.params, name, root=ui.runs_root())
            except Exception as exc:  # disk full, permissions...: say so, keep the draft
                st.error(f"Couldn't queue the run: {exc}. Nothing was created.")
            else:
                ui.go_to("runs", new=run_id)


# --- page ----------------------------------------------------------------------------------

st.title("New run")
st.caption("Choose the settings for one simulated year, then press Run.")
ensure_draft()

top_left, top_right = st.columns([4, 1], vertical_alignment="center")
source = st.session_state.get(SOURCE)
if source:
    top_left.info(f"Started from '{source[0]}' (cloned).", icon=":material/content_copy:")
top_right.button("Reset to defaults", on_click=reset_to_defaults, width="stretch")

for left, right in ui.FORM_ROWS:
    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        render_group(left)
    if right is not None:
        with col_right:
            render_group(right)

check = ui.validate_draft(
    st.session_state[DRAFT], st.session_state.get(ADVANCED, ""), ui.defaults()
)

with st.expander("Advanced settings (YAML)", expanded=bool(check.advanced_errors)):
    edit, full = st.tabs(["Edit", "Full parameter set"])
    with edit:
        st.caption("Settings not in the form above. Basic settings are set in the form, not here.")
        st.text_area(
            "Advanced settings",
            key=W_ADVANCED,
            height=400,
            on_change=_sync_text,
            args=(W_ADVANCED, ADVANCED),
        )
        for message in check.advanced_errors:
            st.error(message)
    with full:
        if check.params is not None:
            st.caption("Exactly what will be written to params.yaml.")
            st.code(params_to_yaml(check.params), language="yaml")
        else:
            st.caption("Fix the problems listed below to see the merged parameters.")

render_review(check)
