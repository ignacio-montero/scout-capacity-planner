"""Streamlit components shared by every page (DESIGN_SYSTEM.md section 8).

The *pure* half (formatters, labels, validation, tables, takeaways) lives in
``viewmodel.py`` and is re-exported here, so pages write ``ui.fmt_pct(...)``.
This module adds what needs Streamlit: navigation, badges, the sidebar queue
indicator, empty states, cached loaders.

Paths: the runs folder is ``data/runs`` unless ``SCOUT_RUNS_ROOT`` is set, and
published sweeps live in ``data/published`` unless ``SCOUT_PUBLISHED_DIR`` is
set. Tests and the demo point both at throwaway folders.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import streamlit as st
from viewmodel import *  # noqa: F403  (re-export: the pure helpers are part of ui's API)
from viewmodel import (
    badge_spec,
    kpi_cards,
    outcome_spec,
    stage_line,
    time_left,
)

from scout_planner import runs
from scout_planner.config import Params, load_default_params
from scout_planner.formatting import fmt_pct
from scout_planner.results import Loaded, Summary, load_summary, load_table

REPO_ROOT = runs.REPO_ROOT
PAGES: dict[str, str] = {
    "sweep": "pages/sweep.py",
    "new": "pages/new_run.py",
    "runs": "pages/runs.py",
    "run": "pages/run_detail.py",
    "compare": "pages/compare.py",
}
TOAST_KEY = "_pending_toasts"
KNOWN_STATES_KEY = "_known_states"


# --- paths -----------------------------------------------------------------------------


def runs_root() -> Path:
    return Path(os.environ.get("SCOUT_RUNS_ROOT") or runs.DEFAULT_ROOT)


def published_dir() -> Path:
    return Path(os.environ.get("SCOUT_PUBLISHED_DIR") or REPO_ROOT / "data" / "published")


def sweeps_dir() -> Path:
    return REPO_ROOT / "config" / "sweeps"


# --- navigation --------------------------------------------------------------------------


def go_to(page: str, **query: str) -> None:
    """Switch page and put the selection in the URL (``?run=...``), so reloads keep it.

    Uses ``st.switch_page(..., query_params=...)`` directly: recent Streamlit
    carries the query across the switch, so no session-state hand-off is needed.
    """
    st.switch_page(PAGES[page], query_params={k: v for k, v in query.items() if v})


def query_value(name: str) -> str | None:
    value = st.query_params.get(name)
    return value or None


# --- cached loaders -------------------------------------------------------------------
# Results are immutable once a run is done, so caching by (run, file mtime) is
# safe: a new mtime means a new cache entry. params.yaml never changes.


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, max_entries=512)
def _summary_cached(root: str, run_id: str, mtime: float) -> Loaded[Summary]:
    return load_summary(runs.run_dir(run_id, root))


def run_summary(run_id: str, root: Path | None = None) -> Loaded[Summary]:
    base = root or runs_root()
    path = runs.run_dir(run_id, base) / "summary.json"
    return _summary_cached(str(base), run_id, _mtime(path))


@st.cache_data(show_spinner=False, max_entries=128)
def _table_cached(root: str, run_id: str, table: str, mtime: float) -> Loaded[Any]:
    return load_table(runs.run_dir(run_id, root), table)


def run_table(run_id: str, table: str, root: Path | None = None) -> Loaded[Any]:
    from scout_planner.results import TABLES

    base = root or runs_root()
    path = runs.run_dir(run_id, base) / TABLES[table].path
    return _table_cached(str(base), run_id, table, _mtime(path))


@st.cache_data(show_spinner=False, max_entries=512)
def _params_cached(root: str, run_id: str, mtime: float) -> Params | None:
    try:
        return runs.read_params(run_id, root)
    except Exception:
        return None


def run_params(run_id: str, root: Path | None = None) -> Params | None:
    base = root or runs_root()
    return _params_cached(str(base), run_id, _mtime(runs.run_dir(run_id, base) / "params.yaml"))


@st.cache_data(show_spinner=False)
def defaults() -> Params:
    return load_default_params()


def list_runs() -> list[runs.RunStatus]:
    """Every run, newest first. Not cached: it reads one small JSON per folder and
    must always be fresh."""
    return runs.list_runs(runs_root())


def run_display_state(status: runs.RunStatus) -> str:
    if status.state == "unreadable":
        return "unreadable"
    try:
        flag = runs.cancel_requested(status.run_id, runs_root())
    except runs.RunNotFound:
        flag = False
    return runs.display_state(status, flag)


# --- small components ---------------------------------------------------------------


def status_badge(state: str) -> None:
    color, icon, label = badge_spec(state)
    st.badge(label, icon=icon, color=color)  # type: ignore[arg-type]


def outcome_badge(summary: Summary | None) -> None:
    spec = outcome_spec(summary.meets_target if summary else None)
    if spec is not None:
        color, icon, label = spec
        st.badge(label, icon=icon, color=color)  # type: ignore[arg-type]


def empty_state(
    title: str, body: str, actions: Sequence[tuple[str, str, bool]] = (), caption: str = ""
) -> None:
    """Empty state as onboarding: what goes here, why it's empty, the next action.

    ``actions``: ``(label, page, primary)``; each becomes a button that moves there.
    """
    with st.container(border=True):
        st.subheader(title)
        st.write(body)
        if actions:
            cols = st.columns(len(actions), gap="medium")
            for col, (label, page, primary) in zip(cols, actions, strict=True):
                if col.button(
                    label, type="primary" if primary else "secondary", key=f"empty-{page}-{label}"
                ):
                    go_to(page)
        if caption:
            st.caption(caption)


def no_runs_state() -> None:
    empty_state(
        "No runs yet",
        "A run simulates one year of the agency with the settings you choose. It takes from "
        "a few seconds to a few minutes.",
        [("Start a new run", "new", True), ("See the headline results", "sweep", False)],
        "Or from a terminal: `make sweep SWEEP=quick`",
    )


def confirm_popover(
    label: str,
    message: str,
    confirm_label: str,
    on_confirm: Callable[[], None],
    *,
    disabled: bool = False,
    help: str | None = None,
    key: str,
) -> None:
    """Destructive action: confirm in place, naming what will be lost."""
    with st.popover(label, disabled=disabled, help=help):
        st.write(message)
        if st.button(confirm_label, type="primary", key=f"{key}-confirm"):
            on_confirm()
            st.rerun()


def kpi_row(summary: Summary, params: Params, late: Any = None) -> None:
    cards = kpi_cards(summary, params, late)
    for col, card in zip(st.columns(len(cards)), cards, strict=True):
        with col:
            st.metric(
                card.label,
                card.value,
                card.delta,
                delta_color=card.delta_color,  # type: ignore[arg-type]
                help=card.help,
                border=True,
            )
            st.caption(card.caption)


def run_progress(status: runs.RunStatus) -> None:
    """Progress bar with words, the stage line, elapsed time and an estimate."""
    from scout_planner.formatting import fmt_duration

    pct = f"{round(status.progress * 100)}%"
    message = status.message or "working"
    st.progress(min(max(status.progress, 0.0), 1.0), text=f"{pct} · {message}")
    bits = [stage_line(status.stage)]
    if status.duration_s is not None:
        bits.append(fmt_duration(status.duration_s))
    left = time_left(status.progress, status.duration_s)
    if left:
        bits.append(left)
    st.caption(" · ".join(bits))


def worker_down_warning() -> None:
    st.warning(
        "Background worker is not running, so queued runs will not start. Restart with `make app`.",
        icon=":material/warning:",
    )


# --- sidebar queue indicator (every page) ----------------------------------------------


def _finished_toasts(statuses: Sequence[runs.RunStatus]) -> list[str]:
    """Runs that were queued/running at the last look and are final now."""
    known: dict[str, str] = st.session_state.setdefault(KNOWN_STATES_KEY, {})
    toasts = []
    first_look = not known
    for status in statuses:
        before = known.get(status.run_id)
        known[status.run_id] = status.state
        if first_look or before not in ("queued", "running") or not status.is_terminal:
            continue
        if status.state == "done":
            summary = run_summary(status.run_id)
            rate = summary.value.mean("on_time_rate") if summary.value else None
            extra = f": {fmt_pct(rate)} on time" if rate is not None else ""
            toasts.append(f"'{status.name}' finished{extra}")
        elif status.state == "failed":
            first_line = (status.error or "unknown error").splitlines()[0]
            toasts.append(f"'{status.name}' failed: {first_line}")
        else:
            toasts.append(f"'{status.name}' was cancelled")
    return toasts


@st.fragment(run_every="5s")
def queue_indicator() -> None:
    """Running run + progress, queue length, worker liveness; toasts when a run ends."""
    root = runs_root()
    statuses = runs.list_runs(root)
    for message in _finished_toasts(statuses):
        st.toast(message)
    running = [s for s in statuses if s.state == "running"]
    queued = runs.queued_runs(root)
    if running:
        current = running[0]
        st.markdown(f"**Running: {current.name}**")
        st.progress(min(max(current.progress, 0.0), 1.0))
        bits = [current.message or "working"]
        left = time_left(current.progress, current.duration_s)
        if left:
            bits.append(left)
        if queued:
            bits.append(f"{len(queued)} more queued")
        st.caption(" · ".join(bits))
    elif queued:
        st.caption(f"{len(queued)} queued, waiting to start.")
    else:
        st.caption("No runs in progress.")
    if (running or queued) and not runs.worker_alive(root):
        worker_down_warning()
    if running or queued:
        st.page_link(PAGES["runs"], label="View runs", icon=":material/list:")
