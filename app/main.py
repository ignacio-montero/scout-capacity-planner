"""Entry point of the simulator app: ``streamlit run app/main.py`` (or ``make app``).

Declares the five pages with ``st.navigation`` (DESIGN_SYSTEM.md section 1.1),
draws the sidebar header and the live queue indicator, then runs the page the
URL points at. Every page re-executes from the top on each interaction
(Streamlit's rerun model); state that must survive a reload lives in the URL.

The app never computes a simulation (D-013): "Run" only writes a queued run
folder through ``scout_planner.runs``; the background worker does the rest.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:  # pages import `ui` / `viewmodel` from this folder
    sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402
import ui  # noqa: E402

st.set_page_config(page_title="Scout Capacity Planner", layout="wide")

pages = [
    st.Page(
        ui.PAGES["sweep"],
        title="Sweep results",
        icon=":material/scatter_plot:",
        url_path="sweep",
        default=True,
    ),
    st.Page(ui.PAGES["new"], title="New run", icon=":material/tune:", url_path="new"),
    st.Page(ui.PAGES["runs"], title="Runs", icon=":material/list:", url_path="runs"),
    st.Page(ui.PAGES["run"], title="Run detail", icon=":material/insights:", url_path="run"),
    st.Page(
        ui.PAGES["compare"], title="Compare", icon=":material/compare_arrows:", url_path="compare"
    ),
]
current = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.markdown("**Scout Capacity Planner**")
    st.caption("Fictional agency · synthetic data")
    ui.queue_indicator()

current.run()
