"""M5 acceptance, end to end: the app queues, the worker executes, the app shows the result.

This is the only test that crosses *every* boundary of the simulator with the
real code on both sides: Streamlit form -> run store (folder + status.json) ->
background worker (child process, real ``run_pipeline``) -> results on disk ->
Run detail / Runs / Compare pages. Existing tests cover each hop alone (pages
on hand-written demo folders, the worker with fake pipelines); this one checks
that the hops fit together.

The browser is simulated with Streamlit's ``AppTest`` (no real browser). The
AppTest session is thrown away before the worker runs, which is how "closing
the browser tab does not stop a run" is exercised: the run's lifecycle has no
link to any Streamlit session.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from acceptance_helpers import check_run_folder
from scout_planner import pipeline, runs
from scout_planner.worker import Worker
from sim_fixtures import tiny_params

MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"
TIMEOUT = 90

pytestmark = pytest.mark.slow


def _open(page: str, **query: str) -> AppTest:
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    # Run once first: st.navigation registers pages (with their url_path) only on
    # a run; before that, switch_page cannot match a views/ page by file name.
    at.run()
    at.switch_page(page)
    for key, value in query.items():
        at.query_params[key] = value
    return at.run()


def _clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _button(at: AppTest, label: str):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button {label!r}; have {[b.label for b in at.button]}"
    return matches[0]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setenv("SCOUT_RUNS_ROOT", str(root))
    monkeypatch.setenv("SCOUT_PUBLISHED_DIR", str(tmp_path / "published"))
    return root


def test_form_to_queue_to_worker_to_run_detail(env: Path) -> None:
    root = env
    # 1. New run: basic widgets + advanced YAML for a tiny, fast run
    at = _open("views/new_run.py")
    _clean(at)
    at.radio(key="w:assignment.policy").set_value("edf").run()
    at.number_input(key="w:team.full_time_count").set_value(8).run()
    at.number_input(key="w:team.freelance_count").set_value(4).run()
    at.slider(key="w:sim.seeds").set_value(1).run()
    at.text_area(key="w:advanced").input("sim:\n  months: 1\n").run()
    at.text_input(key="w:name").input("E2E tiny").run()
    _clean(at)

    # 2. Run returns immediately: the folder is queued, nothing was computed
    started = time.perf_counter()
    _button(at, "Run").click().run()
    assert time.perf_counter() - started < 15, "Run must only queue, never compute (D-013)"
    _clean(at)
    (status,) = runs.list_runs(root)
    folder = runs.run_dir(status.run_id, root)
    assert status.state == "queued" and status.name == "E2E tiny"
    assert not (folder / "summary.json").exists() and not (folder / "raw").exists()
    params = runs.read_params(status.run_id, root)
    assert params.sim.months == 1 and params.team.full_time_count == 8
    assert params.assignment.policy == "edf" and params.sim.seeds == 1

    # 3. "Close the tab": drop the session, then the worker executes the run on its own
    del at
    assert Worker(root=root, poll_s=0.1).drain() == [status.run_id]
    done = runs.read_status(status.run_id, root)
    assert done.state == "done", runs.read_log_tail(status.run_id, root=root)
    assert done.progress == 1.0 and done.started_at and done.finished_at
    check_run_folder(folder, params)

    # 4. Run detail shows every PRD M5 element for the real result
    detail = _open("views/run_detail.py", run=status.run_id)
    _clean(detail)
    takeaway = detail.subheader[0].value
    assert takeaway.startswith("On time ") and "95% target" in takeaway  # on-time vs 95%
    assert [m.label for m in detail.metric][:4] == [
        "On-time rate",
        "Turnaround P90",
        "Total cost",
        "Late reports",
    ]
    assert [t.label for t in detail.tabs] == [
        "Service",
        "Team & cost",
        "Demand & hiring plan",
        "Parameters",
    ]
    # cost split, backlog, on time by month, turnaround, team, utilisation,
    # demand vs forecast, capacity heatmap
    assert len(detail.get("plotly_chart")) >= 8
    assert any(m.value == "**Hiring plan**" for m in detail.markdown)
    assert not any("Couldn't load" in w.value for w in detail.warning)
    frames = [d.value for d in detail.dataframe]
    hiring = [f for f in frames if "Start recruiting" in f.columns]
    no_hires = any("made no hires" in c.value for c in detail.caption)
    assert hiring or no_hires, "the hiring table (or its empty state) is missing"
    (changed,) = [f for f in frames if "Setting" in f.columns]  # the run's parameters
    shown = dict(zip(changed["Setting"], changed["Value"], strict=True))
    assert shown["sim.months"] == "1" and shown["Full-time scouts"] == "8"

    # 5. Runs page lists it as finished
    listing = _open("views/runs.py")
    _clean(listing)
    assert any("E2E tiny" in str(df.value.to_numpy()) for df in listing.dataframe)


def test_queue_runs_one_at_a_time_oldest_first_with_the_real_pipeline(env: Path) -> None:
    root = env
    ids = [
        runs.create_run(tiny_params(sim__months=1, sim__seeds=1, sim__seed=s), f"q{s}", root=root)
        for s in (1, 2, 3)
    ]
    assert Worker(root=root, poll_s=0.1).drain() == ids  # oldest first
    statuses = [runs.read_status(i, root) for i in ids]
    assert {s.state for s in statuses} == {"done"}
    for earlier, later in zip(statuses, statuses[1:], strict=False):
        assert earlier.finished_at <= later.started_at  # never two at once


def _done_run(root: Path, name: str, **overrides) -> str:
    params = tiny_params(sim__months=1, sim__seeds=1, **overrides)
    run_id = runs.create_run(params, name, root=root)
    runs.update_status(run_id, root=root, state="running")
    pipeline.run_pipeline(
        params, runs.run_dir(run_id, root), lambda *_: None, lambda: False, max_workers=1
    )
    runs.update_status(run_id, root=root, state="done")
    return run_id


def test_compare_shows_up_to_four_real_runs_side_by_side(env: Path) -> None:
    root = env
    variants = [
        ("EDF", {}),
        ("FCFS", {"assignment__policy": "fcfs"}),
        ("Pre-screen", {"automation__enabled": True}),
        ("Bigger team", {"team__full_time_count": 12}),
        ("Fifth", {"demand__actual_growth": 1.0}),
    ]
    ids = [_done_run(root, name, **ov) for name, ov in variants]
    at = _open("views/compare.py", runs=",".join(ids))  # five asked, four shown
    _clean(at)
    metric_table = at.dataframe[1].value
    assert list(metric_table.columns) == ["Metric", "Better is", "A", "B", "C", "D"]
    diff = at.dataframe[0].value  # parameter diff: only what differs
    flat = " ".join(map(str, diff.to_numpy().ravel()))
    for differing in ("Full-time scouts", "Who does what", "Video pre-screen"):
        assert differing in flat
    assert "First come, first served" in flat
    assert "growth" not in flat.lower(), "the fifth run (growth 1x) must not be compared"
    assert "Starting workload" not in flat, "settings shared by all runs are not listed"
