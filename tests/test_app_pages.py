"""Streamlit AppTest smoke tests: every page renders on demo data and on an empty root,
and the New run page queues a run through the run store (never computing one, D-013).

AppTest runs the real script in-process, without a browser. The runs root and
published folder point at temp folders through ``SCOUT_RUNS_ROOT`` /
``SCOUT_PUBLISHED_DIR``, so real data is never touched.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ui_fixtures
from scout_planner import runs

MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"
TIMEOUT = 60
PAGES = [
    "pages/sweep.py",
    "pages/new_run.py",
    "pages/runs.py",
    "pages/run_detail.py",
    "pages/compare.py",
]


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    base = tmp_path_factory.mktemp("app")
    root = base / "runs"
    ids = ui_fixtures.write_demo_runs(root, include_broken=True)
    published = base / "published"
    ui_fixtures.write_demo_published(published)
    return {"root": root, "published": published, "ids": ids}


@pytest.fixture
def demo_env(demo, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setenv("SCOUT_RUNS_ROOT", str(demo["root"]))
    monkeypatch.setenv("SCOUT_PUBLISHED_DIR", str(demo["published"]))
    return demo


@pytest.fixture
def empty_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setenv("SCOUT_RUNS_ROOT", str(root))
    monkeypatch.setenv("SCOUT_PUBLISHED_DIR", str(tmp_path / "published"))
    return root


def open_page(page: str, **query: str) -> AppTest:
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.switch_page(page)
    for key, value in query.items():
        at.query_params[key] = value
    return at.run()


def no_exception(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def button(at: AppTest, label: str):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button {label!r}; have {[b.label for b in at.button]}"
    return matches[0]


# --- every page renders --------------------------------------------------------------


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_on_demo_data(demo_env, page: str) -> None:
    at = open_page(page)
    no_exception(at)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_on_empty_root(empty_env, page: str) -> None:
    at = open_page(page)
    no_exception(at)
    assert not at.error


def test_empty_states_offer_next_action(empty_env) -> None:
    runs_page = open_page("pages/runs.py")
    assert "No runs yet" in [s.value for s in runs_page.subheader]
    button(runs_page, "Start a new run")
    sweep = open_page("pages/sweep.py")
    assert "No sweep results yet" in [s.value for s in sweep.subheader]
    compare = open_page("pages/compare.py")
    assert any("at least two finished runs" in i.value for i in compare.info)


@pytest.mark.parametrize(
    "name",
    [
        "Demo: Defaults",
        "Demo: Broken run",
        "Demo: Interrupted run",
        "Demo: Cancelled run",
        "Demo: Very cautious plan",
        "Demo: Next in line",
        "Demo: Changed my mind",
        "(broken)",
    ],
)
def test_run_detail_in_every_state(demo_env, name: str) -> None:
    at = open_page("pages/run_detail.py", run=demo_env["ids"][name])
    no_exception(at)


def test_run_detail_done_leads_with_the_answer(demo_env) -> None:
    at = open_page("pages/run_detail.py", run=demo_env["ids"]["Demo: Hire for 2x, get 4x"])
    no_exception(at)
    assert at.subheader[0].value.startswith("On time ")
    assert [m.label for m in at.metric][:4] == [
        "On-time rate",
        "Turnaround P90",
        "Total cost",
        "Late reports",
    ]
    assert len(at.get("plotly_chart")) >= 7


def test_run_detail_interrupted_explains_and_offers_rerun(demo_env) -> None:
    at = open_page("pages/run_detail.py", run=demo_env["ids"]["Demo: Interrupted run"])
    assert "background worker stopped" in at.error[0].value
    button(at, "Run again")


def test_run_detail_unknown_run(demo_env) -> None:
    at = open_page("pages/run_detail.py", run="20990101-000000-gone")
    no_exception(at)
    assert "doesn't exist" in at.error[0].value


def test_compare_two_runs(demo_env) -> None:
    ids = demo_env["ids"]
    at = open_page(
        "pages/compare.py", runs=f"{ids['Demo: Defaults']},{ids['Demo: Earliest deadline first']}"
    )
    no_exception(at)
    assert any("how work is assigned" in i.value for i in at.info)
    metric_table = at.dataframe[1].value
    assert list(metric_table.columns) == ["Metric", "Better is", "A", "B"]


def test_compare_drops_missing_runs(demo_env) -> None:
    ids = demo_env["ids"]
    at = open_page("pages/compare.py", runs=f"{ids['Demo: Defaults']},20990101-000000-gone")
    no_exception(at)
    assert any("no longer exists" in w.value for w in at.warning)


def test_sweep_page_all_growth_levels_and_local_sweep(demo_env) -> None:
    at = open_page("pages/sweep.py")
    no_exception(at)
    radio = at.radio[0]
    radio.set_value("all").run()
    no_exception(at)
    local = open_page("pages/sweep.py", sweep=f"local:{ui_fixtures.DEMO_SWEEP_ID}")
    no_exception(local)


def test_runs_page_banner_and_tables(demo_env) -> None:
    at = open_page("pages/runs.py", new=demo_env["ids"]["Demo: Next in line"])
    no_exception(at)
    assert at.success[0].value.startswith("Queued 'Demo: Next in line'. It is #1 in line.")
    assert any("couldn't be read" in c.value for c in at.caption)  # the broken folder


# --- New run queues through the run store ------------------------------------------------


def test_new_run_submit_creates_a_queued_folder(empty_env) -> None:
    at = open_page("pages/new_run.py")
    no_exception(at)
    button(at, "Run").click().run()
    no_exception(at)
    (status,) = runs.list_runs(empty_env)
    assert status.state == "queued" and status.name == "Defaults"
    assert status.run_id in str(at.query_params["new"])


def test_new_run_change_widget_then_submit(empty_env) -> None:
    at = open_page("pages/new_run.py")
    at.radio(key="w:assignment.policy").set_value("edf").run()
    at.text_input(key="w:name").input("Demo: EDF from the form").run()
    button(at, "Run").click().run()
    no_exception(at)
    (status,) = runs.list_runs(empty_env)
    assert status.name == "Demo: EDF from the form"
    assert runs.read_params(status.run_id, empty_env).assignment.policy == "edf"


def test_new_run_invalid_advanced_yaml_blocks_run(empty_env) -> None:
    at = open_page("pages/new_run.py")
    at.text_area(key="w:advanced").input("sim:\n  seeds: 5\n").run()
    no_exception(at)
    assert button(at, "Run").disabled
    assert any("set in the form above" in e.value for e in at.error)
    assert runs.list_runs(empty_env) == []


def test_new_run_clone_prefills_and_flags_the_experiment(demo_env) -> None:
    at = open_page("pages/new_run.py", clone=demo_env["ids"]["Demo: Hire for 2x, get 4x"])
    no_exception(at)
    assert any("Started from 'Demo: Hire for 2x, get 4x'" in i.value for i in at.info)
    assert any("Forecast error experiment" in i.value for i in at.info)
    assert at.checkbox(key="w:_same_growth").value is False


def test_new_run_clone_missing_falls_back_to_defaults(demo_env) -> None:
    at = open_page("pages/new_run.py", clone="20990101-000000-gone")
    no_exception(at)
    assert any("no longer exists" in w.value for w in at.warning)


def test_new_run_from_published_sweep_row(demo_env) -> None:
    at = open_page("pages/new_run.py", sweep="published:headline", row="5")
    no_exception(at)
    assert any("from sweep 'headline'" in i.value for i in at.info)


def test_app_code_never_imports_the_simulation() -> None:
    """D-013: the app only queues runs; no simulation stage may be imported by app code."""
    banned = {"generate", "forecast", "plan", "simulate", "pipeline", "assign", "worker"}
    # Explicit exception: a closed-form calibration formula (microseconds, no world is
    # generated), used for the "about N% in January" note on New run.
    allowed = {("scout_planner.generate", "month0_peak_load")}
    for path in MAIN.parent.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if all((node.module, a.name) in allowed for a in node.names):
                    continue
                names = {node.module.split(".")[-1]} | {a.name for a in node.names}
                if node.module.startswith("scout_planner"):
                    assert not names & banned, f"{path.name} imports {names & banned}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[-1] not in banned or not alias.name.startswith(
                        "scout_planner"
                    ), f"{path.name} imports {alias.name}"


# --- a real run from the M4 pipeline (contract test between the engine and the UI) -------


@pytest.fixture(scope="module")
def real_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real run: EDF, one repeat (a few seconds), executed like the worker does."""
    from scout_planner.config import apply_overrides, load_default_params
    from scout_planner.pipeline import run_pipeline

    root = tmp_path_factory.mktemp("real") / "runs"
    params = apply_overrides(load_default_params(), {"assignment.policy": "edf", "sim.seeds": 1})
    run_id = runs.create_run(params, "Real: EDF one repeat", root=root)
    runs.update_status(run_id, root=root, state="running")
    run_pipeline(params, runs.run_dir(run_id, root), lambda *_: None, lambda: False, max_workers=1)
    runs.update_status(run_id, root=root, state="done")
    return root


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_pages_render_a_real_run(
    real_root: Path, monkeypatch: pytest.MonkeyPatch, page: str
) -> None:
    monkeypatch.setenv("SCOUT_RUNS_ROOT", str(real_root))
    monkeypatch.setenv("SCOUT_PUBLISHED_DIR", str(real_root.parent / "published"))
    at = open_page(page)
    no_exception(at)
    assert not at.error, [e.value for e in at.error]
    if page == "pages/run_detail.py":
        assert at.subheader[0].value.startswith("On time ")
        assert not any("Couldn't load" in w.value for w in at.warning)  # every file matched
        assert len(at.get("plotly_chart")) >= 8
