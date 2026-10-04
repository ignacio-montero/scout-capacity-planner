"""Sweeps: expansion (cross-product, fixed overrides), sweep_id tagging, published flattening."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scout_planner import cli, runs, sweep
from scout_planner.config import load_default_params
from scout_planner.sweep import SWEEPS_DIR


def _write(tmp_path: Path, data: dict, name: str = "s.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


SPEC = {
    "name": "mini",
    "base": "config/default.yaml",
    "vary": {"assignment.policy": ["fcfs", "edf", "optimiser"], "demand.actual_growth": [1.0, 4.0]},
    "fixed": {"sim.seeds": 2, "cost.late_penalty": 500.0},
}


def test_expansion_is_the_full_cross_product_in_file_order(tmp_path) -> None:
    runs_ = sweep.expand(sweep.load_sweep(_write(tmp_path, SPEC)))
    assert len(runs_) == 6
    combos = [(r.params.assignment.policy, r.params.demand.actual_growth) for r in runs_]
    assert combos == [
        ("fcfs", 1.0), ("fcfs", 4.0), ("edf", 1.0), ("edf", 4.0),
        ("optimiser", 1.0), ("optimiser", 4.0),
    ]  # fmt: skip
    assert runs_[0].name == "mini: policy=fcfs, actual_growth=1.0"
    assert runs_[0].values == {"assignment.policy": "fcfs", "demand.actual_growth": 1.0}


def test_fixed_overrides_apply_to_every_run_and_nothing_else_changes(tmp_path) -> None:
    default = load_default_params()
    for r in sweep.expand(sweep.load_sweep(_write(tmp_path, SPEC))):
        assert r.params.sim.seeds == 2 and r.params.cost.late_penalty == 500.0
        assert r.params.team == default.team and r.params.automation == default.automation


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"vary": {}}, "vary"),
        ({"vary": {"assignment.policy": []}}, "non-empty list"),
        ({"vary": {"assignment.policy": ["edf", "edf"]}}, "duplicate"),
        ({"fixed": {"assignment.policy": "edf"}}, "both varied and fixed"),
        ({"surprise": 1}, "unknown sweep keys"),
        ({"name": ""}, "name"),
    ],
)
def test_malformed_sweeps_are_rejected(change, message) -> None:
    with pytest.raises(ValueError, match=message):
        sweep.parse_sweep({**SPEC, **change})


def test_unknown_or_invalid_parameters_fail_at_expansion(tmp_path) -> None:
    bad_key = {**SPEC, "vary": {"assignment.polcy": ["edf"]}}
    with pytest.raises(ValueError, match="unknown parameter"):
        sweep.expand(sweep.load_sweep(_write(tmp_path, bad_key)))
    bad_value = {**SPEC, "vary": {"demand.actual_growth": [1.0, 9.0]}}
    with pytest.raises(ValueError):
        sweep.expand(sweep.load_sweep(_write(tmp_path, bad_value)))


@pytest.mark.parametrize(
    "name",
    ["quick", "cadence", "headline", "growth", "forecast_error", "baseline", "late_penalty"],
)
def test_shipped_sweep_files_expand(name) -> None:
    spec = sweep.load_sweep(SWEEPS_DIR / f"{name}.yaml")
    expanded = sweep.expand(spec)
    assert spec.name == name and len(expanded) >= 4
    assert len({r.name for r in expanded}) == len(expanded)


def test_queue_sweep_tags_every_run_with_the_sweep_id(tmp_path) -> None:
    root = tmp_path / "runs"
    spec = sweep.load_sweep(_write(tmp_path, SPEC))
    sweep_id, run_ids = sweep.queue_sweep(spec, root)
    assert sweep_id.startswith("mini-") and len(run_ids) == 6
    statuses = {s.run_id: s for s in runs.list_runs(root)}
    assert set(statuses) == set(run_ids)
    assert {s.sweep_id for s in statuses.values()} == {sweep_id}
    assert {s.state for s in statuses.values()} == {"queued"}
    assert runs.read_params(run_ids[-1], root).assignment.policy == "optimiser"


def _finish(run_id: str, root: Path, on_time: float) -> None:
    """Pretend the worker ran it: mark done and drop a summary.json."""
    runs.update_status(run_id, root=root, state="running")
    runs.update_status(run_id, root=root, state="done")
    summary = {
        "on_time_rate": {"mean": on_time, "min": on_time - 0.01, "max": on_time + 0.01},
        "cost_total": {"mean": 1e6, "min": 9e5, "max": 1.1e6},
        "n_late": {"mean": 10.0, "min": 8.0, "max": 12.0},
        "meets_target": on_time >= 0.95,
        "n_seeds": 2,
        "diagnostics": {"optimiser_solves": 0, "per_seed": []},
    }
    (runs.run_dir(run_id, root) / "summary.json").write_text(json.dumps(summary))


def test_publish_flattens_one_row_per_finished_run(tmp_path) -> None:
    root = tmp_path / "runs"
    spec = sweep.load_sweep(_write(tmp_path, SPEC))
    sweep_id, run_ids = sweep.queue_sweep(spec, root)
    for k, run_id in enumerate(run_ids[:-1]):
        _finish(run_id, root, 0.90 + 0.01 * k)
    # the last one stays queued: refused by default, published with allow_partial
    with pytest.raises(sweep.IncompleteSweep, match="5 of 6 runs are done"):
        sweep.publish(spec, sweep_id, root, out_dir=tmp_path / "published")
    path = sweep.publish(spec, sweep_id, root, out_dir=tmp_path / "published", allow_partial=True)
    assert path.name == "mini_summary.parquet"
    df = pd.read_parquet(path)
    assert len(df) == 5
    for col in (
        "name", "run_id", "assignment.policy", "demand.actual_growth", "sim.seeds",
        "cost.late_penalty", "automation.enabled", "sim.target_on_time",
        "on_time_rate_mean", "on_time_rate_min", "on_time_rate_max",
        "cost_total_mean", "n_late_mean", "meets_target",
    ):  # fmt: skip
        assert col in df.columns, col
    assert not any(c.startswith("diagnostics") or c.startswith("n_seeds") for c in df.columns)
    assert set(df["sim.seeds"]) == {2} and set(df["cost.late_penalty"]) == {500.0}
    assert set(df["automation.enabled"]) == {False}
    row = df.set_index("run_id").loc[run_ids[0]]
    assert row["assignment.policy"] == "fcfs" and row["on_time_rate_mean"] == pytest.approx(0.90)
    assert not row["meets_target"]
    assert set(df["sweep_expected_runs"]) == {6} and set(df["sweep_done_runs"]) == {5}
    assert set(df["sweep_id"]) == {sweep_id}
    assert sweep.read_publish_info(path) == {
        "sweep": "mini", "sweep_id": sweep_id, "expected_runs": 6, "done_runs": 5,
        "partial": True,
    }  # fmt: skip
    assert sweep.latest_sweep_id("mini", root) == sweep_id


def test_publish_without_finished_runs_is_an_error(tmp_path) -> None:
    root = tmp_path / "runs"
    spec = sweep.load_sweep(_write(tmp_path, SPEC))
    sweep_id, _ = sweep.queue_sweep(spec, root)
    with pytest.raises(sweep.IncompleteSweep, match="0 of 6"):
        sweep.publish(spec, sweep_id, root, out_dir=tmp_path / "published")
    with pytest.raises(ValueError, match="no finished runs"):
        sweep.publish(spec, sweep_id, root, out_dir=tmp_path / "published", allow_partial=True)


def test_cli_sweep_queue_only_queues(tmp_path, capsys) -> None:
    root = tmp_path / "runs"
    path = _write(tmp_path, SPEC)
    assert cli.main(["sweep", "--sweep", str(path), "--root", str(root), "--queue"]) == 0
    assert {s.state for s in runs.list_runs(root)} == {"queued"}
    assert "6 runs" in capsys.readouterr().out


def test_cli_sweep_rejects_a_bad_file(tmp_path, capsys) -> None:
    path = _write(tmp_path, {**SPEC, "vary": {"assignment.polcy": ["edf"]}})
    assert cli.main(["sweep", "--sweep", str(path), "--root", str(tmp_path / "r")]) == 2
    assert "invalid sweep" in capsys.readouterr().err


@pytest.mark.slow
def test_cli_sweep_executes_and_publishes(tmp_path) -> None:
    tiny = {
        "name": "tiny",
        "base": "config/default.yaml",
        "vary": {"assignment.policy": ["fcfs", "edf"]},
        "fixed": {
            "sim.months": 1, "sim.seeds": 1, "team.full_time_count": 8,
            "team.freelance_count": 4, "demand.start_load": 0.3,
            "team.follow_hiring_plan": False,
        },
    }  # fmt: skip
    root, published = tmp_path / "runs", tmp_path / "published"
    args = ["sweep", "--sweep", str(_write(tmp_path, tiny)), "--root", str(root)]
    assert cli.main([*args, "--publish", "--published-dir", str(published)]) == 0
    assert {s.state for s in runs.list_runs(root)} == {"done"}
    df = pd.read_parquet(published / "tiny_summary.parquet")
    assert sorted(df["assignment.policy"]) == ["edf", "fcfs"]
    assert set(df["sim.months"]) == {1}
    # --publish-only republishes the latest execution
    assert cli.main([*args, "--publish-only", "--published-dir", str(tmp_path / "p2")]) == 0
    assert (tmp_path / "p2" / "tiny_summary.parquet").is_file()


# --- link: target <- source ----------------------------------------------------------

LINKED = {
    "name": "linked",
    "base": "config/default.yaml",
    "vary": {"demand.actual_growth": [1.0, 2.0, 4.0], "assignment.policy": ["edf"]},
    "fixed": {"sim.seeds": 2},
    "link": {"capacity_plan.assumed_growth": "demand.actual_growth"},
}


def test_link_copies_the_source_into_every_combination(tmp_path) -> None:
    expanded = sweep.expand(sweep.load_sweep(_write(tmp_path, LINKED)))
    pairs = [
        (r.params.demand.actual_growth, r.params.capacity_plan.assumed_growth) for r in expanded
    ]
    assert pairs == [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]
    assert "assumed_growth" not in expanded[0].name  # names show what is varied


def test_link_chains_resolve_to_their_root_and_can_read_fixed_values(tmp_path) -> None:
    spec = {
        **LINKED,
        "vary": {"assignment.policy": ["edf", "fcfs"]},
        "fixed": {"sim.seeds": 2, "demand.actual_growth": 3.0},
        "link": {
            "capacity_plan.assumed_growth": "demand.actual_growth",
            "demand.history_growth_per_year": "automation.desk_reduction",
            "automation.desk_reduction": "demand.live_view_share",
        },
    }
    for r in sweep.expand(sweep.load_sweep(_write(tmp_path, spec))):
        assert r.params.capacity_plan.assumed_growth == 3.0
        share = r.params.demand.live_view_share
        assert r.params.automation.desk_reduction == share
        assert r.params.demand.history_growth_per_year == share


@pytest.mark.parametrize(
    ("link", "message"),
    [
        ({"capacity_plan.assumed_growth": "demand.actual_grwth"}, "not a parameter"),
        ({"capacity_plan.assumed_grwth": "demand.actual_growth"}, "not a parameter"),
        ({"assignment.weights": "demand.actual_growth"}, "not a parameter"),  # a group
        ({"demand.actual_growth": "demand.actual_growth"}, "itself"),
        ({"demand.actual_growth": "sim.seed"}, "also varied"),
        ({"sim.seeds": "sim.seed"}, "also fixed"),
        (
            {
                "capacity_plan.quantile": "capacity_plan.target_utilisation",
                "capacity_plan.target_utilisation": "capacity_plan.quantile",
            },
            "cycle",
        ),  # fmt: skip
        (["capacity_plan.assumed_growth"], "must map"),
    ],
)
def test_bad_links_are_rejected(link, message) -> None:
    with pytest.raises(ValueError, match=message):
        sweep.parse_sweep({**LINKED, "link": link})


def test_linked_value_still_goes_through_validation(tmp_path) -> None:
    # sim.seeds (int, 1..5) <- demand.actual_growth (float): strict validation refuses it.
    spec = {**LINKED, "fixed": {}, "link": {"sim.seeds": "demand.actual_growth"}}
    with pytest.raises(ValueError):
        sweep.expand(sweep.load_sweep(_write(tmp_path, spec)))


# --- published columns: named + everything that differs from the defaults ------------------


def _publish(tmp_path: Path, spec: dict, base: Path | None = None) -> pd.DataFrame:
    root = tmp_path / "runs"
    path = _write(tmp_path, {**spec, **({"base": str(base)} if base else {})})
    parsed = sweep.load_sweep(path)
    sweep_id, run_ids = sweep.queue_sweep(parsed, root)
    for k, run_id in enumerate(run_ids):
        _finish(run_id, root, 0.9 + 0.01 * k)
    return pd.read_parquet(sweep.publish(parsed, sweep_id, root, out_dir=tmp_path / "pub"))


def test_published_table_has_linked_columns(tmp_path) -> None:
    df = _publish(tmp_path, LINKED)
    assert "capacity_plan.assumed_growth" in df.columns
    assert (df["capacity_plan.assumed_growth"] == df["demand.actual_growth"]).all()


def test_published_table_adds_every_parameter_changed_by_a_custom_base(tmp_path) -> None:
    from scout_planner.config import apply_overrides, dump_params

    base = tmp_path / "base.yaml"
    custom = apply_overrides(
        load_default_params(),
        {"team.full_time_count": 10, "demand.desk_hours": [3.0, 9.0],
         "assignment.weights": {"churn": 7.0}},
    )  # fmt: skip
    dump_params(custom, base)
    df = _publish(tmp_path, {**SPEC, "vary": {"assignment.policy": ["edf", "fcfs"]}}, base)
    assert set(df["team.full_time_count"]) == {10}
    assert set(df["assignment.weights.churn"]) == {7.0}
    assert [list(v) for v in df["demand.desk_hours"]] == [[3.0, 9.0]] * 2
    # parameters left at their defaults are not published
    assert "team.freelance_count" not in df.columns and "assignment.weights.cost" not in df.columns
    assert sweep.differing_leaves(load_default_params()) == {}


def test_published_rows_with_a_null_parameter_rebuild_exactly(tmp_path) -> None:
    from scout_planner.results import normalise_sweep_frame, params_for_sweep_row

    spec = {**SPEC, "vary": {"assignment.commit_buffer_days": [None, 2]}, "fixed": {}}
    path = _write(tmp_path, spec)
    df = _publish(tmp_path, spec)
    root = tmp_path / "runs"
    for row in normalise_sweep_frame(df, load_default_params()).to_dict("records"):
        rebuilt = params_for_sweep_row(row, load_default_params(), path)
        assert rebuilt == runs.read_params(row["run_id"], root)


def test_publish_is_atomic_and_leaves_no_temp_files(tmp_path, monkeypatch) -> None:
    root, out = tmp_path / "runs", tmp_path / "pub"
    spec = sweep.load_sweep(_write(tmp_path, SPEC))
    sweep_id, run_ids = sweep.queue_sweep(spec, root)
    for run_id in run_ids:
        _finish(run_id, root, 0.9)
    first = sweep.publish(spec, sweep_id, root, out_dir=out)
    before = first.read_bytes()

    def broken(table, path, **kwargs):
        Path(path).write_bytes(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(sweep.pq, "write_table", broken)
    with pytest.raises(OSError, match="disk full"):
        sweep.publish(spec, sweep_id, root, out_dir=out)
    assert first.read_bytes() == before  # the old file is untouched
    assert [p.name for p in out.iterdir()] == [first.name]  # temp file cleaned up


@pytest.mark.slow
def test_sweeps_run_concurrently_into_different_roots_and_publish_together(tmp_path) -> None:
    """Two CLI sweeps at once, separate --root folders, one shared published folder."""
    import subprocess
    import sys

    fixed = {
        "sim.months": 1, "sim.seeds": 1, "team.full_time_count": 8, "team.freelance_count": 4,
        "demand.start_load": 0.3, "team.follow_hiring_plan": False,
    }  # fmt: skip
    published = tmp_path / "published"
    procs = []
    for name in ("left", "right"):
        spec = {"name": name, "vary": {"assignment.policy": ["fcfs", "edf"]}, "fixed": fixed}
        argv = [
            sys.executable, "-m", "scout_planner.cli", "sweep",
            "--sweep", str(_write(tmp_path, spec, f"{name}.yaml")),
            "--root", str(tmp_path / f"runs-{name}"), "--workers", "1",
            "--publish", "--published-dir", str(published),
        ]  # fmt: skip
        procs.append(subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
    for proc in procs:
        out, _ = proc.communicate(timeout=300)
        assert proc.returncode == 0, out.decode()
    assert sorted(p.name for p in published.iterdir()) == [
        "left_summary.parquet",
        "right_summary.parquet",
    ]
    for name in ("left", "right"):
        assert len(pd.read_parquet(published / f"{name}_summary.parquet")) == 2


# --- completeness, diagnostics, latest execution ----------------------------------------


def test_publish_refuses_a_sweep_with_a_failed_run_and_cli_reports_it(tmp_path, capsys) -> None:
    root, out = tmp_path / "runs", tmp_path / "pub"
    path = _write(tmp_path, SPEC)
    spec = sweep.load_sweep(path)
    sweep_id, run_ids = sweep.queue_sweep(spec, root)
    for run_id in run_ids[1:]:
        _finish(run_id, root, 0.96)
    runs.update_status(run_ids[0], root=root, state="running")
    runs.update_status(run_ids[0], root=root, state="failed", error="boom")
    argv = ["sweep", "--sweep", str(path), "--root", str(root), "--publish-only",
            "--published-dir", str(out)]  # fmt: skip
    assert cli.main(argv) == 2
    assert "5 of 6 runs are done" in capsys.readouterr().err
    assert not out.exists() or not list(out.iterdir())
    assert cli.main([*argv, "--allow-partial"]) == 0
    assert "5/6 runs (PARTIAL)" in capsys.readouterr().out
    assert len(pd.read_parquet(out / "mini_summary.parquet")) == 5


def test_published_table_carries_each_runs_solver_diagnostics(tmp_path) -> None:
    root = tmp_path / "runs"
    spec = sweep.load_sweep(_write(tmp_path, SPEC))
    sweep_id, run_ids = sweep.queue_sweep(spec, root)
    for k, run_id in enumerate(run_ids):
        _finish(run_id, root, 0.9)
        folder = runs.run_dir(run_id, root)
        summary = json.loads((folder / "summary.json").read_text())
        summary["diagnostics"] = {
            "optimiser_solves": 100, "fallbacks": k, "fallback_share": k / 100,
            "wall_clock_hits": 0, "status_FEASIBLE": 90 - k, "status_UNKNOWN": k,
            "per_seed": [{"seed": 0}],
        }  # fmt: skip
        (folder / "summary.json").write_text(json.dumps(summary))
    df = pd.read_parquet(sweep.publish(spec, sweep_id, root, out_dir=tmp_path / "pub"))
    for col in ("diag_optimiser_solves", "diag_fallbacks", "diag_fallback_share",
                "diag_wall_clock_hits", "diag_status_FEASIBLE", "diag_status_UNKNOWN"):  # fmt: skip
        assert col in df.columns, col
    assert "diag_per_seed" not in df.columns
    row = df.set_index("run_id").loc[run_ids[3]]
    assert row["diag_fallbacks"] == 3 and row["diag_fallback_share"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        # optimiser_eval vs optimiser_eval_fast: both slugs start "optimiser-eval-"
        (["optimiser-eval-20261001-090000", "optimiser-eval-fast-20261003-090000"],
         "optimiser-eval-20261001-090000"),
        (["optimiser-eval-20261002-235959", "optimiser-eval-20261003-000001",
          "optimiser-eval-20261001-120000"], "optimiser-eval-20261003-000001"),
        (["optimiser-eval-fast-20261003-090000"], None),
        (["optimiser-eval-2026-oops"], None),
    ],
)  # fmt: skip
def test_latest_sweep_id_matches_the_exact_name_and_sorts_by_time(tmp_path, ids, expected) -> None:
    root = tmp_path / "runs"
    for sweep_id in ids:
        runs.create_run(load_default_params(), "r", sweep_id, root=root)
    assert sweep.latest_sweep_id("optimiser_eval", root) == expected
