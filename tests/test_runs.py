"""Run store: ids, the state machine, atomic writes, robustness, heartbeat."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scout_planner import runs
from scout_planner.config import Params, apply_overrides, params_from_yaml
from scout_planner.errors import IllegalTransition, RunNotFound, RunStoreError

T0 = datetime(2026, 10, 3, 10, 15, 0)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def _create(root: Path, name: str = "Hire for 2x, get 4x", at: datetime = T0, **kw: object) -> str:
    return runs.create_run(Params(), name, root=root, now=at, **kw)  # type: ignore[arg-type]


def _write_raw_status(root: Path, run_id: str, text: str) -> None:
    (root / run_id / runs.STATUS_FILE).write_text(text, encoding="utf-8")


# --- run ids -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Hire for 2x, get 4x", "hire-for-2x-get-4x"),
        ("  EDF / 4x growth!! ", "edf-4x-growth"),
        ("Ünïcode only é", "n-code-only"),
        ("???", "run"),
        ("x" * 80, "x" * 40),
    ],
)
def test_slugify(name: str, slug: str) -> None:
    assert runs.slugify(name) == slug


def test_run_id_has_timestamp_and_slug(root: Path) -> None:
    run_id = _create(root)
    assert run_id == "20261003-101500-hire-for-2x-get-4x"
    assert re.fullmatch(r"\d{8}-\d{6}-[a-z0-9-]+", run_id)


def test_same_name_same_second_gets_suffixes(root: Path) -> None:
    ids = [_create(root, "Same") for _ in range(3)]
    assert ids == ["20261003-101500-same", "20261003-101500-same-2", "20261003-101500-same-3"]
    assert len({runs.read_status(i, root).run_id for i in ids}) == 3


# --- creating ----------------------------------------------------------------------


def test_create_writes_params_and_queued_status(root: Path) -> None:
    params = apply_overrides(Params(), {"assignment.policy": "edf", "sim.seeds": 5})
    run_id = runs.create_run(params, "EDF five seeds", sweep_id="quick", root=root, now=T0)

    assert runs.read_params(run_id, root) == params
    raw = (root / run_id / runs.PARAMS_FILE).read_text(encoding="utf-8")
    assert params_from_yaml(raw) == params

    status = runs.read_status(run_id, root)
    assert status.state == "queued"
    assert status.name == "EDF five seeds"
    assert status.created_at == T0
    assert status.started_at is None and status.finished_at is None
    assert status.progress == 0.0
    assert status.sweep_id == "quick"
    assert status.code_version  # a short SHA, or "unknown" outside git
    # status.json is plain JSON with the contract's keys
    data = json.loads((root / run_id / runs.STATUS_FILE).read_text(encoding="utf-8"))
    for key in ("run_id", "name", "state", "created_at", "progress", "stage", "message", "error"):
        assert key in data


def test_create_leaves_no_staging_folder(root: Path) -> None:
    _create(root)
    assert [p.name for p in root.iterdir()] == ["20261003-101500-hire-for-2x-get-4x"]


@pytest.mark.parametrize("name", ["", "   "])
def test_create_rejects_empty_name(root: Path, name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        runs.create_run(Params(), name, root=root)
    assert not root.exists() or not any(root.iterdir())


def test_create_rejects_bad_sweep_id(root: Path) -> None:
    with pytest.raises(ValueError, match="sweep id"):
        runs.create_run(Params(), "x", sweep_id="../evil", root=root)


def test_create_cleans_up_on_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(runs.os, "rename", boom)
    with pytest.raises(OSError, match="disk full"):
        _create(root)
    assert list(root.iterdir()) == []


# --- atomic writes -----------------------------------------------------------------


def test_crash_between_temp_write_and_replace_keeps_old_file(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = _create(root)
    before = (root / run_id / runs.STATUS_FILE).read_bytes()

    def crash(*_a: object, **_k: object) -> None:
        raise KeyboardInterrupt("simulated crash just before the swap")

    monkeypatch.setattr(runs.os, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        runs.update_status(run_id, root=root, state="running")
    monkeypatch.undo()

    # The real file is untouched and still valid; the temp file was removed.
    assert (root / run_id / runs.STATUS_FILE).read_bytes() == before
    assert runs.read_status(run_id, root).state == "queued"
    assert sorted(p.name for p in (root / run_id).iterdir()) == ["params.yaml", "status.json"]


def test_atomic_write_temp_file_is_in_same_folder(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = _create(root)
    seen: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src: str, dst: str) -> None:
        seen.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(runs.os, "replace", spy)
    runs.update_status(run_id, root=root, state="running")
    (src, dst), *_ = seen
    assert Path(src).parent == Path(dst).parent  # os.replace is atomic only within one filesystem
    assert Path(src).name.startswith(".")


# --- state machine -----------------------------------------------------------------

LEGAL = [
    ("queued", "running"),
    ("queued", "cancelled"),
    ("running", "done"),
    ("running", "failed"),
    ("running", "cancelled"),
]
ILLEGAL = [
    ("queued", "done"),
    ("queued", "failed"),
    ("running", "queued"),
    ("done", "running"),
    ("done", "failed"),
    ("failed", "running"),
    ("failed", "queued"),
    ("cancelled", "running"),
    ("cancelled", "done"),
]


def _put_in_state(root: Path, run_id: str, state: str) -> None:
    path = {
        "queued": [],
        "running": ["running"],
        "done": ["running", "done"],
        "failed": ["running", "failed"],
        "cancelled": ["cancelled"],
    }[state]
    for step in path:
        runs.update_status(run_id, root=root, state=step)


@pytest.mark.parametrize(("old", "new"), LEGAL)
def test_legal_transitions(root: Path, old: str, new: str) -> None:
    run_id = _create(root)
    _put_in_state(root, run_id, old)
    assert runs.update_status(run_id, root=root, state=new).state == new


@pytest.mark.parametrize(("old", "new"), ILLEGAL)
def test_illegal_transitions_are_rejected(root: Path, old: str, new: str) -> None:
    run_id = _create(root)
    _put_in_state(root, run_id, old)
    with pytest.raises(IllegalTransition):
        runs.update_status(run_id, root=root, state=new)
    assert runs.read_status(run_id, root).state == old


@pytest.mark.parametrize("final", ["done", "failed", "cancelled"])
def test_final_states_reject_even_progress_updates(root: Path, final: str) -> None:
    run_id = _create(root)
    _put_in_state(root, run_id, final)
    with pytest.raises(IllegalTransition):
        runs.update_status(run_id, root=root, progress=0.5, message="late progress")


def test_progress_update_while_running(root: Path) -> None:
    run_id = _create(root)
    runs.update_status(run_id, root=root, state="running")
    status = runs.update_status(
        run_id, root=root, progress=0.63, stage="simulate", message="seed 2/3, week 31/52"
    )
    assert (status.state, status.progress, status.stage) == ("running", 0.63, "simulate")
    assert status.message == "seed 2/3, week 31/52"


def test_timestamps_and_progress_are_filled_in(root: Path) -> None:
    run_id = _create(root)
    t1, t2 = T0 + timedelta(seconds=5), T0 + timedelta(seconds=65)
    running = runs.update_status(run_id, root=root, state="running", now=t1)
    assert running.started_at == t1 and running.finished_at is None
    done = runs.update_status(run_id, root=root, state="done", now=t2)
    assert done.finished_at == t2
    assert done.progress == 1.0
    assert done.duration_s == 60.0


def test_progress_is_clamped(root: Path) -> None:
    run_id = _create(root)
    runs.update_status(run_id, root=root, state="running")
    assert runs.update_status(run_id, root=root, progress=1.7).progress == 1.0
    assert runs.update_status(run_id, root=root, progress=-0.2).progress == 0.0


@pytest.mark.parametrize("field", ["run_id", "name", "created_at", "sweep_id", "code_version"])
def test_identity_fields_cannot_change(root: Path, field: str) -> None:
    run_id = _create(root)
    with pytest.raises(ValueError, match="can't be changed"):
        runs.update_status(run_id, root=root, **{field: "x"})


def test_unknown_state_value_is_rejected(root: Path) -> None:
    run_id = _create(root)
    with pytest.raises((IllegalTransition, ValueError)):
        runs.update_status(run_id, root=root, state="paused")


# --- unreadable status -------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (None, "missing"),
        ("{ not json", "corrupt"),
        ('{"run_id": "x"', "corrupt"),
        ("", "corrupt"),
        (
            '{"run_id": "RID", "name": "n", "state": "paused", "created_at": "2026-10-03"}',
            "corrupt",
        ),
        ('{"run_id": "other", "name": "n", "state": "done", "created_at": "2026-10-03"}', "folder"),
    ],
)
def test_bad_status_reads_as_unreadable(root: Path, content: str | None, reason: str) -> None:
    run_id = _create(root)
    path = root / run_id / runs.STATUS_FILE
    if content is None:
        path.unlink()
    else:
        path.write_text(content.replace("RID", run_id), encoding="utf-8")

    status = runs.read_status(run_id, root)
    assert status.state == "unreadable"
    assert reason in (status.error or "")
    assert status.run_id == run_id
    assert runs.display_state(status) == "unreadable"
    # One bad folder never breaks the listing.
    assert [s.state for s in runs.list_runs(root)] == ["unreadable"]
    with pytest.raises(IllegalTransition, match="unreadable"):
        runs.update_status(run_id, root=root, state="running")


@pytest.mark.parametrize("bad_id", ["../etc", "a/b", "_worker", ".hidden", "", "x" * 150])
def test_invalid_run_ids_never_touch_the_filesystem(root: Path, bad_id: str) -> None:
    with pytest.raises(RunNotFound):
        runs.read_status(bad_id, root)
    with pytest.raises(RunNotFound):
        runs.delete_run(bad_id, root)


def test_missing_run_raises_not_found(root: Path) -> None:
    with pytest.raises(RunNotFound):
        runs.read_status("20261003-101500-nope", root)


# --- listing -----------------------------------------------------------------------


def test_list_runs_newest_first_and_skips_non_runs(root: Path) -> None:
    a = _create(root, "first", T0)
    b = _create(root, "second", T0 + timedelta(minutes=1))
    c = _create(root, "third", T0 + timedelta(minutes=2))
    (root / "dev").mkdir()
    (root / "_new-abc").mkdir()
    (root / ".hidden").mkdir()
    (root / "stray.txt").write_text("x", encoding="utf-8")
    runs.write_heartbeat(123, None, root)

    assert [s.run_id for s in runs.list_runs(root)] == [c, b, a]


def test_list_runs_on_missing_root_is_empty(tmp_path: Path) -> None:
    assert runs.list_runs(tmp_path / "nothing-here") == []


def test_queued_runs_oldest_first_with_run_id_tiebreak(root: Path) -> None:
    late = _create(root, "late", T0 + timedelta(minutes=5))
    b = _create(root, "bbb", T0)
    a = _create(root, "aaa", T0)  # same second as bbb, created after it
    done = _create(root, "finished", T0 - timedelta(hours=1))
    _put_in_state(root, done, "done")

    assert [s.run_id for s in runs.queued_runs(root)] == [a, b, late]


def test_microsecond_timestamps_keep_creation_order(root: Path) -> None:
    first = _create(root, "zzz", T0.replace(microsecond=1))
    second = _create(root, "aaa", T0.replace(microsecond=2))
    assert [s.run_id for s in runs.queued_runs(root)] == [first, second]


def test_queued_runs_leave_out_cancel_requests(root: Path) -> None:
    a = _create(root, "a", T0)
    b = _create(root, "b", T0 + timedelta(seconds=1))
    runs.request_cancel(a, root)
    assert [s.run_id for s in runs.queued_runs(root)] == [b]
    assert [s.run_id for s in runs.queued_runs(root, include_cancel_requested=True)] == [a, b]


# --- cancel and display state ------------------------------------------------------


def test_request_cancel_on_queued_and_running(root: Path) -> None:
    run_id = _create(root)
    assert not runs.cancel_requested(run_id, root)
    assert runs.request_cancel(run_id, root) is True
    assert runs.cancel_requested(run_id, root)


@pytest.mark.parametrize("final", ["done", "failed", "cancelled"])
def test_request_cancel_on_finished_run_is_a_no_op(root: Path, final: str) -> None:
    run_id = _create(root)
    _put_in_state(root, run_id, final)
    assert runs.request_cancel(run_id, root) is False
    assert not runs.cancel_requested(run_id, root)


@pytest.mark.parametrize(
    ("state", "cancel", "shown"),
    [
        ("queued", False, "queued"),
        ("queued", True, "cancelling"),
        ("running", False, "running"),
        ("running", True, "cancelling"),
        ("done", False, "done"),
        ("done", True, "done"),
        ("failed", True, "failed"),
        ("cancelled", True, "cancelled"),
        ("unreadable", True, "unreadable"),
    ],
)
def test_display_state(state: str, cancel: bool, shown: str) -> None:
    status = runs.RunStatus(run_id="r", name="r", state=state, created_at=T0)  # type: ignore[arg-type]
    assert runs.display_state(status, cancel_requested=cancel) == shown
    assert shown in runs.DISPLAY_STATES


# --- delete ------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["queued", "done", "failed", "cancelled"])
def test_delete_run(root: Path, state: str) -> None:
    run_id = _create(root)
    _put_in_state(root, run_id, state)
    runs.delete_run(run_id, root)
    assert not (root / run_id).exists()
    assert list(root.iterdir()) == []


def test_delete_refuses_running(root: Path) -> None:
    run_id = _create(root)
    runs.update_status(run_id, root=root, state="running")
    with pytest.raises(RunStoreError, match="running"):
        runs.delete_run(run_id, root)
    assert (root / run_id).is_dir()


def test_delete_unreadable_run(root: Path) -> None:
    run_id = _create(root)
    _write_raw_status(root, run_id, "garbage")
    runs.delete_run(run_id, root)
    assert not (root / run_id).exists()


def test_cleanup_stale_temp_only_removes_old_folders(root: Path) -> None:
    _create(root)
    old = root / "_new-old"
    fresh = root / "_trash-fresh"
    old.mkdir()
    fresh.mkdir()
    an_hour_ago = time.time() - 3600
    os.utime(old, (an_hour_ago, an_hour_ago))
    assert runs.cleanup_stale_temp(root, older_than_s=600) == 1
    assert not old.exists() and fresh.exists()


# --- log ---------------------------------------------------------------------------


def test_log_append_and_tail(root: Path) -> None:
    run_id = _create(root)
    assert runs.read_log_tail(run_id, root=root) == ""
    for i in range(60):
        runs.append_log(run_id, f"line {i}", root)
    tail = runs.read_log_tail(run_id, lines=50, root=root).splitlines()
    assert len(tail) == 50
    assert tail[-1].endswith("line 59") and tail[0].endswith("line 10")


# --- heartbeat ---------------------------------------------------------------------


def test_heartbeat_round_trip(root: Path) -> None:
    runs.write_heartbeat(4242, "some-run", root, now=T0)
    beat = runs.read_heartbeat(root)
    assert beat == runs.Heartbeat(pid=4242, heartbeat_at=T0, current_run="some-run")


@pytest.mark.parametrize(
    ("age_s", "alive"), [(0, True), (9.9, True), (10.1, False), (3600, False), (-3600, False)]
)
def test_worker_alive_depends_on_heartbeat_age(root: Path, age_s: float, alive: bool) -> None:
    runs.write_heartbeat(1, None, root, now=T0)
    assert runs.worker_alive(root, max_age_s=10, now=T0 + timedelta(seconds=age_s)) is alive


def test_no_or_corrupt_heartbeat_means_not_alive(root: Path) -> None:
    assert runs.read_heartbeat(root) is None
    assert not runs.worker_alive(root)
    root.mkdir(parents=True)
    (root / runs.HEARTBEAT_FILE).write_text("{half a heartbe", encoding="utf-8")
    assert runs.read_heartbeat(root) is None
    assert not runs.worker_alive(root)


def test_remove_heartbeat_only_removes_own(root: Path) -> None:
    runs.write_heartbeat(111, None, root)
    runs.remove_heartbeat(root, pid=222)
    assert runs.read_heartbeat(root) is not None
    runs.remove_heartbeat(root, pid=111)
    assert runs.read_heartbeat(root) is None
