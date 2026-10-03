"""Worker: queue order, crash recovery, failures, cancellation, shutdown.

Fast tests drive :class:`Worker` step by step in-process; runs still execute in
a real spawned child process with a fake pipeline from ``fake_pipelines.py``.
Tests marked ``slow`` start the worker (and ``serve``) as real subprocesses.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scout_planner import runs, serve, worker
from scout_planner.config import Params
from scout_planner.errors import RunStoreError
from scout_planner.worker import (
    INTERRUPTED,
    PipelineUnavailable,
    ThrottledProgress,
    Worker,
    WorkerAlreadyRunning,
    executor_lease,
    load_pipeline,
    run_one_now,
    runs_lock,
)

TESTS_DIR = Path(__file__).resolve().parent
T0 = datetime(2026, 10, 3, 10, 0, 0)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def wait_until(predicate: Callable[[], bool], timeout: float = 20.0, step: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def queue(root: Path, name: str, at: datetime | None = None) -> str:
    return runs.create_run(Params(), name, root=root, now=at)


def state(root: Path, run_id: str) -> runs.RunStatus:
    return runs.read_status(run_id, root)


def make_worker(root: Path, pipeline: str = "succeed", **kw: object) -> Worker:
    spec = pipeline if ":" in pipeline else f"fake_pipelines:{pipeline}"
    return Worker(root=root, pipeline=spec, poll_s=0.05, **kw)  # type: ignore[arg-type]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --- pipeline lookup ---------------------------------------------------------------


def test_load_pipeline_resolves_module_function() -> None:
    import fake_pipelines

    assert load_pipeline("fake_pipelines:succeed") is fake_pipelines.succeed


@pytest.mark.parametrize("spec", ["no_colon", ":func", "module:", ""])
def test_load_pipeline_rejects_malformed_spec(spec: str) -> None:
    with pytest.raises(PipelineUnavailable, match="module:function"):
        load_pipeline(spec)


def test_load_pipeline_missing_module_or_function() -> None:
    with pytest.raises(PipelineUnavailable, match="not found"):
        load_pipeline("no_such_module_xyz:run")
    with pytest.raises(PipelineUnavailable, match="no function"):
        load_pipeline("fake_pipelines:no_such_function")


def test_missing_default_pipeline_says_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = "scout_planner.no_such_pipeline_yet:run_pipeline"
    monkeypatch.setattr(worker, "DEFAULT_PIPELINE", missing)
    with pytest.raises(PipelineUnavailable, match="pipeline not implemented yet"):
        load_pipeline(missing)


def test_broken_import_inside_pipeline_is_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "broken_pipeline_mod.py").write_text("import missing_dependency_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="missing_dependency_xyz"):
        load_pipeline("broken_pipeline_mod:run")


# --- progress throttling -----------------------------------------------------------


def test_progress_is_throttled_but_stage_changes_go_through(root: Path) -> None:
    run_id = queue(root, "throttle")
    runs.update_status(run_id, root=root, state="running")
    clock = iter([0.0, 0.1, 0.2, 0.3, 0.6, 0.9]).__next__
    progress = ThrottledProgress(run_id, root, min_interval_s=0.5, clock=clock)

    progress(0.10, "generate", "a")  # t=0.0 first call: written
    progress(0.11, "generate", "b")  # t=0.1 suppressed
    progress(0.12, "generate", "c")  # t=0.2 suppressed
    assert progress.writes == 1
    assert state(root, run_id).message == "a"
    assert progress.latest() == {"progress": 0.12, "stage": "generate", "message": "c"}

    progress(0.30, "forecast", "d")  # t=0.3 new stage: written at once
    assert (progress.writes, state(root, run_id).stage) == (2, "forecast")
    progress(0.31, "forecast", "e")  # t=0.6, only 0.3 s after the last write: suppressed
    assert progress.writes == 2
    progress(0.40, "forecast", "f")  # t=0.9 >= 0.3 + 0.5: written
    assert progress.writes == 3
    assert state(root, run_id).message == "f"
    assert progress.latest() == {}


def test_progress_after_run_ended_does_not_raise(root: Path) -> None:
    run_id = queue(root, "ended")
    runs.update_status(run_id, root=root, state="cancelled")
    ThrottledProgress(run_id, root)(0.5, "simulate", "late")  # logged, not raised


# --- queue steps (no child process needed) -----------------------------------------


def test_recover_marks_running_runs_interrupted(root: Path) -> None:
    left_running = queue(root, "was running")
    untouched = queue(root, "still queued")
    runs.update_status(left_running, root=root, state="running", progress=0.4, message="week 20")

    assert make_worker(root).recover() == [left_running]
    status = state(root, left_running)
    assert (status.state, status.error) == ("failed", INTERRUPTED)
    assert status.message == "week 20"  # where it stopped stays visible
    assert "interrupted" in runs.read_log_tail(left_running, root=root)
    assert state(root, untouched).state == "queued"


def test_cancel_flagged_queued_runs(root: Path) -> None:
    keep = queue(root, "keep", T0)
    drop = queue(root, "drop", T0 + timedelta(seconds=1))
    runs.request_cancel(drop, root)

    w = make_worker(root)
    assert w.cancel_flagged_queued() == [drop]
    assert state(root, drop).state == "cancelled"
    assert state(root, keep).state == "queued"
    assert w.next_run() == keep


def test_next_run_is_the_oldest_queued(root: Path) -> None:
    newest = queue(root, "newest", T0 + timedelta(minutes=2))
    oldest = queue(root, "oldest", T0)
    middle = queue(root, "middle", T0 + timedelta(minutes=1))
    w = make_worker(root)
    assert w.next_run() == oldest
    runs.update_status(oldest, root=root, state="cancelled")
    assert w.next_run() == middle
    assert newest != w.next_run()


def test_idle_poll_writes_heartbeat(root: Path) -> None:
    w = make_worker(root)
    assert w.poll_once() is None
    beat = runs.read_heartbeat(root)
    assert beat is not None and beat.pid == os.getpid() and beat.current_run is None
    assert runs.worker_alive(root)


def test_second_worker_on_same_root_is_refused(root: Path) -> None:
    first, second = make_worker(root), make_worker(root)
    with first.exclusive(), pytest.raises(WorkerAlreadyRunning), second.exclusive():
        pass
    with make_worker(root).exclusive():  # released when the first one ended
        pass


# --- executing runs (spawned child process) ----------------------------------------


def test_successful_run(root: Path) -> None:
    run_id = queue(root, "ok")
    assert make_worker(root, "succeed").poll_once() == run_id

    status = state(root, run_id)
    assert status.state == "done"
    assert status.progress == 1.0
    assert status.started_at is not None and status.finished_at is not None
    assert status.error is None
    child_pid = int((root / run_id / "fake_result.txt").read_text().split()[0])
    assert child_pid != os.getpid()  # really ran in a child process
    assert runs.read_heartbeat(root).current_run is None  # type: ignore[union-attr]


def test_failed_run_records_error_and_traceback(root: Path) -> None:
    run_id = queue(root, "boom")
    make_worker(root, "fail").poll_once()

    status = state(root, run_id)
    assert status.state == "failed"
    assert status.error == "ValueError: demand model did not converge"
    assert status.message == "fitting the demand model"  # last progress kept
    log_text = (root / run_id / runs.LOG_FILE).read_text()
    assert "Traceback (most recent call last)" in log_text
    assert "fake_pipelines.py" in log_text


def test_hard_crash_in_child_does_not_kill_worker(root: Path) -> None:
    crashing = queue(root, "crash", T0)
    after = queue(root, "after", T0 + timedelta(seconds=1))
    w = make_worker(root, "crash")
    w.poll_once()
    status = state(root, crashing)
    assert status.state == "failed"
    assert "exit code 7" in (status.error or "")
    # The worker is still fine and moves on to the next run.
    assert w.next_run() == after


def test_missing_pipeline_fails_the_run_not_the_worker(root: Path) -> None:
    run_id = queue(root, "nothing to run")
    make_worker(root, "no_such_pipeline_module:run").poll_once()
    status = state(root, run_id)
    assert status.state == "failed"
    assert "no_such_pipeline_module" in (status.error or "")


@pytest.mark.skipif(
    importlib.util.find_spec("scout_planner.pipeline") is not None,
    reason="the real pipeline exists now",
)
def test_default_pipeline_not_built_yet_gives_clear_error(root: Path) -> None:
    run_id = queue(root, "default pipeline")
    Worker(root=root, poll_s=0.05).poll_once()
    status = state(root, run_id)
    assert status.state == "failed"
    assert (status.error or "").startswith("pipeline not implemented yet")


def test_invalid_params_file_fails_the_run(root: Path) -> None:
    run_id = queue(root, "bad params")
    (root / run_id / runs.PARAMS_FILE).write_text("sim:\n  seeds: 99\n", encoding="utf-8")
    make_worker(root, "succeed").poll_once()
    status = state(root, run_id)
    assert status.state == "failed"
    assert "ValidationError" in (status.error or "")


def _in_background(fn: Callable[[], None]) -> threading.Thread:
    thread = threading.Thread(target=fn, daemon=True)
    thread.start()
    return thread


def test_cancel_running_run(root: Path) -> None:
    run_id = queue(root, "long")

    def cancel_once_started() -> None:
        assert wait_until(lambda: (root / run_id / "fake_started.txt").exists())
        assert wait_until(lambda: state(root, run_id).progress > 0)
        runs.request_cancel(run_id, root)

    helper = _in_background(cancel_once_started)
    make_worker(root, "wait_for_cancel").poll_once()
    helper.join(5)

    status = state(root, run_id)
    assert status.state == "cancelled"
    assert (status.message or "").startswith("simulating week")  # where it stopped
    assert status.finished_at is not None


def test_run_ignoring_cancel_is_killed_with_its_children(root: Path) -> None:
    run_id = queue(root, "stuck")

    def cancel_once_started() -> None:
        assert wait_until(lambda: (root / run_id / "fake_started.txt").exists())
        runs.request_cancel(run_id, root)

    _in_background(cancel_once_started)
    started = time.monotonic()
    make_worker(root, "ignore_cancel", cancel_grace_s=0.3).poll_once()
    assert time.monotonic() - started < 15

    status = state(root, run_id)
    assert status.state == "cancelled"
    assert "by force" in (status.message or "")
    grandchild = int((root / run_id / "grandchild.pid").read_text())
    assert wait_until(lambda: not pid_alive(grandchild), timeout=5), "grandchild survived"


def test_stopping_worker_mid_run_marks_it_interrupted(root: Path) -> None:
    run_id = queue(root, "will be interrupted")
    w = make_worker(root, "wait_for_cancel")

    def stop_once_started() -> None:
        assert wait_until(lambda: (root / run_id / "fake_started.txt").exists())
        w.stop()

    _in_background(stop_once_started)
    w.poll_once()
    status = state(root, run_id)
    assert (status.state, status.error) == ("failed", INTERRUPTED)


def test_drain_runs_queue_oldest_first_and_survives_failures(root: Path) -> None:
    third = queue(root, "ok three", T0 + timedelta(seconds=2))
    first = queue(root, "ok one", T0)
    second = queue(root, "fail two", T0 + timedelta(seconds=1))
    skipped = queue(root, "ok cancelled", T0 + timedelta(seconds=3))
    runs.request_cancel(skipped, root)

    executed = make_worker(root, "by_name").drain()

    assert executed == [first, second, third]
    assert [state(root, r).state for r in (first, second, third, skipped)] == [
        "done",
        "failed",
        "done",
        "cancelled",
    ]
    starts = [state(root, r).started_at for r in executed]
    assert starts == sorted(starts)  # type: ignore[type-var]
    assert runs.read_heartbeat(root) is None  # removed on clean exit


# --- robustness: poison pills, failing writes, the loop never dies -----------------


@pytest.fixture
def read_only(tmp_path: Path) -> Iterator[Callable[[Path], None]]:
    locked: list[Path] = []

    def lock(path: Path) -> None:
        path.chmod(0o555)
        locked.append(path)

    yield lock
    for path in locked:
        path.chmod(0o755)


def test_unwritable_run_folder_is_skipped_not_fatal(
    root: Path, read_only: Callable[[Path], None]
) -> None:
    poisoned = queue(root, "poisoned", T0)
    healthy = queue(root, "healthy", T0 + timedelta(seconds=1))
    read_only(root / poisoned)

    w = make_worker(root, "succeed")
    assert w.poll_once() == poisoned  # tried, did not raise
    assert poisoned in w.skipped
    assert state(root, poisoned).state == "queued"  # can't even be marked failed
    assert w.poll_once() == healthy  # the queue moves on
    assert state(root, healthy).state == "done"
    assert w.next_run() is None


def test_oserror_starting_a_run_marks_it_failed_when_possible(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = queue(root, "flaky disk")
    real_update = runs.update_status
    calls = {"n": 0}

    def flaky(rid: str, /, **kw: object) -> runs.RunStatus:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        return real_update(rid, **kw)

    monkeypatch.setattr(runs, "update_status", flaky)
    w = make_worker(root, "succeed")
    w.poll_once()
    assert run_id in w.skipped
    status = state(root, run_id)
    assert status.state == "failed"
    assert "No space left on device" in (status.error or "")


def test_failing_log_and_heartbeat_writes_do_not_break_runs(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_a: object, **_k: object) -> None:
        raise PermissionError("read-only")

    left_running = queue(root, "left running", T0)
    runs.update_status(left_running, root=root, state="running")
    crashing = queue(root, "crashes", T0 + timedelta(seconds=1))
    monkeypatch.setattr(runs, "append_log", broken)
    monkeypatch.setattr(runs, "write_heartbeat", broken)

    w = make_worker(root, "crash")
    assert w.recover() == [left_running]
    w.poll_once()
    assert state(root, crashing).state == "failed"  # _finalise still recorded it
    assert "exit code 7" in (state(root, crashing).error or "")


def _serve_in_thread(w: Worker) -> threading.Thread:
    thread = threading.Thread(target=w.serve_forever, daemon=True)
    thread.start()
    return thread


def test_serve_loop_survives_unexpected_errors(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = make_worker(root, "succeed")
    real_poll = w.poll_once
    failures = {"left": 3}

    def sometimes_broken() -> str | None:
        if failures["left"]:
            failures["left"] -= 1
            raise OSError("runs folder temporarily unreadable")
        return real_poll()

    monkeypatch.setattr(w, "poll_once", sometimes_broken)
    thread = _serve_in_thread(w)
    run_id = queue(root, "after the errors")
    try:
        assert wait_until(lambda: state(root, run_id).state == "done", timeout=20)
        assert thread.is_alive()
        assert failures["left"] == 0
    finally:
        w.stop()
        thread.join(10)
    assert not thread.is_alive()


def test_serving_worker_waits_for_the_lock_as_standby(root: Path) -> None:
    run_id = queue(root, "waits for the lock")
    w = make_worker(root, "succeed")
    with runs_lock(root):  # e.g. a CLI run in progress
        thread = _serve_in_thread(w)
        time.sleep(0.5)
        assert thread.is_alive()
        assert state(root, run_id).state == "queued"
    try:
        assert wait_until(lambda: state(root, run_id).state == "done", timeout=20)
    finally:
        w.stop()
        thread.join(10)


def test_worker_stops_while_waiting_for_the_lock(root: Path) -> None:
    w = make_worker(root, "succeed")
    with runs_lock(root):
        thread = _serve_in_thread(w)
        time.sleep(0.2)
        w.stop()
        thread.join(5)
        assert not thread.is_alive()


def test_run_folder_deleted_mid_run_is_not_resurrected(root: Path) -> None:
    run_id = queue(root, "vanishing", T0)
    after = queue(root, "next", T0 + timedelta(seconds=1))
    w = make_worker(root, "vanish")
    assert w.execute(run_id) is None  # nothing to record: the run is gone
    assert not (root / run_id).exists(), "pipeline writer debris was left behind"
    assert w.next_run() == after


def test_deleting_the_run_being_executed_is_refused(root: Path) -> None:
    run_id = queue(root, "busy")
    outcome: dict[str, object] = {}

    def try_delete() -> None:
        assert wait_until(lambda: (root / run_id / "fake_started.txt").exists())
        try:
            runs.delete_run(run_id, root)
        except RunStoreError as exc:
            outcome["error"] = exc
        runs.request_cancel(run_id, root)

    _in_background(try_delete)
    make_worker(root, "wait_for_cancel").poll_once()
    assert isinstance(outcome.get("error"), RunStoreError)
    assert state(root, run_id).state == "cancelled"


# --- helpers for the CLI: executor_lease and run_one_now ----------------------------


def test_executor_lease_locks_recovers_and_beats(root: Path) -> None:
    stale = queue(root, "stale", T0)
    runs.update_status(stale, root=root, state="running")
    run_id = queue(root, "cli run", T0 + timedelta(seconds=1))

    with executor_lease(root) as lease:
        assert state(root, stale).error == INTERRUPTED  # recovered on entry
        with pytest.raises(WorkerAlreadyRunning), make_worker(root).exclusive():
            pass
        lease.current_run = run_id
        runs.update_status(run_id, root=root, state="running")
        assert runs.worker_alive(root)
        assert runs.run_is_live(run_id, root)
        with pytest.raises(RunStoreError):
            runs.delete_run(run_id, root)
        runs.update_status(run_id, root=root, state="done")
        lease.current_run = None
    assert runs.read_heartbeat(root) is None
    with make_worker(root).exclusive():  # released
        pass


def test_executor_lease_refused_while_a_worker_holds_the_lock(root: Path) -> None:
    with make_worker(root).exclusive(), pytest.raises(WorkerAlreadyRunning), executor_lease(root):
        pass


def test_run_one_now(root: Path) -> None:
    run_id = queue(root, "now")
    status = run_one_now(run_id, root, "fake_pipelines:succeed")
    assert status is not None and status.state == "done"
    assert runs.read_heartbeat(root) is None
    assert run_one_now(run_id, root, "fake_pipelines:succeed") is None  # not queued any more


def test_run_one_now_refuses_while_a_worker_runs(root: Path) -> None:
    run_id = queue(root, "blocked")
    with runs_lock(root), pytest.raises(WorkerAlreadyRunning):
        run_one_now(run_id, root, "fake_pipelines:succeed")
    assert state(root, run_id).state == "queued"


# --- integration: real worker / serve processes ------------------------------------


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(TESTS_DIR), env.get("PYTHONPATH")]))
    return env


@pytest.fixture
def start_worker(root: Path) -> Iterator[Callable[[], subprocess.Popen[bytes]]]:
    started: list[subprocess.Popen[bytes]] = []

    def start() -> subprocess.Popen[bytes]:
        cmd = [
            sys.executable,
            "-m",
            "scout_planner.worker",
            "--root",
            str(root),
            "--pipeline",
            "fake_pipelines:by_name",
            "--poll",
            "0.1",
        ]
        proc = subprocess.Popen(cmd, env=_env(), stderr=subprocess.DEVNULL)
        started.append(proc)
        assert wait_until(lambda: runs.worker_alive(root), timeout=20), "worker did not start"
        return proc

    yield start
    for proc in started:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _child_pid(root: Path, run_id: str) -> int:
    path = root / run_id / "fake_started.txt"
    assert wait_until(path.exists, timeout=20)
    return int(path.read_text().split()[0])


@pytest.mark.slow
def test_worker_process_runs_queue_in_order(
    root: Path, start_worker: Callable[[], subprocess.Popen[bytes]]
) -> None:
    proc = start_worker()
    ids = [queue(root, f"slow {i}") for i in range(3)]
    assert wait_until(lambda: all(state(root, r).state == "done" for r in ids), timeout=60)

    statuses = [state(root, r) for r in ids]
    for earlier, later in zip(statuses, statuses[1:], strict=False):
        # oldest first, and strictly one at a time
        assert later.started_at >= earlier.finished_at  # type: ignore[operator]

    proc.send_signal(signal.SIGTERM)
    assert proc.wait(15) == 0
    assert runs.read_heartbeat(root) is None


@pytest.mark.slow
def test_killed_worker_run_is_recovered_on_restart(
    root: Path, start_worker: Callable[[], subprocess.Popen[bytes]]
) -> None:
    first = start_worker()
    run_id = queue(root, "hang until killed")
    child = _child_pid(root, run_id)

    # Hard crash of everything: worker and the run's process tree.
    first.kill()
    first.wait()
    os.killpg(child, signal.SIGKILL)
    assert wait_until(lambda: not pid_alive(child), timeout=5)
    assert state(root, run_id).state == "running"  # stuck, until a worker restarts

    second = start_worker()
    assert wait_until(lambda: state(root, run_id).state == "failed", timeout=20)
    assert state(root, run_id).error == INTERRUPTED

    # The restarted worker works normally...
    after = queue(root, "ok after restart")
    assert wait_until(lambda: state(root, after).state == "done", timeout=30)

    # ...and SIGTERM mid-run stops the run as interrupted and exits cleanly.
    last = queue(root, "hang until stopped")
    last_child = _child_pid(root, last)
    second.send_signal(signal.SIGTERM)
    assert second.wait(20) == 0
    assert (state(root, last).state, state(root, last).error) == ("failed", INTERRUPTED)
    assert wait_until(lambda: not pid_alive(last_child), timeout=5)
    assert runs.read_heartbeat(root) is None


@pytest.mark.slow
def test_orphaned_run_stops_itself_when_worker_dies(
    root: Path, start_worker: Callable[[], subprocess.Popen[bytes]]
) -> None:
    proc = start_worker()
    run_id = queue(root, "hang orphan")
    child = _child_pid(root, run_id)

    proc.kill()  # only the worker; the run's process is left behind
    proc.wait()
    assert wait_until(lambda: not pid_alive(child), timeout=10), "orphaned run kept going"
    assert (state(root, run_id).state, state(root, run_id).error) == ("failed", INTERRUPTED)


@pytest.mark.slow
def test_serve_stops_the_worker_when_the_app_exits(root: Path, tmp_path: Path) -> None:
    pid_file = tmp_path / "worker.pid"
    fake_app = (
        "import sys, time\n"
        "from scout_planner import runs\n"
        f"root = {str(root)!r}\n"
        "deadline = time.monotonic() + 20\n"
        "while time.monotonic() < deadline and not runs.worker_alive(root):\n"
        "    time.sleep(0.05)\n"
        "beat = runs.read_heartbeat(root)\n"
        "if beat is None:\n"
        "    sys.exit(1)\n"
        f"open({str(pid_file)!r}, 'w').write(str(beat.pid))\n"
    )
    code = serve.main(
        ["--root", str(root), "--pipeline", "fake_pipelines:by_name"],
        app_cmd=[sys.executable, "-c", fake_app],
    )
    assert code == 0, "the fake app never saw a live worker"
    worker_pid = int(pid_file.read_text())
    assert not pid_alive(worker_pid), "worker outlived the app"
    assert runs.read_heartbeat(root) is None


@pytest.mark.slow
def test_serve_with_a_live_worker_keeps_it(
    root: Path, start_worker: Callable[[], subprocess.Popen[bytes]]
) -> None:
    existing = start_worker()
    code = serve.main(
        ["--root", str(root)],
        app_cmd=[sys.executable, "-c", "import time; time.sleep(1)"],
    )
    assert code == 0
    # serve's own worker only waited as a standby and was stopped with the app;
    # the existing worker kept the lock and lives on.
    assert existing.poll() is None
    beat = runs.read_heartbeat(root)
    assert beat is not None and beat.pid == existing.pid
    assert runs.worker_alive(root)
