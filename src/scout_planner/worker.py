"""Background worker: executes queued runs one at a time, oldest first (D-013).

The worker is a **supervisor**. It never runs the simulation in its own
process; for each run it starts a **child process** (``multiprocessing``,
``spawn`` start method) that calls the pipeline and reports progress by
writing ``status.json`` itself. The worker meanwhile keeps its heartbeat
fresh, cancels flagged queued runs, and watches the child:

* child exits after recording ``done`` / ``failed`` / ``cancelled`` -> nothing to do;
* child dies without recording a final state (segfault, out of memory, killed)
  -> the worker marks the run ``failed`` with the exit code;
* user asked to cancel but the child does not stop within ``cancel_grace_s``
  (stuck inside one long solver call) -> the worker kills it and marks ``cancelled``;
* the worker itself is asked to stop (SIGTERM / SIGINT / SIGHUP) -> it kills the
  child and marks the run ``failed`` with error ``"interrupted"``, the same
  outcome crash recovery gives, so "app closed mid-run" always looks the same.

Each child is the leader of its own **process group**, so the worker can stop
the child *and* any processes the pipeline started (one per seed) with one
signal, and a Ctrl-C in the terminal reaches the worker only, not the run.

Usage::

    python -m scout_planner.worker [--root data/runs] [--pipeline module:function]
                                   [--poll 1.0] [--cancel-grace 30] [--once]
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib
import logging
import math
import multiprocessing
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from scout_planner import runs
from scout_planner.errors import IllegalTransition, RunCancelled, RunNotFound, RunStoreError

DEFAULT_PIPELINE = "scout_planner.pipeline:run_pipeline"
DEFAULT_POLL_S = 1.0
DEFAULT_CANCEL_GRACE_S = 30.0
# Progress writes from the child: at most one per this many seconds (<= 2/s),
# except that a change of stage is always written at once.
PROGRESS_MIN_INTERVAL_S = 0.5
# How long a child gets to exit after SIGTERM before it gets SIGKILL.
KILL_TIMEOUT_S = 5.0
INTERRUPTED = "interrupted"  # status.error when the worker stopped mid-run (DATA_CONTRACTS)

log = logging.getLogger("scout_planner.worker")

Pipeline = Callable[..., None]


class PipelineUnavailable(Exception):
    """The pipeline callable named by ``--pipeline`` can't be imported."""


class WorkerAlreadyRunning(RuntimeError):
    """Another worker holds the lock on this runs root."""


# --- pipeline lookup ---------------------------------------------------------------


def load_pipeline(spec: str) -> Pipeline:
    """Resolve ``"package.module:function"`` to the function.

    Done inside the run's child process, so every run imports the current code
    (edit the pipeline, queue a run, no worker restart needed).
    """
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise PipelineUnavailable(f"pipeline must look like 'module:function', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and module_name.startswith(exc.name):
            if spec == DEFAULT_PIPELINE:
                raise PipelineUnavailable(
                    f"pipeline not implemented yet: module {module_name} does not exist"
                ) from exc
            raise PipelineUnavailable(f"pipeline module {module_name!r} not found") from exc
        raise  # the module exists but one of *its* imports is missing: a real bug
    func = getattr(module, attr, None)
    if not callable(func):
        raise PipelineUnavailable(f"{module_name} has no function {attr!r}")
    return func


# --- the child process (one per run) -----------------------------------------------


class ThrottledProgress:
    """The ``progress`` callback handed to the pipeline: writes ``status.json``,
    but at most once per ``min_interval_s`` (a stage change always goes through).

    The pipeline may report every simulated day; without the throttle that is
    hundreds of fsync'ed file writes per second for no visible gain. The last
    suppressed update is kept in :attr:`pending` so the final status can include it.
    """

    def __init__(
        self,
        run_id: str,
        root: Path,
        min_interval_s: float = PROGRESS_MIN_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_id = run_id
        self.root = root
        self.min_interval_s = min_interval_s
        self.clock = clock
        self.pending: dict[str, Any] | None = None
        self.writes = 0
        self._last_write = -math.inf
        self._last_stage: str | None = None

    def __call__(self, fraction: float, stage: str, message: str) -> None:
        values = {
            "progress": min(1.0, max(0.0, float(fraction))),
            "stage": stage,
            "message": message,
        }
        now = self.clock()
        if stage == self._last_stage and now - self._last_write < self.min_interval_s:
            self.pending = values
            return
        self._write(values, now)

    def latest(self) -> dict[str, Any]:
        """The most recent progress values not yet written (empty if none)."""
        return dict(self.pending or {})

    def _write(self, values: dict[str, Any], now: float) -> None:
        self._last_write = now
        self._last_stage = values["stage"]
        self.pending = None
        try:
            runs.update_status(self.run_id, root=self.root, **values)
            self.writes += 1
        except (RunStoreError, OSError) as exc:
            # A progress line is not worth failing a run over; the final
            # status write will surface any real problem.
            log.warning("progress update not recorded: %s", exc)


def _short_error(exc: BaseException) -> str:
    """One line for status.error; the full traceback goes to log.txt."""
    if isinstance(exc, PipelineUnavailable):
        text = str(exc)
    else:
        text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    first = text.strip().splitlines()[0] if text.strip() else type(exc).__name__
    return first[:300]


def _redirect_output_to(path: Path) -> None:
    """Send this process's stdout/stderr (file descriptors 1 and 2) to ``path``.

    At the descriptor level, so output from C extensions (solver logs) is
    captured too, not just Python ``print``.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(line_buffering=True)  # type: ignore[union-attr]


def _child_main(root: str, run_id: str, pipeline_spec: str, progress_interval_s: float) -> None:
    """Entry point of a run's child process. Records the final state itself."""
    # Own process group: Ctrl-C in the terminal goes to the worker, which then
    # decides; and the worker can signal this whole tree (pipeline's seed
    # processes included) with one os.killpg.
    with contextlib.suppress(OSError):
        os.setpgid(0, 0)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    root_path = Path(root)
    folder = runs.run_dir(run_id, root_path)
    _redirect_output_to(folder / runs.LOG_FILE)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    run_log = logging.getLogger("scout_planner.run")
    run_log.info("run %s: child pid %d, pipeline %s", run_id, os.getpid(), pipeline_spec)

    parent_pid = os.getppid()
    orphaned = False

    def should_cancel() -> bool:
        nonlocal orphaned
        # If the worker died (SIGKILL), we are re-parented: stop instead of
        # computing a result nobody is waiting for.
        if os.getppid() != parent_pid:
            orphaned = True
            return True
        return runs.cancel_requested(run_id, root_path)

    progress = ThrottledProgress(run_id, root_path, progress_interval_s)
    final: dict[str, Any]
    try:
        pipeline = load_pipeline(pipeline_spec)
        params = runs.read_params(run_id, root_path)
        pipeline(params, folder, progress, should_cancel)
    except RunCancelled:
        if orphaned:
            run_log.warning("worker is gone; stopping this run")
            final = {**progress.latest(), "state": "failed", "error": INTERRUPTED}
        else:
            run_log.info("cancelled on request")
            final = {**progress.latest(), "state": "cancelled"}
    except Exception as exc:
        run_log.exception("run failed")
        final = {**progress.latest(), "state": "failed", "error": _short_error(exc)}
    else:
        run_log.info("run finished")
        final = {"state": "done", "stage": None, "message": "finished"}

    try:
        runs.update_status(run_id, root=root_path, **final)
    except (RunStoreError, OSError) as exc:
        run_log.warning("final state %s not recorded: %s", final["state"], exc)


# --- process-tree helpers ----------------------------------------------------------


def _signal_tree(pid: int, sig: int) -> None:
    """Signal the child's whole process group; fall back to the child alone
    (it may not have created its group yet)."""
    try:
        os.killpg(pid, sig)
        return
    except (ProcessLookupError, PermissionError):
        pass
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, sig)


def _stop_child(proc: multiprocessing.process.BaseProcess) -> None:
    """SIGTERM the run's process tree, then SIGKILL whatever is left."""
    if proc.pid is None:
        return
    _signal_tree(proc.pid, signal.SIGTERM)
    proc.join(KILL_TIMEOUT_S)
    if proc.exitcode is None:
        _signal_tree(proc.pid, signal.SIGKILL)
        proc.join()


def _reap_group(pid: int | None) -> None:
    """Kill any processes the pipeline left behind in the child's group."""
    if pid is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)


def _describe_exit(code: int | None) -> str:
    if code is None:
        return "still running"
    if code < 0:
        try:
            return f"killed by signal {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exit code {code}"


# --- the worker --------------------------------------------------------------------


class Worker:
    """The queue loop. One instance per runs root; one run at a time."""

    def __init__(
        self,
        root: Path | str | None = None,
        pipeline: str = DEFAULT_PIPELINE,
        poll_s: float = DEFAULT_POLL_S,
        cancel_grace_s: float = DEFAULT_CANCEL_GRACE_S,
        progress_interval_s: float = PROGRESS_MIN_INTERVAL_S,
        exit_with_parent: bool = False,
    ) -> None:
        self.root = Path(root) if root is not None else runs.DEFAULT_ROOT
        self.pipeline = pipeline
        self.poll_s = poll_s
        self.cancel_grace_s = cancel_grace_s
        self.progress_interval_s = progress_interval_s
        self.exit_with_parent = exit_with_parent
        self.pid = os.getpid()
        self._parent_pid = os.getppid()
        self._stop = threading.Event()
        self._mp = multiprocessing.get_context("spawn")

    # -- lifecycle --

    def stop(self) -> None:
        """Ask the loop to stop (safe from a signal handler)."""
        self._stop.set()

    def stopping(self) -> bool:
        if self.exit_with_parent and not self._stop.is_set() and os.getppid() != self._parent_pid:
            log.warning("parent process is gone; stopping")
            self._stop.set()
        return self._stop.is_set()

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold an exclusive lock on ``<root>/_worker.lock`` for the worker's lifetime.

        Two workers on one root would both pick the same queued run. The OS
        releases a ``flock`` when the process dies, however it dies, so a crashed
        worker never leaves a stale lock behind (unlike a "pid file").
        """
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.root / runs.LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise WorkerAlreadyRunning(
                    f"another worker is already running on {self.root}"
                ) from None
            os.ftruncate(fd, 0)
            os.write(fd, f"{self.pid}\n".encode())
            yield
        finally:
            os.close(fd)  # closing the descriptor releases the lock

    @contextlib.contextmanager
    def signal_handlers(self) -> Iterator[None]:
        """SIGTERM / SIGINT / SIGHUP -> graceful stop; a second Ctrl-C stops hard.

        The previous handlers are restored on exit, so calling the loop from
        another program (or a test) leaves its signal handling as it was.
        """

        def handle(signum: int, _frame: object) -> None:
            log.info("received %s; stopping", signal.Signals(signum).name)
            self.stop()
            signal.signal(signal.SIGINT, signal.default_int_handler)

        previous = {}
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            with contextlib.suppress(ValueError):  # only possible in the main thread
                previous[sig] = signal.signal(sig, handle)
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    # -- queue steps (each one small and testable on its own) --

    def beat(self, current_run: str | None = None) -> None:
        """Rewrite the heartbeat file."""
        runs.write_heartbeat(self.pid, current_run, self.root)

    def recover(self) -> list[str]:
        """Crash recovery: runs left ``running`` by a previous worker become ``failed``."""
        recovered = []
        for status in runs.list_runs(self.root):
            if status.state != "running":
                continue
            try:
                runs.update_status(status.run_id, root=self.root, state="failed", error=INTERRUPTED)
            except (RunStoreError, OSError) as exc:
                log.warning("could not recover %s: %s", status.run_id, exc)
                continue
            runs.append_log(
                status.run_id,
                "worker restarted: this run was in progress and is marked failed (interrupted)",
                self.root,
            )
            log.info("marked %s as failed (interrupted)", status.run_id)
            recovered.append(status.run_id)
        runs.cleanup_stale_temp(self.root)
        return recovered

    def cancel_flagged_queued(self) -> list[str]:
        """Queued runs with a ``cancel`` flag go straight to ``cancelled`` (gap G4)."""
        cancelled = []
        for status in runs.queued_runs(self.root, include_cancel_requested=True):
            if not runs.cancel_requested(status.run_id, self.root):
                continue
            try:
                runs.update_status(
                    status.run_id,
                    root=self.root,
                    state="cancelled",
                    message="cancelled before it started",
                )
            except (RunStoreError, OSError) as exc:
                log.warning("could not cancel %s: %s", status.run_id, exc)
                continue
            log.info("cancelled queued run %s", status.run_id)
            cancelled.append(status.run_id)
        return cancelled

    def next_run(self) -> str | None:
        """The oldest queued run without a cancel request, or None."""
        queue = runs.queued_runs(self.root)
        return queue[0].run_id if queue else None

    def poll_once(self) -> str | None:
        """One poll: heartbeat, cancel flagged queued runs, execute the oldest run.

        Returns the id of the run it executed, or None if the queue was empty.
        """
        self.beat()
        self.cancel_flagged_queued()
        run_id = self.next_run()
        if run_id is not None and not self._stop.is_set():
            self.execute(run_id)
            return run_id
        return None

    # -- executing one run --

    def execute(self, run_id: str) -> runs.RunStatus | None:
        """Run one queued run in a child process and make sure it ends in a final state."""
        try:
            runs.update_status(
                run_id,
                root=self.root,
                state="running",
                stage=None,
                message="starting",
                progress=0.0,
            )
        except (RunNotFound, IllegalTransition) as exc:
            log.warning("skipping %s: %s", run_id, exc)  # deleted or changed since we looked
            return None
        log.info("running %s", run_id)
        runs.append_log(run_id, f"worker pid {self.pid}: starting ({self.pipeline})", self.root)
        self.beat(run_id)

        proc = self._mp.Process(
            target=_child_main,
            args=(str(self.root), run_id, self.pipeline, self.progress_interval_s),
            name=f"scout-run-{run_id}",
        )
        try:
            proc.start()
        except Exception as exc:
            log.exception("could not start the run process for %s", run_id)
            result = self._finalise(
                run_id, None, {"state": "failed", "error": f"could not start run process: {exc}"}
            )
            self.beat(None)
            return result

        override: dict[str, Any] | None = None
        try:
            override = self._supervise(run_id, proc)
        except BaseException:
            # Anything unexpected in the supervisor (even a second Ctrl-C):
            # never leave an unsupervised run behind.
            _stop_child(proc)
            override = {"state": "failed", "error": INTERRUPTED}
            raise
        finally:
            _reap_group(proc.pid)
            result = self._finalise(run_id, proc.exitcode, override)
            self.beat(None)
        return result

    def _supervise(
        self, run_id: str, proc: multiprocessing.process.BaseProcess
    ) -> dict[str, Any] | None:
        """Wait for the child, staying responsive. Returns a forced final state, if any."""
        cancel_seen: float | None = None
        while True:
            proc.join(self.poll_s)
            if proc.exitcode is not None:
                return None
            self.beat(run_id)
            self.cancel_flagged_queued()
            if self.stopping():
                log.info("stopping %s: worker is shutting down", run_id)
                _stop_child(proc)
                return {"state": "failed", "error": INTERRUPTED}
            if runs.cancel_requested(run_id, self.root):
                cancel_seen = cancel_seen if cancel_seen is not None else time.monotonic()
                if time.monotonic() - cancel_seen > self.cancel_grace_s:
                    log.warning(
                        "%s ignored cancel for %.0f s; killing it", run_id, self.cancel_grace_s
                    )
                    _stop_child(proc)
                    return {
                        "state": "cancelled",
                        "message": f"stopped by force after {self.cancel_grace_s:.0f} s",
                    }

    def _finalise(
        self, run_id: str, exitcode: int | None, override: dict[str, Any] | None
    ) -> runs.RunStatus | None:
        """If the child didn't record a final state, record one for it."""
        try:
            status = runs.read_status(run_id, self.root)
        except RunNotFound:
            log.warning("%s disappeared while running", run_id)
            return None
        if status.state != "running":
            log.info("%s finished: %s", run_id, status.state)
            return status
        changes = override or {
            "state": "failed",
            "error": f"run process ended without a result ({_describe_exit(exitcode)})",
        }
        runs.append_log(
            run_id,
            f"worker: child {_describe_exit(exitcode)}; marking {changes['state']}"
            + (f" ({changes['error']})" if changes.get("error") else ""),
            self.root,
        )
        try:
            status = runs.update_status(run_id, root=self.root, **changes)
        except (RunStoreError, OSError) as exc:
            log.error("could not record final state of %s: %s", run_id, exc)
            return None
        log.info("%s finished: %s", run_id, status.state)
        return status

    # -- loops --

    def serve_forever(self) -> None:
        """The long-running loop used by ``make app``. Returns after a stop request."""
        with self.exclusive(), self.signal_handlers():
            self.recover()
            log.info("worker %d watching %s (pipeline %s)", self.pid, self.root, self.pipeline)
            try:
                while not self.stopping():
                    if self.poll_once() is None:
                        self._stop.wait(self.poll_s)
            finally:
                runs.remove_heartbeat(self.root, pid=self.pid)
                log.info("worker %d stopped", self.pid)

    def drain(self) -> list[str]:
        """``--once``: recover, execute everything queued right now, then return."""
        executed: list[str] = []
        with self.exclusive(), self.signal_handlers():
            self.recover()
            try:
                while not self.stopping():
                    run_id = self.poll_once()
                    if run_id is None:
                        break
                    executed.append(run_id)
            finally:
                runs.remove_heartbeat(self.root, pid=self.pid)
        return executed


# --- command line ------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scout_planner.worker",
        description="Execute queued simulation runs one at a time, oldest first.",
    )
    parser.add_argument("--root", type=Path, default=runs.DEFAULT_ROOT, help="runs folder")
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE, help="pipeline as module:function")
    parser.add_argument("--poll", type=float, default=DEFAULT_POLL_S, help="seconds between polls")
    parser.add_argument(
        "--cancel-grace",
        type=float,
        default=DEFAULT_CANCEL_GRACE_S,
        help="seconds a run may ignore a cancel request before it is killed",
    )
    parser.add_argument("--once", action="store_true", help="execute what is queued now, then exit")
    parser.add_argument(
        "--exit-with-parent",
        action="store_true",
        help="stop when the process that started the worker exits (used by serve.py)",
    )
    args = parser.parse_args(argv)
    if ":" not in args.pipeline:
        parser.error("--pipeline must look like module:function")
    if args.poll <= 0:
        parser.error("--poll must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s worker %(levelname)s: %(message)s")
    worker = Worker(
        root=args.root,
        pipeline=args.pipeline,
        poll_s=args.poll,
        cancel_grace_s=args.cancel_grace,
        exit_with_parent=args.exit_with_parent,
    )
    try:
        if args.once:
            done = worker.drain()
            log.info("executed %d run(s)", len(done))
        else:
            worker.serve_forever()
    except WorkerAlreadyRunning as exc:
        log.error("%s", exc)
        return 3
    return 0


if __name__ == "__main__":
    # Run main() from the importable module, not from this __main__ copy, so the
    # child-process entry point pickles as scout_planner.worker._child_main.
    from scout_planner import worker as _worker

    sys.exit(_worker.main())
