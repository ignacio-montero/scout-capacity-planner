"""``make app``: start the background worker and the Streamlit app together.

The worker runs as a separate process (D-013) and must not outlive the app:

* Ctrl-C reaches every process in the terminal's foreground group (this
  launcher, Streamlit, the worker). Streamlit and the worker shut themselves
  down; this launcher waits for them and forces whatever is left.
* ``kill`` / closing the terminal (SIGTERM / SIGHUP) to this launcher alone:
  it stops Streamlit and the worker itself.
* This launcher killed with SIGKILL (no cleanup code can run): the worker was
  started with ``--exit-with-parent``, notices it was orphaned within one
  poll, and stops on its own.

If a live worker heartbeat already exists (e.g. a worker started by hand with
``python -m scout_planner.worker``), no second worker is started; the worker's
own lock would refuse a second one anyway.

Usage::

    python -m scout_planner.serve [--root data/runs] [--pipeline module:function]
                                  [-- extra streamlit arguments]
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import logging
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from scout_planner import runs
from scout_planner.worker import DEFAULT_PIPELINE

# The Streamlit entry script (built by the app milestone, M5).
APP_ENTRY = runs.REPO_ROOT / "app" / "main.py"
# The worker may need a few seconds: it stops the current run (up to 5 s
# SIGTERM grace) and records it as interrupted before exiting.
WORKER_STOP_TIMEOUT_S = 15.0
APP_STOP_TIMEOUT_S = 10.0

log = logging.getLogger("scout_planner.serve")


def worker_command(root: Path, pipeline: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scout_planner.worker",
        "--root",
        str(root),
        "--pipeline",
        pipeline,
        "--exit-with-parent",
    ]


def app_command(entry: Path, extra: Sequence[str] = ()) -> list[str]:
    return [sys.executable, "-m", "streamlit", "run", str(entry), *extra]


def _stop(proc: subprocess.Popen[bytes] | None, timeout: float, what: str) -> None:
    """SIGTERM, wait up to ``timeout``, then SIGKILL. Safe to call twice."""
    if proc is None or proc.poll() is not None:
        return
    log.info("stopping %s (pid %d)", what, proc.pid)
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        log.warning("%s did not stop within %.0f s; killing it", what, timeout)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        proc.wait()


def _wait_after_ctrl_c(proc: subprocess.Popen[bytes], timeout: float) -> None:
    """Ctrl-C already reached ``proc``; give it time to exit cleanly."""
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout)


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="python -m scout_planner.serve",
        description="Start the simulator app and its background worker.",
    )
    parser.add_argument("--root", type=Path, default=runs.DEFAULT_ROOT, help="runs folder")
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE, help="pipeline as module:function")
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    return args, extra


def main(argv: Sequence[str] | None = None, app_cmd: Sequence[str] | None = None) -> int:
    """Run the app until it exits; always stop the worker we started.

    ``app_cmd`` replaces the Streamlit command (tests use a tiny script).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s serve %(levelname)s: %(message)s")
    args, extra = _parse_args(argv)
    if app_cmd is None:
        if not APP_ENTRY.is_file():
            log.error("app entry %s does not exist yet", APP_ENTRY)
            return 2
        app_cmd = app_command(APP_ENTRY, extra)

    worker: subprocess.Popen[bytes] | None = None
    app: subprocess.Popen[bytes] | None = None

    def cleanup() -> None:
        _stop(app, APP_STOP_TIMEOUT_S, "app")
        _stop(worker, WORKER_STOP_TIMEOUT_S, "worker")

    def on_signal(signum: int, _frame: object) -> None:
        # Turn SIGTERM / SIGHUP into a normal exit so the finally block runs.
        raise SystemExit(128 + signum)

    previous: dict[int, Callable[..., object] | int | None] = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        with contextlib.suppress(ValueError):  # only possible in the main thread
            previous[sig] = signal.signal(sig, on_signal)
    # Last line of defence if the interpreter exits some other way.
    atexit.register(cleanup)

    try:
        args.root.mkdir(parents=True, exist_ok=True)
        beat = runs.read_heartbeat(args.root)
        if runs.worker_alive(args.root) and beat is not None:
            log.info("a worker is already running (pid %d); not starting another", beat.pid)
        else:
            worker = subprocess.Popen(worker_command(args.root, args.pipeline))
            log.info("started worker (pid %d)", worker.pid)

        # The app reads its runs folder from SCOUT_RUNS_ROOT; keep it in step with the worker.
        app_env = {**os.environ, "SCOUT_RUNS_ROOT": str(args.root)}
        app = subprocess.Popen(list(app_cmd), env=app_env)
        try:
            code = app.wait()
        except KeyboardInterrupt:
            log.info("Ctrl-C: shutting down")
            _wait_after_ctrl_c(app, APP_STOP_TIMEOUT_S)
            if worker is not None:
                _wait_after_ctrl_c(worker, WORKER_STOP_TIMEOUT_S)
            code = 130
        if worker is not None and worker.poll() not in (None, 0):
            log.warning("the worker exited early with code %s", worker.returncode)
        return code
    finally:
        cleanup()
        atexit.unregister(cleanup)
        for sig, handler in previous.items():
            with contextlib.suppress(ValueError, TypeError):
                signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
