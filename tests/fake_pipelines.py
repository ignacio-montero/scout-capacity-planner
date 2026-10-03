"""Stand-ins for ``scout_planner.pipeline.run_pipeline`` (which arrives in M4).

Each has the contract signature ``(params, run_dir, progress, should_cancel)``
and one scripted behaviour, so the worker can be tested before the real
pipeline exists. The worker imports them by name (``fake_pipelines:succeed``)
inside its child process, so this module must stay importable from ``tests/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from scout_planner.config import Params
from scout_planner.errors import RunCancelled

Progress = Callable[[float, str, str], None]
ShouldCancel = Callable[[], bool]

WEEKS = 8


def _mark(run_dir: Path, name: str) -> None:
    """Leave evidence that the pipeline ran, with its pid and a timestamp."""
    (run_dir / name).write_text(f"{os.getpid()} {time.time_ns()}\n", encoding="utf-8")


def succeed(params: Params, run_dir: Path, progress: Progress, should_cancel: ShouldCancel) -> None:
    """Report a few weeks of progress, write a result file, return."""
    assert isinstance(params, Params)
    for week in range(1, WEEKS + 1):
        progress(week / WEEKS, "simulate", f"simulating week {week}/{WEEKS}")
    _mark(run_dir, "fake_result.txt")


def slow(params: Params, run_dir: Path, progress: Progress, should_cancel: ShouldCancel) -> None:
    """Like ``succeed`` but takes about half a second (for queue-order tests)."""
    _mark(run_dir, "fake_started.txt")
    for week in range(1, WEEKS + 1):
        if should_cancel():
            raise RunCancelled
        progress(week / WEEKS, "simulate", f"simulating week {week}/{WEEKS}")
        time.sleep(0.06)
    _mark(run_dir, "fake_result.txt")


def fail(params: Params, run_dir: Path, progress: Progress, should_cancel: ShouldCancel) -> None:
    """Raise partway through, like a bug in a stage would."""
    progress(0.3, "forecast", "fitting the demand model")
    raise ValueError("demand model did not converge")


def crash(params: Params, run_dir: Path, progress: Progress, should_cancel: ShouldCancel) -> None:
    """Die without any Python cleanup, like a segfault in a native solver."""
    progress(0.2, "simulate", "about to crash")
    os._exit(7)


def wait_for_cancel(
    params: Params, run_dir: Path, progress: Progress, should_cancel: ShouldCancel
) -> None:
    """A long simulation that checks ``should_cancel`` every simulated week."""
    _mark(run_dir, "fake_started.txt")
    for week in range(1, 100_000):
        if should_cancel():
            raise RunCancelled
        progress(min(0.99, week / 1000), "simulate", f"simulating week {week}")
        time.sleep(0.02)
    raise TimeoutError("fake pipeline was never cancelled")


def ignore_cancel(
    params: Params, run_dir: Path, progress: Progress, should_cancel: ShouldCancel
) -> None:
    """Stuck in one long call (think: a solver) and never checks ``should_cancel``.

    Starts a grandchild process too, so tests can check the whole process
    tree is stopped, not just the run's own process.
    """
    progress(0.1, "simulate", "inside a very long solve")
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    (run_dir / "grandchild.pid").write_text(str(helper.pid), encoding="utf-8")
    _mark(run_dir, "fake_started.txt")
    time.sleep(120)


def by_name(params: Params, run_dir: Path, progress: Progress, should_cancel: ShouldCancel) -> None:
    """Dispatch on the run's name prefix, so one worker can run mixed behaviours.

    ``slow ...`` -> :func:`slow`, ``hang ...`` -> :func:`wait_for_cancel`,
    ``fail ...`` -> :func:`fail`, anything else -> :func:`succeed`.
    """
    name = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["name"]
    if name.startswith("slow"):
        slow(params, run_dir, progress, should_cancel)
    elif name.startswith("hang"):
        wait_for_cancel(params, run_dir, progress, should_cancel)
    elif name.startswith("fail"):
        fail(params, run_dir, progress, should_cancel)
    else:
        succeed(params, run_dir, progress, should_cancel)


def vanish(params: Params, run_dir: Path, progress: Progress, should_cancel: ShouldCancel) -> None:
    """The run folder is deleted mid-run, then a stage writer recreates it.

    Mimics ``delete_run`` (rename to a trash name) racing a pipeline whose
    writers call ``mkdir(parents=True)``. The next progress report must stop it.
    """
    progress(0.1, "generate", "writing the world")
    run_dir.rename(run_dir.parent / f"_trash-{run_dir.name}")
    (run_dir / "raw").mkdir(parents=True)  # a writer resurrecting the folder
    (run_dir / "raw" / "scouts.parquet").write_bytes(b"partial")
    progress(0.2, "forecast", "should never get here")
    _mark(run_dir, "fake_result.txt")
