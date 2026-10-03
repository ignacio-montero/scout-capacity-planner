"""``run_pipeline``: one parameter set in, one complete run folder out (DATA_CONTRACTS §0).

The *imperative shell* around the pure stages (pipes and filters)::

    generate -> raw/            forecast -> forecast/backtest    capacity plan -> plans
    simulate (N replications, in parallel processes) -> results/*.parquet + summary.json

Called by the worker (inside the run's child process) and by the CLI. Its
signature is the contract the run store and worker were built against, so it
takes plain callbacks and knows nothing about ``status.json``.

Parallel replications
---------------------
Replications are independent, so they run in a **process pool**
(``concurrent.futures.ProcessPoolExecutor``, ``spawn`` start method,
``max_workers = min(N, cpu_count - 1)``). Processes rather than threads
because the simulation is pure-Python CPU work and the GIL would serialise
threads. ``spawn`` (a fresh interpreter) rather than ``fork``: forking a
process that has threads (CP-SAT, pyarrow) can deadlock, and spawn behaves
the same on every OS.

* **Progress** flows child -> parent through a ``multiprocessing.Queue``:
  each replication posts ``(replication, weeks done, total weeks)`` once per
  simulated week; the parent averages the fractions (monotonic by
  construction) and calls ``progress``.
* **Cancellation** flows parent -> children through a shared
  ``multiprocessing.Event``: the parent polls ``should_cancel()`` a few times
  per second, sets the event, and each replication checks it at the start of
  every simulated week and raises ``RunCancelled``. The parent then cancels
  not-yet-started replications, waits for the pool to shut down (no orphan
  processes) and raises ``RunCancelled`` itself.

The worker runs this function in a non-daemonic child, so starting a pool from
there is allowed; the worker's process-group kill also reaches the pool.
With one replication (or one usable core) everything runs in-process.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing
import os
import queue
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scout_planner import forecast, generate, metrics, plan, simulate
from scout_planner.config import Params
from scout_planner.errors import RunCancelled
from scout_planner.generate import World

Progress = Callable[[float, str, str], None]
ShouldCancel = Callable[[], bool]

RESULTS_DIR = "results"
SUMMARY_FILE = "summary.json"
POLL_S = 0.2  # parent's poll interval for progress and cancellation

# Share of the progress bar per stage (simulate dominates the wall time).
_P_FORECAST, _P_PLAN, _P_SIM, _P_WRITE = 0.03, 0.08, 0.10, 0.98

RESULT_SCHEMAS: dict[str, pa.Schema] = {
    "weekly": pa.schema(
        [
            ("seed", pa.int64()),
            ("week_start", pa.date32()),
            ("open_requests", pa.int64()),
            ("open_hours", pa.float64()),
            ("late_requests", pa.int64()),
            ("team_full_time", pa.int64()),
            ("team_freelance", pa.int64()),
        ]
    ),
    "requests": pa.schema(
        [
            ("seed", pa.int64()),
            ("request_id", pa.string()),
            ("completed_date", pa.date32()),
            ("turnaround_days", pa.float64()),
            ("on_time", pa.bool_()),
            ("at_risk_day_one", pa.bool_()),
            ("scouts_involved", pa.int64()),
            ("scored", pa.bool_()),
            ("received_date", pa.date32()),
            ("due_date", pa.date32()),
            ("skill_type", pa.string()),
            ("needs_live_view", pa.bool_()),
        ]
    ),
}


def default_max_workers(n_replications: int) -> int:
    """``min(N, cpu_count - 1)``, at least 1: leave a core for the app and the worker."""
    return max(1, min(n_replications, (os.cpu_count() or 2) - 1))


def _check(should_cancel: ShouldCancel) -> None:
    if should_cancel():
        raise RunCancelled


# --- the contract entry point -------------------------------------------------------


def run_pipeline(
    params: Params,
    run_dir: Path,
    progress: Progress,
    should_cancel: ShouldCancel,
    *,
    max_workers: int | None = None,
) -> None:
    """Run every stage for ``params`` and write the complete run folder into ``run_dir``.

    ``progress(fraction, stage, message)`` is called at each stage change and
    about once per simulated week; ``should_cancel()`` is polled between stages
    and about weekly during the simulation; raises ``RunCancelled`` if it says so.
    ``max_workers`` (keyword-only, outside the contract) caps the process pool.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    progress(0.0, "generate", "generating the synthetic world")
    _check(should_cancel)
    world = generate.generate_world(params)
    generate.write_world(world, run_dir)

    progress(_P_FORECAST, "forecast", "fitting the demand forecast and backtest")
    _check(should_cancel)
    fc, bt = forecast.build_forecast(world, params)
    forecast.write_forecast(fc, bt, run_dir)

    progress(_P_PLAN, "capacity_plan", "building the capacity plan")
    _check(should_cancel)
    capacity, hiring = plan.build_capacity_plan(
        world.scouts, world.scout_unavailability, fc, params
    )
    plan.write_plan(capacity, hiring, run_dir)

    simulate_stage(
        params, run_dir, progress, should_cancel, world=world, hiring=hiring,
        max_workers=max_workers,
    )  # fmt: skip


def simulate_stage(
    params: Params,
    run_dir: Path,
    progress: Progress,
    should_cancel: ShouldCancel,
    *,
    world: World | None = None,
    hiring: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Stage [5] on a run folder that already has ``raw/`` and the plans; returns the summary.

    ``world`` / ``hiring`` default to what is on disk (``make simulate``).
    """
    run_dir = Path(run_dir)
    world = generate.read_world(run_dir) if world is None else world
    hiring = plan.read_plan(run_dir)[1] if hiring is None else hiring

    def sim_progress(fraction: float, message: str) -> None:
        progress(_P_SIM + (_P_WRITE - _P_SIM) * fraction, "simulate", message)

    sim_progress(0.0, f"simulating {params.sim.seeds} seed(s)")
    _check(should_cancel)
    results = simulate_replications(
        params, world, hiring, sim_progress, should_cancel, max_workers=max_workers
    )
    progress(_P_WRITE, "simulate", "writing results")
    summary = write_results(results, params, run_dir)
    progress(1.0, "simulate", "finished")
    return summary


# --- replications: in-process or in a process pool ----------------------------------

_child_queue: Any = None
_child_cancel: Any = None


def _init_child(progress_queue: Any, cancel_event: Any) -> None:
    """Pool initializer: keep the shared queue and event in module globals."""
    global _child_queue, _child_cancel
    _child_queue, _child_cancel = progress_queue, cancel_event


def _run_replication(
    params: Params,
    world0: World | None,
    hiring: pd.DataFrame,
    replication: int,
    on_week: simulate.OnWeek | None,
    should_cancel: ShouldCancel | None,
) -> simulate.ReplicationResult:
    world = simulate.replication_world(params, replication, world0)
    inputs = simulate.prepare_replication(params, world, hiring, replication)
    return simulate.simulate_replication(inputs, on_week=on_week, should_cancel=should_cancel)


def _child_replication(
    params: Params, world0: World | None, hiring: pd.DataFrame, replication: int
) -> simulate.ReplicationResult:
    """Entry point inside a pool process (module level, so ``spawn`` can pickle it)."""
    q, ev = _child_queue, _child_cancel

    def on_week(done: int, total: int) -> None:
        q.put((replication, done, total))

    return _run_replication(params, world0, hiring, replication, on_week, ev.is_set)


def simulate_replications(
    params: Params,
    world0: World,
    hiring: pd.DataFrame,
    progress: Callable[[float, str], None],
    should_cancel: ShouldCancel,
    *,
    max_workers: int | None = None,
) -> list[simulate.ReplicationResult]:
    """All ``sim.seeds`` replications; replication 0 uses ``world0`` (the one in ``raw/``)."""
    n = params.sim.seeds
    workers = default_max_workers(n) if max_workers is None else max(1, min(max_workers, n))
    fractions = [0.0] * n

    def report(replication: int, done: int, total: int) -> None:
        fractions[replication] = max(fractions[replication], done / total)
        progress(
            sum(fractions) / n,
            f"simulating seed {replication + 1}/{n}, week {done}/{total}",
        )

    if workers == 1:
        results = []
        for k in range(n):
            results.append(
                _run_replication(
                    params,
                    world0,
                    hiring,
                    k,
                    lambda done, total, k=k: report(k, done, total),
                    should_cancel,
                )  # fmt: skip
            )
        return results
    return _pool_replications(params, world0, hiring, n, workers, report, should_cancel)


def _pool_replications(
    params: Params,
    world0: World,
    hiring: pd.DataFrame,
    n: int,
    workers: int,
    report: Callable[[int, int, int], None],
    should_cancel: ShouldCancel,
) -> list[simulate.ReplicationResult]:
    ctx = multiprocessing.get_context("spawn")
    progress_queue = ctx.Queue()
    cancel_event = ctx.Event()
    executor = cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_child,
        initargs=(progress_queue, cancel_event),
    )

    def drain() -> None:
        while True:
            try:
                report(*progress_queue.get_nowait())
            except queue.Empty:
                return

    cancelled = False
    try:
        futures = [
            # Only replication 0 needs the parent's world; the others regenerate theirs.
            executor.submit(_child_replication, params, world0 if k == 0 else None, hiring, k)
            for k in range(n)
        ]
        pending = set(futures)
        while pending:
            _, pending = cf.wait(pending, timeout=POLL_S, return_when=cf.FIRST_EXCEPTION)
            drain()
            if not cancelled and should_cancel():
                cancelled = True
                cancel_event.set()
                for f in futures:
                    f.cancel()  # not started yet: never starts
            failed = [
                f
                for f in futures
                if f.done() and not f.cancelled() and f.exception() is not None
                and not isinstance(f.exception(), RunCancelled)
            ]  # fmt: skip
            if failed:
                cancel_event.set()  # a bug in one replication: stop the others too
                raise failed[0].exception()  # type: ignore[misc]
        drain()
        if cancelled:
            raise RunCancelled
        return [f.result() for f in futures]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        progress_queue.close()
        progress_queue.join_thread()


# --- results I/O ---------------------------------------------------------------------


def _write_table(df: pd.DataFrame, path: Path, schema: pa.Schema | None = None) -> None:
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path)


def write_results(
    results: list[simulate.ReplicationResult], params: Params, run_dir: Path
) -> dict[str, Any]:
    """Write ``results/*.parquet`` (seeds, weekly, requests) and ``summary.json``; return it."""
    out = Path(run_dir) / RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    seeds = metrics.seeds_frame(results, params)
    _write_table(seeds, out / "seeds.parquet")
    _write_table(metrics.weekly_frame(results), out / "weekly.parquet", RESULT_SCHEMAS["weekly"])
    _write_table(
        metrics.requests_frame(results), out / "requests.parquet", RESULT_SCHEMAS["requests"]
    )
    summary = metrics.summarise(seeds, params)
    tmp = Path(run_dir) / f".{SUMMARY_FILE}.tmp"
    tmp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, Path(run_dir) / SUMMARY_FILE)
    return summary


def read_summary(run_dir: Path) -> dict[str, Any]:
    """``summary.json`` of a finished run."""
    return json.loads((Path(run_dir) / SUMMARY_FILE).read_text(encoding="utf-8"))


# --- executing a queued run in this process (CLI `run` and `sweep`) -----------------


def execute_run_inline(
    run_id: str,
    root: Path | str | None = None,
    *,
    echo: Callable[[float, str, str], None] | None = None,
    max_workers: int | None = None,
) -> str:
    """Execute one queued run in this process, recording its lifecycle in ``status.json``.

    Call it only while holding ``worker.executor_lease(root)`` (the runs
    lock): the background worker's crash recovery assumes nobody else is
    executing runs in that folder.

    The CLI's synchronous path: the same state machine as the worker
    (``queued -> running -> done | failed | cancelled``), progress through the
    worker's throttled status writer, errors to ``log.txt``. Returns the final
    state, or ``"skipped"`` if the run was no longer queued (e.g. a background
    worker picked it up first). Ctrl-C marks the run failed (interrupted).
    """
    from scout_planner import runs
    from scout_planner.errors import IllegalTransition
    from scout_planner.worker import INTERRUPTED, ThrottledProgress

    try:
        runs.update_status(
            run_id, root=root, state="running", stage=None, message="starting", progress=0.0
        )
    except IllegalTransition:
        return "skipped"
    root_path = Path(root) if root is not None else runs.DEFAULT_ROOT
    folder = runs.run_dir(run_id, root_path)
    runs.append_log(run_id, f"cli pid {os.getpid()}: executing in-process", root_path)
    writer = ThrottledProgress(run_id, root_path)

    def progress(fraction: float, stage: str, message: str) -> None:
        writer(fraction, stage, message)
        if echo is not None:
            echo(fraction, stage, message)

    def should_cancel() -> bool:
        return runs.cancel_requested(run_id, root_path)

    started = time.perf_counter()
    try:
        params = runs.read_params(run_id, root_path)
        run_pipeline(params, folder, progress, should_cancel, max_workers=max_workers)
    except RunCancelled:
        final = {**writer.latest(), "state": "cancelled"}
    except KeyboardInterrupt:
        runs.update_status(run_id, root=root_path, state="failed", error=INTERRUPTED)
        raise
    except Exception as exc:
        runs.append_log(run_id, traceback.format_exc(), root_path)
        error = f"{type(exc).__name__}: {exc}".strip().splitlines()[0][:300]
        final = {**writer.latest(), "state": "failed", "error": error}
    else:
        final = {"state": "done", "stage": None, "message": "finished"}
    runs.append_log(
        run_id, f"{final['state']} after {time.perf_counter() - started:.1f} s", root_path
    )
    runs.update_status(run_id, root=root_path, **final)
    return str(final["state"])
