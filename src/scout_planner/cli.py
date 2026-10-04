"""Command-line front door: ``python -m scout_planner.cli <stage> [options]``.

The *imperative shell* around the pure stages: it parses arguments, loads the
parameter file, calls the stage, writes files and prints a short summary.
No modelling logic lives here. Stage commands: ``data`` [1], ``forecast`` [2],
``plan`` [3], ``simulate`` [5]; each later stage reads the run folder the
earlier ones wrote (its ``params.yaml``, ``raw/`` and plan files). ``run``
creates a run folder in the run store and executes it synchronously;
``sweep`` expands a sweep file into runs (executed here, or only queued for
the background worker with ``--queue``).

Exit codes: 0 success, 1 a run failed, 2 invalid parameters, missing inputs or
usage (argparse's convention), 3 a background worker holds the runs folder.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from scout_planner import forecast, generate, pipeline, plan, runs, sweep, worker
from scout_planner.config import DEFAULT_CONFIG_PATH, Params, dump_params, load_params

DEFAULT_OUT = Path("data/runs/dev")


def _load(path: Path) -> Params | None:
    """Load parameters, printing a readable error instead of a traceback."""
    try:
        return load_params(path)
    except FileNotFoundError:
        print(f"error: parameter file not found: {path}", file=sys.stderr)
    except (ValidationError, ValueError) as exc:
        print(f"error: invalid parameters in {path}:\n{exc}", file=sys.stderr)
    return None


def cmd_data(args: argparse.Namespace) -> int:
    """[1] generate the synthetic world into ``<out>/raw/``."""
    params = _load(args.params)
    if params is None:
        return 2
    started = time.perf_counter()
    world = generate.generate_world(params)
    paths = generate.write_world(world, args.out)
    dump_params(params, Path(args.out) / "params.yaml")
    elapsed = time.perf_counter() - started

    req = world.requests
    print(f"wrote {Path(args.out) / 'raw'} in {elapsed:.2f} s")
    for name, df in world.tables().items():
        print(f"  {paths[name].name:<30} {len(df):>7,} rows")
    print(
        f"  history requests {int((req['period'] == 'history').sum()):,}, "
        f"future requests {int((req['period'] == 'future').sum()):,}"
    )
    print(
        f"base volume (deseasonalised, month 0): {generate.base_monthly_volume(params):.1f}/month"
    )
    print(
        f"start load: target {params.demand.start_load:.2f}, "
        f"realised month 0 {generate.realised_start_load(world, params):.3f} (deseasonalised)"
    )
    print(
        f"January load incl. seasonality: expected {generate.month0_peak_load(params):.3f}, "
        f"realised {generate.realised_start_load(world, params, deseasonalised=False):.3f}"
    )
    urgent = req["urgent"].mean() if len(req) else 0.0
    print(f"urgent requests: {urgent:.1%} (due in {params.demand.urgent_turnaround_days} days)")
    print(f"at risk from day one (future live views): {generate.at_risk_share(world):.1%}")
    return 0


def _load_run(run: Path, *needs: str) -> Params | None:
    """Parameters of an existing run folder, after checking its input files exist."""
    for rel in ("params.yaml", *needs):
        if not (run / rel).exists():
            hint = "run `data` first" if rel in ("params.yaml", "raw") else "run `forecast` first"
            print(f"error: {run / rel} not found ({hint})", file=sys.stderr)
            return None
    return _load(run / "params.yaml")


def _pct(x: float) -> str:
    return f"{x:6.1%}"


def cmd_forecast(args: argparse.Namespace) -> int:
    """[2] forecast + backtest from ``<run>/raw/`` into ``<run>/*.parquet``."""
    run = Path(args.run)
    params = _load_run(run, "raw")
    if params is None:
        return 2
    started = time.perf_counter()
    world = generate.read_world(run)
    fc, bt = forecast.build_forecast(world, params)
    paths = forecast.write_forecast(fc, bt, run)
    elapsed = time.perf_counter() - started
    print(f"wrote {paths['forecast']} and {paths['backtest'].name} in {elapsed:.2f} s")
    if (fc["method"] != forecast.METHOD_ETS).any():
        print("warning: ETS fit failed for the plan year; seasonal naive fallback used")

    summary = forecast.backtest_summary(bt)
    origins = sorted(bt["origin"].unique())
    print(
        f"\nbacktest: rolling origin, {len(origins)} origins "
        f"({origins[0]:%Y-%m}..{origins[-1]:%Y-%m}) x {forecast.BACKTEST_HORIZON} months; "
        "WAPE, lower is better"
    )
    print(f"  {'skill':<24} {'model':>7} {'naive':>7}  better")
    for row in summary.itertuples(index=False):
        better = "model" if row.model_beats_naive else "naive"
        print(f"  {row.skill_type:<24} {_pct(row.wape_model)} {_pct(row.wape_naive)}  {better}")
    agg = summary.iloc[0]
    skills = summary.iloc[1:]
    verdict = "beats" if agg["model_beats_naive"] else "does NOT beat"
    print(
        f"  -> ETS {verdict} seasonal naive on aggregate "
        f"({agg['wape_model']:.1%} vs {agg['wape_naive']:.1%}); "
        f"better on {int(skills['model_beats_naive'].sum())}/{len(skills)} skills"
    )

    cp, dm = params.capacity_plan, params.demand
    print(
        f"\nplan-year forecast (assumed growth {cp.assumed_growth:g}x, P{cp.quantile * 100:g}) "
        f"vs realised (actual growth {dm.actual_growth:g}x; realised is never read by the forecast)"
    )
    totals = fc.groupby("month", sort=True)[["requests_p50", "requests_pq"]].sum()
    totals["realised"] = forecast.realised_monthly_totals(world.requests, params).to_numpy()
    quarter = [f"{m.year}-Q{(m.month - 1) // 3 + 1}" for m in totals.index]
    by_q = totals.groupby(quarter, sort=True).sum()
    by_q.loc["year"] = totals.sum()
    print(f"  {'period':<8} {'P50':>8} {'Pq':>8} {'realised':>9} {'P50 err':>8}")
    for label, r in by_q.iterrows():
        err = (r["requests_p50"] - r["realised"]) / r["realised"] if r["realised"] else float("nan")
        print(
            f"  {label:<8} {r['requests_p50']:8.0f} {r['requests_pq']:8.0f} "
            f"{r['realised']:9.0f} {err:+8.1%}"
        )
    print(f"hours per request: {forecast.hours_per_request(params):.2f}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """[3] capacity plan + hiring table from ``raw/`` and ``forecast.parquet``."""
    run = Path(args.run)
    params = _load_run(run, "raw", "forecast.parquet")
    if params is None:
        return 2
    started = time.perf_counter()
    world = generate.read_world(run)
    fc, _ = forecast.read_forecast(run)
    capacity, hiring = plan.build_capacity_plan(
        world.scouts, world.scout_unavailability, fc, params
    )
    paths = plan.write_plan(capacity, hiring, run)
    elapsed = time.perf_counter() - started
    print(f"wrote {paths['capacity_plan']} and {paths['hiring_plan'].name} in {elapsed:.2f} s")

    cols = ["required_hours", "available_hours", "gap_hours", "hired_hours"]
    monthly = capacity.groupby("month", sort=True)[cols].sum()
    short = capacity.assign(short=capacity["gap_hours"] > 0).groupby("month", sort=True)["short"]
    after = capacity.assign(short=capacity["gap_after_hires_hours"] > 0)
    still_short = after.groupby("month", sort=True)["short"].sum()
    print(
        f"\ncapacity plan, all skills (hours; required = P{params.capacity_plan.quantile * 100:g}, "
        f"available at {params.capacity_plan.target_utilisation:.0%} utilisation)"
    )
    print(
        f"  {'month':<8} {'required':>9} {'available':>9} {'gap':>8} {'hired':>8} "
        f"{'short skills':>13}"
    )
    for month, r in monthly.iterrows():
        print(
            f"  {month:%Y-%m}  {r['required_hours']:9.0f} {r['available_hours']:9.0f} "
            f"{r['gap_hours']:8.0f} {r['hired_hours']:8.0f} "
            f"{int(short.sum()[month]):>5} -> {int(still_short[month]):<2}"
        )

    print(f"\nhiring plan: {len(hiring)} rows (full table in {paths['hiring_plan'].name})")
    if hiring.empty:
        print("  no hires needed")
        return 0
    late = hiring["reason"].str.contains("late")
    grouped = (
        hiring.assign(late=late)
        .groupby(["month_to_act", "joins_month", "hire_type"], sort=True)
        .agg(count=("count", "sum"), skills=("skill_type", "nunique"), late=("late", "any"))
        .reset_index()
    )
    print(f"  {'act':>3} {'joins':>5}  {'type':<10} {'hires':>5} {'skills':>6}")
    for r in grouped.itertuples(index=False):
        flag = "  late" if r.late else ""
        print(
            f"  {r.month_to_act:>3} {r.joins_month:>5}  {r.hire_type:<10} {r.count:>5} "
            f"{r.skills:>6}{flag}"
        )
    totals = hiring.groupby("hire_type")["count"].sum()
    print(
        "  total: "
        + ", ".join(f"{int(totals.get(t, 0))} {t}" for t in ("full_time", "freelance"))
        + f" (initial team {len(world.scouts)})"
    )
    return 0


# --- [5] simulate, run, sweep ------------------------------------------------------


class _Echo:
    """Prints pipeline progress to the terminal: every stage change, then every ~10%."""

    def __init__(self, step: float = 0.1) -> None:
        self.step = step
        self.next = 0.0
        self.stage: str | None = None

    def __call__(self, fraction: float, stage: str, message: str) -> None:
        if stage != self.stage or fraction >= self.next or fraction >= 1.0:
            print(f"  [{fraction:4.0%}] {stage}: {message}", flush=True)
            self.stage = stage
            self.next = (int(fraction / self.step) + 1) * self.step


def _fmt_range(stat: dict, fmt: str) -> str:
    if stat["mean"] is None:
        return "n/a"
    text = format(stat["mean"], fmt)
    if stat["min"] != stat["max"]:
        text += f" [{format(stat['min'], fmt)} .. {format(stat['max'], fmt)}]"
    return text


def print_summary(summary: dict, params: Params) -> None:
    """Headline metrics of a run (mean [min .. max] over seeds)."""
    target = params.sim.target_on_time
    verdict = "meets" if summary["meets_target"] else "MISSES"
    print(
        f"on time        {_fmt_range(summary['on_time_rate'], '.1%')} "
        f"-> {verdict} the {target:.0%} target"
    )
    print(
        f"requests       {_fmt_range(summary['n_requests'], ',.0f')} scored, "
        f"{_fmt_range(summary['n_late'], ',.0f')} late, "
        f"{_fmt_range(summary['n_at_risk_day_one'], ',.0f')} at risk from day one, "
        f"{_fmt_range(summary['n_censored'], ',.0f')} censored (due after the horizon)"
    )
    print(
        f"turnaround     mean {_fmt_range(summary['mean_turnaround_days'], '.1f')} d, "
        f"P90 {_fmt_range(summary['p90_turnaround_days'], '.1f')} d"
    )
    print(
        f"utilisation    full-time {_fmt_range(summary['util_full_time'], '.0%')}, "
        f"freelance {_fmt_range(summary['util_freelance'], '.0%')}"
    )
    print(
        "cost           total "
        + _fmt_range(summary["cost_total"], ",.0f")
        + " = salaried "
        + _fmt_range(summary["cost_salaried"], ",.0f")
        + " + freelance "
        + _fmt_range(summary["cost_freelance"], ",.0f")
        + " + automation "
        + _fmt_range(summary["cost_automation"], ",.0f")
        + " + late penalty "
        + _fmt_range(summary["cost_late_penalty"], ",.0f")
    )
    print(
        f"hires          {_fmt_range(summary['n_hires_full_time'], '.0f')} full-time, "
        f"{_fmt_range(summary['n_hires_freelance'], '.0f')} freelance joined"
    )
    d = summary["diagnostics"]
    print(
        f"assignment     {d['assignment_runs']} runs, {d['optimiser_solves']} optimiser solves "
        f"(mean {d['mean_solve_seconds']:.2f} s, max {d['max_solve_seconds']:.2f} s), "
        f"{d['wall_clock_hits']} wall-clock stops, {d['fallbacks']} EDF fallbacks, "
        f"{d['live_view_retargets']} live views re-targeted"
    )


def cmd_simulate(args: argparse.Namespace) -> int:
    """[5] simulate on an existing run folder (raw/ + plans) -> results/ + summary.json."""
    run = Path(args.run)
    params = _load_run(run, "raw", "forecast.parquet", "hiring_plan.parquet")
    if params is None:
        return 2
    started = time.perf_counter()
    print(f"simulating {params.sim.seeds} seed(s), policy {params.assignment.policy}")
    summary = pipeline.simulate_stage(params, run, _Echo(), lambda: False, max_workers=args.workers)
    print(
        f"wrote {run / pipeline.RESULTS_DIR} and {pipeline.SUMMARY_FILE} "
        f"in {time.perf_counter() - started:.1f} s\n"
    )
    print_summary(summary, params)
    return 0


def _report(run_id: str, root: Path, state: str, elapsed: float) -> None:
    print(f"run {run_id}: {state} in {elapsed:.1f} s")
    if state == "done":
        print_summary(
            pipeline.read_summary(runs.run_dir(run_id, root)), runs.read_params(run_id, root)
        )
    elif state == "failed":
        print(f"  error: {runs.read_status(run_id, root).error} (see log.txt)", file=sys.stderr)


def _execute(lease: worker.Lease, run_id: str, root: Path, workers: int | None) -> str:
    """Execute a queued run here, under the runs lock held by ``lease``."""
    started = time.perf_counter()
    lease.current_run = run_id  # heartbeat names the run before it is marked running
    try:
        state = pipeline.execute_run_inline(run_id, root, echo=_Echo(), max_workers=workers)
    finally:
        lease.current_run = None
    _report(run_id, root, state, time.perf_counter() - started)
    return state


def _create_and_execute(
    items: list[tuple[Params, str]], root: Path, sweep_id: str | None, workers: int | None
) -> tuple[list[str], int] | None:
    """Create runs and execute them in this process; ``(run_ids, failures)``, or None.

    The CLI becomes the runs folder's executor (``worker.executor_lease``:
    the single-executor lock, crash recovery, a heartbeat), so a background
    worker can neither pick these runs up halfway nor mistake them for crashed
    ones. The runs are created inside the lease for the same reason. If the
    app's worker already holds the lock, nothing is created and None is returned.
    """
    try:
        with worker.executor_lease(root) as lease:
            run_ids = [runs.create_run(p, name, sweep_id, root=root) for p, name in items]
            failures = 0
            for i, run_id in enumerate(run_ids, 1):
                if len(run_ids) > 1:
                    print(f"\n[{i}/{len(run_ids)}] {run_id}")
                failures += _execute(lease, run_id, root, workers) not in ("done", "skipped")
            return run_ids, failures
    except worker.WorkerAlreadyRunning:
        print(
            "error: a background worker is running on this runs folder; "
            "use --queue (sweep) or stop `make app` first",
            file=sys.stderr,
        )
        return None


def cmd_run(args: argparse.Namespace) -> int:
    """Create a run from a parameter file and execute it now."""
    params = _load(args.params)
    if params is None:
        return 2
    name = args.name or Path(args.params).stem
    print(f"run {name!r}: executing in {args.root}")
    outcome = _create_and_execute([(params, name)], args.root, None, args.workers)
    if outcome is None:
        return 3
    return 0 if outcome[1] == 0 else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    """Expand a sweep file into runs; execute them here (default) or only queue them."""
    try:
        spec = sweep.load_sweep(args.sweep)
        expanded = sweep.expand(spec)
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"error: invalid sweep {args.sweep}:\n{exc}", file=sys.stderr)
        return 2

    if args.publish_only:
        sweep_id = args.sweep_id or sweep.latest_sweep_id(spec.name, args.root)
        if sweep_id is None:
            print(f"error: no runs of sweep {spec.name!r} in {args.root}", file=sys.stderr)
            return 2
        return _publish(spec, sweep_id, args)

    sweep_id = sweep.new_sweep_id(spec.name)
    print(f"sweep {spec.name}: {len(expanded)} runs, sweep_id {sweep_id}")
    for r in expanded:
        print(f"  {r.name}")
    if args.queue:
        sweep.queue_sweep(spec, args.root, sweep_id=sweep_id)
        print("queued for the background worker (make app, or python -m scout_planner.worker)")
        return 0
    started = time.perf_counter()
    items = [(r.params, r.name) for r in expanded]
    outcome = _create_and_execute(items, args.root, sweep_id, args.workers)
    if outcome is None:
        return 3
    run_ids, failures = outcome
    print(
        f"\nsweep {spec.name}: {len(run_ids) - failures}/{len(run_ids)} runs done "
        f"in {time.perf_counter() - started:.0f} s"
    )
    if args.publish and _publish(spec, sweep_id, args) != 0:
        return 1
    return 0 if failures == 0 else 1


def _publish(spec: sweep.SweepSpec, sweep_id: str, args: argparse.Namespace) -> int:
    try:
        path = sweep.publish(
            spec, sweep_id, args.root, args.published_dir, allow_partial=args.allow_partial
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    info = sweep.read_publish_info(path) or {}
    partial = " (PARTIAL)" if info.get("partial") else ""
    print(f"published {path}: {info.get('done_runs')}/{info.get('expected_runs')} runs{partial}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, separate from :func:`main` so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog="python -m scout_planner.cli",
        description="Scout capacity planner: pipeline stages from the command line.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    data = sub.add_parser("data", help="[1] generate the synthetic world into OUT/raw/")
    data.add_argument("--params", type=Path, default=DEFAULT_CONFIG_PATH, help="parameter YAML")
    data.add_argument("--out", type=Path, default=DEFAULT_OUT, help="run folder to write into")
    data.set_defaults(func=cmd_data)

    fc = sub.add_parser("forecast", help="[2] forecast + backtest from RUN/raw/ into RUN/")
    fc.add_argument("--run", type=Path, default=DEFAULT_OUT, help="run folder (written by data)")
    fc.set_defaults(func=cmd_forecast)

    pl = sub.add_parser("plan", help="[3] capacity plan + hiring table into RUN/")
    pl.add_argument("--run", type=Path, default=DEFAULT_OUT, help="run folder (after forecast)")
    pl.set_defaults(func=cmd_plan)

    sim = sub.add_parser("simulate", help="[5] simulate RUN (after plan) into RUN/results/")
    sim.add_argument("--run", type=Path, default=DEFAULT_OUT, help="run folder (after plan)")
    sim.add_argument("--workers", type=int, default=None, help="max parallel seed processes")
    sim.set_defaults(func=cmd_simulate)

    run = sub.add_parser("run", help="create a run from PARAMS and execute it now")
    run.add_argument("--params", type=Path, default=DEFAULT_CONFIG_PATH, help="parameter YAML")
    run.add_argument("--name", default=None, help="run name (default: the file name)")
    run.add_argument("--root", type=Path, default=runs.DEFAULT_ROOT, help="runs folder")
    run.add_argument("--workers", type=int, default=None, help="max parallel seed processes")
    run.set_defaults(func=cmd_run)

    sw = sub.add_parser("sweep", help="expand a sweep file into runs and execute them")
    sw.add_argument("--sweep", type=Path, required=True, help="config/sweeps/<name>.yaml")
    sw.add_argument("--root", type=Path, default=runs.DEFAULT_ROOT, help="runs folder")
    sw.add_argument("--workers", type=int, default=None, help="max parallel seed processes")
    mode = sw.add_mutually_exclusive_group()
    mode.add_argument("--queue", action="store_true", help="only queue the runs for the worker")
    mode.add_argument(
        "--publish-only",
        action="store_true",
        help="publish an earlier execution of this sweep (latest, or --sweep-id)",
    )
    sw.add_argument("--publish", action="store_true", help="write the published summary")
    sw.add_argument("--sweep-id", default=None, help="with --publish-only: which execution")
    sw.add_argument(
        "--allow-partial",
        action="store_true",
        help="publish even if some runs of the sweep are not done (recorded in the file)",
    )
    sw.add_argument(
        "--published-dir", type=Path, default=sweep.PUBLISHED_DIR, help="where --publish writes"
    )
    sw.set_defaults(func=cmd_sweep)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
