"""Command-line front door: ``python -m scout_planner.cli <stage> [options]``.

The *imperative shell* around the pure stages: it parses arguments, loads the
parameter file, calls the stage, writes files and prints a short summary.
No modelling logic lives here. Stage commands: ``data`` [1], ``forecast`` [2],
``plan`` [3]; ``forecast`` and ``plan`` read the run folder that ``data``
wrote (its ``params.yaml`` and ``raw/``). Later milestones add ``run`` and
``sweep`` in the same shape.

Exit codes: 0 success, 2 invalid parameters, missing inputs or usage
(argparse's convention).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from scout_planner import forecast, generate, plan
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
        f"realised month 0 {generate.realised_start_load(world, params):.3f}"
    )
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
