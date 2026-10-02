"""Command-line front door: ``python -m scout_planner.cli <stage> [options]``.

The *imperative shell* around the pure stages: it parses arguments, loads the
parameter file, calls the stage, writes files and prints a short summary.
No modelling logic lives here. Later milestones add ``forecast``, ``plan``,
``run`` and ``sweep`` subcommands in the same shape.

Exit codes: 0 success, 2 invalid parameters or usage (argparse's convention).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from scout_planner import generate
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
