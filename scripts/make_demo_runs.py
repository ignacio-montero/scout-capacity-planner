"""Populate a throwaway runs folder with demo runs, to try the app before M4 exists.

    uv run python scripts/make_demo_runs.py                 # data/runs_demo + data/published_demo
    SCOUT_RUNS_ROOT=data/runs_demo SCOUT_PUBLISHED_DIR=data/published_demo \\
        uv run streamlit run app/main.py

Never writes into the real ``data/runs`` or ``data/published`` unless you
pass ``--allow-real``. Every demo run name starts with "Demo:". The outputs
come from a toy model in ``tests/ui_fixtures.py`` (not the real simulation).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

import ui_fixtures  # noqa: E402

REAL = {(REPO / "data" / "runs").resolve(), (REPO / "data" / "published").resolve()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO / "data" / "runs_demo")
    parser.add_argument("--published", type=Path, default=REPO / "data" / "published_demo")
    parser.add_argument("--clean", action="store_true", help="empty both folders first")
    parser.add_argument("--broken", action="store_true", help="add a folder with a corrupt status")
    parser.add_argument("--allow-real", action="store_true", help="permit the real data folders")
    args = parser.parse_args(argv)

    for folder in (args.root, args.published):
        if folder.resolve() in REAL and not args.allow_real:
            print(f"refusing to write demo data into {folder} (use --allow-real)", file=sys.stderr)
            return 2
    if args.clean:
        for folder in (args.root, args.published):
            shutil.rmtree(folder, ignore_errors=True)
    ids = ui_fixtures.write_demo_runs(args.root, include_broken=args.broken)
    path = ui_fixtures.write_demo_published(args.published)
    print(f"wrote {len(ids)} demo runs to {args.root}")
    print(f"wrote published demo sweep to {path}")
    print(
        f"try it: SCOUT_RUNS_ROOT={args.root} SCOUT_PUBLISHED_DIR={args.published} "
        "uv run streamlit run app/main.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
