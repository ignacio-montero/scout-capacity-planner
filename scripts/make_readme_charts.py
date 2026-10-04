"""README charts and key numbers from the published sweep summaries (M6).

    uv run python scripts/make_readme_charts.py            # or: make charts

Reads ``data/published/<sweep>_summary.parquet`` (headline, growth,
forecast_error / forecast-error, baseline, cadence, late_penalty), writes PNGs (1200x675,
scale 2, white background) and ``key_numbers.md`` into ``docs/img/``, and
prints the key numbers. A chart whose input is missing is skipped with a
message. Static export needs a Chrome for kaleido: ``BROWSER_PATH`` if set,
else a Chrome for Testing in the puppeteer cache, else a standard Chrome
install. Nothing is downloaded.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from scout_planner import readme_charts as rc
from scout_planner.config import Params, load_default_params
from scout_planner.results import load_published

REPO = Path(__file__).resolve().parents[1]
PUBLISHED = REPO / "data" / "published"
OUT = REPO / "docs" / "img"
WIDTH, HEIGHT, SCALE = 1200, 675, 2

BROWSER_CANDIDATES = (
    "~/.cache/puppeteer/chrome/*/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/"
    "Google Chrome for Testing",
    "~/.cache/puppeteer/chrome/*/chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/"
    "Google Chrome for Testing",
    "~/.cache/puppeteer/chrome/*/chrome-linux64/chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


def find_browser(env: dict[str, str] | None = None) -> Path | None:
    """``BROWSER_PATH`` if it points at a file, else the newest known install, else None."""
    env = os.environ if env is None else env
    explicit = env.get("BROWSER_PATH")
    if explicit:
        return Path(explicit) if Path(explicit).is_file() else None
    for pattern in BROWSER_CANDIDATES:
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if matches:
            return Path(matches[-1])
    return None


def published_path(name: str, published: Path) -> Path | None:
    """``<name>_summary.parquet``; the sweep name may be slugified (``forecast-error``)."""
    for candidate in (name, name.replace("_", "-")):
        path = published / f"{candidate}_summary.parquet"
        if path.is_file():
            return path
    return None


def load_frames(published: Path, defaults: Params) -> dict[str, pd.DataFrame | None]:
    frames: dict[str, pd.DataFrame | None] = {}
    for name in rc.SWEEPS:
        path = published_path(name, published)
        if path is None:
            frames[name] = None
            continue
        loaded = load_published(path)
        if loaded.value is None:
            print(f"warning: {loaded.problem}", file=sys.stderr)
            frames[name] = None
            continue
        frames[name] = rc.prepare(loaded.value, defaults)
    return frames


Builder = Callable[[dict[str, pd.DataFrame | None], Params], go.Figure]


def chart_specs() -> list[tuple[str, str, Builder]]:
    """``(file name, required sweep, builder)`` for every README chart."""

    def target(df: pd.DataFrame, defaults: Params) -> float:
        return rc.target_of(df, defaults)

    return [
        ("headline.png", "headline",
         lambda f, d: rc.fig_headline(f["headline"], target(f["headline"], d))),
        ("mix.png", "headline", lambda f, d: rc.fig_mix(f["headline"], target(f["headline"], d))),
        ("policies.png", "headline",
         lambda f, d: rc.fig_policies(f["headline"], target(f["headline"], d), d)),
        ("growth.png", "growth", lambda f, d: rc.fig_growth(f["growth"], target(f["growth"], d))),
        ("forecast_error.png", "forecast_error",
         lambda f, d: rc.fig_forecast_error(f["forecast_error"], target(f["forecast_error"], d))),
        ("baseline.png", "baseline",
         lambda f, d: rc.fig_baseline(f["baseline"], target(f["baseline"], d))),
        ("cadence.png", "cadence",
         lambda f, d: rc.fig_cadence(f["cadence"], target(f["cadence"], d))),
        ("policy_gain.png", "headline",
         lambda f, d: rc.fig_policy_gain(f["headline"], target(f["headline"], d))),
        ("paired.png", "headline",
         lambda f, d: rc.fig_paired(f["headline"], target(f["headline"], d))),
        ("late_penalty.png", "late_penalty",
         lambda f, d: rc.fig_late_penalty(f["late_penalty"], target(f["late_penalty"], d))),
    ]  # fmt: skip


def build_figures(
    frames: dict[str, pd.DataFrame | None], defaults: Params
) -> tuple[dict[str, go.Figure], list[str]]:
    """Figures for every chart whose input exists, plus the skip messages."""
    figures, skipped = {}, []
    for file_name, sweep, builder in chart_specs():
        if frames.get(sweep) is None:
            skipped.append(f"skip {file_name}: no published '{sweep}' summary")
            continue
        figures[file_name] = builder(frames, defaults)
    return figures, skipped


def title_text(fig: go.Figure) -> str:
    """The figure's action title as plain text (subtitle and line breaks removed)."""
    text = str(fig.layout.title.text or "")
    return text.split("<br><span")[0].replace("<br>", " ")


def takeaways_markdown(figures: dict[str, go.Figure]) -> str:
    if not figures:
        return ""
    lines = ["", "## Chart takeaways (the computed titles)", ""]
    lines += [f"- `{name}`: {title_text(fig)}" for name, fig in figures.items()]
    return "\n".join(lines) + "\n"


def export(figures: dict[str, go.Figure], out: Path, browser: Path) -> list[Path]:
    os.environ["BROWSER_PATH"] = str(browser)  # kaleido >= 1 drives this Chrome
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for file_name, fig in figures.items():
        path = out / file_name
        fig.write_image(path, width=WIDTH, height=HEIGHT, scale=SCALE)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--published", type=Path, default=PUBLISHED)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--no-images", action="store_true", help="key numbers only")
    args = parser.parse_args(argv)

    defaults = load_default_params()
    frames = load_frames(args.published, defaults)
    if all(df is None for df in frames.values()):
        print(f"no published summaries in {args.published}; run the sweeps first", file=sys.stderr)
        return 2
    figures, skipped = build_figures(frames, defaults)
    markdown = rc.key_numbers_markdown(frames, defaults) + takeaways_markdown(figures)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "key_numbers.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {args.out / 'key_numbers.md'}")
    for message in skipped:
        print(message)
    if args.no_images:
        return 0
    browser = find_browser()
    if browser is None:
        print(
            "error: no Chrome found for static export. Set BROWSER_PATH to a Chrome or "
            "Chromium executable (nothing is downloaded automatically).",
            file=sys.stderr,
        )
        return 3
    for path in export(figures, args.out, browser):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
