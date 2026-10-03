"""M0 acceptance: `make setup && make test && make lint` on a fresh clone; Makefile <-> CLI.

`/usr/bin/make` is unusable on this machine (Xcode licence prompt), so the
Makefile is tested as *data*: every recipe that calls the CLI is parsed and its
arguments are fed to the real argument parser. That is a *contract test*
between two files that otherwise only meet when somebody types `make`.

The ``slow`` fresh-clone test is the real thing: `git clone` the committed
HEAD into a temp folder (no `.venv`, no `.private/`, no untracked files), run
the setup recipe, lint, the guardrail tests (which must *skip*, not fail,
without the private term list) and the stage pipeline on a tiny parameter file.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from scout_planner import cli

REPO = Path(__file__).resolve().parents[1]
MAKEFILE = REPO / "Makefile"
VARS = {
    "DEV_RUN": "data/runs/dev",
    "PARAMS": "config/default.yaml",
    "SWEEP": "quick",
    "SWEEP_ARGS": "",
}


def _recipes() -> dict[str, list[str]]:
    """``target -> recipe lines`` from the Makefile (tab-indented lines under ``target:``)."""
    out: dict[str, list[str]] = {}
    target = None
    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"^([a-z][a-z-]*):", line):
            target = m.group(1)
            out[target] = []
        elif line.startswith("\t") and target:
            out[target].append(line.strip())
    return out


def _expand(line: str) -> str:
    while (start := line.find("$(if ")) >= 0:  # drop optional flags such as NAME=...
        depth, end = 0, start
        for end in range(start, len(line)):
            depth += {"(": 1, ")": -1}.get(line[end], 0)
            if depth == 0 and line[end] == ")":
                break
        line = line[:start] + line[end + 1 :]
    for key, value in VARS.items():
        line = line.replace(f"$({key})", value)
    return line


def test_every_m0_target_exists() -> None:
    recipes = _recipes()
    for target in ("setup", "test", "lint", "data", "forecast", "plan", "simulate", "run"):
        assert target in recipes, target
    for target in ("sweep", "app", "all"):
        assert target in recipes, target
    assert "uv sync --locked" in recipes["setup"]
    assert any(r.startswith("uv run pytest") for r in recipes["test"])
    assert "uv run pytest" in recipes["test-all"]
    assert recipes["lint"] == ["uv run ruff check .", "uv run ruff format --check ."]


def test_every_cli_recipe_parses_with_the_real_parser() -> None:
    """A renamed flag or subcommand in cli.py would break `make` silently; not any more."""
    parser = cli.build_parser()
    seen = set()
    for target, lines in _recipes().items():
        for line in lines:
            if "scout_planner.cli" not in line:
                continue
            argv = shlex.split(_expand(line).split("scout_planner.cli", 1)[1])
            args = parser.parse_args(argv)  # SystemExit (= test error) if invalid
            seen.add(args.command)
            for attr in ("params", "sweep"):
                path = getattr(args, attr, None)
                if path is not None:
                    assert (REPO / path).is_file(), f"make {target}: {path} does not exist"
    assert seen == {"data", "forecast", "plan", "simulate", "run", "sweep"}


def test_stage_targets_share_the_dev_run_folder() -> None:
    """PRD M1: `make data` writes to data/runs/dev/raw; later stages read the same folder."""
    parser = cli.build_parser()
    assert parser.parse_args(["data"]).out == Path("data/runs/dev")
    for cmd in ("forecast", "plan", "simulate"):
        assert parser.parse_args([cmd]).run == Path("data/runs/dev")


def test_make_all_runs_tests_and_lint_first_then_the_pipeline_then_the_quick_sweep() -> None:
    recipes = _recipes()
    head = MAKEFILE.read_text(encoding="utf-8").split("\nall:", 1)[1].splitlines()[0]
    assert head.split("##")[0].split() == ["test", "lint"]
    joined = " ".join(recipes["all"])
    assert joined.index("data forecast plan simulate") < joined.index("sweep SWEEP=quick")


def test_lockfile_is_in_sync_with_pyproject() -> None:
    """`uv sync --locked` (make setup) fails if uv.lock is stale; check that cheaply offline."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    out = subprocess.run(
        [uv, "lock", "--check", "--offline"], cwd=REPO, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr


def test_lint_passes_on_the_working_tree() -> None:
    """PRD M0: `make lint` passes (both of its recipe lines)."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    for cmd in (["ruff", "check", "."], ["ruff", "format", "--check", "."]):
        out = subprocess.run([uv, "run", *cmd], cwd=REPO, capture_output=True, text=True)
        assert out.returncode == 0, f"{' '.join(cmd)}:\n{out.stdout}{out.stderr}"


# --- fresh clone ----------------------------------------------------------------------

TINY_YAML = """\
schema_version: 1
team: {full_time_count: 8, freelance_count: 4, follow_hiring_plan: false}
demand: {start_load: 0.3, actual_growth: 2.0}
assignment: {policy: edf}
sim: {months: 1, seeds: 1}
"""


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    return env


def _sh(cmd: list[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=_env(), capture_output=True, text=True, timeout=timeout)


@pytest.mark.slow
def test_fresh_clone_sets_up_lints_skips_the_guardrail_and_runs_the_pipeline(
    tmp_path: Path,
) -> None:
    uv, git = shutil.which("uv"), shutil.which("git")
    if uv is None or git is None:
        pytest.skip("uv and git are needed for the fresh-clone check")
    clone = tmp_path / "clone"
    out = _sh([git, "clone", "--quiet", "--no-hardlinks", str(REPO), str(clone)], tmp_path)
    assert out.returncode == 0, out.stderr
    assert not (clone / ".private").exists() and not (clone / ".venv").exists()

    # make setup (offline: the lock must resolve from the local uv cache)
    out = _sh([uv, "sync", "--locked", "--offline", "--quiet"], clone)
    if out.returncode != 0 and "offline" in out.stderr.lower():
        pytest.skip(f"uv cache cannot satisfy an offline sync here: {out.stderr[:200]}")
    assert out.returncode == 0, out.stderr
    assert _sh([git, "config", "core.hooksPath", ".githooks"], clone).returncode == 0

    # make lint (on the committed code)
    for cmd in (["ruff", "check", "."], ["ruff", "format", "--check", "."]):
        out = _sh([uv, "run", "--offline", *cmd], clone)
        assert out.returncode == 0, f"{cmd}: {out.stdout}{out.stderr}"

    # make test, guardrail part: must skip cleanly without .private/banned-terms.txt
    out = _sh([uv, "run", "--offline", "pytest", "-q", "tests/test_guardrails.py"], clone)
    assert out.returncode == 0, out.stdout[-2000:]
    assert "skipped" in out.stdout and "banned-terms.txt not present" in out.stdout

    # make data forecast plan simulate, on a tiny parameter file
    # Hand-written, minimal: the clone is the *committed* code, so a file dumped by the
    # working tree's config (which may already have new keys) could be refused there.
    params_file = tmp_path / "tiny.yaml"
    params_file.write_text(TINY_YAML, encoding="utf-8")
    run = clone / "data/runs/dev"
    stages = [
        ["data", "--params", str(params_file), "--out", str(run)],
        ["forecast", "--run", str(run)],
        ["plan", "--run", str(run)],
        ["simulate", "--run", str(run)],
    ]
    for stage in stages:
        out = _sh([uv, "run", "--offline", "python", "-m", "scout_planner.cli", *stage], clone)
        assert out.returncode == 0, f"{stage[0]}: {out.stderr[-2000:]}"
    assert (run / "summary.json").is_file()
