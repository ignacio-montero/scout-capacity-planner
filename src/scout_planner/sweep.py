"""Sweeps: one YAML file -> many ordinary runs sharing a ``sweep_id`` (DATA_CONTRACTS §6).

A sweep file::

    name: quick
    base: config/default.yaml        # starting parameters (relative to the repo root)
    vary:                            # full cross-product, in file order
      assignment.policy: [edf, optimiser]
      demand.actual_growth: [1.0, 4.0]
    fixed:                           # overrides applied to every run
      sim.seeds: 2

expands into ``2 x 2 = 4`` parameter sets: ``base`` + ``fixed`` + one
combination of ``vary``, each validated by ``config.apply_overrides``. The
first ``vary`` key changes slowest (like nested loops), so the run list reads
in a predictable order. This is a *full factorial design*: every combination,
so each parameter's effect can be read at every level of the others.

Expansion (:func:`load_sweep`, :func:`expand`) is pure. Queueing goes
through the run store; :func:`publish` flattens the finished runs into
``data/published/<sweep>_summary.parquet`` (G5).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scout_planner import metrics, runs
from scout_planner.config import Params, apply_overrides, load_params

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEPS_DIR = REPO_ROOT / "config" / "sweeps"
PUBLISHED_DIR = REPO_ROOT / "data" / "published"
PUBLISHED_SUFFIX = "_summary.parquet"
# Always present as columns in a published summary, varied or not (the app
# encodes them in the sweep chart).
ALWAYS_PUBLISHED: tuple[str, ...] = (
    "assignment.policy",
    "demand.actual_growth",
    "automation.enabled",
    "sim.seeds",
    "sim.target_on_time",
)


@dataclass(frozen=True)
class SweepSpec:
    """A parsed, validated sweep file."""

    name: str
    base: Path
    vary: dict[str, list[Any]]
    fixed: dict[str, Any]


@dataclass(frozen=True)
class SweepRun:
    """One expanded combination: its run name, the varied values, the full parameters."""

    name: str
    values: dict[str, Any]
    params: Params


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_sweep(data: Any) -> SweepSpec:
    """Validate the structure of a decoded sweep file (values are checked by :func:`expand`)."""
    if not isinstance(data, Mapping):
        raise ValueError("a sweep file must be a mapping")
    unknown = set(data) - {"name", "base", "vary", "fixed"}
    if unknown:
        raise ValueError(f"unknown sweep keys: {sorted(unknown)}")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("a sweep needs a non-empty 'name'")
    vary = data.get("vary") or {}
    fixed = data.get("fixed") or {}
    if not isinstance(vary, Mapping) or not vary:
        raise ValueError("'vary' must map at least one parameter to a list of values")
    for key, values in vary.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"vary.{key} must be a non-empty list")
        if len({json.dumps(v, sort_keys=True) for v in values}) != len(values):
            raise ValueError(f"vary.{key} has duplicate values")
    if not isinstance(fixed, Mapping):
        raise ValueError("'fixed' must be a mapping")
    both = set(vary) & set(fixed)
    if both:
        raise ValueError(f"parameters both varied and fixed: {sorted(both)}")
    return SweepSpec(
        name=name.strip(),
        base=_resolve(data.get("base", "config/default.yaml")),
        vary={str(k): list(v) for k, v in vary.items()},
        fixed={str(k): v for k, v in fixed.items()},
    )


def load_sweep(path: str | Path) -> SweepSpec:
    """Read and validate a sweep YAML file."""
    return parse_sweep(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _short(key: str) -> str:
    """Last part of a dotted key, but ``automation.enabled`` -> ``automation``."""
    parts = key.split(".")
    return parts[-2] if parts[-1] == "enabled" and len(parts) > 1 else parts[-1]


def run_name(sweep: str, values: Mapping[str, Any]) -> str:
    """``"quick: policy=edf, actual_growth=4.0"``: readable in the runs list."""
    return f"{sweep}: " + ", ".join(f"{_short(k)}={v}" for k, v in values.items())


def expand(spec: SweepSpec, base: Params | None = None) -> list[SweepRun]:
    """Every combination of ``vary`` on top of ``base`` + ``fixed`` (validated, file order)."""
    start = apply_overrides(load_params(spec.base) if base is None else base, spec.fixed)
    keys = list(spec.vary)
    out = []
    for combo in itertools.product(*(spec.vary[k] for k in keys)):
        values = dict(zip(keys, combo, strict=True))
        out.append(SweepRun(run_name(spec.name, values), values, apply_overrides(start, values)))
    return out


def new_sweep_id(name: str, when: datetime | None = None) -> str:
    """``<slug>-YYYYMMDD-HHMMSS``: a valid run-store id shared by the sweep's runs."""
    return f"{runs.slugify(name)}-{(when or datetime.now()):%Y%m%d-%H%M%S}"


def queue_sweep(
    spec: SweepSpec, root: Path | str | None = None, *, sweep_id: str | None = None
) -> tuple[str, list[str]]:
    """Create one queued run per combination; returns ``(sweep_id, run_ids)``."""
    sweep_id = sweep_id or new_sweep_id(spec.name)
    run_ids = [runs.create_run(r.params, r.name, sweep_id, root=root) for r in expand(spec)]
    return sweep_id, run_ids


# --- publishing ------------------------------------------------------------------


def _get(params: Params, dotted: str) -> Any:
    node: Any = params.model_dump(mode="json")
    for part in dotted.split("."):
        node = node[part]
    return node


def published_columns(spec: SweepSpec) -> list[str]:
    """Parameter columns of the published table: varied, fixed, then the always-present ones."""
    cols = list(spec.vary) + [k for k in spec.fixed if k not in spec.vary]
    cols += [k for k in ALWAYS_PUBLISHED if k not in cols]
    return cols


def summary_rows(spec: SweepSpec, sweep_id: str, root: Path | str | None = None) -> pd.DataFrame:
    """One row per finished run of ``sweep_id``: name, run_id, parameters, flat metrics (G5).

    Runs that are not ``done`` (failed, cancelled, still queued) are skipped.
    Parameter columns hold the run's actual value from its ``params.yaml``; a
    ``fixed`` key whose value is a mapping is stored as JSON text.
    """
    rows = []
    for status in runs.list_runs(root):
        if status.sweep_id != sweep_id or status.state != "done":
            continue
        folder = runs.run_dir(status.run_id, root)
        params = runs.read_params(status.run_id, root)
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        row: dict[str, Any] = {"name": status.name, "run_id": status.run_id}
        for key in published_columns(spec):
            value = _get(params, key)
            row[key] = (
                json.dumps(value, sort_keys=True) if isinstance(value, dict | list) else value
            )
        row.update(metrics.flatten_summary(summary))
        rows.append(row)
    return pd.DataFrame(rows)


def publish(
    spec: SweepSpec,
    sweep_id: str,
    root: Path | str | None = None,
    out_dir: Path | str = PUBLISHED_DIR,
) -> Path:
    """Write ``<out_dir>/<sweep name>_summary.parquet`` from the sweep's finished runs."""
    df = summary_rows(spec, sweep_id, root)
    if df.empty:
        raise ValueError(f"sweep {sweep_id!r} has no finished runs to publish")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{runs.slugify(spec.name)}{PUBLISHED_SUFFIX}"
    df.to_parquet(path, index=False)
    return path


def latest_sweep_id(name: str, root: Path | str | None = None) -> str | None:
    """The most recent ``sweep_id`` created for sweep ``name`` (for ``--publish`` alone)."""
    prefix = f"{runs.slugify(name)}-"
    ids = sorted(
        {s.sweep_id for s in runs.list_runs(root) if s.sweep_id and s.sweep_id.startswith(prefix)}
    )
    return ids[-1] if ids else None


__all__: Sequence[str] = (
    "SweepRun",
    "SweepSpec",
    "expand",
    "load_sweep",
    "new_sweep_id",
    "parse_sweep",
    "publish",
    "queue_sweep",
    "summary_rows",
)
