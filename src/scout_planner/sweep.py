"""Sweeps: one YAML file -> many ordinary runs sharing a ``sweep_id`` (DATA_CONTRACTS §6).

A sweep file::

    name: quick
    base: config/default.yaml        # starting parameters (relative to the repo root)
    vary:                            # full cross-product, in file order
      assignment.policy: [edf, optimiser]
      demand.actual_growth: [1.0, 4.0]
    fixed:                           # overrides applied to every run
      sim.seeds: 2
    link:                            # optional: target <- source, per combination
      capacity_plan.assumed_growth: demand.actual_growth

expands into ``2 x 2 = 4`` parameter sets: ``base`` + ``fixed`` + one
combination of ``vary``, then ``link`` (each target takes its source's value
in that combination: "the agency assumes the growth that actually happens"),
each validated by ``config.apply_overrides``. The first ``vary`` key changes
slowest (like nested loops), so the run list reads in a predictable order.
This is a *full factorial design*: every combination, so each parameter's
effect can be read at every level of the others. ``link`` keeps a factorial
design from wasting runs on combinations nobody wants (assumed 1x with actual
4x) when the question is about something else.

Expansion (:func:`load_sweep`, :func:`expand`) is pure. Queueing goes
through the run store; :func:`publish` flattens the finished runs into
``data/published/<sweep>_summary.parquet`` (G5), written atomically (temp
file + ``os.replace``), so sweeps running at the same time into different
``--root`` folders can all publish into the shared folder safely.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
    link: dict[str, str] = field(default_factory=dict)  # target key -> source key


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
    unknown = set(data) - {"name", "base", "vary", "fixed", "link"}
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
    link = data.get("link") or {}
    _check_link(link, vary, fixed)
    return SweepSpec(
        name=name.strip(),
        base=_resolve(data.get("base", "config/default.yaml")),
        vary={str(k): list(v) for k, v in vary.items()},
        fixed={str(k): v for k, v in fixed.items()},
        link={str(k): str(v) for k, v in link.items()},
    )


def _leaves(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested parameter dict -> ``{"team.full_time_count": 24, ...}`` (leaf settings only)."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            flat.update(_leaves(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


PARAM_KEYS: frozenset[str] = frozenset(_leaves(Params().model_dump(mode="json")))


def _check_link(link: Any, vary: Mapping[str, Any], fixed: Mapping[str, Any]) -> None:
    """``link`` is ``{target: source}`` over existing leaf parameters, without cycles.

    A target may not also be varied or fixed: its value would be overwritten
    by the link, so the file would say two contradictory things.
    """
    if not isinstance(link, Mapping):
        raise ValueError("'link' must map target parameters to source parameters")
    for target, source in link.items():
        for what, key in (("target", target), ("source", source)):
            if not isinstance(key, str) or key not in PARAM_KEYS:
                raise ValueError(f"link {what} {key!r} is not a parameter")
        if target == source:
            raise ValueError(f"link {target!r} points to itself")
        if target in vary:
            raise ValueError(f"link target {target!r} is also varied")
        if target in fixed:
            raise ValueError(f"link target {target!r} is also fixed")
    for start in link:
        seen, key = [start], link[start]
        while key in link:
            if key in seen:
                raise ValueError(f"link cycle: {' -> '.join([*seen, key])}")
            seen.append(key)
            key = link[key]


def apply_links(params: Params, link: Mapping[str, str]) -> Params:
    """Set each link target to its source's value (a chain resolves to its root)."""
    if not link:
        return params
    flat = _leaves(params.model_dump(mode="json"))

    def root_value(key: str) -> Any:
        while key in link:
            key = link[key]
        return flat[key]

    return apply_overrides(params, {target: root_value(target) for target in link})


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
    """Every combination of ``vary`` on top of ``base`` + ``fixed``, then ``link`` (file order)."""
    start = apply_overrides(load_params(spec.base) if base is None else base, spec.fixed)
    keys = list(spec.vary)
    out = []
    for combo in itertools.product(*(spec.vary[k] for k in keys)):
        values = dict(zip(keys, combo, strict=True))
        params = apply_links(apply_overrides(start, values), spec.link)
        out.append(SweepRun(run_name(spec.name, values), values, params))
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
    """Named parameter columns: varied, fixed, linked, then the always-present ones.

    :func:`summary_rows` adds a column for every other leaf parameter that
    differs from the code defaults, so a row describes its run completely.
    """
    cols = list(spec.vary) + [k for k in spec.fixed if k not in spec.vary]
    cols += [k for k in spec.link if k not in cols]
    cols += [k for k in ALWAYS_PUBLISHED if k not in cols]
    return cols


_DEFAULT_LEAVES = _leaves(Params().model_dump(mode="json"))


def differing_leaves(params: Params) -> dict[str, Any]:
    """Every leaf parameter whose value differs from ``Params()``, in parameter order."""
    flat = _leaves(params.model_dump(mode="json"))
    return {k: v for k, v in flat.items() if v != _DEFAULT_LEAVES[k]}


def summary_rows(spec: SweepSpec, sweep_id: str, root: Path | str | None = None) -> pd.DataFrame:
    """One row per finished run of ``sweep_id``: name, run_id, parameters, flat metrics (G5).

    Runs that are not ``done`` (failed, cancelled, still queued) are skipped.
    Parameter columns hold the run's actual value from its ``params.yaml``: the
    named columns of :func:`published_columns` (a ``fixed`` key whose value is
    a mapping is stored as JSON text), then a dotted leaf column for every
    parameter that differs from the code defaults in any run, e.g. the
    changes a custom ``base`` file makes. Defaults + those columns rebuild each
    run's parameters exactly (``results.params_for_sweep_row``). Ranges stay
    lists.
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
            row[key] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
        rows.append((row, params, metrics.flatten_summary(summary)))
    named = set(published_columns(spec))
    extra: dict[str, None] = {}  # ordered set: parameter order, first seen first
    for _, params, _ in rows:
        extra.update(dict.fromkeys(k for k in differing_leaves(params) if k not in named))
    out = []
    for row, params, flat_metrics in rows:
        leaves = _leaves(params.model_dump(mode="json"))
        out.append({**row, **{k: leaves[k] for k in extra}, **flat_metrics})
    return pd.DataFrame(out)


# Key of the parquet file metadata that records how complete a published sweep is.
PUBLISH_METADATA_KEY = b"scout_planner.sweep"


@dataclass(frozen=True)
class SweepProgress:
    """How many of a sweep execution's runs exist and how many finished (``done``)."""

    expected: int  # combinations in the sweep file (or runs queued, if more)
    queued: int  # runs in the run store with this sweep_id, any state
    done: int

    @property
    def complete(self) -> bool:
        return self.done >= self.expected


def sweep_progress(spec: SweepSpec, sweep_id: str, root: Path | str | None = None) -> SweepProgress:
    """Count a sweep execution's runs against what the sweep file expands to."""
    mine = [s for s in runs.list_runs(root) if s.sweep_id == sweep_id]
    done = sum(1 for s in mine if s.state == "done")
    return SweepProgress(max(len(expand(spec)), len(mine)), len(mine), done)


class IncompleteSweep(ValueError):
    """``publish`` refused: some runs of the sweep are not ``done``."""


def publish(
    spec: SweepSpec,
    sweep_id: str,
    root: Path | str | None = None,
    out_dir: Path | str = PUBLISHED_DIR,
    *,
    allow_partial: bool = False,
) -> Path:
    """Write ``<out_dir>/<sweep name>_summary.parquet`` from the sweep's finished runs.

    Refuses (:class:`IncompleteSweep`) when any expected run is missing,
    failed, cancelled or unfinished, unless ``allow_partial``: a published
    table with holes would quietly answer a different question (the cheapest
    passing run may be the one that failed). Completeness is recorded either
    way, as columns ``sweep_id``, ``sweep_expected_runs``, ``sweep_done_runs``
    (easy for the app: plain ``pd.read_parquet``) and as parquet file metadata
    under :data:`PUBLISH_METADATA_KEY` (JSON).
    """
    progress = sweep_progress(spec, sweep_id, root)
    if not progress.complete and not allow_partial:
        raise IncompleteSweep(
            f"sweep {sweep_id!r}: {progress.done} of {progress.expected} runs are done "
            f"({progress.queued} in the run store); finish or re-run the rest, "
            "or publish anyway with --allow-partial"
        )
    df = summary_rows(spec, sweep_id, root)
    if df.empty:
        raise ValueError(f"sweep {sweep_id!r} has no finished runs to publish")
    df.insert(2, "sweep_id", sweep_id)
    df.insert(3, "sweep_expected_runs", progress.expected)
    df.insert(4, "sweep_done_runs", progress.done)
    table = pa.Table.from_pandas(df, preserve_index=False)
    info = {
        "sweep": spec.name,
        "sweep_id": sweep_id,
        "expected_runs": progress.expected,
        "done_runs": progress.done,
        "partial": not progress.complete,
    }
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), PUBLISH_METADATA_KEY: json.dumps(info).encode()}
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{runs.slugify(spec.name)}{PUBLISHED_SUFFIX}"
    # Atomic publish: readers (the app) and other sweeps publishing at the same
    # time never see a half-written file; a unique temp name per writer.
    fd, tmp = tempfile.mkstemp(dir=out, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_publish_info(path: Path | str) -> dict[str, Any] | None:
    """The completeness record of a published summary (``None`` for older files)."""
    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(PUBLISH_METADATA_KEY)
    return json.loads(raw) if raw else None


def latest_sweep_id(name: str, root: Path | str | None = None) -> str | None:
    """The most recent ``sweep_id`` created for sweep ``name`` (for ``--publish-only``).

    Matches ``<slug>-YYYYMMDD-HHMMSS`` exactly, so sweep ``optimiser_eval``
    never picks up an execution of ``optimiser_eval_fast`` (a plain prefix
    match would: both slugs start ``optimiser-eval-``), and sorts by the
    timestamp part.
    """
    pattern = re.compile(rf"^{re.escape(runs.slugify(name))}-(\d{{8}}-\d{{6}})$")
    stamped = {
        (m.group(1), s.sweep_id)
        for s in runs.list_runs(root)
        if s.sweep_id and (m := pattern.match(s.sweep_id))
    }
    return max(stamped)[1] if stamped else None


__all__: Sequence[str] = (
    "IncompleteSweep",
    "SweepProgress",
    "SweepRun",
    "SweepSpec",
    "apply_links",
    "differing_leaves",
    "expand",
    "load_sweep",
    "new_sweep_id",
    "parse_sweep",
    "publish",
    "queue_sweep",
    "read_publish_info",
    "sweep_progress",
    "summary_rows",
)
