"""Read-only loaders for what a finished run (or a sweep) left on disk.

The run store (:mod:`scout_planner.runs`) owns the *lifecycle* files of a run
folder (``status.json``, ``params.yaml``, ``cancel``, ``log.txt``). This module
reads the *outputs* the pipeline writes next to them (DATA_CONTRACTS.md
sections 2, 3, 5 and 6) and the published sweep summaries. It never writes.

Two rules shape every loader:

* **Tolerant, never raising.** A missing or malformed file comes back as a
  :class:`Loaded` with ``value=None`` and a human-readable ``problem``. The app
  turns that into one local warning in place of one chart, and the rest of the
  page still renders (*graceful degradation*).
* **Schema checked at the edge.** Each table lists the columns the contract
  promises; a file missing one of them is reported as a problem here, instead
  of surfacing later as a ``KeyError`` deep inside a chart.

Also here: small pure transforms that join outputs into the shapes the charts
need (on-time rate by due month, demand per month, a sweep table built from
local run folders).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scout_planner import runs
from scout_planner.config import Params, apply_overrides

__all__ = [
    "COST_METRICS",
    "SEED_METRICS",
    "SUMMARY_METRICS",
    "TABLES",
    "Loaded",
    "MetricStat",
    "PublishedSweep",
    "Summary",
    "flatten_params",
    "list_published",
    "load_published",
    "load_summary",
    "load_table",
    "local_sweep_frame",
    "local_sweeps",
    "monthly_demand",
    "monthly_on_time",
    "normalise_sweep_frame",
    "params_for_sweep_row",
    "scored_outcomes",
    "summary_from_seeds",
]

# --- contract constants ----------------------------------------------------------

# seeds.parquet contract columns (DATA_CONTRACTS.md section 5), minus "seed".
SEED_METRICS: tuple[str, ...] = (
    "n_requests",
    "n_at_risk_day_one",
    "on_time_rate",
    "mean_turnaround_days",
    "p90_turnaround_days",
    "util_full_time",
    "util_freelance",
    "cost_salaried",
    "cost_freelance",
    "cost_automation",
    "cost_late_penalty",
    "cost_total",
    "n_hires_full_time",
    "n_hires_freelance",
    "n_late",
)
# summary.json holds each of these as {mean, min, max}; n_censored = arrived but
# due after the last simulated day, so not scored (right-censored).
SUMMARY_METRICS: tuple[str, ...] = (*SEED_METRICS, "n_censored")
# summary.json keys that are not metrics.
SUMMARY_EXTRAS = frozenset({"meets_target", "n_seeds", "diagnostics"})
COST_METRICS: tuple[str, ...] = (
    "cost_salaried",
    "cost_freelance",
    "cost_automation",
    "cost_late_penalty",
)


@dataclass(frozen=True)
class TableSpec:
    """Where a table lives inside a run folder and the columns it must have."""

    path: str
    columns: tuple[str, ...]
    date_columns: tuple[str, ...] = ()


TABLES: dict[str, TableSpec] = {
    "seeds": TableSpec("results/seeds.parquet", ("seed", *SEED_METRICS)),
    "weekly": TableSpec(
        "results/weekly.parquet",
        (
            "seed",
            "week_start",
            "open_requests",
            "open_hours",
            "late_requests",
            "team_full_time",
            "team_freelance",
        ),
        ("week_start",),
    ),
    "outcomes": TableSpec(
        "results/requests.parquet",
        (
            "seed",
            "request_id",
            "completed_date",
            "turnaround_days",
            "on_time",
            "at_risk_day_one",
            "scouts_involved",
            # Each repeat simulates its own requests, so the attributes the
            # charts need travel with the outcome (no join to raw/).
            "scored",
            "received_date",
            "due_date",
            "skill_type",
            "needs_live_view",
        ),
        ("completed_date", "received_date", "due_date"),
    ),
    "forecast": TableSpec(
        "forecast.parquet",
        ("month", "skill_type", "requests_p50", "requests_pq", "hours_p50", "hours_pq"),
        ("month",),
    ),
    "backtest": TableSpec(
        "backtest.parquet",
        ("origin", "month", "skill_type", "actual", "model", "naive"),
        ("origin", "month"),
    ),
    "capacity_plan": TableSpec(
        "capacity_plan.parquet",
        ("month", "skill_type", "required_hours", "available_hours", "gap_hours"),
        ("month",),
    ),
    "hiring_plan": TableSpec(
        "hiring_plan.parquet",
        ("month_to_act", "joins_month", "skill_type", "hire_type", "count", "reason"),
        ("month_to_act", "joins_month"),
    ),
    "raw_requests": TableSpec(
        "raw/requests.parquet",
        ("request_id", "skill_type", "received_date", "due_date", "period", "at_risk_day_one"),
        ("received_date", "due_date"),
    ),
}

SUMMARY_FILE = "summary.json"
PUBLISHED_SUFFIX = "_summary.parquet"
HEADLINE_SWEEP = "headline"

# Parameters every sweep table is normalised to have, because the scatter
# encodes them (colour + symbol = policy, fill = pre-screen, panel = growth).
SWEEP_KEY_PARAMS: tuple[str, ...] = (
    "assignment.policy",
    "demand.actual_growth",
    "automation.enabled",
)


# --- result containers -------------------------------------------------------------


@dataclass(frozen=True)
class Loaded[T]:
    """A value, or ``None`` plus the reason it couldn't be loaded."""

    value: T | None
    problem: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class MetricStat:
    """One metric aggregated over repeats (seeds)."""

    mean: float
    min: float
    max: float

    @property
    def has_range(self) -> bool:
        return not math.isclose(self.min, self.max)


@dataclass(frozen=True)
class Summary:
    """``summary.json``: each metric's mean/min/max over repeats, plus the verdict."""

    metrics: Mapping[str, MetricStat] = field(default_factory=dict)
    meets_target: bool | None = None
    n_seeds: int | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def stat(self, name: str) -> MetricStat | None:
        return self.metrics.get(name)

    def mean(self, name: str) -> float | None:
        stat = self.metrics.get(name)
        return None if stat is None else stat.mean

    def late_reports(self) -> float | None:
        """Mean late reports per repeat: the exact ``n_late`` when the run has it.

        Older summaries without ``n_late`` fall back to
        ``n_requests * (1 - on_time_rate)`` from the means (close, not exact).
        """
        if (late := self.mean("n_late")) is not None:
            return late
        n, rate = self.mean("n_requests"), self.mean("on_time_rate")
        if n is None or rate is None:
            return None
        return n * (1 - rate)

    def cost_parts(self) -> dict[str, float]:
        """Mean of each cost component (missing components count as 0)."""
        return {name: float(self.mean(name) or 0.0) for name in COST_METRICS}

    def to_flat(self) -> dict[str, Any]:
        """``{"on_time_rate_mean": .., "on_time_rate_min": .., ..., "meets_target": ..}``."""
        flat: dict[str, Any] = {}
        for name, stat in self.metrics.items():
            flat[f"{name}_mean"] = stat.mean
            flat[f"{name}_min"] = stat.min
            flat[f"{name}_max"] = stat.max
        flat["meets_target"] = self.meets_target
        return flat


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def parse_summary(data: Any) -> Loaded[Summary]:
    """Validate the decoded content of ``summary.json``."""
    if not isinstance(data, Mapping):
        return Loaded(None, "summary.json is not a JSON object")
    metrics: dict[str, MetricStat] = {}
    for name, entry in data.items():
        if name in SUMMARY_EXTRAS:
            continue
        if isinstance(entry, Mapping):
            values = [_as_float(entry.get(k)) for k in ("mean", "min", "max")]
            if any(v is None for v in values):
                continue
            mean, lo, hi = values
            metrics[name] = MetricStat(mean, lo, hi)  # type: ignore[arg-type]
        elif (number := _as_float(entry)) is not None:
            metrics[name] = MetricStat(number, number, number)
    if "on_time_rate" not in metrics:
        return Loaded(None, "summary.json has no on_time_rate")
    verdict = data.get("meets_target")
    n_seeds = data.get("n_seeds")
    diagnostics = data.get("diagnostics")
    return Loaded(
        Summary(
            metrics,
            verdict if isinstance(verdict, bool) else None,
            n_seeds if isinstance(n_seeds, int) and not isinstance(n_seeds, bool) else None,
            dict(diagnostics) if isinstance(diagnostics, Mapping) else {},
        )
    )


def summary_from_seeds(seeds: pd.DataFrame, target_on_time: float) -> Summary:
    """Aggregate ``seeds.parquet`` the way ``summary.json`` is defined (section 5)."""
    metrics = {
        name: MetricStat(
            float(seeds[name].mean()), float(seeds[name].min()), float(seeds[name].max())
        )
        for name in SUMMARY_METRICS
        if name in seeds.columns
    }
    on_time = metrics.get("on_time_rate")
    meets = None if on_time is None else bool(on_time.mean >= target_on_time)
    return Summary(metrics, meets, len(seeds))


# --- run folder loaders --------------------------------------------------------------


def load_summary(run_path: Path) -> Loaded[Summary]:
    """``<run>/summary.json`` as a :class:`Summary`."""
    path = Path(run_path) / SUMMARY_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Loaded(None, "summary.json is missing")
    except (OSError, ValueError) as exc:
        return Loaded(None, f"summary.json could not be read: {exc}")
    return parse_summary(data)


def read_checked_parquet(path: Path, spec: TableSpec) -> Loaded[pd.DataFrame]:
    """Read one parquet file and check it has the contract's columns."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        return Loaded(None, f"{spec.path} is missing")
    except Exception as exc:  # pyarrow raises several types for a corrupt file
        return Loaded(None, f"{spec.path} could not be read: {type(exc).__name__}")
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        return Loaded(None, f"{spec.path} lacks columns: {', '.join(missing)}")
    for column in spec.date_columns:
        if column not in df.columns:
            continue
        if pd.api.types.is_integer_dtype(df[column]):
            # hiring_plan stores months as indices (0 = first plan month).
            df[column] = [month_from_index(int(i)) for i in df[column]]
        # date32 arrives as Python dates (object dtype); plotly and groupby
        # want real datetimes.
        df[column] = pd.to_datetime(df[column])
    return Loaded(df)


def month_from_index(index: int) -> pd.Timestamp:
    """Month index -> first day of that month (0 = the plan year's first month)."""
    from scout_planner.generate import SIM_START  # a constant; nothing is generated

    total = SIM_START.year * 12 + SIM_START.month - 1 + index
    return pd.Timestamp(year=total // 12, month=total % 12 + 1, day=1)


def load_table(run_path: Path, table: str) -> Loaded[pd.DataFrame]:
    """One of the tables in :data:`TABLES` from a run folder."""
    spec = TABLES.get(table)
    if spec is None:
        raise KeyError(f"unknown table {table!r}; known: {sorted(TABLES)}")
    return read_checked_parquet(Path(run_path) / spec.path, spec)


# --- parameters as flat dotted keys ------------------------------------------------


def flatten_params(params: Params | Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """``Params`` -> ``{"team.full_time_count": 24, ..., "assignment.weights.cost": 1.0}``.

    Ranges (``[min, max]`` pairs) stay as tuples: they are one setting.
    """
    data = params.model_dump(mode="python") if isinstance(params, Params) else params
    flat: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, f"{dotted}."))
        elif isinstance(value, list):
            flat[dotted] = tuple(value)
        else:
            flat[dotted] = value
    return flat


def _python_value(value: Any) -> Any:
    """numpy values (from a parquet row) -> plain Python; lists become range tuples."""
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "tolist"):  # numpy scalar or array
        value = value.tolist()
    if isinstance(value, list):
        return tuple(value)
    # Parquet stores a missing value in a numeric column as NaN; a null parameter
    # (e.g. commit_buffer_days: null) must come back as None, not float("nan").
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


# --- published and local sweeps ------------------------------------------------------


@dataclass(frozen=True)
class PublishedSweep:
    name: str
    path: Path


def list_published(published_dir: Path) -> list[PublishedSweep]:
    """``<dir>/<sweep>_summary.parquet`` files, the headline sweep first."""
    base = Path(published_dir)
    if not base.is_dir():
        return []
    found = [
        PublishedSweep(p.name.removesuffix(PUBLISHED_SUFFIX), p)
        for p in base.iterdir()
        if p.is_file() and p.name.endswith(PUBLISHED_SUFFIX)
    ]
    return sorted(found, key=lambda s: (s.name != HEADLINE_SWEEP, s.name))


SWEEP_REQUIRED = ("on_time_rate_mean", "cost_total_mean")


def load_published(path: Path) -> Loaded[pd.DataFrame]:
    """A published sweep summary (DATA_CONTRACTS.md section 6), checked for its key columns."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        return Loaded(None, f"{Path(path).name} is missing")
    except Exception as exc:
        return Loaded(None, f"{Path(path).name} could not be read: {type(exc).__name__}")
    missing = [c for c in SWEEP_REQUIRED if c not in df.columns]
    if missing:
        return Loaded(None, f"{Path(path).name} lacks columns: {', '.join(missing)}")
    return Loaded(df)


def normalise_sweep_frame(df: pd.DataFrame, defaults: Params) -> pd.DataFrame:
    """Give every sweep table the same shape, whatever produced it.

    Adds ``policy``, ``actual_growth``, ``automation_enabled`` (from the dotted
    parameter columns, else the default value: a parameter that was not varied
    has its default everywhere unless the sweep's ``fixed`` changed it),
    ``row_id``, and ``name`` / ``run_id`` if absent. Also fills
    ``meets_target`` and the ``_min`` / ``_max`` columns when a file has only means.
    """
    out = df.reset_index(drop=True).copy()
    flat_defaults = flatten_params(defaults)
    for dotted, short in zip(
        SWEEP_KEY_PARAMS, ("policy", "actual_growth", "automation_enabled"), strict=True
    ):
        if short not in out.columns:
            out[short] = out[dotted] if dotted in out.columns else flat_defaults[dotted]
    out["automation_enabled"] = out["automation_enabled"].astype(bool)
    out["actual_growth"] = out["actual_growth"].astype(float)
    out["policy"] = out["policy"].astype(str)
    for metric in SUMMARY_METRICS:
        mean = f"{metric}_mean"
        if mean in out.columns:
            for bound in ("min", "max"):
                col = f"{metric}_{bound}"
                if col not in out.columns:
                    out[col] = out[mean]
    if "meets_target" not in out.columns:
        target = (
            out["sim.target_on_time"]
            if "sim.target_on_time" in out.columns
            else flat_defaults["sim.target_on_time"]
        )
        out["meets_target"] = out["on_time_rate_mean"] >= target
    out["meets_target"] = out["meets_target"].astype(bool)
    if "run_id" not in out.columns:
        out["run_id"] = None
    if "name" not in out.columns:
        out["name"] = [f"Run {i + 1}" for i in range(len(out))]
    out["row_id"] = range(len(out))
    return out


def local_sweeps(root: Path | str | None = None) -> dict[str, list[runs.RunStatus]]:
    """Run folders grouped by ``sweep_id`` (only runs created by a sweep)."""
    groups: dict[str, list[runs.RunStatus]] = {}
    for status in runs.list_runs(root):
        if status.sweep_id:
            groups.setdefault(status.sweep_id, []).append(status)
    return groups


def local_sweep_frame(
    statuses: Iterable[runs.RunStatus], root: Path | str | None = None
) -> pd.DataFrame:
    """A sweep summary table built from local run folders (finished runs only).

    Same shape as a published summary: the parameters that differ between the
    runs (plus :data:`SWEEP_KEY_PARAMS`), the flattened summary metrics,
    ``meets_target``, ``run_id`` and ``name``. Runs whose summary or
    parameters can't be read are skipped.
    """
    rows: list[dict[str, Any]] = []
    flats: list[dict[str, Any]] = []
    for status in statuses:
        if status.state != "done":
            continue
        try:
            params = runs.read_params(status.run_id, root)
        except Exception:
            continue
        summary = load_summary(runs.run_dir(status.run_id, root))
        if summary.value is None:
            continue
        flat = flatten_params(params)
        flats.append(flat)
        rows.append(
            {
                "run_id": status.run_id,
                "name": status.name,
                **summary.value.to_flat(),
                "sim.target_on_time": flat["sim.target_on_time"],
                "sim.seeds": flat["sim.seeds"],
            }
        )
    if not rows:
        return pd.DataFrame()
    varied = [
        key for key in flats[0] if key in SWEEP_KEY_PARAMS or len({repr(f[key]) for f in flats}) > 1
    ]
    for row, flat in zip(rows, flats, strict=True):
        for key in varied:
            value = flat[key]
            row[key] = list(value) if isinstance(value, tuple) else value
    return pd.DataFrame(rows)


def params_for_sweep_row(
    row: Mapping[str, Any], defaults: Params, sweep_file: Path | None = None
) -> Params:
    """Rebuild a sweep row's full parameters: defaults + sweep ``fixed`` + the row's values.

    Every column whose name is a parameter key (``assignment.policy``) is
    applied; other columns (metrics, names) are ignored. ``sweep_file`` is the
    sweep's YAML; its ``fixed`` block is applied when the file exists.
    """
    flat_defaults = flatten_params(defaults)
    params = defaults
    if sweep_file is not None and Path(sweep_file).is_file():
        spec = yaml.safe_load(Path(sweep_file).read_text(encoding="utf-8")) or {}
        fixed = spec.get("fixed") or {}
        if isinstance(fixed, Mapping):
            params = apply_overrides(params, dict(fixed))
    overrides: dict[str, Any] = {}
    for key, value in row.items():
        if key not in flat_defaults:
            continue
        value = _python_value(value)
        default = flat_defaults[key]
        # parquet stores every number in a column with one type: an int
        # setting in a mixed column may come back as 3.0.
        if isinstance(default, int) and not isinstance(default, bool) and isinstance(value, float):
            value = int(value)
        overrides[key] = value
    return apply_overrides(params, overrides) if overrides else params


# --- joins for charts (pure) -------------------------------------------------------


def scored_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Only the requests whose due date fell inside the simulated period.

    Requests due later are right-censored: the year ended before we could know
    whether they would be on time, so they count neither way.
    """
    if "scored" not in outcomes.columns:
        return outcomes
    return outcomes[outcomes["scored"].astype(bool)]


def monthly_on_time(outcomes: pd.DataFrame) -> pd.DataFrame:
    """On-time rate per repeat and **due** month: ``seed, month, on_time_rate, n``.

    Scored requests only. Due month, not completion month: a report that is
    never delivered still counts against the month it was promised for. Uses
    the outcome's own ``due_date``: each repeat simulates different requests,
    so joining to ``raw/`` by ``request_id`` would be wrong for repeats >= 2.
    """
    scored = scored_outcomes(outcomes)
    if scored.empty:
        return pd.DataFrame(columns=["seed", "month", "on_time_rate", "n"])
    df = scored[["seed", "on_time"]].copy()
    df["month"] = pd.to_datetime(scored["due_date"]).dt.to_period("M").dt.to_timestamp()
    return df.groupby(["seed", "month"], as_index=False).agg(
        on_time_rate=("on_time", "mean"), n=("on_time", "size")
    )


def monthly_demand(
    raw_requests: pd.DataFrame,
    outcomes: pd.DataFrame | None = None,
    history_months_shown: int = 12,
) -> pd.DataFrame:
    """Requests received per month: ``month, requests, requests_min, requests_max, period``.

    History (shared by every repeat) comes from ``raw/requests``; the last
    ``history_months_shown`` months are kept. Plan-year months come from the
    outcomes when given: every arrived request, per repeat, as mean / min /
    max over repeats (each repeat draws its own demand). Without outcomes the
    plan year falls back to ``raw/`` (which is repeat 1's world).
    """

    def per_month(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        months = pd.to_datetime(df["received_date"]).dt.to_period("M").dt.to_timestamp()
        return df.assign(month=months).groupby([*by, "month"], as_index=False).size()

    raw = raw_requests
    history = per_month(raw[raw["period"] == "history"], []).rename(columns={"size": "requests"})
    history = history.sort_values("month").tail(history_months_shown)
    history = history.assign(
        requests_min=history["requests"], requests_max=history["requests"], period="history"
    )
    if outcomes is not None and not outcomes.empty:
        counts = per_month(outcomes, ["seed"])
        future = counts.groupby("month", as_index=False)["size"].agg(["mean", "min", "max"])
        future = future.rename(
            columns={"mean": "requests", "min": "requests_min", "max": "requests_max"}
        )
    else:
        future = per_month(raw[raw["period"] != "history"], []).rename(columns={"size": "requests"})
        future = future.assign(requests_min=future["requests"], requests_max=future["requests"])
    future = future.assign(period="future").sort_values("month")
    columns = ["month", "requests", "requests_min", "requests_max", "period"]
    return pd.concat([history[columns], future[columns]], ignore_index=True)


def forecast_totals(forecast: pd.DataFrame) -> pd.DataFrame:
    """Forecast summed over skill types: ``month, requests_p50, requests_pq``."""
    return (
        forecast.groupby("month", as_index=False)[["requests_p50", "requests_pq"]]
        .sum()
        .sort_values("month")
    )


def seeds_late_reports(seeds: pd.DataFrame) -> MetricStat:
    """Late reports per repeat, exact from ``seeds.parquet`` (``n_late``)."""
    if "n_late" in seeds.columns:
        late = seeds["n_late"].astype(float)
    else:
        late = seeds["n_requests"] * (1 - seeds["on_time_rate"])
    return MetricStat(float(late.mean()), float(late.min()), float(late.max()))


def describe_missing(problems: Sequence[str | None]) -> list[str]:
    """Drop ``None`` entries: the list of problems worth showing."""
    return [p for p in problems if p]
