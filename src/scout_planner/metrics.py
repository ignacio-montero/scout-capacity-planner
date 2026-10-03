"""Metrics: replication results -> the result tables and ``summary.json``. Spec: DATA_CONTRACTS §5.

Pure functions only (no file access; ``pipeline.py`` writes the files).

Scoring rule and censoring
--------------------------
A request is **scored** iff its due date is on or before the last simulated
day. Requests due later are *censored*: the simulation stops before we can
know whether they would have been on time, so counting them either way would
bias the result (an unfinished request due in three weeks is not late yet; a
finished one only finished because it happened to be quick). Scored requests
that are still unfinished at the end count as **late**: their due date has
passed. This is the standard way to handle *right-censored* outcomes in a
fixed-horizon simulation.

* ``on_time_rate`` = on-time scored requests / scored requests.
* ``n_requests`` = scored requests (the denominator); ``n_late`` = scored
  requests that were late (unfinished included); ``n_censored`` = arrived but
  not scored.
* Turnaround = completion date - received date, in days, over **scored,
  completed** requests. Unfinished requests are excluded from turnaround
  (there is no completion date) but count as late.
* ``n_at_risk_day_one`` counts scored requests flagged at generation.

Utilisation and cost
--------------------
* Utilisation by employment type = hours worked / hours offered, summed over
  the scouts of that type, while they were on the team. A live view counts
  its full ``live_view_hours`` as worked, including the part still owed as
  time in lieu at the end, so utilisation can exceed 1 by that sliver.
* Salaried cost = sum over full-time scouts of ``monthly_salary`` x months on
  the team (``sim.months - joined_month``): paid whether busy or idle.
* Freelance cost = hours worked x the freelancer's hourly rate.
* Automation cost = ``automation.monthly_cost`` x ``sim.months`` if enabled.
* Late penalty = ``n_late`` x ``cost.late_penalty``.
* ``cost_total`` = salaried + freelance + automation + late penalty (G2).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from scout_planner.config import Params
from scout_planner.simulate import ReplicationResult

# seeds.parquet: the contract columns (DATA_CONTRACTS §5) in order, then extras.
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
# Extra per-replication columns: censoring counts and solver diagnostics.
SEED_EXTRAS: tuple[str, ...] = (
    "n_arrived",
    "n_censored",
    "n_unfinished_scored",
    "assignment_runs",
    "optimiser_solves",
    "wall_clock_hits",
    "fallbacks",
    "mean_solve_seconds",
    "max_solve_seconds",
    "live_view_retargets",
    "reworks",
    "sim_seconds",
)
INT_COLUMNS = frozenset(
    {
        "seed",
        "n_requests",
        "n_at_risk_day_one",
        "n_hires_full_time",
        "n_hires_freelance",
        "n_late",
        "n_arrived",
        "n_censored",
        "n_unfinished_scored",
        "assignment_runs",
        "optimiser_solves",
        "wall_clock_hits",
        "fallbacks",
        "live_view_retargets",
        "reworks",
    }
)
# Metrics summarised in summary.json as {mean, min, max}.
SUMMARY_METRICS: tuple[str, ...] = (*SEED_METRICS, "n_censored")
DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "assignment_runs",
    "optimiser_solves",
    "wall_clock_hits",
    "fallbacks",
    "mean_solve_seconds",
    "max_solve_seconds",
    "live_view_retargets",
    "reworks",
    "sim_seconds",
)

REQUEST_COLUMNS: tuple[str, ...] = (
    "seed",
    "request_id",
    "completed_date",
    "turnaround_days",
    "on_time",
    "at_risk_day_one",
    "scouts_involved",
    # extras: each replication has its own requests (D-023 draft), so the
    # attributes a chart needs travel with the outcome instead of a join to raw/.
    "scored",
    "received_date",
    "due_date",
    "skill_type",
    "needs_live_view",
)
WEEKLY_COLUMNS: tuple[str, ...] = (
    "seed",
    "week_start",
    "open_requests",
    "open_hours",
    "late_requests",
    "team_full_time",
    "team_freelance",
)


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else math.nan


def costs(scouts: pd.DataFrame, n_late: int, params: Params) -> dict[str, float]:
    """The cost split of one replication (see the module docstring)."""
    months = params.sim.months
    ft = scouts[scouts["employment"] == "full_time"]
    fl = scouts[scouts["employment"] == "freelance"]
    months_on_team = (months - ft["joined_month"]).clip(lower=0)
    salaried = float((ft["monthly_salary"].astype(float) * months_on_team).sum())
    freelance = float((fl["hours_worked"] * fl["hourly_rate"].astype(float)).sum())
    automation = params.automation.monthly_cost * months if params.automation.enabled else 0.0
    late = n_late * params.cost.late_penalty
    return {
        "cost_salaried": salaried,
        "cost_freelance": freelance,
        "cost_automation": float(automation),
        "cost_late_penalty": float(late),
        "cost_total": salaried + freelance + float(automation) + float(late),
    }


def seed_row(result: ReplicationResult, params: Params) -> dict[str, Any]:
    """One ``seeds.parquet`` row. ``seed`` is the replication index (0..N-1).

    (Streams are keyed by ``(sim.seed, replication)``, not ``sim.seed + k``,
    D-021; the column name is the contract's.)
    """
    o = result.outcomes
    scored = o[o["scored"]]
    done = scored[scored["completed_date"].notna()]
    n_scored = len(scored)
    n_on_time = int(scored["on_time"].sum())
    n_late = n_scored - n_on_time
    turnaround = done["turnaround_days"].to_numpy(dtype=float)

    sc = result.scouts
    util = {}
    for kind in ("full_time", "freelance"):
        rows = sc[sc["employment"] == kind]
        util[kind] = _ratio(rows["hours_worked"].sum(), rows["hours_offered"].sum())
    # Hires are identified by id (H###), not join month: with a zero lead time a
    # hire can join in month 0, alongside the initial team.
    hires = sc[sc["scout_id"].astype(str).str.startswith("H")]
    d = result.diagnostics
    solves = d.get("optimiser_solves", 0)
    row: dict[str, Any] = {
        "seed": result.replication,
        "n_requests": n_scored,
        "n_at_risk_day_one": int(scored["at_risk_day_one"].sum()),
        "on_time_rate": _ratio(n_on_time, n_scored),
        "mean_turnaround_days": float(turnaround.mean()) if len(turnaround) else math.nan,
        "p90_turnaround_days": float(np.quantile(turnaround, 0.9)) if len(turnaround) else math.nan,
        "util_full_time": util["full_time"],
        "util_freelance": util["freelance"],
        **costs(sc, n_late, params),
        "n_hires_full_time": int((hires["employment"] == "full_time").sum()),
        "n_hires_freelance": int((hires["employment"] == "freelance").sum()),
        "n_late": n_late,
        "n_arrived": len(o),
        "n_censored": len(o) - n_scored,
        "n_unfinished_scored": n_scored - len(done),
        "assignment_runs": int(d.get("assignment_runs", 0)),
        "optimiser_solves": int(solves),
        "wall_clock_hits": int(d.get("wall_clock_hits", 0)),
        "fallbacks": int(d.get("fallbacks", 0)),
        "mean_solve_seconds": _ratio(d.get("solve_seconds_total", 0.0), solves) if solves else 0.0,
        "max_solve_seconds": float(d.get("solve_seconds_max", 0.0)),
        "live_view_retargets": int(d.get("live_view_retargets", 0)),
        "reworks": int(d.get("reworks", 0)),
        "sim_seconds": float(d.get("sim_seconds", 0.0)),
    }
    return row


def seeds_frame(results: Sequence[ReplicationResult], params: Params) -> pd.DataFrame:
    """``seeds.parquet``: one row per replication, contract columns first."""
    rows = [seed_row(r, params) for r in sorted(results, key=lambda r: r.replication)]
    df = pd.DataFrame(rows, columns=["seed", *SEED_METRICS, *SEED_EXTRAS])
    for col in df.columns:
        df[col] = df[col].astype("int64" if col in INT_COLUMNS else "float64")
    return df


def requests_frame(results: Sequence[ReplicationResult]) -> pd.DataFrame:
    """``requests.parquet``: per-request outcome per replication."""
    frames = [
        r.outcomes.assign(seed=r.replication) for r in sorted(results, key=lambda r: r.replication)
    ]
    df = pd.concat(frames, ignore_index=True)[list(REQUEST_COLUMNS)]
    df["seed"] = df["seed"].astype("int64")
    df["scouts_involved"] = df["scouts_involved"].astype("int64")
    for col in ("on_time", "at_risk_day_one", "scored", "needs_live_view"):
        df[col] = df[col].astype(bool)
    df["turnaround_days"] = df["turnaround_days"].astype("float64")
    return df


def weekly_frame(results: Sequence[ReplicationResult]) -> pd.DataFrame:
    """``weekly.parquet``: backlog at the start of each week, per replication."""
    frames = [
        r.weekly.assign(seed=r.replication) for r in sorted(results, key=lambda r: r.replication)
    ]
    df = pd.concat(frames, ignore_index=True)[list(WEEKLY_COLUMNS)]
    for col in ("seed", "open_requests", "late_requests", "team_full_time", "team_freelance"):
        df[col] = df[col].astype("int64")
    df["open_hours"] = df["open_hours"].astype("float64")
    return df


def _clean(x: float) -> float | None:
    """JSON has no NaN: a metric that is undefined (nothing to measure) becomes null."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def summarise(seeds: pd.DataFrame, params: Params) -> dict[str, Any]:
    """``summary.json`` content: every metric as ``{mean, min, max}`` over replications.

    Also ``meets_target`` (mean on-time rate >= ``sim.target_on_time``),
    ``n_seeds``, and a ``diagnostics`` block (solver counters totalled over
    replications, plus the per-replication list) so a run can show how often
    the optimiser hit its wall-clock backstop or fell back to EDF.
    """
    out: dict[str, Any] = {}
    for name in SUMMARY_METRICS:
        col = seeds[name].astype(float)
        out[name] = {
            "mean": _clean(col.mean()),
            "min": _clean(col.min()),
            "max": _clean(col.max()),
        }
    on_time = out["on_time_rate"]["mean"]
    out["meets_target"] = bool(on_time is not None and on_time >= params.sim.target_on_time)
    out["n_seeds"] = len(seeds)
    totals: dict[str, Any] = {}
    for key in ("assignment_runs", "optimiser_solves", "wall_clock_hits", "fallbacks"):
        totals[key] = int(seeds[key].sum())
    solves = seeds["optimiser_solves"].to_numpy(dtype=float)
    mean_s = seeds["mean_solve_seconds"].to_numpy(dtype=float)
    totals["mean_solve_seconds"] = (
        float((solves * mean_s).sum() / solves.sum()) if solves.sum() > 0 else 0.0
    )
    totals["max_solve_seconds"] = float(seeds["max_solve_seconds"].max())
    totals["live_view_retargets"] = int(seeds["live_view_retargets"].sum())
    totals["reworks"] = int(seeds["reworks"].sum())
    totals["per_seed"] = [
        {k: _json_value(row[k]) for k in ("seed", *DIAGNOSTIC_KEYS)}
        for row in seeds.to_dict("records")
    ]
    out["diagnostics"] = totals
    return out


def _json_value(v: Any) -> Any:
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return _clean(float(v))
    return v


def flatten_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """``{"on_time_rate_mean": .., "on_time_rate_min": .., ..., "meets_target": ..}`` (G5)."""
    flat: dict[str, Any] = {}
    for name, entry in summary.items():
        if isinstance(entry, Mapping) and {"mean", "min", "max"} <= set(entry):
            for stat in ("mean", "min", "max"):
                flat[f"{name}_{stat}"] = entry[stat]
    flat["meets_target"] = bool(summary.get("meets_target", False))
    return flat
