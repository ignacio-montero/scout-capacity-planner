"""Demo run folders for building and testing the app before the M4 pipeline exists.

Writes run folders that follow DATA_CONTRACTS.md exactly (section 0 layout,
``status.json`` via the run store, sections 1-3 and 5 tables, ``summary.json``)
and a published sweep summary (section 6). Nothing here is the real
simulation: ``raw/`` comes from the real generator, everything downstream
(forecast, capacity plan, hiring plan, results) from a small **toy model**
that is only meant to look plausible and stay consistent across files:

* ``results/requests.parquet`` decides every outcome; ``seeds.parquet``,
  ``weekly.parquet`` and ``summary.json`` are aggregated from it, so on-time
  rates, backlogs and late counts agree with each other.
* Team sizes, hires and salaried cost all follow ``hiring_plan.parquet``.
* The toy model responds to the parameters in the right direction (more
  growth than planned for: worse; optimiser > EDF > FCFS; pre-screen saves
  hours), so the demo tells the same story the real simulator will.

Used by ``scripts/make_demo_runs.py`` and by the app tests. Every demo run
name starts with ``"Demo:"`` so demo folders are never mistaken for real runs.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scout_planner import generate, runs
from scout_planner.config import Params, apply_overrides, load_default_params
from scout_planner.results import SUMMARY_METRICS, summary_from_seeds

FULL_TIME_MONTHLY_HOURS = 37.5 * 52 / 12 * 0.86  # net of leave and admin, toy value
FREELANCE_MONTHLY_HOURS = 16.5 * 52 / 12 * 0.8
POLICY_THRESHOLD = {"fcfs": 0.70, "edf": 0.80, "optimiser": 0.92}
DEMO_NOW = dt.datetime(2026, 10, 3, 12, 0, 0)


@dataclass(frozen=True)
class DemoOutputs:
    """Everything a finished run writes, as DataFrames (dates as ``datetime.date``)."""

    forecast: pd.DataFrame
    backtest: pd.DataFrame
    capacity_plan: pd.DataFrame
    hiring_plan: pd.DataFrame
    seeds: pd.DataFrame
    weekly: pd.DataFrame
    outcomes: pd.DataFrame
    summary: dict[str, Any]


# --- the toy model -------------------------------------------------------------------


def _month_index(dates: pd.Series) -> np.ndarray:
    start = generate.SIM_START
    return ((dates.dt.year - start.year) * 12 + dates.dt.month - 1).to_numpy()


def _month_start(index: int) -> dt.date:
    return generate.add_months(generate.SIM_START, index)


def _request_hours(req: pd.DataFrame, params: Params) -> np.ndarray:
    desk = req["desk_hours"].to_numpy()
    auto = params.automation
    if auto.enabled:
        failed = req["rework_draw"].to_numpy() < auto.rework_rate
        desk = np.where(failed, desk + auto.rework_overhead_hours, desk * (1 - auto.desk_reduction))
    live = req["needs_live_view"].to_numpy() * params.demand.live_view_hours
    return desk + req["writeup_hours"].to_numpy() + live


def _forecast(world: generate.World, params: Params, months: int) -> pd.DataFrame:
    req = world.requests
    hist = req[req["period"] == "history"].copy()
    hist["received_date"] = pd.to_datetime(hist["received_date"])
    last_year = hist[
        hist["received_date"] >= pd.Timestamp(generate.add_months(generate.SIM_START, -12))
    ]
    base = (
        last_year.groupby("skill_type").size()
        / 12
        * (1 + params.demand.history_growth_per_year) ** 0.5
    )
    hours_per = (
        req[req["period"] == "history"]
        .assign(h=lambda d: _request_hours(d, params))
        .groupby("skill_type")["h"]
        .mean()
    )
    season = generate.seasonal_index(params.demand.seasonality_strength)
    z = statistics.NormalDist().inv_cdf(params.capacity_plan.quantile)
    rows = []
    for m in range(months):
        month = _month_start(m)
        ramp = params.capacity_plan.assumed_growth ** ((m + 0.5) / 12)
        for skill in generate.SKILL_TYPES:
            p50 = float(base.get(skill, 0.0)) * season[month.month - 1] * ramp
            pq = p50 * (1 + 0.11 * z)
            h = float(hours_per.get(skill, 12.0))
            rows.append(
                {
                    "month": month,
                    "skill_type": skill,
                    "requests_p50": p50,
                    "requests_pq": pq,
                    "hours_p50": p50 * h,
                    "hours_pq": pq * h,
                }
            )
    return pd.DataFrame(rows)


def _backtest(world: generate.World, rng: np.random.Generator) -> pd.DataFrame:
    req = world.requests
    hist = req[req["period"] == "history"].copy()
    hist["month"] = pd.to_datetime(hist["received_date"]).dt.to_period("M").dt.to_timestamp()
    counts = hist.groupby("month").size()
    rows = []
    for back in (12, 9, 6):
        origin = generate.add_months(generate.SIM_START, -back)
        for h in range(6):
            month = pd.Timestamp(generate.add_months(origin, h))
            last_year = month - pd.DateOffset(years=1)
            if month not in counts.index or last_year not in counts.index:
                continue
            actual = float(counts[month])
            rows.append(
                {
                    "origin": origin,
                    "month": month.date(),
                    "skill_type": "ALL",
                    "actual": actual,
                    "model": actual * (1 + rng.normal(0, 0.06)),
                    "naive": float(counts[last_year]),
                }
            )
    return pd.DataFrame(rows)


def _initial_hours_by_skill(world: generate.World) -> dict[str, float]:
    hours = dict.fromkeys(generate.SKILL_TYPES, 0.0)
    for _, scout in world.scouts.iterrows():
        per_month = (
            FULL_TIME_MONTHLY_HOURS
            if scout["employment"] == "full_time"
            else FREELANCE_MONTHLY_HOURS
        )
        skills = list(scout["skills"])
        for skill in skills:
            hours[skill] = hours.get(skill, 0.0) + per_month / len(skills)
    return hours


def _capacity_and_hiring(
    forecast: pd.DataFrame, world: generate.World, params: Params, months: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cp = params.capacity_plan
    initial = _initial_hours_by_skill(world)
    added = {s: np.zeros(months) for s in generate.SKILL_TYPES}
    plan_rows, hire_rows = [], []
    for m in range(months):
        month = _month_start(m)
        for skill in generate.SKILL_TYPES:
            f = forecast[(forecast["skill_type"] == skill) & (forecast["month"] == month)]
            required = float(f["hours_pq"].iloc[0]) / cp.target_utilisation
            available = initial[skill] + added[skill][m]
            shortfall = required - available
            if shortfall > 0:
                if m >= cp.lead_time_full_time_months and m >= cp.persistent_gap_months - 1:
                    kind, per, lead = (
                        "full_time",
                        FULL_TIME_MONTHLY_HOURS,
                        cp.lead_time_full_time_months,
                    )
                elif m >= cp.lead_time_freelance_months:
                    kind, per, lead = (
                        "freelance",
                        FREELANCE_MONTHLY_HOURS,
                        cp.lead_time_freelance_months,
                    )
                else:
                    kind = None
                if kind is not None:
                    count = math.ceil(shortfall / per)
                    added[skill][m:] += count * per
                    available += count * per
                    hire_rows.append(
                        {
                            # month indices (0 = first plan month), as the pipeline writes
                            "month_to_act": m - lead,
                            "joins_month": m,
                            "skill_type": skill,
                            "hire_type": kind,
                            "count": count,
                            "reason": (
                                f"gap of {shortfall:,.0f} h in {month:%b %Y}"
                                + (
                                    " lasting 3+ months"
                                    if kind == "full_time"
                                    else ", short notice"
                                )
                            ),
                        }
                    )
            plan_rows.append(
                {
                    "month": month,
                    "skill_type": skill,
                    "required_hours": required,
                    "available_hours": available,
                    "gap_hours": required - available,
                }
            )
    hiring = pd.DataFrame(
        hire_rows,
        columns=["month_to_act", "joins_month", "skill_type", "hire_type", "count", "reason"],
    )
    hiring["count"] = hiring["count"].astype("int64")
    return pd.DataFrame(plan_rows), hiring


def _team_by_month(
    params: Params, hiring: pd.DataFrame, months: int
) -> tuple[np.ndarray, np.ndarray]:
    ft = np.full(months, float(params.team.full_time_count))
    fl = np.full(months, float(params.team.freelance_count))
    if params.team.follow_hiring_plan:
        for _, hire in hiring.iterrows():
            m = int(hire["joins_month"])
            target = ft if hire["hire_type"] == "full_time" else fl
            target[m:] += hire["count"]
    return ft, fl


def simulate_demo(params: Params, world: generate.World) -> DemoOutputs:
    """The toy model: forecast -> plan -> outcomes per seed -> aggregates."""
    months = params.sim.months
    end = pd.Timestamp(generate.sim_end(params))
    rng = np.random.default_rng(params.sim.seed)
    forecast = _forecast(world, params, months)
    backtest = _backtest(world, rng)
    capacity_plan, hiring = _capacity_and_hiring(forecast, world, params, months)
    ft, fl = _team_by_month(params, hiring, months)

    req = world.requests[world.requests["period"] == "future"].reset_index(drop=True).copy()
    received = pd.to_datetime(req["received_date"])
    due = pd.to_datetime(req["due_date"])
    month_of = np.clip(_month_index(received), 0, months - 1)
    hours = _request_hours(req, params)
    demand_hours = np.bincount(month_of, weights=hours, minlength=months)
    capacity = ft * FULL_TIME_MONTHLY_HOURS + fl * FREELANCE_MONTHLY_HOURS
    load = demand_hours / np.maximum(capacity, 1.0)

    threshold = POLICY_THRESHOLD.get(params.assignment.policy, 0.85)
    threshold += -0.03 if params.assignment.cadence == "weekly" else 0.0
    threshold += 0.01 if params.assignment.unit == "bundle" else 0.0
    carry, stress = 0.0, np.zeros(months)
    for m in range(months):
        carry = max(0.0, carry * 0.6 + (load[m] - 1.0))
        stress[m] = load[m] + 0.5 * carry
    p_month = np.clip(0.995 - 1.0 * np.clip(stress - threshold, 0, None), 0.05, 0.995)

    at_risk = req["at_risk_day_one"].to_numpy()
    td = params.demand.turnaround_days
    seeds_rows, weekly_frames, outcome_frames = [], [], []
    week_starts = pd.to_datetime(pd.Series(generate.sim_week_starts(params)))
    for k in range(params.sim.seeds):
        seed = params.sim.seed + k
        srng = np.random.default_rng([params.sim.seed, k, 7])
        p = np.clip(p_month + srng.normal(0, 0.006) + srng.normal(0, 0.004, months), 0.02, 0.998)
        p_req = p[month_of] * np.where(at_risk, 0.5, 1.0)
        on_time = srng.random(len(req)) < p_req
        late_days = 1 + srng.exponential(4 + 6 * np.clip(stress[month_of] - 1, 0, None))
        turnaround = np.where(
            on_time,
            np.round(srng.triangular(2, 0.65 * td, td, len(req))),
            np.minimum(np.round(td + late_days), 75),
        ).astype(float)
        completed = received + pd.to_timedelta(turnaround, unit="D")
        # Censoring, as in the real metrics: scored iff due by the last simulated
        # day; work not finished by then has no completion date (and is late).
        unfinished = (completed > end).to_numpy()
        completed = completed.mask(unfinished)
        turnaround = np.where(unfinished, np.nan, turnaround)
        on_time = ((completed <= due) & ~pd.Series(unfinished)).to_numpy()
        scored = (due <= end).to_numpy()
        # Each repeat simulates its own requests: give repeats 2+ their own ids
        # so any accidental join to raw/ by request_id fails loudly in tests.
        ids = req["request_id"].astype(str) + ("" if k == 0 else f"-r{k + 1}")
        outcome_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "request_id": ids,
                    "completed_date": completed.dt.date,
                    "turnaround_days": turnaround,
                    "on_time": on_time,
                    "at_risk_day_one": at_risk,
                    "scouts_involved": np.where(req["needs_live_view"].to_numpy(), 2, 1)
                    + (srng.random(len(req)) < 0.1),
                    "scored": scored,
                    "received_date": received.dt.date,
                    "due_date": due.dt.date,
                    "skill_type": req["skill_type"].astype(str),
                    "needs_live_view": req["needs_live_view"].to_numpy(),
                }
            )
        )
        # Backlog at each week start, straight from the outcomes above.
        ws = week_starts.to_numpy()[:, None]
        rec, comp, due_ = (
            received.to_numpy()[None, :],
            completed.to_numpy()[None, :],
            due.to_numpy()[None, :],
        )
        open_mask = (rec <= ws) & ~(comp <= ws)  # unfinished (NaT) stays open
        ws_month = np.clip(_month_index(week_starts), 0, months - 1)
        weekly_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "week_start": week_starts.dt.date,
                    "open_requests": open_mask.sum(axis=1).astype("int64"),
                    "open_hours": (open_mask * hours[None, :]).sum(axis=1),
                    "late_requests": (open_mask & (due_ < ws)).sum(axis=1).astype("int64"),
                    "team_full_time": ft[ws_month].astype("int64"),
                    "team_freelance": fl[ws_month].astype("int64"),
                }
            )
        )
        util_ft = float(
            np.clip(np.mean(np.minimum(load, 1.05)) * 0.92 + srng.normal(0, 0.01), 0.3, 0.97)
        )
        util_fl = float(np.clip(util_ft - 0.1 + srng.normal(0, 0.01), 0.2, 0.95))
        n_late = int((scored & ~on_time).sum())
        done_scored = turnaround[scored & ~unfinished]
        salaried = float(ft.sum() * params.cost.full_time_monthly_salary)
        freelance = float(
            fl.sum() * FREELANCE_MONTHLY_HOURS * util_fl * params.cost.freelance_hourly_rate
        )
        automation = params.automation.monthly_cost * months if params.automation.enabled else 0.0
        late_cost = n_late * params.cost.late_penalty
        hires = hiring if params.team.follow_hiring_plan else hiring.iloc[0:0]
        seeds_rows.append(
            {
                "seed": seed,
                "n_requests": int(scored.sum()),
                "n_at_risk_day_one": int((at_risk & scored).sum()),
                "on_time_rate": float(on_time[scored].mean()),
                "mean_turnaround_days": float(done_scored.mean()),
                "p90_turnaround_days": float(np.quantile(done_scored, 0.9)),
                "util_full_time": util_ft,
                "util_freelance": util_fl,
                "cost_salaried": salaried,
                "cost_freelance": freelance,
                "cost_automation": float(automation),
                "cost_late_penalty": float(late_cost),
                "cost_total": salaried + freelance + automation + late_cost,
                "n_hires_full_time": int(
                    hires.loc[hires["hire_type"] == "full_time", "count"].sum()
                ),
                "n_hires_freelance": int(
                    hires.loc[hires["hire_type"] == "freelance", "count"].sum()
                ),
                "n_late": n_late,
                "n_censored": int((~scored).sum()),
            }
        )
    seeds = pd.DataFrame(seeds_rows)
    summary = summary_from_seeds(seeds, params.sim.target_on_time)
    summary_json: dict[str, Any] = {
        name: {"mean": s.mean, "min": s.min, "max": s.max} for name, s in summary.metrics.items()
    }
    summary_json["meets_target"] = summary.meets_target
    summary_json["n_seeds"] = params.sim.seeds
    optimiser = params.assignment.policy == "optimiser"
    rounds = 365 if params.assignment.cadence == "daily" else 52
    summary_json["diagnostics"] = {
        "assignment_runs": rounds * params.sim.seeds,
        "optimiser_solves": rounds * params.sim.seeds if optimiser else 0,
        # toy: the pre-screen run shows what the time-limit warning looks like
        "wall_clock_hits": 3 if optimiser and params.automation.enabled else 0,
        "fallbacks": 0,
        "mean_solve_seconds": 0.21 if optimiser else 0.0,
        "max_solve_seconds": 1.0 if optimiser else 0.0,
        "live_view_retargets": 60 * params.sim.seeds,
        "reworks": 0,
        "per_seed": [],
    }
    return DemoOutputs(
        forecast=forecast,
        backtest=backtest,
        capacity_plan=capacity_plan,
        hiring_plan=hiring,
        seeds=seeds[["seed", *SUMMARY_METRICS]],  # contract columns + n_censored
        weekly=pd.concat(weekly_frames, ignore_index=True),
        outcomes=pd.concat(outcome_frames, ignore_index=True),
        summary=summary_json,
    )


# --- writing run folders ---------------------------------------------------------------


def _write_stage_outputs(run_path: Path, out: DemoOutputs) -> None:
    out.forecast.to_parquet(run_path / "forecast.parquet", index=False)
    out.backtest.to_parquet(run_path / "backtest.parquet", index=False)
    out.capacity_plan.to_parquet(run_path / "capacity_plan.parquet", index=False)
    out.hiring_plan.to_parquet(run_path / "hiring_plan.parquet", index=False)


def _write_results(run_path: Path, out: DemoOutputs) -> None:
    results = run_path / "results"
    results.mkdir(exist_ok=True)
    out.seeds.to_parquet(results / "seeds.parquet", index=False)
    out.weekly.to_parquet(results / "weekly.parquet", index=False)
    out.outcomes.to_parquet(results / "requests.parquet", index=False)
    (run_path / "summary.json").write_text(
        json.dumps(out.summary, indent=2) + "\n", encoding="utf-8"
    )


def make_params(overrides: Mapping[str, Any] | None = None) -> Params:
    return apply_overrides(load_default_params(), dict(overrides or {}))


def write_run(
    root: Path,
    name: str,
    overrides: Mapping[str, Any] | None = None,
    *,
    state: str = "done",
    created: dt.datetime = DEMO_NOW,
    duration_s: float = 95.0,
    sweep_id: str | None = None,
    error: str | None = None,
    cancel_flag: bool = False,
) -> str:
    """Create one run folder through the run store and fill it for ``state``.

    ``done``: every output; ``running``: raw + forecast + plans, status at
    63%; ``failed``/``cancelled``: partial outputs and a log; ``queued``:
    params + status only. Returns the run id.
    """
    params = make_params(overrides)
    run_id = runs.create_run(params, name, sweep_id, root=root, now=created)
    run_path = runs.run_dir(run_id, root)
    if state == "queued":
        if cancel_flag:
            runs.request_cancel(run_id, root)
        return run_id
    started = created + dt.timedelta(seconds=20)
    runs.update_status(
        run_id,
        root=root,
        now=started,
        state="running",
        stage="generate",
        progress=0.02,
        message="generating the synthetic world",
    )
    world = generate.generate_world(params)
    generate.write_world(world, run_path)
    runs.append_log(run_id, "generate: done", root)
    out = simulate_demo(params, world)
    _write_stage_outputs(run_path, out)
    runs.append_log(run_id, "forecast and capacity plan: done", root)
    finished = started + dt.timedelta(seconds=duration_s)
    if state == "running":
        runs.update_status(
            run_id,
            root=root,
            now=started,
            progress=0.63,
            stage="simulate",
            message=f"simulating seed 2/{params.sim.seeds}, week 31/52",
        )
        if cancel_flag:
            runs.request_cancel(run_id, root)
        return run_id
    if state == "done":
        _write_results(run_path, out)
        runs.append_log(run_id, "simulate: done", root)
        runs.update_status(
            run_id, root=root, now=finished, state="done", stage="simulate", message="finished"
        )
    elif state == "failed":
        message = error or "RuntimeError: no feasible assignment for day 2027-03-14"
        runs.append_log(run_id, "Traceback (most recent call last):", root)
        runs.append_log(run_id, '  File "pipeline.py", line 88, in run_pipeline', root)
        runs.append_log(run_id, message, root)
        runs.update_status(
            run_id,
            root=root,
            now=finished,
            state="failed",
            stage="simulate",
            progress=0.41,
            message="failed",
            error=message,
        )
    elif state == "cancelled":
        runs.update_status(
            run_id,
            root=root,
            now=finished,
            state="cancelled",
            stage="simulate",
            progress=0.6,
            message="cancelled at week 31/52",
        )
    else:
        raise ValueError(f"unknown demo state {state!r}")
    return run_id


DEMO_DONE: tuple[tuple[str, dict[str, Any]], ...] = (
    ("Demo: Defaults", {}),
    ("Demo: Hire for 2x, get 4x", {"capacity_plan.assumed_growth": 2.0}),
    ("Demo: Earliest deadline first", {"assignment.policy": "edf"}),
    ("Demo: First come, first served", {"assignment.policy": "fcfs"}),
    ("Demo: Pre-screen on", {"automation.enabled": True}),
    ("Demo: Planned hires don't join", {"team.follow_hiring_plan": False}),
)
DEMO_SWEEP_ID = "demo-quick"


def write_demo_runs(
    root: Path,
    *,
    now: dt.datetime = DEMO_NOW,
    include_active: bool = True,
    include_sweep: bool = True,
    include_broken: bool = False,
) -> dict[str, str]:
    """Populate ``root`` with demo runs. Returns ``{name: run_id}``.

    Done runs (6), a local sweep (4 runs, ``sweep_id="demo-quick"``), one
    failed, one interrupted, one cancelled, one running, two queued (one of
    them with a cancel request), and optionally a folder with a corrupt
    ``status.json``.
    """
    root = Path(root)
    ids: dict[str, str] = {}
    minute = dt.timedelta(minutes=1)
    t = now - 300 * minute
    for name, overrides in DEMO_DONE:
        ids[name] = write_run(
            root, name, overrides, created=t, duration_s=140 if "fcfs" not in str(overrides) else 22
        )
        t += 12 * minute
    if include_sweep:
        for policy in ("edf", "optimiser"):
            for enabled in (False, True):
                name = f"Demo sweep: {policy} · pre-screen {'on' if enabled else 'off'}"
                ids[name] = write_run(
                    root,
                    name,
                    {"assignment.policy": policy, "automation.enabled": enabled, "sim.seeds": 2},
                    created=t,
                    sweep_id=DEMO_SWEEP_ID,
                )
                t += 4 * minute
    if include_active:
        ids["Demo: Broken run"] = write_run(
            root, "Demo: Broken run", {"assignment.policy": "edf"}, state="failed", created=t
        )
        t += 5 * minute
        ids["Demo: Interrupted run"] = write_run(
            root, "Demo: Interrupted run", {}, state="failed", created=t, error="interrupted"
        )
        t += 5 * minute
        ids["Demo: Cancelled run"] = write_run(
            root, "Demo: Cancelled run", {"demand.actual_growth": 6.0}, state="cancelled", created=t
        )
        t += 5 * minute
        ids["Demo: Very cautious plan"] = write_run(
            root,
            "Demo: Very cautious plan",
            {"capacity_plan.quantile": 0.9},
            state="running",
            created=now - 3 * minute,
        )
        ids["Demo: Next in line"] = write_run(
            root,
            "Demo: Next in line",
            {"cost.late_penalty": 2000.0},
            state="queued",
            created=now - 2 * minute,
        )
        ids["Demo: Changed my mind"] = write_run(
            root,
            "Demo: Changed my mind",
            {"assignment.policy": "fcfs"},
            state="queued",
            created=now - minute,
            cancel_flag=True,
        )
    if include_broken:
        broken = root / "20261003-090000-demo-broken-folder"
        broken.mkdir(parents=True, exist_ok=True)
        (broken / "status.json").write_text("{not json", encoding="utf-8")
        ids["(broken)"] = broken.name
    return ids


def write_demo_published(published_dir: Path, name: str = "headline") -> Path:
    """A published sweep summary: policy x actual growth x pre-screen, 3 repeats each."""
    published_dir = Path(published_dir)
    published_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for growth in (1.0, 2.0, 4.0):
        for policy in ("fcfs", "edf", "optimiser"):
            for enabled in (False, True):
                overrides = {
                    "assignment.policy": policy,
                    "demand.actual_growth": growth,
                    "capacity_plan.assumed_growth": growth,
                    "automation.enabled": enabled,
                    "sim.seeds": 3,
                }
                params = make_params(overrides)
                out = simulate_demo(params, generate.generate_world(params))
                flat: dict[str, Any] = {}
                for metric, stat in out.summary.items():
                    if metric in ("meets_target", "n_seeds", "diagnostics"):
                        continue
                    for bound in ("mean", "min", "max"):
                        flat[f"{metric}_{bound}"] = stat[bound]
                rows.append(
                    {
                        "run_id": None,
                        "name": f"{policy} · {growth:g}x · pre-screen {'on' if enabled else 'off'}",
                        **overrides,
                        **flat,
                        "meets_target": out.summary["meets_target"],
                    }
                )
    path = published_dir / f"{name}_summary.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
