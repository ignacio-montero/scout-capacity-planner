# ARCHITECTURE — Scout Capacity Planner

> Owner: Architect persona. Grounded in `PRD.md`. Table schemas, the
> assignment interface and the run-folder layout live in `DATA_CONTRACTS.md`;
> every tunable input is catalogued in `PARAMETERS.md`. Every deviation from
> the original brief has a `D-xxx` entry in `DECISIONS.md`.

## 1. Shape: a simulation engine with two front doors

The core is a **batch pipeline** (the "pipes and filters" pattern): five
stages run in order, each a module with one clear input and output.
`run_pipeline(params) → run folder` executes all of them for one set of
parameters. Around that core sit two "front doors" that only call it:

- **CLI** (`make run`, `make sweep`) — scripted runs, the README sweep, CI.
- **Simulator app** — a Streamlit dashboard where you fill in parameters and
  click Run, plus a **background worker** that executes queued runs (D-012, D-013).

```
                 ┌─────────────── Streamlit app ───────────────┐
                 │ New run form → writes params.yaml + status   │
  you ──────────►│ Runs list / Run detail / Compare / Sweep     │◄─── reads run folders
                 └──────────────────────┬───────────────────────┘
                                        │ status = queued
                                        ▼
                         data/runs/<run_id>/   ◄── the job queue is just folders
                                        ▲
                                        │ picks oldest queued, one at a time
                                ┌───────┴────────┐
                                │ worker process │── run_pipeline(params) ──┐
                                └────────────────┘                          │
                                                                            ▼
 params ─► [1] generate ─► [2] forecast ─► [3] capacity plan ─► [5] simulate ──► results
            synthetic       organic demand    gaps + hiring        day by day,      in the
            world           × ASSUMED growth  table                calls [4] assign run folder
            (ACTUAL growth)                                        each day
```

Principles:
- **Parameter-driven.** Every number that matters is a parameter
  (`PARAMETERS.md`), defaults in `config/default.yaml`, validated by pydantic.
  A run is fully described by its `params.yaml`.
- **Deterministic.** Every random draw goes through a `numpy.random.Generator`
  passed in explicitly. One independent stream per purpose (arrivals, task
  durations, freelancer hours, rework…), spawned from the run seed with
  `numpy.random.SeedSequence` — so changing one parameter doesn't reshuffle
  unrelated randomness (*common random numbers*, D-015).
- **Functional core, imperative shell.** Stages are pure functions on plain
  dataframes / dataclasses. File I/O lives only in `runs.py` (run store), the
  CLI and the app. The worker and the CLI call the exact same `run_pipeline`.
- Type hints throughout; docstrings on public functions.

## 2. Repo layout

```
scout-capacity-planner/
├── CLAUDE.md               # handoff doc for Claude sessions
├── README.md               # M6 — written for the reviewer
├── LICENSE                 # MIT
├── pyproject.toml          # M0 — uv-managed, src layout
├── Makefile                # M0
├── .python-version         # 3.12
├── config/
│   ├── default.yaml        # default value of every parameter
│   └── sweeps/             # sweep definitions (README headline, cadence experiment)
├── src/scout_planner/
│   ├── config.py           # pydantic parameter models (single source of truth)
│   ├── domain.py           # dataclasses: Scout, Request, Task, WorkItem, Fixture, …
│   ├── generate.py         # [1]
│   ├── forecast.py         # [2]
│   ├── plan.py             # [3] capacity plan + hiring triggers
│   ├── assign/
│   │   ├── __init__.py     # common interface + policy registry
│   │   ├── greedy.py       # fcfs, edf
│   │   └── optimiser.py    # CP-SAT
│   ├── simulate.py         # [5]
│   ├── metrics.py
│   ├── pipeline.py         # run_pipeline(params, progress_cb) → RunResult
│   ├── runs.py             # run store: create/list/read/update status, atomic writes
│   ├── worker.py           # background worker: queue loop, crash recovery
│   ├── sweep.py            # expand a sweep definition into queued/CLI runs
│   ├── serve.py            # `make app`: starts worker + Streamlit, stops both
│   ├── cli.py              # `make run`, `make sweep`, stage targets
│   └── charts.py           # plotly figures shared by README and app
├── app/                    # Streamlit app: main.py + views/ (new run, runs, detail, compare, sweep)
├── scripts/make_readme_charts.py
├── data/                   # gitignored, except data/published/
│   ├── runs/<run_id>/      # one folder per run (gitignored)
│   └── published/          # committed: headline sweep summary for README + app
├── docs/                   # PRD, ARCHITECTURE, DATA_CONTRACTS, PARAMETERS, DECISIONS, OPTIMISATION_CASE_STUDY, img/
├── tests/
└── .private/               # gitignored: original brief, guardrail term list
```

## 3. Tech stack

Python 3.12, `uv`, `pandas`, `numpy`, `pyarrow` (parquet engine), `pydantic`,
`pyyaml`, `ortools` (CP-SAT), `statsmodels`, `plotly`, `kaleido` (static PNG
export for the README), `streamlit`, `pytest`, `ruff`.
Stretch only, optional extra: `anthropic`.

`pyarrow` and `kaleido` are additions to the brief's list (parquet I/O, README
images). The worker and job queue use only the standard library
(`subprocess`, `multiprocessing`, `os.replace`) — no Celery/Redis (D-013).

## 4. Domain model

**Skills.** Position group (GK/DEF/MID/FWD) × invented league region (4) ×
language where required → collapsed to **~15 skill types actually used**, e.g.
`MID-Iberia-ES`. Each scout holds 1–4.

**Scouts (~40 by default).** `full_time` (37.5 h/week, fixed monthly salary) or
`freelance` (8–25 h/week sampled weekly, hourly rate ≈1.5× salaried
equivalent). Plus `home_region`, unavailable days, `former_clubs` (conflict of
interest). Team size and mix are parameters.

**Requests.** `client_club`, `player_club`, `skill_type`, `received_date`,
`due_date = received + 14 calendar days`, `needs_live_view` (~40%).

**Tasks per request.**
1. Desk review — 4–10 h, matching skill.
2. Live view (optional) — a specific fixture date; consumes the scout's whole
   day, counted as 8 h; scout must be in the fixture's region (D-008).
3. Write-up — 2–4 h, matching skill; only after 1 (and 2) are done.

On time ⇔ write-up completes on or before `due_date`.

**Fixtures.** Weekly rounds per invented league, each club ~once a week. If no
fixture exists between receipt and due date, the request is **at risk from day
one** — counted separately in metrics.

**Automation (video pre-screen).** If on: desk hours × (1 − reduction),
default 40%. With probability = rework rate (default 15%) the output is
unusable and the full desk hours are needed anyway, plus an overhead. Rework is
sampled when the desk task completes.

**Demand — actual vs assumed (D-014).** Two separate parameters:
- **Actual growth** (`demand.actual_growth`) — what really arrives in the
  simulated year: baseline × seasonality (peaks Jan, Jun–Aug) × growth ramp ×
  Poisson noise. Used by [1] generate.
- **Assumed growth** (`capacity_plan.assumed_growth`) — what the agency
  believes is coming and hires for. Used by [2] forecast overlay and [3]
  capacity plan. Default: equal to actual (perfect foresight on growth).
- History: **48 months** with modest organic growth (D-004). Base volume is
  **calibrated to the team**: the default team starts the year at
  `demand.start_load` (~70%) of usable hours (D-003).

## 5. Stage details

### [2] Forecast
- **Statistical baseline:** ETS(A,Ad,A) (`statsmodels`) on **log** monthly
  counts (= multiplicative seasonality; lognormal mean correction), fitted on
  the **aggregate** series;
  per-skill = aggregate × trailing-12-month skill shares (**top-down**, D-005).
  Seasonal-naive fallback if a fit fails.
- **Growth overlay:** baseline × the **assumed** growth curve.
- **Backtest:** rolling origin over the history (≥24 months training at every
  origin, 12-month horizon) vs seasonal naive; WAPE per skill and aggregate.
  If the model loses on aggregate, say so in the README.
- **Hours:** requests × mean hours per request (desk + live-view share × 8 +
  write-up, adjusted for automation incl. expected rework).
- **Quantiles:** P50 and the planning quantile (default P80) from a normal approximation
  whose variance adds level (model), Poisson arrival, share-estimate and
  hours-per-request terms; coverage is tested across seeds (D-023).

### [3] Capacity plan
- Available hours per skill per month = Σ (scout weekly hours × weeks −
  leave hours) × target utilisation (80%).
- Coverage (D-023): a min-cost max-flow over scouts → skills they hold finds
  the true shortfall per skill (spare hours move across skills); gaps below
  25% of a full-timer's month are ignored.
- Gap = required hours (planning quantile) − available hours. Persistence is judged on the
  deseasonalised full-time-only gap; a gap reaching the horizon is open-ended.
- Hiring triggers: gap in month *m* → full-time hire by *m − 3* or freelancer
  by *m − 1* (lead times are parameters). Persistent gap (≥ N consecutive
  months) → full-time; peak-only → freelance.
- Sizing (D-022): full-time hires cover the smallest gap of a persistent run
  (base load), freelancers the rest (peaks), plus "bridge" freelancers until
  full-timers join. Hires are single-skilled in the skill's region.
- Output: `month_to_act, skill_type, hire_type, count, reason`.
- If `team.follow_hiring_plan` is on, hires **join the simulated team** at
  `month_to_act + lead time`, with the right skill. This is how "planned for
  2x, got 4x" plays out in [5].

### [4] Assignment engine
Common interface, three policies (`fcfs`, `edf`, `optimiser`); signature in
`DATA_CONTRACTS.md`. Unassigned work items roll over to the next assignment run.

**Assignment cadence and unit (D-010).** Two independent parameters:
- `assignment.cadence: daily | weekly` — how often `simulate.py` calls
  `assign()`. Default **daily**.
- `assignment.unit: task | bundle` — how the open pool is built. `task`: desk
  review, live view and write-up are separate items; a write-up becomes
  eligible only once its prerequisites are done. `bundle`: desk review +
  write-up form one item for one scout, worked back-to-back (live view stays
  separate). Default **task**; `bundle` is a nice-to-have.

**Rolling horizon + frozen zone.** Each assignment run looks at the next 7 days
of each scout's free hours (*rolling horizon*). Work a scout has **started**
is frozen. The optimiser may move assigned-but-not-started items to another
scout at a churn penalty; FCFS/EDF never move anything.

CP-SAT model (per assignment run):
- Decision `x[i, s] ∈ {0,1}`, created only for **eligible** pairs (skill,
  conflict of interest, live-view date + region pre-filtered in Python).
- Hard constraints: at most one scout per item; each scout's hours in the
  7-day window not exceeded (frozen work counted first); precedence.
  Plus **due-date-aware capacity** (D-021): per scout and window day *d*, work
  due by *d* fits the hours available by *d*. Live views whose match falls
  before the next assignment run are priced as lost if left unassigned.
- Objective (minimise): lateness = `cost.late_penalty × weights.lateness ×
  urgency` per request (urgency = 1/(1+slack), 1.25 if overdue), split across
  its items by hours +
  cost (freelance hours dearer) + continuity (one scout per request) + churn.
- Time limit per solve (parameter, default 1 s); accept best feasible.

Within a scout's queue, assigned work is done earliest-due-first.

### [5] Simulation
Plain day-by-day loop over 12 months (no SimPy). Each day: arrivals; hires
join if due; scouts burn hours on assigned work; completions unlock
dependants; rework sampled; assignment run on the configured cadence. Reports
progress through a callback (the worker turns it into `status.json` updates).

A run simulates `sim.seeds` replications (default 3, max 5) in parallel
processes and reports mean and min–max across them.

**Metrics:** on-time rate, mean & P90 turnaround, utilisation by employment
type, cost (salaried / freelance / late penalty), backlog over time, at-risk-
from-day-one count, demand actual vs forecast.

### Cost model
Salaried = fixed monthly per full-time scout (busy or idle). Freelance = rate ×
hours used. Late penalty = fixed amount per late report (stands in for churn
and refunds). Total cost puts runs on one axis against on-time rate.

### Sweeps (replaces the brief's fixed scenario grid — D-006, D-012)
A **sweep** is a YAML file in `config/sweeps/` listing parameter values to
vary; it expands into many ordinary runs tagged with a `sweep_id`.
- `headline.yaml` — the README answer (policy × growth × automation × team
  mix). Run once via CLI; its summary is committed to `data/published/`.
- `quick.yaml` — small subset used by `make all` (minutes).
- `cadence.yaml` — weekly vs daily side experiment at 4x (D-010).

## 6. Simulator app (Streamlit) and worker

**Pages:**
1. **New run** — a form with the *basic* parameters (grouped as in
   `PARAMETERS.md`), an *advanced* expander with the full YAML (validated by
   pydantic before queueing), a run name, and **Run**. Can start from
   defaults or by cloning an existing run's parameters.
2. **Runs** — every run with name, status, progress bar, duration, headline
   metrics; actions: open, clone, cancel (if queued/running), delete.
3. **Run detail** — headline numbers vs the 95% target; backlog over time;
   cost split; demand actual vs forecast; capacity heatmap (skill × month,
   coloured by gap) + hiring table; the run's parameters.
4. **Compare** — pick 2–4 runs: parameter diff (only what differs), metric
   table, overlaid backlog and on-time charts.
5. **Sweep results** — on-time vs total cost scatter, one dot per run, 95%
   line, cheapest passing run highlighted (the README headline).

**Execution model (D-013):** the app never computes a run itself. "Run"
creates `data/runs/<run_id>/` with `params.yaml` and `status.json`
(`queued`). A single long-lived **worker** process polls for the oldest queued
run, marks it `running`, calls `run_pipeline`, writes results, marks it `done`
or `failed` (with the error). One run at a time; each run parallelises its
seeds across cores.
- **Atomic status writes:** write to a temp file, then `os.replace` — the app
  never reads a half-written file.
- **Heartbeat:** the worker rewrites `data/runs/_worker.json` every poll; the
  app warns "worker not running" if it is stale (>10 s).
- **Crash recovery:** on start, the worker marks any `running` run as
  `failed (interrupted)`.
- **Cancel:** the app writes a `cancel` flag file; the simulation loop checks
  it once per simulated week.
- **Live progress:** the Runs page refreshes itself every few seconds
  (`st.fragment(run_every=…)`).
- `make app` starts both processes via `serve.py` and stops the worker when
  Streamlit exits.

No auth, no database, local only (deployment out of scope).

## 7. Performance budget

| Step | Budget (laptop) |
|---|---|
| generate + forecast + capacity plan | < 30 s |
| one optimiser solve | ≤ time limit (default 1 s) |
| one run, greedy policy | ~5–20 s |
| one run, optimiser | ~1–6 min (seeds in parallel) |
| `make all` (tests + quick sweep + charts) | a few minutes |
| headline sweep | allowed to take longer; run once, summary committed |

## 8. Guardrail enforcement

The "football only" guardrail is enforced as code (D-002):
- `tests/test_guardrails.py` scans every git-tracked file (contents and paths)
  for terms listed in the **untracked** `.private/banned-terms.txt`; skipped
  with a clear message if that file is missing (e.g. CI).
- A `commit-msg` hook in a tracked `.githooks/` dir (enabled via
  `git config core.hooksPath .githooks`) checks commit messages the same way.
