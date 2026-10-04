# DATA CONTRACTS — Scout Capacity Planner

> This project has no HTTP API, so this file plays the role `API_SPEC.md` plays
> elsewhere: it is the **contract between pipeline stages, and between the app
> and the worker**. If two parts are built in parallel, they build against this
> file. Status: **draft**, firmed up in M0 (domain dataclasses), M1 (generator
> output) and M4 (run folder). Change it here first, then in code.

Conventions: dates are `date` (no time component; the simulation is daily).
IDs are strings with a prefix (scouts `FT001` / `FL001` by employment, hires `H001`, `R00001`, `F00001`, `T00001-desk`).
Money is in one cost unit (think €), stored as float. Hours are float.

## 0. The run folder: the unit of everything

Every run, whether from the app, the CLI or a sweep, is one folder. The app
and the worker communicate **only** through these folders: the folder is the
job queue (D-013).

```
data/runs/<run_id>/                 run_id = YYYYMMDD-HHMMSS-<slug>, e.g. 20261003-101500-hire-for-2x
├── params.yaml        # full, validated parameter set; written once, never edited (starts with schema_version)
├── status.json        # lifecycle, rewritten atomically (temp file + os.replace)
├── cancel             # optional flag file written by the app; worker stops at next week
├── log.txt            # worker log for this run (errors end up here too)
├── raw/               # [1] scouts, scout_unavailability, scout_weekly_hours, fixtures, requests (.parquet)
├── forecast.parquet   # [2]
├── backtest.parquet   # [2]
├── capacity_plan.parquet   # [3]
├── hiring_plan.parquet     # [3]
├── results/
│   ├── seeds.parquet  # [5] one row per seed
│   ├── weekly.parquet # [5] backlog over time per seed
│   └── requests.parquet  # [5] per-request outcome per seed (turnaround, on time, cost)
└── summary.json       # headline metrics aggregated over seeds (mean, min, max)
```

`data/runs/dev/` is the fixed folder the stage-by-stage make targets
(`make data`, `make forecast`, …) write to during development.

### `status.json`
```json
{
  "run_id": "20261003-101500-hire-for-2x",
  "name": "Hire for 2x, get 4x",
  "state": "queued | running | done | failed | cancelled",
  "created_at": "2026-10-03T10:15:00", "started_at": null, "finished_at": null,
  "progress": 0.0,                 // 0..1, updated about weekly in simulated time
  "stage": "generate | forecast | capacity_plan | simulate | null",
  "message": "simulating seed 2/3, week 31/52",
  "error": null,                   // set when failed; full traceback in log.txt
  "sweep_id": null,                // set when created by a sweep
  "code_version": "c5b4637"        // git SHA at run time: reproducibility
}
```

**State machine.** `queued → running → done | failed | cancelled`. Only the
worker moves a run out of `queued`. On worker start, `running` → `failed`
with error `"interrupted"`. On **every poll** the worker also moves queued runs
that have a `cancel` flag to `cancelled` (G4). The app may delete a run folder
that is not `running`.

**Worker heartbeat (G3).** The worker rewrites `data/runs/_worker.json`
(`{"pid": int, "heartbeat_at": iso-datetime, "current_run": str|null}`) every
poll (~1 s). The app treats the worker as down if the heartbeat is older than
10 s and says so next to queued runs.

The app reads the runs root from `SCOUT_RUNS_ROOT` (default `data/runs`) and
the published folder from `SCOUT_PUBLISHED_DIR` (default `data/published`);
`serve.py` passes its root to the app so worker and UI always agree.

### Pipeline entry point (called by the worker and the CLI)
```python
def run_pipeline(
    params: Params,
    run_dir: Path,                                   # existing run folder; outputs go here
    progress: Callable[[float, str, str], None],     # (fraction 0..1, stage, message)
    should_cancel: Callable[[], bool],               # polled about once per simulated week
) -> None: ...                                       # raises RunCancelled if cancelled
```
Lives in `scout_planner/pipeline.py` (M4). The run store (`runs.py`) and
worker (`worker.py`) depend only on this signature, so they can be built and
tested with a fake pipeline before M4 exists.

### Run store and worker: details fixed by the implementation
- `RunCancelled` lives in `scout_planner/errors.py` (re-exported by `runs.py`),
  so the pipeline can raise it without importing the run store.
- The pipeline runs in a **child process** of the worker, in its own process
  group. `progress` writes `status.json` at most ~2×/s (a stage change is
  always written). Anything printed to stdout/stderr lands in `log.txt`.
- Timestamps are naive local ISO datetimes **with microseconds**, so "oldest
  first" (`created_at`, then `run_id`) is exact even within one second.
- Listing ignores `dev/` and every name starting with `_` or `.`
  (`_worker.json`, `_worker.lock`, `_new-*` staging and `_trash-*` folders,
  temp files). A run is assembled in `_new-*` and renamed into place.
- `_worker.lock` (an OS file lock) guarantees **one executor per runs folder**:
  the background worker, or a CLI run via `worker.executor_lease()` /
  `worker.run_one_now()`. Crash recovery runs only while holding it. A worker
  started while the lock is held waits as a standby. The heartbeat file is
  removed on clean shutdown; its `current_run` is set *before* a run is
  marked `running`.
- Delete: refused only while a live heartbeat names the run. A `running` run
  with no live executor is stale and may be deleted. `delete_run` renames to
  `_trash-*`, re-reads the status and restores the folder if the run started
  meanwhile. A run process whose `status.json` disappears stops at its next
  `progress`/`should_cancel` call and removes folders that writers recreated.
- A run the worker cannot even start (unwritable folder, disk full) is
  skipped for the rest of that worker's life and marked `failed` if possible;
  the worker loop logs, backs off (up to 30 s) and never exits on an error.
- Cancel of a running run: if the pipeline doesn't stop within
  `--cancel-grace` (30 s) of the flag, the worker kills it → `cancelled`.
- Worker stopped mid-run (Ctrl-C, SIGTERM, parent gone) → run `failed`,
  error `"interrupted"`: the same outcome as crash recovery.
- A child that dies without recording a final state → `failed`, error
  `"run process ended without a result (exit code N)"`.

## 1. Stage [1] generate → `raw/`

### `scouts.parquet`: one row per scout (initial team; hires are added in [5])
| column | type | notes |
|---|---|---|
| scout_id | str | `FT001`… / `FL001`… (hires `H001`…) |
| name | str | synthetic |
| employment | str | `full_time` \| `freelance` |
| skills | list[str] | 1–4 skill types |
| home_region | str | one of the 4 invented regions |
| min_weekly_hours, max_weekly_hours | float | freelance weekly draw range (FT: both 37.5) |
| former_clubs | list[str] | conflict of interest |
| monthly_salary | float \| null | FT only |
| hourly_rate | float \| null | freelance only |
| joined_month | int | 0 for the initial team; set for hires |

### `scout_unavailability.parquet`: one row per scout-day off
`scout_id: str, date: date, reason: str` (`leave` \| `other`)

### `scout_weekly_hours.parquet`: pre-drawn hours per scout per week (D-015)
`scout_id: str, week_start: date (Monday), hours: float`: every scout × every
week overlapping the simulated period. Full-time rows are 37.5 and are **not**
net of leave (leave lives in `scout_unavailability`). A freelancer's days off
constrain *which* days they work, not their weekly total.

### `fixtures.parquet`
`fixture_id: str, date: date, league: str, region: str, home_club: str, away_club: str`

### `requests.parquet`: history + simulated year, one row per request
| column | type | notes |
|---|---|---|
| request_id | str | |
| client_club, player_club | str | |
| skill_type | str | |
| received_date | date | |
| due_date | date | received + turnaround days (urgent: + urgent_turnaround_days) |
| urgent | bool | express request (D-025); urgent ⇒ no live view (D-026) |
| needs_live_view | bool | |
| desk_hours | float | true duration, sampled at generation |
| writeup_hours | float | |
| period | str | `history` \| `future` |
| rework_draw | float | pre-drawn uniform; rework iff `rework_draw < rework_rate` (D-015) |
| at_risk_day_one | bool | live view needed and no player-club fixture in [received+2, due−1] |

Future rows follow **actual** growth (D-014). Each run has exactly one actual
growth, so there is no scenario column.

## 2. Stage [2] forecast → `forecast.parquet`, `backtest.parquet`

`forecast`: `month: date (1st), skill_type: str, requests_p50: float,
requests_pq: float, hours_p50: float, hours_pq: float`, `season_index: float`, `method: str` (`ets` | `seasonal_naive` fallback).
`*_p50` is the expected value; quantiles use a normal approximation with
variance = level (model) + Poisson arrivals + share-estimate + hours-per-request
terms (D-023) where `pq` = planning
quantile (default P80). Built from history × **assumed** growth.

`backtest`: `origin: date, month: date, skill_type: str ('ALL' for aggregate),
actual: float, model: float, naive: float`, `method: str`. `origin` = first forecast month; WAPE is computed from this.

## 3. Stage [3] capacity plan → `capacity_plan.parquet`, `hiring_plan.parquet`

`capacity_plan`: `month, skill_type, required_hours, available_hours, gap_hours,
hired_hours, gap_after_hires_hours`. `available_hours` = hours the current team
puts on the skill under the best (max-flow) allocation plus spare hours spread
by demand; `gap_hours` = required − available (negative = slack);
`gap_after_hires_hours` re-allocates with hires

`hiring_plan`: `month_to_act, joins_month, skill_type, hire_type
('full_time'|'freelance'), count: int, reason: str`. Month fields are int month
indices (0 = first simulated month). Reasons include "late" (act month clipped to 0),
"bridge" (freelancers covering until full-timers join), "open-ended" (gap
reaches the plan horizon) and "peak-only".

## 4. Stage [4] assign: the interface

```python
def assign(
    pool: list[WorkItem],            # eligible items: unassigned + movable (assigned, not started)
    scouts: list[ScoutState],        # scout + free hours per day in window + frozen work + calendar
    window: AssignmentWindow,        # today .. today + 6 (rolling 7-day horizon)
    cfg: AssignmentParams,
    rng: np.random.Generator,        # the run's "assignment" stream; optimiser draws exactly 1 number per call
    *,
    cost: CostParams,                # required: the optimiser prices lateness with cost.late_penalty
) -> list[Assignment]:               # (item_id, scout_id, planned_date | None)
    ...
```

`assign_with_report(...)` (same arguments) returns `AssignOutcome(assignments,
solve)`, where `solve` is the optimiser's `SolveReport` (status, objective,
best bound, gap, wall/deterministic time, `fell_back_to_edf`, `hit_wall_clock`)
or `None` for greedy policies, so a run can count fallbacks and wall-clock stops.
Inputs are validated on entry (unique ids; `hours_by_day` only inside the
window; 0 h on leave days) and sorted by id, so list order never changes the answer.

- A `WorkItem` is either a single `Task` (`assignment.unit = task`) or a
  bundle of desk review + write-up for one request (`assignment.unit = bundle`).
  Policies only see `WorkItem`s, so they don't care which unit is configured.
- Each `WorkItem` carries `current_scout_id: str | None`. Greedy policies keep
  it as is while the scout can still take it, and re-place it like a new item
  when that is no longer possible (a forced move, not churn); the optimiser may
  move or drop it at a churn penalty. Started work never
  appears in `pool`; it sits in `ScoutState` as frozen hours.
- Items left without an `Assignment` stay in the pool for the next run.
- `simulate.py` decides **when** `assign()` is called (`assignment.cadence`);
  the pool builder decides **what** is in the pool (`assignment.unit`) (D-010).

Every policy has exactly this signature and is looked up by name from a
registry (`"fcfs" | "edf" | "optimiser"`): the **strategy pattern**. The
simulation never imports a policy directly.

### Domain dataclasses (`src/scout_planner/domain.py`, firmed up in M0)
All are `@dataclass(frozen=True, slots=True)`: immutable snapshots; the
simulation keeps its own mutable bookkeeping and builds fresh ones per
assignment run. Lists passed in are normalised to `frozenset`/`tuple`.
`cfg` is `config.AssignmentParams`.

| class | fields |
|---|---|
| `Scout` | `scout_id, name, employment ('full_time'\|'freelance'), skills: frozenset, home_region, min_weekly_hours, max_weekly_hours, former_clubs: frozenset, monthly_salary?, hourly_rate?, joined_month=0` |
| `Fixture` | `fixture_id, date, league, region, home_club, away_club` |
| `Request` | `request_id, client_club, player_club, skill_type, received_date, due_date, needs_live_view, desk_hours, writeup_hours, period='future'`; `conflict_clubs` = {client, player}; `is_on_time(d)` ⇔ `d <= due_date` |
| `Task` | `task_id, request_id, kind ('desk'\|'live'\|'writeup'), hours, skill_type, due_date, fixture_id?, fixture_date?, region?, prerequisites: tuple[task_id]`; fixture fields set iff `kind='live'`. `tasks_for_request()` builds them: write-up depends on desk (+ live). |
| `WorkItem` | `item_id, request_id, kind ('desk'\|'live'\|'writeup'\|'bundle'), task_ids, hours` (remaining), `skill_type, received_date` (FCFS), `due_date` (EDF), `conflict_clubs, current_scout_id?, fixed_date?, region?` (both iff live), `depends_on: tuple[item_id]` (unfinished prerequisites, in the pool or frozen; empty = ready), `request_scout_ids: frozenset` (M3: scouts already on this request *outside the pool*, i.e. holding started work or having finished part of it; feeds the optimiser's continuity term; default empty). Built with `WorkItem.from_task()` (item_id = task_id) or `WorkItem.bundle()` (`T00001-bundle`). |
| `ScoutState` | `scout, hours_by_day: {date: hours}` (offered per window day, 0 on days off), `frozen_hours` (started work, counted first), `unavailable_dates`; `free_hours = max(0, Σ hours_by_day − frozen_hours)` |
| `AssignmentWindow` | `start, end` (inclusive); `starting(today, horizon_days)`, `days`, `n_days`, `d in window` |
| `Assignment` | `item_id, scout_id, planned_date?` (fixture date for live views; `None` = as the queue allows) |

`unavailable_dates` is separate from zero-hour days on purpose: a live view
may fall on a day with no desk hours (a weekend fixture) but never on leave.

## 5. Stage [5] simulate → `results/`, `summary.json`

`seeds.parquet`, one row per seed:
`seed, n_requests, n_late, n_at_risk_day_one, on_time_rate, mean_turnaround_days,
p90_turnaround_days, util_full_time, util_freelance, cost_salaried,
cost_freelance, cost_automation, cost_late_penalty, cost_total, n_hires_full_time,
n_hires_freelance`. `cost_total` = salaried + freelance + automation + late penalty (G2).

`weekly.parquet`: `seed, week_start, open_requests, open_hours, late_requests,
team_full_time, team_freelance` (G1)

`requests.parquet`: `seed, request_id, completed_date, turnaround_days,
on_time, at_risk_day_one, scouts_involved`

`summary.json`: each `seeds.parquet` metric as `{mean, min, max}` plus
`meets_target: bool` (mean on-time ≥ `sim.target_on_time`).
Also `n_censored` ({mean,min,max}), `n_seeds` (int) and `diagnostics` (dict:
optimiser solves, wall-clock stops, EDF fallbacks, mean/max solve time).
`seeds.parquet`: `seed` = replication index 0..N−1; `n_requests` = scored
requests (on-time denominator). `results/requests.parquet` also carries
`scored, received_date, due_date, skill_type, needs_live_view` (replications ≥1
have different requests from `raw/`, so never join outcomes to `raw/`).

## 6. Sweeps

`config/sweeps/<name>.yaml`:
```yaml
name: headline
base: config/default.yaml          # starting parameters
vary:                              # full cross-product of these
  assignment.policy: [fcfs, edf, optimiser]
  demand.actual_growth: [1.0, 2.0, 4.0]
  automation.enabled: [false, true]
fixed:                             # overrides applied to every run
  sim.seeds: 5
```
Optional `link: {target: source}` copies a varied/fixed value into a derived
key for every combination (e.g. `capacity_plan.assumed_growth: demand.actual_growth`);
targets may not be varied or fixed; cycles are rejected.
Expands into ordinary run folders sharing a `sweep_id`.
`data/published/<sweep>_summary.parquet` (committed) holds one row per run:
the varied parameters plus the `summary.json` metrics flattened as
`<metric>_mean`, `<metric>_min`, `<metric>_max`, plus `meets_target` (G5).
Also `name`, `run_id`, and dotted parameter columns for every varied key,
every `fixed` key (constant columns), and always `assignment.policy`,
`demand.actual_growth`, `automation.enabled`, `sim.seeds`, `sim.target_on_time`
(so "Clone settings" can rebuild a row exactly). Publishing refuses a
partial grid unless `--allow-partial`; each file carries `sweep_id`,
`sweep_expected_runs`, `sweep_done_runs` columns and parquet metadata key
`scout_planner.sweep`, plus `diag_*` solver-diagnostic columns (fallbacks,
fallback share, wall-clock hits, CP-SAT status counts). Every parameter that differs from
the defaults in any run is also published (ranges as lists), so each row is a
complete recipe. The file is written atomically (temp + `os.replace`), so
concurrent sweeps in different roots can publish safely.
