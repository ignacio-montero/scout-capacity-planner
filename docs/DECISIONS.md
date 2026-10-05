# DECISIONS — Scout Capacity Planner

Running log of notable decisions and *why*. Newest last. "Rejected" names the
alternative so a future reader can see it was considered. When a modelling
choice is ambiguous: pick the simpler option, log it here, move on.

---

### D-001 — The brief is split into canonical docs and taken out of git (2026-10-02)
**What.** The original brief now lives in `.private/original-brief.md`
(gitignored) and is superseded by `PRD.md` (what/why), `ARCHITECTURE.md`
(how), `DATA_CONTRACTS.md` (stage-to-stage schemas) and this log.
**Why.** One source of truth per question; changes to the brief live here
rather than as silent edits. The brief itself also quotes the guardrail's
banned vocabulary verbatim, so committing it would break the guardrail.
**Rejected:** keeping the brief in the repo with a "superseded" banner — two
documents describing the same system drift apart, and see above.

---

### D-002 — The guardrail is enforced as code (2026-10-02)
**What.** A pytest check scans tracked file contents and paths, and a
`commit-msg` hook scans commit messages, for terms listed in the untracked
`.private/banned-terms.txt`. The check skips with a message if that file is absent.
**Why.** "Never mention X anywhere, including commit messages" is the kind of
rule humans and models forget at commit 40. A check that fails loudly is cheaper
than a review pass. The list stays untracked because a tracked list would
itself violate the rule.
**Cost.** The check only runs where the private file exists (this laptop).

---

### D-003 — Demand is calibrated to the team, not to "~600 requests in year one" (2026-10-02)
**What.** Base volume is set so the **current** team runs at ~65–75% of usable
hours at the start of the plan year; a test asserts this.
**Why.** Arithmetic. Mean work per request ≈ 7 h desk + 0.4 × 8 h live view +
3 h write-up ≈ 13 h. Usable capacity ≈ 24 FT × 37.5 h × 46 wk × 0.8 + 16 FL ×
~16 h × 46 wk × 0.8 ≈ 43k h/yr. 600 requests ≈ 8k h → the team would be ~20%
loaded, and the 14-day target would be met in nearly every scenario; the
question would have no interesting answer. Realised with defaults: ~248 requests/month (deseasonalised) at the start
of the plan year; available hours are counted without the 80% utilisation
factor, so no capacity-plan parameter can change arrivals. Calibration is
against a fixed reference world (D-018).
**Rejected:** shrinking the team to ~10 scouts — with 15 skill types that leaves
1–2 scouts per skill, so results would be driven by coverage holes, not by
policy or staffing mix.

---

### D-004 — 48 months of history, not 24 (2026-10-02)
**What.** Generate 48 months of history + 12 future (60 total).
**Why.** A model with yearly seasonality needs ≥2 full years to fit. With only
24 months, every rolling-origin backtest origin would have <24 months of
training, the model would fall back to seasonal naive, and the backtest would
compare the baseline with itself. The data is synthetic, so more history is free.
**Rejected:** pooling seasonality across skills with 24 months — workable but
more bespoke code for no gain.

---

### D-005 — Forecast = statistical baseline × growth overlay; top-down by skill (2026-10-02)
**What.** ETS forecasts the *organic* aggregate demand learned from history.
The growth scenario (1x/2x/4x by month 12) is applied as a multiplier on top.
Per-skill forecasts = aggregate × trailing-12-month skill shares.
**Why.** (a) A model can't learn a 4x ramp that hasn't started — that ramp is
a commercial assumption, which is how real demand planning treats it
("statistical forecast + judgemental overlay"). (b) Per-skill monthly series
are small counts (~10/month, many lower) and very noisy; **top-down
forecasting** from a smooth aggregate is more stable.
**Rejected:** 15 independent per-skill ETS models — fragile on sparse series and
mostly fall back to naive anyway. **Trade-off:** top-down misses shifts in
skill mix; acceptable because the synthetic mix is stable, noted in README.

---

### D-006 — Two scenario profiles: `quick` (make all) and `full` (committed) (2026-10-02) — *reframed by D-012: profiles are now sweep files (`quick.yaml`, `headline.yaml`)*
**What.** `make simulate` runs a `quick` profile (subset of the grid, 2 seeds).
`make simulate-full` runs the whole grid × 5 seeds in parallel; its small
summary is committed to `data/published/`.
**Why.** The full grid is ≥72 scenarios × 5 seeds, and each optimiser run is
hundreds of CP-SAT solves. Even at 0.3 s/solve that is hours, not "a few
minutes". Precomputing and shipping the summary lets a fresh clone render the
dashboard and README instantly.
**Rejected:** shrinking the grid to fit in minutes — loses the answer;
dropping seeds — loses the min–max spread the brief asks for.

---

### D-007 — Dashboard what-if reads precomputed results only (2026-10-02) — *SUPERSEDED by D-012*
**What.** Sliders snap to the nearest precomputed scenario. No live simulation
from the app.
**Why.** A single optimiser run takes minutes, which is not "responsive"; a
greedy-only live run would show results inconsistent with the grid.
**Consequence.** Every slider must map to a grid lever (see Q2 on the
freelancer-pool slider).

---

### D-008 — Live-view geography: same region only; a live view costs a full day (2026-10-02)
**What.** A scout can attend a fixture only in their `home_region` (optional
adjacency list in config, empty by default). A live view blocks that date and
counts as 8 h against the scout's week.
**Why.** "In or near" needs a definition; same-region is the simplest one that
still creates the geographic bottleneck the brief wants.

---

### D-009 — Repo hygiene (2026-10-02)
**What.** Python 3.12 via uv; **src layout**; `data/` gitignored except
`data/published/`; `pyarrow` and `kaleido` added to the stack; doc file is
`DECISIONS.md` (uppercase, like the other projects), not `decisions.md`.
GitHub repo is **private** until the README is ready. No homelab deployment
(out of scope per PRD).

---

### D-010 — Daily rolling-horizon assignment runs; cadence and bundling are config settings (2026-10-02)
**What.** (1) The assignment run happens **every day** over a rolling **7-day window** of
each scout's free hours. Tasks a scout has **started** are frozen; the optimiser
may reassign assigned-but-not-started tasks, paying a small churn penalty per
move. Greedy policies (FCFS, EDF) never reassign. (2) Two independent config
settings: `assignment.cadence: daily | weekly` (where `simulate.py` calls
`assign()`) and `assignment.unit: task | bundle` (how the open pool is built;
`bundle` = desk review + write-up to one scout, back-to-back). Defaults
`daily` + `task`; only the default must work in M3, `bundle` is an M4 nice-to-have.
(3) Cadence is **not** a full-grid lever; one side experiment (weekly vs daily,
all policies, 4x growth) feeds the README.
**Why.** Weekly assignment runs + "write-up only after desk review is done" costs
about half the 14-day window structurally: a Tuesday arrival finishes its
write-up on day 13–14 even with idle scouts; daily assignment runs finish it by day
~5. Freezing every assignment would leave the optimiser only the day's new
arrivals to decide, too little room to beat EDF; the frozen zone + churn
penalty is the standard fix (*time fence*, *plan nervousness*).
**Rejected:** (A) weekly as briefed — structurally late; (D) weekly assignment run +
daily top-up — two planners to keep consistent for no gain over daily;
freezing all assignments — simpler but starves the optimiser; cadence as a
full-grid lever — doubles compute for one sentence of README.

---

### D-011 — Fixed vocabulary: "capacity plan" vs "assignment run" (2026-10-02)
**What.** "Capacity plan" always means the monthly headcount decision (stage
[3]: gaps, hiring table). "Assignment run" always means the daily
who-does-which-task decision (stage [4]). Config keys are `assignment.cadence`
/ `assignment.unit`; the window type is `AssignmentWindow`. The bare word
"planning" is avoided for stage [4].
**Why.** Both were being called "planning", which made it unclear whether a
change was about hiring or about dispatching tasks.

---

### D-012 — The product is a scenario simulator, not a fixed answer (2026-10-03)
**What.** The app lets the user set parameters, launch a run, see results,
change parameters, run again and compare runs. Every run, from the app, the
CLI or a sweep, goes through one function, `run_pipeline(params)`, and
produces one run folder. The brief's fixed scenario grid becomes **sweeps**:
YAML files that expand into many ordinary runs (`headline`, `quick`, `cadence`).
Milestones re-cut: M4 = run pipeline + sweeps, M5 = simulator app,
M6 = headline sweep + README.
**Why.** User direction: the value is in asking new "what if" questions, not
only reading one precomputed answer. The engine was already parameter-driven
with a functional core, so the change is mostly in the shell around it.
Supersedes D-007; reframes D-006. Absorbs Q2 (team mix is now just
parameters; the headline sweep picks representative mixes) and Q3.
**Cost.** Roughly +½–1 weekend (form, queue, runs list, detail, compare).
Cut line in `PRD.md` §4.

---

### D-013 — Runs execute in a background worker; the job queue is folders (2026-10-03)
**What.** "Run" in the app only writes `data/runs/<id>/params.yaml` and
`status.json` (`queued`). One long-lived worker process (started with the app by
`make app`) picks the oldest queued run, executes it, and updates
`status.json` atomically (temp file + `os.replace`). One run at a time; seeds
run in parallel inside a run. On start, the worker marks leftover `running`
runs as `failed (interrupted)`. Cancel = a flag file checked weekly.
**Why.** Streamlit re-runs its script on every interaction, so a multi-minute
computation inside the app would be interrupted by any click or by closing the
tab. A separate process makes runs survive the UI, lets the user queue
variations, and gives each run a recorded lifecycle. User's choice (Option 2:
"more robust and a better artifact").
**Rejected:** (a) computing inside the Streamlit session with a progress bar:
simplest, but fragile for the reason above; (b) one subprocess per run with no
queue: concurrent runs fight over CPU cores and there is no ordering;
(c) Celery/RQ + Redis: a real queue, but a broker service for a single-user
laptop tool is unjustified. Same idea as Mise's "the job queue is a database
table", with folders instead of a table.

---

### D-014 — Actual demand and assumed demand are separate parameters (2026-10-03)
**What.** `demand.actual_growth` drives what arrives in the simulated year
([1] generate). `capacity_plan.assumed_growth` drives the forecast overlay and
hiring plan ([2], [3]). Default: equal. Hires join the simulated team when the
plan says they would.
**Why.** User's choice. It lets the simulator answer "we hired for 2x and 4x
showed up" (or over-hiring): the cost of forecast error, which is the real
risk behind the brief's question.

---

### D-015 — Independent random streams per purpose (common random numbers) (2026-10-03)
**What.** The run seed feeds a `numpy.random.SeedSequence` that spawns one
generator per purpose (arrivals, task durations, freelancer hours,
unavailability, rework, …). Each component only draws from its own stream.
Where possible, outcomes are **pre-drawn per entity** at generation: each
request's rework outcome, each freelancer's hours for each week. That way they
are identical no matter when, or in what order, the simulation asks for them.
**Why.** When comparing two runs that differ in one parameter (say, policy),
the difference in results should come from that parameter, not from different
luck. With a single shared stream, any change in the number of draws (e.g. the
optimiser triggering a rework sample earlier) shifts every later random number,
so you'd compare different worlds. This is the *common random numbers*
technique for reducing noise when comparing simulations; it makes run-to-run
comparisons meaningful with fewer seeds.

---

### D-016 — Greedy baselines pick salaried first, then least loaded (2026-10-03)
**What.** FCFS/EDF give each item to the eligible scout chosen by: salaried
before freelance → most remaining hours → scout id. A committed item that no
longer fits stays unassigned for this run (never moved).
**Why.** "First eligible scout in list order" would make results depend on how
the team list happens to be sorted. The rule stays deliberately naive (it
ignores that multi-skilled scouts are scarce), which is what the optimiser can
exploit.

---

### D-017 — Optimiser determinism, urgency curve, churn and continuity (2026-10-03) — *urgency scale and lateness weight superseded by D-021*
**What.**
- *Determinism:* CP-SAT runs single-threaded with a seed drawn from the
  `assignment` stream and a **deterministic** time limit
  (`max_deterministic_time = assignment.time_limit_s`, ≈0.8 s wall on the dev
  Mac), plus a 2× wall-clock backstop. EDF's answer is passed as a full
  solution hint (warm start).
- *Urgency:* unassigned item penalty = `lateness × urgency`, urgency =
  `10 / (1 + slack_days)` (slack < 0 → 0; overdue → 20). At the default weight
  a zero-slack item costs 1000 = `cost.late_penalty`.
- *Churn* applies when a committed not-started item is moved **or dropped**
  (a model that only penalised moves dropped work to dodge costs).
- *Continuity* reads a new `WorkItem.request_scout_ids` (scouts already on the
  request outside the pool), keeping the policy signature unchanged.
- *Precedence* is enforced by pre-filtering the pool (`depends_on`), so a
  write-up becomes assignable in the run after its prerequisites finish.
**Why.** Wall-clock limits make results depend on machine speed; parallel
search is a race between threads. Both would break "same params → same
results". The urgency curve is convex because a day of slack matters far more
at 2 days left than at 12.
**Risk to check in M4.** Salaried hours have zero marginal cost, so the
optimiser defers work that only a freelancer could do now until slack is
~3 days. That saves money but may lose on-time rate at 4x growth. To be tested
with real runs before the defaults are frozen.

---

### D-018 — Demand is calibrated once, against the default (reference) world (2026-10-03)
**What.** `base_monthly_volume` uses the **default** team and demand shape
(team counts, freelancer hours, leave, live-view share, task hours). Only
`demand.start_load` comes from the run. `start_load` = load of the reference
team at month 0.
**Why.** If calibration used the run's own team, "what if we add 10
freelancers?" would raise demand to keep load at 70% and cancel the what-if.
The same applies to the live-view share and task durations. A simulator's
what-ifs must change the *load*, not the *demand*.
**Rejected:** calibrating per run (the M1 first cut): technically consistent,
but it neutralises the team-size and demand-shape levers.

---

### D-019 — Arrival thinning keeps worlds nested across growth values (2026-10-03)
**What.** Future arrivals are drawn at the maximum growth rate (6x) and each
candidate is kept with probability `rate / ceiling` using its own pre-drawn
uniform (Lewis–Shedler *thinning*). Streams are per period and per purpose,
with a fixed number of draws per entity; fixtures come from a fixed epoch.
**Why.** A 2x world's requests are an exact subset of the 4x world's, so
growth comparisons differ only by the extra requests, not by different luck
(common random numbers, D-015). Changing `sim.months` or `history_months`
leaves the other period's draws untouched.

---

### D-020 — Plan-year forecast: the growth assumption replaces the statistical trend (2026-10-03)
**What.** Plan-year forecast = ETS level and seasonal profile at the forecast
origin × the **assumed** growth ramp. The full ETS model (damped trend +
seasonality) is what is backtested on history. The forecast reads history
rows only.
**Why.** ETS learns the organic growth in the history as a trend. Multiplying
the business growth assumption on top would count that growth twice. Treating
the assumption as the plan year's trend is how a "statistical forecast +
judgemental overlay" works in practice. Refines D-005.
**Rejected:** baseline-with-trend × growth overlay (double counting).

---

### D-021 — Fixes from the M0/M3 red-team review (2026-10-03)
**What.**
- **Due-date-aware capacity:** the optimiser now enforces, per scout and per
  window day *d*, "work due by *d* fits the hours available by *d*" (the EDF
  feasibility condition). Before, capacity was one 7-day total, so "assigned"
  quietly stood in for "on time" (*proxy objective*).
- **Perishable live views:** a live view whose fixture is before the next
  assignment run is priced as lost if not assigned now.
- **Lateness tied to the late penalty:** `assignment.weights.lateness` is now a
  multiplier (default 1.0) on `cost.late_penalty`. The penalty is **per
  request**, split across its pool items by hours share.
- **Greedy forced moves:** greedy keeps a commitment if it's still feasible, but
  re-places it when it is *forced* off (scout on leave, no longer eligible).
  Refines D-016.
- **Visible degradation:** wall-clock backstop widened to `max(10×limit,
  limit+5 s)`; wall-cap hits and EDF fallbacks are counted per run;
  invalid/infeasible models raise. Scouts sorted on entry, so list order
  can't change the answer.
- **Config hardening:** items must fit one full-timer's horizon; duplicate YAML
  keys rejected; overrides deep-merge; numpy values coerced; `schema_version`
  added for future migrations. Replications derive streams from
  `(replication, name)` rather than `seed + k`. Guardrail matches inflected
  forms and scans commit messages.
**Why.** A red-team review found that the optimiser could look better than
EDF in tests while being worse in simulation, and that `late_penalty`, the
most answer-shaping number, didn't affect assignment behaviour.
**Simulation contract (M4):** clear `current_scout_id` for items left
unassigned; re-target a live view whose fixture has passed to the player
club's next feasible fixture; known limitation: in `bundle` mode the desk
review waits for the live view.

---

### D-022 — Forecast model and hiring sizing (M2) (2026-10-03) — *model and numbers refined by D-023 (log-ETS, calibrated quantiles, max-flow coverage)*
**What.**
- Forecast model is **ETS(A,Ad,A)** (additive error, damped additive trend,
  additive yearly seasonality): the one family with exact closed-form
  prediction intervals in statsmodels (no simulation, no extra random stream).
  Planning quantile = ETS relative standard deviation × z(q) applied to the
  overlaid mean; per-skill quantiles assume skills are fully correlated.
- Backtest baseline per skill = that skill's own seasonal naive (scores the
  whole top-down approach honestly).
- Available hours = (weekly hours × weeks − leave hours) × target utilisation.
- **Hiring sizing:** a persistent gap → full-time hires sized to the
  **smallest** gap month of the run (the base load); remaining gaps →
  freelancers sized to the **largest** (the peak). "Bridge" freelancers cover
  the months before full-timers can join. Hires are single-skilled, in the
  skill's region.
**Why.** Full-timers cost money idle, freelancers don't: base load on salary,
peaks on hourly pay is the classic workforce-mix pattern. Sizing full-timers to
the run's maximum would hire December's headcount in March under a 4x ramp.
**Result with defaults:** ETS beats seasonal naive on aggregate (WAPE 10.2% vs
13.9%) and on all 15 skills; the plan-year forecast is +2.5% vs the realised
year; plan = 44 full-time + 50 freelance hires; January is unfixable (no lead
time), which is realistic and visible.

---

### D-023 — Fixes from the M1/M2/run-store red-team review (2026-10-03)
**What.**
- *Capacity allocation:* per-skill coverage computed as a min-cost max-flow (a transportation problem)
  (scouts → skills they hold), not a fixed proportional split, plus a hiring
  tolerance. The fixed split invented gaps (at 0.5x growth it still hired 14).
- *Quantiles:* per-skill variance includes Poisson noise and hours-per-request
  spread, with a multi-seed coverage test. The old "P80" covered ~56%.
- *Forecast:* ETS on log counts (multiplicative seasonality), because additive
  seasonality under-forecast January by ~14%.
- *Hiring horizon:* a gap run reaching the last month is open-ended
  (persistent); persistence is judged on the deseasonalised gap.
- *Generator (after M4):* per-scout stable random streams so changing team size
  doesn't reshuffle existing scouts; thinning ceiling also covers
  `start_load` and `seasonality_strength`.
- *Worker:* a run that throws an I/O error is skipped and marked, never
  crash-loops the worker; delete/finish races closed.
- `demand.start_load` keeps its **average-month** meaning (deseasonalised).
  January (seasonal peak ×1.35) therefore starts at ~100% load with no time to
  hire. That's realistic for a transfer window, but the UI and README must say
  so explicitly.
**Why.** Without these, low-growth and "planned 2x, got 1x" comparisons would
over-hire, "P80" wouldn't mean P80, and the team-size lever would compare
different teams rather than bigger ones.

---

### D-024 — Simulation mechanics (M4) (2026-10-03)
**What.**
- *Replications vary luck, not decisions:* the team (scouts), fixtures,
  forecast, capacity plan and hires are fixed per run; arrivals, durations,
  live-view flags, rework draws, leave and freelancer hours vary per
  replication (streams keyed `(seed, replication)`).
- *Daily step order:* weekly tick (snapshot, progress, cancel check) →
  arrivals → re-target live views whose match passed → assignment run (on
  cadence days) → execution (today's live views, then time-in-lieu repayment,
  then the queue: started work first, then earliest due).
- *`daily` cadence runs every calendar day, weekends included*: with
  weekday-only runs, Friday's run left weekend fixtures for a Saturday run
  that never came (934 vs 27 re-targets). Scouts offer no desk hours at
  weekends, so greedy policies are unaffected.
- *Time in lieu:* a weekend live view's hours are repaid from the scout's
  next working days.
- *Censoring:* requests due after the simulated year are not scored; scored
  requests unfinished at the end count as late.
**Why.** Fixed daily steps make invariants ("no scout over hours", "write-up
never before its prerequisites") checkable in tests; spawn-context process
pools parallelise replications without fork-with-threads deadlocks.

---

### D-025 — Urgent requests (7-day) make prioritisation meaningful (2026-10-03)
**What.** `demand.urgent_share` (default 0.15) of requests are express, with
`demand.urgent_turnaround_days` (default 7) instead of 14. Urgency is a
pre-drawn per-request uniform (worlds nest across shares).
**Why.** With one turnaround for everyone, due-date order = arrival order, so
FCFS and EDF were literally the same policy (byte-identical results). Express
requests near transfer deadlines are realistic and give the policies
something to differ on. `urgent_share: 0` restores the brief's original
behaviour.
**Rejected:** presenting FCFS ≡ EDF as two policies (misleading); dropping FCFS
(loses the naive baseline).

---

### D-026 — Per-scout random streams; wider thinning envelope; express = no live view (2026-10-03)
**What.**
- Each scout's attributes come from streams keyed by a stable id (`FT001`,
  `FL001`, …): adding people never reshuffles existing scouts (extends D-015).
- The thinning envelope (D-019) also covers `start_load` (up to 1.2) and
  `seasonality_strength` (0–2), so worlds nest across those levers too.
- Urgent (express) requests never need a live view: with a 7-day turnaround
  the live-view window is 5 days, so ~32% of urgent live-view requests were
  impossible from day one. That capped achievable on-time near 96% in every
  scenario.
**Why.** Every lever the simulator offers must compare like with like, and
the 95% target must stay reachable for a good setup.

---

### D-027 — Optimiser calibrated in full simulation: premium pricing, commitment horizon, queue-aware urgency (2026-10-03)
**What.** Three advanced parameters, defaults chosen by an ablation sweep in
which every combination was re-run against EDF in the same sweep:
- `assignment.cost_basis: premium`: a freelance hour costs its premium over a
  salaried hour (plus a 0.01 salaried-first tie-break), not the full rate.
- `assignment.commit_buffer_days: 2`: only commit work a scout can finish
  within 2 days after the next run; re-plan the rest (*plan long, commit
  short*).
- `assignment.load_aware: true`: slack is reduced by the queue wait behind
  earlier-due work of the same skill on the salaried team.
**Why.** The real root cause was not freelance pricing. The optimiser had no
reason to start work early (finishing on day 1 or day 6 of the window scored
the same), so it parked work late and write-ups lost their buffer. EDF's
least-loaded rule starts work early by accident.
**Result (3 seeds, paired with EDF):** default 4x with hiring plan: 95.5% vs
EDF 95.1% on time, +0.9% cost. Tight (2x, no hiring): 46.0% vs 40.8%, −196k.
The old objective: 75.9% / 22.5%. The time limit can drop to 0.3 s with the
same conclusions at half the runtime.
**Rejected:** cost weight > 1 (worse on both axes); demand-proportional or
all-scout queue waits (slightly worse); no commitment horizon (93.8%).

---

### D-028 — Headline sweep design (closes Q6) (2026-10-03)
**What.** At 4x actual growth with assumed = actual (sweep `link:`), vary:
policy {fcfs, edf, edf_feasible, optimiser} × `capacity_plan.hire_mix`
{rule, freelance_only, full_time_only} × planning quantile {P50, P80, P90} ×
tool {off, on}. 72 runs × 3 seeds (edf_feasible added by D-029), optimiser at 0.3 s. Companion sweeps: `growth`
(1x/2x/4x), `forecast_error` (assumed 2x/4x/6x vs actual 4x), `baseline` (no
hiring).
**Why.** The question asks which *mix* of salaried, freelance and automation
is cheapest at ≥95%; hire mix, planning caution and the tool are exactly
those levers, and policy answers "how should work be assigned". 3 seeds
instead of 5 keeps the sweep around an hour; ranges are still shown.

---

### D-029 — Fixes from the final red-team: one rework model, a fair baseline, honest publishing (2026-10-04)
**What.**
- *Rework:* a failed pre-screen costs the full desk review **in total** plus
  the overhead (the plan's model, the brief and the UI copy). The simulation
  had charged the reduced hours twice (+6% work with the tool on). A
  cross-module test now pins sim ≈ plan.
- *Fourth policy `edf_feasible`:* EDF that only promises work a scout can
  finish by its due date (the optimiser's due-date check, without the search).
  A 20-line version of it beat the optimiser with the tool off, so the
  optimiser's advantage must be measured against it.
- *Publishing integrity:* per-run solver diagnostics (fallbacks, status
  counts) are published; publishing refuses partial grids unless
  `--allow-partial`; sweep-id matching is exact.
- *Config:* weekly cadence requires `horizon_days ≥ 7`.
- *Late-penalty sensitivity sweep* (250/1000/4000), because the penalty is
  part of total cost and the optimiser optimises it directly while EDF doesn't.
**Why.** README claims must survive a fair baseline, a consistent model and
visible degradation. Headline and baseline sweeps are re-run after these
fixes.

---

### D-030 — `edf_feasible` is the default policy; the optimiser is the comparison (2026-10-04)
**What.** `assignment.policy` defaults to `edf_feasible` (EDF + a per-scout
due-date check, whole window; D-029). The CP-SAT optimiser stays available and
in every sweep.
**Why.** Paired runs (3 seeds): at 4x with the hiring plan, `edf_feasible`
96.1% at 3,232k vs optimiser 95.5% at 3,308k vs EDF 95.1% at 3,266k; tight
(2x, no hiring) 50.5% / 3,779k vs 46.2% / 3,973k vs 40.8% / 4,186k. A simple
heuristic with the right constraint beat the optimiser at a fraction of the
runtime (seconds vs minutes). The optimiser's earlier "win" was against a
heuristic that ignored a constraint the optimiser enforced. The commitment
horizon that fixed the optimiser (D-027) *hurt* the greedy policy, so
`edf_feasible` ignores it ("a fix belongs to the method whose flaw it fixes").
**Resolved by the final sweeps:** `edf_feasible` wins with the tool off and a
plan that keeps up with demand. The optimiser is ahead (by ~1–3% cost or a few
pts) with the tool on (97.0% / 2,735k vs 96.1% / 2,765k), at every late-penalty
level, without hiring and under weekly assignment. The default stays
`edf_feasible` (seconds vs minutes, never far behind); the README says the
optimiser's edge is small and situational.
