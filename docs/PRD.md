# PRD — Scout Capacity Planner

> Owner: PM persona. Source: the original project brief (kept privately in
> `.private/original-brief.md`, superseded by this file + `ARCHITECTURE.md`).
> Changes vs the brief are logged in `DECISIONS.md`.

## 1. Problem & goal

A fictional **football scouting agency** promises clubs a scouting report on a
player **within 14 calendar days**. Reports are produced by full-time and
freelance scouts with different specialisms; some need a live match visit; a
video pre-screen tool can automate part of the desk work.

**The question the prototype answers:**

> As request volume grows 4x over the next year, what mix of salaried scouts,
> freelancers and automation keeps **≥95% of reports within 14 days** at the
> lowest cost — and how should work be assigned week to week?

Everything in the repo serves that question. The README opens with it and one
headline chart that answers it.

**How it answers it: a simulator (D-012).** The product is a scenario
simulator: set parameters (team, demand actually arriving, demand assumed when
hiring, automation, assignment policy, costs) → run → see results → change
parameters → run again → compare. The README headline is one curated sweep of
runs through the same engine.

**Success looks like:** a reader who never runs the code can read the README in
five minutes and come away with (a) the answer, with numbers, (b) how it was
reached, and (c) what was simplified. A user of the app can answer their own
"what if" question in a few clicks and minutes. `make all` works on a fresh clone.

## 2. Target users

- **The agency's head of operations (fictional persona).** Decides hiring,
  freelancer budget, whether to switch on the pre-screen tool, and which
  assignment policy to run. Uses the **simulator** to test "what if" questions
  before committing money. Wants a cost-vs-service trade-off, not a black box.
- **The real reader: a technical reviewer short on time.** Reads the README,
  maybe opens the simulator. Judges modelling judgement, honesty about
  limitations, and code quality.

## 3. Guardrails (non-negotiable)

- **Football scouting only.** No reference to any other industry, real company,
  or real-world institution anywhere — code, comments, docstrings, commit
  messages, README, file names. Enforced by an automated check (see
  `ARCHITECTURE.md` §8).
- **All data is synthetic.** Invented clubs, leagues, players and scouts only.
- **Small and finished beats ambitious and half-built.** Budget: two weekends.
- Licence: MIT.

## 4. MVP scope

In:
1. Synthetic world: scouts, fixtures, request history + future (M1).
2. Monthly demand forecast with an honest backtest; conversion to hours (M2).
3. Capacity plan with hiring triggers and a hiring table (M2).
4. Three assignment policies behind one interface: FCFS, EDF, CP-SAT (M3).
5. Day-by-day simulation over 12 months with seeds; `run_pipeline` + run
   folders; CLI runs and sweeps (M4).
6. Simulator app: parameter form, background worker + queue, runs list, run
   detail, compare, sweep results (M5).
7. Headline sweep, README with exported charts (M6).

Stretch (only after M6): LLM parsing of free-text club requests, with an
accuracy vs manual-triage trade-off (see §7).

**Explicitly out of scope:** user accounts, databases, deployment (no homelab
service for this one), multiple users or concurrent
runs, deep learning, real-world data, a separate web frontend (Streamlit only).

**Cut line if time runs short** (cut from the bottom first):
1. Stretch — AI intake.
2. `bundle` assignment unit; cancel button; clone-run button.
3. Compare page charts (keep the parameter diff + metric table).
4. Headline sweep breadth (keep policy × growth × automation on/off).
Never cut: the backtest vs baseline, the constraint tests, the run button +
worker, the README.

## 5. User stories & acceptance criteria

Acceptance criteria are the tester's checklist per milestone.

### M0 — Scaffolding
*As a contributor, I want a reproducible environment so that anyone can run the
project from a fresh clone.*
- [ ] `make setup && make test && make lint` pass on a fresh clone.
- [ ] `config/default.yaml` loads into typed pydantic models; invalid values fail loudly.
- [ ] Domain dataclasses exist for Scout, Request, Task, Fixture.
- [ ] Guardrail check runs in `make test` (skips cleanly if the private term list is absent).

### M1 — Synthetic world
*As the ops lead, I want a realistic synthetic history so that forecasts and
plans have something credible to learn from.*
- [ ] `make data` writes scouts, fixtures and requests to `data/runs/dev/raw/*.parquet`.
- [ ] Same seed → byte-identical outputs (determinism test).
- [ ] Scout mix ~60/40 full-time/freelance; every skill type covered by ≥2 scouts.
- [ ] Monthly volume peaks in transfer windows (Jan; Jun–Aug).
- [ ] **Calibration:** at the start of the plan year the current team's load is
      ~65–75% of usable hours (see D-003) — tested, not eyeballed.
- [ ] Some live-view requests have no feasible fixture before their due date
      and are flagged "at risk from day one".

### M2 — Forecast and capacity plan
*As the ops lead, I want a demand forecast and a hiring plan so that I know
what to hire, what kind, and when to act.*
- [ ] `make forecast plan` writes the forecast, capacity plan and hiring table.
- [ ] Backtest summary is printed: model vs seasonal-naive, WAPE per skill and aggregate.
- [ ] If the model does not beat the baseline on aggregate, the README says so.
- [ ] Tests cover the requests→hours conversion and each hiring-trigger rule
      (persistent gap → full-time by m−3; peak-only gap → freelance by m−1).

### M3 — Assignment engine
*As the ops lead, I want an assignment policy that respects every real-world
constraint so that I can trust its output.*
- [ ] FCFS, EDF and CP-SAT share one interface (see `DATA_CONTRACTS.md`).
- [ ] Hand-built unit tests prove each hard constraint for **every** policy:
      skill match, hours cap, conflict of interest, live-view date + region, precedence.
- [ ] On a constructed tight-deadline case, the optimiser beats EDF on on-time count.
- [ ] Each optimiser solve respects its time limit.
- [ ] Started work is never reassigned; FCFS/EDF never reassign anything (D-010).
- [ ] Assignment cadence and unit are parameters; `daily` + `task` works end to end.

### M4 — Simulation, run pipeline and sweeps
*As the ops lead, I want one call that turns a set of parameters into a full
simulated year so that every "what if" is just a different parameter file.*
- [ ] `run_pipeline(params)` runs generate → forecast → capacity plan →
      simulate and writes a complete run folder (`DATA_CONTRACTS.md`).
- [ ] `make run` (optionally `PARAMS=…`) produces a run folder from the CLI.
- [ ] Same `params.yaml` → identical results (determinism, per seed).
- [ ] Actual and assumed growth are independent: assumed drives the forecast
      and hiring plan, actual drives arrivals (D-014); hires join the team when due.
- [ ] Changing only the policy leaves the request stream identical (D-015).
- [ ] `make sweep SWEEP=quick` expands a sweep file into runs (minutes).
- [ ] Sanity checks hold: more capacity never lowers mean on-time rate;
      automation with 0% rework beats automation off.
- [ ] Metrics: on-time rate, mean & P90 turnaround, utilisation by employment
      type, cost split (salaried / freelance / late penalty), backlog over time.

### M5 — Simulator app
*As the ops lead, I want to set parameters, launch a run, and come back to
the results, so that I can explore trade-offs myself without touching code.*
- [ ] `make app` starts the dashboard **and** the background worker; quitting
      the app stops the worker.
- [ ] New run: basic-parameter form + advanced YAML; invalid input is
      rejected with a clear message **before** queueing.
- [ ] Clicking Run returns immediately; the run shows as queued → running
      (with progress) → done/failed without a manual page reload.
- [ ] Several runs can be queued; they execute one at a time, oldest first.
- [ ] Closing the browser tab does not stop a run.
- [ ] Killing the worker mid-run leaves that run `failed (interrupted)` on
      restart, never stuck at `running`.
- [ ] Run detail shows on-time vs 95% target, backlog, cost split, demand
      actual vs forecast, capacity heatmap + hiring table, and the parameters.
- [ ] Compare shows the parameter diff and metrics of 2–4 runs side by side.
- [ ] Worker queue logic is unit-tested without Streamlit (functional core).

### M6 — Headline sweep and README
*As a reviewer, I want a README that answers the question by itself.*
- [ ] Headline sweep run once; summary committed to `data/published/`.
- [ ] Sweep results page and README headline chart: on-time vs cost, one dot
      per run, 95% line, cheapest passing run highlighted.
- [ ] Cadence side experiment result (weekly vs daily) in the README.
- [ ] `make all` then `make app` works from a fresh clone.
- [ ] README follows the outline in §6 and stands on its own.

## 6. README outline (deliverable spec)

1. **The question** (one sentence) and the **headline chart**.
2. **Key results:** 3–5 bullets with numbers.
3. **How it works:** the five stages and the simulator (worker + run folders),
   one short paragraph each, with the diagram.
4. **Design decisions and trade-offs:** why CP-SAT over a heuristic, why the
   daily assignment cadence, why P80 for capacity, actual vs assumed demand,
   why a background worker, how automation reliability is
   modelled, what was deliberately simplified.
5. **Limitations and what production would need:** real data contracts, scout
   preferences and fairness, uncertainty in task durations, re-assigning work
   mid-week, integration with hiring systems, monitoring forecast drift.
6. **Run it:** `make setup && make all && make app`.
7. **Project structure.**

Tone: lead with results, short paragraphs, no marketing language.

## 7. Stretch — AI request intake (after M6 only)

Clubs send free-text requests (e.g. *"Need eyes on a left-back at Puerto Azul,
Spanish speaker, live viewing if possible, ideally before the window opens"*).
An LLM call parses them into `position_group`, `region`, `language`,
`needs_live_view`, `urgency`, plus a confidence score.
- ~50 synthetic labelled requests generated from templates with known truth.
- Report field-level accuracy, confidence calibration, and accuracy vs
  manual-triage workload at several confidence thresholds.
- Anthropic Python SDK behind `ANTHROPIC_API_KEY`; if absent, skip cleanly and
  say so. Nothing else in the pipeline may depend on it.
- Done when `make intake-eval` prints the tables and the README has a short section.

## 8. Open questions

Tracked in `NEXT_STEPS.md` → "Open decisions".
