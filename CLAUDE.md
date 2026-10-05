# CLAUDE.md — scout-capacity-planner

Handoff doc for Claude sessions. Concise orientation; details live in `docs/`.

## What this is
A **scenario simulator** for a fictional **football scouting agency** that
promises reports within 14 days. As demand grows 4x in a year, what mix of
salaried scouts, freelancers and automation keeps ≥95% on time at lowest
cost, and how should work be assigned? You set parameters in a Streamlit app,
click Run, a background worker simulates the year, you compare runs.
Engine: generate → forecast → capacity plan → assign (daily) → simulate.
Python 3.12, uv, pandas, statsmodels, OR-Tools CP-SAT, plotly, streamlit.

## Read first
- `docs/PRD.md`: goal, scope, milestones M0–M6 with acceptance criteria.
- `docs/ARCHITECTURE.md`: engine, simulator app + worker, layout, modelling.
- `docs/DATA_CONTRACTS.md`: run folder, `status.json`, parquet schemas, `assign()`.
- `docs/PARAMETERS.md`: every simulator input with defaults (spec for `config.py`).
- `docs/DECISIONS.md`: every deviation from the original brief (D-001…D-030).
- `docs/OPTIMISATION_CASE_STUDY.md`: optimised assignment vs FCFS (business reader).

## Guardrails (hard rules)
- **Football scouting only.** No reference to any other industry, real
  company or real-world institution anywhere: code, comments, docstrings,
  commit messages, README, file names. The banned vocabulary is listed in
  `.private/banned-terms.txt` (gitignored); the guardrail test and commit-msg
  hook enforce it (D-002). Check commit messages against it.
- **Synthetic data only**: invented clubs, leagues, players, scouts.
- Small and finished over ambitious. **Stop after each milestone** and summarise.
- Parameter-driven (`PARAMETERS.md` → `config/default.yaml`), deterministic
  (one seeded stream per purpose, D-015), functional core: stages are pure;
  I/O only in `runs.py`, CLI, app.
- Tests for every constraint and every rule a reviewer could question.

## Vocabulary (D-011)
"Capacity plan" = monthly headcount/hiring decision (stage 3). "Assignment
run" = daily who-does-which-task decision (stage 4). "Run" = one full
simulation for one parameter set (= one folder in `data/runs/`). "Sweep" =
many runs from a YAML file.

## Current state (2026-10-03, overnight autonomous session)
- Done + committed: M0–M5 (generator, forecast + capacity plan, 4 assignment
  policies, day-by-day simulation, run pipeline, sweeps, run store + worker,
  Streamlit app), README chart export, published sweep summaries. Three red-team
  rounds (D-021, D-023, D-029). Default policy is `edf_feasible` (D-030).
  M6 done: sweeps re-run, charts exported, README + case study written and
  committed.
- Demo the UI: `uv run python scripts/make_demo_runs.py --clean` then
  `SCOUT_RUNS_ROOT=data/runs_demo SCOUT_PUBLISHED_DIR=data/published_demo uv run streamlit run app/main.py`.
- GitHub: `ignacio-montero/scout-capacity-planner` (private), branch `main`.
- Docs are committed (2026-10-04, on the user's go-ahead). Keep them current
  and commit doc updates with the related code.
- `.private/` holds the original brief (superseded; do not implement from it,
  it describes a weekly, precomputed-grid design that was replaced).
- Not deployed anywhere; deployment is out of scope (no homelab service).

## Run it (from M0 on)
`make setup && make all && make app`. `make test` = fast suite (~1 min);
`make test-all` adds slow end-to-end tests (~7 min). `make run PARAMS=…`,
`make sweep SWEEP=quick`, `make charts` (needs Chrome; uses `BROWSER_PATH` or
the puppeteer Chrome for Testing in `~/.cache/puppeteer`).

## Gotchas
- Xcode licence accepted (2026-10-04): `make` works. Still use `uv run`, never
  the system python, for project code.
- Streamlit view scripts live in `app/views/`, NOT `app/pages/` (a `pages/`
  folder triggers Streamlit's legacy auto-discovery → "Page not found" on deep
  links). Streamlit runs headless (first-run email prompt kills `make app`).
  In AppTest, call `run()` once before `switch_page()`.
- kaleido ≥1 needs a local Chrome to export PNGs (matters for README charts).
- `/usr/bin/python3` on this Mac triggers the Xcode licence prompt; always go
  through `uv run` / the uv-managed 3.12, never the system python.
- Assignment runs **daily on a rolling 7-day window** with started work frozen
  (D-010), not weekly as the original brief says.
- The app never computes a run; it only writes a queued run folder. The worker
  executes (D-013). Don't put simulation calls in Streamlit code. A test
  enforces that app code never imports pipeline stages.
- `assign()` requires keyword `cost=params.cost`; use `assign_with_report` for
  solver diagnostics.
- Streamlit AppTest paths must be absolute or relative to the test file.
- The user wants to steer decisions with full understanding: explain in plain
  language before acting on significant choices; tutor mode is on.
