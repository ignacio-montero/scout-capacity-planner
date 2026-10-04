# Overnight report — autonomous session (started 2026-10-03)

> Written for the user to read on return. What got built, every decision taken
> without you (with the alternative and how to reverse it), what's open, and
> what each piece taught. Decisions that change the design are also logged in
> `DECISIONS.md`.

## Ground rules followed
- Markdown docs are **still not committed** (your earlier instruction); code,
  config and tests are committed and pushed to `main` at milestone checkpoints.
- Nothing irreversible: no force-pushes, no history rewrites, repo stays
  private, no deletions outside generated data.
- Specialist subagents did the implementation; the orchestrator integrated,
  reviewed, ran tests + guardrail scan before every commit.

## The answer (from the final sweeps, 3 simulated years per setting)
At 4x growth, keeping ≥95% of reports on time:
- **With the pre-screen tool on, ~2.7M a year.** Cheapest point: optimiser +
  freelancers-only hiring + P90 (2,678k, 95.9%). It's fragile, though: most
  freelancers-only setups fail badly. The robust choice is the planned
  (rule-based) hiring mix with the tool: optimiser 2,735k (97.0%) or EDF +
  due-date check 2,765k (96.1%).
- **Without the tool:** EDF + due-date check, planned mix, P80, at 3,232k
  (96.1%), about 18% more.
- **Assignment:** EDF with a due-date check (`edf_feasible`, now the default)
  beats the CP-SAT optimiser with the tool off; the optimiser is ahead by a
  small margin (~1–3%) with the tool on, at higher late penalties, and in
  overloaded worlds. It takes minutes per run vs seconds. Plain EDF and FCFS
  fail once the tool shrinks the plan.
- **Forecast error is expensive:** hiring for 2x when 4x arrives gives 53–65% on
  time and costs +1.9M. **Daily vs weekly assignment:** +28 pts on time.
- All numbers: `docs/img/key_numbers.md`; charts: `docs/img/*.png`.

## Progress log
| Time | Milestone / step | Result | Commit |
|---|---|---|---|
| Phase A | M0 scaffolding (backend-developer): config, random streams, domain model, guardrail test + hook; 113 tests green | ✅ | `ecc33a4` |
| Phase A | App UX spec (designer) → `docs/DESIGN_SYSTEM.md` | ✅ (doc, uncommitted) | — |
| Phase B | M3 assignment engine (backend-developer): FCFS, EDF, CP-SAT + shared eligibility rules and validator; optimiser beats EDF on a constructed tight case; 400×60 solve in 0.77 s | ✅ | `820fe39` |
| Phase B | M1 synthetic world (backend-developer): 40 scouts, 16k fixtures, 15k requests in 0.1 s; calibrated load 0.71; thinning keeps growth worlds nested | ✅ | `c12ddb6` |
| Phase C | Run store + background worker + `make app` launcher (backend-developer): child process per run, crash recovery, cancel, heartbeat, single-worker lock; 118 tests incl. real-process tests | ✅ | `c6e1c9e` |
| Phase C | Critic red-team of M0+M3 → 2 blockers + 8 should-fixes, all fixed (M3 + M0 agents resumed) | ✅ | `00c8c9a` |
| Phase C | M2 forecast + capacity plan (backend-developer): ETS beats naive 10.2% vs 13.9% WAPE; plan-year forecast +2.5% vs realised; 44 FT + 50 FL hires at 4x | ✅ | `2c10cf4` |
| (pause) | API usage limit hit mid-phase; 3 agents interrupted and resumed after reset, no work lost | — | — |
| Phase D | Simulator app UI (frontend-developer): 5 pages on demo data, 154 tests incl. Streamlit AppTest; visually checked in the browser | ✅ | `941094b` |
| Phase D | Second critic red-team (generator, forecast, plan, worker): 2 blockers + 8 should-fixes | ✅ review | — |
| Phase D | Worker robustness fixes (poison-pill skip, delete races, standby worker, CLI lease) | ✅ | `b8e1c7b` |
| Phase D | Forecast + plan fixes: per-skill P80 coverage 0.55 → 0.77; January bias −12.8% → −3.7%; 0.5x growth hires 2 (was 14); max-flow coverage | ✅ | `60a2846` |
| (pause) | Second API usage limit; M4 agent interrupted and resumed after reset | — | — |
| Phase D | M4 simulation + pipeline + sweeps (backend-developer): invariant-tested engine; EDF 97.1% on time at 4x with the plan; optimiser 89.0% (loses: myopic deferral); FCFS ≡ EDF found | ✅ | `6325352` |
| Phase E | Generator follow-ups (per-scout streams, wider thinning, express requests without live views, January load) + app fixes (censoring-correct charts, diagnostics, polish, real-run page test) | ✅ | `c457288` |
| Phase E | Acceptance-criteria review (tester): traceability table M0–M5, fresh-clone + end-to-end + 21 boundary cases; found month-0 hire-count bug (fixed) | ✅ | `c08ea25` |
| Phase E | Optimiser calibration (premium pricing + commit horizon + queue-aware urgency): 95.5% vs EDF 95.1% at 4x; tight case 46.0% vs 40.8% and −196k | ✅ | `54c686c` |
| Phase F | Sweep links, hire-mix lever, README sweeps | ✅ | `a37894b` |
| Phase F | Ran 5 sweeps (78 runs × 3 seeds, ~1 h): headline answer = optimiser + rule-based hire mix + P80 + pre-screen tool → 95.7% on time at 2,871k (cheapest without tool: EDF, 3,266k) | ✅ | `2042625` |
| (pause) | Third and fourth API usage limits; agents resumed after reset | — | — |
| Phase F | Final red-team (simulation + optimiser): rework double-count, weak EDF baseline, silent fallbacks/partial publishes | ✅ review | — |
| Phase F | Fixes: one rework model + cross-module test; new `edf_feasible` policy (now default — it beats the optimiser); honest publishing | ✅ | `429bc95` |
| Phase F | README chart export (7–8 charts + key numbers, computed titles, honest near-target precision) | ✅ | `6e79dfe` |
| Phase G | Re-ran all 6 sweeps on one code version (headline 72 runs, 3 seeds) and regenerated charts | ✅ | `7f4b49d`, `f7c26e3` |
| Phase G | README written (technical-writer), numbers checked against `key_numbers.md`; docs corrected where I had overstated the EDF-vs-optimiser result | ✅ (uncommitted .md) | — |

## Decisions taken without you
(numbered O-1, O-2, … — each with: what, why, alternative, how to reverse)

**O-1 — Conflict of interest covers the client club too.** A scout can't
work a request if they used to be at the player's club *or* the client club
(the brief only said the player's club). Why: an ex-employee writing for their
old club is the same bias in the other direction. Reverse: one line in
`Request.conflict_clubs` (domain.py).

**O-2 — Ranges the parameter spec left open** were fixed in code: task hours
0–40, live-view hours 0–24, weights/penalty/licence ≥ 0, salary > 0, seed ≥ 0,
team needs ≥ 1 scout. Reverse: edit `config.py`.

**O-3 — Design spec accepted with its proposals:** Sweep results is the app's
landing page (answer first, then "clone and change one thing"); the tool's
licence cost is a basic form field (otherwise automation looks free); data
contracts gained team-size columns, an automation cost column, a worker
heartbeat file, cancel-while-queued handling, and flat summary column names.
Reverse: edit `DESIGN_SYSTEM.md` / `DATA_CONTRACTS.md`.

**O-4 — Assignment engine design choices** (D-016, D-017): greedy picks
salaried-first then least-loaded; the optimiser is single-threaded with a
deterministic time budget so results are reproducible; churn penalises
dropping committed work as well as moving it. Reverse: config weights or the
named functions in `assign/`.

**O-5 — Demand is calibrated against the default team, not the run's team**
(D-018). The first cut sized demand so *every* run's team started at 70% load,
which made "what if we add 10 freelancers?" pointless: demand grew to match.
Now 10 more freelancers means the same requests and a lower load (0.71 → 0.62).
Reverse: `base_monthly_volume` in `generate.py`.

**O-6 — Plan-year forecast = learned level & seasonality × assumed growth**
(D-020): the growth assumption replaces the statistical trend instead of being
stacked on it (which would double-count history's organic growth).

**O-7 — Red-team fixes to the assignment engine** (D-021). A critic found the
optimiser's "beats EDF" test counted assignments, not on-time completions
(capacity was a weekly total with no notion of *when* work finishes); that it
could let a live-view fixture lapse rather than pay a freelancer; and that the
late penalty didn't influence it. Fixes: due-date-aware capacity, perishable
fixtures, lateness = multiplier on the late penalty (per request), greedy may
re-place work forced off a scout, plus config/guardrail hardening. Reverse:
individual commits after `c6e1c9e`.

**O-8 — App listens on localhost only, and follows the worker's runs folder.**
Streamlit by default listens on every network interface (visible on your LAN);
this is a local tool, so `.streamlit/config.toml` binds it to `localhost`.
`serve.py` passes its runs folder to the app so the two can never disagree.
Reverse: delete the `[server]` block.

**O-9 — Second red-team (generator, forecast, plan, worker) → fixes** (D-023).
Biggest: the hiring plan invented skill gaps (fixed split of multi-skilled
scouts' hours; now a max-flow allocation), "P80" was really ~P56 (now
calibrated per skill, ~0.77 coverage), and adding scouts reshuffled the
existing team (being fixed). Decision taken without you: keep `start_load`
meaning "average month" (so January starts ~100% loaded, realistic for a
transfer window) and label it clearly, rather than recalibrating to January.

**O-10 — Daily assignment runs include weekends** (D-024): otherwise weekend
matches were systematically missed (934 vs 27 re-targeted live views).

**O-11 — 15% urgent (7-day) requests** (D-025): with one turnaround for
everyone, FCFS and EDF were literally the same policy. Reverse:
`demand.urgent_share: 0`.

**O-12 — Optimiser fixed and re-measured** (D-027). Root cause: it had no
reason to start work early. Fix: freelance premium pricing + "plan long,
commit short" + queue-aware urgency. Now 95.5% vs EDF 95.1% at 4x (+0.9% cost)
and clearly better when capacity is tight (46.0% vs 40.8%, −196k).

**O-13 — Headline sweep design** (D-028, closes Q6): 4x growth; policy × hire
mix (new lever: rule / freelance-only / full-time-only) × P50/P80/P90 × tool
on/off; 3 seeds. Plus growth, forecast-error and no-hiring sweeps.

**O-14 — `edf_feasible` is the default policy** (D-029, D-030). A final review
showed the optimiser's edge came from a weak EDF baseline. EDF with a
due-date check beats the optimiser at 4x with the tool off (96.1% vs 95.5%,
cheaper) and when capacity is tight (50.5% vs 46.2%), in seconds instead of
minutes; with the tool on the optimiser keeps a small edge. Also fixed: the
simulation charged failed pre-screens twice (now matches the plan, pinned by
a cross-module test). Reverse: `assignment.policy: optimiser`.

## Open questions for you
- **`make` doesn't run on this Mac yet.** `/usr/bin/make` is the Xcode stub and
  stops at the licence prompt (like the system python). Either run
  `sudo xcodebuild -license` once (your call — it's accepting Apple's terms),
  or `brew install make` and use `gmake`. Until then every Makefile recipe was
  verified by running its commands directly.
- Q5 — the ⚑ defaults in `PARAMETERS.md` (still the drafts; used as-is).

## What this taught (merged from all subagents)

The full explanations, with file:function pointers, live in the study guide:
`~/Development/docs/learning-guide/12-scout-capacity-planner.md` (written
tonight) and ~45 new glossary entries in `00-glossary.md`. The ones that
mattered most:

**Engineering patterns**
- **Functional core, imperative shell:** pure stages (`generate`, `forecast`,
  `plan`, `assign`, `simulate`) with I/O only in `runs.py`, the CLI and the app.
  It made ~1,000 tests cheap, and let the CLI and the background worker share
  one `run_pipeline`.
- **Job queue as folders + supervisor/child process:** atomic writes
  (temp + `os.replace`), a state machine as data, heartbeat (liveness) vs
  `flock` (exclusivity), poison-pill skipping. See `runs.py`, `worker.py`.
- **Strategy pattern + registry:** four assignment policies behind one
  `assign()` signature (`assign/__init__.py`).
- **Parse, don't validate:** strict pydantic parameters with cross-field rules
  and schema versioning (`config.py`).

**Modelling & simulation**
- **Common random numbers:** named streams per purpose *and per entity*, plus
  thinning, so two runs differ only by the lever you changed (`rng.py`,
  `generate.py`).
- **Calibrate quantiles empirically:** "P80" is a testable claim. It was really
  P56 until the variance was built from its sources (`forecast.py`).
- **Transportation problem / max-flow** to find real (not phantom) capacity
  gaps (`plan.allocate`).
- **Constraint programming:** pre-filtering, the EDF feasibility condition,
  deterministic vs wall-clock limits, warm starts (`assign/optimiser.py`).
- **Myopic rolling horizons:** plan long, commit short; perishable options
  (match dates) can't be deferred.

**Lessons about checking your own work (the big ones)**
- **Proxy objectives:** a test that counted *assignments* instead of
  *on-time completions* made the optimiser look better than it was.
- **Baseline strength decides the result:** a 20-line due-date check made
  EDF beat CP-SAT. Always try the cheapest smarter heuristic first.
- **Two sources of truth for one assumption** (rework hours in plan vs
  simulation): only a cross-module consistency test catches it.
- **Silent degradation and silent omission:** fallbacks and partial sweeps
  must be counted and published, not hidden.
- **Rounding is a claim:** 94.97% shown as "95.0%" next to a 95% target says
  "passed" when one simulated year didn't.
