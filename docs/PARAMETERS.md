# PARAMETERS — Scout Capacity Planner

> The simulator's inputs. **Draft for review**: defaults marked ⚑ are business
> judgement calls the user should own; the rest are modelling defaults.
> This table is the spec for `config.py` (pydantic models) and
> `config/default.yaml`; the app's form is built from the **Basic** rows, and
> everything else is reachable through the advanced YAML editor.
> Cost unit: one invented currency (think €).

## Team (initial)
| Key | Default | Range | Basic | Meaning |
|---|---|---|---|---|
| `team.full_time_count` | 24 | 0–100 | ✓ | Salaried scouts at the start of the year |
| `team.freelance_count` | 16 | 0–100 | ✓ | Freelancers in the pool at the start |
| `team.freelance_weekly_hours` | [8, 25] | 0–40 | | Weekly hours a freelancer offers, drawn each week |
| `team.skills_per_scout` | [1, 4] | 1–6 | | How many skill types each scout covers |
| `team.leave_days_per_year` | 25 | 0–60 | | Days off per full-time scout |
| `team.follow_hiring_plan` | true | bool | ✓ | Do hires from the capacity plan actually join during the year? |

## Demand (what actually arrives)
| Key | Default | Range | Basic | Meaning |
|---|---|---|---|---|
| `demand.start_load` | 0.70 | 0.3–1.2 | ✓ ⚑ | Average-month (deseasonalised) load of the reference team at month 0; sets base volume (D-003, D-018). January peaks ~1.35× this. A different team starts at a different load |
| `demand.actual_growth` | 4.0 | 0.5–6 | ✓ | Run-rate at month 12 ÷ run-rate at month 0 (D-014) |
| `demand.live_view_share` | 0.40 | 0–1 | ✓ | Share of **non-urgent** requests needing a live view; effective share = share × (1 − urgent_share) ≈ 0.34 |
| `demand.seasonality_strength` | 1.0 | 0–2 | | 0 = flat year, 1 = default transfer-window peaks |
| `demand.history_months` | 48 | 36–72 | | Months of history for the forecast (D-004) |
| `demand.history_growth_per_year` | 0.15 | −0.5–1.0 | | Organic growth in the history (before the simulated year) |
| `demand.turnaround_days` | 14 | 7–28 | | Promised delivery time |
| `demand.urgent_share` | 0.15 | 0–1 | | Share of express requests (D-025); express requests never need a live view (D-026) |
| `demand.urgent_turnaround_days` | 7 | 3–28 | | Express delivery time; must be ≤ `turnaround_days` |
| `demand.desk_hours` | [4, 10] | | | Desk review duration range |
| `demand.writeup_hours` | [2, 4] | | | Write-up duration range |
| `demand.live_view_hours` | 8 | | | A live view costs the scout this many hours that day |

## Capacity plan (what the agency believes and hires for)
| Key | Default | Range | Basic | Meaning |
|---|---|---|---|---|
| `capacity_plan.assumed_growth` | 4.0 | 0.5–6 | ✓ | Growth the agency hires for; ≠ actual = forecast error (D-014) |
| `capacity_plan.quantile` | 0.80 | 0.5–0.95 | ✓ ⚑ | How cautious the hiring plan is (P50 / P80 / P90) |
| `capacity_plan.target_utilisation` | 0.80 | 0.5–1.0 | ✓ ⚑ | Max planned load per scout before hiring |
| `capacity_plan.lead_time_full_time_months` | 3 | 0–6 | | Recruitment time, full-time |
| `capacity_plan.lead_time_freelance_months` | 1 | 0–3 | | Onboarding time, freelancer |
| `capacity_plan.persistent_gap_months` | 3 | 1–12 | | Consecutive gap months that justify a full-time hire |
| `capacity_plan.hire_mix` | rule | rule / freelance_only / full_time_only | | Which contract types the hiring plan may use (D-028) |

## Automation (video pre-screen)
| Key | Default | Range | Basic | Meaning |
|---|---|---|---|---|
| `automation.enabled` | false | bool | ✓ | Tool on or off |
| `automation.desk_reduction` | 0.40 | 0–0.9 | ✓ | Share of desk hours saved when it works |
| `automation.rework_rate` | 0.15 | 0–0.6 | ✓ | Chance its output is unusable |
| `automation.rework_overhead_hours` | 1.0 | 0–4 | | Extra hours lost when it fails |
| `automation.monthly_cost` | 1500 | 0– | ✓ ⚑ | Licence cost of the tool (not in the brief; without it the tool is free) |

## Assignment
| Key | Default | Range | Basic | Meaning |
|---|---|---|---|---|
| `assignment.policy` | edf_feasible | fcfs / edf / edf_feasible / optimiser | ✓ | Who-does-what rule; `edf_feasible` = EDF that only promises work a scout can finish by its due date (D-029, D-030) |
| `assignment.cadence` | daily | daily / weekly | ✓ | How often assignment runs: daily = every calendar day; weekly = Mondays (D-010, D-024) |
| `assignment.unit` | task | task / bundle | ✓ | Assign tasks one by one, or desk + write-up together |
| `assignment.horizon_days` | 7 | 1–14 | | Rolling window each run looks ahead |
| `assignment.weights.lateness` | 1.0 | >0–10 | | Optimiser: multiplier on `cost.late_penalty` for leaving urgent work unassigned (D-021) |
| `assignment.weights.cost` | 1 | | | Optimiser: per cost unit of freelance hours |
| `assignment.weights.continuity` | 5 | | | Optimiser: per extra scout on one request |
| `assignment.weights.churn` | 3 | | | Optimiser: per reassigned not-started item |
| `assignment.time_limit_s` | 1.0 | 0.1–10 | | Max seconds per optimiser solve |
| `assignment.cost_basis` | premium | full / premium | | Freelance hours priced at full rate or at the premium over salaried (D-027) |
| `assignment.commit_buffer_days` | 2 | int ≥ 0 or null | | Commit only work finishable within N days after the next run; null = whole window (D-027) |
| `assignment.load_aware` | true | bool | | Urgency accounts for the queue of earlier-due same-skill work (D-027) |

## Costs
| Key | Default | Range | Basic | Meaning |
|---|---|---|---|---|
| `cost.full_time_monthly_salary` | 4500 | | ✓ ⚑ | Fully loaded monthly cost per salaried scout (≈28/h) |
| `cost.freelance_premium` | 1.5 | 1–3 | ✓ ⚑ | Freelance hourly rate ÷ salaried hourly equivalent (≈42/h) |
| `cost.late_penalty` | 1000 | 0– | ✓ ⚑ | Cost per late report (refunds + churn). ~2.5× the labour cost of a report (~13 h ≈ 400). **The single most answer-shaping number.** |

## Simulation
| Key | Default | Range | Basic | Meaning |
|---|---|---|---|---|
| `sim.seeds` | 3 | 1–5 | ✓ | Replications with different luck; more = slower, tighter ranges |
| `sim.seed` | 42 | | | Base seed; all random streams derive from it (D-015) |
| `sim.months` | 12 | 1–24 | | Simulated horizon |
| `sim.target_on_time` | 0.95 | 0.5–1 | ✓ | Service target used for pass/fail |

## Validation rules across groups
- `schema_version` (int, currently 1) is the first key of every params file;
  older files are migrated on load (`config.migrate`), newer ones rejected.
- Every single work item must fit one full-timer's horizon:
  `max(desk max, write-up max, live_view_hours) ≤ 37.5 × horizon_days / 7`
  (and desk max + write-up max in `bundle` mode).
- Duplicate YAML keys are rejected; dotted overrides deep-merge into groups.

## Not parameters (fixed in code, on purpose)
Skill-type list, region names, club and scout names, and the fixture
structure. They define the synthetic world's *shape* rather than a decision;
exposing them would multiply the input surface for no insight.
