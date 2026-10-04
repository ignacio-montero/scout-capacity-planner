# DESIGN SYSTEM — Scout Capacity Planner (simulator app + README charts)

> Owner: Designer. Grounded in `PRD.md` (M5, M6), `ARCHITECTURE.md` §6,
> `DATA_CONTRACTS.md` §0/§5/§6 and `PARAMETERS.md`. This is the spec the
> frontend builds against: what each page shows, in what order, with which
> Streamlit component, and what it says when there is nothing, something
> loading, or something broken. Status: **draft for M5**.
>
> Scope guard: native Streamlit + plotly only, **no custom CSS in v1**. If a
> line in this doc would need CSS or a custom component, it is a bug in this
> doc. Requires a recent Streamlit (uses `st.navigation`, `st.fragment(run_every=…)`,
> `st.badge`, `st.popover`, dataframe and plotly selection events).

---

## 0. Design principles (the five rules every screen follows)

1. **Lead with the answer.** Every results view opens with one plain sentence
   that states the takeaway ("On time 93.4%, 1.6 pts below target"), then the
   numbers, then the charts. Pattern: *inverted pyramid* / *action titles*.
2. **Overview first, details on demand** (*Shneiderman's mantra*). Sweep
   results (many runs) → Run detail (one run) → Parameters / per-repeat table
   (raw detail). Each step is one click deeper.
3. **Always show the system's state** (*Nielsen heuristic #1, visibility of
   system status*). A run takes 10 s to 6 min in another process; the user
   must always see whether it is waiting, working, done or broken, on every
   page (sidebar queue indicator, §1.3).
4. **One meaning per colour, everywhere.** Blue always means "optimiser",
   orange always means "full-time". Colour is a data variable, not decoration
   (§2.2). Never rely on colour alone (*redundant encoding*): every colour has
   a shape, dash or text label alongside it.
5. **Prevent errors instead of reporting them** (*error prevention*). Widgets
   are bounded to valid ranges; the only free text (advanced YAML, run name) is
   validated live, before the Run button can be pressed.

---

## 1. Information architecture and navigation

### 1.1 Pages

| # | Nav label | Purpose (one line) | URL path | Query params | Material icon |
|---|---|---|---|---|---|
| 1 | **Sweep results** (default page) | The headline answer: on-time vs cost, one dot per run | `/sweep` | `?sweep=<id>` | `:material/scatter_plot:` |
| 2 | **New run** | Set parameters, review, queue a run | `/new` | `?clone=<run_id>` | `:material/tune:` |
| 3 | **Runs** | Everything queued, running and finished; actions | `/runs` | `?new=<run_id>` (just-queued banner) | `:material/list:` |
| 4 | **Run detail** | One run's results, or its live progress | `/run` | `?run=<run_id>` | `:material/insights:` |
| 5 | **Compare** | 2–4 runs side by side | `/compare` | `?runs=<id>,<id>[,…]` | `:material/compare_arrows:` |

Built with `st.navigation([...], position="sidebar")` and one `st.Page` per
file in `app/`. Flat list in the order above (no nav sections: five pages do
not need grouping). Sidebar header: app name "Scout Capacity Planner" + caption
"Fictional agency · synthetic data".

**Why Sweep results is the landing page.** A first-time user faced with a
blank form of 20 settings does not know which ones matter (the *blank-slate
problem*). Landing on the answer shows what the tool is for, and every dot is a
door into "clone these settings and change one thing" (*start from an
example*). Discarded: landing on New run (fast for experts, disorienting for
everyone else) and landing on Runs (empty on a fresh clone, so the first
impression would be an empty state).

> **Concept: URL as state (deep linking).** Putting the selected run in the
> URL (`/run?run=20261003-101500-hire-for-2x`) instead of only in memory means
> a browser reload, the back button, or a bookmark all land on the same view.
> In Streamlit, read/write it with `st.query_params`; `st.session_state` is
> lost on reload and is per browser tab.

### 1.2 Moves between pages

All cross-page moves go through one helper, `go_to(page, **query)` (§8): it
stores the query in `st.session_state["nav"]`, calls `st.switch_page`, and the
target page copies it into `st.query_params` on load.

| From | Trigger | To | What the user sees on arrival |
|---|---|---|---|
| New run | **Run** clicked (valid) | Runs `?new=<id>` | Success banner "Queued 'Hire for 2x, get 4x'. It is #2 in line." + **Open run** button; the run appears in "In progress" |
| Runs | Select 1 row → **Open** | Run detail `?run=<id>` | Results, or live progress if still running |
| Runs | Select 1 row → **Clone** | New run `?clone=<id>` | Form prefilled with that run's parameters; info banner "Started from 'X'" |
| Runs | Select 2–4 rows → **Compare** | Compare `?runs=…` | Those runs preselected; first selected = baseline A |
| Run detail | **Clone to new run** | New run `?clone=<id>` | as above |
| Run detail | **Compare with…** | Compare `?runs=<id>` | This run preselected as A; picker focused for the rest |
| Run detail (failed) | **Run again** | Runs `?new=<new id>` | New run with identical parameters, name + " (again)" |
| Sweep results | Click a dot / select a table row | (stays) detail panel below chart | **Open run** (if the run folder exists locally) and **Clone settings** |
| Sweep results | **Clone settings** | New run `?clone=…` | Prefilled from that sweep row (default + sweep `fixed` + varied values) |
| Compare | Click a run's header card name | Run detail `?run=<id>` | |
| Any page | Sidebar queue indicator → "View" | Runs | |

### 1.3 Sidebar queue indicator (global, every page)

Below the nav, a small block inside `st.fragment(run_every="5s")`:

- Running: **"Running: Hire for 2x, get 4x"** + `st.progress(0.63)` + caption "seed 2/3 · week 31/52 · about 2 min left" + "2 more queued".
- Idle: caption "No runs in progress."
- Worker down (see §10, gap G3): `st.warning("Background worker is not running, so queued runs will not start. Restart with make app.")`.
- When a run finishes while the user is on any page: `st.toast("'Hire for 2x, get 4x' finished: 93.4% on time")` (or "failed", with the error's first line).

Why: the user should be able to start a run, wander to Compare, and still know
when it is done without going back to Runs. Discarded: a top-of-page banner on
every page (pushes content down, competes with page titles).

---

## 2. Visual language

### 2.1 Theme, type, spacing

- `st.set_page_config(layout="wide", page_title="Scout Capacity Planner")`. Wide
  layout because the form uses two columns and tables have 10+ columns.
- `.streamlit/config.toml`: set only `primaryColor = "#237A4B"` ("pitch
  green"). Leave `base` unset so the app follows the user's light/dark OS
  setting. Why not Streamlit's default red primary: red buttons and sliders
  read as "danger", and red is reserved here for "late" (§2.2). #237A4B passes
  ≥3:1 against both Streamlit backgrounds and carries white button text at
  5.2:1.
- **Typography:** Streamlit's default font, no overrides. Hierarchy uses only
  native elements: `st.title` (page), `st.subheader` (section), `**bold**`
  markdown (card titles), body text, `st.caption` (secondary info, units,
  ranges). Never more than these four levels on a page.
- **Spacing and grouping:** rely on Streamlit's built-in spacing; group related
  controls in `st.container(border=True)` (*Gestalt principle of common
  region*: things in one box read as one thing). `st.divider()` only between
  major page sections, at most two per page. `st.columns(..., gap="medium")`.
- **Icons:** Material icons via `:material/name:` only. No emoji anywhere in the
  UI or charts.

### 2.2 Colour tokens

> **Concept: design tokens.** A token is a named design value
> (`POLICY_COLORS["optimiser"] = "#0072B2"`) defined once and referenced
> everywhere, instead of hex codes scattered through the code. Changing the
> palette becomes a one-line edit, and the app and the README cannot drift
> apart. Here the tokens live as constants at the top of `charts.py`.

> **Concept: colour-blind-safe palette and contrast ratio.** About 1 in 12 men
> has a colour-vision deficiency, mostly red–green. The *Okabe–Ito* palette was
> designed so its colours stay distinguishable under the common types. The
> *contrast ratio* (WCAG) measures how much a colour stands out from its
> background, from 1:1 to 21:1; lines and markers should reach **3:1**, text
> **4.5:1**. Our charts must pass on both Streamlit backgrounds: light
> `#FFFFFF` and dark `#0E1117`.

All series colours come from Okabe–Ito. Contrast figures are computed against
white / dark background.

**Assignment policy** (lines and markers; all ≥3:1 on both themes). Order is
always fcfs → edf → optimiser (simplest to smartest) in legends, tables and
radio buttons.

| Token | Hex | Okabe–Ito name | Marker symbol | Line dash (Run detail) | Contrast W / D |
|---|---|---|---|---|---|
| `fcfs` | `#CC79A7` | reddish purple | circle | solid | 3.1 / 6.2 |
| `edf` | `#009E73` | bluish green | square | solid | 3.4 / 5.5 |
| `optimiser` | `#0072B2` | blue | diamond | solid | 5.2 / 3.6 |

**Employment type and cost components** (fills: bars, stacked areas; these
are large areas with direct labels, so the lighter colours are acceptable;
never use them for thin lines).

| Token | Hex | Used for |
|---|---|---|
| `full_time` / `cost_salaried` | `#E69F00` (orange) | full-time scouts, salaried cost |
| `freelance` / `cost_freelance` | `#56B4E9` (sky blue) | freelancers, freelance cost |
| `cost_late_penalty` | `#D55E00` (vermillion) | late penalties; the only "bad" colour |
| `cost_automation` | `#999999` (grey) | pre-screen licence (if added, gap G2) |

**Reference marks and neutrals**

| Token | Value | Used for |
|---|---|---|
| `TARGET_LINE` | `#808080`, dash `"dash"`, width 1.5 | the on-time target line (3.9:1 / 4.8:1) |
| `PROMISE_LINE` | `#808080`, dash `"dot"`, width 1.5 | the 14-day promise on turnaround charts |
| `BAND_OPACITY` | `0.2` of the series colour, no outline | min–max across repeats |
| `NEUTRAL_MID` | `#E8E8E8` | heatmap zero point |

**Status badges** (`st.badge` colours; icon + word, so colour is never the only cue):

| Display state | Badge colour | Icon | Label |
|---|---|---|---|
| queued | gray | `:material/schedule:` | Queued |
| running | blue | `:material/progress_activity:` | Running |
| cancelling (derived: queued/running + `cancel` file exists) | orange | `:material/hourglass_top:` | Cancelling |
| done | green | `:material/check_circle:` | Done |
| failed | red | `:material/error:` | Failed |
| cancelled | gray | `:material/block:` | Cancelled |
| unreadable (derived: `status.json` missing/corrupt) | gray | `:material/help:` | Unreadable |

**Lifecycle state is not outcome.** "Done" says the simulation finished; it
says nothing about whether the run met 95%. The outcome gets its own column
("Meets target") and its own badge on Run detail ("Meets target" green /
"Misses target" orange, never red: missing a target is a result, not an
error). Discarded: a green/red "Done" depending on outcome; it overloads one
signal with two meanings, and a red "Done" looks like a crash.

---

## 3. Chart conventions (app and README share these)

All figures are built by pure functions in `src/scout_planner/charts.py` that
return a `plotly.graph_objects.Figure` and never import Streamlit. The app
renders them with `st.plotly_chart(fig, theme="streamlit")` at full container
width; the README script exports the **same function's output** to PNG.

### 3.1 Colour rules

- Set every trace colour explicitly from the tokens in §2.2; never rely on
  plotly's or Streamlit's default colourway.
- **Single-run charts** (Run detail) draw the run's series in its policy colour.
- **Compare charts**: colour = policy, **line dash = run slot** (A solid,
  B `dash`, C `dot`, D `dashdot`), and each line is **directly labelled** with
  its slot letter at its right end. This keeps "blue = optimiser" true on every
  page. Discarded: colouring by slot (A blue, B orange…), which is the usual
  default but would make a blue line on Compare mean something different from
  a blue dot on Sweep results. Accepted cost: four runs with the same policy
  differ only by dash and end label, so Compare recommends ≤3 runs when
  policies match (caption).
- **Heatmap** (capacity gap): diverging scale centred on zero,
  `[[0, "#0072B2"], [0.5, "#E8E8E8"], [1, "#D55E00"]]`, `zmid=0`, symmetric
  limits ±max|gap|. Blue = spare hours, vermillion = shortfall. Blue/vermillion
  stays distinct under red–green colour blindness, unlike a red/green scale.

### 3.2 Uncertainty across repeats (seeds)

- **Time series** (backlog, on-time by month, demand): the line is the **mean**
  over repeats; a **band** (filled area between min and max, `BAND_OPACITY`, no
  border, `hoverinfo="skip"`) shows the range. Subtitle says "band = range over
  3 repeats".
- **Point estimates** (sweep scatter dots, utilisation bars): **error bars**
  (min–max), thin, same colour as the marker. On the sweep scatter they are
  **off by default** behind a toggle "Show range over repeats", because 36
  dots with crosshairs become unreadable.
- **Compare overlays**: bands off by default (toggle), for the same reason.
- If `sim.seeds = 1`: no band, no error bars, subtitle says "single repeat, no
  range".

Why min–max and not a confidence interval: with 3–5 repeats a CI is mostly
noise, and "the worst and best of the repeats" is something a business user
reads correctly without statistics.

### 3.3 Number formats (one formatter each, §8)

| Quantity | Stored as | Display | Examples |
|---|---|---|---|
| Rates (on-time, utilisation, shares) | fraction 0–1 | percent, **1 decimal** | `94.3%` |
| Difference between two rates | fraction | **percentage points**, 1 decimal, signed | `-1.6 pts`, `+0.4 pts` |
| Money | cost units | **thousands with "k"**, thousands separator, 0 decimals; under 10k: 1 decimal | `1,240k`, `4.5k` |
| Money in form inputs | cost units | plain integer with separator | `4,500` |
| Days (turnaround) | float | 1 decimal + "days" (KPI) / "d" (tables, axes) | `17.0 days`, `9.8 d` |
| Growth multipliers | float | up to 1 decimal, trailing zero dropped, "x" | `4x`, `2.5x` |
| Counts | int | thousands separator | `3,120` |
| Months (axes) | date | `%b %Y`, year shown on first tick and January | `Mar 2027` |
| Timestamps | datetime | `%d %b %H:%M` | `03 Oct 10:15` |
| Durations | seconds | `Xm Ys`, or `Ys` under a minute | `2m 14s`, `18s` |

Plotly specifics: rates on axes `tickformat=".0%"`, in hover `".1%"`; money is
divided by 1,000 before plotting with `ticksuffix="k"`, `tickformat=",.0f"`.
Hover templates show the full formatted value plus the range, e.g.
`On time 94.3% (range 92.1–95.0%)`.

> **Concept: percent vs percentage points.** Going from 93% to 95% on-time is
> a change of **2 percentage points**, but a **2.2% relative** increase.
> Mixing the two is one of the most common errors in business dashboards. Every
> difference between two rates in this app is in "pts".

**Gotcha for `st.metric`:** Streamlit decides the delta arrow's direction from
a leading ASCII hyphen `-`. Format negative deltas with `-`, not the Unicode
minus sign, or a drop shows a green up-arrow.

### 3.4 Titles state the takeaway

> **Concept: action titles.** A chart title that states the conclusion ("On
> time drops below 95% from June") instead of the topic ("On-time rate by
> month"). The reader gets the point in one second and uses the chart to check
> it. The descriptive part moves to a smaller subtitle.

Every figure has:
- **Title** (18 px): the takeaway, generated from the data by a small rule (§5.3 lists them). If no rule fires, fall back to a descriptive title.
- **Subtitle** (13 px, `#808080`, rendered via `<br><sup>` in the plotly title): what is plotted, units, aggregation, e.g. "Open requests per week · mean of 3 repeats · band = range".
- Axis titles in sentence case with units: "Total cost for the year (k)", "On-time rate", "Week".
- Legend: horizontal, above the plot area, left-aligned (`orientation="h", y=1.02`). Hidden when there is only one series or when lines are directly labelled.

### 3.5 Target line

Horizontal line at `sim.target_on_time` (default 0.95) using `TARGET_LINE`,
drawn with `fig.add_hline`, annotated at the right end "95% target" (font
12 px, same grey). Drawn **below** data traces (`layer="below"`). The target
value is read from the run's params, never hard-coded; if compared runs have
different targets, draw one line per distinct target and label each.

### 3.6 Light and dark mode

- In the app, `theme="streamlit"` makes background, grid and font follow the
  active theme. Only trace colours are fixed, and every token was chosen to
  pass on both backgrounds (§2.2). Never use pure black or white for data.
- Annotations and subtitles use `#808080` (readable on both) or inherit the
  theme font colour; never hard-code `#000`.
- **README export:** `template="plotly_white"`, **solid white background**,
  1200×675 px, `scale=2`, into `docs/img/`. Discarded: a transparent PNG,
  which would put dark axis text on GitHub's dark background. The README
  figures add a source footnote: "Synthetic data · headline sweep · 5 repeats
  per setting".

### 3.7 Axis rules

- Bar charts start at zero, always.
- Scatter and line charts of rates may start above zero (e.g. y from 60% to
  100%) when the axis is labelled; starting at 0 would flatten every
  difference that matters near 95%. Upper bound for rates: 100%.
- Time axes cover the simulated year; the demand chart also shows the last 12
  history months, separated by a vertical dotted line labelled "Plan year starts".

---

## 4. Copy and tone

### 4.1 Voice

- Plain words for a head of operations, not for a modeller. Sentence case
  everywhere ("New run", not "New Run").
- Numbers with units, always. "17.0 days", not "17.0".
- Internal names (`assignment.policy`, CP-SAT, quantile, seed) never appear as
  **labels**; they appear only as the last line of a widget's tooltip, in code
  font, for people who use the YAML editor.
- Don't use "assignment run" on screen: the word "run" means one simulation
  here (D-011). The on-screen phrase is "assignment round" or "how often work
  is assigned".
- **Messages follow "what happened → why → what to do":** "Can't queue this
  run: 2 advanced settings are invalid. Fix the lines marked below." Never
  "Error: validation failed".
- No exclamation marks, no "Oops", no marketing words.

### 4.2 Glossary of on-screen labels

| On-screen label | Meaning shown in help | Source |
|---|---|---|
| **On-time rate** | Share of reports delivered within the promised 14 days | `on_time_rate` |
| **On-time target** | The service level a run must reach to pass (default 95%) | `sim.target_on_time` |
| **Meets target** / **Misses target** | Average on-time rate across repeats is at or above the target | `meets_target` |
| **Turnaround P90** | 9 in 10 reports were delivered within this many days | `p90_turnaround_days` |
| **Average turnaround** | Mean days from request to delivered report | `mean_turnaround_days` |
| **Total cost** | Salaries + freelance hours + late penalties (+ tool licence) for the year | `cost_total` |
| **Salaried cost / Freelance cost / Late penalties** | The three parts of total cost | `cost_*` |
| **Late reports** | Reports delivered after the promised date | derived |
| **At risk from day one** | Requests with no suitable match before the due date; nobody could have made them on time | `n_at_risk_day_one` |
| **Backlog** / **Open requests** | Requests received but not yet delivered, at week start | `open_requests` |
| **Utilisation** | Share of available hours actually spent on work | `util_*` |
| **Actual growth (what arrives)** | How much request volume really grows over the year | `demand.actual_growth` |
| **Assumed growth (what we hire for)** | The growth the agency believes in and plans hiring around | `capacity_plan.assumed_growth` |
| **Starting workload** | How busy the starting team is in month one | `demand.start_load` |
| **Planning caution** | How much buffer the hiring plan keeps (Average / Cautious / Very cautious) | `capacity_plan.quantile` |
| **Target workload per scout** | The plan hires once scouts would be busier than this | `capacity_plan.target_utilisation` |
| **Hiring plan** | Which scouts to hire, of which kind, and when to start recruiting | `hiring_plan` |
| **Capacity gap** | Hours of work needed minus hours available, per skill and month | `gap_hours` |
| **Video pre-screen** | Tool that does part of the desk review; sometimes its output is unusable | `automation.*` |
| **First come, first served** / **Earliest deadline first** / **Optimiser** | The three ways of deciding who does which task | `fcfs` / `edf` / `optimiser` |
| **Repeats** | Times the year is simulated with different luck; results show average and range | `sim.seeds` |
| **Run** | One simulated year for one set of settings | run folder |
| **Sweep** | A batch of runs that vary a few settings systematically | `sweep_id` |
| **Baseline (A)** | On Compare, the run every other run is measured against | — |

---

## 5. Pages

Each page section gives: a wireframe (top to bottom), the components, and the
empty / loading / error states.

### 5.1 New run

```
┌──────────────────────────────────────────────────────────────────────┐
│ New run                                                              │
│ caption: Choose the settings for one simulated year, then press Run. │
│ [info] Started from "Hire for 2x" (cloned).      [Reset to defaults] │
├───────────────────────────────┬──────────────────────────────────────┤
│ ▢ Demand (what arrives)       │ ▢ Capacity plan (what we hire for)   │
├───────────────────────────────┼──────────────────────────────────────┤
│ ▢ Team                        │ ▢ Assignment                         │
├───────────────────────────────┼──────────────────────────────────────┤
│ ▢ Video pre-screen            │ ▢ Costs                              │
├───────────────────────────────┼──────────────────────────────────────┤
│ ▢ Simulation                  │                                      │
├───────────────────────────────┴──────────────────────────────────────┤
│ > Advanced settings (YAML)                         [expander, closed]│
├──────────────────────────────────────────────────────────────────────┤
│ ▢ What you are about to simulate                                     │
│   4 plain sentences · "Changed from defaults: …" · warnings/errors   │
│   Run name [ placeholder: optimiser · 4x actual · 2x assumed ]       │
│   [            Run  (primary, full width)            ]               │
│   caption: Rough run time: 1–6 min. You can leave this page.        │
└──────────────────────────────────────────────────────────────────────┘
```

**Why Demand and Capacity plan share the first row.** The core idea of the
simulator (D-014) is the contrast between what arrives and what the agency
hires for. Putting the two side by side makes "4x arrives / 2x planned"
visible at a glance. Discarded: following `PARAMETERS.md` order (Team first),
which buries the most important pair in the middle of the page.

**Why no `st.form`.** `st.form` batches all inputs until submit, so the
summary and the warnings could not update as the user types. This page does no
computation, so rerunning the script on every change costs milliseconds.
Discarded: `st.form` (cleaner batching, but a stale summary defeats the review
step). Discarded: one tab per group (hides 6 of 7 groups; the user can't see
the whole setup at once).

> **Concept: the Streamlit rerun model.** Every widget interaction reruns the
> page script from top to bottom with the new widget values. That is what
> makes live summaries free here, and why expensive work (loading parquet)
> must be cached with `st.cache_data`. `st.form` exists to suppress those
> reruns until a submit button is pressed.

**State gotcha.** Streamlit deletes the state of widgets that are not rendered,
so switching to Runs and back would reset the form. Keep the draft in a
non-widget key, `st.session_state["draft"]` (a dict of parameter values),
initialise each widget from it, and write back in `on_change`. The draft is
seeded from defaults, or from `?clone=<id>`'s `params.yaml` on first load.

#### Basic parameters: widgets, ranges, help text

Help-text style: the **label** says *what* in plain words; the `help=` tooltip
says *what it does and which direction is "more"*, then the YAML key on its
own line in code font; a **caption** under the widget is used only for derived
values or a must-see warning. Percent widgets show integers (70%) and store
fractions (0.70); conversion happens in `param_form`, nowhere else.

| Group (container title) | Label | Widget | Range / step / format | Default | Tooltip (`help=`) |
|---|---|---|---|---|---|
| **Demand (what actually arrives)** | Actual growth (what arrives) | `st.select_slider` | 0.5x–6x, steps of 0.5, shown "4x" | 4x | How much request volume grows over the year: volume in month 12 ÷ volume in month 1. `demand.actual_growth` |
| | Starting workload | `st.slider` | 30–120%, step 5, `"%d%%"` | 70% | How busy the starting team is in month one. 70% means arriving work needs 70% of the team's usable hours. Sets the base volume. `demand.start_load` |
| | Live-view share | `st.slider` | 0–100%, step 5 | 40% | Share of requests that need a scout at a match. A live view takes the scout's whole day, in the match's region. `demand.live_view_share` |
| **Capacity plan (what the agency hires for)** | Same as actual growth | `st.checkbox` | — | on | On: the agency predicts growth perfectly. Off: set a different figure to test a wrong forecast. |
| | Assumed growth (what we hire for) | `st.select_slider` | as actual growth | mirrors actual; **disabled** while the checkbox is on | The growth the hiring plan is built on. `capacity_plan.assumed_growth` |
| | Planning caution | `st.radio(horizontal=True)` | Average (P50) / Cautious (P80) / Very cautious (P90) | Cautious (P80) | How much buffer the plan keeps. Cautious plans for a month busier than 80% of likely outcomes: more hires, fewer late reports. `capacity_plan.quantile` |
| | Target workload per scout | `st.slider` | 50–100%, step 5 | 80% | The plan hires once scouts would be busier than this. Lower means more slack and more cost. `capacity_plan.target_utilisation` |
| **Team (on day one)** | Full-time scouts | `st.number_input` | 0–100, step 1 | 24 | Salaried scouts on day one. Fixed monthly cost, busy or not. `team.full_time_count` |
| | Freelancers | `st.number_input` | 0–100, step 1 | 16 | Freelancers on day one. They offer 8–25 hours a week and are paid only for hours used, at a premium. `team.freelance_count` |
| | Planned hires actually join | `st.toggle` | — | on | On: the scouts the hiring plan asks for join when recruitment finishes. Off: the team never grows. `team.follow_hiring_plan` |
| **Assignment** | Who does what | `st.radio` with `captions=` | First come, first served (*oldest request first*) / Earliest deadline first (*most urgent first*) / Optimiser (*plans the best fit each round; slowest*) | Optimiser | The rule that decides which scout does which task. `assignment.policy` |
| | How often work is assigned | `st.radio(horizontal=True)` | Every day / Once a week | Every day | `assignment.cadence` |
| | How work is handed out | `st.radio(horizontal=True)` | Task by task / Desk review and write-up together | Task by task | Together keeps one scout on a report's desk work and write-up. `assignment.unit`. If `bundle` is cut (PRD cut line), show only the default as caption text, not a widget. |
| **Video pre-screen** | Use the video pre-screen tool | `st.toggle` | — | off | Software that does part of the desk review. `automation.enabled` |
| | Desk time saved when it works | `st.slider` | 0–90%, step 5; **disabled** when tool off | 40% | `automation.desk_reduction` |
| | Chance its output is unusable | `st.slider` | 0–60%, step 5; **disabled** when tool off | 15% | When it fails, the scout does the full desk review anyway, plus about an hour lost. `automation.rework_rate` |
| **Costs** | Monthly cost per full-time scout | `st.number_input` | 0–20,000, step 100 | 4,500 | Fully loaded salary. `cost.full_time_monthly_salary`. Caption: "≈ 28 per hour" (derived live) |
| | Freelance premium | `st.slider` | 1.0–3.0, step 0.1, `"%.1fx"` | 1.5x | Freelance hourly rate ÷ salaried hourly cost. `cost.freelance_premium`. Caption: "Freelance rate ≈ 42 per hour" |
| | Cost of one late report | `st.number_input` | 0–20,000, step 100 | 1,000 | Refunds and lost clients per late report. `cost.late_penalty`. **Visible caption:** "This number shapes the answer most: it decides whether paying for capacity beats paying for lateness." |
| **Simulation** | Repeats | `st.slider` | 1–5 | 3 | The year is simulated this many times with different luck; results show the average and the range. More repeats: slower, more reliable. `sim.seeds` |
| | On-time target | `st.slider` | 50–100%, step 1 | 95% | `sim.target_on_time` |

Widget-type rules used above: **number input** for counts and money (exact
values, typed); **slider** for bounded rates where the rough position matters
more than the exact value; **select slider** for an ordered set of discrete
values (growth); **radio** for 2–3 named choices (all options visible, no
click to open, unlike a selectbox); **toggle** for on/off switches that take
effect, **checkbox** for "link this value to that one". Dependent widgets are
**disabled, not hidden**, when their switch is off, so the layout does not
jump and the user sees what turning it on unlocks.

If a cloned value is not on a widget's grid (e.g. quantile 0.85 from YAML), add
it as an extra option labelled "Custom (P85)" rather than silently snapping it.

#### Advanced settings (YAML) — *progressive disclosure*

`st.expander("Advanced settings (YAML)", expanded=False)` containing
`st.tabs(["Edit", "Full parameter set"])`:

- **Edit:** `st.text_area` (monospace, height ≈ 400 px) holding **only the
  non-basic keys** (`PARAMETERS.md` rows without "Basic"), prefilled with their
  current values. Caption: "Settings not in the form above. Basic settings are
  set in the form, not here."
- **Full parameter set:** read-only `st.code(yaml, language="yaml")` of the
  merged result: exactly what will be written to `params.yaml`.

Rule: a basic key typed into the YAML is a validation error ("`sim.seeds` is
set in the form above"). Why: two editors for the same value need two-way
syncing, which Streamlit makes fragile (a widget's value can't be changed after
it renders in the same rerun). Discarded: full two-way sync between form and
YAML. The expander auto-opens if its content has errors.

#### Validation (live, before queueing)

On every rerun, merge defaults + basic widgets + YAML overrides and validate
with the pydantic models. Three kinds of message:

| Kind | Component | Placement | Blocks Run? | Example |
|---|---|---|---|---|
| YAML syntax error | `st.error` | directly under the text area | yes | "Line 7: expected a number after `time_limit_s:`." |
| Invalid value | `st.error`, one bullet per problem | review card, and under the text area when the key is in the YAML | yes | "Advanced › `assignment.time_limit_s`: must be between 0.1 and 10 (you entered 30)." |
| Warning (valid but suspicious) | `st.warning` | review card | no | "No scouts at all: every report will be late." / "Late reports cost 0: understaffing will look cheapest." / "One repeat: results will show no range." |
| Note (deliberate experiment) | `st.info` | review card | no | "Forecast error experiment: hiring for 2x while 4x arrives." |
| Duplicate (nice-to-have) | `st.info` + **Open it** button | review card | no | "A finished run with identical settings exists: 'Defaults' (03 Oct 10:15). Results would be identical." |

While errors exist, the **Run** button is disabled and a caption next to it
says why: "Fix 2 problems above to run." Discarded: an always-enabled button
that shows errors after the click (the errors are already on screen; a click
that does nothing is worse). The duplicate check works because runs are
deterministic: same parameters, same code version, same results.

#### Review card: "What you are about to simulate"

`st.container(border=True)` with `st.subheader("What you are about to simulate")`,
then 4 generated sentences in plain words (bold the values), e.g.:

> Demand grows **4x** over the year, starting at **70%** of the team's capacity.
> The agency **hires for 2x**, **cautiously (P80)**, and planned hires **join**.
> **24 full-time scouts and 16 freelancers** to start; work is assigned **every
> day, task by task**, by the **optimiser**; the pre-screen tool is **off**.
> A late report costs **1,000**; **3 repeats**; target **95%** on time.

Then a caption list **"Changed from defaults"** (or "from 'X'" when cloned):
`Assumed growth 4x → 2x`, one per line; "Nothing changed" otherwise. This is
*recognition over recall*: the user doesn't have to remember what the defaults
were. Then messages (table above), then:

- **Run name:** `st.text_input` with `placeholder` = auto-name (§8
  `auto_run_name`, e.g. "optimiser · 4x actual · 2x assumed"); empty field
  means the auto-name is used. Max 60 characters; the slug for `run_id` is
  derived from it.
- **Run** button: `type="primary"`, full width (*Fitts's law*: the most
  important action is the biggest target, right next to the summary it
  commits to).
- Caption: rough run time from `estimate_runtime(params)` ("Rough run time:
  under 30 s" for fcfs/edf; "1–6 min" for the optimiser) + "You can leave this
  page or close the tab; the run keeps going."
- **Reset to defaults** sits at the top of the page (secondary button), far
  from Run, so it can't be hit by mistake.

On click: create the run folder (state `queued`), then
`go_to("runs", new=run_id)`. The page change itself prevents double submits.
If the folder can't be written: `st.error("Couldn't queue the run: <reason>. Nothing was created.")` and stay on the page with the draft intact.

**States:** loading the clone source: `st.spinner("Loading settings from 'X'…")`;
clone id not found: `st.warning("Run 'X' no longer exists; starting from defaults.")`;
there is no empty state (defaults always exist).

### 5.2 Runs

```
┌──────────────────────────────────────────────────────────────────────┐
│ Runs                                                                 │
│ caption: Runs execute one at a time, oldest first. Closing the tab   │
│          does not stop them.                                         │
│ [success] Queued "Hire for 2x". It is #2 in line.  [Open run]        │
├──────────────────────────────────────────────────────────────────────┤
│ In progress                                     (fragment, every 2s) │
│ ▢ Hire for 4x  [Running]                          [Open] [Cancel ▾]  │
│   ███████████░░░░░░  63% · simulating seed 2/3, week 31/52           │
│   Generate › Forecast › Capacity plan › **Simulate** · 1m 52s ·      │
│   about 1 min left                                                   │
│ Queued (2)  table: # · Name · Policy · Queued at   [Cancel selected] │
├──────────────────────────────────────────────────────────────────────┤
│ Finished                                                             │
│ [search name] [status ▾ Done, Failed, Cancelled] [Show sweep runs ○] │
│ dataframe (multi-row selection)                                      │
│ [Open] [Compare] [Clone] [Delete ▾]       caption: "3 selected"      │
└──────────────────────────────────────────────────────────────────────┘
```

**Why split "In progress" from "Finished".** Only the in-progress part
changes every few seconds, so only it lives in `st.fragment(run_every="2s")`.
If the whole table refreshed, the user's row selection would jump whenever a
new run appeared. When the fragment sees a run change state to
done/failed/cancelled, it stores a toast message in session state and calls
`st.rerun(scope="app")` so the Finished table picks it up. Discarded: one
auto-refreshing table for everything (simpler code, broken selection).

#### In progress

- **Running run card** (`st.container(border=True)`, at most one, since the
  worker runs one at a time): bold name + `status_badge`; `st.progress(p,
  text="63% · simulating seed 2/3, week 31/52")` (text from `status.message`);
  caption with the stage line (current stage bold), elapsed time, and "about N
  min left" (extrapolated from elapsed / progress, shown only after 10% so the
  first guess isn't wild); buttons **Open** and **Cancel**.
- **Cancel** on a running run uses `st.popover("Cancel")` containing "Stop this
  run? Work done so far is discarded." + **Stop run** button. After confirming,
  the badge becomes **Cancelling** with the caption "Stops at the end of the
  current simulated week." until the worker writes `cancelled`. That
  intermediate state matters: without it the user clicks Cancel again and
  thinks it is broken.
- **Queued table** (shown if any): `st.dataframe`, multi-row selection,
  columns: `#` (queue position), Name, Policy, Actual growth, Queued at, Sweep.
  Button **Cancel selected** (no confirmation: nothing has been computed yet).
  Rows with a cancel request show "Cancelling" and leave the numbering.
- **Empty:** caption "Nothing running." + `st.page_link` to New run.

#### Finished table

`st.dataframe(hide_index=True, on_select="rerun", selection_mode="multi-row")`,
newest first. Keep numeric columns numeric (so sorting works) and format them
with `st.column_config.NumberColumn(format=…)`.

| Column | Config | Notes |
|---|---|---|
| Name | `TextColumn`, pinned | |
| Status | `TextColumn` | Done / Failed / Cancelled / Unreadable (word, not colour) |
| Meets target | `CheckboxColumn` (read-only) | blank unless Done |
| On-time rate | `NumberColumn`, stored ×100, `"%.1f%%"` | mean over repeats |
| Turnaround P90 | `NumberColumn`, `"%.1f d"` | |
| Total cost | `NumberColumn`, stored ÷1000, `"%.0fk"` | |
| Policy | `TextColumn` | plain label ("Optimiser") |
| Actual growth | `TextColumn` | "4x" |
| Assumed growth | `TextColumn` | "2x" |
| Pre-screen | `TextColumn` | On / Off |
| Repeats | `NumberColumn` | |
| Finished | `DatetimeColumn`, `"DD MMM HH:mm"` | |
| Duration | `TextColumn` | "2m 14s" |
| Sweep | `TextColumn` | shown only when "Show sweep runs" is on |

Filters above the table (one `st.columns` row): name search (`st.text_input`,
substring, case-insensitive), status (`st.multiselect`, default all), and
`st.toggle("Show sweep runs")`, **off by default**, with the hidden count in
its label ("Show sweep runs (36 hidden)"), because one sweep adds dozens of
rows that drown the user's own runs.

**Actions** (*selection then action*, the bulk-action bar pattern from mail
clients): a row of buttons below the table, each enabled only when the
selection fits, with `help=` explaining why when disabled.

| Button | Enabled when | Does |
|---|---|---|
| **Open** | exactly 1 selected | Run detail |
| **Compare** | 2–4 selected, all Done | Compare |
| **Clone** | exactly 1 selected | New run prefilled |
| **Delete** | ≥1 selected, none running/queued | `st.popover`: "Delete 3 runs? Their folders are removed for good." + list of names + **Delete** |

Discarded: buttons inside each row. `st.dataframe` can't hold buttons, and
hand-building rows from `st.columns` is slow and breaks sorting.

**Just-queued banner:** if `?new=<id>`, `st.success` with the name, queue
position and **Open run**; dismissed when the user navigates away (remove the
query param).

**States**
- **Empty (no runs at all):** a bordered container: subheader "No runs yet",
  text "A run simulates one year of the agency with the settings you choose.
  It takes from a few seconds to a few minutes.", buttons **Start a new run**
  (primary) and **See the headline results**, caption "Or from a terminal:
  `make sweep SWEEP=quick`". (*Empty state as onboarding*, not a dead end.)
- **Filters match nothing:** caption "No runs match these filters." + **Clear filters**.
- **Loading:** reading `status.json` and `summary.json` for every folder is
  fast; wrap in `st.cache_data` keyed by each file's modification time. No
  spinner unless > 0.5 s.
- **Error:** a folder with missing or corrupt `status.json` shows as
  "Unreadable" with blank metrics, plus a caption under the table: "1 run
  folder couldn't be read: `<run_id>`. Delete it or check `log.txt`." The page
  never crashes because of one bad folder (*graceful degradation*).

### 5.3 Run detail

```
┌──────────────────────────────────────────────────────────────────────┐
│ Run [selectbox: Hire for 2x, get 4x · 03 Oct 10:15            ▾]    │
├──────────────────────────────────────┬───────────────────────────────┤
│ Hire for 2x, get 4x                  │ [Clone to new run]            │
│ [Done] [Misses target]               │ [Compare with…]               │
│ caption: run id · finished 03 Oct    │ [Download parameters]         │
│ 10:21 · took 4m 02s · code c5b4637   │                               │
├──────────────────────────────────────┴───────────────────────────────┤
│ ## On time 93.4%: 1.6 pts below the 95% target, at 1,240k a year.    │
│ ┌ On-time rate ┐┌ Turnaround P90 ┐┌ Total cost ┐┌ Late reports ┐      │
│ └──────────────┘└────────────────┘└────────────┘└──────────────┘      │
│ [cost split: one horizontal 100% stacked bar]                        │
├──────────────────────────────────────────────────────────────────────┤
│ [Service] [Team & cost] [Demand & hiring plan] [Parameters]          │
│   charts per tab                                                     │
└──────────────────────────────────────────────────────────────────────┘
```

- **Run picker** at the top (`st.selectbox`, newest first, options formatted
  "name · 03 Oct 10:15"), bound to `?run=`. With no query param it opens the
  newest Done run.
- **Header:** `st.columns([3, 1])`. Left: `st.title(name)`, `status_badge`,
  outcome badge (Done runs only), caption with run id, finished time, duration,
  code version. Right: **Clone to new run**, **Compare with…**,
  `st.download_button("Download parameters", params.yaml)`; **Cancel** (popover)
  if queued/running. No Delete here: destructive actions live only on Runs.

#### Body by state

| State | Body |
|---|---|
| queued | `st.info("Waiting in line: #2. It starts when 'X' finishes.")` + Parameters section only |
| running | `st.status("Running: simulating seed 2/3, week 31/52", state="running", expanded=True)` containing `st.progress` and the stage line; inside a 2 s fragment; on finish, `st.rerun(scope="app")` to render results. Parameters below. |
| cancelling | as running, badge Cancelling, caption "Stops at the end of the current simulated week." |
| failed | `st.error("This run failed: <error>.")`. If error is `interrupted`: "The background worker stopped while this run was in progress, for example because the app was closed. Its settings are fine; run it again." Then `st.expander("Log (last 50 lines)")` with `st.code(tail)`, and **Run again** (primary). Parameters below. |
| cancelled | `st.info("Cancelled at week 31 of 52. No results were kept.")` + **Run again** + Parameters |
| done | everything below |

#### Takeaway sentence

`st.subheader`, generated by `run_takeaway(summary, params)`:
- passes: "On time 96.1%: 1.1 pts above the 95% target, at 1,310k a year."
- fails: "On time 93.4%: 1.6 pts below the 95% target, at 1,240k a year."
- range straddles the target (min < target ≤ max): append "Close call: some repeats pass, some don't."

#### KPI row

`st.columns(4)`, each an `st.metric(..., border=True)` with a `st.caption`
underneath for the range over repeats.

| Metric | Value | Delta | `delta_color` | Caption under it |
|---|---|---|---|---|
| On-time rate | `93.4%` | `-1.6 pts vs 95% target` | `normal` (below target = red down arrow) | "Range 92.1–94.8% over 3 repeats · 41 requests at risk from day one" |
| Turnaround P90 | `17.0 days` | `+3.0 days vs 14-day promise` | `inverse` (above promise = red) | "Average 9.8 days · range 16.0–18.0" |
| Total cost | `1,240k` | none (no natural reference) | — | "Range 1,210k–1,275k" |
| Late reports | `206` | none | — | "of 3,120 requests (6.6%)" |

Below the row, the **cost split**: one horizontal 100% stacked bar, about 90
px tall, segments salaried / freelance / late penalties (/ tool licence),
colours from §2.2, each segment labelled inside "Salaried 780k (63%)" (labels
hidden for segments under 8%; hover always works). Title (takeaway): "Late
penalties are 18% of the cost". No axes, no legend (direct labels instead).
Why a bar and not a pie: lengths are compared more accurately than angles, and
the same bar style stacks naturally on Compare (one bar per run).

#### Tabs

`st.tabs(["Service", "Team & cost", "Demand & hiring plan", "Parameters"])`.
Why tabs: four groups of charts would make a very long scroll. Discarded: one
long page (everything visible, but the KPIs and the Parameters end up 3
screens apart). Accepted cost: content in tabs is less discoverable, so tab
names say exactly what is inside.

Charts (all in the run's policy colour unless stated; mean line + min–max band):

| Tab | Chart | x axis | y axis | Extras | Takeaway title rule |
|---|---|---|---|---|---|
| Service | **Backlog over time** (line + band) | Week (`week_start`) | Open requests (count, from 0) | thin `#808080` line for late open requests | "Backlog peaks at {max} open requests in {month}"; if last-week backlog ≤ 1.2× first: "Backlog stays under control all year" |
| Service | **On-time rate by month** (line + markers + band) | Month the report was due | On-time rate (%, from min(60%, data) to 100%) | target line §3.5 | "On time drops below 95% from {first failing month}" / "On time stays above 95% every month" |
| Service | **Turnaround distribution** (histogram, all repeats pooled) | Turnaround (days, 1-day bins) | Reports | `PROMISE_LINE` at 14 "14-day promise"; marker at P90 | "{x}% of reports arrive within 14 days" |
| Service | caption | | | "41 requests (1.3%) were at risk from day one: no suitable match before their due date. They count in the on-time rate." | |
| Team & cost | **Team size over time** (step line; stacked area by employment type if gap G1 is closed) | Week | Scouts | hire arrivals as markers with hover "2 full-time joined (planned in March)" | "The team grows from 40 to {n} scouts" |
| Team & cost | **Utilisation by employment type** (bars + error bars) | Employment type (Full-time, Freelance) | Utilisation (%, 0–100) | dashed line at target workload per scout | "Full-time scouts are {x}% busy, freelancers {y}%" |
| Team & cost | **Hires** (`st.metric` pair) | | | "Full-time hires 6 · Freelance hires 9" | |
| Demand & hiring plan | **Demand: actual vs forecast** (lines) | Month (last 12 history months + plan year) | Requests per month | Actual (policy colour, solid, band if per-repeat data exist); Forecast P50 (`#808080` dashed); Planning forecast (P80) (`#808080` dotted); vertical dotted line "Plan year starts" | ratio r = actual ÷ P50 over the plan year: "Demand came in {r}x above the plan" if r > 1.1; "below" if r < 0.9; else "The forecast tracked demand within 10%" |
| Demand & hiring plan | **Capacity gap heatmap** | Month | Skill type (sorted by total shortfall, worst on top) | diverging scale §3.1; colour bar titled "Hours short (+) / spare (−)"; hover: required, available, gap | "Shortfalls concentrate in {top skill} from {month}" / "No skill runs short of capacity" |
| Demand & hiring plan | **Hiring plan** (`st.dataframe`) | | | columns: Start recruiting (month), Joins (month), Skill, Hire type, Count, Reason | — |
| Parameters | `param_table(params, compare_to=defaults)` (`st.dataframe`) | | | columns: Group, Setting (plain label), Value, Default, Changed (checkbox); toggle "Only show changed settings" **on** by default; `st.expander("Full params.yaml")` with `st.code` | — |
| Parameters | **Per-repeat results** (`st.expander`, closed) | | | `seeds.parquet` as a table | — |

Hiring plan states: empty → caption "The plan made no hires: the starting
team covers the assumed demand."; follow-plan off → `st.info("Planned hires
did not join in this run (setting 'Planned hires actually join' is off).")`.

**Loading:** results are immutable once a run is Done, so cache every parquet
read with `st.cache_data` keyed by run id (no expiry). First load under
`st.spinner("Loading results…")`.

**Errors:** missing result file for a Done run → that chart is replaced by
`st.warning("Couldn't load <file> for this run.")`; the rest of the page still
renders. Run id not found → `st.error("Run '<id>' doesn't exist; it may have been deleted.")` + link to Runs.

**Empty (no runs at all):** same empty state as Runs.

### 5.4 Compare

```
┌──────────────────────────────────────────────────────────────────────┐
│ Compare                                                              │
│ Runs to compare (2–4): [A Hire for 2x ×] [B Hire for 4x ×] [+ …]     │
│ caption: The first run is the baseline (A); others show differences. │
├────────────────┬────────────────┬──────────────────────────────────┤
│ A Hire for 2x  │ B Hire for 4x  │  header cards: policy, outcome badge,│
│ Misses target  │ Meets target   │  on-time, total cost                 │
├──────────────────────────────────────────────────────────────────────┤
│ What's different                       [Show all settings ○]         │
│ table: Setting · A · B                                               │
│ [info] Same demand in both runs, so differences come from …          │
├──────────────────────────────────────────────────────────────────────┤
│ Results   table: Metric · Better is · A · B (Δ vs A)                 │
├──────────────────────────────────────────────────────────────────────┤
│ Charts    [Show range over repeats ○]                                │
│ backlog overlay · on-time by month overlay · cost split bars         │
└──────────────────────────────────────────────────────────────────────┘
```

- **Picker:** `st.multiselect` of Done runs only, `max_selections=4`, options
  "name · 03 Oct 10:15", bound to `?runs=`. Slot letters A–D follow
  selection order; A is the **baseline** that every delta is measured against
  (*anchoring on a baseline*: one reference point is easier to read than all
  pairwise differences). To change the baseline, reorder by removing and
  re-adding (enough for v1).
- **Header cards:** `st.columns(n)`, each a bordered container: slot letter +
  name (as `st.page_link` to Run detail), policy label, outcome badge, on-time
  and total cost in one line.
- **What's different** (`param_diff(params_list)`): `st.dataframe` with rows
  only for settings whose values differ, columns Setting (plain label),
  Group, A, B, …; toggle "Show all settings" (off). Code version appears as a
  row if it differs, with caption "Different code versions: results may
  differ for reasons other than the settings."
  - If **only `assignment.*` settings differ**: `st.info("Same requests arrive
    in all these runs, so differences come from how work is assigned.")`. This
    is true by design (common random numbers, D-015) and is the most useful
    sentence on the page for the policy comparison.
  - If **nothing differs** (same code): `st.info("These runs have identical
    settings, so their results are identical: the simulator is deterministic.")`.
- **Results** (`metric_table(summaries, baseline=0)`): rows On-time rate,
  Meets target, Turnaround P90, Average turnaround, Late reports, At risk from
  day one, Total cost, Salaried cost, Freelance cost, Late penalties,
  Utilisation full-time, Utilisation freelance, Full-time hires, Freelance
  hires. Columns: Metric, "Better is" (higher / lower), A, then B… as text
  cells "91.0% (-2.4 pts)". Deltas carry signs, not colours (red/green
  colouring would fail for colour-blind users, and "better" depends on the
  metric's direction, which the "Better is" column states).
- **Charts** (cut line #3 in the PRD: build last, cut first). Colour = policy,
  dash = slot, direct slot labels at line ends (§3.1). Toggle "Show range over
  repeats" (off).
  1. **Backlog over time:** x Week, y Open requests. Title: "B's backlog stays {x}% lower than A's at the peak" style, fallback "Backlog over time".
  2. **On-time rate by month:** x Month due, y On-time rate (%), target line. Title: "A drops below 95% from {month}; B never does".
  3. **Cost split:** horizontal stacked bars (absolute k, not 100%), one bar per run labelled "A · name", cost colours. Title: "B costs {Δ}k more, mostly salaries".
  Caption when ≥3 runs share a policy: "Several runs share a policy colour; tell them apart by line style and the letter at each line's end."
- **States:** fewer than 2 selected → `st.info("Pick at least two finished runs to compare.")` + caption "Tip: on Runs, select rows and press Compare."; fewer than 2 Done runs exist → `st.info("You need at least two finished runs.")` + **Start a new run**; a run in `?runs=` that no longer exists → dropped from the picker with `st.warning("Run '<id>' no longer exists and was removed from the comparison.")`.

### 5.5 Sweep results

```
┌──────────────────────────────────────────────────────────────────────┐
│ Sweep results                                                        │
│ caption: A sweep is a batch of runs that vary a few settings.        │
│ Sweep [Headline (published) ▾]   Actual growth (●4x ○2x ○1x ○All)    │
│ [Show range over repeats ○]                                          │
├──────────────────────────────────────────────────────────────────────┤
│ ## Cheapest way to stay on time at 4x: optimiser, pre-screen on,     │
│ ##   1,180k a year                                                   │
│ scatter: x Total cost (k) · y On-time rate · 95% line · highlight    │
├──────────────────────────────────────────────────────────────────────┤
│ Selected: optimiser · 4x · pre-screen on   [Open run] [Clone settings]│
├──────────────────────────────────────────────────────────────────────┤
│ All runs in this sweep (table, sorted by total cost)                 │
└──────────────────────────────────────────────────────────────────────┘
```

- **Sweep picker** (`st.selectbox`): "Headline (published)" from
  `data/published/headline_summary.parquet` first, then any other published
  summaries, then local sweeps found via `sweep_id` in run folders (labelled
  "quick (local, 03 Oct)"). Bound to `?sweep=`.
- **Actual growth** selector: `st.radio(horizontal=True)` with one option per
  growth value in the sweep plus **All**, default = the highest growth (the
  question is about 4x). "All" renders **small multiples**: one panel per
  growth level side by side, shared y axis, each with its own highlight.
  Why filter by growth: across growth levels, the cheapest passing run would
  always be a 1x run, which answers a question nobody asked.

> **Concept: small multiples.** The same chart repeated for each value of one
> variable, on shared axes, side by side (Tufte's term). The eye compares the
> panels directly, which is easier than reading one chart with every
> variable crammed into colours and shapes.

**The headline scatter** (`fig_sweep_scatter`, also the README headline):

| Element | Spec |
|---|---|
| x | Total cost for the year (k); range padded 5% around the data |
| y | On-time rate (mean over repeats); range from min(data) − 2 pts to 100% |
| One dot per run | colour + symbol = policy (§2.2); **filled = pre-screen on, hollow = off** (`symbol="circle-open"` etc.); size 10 |
| Target line | §3.5, "95% target" |
| Highlight: cheapest passing | the lowest-cost run with mean on-time ≥ target: size 18, 2 px `#808080` outline, plus an annotation with arrow: "Cheapest that meets 95%: Optimiser, pre-screen on · 1,180k · 96.1%" |
| Error bars | min–max on both axes, toggle, off by default |
| Hover | run name, every varied setting in plain words, on-time with range, total cost with range, Turnaround P90 |
| Legend | two small legends in words: policy (colour+symbol), "Filled = pre-screen on" as a caption under the chart, because plotly's legend can't show fill meaning cleanly |
| Selection | `st.plotly_chart(..., on_select="rerun", selection_mode="points")`; the clicked dot fills the "Selected" panel |
| Title (takeaway) | "Cheapest way to stay on time at 4x: {policy}, pre-screen {on/off}, {cost}"; no passing run: "No setting reaches 95% at 4x; best is {policy} at {x}%" (and no highlight, the best run gets the annotation instead) |
| Subtitle | "One dot per run · {n} runs · mean of {k} repeats · synthetic data" |

Team mix (if varied in the headline sweep, Q6) is **not** given a visual
channel: colour, shape and fill are taken, and a fourth encoding would make the
chart unreadable. It appears in hover and in the table; if it turns out to
matter, it becomes a filter like growth.

- **Selected panel:** shown once a dot or table row is selected: a one-line
  description in plain words and the run's KPIs, plus **Open run** (only if
  the run folder exists locally; published headline runs usually only exist
  as summary rows) and **Clone settings** (always: rebuilt from default +
  sweep `fixed` + the row's varied values).
- **Table:** `st.dataframe` of all runs in the sweep, single-row selection
  synced with the chart's selection, sorted by total cost; columns: varied
  settings (plain labels), Meets target (checkbox), On-time rate, Turnaround
  P90, Total cost, Late penalties. Respects the growth filter.

**States**
- **Empty** (no published summary and no local sweeps): bordered container,
  "No sweep results yet. A sweep runs many settings in one go. From a
  terminal, run `make sweep SWEEP=quick` (a few minutes) or `make all`." +
  **Start a single run instead**.
- **Local sweep still running:** `st.progress(24/36, text="24 of 36 runs
  finished")` above the chart, chart shows finished runs only, fragment
  refresh every 5 s, title prefixed "So far:".
- **Error:** published file unreadable → `st.error("Couldn't read the
  published results file <path>. Re-create it with make sweep SWEEP=headline.")`;
  other sweeps in the picker remain usable.

---

## 6. Shared state patterns (summary)

| Situation | Pattern | Component |
|---|---|---|
| Nothing to show yet | *Empty state*: say what goes here, why it's empty, and offer the next action as a button | bordered container + subheader + text + 1–2 buttons |
| Work in another process | progress with words, elapsed time, estimate | `st.progress(text=…)`, `st.status`, fragment refresh |
| Local wait over 0.5 s | spinner with a verb | `st.spinner("Loading results…")` |
| Something failed, page still usable | inline, local message | `st.warning` in place of the broken part |
| Something failed, page can't continue | what happened → why → what to do | `st.error` + one recovery button |
| Background event while elsewhere | non-blocking notice | `st.toast` |
| Destructive action | confirm in place, name what will be lost | `st.popover` with a second button |

---

## 7. Key flows

### 7.1 First-time user
1. `make app` opens the browser on **Sweep results**. The takeaway title and
   the highlighted dot give the answer to the project's question.
2. Clicks the highlighted dot → Selected panel → **Clone settings**.
3. **New run** opens prefilled, with the "Started from…" banner. Changes one
   thing, e.g. Cost of one late report 1,000 → 2,000. The review card shows
   "Changed: Cost of one late report 1,000 → 2,000" and the auto-name updates.
4. Presses **Run** → **Runs** with "Queued … #1 in line"; the run card shows
   progress; the sidebar shows it on every page.
5. A toast says it finished → **Open** → **Run detail**: takeaway sentence,
   KPIs, tabs.
6. Optional: **Compare with…** another finished run.

### 7.2 "Plan for 2x, get 4x"
1. **New run**, defaults (actual growth already 4x).
2. Capacity plan: untick **Same as actual growth**, set **Assumed growth (what
   we hire for)** to 2x. The info note appears: "Forecast error experiment:
   hiring for 2x while 4x arrives." Name: "Hire for 2x, get 4x". **Run**.
3. On Runs, select it → **Clone**; tick **Same as actual growth** again; name
   "Hire for 4x, get 4x"; **Run**. It queues behind the first.
4. When both are Done: select both on Runs → **Compare**.
5. What's different shows exactly one row (Assumed growth 2x vs 4x). Results:
   A's on-time lower, cost lower. Backlog chart: the lines split around the
   middle of the year, when the gap appears and a 3-month recruitment lead
   time can't close it.
6. Open A's **Demand & hiring plan** tab: actual demand pulls away from the
   forecast lines; the heatmap turns vermillion from that month on.

### 7.3 Comparing two policies
1. Open an optimiser run on **Run detail** → **Clone to new run**.
2. Set **Who does what** to Earliest deadline first; the run-time caption
   drops to "under 30 s". **Run**.
3. Select both on Runs → **Compare**. The note "Same requests arrive in all
   these runs…" confirms it is a fair test.
4. Charts use the policy colours (blue optimiser, green EDF), matching the
   Sweep results page. Results table shows the on-time difference in pts and
   the cost difference in k.
5. For the full picture across growth levels, go to **Sweep results**, choose
   **All**: one panel per growth level, policies by colour.

---

## 8. Components and helpers to build

Two modules keep the UI consistent. Everything marked *pure* has no Streamlit
import and gets unit tests (it is the functional core of the UI).

**`src/scout_planner/charts.py`** (shared with the README script; pure)

| Function / constant | Purpose |
|---|---|
| `POLICY_COLORS`, `POLICY_SYMBOLS`, `POLICY_ORDER`, `EMPLOYMENT_COLORS`, `COST_COLORS`, `SLOT_DASHES`, `TARGET_LINE`, `PROMISE_LINE`, `BAND_OPACITY`, `GAP_COLORSCALE` | design tokens (§2.2) |
| `style_figure(fig, title, subtitle, *, for_readme=False)` | fonts, legend position, margins; README mode adds white template and source footnote |
| `add_target_line(fig, target, label=None)` | §3.5 |
| `add_range_band(fig, x, lo, hi, color, name)` | min–max band (§3.2) |
| `fig_backlog(weekly: dict[slot, df], policies, show_range)` | single run or overlay |
| `fig_on_time_by_month(...)`, `fig_turnaround_hist(...)`, `fig_cost_split(summaries, mode="share"/"absolute")`, `fig_team_size(...)`, `fig_utilisation(...)`, `fig_demand_vs_forecast(...)`, `fig_capacity_heatmap(...)` | §5.3 / §5.4 charts |
| `fig_sweep_scatter(summary_df, target, growth=None, show_range=False)` | the headline (§5.5) |
| `takeaway_*` functions, one per chart | the title rules in §5.3; return `None` → descriptive fallback |

**`app/ui.py`** (Streamlit components + pure helpers)

| Function | Pure? | Purpose |
|---|---|---|
| `fmt_pct`, `fmt_pts`, `fmt_cost_k`, `fmt_days`, `fmt_growth`, `fmt_count`, `fmt_when`, `fmt_duration` | pure | §3.3; `fmt_pts` uses ASCII `-` |
| `PARAM_UI` | pure (data) | per basic key: label, group, widget kind, range, step, format, help text, display↔stored conversion. **Single source** for the form, the parameter table and the diff labels |
| `METRIC_UI` | pure (data) | per metric: label, formatter, "better is", help |
| `display_state(status, run_dir) -> str` | pure | adds derived states: cancelling, unreadable |
| `status_badge(state)` | UI | `st.badge` per §2.2 |
| `outcome_badge(summary)` | UI | Meets / Misses target |
| `param_form(draft) -> dict` | UI | renders §5.1 groups from `PARAM_UI` |
| `advanced_yaml_editor(draft) -> (overrides, errors)` | UI | §5.1 expander |
| `validate_draft(draft, overrides) -> (params, errors, warnings, notes)` | pure | pydantic + warning rules |
| `describe_params(params) -> str` | pure | the review card sentences |
| `changed_settings(params, reference) -> list[(label, old, new)]` | pure | "Changed from defaults" and Parameters tab |
| `auto_run_name(params, defaults) -> str` | pure | up to 3 changed settings in order policy, actual growth, assumed growth, pre-screen, team; "Defaults" if none |
| `estimate_runtime(params) -> (lo_s, hi_s)` | pure | rough ranges from §7 of ARCHITECTURE |
| `run_takeaway(summary, params) -> str` | pure | §5.3 sentence |
| `kpi_row(summary, params)` | UI | §5.3 KPI row |
| `run_progress(status)` | UI | progress bar + stage line + time left |
| `param_diff(params_list, show_all=False) -> DataFrame` | pure | §5.4 |
| `metric_table(summaries, baseline=0) -> DataFrame` | pure | §5.4 |
| `queue_indicator()` | UI | sidebar fragment §1.3 |
| `empty_state(title, body, actions)` | UI | §6 |
| `confirm_popover(label, message, confirm_label, on_confirm)` | UI | §6 |
| `go_to(page, **query)` / `read_nav()` | UI | §1.2 |
| `load_run(run_id)` / `list_runs()` | UI (cached) | thin `st.cache_data` wrappers over `runs.py`; cache key includes file mtimes |

---

## 9. Layout, responsiveness and accessibility

- **Target screen:** a laptop, 1280 px wide and up, wide layout. Streamlit
  stacks `st.columns` vertically on narrow windows automatically; that is the
  whole responsive strategy. Mobile is not a goal (single local user).
- **Two-column maximum** for forms, four for KPI cards; charts always full
  width of their container. One chart per row, so each is wide enough to read.
- **Reading order** on every results page: takeaway sentence → numbers →
  charts → raw detail.
- **Accessibility checklist:** every widget has a visible label (never
  `label_visibility="collapsed"`); colour always paired with a shape, dash,
  fill or word; tokens meet the contrast targets in §2.2; information needed
  to make a decision is on screen, not only in a tooltip (e.g. the late-penalty
  caption); native Streamlit widgets give keyboard navigation for free, which
  is another reason to avoid custom components.

---

## 10. Open questions and contract gaps (for the Architect / user)

| # | Gap | Why the UI needs it | Suggested fix |
|---|---|---|---|
| G1 | `weekly.parquet` has only `team_size` | Team chart can't split full-time vs freelance | add `team_full_time`, `team_freelance` |
| G2 | `seeds.parquet` has no `cost_automation`, but `automation.monthly_cost` exists | Cost split segments wouldn't add up to total cost | add `cost_automation` and include it in `cost_total` |
| G3 | No worker liveness signal | Queued runs sit forever if the worker isn't running, with no explanation | worker rewrites `data/runs/.worker.json` (`pid`, `last_seen`) every few seconds; app warns if older than 30 s |
| G4 | Cancelling a *queued* run waits until the worker reaches it | "Cancelling" could show for minutes | worker checks cancel flags on queued runs on each poll and marks them `cancelled` without running |
| G5 | Published summary column names | Error bars and hover need min/max | flatten as `<metric>_mean`, `<metric>_min`, `<metric>_max`, plus `name`, `policy` |
| Q-D1 | `automation.monthly_cost` is marked ⚑ (answer-shaping) but not Basic | Without it in the form, the pre-screen looks free to form users | promote it to Basic under "Video pre-screen" (user decision, ties to Q5) |
| Q-D2 | Landing page | Chosen: Sweep results. Alternative: Runs | user preference |

---

## Implementation notes (added after M5 build, 2026-10-03)
- Pure helpers live in `app/viewmodel.py` (re-exported by `app/ui.py`);
  formatters live in `src/scout_planner/formatting.py` so `charts.py` and the
  README script can share them.
- `go_to` uses `st.switch_page(page, query_params=...)` directly (Streamlit
  1.65 carries query params across page switches).
- New run includes "Tool licence per month" (`automation.monthly_cost`) under
  Video pre-screen. Q-D1 closed. G1–G4 closed (in DATA_CONTRACTS).
- Sweep picker keys: `published:<name>` / `local:<sweep_id>`; "Clone settings"
  opens `/new?sweep=…&row=…`.
- The worker-down warning shows only when something is queued or running.
- Polish backlog (seen in a visual check at ~800 px wide): KPI cards truncate
  labels/values; long chart titles are clipped; hide Streamlit's "Deploy"
  toolbar button (`client.toolbarMode = "minimal"`).
