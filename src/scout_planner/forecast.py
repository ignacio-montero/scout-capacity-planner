"""Stage [2]: monthly demand forecast and its backtest. Spec: DATA_CONTRACTS section 2.

``build_forecast(requests, params)`` is pure: history rows in, two tables out
(``forecast``, ``backtest``), no file access. ``write_forecast`` /
``read_forecast`` are the thin I/O edge, as in ``generate``.

How the forecast is built
-------------------------
* **Only history is read.** Rows with ``period == "future"`` are dropped before
  anything else, so the forecast cannot peek at what will actually arrive
  (no *leakage*; a test changes ``demand.actual_growth`` and expects an
  identical forecast).
* **Top-down (D-005).** One model for the aggregate monthly request count;
  each skill gets ``aggregate x its share of the trailing 12 months``. Skill
  series are small, noisy counts; the aggregate is smooth.
* **Statistical model.** ETS(A,Ad,A) from ``statsmodels``: additive error,
  damped additive trend, additive yearly seasonality. All-additive is the
  family with *exact* (closed-form) prediction intervals, so no simulation
  and no extra randomness. A multiplicative-season variant was tried; it was
  not consistently better across seeds and needs simulated intervals.
* **Plan year = statistical base x assumed growth (D-020).** For the plan year
  the model's damped trend is *replaced* by the business assumption: the
  forecast for plan month ``m`` is the ETS level plus the seasonal term at the
  forecast origin (what ETS says month ``m`` would be with no further trend),
  times the growth ramp ``trend(m) / trend(origin)``. The ramp has the
  generator's functional form (organic growth before ``SIM_START``,
  ``assumed_growth ** t`` after it), with ``capacity_plan.assumed_growth``.
  Multiplying the full ETS forecast (trend included) by the ramp would count
  the history's organic growth twice.
* **Uncertainty.** ETS prediction intervals give a standard deviation per
  horizon. Its size *relative to* the ETS mean is applied to the overlaid
  mean: ``q = mean x (1 + z_q x sd_h / ets_mean_h)``. P50 is the mean (the
  error is symmetric). Per-skill quantiles are the aggregate quantiles times
  the share, i.e. skills are treated as perfectly correlated (a
  simplification: it understates the relative noise of small skills).
* **Fallback.** If the ETS fit fails (exception, non-finite output, too little
  data), the seasonal naive forecast is used instead (same month last year,
  scaled by the growth ramp), its interval from the spread of year-on-year
  differences. It is logged and flagged in the ``method`` column.
* **Hours.** ``hours = requests x hours_per_request(params)``, where hours per
  request uses the run's demand shape and automation settings.

The backtest
------------
*Rolling origin* evaluation on the history only: at origins every 3 months
with at least 24 months of training data and a full 12-month horizon inside
the history, fit the full ETS (trend included) on the data before the origin,
forecast 12 months, and compare with what happened. The baseline is the
*seasonal naive* forecast (same month last year). Skill rows use the
top-down model vs each skill's own seasonal naive. WAPE (weighted absolute
percentage error, ``sum |actual - forecast| / sum actual``) summarises it.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from scout_planner import generate
from scout_planner.config import Params, apply_overrides
from scout_planner.generate import SIM_START, SKILL_TYPES, World, add_months

logger = logging.getLogger(__name__)

SEASON_LENGTH = 12  # months in a seasonal cycle
SHARE_WINDOW_MONTHS = 12  # trailing window for the top-down skill shares
MIN_ETS_MONTHS = 2 * SEASON_LENGTH  # below this, seasonal ETS is not attempted
BACKTEST_MIN_TRAIN_MONTHS = 24
BACKTEST_HORIZON = 12
BACKTEST_STEP_MONTHS = 3
ALL = "ALL"  # skill_type label of aggregate backtest rows
METHOD_ETS = "ets"
METHOD_NAIVE = "seasonal_naive"

# --- table schemas (the contract, as code) -----------------------------------------
# ``method`` extends DATA_CONTRACTS section 2: it flags rows produced by the fallback.

FORECAST_SCHEMAS: dict[str, pa.Schema] = {
    "forecast": pa.schema(
        [
            ("month", pa.date32()),
            ("skill_type", pa.string()),
            ("requests_p50", pa.float64()),
            ("requests_pq", pa.float64()),
            ("hours_p50", pa.float64()),
            ("hours_pq", pa.float64()),
            ("method", pa.string()),
        ]
    ),
    "backtest": pa.schema(
        [
            ("origin", pa.date32()),
            ("month", pa.date32()),
            ("skill_type", pa.string()),
            ("actual", pa.float64()),
            ("model", pa.float64()),
            ("naive", pa.float64()),
            ("method", pa.string()),
        ]
    ),
}


# --- requests -> hours ---------------------------------------------------------------


def automation_factor(params: Params) -> float:
    """Expected desk hours with the pre-screen tool, as a share of desk hours without it.

    ``1`` when automation is off. When on: with probability ``1 - rework`` the
    tool works and saves ``desk_reduction``; with probability ``rework`` its
    output is unusable, so the full desk review is done anyway plus
    ``rework_overhead_hours``. Hence ``(1 - r)(1 - d) + r (1 + overhead / E[desk])``.
    """
    a = params.automation
    if not a.enabled:
        return 1.0
    mean_desk = float(np.mean(params.demand.desk_hours))
    r = a.rework_rate
    return (1 - r) * (1 - a.desk_reduction) + r * (1 + a.rework_overhead_hours / mean_desk)


def hours_per_request(params: Params) -> float:
    """Expected scout hours per request under the run's demand shape and automation.

    ``E[desk] x automation_factor + live_view_share x live_view_hours + E[write-up]``.
    """
    d = params.demand
    return (
        float(np.mean(d.desk_hours)) * automation_factor(params)
        + d.live_view_share * d.live_view_hours
        + float(np.mean(d.writeup_hours))
    )


# --- monthly series ------------------------------------------------------------------


def history_months(params: Params) -> list[dt.date]:
    """First day of every history month, oldest first."""
    n = params.demand.history_months
    return [add_months(SIM_START, k - n) for k in range(n)]


def plan_months(params: Params) -> list[dt.date]:
    """First day of every month of the plan year (the simulated period)."""
    return [add_months(SIM_START, k) for k in range(params.sim.months)]


def monthly_counts(requests: pd.DataFrame, months: list[dt.date]) -> pd.DataFrame:
    """Requests per month (rows, first-of-month dates) and skill (columns, canonical order).

    Months or skills with no request are zeros. Requests outside ``months`` are
    ignored, so pass history months to get history counts.
    """
    skills = list(SKILL_TYPES) + sorted(set(requests["skill_type"]) - set(SKILL_TYPES))
    if requests.empty:
        return pd.DataFrame(0.0, index=months, columns=skills)
    month = [dt.date(d.year, d.month, 1) for d in requests["received_date"]]
    table = requests.groupby([month, requests["skill_type"].to_numpy()]).size().unstack()
    return table.reindex(index=months, columns=skills).fillna(0).astype(float)


def skill_shares(counts: pd.DataFrame, window: int = SHARE_WINDOW_MONTHS) -> pd.Series:
    """Each skill's share of requests over the last ``window`` rows of ``counts``.

    Falls back to the whole frame, then to equal shares, if the window is empty.
    """
    for frame in (counts.iloc[-window:], counts):
        total = float(frame.to_numpy().sum())
        if total > 0:
            return frame.sum() / total
    return pd.Series(1.0 / counts.shape[1], index=counts.columns)


# --- the statistical model -----------------------------------------------------------


@dataclass(frozen=True)
class ModelForecast:
    """An aggregate forecast for horizons 1..h made at one origin.

    ``mean``: the model's own forecast (trend included; this is what the
    backtest scores). ``base``: the same without any further trend, i.e. level
    and seasonal term at the origin (what the growth ramp multiplies).
    ``sd``: forecast standard deviation per horizon. ``method``: ``ets`` or
    ``seasonal_naive``.
    """

    mean: np.ndarray
    base: np.ndarray
    sd: np.ndarray
    method: str


def _series(y: np.ndarray) -> pd.Series:
    # A dated, fixed-frequency index: statsmodels' prediction intervals need it
    # (with a bare numpy array, get_prediction fails). The dates are arbitrary.
    return pd.Series(y, index=pd.date_range("2000-01-01", periods=len(y), freq="MS"))


def fit_ets(y: np.ndarray, horizon: int) -> ModelForecast:
    """Fit ETS(A,Ad,A) with a yearly season to ``y`` and forecast ``horizon`` months.

    Raises ``ValueError`` if there is too little data or the result is not finite.
    """
    if len(y) < MIN_ETS_MONTHS:
        raise ValueError(f"ETS needs >= {MIN_ETS_MONTHS} months, got {len(y)}")
    with warnings.catch_warnings():
        # Optimiser chatter (convergence, bounds) is judged by the finiteness
        # check below, not printed for every backtest origin.
        warnings.simplefilter("ignore")
        model = ETSModel(
            _series(np.asarray(y, dtype=float)),
            error="add",
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=SEASON_LENGTH,
        )
        res = model.fit(disp=False)
        pred = res.get_prediction(start=len(y), end=len(y) + horizon - 1)
    mean = np.asarray(pred.predicted_mean, dtype=float)
    sd = np.sqrt(np.asarray(pred.forecast_variance, dtype=float))
    # Level + seasonal term at the origin; seasonal state for horizon h is the
    # one from the same month a year earlier. Checked against ETS's own mean in
    # the tests (mean == base + damped trend sum).
    season = np.asarray(res.season, dtype=float)[-SEASON_LENGTH:]
    base = float(np.asarray(res.level)[-1]) + np.resize(season, horizon)
    if not (np.isfinite(mean).all() and np.isfinite(sd).all() and np.isfinite(base).all()):
        raise ValueError("ETS fit produced non-finite values")
    return ModelForecast(mean=mean, base=base, sd=sd, method=METHOD_ETS)


def seasonal_naive(y: np.ndarray, horizon: int) -> ModelForecast:
    """Same month last year, repeated; sd from the year-on-year differences."""
    y = np.asarray(y, dtype=float)
    if len(y) < SEASON_LENGTH:
        raise ValueError(f"seasonal naive needs >= {SEASON_LENGTH} months, got {len(y)}")
    mean = np.resize(y[-SEASON_LENGTH:], horizon)
    diffs = y[SEASON_LENGTH:] - y[:-SEASON_LENGTH]
    sd_value = float(np.sqrt(np.mean(diffs**2))) if len(diffs) else float(np.std(y))
    # Error of a seasonal naive forecast compounds once per completed season.
    sd = sd_value * np.sqrt(np.arange(horizon) // SEASON_LENGTH + 1)
    return ModelForecast(mean=mean, base=mean.copy(), sd=sd, method=METHOD_NAIVE)


def forecast_with_fallback(y: np.ndarray, horizon: int, *, label: str = "") -> ModelForecast:
    """ETS if it fits, seasonal naive otherwise (logged)."""
    try:
        return fit_ets(y, horizon)
    except Exception as exc:  # any fit failure: numerical, data, library
        logger.warning("ETS fit failed%s (%s); using seasonal naive", label, exc)
        return seasonal_naive(y, horizon)


# --- growth overlay --------------------------------------------------------------------


def monthly_growth_ramp(params: Params, months: list[dt.date], growth: float) -> np.ndarray:
    """Mean trend multiplier over each month's days, relative to the run-rate at SIM_START.

    Uses the generator's own demand curve with seasonality switched off, so
    the ramp has exactly the form the world is generated with: organic growth
    (``history_growth_per_year``) before ``SIM_START`` and ``growth ** t`` after.
    """
    flat = apply_overrides(params, {"demand.seasonality_strength": 0.0})
    anchor = generate.daily_rate(flat, np.array([SIM_START], dtype="datetime64[D]"))[0]
    out = []
    for month in months:
        days = np.arange(np.datetime64(month, "D"), np.datetime64(add_months(month, 1), "D"))
        out.append(float(generate.daily_rate(flat, days, growth=growth).mean() / anchor))
    return np.array(out)


# --- the stage -------------------------------------------------------------------------


def _history_requests(world_or_requests: World | pd.DataFrame) -> pd.DataFrame:
    req = world_or_requests.requests if isinstance(world_or_requests, World) else world_or_requests
    return req[req["period"] == "history"]


def plan_year_forecast(counts: pd.DataFrame, params: Params) -> pd.DataFrame:
    """The ``forecast`` table from history counts (months x skills) and the run's params."""
    months = plan_months(params)
    horizon = len(months)
    y = counts.sum(axis=1).to_numpy()
    fit = forecast_with_fallback(y, horizon, label=" (plan year)")

    cp = params.capacity_plan
    origin_month = add_months(SIM_START, -1)  # the last history month
    ramp_origin = monthly_growth_ramp(params, [origin_month], cp.assumed_growth)[0]
    if fit.method == METHOD_ETS:
        # ETS base is the level at the origin: scale it from there.
        ramp = monthly_growth_ramp(params, months, cp.assumed_growth) / ramp_origin
    else:
        # Naive base is last year's value for that month: scale it from that month.
        year_before = [add_months(m, -SEASON_LENGTH) for m in months]
        ramp = monthly_growth_ramp(params, months, cp.assumed_growth) / monthly_growth_ramp(
            params, year_before, cp.assumed_growth
        )
    p50 = np.clip(fit.base * ramp, 0.0, None)
    z = NormalDist().inv_cdf(cp.quantile)
    scale = np.where(fit.mean > 0, fit.mean, np.maximum(fit.base, 1.0))
    relative_sd = fit.sd / np.maximum(scale, 1e-9)
    pq_ = p50 * (1.0 + z * relative_sd)

    shares = skill_shares(counts)
    hpr = hours_per_request(params)
    rows = {
        "month": [],
        "skill_type": [],
        "requests_p50": [],
        "requests_pq": [],
        "hours_p50": [],
        "hours_pq": [],
        "method": [],
    }
    for k, month in enumerate(months):
        for skill, share in shares.items():
            rows["month"].append(month)
            rows["skill_type"].append(skill)
            rows["requests_p50"].append(float(p50[k] * share))
            rows["requests_pq"].append(float(pq_[k] * share))
            rows["hours_p50"].append(float(p50[k] * share * hpr))
            rows["hours_pq"].append(float(pq_[k] * share * hpr))
            rows["method"].append(fit.method)
    return _frame("forecast", rows)


def backtest_origins(n_months: int) -> list[int]:
    """Training lengths (= index of the first forecast month) of every backtest origin."""
    last = n_months - BACKTEST_HORIZON
    return list(range(BACKTEST_MIN_TRAIN_MONTHS, last + 1, BACKTEST_STEP_MONTHS))


def run_backtest(counts: pd.DataFrame) -> pd.DataFrame:
    """Rolling-origin backtest of the aggregate model (and top-down skills) vs seasonal naive.

    ``counts``: history months (rows, first-of-month dates) x skills. Rows of
    the result: one per origin x horizon month x (``ALL`` + each skill).
    ``origin`` is the first forecast month (training data ends the month before).
    """
    months = list(counts.index)
    total = counts.sum(axis=1).to_numpy()
    values = counts.to_numpy()
    rows: dict[str, list] = {c: [] for c in FORECAST_SCHEMAS["backtest"].names}
    for n_train in backtest_origins(len(months)):
        origin = months[n_train]
        fit = forecast_with_fallback(total[:n_train], BACKTEST_HORIZON, label=f" at {origin}")
        shares = skill_shares(counts.iloc[:n_train]).to_numpy()
        for h in range(BACKTEST_HORIZON):
            t = n_train + h
            naive_t = t - SEASON_LENGTH * (h // SEASON_LENGTH + 1)  # latest same month seen
            series = [(ALL, total[t], fit.mean[h], total[naive_t])]
            series += [
                (skill, values[t, j], fit.mean[h] * shares[j], values[naive_t, j])
                for j, skill in enumerate(counts.columns)
            ]
            for skill, actual, model, naive in series:
                rows["origin"].append(origin)
                rows["month"].append(months[t])
                rows["skill_type"].append(skill)
                rows["actual"].append(float(actual))
                rows["model"].append(float(model))
                rows["naive"].append(float(naive))
                rows["method"].append(fit.method)
    return _frame("backtest", rows)


def build_forecast(
    world_or_requests: World | pd.DataFrame, params: Params
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage [2]: ``(forecast, backtest)`` from the history requests (pure).

    Accepts a ``World`` or a ``requests`` frame; only ``period == "history"``
    rows are used.
    """
    counts = monthly_counts(_history_requests(world_or_requests), history_months(params))
    return plan_year_forecast(counts, params), run_backtest(counts)


# --- summaries (pure) --------------------------------------------------------------------


def wape(actual: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> float:
    """Weighted absolute percentage error: ``sum |actual - forecast| / sum actual``."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    denom = float(np.abs(a).sum())
    return float(np.abs(a - f).sum() / denom) if denom > 0 else math.nan


def backtest_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    """WAPE of model and naive per ``skill_type`` (``ALL`` first), plus who wins."""
    out = []
    for skill, grp in backtest.groupby("skill_type", sort=False):
        m, n = wape(grp["actual"], grp["model"]), wape(grp["actual"], grp["naive"])
        out.append(
            {
                "skill_type": skill,
                "wape_model": m,
                "wape_naive": n,
                "model_beats_naive": bool(m < n),
                "n_origins": int(grp["origin"].nunique()),
            }
        )
    summary = pd.DataFrame(out)
    order = (summary["skill_type"] != ALL).astype(int)
    return summary.assign(_o=order).sort_values(["_o"], kind="stable").drop(columns="_o")


def realised_monthly_totals(requests: pd.DataFrame, params: Params) -> pd.Series:
    """Requests that actually arrived per plan month (for reporting, never for fitting)."""
    fut = requests[requests["period"] == "future"]
    return monthly_counts(fut, plan_months(params)).sum(axis=1)


# --- I/O edge --------------------------------------------------------------------------


def _frame(table: str, columns: dict[str, list]) -> pd.DataFrame:
    """A table with exactly the contract's columns, in order, with stable dtypes."""
    schema = FORECAST_SCHEMAS[table]
    if list(columns) != schema.names:
        raise ValueError(f"{table}: columns {list(columns)} != contract {schema.names}")
    data = {}
    for f in schema:
        if pa.types.is_date32(f.type):
            data[f.name] = pd.Series(list(columns[f.name]), dtype=object)
        elif pa.types.is_string(f.type):
            data[f.name] = pd.Series(columns[f.name], dtype="str")
        else:
            data[f.name] = pd.Series(columns[f.name], dtype="float64")
    return pd.DataFrame(data)


def write_forecast(forecast: pd.DataFrame, backtest: pd.DataFrame, run_dir: str | Path) -> dict:
    """Write ``forecast.parquet`` and ``backtest.parquet`` into the run folder."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in (("forecast", forecast), ("backtest", backtest)):
        table = pa.Table.from_pandas(df, schema=FORECAST_SCHEMAS[name], preserve_index=False)
        paths[name] = run_dir / f"{name}.parquet"
        pq.write_table(table, paths[name])
    return paths


def read_forecast(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read back what :func:`write_forecast` wrote: ``(forecast, backtest)``."""
    out = []
    for name in ("forecast", "backtest"):
        schema = FORECAST_SCHEMAS[name]
        df = pq.read_table(Path(run_dir) / f"{name}.parquet", schema=schema).to_pandas()
        out.append(_frame(name, {c: df[c].tolist() for c in schema.names}))
    return out[0], out[1]
