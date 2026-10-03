"""Stage [2] forecast: hours conversion, no leakage, growth overlay, backtest, fallback."""

from __future__ import annotations

import datetime as dt
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from scout_planner import forecast as f
from scout_planner import generate as g
from scout_planner.config import Params, apply_overrides

P = Params()


@pytest.fixture(scope="module")
def world() -> g.World:
    return g.generate_world(P)


@pytest.fixture(scope="module")
def built(world: g.World) -> tuple[pd.DataFrame, pd.DataFrame]:
    return f.build_forecast(world, P)


def monthly_total(fc: pd.DataFrame, column: str = "requests_p50") -> pd.Series:
    return fc.groupby("month", sort=True)[column].sum()


# --- requests -> hours (by hand) -------------------------------------------------------


def test_hours_per_request_without_automation_by_hand() -> None:
    # desk (4+10)/2 = 7, live 0.4 x 8 = 3.2, write-up (2+4)/2 = 3
    assert f.automation_factor(P) == 1.0
    assert f.hours_per_request(P) == pytest.approx(13.2)


def test_hours_per_request_with_automation_by_hand() -> None:
    p = apply_overrides(P, {"automation.enabled": True})
    # works (0.85): 7 x 0.6 = 4.2 h; fails (0.15): 7 + 1 overhead = 8 h
    expected_desk = 0.85 * 4.2 + 0.15 * 8.0  # 4.77
    assert f.automation_factor(p) * 7 == pytest.approx(expected_desk)
    assert f.hours_per_request(p) == pytest.approx(expected_desk + 3.2 + 3.0)


@pytest.mark.parametrize(
    ("rework", "reduction", "overhead", "factor"),
    [(0.0, 0.4, 1.0, 0.6), (0.6, 0.5, 3.5, 0.4 * 0.5 + 0.6 * 1.5), (0.5, 0.0, 0.0, 1.0)],
)
def test_automation_factor_edge_cases(
    rework: float, reduction: float, overhead: float, factor: float
) -> None:
    p = apply_overrides(
        P,
        {
            "automation.enabled": True,
            "automation.rework_rate": rework,
            "automation.desk_reduction": reduction,
            "automation.rework_overhead_hours": overhead,
        },
    )
    assert f.automation_factor(p) == pytest.approx(factor)


def test_forecast_hours_are_requests_times_hours_per_request(
    built: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    fc, _ = built
    hpr = f.hours_per_request(P)
    np.testing.assert_allclose(fc["hours_p50"], fc["requests_p50"] * hpr)
    np.testing.assert_allclose(fc["hours_pq"], fc["requests_pq"] * hpr)


def test_automation_lowers_hours_but_not_requests(world: g.World) -> None:
    base, _ = f.build_forecast(world, P)
    auto, _ = f.build_forecast(world, apply_overrides(P, {"automation.enabled": True}))
    pd.testing.assert_series_equal(base["requests_pq"], auto["requests_pq"])
    assert (auto["hours_pq"] < base["hours_pq"]).all()


# --- shape of the forecast table -------------------------------------------------------


def test_forecast_table_contract(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    fc, _ = built
    assert list(fc.columns) == f.FORECAST_SCHEMAS["forecast"].names
    assert sorted(set(fc["month"])) == f.plan_months(P)
    assert all(type(m) is dt.date and m.day == 1 for m in fc["month"])
    assert len(fc) == 12 * len(g.SKILL_TYPES)
    assert set(fc["method"]) == {f.METHOD_ETS}
    assert (fc["requests_pq"] > fc["requests_p50"]).all()  # P80 above the median
    assert (fc["requests_p50"] > 0).all()


def test_planning_quantile_widens_with_the_quantile(world: g.World) -> None:
    p50, _ = f.build_forecast(world, apply_overrides(P, {"capacity_plan.quantile": 0.5}))
    p95, _ = f.build_forecast(world, apply_overrides(P, {"capacity_plan.quantile": 0.95}))
    np.testing.assert_allclose(p50["requests_pq"], p50["requests_p50"])
    assert (p95["requests_pq"] > p95["requests_p50"] * 1.02).all()


def test_top_down_split_uses_trailing_twelve_month_shares(
    world: g.World, built: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    fc, _ = built
    hist = world.requests[world.requests["period"] == "history"]
    last_year = hist[hist["received_date"] >= dt.date(2026, 1, 1)]
    expected = last_year["skill_type"].value_counts(normalize=True)
    for _, month in fc.groupby("month"):
        shares = month.set_index("skill_type")["requests_p50"] / month["requests_p50"].sum()
        pd.testing.assert_series_equal(
            shares.sort_index(), expected.sort_index(), check_names=False, rtol=1e-9
        )


# --- leakage and the growth overlay ------------------------------------------------------


def test_forecast_reads_history_only(world: g.World) -> None:
    """No leakage: the realised future (and actual growth) never reaches the forecast."""
    other = g.generate_world(apply_overrides(P, {"demand.actual_growth": 1.0}))
    assert not other.requests.equals(world.requests)  # the futures really differ
    a = f.build_forecast(world, P)
    b = f.build_forecast(other, apply_overrides(P, {"demand.actual_growth": 1.0}))
    c = f.build_forecast(world.requests[world.requests["period"] == "history"], P)
    for x, y in ((a, b), (a, c)):
        pd.testing.assert_frame_equal(x[0], y[0])
        pd.testing.assert_frame_equal(x[1], y[1])


def test_assumed_growth_changes_forecast_but_not_backtest_or_history(world: g.World) -> None:
    low_params = apply_overrides(P, {"capacity_plan.assumed_growth": 2.0})
    high, bt_high = f.build_forecast(world, P)
    low, bt_low = f.build_forecast(world, low_params)
    pd.testing.assert_frame_equal(bt_high, bt_low)
    ratio = monthly_total(high) / monthly_total(low)
    # Same run-rate at the origin; by December the ramps differ by ~(4/2)^(11.5/12).
    assert ratio.iloc[0] == pytest.approx(2 ** (0.5 / 12), rel=0.01)
    assert ratio.iloc[-1] == pytest.approx(2 ** (11.5 / 12), rel=0.01)
    assert ratio.is_monotonic_increasing
    # The world (history included) does not depend on the planning assumption.
    pd.testing.assert_frame_equal(g.generate_world(low_params).requests, world.requests)


@pytest.mark.parametrize("growth", [4.0, 2.0, 1.0])
def test_assumed_equal_actual_is_roughly_unbiased(growth: float) -> None:
    p = apply_overrides(P, {"demand.actual_growth": growth, "capacity_plan.assumed_growth": growth})
    w = g.generate_world(p)
    fc, _ = f.build_forecast(w, p)
    forecast_total = monthly_total(fc)
    realised = f.realised_monthly_totals(w.requests, p)
    assert forecast_total.sum() == pytest.approx(realised.sum(), rel=0.08)
    quarters = np.arange(12) // 3
    for q in range(4):
        fq = forecast_total.to_numpy()[quarters == q].sum()
        rq = realised.to_numpy()[quarters == q].sum()
        assert fq == pytest.approx(rq, rel=0.15), f"Q{q + 1}"


def test_assumed_below_actual_underforecasts() -> None:
    p = apply_overrides(P, {"capacity_plan.assumed_growth": 2.0})  # actual stays 4x
    w = g.generate_world(p)
    fc, _ = f.build_forecast(w, p)
    assert monthly_total(fc).sum() < 0.85 * f.realised_monthly_totals(w.requests, p).sum()


def test_growth_ramp_matches_the_generator() -> None:
    months = f.plan_months(P)
    ramp = f.monthly_growth_ramp(P, months, 4.0)
    # Continuous at SIM_START, then 4 ** t: January's mean is just above 1.
    assert 1.0 < ramp[0] < 4 ** (31 / 365)
    assert ramp[-1] == pytest.approx(np.mean(4 ** (np.arange(334, 365) / 365)), rel=1e-9)
    history = f.monthly_growth_ramp(P, [dt.date(2026, 1, 1)], 4.0)[0]
    assert history < 1.0 and history == pytest.approx(1.15 ** (-(365 - 15) / 365), rel=0.01)


# --- the model -----------------------------------------------------------------------------


def test_ets_base_plus_damped_trend_reproduces_the_ets_mean(world: g.World) -> None:
    """``base`` (level + season at the origin) is ETS's forecast minus its trend part."""
    counts = f.monthly_counts(
        world.requests[world.requests["period"] == "history"], f.history_months(P)
    )
    y = counts.sum(axis=1).to_numpy()
    fit = f.fit_ets(y, 12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ETSModel(
            f._series(y), error="add", trend="add", damped_trend=True,
            seasonal="add", seasonal_periods=12,
        ).fit(disp=False)  # fmt: skip
    phi = float(np.asarray(res.params)[res.param_names.index("damping_trend")])
    trend = np.cumsum(phi ** np.arange(1, 13)) * float(np.asarray(res.slope)[-1])
    np.testing.assert_allclose(fit.base + trend, fit.mean, atol=1e-6)
    assert (np.diff(fit.sd) >= -1e-9).all()  # uncertainty does not shrink with horizon


def test_seasonal_naive_repeats_last_year() -> None:
    y = np.arange(30, dtype=float)
    fit = f.seasonal_naive(y, 14)
    np.testing.assert_array_equal(fit.mean, np.r_[y[-12:], y[-12:-10]])
    assert fit.sd[0] == pytest.approx(12.0)  # every year-on-year difference is 12
    assert fit.sd[12] == pytest.approx(12.0 * np.sqrt(2))


def test_ets_refuses_too_short_series_and_fallback_catches_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(ValueError, match="ETS needs"):
        f.fit_ets(np.ones(20), 12)
    with caplog.at_level(logging.WARNING, logger="scout_planner.forecast"):
        fit = f.forecast_with_fallback(np.arange(20, dtype=float), 12)
    assert fit.method == f.METHOD_NAIVE
    assert "seasonal naive" in caplog.text


def test_fallback_path_flags_and_scales_last_year(
    world: g.World, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(y: np.ndarray, horizon: int) -> f.ModelForecast:
        raise RuntimeError("optimiser exploded")

    monkeypatch.setattr(f, "fit_ets", broken)
    with caplog.at_level(logging.WARNING, logger="scout_planner.forecast"):
        fc, bt = f.build_forecast(world, P)
    assert "optimiser exploded" in caplog.text
    assert set(fc["method"]) == {f.METHOD_NAIVE} and set(bt["method"]) == {f.METHOD_NAIVE}
    np.testing.assert_array_equal(bt["model"][bt["skill_type"] == f.ALL],
                                  bt["naive"][bt["skill_type"] == f.ALL])  # fmt: skip
    # Plan year = last year's month x (ramp this month / ramp a year earlier).
    hist = world.requests[world.requests["period"] == "history"]
    last_year = f.monthly_counts(hist, f.history_months(P)).sum(axis=1).to_numpy()[-12:]
    months = f.plan_months(P)
    year_before = [g.add_months(m, -12) for m in months]
    ramp = f.monthly_growth_ramp(P, months, 4.0) / f.monthly_growth_ramp(P, year_before, 4.0)
    np.testing.assert_allclose(monthly_total(fc).to_numpy(), last_year * ramp)
    assert (monthly_total(fc, "requests_pq") > monthly_total(fc)).all()


# --- backtest --------------------------------------------------------------------------------


def test_backtest_frame_shape(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, bt = built
    assert list(bt.columns) == f.FORECAST_SCHEMAS["backtest"].names
    origins = sorted(set(bt["origin"]))
    # 48 history months: training 24, 27, 30, 33, 36 months, each with a full year ahead.
    assert origins == [g.add_months(dt.date(2023, 1, 1), n) for n in (24, 27, 30, 33, 36)]
    assert len(bt) == 5 * 12 * (1 + len(g.SKILL_TYPES))
    assert max(bt["month"]) == dt.date(2026, 12, 1)  # never past the history
    for origin, grp in bt.groupby("origin"):
        assert min(grp["month"]) == origin and grp["month"].nunique() == 12


def test_backtest_actuals_and_top_down_rows_add_up(
    built: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, bt = built
    skills = bt[bt["skill_type"] != f.ALL].groupby(["origin", "month"])[["actual", "model"]].sum()
    agg = bt[bt["skill_type"] == f.ALL].set_index(["origin", "month"])[["actual", "model"]]
    pd.testing.assert_frame_equal(skills, agg)


def test_backtest_on_a_toy_series_by_hand() -> None:
    """Pure repeating season, no noise: seasonal naive is exact; values check by hand."""
    months = [g.add_months(dt.date(2020, 1, 1), k) for k in range(39)]
    season = np.array([30, 10, 10, 10, 10, 20, 30, 30, 10, 10, 10, 10], dtype=float)
    a = np.resize(season, 39)
    counts = pd.DataFrame({"A": a, "B": 3 * a}, index=months)
    bt = f.run_backtest(counts)
    # 39 months: origins at 24 and 27 (27 + 12 = 39); 2 x 12 x (ALL, A, B).
    assert sorted(set(bt["origin"])) == [months[24], months[27]]
    assert len(bt) == 2 * 12 * 3
    np.testing.assert_allclose(bt["naive"], bt["actual"])
    row = bt[(bt["origin"] == months[24]) & (bt["month"] == months[30]) & (bt["skill_type"] == "B")]
    assert row["actual"].item() == 3 * 30  # July
    ab = bt[bt["skill_type"] != f.ALL]
    total = bt[bt["skill_type"] == f.ALL]["model"].to_numpy()
    np.testing.assert_allclose(ab[ab["skill_type"] == "A"]["model"], total * 0.25)
    summary = f.backtest_summary(bt).set_index("skill_type")
    assert summary.loc[f.ALL, "wape_naive"] == 0.0
    assert list(summary.index) == [f.ALL, "A", "B"]


def test_wape_by_hand() -> None:
    assert f.wape([10, 20, 30], [12, 18, 30]) == pytest.approx(4 / 60)
    assert f.wape([0, 0], [1, 2]) != f.wape([0, 0], [1, 2])  # nan: undefined


def test_backtest_summary_by_hand() -> None:
    d = dt.date(2025, 1, 1)
    bt = pd.DataFrame(
        {
            "origin": [d] * 4,
            "month": [d, g.add_months(d, 1)] * 2,
            "skill_type": ["X", "X", f.ALL, f.ALL],
            "actual": [10.0, 30.0, 100.0, 100.0],
            "model": [12.0, 27.0, 90.0, 100.0],
            "naive": [10.0, 20.0, 120.0, 80.0],
            "method": ["ets"] * 4,
        }
    )
    s = f.backtest_summary(bt).set_index("skill_type")
    assert list(s.index) == [f.ALL, "X"]
    assert s.loc["X", "wape_model"] == pytest.approx(5 / 40)
    assert s.loc["X", "wape_naive"] == pytest.approx(10 / 40)
    assert s.loc[f.ALL, "wape_model"] == pytest.approx(10 / 200)
    assert bool(s.loc["X", "model_beats_naive"]) and bool(s.loc[f.ALL, "model_beats_naive"])


def test_model_beats_seasonal_naive_on_aggregate(built: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Not a law: if this ever fails, the README must say so (PRD M2)."""
    agg = f.backtest_summary(built[1]).iloc[0]
    assert agg["skill_type"] == f.ALL and agg["model_beats_naive"]


def test_shorter_history_gives_one_origin() -> None:
    p = apply_overrides(P, {"demand.history_months": 36})
    _, bt = f.build_forecast(g.generate_world(p), p)
    assert bt["origin"].nunique() == 1


# --- determinism, I/O, runtime ----------------------------------------------------------------


def test_forecast_is_deterministic_and_round_trips(
    world: g.World, built: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    again = f.build_forecast(world, P)
    pd.testing.assert_frame_equal(built[0], again[0])
    pd.testing.assert_frame_equal(built[1], again[1])
    paths_a = f.write_forecast(*built, tmp_path / "a")
    paths_b = f.write_forecast(*again, tmp_path / "b")
    for name in paths_a:
        assert paths_a[name].read_bytes() == paths_b[name].read_bytes()
    back = f.read_forecast(tmp_path / "a")
    pd.testing.assert_frame_equal(back[0], built[0])
    pd.testing.assert_frame_equal(back[1], built[1])


def test_forecast_and_plan_run_well_within_budget(world: g.World) -> None:
    from scout_planner import plan

    started = time.perf_counter()
    fc, _ = f.build_forecast(world, P)
    plan.build_capacity_plan(world.scouts, world.scout_unavailability, fc, P)
    assert time.perf_counter() - started < 30  # ARCHITECTURE section 7 budget (~0.5 s)
