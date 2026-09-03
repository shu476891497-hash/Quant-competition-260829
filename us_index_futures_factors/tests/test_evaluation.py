from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futures_factors.evaluation import backtest_symbol, performance_metrics
from futures_factors.low_turnover import OverlaySpec, add_cash_collateral_returns, build_overlay


def test_backtest_uses_previous_day_signal_and_charges_turnover() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="B"),
            "symbol": "ES",
            "log_return": [0.0, 0.10, -0.05, 0.02],
            "composite": [1.0, -1.0, -1.0, 0.0],
        }
    )
    result = backtest_symbol(frame, cost_bps=1.0)
    assert result.loc[0, "position"] == 0.0
    assert result.loc[1, "position"] == 1.0
    assert result.loc[2, "position"] == -1.0
    assert result.loc[1, "gross_log_return"] == pytest.approx(0.10)
    assert result.loc[2, "gross_log_return"] == pytest.approx(0.05)
    assert result.loc[2, "turnover"] == pytest.approx(2.0)
    assert result.loc[2, "net_log_return"] == pytest.approx(0.05 - 0.0002)


def test_performance_metrics_uses_geometric_annual_return() -> None:
    returns = pd.Series([np.log(1.01)] * 252)
    metrics = performance_metrics(returns, turnover=pd.Series([0.0] * 252))
    assert metrics["annual_return"] == pytest.approx(1.01**252 - 1, rel=1e-9)
    assert metrics["max_drawdown"] == pytest.approx(0.0)


def test_performance_metrics_counts_loss_before_first_high_water_mark() -> None:
    metrics = performance_metrics(pd.Series([np.log(0.9), np.log(1.0)]))
    assert metrics["max_drawdown"] == pytest.approx(-0.1)


def _overlay_fixture(periods: int = 260) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    price_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    for offset, symbol in enumerate(("ES", "NQ", "RTY", "YM")):
        close = 100.0 * np.exp(np.arange(periods) * (0.0005 + offset * 0.0001))
        log_return = np.r_[np.nan, np.diff(np.log(close))]
        for date, value, ret in zip(dates, close, log_return, strict=True):
            price_rows.append({"date": date, "symbol": symbol, "close": value, "log_return": ret})
            factor_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": value,
                    "liquidity_factor": 1.0,
                }
            )
    return pd.DataFrame(price_rows), pd.DataFrame(factor_rows)


def test_low_turnover_overlay_is_lagged_and_causal() -> None:
    prices, factors = _overlay_fixture()
    spec = OverlaySpec(
        name="test",
        top_n=2,
        trend_windows=(5, 10, 20),
        liquidity_halflife=2,
        allocation_interval=5,
        volatility_window=10,
        volatility_interval=5,
    )
    full, positions = build_overlay(prices, factors, spec)
    cutoff = prices["date"].sort_values().unique()[199]
    prefix_prices = prices.loc[prices["date"] <= cutoff]
    prefix_factors = factors.loc[factors["date"] <= cutoff]
    prefix, prefix_positions = build_overlay(prefix_prices, prefix_factors, spec)
    pd.testing.assert_frame_equal(full.iloc[:200], prefix)
    pd.testing.assert_frame_equal(
        positions.loc[positions["date"] <= prefix["date"].max()].reset_index(drop=True),
        prefix_positions,
    )
    first_position = positions.loc[positions["position"] > 0, "date"].min()
    assert first_position > prices["date"].min()


def test_low_turnover_future_mutation_does_not_change_past() -> None:
    prices, factors = _overlay_fixture()
    spec = OverlaySpec(
        name="test",
        top_n=2,
        trend_windows=(5, 10, 20),
        liquidity_halflife=2,
        allocation_interval=5,
        volatility_window=10,
        volatility_interval=5,
    )
    cutoff = prices["date"].sort_values().unique()[199]
    baseline, _ = build_overlay(prices, factors, spec)
    mutated = prices.copy()
    mutated.loc[mutated["date"] > cutoff, "close"] *= 10.0
    changed, _ = build_overlay(mutated, factors, spec)
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["date"] <= cutoff].reset_index(drop=True),
        changed.loc[changed["date"] <= cutoff].reset_index(drop=True),
    )


def test_cash_collateral_rate_is_lagged_one_session() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    daily = pd.DataFrame(
        {
            "date": dates,
            "net_log_return": [0.0, 0.01, -0.01, 0.0],
            "one_way_turnover": 0.0,
        }
    )
    cash_rate = pd.DataFrame(
        {
            "date": dates,
            "annual_rate_pct": [2.52, 5.04, 7.56, 10.08],
        }
    )
    result = add_cash_collateral_returns(daily, cash_rate)
    assert np.isnan(result.loc[0, "cash_collateral_log_return"])
    assert result.loc[1, "cash_rate_pct_known"] == pytest.approx(2.52)
    assert result.loc[1, "cash_collateral_log_return"] == pytest.approx(np.log1p(0.0001))
    assert result.loc[1, "total_account_net_log_return"] == pytest.approx(
        0.01 + np.log1p(0.0001)
    )
    assert result.loc[1, "excess_over_cash_log_return"] == pytest.approx(0.01)


def test_low_turnover_cost_sensitivity_changes_only_net_returns() -> None:
    prices, factors = _overlay_fixture()
    spec = OverlaySpec(
        name="test",
        top_n=2,
        trend_windows=(5, 10, 20),
        liquidity_halflife=2,
        allocation_interval=5,
        volatility_window=10,
        volatility_interval=5,
    )
    zero_cost, zero_positions = build_overlay(prices, factors, spec, cost_bps=0.0)
    five_bps, five_positions = build_overlay(prices, factors, spec, cost_bps=5.0)
    pd.testing.assert_frame_equal(zero_positions, five_positions)
    pd.testing.assert_series_equal(
        zero_cost["gross_log_return"],
        five_bps["gross_log_return"],
    )
    expected = zero_cost["trading_notional"] * 5.0 / 10_000.0
    valid = zero_cost["net_log_return"].notna()
    np.testing.assert_allclose(
        (zero_cost.loc[valid, "net_log_return"] - five_bps.loc[valid, "net_log_return"]),
        expected.loc[valid],
    )
