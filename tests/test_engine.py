import pandas as pd
import pytest

from quantcta import BacktestConfig, CostModel, InstrumentSpec, run_backtest


def test_futures_multiplier_costs_and_one_bar_lag() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    prices = pd.DataFrame({"TEST": [100.0, 101.0, 103.0, 102.0]}, index=index)
    targets = pd.DataFrame({"TEST": [1.0, 1.0, 0.0, 0.0]}, index=index)
    specs = {"TEST": InstrumentSpec("TEST", multiplier=10.0, tick_size=0.5)}
    costs = {"TEST": CostModel(2.0, 1.0, 1.0)}
    result = run_backtest(
        prices,
        targets,
        specs,
        costs,
        BacktestConfig(initial_capital=1_000.0, execution_lag=1),
    )
    assert result.positions["TEST"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert result.trades["TEST"].tolist() == [0.0, 1.0, 0.0, -1.0]
    assert result.gross_pnl_by_instrument["TEST"].tolist() == [0.0, 0.0, 20.0, -10.0]
    assert result.costs_by_instrument["TEST"].tolist() == [0.0, 8.0, 0.0, 8.0]
    assert result.nav.tolist() == pytest.approx([1000.0, 992.0, 1012.0, 994.0])


def test_same_bar_execution_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one bar"):
        BacktestConfig(execution_lag=0)


def test_shape_mismatch_fails_loudly() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    prices = pd.DataFrame({"A": [1.0, 2.0]}, index=index)
    targets = pd.DataFrame({"B": [0.0, 1.0]}, index=index)
    with pytest.raises(ValueError, match="identical"):
        run_backtest(prices, targets, {"A": InstrumentSpec("A", 1.0, 0.01)})
