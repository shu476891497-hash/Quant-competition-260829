import pandas as pd

from single_file_backtest import Contract, backtest


def test_single_file_engine_has_lag_multiplier_and_costs() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    close = pd.DataFrame({"X": [100.0, 101.0, 103.0, 102.0]}, index=index)
    target = pd.DataFrame({"X": [1.0, 1.0, 0.0, 0.0]}, index=index)
    contracts = {"X": Contract(multiplier=10.0, tick_size=0.5, commission=2.0)}
    result = backtest(close, target, contracts, initial_capital=1_000.0)
    assert result["position"]["X"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert result["gross_pnl"]["X"].tolist() == [0.0, 0.0, 20.0, -10.0]
    assert result["cost"]["X"].tolist() == [0.0, 7.0, 0.0, 7.0]
