"""Run the complete research chain on deterministic synthetic futures prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcta import BacktestConfig, CostModel, InstrumentSpec, run_backtest
from quantcta.factors import dual_ema_momentum
from quantcta.portfolio import cap_gross_notional, volatility_target_positions
from quantcta.signals import sign_signal


def main() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-01", periods=500, freq="B", tz="UTC")
    returns = pd.DataFrame(
        rng.normal([0.0002, 0.00025], [0.008, 0.011], size=(len(index), 2)),
        index=index,
        columns=["MES", "MNQ"],
    )
    prices = (1.0 + returns).cumprod().mul([4800.0, 16800.0])
    specs = {
        "MES": InstrumentSpec("MES", multiplier=5.0, tick_size=0.25),
        "MNQ": InstrumentSpec("MNQ", multiplier=2.0, tick_size=0.25),
    }
    costs = {
        symbol: CostModel(commission_per_contract=1.0, slippage_ticks=1.0)
        for symbol in prices.columns
    }
    fx_rates = pd.DataFrame(7.8, index=prices.index, columns=prices.columns)
    factor = dual_ema_momentum(prices, fast=20, slow=100)
    signal = sign_signal(factor)
    targets = volatility_target_positions(
        prices,
        signal,
        specs,
        capital=1_000_000.0,
        target_volatility=0.10,
        lookback=20,
        fx_rates=fx_rates,
    )
    targets = cap_gross_notional(targets, prices, specs, 1_000_000.0, 2.0, fx_rates=fx_rates)
    result = run_backtest(
        prices,
        targets,
        specs,
        costs,
        BacktestConfig(initial_capital=1_000_000.0, base_currency="HKD"),
        fx_rates=fx_rates,
    )
    print(pd.Series(result.metrics).round(4))


if __name__ == "__main__":
    main()
