import numpy as np
import pandas as pd
import pandas.testing as pdt

from quantcta import InstrumentSpec, run_backtest
from quantcta.factors import dual_ema_momentum
from quantcta.signals import sign_signal


def test_future_mutation_cannot_change_past_factor_or_positions() -> None:
    index = pd.date_range("2020-01-01", periods=160, freq="D", tz="UTC")
    close = pd.DataFrame({"X": np.linspace(100.0, 150.0, len(index))}, index=index)
    mutated = close.copy()
    cutoff = index[130]
    mutated.loc[mutated.index > cutoff, "X"] *= 10.0
    factor = dual_ema_momentum(close, fast=10, slow=30)
    mutated_factor = dual_ema_momentum(mutated, fast=10, slow=30)
    pdt.assert_frame_equal(factor.loc[:cutoff], mutated_factor.loc[:cutoff])
    signal = sign_signal(factor)
    mutated_signal = sign_signal(mutated_factor)
    specs = {"X": InstrumentSpec("X", multiplier=5.0, tick_size=0.25)}
    result = run_backtest(close, signal, specs)
    mutated_result = run_backtest(mutated, mutated_signal, specs)
    pdt.assert_frame_equal(result.positions.loc[:cutoff], mutated_result.positions.loc[:cutoff])
