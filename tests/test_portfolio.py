import pandas as pd

from quantcta import InstrumentSpec
from quantcta.portfolio import cap_gross_notional


def test_gross_notional_cap_scales_positions() -> None:
    index = pd.date_range("2025-01-01", periods=1, tz="UTC")
    prices = pd.DataFrame({"A": [100.0], "B": [100.0]}, index=index)
    positions = pd.DataFrame({"A": [10.0], "B": [-10.0]}, index=index)
    specs = {
        "A": InstrumentSpec("A", multiplier=10.0, tick_size=1.0),
        "B": InstrumentSpec("B", multiplier=10.0, tick_size=1.0),
    }
    capped = cap_gross_notional(positions, prices, specs, 10_000.0, 1.0)
    assert capped.loc[index[0]].abs().tolist() == [5.0, 5.0]


def test_gross_notional_cap_converts_to_base_currency() -> None:
    index = pd.date_range("2025-01-01", periods=1, tz="UTC")
    prices = pd.DataFrame({"USD_FUT": [100.0]}, index=index)
    positions = pd.DataFrame({"USD_FUT": [10.0]}, index=index)
    fx_rates = pd.DataFrame({"USD_FUT": [8.0]}, index=index)
    specs = {"USD_FUT": InstrumentSpec("USD_FUT", multiplier=10.0, tick_size=1.0)}
    capped = cap_gross_notional(
        positions, prices, specs, capital=40_000.0, max_gross_leverage=1.0, fx_rates=fx_rates
    )
    assert capped.loc[index[0], "USD_FUT"] == 5.0
