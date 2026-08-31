import numpy as np
import pandas as pd

from quantcta.research import evaluate_factor, forward_returns


def test_forward_return_uses_next_bar_as_entry() -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    prices = pd.DataFrame({"ES": [100.0, 110.0, 121.0, 133.1, 146.41, 161.051]}, index=index)

    labels = forward_returns(prices, horizon=2, availability_lag=1)

    assert np.isclose(labels.iloc[0, 0], np.log(133.1 / 110.0))
    assert np.isclose(labels.iloc[1, 0], np.log(146.41 / 121.0))


def test_forward_return_rejects_same_bar_execution() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    prices = pd.DataFrame({"ES": [100.0, 101.0, 102.0, 103.0]}, index=index)

    try:
        forward_returns(prices, horizon=1, availability_lag=0)
    except ValueError as error:
        assert "at least one bar" in str(error)
    else:
        raise AssertionError("same-bar execution should be rejected")


def test_forward_return_drops_samples_crossing_contract_roll() -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    prices = pd.DataFrame({"ES": [100.0, 101.0, 102.0, 200.0, 202.0, 204.0]}, index=index)
    contracts = pd.DataFrame({"ES": ["ESH5", "ESH5", "ESH5", "ESM5", "ESM5", "ESM5"]}, index=index)

    labels = forward_returns(
        prices, horizon=1, availability_lag=1, contract_ids=contracts
    )

    assert labels.iloc[:3, 0].isna().tolist() == [False, True, True]
    assert labels.iloc[3, 0] == np.log(204.0 / 202.0)


def test_evaluate_factor_returns_team_standard_columns() -> None:
    index = pd.date_range("2020-01-01", periods=150, freq="D", tz="UTC")
    prices = pd.DataFrame({"NQ": 100.0 * np.exp(np.arange(150) * 0.001)}, index=index)
    factor = pd.DataFrame({"NQ": np.sin(np.arange(150) / 9.0)}, index=index)

    result = evaluate_factor(
        factor,
        prices,
        horizons=(1, 5),
        factor_name="curve_curvature",
        min_samples=20,
    )

    assert result["horizon"].tolist() == [1, 5]
    assert set(result.columns) == {
        "factor",
        "symbol",
        "horizon",
        "n",
        "ic",
        "rank_ic",
        "newey_west_t",
        "q5_minus_q1_bps",
        "availability_lag",
    }
