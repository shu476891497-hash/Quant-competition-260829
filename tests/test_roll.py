import pandas as pd

from quantcta.data.roll import expand_root_positions


def test_contract_mapping_creates_explicit_roll_trade() -> None:
    index = pd.date_range("2025-09-10", periods=3, freq="D", tz="UTC")
    root_positions = pd.DataFrame({"ES": [1.0, 1.0, 1.0]}, index=index)
    mapping = pd.DataFrame({"ES": ["ESU5", "ESU5", "ESZ5"]}, index=index)
    actual = expand_root_positions(root_positions, mapping)
    assert actual["ESU5"].tolist() == [1.0, 1.0, 0.0]
    assert actual["ESZ5"].tolist() == [0.0, 0.0, 1.0]
    trades = actual.diff().fillna(actual)
    assert trades.loc[index[-1], "ESU5"] == -1.0
    assert trades.loc[index[-1], "ESZ5"] == 1.0
