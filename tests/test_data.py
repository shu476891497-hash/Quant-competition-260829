import pandas as pd
import pytest

from quantcta.data.schema import point_in_time_view, validate_bars


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2025-01-02 21:00:00+00:00", "2025-01-03 21:00:00+00:00"],
            "available_at": ["2025-01-02 21:00:01+00:00", "2025-01-03 21:00:02+00:00"],
            "instrument": ["MES", "MES"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10, 20],
        }
    )


def test_point_in_time_view_uses_available_at() -> None:
    result = point_in_time_view(_bars(), pd.Timestamp("2025-01-03 21:00:01", tz="UTC"))
    assert len(result) == 1
    assert result.iloc[0]["close"] == 101.0


def test_duplicate_bars_are_rejected() -> None:
    bars = pd.concat([_bars(), _bars().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_bars(bars)
