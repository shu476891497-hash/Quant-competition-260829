"""Reference trend factors."""

from __future__ import annotations

import pandas as pd


def dual_ema_momentum(close: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.DataFrame:
    """Return the relative fast/slow EMA spread for each instrument.

    The value at t uses observations only through t. The engine applies the
    mandatory execution lag before the resulting target can be held.
    """

    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("require 0 < fast < slow")
    if close.empty or close.isna().all().all():
        raise ValueError("close cannot be empty")
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return fast_ema.div(slow_ema).sub(1.0)
