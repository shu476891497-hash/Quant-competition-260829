"""Reusable factor-to-signal transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore_signal(
    factor: pd.DataFrame, lookback: int = 60, clip: float = 3.0
) -> pd.DataFrame:
    """Transform a factor to a bounded [-1, 1] signal using past rolling data."""

    if lookback < 2:
        raise ValueError("lookback must be at least two")
    if clip <= 0:
        raise ValueError("clip must be positive")
    mean = factor.rolling(lookback, min_periods=lookback).mean()
    std = factor.rolling(lookback, min_periods=lookback).std(ddof=0).replace(0.0, np.nan)
    zscore = factor.sub(mean).div(std).clip(-clip, clip)
    return zscore.div(clip).fillna(0.0)


def sign_signal(factor: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Convert a factor into -1/0/1 without changing its timestamps."""

    if threshold < 0:
        raise ValueError("threshold cannot be negative")
    result = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    result[factor > threshold] = 1.0
    result[factor < -threshold] = -1.0
    return result
