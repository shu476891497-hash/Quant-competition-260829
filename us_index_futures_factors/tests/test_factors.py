from __future__ import annotations

import numpy as np
import pandas as pd

from futures_factors.factors import trailing_zscore


def test_trailing_zscore_is_prefix_invariant() -> None:
    series = pd.Series(np.sin(np.arange(200) / 7) + np.arange(200) / 100)
    full = trailing_zscore(series, window=40, min_periods=20)
    prefix = trailing_zscore(series.iloc[:120], window=40, min_periods=20)
    pd.testing.assert_series_equal(full.iloc[:120], prefix)


def test_future_perturbation_cannot_change_past_factor() -> None:
    series = pd.Series(np.arange(100, dtype=float))
    changed = series.copy()
    changed.iloc[80:] += 10_000
    base = trailing_zscore(series, window=20, min_periods=10)
    perturbed = trailing_zscore(changed, window=20, min_periods=10)
    pd.testing.assert_series_equal(base.iloc[:80], perturbed.iloc[:80])


def test_trailing_zscore_clips_outliers() -> None:
    series = pd.Series([1.0] * 20 + [1000.0])
    result = trailing_zscore(series, window=21, min_periods=10, clip=3.0)
    assert result.iloc[-1] == 3.0
