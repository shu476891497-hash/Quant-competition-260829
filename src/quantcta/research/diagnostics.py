"""Small, causal diagnostics for factor robustness."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def causal_winsorize(
    factor: pd.DataFrame,
    *,
    window: int = 252,
    min_periods: int = 60,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Clip each observation using quantiles estimated strictly before it."""

    _validate_factor(factor)
    if window <= 1 or not 2 <= min_periods <= window:
        raise ValueError("require 2 <= min_periods <= window")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("require 0 <= lower_quantile < upper_quantile <= 1")
    history = factor.shift(1)
    lower = history.rolling(window, min_periods=min_periods).quantile(lower_quantile)
    upper = history.rolling(window, min_periods=min_periods).quantile(upper_quantile)
    return factor.clip(lower=lower, upper=upper, axis=None)


def factor_autocorrelation(
    factor: pd.DataFrame, lags: Iterable[int] = (1, 5, 21)
) -> pd.DataFrame:
    """Return time-series autocorrelation by instrument and lag."""

    _validate_factor(factor)
    requested_lags = tuple(dict.fromkeys(int(value) for value in lags))
    if not requested_lags or any(value <= 0 for value in requested_lags):
        raise ValueError("lags must contain positive integers")
    records: list[dict[str, float | int | str]] = []
    for symbol in factor.columns:
        values = factor[symbol]
        for lag in requested_lags:
            paired = pd.concat(
                {"current": values, "lagged": values.shift(lag)}, axis=1
            ).dropna()
            correlation = (
                float(paired["current"].corr(paired["lagged"]))
                if len(paired) >= 3
                else float("nan")
            )
            records.append(
                {
                    "symbol": str(symbol),
                    "lag": lag,
                    "n": len(paired),
                    "autocorrelation": correlation,
                }
            )
    return pd.DataFrame.from_records(records)


def _validate_factor(factor: pd.DataFrame) -> None:
    if not isinstance(factor, pd.DataFrame) or factor.empty:
        raise ValueError("factor must be a non-empty DataFrame")
    if factor.index.has_duplicates or not factor.index.is_monotonic_increasing:
        raise ValueError("factor index must be unique and increasing")
    if factor.columns.has_duplicates:
        raise ValueError("factor columns must be unique")
    try:
        values = factor.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("factor must be numeric") from error
    if np.isinf(values).any():
        raise ValueError("factor cannot contain infinite values")
