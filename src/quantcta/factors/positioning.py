"""Open-interest and CFTC positioning factors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def open_interest_change(open_interest: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Return the log change in open interest over ``lookback`` bars."""

    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if open_interest.empty or (open_interest.dropna() <= 0).any().any():
        raise ValueError("open_interest must contain positive observations")
    return np.log(open_interest / open_interest.shift(lookback))


def price_oi_confirmation(
    close: pd.DataFrame,
    open_interest: pd.DataFrame,
    price_lookback: int = 20,
    oi_lookback: int = 5,
    scale_window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Measure whether changes in OI confirm the direction of price momentum.

    Rising OI produces a positive confirmation for an uptrend and a negative
    confirmation for a downtrend. Falling OI weakens or reverses that reading.
    The OI scale uses only observations before the current bar.
    """

    _require_aligned(close, open_interest)
    if price_lookback <= 0 or oi_lookback <= 0:
        raise ValueError("lookbacks must be positive")
    if scale_window <= 1 or not 2 <= min_periods <= scale_window:
        raise ValueError("require 2 <= min_periods <= scale_window")

    price_momentum = np.log(close / close.shift(price_lookback))
    oi_delta = open_interest_change(open_interest, oi_lookback)
    oi_scale = oi_delta.shift(1).rolling(
        scale_window, min_periods=min_periods
    ).std(ddof=0)
    scaled_oi_delta = oi_delta.div(oi_scale.where(oi_scale > 0))
    return np.sign(price_momentum) * scaled_oi_delta


def cot_net_share(
    long_positions: pd.DataFrame,
    short_positions: pd.DataFrame,
    total_open_interest: pd.DataFrame,
) -> pd.DataFrame:
    """Return a CFTC participant group's net position as a share of total OI.

    Input rows must be indexed by the date when the report became public, not
    the Tuesday report-as-of date. For the standard weekly COT release this
    normally means aligning the observation to its Friday publication time.
    """

    _require_aligned(long_positions, short_positions, total_open_interest)
    if (total_open_interest.dropna() <= 0).any().any():
        raise ValueError("total_open_interest must be positive")
    return long_positions.sub(short_positions).div(total_open_interest)


def cot_crowding_zscore(
    long_positions: pd.DataFrame,
    short_positions: pd.DataFrame,
    total_open_interest: pd.DataFrame,
    window: int = 156,
    min_periods: int = 52,
) -> pd.DataFrame:
    """Standardize CFTC net positioning against the preceding weekly history."""

    if window <= 1 or not 2 <= min_periods <= window:
        raise ValueError("require 2 <= min_periods <= window")
    share = cot_net_share(long_positions, short_positions, total_open_interest)
    mean = share.shift(1).rolling(window, min_periods=min_periods).mean()
    std = share.shift(1).rolling(window, min_periods=min_periods).std(ddof=0)
    return share.sub(mean).div(std.where(std > 0))


def align_cot_publication(
    report_values: pd.DataFrame, publication_lag_days: int = 3
) -> pd.DataFrame:
    """Move CFTC report-as-of dates to their earliest allowed publication date."""

    if report_values.empty:
        raise ValueError("report_values cannot be empty")
    if publication_lag_days < 0:
        raise ValueError("publication_lag_days cannot be negative")
    aligned = report_values.copy()
    aligned.index = pd.DatetimeIndex(aligned.index) + pd.Timedelta(
        days=publication_lag_days
    )
    if aligned.index.has_duplicates or not aligned.index.is_monotonic_increasing:
        raise ValueError("publication dates must be unique and increasing")
    return aligned


def _require_aligned(reference: pd.DataFrame, *others: pd.DataFrame) -> None:
    if reference.empty:
        raise ValueError("inputs cannot be empty")
    for values in others:
        if not reference.index.equals(values.index) or list(reference.columns) != list(
            values.columns
        ):
            raise ValueError("all inputs must have identical index and columns")
