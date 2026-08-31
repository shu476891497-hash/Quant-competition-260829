"""Futures curve and carry factors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_curve_carry(
    front: pd.DataFrame,
    back: pd.DataFrame,
    front_expiry: pd.DataFrame,
    back_expiry: pd.DataFrame,
) -> pd.DataFrame:
    """Return annualized log carry between the first and second contracts.

    The sign convention is positive in backwardation (front > back) and
    negative in contango (front < back). All four inputs must be point-in-time
    snapshots with identical index and columns.
    """

    _require_same_shape(front, back, front_expiry, back_expiry)
    if ((front <= 0) | (back <= 0)).any().any():
        raise ValueError("front and back prices must be positive")

    days = (back_expiry - front_expiry) / np.timedelta64(1, "D")
    days = days.astype(float).where(lambda values: values > 0)
    carry = np.log(front / back) * 365.0 / days
    return carry.replace([np.inf, -np.inf], np.nan)


def curve_carry_zscore(
    front: pd.DataFrame,
    back: pd.DataFrame,
    front_expiry: pd.DataFrame,
    back_expiry: pd.DataFrame,
    window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Standardize carry against information available before the current bar."""

    if window <= 1 or not 2 <= min_periods <= window:
        raise ValueError("require 2 <= min_periods <= window")
    carry = annualized_curve_carry(front, back, front_expiry, back_expiry)
    mean = carry.shift(1).rolling(window, min_periods=min_periods).mean()
    std = carry.shift(1).rolling(window, min_periods=min_periods).std(ddof=0)
    return carry.sub(mean).div(std.where(std > 0))


def annualized_curve_curvature(
    front: pd.DataFrame,
    middle: pd.DataFrame,
    back: pd.DataFrame,
    front_expiry: pd.DataFrame,
    middle_expiry: pd.DataFrame,
    back_expiry: pd.DataFrame,
) -> pd.DataFrame:
    """Return front-middle carry minus middle-back carry.

    Contract spacing is normalized separately on each curve segment, so an
    irregular expiry gap cannot masquerade as curvature.
    """

    first_segment = annualized_curve_carry(
        front, middle, front_expiry, middle_expiry
    )
    second_segment = annualized_curve_carry(
        middle, back, middle_expiry, back_expiry
    )
    return first_segment - second_segment


def normalized_curve_curvature(
    front: pd.DataFrame,
    middle: pd.DataFrame,
    back: pd.DataFrame,
    front_expiry: pd.DataFrame,
    middle_expiry: pd.DataFrame,
    back_expiry: pd.DataFrame,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Return dimensionless curvature while preserving its economic sign."""

    first_segment = annualized_curve_carry(
        front, middle, front_expiry, middle_expiry
    )
    second_segment = annualized_curve_carry(
        middle, back, middle_expiry, back_expiry
    )
    scale = first_segment.abs() + second_segment.abs()
    return (first_segment - second_segment).div(scale.where(scale > epsilon))


def open_interest_weighted_maturity(
    days_to_expiry: pd.DataFrame, open_interest: pd.DataFrame
) -> pd.Series:
    """Return the point-in-time OI-weighted average days to expiry per row."""

    _require_same_shape(days_to_expiry, open_interest)
    if (days_to_expiry.dropna() <= 0).any().any():
        raise ValueError("days_to_expiry must be positive")
    if (open_interest.dropna() < 0).any().any():
        raise ValueError("open_interest cannot be negative")
    valid = days_to_expiry.notna() & open_interest.notna()
    weights = open_interest.where(valid)
    denominator = weights.sum(axis=1).where(lambda values: values > 0)
    return days_to_expiry.mul(weights).sum(axis=1).div(denominator)


def front_open_interest_share(open_interest: pd.DataFrame) -> pd.Series:
    """Return front-contract OI divided by OI across the supplied curve."""

    if open_interest.empty or open_interest.shape[1] < 2:
        raise ValueError("open_interest must contain at least two contracts")
    if (open_interest.dropna() < 0).any().any():
        raise ValueError("open_interest cannot be negative")
    total = open_interest.sum(axis=1).where(lambda values: values > 0)
    return open_interest.iloc[:, 0].div(total)


def next_to_front_open_interest_ratio(open_interest: pd.DataFrame) -> pd.Series:
    """Return log(next-contract OI / front-contract OI)."""

    if open_interest.empty or open_interest.shape[1] < 2:
        raise ValueError("open_interest must contain at least two contracts")
    front = open_interest.iloc[:, 0]
    second = open_interest.iloc[:, 1]
    if ((front.dropna() <= 0).any() or (second.dropna() <= 0).any()):
        raise ValueError("front and next open interest must be positive")
    return np.log(second / front)


def volume_minus_open_interest_maturity(
    days_to_expiry: pd.DataFrame,
    volume: pd.DataFrame,
    open_interest: pd.DataFrame,
) -> pd.Series:
    """Return volume-weighted minus OI-weighted curve maturity.

    Positive values mean today's trading flow is concentrated farther along
    the curve than the existing stock of open positions.
    """

    _require_same_shape(days_to_expiry, volume, open_interest)
    if (volume.dropna() < 0).any().any():
        raise ValueError("volume cannot be negative")
    volume_maturity = open_interest_weighted_maturity(days_to_expiry, volume)
    oi_maturity = open_interest_weighted_maturity(days_to_expiry, open_interest)
    return volume_maturity - oi_maturity


def volume_to_open_interest(
    volume: pd.DataFrame, open_interest: pd.DataFrame
) -> pd.DataFrame:
    """Return aggregate contract volume divided by aggregate open interest."""

    _require_same_shape(volume, open_interest)
    if (volume.dropna() < 0).any().any():
        raise ValueError("volume cannot be negative")
    if (open_interest.dropna() <= 0).any().any():
        raise ValueError("open_interest must be positive")
    return volume / open_interest


def _require_same_shape(reference: pd.DataFrame, *others: pd.DataFrame) -> None:
    if reference.empty:
        raise ValueError("inputs cannot be empty")
    for values in others:
        if not reference.index.equals(values.index) or list(reference.columns) != list(
            values.columns
        ):
            raise ValueError("all inputs must have identical index and columns")
