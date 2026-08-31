"""Uniform factor-to-forward-return evaluation for the whole team."""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt

import numpy as np
import pandas as pd


def forward_returns(
    prices: pd.DataFrame,
    horizon: int,
    *,
    availability_lag: int = 1,
    contract_ids: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create tradable forward log returns without using the decision-bar close.

    A factor observed on row t enters at t plus availability_lag and exits
    horizon rows later. With futures contract IDs supplied, observations
    crossing a roll between decision, entry, and exit are set to NaN.
    """

    _validate_numeric_frame(prices, "prices")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if availability_lag < 1:
        raise ValueError("availability_lag must be at least one bar")
    if (prices <= 0).any().any():
        raise ValueError("prices must be positive")

    entry = prices.shift(-availability_lag)
    exit_prices = prices.shift(-(availability_lag + horizon))
    labels = np.log(exit_prices / entry)
    if contract_ids is not None:
        _require_aligned(prices, contract_ids)
        entry_contract = contract_ids.shift(-availability_lag)
        exit_contract = contract_ids.shift(-(availability_lag + horizon))
        same_contract = contract_ids.eq(entry_contract) & entry_contract.eq(exit_contract)
        labels = labels.where(same_contract)
    return labels


def evaluate_factor(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 5, 21),
    availability_lag: int = 1,
    contract_ids: pd.DataFrame | None = None,
    factor_name: str = "factor",
    min_samples: int = 100,
) -> pd.DataFrame:
    """Return IC, RankIC, overlap-robust t-stat and Q5-Q1 for each instrument."""

    _validate_numeric_frame(factor, "factor")
    _validate_numeric_frame(prices, "prices")
    _require_aligned(prices, factor)
    if min_samples < 20:
        raise ValueError("min_samples must be at least 20")
    requested_horizons = tuple(dict.fromkeys(int(value) for value in horizons))
    if not requested_horizons:
        raise ValueError("horizons cannot be empty")

    records: list[dict[str, float | int | str]] = []
    for horizon in requested_horizons:
        labels = forward_returns(
            prices,
            horizon,
            availability_lag=availability_lag,
            contract_ids=contract_ids,
        )
        for symbol in prices.columns:
            sample = pd.concat(
                {"factor": factor[symbol], "forward_return": labels[symbol]}, axis=1
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if len(sample) < min_samples or sample["factor"].nunique() < 2:
                continue
            x = sample["factor"].to_numpy(dtype=float)
            y = sample["forward_return"].to_numpy(dtype=float)
            ic = float(np.corrcoef(x, y)[0, 1])
            rank_ic = float(
                sample["factor"].rank(method="average").corr(
                    sample["forward_return"].rank(method="average")
                )
            )
            low, high = np.quantile(x, [0.2, 0.8])
            spread = float(np.mean(y[x >= high]) - np.mean(y[x <= low])) * 10_000.0
            records.append(
                {
                    "factor": factor_name,
                    "symbol": str(symbol),
                    "horizon": horizon,
                    "n": len(sample),
                    "ic": ic,
                    "rank_ic": rank_ic,
                    "newey_west_t": _newey_west_ic_t(x, y, horizon - 1),
                    "q5_minus_q1_bps": spread,
                    "availability_lag": availability_lag,
                }
            )
    return pd.DataFrame.from_records(records)


def _newey_west_ic_t(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)
    if x_std <= 0 or y_std <= 0:
        return float("nan")
    products = ((x - np.mean(x)) / x_std) * ((y - np.mean(y)) / y_std)
    centered = products - np.mean(products)
    n = len(products)
    variance = float(np.dot(centered, centered) / n)
    max_lag = min(max(lag, 0), n - 2)
    for offset in range(1, max_lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        variance += 2.0 * (1.0 - offset / (max_lag + 1.0)) * covariance
    if variance <= 0:
        return float("nan")
    return float(np.mean(products) / sqrt(variance / n))


def _validate_numeric_frame(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} columns must be unique")
    try:
        frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


def _require_aligned(reference: pd.DataFrame, other: pd.DataFrame) -> None:
    if not reference.index.equals(other.index) or list(reference.columns) != list(
        other.columns
    ):
        raise ValueError("inputs must have identical index and ordered columns")
