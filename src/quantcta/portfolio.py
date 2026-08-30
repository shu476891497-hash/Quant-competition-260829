"""Transparent conversion from signals to futures contract targets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from quantcta.config import InstrumentSpec


def volatility_target_positions(
    close: pd.DataFrame,
    signal: pd.DataFrame,
    specs: Mapping[str, InstrumentSpec],
    capital: float,
    target_volatility: float = 0.10,
    lookback: int = 20,
    annualization: int = 252,
    risk_budget: pd.Series | None = None,
    fx_rates: pd.DataFrame | None = None,
    integer_contracts: bool = True,
) -> pd.DataFrame:
    """Size each signal by trailing annualized volatility and contract notional."""

    _require_aligned(close, signal)
    if capital <= 0 or target_volatility <= 0:
        raise ValueError("capital and target_volatility must be positive")
    if lookback < 2 or annualization <= 0:
        raise ValueError("invalid volatility settings")
    _require_specs(close.columns, specs)
    fx_rates = _resolve_fx(close, fx_rates)
    if risk_budget is None:
        risk_budget = pd.Series(1.0 / len(close.columns), index=close.columns)
    risk_budget = risk_budget.reindex(close.columns)
    if risk_budget.isna().any() or (risk_budget < 0).any() or risk_budget.sum() > 1.0 + 1e-12:
        raise ValueError("risk_budget must be complete, non-negative, and sum to at most one")
    annualized_vol = close.pct_change(fill_method=None).rolling(lookback, min_periods=lookback).std(
        ddof=0
    ) * np.sqrt(annualization)
    multipliers = pd.Series({symbol: specs[symbol].multiplier for symbol in close.columns})
    risk_cash = capital * target_volatility * risk_budget
    contract_notional_base = close.mul(multipliers, axis=1).mul(fx_rates)
    positions = signal.mul(risk_cash, axis=1).div(annualized_vol * contract_notional_base)
    positions = positions.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if integer_contracts:
        positions = positions.map(np.trunc)
    return positions.astype(float)


def cap_gross_notional(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    specs: Mapping[str, InstrumentSpec],
    capital: float,
    max_gross_leverage: float,
    fx_rates: pd.DataFrame | None = None,
    integer_contracts: bool = True,
) -> pd.DataFrame:
    """Proportionally scale target positions to a gross-notional limit."""

    _require_aligned(prices, positions)
    _require_specs(prices.columns, specs)
    if capital <= 0 or max_gross_leverage <= 0:
        raise ValueError("capital and max_gross_leverage must be positive")
    fx_rates = _resolve_fx(prices, fx_rates)
    multipliers = pd.Series({symbol: specs[symbol].multiplier for symbol in prices.columns})
    gross = positions.abs().mul(prices).mul(multipliers, axis=1).mul(fx_rates).sum(axis=1)
    limit = capital * max_gross_leverage
    scale = (limit / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    capped = positions.mul(scale, axis=0)
    if integer_contracts:
        capped = capped.map(np.trunc)
    return capped.astype(float)


def _require_aligned(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if not left.index.equals(right.index) or list(left.columns) != list(right.columns):
        raise ValueError("input matrices must have identical index and columns")


def _require_specs(columns: pd.Index, specs: Mapping[str, InstrumentSpec]) -> None:
    missing = set(columns).difference(specs)
    if missing:
        raise ValueError(f"missing instrument specs: {sorted(missing)}")


def _resolve_fx(prices: pd.DataFrame, fx_rates: pd.DataFrame | None) -> pd.DataFrame:
    if fx_rates is None:
        return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    _require_aligned(prices, fx_rates)
    if fx_rates.isna().any().any() or (fx_rates <= 0).any().any():
        raise ValueError("fx_rates must be finite and positive")
    return fx_rates.astype(float)
