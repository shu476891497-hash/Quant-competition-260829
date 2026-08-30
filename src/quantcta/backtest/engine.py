"""Small vectorized futures accounting engine with strict timing semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

import numpy as np
import pandas as pd

from quantcta.backtest.result import BacktestResult
from quantcta.config import BacktestConfig, CostModel, InstrumentSpec
from quantcta.metrics import calculate_metrics


def run_backtest(
    prices: pd.DataFrame,
    target_positions: pd.DataFrame,
    specs: Mapping[str, InstrumentSpec],
    costs: Mapping[str, CostModel] | None = None,
    config: BacktestConfig | None = None,
    fx_rates: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run a close-to-close vectorized futures backtest.

    Targets decided at t become actual at t + execution_lag. PnL from t-1 to t
    belongs to the actual position held at t-1. Costs are charged on delayed trades.
    """

    config = config or BacktestConfig()
    prices = _validate_matrix(prices, "prices")
    target_positions = _validate_matrix(target_positions, "target_positions")
    _require_same_shape(prices, target_positions)
    _require_specs(prices.columns, specs)
    costs = costs or {}
    unknown_costs = set(costs).difference(prices.columns)
    if unknown_costs:
        raise ValueError(f"cost models supplied for unknown instruments: {sorted(unknown_costs)}")
    if fx_rates is None:
        fx_rates = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    else:
        fx_rates = _validate_matrix(fx_rates, "fx_rates")
        _require_same_shape(prices, fx_rates)
        if (fx_rates <= 0).any().any():
            raise ValueError("fx_rates must be positive")

    positions = target_positions.shift(config.execution_lag, fill_value=0.0)
    trades = positions.diff()
    trades.iloc[0] = positions.iloc[0]
    multipliers = pd.Series({symbol: specs[symbol].multiplier for symbol in prices.columns})
    gross_pnl_by_instrument = (
        positions.shift(1, fill_value=0.0)
        .mul(prices.diff().fillna(0.0))
        .mul(multipliers, axis=1)
        .mul(fx_rates)
    )
    costs_by_instrument = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for symbol in prices.columns:
        model = costs.get(symbol, CostModel())
        spec = specs[symbol]
        cash_per_contract = (
            model.commission_per_contract
            + model.exchange_fee_per_contract
            + (model.slippage_ticks + model.half_spread_ticks) * spec.tick_size * spec.multiplier
        )
        costs_by_instrument[symbol] = trades[symbol].abs() * cash_per_contract * fx_rates[symbol]

    net_pnl = gross_pnl_by_instrument.sum(axis=1) - costs_by_instrument.sum(axis=1)
    nav = config.initial_capital + net_pnl.cumsum()
    previous_nav = nav.shift(1, fill_value=config.initial_capital)
    if (previous_nav <= 0).any() or (nav <= 0).any():
        raise RuntimeError("portfolio NAV became non-positive")
    returns = net_pnl.div(previous_nav)
    notional = positions.mul(prices).mul(multipliers, axis=1).mul(fx_rates)
    traded_notional = trades.abs().mul(prices).mul(multipliers, axis=1).mul(fx_rates)
    turnover = traded_notional.sum(axis=1).div(previous_nav)
    gross_exposure = notional.abs().sum(axis=1).div(nav)
    net_exposure = notional.sum(axis=1).div(nav)
    metrics = calculate_metrics(
        returns,
        nav,
        turnover,
        gross_exposure,
        float(costs_by_instrument.to_numpy().sum()),
        config.annualization,
    )
    manifest = {
        "engine": "quantcta-vectorized-v1",
        "config": asdict(config),
        "instruments": {symbol: asdict(specs[symbol]) for symbol in prices.columns},
        "costs": {symbol: asdict(costs.get(symbol, CostModel())) for symbol in prices.columns},
        "start": prices.index[0],
        "end": prices.index[-1],
        "rows": len(prices),
    }
    return BacktestResult(
        positions,
        trades,
        gross_pnl_by_instrument,
        costs_by_instrument,
        net_pnl,
        nav,
        returns,
        notional,
        turnover,
        gross_exposure,
        net_exposure,
        metrics,
        manifest,
    )


def _validate_matrix(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} columns must be unique")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain only finite values")
    return frame.astype(float).copy()


def _require_same_shape(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if not left.index.equals(right.index) or list(left.columns) != list(right.columns):
        raise ValueError("all matrices must have identical index and ordered columns")


def _require_specs(columns: pd.Index, specs: Mapping[str, InstrumentSpec]) -> None:
    if set(columns) != set(specs):
        missing = set(columns).difference(specs)
        extra = set(specs).difference(columns)
        raise ValueError(
            f"instrument specs mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
