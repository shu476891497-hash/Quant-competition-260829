"""Small, explicit configuration objects used by the research core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    """Static contract metadata required for futures accounting."""

    symbol: str
    multiplier: float
    tick_size: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")


@dataclass(frozen=True)
class CostModel:
    """Per-contract, per-side futures transaction cost assumptions."""

    commission_per_contract: float = 0.0
    exchange_fee_per_contract: float = 0.0
    slippage_ticks: float = 0.0
    half_spread_ticks: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.commission_per_contract,
            self.exchange_fee_per_contract,
            self.slippage_ticks,
            self.half_spread_ticks,
        )
        if any(value < 0 for value in values):
            raise ValueError("transaction cost inputs cannot be negative")


@dataclass(frozen=True)
class BacktestConfig:
    """Settings that define one reproducible backtest run."""

    initial_capital: float = 1_000_000.0
    base_currency: str = "HKD"
    execution_lag: int = 1
    annualization: int = 252

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.execution_lag < 1:
            raise ValueError("execution_lag must be at least one bar")
        if self.annualization <= 0:
            raise ValueError("annualization must be positive")
