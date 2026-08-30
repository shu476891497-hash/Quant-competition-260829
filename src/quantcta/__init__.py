"""QuantCTA research and futures backtesting toolkit."""

from quantcta.backtest.engine import run_backtest
from quantcta.config import BacktestConfig, CostModel, InstrumentSpec

__all__ = ["BacktestConfig", "CostModel", "InstrumentSpec", "run_backtest"]
__version__ = "0.1.0"
