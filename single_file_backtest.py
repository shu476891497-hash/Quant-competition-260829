"""一个文件完成 Factor → Signal → Position → PnL → Metrics。

队员通常只需要修改 ``make_factor``。运行：

    pip install pandas numpy
    python single_file_backtest.py

真实数据 CSV 格式：timestamp,instrument,close。时间必须从早到晚，不能重复。
本文件使用收盘价决策、下一根 bar 收盘成交，因此至少延迟一根 bar。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ======================== 队员主要修改这里 ========================


def make_factor(close: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.DataFrame:
    """示例：快慢 EMA 趋势因子。输出必须与 close 同 index、同 columns。"""
    if not 0 < fast < slow:
        raise ValueError("require 0 < fast < slow")
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return fast_ema.div(slow_ema).sub(1.0)


def make_signal(factor: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """正因子做多、负因子做空；可以按研究假设改阈值。"""
    signal = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    signal[factor > threshold] = 1.0
    signal[factor < -threshold] = -1.0
    return signal


# ======================== 统一回测口径，通常不要修改 ========================


@dataclass(frozen=True)
class Contract:
    multiplier: float
    tick_size: float
    commission: float = 0.0  # 每张、每边、基础货币
    slippage_ticks: float = 1.0  # 每次交易的不利滑点


def load_close_csv(path: str | Path) -> pd.DataFrame:
    """读取 timestamp,instrument,close 的长表 CSV，转换为价格矩阵。"""
    raw = pd.read_csv(path)
    required = {"timestamp", "instrument", "close"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
    if raw[["timestamp", "instrument"]].duplicated().any():
        raise ValueError("duplicate timestamp/instrument rows")
    return raw.pivot(index="timestamp", columns="instrument", values="close").sort_index()


def size_positions(
    close: pd.DataFrame,
    signal: pd.DataFrame,
    contracts: dict[str, Contract],
    capital: float,
    fx: pd.DataFrame | None = None,
    target_vol: float = 0.10,
    vol_lookback: int = 20,
    max_gross_leverage: float = 2.0,
) -> pd.DataFrame:
    """波动率目标仓位，并限制组合 gross notional。"""
    _validate_inputs(close, signal, contracts)
    fx = _fx_matrix(close, fx)
    annual_vol = close.pct_change(fill_method=None).rolling(
        vol_lookback, min_periods=vol_lookback
    ).std(ddof=0) * np.sqrt(252)
    multiplier = pd.Series({s: contracts[s].multiplier for s in close.columns})
    risk_budget = 1.0 / len(close.columns)
    contract_notional = close.mul(multiplier, axis=1).mul(fx)
    target = signal * (capital * target_vol * risk_budget) / (annual_vol * contract_notional)
    target = target.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    gross = target.abs().mul(contract_notional).sum(axis=1)
    scale = (capital * max_gross_leverage / gross.replace(0.0, np.nan)).clip(upper=1.0)
    target = target.mul(scale.fillna(1.0), axis=0)
    return target.map(np.trunc).astype(float)


def backtest(
    close: pd.DataFrame,
    target: pd.DataFrame,
    contracts: dict[str, Contract],
    initial_capital: float,
    fx: pd.DataFrame | None = None,
    execution_lag: int = 1,
) -> dict[str, object]:
    """向量化期货回测；收益率基于组合 NAV，不是期货价格。"""
    _validate_inputs(close, target, contracts)
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if execution_lag < 1:
        raise ValueError("execution_lag must be at least one bar")
    fx = _fx_matrix(close, fx)
    multiplier = pd.Series({s: contracts[s].multiplier for s in close.columns})

    position = target.shift(execution_lag, fill_value=0.0)
    trades = position.diff()
    trades.iloc[0] = position.iloc[0]
    gross_pnl = (
        position.shift(1, fill_value=0.0)
        .mul(close.diff().fillna(0.0))
        .mul(multiplier, axis=1)
        .mul(fx)
    )

    cost = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for symbol in close.columns:
        spec = contracts[symbol]
        per_contract = spec.commission + spec.slippage_ticks * spec.tick_size * spec.multiplier
        cost[symbol] = trades[symbol].abs() * per_contract * fx[symbol]

    net_pnl = gross_pnl.sum(axis=1) - cost.sum(axis=1)
    nav = initial_capital + net_pnl.cumsum()
    previous_nav = nav.shift(1, fill_value=initial_capital)
    if (nav <= 0).any():
        raise RuntimeError("NAV became non-positive")
    returns = net_pnl / previous_nav
    notional = position.mul(close).mul(multiplier, axis=1).mul(fx)
    traded_notional = trades.abs().mul(close).mul(multiplier, axis=1).mul(fx)
    turnover = traded_notional.sum(axis=1) / previous_nav
    gross_exposure = notional.abs().sum(axis=1) / nav

    return {
        "factor_target": target,
        "position": position,
        "trades": trades,
        "gross_pnl": gross_pnl,
        "cost": cost,
        "net_pnl": net_pnl,
        "nav": nav,
        "returns": returns,
        "notional": notional,
        "turnover": turnover,
        "metrics": performance_metrics(returns, nav, turnover, gross_exposure, cost),
    }


def performance_metrics(
    returns: pd.Series,
    nav: pd.Series,
    turnover: pd.Series,
    gross_exposure: pd.Series,
    cost: pd.DataFrame,
) -> dict[str, float]:
    annual_return = float(returns.mean() * 252)
    annual_vol = float(returns.std(ddof=1) * np.sqrt(252))
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    return {
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": annual_return / annual_vol if annual_vol > 0 else 0.0,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0,
        "average_turnover": float(turnover.mean()),
        "max_gross_exposure": float(gross_exposure.max()),
        "total_cost": float(cost.to_numpy().sum()),
    }


def _validate_inputs(
    close: pd.DataFrame, values: pd.DataFrame, contracts: dict[str, Contract]
) -> None:
    if close.empty or not close.index.is_monotonic_increasing or close.index.has_duplicates:
        raise ValueError("close needs a non-empty, unique, increasing index")
    if not close.index.equals(values.index) or list(close.columns) != list(values.columns):
        raise ValueError("close and factor/signal/position must align exactly")
    if set(close.columns) != set(contracts):
        raise ValueError("contracts must match close columns")
    if not np.isfinite(close.to_numpy(dtype=float)).all():
        raise ValueError("close must contain finite values")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("factor/signal/position must contain finite values")


def _fx_matrix(close: pd.DataFrame, fx: pd.DataFrame | None) -> pd.DataFrame:
    if fx is None:
        return pd.DataFrame(1.0, index=close.index, columns=close.columns)
    if not close.index.equals(fx.index) or list(close.columns) != list(fx.columns):
        raise ValueError("fx must align exactly with close")
    if not np.isfinite(fx.to_numpy(dtype=float)).all() or (fx <= 0).any().any():
        raise ValueError("fx must be finite and positive")
    return fx.astype(float)


def demo() -> None:
    """无需真实数据即可运行的 MES/MNQ 演示。"""
    rng = np.random.default_rng(7)
    index = pd.date_range("2022-01-01", periods=750, freq="B", tz="UTC")
    random_returns = pd.DataFrame(
        rng.normal([0.0002, 0.00025], [0.008, 0.011], size=(len(index), 2)),
        index=index,
        columns=["MES", "MNQ"],
    )
    close = (1.0 + random_returns).cumprod().mul([4800.0, 16800.0])
    contracts = {
        "MES": Contract(multiplier=5.0, tick_size=0.25, commission=1.0),
        "MNQ": Contract(multiplier=2.0, tick_size=0.25, commission=1.0),
    }
    fx = pd.DataFrame(7.8, index=index, columns=close.columns)
    factor = make_factor(close)
    signal = make_signal(factor)
    target = size_positions(close, signal, contracts, 1_000_000.0, fx)
    result = backtest(close, target, contracts, 1_000_000.0, fx)
    print("\n=== QuantCTA 单文件回测结果 ===")
    print(pd.Series(result["metrics"]).round(4))


if __name__ == "__main__":
    demo()
