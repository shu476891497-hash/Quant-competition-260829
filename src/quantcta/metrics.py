"""Portfolio-level performance and risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(
    returns: pd.Series,
    nav: pd.Series,
    turnover: pd.Series,
    gross_exposure: pd.Series,
    total_cost: float,
    annualization: int,
) -> dict[str, float]:
    """Calculate explicit metrics from portfolio returns and NAV."""

    volatility = float(returns.std(ddof=1) * np.sqrt(annualization)) if len(returns) > 1 else 0.0
    annual_return = float(returns.mean() * annualization)
    sharpe = annual_return / volatility if volatility > 0 else 0.0
    downside = returns.clip(upper=0).std(ddof=1) * np.sqrt(annualization)
    sortino = annual_return / downside if downside > 0 else 0.0
    drawdown = nav.div(nav.cummax()).sub(1.0)
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    periods = max(len(returns), 1)
    start_value = nav.iloc[0]
    cagr = float((nav.iloc[-1] / start_value) ** (annualization / periods) - 1.0)
    return {
        "total_return": float(nav.iloc[-1] / start_value - 1.0),
        "cagr": cagr,
        "annualized_return": annual_return,
        "annualized_volatility": volatility,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "average_turnover": float(turnover.mean()),
        "total_turnover": float(turnover.sum()),
        "average_gross_exposure": float(gross_exposure.mean()),
        "max_gross_exposure": float(gross_exposure.max()),
        "total_cost": float(total_cost),
    }
