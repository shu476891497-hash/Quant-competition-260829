from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr


def backtest_symbol(frame: pd.DataFrame, cost_bps: float = 1.0) -> pd.DataFrame:
    result = frame.sort_values("date").copy().reset_index(drop=True)
    signal = result["composite"].clip(-1.0, 1.0)
    known_signal = signal.shift(1).notna()
    started = known_signal.cummax()
    result["position"] = signal.shift(1).fillna(0.0)
    result["turnover"] = result["position"].diff().abs().fillna(result["position"].abs())
    result["gross_log_return"] = result["position"] * result["log_return"].fillna(0.0)
    result["cost"] = result["turnover"] * float(cost_bps) / 10_000.0
    result["net_log_return"] = result["gross_log_return"] - result["cost"]
    result["benchmark_log_return"] = result["log_return"].fillna(0.0)
    result.loc[~started, ["gross_log_return", "net_log_return", "benchmark_log_return"]] = np.nan
    return result


def performance_metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return {key: np.nan for key in ("annual_return", "annual_volatility", "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate", "annual_turnover", "observations")}
    n = len(clean)
    annual_return = float(np.exp(clean.sum() * 252.0 / n) - 1.0)
    annual_volatility = float(clean.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(clean.mean() / clean.std(ddof=0) * np.sqrt(252.0)) if clean.std(ddof=0) > 0 else np.nan
    downside = clean.loc[clean < 0].std(ddof=0)
    sortino = float(clean.mean() / downside * np.sqrt(252.0)) if pd.notna(downside) and downside > 0 else np.nan
    nav = np.exp(clean.cumsum())
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan
    hit_rate = float((clean > 0).mean())
    annual_turnover = (
        float(turnover.reindex(clean.index).fillna(0).mean() * 252.0) if turnover is not None else np.nan
    )
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "hit_rate": hit_rate,
        "annual_turnover": annual_turnover,
        "observations": float(n),
    }


def _ic_and_t(factor: pd.Series, forward: pd.Series, horizon: int) -> tuple[float, float, float]:
    valid = pd.concat([factor.rename("factor"), forward.rename("forward")], axis=1).dropna()
    if len(valid) < 30 or valid["factor"].nunique() < 2 or valid["forward"].nunique() < 2:
        return np.nan, np.nan, np.nan
    pearson = float(pearsonr(valid["factor"], valid["forward"]).statistic)
    spearman = float(spearmanr(valid["factor"], valid["forward"]).statistic)
    model = sm.OLS(valid["forward"], sm.add_constant(valid[["factor"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon}
    )
    return pearson, spearman, float(model.tvalues["factor"])


def factor_diagnostics(
    panel: pd.DataFrame,
    factor_columns: list[str],
    periods: dict[str, tuple[str, str]],
    horizons: tuple[int, ...] = (1, 5, 20),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ic_rows: list[dict[str, object]] = []
    quintile_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    for symbol, symbol_frame in panel.groupby("symbol", sort=True):
        symbol_frame = symbol_frame.sort_values("date").copy()
        for horizon in horizons:
            symbol_frame[f"forward_{horizon}"] = np.log(symbol_frame["close"].shift(-horizon) / symbol_frame["close"])
        for period, (start, end) in periods.items():
            sample = symbol_frame.loc[symbol_frame["date"].between(start, end)]
            for factor in factor_columns:
                for horizon in horizons:
                    pearson, spearman, tvalue = _ic_and_t(sample[factor], sample[f"forward_{horizon}"], horizon)
                    ic_rows.append(
                        {"period": period, "symbol": symbol, "factor": factor, "horizon": horizon, "pearson_ic": pearson, "spearman_ic": spearman, "hac_t": tvalue, "observations": int(sample[[factor, f"forward_{horizon}"]].dropna().shape[0])}
                    )
                valid = sample[[factor, "forward_5"]].dropna().copy()
                if len(valid) >= 50 and valid[factor].nunique() >= 5:
                    valid["quintile"] = pd.qcut(valid[factor], 5, labels=False, duplicates="drop") + 1
                    for quintile, group in valid.groupby("quintile"):
                        quintile_rows.append({"period": period, "symbol": symbol, "factor": factor, "quintile": int(quintile), "mean_forward_5d": float(group["forward_5"].mean()), "observations": len(group)})
                for year, group in sample.groupby(sample["date"].dt.year):
                    _, spearman, tvalue = _ic_and_t(group[factor], group["forward_5"], 5)
                    yearly_rows.append({"period": period, "symbol": symbol, "factor": factor, "year": int(year), "spearman_ic_5d": spearman, "hac_t_5d": tvalue})
    return pd.DataFrame(ic_rows), pd.DataFrame(quintile_rows), pd.DataFrame(yearly_rows)


def extreme_state_diagnostics(
    panel: pd.DataFrame,
    factor_columns: list[str],
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, group in panel.groupby("symbol"):
        group = group.sort_values("date").copy()
        group["forward_5"] = np.log(group["close"].shift(-5) / group["close"])
        for period, (start, end) in periods.items():
            sample = group.loc[group["date"].between(start, end)]
            for factor in factor_columns:
                for state, mask in {
                    "LOW_LE_-1": sample[factor] <= -1.0,
                    "HIGH_GE_1": sample[factor] >= 1.0,
                }.items():
                    values = sample.loc[mask, "forward_5"].dropna()
                    rows.append(
                        {
                            "period": period,
                            "symbol": symbol,
                            "factor": factor,
                            "state": state,
                            "mean_forward_5d": float(values.mean()) if not values.empty else np.nan,
                            "median_forward_5d": float(values.median()) if not values.empty else np.nan,
                            "positive_rate": float((values > 0).mean()) if not values.empty else np.nan,
                            "observations": len(values),
                        }
                    )
    return pd.DataFrame(rows)
