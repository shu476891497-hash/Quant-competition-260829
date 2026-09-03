from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import PERIODS, ProjectPaths
from .data import download_cash_rate
from .evaluation import performance_metrics
from .utils import sha256_file, write_json


@dataclass(frozen=True)
class OverlaySpec:
    name: str
    top_n: int
    trend_windows: tuple[int, ...] = (60, 120, 200)
    liquidity_halflife: int = 5
    liquidity_threshold: float = 0.5
    allocation_interval: int = 10
    volatility_window: int = 40
    volatility_target: float = 0.10
    leverage_cap: float = 1.0
    volatility_interval: int = 5


PREREGISTERED_SPECS = (
    OverlaySpec(name="top2_liquidity_vol", top_n=2),
    OverlaySpec(name="top1_liquidity_vol", top_n=1),
    OverlaySpec(
        name="top2_liquidity_monthly",
        top_n=2,
        allocation_interval=21,
        volatility_interval=21,
    ),
)


def _periodic_hold(frame: pd.DataFrame | pd.Series, interval: int) -> pd.DataFrame | pd.Series:
    if interval < 1:
        raise ValueError("interval must be positive")
    held = frame.copy() * np.nan
    held.iloc[::interval] = frame.iloc[::interval]
    return held.ffill()


def build_overlay(
    prices: pd.DataFrame,
    factor_panel: pd.DataFrame,
    spec: OverlaySpec,
    cost_bps: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a causal cross-index overlay; close-t decisions earn from t+1."""
    symbols = sorted(prices["symbol"].dropna().unique())
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    returns = prices.pivot(index="date", columns="symbol", values="log_return").sort_index()
    close = close.reindex(columns=symbols)
    returns = returns.reindex(columns=symbols)

    liquidity = (
        factor_panel.groupby("date", sort=True)["liquidity_factor"]
        .first()
        .reindex(close.index)
        .ewm(
            halflife=spec.liquidity_halflife,
            min_periods=spec.liquidity_halflife,
            adjust=False,
        )
        .mean()
    )
    trend_parts = [np.sign(np.log(close / close.shift(window))) for window in spec.trend_windows]
    trend = sum(trend_parts) / len(trend_parts)
    rank = trend.rank(axis=1, ascending=False, method="first")
    selected = (rank <= spec.top_n).astype(float) / float(spec.top_n)
    positive_trend = (trend > 0.0).astype(float)
    liquidity_gate = (liquidity > spec.liquidity_threshold).astype(float)
    target_allocation = (selected * positive_trend).mul(liquidity_gate, axis=0)
    held_allocation = _periodic_hold(target_allocation, spec.allocation_interval)

    base_position = held_allocation.shift(1)
    base_return = (base_position * returns).sum(axis=1, min_count=1)
    realized_volatility = (
        base_return.rolling(
            spec.volatility_window,
            min_periods=max(5, spec.volatility_window // 2),
        ).std(ddof=0)
        * np.sqrt(252.0)
    )
    leverage = (spec.volatility_target / realized_volatility).clip(upper=spec.leverage_cap)
    leverage = _periodic_hold(leverage, spec.volatility_interval)

    decision_weights = held_allocation.mul(leverage, axis=0)
    weights = decision_weights.shift(1).fillna(0.0)
    trading_notional = weights.diff().abs().sum(axis=1)
    trading_notional.iloc[0] = weights.iloc[0].abs().sum()
    one_way_turnover = trading_notional / 2.0
    gross_return = (weights * returns).sum(axis=1, min_count=1)
    cost = trading_notional * float(cost_bps) / 10_000.0
    net_return = gross_return - cost
    started = weights.abs().sum(axis=1).gt(0).cummax()
    gross_return = gross_return.where(started)
    net_return = net_return.where(started)
    cost = cost.where(started)

    daily = pd.DataFrame(
        {
            "date": close.index,
            "liquidity_smoothed": liquidity,
            "liquidity_gate": liquidity_gate,
            "leverage": leverage,
            "gross_log_return": gross_return,
            "cost": cost,
            "net_log_return": net_return,
            "trading_notional": trading_notional,
            "one_way_turnover": one_way_turnover,
        }
    ).reset_index(drop=True)
    weight_panel = weights.stack(future_stack=True).rename("position").reset_index()
    trend_panel = trend.stack(future_stack=True).rename("trend_score").reset_index()
    details = weight_panel.merge(trend_panel, on=["date", "symbol"], how="left")
    return daily, details


def _metrics_for_periods(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, (start, end) in PERIODS.items():
        sample = daily.loc[daily["date"].between(start, end)]
        values = performance_metrics(sample["net_log_return"], sample["one_way_turnover"])
        rows.append({"period": period, **values})
    return pd.DataFrame(rows)


def add_cash_collateral_returns(
    daily: pd.DataFrame,
    cash_rate: pd.DataFrame,
) -> pd.DataFrame:
    """Add cash collateral return using only the prior session's known Treasury yield."""
    result = daily.sort_values("date").copy().reset_index(drop=True)
    rate = (
        cash_rate.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")["annual_rate_pct"]
        .reindex(result["date"])
        .ffill()
        .shift(1)
    )
    result["cash_rate_pct_known"] = rate.to_numpy()
    result["cash_collateral_log_return"] = np.log1p(
        result["cash_rate_pct_known"] / 100.0 / 252.0
    )
    result["total_account_net_log_return"] = (
        result["net_log_return"] + result["cash_collateral_log_return"]
    )
    result["excess_over_cash_log_return"] = (
        result["total_account_net_log_return"] - result["cash_collateral_log_return"]
    )
    return result


def _account_metrics_for_periods(daily: pd.DataFrame) -> pd.DataFrame:
    return_columns = {
        "price_only": "net_log_return",
        "total_account_zero_hurdle": "total_account_net_log_return",
        "excess_over_cash": "excess_over_cash_log_return",
    }
    rows: list[dict[str, object]] = []
    for period, (start, end) in PERIODS.items():
        sample = daily.loc[daily["date"].between(start, end)]
        for return_basis, column in return_columns.items():
            values = performance_metrics(sample[column], sample["one_way_turnover"])
            rows.append({"period": period, "return_basis": return_basis, **values})
    return pd.DataFrame(rows)


def _benchmark_comparison(
    prices: pd.DataFrame,
    cash_benchmarks: pd.DataFrame,
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    futures = prices.pivot(index="date", columns="symbol", values="log_return").sort_index()
    cash_index = cash_benchmarks.pivot(
        index="date", columns="symbol", values="log_return"
    ).sort_index()
    comparison = daily.set_index("date")[
        ["total_account_net_log_return", "cash_collateral_log_return"]
    ].join(futures[["ES", "NQ", "RTY", "YM"]], how="left")
    comparison = comparison.join(cash_index[["GSPC"]], how="left")
    comparison["LOW_TURNOVER"] = comparison["total_account_net_log_return"]
    comparison["EQUAL_WEIGHT_FUTURES"] = (
        comparison[["ES", "NQ", "RTY", "YM"]].mean(axis=1, skipna=False)
        + comparison["cash_collateral_log_return"]
    )
    comparison["ES_FUTURES"] = (
        comparison["ES"] + comparison["cash_collateral_log_return"]
    )
    comparison["GSPC_CASH_INDEX"] = comparison["GSPC"]
    return_columns = [
        "LOW_TURNOVER",
        "EQUAL_WEIGHT_FUTURES",
        "ES_FUTURES",
        "GSPC_CASH_INDEX",
    ]
    comparison = comparison[return_columns].reset_index()

    rows: list[dict[str, object]] = []
    for period, (start, end) in PERIODS.items():
        sample = comparison.loc[comparison["date"].between(start, end)].dropna()
        for name in return_columns:
            values = performance_metrics(sample[name])
            rows.append({"period": period, "benchmark": name, **values})
    return pd.DataFrame(rows), comparison


def _create_oos_figure(comparison: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    sample = comparison.loc[comparison["date"] >= "2023-01-01"].dropna().copy()
    return_columns = [column for column in comparison.columns if column != "date"]
    nav = np.exp(sample[return_columns].cumsum())
    nav = nav.div(nav.iloc[0])
    drawdown = nav.div(nav.cummax()) - 1.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for column in return_columns:
        axes[0].plot(sample["date"], nav[column], label=column)
        axes[1].plot(sample["date"], drawdown[column], label=column)
    axes[0].set_title("OOS fully collateralized NAV / cash-index benchmark")
    axes[0].set_ylabel("NAV (start = 1)")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].set_title("OOS drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _liquidity_ic(factor_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, group in factor_panel.groupby("symbol", sort=True):
        group = group.sort_values("date").copy()
        group["forward_5"] = np.log(group["close"].shift(-5) / group["close"])
        for period, (start, end) in PERIODS.items():
            sample = group.loc[
                group["date"].between(start, end), ["liquidity_factor", "forward_5"]
            ].dropna()
            value = np.nan
            if len(sample) >= 30 and sample["liquidity_factor"].nunique() > 1:
                value = float(spearmanr(sample["liquidity_factor"], sample["forward_5"]).statistic)
            rows.append(
                {
                    "period": period,
                    "symbol": symbol,
                    "factor": "liquidity_factor",
                    "horizon": 5,
                    "spearman_ic": value,
                    "observations": len(sample),
                }
            )
    result = pd.DataFrame(rows)
    medians = (
        result.groupby("period", as_index=False)["spearman_ic"]
        .median()
        .assign(
            symbol="MEDIAN",
            factor="liquidity_factor",
            horizon=5,
            observations=np.nan,
        )
    )
    return pd.concat([result, medians[result.columns]], ignore_index=True)


def _overlay_score_ic(
    factor_panel: pd.DataFrame,
    details: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    scores = details.merge(
        daily[["date", "liquidity_gate"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    scores["overlay_tradable_score"] = scores["trend_score"] * scores["liquidity_gate"]
    scores = scores.rename(columns={"position": "overlay_position_score"})
    panel = factor_panel[["date", "symbol", "close"]].merge(
        scores[
            [
                "date",
                "symbol",
                "overlay_tradable_score",
                "overlay_position_score",
            ]
        ],
        on=["date", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        group = group.sort_values("date").copy()
        group["forward_5"] = np.log(group["close"].shift(-5) / group["close"])
        for factor in ("overlay_tradable_score", "overlay_position_score"):
            for period, (start, end) in PERIODS.items():
                sample = group.loc[
                    group["date"].between(start, end), [factor, "forward_5"]
                ].dropna()
                value = np.nan
                if len(sample) >= 30 and sample[factor].nunique() > 1:
                    value = float(spearmanr(sample[factor], sample["forward_5"]).statistic)
                rows.append(
                    {
                        "period": period,
                        "symbol": symbol,
                        "factor": factor,
                        "horizon": 5,
                        "spearman_ic": value,
                        "observations": len(sample),
                    }
                )
    result = pd.DataFrame(rows)
    medians = (
        result.groupby(["period", "factor"], as_index=False)["spearman_ic"]
        .median()
        .assign(symbol="MEDIAN", horizon=5, observations=np.nan)
    )
    return pd.concat([result, medians[result.columns]], ignore_index=True)


def evaluate_low_turnover(paths: ProjectPaths, cost_bps: float = 1.0) -> dict[str, pd.DataFrame]:
    prices = pd.read_parquet(paths.processed / "prices.parquet")
    factor_panel = pd.read_parquet(paths.processed / "final_factor_panel.parquet")
    cash_benchmarks = pd.read_parquet(paths.processed / "cash_benchmarks.parquet")
    cash_rate, cash_rate_source = download_cash_rate(paths, refresh=False)
    output = paths.outputs / "low_turnover"
    output.mkdir(parents=True, exist_ok=True)

    evidence_rows: list[dict[str, object]] = []
    built: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for spec in PREREGISTERED_SPECS:
        daily, details = build_overlay(prices, factor_panel, spec, cost_bps=cost_bps)
        built[spec.name] = (daily, details)
        metrics = _metrics_for_periods(daily)
        for period in ("IS", "VALIDATION"):
            row = metrics.loc[metrics["period"] == period].iloc[0]
            evidence_rows.append(
                {
                    "candidate": spec.name,
                    "period": period,
                    "sharpe": row["sharpe"],
                    "annual_return": row["annual_return"],
                    "max_drawdown": row["max_drawdown"],
                    "annual_one_way_turnover": row["annual_turnover"],
                    "selection_eligible": period == "IS" and row["annual_turnover"] < 30.0,
                }
            )
    evidence = pd.DataFrame(evidence_rows)
    is_evidence = evidence.loc[evidence["selection_eligible"]].sort_values(
        ["sharpe", "annual_one_way_turnover", "candidate"],
        ascending=[False, True, True],
    )
    if is_evidence.empty:
        raise RuntimeError("No preregistered candidate passed the IS turnover constraint")
    selected_name = str(is_evidence.iloc[0]["candidate"])
    selected_spec = next(spec for spec in PREREGISTERED_SPECS if spec.name == selected_name)
    daily, details = built[selected_name]
    daily = add_cash_collateral_returns(daily, cash_rate)
    metrics = _account_metrics_for_periods(daily)
    metrics["cost_bps"] = cost_bps
    sensitivity_frames: list[pd.DataFrame] = []
    for scenario_cost in (0.0, 1.0, 2.0, 5.0):
        scenario_daily, _ = build_overlay(
            prices,
            factor_panel,
            selected_spec,
            cost_bps=scenario_cost,
        )
        scenario_daily = add_cash_collateral_returns(scenario_daily, cash_rate)
        scenario_metrics = _account_metrics_for_periods(scenario_daily)
        scenario_metrics["cost_bps"] = scenario_cost
        sensitivity_frames.append(scenario_metrics)
    cost_sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    benchmark_metrics, benchmark_daily = _benchmark_comparison(
        prices,
        cash_benchmarks,
        daily,
    )
    ic = pd.concat(
        [
            _liquidity_ic(factor_panel),
            _overlay_score_ic(factor_panel, details, daily),
        ],
        ignore_index=True,
    )

    evidence.to_csv(output / "selection_evidence.csv", index=False)
    daily.to_csv(output / "backtest_daily.csv", index=False)
    daily.to_parquet(output / "backtest_daily.parquet", index=False)
    details.to_csv(output / "positions.csv", index=False)
    details.to_parquet(output / "positions.parquet", index=False)
    metrics.to_csv(output / "performance_metrics.csv", index=False)
    cost_sensitivity.to_csv(output / "cost_sensitivity_metrics.csv", index=False)
    benchmark_metrics.to_csv(output / "benchmark_comparison.csv", index=False)
    benchmark_daily.to_parquet(output / "benchmark_daily.parquet", index=False)
    ic.to_csv(output / "factor_ic.csv", index=False)
    _create_oos_figure(benchmark_daily, output / "nav_drawdown_oos.png")
    write_json(output / "cash_rate_source.json", cash_rate_source[0])
    write_json(
        output / "frozen_overlay.json",
        {
            "selection_rule": "highest IS Sharpe subject to annual one-way turnover < 30; validation and OOS do not select",
            "cost_bps": cost_bps,
            "selected": asdict(selected_spec),
            "preregistered": [asdict(spec) for spec in PREREGISTERED_SPECS],
        },
    )

    oos_total = metrics.loc[
        (metrics["period"] == "OOS")
        & (metrics["return_basis"] == "total_account_zero_hurdle")
    ].iloc[0]
    oos_excess = metrics.loc[
        (metrics["period"] == "OOS") & (metrics["return_basis"] == "excess_over_cash")
    ].iloc[0]
    oos_ic = ic.loc[
        (ic["period"] == "OOS")
        & (ic["symbol"] == "MEDIAN")
        & (ic["factor"] == "overlay_tradable_score"),
        "spearman_ic",
    ].iloc[0]
    status = {
        "ic_definition": "median four-symbol 5-day Spearman IC of the frozen trend-times-liquidity tradable score",
        "oos_ic": oos_ic,
        "sharpe_gate_definition": "fully cash-collateralized total-account return with zero hurdle",
        "oos_sharpe": oos_total["sharpe"],
        "oos_excess_over_cash_sharpe": oos_excess["sharpe"],
        "oos_annual_one_way_turnover": oos_total["annual_turnover"],
        "targets": {
            "oos_ic_min": 0.015,
            "oos_sharpe_min": 1.5,
            "annual_one_way_turnover_max": 30.0,
        },
        "passed": bool(
            oos_ic >= 0.015
            and oos_total["sharpe"] >= 1.5
            and oos_total["annual_turnover"] < 30.0
        ),
        "label": "PSEUDO_OOS_CURRENT_VINTAGE",
    }
    write_json(output / "target_audit.json", status)
    _write_report(
        output / "REPORT_LOW_TURNOVER_CN.md",
        selected_spec,
        evidence,
        metrics,
        ic,
        cost_sensitivity,
        benchmark_metrics,
        status,
    )
    write_json(
        output / "hashes.json",
        {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "hashes.json"
        },
    )
    return {
        "evidence": evidence,
        "metrics": metrics,
        "ic": ic,
        "cost_sensitivity": cost_sensitivity,
        "benchmarks": benchmark_metrics,
        "daily": daily,
        "positions": details,
    }


def _write_report(
    path: Path,
    spec: OverlaySpec,
    evidence: pd.DataFrame,
    metrics: pd.DataFrame,
    ic: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
    status: dict[str, object],
) -> None:
    metric_view = metrics[
        [
            "period",
            "return_basis",
            "annual_return",
            "sharpe",
            "max_drawdown",
            "annual_turnover",
        ]
    ].copy()
    for column in ("annual_return", "max_drawdown"):
        metric_view[column] = metric_view[column].map(lambda value: f"{value:.2%}")
    metric_view["sharpe"] = metric_view["sharpe"].map(lambda value: f"{value:.2f}")
    metric_view["annual_turnover"] = metric_view["annual_turnover"].map(
        lambda value: f"{value:.2f}"
    )
    ic_view = ic.loc[
        ic["symbol"] == "MEDIAN", ["period", "factor", "spearman_ic"]
    ].copy()
    ic_view["spearman_ic"] = ic_view["spearman_ic"].map(lambda value: f"{value:.4f}")
    evidence_view = evidence.copy()
    for column in ("sharpe", "annual_return", "max_drawdown", "annual_one_way_turnover"):
        evidence_view[column] = evidence_view[column].map(lambda value: f"{value:.4f}")
    sensitivity_view = cost_sensitivity.loc[
        (cost_sensitivity["period"] == "OOS")
        & (cost_sensitivity["return_basis"] == "total_account_zero_hurdle"),
        ["cost_bps", "annual_return", "sharpe", "max_drawdown", "annual_turnover"],
    ].copy()
    for column in ("annual_return", "max_drawdown"):
        sensitivity_view[column] = sensitivity_view[column].map(lambda value: f"{value:.2%}")
    sensitivity_view["sharpe"] = sensitivity_view["sharpe"].map(lambda value: f"{value:.2f}")
    sensitivity_view["annual_turnover"] = sensitivity_view["annual_turnover"].map(
        lambda value: f"{value:.2f}"
    )
    benchmark_view = benchmark_metrics.loc[
        benchmark_metrics["period"] == "OOS",
        ["benchmark", "annual_return", "sharpe", "max_drawdown", "observations"],
    ].copy()
    for column in ("annual_return", "max_drawdown"):
        benchmark_view[column] = benchmark_view[column].map(lambda value: f"{value:.2%}")
    benchmark_view["sharpe"] = benchmark_view["sharpe"].map(lambda value: f"{value:.2f}")

    text = f"""# 低换手候选策略独立审计

## 结论

- 冻结候选：`{spec.name}`。
- OOS 5 日 Spearman IC（四品种中位数）：{float(status['oos_ic']):.4f}。
- OOS 组合 Sharpe（1 bp）：{float(status['oos_sharpe']):.2f}。
- OOS 扣除现金基准后的超额 Sharpe：{float(status['oos_excess_over_cash_sharpe']):.2f}。
- OOS 年化单边换手：{float(status['oos_annual_one_way_turnover']):.2f}。
- 三项门槛是否同时通过：**{'是' if status['passed'] else '否'}**。

## 冻结规则

本策略不改写原三因子基线。候选仅按 IS 价格损益 Sharpe 排序，并要求年化单边换手低于 30；验证期与 OOS 均不参与选择。信号在 t 日收盘形成，整体滞后一交易日后赚取收益。每 10 个全局锚定交易日更新一次配置：以 60/120/200 日趋势均值排序，持有正趋势前两名；5 日半衰期流动性平滑值高于 0.5 时才承担风险；40 日实现波动率目标为 10%，杠杆上限 1.0，每 5 个交易日更新风险倍率。

Sharpe 门槛按全额现金抵押期货账户的总收益、零收益门槛计算；现金抵押收益使用 FRED `DGS3MO`，且只在下一交易日计入。为了避免口径美化，报告同时列出价格损益和扣除现金基准的超额 Sharpe。总账户 Sharpe 通过 1.50 不代表超额 Sharpe也通过。

## 候选证据链

{evidence_view.to_markdown(index=False)}

## 分段绩效

{metric_view.to_markdown(index=False)}

## 因子 IC

{ic_view.to_markdown(index=False)}

## OOS 成本敏感性

{sensitivity_view.to_markdown(index=False)}

## OOS 基准对比

{benchmark_view.to_markdown(index=False)}

所有基准在各分段使用共同有效日期。`EQUAL_WEIGHT_FUTURES` 与 `ES_FUTURES` 加入同一现金抵押收益；`GSPC_CASH_INDEX` 是标普500现金价格指数，不含股息。

## 限制

Yahoo 连续合约是研究代理；FRED/CFTC 为当前历史版本，因此标签为 `PSEUDO_OOS_CURRENT_VINTAGE`。策略允许空仓但不做净空；换月成本包含在 Yahoo 连续价格代理中，不另作普通调仓费重复扣除。结果仅供量化研究，不构成投资建议。
"""
    path.write_text(text, encoding="utf-8")
