from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    if frame.empty:
        return "暂无可用结果。"
    display = frame.copy()
    for column in display.select_dtypes(include=["float", "float64", "float32"]):
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}")
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def create_figures(
    backtest: pd.DataFrame,
    final_panel: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    primary = backtest.loc[backtest["cost_bps"] == 1.0].copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for symbol, group in primary.groupby("symbol"):
        group = group.sort_values("date")
        nav = np.exp(group["net_log_return"].fillna(0).cumsum())
        if group["net_log_return"].notna().any():
            axes[0].plot(group["date"], nav, label=symbol)
            axes[1].plot(group["date"], nav / nav.cummax().clip(lower=1.0) - 1.0, label=symbol)
    axes[0].set_title("Three-factor strategy cumulative NAV (1 bp one-way cost)")
    axes[0].set_ylabel("NAV")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[0].legend(ncol=3)
    axes[1].legend(ncol=3)
    fig.tight_layout()
    path = figures_dir / "nav_and_drawdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    rolling_rows: list[pd.DataFrame] = []
    for symbol, group in final_panel.groupby("symbol"):
        group = group.sort_values("date").copy()
        forward = np.log(group["close"].shift(-5) / group["close"])
        for factor in ("put_call_factor", "liquidity_factor", "positioning_factor"):
            rolling = group[factor].rolling(252, min_periods=126).corr(forward)
            rolling_rows.append(pd.DataFrame({"date": group["date"], "symbol": symbol, "factor": factor, "rolling_ic": rolling}))
    rolling_frame = pd.concat(rolling_rows, ignore_index=True)
    median_rolling = rolling_frame.groupby(["date", "factor"], as_index=False)["rolling_ic"].median()
    fig, ax = plt.subplots(figsize=(12, 5))
    for factor, group in median_rolling.groupby("factor"):
        ax.plot(group["date"], group["rolling_ic"], label=factor)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Cross-index median 252-day rolling 5-day IC")
    ax.set_ylabel("Pearson IC")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "rolling_ic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    rolling_frame.to_csv(figures_dir.parent / "rolling_ic.csv", index=False)
    return paths


def write_chinese_report(
    path: Path,
    selections: list[dict[str, object]],
    metrics: pd.DataFrame,
    ic: pd.DataFrame,
    quality: pd.DataFrame,
    coverage: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> None:
    selected_table = pd.DataFrame(selections)[["family", "candidate", "orientation", "selection_period", "horizon"]]
    oos_metrics = metrics.loc[
        (metrics["period"] == "OOS") & (metrics["cost_bps"] == 1.0) & (metrics["kind"].isin(["strategy", "benchmark", "cash_benchmark"]))
    ][["symbol", "kind", "annual_return", "annual_volatility", "sharpe", "max_drawdown", "calmar", "annual_turnover", "observations"]]
    for column in ("annual_return", "annual_volatility", "max_drawdown"):
        oos_metrics[column] = oos_metrics[column] * 100.0
    oos_metrics = oos_metrics.rename(
        columns={
            "annual_return": "annual_return_pct",
            "annual_volatility": "annual_volatility_pct",
            "max_drawdown": "max_drawdown_pct",
        }
    )
    oos_ic = ic.loc[(ic["period"] == "OOS") & (ic["horizon"] == 5)][
        ["symbol", "factor", "spearman_ic", "hac_t", "observations"]
    ]
    text = f"""# 美股股指期货三因子研究报告

## 结论先行

本项目从 Put/Call、美元短期流动性和 CFTC 持仓三个主题族中，只使用 2010-06-15 至 2018-12-31 的 IS 数据冻结一个候选及其方向。2019—2022 为验证期，2023—{latest_date:%Y-%m-%d} 为只读 OOS。下表为程序实际选出的三个因子；OOS 好坏没有参与选择。

{_markdown_table(selected_table, 3)}

## OOS 策略与基准

策略允许多空，三个因子等权，仓位为 `clip(mean(z), -1, 1)`，信号滞后一交易日执行。主结果按单边换手 1 bp 扣费；benchmark 为同样本期货连续合约长期持有。

{_markdown_table(oos_metrics, 4)}

## OOS 五日预测能力

{_markdown_table(oos_ic, 4)}

## 数据覆盖与质量

{_markdown_table(coverage, 3)}

{_markdown_table(quality, 3)}

## 方法与边界

- 行情来自 Yahoo Finance 的公开免费连续合约代理，滚动方法不透明；结果不是 CME 官方连续合约回测。
- Cboe 当日 Put/Call 只在下一交易日进入收益；FRED 周三数据按周四发布并再滞后一交易日；CFTC 周二持仓从下一周一开始使用。
- FRED/CFTC 历史下载可能包含事后修订，所有宏观与持仓历史结果统一标记为 `PSEUDO_OOS_CURRENT_VINTAGE`，不声称是真正 point-in-time 数据。
- Yahoo 连续合约可能已隐含换月跳变，交易成本只针对策略仓位变化，不把未知换月成本伪装成普通换手成本。
- 本研究讨论衍生品情绪、宏观流动性和机构持仓，不是上市公司财务报表基本面。
- 仅供量化研究参考，不构成投资建议或实盘收益保证。

## 复现

```powershell
python run_research.py download
python run_research.py build-factors
python run_research.py evaluate
```

图表见 `outputs/figures/nav_and_drawdown.png` 与 `outputs/figures/rolling_ic.png`。完整 IS/验证/OOS/Full 指标和证据链均位于 `outputs/`。
"""
    path.write_text(text, encoding="utf-8")
