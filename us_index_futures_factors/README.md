# US Index Futures Three-Factor Research

研究 ES、NQ、YM、RTY 四个公开免费连续期货代理的三类因子：Cboe Put/Call、FRED 美元流动性、CFTC TFF 持仓。

## Git delivery / QuantConnect status

This repository contains a local Python research pipeline, not a validated
QuantConnect/LEAN algorithm. No QuantConnect compilation, cloud backtest or live
deployment has been performed. Git publication does not imply platform validation.

Git 中保留源码、测试、研究报告快照、三因子冻结配置与 IS 选参证据；
不包含原始行情、清洗数据、逐日面板、密钥或本机缓存。克隆后需先安装依赖并下载数据。
公开免费访问不等于数据具有再分发许可。

历史报告中的 2.12 是假设全额现金抵押收益的零门槛收益/波动比，
不能等同于扣除无风险收益的标准 Sharpe；报告列示的超额 Sharpe 为 1.26。
这些数值是本地研究代理回测，不是 QuantConnect 平台结果。

## Quick start

```powershell
cd C:\Users\Hu\Desktop\oot\us_index_futures_factors
python -m pip install -e .
python run_research.py all
```

分步运行：

```powershell
python run_research.py download
python run_research.py build-factors
python run_research.py evaluate
python run_research.py evaluate-low-turnover
python -m pytest
```

`build-factors` 第一次运行时只用 2010-06-15—2018-12-31 的 ES/NQ/YM 数据选取候选，并将结果冻结在 `outputs/frozen_selection.json`。普通更新不会重新选参；只有明确执行 `python run_research.py build-factors --reselect` 才会替换冻结选择。

## Outputs

- `REPORT_CN.md`：中文研究报告。
- `RESULT_SUMMARY_CN.md`：基线、低换手候选和三项硬门槛的一页整合结论。
- `outputs/candidate_selection_evidence.csv`：九个预注册候选的 IS 证据链。
- `outputs/final_factor_panel.csv`：三个最终因子、综合信号和行情。
- `outputs/factor_ic.csv`：IS/验证/OOS/Full 的 1/5/20 日 IC 与 HAC t 值。
- `outputs/factor_extreme_states.csv`：因子大于 1 或小于 -1 时的五日条件收益。
- `outputs/performance_metrics.csv`：策略与基准绩效。
- `outputs/backtest_daily.csv`：0/1/2/5 bp 成本情景的逐日回测。
- `data/raw/manifest.json`：下载来源、抓取时间、行数和 SHA256。
- `outputs/delivery_manifest.json`：源码、报告和主要输出文件的 SHA256。
- `outputs/low_turnover/REPORT_LOW_TURNOVER_CN.md`：独立低换手候选与三门槛审计，不覆盖原基线。
- `outputs/low_turnover/cash_rate_source.json`：FRED 三个月国债现金抵押利率的来源、抓取时间与 SHA256。
- `outputs/low_turnover/cost_sensitivity_metrics.csv`：0/1/2/5 bp 成本敏感性。
- `outputs/low_turnover/benchmark_comparison.csv`：共同日期上的期货与标普500基准对比。
- `outputs/low_turnover/nav_drawdown_oos.png`：OOS 净值与回撤图。

## Research discipline

- IS 只用于选参；验证期和 OOS 不参与候选、方向、窗口或权重选择。
- PCR、FRED 和 CFTC 信号均按公开时间滞后，策略再滞后一交易日持仓。
- 因 Yahoo 连续合约方法不透明，结果是研究代理，不是 CME 官方行情回测。
- FRED/CFTC 当前历史可能包含修订，因此标记为 `PSEUDO_OOS_CURRENT_VINTAGE`。
- 仅供研究参考，不构成投资建议。
