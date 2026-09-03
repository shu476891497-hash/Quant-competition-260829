# 美股股指期货三因子研究报告

## 结论先行

本项目从 Put/Call、美元短期流动性和 CFTC 持仓三个主题族中，只使用 2010-06-15 至 2018-12-31 的 IS 数据冻结一个候选及其方向。2019—2022 为验证期，2023—2026-08-31 为只读 OOS。下表为程序实际选出的三个因子；OOS 好坏没有参与选择。

| family | candidate | orientation | selection_period | horizon |
| --- | --- | --- | --- | --- |
| put_call | pcr_deviation | 1 | 2010-06-15/2018-12-31 | 5 |
| liquidity | liq_net_rrp | 1 | 2010-06-15/2018-12-31 | 5 |
| positioning | cot_divergence | 1 | 2010-06-15/2018-12-31 | 5 |

## OOS 策略与基准

策略允许多空，三个因子等权，仓位为 `clip(mean(z), -1, 1)`，信号滞后一交易日执行。主结果按单边换手 1 bp 扣费；benchmark 为同样本期货连续合约长期持有。

| symbol | kind | annual_return_pct | annual_volatility_pct | sharpe | max_drawdown_pct | calmar | annual_turnover | observations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ES | benchmark | 20.7847 | 14.8967 | 1.2677 | -18.5421 | 1.1209 |  | 921.0000 |
| NQ | benchmark | 30.9294 | 20.3017 | 1.3274 | -22.5017 | 1.3745 |  | 921.0000 |
| PORTFOLIO | benchmark | 19.9399 | 16.1465 | 1.1261 | -20.2986 | 0.9823 |  | 921.0000 |
| RTY | benchmark | 15.0776 | 20.8845 | 0.6724 | -27.6214 | 0.5459 |  | 921.0000 |
| YM | benchmark | 13.7142 | 13.6016 | 0.9449 | -16.0599 | 0.8539 |  | 921.0000 |
| ES | strategy | 4.5070 | 9.5573 | 0.4613 | -11.6560 | 0.3867 | 78.0414 | 921.0000 |
| NQ | strategy | 4.8597 | 11.0985 | 0.4276 | -12.1220 | 0.4009 | 79.5895 | 921.0000 |
| PORTFOLIO | strategy | 1.9930 | 8.3233 | 0.2371 | -12.3656 | 0.1612 | 79.3458 | 921.0000 |
| RTY | strategy | -1.5948 | 10.7165 | -0.1500 | -19.7715 | -0.0807 | 78.9038 | 921.0000 |
| YM | strategy | 0.3483 | 7.1437 | 0.0487 | -14.2980 | 0.0244 | 80.8487 | 921.0000 |
| GSPC | cash_benchmark | 20.9892 | 14.7335 | 1.2932 | -18.9022 | 1.1104 |  | 918.0000 |

## OOS 五日预测能力

| symbol | factor | spearman_ic | hac_t | observations |
| --- | --- | --- | --- | --- |
| ES | put_call_factor | 0.0922 | 2.2845 | 914 |
| ES | liquidity_factor | 0.0391 | 0.1319 | 916 |
| ES | positioning_factor | -0.1312 | -1.7429 | 916 |
| NQ | put_call_factor | 0.0854 | 2.1671 | 914 |
| NQ | liquidity_factor | 0.0090 | -0.2514 | 916 |
| NQ | positioning_factor | -0.0465 | -0.7696 | 916 |
| RTY | put_call_factor | 0.0491 | 1.4930 | 914 |
| RTY | liquidity_factor | 0.0426 | 0.1358 | 916 |
| RTY | positioning_factor | -0.1184 | -1.7889 | 916 |
| YM | put_call_factor | 0.0525 | 1.5421 | 914 |
| YM | liquidity_factor | 0.0685 | 0.6512 | 916 |
| YM | positioning_factor | 0.0172 | 0.1898 | 916 |

## 数据覆盖与质量

| symbol | field | start | end | observations | coverage |
| --- | --- | --- | --- | --- | --- |
| ES | put_call_factor | 2007-05-17 | 2026-08-31 | 4852 | 0.932 |
| ES | liquidity_factor | 2006-01-03 | 2026-08-31 | 5204 | 1.000 |
| ES | positioning_factor | 2011-12-09 | 2026-08-31 | 3702 | 0.711 |
| ES | composite | 2011-12-09 | 2026-08-31 | 3700 | 0.711 |
| NQ | put_call_factor | 2007-05-17 | 2026-08-31 | 4852 | 0.932 |
| NQ | liquidity_factor | 2006-01-03 | 2026-08-31 | 5204 | 1.000 |
| NQ | positioning_factor | 2011-12-09 | 2026-08-31 | 3702 | 0.711 |
| NQ | composite | 2011-12-09 | 2026-08-31 | 3700 | 0.711 |
| RTY | put_call_factor | 2017-07-10 | 2026-08-31 | 2300 | 0.999 |
| RTY | liquidity_factor | 2017-07-10 | 2026-08-31 | 2302 | 1.000 |
| RTY | positioning_factor | 2017-07-10 | 2026-08-31 | 2302 | 1.000 |
| RTY | composite | 2017-07-10 | 2026-08-31 | 2300 | 0.999 |
| YM | put_call_factor | 2007-05-17 | 2026-08-31 | 4852 | 0.933 |
| YM | liquidity_factor | 2006-01-03 | 2026-08-31 | 5198 | 1.000 |
| YM | positioning_factor | 2011-12-09 | 2026-08-31 | 3702 | 0.712 |
| YM | composite | 2011-12-09 | 2026-08-31 | 3700 | 0.712 |

| source | rows | start | end | duplicate_rows | duplicate_keys | missing_cells |
| --- | --- | --- | --- | --- | --- | --- |
| prices | 17908 | 2006-01-03 | 2026-08-31 | 0 | 0 | 4 |
| cboe_pcr | 4988 | 2006-11-01 | 2026-08-31 | 0 | 0 | 0 |
| fred | 6155 | 2002-12-18 | 2026-08-31 | 0 | 0 | 17593 |
| cftc | 3593 | 2006-06-13 | 2026-08-25 | 0 | 0 | 0 |
| cash_benchmarks | 5197 | 2006-01-03 | 2026-08-31 | 0 | 0 | 1 |

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
