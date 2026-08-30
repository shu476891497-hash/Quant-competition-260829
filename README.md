# Quant Competition 260829

面向 IBKR 模拟盘比赛的轻量期货 CTA 研究与回测框架。

```text
Data → Factor → Signal → Target Position → Actual Position → PnL → Metrics
```

## 已实现

- pandas/NumPy 向量化多品种期货回测；
- 强制至少一根 bar 的决策—成交延迟；
- 合约乘数、tick size、FX、每张费用、交易所费、滑点和价差；
- notional、gross/net exposure、turnover、NAV 与成本分解；
- 波动率目标仓位和 gross-notional 上限；
- root target 到实际交割合约的显式换月；
- point-in-time `available_at` 数据校验；
- 防前视、期货记账、数据校验和换月测试；
- Tushare 中国期货 adapter；IBKR 执行与研究核心隔离。

## 安装与验证

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest
python examples/run_synthetic.py
```

需要 Tushare 时安装 `.[tushare,dev]`。令牌只通过环境变量
`TUSHARE_TOKEN` 提供，不得写入仓库。合成示例不需要 TWS 或 API 密钥。

## 新增 factor

1. 在 `src/quantcta/factors/` 新增普通函数；
2. 输入和输出都是时间 × instrument 的同形 DataFrame；
3. 只能使用当前及过去数据；
4. 在 `tests/` 增加 future-mutation 测试；
5. factor 不直接计算仓位或 PnL。

不需要修改 backtest engine，也不需要继承基类。

## 文档

- [架构与时间契约](docs/ARCHITECTURE.md)
- [数据源选择](docs/DATA_SOURCES.md)
- [团队协作](TEAM_GUIDE.md)

v0.1 面向日线/小时线 CTA。L2 队列、部分成交、保证金追缴、组合优化和
IBKR 下单将在真实数据记账核对后加入。
