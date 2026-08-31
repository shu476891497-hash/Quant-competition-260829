# Quant Competition 260829

面向 IBKR 模拟盘比赛的轻量期货 CTA 研究与回测框架。

只想研究一个因子、不想安装完整项目结构的队员，可以只下载根目录的
[`single_file_backtest.py`](single_file_backtest.py)，安装 pandas/numpy 后直接运行。

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
python examples/run_factor_research.py
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

统一因子检验只需一行：

```python
result = evaluate_factor(
    factor,
    front_contract_close,
    horizons=(1, 5, 21),
    availability_lag=1,
    contract_ids=front_contract_symbol,
    factor_name="curve_curvature",
)
```

`availability_lag` 强制至少为 1；合约代码变化的标签会自动删除。输出统一为
样本数、IC、RankIC、Newey-West t 值和 Q5-Q1，不计算策略 Sharpe。

稳健性检查继续使用三个小函数：

- `evaluate_factor_by_year`：按因子决策年份输出同一套指标；
- `causal_winsorize`：只用当前时点之前的滚动分位数处理极端值；
- `factor_autocorrelation`：检查因子衰减和重复持仓风险。

`save_factor_results` 会把指标和数据源、进场延迟、换月规则等假设保存到同一个
运行目录。完整用法见 `examples/run_factor_research.py`。

## 文档

- [架构与时间契约](docs/ARCHITECTURE.md)
- [数据源选择](docs/DATA_SOURCES.md)
- [团队协作](TEAM_GUIDE.md)

v0.1 面向日线/小时线 CTA。L2 队列、部分成交、保证金追缴、组合优化和
IBKR 下单将在真实数据记账核对后加入。
