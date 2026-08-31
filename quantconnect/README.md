# QuantConnect 纯因子研究脚本

这些脚本用于 QuantConnect 云端数据上的 ES/NQ 研究，全部为 **0-order 因子检验**，
不是交易策略。

## 使用

QuantConnect 项目的入口固定为 `main.py`：

1. 选择一个编号脚本，把内容复制到云端 `main.py`；
2. 运行 CFTC 脚本时，同时上传 `cftc_tff_embedded.py`；
3. Cloud Build 后运行 Backtest；
4. 在结果顶部读取自定义统计。PnL/Sharpe 为 0 是正常的，因为脚本不下单。

## 文件顺序

- `13_`：期限结构、OI 与 volume/OI 首轮批量检验；
- `14_`：CFTC TFF 分类持仓批量检验；
- `15_`：候选的时间分段与距到期日过滤；
- `16_`：曲率标准化、变化和正交化扩展；
- `17_`：ES 正交曲率与 NQ 原始曲率的最终分段复核。
- `18_`：强制下一根 bar 进场的 NQ 曲率最终口径；后续以此为准。
- `19_`：检验前冻结的第二批期限结构 / 交易所 OI 因子，严格下一根 bar。
- `20_`：第二批 CFTC TFF spreading / 资金分歧因子，严格发布时间与下一根 bar。
- `21_`：NQ 标准化曲率 H5 的分段、距到期日和年度方向稳健性复核。

注意：以上历史脚本的未来收益从因子日收盘开始，仅用于复现已经取得的研究结果。
仓库内新的统一口径 `quantcta.research.evaluate_factor` 更保守，强制下一根 bar
才作为进场价。NQ CURV 已用 `18_` 重跑：H5 IC=0.057、RankIC=0.081、
Newey-West t=2.2；距到期至少 30 天时 t=2.0。
