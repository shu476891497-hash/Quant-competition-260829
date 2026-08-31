# 舒歆 ES/NQ 因子批次 02（检验前冻结）

本批次只研究舒歆负责的 **期限结构、资金/持仓**。不研究价格动量、反转、
波动率或交易微观结构，也不把多个因子拼成交易策略。

## A. 日频期限结构 / 交易所 OI 家族

数据是当日 ES/NQ 各实际到期合约的收盘价、到期日、成交量和持仓量。

| 编号 | 因子 | 定义 | 经济含义 |
|---|---|---|---|
| NCURV | 标准化曲率 | `(carry12-carry23)/(|carry12|+|carry23|)` | 曲线弯曲而非绝对斜率 |
| OIMAT | OI 加权期限 | `sum(OI_i*DTE_i)/sum(OI_i)` | 存量资金位于曲线的远近 |
| OIMD5 | OI 加权期限 5 日变化 | `OIMAT_t-OIMAT_t-5` | 持仓向远月或近月迁移 |
| FOID5 | 近月 OI 占比 5 日变化 | `front_share_t-front_share_t-5` | 换月/资金迁移压力 |
| OIRAT | 次近月/近月 OI 对数比 | `log(OI_2/OI_1)` | 当前换月阶段与持仓分布 |
| VMOI | 成交期限减持仓期限 | `volume_weighted_DTE-OI_weighted_DTE` | 当日新增交易相对存量的位置 |

预先固定：ES、NQ；未来收益 H1/H5/H21；共 `6×2×3=36` 个检验，作为一个
Benjamini-Hochberg FDR 家族。因子在日 t 收盘形成，最早用 t+1 收盘作为进入价，
退出为 t+1+h；决策、进入和退出必须属于同一个实际近月合约。所有因子方向均
双侧检验，不因结果事后翻转。

## B. 周频 CFTC TFF 资金 / 持仓家族

只用 CFTC TFF Futures-Only。报告对应周二持仓，固定在周五发布后才可用。

| 编号 | 因子 | 定义 |
|---|---|---|
| ASPR | Asset Manager spreading / OI | 资管跨月价差持仓占比 |
| LSPR | Leveraged Money spreading / OI | 杠杆资金跨月价差持仓占比 |
| DSPR | Dealer spreading / OI | 做市/中介跨月价差持仓占比 |
| ALDIV | Asset Manager net share - Leveraged Money net share | 两类资金方向分歧 |
| DAS1 | Asset Manager spreading share 一周变化 | 资管价差持仓增减 |
| DLS1 | Leveraged Money spreading share 一周变化 | 杠杆资金价差持仓增减 |

预先固定：ES、NQ；未来收益 H5/H21；共 `6×2×2=24` 个检验，作为另一个 FDR
家族。因子从发布后的下一根日线才可进入。报告修订、缺失值和合约换月不得回填
未来信息。两个家族分别控制 FDR，不在批次之间挑最好看的结果。

## 通过标准

单项 `|Newey-West t| >= 2` 只算线索；必须同时报告 IC、Rank IC、五分位差、
样本数、年度分段和 FDR q 值。`q <= 0.10` 才能称为本批次统计通过；否则保留为
失败结果或前瞻观察候选，不能称为已发现 alpha。
