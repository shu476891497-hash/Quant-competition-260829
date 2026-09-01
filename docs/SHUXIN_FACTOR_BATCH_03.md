# 舒歆 ES/NQ 因子批次 03（检验前冻结）

本批次仍只研究期限结构、资金/持仓，不修改批次 02 的 NQ 曲率公式。

## A. ES–NQ 跨市场期限结构

预测目标固定为下一根日线进入的 `NQ 对数收益 - ES 对数收益`。两边的决策、
进入、退出均须是同一实际近月合约，任一市场换月即删除样本。

| 因子 | 定义 | 假设 |
|---|---|---|
| CARRYDIFF | `carry12_NQ-carry12_ES` | 两指数的隐含持有收益差包含相对资金需求 |
| NCURVDIFF | `ncurv_NQ-ncurv_ES` | NQ 特异的曲线弯曲预测相对收益 |
| OIMIGDIFF | `FOI_change5_NQ-FOI_change5_ES` | 两市场换月资金迁移差预测相对需求 |

H1/H5/H21，共 `3×3=9` 项，作为一个 BH FDR 家族。方向双侧，不事后翻转。

## B. CFTC 交易者参与度 / 集中度

仍用 TFF Futures-Only，报告周二观察、周五发布、下一根日线进入；任一预测区间
跨实际近月换月即删除。

| 因子 | 定义 |
|---|---|
| APTR | Asset Manager net OI share / (long traders + short traders) |
| LPTR | Leveraged Money net OI share / (long traders + short traders) |
| ALPTR | APTR - LPTR |
| C4NET | Top-4 net long concentration - net short concentration |
| C8NET | Top-8 net long concentration - net short concentration |
| C4RATIO | Top-4 gross concentration / Top-8 gross concentration |

ES、NQ；H5/H21，共 `6×2×2=24` 项，作为第二个 BH FDR 家族。交易者数量是
CFTC 报告值，不从未来报告回填。

## 通过标准

继续采用 `FDR q <= 0.10`。原始 `|t|>=2` 仅为线索；必须保存全部结果。
