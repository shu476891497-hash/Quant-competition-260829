# 团队协作指南

本仓库是量化竞赛团队代码的同步仓库。所有成员通过 GitHub 协作，请按本指南操作。

## 一、加入仓库

1. 将你的 **GitHub 用户名**发给项目管理员（yyf0507）。
2. 管理员在仓库 **Settings → Collaborators** 中将你添加为协作者。
3. 你收到 GitHub 邀请邮件后，点击 **Accept invitation** 即可访问本仓库。

## 二、获取代码

首次使用，克隆仓库到本地：

```bash
git clone https://github.com/shu476891497-hash/Quant-competition-260829.git
```

如果你已有本地文件，把它们移入克隆出来的目录即可。

## 三、日常协作流程

每次开发前，先同步最新代码：

```bash
git pull
```

**在独立分支上开发，不要直接在 main 上改。** 新建分支：

```bash
git checkout -b 你的分支名   # 例如 feature/preprocess
```

提交你的改动：

```bash
git add .
git commit -m "简要描述你的改动"
```

推送分支：

```bash
git push origin 你的分支名
```

然后在 GitHub 网页上发起 **Pull Request**，由其他成员审查后合并到 main。

发起 Pull Request 前必须通过：

```bash
python -m ruff check .
python -m pytest
```

新增因子时，只在 `src/quantcta/factors/` 增加普通函数及对应测试，不要修改
`backtest/engine.py`。如果确实需要改变统一回测口径，应单独发 PR 并由至少一名
核心框架负责人复核。

纯因子研究统一调用 `quantcta.research.evaluate_factor`，禁止各自重新实现未来
收益标签。日线因子默认 `availability_lag=1`，必须传入实际前月合约代码以自动
剔除跨换月样本。先报告 IC/RankIC/Newey-West t/Q5-Q1；形成明确 Signal 后才进入
回测引擎讨论 Sharpe、成本和风控。

提交因子结论前还必须附上：`evaluate_factor_by_year` 的年度结果、
`factor_autocorrelation` 的 1/5/21 bar 衰减，以及原始因子和
`causal_winsorize` 版本的对照。用 `save_factor_results` 保存 manifest，
不要只发截图或手抄一个最好看的数字。

## 四、提交规范

- 提交信息简短描述改了什么（中文或英文均可）。
- 不要把数据文件、日志、密钥提交入库（`.gitignore` 已忽略常见类型）。
- 禁止在代码、notebook、截图或 Issue 中粘贴 Tushare/IBKR 密钥。
- 每次 push 前先 `git pull`，减少冲突。
- 冲突时先手动解决冲突文件，再提交合并。

## 五、常见问题

- 推送或拉取网络失败：稍后重试；若长期无法访问，联系管理员配置代理。
- 不确定操作是否影响他人：先问再改，优先 Pull Request 而不是直接推 main。
