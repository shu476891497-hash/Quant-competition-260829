# 团队协作指南

本仓库是量化竞赛团队代码的同步仓库。所有成员通过 GitHub 协作，请按本指南操作。

## 一、加入仓库

1. 将你的 **GitHub 用户名**发给项目管理员（yyf0507）。
2. 管理员在仓库 **Settings → Collaborators** 中将你添加为协作者。
3. 你收到 GitHub 邀请邮件后，点击 **Accept invitation** 即可访问本仓库。

## 二、获取代码

首次使用，克隆仓库到本地：

```bash
git clone https://github.com/yyf0507/Quant-competition-260829.git
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

## 四、提交规范

- 提交信息简短描述改了什么（中文或英文均可）。
- 不要把数据文件、日志、密钥提交入库（`.gitignore` 已忽略常见类型）。
- 每次 push 前先 `git pull`，减少冲突。
- 冲突时先手动解决冲突文件，再提交合并。

## 五、常见问题

- 推送或拉取网络失败：稍后重试；若长期无法访问，联系管理员配置代理。
- 不确定操作是否影响他人：先问再改，优先 Pull Request 而不是直接推 main。
