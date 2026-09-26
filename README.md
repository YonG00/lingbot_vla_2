# lingbot_vla_2 — LingBot-VLA 2.0 + RoboTwin-lingbot 工作快照

本仓库为多项目合集快照, 由服务器 `gpu-kvm` 抓取并整理。
各子目录已剥离原有 git 历史, 统一以单个快照提交保存。

| 子目录 | 来源 | 说明 |
| :--- | :--- | :--- |
| `lingbot-vla-v2/` | https://github.com/Robbyant/lingbot-vla-v2 | LingBot-VLA 2.0 官方代码 (upstream `main`, 含 1 处本地改动: `experiment/robotwin/eval_policy_client_lingbotvla.py`) |
| `RoboTwin-lingbot/` | https://github.com/RoboTwin-Platform/RoboTwin | RoboTwin + LingBot 评测集成 (@13c3c47), 含 `script/deploy/`、`script/eval_policy_client_lingbotvla.py` |

> **上游 RoboTwin 本体有意未纳入本仓库。**
> 服务器上另有源码副本位于 `/data/code/RoboTwin`, 供源码编译链接使用, 属本地专用依赖。

## 快照时排除的内容

以下内容体积过大或可重新生成, 未纳入本仓库, 需要时请通过各自上游重新获取:

- 各项目 `assets/` 素材 (RoboTwin 9.0G, RoboTwin-lingbot 5.3G, lingbot-vla-v2 16M)
- `RoboTwin/assets/objects.zip` (3.5G)、`embodiments.zip` (210M)
- 上游 RoboTwin 本体 (`/data/code/RoboTwin`)
- `__pycache__/`、`*.pyc`、`*.egg-info/`、`*.zip`、`*.bak`
- 模型权重与训练产出 (各项目 `.gitignore` 另行排除 `models/`、`result/`、`checkpoints/`、`eval_result/` 等)

## 环境

训练环境为 conda 环境 `lingbotvla` (Python 3.12 + PyTorch 2.8.0), 由
`lingbot-vla-v2/tools/create_train_env.sh` 创建。
