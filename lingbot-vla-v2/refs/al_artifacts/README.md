# AL 派生产物「离线复现包」（避免重建后重复计算）

> **为什么有这个目录**：`task_splits` / `scout` / `hardness` / 阈值表 / `datasets.txt` 这些
> **都不是源码，而是"由数据集 + 配置算出来的派生数据"** —— 它们不进主仓库的历史，
> 但**重建实例后会全部丢失，重算一次要几十分钟到几小时**（2026-10-11 实测：找 + 重算
> 花了约 1 小时，其中 `task_baseline` 单任务 ~90 s、hardness 单任务单卡 ~1000 s）。
> 因此把它们**打包入库**，重建后一条命令恢复，**不必重算**。

体积：**109 个文件 / 约 216 KB**（压缩后更小）。全部是 JSON / 文本，无大文件。

---

## 0. 一条命令恢复

```bash
# 本包位于【仓库根】下的 refs/al_artifacts/（仓库 = lingbot-vla-v2）
# ① 仓库已在训练机上（推荐，不用再走一遍 SSH）：
cd /workspace/lingbot_vla_2/lingbot-vla-v2
bash refs/al_artifacts/restore.sh --local

# ② 从开发机推到训练机：
bash refs/al_artifacts/restore.sh root@<host> <port> <本机私钥路径>
# 例（本会话）：bash refs/al_artifacts/restore.sh root@36.150.116.206 32763 .ssh/cpu1_ed25519
```

恢复脚本会：① 把文件放到下方「落位表」的路径；② 用 `SHA256SUMS` 校验；
③ 打印还缺什么（例如 hardness 缓存未覆盖的任务）。

---

## 1. 内容与落位表

| 包内路径 | 训练机落位 | 是什么 | 丢了要多久才能重算 |
|---|---|---|---|
| `al/task_splits_50/manifest.json` | `/workspace/al/task_splits_50/manifest.json` | 50 任务 + train/val 回合划分（`tools/task_split.py` 的产物） | 秒级（但要先有聚合数据集） |
| `al/task_splits_50/*.train_ids.json` / `*.val_ids.json` | 同上目录 | 每任务的 train/val 回合白名单（104 个） | 同上 |
| `al/task_splits_50/task_baseline.json` | 同上目录 | **每任务训练前基线 MSE**（NMSE 的分母） | **~90 s/任务 × 50 ≈ 75 分钟**（纯 CPU） |
| `al/phases_al/datasets.txt` | `/workspace/al/phases_al/datasets.txt` | **单行聚合数据集**清单（训练侧 `--data.train_path` 指向它） | 秒级（但要知道"必须单行"这个约束） |
| `al/scout_cache/scout.json` | `/workspace/al/scout_cache/scout.json` | scout 开环指标缓存（50/50 覆盖） | **~4 分 44 秒**（50 任务 / 8 分片首扫） |
| `al/hardness_cache/hardness.json` | `/workspace/al/hardness_cache/hardness.json` | hardness 逐样本难度缓存（模型名判定有效性） | **单卡 ~1000 s/任务**（8 卡 rank 分片约 1/8） |
| `eval_results/open_loop/ref50k/pass_thresholds_gmean100_warn.json` | `/workspace/eval_results/open_loop/ref50k/…` | **通过线阈值表**（GMean×100） | 秒级（但依赖 `ref_per_traj.jsonl`） |
| `eval_results/open_loop/ref50k/ref_per_traj.jsonl` | 同上目录 | 50k 参考模型逐轨迹 MSE（生成阈值表的唯一输入） | **不可重算**（需 50k 参考模型；必须从历史机器带出来） |

---

## 2. 关键不变量（**恢复后必须核对，否则会静默错判**）

| # | 不变量 | 怎么核对 |
|---|---|---|
| 1 | **`task_baseline.json` 与 `pass_thresholds_*.json` 的 `config_fingerprint` 必须相同** | `python3 -c "import json;print(json.load(open('…task_baseline.json'))['config_fingerprint'], json.load(open('…pass_thresholds_gmean100_warn.json'))['config_fingerprint'])"` —— 本包内两者同为 **`0e9f13836c74914d`** |
| 2 | **`scout.json` / `hardness.json` 的 `model` 必须等于 `--model-name`** | 本包内为 **`robbyant_lingbot-vla-v2-6b-bf16`**；不一致 ⇒ 缓存整体判为未命中（会真扫一遍） |
| 3 | **`datasets.txt` 必须是「单行 + 聚合数据集」** | `wc -l` 应输出 1；给 50 行分任务清单会导致 `resolver` 报 `下钻不到 hf_dataset` |
| 4 | **`manifest.json` 的回合号是聚合数据集的全局编号（0–2499）** | 与 `datasets.txt` 指向的聚合数据集必须是同一份（本包对应 `…/agg_lerobot_v30/RoboTwin_lerobot_v30`） |

> ⚠️ **不变量 1 的后果**：若只有一边重算过（例如重算了 baseline 却没重算阈值表），
> 通过线会与 baseline 错位，而**程序不会报错** ⇒ 只能靠指纹对拍发现。所以本包把两者一起带。

---

## 3. 各项的重算命令（想重算时照抄）

```bash
cd /workspace/lingbot_vla_2/lingbot-vla-v2
PY=/opt/robotwin-env/bin/python
AGG=/models/robotwin-persistent/data/agg_lerobot_v30/RoboTwin_lerobot_v30

# ① 任务划分（manifest + 每任务 ids）
$PY tools/task_split.py --task all --dataset "$AGG" --out /workspace/al/task_splits_50

# ② baseline（每任务训练前 MSE；纯 CPU，~90 s/任务）
$PY -m lingbotvla.auto_learning.tools.compute_task_baseline \
    --manifest /workspace/al/task_splits_50/manifest.json \
    --config configs/rocm/robotwin_official_paths_rocm.yaml \
    --out /workspace/al/task_splits_50/task_baseline.json

# ③ datasets.txt（**必须单行**）
printf 'robotwin %s\n' "$AGG" > /workspace/al/phases_al/datasets.txt

# ④ scout / hardness：不用单独跑，launcher 会按缓存覆盖情况自动补扫
#    全量重扫：给 al_launch.py 加 --no-cache

# ⑤ 阈值表（需要 ref_per_traj.jsonl）
$PY -m lingbotvla.auto_learning.tools.build_gmean_thresholds \
    --ref-per-traj /workspace/eval_results/open_loop/ref50k/ref_per_traj.jsonl \
    --baseline /workspace/al/task_splits_50/task_baseline.json \
    --reference global_step_50000 --multiplier 100 \
    -o /workspace/eval_results/open_loop/ref50k/pass_thresholds_gmean100_warn.json
```

---

## 4. 快照口径与更新方式

| 项 | 说明 |
|---|---|
| 快照时间 | **2026-10-11 03:40 UTC**（重建后首次把 8 卡训练跑通的时刻） |
| `hardness.json` 状态 | ⚠️ **部分填充**（当时训练刚起步，只有少数任务扫完）；训练会持续补全。**训练完成/暂停后应重新快照并覆盖本包** |
| `scout.json` 状态 | ✅ 50/50 完整 |
| 更新方式 | 在训练机上 `tar czf` 打包后拉回本目录覆盖，再 `python3` 重跑 `SHA256SUMS` 生成脚本（见仓库提交历史），最后 `git add refs/al_artifacts && git commit` |

> 🔴 **`ref_per_traj.jsonl` 属于"不可再生资产"**（需要 50k 参考模型）—— 任何清理 `/workspace`
> 的操作前，先确认它已被本包收录。
