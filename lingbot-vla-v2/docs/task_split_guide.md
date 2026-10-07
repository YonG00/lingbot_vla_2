# 按任务划分训练/验证集 使用文档

> 适用版本：`YonG00/lingbot_vla_2`（LingBot-VLA-v2 post-training）
> 相关文件：`tools/task_split.py`（本工具）、`tools/robotwin_curriculum.py`（复用其块结构常量）
> 产物目录：`/data/train/task_splits/`（默认）

---

## 1. 这个功能解决什么问题

做**单任务训练**或**开环评测**时，需要把某个任务的 50 个回合切成「训练用」和「验证用」两部分，并且要求：

- **可复现** —— 今天切的和下周切的是同一批，否则两次结果没法比
- **验证集能代表整个任务** —— 10 个样本时，"分布是否代表整体"比"随机不随机"重要得多
- **与课程解耦** —— 单任务训练不该被 `phases` / `skill_levels` 那套牵连

本工具就是干这件事：**给一个任务名，输出训练白名单 + 验证白名单 + 一份可审计的 manifest**。

---

## 2. 一句话原理

> 数据集是**定长块**结构：`block_id = episode_index // 50`，`task = TASK_ORDER[block_id]`
> ⇒ 每个任务恰好 50 个回合、整块属于该任务。
> 在该任务的 50 个回合内，**按长度升序分层**、等分成 `n_val` 层、每层取层内中位那一条作验证集。

---

## 3. 快速开始

```bash
cd /data/code/lingbot-vla-v2
conda activate lingbotvla

# 列出全部 50 个任务名（含 block 与回合区间）
python tools/task_split.py --list

# 生成 click_bell 的划分（默认 val_ratio=0.2 ⇒ 40 训练 / 10 验证）
python tools/task_split.py --task click_bell

# 多任务子集（逗号分隔）—— 多任务 Auto Learning / 课程子集用这个
python tools/task_split.py --task click_bell,click_alarmclock --out /data/train/task_splits_2task

# 全部 50 个任务都生成
python tools/task_split.py --task all
```

产出：

```
/data/train/task_splits/
├── click_bell.train_ids.json   裸列表 [50, 51, ...] ⇒ 直接喂 --data.episode_ids_file
├── click_bell.val_ids.json     裸列表 [53, 60, ...] ⇒ 开环评测用
└── manifest.json               参数 + 各任务明细 + 帧数 + sha256（审计 / 基线匹配）
```

**多个任务时**额外产出一份合并白名单（单任务不产出，避免多余文件）：

```
/data/train/task_splits_2task/
├── click_bell.train_ids.json / .val_ids.json
├── click_alarmclock.train_ids.json / .val_ids.json
├── combined.train_ids.json     ← 两个任务 train 回合的并集，喂 --data.episode_ids_file
├── combined.val_ids.json       ← 两个任务 val 回合的并集
└── manifest.json               ← **只含这 2 个任务**
```

> ⚠️ **manifest 必须只含你要用的那几个任务**：`TaskCatalog.attach_samples()` 要求
> manifest 里**每个**任务在当前数据集白名单里都有样本，多列一个任务就直接报错
> （`[<task>] 回合 N 在当前数据集里一个样本都没有`）。
> 所以「多任务」要用 `--task a,b` 生成**配套的** manifest，**不要**拿 `--task all`
> 的全量 manifest 去配一个 2 任务的白名单。

---

## 4. 取开环评测用的轨迹索引

**stdout 只输出数字、日志全部走 stderr**，所以可以直接内联进命令：

```bash
# 从训练集等间隔抽 5 条（测「拟合」）
python tools/task_split.py --task click_bell --pick-train 5
# 从验证集等间隔抽 10 条（测「泛化」；val 恰好 10 条 ⇒ 全取）
python tools/task_split.py --task click_bell --pick-val 10

# 串起来一次跑完
python scripts/open_loop_eval.py \
  --model_path <hf_ckpt> --robo_name robotwin --data_path <数据集根> \
  --traj_ids $(python tools/task_split.py --task click_bell --pick-train 5) \
             $(python tools/task_split.py --task click_bell --pick-val 10) \
  --use_length 50 --chunk_ret true --save_plot_path <输出目录>
```

> 抽 n 条是**按 episode_index 升序等间隔取**（含首尾），确定性；不是随机。

---

## 5. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--task` | 必填 | 任务名、**逗号分隔的多个任务名**（如 `click_bell,click_alarmclock`），或 `all`（全部 50 个）。名字写错会直接报错并提示用 `--list` 查 |
| `--list` | — | 列出全部任务名后退出 |
| `--val-ratio` | `0.2` | 验证集比例（1/5 ⇒ 50 个回合里 10 条） |
| `--strategy` | `quantile` | `quantile`=按长度分层（推荐）｜`seed`=固定种子随机｜`stride`=等间隔 |
| `--seed` | `20261004` | 仅 `--strategy seed` 用 |
| `--out` | `/data/train/task_splits` | 产物目录 |
| `--dataset` | `/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30` | 数据集根 |
| `--pick-train N` | `0` | 从 train 里等间隔抽 N 条打印到 stdout |
| `--pick-val N` | `0` | 从 val 里等间隔抽 N 条打印到 stdout |

**三种策略怎么选**：

| 策略 | 优点 | 缺点 |
|---|---|---|
| **`quantile`（默认）** | val 的长度分布与整体一致（均值/分位都对得上） | 层内取中位 ⇒ **最极端的少数几条会留在 train** |
| `seed` | 无周期偏置 | 小样本下分布可能聚簇（实测 click_bell 出现 60/61/62 三连） |
| `stride` | 最均匀、可手算 | 若数据按生成变体循环、且周期能被整除 ⇒ 系统性偏 |

实测 `click_bell`（`quantile`）：train 40 条 / 3,085 帧（均值 77.1）｜val 10 条 / 770 帧（**均值 77.0**）⇒ 均值几乎完全一致。

---

## 6. 注意事项

1. **和课程 `phases/` 的关系**：两套白名单**不要混用**。`--data.episode_ids_file` 只能指一份 —— 单任务训练用 `task_splits/`，分阶段训练用 `phases/`。
2. **现有 ckpt 已污染**：`phase1_L1` 等阶段把 L1 每个任务的 **50 个回合全训过**，所以那批 ckpt 在 L1 任务上的开环结果测的是「拟合」不是「泛化」。要真正的同分布留出，**只能对之后的训练生效**。
3. **数据集结构硬校验**：脚本会断言「2500 回合 / 548,893 帧 / 每任务恰好 50 条」，任一不符直接报错退出 —— 这是防止块映射静默错位。
4. **幂等**：同输入同参数必得同输出；重跑时会优先复用 `manifest.json` 里已有的划分（参数一致才复用）。
5. **`--val-ratio` 改了就重算**：manifest 里记了 `val_ratio` / `strategy`，参数不一致会自动重新划分并覆盖。
6. 本工具**只产划分文件**，不产闭环评测用的 `phase*_eval.txt`（那是课程 `prepare_phase.py` 的职责）。
