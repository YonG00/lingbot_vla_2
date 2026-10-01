# L1–L4 阶段训练使用说明

本文说明如何在 RoboTwin clean 数据集上，按 **L1 → L1+L2 → L1+L2+L3 → All** 四个阶段
做累积式（curriculum）后训练。

配套文档：
- 课程设计（等级划分与依据）：[`robotwin_curriculum_v1.md`](./robotwin_curriculum_v1.md)
- 课程配置文件：[`configs/curriculum/robotwin_curriculum_v1.yaml`](../configs/curriculum/robotwin_curriculum_v1.yaml)

> ⚠️ **注意**：`robotwin_curriculum_v1.md` 的 **第 5 节（L1–L4 等级划分）仍然有效**，
> 但其 **第 6 节的「8 阶段 C1–C8」设计已废弃**。现行四阶段设计见本文。
> 该文件是历史设计记录，暂未改动；两者的换算关系见本文第 8 节。

---

## 1. 一句话概括

**四个阶段共用同一份数据集，只是每个阶段读不同的回合子集。**

阶段选择由一份「回合号白名单」表达，训练时通过
`--data.episode_ids_file` 传给加载器，最终透传给 LeRobot 原生的
`LeRobotDataset(episodes=...)`。

**不复制数据、不重编码视频、不额外占磁盘。**

---

## 2. 原理：两个正交的问题

数据集 `RoboTwin_lerobot_v30` 是把 50 个 RoboTwin 任务各 50 个回合**合并成的一份**
LeRobot v3 数据集，且**按任务连续分块**：

```
block_id = episode_index // 50
task     = TASK_ORDER[block_id]
```

| 特征 | 值 |
|---|---|
| 任务数 | 50（每任务恰好 50 回合） |
| 回合数 | 2500 |
| 帧数 | 548,893 |
| fps | 15 |
| 相机 | 3 路（cam_high / cam_left_wrist / cam_right_wrist） |
| state / action 维度 | 14 |

边界示例：`ep49 → block0 adjust_bottle`，`ep50 → block1 click_bell`，
`ep100 → block2 hanging_mug`。

「数据在哪」和「读哪些回合」是**两个独立问题**，所以用两个文件：

| | `datasets.txt` | `phase<N>_<levels>.episode_ids.json` |
|---|---|---|
| 回答的问题 | 数据**在哪** | 读**哪些**回合 |
| 格式 | 每行 `名称 路径` | JSON 整数数组（升序去重） |
| 传给训练的参数 | `--data.train_path` | `--data.episode_ids_file` |
| 四个阶段是否变化 | ❌ 恒定同一份 | ✅ 每阶段不同 |
| 代码消费方 | `get_all_tasks()` → `MultiVLADataset` | `LeRobotDataset(episodes=[...])` |

> `datasets.txt` 的**第一列是 `data_name`**，它决定
> `configs/robot_configs/<data_name>.yaml`。对 RoboTwin 必须是 **`robotwin`**，
> 不能写成数据集名。

---

## 3. 快速开始

### 步骤 1：生成四个阶段的数据文件

```bash
cd /data/code/lingbot-vla-v2

python tools/prepare_phase.py --phase all --gbs 112
```

产出到 `/data/train/phases/`：

```
datasets.txt                          数据集清单（共享）
phase1_L1.episode_ids.json            700 个回合
phase2_L1_L2.episode_ids.json        1450
phase3_L1_L2_L3.episode_ids.json     2000
phase4_all.episode_ids.json          2500
README.md                             自动生成的同目录说明
```

终端会打印摘要：

```
  阶段         等级               任务     回合        帧数  steps/epoch       预计耗时
  P1         L1               14    700     87266          779        43分
  P2         L1+L2            29   1450    207872         1856       1.7时
  P3         L1+L2+L3         40   2000    360357         3217       3.0时
  P4         L1+L2+L3+L4      50   2500    548893         4901       4.5时
```

`--gbs` **只影响打印出来的 steps/epoch 与耗时预估，不改变任何数据**。
耗时参考值来自 *4×RTX PRO 6000 96G，micro=28，gbs=112*，可用
`--step-time` 覆盖。

### 步骤 2：启动 Phase 1

```bash
cd /data/code/lingbot-vla-v2
export PATH=/data/miniconda3/envs/lingbotvla/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash train.sh tasks/vla/train_lingbotvla.py \
  /data/train/configs/robotwin_official_paths.yaml \
  --data.train_path        /data/train/phases/datasets.txt \
  --data.episode_ids_file  /data/train/phases/phase1_L1.episode_ids.json \
  --train.output_dir       /data/outputs/phase1_L1 \
  --train.micro_batch_size 28 \
  --train.gradient_accumulation_steps 1 \
  --train.global_batch_size 112 \
  --train.train_expert_only true \
  --data.image_augment true
```

相比原来的训练命令，**只多了两个 `--data.*` 参数**。

> `global_batch_size` 必须等于
> `micro_batch_size × 卡数 × gradient_accumulation_steps`，
> 否则会被官方校验直接拒绝。

### 步骤 3：切换阶段

只改两处：**白名单文件** 和 **输出目录**。

```bash
  --data.episode_ids_file  /data/train/phases/phase2_L1_L2.episode_ids.json \
  --train.output_dir       /data/outputs/phase2_L1_L2 \
```

阶段必须**累积**（L1 → L1+L2 → L1+L2+L3 → All），不要跳级，避免灾难性遗忘。

> **Phase 4 提示**：`phase4_all` 等于全量，可以**不传** `--data.episode_ids_file`
> 直接训练全量数据，效果等价。

---

## 4. 验证筛选是否生效

启动后日志里应出现：

```
[episode_ids] 生效白名单: 700 个回合, 范围 50..2499  (phase1_L1.episode_ids.json)
```

以及步数分母：

| Phase | gbs=64 | gbs=112 | gbs=128 |
|---|---:|---:|---:|
| P1 (L1) | 1,364 | **779** | 682 |
| P2 (L1+L2) | 3,248 | 1,856 | 1,624 |
| P3 (+L3) | 5,631 | 3,217 | 2,815 |
| P4 (All) | 8,576 | 4,901 | 4,288 |

**分母与全量不同，就说明筛选生效了。**

---

## 5. 四阶段对照

| Phase | 等级 | 任务数 | 回合数 | 帧数 |
|---|---|---:|---:|---:|
| P1 | L1 基础原子技能 | 14 | 700 | 87,266 |
| P2 | L1+L2 带约束基础技能 | 29 | 1,450 | 207,872 |
| P3 | +L3 技能组合 | 40 | 2,000 | 360,357 |
| P4 | +L4 复杂组合技能（=全量） | 50 | 2,500 | 548,893 |

各等级的完整任务清单见
[`robotwin_curriculum_v1.md`](./robotwin_curriculum_v1.md) 第 5 节，
或运行：

```bash
python -c "
import sys; sys.path.insert(0,'tools')
from robotwin_curriculum import *
cfg = load_curriculum('configs/curriculum/robotwin_curriculum_v1.yaml')
for lv, info in cfg['skill_levels'].items():
    print(lv, len(info['tasks']), sorted(info['tasks']))
"
```

---

## 6. 常用参数

```
python tools/prepare_phase.py --help

  --phase {1,2,3,4,all}   要生成哪个阶段 (默认 all)
  --out-dir DIR           输出目录 (默认 /data/train/phases)
  --gbs N                 可选; 仅用于打印 steps/epoch 与耗时预估
  --step-time S           可选; 单步秒数, 用于耗时预估 (默认 3.31)
  --check-csv PATH        可选; 与旧 8 阶段 manifest 交叉验证
  --curriculum PATH       可选; 课程配置路径
```

`--check-csv` 会断言「按等级算出的回合集合」与「旧构建脚本产出的
`first_stage_num <= 2*phase` 累积集合」完全一致，用于防止回归。

---

## 7. 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `episode_ids_file 不存在` | 还没生成阶段文件 | 先跑 `prepare_phase.py` |
| `最大回合号 N 超出数据集总回合数` | 白名单来自别的数据集 | 重新用本数据集的 `prepare_phase.py` 生成 |
| `episode_ids_file 是空数组` | 文件损坏或被清空 | 重新生成 |
| `episode_ids_file 只支持单数据集清单` | `datasets.txt` 有多行 | 本期白名单是全局回合号，只支持单数据集；删到一行 |
| `manifest 第一列 X 对应的机器人配置不存在` | `dataset.data_name` 写错 | 改成 `robotwin` |
| 训练集大小与预期不符 | 忘了传 `--data.episode_ids_file`，或传错阶段 | 看日志的 `[episode_ids] 生效白名单` 与 `Step 1/N` 分母 |
| 不传白名单时行为是否变了 | **没变**，仍加载全部 2500 回合 / 548,893 帧 | 已回归验证 |

---

## 8. 设计取舍

### 为什么用回合号白名单，而不是物理切分数据集？

LeRobot v3 的视频**不是按回合分文件的** —— 12 个 mp4 承载了 2500 个回合的片段，
靠 `meta/episodes` 里的 `from_timestamp` / `to_timestamp` 定位。

物理切分意味着**重编码全部视频**（3.8 GB），代价高且容易引入误差。
LeRobot 0.4.2 原生支持 `LeRobotDataset(episodes=[...])`，因此选择加载期筛选：
零数据复制、零磁盘开销、可随时增删阶段。

### 为什么不再分短/长轨迹？

早期设计把每个等级再按轨迹长度拆 short/long，共 8 阶（即
`robotwin_curriculum_v1.md` 第 6 节的 C1–C8）。项目规范已明确
**不采用「短轨迹 → 长轨迹」的二级 curriculum**（理由：短轨迹不一定简单，
长轨迹可能只是起点更远或停顿更多），故收敛为 4 阶。

短/长**仅作为长度统计维度保留**（见课程配置的 `trajectory_split`，其
`used_for_curriculum` 已置为 `false`），不参与分阶段。

**新旧换算关系**：四个阶段恰好等于旧 8 阶段的偶数累积阶 ——

```
P1 == C2 (700)    P2 == C4 (1450)    P3 == C6 (2000)    P4 == C8 (2500)
```

这不是巧合，而是「等级累积」与「等级×长短累积」在含双长短时必然相等。
`prepare_phase.py --check-csv <旧 manifest.csv>` 会断言这个等式，
所以从旧设计迁移是**可验证**的，不是靠假设。

### 与 `seen` / `unseen` 的关系

两者**正交**，不要混淆：

`seen` / `unseen` 指的是**背景纹理域**（`assets/background_texture/{seen,unseen}/`）。
采集数据时 `eval_mode=False`，因此**这 2500 条训练轨迹属于 `seen` 域**；
评测时切换到 `unseen` 纹理，用来衡量对新背景的泛化。

L1–L4 描述的是**操作技能复杂度**，与纹理域无关。

---

## 9. 相关文件

| 路径 | 作用 |
|---|---|
| `tools/prepare_phase.py` | 阶段文件生成器（CLI） |
| `tools/robotwin_curriculum.py` | 共享映射模块：`TASK_ORDER` + 回合→任务→等级 |
| `tools/build_robotwin_curriculum_manifest.py` | 旧的 manifest 构建脚本（保留） |
| `configs/curriculum/robotwin_curriculum_v1.yaml` | 课程配置：等级划分 + 4 个 phase 定义 |
| `lingbotvla/utils/arguments.py` | `DataArguments.episode_ids_file` 字段 |
| `lingbotvla/data/vla_data/base_dataset.py` | 白名单读取与校验，透传 `episodes=` |
| `lingbotvla/data/vla_data/multi_vla_dataset.py` | 单数据集守卫 |
