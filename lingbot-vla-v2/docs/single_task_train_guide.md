# 单任务训练探针 使用文档

> 适用版本：`YonG00/lingbot_vla_2`（LingBot-VLA-v2 post-training）
> 相关文件：`experiment/robotwin/single_task_train.sh`（本脚本）、`tools/task_split.py`（前置，产划分）
> 相关参数：`--data.episode_ids_file` / `--train.max_steps` / `--train.save_steps` / `--train.global_batch_size`

---

## 1. 这个功能解决什么问题

L1 整段（14 个任务 / 700 回合）训练 2 个 epoch **完全没有效果**：
闭环 0/12，开环 MSE 仍比「输出常数均值」差 87%。

为了**便宜地**回答「模型到底能不能学会一个任务」，把范围缩到**单个任务**：
一次只训一个 task（40 回合），跑几百到一千多步，然后用开环诊断看有没有动起来。

**本脚本 = 单任务训练探针**：不做闭环评测（那一步手动跑），只负责把训练跑起来。

---

## 2. 一句话原理

> 用**一个 `TASK` 变量**驱动全部：读 `task_split` 产出的 `train_ids` ⇒ 自动算每轮步数、
> 自动定 epoch 数（让 `max_steps` 真正生效）、自动选一个**能整除 `max_steps` 的 `save_steps`**
> （保证末步一定有存档），然后交给官方 `train.sh` 跑。

---

## 3. 前置：先生成数据划分

```bash
cd /data/code/lingbot-vla-v2
python tools/task_split.py --task click_bell
```

详见 `docs/task_split_guide.md`。

---

## 4. 快速开始

```bash
cd /data/code/lingbot-vla-v2
conda activate lingbotvla

# ① 先探测显存（3 步就停；3 < save_steps 所以不会写任何存档）
MAX_STEPS=3 TASK=click_bell bash experiment/robotwin/single_task_train.sh

# ② 正式跑
TASK=click_bell bash experiment/robotwin/single_task_train.sh

# ③ 只打印计划、不训练
DRY_RUN=1 TASK=click_bell bash experiment/robotwin/single_task_train.sh
```

`click_bell` 的实测计划（`DRY_RUN=1` 输出）：

```
训练规模    = 9 epoch × 192 步/轮，max_steps=1500 ⇒ 总 1500 步
批大小      = micro 16 × gas 1 × 1 卡 = gbs 16
存档计划    = 每 500 步一份（目标 600，已对齐 max_steps）× 3 份；轮末不存
剪枝看门狗  = PRUNE=1  keep-last=0  ⇒ 单份 72G → 24G
```

---

## 5. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `TASK` | 必填（或位置参数 `$1`） | 任务名 |
| `MICRO` / `GAS` / `N_GPU` | `16` / `1` / `1` | `gbs = MICRO × GAS × N_GPU` |
| `MAX_STEPS` | `1500` | 总步数（由 `max_steps` 驱动，不被 epoch 截断） |
| `SAVE_EVERY` | `600` | **期望**的存档间隔；脚本会自动调整成 `MAX_STEPS` 的约数 |
| `AUGMENT` | `false` | `--data.image_augment`（与官方一致） |
| `PRUNE` / `PRUNE_KEEP` | `1` / `0` | 剪枝看门狗开关 / 保留几份完整 DCP（0 = 全剪，放弃续训） |
| `TRAIN_OUT` | `/data/outputs/single/$TASK` | 训练输出目录 |
| `SPLIT_DIR` | `/data/train/task_splits` | 划分文件目录 |
| `DRY_RUN` | `0` | `1` = 只打印计划 |

---

## 6. 与 `phase1_L1_vit_frozen_train_then_eval.sh`（昨晚对照组）的差异

冻结配置**完全一致**（`train_expert_only=false` + `freeze_vision_encoder=true`），只有三点不同：

| | 昨晚对照组 | 单任务探针 |
|---|---|---|
| 数据 | 14 个 L1 任务（700 回合） | **单个 task**（40 回合） |
| 卡数/批 | 4 卡 micro 14 / gas 2 ⇒ gbs 112 | **单卡 micro 16 / gas 1 ⇒ gbs 16** |
| `image_augment` | `true` | **`false`**（与官方一致） |

> ⚠️ **归因局限**：同时动了两处（数据 + `image_augment`），若探针成功**无法单独归因**。
> 补救很便宜：用 `AUGMENT=true` 反跑一次即可分离。

---

## 7. 两个自动推导（为什么不能照抄旧脚本）

1. **`STEPS_PER_EPOCH = train_frames / gbs`**（读 `manifest.json` 里的 `train_frames`）。
   单任务只有 40 回合 ⇒ 一轮只有 ~28–63 步（整段 L1 是 779 步）⇒ **"跑 N 轮"完全不是同一个量纲**，必须按**目标步数**反推。
2. **`SAVE_STEPS` = 「≤ `SAVE_EVERY` 的 `MAX_STEPS` 最大约数」**。
   例：`MAX_STEPS=1500`、`SAVE_EVERY=600` ⇒ 取 **500**（600 不整除 1500，直接写 600 会**丢掉末份存档**）。
   配 `SAVE_EPOCHS=0` 就安全了。

---

## 8. 磁盘与剪枝看门狗

单份完整存档 = **72G**（`model/` 24 + `optimizer/` 24 + `hf_ckpt/` 24）。
3 份 = 216G —— **通常放不下**，所以脚本**默认开剪枝看门狗**（`PRUNE=1`）：

- 拉起 `tools/prune_dcp.py --keep-last 0 --min-age-seconds 300 --interval 120` 常驻
- 每份 `hf_ckpt` **完整性校验通过后**剪掉 DCP ⇒ **单份降到 24G**，3 份 = 72G
- 训练结束自动停掉看门狗；日志在 `$TRAIN_OUT/prune_dcp.log`

> `PRUNE_KEEP=1` 可保留最新一份完整 DCP（保住"从最近存档续训"的能力）。

---

## 9. 显存

单卡要把**全部**权重 + 梯度 + 优化器放进一张卡（FSDP2 在单卡上不分片）：

| 项 | 大小 |
|---|---|
| 权重 F32（6.376B） | 25.5G |
| 梯度 F32（5.961B 可训） | 23.8G |
| 优化器状态 | 23.7G |
| **小计** | **≈73G** |
| 留给激活值 + CUDA context | 仅 ~23G |

若 OOM：
1. 降 `MICRO`（如 `MICRO=12`）
2. 加 `--train.enable_gradient_checkpointing true`（会变慢）

**建议先跑 `MAX_STEPS=3` 探测一次**，别直接上 1500 步。

---

## 10. 训练之后（手动）

```bash
# 开环诊断：base 基线 + 训练后的 ckpt，用同一批 traj_ids
python tools/collect_open_loop.py          # 汇总所有开环结果 → registry.json
```

详见 `docs/open_loop_guide.md`。**闭环评测本脚本不跑**，需要时手动执行 launcher。

---

## 11. 注意事项

1. **`TRAIN_OUT` 默认是 `/data/outputs/single/$TASK`** —— 不要指向 `phase1_*` 的老目录，否则会污染现有 ckpt。
2. **训练 stdout 不落盘**（与官方 `train.sh` 一致）。要看日志请自己 `| tee`，或看 TB 的 `$TRAIN_OUT/runs/`。
3. **`gbs≠112` 会与昨晚对照组不可比** —— 单任务探针本来就不跟它比，但跨探针之间请保持同一组 `MICRO/GAS`。
4. 脚本尾部会打印**开环评测的现成命令**（含 train/val 两组的 `--traj_ids`），可直接复制。
