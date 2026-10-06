# Stage B1 —— Auto Learning 接入真实训练循环 · 集成设计

> 适用版本：`YonG00/lingbot_vla_2` · 分支 `auto-learning/b1-integration`
> 上游：`stage_b0_adapter_design.md`（B0 接口层，已完成）、`stage_b1_direction_design_test_plan_v0_1.md`（本阶段方向与测试计划）
> 目标环境：**单卡 48GB BF16**；本文件**不**涉及 96GB / 4-task 正式实验（那是 B2）

---

## 0. 一句话

> 用**一个可关闭的附加层**，把已经验证过的 Auto Learning Scheduler 接到真实训练循环上：
> **换一个 sampler**（7 NEW + 3 Replay）、**在 unit 边界调一次已有的 OpenLoopValidator**、
> **状态进 `extra_state`**；`enabled=false` 时一行都不生效。

---

## 1. 插入点在哪里（真实训练循环）

`tasks/vla/train_lingbotvla.py` 的循环骨架（行号为当前 HEAD）：

```
488  train_dataset = build_vla_dataset(...)                     # ← 数据集只建一次
491  train_dataloader = build_dataloader(...)                   # ← ① 在这里换 sampler
...
828  for epoch in range(...):
841      data_iterator = iter(train_dataloader)
842      for epoch_step in range(start_step, args.train.train_steps):
844          global_step += 1
863          micro_batches = next(data_iterator)                 # ← ② 一个 optimizer step 的样本
882          for micro_batch in micro_batches:                   # ← 梯度累积（B1 下恒为 1）
925              model_outputs = model(**micro_batch)            # ← forward
963              loss.backward()                                 # ← backward
1023         optimizer.step(); lr_scheduler.step()               # ← ③ optimizer step
1271         if open_loop_validator and global_step % open_loop_eval_steps == 0:
1273             open_loop_validator.validate(global_step)       # ← ④ 已有的安全评测入口
```

**三个插入点**：

| # | 位置 | 做什么 |
|---|---|---|
| ① | `build_dataloader(...)` 调用处 | `auto_learning.enabled` 时：**强制 `rmpad=false`** + 传入 `AutoLearnSampler` + 断言 `num_micro_batch == 1` |
| ② | `next(data_iterator)` 之前 | **只在 unit 边界**：驱动 Scheduler 做 select/review/rollover/eval，并 `iter(train_dataloader)` **重建迭代器**（见 §5） |
| ③ | `optimizer.step()` 之后 | 累计 `samples_seen`、写 unit 级统计与 JSONL 事件 |
| ④ | 已有 `open_loop_eval_steps` 分支 | **不新开评测入口**：Auto Learning 的评测复用 `EvaluatorAdapter → OpenLoopValidator.evaluate_ids`（内含 `safe_eval_context`） |

> 🔑 **`train_steps` / LR horizon 不变**：`dataloader_batch_size = gbs // dp`（`arguments.py:661`），
> 我们**不动 `global_batch_size`**，所以 LR 调度 horizon 与改造前**逐值相同**。

---

## 2. 每个 optimizer step 如何拿到「独立的一批」

现状：`dataloader_batch_size = gbs // dp`，`num_micro_batch = gbs // (micro × dp)`。
单卡 `gbs=10, micro=10, dp=1` ⇒ **`dataloader_batch_size=10`、`num_micro_batch=1`**
⇒ **一次 `next(data_iterator)` 恰好 = 一个 optimizer step 的 10 个样本**。

所以「每步独立采样」等价于「**sampler 每产出 10 个 index 就换一批**」。

```python
class AutoLearnSampler(Sampler[int]):
    def __iter__(self):
        while True:
            plan = self.planner.next_plan()     # 长度 == new_slots + replay_slots
            yield from plan                     # DataLoader 按 batch_size 取走
```

- DataLoader 在**主进程**按 `batch_size` 消费 sampler ⇒ `next_plan()` 恰好一次/step ✅
- `state_dict()/load_state_dict()` 让 `StatefulDataLoader` 能 resume ✅

**硬前置（fail-fast）**：
```
assert rmpad is False                    # 决策③；否则 dyn bsz 会打散 7+3
assert num_micro_batch == 1              # 否则一个 optimizer step 跨多批，7+3 语义不清
assert dataloader_batch_size == new_slots + replay_slots
```

---

## 3. Dataset 保持**一个稳定全量索引空间**

**问题**：Auto Learning 要按 `(task, episode, frame)` 取样本；如果每个 task 各建一个 Dataset，
索引空间会随 task 漂移，replay snapshot 里存的 index 全部作废。

**做法**：训练数据集**只建一次**，白名单 = **所有参与任务的 train 回合并集**
（`episode_ids_file` 由 `TaskCatalog` 生成）。于是：

```
local_idx  →  (task, episode, absolute frame)      # SampleResolver（B0 已实现）
```

- 索引空间**全程不变**（不随 active task / replay / round 变化）
- `SampleResolver` 就是 B0 那个，**不重新编码 sample id**
- PASS 时保存的 sampling snapshot 存的是 **local_idx**（稳定）+ 绝对帧号（可审计）

---

## 4. NEW / Replay 如何映射为 dataset indices

```text
TaskRegistry + FixedBaseline + SampleResolver
                    │
                    ▼
              TaskRegistry（B1-1）
                    │
                    ▼
                Scheduler（B1-1/4）
             ┌──────┴──────┐
             ▼             ▼
   HardnessScorer    EvaluatorAdapter        ← B0 已实现（真实模型 / OpenLoopValidator）
             └──────┬──────┘
                    ▼
           AutoLearnBatchPlanner              ← B1 新增：把 TrainRequest 变成 10 个 local_idx
                    │
                    ▼
           AutoLearnSampler                   ← B1 新增：每 step 一批
                    │
                    ▼
            真实训练循环 / optimizer.step()
                    │
                    ▼
          Checkpoint / extra_state
```

`AutoLearnBatchPlanner.next_plan()`：

| slot | 来源 | 算法 |
|---|---|---|
| NEW (`new_slots=7`) | active task 的 **train** 样本 | hardness 权重 `w_i = w_min + (w_max-w_min)·d_i^α` 的 alias 抽样（B0 已移植 alias method），**每步重抽** |
| Replay (`replay_slots=3`) | PASS pool | task-level 先选 task（**尽量 3 个不同 task**，PASS 不足则允许重复），再按该 task 的 PASS snapshot 抽样本；**没有 snapshot 就 uniform** |
| 无 PASS task | — | `10 NEW + 0 Replay` |

统计（每 step 记录，unit 级汇总）：`n_new` / `n_old` / `new_slot_counts_by_task` /
`old_slot_counts_by_task` / `unique_replay_tasks_per_batch` / hardness 采样熵。

**不变量**：`n_new + n_old == samples_seen`（每步与 unit 级都查）。

---

## 5. 如何避免重建 Dataset/DataLoader（以及**必须重建**的那一处）

| 对象 | 是否重建 | 原因 |
|---|---|---|
| **Dataset** | ❌ 全程一次 | 索引空间必须稳定（§3）；重建代价极高（LeRobot 元数据 + HF 表） |
| **DataLoader 对象** | ❌ 一次 | 与 Dataset 同生命周期 |
| **`iter(train_dataloader)` 迭代器** | ✅ **每个 learning unit 边界重建一次** | 🔴 见下 |

### 🔴 为什么 unit 边界必须重建迭代器

DataLoader 有 **prefetch**（`prefetch_factor × num_workers` 批）。
若在 unit 边界只是「换一个 request」，**已经预取好的批次仍属于旧 unit** ⇒
scheduler 认为在训新 task，实际前几个 step 用的是旧 task 的样本。

⇒ 在 unit 边界调用 `iter(train_dataloader)` 重建迭代器，**丢弃预取**（与 resume 路径
`iter(train_dataloader)  # clear resume state and prefetch data` 同一手法）。

**代价**：每个 unit（50 step）重建一次 worker（默认 8 个），实测 worker 启动 = 读 LeRobot
元数据 + 打开视频容器。**B1 要实测这个开销**；若显著，B2 再优化（例如 `persistent_workers`
+ 显式 `reset()`）。

---

## 6. Scheduler 在哪个层调 Evaluator

**只在训练循环的 unit 边界**（§1 的插入点 ②），**不在 sampler 内部**。
理由：评测要切 `model.eval()`、跑 `sample_actions`、临时改 `use_cache`/attention，
这些**必须**在「一步训练之外的干净时刻」做，且**单卡限定**。

```text
unit 结束
  → EvaluatorAdapter.evaluate_task(task, "train_monitor")   # 4 条 train-monitor
  → EvaluatorAdapter.evaluate_task(task, "val")             # 4 条 val
  → LP50 / overfit 诊断 / PASS-DEFER-EXHAUSTED 判定
  → 可能触发 review（2 条 → 可疑再 4 条）
  → 可能 select 下一个 task / round rollover
  → 发布新的 TrainRequest 给 planner
  → 重建 DataLoader 迭代器
```

- **不新写评测入口**：全部走 `EvaluatorAdapter` → `OpenLoopValidator.evaluate_ids`
  （内含 `safe_eval_context`：RNG / training flag / use_cache / attention / 视觉网格缓存）
- 每次评测后调 **`validator.clear_dataset_cache()`**（B0 新增）——
  `_ds_cache` 只增不减，Auto Learning 按 `(task, split, ids)` 反复评测必然 OOM
- 评测失败**向上抛**（`evaluate_ids` 的语义），由调度层决定是否终止

---

## 7. eval 前后如何复用现有 state restoration

**完全复用，不新增机制**：`EvaluatorAdapter` 只调 `validator.evaluate_ids()`，
而它内部就是

```python
with safe_eval_context(model=..., logger=..., ft_aug_registry=..., ...):
    with torch.inference_mode():
        return self._evaluate_ids(ids, tag)
```

`safe_eval_context` 负责 `use_cache` / eager attention / 视觉网格缓存 / 逐模块 training 标志 /
torch·numpy·python 三套 RNG 的 **snapshot → 强制 → finally 恢复 → 审计**。

⇒ **B1 不新增任何 eval 状态管理代码**。G8 只做「train → eval → train」的端到端验证。

---

## 8. `extra_state["auto_learning"]` 加哪些字段

`checkpointer` **不改**（B0 已确认 `extra_state` 原生支持）。训练循环里追加一个 key：

```python
state["extra_state"]["auto_learning"] = {
    "version": 1,
    "registry": registry.to_state(),          # 每 task: status/attempt/reopen/best/last/prev/...
    "round": state.round,
    "current_task": state.current_task,
    "attempt_count": ..., "attempt_step": ...,
    "transition_count": ..., "new_tasks_attempted": ..., "reopen_counts": ...,
    "sampler_state": sampler.state_dict(),    # 含 planner 的 step 计数
    "sampler_rng": rng.getstate(),            # 单一权威 RNG（Stage A 的教训）
    "pass_snapshots": {...},                  # PASS sampling snapshot（local_idx 列表）
    "hardness": {...},                        # probe identity + weights 或 (seed, meta) 可重建
    "baseline_fingerprint": ...,              # 防 resume 后换了分母
    "events_offset": ...,                     # JSONL 事件流已写到哪里
}
```

**不做**：新建并行恢复系统、改 DCP 格式、改 HF 存档。

---

## 9. resume 后如何恢复 sampler RNG 与 next decision

顺序（与 B0 的「配置指纹不一致就拒绝」一致）：

```text
load extra_state["auto_learning"]
  → 校验 baseline_fingerprint / 关键语义配置（不一致 ⇒ 报错，不静默继续）
  → 恢复 registry / round / current_task / counters
  → 恢复 scheduler RNG（**单一权威**，rebind 给 sampler/planner）
  → sampler.load_state_dict(...) ⇒ planner 的 step 计数与 unit 内位置
  → 重建 DataLoader 迭代器（丢弃预取）
  → 下一次 next_plan() 必须与「不中断」逐位一致
```

**验收**：`interrupted vs uninterrupted` 对拍 —— registry / current_task / attempt counters /
global_step / 下一次决策 / sampler 的下 N 个 index / RNG / PASS pool / replay 分布 **逐项一致**。

---

## 10. `auto_learning=false` 如何走原路径

| 项 | `enabled=false` | `enabled=true` |
|---|---|---|
| sampler | `StatefulDistributedSampler`（原样） | `AutoLearnSampler` |
| `rmpad` | 由原参数决定（**不改**） | **强制 false**（不改 yaml，只在 build 时覆盖） |
| `extra_state` | **不加** `auto_learning` key | 加 |
| 评测 | 原 `open_loop_eval_steps` 分支（原样） | 由 Scheduler 驱动（仍走同一个 `OpenLoopValidator`） |
| 日志 | 原样 | 追加 Auto Learning 面板 + JSONL 事件 |
| 新建对象 | **一个都不建**（不 import scheduler、不读 manifest） | — |

**验收（§7.9）**：固定 seed 下，`enabled=false` 的关键初始化结果（dataset 长度、
`train_steps`、LR horizon、首个 batch 的 index 序列）与改造前**一致**。

---

## 11. 48GB BF16 下的显存热点（预判）

| 热点 | 说明 | 缓解 |
|---|---|---|
| 训练步本体 | 与改造前**完全相同**（micro/gbs 不变） | — |
| **Open-loop 评测** | `sample_actions` + KV cache（`use_cache=True`）逐样本跑；峰值通常**高于**训练步 | 评测条数固定（4+4）；必要时缩短 val 集 |
| 视觉网格缓存被清 | 评测按单样本网格重算，不额外占 | — |
| 评测与训练**不重叠** | 单卡、同一进程、`inference_mode` | 不叠加 |
| Hardness probe | `no_grad` + 固定 noise/time，一次前向（micro=probe 条数） | probe 只抽 1/3 轨迹 |

**B1 只记录 `peak_vram`，不做吞吐优化。**

---

## 12. 哪些测试无卡跑、哪些必须 GPU

| 类别 | 内容 | 环境 |
|---|---|---|
| **无卡（本阶段先全绿）** | §7.1–§7.11：B0 回归 / catalog·split / resolver / baseline / state machine / sampler（7+3·diversity·fairness·hardness 频率）/ logging invariant / resume 确定性 / disabled 回归 / E2E dry-run / Monte Carlo 50×100 | 无卡模式（**2 GiB 内存**，见下） |
| **必须 GPU（等用户开 48GB）** | G0–G10：环境 smoke / 既有 OL selfcheck·parity / BF16 hardness / real baseline cache / 2→4 eval / 1·3-step training / 7+3 provenance / mini scheduler E2E / eval-after-train 恢复 / checkpoint-resume / 50-step unit | 单卡 48GB BF16 |

> ⚠️ **无卡模式内存约束（实测）**：cgroup `memory.max = 2048 MB`；
> 仅 `import` 到 `LingbotVLAV2Config` 就 ~800 MB，+processor ~940 MB。
> ⇒ 无卡测试**一律用 fake/mock 数据集与假模型**，不建真实 VLA 数据集。

---

## 13. 拟修改文件与最小改动

### 新增（`lingbotvla/auto_learning/`）

| 文件 | 内容 | 来源 |
|---|---|---|
| `config.py` `types.py` `ports.py`(合并) | 配置 / DTO / 5 Protocol + `ReplayPlan` | 移植自 Stage A（**复制，不 import Demo**） |
| `decision/{metrics,state_machine,review}.py` | LP50 / 状态机 / review | 移植 |
| `state/{registry,persistence}.py` | TaskRegistry / 存档 | 移植 |
| `sampling/{rng,replay,sampler,hardness_scanner}.py` | alias 抽样 / replay 槽位 / hardness 扫描 | 移植 |
| `orchestration/scheduler.py` | Scheduler（原子动作） | 移植 |
| `obs/{logger,report}.py` | 事件流 / 报表 | 移植 |
| **`real/sampler.py`** | `AutoLearnSampler` + `AutoLearnBatchPlanner` | **新写** |
| **`real/loop_hook.py`** | 训练循环接线：unit 边界驱动 + iterator 重建 + extra_state | **新写** |
| **`real/build.py`** | 从真实 config 构造整套 Backend | **新写** |

### 修改（真实仓库，**逐处最小**）

| 文件 | 改动 | 风险 |
|---|---|---|
| `lingbotvla/data/data_loader.py` | `build_dataloader(..., sampler=None)` 透传（默认 None = 原行为） | 极低 |
| `tasks/vla/train_lingbotvla.py` | ① enabled 时覆盖 `rmpad=false` + 传 sampler；② unit 边界 hook；③ `extra_state` 加一个 key | 中（有 disabled 回归守） |
| `train_lingbotvla.py::MyTrainingArguments` | 加 `auto_learning_*` 开关 | 低 |
| ~~`checkpointer.py`~~ / ~~collator~~ / ~~model forward~~ / ~~evaluator~~ | **不改** | — |

---

## 14. 风险点

| # | 风险 | 缓解 |
|---|---|---|
| 1 | 🔴 **prefetch 跨 unit 边界**（旧 unit 的样本混进新 unit） | unit 边界**重建迭代器**；G6 审计 sample provenance |
| 2 | 🔴 **只支持单卡**（`OpenLoopValidator` 多卡构造即抛） | B1 就按单卡设计；多卡留 B2+ |
| 3 | 🔴 `_ds_cache` 只增不减 ⇒ 反复评测 OOM | 每次评测后 `clear_dataset_cache()` |
| 4 | ⚠️ 重建迭代器的开销（每 unit 一次） | B1 实测；必要时 B2 用 `persistent_workers` + `reset()` |
| 5 | ⚠️ `train_steps` 由 dataset 长度推导；换 sampler 后语义需重新确认 | **不动 gbs**，LR horizon 不变；`train_steps` 用原推导 |
| 6 | ⚠️ `drop_last=True` + 自定义 sampler | sampler 无限产出，`drop_last` 不触发；需断言 |
| 7 | ⚠️ 视频解码很慢（实测 6 回合 ≈ 2–3 分钟） | 评测条数固定 4+4；probe 只抽 1/3 |
| 8 | ⚠️ 48GB 显存是否够（训练 micro=10 峰值 78.7G 是 **F32**；bf16 micro=1 峰值 43.3G） | **B1 smoke 用 bf16 + 小 micro**，如实记录；不改算法语义 |
| 9 | ⚠️ hardness 的 dtype（BF16 路径） | B0 已修：`forward_dtype()`；G2 验证 |
| 10 | ⚠️ resume 后决策漂移 | 单一权威 RNG + `sampler_state`；§7.8 逐项对拍 |

---

## 15. 提交切分（按测试计划 §13）

```text
B1-1  registry / scheduler skeleton（移植 + 接真实 Backend）
B1-2  dynamic sampler + replay（AutoLearnSampler / Planner）
B1-3  hardness integration（接真实 HardnessScorer）
B1-4  evaluator decision loop（unit 边界驱动）
B1-5  checkpoint / resume（extra_state）
B1-6  logging + invariants
B1-7  no-GPU test suite（§7.1–§7.11）
B1-8  48GB BF16 smoke fixes（等用户开卡）
```

---

## 16. 验收 Gate（对应测试计划 §15）

| Gate | 内容 | 本阶段能否完成 |
|---|---|---|
| **A** B0 fixes closed | 4 条 P0/P1 + regression 全绿 | ✅ 已完成（41 contract tests + 无卡实跑 9/9、5/5） |
| **B** Design approved | 本文件 | ⏳ 等 review |
| **C** No-GPU tests green | §7.1–§7.11 全绿 | 🎯 **本阶段目标** |
| **D** 48GB interface smoke | G0–G6 | ⏸️ 等用户开卡 |
| **E** 48GB full B1 smoke | G7–G10 | ⏸️ 等用户开卡 |

---

## 17. 明确不做（本阶段边界）

- ❌ 不做 96GB / 4-task 正式对比实验（B2）
- ❌ 不新增未经讨论的 curriculum / 算法机制
- ❌ 不重写 preprocess / evaluator / checkpoint / 训练 forward-backward
- ❌ 不做多卡 Scheduler / all-rank evaluation
- ❌ 不做 dynamic batch packing（v0 要求 `rmpad=false`）
- ❌ 不为「更优雅」而重构既有已验证代码
