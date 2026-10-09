# 阶段 3：四任务真实 Replay 验收报告（2026-10-09 08:26–08:36）

## 0. 结论速览

| 项 | 结果 |
|---|---|
| **工具判定（`result.json`）** | **BLOCKED**（`diagnosis = train_unit_without_replay_pool`）⇒ 按用户规则**不得判 PASS** |
| **独立证据（训练器实际指标）** | **真实 Replay 确已消费**：5 个 Train Unit、ReplayPlan 来自 2 个真实 Bootstrap PASS 任务、每 Unit 21 个 Replay 样本、总计 360 = 15×24 完全自洽、Replay 占比 29.2% ≈ 30% |
| **不一致的根因** | **工具与训练器的字段口径不匹配**（工具读 `batches_built`/`samples_seen`/`old_tasks` 与"逐步" `sampling/replay_samples_per_unit`；训练器发 `steps`/`request`（内含 ReplayPlan）与"按 Unit 聚合"的该指标）⇒ 工具算出 `samples_seen=0`、`replay=None`、`actual_new=-21` ⇒ 结构性 BLOCKED，**与训练是否消费 Replay 无关** |
| **是否伪造 PASS** | 否。工具 fail-closed 设计正确生效 ✓ |
| **误写权重** | 无 ✓（`checkpoints/global_step_*` 0 个、`hf_milestones/*` 0 个、`*.safetensors` 0 个、`*.distcp` 0 个；目录仅 16.3 MiB 日志） |

## 1. 执行环境与命令

- Git HEAD：`648558f379c7f6845b69d080371a2796af43bd58`（本机 = origin = 机器）
- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition，97,887 MiB，初始占用 0
- 峰值显存 **87,107 MiB**｜耗时 **617.03 s**（约 10.3 分钟）｜`returncode=0, reason=exit`
- 命令（用户批准，仅一次）：
  `ratio-gpu --micro 24 --gas 1 --al-config configs/auto_learning/gpu96_ratio_4task_acceptance.yaml --target-total-passed-tasks 4 --max-named-tasks 4 --steps 15 --execute`
- 输出目录：`/data/outputs/gpu96_acceptance/ratio_real_micro24_gas1_target4_steps15`

## 2. Bootstrap PASS 名单（4 任务）

| 任务 | scout NMSE（2 轨迹） | confirm NMSE（4 轨迹） | 结果 | vs 阈值 1.66 |
|---|---|---|---|---|
| click_bell | 0.03403 | **0.03402** | **pass**（source=confirm） | 远低 |
| click_alarmclock | 0.06871 | **0.10189** | **pass**（source=confirm） | 远低 |
| turn_switch | **1.76268** | — | **candidate（未通过）** | **>1.66** |
| put_object_cabinet | **2.16435** | — | **candidate（未通过）** | **>1.66** |

⇒ 2 PASS 构成真实 Replay 池；2 个未通过任务迫使进入 Train Unit（设计目标达成 ✓）

## 3. Train Unit 明细（5 个 Unit，全部被真实消费）

| step | task | steps | deferred | ReplayPlan.tasks |
|---|---|---|---|---|
| 0 | turn_switch | 3 | true | `['click_bell','click_alarmclock']` |
| 3 | turn_switch | 3 | true | 同上 |
| 6 | turn_switch | 3 | true | 同上 |
| 9 | turn_switch | 3 | true | 同上 |
| 12 | put_object_cabinet | 3 | true | 同上 |

（`old_tasks` 字段训练器未直接发出；ReplayPlan 以字符串形式在 `request` 内，任务列表即真实 PASS 池 ✓）

## 4. 实际消费计数（训练器 metric，独立于工具对账）

```
replay/available_tasks                = 2      ← 真实 PASS 池任务数
system/replay_unique_tasks            = 2
sampling/replay_distinct_tasks_per_unit = 2
sampling/replay_samples_per_unit      = 21     ← 7/step × 3 step = 21 ✓
system/replay_slots                   = 21
sampling/total_samples                = 360    ← 15 步 × GBS24 = 360 ✓
system/global_samples_seen            = 360    ← 与上式一致 ✓
system/units_run = 5   curriculum/units_completed = 5
```
- **三方对账**：步数 15 × GBS 24 = 360 = `total_samples` = `global_samples_seen` ✓
- **比例对账**：每步 17 NEW + 7 Replay（`expected_per_step`）⇒ NEW 占比 17/24 = **70.8%**、Replay 7/24 = **29.2%** ≈ 目标 70:30 ✓
- **总量对账**：5 Unit × 21 = **105 个真实 Replay 样本**（105/360 = 29.2% ✓）

## 5. 工具 BLOCKED 的精确根因（字段口径契约差）

工具 `ratio_unit_evidence()` 期望 | 训练器实际发出 | 后果
---|---|---
事件字段 `batches_built` | `steps` | `optimizer_steps=0`
事件字段 `samples_seen` | 无（只有聚合 metric） | `samples_seen=0` ⇒ `actual_new = 0-21 = -21`
事件字段 `old_tasks` | ReplayPlan 在 `request` 字符串内 | `replay_tasks=[]` ⇒ `replay_ratio_valid=False`
metric `sampling/replay_samples_per_unit` **逐步**（每 step 一条） | **按 Unit 聚合一条** | 只有 step 0 能取到 21，其余 Unit 取不到 ⇒ `replay=None`

⇒ `full_batch_valid=False`、`replay_ratio_valid=False` ⇒ `diagnosis=train_unit_without_replay_pool` ⇒ **BLOCKED**
（**修正上述任一契约即可得到真实 PASS，但这属于工具改动，需用户批准后另做最小补丁 + CPU 测试**）

## 6. 本轮未做（严格遵守约束）

未追加训练步数（用满 `steps=15` 预算即自然结束）✓；未改 PASS 阈值（仍 1.66）✓；未重跑 ✓；未启动正式训练 ✓；
未修改任何正式训练配置 ✓；未自行开关机 ✓（GPU 已释放，0 MiB）。

## 7. 产物清单（本目录）

`result.json`（工具判定）· `auto_learning_events.jsonl`（269+ 事件，含 Bootstrap 与 5 个 train_unit）·
`isolated_ratio_smoke.yaml`（运行中生效的隔离配置：四任务 + nmse + 1.66 + target 4 + max_global_steps 515）·
`train.log`（完整训练日志，含 4 次评测与 15 步训练）

## 8. 建议的下一步（待用户决定）

1. **修正工具对账口径**（读 `steps` + 解析 `request` 内 ReplayPlan + 按 Unit 聚合指标）⇒ 最小补丁 + CPU 测试 ⇒ **再跑一次**（约 10 分钟）即可拿到工具认可的 **PASS**；
2. 或**接受独立证据**（真实消费 + 计数自洽）并把该契约差记录为已知问题，先继续 GMean / HF 专项；
3. 不建议：为凑 PASS 而放宽判定逻辑或伪造字段。
