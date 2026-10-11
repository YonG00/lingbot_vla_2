# B. Replay 对账 —— 实测结论（数据来源：TensorBoard 标量 + 代码）

## 关键答案：**不是 n_old=0；Bootstrap PASS 确实进了 Replay 池并被采样**

| 证据（TB 标量，成功 run） | 值 | 含义 |
|---|---|---|
| `memory/pass_pool_size` | @505 = **2.0**, @510 = **2.0** | PASS 池恒有 2 个任务（click_bell + click_alarmclock） |
| `system/replay_slots` | @505 = **5.0**, @510 = **5.0** | 每个 5 步 unit 分配了 **5 个 replay slot**（1 slot/step） |
| `system/global_samples_seen` | @505 = **20.0**, @510 = **40.0** | 每 unit 消费 20 样本 = 5 步 × batch 4 ⇒ 与 AL batch 契约一致 |
| `system/replay_unique_tasks` | @505/510 = 1.0 | **每 batch** 只有 1 个 replay slot ⇒ 该指标恒为 1（非异常） |

⇒ 每 unit：`n_new = 3×5 = 15`、`n_old = 1×5 = `**5**；两 unit 合计 **10 个 OLD 样本**。
⇒ `n_new + n_old == samples_seen`（现实现在 `real/sampler.py:85` 的不变量）成立。

## 为什么 Bootstrap PASS 能进池（代码链）
1. `Scheduler.replay_plan()`：PASS 池 = `registry.by_status(TaskStatus.PASS)`，**含 bootstrap 的免费 PASS**；
2. `pass_sampling_snapshot` 只在 `decision/state_machine.py:172` 由 `record.sample_probs` 生成，而 `sample_probs` 仅由 `_select` 的 Hardness 扫描赋值
   ⇒ **bootstrap PASS 的任务 snapshot = None**（它从没被扫描过）；
3. `sampling/replay.py:58-72`：`raw` 为空 ⇒ `table=None` ⇒ **均匀采样**（注释原文 "None ⇒ 均匀"），
   `sample_replay_refs` 对 `table is None` 走 `rng.randrange(len(sample_ids))`。

⇒ **snapshot 缺失只是把"样本级分布"退化为均匀，不会让任务被跳过** ⇒ 与实测 n_old=5 一致。

## 仍存在的两个观察（非 Bug，建议项）
1. **每任务的 OLD 计数没有上报**：`TrainResult.old_slot_counts`（按任务）与 `new_slot_counts`（按样本）只在进程内，
   未写入 TB/事件 ⇒ 目前**无法从产物里回答"这 5 个 OLD 样本分别来自哪个 PASS 任务"**。建议加
   `system/replay_old_slots/<task>` 与 `system/replay_new_unique_samples` 两个标量（仅日志，不改调度）。
2. `system/replay_unique_tasks` 的语义是**每 batch** 唯一数；`replay_slots=1` 时恒为 1，
   容易被误读成"跨 unit 只复习了 1 个任务"。建议改名为 `system/replay_unique_tasks_per_batch` 或另加 unit 级统计。

> 本结论**只基于实测产物与源码**；未修改任何 Scheduler 代码（遵守"发现 Bug 先报告"）。
