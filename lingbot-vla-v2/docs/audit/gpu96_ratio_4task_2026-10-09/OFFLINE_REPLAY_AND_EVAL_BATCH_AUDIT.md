# 离线 Replay 复核 + 自适应 Eval Batch 审计（无卡，2026-10-09）

基线 HEAD：合入前 `e362f4b`；本补丁为**纯新增**（6 文件），未改任何生产逻辑。

## 一、离线 Replay 严格复核结果：**BLOCKED**（fail-closed，正确）

命令（无卡，CPU-only；读**原始 JSONL + 原始未修改 TensorBoard 事件**）：

```
tools/replay_offline_verify.py --audit-dir <run_dir> --tb-logdir <run_dir>/runs \
    --gbs 24 --new 17 --replay 7 --step-offset 500 --output <run_dir>/replay_offline_verify.json
```

结果：`status=BLOCKED`、`diagnosis=evidence_incomplete_or_inconsistent`、`source_count=2`
（JSONL sha `9ef3a5b5…`、TB 事件 sha `4da5124b…`）、**`historical_result_unchanged: true`**（历史 result.json 的 `status=BLOCKED` 与摘要均未被改动 ✓）。

### 具体缺失字段/原因（提供方要求的精确说明）

| # | 报错 | **精确根因** |
|---|---|---|
| 1 | `unit@0/3/6/9/12: no/duplicate observed ReplayPlan tasks` | 训练器把 ReplayPlan 放在 `TrainRequest` 的**字符串** repr 里，且该字符串长度 **161,609 字符**；验证器有 `len(req) > 100_000 ⇒ return []` 的保护，直接放弃解析 ✗。ReplayPlan 实际存在：`ReplayPlan(tasks=['click_bell','click_alarmclock'], probs={...}, sample_ids={...})`，同一字符串还含 `batch_size=24, new_slots=17, replay_slots=7` ✓ |
| 2 | `unit-to-metric step mapping ambiguous/missing: unit starts [0,3,6,9,12], metric steps [3,6,9,12,15,503,506,509,512,515], candidates []` | `sampling/replay_samples_per_unit` 在**每个 Unit 结束时**（step 3/6/9/12/15）各写一条，而 Unit 的 `start` 记为 0/3/6/9/12 ⇒ 验证器要求唯一映射（同一 offset 约定下 start↔metric 逐步对应），当前约定下无候选 ✗ |

**⇒ 结论：不能判 PASS**（与历史工具判定一致），但**缺的是"离线复核器的解析/映射口径"，不是"训练时没消费 Replay"**。
事件与指标显示的消费事实（5 Unit、ReplayPlan 含 2 个真实 PASS 任务、`replay_samples_per_unit=21`、`total_samples=360=15×24`）不变；
本轮**未**为取得 PASS 而重跑 GPU ✓（遵守提供方"不得为验收判 PASS 重跑 GPU"）。

### 若要让它 PASS（需批准；属复核器口径修正，不改生产代码）

1. 解析 `request` 时不要用 100k 硬上限（改为"先定位 `ReplayPlan(tasks=[…])` 再解析该子串"，或对流式前缀做有界搜索）；
2. Unit↔metric 映射改为支持 **unit-end** 约定（start+steps == metric step），并在歧义时仍 fail-closed；
3. 反向补一条**负例测试**（伪造/缺失指标必须仍 BLOCKED）。

## 二、自适应 Eval Batch 能力门控（不伪造 batched backend）

| 调用 | 输出 | 退出码 |
|---|---|---|
| `--mode auto --candidates 1 2 4 8 --reserve-gib 10` | `status=PLAN_ONLY`，`note="This tool never claims GPU multi-trajectory inference exists."` | 0 |
| `--mode auto --execute`（无 backend 证明） | `status=BLOCKED`，`reason="missing real inference batch adapter + numerical parity proof"` | **2** ✓ fail-closed |
| `--mode serial` | `status=PLAN_ONLY`（现行生产语义） | 0 |

## 三、生产适配接口清单（把 `eval_batch_policy` 接进真实评测要动哪里）

> 结论：**当前生产评测是单轨迹**（batch 维硬编码 1），接入需要"新增 batched 推理路径 + 策略咨询点 + 数值对拍证明"，**不是改一个常数**。

| 位置（file:line） | 现状 | 接入所需改动 |
|---|---|---|
| `lingbotvla/utils/open_loop_validation.py:918` `_infer_one()` | 单轨迹；`noise` 形状 `(1, n_action_steps, max_action_dim)`；L956-962 对 images/masks/tokens/state 逐个 `unsqueeze(0)` ⇒ batch≡1 | **新增** `_infer_batch(items)`：按 trajectory padding + **逐轨迹固定种子** + 逐轨迹 chunk 归属；不得改变 `_infer_one` 语义 |
| 同文件 `:1108` `pred = self._infer_one(item, ft)`（驱动循环） | 逐 item 串行调用；`:1153` `chunks.append((ep_key, gt, pr))` | 改为"按 split 分组 → 咨询策略得 batch → 组内批推 → 仍按 `(ep_key, gt, pr)` 逐条入 chunks"（保持 `aggregate_chunks` 不变） |
| 同文件 `:1301 collect_gt_chunks()` / `:1353 evaluate_ids()` | 评测入口 | 增加可选 `batch_size` 参数（默认 1 ⇒ 行为不变） |
| 同文件 `:297 aggregate_chunks()` | 指标聚合（已 fp64） | **无需改动**（只要 chunk 三元组与串行一致） |
| `lingbotvla/auto_learning/ports.py:121 EvalResult` / `:220 evaluate(task, split, episode_ids)` | 端口签名 | 建议**保持签名不变**，把 batch 决策放在实现内部（避免改 Scheduler） |
| `lingbotvla/auto_learning/orchestration/scheduler.py:312/364/456/457/605/606/850/877`（8 处 `self.evaluator.evaluate(...)`） | scout / confirm / train_monitor / active_val | **无需改动**（若坚持在端口传 batch，则这 8 处都要动 ⇒ 不推荐） |
| `lingbotvla/auto_learning/eval_batch_policy.py:14/59/74/148`（新增） | `EvalBatchSettings` / `outputs_close` / `profile_batch_candidates` / `allowed_runtime_batch` | 作为评测实现内部的策略咨询 API；`reserve_gib` 使用**实测**空闲显存 |

**未来接入所需的 GPU 验收步骤（另提小补丁时一并给出）**：
① 固定逐轨迹种子与轨迹集合，串行 vs batched 的 **NMSE / GMean-MSE 逐轨迹一致**（`outputs_close(atol,rtol)` 必须通过，否则 fail-closed 退回串行）；
② 记录 batch=1/2/4/8 的峰值显存与单 chunk 延迟，验证 `reserve_gib` 与 `allowed_runtime_batch` 的预测；
③ PASS/DEFER/REOPEN 判定在两种模式下**逐任务一致**（对同一 ckpt、同一轨迹集合）；
④ 任何数值不一致 ⇒ 该 batch 尺寸禁用并回退串行，不得静默降级。

## 四、本轮未做（遵守约束）

未改正式 YAML / NMSE 判据 / DCP-HF / NEW:Replay / Scheduler / Hardness ✓；未启动 GPU（无卡模式）✓；
未自动关机 ✓；未把 batch 维改成 4、未接入未验证的 auto 模式 ✓；未为取得 PASS 重跑 GPU ✓。
