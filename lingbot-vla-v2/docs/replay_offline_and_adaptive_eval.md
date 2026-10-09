# Replay 历史证据离线复核 + Adaptive Eval Batch 安全门控

适用基线（来自用户报告）：`e362f4b0e6ce7a0fc8f522888ce72462b9a582ee`。
**本地未取得该 HEAD 的完整 Git 对象；补丁只新增文件，训练机需要复验**。

## A. 无卡离线 Replay 复核

```bash
PY=/data/miniconda3/envs/lingbotvla/bin/python
AUDIT=docs/audit/gpu96_ratio_4task_2026-10-09
$PY tools/replay_offline_verify.py --audit-dir "$AUDIT" \
  --gbs 24 --new 17 --replay 7 --step-offset 500 \
  --output "$AUDIT/offline_verification.json"
```

若 `auto_learning_events.jsonl` **没有逐 Unit 的真实指标**，可选择追加一种可信原始指标来源：

```bash
# 优先使用真实训练输出里的 TensorBoard event 文件：
$PY tools/replay_offline_verify.py --audit-dir "$AUDIT" \
  --tb-logdir /data/outputs/gpu96_acceptance/ratio_real_micro24_gas1_target4_steps15 \
  --output "$AUDIT/offline_verification.json"

# 或使用从真实 TensorBoard 事件直接导出的 metric JSONL，必须保留来源文件 SHA256：
$PY tools/replay_offline_verify.py --audit-dir "$AUDIT" \
  --metric-jsonl /path/to/observed_tb_scalars.jsonl \
  --output "$AUDIT/offline_verification.json"
```

**注意：两种指标来源只能选一种，不要对同一 step 重复计数。** 本工具记录输入文件 SHA256，禁止覆盖原始 `result.json` 和既有离线结果。首次尝试应先不加 `--output`，检查缺失项，确定文件后再写入一次。

严格 PASS 条件：每个 Unit 有 `steps`、`ReplayPlan.tasks`、唯一且匹配的真实 `sampling/replay_samples_per_unit`；单位 step 可在 Train Unit 开始或结束时，支持 TB step500 偏移，但必须全程唯一映射。还需要独立累积指标 `sampling/total_samples=360`、`system/global_samples_seen=360`、`system/units_run=5`、`curriculum/units_completed=5`。若日志缺一项就 BLOCKED，不得凭 ratio plan 补数。**原始工具的 BLOCKED 不追改**。此程序只能对已有真实证据重作判读，不代表新 GPU 实测。

### 版本来源核对

实际运行 `result.json.git_head=648558f`、报告后续 HEAD `e362f4b`。WorkBuddy 必须用 `git merge-base --is-ancestor 648558f e362f4b` 和 `git show` 校对提交先后及运行时代码，不得直接认定二者相同。

## B. Adaptive Eval Batch：已实现策略与门控，尚未修改正式推理路径

新增 `lingbotvla/auto_learning/eval_batch_policy.py`：

- 默认 `mode=serial`，旧评测路径 batch=1 完全不变。
- `mode=auto` 只在存在真实 `batch_infer(items)` 接口、**每轨迹固定 RNG**、峰值显存传感器、数值对拍成功时，才从 1/2/4/8 中选吞吐最好的实测 Batch。
- 显存预留默认 10GiB；推理**期间**的峰值 free VRAM 和结束后 free VRAM 都得满足门槛。单凭执行前后空闲显存不能证明安全。
- 训练后优化器状态变化、任务输入形状变化必须重新校准/降级。`allowed_runtime_batch()` 提供保守门槛；多卡 FSDP2 默认 BLOCKED / batch=1，需要另加 collective-safe 决策。
- 发生 GPU OOM **不吞异常、不直接在同一个进程内重试**，必须安全退出并由人工决定是否另开进程。
- `outputs_close` 对逐轨迹完整预测张量检查，额外必须对 NMSE/GMean-MSE、PASS/REOPEN、episode IDs 等做线上语义级校验。

### 批量推理能力检查（不使用 GPU）

```bash
$PY tools/adaptive_eval_batch_acceptance.py --mode auto --candidates 1 2 4 8
# PLAN_ONLY: 不启动推理
$PY tools/adaptive_eval_batch_acceptance.py --mode auto --execute
# BLOCKED: 当前补丁没有擅自假设生产 _infer_one 支持多轨迹 batch
```

### 接入真实模型前需完成的工程工作

WorkBuddy 应基于**实际训练机 HEAD** 检查 `scripts/open_loop_eval.py` 的 `_infer_one`、真实 Scheduler eval port、模型 `infer_action`/`generate` 接口及 `use_cache`/KV 状态，并另出**生产批量推理接线补丁**；不要仅修改 `(1, n_action_steps, max_action_dim)` 的 1。合并项目的真实多模态输入/attention mask、不同 chunk 数的活跃样本掩码、逐轨迹独立 noise RNG（seed 不依赖 batch 排序）、输出拆分和 ID 映射，才能真正提速。

严格 GPU 验收：先使用完全相同轨迹 ID/种子、BF16/FP32、bootstrap 与 optimizer 已初始化之后的两个阶段，对 batch1/2/4/8 做真实 `samples/s`、峰值 VRAM、每轨迹 action 输出、NMSE/GMean、PASS/Review 结果对拍。任何指标翻转、无法证明数值一致、FSDP2 collective 不安全时，正式 eval 仍 serial。不要改正式 NMSE 配置、阈值、DCP/HF、训练 GBS。

本补丁**并未实现 6B 模型实际跨轨迹 batch 推理**，不得宣传批量推理加速已实测，也不自动启动 GPU、删除旧文件或关机。
