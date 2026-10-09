# 50-task GMean200 扫描加速（实验、默认关闭）

基于用户上传的 Git HEAD `2650590` 并先叠加前一轮 GMean 200× 选课补丁；**原 NMSE 正式配置不修改**。

> **开 `AL_EVAL_BATCH_MODE=auto` 前必读**：本模式要求通过**随机性验收**（模型自身非确定，
> strict parity 恒不可达）。步骤与门控见 `docs/eval_batch_auto_gate_guide.md`；
> 缺 `AL_EVAL_BATCH_STOCHASTIC_APPROVED=<gate.json>` 时门不开、自动退化为串行。

## 实现边界

1. **多轨迹批量开环推理（已接生产代码但未 GPU 验证）**：`OpenLoopValidator._infer_batch` 把同形状 item 的图像、mask、文本、state 按 batch 维组装，**为每条 chunk 逐次产生单样本噪声**，一次调用真实 `model.sample_actions`，拆分后用原 `ft.unapply` 回到物理 action 空间。`_infer_one` 语义不变，`aggregate_chunks` 不变。第一次遇到每个 batch/输入形状都要和串行输出比较，并测量 peak VRAM 与耗时。`probe` 只产生原串行指标；`auto` 仅在 `AL_EVAL_BATCH_APPROVED=1` 且该批次形状在本次 eval 已通过实时 parity/显存/吞吐检查后，才使用 batched 预测。多卡 FSDP 不支持。
2. **Hardness Batch 自适应**：默认仍固定 batch8；启用后，先用已验证的 batch8，观测每个实际前向的 PyTorch CUDA 峰值 reserved 与可用显存，在至少 10GiB 余量下保守增长/缩小（最多16）。最重要的是**按 sample_id 固定随机 noise**，否则变更 Batch 会破坏样本难度排序口径。旧固定 batch 的噪声实现保留。多卡 FSDP 不启用。
3. **Bootstrap Scout 跨运行缓存**：只接受完整真实 checkpoint 权重分片 SHA256 + eval/model 源码 + 阈值文件 + baseline + manifest + norm 指纹完全相同的缓存；需明确启用、初始 Step500、无 Resume。缓存仅用于初始 Bootstrap Scout；Confirm / Rescan / Review / 训练监测依然真实评测。**现有旧 NMSE JSONL 不能命中**。如无法证明当前内存中的初始模型实际来自所声明的 checkpoint，应保持缓存关闭。

## 无卡入口

```bash
cd /data/code/lingbot-vla-v2
PY=/data/miniconda3/envs/lingbotvla/bin/python
$PY tools/scan_accel_preflight.py  # 默认 PLAN ONLY，不读 12GiB 权重
$PY tools/scan_accel_preflight.py --verify \
  --checkpoint /data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt \
  --manifest /data/train/task_splits_50/manifest.json \
  --baseline /data/train/task_splits_50/task_baseline.json \
  --norm assets/norm_stats/robotwin_competition_clean.json \
  --thresholds /data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json \
  --cache-root /data/outputs/al_gmean200_scout_cache --dtype bfloat16
```

这里 `--verify` 需要读取 checkpoint 全部分片以计算 SHA256，**可能耗费 CPU/磁盘时间，但不会使用 GPU 或写缓存**。本 CLI 给出指纹；实际训练启用需严格核对加载来源与以下环境变量。

## 测试/启用开关（均不会自己开卡）

默认状态无需设置任何变量：开环 Batch1、Hardness Batch8、缓存 OFF。

```bash
# 第一次真实 GPU 验收：只做影子 batch 推理，官方 NMSE/GMean 数值仍来自串行
export AL_EVAL_BATCH_MODE=probe
export AL_EVAL_BATCH_MAX=4
export AL_EVAL_BATCH_RESERVE_GIB=10

# GPU 对拍、物理 action、GMean/NMSE 与训练态 VRAM 均验证通过后再显式批准：
# export AL_EVAL_BATCH_MODE=auto
# export AL_EVAL_BATCH_APPROVED=1

# Hardness 自适应需另行经过真模型 batch-invariant RNG 数值验收：
# export AL_HARDNESS_BATCH_MODE=auto
# export AL_HARDNESS_BATCH_APPROVED=1
# export AL_HARDNESS_BATCH_MAX=16
# export AL_HARDNESS_RESERVE_GIB=10

# Scout 缓存属于可选功能；需严格验证当前模型权重与声明的 shard 一致：
# export AL_SCOUT_CACHE_MODE=bootstrap
# export AL_SCOUT_CACHE_ROOT=/data/outputs/al_gmean200_scout_cache
# export AL_SCOUT_CACHE_CHECKPOINT=/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt
# export AL_SCOUT_CACHE_MANIFEST=/data/train/task_splits_50/manifest.json
# export AL_SCOUT_CACHE_BASELINE=/data/train/task_splits_50/task_baseline.json
# export AL_SCOUT_CACHE_NORM=assets/norm_stats/robotwin_competition_clean.json
# export AL_SCOUT_CACHE_DTYPE=bfloat16
```

注意：`AL_SCOUT_CACHE_MODE` 只有 `bootstrap` 才启用，**没有从旧日志自动迁移**；若重建数据集内容但 manifest 文件未更新，需要重新检查数据来源。是否启用缓存需先确认训练入口确实从指定 checkpoint 加载了模型，而不是误用训练后权重；仅靠文件名相同不能证明相同。

## 第一次 GPU 验收建议

先在独立的 2–3 任务短 Smoke 上运行 `AL_EVAL_BATCH_MODE=probe`，同时记录 B1/B2/B4 的 `parity`、`peak_free_gib`、`serial_seconds`、`batch_seconds`。保持原 NMSE/GMean PASS 和 Step500 checkpoint，不能仅因为 logs 表示 batch2 已执行就称已经提速。实际 Scout2 chunk 可能少到连 B4 都组不出来，必要时用真实 10-trajectory eval 做容量验证。

Hardness 独立测固定8与自适应8/12/16的**逐样本 ID 和 Flow Loss** 对拍（不建议让开发者手工测几十组并行数），确保排序不发生超过容差的变化；当前使用 per-ID seed 的动态模式，与旧固定8的 batch-seed 流有意不同，**不能把两种模式的 raw loss 逐位相等当作前提**。正式启用前确认批量划分改变时 per-ID 值稳定，且训练前后都有安全余量。

如果 Scout 候选的 GMean 200× PASS 已在 Bootstrap 达到4个，Scheduler 仍可能零步结束；本补丁**不擅自修改 stop target**。

## 尚未证明

- **未真实在 6B GPU 上跑过 `_infer_batch`**，所以未声称它已提速、稳定无OOM或能保证全部数值一致。
- 同形状批处理已实现；变长/不同观测形状只会退化串行，不猜 padding。
- CPU 数据多进程预取未启用：当前数据集确定性上下文会暂时修改共享 transform 与 RNG，直接加线程容易造成竞态，需要单独设计后再开发。
- 现有历史 NMSE Scout 记录仍不可作为 GMean200 缓存，第一轮新 GMean Scout 必须重扫。

**不修改正式 NMSE YAML、Reference 阈值、Scheduler 的 PASS 口径、Replay70:30、DCP/HF 方案。不开/关 GPU，不自动开始正式训练。**
