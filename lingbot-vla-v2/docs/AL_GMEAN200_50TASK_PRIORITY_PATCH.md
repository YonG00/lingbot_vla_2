# 50-task GMean 200× 实验：倍率选课与扫描审计（源码 HEAD 2650590）

## 实际完成的内容

1. `Scheduler._candidate_priority()` 在 `pass_metric=gmean_mse` 时按**候选 Scout GMean-MSE / 生效任务 PASS 阈值**升序；默认 `nmse`/`mse` 保持原 Scout NMSE 排序。生效阈值可能被 `baseline_cap=0.99` 限制，因此此值**不是在所有任务上都严格等于 Candidate/(Reference×200)**。它与当前实际 PASS 门槛一致，优先接近但尚未通过的候选。
2. Bootstrap、Rescan、Confirm 失败时的 Scout GMean 存入 `TaskRecord.scout_gmean_mse`；Resume/DCP 通过 `to_dict()/from_dict()` 恢复；GMean-only 的配置指纹绑定 `scout_confirm_enabled`，旧 NMSE/MSE 指纹保持不变。
3. `experiment_50task_gmean200.yaml` 是新的**实验 YAML**；正式 NMSE YAML 未改。50 任务候选，当前 Registry PASS≥4 即停止，含 Bootstrap PASS。启用 Scout2→疑似 PASS 时 Confirm4；70:30；Hardness 0.10；每3次任务切换 Rescan。
4. `tools/gmean50_preflight.py` 只读验证 Reference 阈值的 `calibration.multiplier==200`、`stat=geomean`、`metric=mse`、`config_fingerprint`、50 任务完整覆盖、是否触发 baseline cap/CV 警告，并列出历史 JSONL Bootstrap 事件覆盖率。**历史事件当前不允许自动复用为 Scout 缓存**（无法仅凭日志证明候选权重、dtype、norm、具体 episode 集与噪声等全部一致）。
5. Scout/Rescan 的真实后端评测增加每任务墙钟与轨迹/s 埋点；Hardness 扫描原有同步总计时保留，真实后端新增 CPU 数据读取和 GPU 提交侧耗时（`score_submit_seconds` 是宿主侧提交时间，**不是 CUDA Kernel 耗时**）。

## 无卡执行（先做）

```bash
cd /data/code/lingbot-vla-v2
PY=/data/miniconda3/envs/lingbotvla/bin/python
$PY tools/gmean50_preflight.py \
  --config configs/auto_learning/experiment_50task_gmean200.yaml \
  --thresholds /data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json \
  --baseline /data/train/task_splits_50/task_baseline.json \
  --history docs/audit/gpu96_ratio_4task_2026-10-09/auto_learning_events.jsonl
$PY -m pytest -q tests/test_gmean_ratio_priority.py tests/test_gmean50_preflight.py \
  tests/test_gmean_pass_pipeline.py tests/test_resume.py tests/test_hardness_scan_timing.py
```

**只有 `status=READY` 且退出码 0，才允许考虑后续 GPU 运行。** 缺本地 Reference 表是 `BLOCKED`，必须先在远端核实；`READY` 并不表示历史 Scout 可复用，也不代表 GPU 批量推理已经接线。若 `baseline_capped_tasks` 非空，要清楚这些任务不是严格的裸 200× 门槛。

## GPU 启动门槛与重要边界

- 上述命令**不会运行 GPU、不生成缓存、不修改任何配置文件**。GMean 新 YAML 不会自行启动训练；需另行审批完整训练启动器、模型 step500、96G BF16 micro24/GAS1/GBS24、日志位置与存档空间。
- 一次启动的目标仍为**当前共4个 PASS**，如 Bootstrap 已有4个 PASS 可能**零训练步结束**；需先看 Bootstrap 结果，不得擅改目标或伪造 PASS。
- 历史 50task NMSE Scout 不能直接拿来排序 GMean；真正跨运行权重特定的 GMean Scout 缓存尚未启用，需要带模型权重/评测口径/轨迹ID等指纹的独立验收。
- **没有实现 `_infer_batch()`**；现有 `eval_batch_policy.py` 只是策略与安全门控。本补丁增加计时证据，以便下一次 GPU 测量 Batch=1 真实瓶颈；不宣称资源利用率已提升。
- Hardness 已经最多 8 样本 GPU Batch；本补丁仅分解 CPU 数据准备/打分提交侧耗时，不盲目改 Batch，不更改随机数与训练采样。
- 原有 DCP 每1000步/最终DCP、HF 里程碑、Replay 70:30、正式 `pass_metric=nmse` 均不改变。
- 不自动开启/关闭 GPU，不自动清理任何旧产物。

## 本次没有实现（下一轮根据测量与数值一致性补丁）

生产多轨迹批量 `_infer_batch()`、真正跨运行自动跳过 Scout 任务、Hardness 多 worker 预取。这三项都需要额外 GPU/CPU 审计及轨迹级数值对照，尤其不能让 batch 改变 GMean PASS/DEFER/REOPEN。
