# GMean 开环候选 PASS 线：改动、接入与验证

> 本文是 `feature-auto-learning-v1` 的**试验性离线阈值生成工具**使用说明，不是闭环成功率保证。参考倍数 220 只是待复测的候选参数。

## 修改范围（最小化）

- **新增** `lingbotvla/auto_learning/tools/build_gmean_thresholds.py`：以参考 50k 的逐轨迹 MSE 几何均值为参考，生成原有 `PassThresholds` 兼容的 JSON 表。
- **新增** `tests/test_gmean_thresholds.py`：边界、输入/输出、回归、CLI 端到端测试。
- **新增**本文件。未修改训练主循环、`Scheduler`、Replay、模型/优化器、原有 `compute_pass_thresholds.py` 和配置默认值。

## 算法

对于任务 t 的 n 条 **严格正值**参考 MSE：

```
G_t = exp(mean(log(ref_traj_mse_i)))
raw_t = G_t * multiplier
pass_t = min(raw_t, baseline_cap * baseline_mse_t)
```

默认 `multiplier=必须显式指定`（**可以先输入 220，不代表已经标定完成**）、`baseline_cap=0.99`、`min_trajectories=10`、`max_cv=1.5`。CV 使用 `statistics.stdev(values) / mean(values)`（样本标准差）。

- 参考 CV > 1.5：旧默认 `--high-cv-policy null` 会把阈值置 `null` 并跳过该任务；新的纯开环实验请显式用 `--high-cv-policy warn`，生成有效阈值同时记录 `calibration.tasks.<task>.high_cv_warning=true`。CV 高提示评测不稳定，**不能等同于模型已经掌握任务**。
- 每个任务逐轨迹样本不足 10、同轨迹重复、MSE≤0、NaN/Inf、任务或指纹不匹配：**报错且不覆盖旧输出文件**。
- 元数据包含逐任务 `gmean/CV/算术均值/生效阈值/触发上限情况`、原始文件的 SHA256，便于审计。
- JSON 顶层 `metric="mse"`，被现有 `PassThresholds.load()` 和运行时阈值表读取器兼容接收。

## 前置条件

1. 使用与原参考 50k 模型配套的 `task_baseline.json`（必须覆盖同一批任务）。
2. 必须有**逐轨迹**参考 MSE，而非 `tools/eval_ref_model_open_loop.py` 生成的每任务平均 `eval.jsonl`。
3. 参考 50k 与待测 step500 候选应在相同 held-out 轨迹、相同 fp32/bf16、seed、chunk 和归一化条件下重新对齐；历史锚点的 MSE **暂未对齐**。
4. 如果用现有 `open_loop_eval.log`，要检查它来自对应参考模型；日志本身不自带可验证的权重/推理精度信息。

## 方案 A：已有 `ref_per_traj.jsonl`（推荐）

文件每行格式：

```json
{"task":"click_bell","traj":42,"mse":0.000123,"split":"active_val"}
```

在 `lingbot-vla-v2/` 根目录运行（把路径换成你 GPU 机器的真实路径）：

```bash
python -m lingbotvla.auto_learning.tools.build_gmean_thresholds \
  --ref-per-traj /data/eval_results/open_loop/ref50k/ref_per_traj.jsonl \
  --baseline /data/你的目录/task_baseline.json \
  --reference global_step_50000 --multiplier 220 \
  --min-trajectories 10 --max-cv 1.5 --baseline-cap 0.99 --high-cv-policy warn \
  -o /data/eval_results/open_loop/ref50k/pass_thresholds_gmean220.json
```

## 方案 B：直接解析现有 `open_loop_eval.log`

这个脚本已有的逐条打印行格式是 `MSE for trajectory ID: VALUE`。用每任务 `*.val_ids.json` 建立任务与轨迹 ID 映射：

```bash
python -m lingbotvla.auto_learning.tools.build_gmean_thresholds \
  --ref-log /data/eval_results/open_loop/ref50k/open_loop_eval.log \
  --split-dir /data/train/task_splits_50 --val-trajs 10 \
  --baseline /data/你的目录/task_baseline.json \
  --reference global_step_50000 --multiplier 220 --high-cv-policy warn \
  -o /data/eval_results/open_loop/ref50k/pass_thresholds_gmean220.json
```

**注意**：历史日志已知可能有同一轨迹重复 10 次的情况。这里故意 fail-fast，而不是悄悄按最后一条覆盖。遇到重复请审查日志原因，必要时重新生成干净的逐轨迹数据；不要随意删掉不利结果。

## 使用 JSON 阈值表前的审核

- [ ] 检查 `calibration.source_sha256` 是否对应真实参考日志。
- [ ] 检查实际 `metric="mse"`、参考模型版本和 baseline 配置指纹。
- [ ] 检查所有任务的 `calibration.tasks.*.n`，默认各任务不少于 10 条。
- [ ] 检查 `effective_line` 是否 `< baseline_mse`，没有 `NaN/Inf`。
- [ ] 纯开环实验：检查 `--high-cv-policy warn`，列出 CV>1.5 的 `high_cv_warning` 任务，确认全部 50 个 task 有数值阈值；若仍使用旧 null 表，这些任务会被跳过。
- [ ] 候选模型与参考模型先完成统一精度+轨迹+噪声种子的配对开环重测，再决定是否启用 220 倍。
- [ ] 要用作真正的「任务学会」判定，还需要闭环实验验证。

如果只做离线判定对拍，暂时不用修改训练配置。**只有确认试验成功后**才在相应 Auto Learning YAML 中显式配置：

```yaml
pass_metric: mse
pass_thresholds_file: /data/eval_results/open_loop/ref50k/pass_thresholds_gmean220.json
```

原始 YAML 不被此次补丁改变。`pass_thresholds_file` 将由 `real/build.py` 校验任务覆盖、单位和 `config_fingerprint`。

## 测试方式

无需 GPU，进入 `lingbot-vla-v2/` 执行：

```bash
PYTHONPATH=.:tests python -m pytest -q tests/test_gmean_thresholds.py tests/test_pass_thresholds.py
```

已验证新工具和原有阈值测试一起通过。更广泛的 CPU 回归还覆盖了 Scheduler、Replay、Hook、恢复等模块。完整测试套件会因本地没有 `transformers` 而在其他 GPU 相关测试收集阶段中断；需要在完整训练环境上做额外测试。

## 本次没有实现的能力

- **没有**修改 Scheduler 去自动触发闭环，也**没有**新 `CANDIDATE_PASS` 状态。
- **没有**用真实 50k/step500 原始远端数据复算倍率（远端 `/data` 不在此沙箱）。
- **没有**进行 RoboTwin 闭环/GPU 验证。
- **没有**擅自改动正式训练默认阈值（如旧 `pass_nmse: 0.35`）。


## 2026-10-08 后续：纯开环课程低开销模式

最新增量代码包含 `scout_confirm_enabled` 和 `rescan_every_n_task_switches`。
在 `configs/auto_learning/formal_50task_4pass.yaml` 已选 `false` / `3`，但**保留历史默认 PASS 指标**。
在配对标定完成前，不要误认为此 YAML 已启用 GMean。具体流程参见 `docs/AL_OPENLOOP_UPDATE_GUIDE.md`。
