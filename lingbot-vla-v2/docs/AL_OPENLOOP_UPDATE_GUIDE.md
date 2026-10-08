# Auto Learning 纯开环低开销更新（增量补丁）

> 基线：此前交付的 `feature-auto-learning-v1 + GMean` 完整 ZIP。适用于你们目前只用开环判断 PASS、闭环以后手动评测的研发阶段。

## 变更与范围

1. `config.py` 新增 `scout_confirm_enabled`（默认 `true`，兼容旧行为）。置 `false` 后 Bootstrap 和全候选池 Rescan 只使用两条 Scout 轨迹判断开环 PASS；保留每个 Learning Unit 的 4 条 Active Validation 和遗忘复查。
2. `config.py` 新增 `rescan_every_n_task_switches`（默认 `1` 保持兼容）。正式 50 任务配置设置 `3`；**每完成三次主训练任务切换**才做一次全候选池 Rescan。`round_rollover` 时对被再次纳入候选的任务执行定向重扫，不受这个间隔限制。
3. `scheduler.py` 增加持久化的 `task_switch_count/full_rescan_count`，避免 review/reopen 增加的 `transition_count` 干扰全池 Rescan 节奏；检查点恢复后仍保持“三次触发”。
4. `build_gmean_thresholds.py` 新增 `--high-cv-policy warn`：当 CV>1.5，**继续输出数值阈值**，并在 `calibration.tasks.<name>` 下记录 `high_cv_warning`、`cv`、`gmean`、`effective_line`。原来 `null/error` 保持可用，工具默认仍为 `null`，避免旧脚本意外修改语义。
5. 同工具修复 `--output` 指向目录时抛裸 `IsADirectoryError` 的错误体验：现在报 `ThresholdsError`，且不产生临时文件。
6. 新增 `tests/test_al_openloop_update.py`，重点针对开关、输入输出、CV/错误路径、恰好第 3 次触发、恢复、Replay 和旧行为不回归。
7. 更新 `docs/GMEAN_PASSLINE_GUIDE.md` 并在正式 YAML 加入已批准的 Scout/Rescan 选项。**没有改模型、训练主循环、优化器、Replay 策略，也没有把 220× 设为已标定的正式线。**

## 现在的状态与启用步骤

- 已启用（YAML）：`global_scout_val_trajs: 2`、`scout_confirm_enabled: false`、`rescan_every_n_task_switches: 3`。
- 未启用（YAML）：新的 GMean PASS 指标；当前 `pass_metric` 仍未显式设置，默认是 `nmse`。等待参考/候选一致口径的配对开环标定、选择好倍率之后，再指定：

```yaml
pass_metric: mse
pass_thresholds_file: /data/eval_results/open_loop/ref50k/pass_thresholds_gmean_calibrated.json
```

使用新 CV 策略，**必须重新生成阈值表**。之前的 `gmean220.json` 是 `null 6/50`，直接复用仍会跳过 6 任务。

示例（把 baseline/数据路径换成实际值）：

```bash
python -m lingbotvla.auto_learning.tools.build_gmean_thresholds \
  --ref-per-traj /data/eval_results/open_loop/ref50k/ref_per_traj.jsonl \
  --baseline /data/actual/task_baseline.json \
  --reference global_step_50000 \
  --multiplier 220 --max-cv 1.5 --high-cv-policy warn \
  -o /data/eval_results/open_loop/ref50k/pass_thresholds_gmean220_warn.json
```

`220` 是等待配对标定的**候选倍率**，不能视为正式启用数值。`warn` 允许高 CV 任务参与训练/开环 PASS，但未消除测量波动；实际成功率仍要以后手动闭环评测。

### 需要知道的行为变化

- Bootstrap 和 Rescan 的两轨迹直通可能增加噪声引起的 **误 PASS**；在本阶段用来减少评测开销，需要在实验日志标记 `pass_source=scout_direct`。如果假 PASS 过多，可把 `scout_confirm_enabled` 恢复为 `true`。
- Review 阶段针对遗忘嫌疑的 confirm 仍保留；本轮只取消 Bootstrap/Rescan 的额外 4-val confirm。
- 即使全池 Rescan 每 3 次任务切换一次，`rollover_round` 仍可能做**指定任务的定向扫描**，不等于总扫描次数严格减少三分之二。
- `rescan_every_n_task_switches` 不按 50-step Learning Unit 计数：同一个任务多个 Unit 不会触发全池 Rescan。
- 新代码中直接从已有 checkpoint 恢复时，新增计数器缺失会从 0 开始。正式长跑前须确认新建运行或在 checkpoint 迁移时显式映射进度；不要未经验证热换已运行中的训练配置。
- `--noise_repeats k` 的重复 `(task,traj)` 数据在当前生成器仍需由上游明确聚合，否则会 fail-closed。历史混有重复轨迹的日志不能直接生成阈值。

## 测试与远端验收

在 `lingbot-vla-v2/` 目录：

```bash
PYTHONPATH="$PWD:$PWD/tests" python -m pytest -q tests/test_al_openloop_update.py tests/test_gmean_thresholds.py tests/test_pass_thresholds.py tests/test_scheduler_c.py tests/test_resume.py
PYTHONPATH="$PWD:$PWD/tests" python -m pytest -q tests --ignore=tests/test_disk_guard.py
```

本地 CPU 测试：450 passed、12 skipped（`test_disk_guard.py` 依赖本沙箱缺失的 transformers/torch 环境，被单独排除）。**没有运行 GPU 或正式 50 任务实验**。远端需要复核：

1. 源码与补丁基线一致，保存备份、比较 SHA256、先 `git apply --check`，再应用。
2. 检查 50 个任务均生成阈值（特别是原先 `null` 的 6 个高 CV 任务），全部 `effective_line>0` 且 `< baseline`。
3. 跑模拟 Scheduler，确认第 1/2 次切换不扫描，第 3 次恰好全池重扫，Checkpoint Resume 后节奏保持。
4. 统计真实任务评估调用/时间：Bootstrap、Rescan、Active Validation、Hardness Scan 和 Review；请基于实测决定 3 是否还需改为 5。
5. 重新审查并核准 PASS 指标切换；**不得自动启动正式 GPU 训练**。
