# Auto Learning：当前总 PASS 目标、无存档 Smoke、TensorBoard 与 Hardness 10%

代码对齐 Git HEAD：`a01a40482e6af89aa669a05ca6e82f6cb5532622`（用户 2026-10-08 提供的源码包）。

## 1. 完成条件：当前通过 4 个任务

新配置 `target_total_passed_tasks: 4` 使用 Registry 里**当前** `TaskStatus.PASS` 的不同任务数量，而不是 `newly_passed` 历史累计数。

- Bootstrap 已 PASS 2 个 + 本次训练再 PASS 2 个 => 总数 4，收工。
- Rescan 免费 PASS 也计入；重开（REOPEN）/丧失 PASS 的任务不计入当前总数。
- Bootstrap 未扫描完成时继续扫描，扫描完成后立即检查目标，**不再进入昂贵 Hardness Scan**。真实训练入口已额外保护：首次就收工时不构造缺少 TrainRequest 的 DataLoader 迭代器。
- 训练 Unit 达标触发的 PASS 在全池 Rescan 之前检查停止，避免已达目标却进行大规模额外扫描。训练器在下一轮 `global_step += 1` 之前检查结束状态，防止凭空多算一步。
- 新 `target_total_passed_tasks` 与旧 `max_new_tasks_passed_this_run` 同时设置时，**满足任一停止**。建议实验只配置一个。旧字段逻辑未改。非目标因素（耗尽、预算到顶）仍可以提前结束，不能保证一定达到目标。
- 新目标加入 Resume 的语义指纹；**目标为 None 时不添加新指纹字段**，避免旧 DCP 无故失配。真正更改总目标并恢复旧 Run 时应按指纹变更流程审查。

正式 YAML `configs/auto_learning/formal_50task_4pass.yaml` 现设置 `target_total_passed_tasks: 4`，旧 `max_new_tasks_passed_this_run: null`。**正式 YAML 仍未启用 GMean-MSE（旧 `pass_metric` 默认 nmse）；正式启用 MSE 表需要另行批准。**

## 2. 无存档短跑：Smoke 专用

使用独立 `TRAIN_OUT`，以 `SMOKE_NO_CHECKPOINT=1` 启动。

```bash
export TORCH_COMPILE_DISABLE=1
SMOKE_NO_CHECKPOINT=1 MICRO=1 GAS=4 N_GPU=1 MIXED=false \
  STEP_OFFSET=500 MAX_STEPS=510 \
  AL_CFG=/data/code/lingbot-vla-v2/configs/auto_learning/smoke_gbs4_4task_tb5.yaml \
  TRAIN_OUT=/data/outputs/al_smoke_tb10_nockpt DRY_RUN=1 TB=0 \
  bash experiment/robotwin/al_50task_bf16.sh
```

核对 DRY_RUN 中必须存在：`--train.smoke_no_checkpoint true`、`--train.save_steps 0`、`--train.save_epochs 0`、`--train.save_hf_weights false`、`--train.disk_guard false`。确认后设 `DRY_RUN=0` 才能真实启动。`STEP_OFFSET=500` => 10 个新 optimizer step 以 `MAX_STEPS=510` 结束。

此开关禁用：周期 DCP/HF、收尾 DCP/HF、轮末 DCP/HF、异步 HF 队列、DCP 剪枝进程及无意义的磁盘存档检查。保留：AL Unit 结算、事件 JSONL、TensorBoard（`TB=0` 仅关闭 TB **服务**，不会关闭 events 写入）、日志 flush/close。代码里可能仍创建空 `checkpoints/` 目录及模型 processor/config 资产目录，**不等于写入 DCP/HF**。

**限制：不支持 Resume。** 当 `SMOKE_NO_CHECKPOINT=1` 同时 `RESUME=1`，启动器直接失败；直接调用训练器时也会失败。不要在此模式下下达 `STOP_AND_SAVE` 预期得到 Checkpoint（它只会正常停止而不存档）。

### 对 GPU runner 的要求（非常重要）

之前远端 `/data/tmp/run_smoke_tb10_48g_v2.sh` 的成功判定要求存档落地。当前它**不在此次仓库补丁的受控版本里**。交给 WorkBuddy 时请让它审核并适配 runner：

1. `SMOKE_NO_CHECKPOINT=1` 时，**成功条件改为训练 rc=0、实际到达目标步数或 AL 正常收工、事件/TB 已写盘、训练器退出**，绝不能再要求 DCP/HF 目录存在。
2. 不启动 DCP 剪枝，不触发 HF 完整性验证；如仍有系统自动关机，须在 `JOB_END` 与完整日志/报告落在持久目录后才允许关机。
3. 不要把旧失败 Run 的 TensorBoard events 混入本次新目录。原本已存在的残缺 `.hf_ckpt.tmp.*` 不受该开关修复。

## 3. TensorBoard 新标签

旧标签**全部保留**，新建 Run 可用以下标签过滤：

| 新标签 | 含义 |
| --- | --- |
| `curriculum/current_task_name` | **Text 面板**：当前训练任务的完整名称；事件 JSONL 同样记录 `kind=text`；选择任务完成后和 Unit 末尾更新 |
| `curriculum/active_task/<task>` | Scalars：该任务在本次 Unit 活跃（值 1），用标签名称直观看训练对象 |
| `task/<task>/unit_loss` | 当前任务所辖 Learning Unit 的**混合批次**平均训练 Loss（包括 NEW 与 Replay），**不是仅该任务样本的单任务 Loss** |
| `curriculum/passed_tasks` | Registry 当前 PASS 的不同任务总数 |
| `curriculum/target_passed_tasks` | 当前配置的总 PASS 目标 |
| `curriculum/bootstrap_pass_count` | 当前仍维持 PASS 的 Bootstrap 初始通过任务数（不含 Rescan） |
| `curriculum/newly_passed_count` | 本轮主动训练后首度 PASS 的累计数 |
| `curriculum/units_completed` | 已完成 Learning Unit 数 |
| `replay/available_tasks` | 当前 PASS Replay 池的任务数 |
| `sampling/total_samples` | 累计训练样本数 |
| `sampling/replay_samples_per_unit` | 本 Unit 实际 OLD 样本槽位数 |
| `sampling/replay_distinct_tasks_per_unit` | 本 Unit 的 OLD 样本来源任务数 |
| `curriculum/forgotten_tasks`、`curriculum/reopened_count` | 遗忘/回炉计数 |
| `diagnostics/current_task_val_nmse` | NMSE 辅助指标，不参与 MSE 模式的正式 PASS 决策 |
| `auto_learning/hardness_scan_mean_loss`、`auto_learning/hardness_scan_p90_loss` | 难度扫描样本 Loss 分布概览 |

`training/loss` = 单个 optimizer step 的训练 Loss；`auto_learning/unit_loss` = 本 Unit 所有 step 的平均 Loss，切换任务会导致两点之间上下跳动。`task/<task>/unit_loss` 便于按**所在训练任务**筛选 Unit，但因 GBS 中有 Replay 样本，不能说这是当前任务样本的独立 Loss。

## 4. Hardness 10%

`AutoLearningConfig` 默认和正式/两份 Smoke YAML 中的 `hardness_probe_fraction` 已改为 `0.10`。对 40 条训练轨迹会抽 `ceil(40*0.1)=4` 条完整轨迹，每条仍扫描所有有效帧；其余样本使用原来的默认难度权重。已有的 sampling 权重、RNG 和 Flow Matching 计算逻辑未改。

注意：占用时间可能按样本数下降，但实际耗时仍受任务轨迹长度、数据 IO 和 GPU 前向影响。不要在未测量时声称精确节省 70%。

## 5. 审核与部署

本补丁路径基于 **Git 仓库根目录**，包含 `lingbot-vla-v2/` 前缀。请在目标环境执行：

```bash
cd "$(git rev-parse --show-toplevel)"
git rev-parse HEAD       # 应先核对为 a01a404...
git status --short       # 必须先审核工作区，避免覆盖尚未提交的工作
git apply --check /path/to/al_totalpass_smoke_tb.patch
git apply --stat /path/to/al_totalpass_smoke_tb.patch  # 非 0 files
git apply -v /path/to/al_totalpass_smoke_tb.patch
cd lingbot-vla-v2
python -m compileall -q lingbotvla/auto_learning tasks/vla/train_lingbotvla.py lingbotvla/utils/async_tb_writer.py
bash -n experiment/robotwin/al_50task_bf16.sh
python -m pytest -q tests/test_al_smoke_visibility_total_pass.py
```

首次 GPU 测试尚未在这份新代码上进行，务必由 WorkBuddy 在远端先完整 CPU 回归、DRY_RUN 和 runner 无存档条件审查。禁止把本地 CPU 测试写成 GPU 已验证。
