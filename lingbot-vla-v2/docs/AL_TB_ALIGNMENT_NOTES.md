# Auto Learning × TensorBoard 对齐：应用说明与实测口径

> 覆盖：`al_tb_alignment_fix.patch`（横轴对齐 / unit-loss 分离 / GMean 阈值指标）
> ＋ 2026-10-08 追加的两处：`detailed_loss` 跨 GBS 聚合、Hardness Scan 分项计时。
> 基线：`feature/auto-learning-v1`。

## 1. 🔴 补丁应用：**必须**用仓库根 + `--directory`（否则静默空操作）

该补丁的路径前缀是**相对包目录**的（`a/lingbotvla/...`，没有 `lingbot-vla-v2/`）。
在包目录里直接 `git apply` 会出现**"成功"假象**：

```bash
# ❌ 危险：看起来成功，其实什么都没做
cd /data/code/lingbot-vla-v2
git apply --check /tmp/al_tb_alignment_fix.patch    # rc=0
git apply         /tmp/al_tb_alignment_fix.patch    # 逐文件 "Skipped patch '...'"
git apply --stat  /tmp/al_tb_alignment_fix.patch    # "0 files changed"
```

```bash
# ✅ 正确：从仓库根加 --directory
cd /data/code
git apply --stat  --directory=lingbot-vla-v2 /tmp/al_tb_alignment_fix.patch   # 4 files changed, +140/-4
git apply --check --directory=lingbot-vla-v2 /tmp/al_tb_alignment_fix.patch   # rc=0
git apply -v      --directory=lingbot-vla-v2 /tmp/al_tb_alignment_fix.patch   # 逐文件 "Applied ... cleanly"
# 备选（GNU patch，同样可行）：cd lingbot-vla-v2 && patch -p1 < /tmp/al_tb_alignment_fix.patch
```

**应用后必须再验两件事**（`git apply --check` 只验上下文，不验任何语义）：
1. `git diff --stat` 有预期行数（本次：4 files, +140/−4）；
2. **`python -m py_compile <改动文件>`** —— 2026-10-08 实测过：一份缩进少 4 格的补丁能 `git apply --check` 通过但 `IndentationError`。

## 2. 横轴与标签语义（已实测确认）

| 标签 | 写入者 | 横轴 | 含义 |
|---|---|---|---|
| `training/loss`、`training/lr`、`steptime`、`detailed_loss/<task>` | 训练器直写 | **绝对** `global_step` | 单个 optimizer step |
| `auto_learning/unit_loss` | Scheduler（经适配器改名） | 绝对（`step + tb_offset`） | 该 **learning unit** 的逐 step 平均 loss（`real/hook.py`：`sum(_unit_losses)/len(...)`） |
| `current_skill/*` | Scheduler | 绝对 | **当步"当前任务"** 的 unit 指标 |
| `task/<task>/*`、`debug/<task>/*` | Scheduler | 绝对 | 该任务自己的评测指标 |

- `auto_learning_events.jsonl` 保留 `step`（**AL 相对步**，语义不变）并新增 `tb_step`（绝对）便于审计。
- 校准点：`train_lingbotvla.py` 在建 hook 之后、第一次 `on_step_begin(0)` 之前调用
  `logger.set_tb_step_offset(train_global_step=global_step, al_global_step=scheduler.state.global_step)`；
  该行缩进为 8 ⇒ 在 `if _al_parts is not None:` 内、**不在** `if _al_state is not None:` 内 ⇒ **fresh run 也生效**。
- ⚠️ **读图注意**：`current_skill/*` 会随任务切换**跳变**（实测：step505 对应 easy_pass 0.60，step510 跳到 unlearnable 2.25）。
  单任务学习进度请看 `task/<task>/val_to_pass_threshold`，不要把 `current_skill/*` 当单任务曲线。
- ⚠️ `current_skill/lp50` 是历史命名；若 `eval_interval_steps=5`，它表示"相邻 5 步"的进度，不是 50 步。

## 3. `TB=0` 的准确含义

`TB=0`（启动器）**只表示不自动拉起 TensorBoard 服务**；训练器仍然**无条件**创建
`AsyncTBWriter(log_dir=<TRAIN_OUT>/runs)`（`tasks/vla/train_lingbotvla.py`）⇒
`$TRAIN_OUT/runs/events.out.tfevents.*` 一定会有数据（2026-10-08 实测：`TB=0` 的首跑留档里确有 5702 B events）。
事后手工看板：`python -m tensorboard.main --logdir "$TRAIN_OUT/runs" --port 6006 --host 0.0.0.0`。
（`AsyncTBWriter` 是异步写盘：验收要在 `writer.close()` 之后，并留意后台写失败告警。）

## 4. `detailed_loss/<task>`：跨完整 GBS 聚合（2026-10-08 新增）

- **旧行为**：梯度累积时只统计**最后一个** micro-batch（原代码注释自认 "we only log the last mini batch"）⇒
  `MICRO=1/GAS=4` 时 `detailed_loss/<task>` 只反映 1/4 的 GBS，未出现在最后一批的任务**完全没有点**。
- **新行为**：`lingbotvla/utils/tb_task_loss.py` 的纯函数在**每个 micro-batch** 累积 `(sum, count)`，
  步末写 `detailed_loss/<task> = sum/count`（完整 GBS 的样本均值），并额外写 `detailed_loss_count/<task>` 便于审计。
- 约束：**只动日志**——不参与梯度、反向、优化器、采样与 RNG；张量每个 micro-batch `detach().cpu()` 一次（不逐样本同步）；
  该模块不 import torch/numpy（无 GPU 也能单测）。
- 复现差异（假数据 GAS=4、4 样本分属 2 任务）：旧 `{taskA: 0.6}` vs 新 `{taskA: 0.3333, taskB: 0.5}`。

## 5. Hardness Scan 分项计时（2026-10-08 新增）

标签：`auto_learning/hardness_scan_seconds`（原始）、`..._seconds_warmup`（**首次**，含 torch.compile/预热）、
`..._seconds_steady`（稳态）、`hardness_scan_samples`（`n_scanned`）、`hardness_scan_trajs`（扫描轨迹数）；
事件里另加 `hardness_scan_seconds` 与 `hardness_scan_is_first`。

设计约束：
- **只在扫描前后各 `torch.cuda.synchronize()` 一次**（`_cuda_sync_for_timing`，惰性 import torch，无 CUDA 时 no-op）——
  绝不逐样本同步；否则会把异步排队时间算进/漏出。
- **首次与稳态分开记**：首次扫描包含 train-mode 的 `torch.compile`，不能算作难度扫描成本
  （2026-10-08 首跑实测 `13:13:51 → 13:20:00 = 369 s` 这个窗口，末尾才出现 dynamo 重编译告警）。
- 计时打开/关闭时，扫描结果（probs/losses/样本 ID/轨迹）、调度决策与 RNG 状态必须逐位一致（有回归测试）。

**成本参考（进程内评测，实测）**：约 **0.6 s/chunk**；2 轨迹/4 chunk ≈ 3–5 s；4 轨迹/16 chunk ≈ 10 s。
日志开销实测 **18.5 µs/条**（jsonl append + writer）⇒ 一个 unit ≈30 条 ≈ **0.56 ms**，可忽略。

## 6. 显存实测（RTX 4090 48G，MICRO=1/GAS=4/GBS=4/bf16，`step500` 起）

| 时段 | 峰值 |
|---|---|
| 启动 + 模型/数据集加载 | 15.07 GB |
| **Bootstrap + select + HardnessScan（`max_batch=8`）** | **15.31 GB**（≈ 常驻模型，`no_grad` 下几乎不涨） |
| 训练 1 step + 单元评测 | **38.63 GB** ← 真正的显存瓶颈 |
| 收尾存档 | 27.70 GB |
| 全程峰值 / 余量 | 38.63 GB / 49.14 GB ⇒ **余量 9.36 GB** |

⇒ 打分走 `torch.no_grad()` + `model.eval()` + 固定 noise/time（自带 generator）⇒
**`max_batch=8` 在 48G 上无需下调**；瓶颈是训练步本身（micro=1 已是最小 micro）。

## 7. Smoke 验收清单（5 步 unit / 10 步）

从 `step500` 起跑（`STEP_OFFSET=500`，**`max_steps` 是绝对值**）：

| 再训步数 | `MAX_STEPS` | 期望的 unit 边界 |
|---|---|---|
| 5 | 505 | 1 个 unit @505 |
| **10** | **510** | **2 个 unit @505、@510** |

- Bootstrap 的 `debug/<task>/*` 落在 **500**；
- 每个 unit 的 `current_skill/*` 与 `auto_learning/unit_loss` 落在 505 / 510；
- 训练器 `training/loss` 落在 501…510，且**不得**混入 unit 平均值；
- `task/<task>/val_to_pass_threshold` 应与 `task/<task>/val_mse ÷ pass_threshold_mse` 一致；
- Resume 后 AL 数据**不得**回到 0（`tb_step` 应为恢复后的绝对步）；
- `runs/events.out.tfevents.*` 与 `auto_learning_events.jsonl` 都必须有数据。
