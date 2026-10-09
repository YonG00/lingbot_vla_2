# 自动学习（Auto Learning）多卡支持说明

> 更新：2026-10-09。目标场景：**2 × 48G 卡**先跑通，再考虑更大规模。
> 结论：**DDP（参数复制）已支持**；**FSDP1/2 仍明确拒绝**（参数分片 ⇒ 单 rank 评测结果无效）。

---

## 1. 支持范围

| 并行模式 | 自动学习 | 说明 |
|---|---|---|
| `ddp`（**默认**） | ✅ **支持** | 参数复制 ⇒ 任一 rank 都有完整参数；评测由 rank0 计算并广播 |
| `fsdp1` / `fsdp2` / `fsdp2-vescale` | ❌ **拒绝**（fail-fast，信息明确） | 参数分片 ⇒ rank0 上只有局部参数，评测结果无效；且 rank0 评测时其它 rank 继续训练会与 all-gather 错配 |

**训练/采样侧本来就已经支持 DP**（无需改动）：
* `batch_ratio.ratio_plan(GBS, dp_size, dp_rank, new_ratio)`：全局取整 → 按 rank 均分（保证各 rank `local_batch_size` 相同）；校验 `GBS % dp_size == 0`；
* `dataloader_batch_size = GBS // dp_size`（= micro × gas），并校验 `GBS == dls × dp_size`、`cfg.batch_size == dls`；
* AL hook 在**所有 rank** 构建；`cfg._ratio_dp_rank/_ratio_dp_size` 已设；日志打印 `global ratio plan: GBS=…, DP=…`。

## 2. 评测同步协议（本次新增）

AL 的 hook 在**所有 rank 无条件**驱动调度器 ⇒ 评测也必须在所有 rank 上"步调一致"。协议：

```
所有 rank 一起进入 evaluate_ids
   └─ rank0：跑评测（safe_eval_context + inference_mode）→ 广播 {"ok":True,"result":…}
   └─ rank≠0：不计算，直接等在广播上 → 收到同一份 result
若 rank0 失败 ⇒ 广播 {"ok":False,"error":…} ⇒ **所有 rank 一起抛**（不会有人永久等在广播上 ⇒ 不死锁）
```

* **为什么由 rank0 统一算**：MoE 原子加让不同 rank 的同一次前向有 1e-2 级差异；若各 rank 各算，
  贴线的 PASS/FAIL 可能分叉 ⇒ **各 rank 训练不同任务**（灾难）。统一用 rank0 结果 ⇒ 决策必然一致。
* **为什么所有 rank 都进**：避免有的 rank 先跑进 DDP 的集合通信而别的还在评测 ⇒ collective 错配。
* 单卡（`world_size == 1`）路径**逐字不变**（直接本地跑，无广播）。

## 3. 多卡下的写入守卫

| 写入方 | 守卫 |
|---|---|
| 事件 JSONL `auto_learning_events.jsonl` | **只有 rank0 写**（`SchedulerLoggerAdapter(write_events=…)`，默认按 dist rank 自动判定） |
| Bootstrap Scout 缓存 | **只有 rank0 写**（`BootstrapScoutCache(write_enabled=…)`）；各 rank 仍可读 |
| 评测探针证据 `AL_EVAL_BATCH_PROBE_DIR` | 多卡时自动落到 `…/rank{N}/` 子目录（互不覆盖） |
| TensorBoard | 本来就是只有 rank0 有 writer（其余 rank 是 dummy） |

## 4. 48G × 2 推荐配置

```bash
cd /data/code/lingbot-vla-v2
export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct
export CUDA_VISIBLE_DEVICES=0,1          # ⚠️ 启动器默认是 :0，2 卡必须显式给

setsid nohup env \
  MICRO=1 GAS=2 N_GPU=2 \                # GBS = 1×2×2 = 4（per-rank batch = 2）
  AL_EVAL_BATCH_MAX=4 \                  # ⚠️ 关键：评测批大小与训练 micro 解耦
  TB_PORT=6007 \
  AL_CFG=configs/auto_learning/experiment_50task_gmean200_train2.yaml \
  TRAIN_OUT=/data/outputs/al_50task_2gpu \
  MODEL_PATH=/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt_bf16 \
  STEP_OFFSET=500 MAX_STEPS=20000 \
  AL_EVAL_BATCH_MODE=auto AL_EVAL_BATCH_APPROVED=1 \
  bash experiment/robotwin/al_50task_bf16.sh > /data/tmp/al_2gpu.log 2>&1 < /dev/null &
```

**为什么不用 `MICRO=2`（虽然算式也自洽）**：48G 上历史实测（micro=1、gas=4、GBS=4）训练步峰值已
**38.63 GB**（余 9.36 GB），micro=2 会让激活约翻倍 ⇒ OOM 风险高。先 `MICRO=1 GAS=2` 跑通链路，再单独试 micro=2。

**为什么必须显式 `AL_EVAL_BATCH_MAX`**：评测批大小默认 = `micro_batch_size`，micro=1 时评测会**退化成串行**
（白丢实测的 3.6× 收益）。评测无梯度、显存压力小，批 4/8 安全。

**GBS=4 / DP=2 的 NEW/Replay 分配**（`ratio_plan` 实测口径）：
`global_new = floor(4×0.7+0.5) = 3`、`global_replay = 1` ⇒ **rank0 = 1 NEW + 1 Replay，rank1 = 2 NEW + 0 Replay**
（各 rank `local_batch_size` 都 = 2 ✓，这是 DDP 的硬要求）。

## 5. 2 卡短跑验收清单

跑之前先把 `MAX_STEPS` / eval 间隔调小（快速暴露问题），逐条核对：

| # | 检查 | 期望 |
|---|---|---|
| 1 | 启动日志 | `[open_loop] 多卡评测已启用（world_size=2 + DDP 参数复制）` |
| 2 | 批计划 | `[auto_learning] global ratio plan: GBS=4, DP=2, NEW=3 Replay=1; rank0 local=1+1` |
| 3 | 批一致性校验 | 无 `AL logical batch_size != dataloader_batch_size` 报错 |
| 4 | 评测 | 每个任务只出现**一套**评测日志（rank0 的），各 rank 指标一致 |
| 5 | 事件流 | `auto_learning_events.jsonl` 行数 ≈ 单卡同规模，**不是 2×**（证明只有 rank0 写） |
| 6 | 决策一致 | 各 rank 的 `select/train_unit` 任务相同（日志里 `info_rank0` 只有 rank0 打，重点看无分叉报错） |
| 7 | 显存 | 两卡峰值 < 48G 且有余量（建议留 ≥8 GB） |
| 8 | 训练推进 | `Step:` 正常递增，`StepTime` 与单卡同量级（DDP 会引入 all-reduce 开销） |
| 9 | 编译 | 无异常 dynamo 报错；`use_compile` 与单卡同设置 |
| 10 | 收尾 | 正常收工后存档 **只由 rank0 写一份**（DCP 目录数=1） |

## 6. 已知限制 / 未做

| 项 | 状态 |
|---|---|
| FSDP1/2 多卡评测 | **未支持**（明确拒绝，不做静默降级） |
| `AL_HARDNESS_SELFTEST`（验收钩子） | 明确拒绝多卡（避免 collective 死锁；仅验收用） |
| 评测算力 | 只有 rank0 计算，其余 rank 等待（评测现在很快：一个任务 ≈ 一次前向 2.5 s；不值得为省这点做并行分片评测） |
| 多卡 eval 的"每 rank 各算一部分" | 未做：会让指标不一致 ⇒ 决策分叉风险，刻意不做 |
| 3 卡及以上 | 代码路径同一套（DP=3+），但**未实测**；注意 `GBS % dp_size == 0` |
| 数据读取 | 评测数据集在每个 rank 都会构建（用于 rank0 计算）⇒ 非 rank0 的构建是浪费，先不做优化 |


## 6b. 程序体检（**先跑这个**，CPU + gloo，不需要 GPU）

```bash
PY=/data/miniconda3/envs/lingbotvla/bin/python
$PY tools/multigpu_eval_smoke.py              # 2 进程
$PY tools/multigpu_eval_smoke.py --procs 4    # 3 卡以上同协议
```
它真起 N 个进程 + gloo，逐项验证（**带超时，超时即判"疑似死锁"**）：

| # | 检查 | 期望 |
|---|---|---|
| 1 | rank0 计算 + 广播，所有 rank 收到**完全相同**的结果（含嵌套 dict / 中文 / None） | 一致 |
| 2 | 非 0 rank **不得自己计算**（否则指标可能分叉） | 本地计算次数 = 0 |
| 3 | rank0 失败 ⇒ **所有 rank 一起抛**、且错误信息带原始原因 | 全部抛 RuntimeError |
| 4 | 失败路径之后**通信组仍可用**（协议没把组搞坏） | 第二次广播正常 |
| 5 | 事件 JSONL **只有 rank0 写**（N 个 rank 各写 2 条 ⇒ 应只有 2 行，不是 2N） | 行数 = 2 |
| 6 | scout 缓存**只有 rank0 写** | 文件数 = 1 |

本机实测（2026-10-09，CPU/gloo）：`--procs 2` 与 `--procs 4` **均通过**（用时 1.2 s / 1.8 s）。
已纳入测试套件：`tests/test_multigpu_smoke_tool.py`（2 项，子进程调用工具并断言 rc=0、无死锁标记）。

退出码：`0` 全过 / `2` 失败（含死锁）/ `3` 本机无 `torch.distributed`（跳过）。
**为什么必须真跑进程**：pickle 不了的结果、rank0 失败导致对端死等、事件流被每个 rank 各写一份 —— 这三类坑
用 mock 测不出来（本次就在工具自身抓到一次"把函数对象塞进共享 dict ⇒ PicklingError"）。

## 7. 相关文件

| 路径 | 作用 |
|---|---|
| `lingbotvla/utils/open_loop_validation.py` | `_multirank_eval_payload()`（协议）、`_ddp_replicated()`、`_data_parallel_mode()`、`_broadcast_object()`、构造/批处理守卫、探针目录按 rank |
| `lingbotvla/auto_learning/real/build.py` | `SchedulerLoggerAdapter(write_events=…)`、`_is_rank0()` |
| `lingbotvla/auto_learning/scout_cache.py` | `BootstrapScoutCache(write_enabled=…)` |
| `lingbotvla/auto_learning/batch_ratio.py` | `ratio_plan()`（GBS→per-rank 分配，原有） |
| `tools/multigpu_eval_smoke.py` | **多进程体检**（CPU/gloo，带超时判定死锁） |
| `tests/test_multigpu_eval.py` | 17 项：协议（rank0/非0/失败/空 payload）、模式判定、写入守卫、源码接线 |
| `tests/test_multigpu_smoke_tool.py` | 2 项：子进程真跑体检工具（2/4 进程）并断言无死锁 |
