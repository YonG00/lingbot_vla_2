# 多卡（FSDP2）开环评测：rank 不对称 ⇒ NCCL 看门狗 600 s 的定位与守门

> 2026-10-10，AMD 8×W7900D / ROCm / 8 卡 FSDP2 正式 AutoLearning 训练实测。
> 结论：**故障不在「评测算法」，而在「多卡评测协议的失败可见性」**——
> 只要有一个 rank 没走到评测末尾的那次 `broadcast_object_list`，其余 rank 就会在
> FSDP all-gather 上死等，600 s 后 NCCL 看门狗 abort **整个作业**，而真因会被
> `NCCL communicator was aborted` 完全埋掉。
>
> **第 1 轮**（当天早些时候）新增「一致性预检（fail-closed）+ 每 rank 阶段面包屑 +
> 停滞告警 + 失败绝不静默」。
>
> **第 2 轮**（当天真机 8 卡复跑，`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120` 加速暴露）
> 又抓到三条硬事实，本轮据此修：
> 1. **预检放错了位置**：它写在 `_evaluate_ids` **里面**，而 `unshard()`
>    （`_fsdp_full_params_context`）本身就是集合通信且在它之前 ⇒ 真机日志里
>    `一致性预检通过` 出现 **0 次**、`评测窗口：unshard()` 出现了。
>    **预检必须前移到窗口之外**（本轮第一优先级）。
> 2. **7 个 rank 停在 `unshard()` 等 1 个 rank**：那 1 张卡（rank4）在做**非集合通信的
>    本地重活**（首次前向的 kernel 编译，分钟级），GPU 100%、其余 7 张 0%；而且
>    各 rank 的首次编译会被集合通信**串行化**（谁先到谁先编译）⇒ 全组干等到看门狗。
>    对策：**窗口前对称预热**（预检集合点之后、真实窗口之前，所有 rank 各跑 1 次真实推理）。
> 3. **rank4 一行阶段日志都没有** ⇒ 「每 rank 阶段日志」当时并不成立（logger 级别/
>    `LOCAL_RANK` 门控/输出重定向都可能吞掉某个 rank）。对策：阶段面包屑改为
>    **stderr(flush) + 逐 rank 独立轨迹文件**，不再经过 logger。
>
> ⚠️ 本轮实测发现的一条硬约束（决定了预热的形状）：**FSDP2 根单元的参数在 `unshard()`
> 之前是 DTensor**，直接前向必报
> `aten.mm.default: got mixed torch.Tensor and DTensor`（本机 CPU/gloo 复现）
> ⇒ 「预热」不可能零集合通信，它必须带**自己的一次全参数窗口**；于是顺序定为
> **本地准备 → 预检 A → 预热（自己的窗口）→ 预检 B → 真实窗口**，并在预检 A 里
> **比对「谁打算预热」**（一个 rank 进窗口、另一个不进 = 新的死锁，实测过）。

---

## 1. 真机现象的判读（怎么从日志定位到「哪个评测、卡在哪」）

日志（用户提供）：

```
[rank1]:[E1010 06:10:06 ProcessGroupNCCL.cpp:744] [Rank 1] Some NCCL operations have failed or timed out.
[rank1]:[E1010 06:10:06 ProcessGroupNCCL.cpp:758] [Rank 1] To avoid data inconsistency, we are taking the entire process down.
[rank1]:[E1010 06:10:06 ProcessGroupNCCL.cpp:2059] [PG ID 0 PG GUID 0(default_pg) Rank 1] Process group watchdog thread terminated with exception …
10/10/2026 06:10:06 - WARNING - __main__ - [open_loop] ⚠️ [al_adjust_bottle_val_e10f709e] 评测失败: DistBackendError: NCCL communicator was aborted on rank 1.
```

三条判读（都可离线复核）：

1. **`al_adjust_bottle_val_e10f709e` 里的 8 位指纹 = 评测回合集合的指纹**
   （`EvaluatorAdapter.ids_fingerprint` = `sha256(sorted(set(ids)))[:8]`）。
   `adjust_bottle` 的 val 回合 = `[1, 5, 20, 21, 28, 29, 33, 34, 41, 49]`
   ⇒ 前 2 条 `{1,5}` 的指纹正好是 `e10f709e`；前 4 条的指纹是 `9cdb4cbf`（不是它）。
   `configs/auto_learning/al_50task_gmean100_rocm.yaml` 里
   `global_scout_val_trajs: 2` / `active_val_probe_trajs: 4`
   ⇒ **挂掉的是 scout 评测（2 条回合）**，而 `TASK_ORDER[0] == "adjust_bottle"`
   ⇒ 它就是 bootstrap 的**第一个**评测（scout 缓存未命中时必然真跑模型）。
2. **卡住的集合通信在 `default_pg`**（PG ID 0）：FSDP2 的 all-gather 走的是
   device mesh 自己的 process group（`fsdp_mesh = device_mesh["dp_shard"]`，
   `lingbotvla/distributed/parallel_state.py`），**只有**
   `_multirank_eval_payload` 末尾的 `dist.broadcast_object_list`（以及 `dist.barrier`）
   用 default_pg ⇒ **有人没到那次广播**。
3. 时间线 `plan 05:58:43 → abort 06:10:06` ≈ 600 s = `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`
   默认值 ⇒ 看门狗等满 10 分钟才动手（**不是**评测跑了 12 分钟）。

### 为什么日志里没有真因

`_multirank_eval_payload`（分片并行分支）旧实现：

```python
else:                      # 非 0 rank
    try:
        run()              # 只为参与集合通信
    except BaseException:  # ← 静默吞掉
        pass
```

* 该 rank 失败后**直接去等末尾广播**；其余 rank 还在 all-gather 里（少一个参与者）
  ⇒ 双方互等，谁都不抛；
* 600 s 后看门狗 abort，所有 rank 报 `DistBackendError`；
* 若失败发生在 **all-gather 已经发出之后**，连 `_evaluate_run` 里那条
  「评测失败: <真因>」的 WARNING 都可能来不及/被 abort 掩盖。

⇒ 这一类故障在当前代码里**结构上不可诊断**，必须先修「失败可见性」。

---

## 2. 本次修复（`lingbotvla/utils/open_loop_validation.py`）

| # | 机制 | 作用 | 开关 |
|---|---|---|---|
| ① | **一致性预检** `_multirank_eval_preflight` + `_eval_preflight`：在**进入推理循环（= 集合通信区）之前**用 `all_gather_object` 交换各 rank 的 `tag / ids / 推理次数 n_starts / 数据集长度 len_ds / 准备阶段错误`，**任意不一致或任意 rank 失败 ⇒ 所有 rank 一起抛**（fail-closed） | 把「死锁 600 s + 真因丢失」变成「立刻抛出、点名 rank 与字段」；直接回答「各 rank 的评测集合/顺序是否一致」 | `AL_EVAL_PREFLIGHT=0` 关闭 |
| ② | **准备阶段错误先收集不就地抛**（`prep_error`）：数据集构建 / 索引 / 解码失败时，先记下来，走到①的集合点由所有 rank 一起决定 | 失败 rank 不再脱离集合点 ⇒ 不会出现「它等广播、别人等 all-gather」 | 无（始终生效） |
| ③ | **非 0 rank 失败绝不静默**：异常 + rank + 完整栈**立刻打到 stderr**；`AL_EVAL_FAIL_FAST_NONZERO=1` 且已进集合通信区时**立刻抛**（不再等广播） | 排在第二的失败成因（推理中途 OOM / HIP / 解码）也有完整证据；torchrun 立刻收掉作业，不必等 600 s | `AL_EVAL_FAIL_FAST_NONZERO=1`（默认 0 = 旧语义） |
| ④ | **每 rank 阶段面包屑 + 停滞告警** `_EvalPhaseTracker`：`进入评测 / unshard / 本地准备 / 一致性预检 / 推理 i/N / 聚合 / 广播` 每个阶段各打一行（**所有 rank**，以前非 0 rank 一行都没有）；某阶段停留超过 `AL_EVAL_STALL_SEC`（默认 120 s）每 120 s 打一条 WARNING | 「12 分钟完全静默」直接变成日志里的「rank3 仍停在『推理 2/6』已 137 s」——不需要 py-spy、不需要复跑 | `AL_EVAL_PHASE_LOG=0` 关闭 |
| ⑤ | 顺手修掉一个**诊断日志自杀** bug：`_chunk_dumped` 那段把行号 `25/49` 写死（按 `chunk_size=50`），`chunk_size<50` 时 `IndexError: index 25 is out of bounds` 会把评测搞挂 | CPU 复现（小 chunk）与改配置时不再被日志打断 | 无 |

### 第 2 轮（本次）新增

| # | 机制 | 作用 | 开关 |
|---|---|---|---|
| ⑥ | **窗口前阶段 `_eval_prelude`**：本地准备 → **预检 A**（本进程第一个集合通信）→ **对称预热** → **预检 B** → 才开真实 `unshard()` 窗口。`validate()` 与 `evaluate_ids()` 两条路径都在窗口之前调它 | 修掉「预检护不到 `unshard()`」这个第一优先级问题；真机验收应看到 `一致性预检通过`，且**任何失败都在 FSDP 集合通信之前**摊开 | `无` |
| ⑦ | **对称预热 `_eval_warmup`**：每 rank 用**同一份输入**各跑 1 次真实推理（`_eval_context()` + `no_grad`，RNG / `_noise_gen` / 参数 / 梯度 / training 标志全部还原），一次/进程 | 把「首次前向的 kernel 编译」从真实窗口里挪到窗口之前，并让各 rank **同时**编译（避免 8 次串行编译 ⇒ 全组干等） | `AL_EVAL_WARMUP=0` |
| ⑧ | **预热计划纳入预检**：报告里多一个 `warmup` 字段，逐 rank 不一致 ⇒ 预检 A 直接停 | 预热窗口本身是集合通信；「一 rank 进、另一 rank 不进」是新的死锁形态，必须在**任何集合通信之前**拦 | 无 |
| ⑨ | **阶段日志换通道**：`sys.stderr`（flush）+ 逐 rank 文件 `<output_dir>/_open_loop_phase/rank<R>.log`，**不再经过 logger**；每个阶段进入时立刻打一行；广播阶段也有面包屑 | 「某个 rank 一行都没有」不再可能是日志机制的锅：要么它的文件里有行，要么它连 `▶ 进入评测` 都没到（= 卡在更早的位置，正是 rank4 的判据） | `AL_EVAL_PHASE_LOG=0` / `AL_EVAL_PHASE_DIR` / `AL_EVAL_STALL_POLL_SEC` |

> ③ 的默认值保持旧语义（吞掉、用 rank0 结果），是为了不改变既有协议行为与既有测试；
> **多卡 FSDP2 排障/正式跑建议显式开 `AL_EVAL_FAIL_FAST_NONZERO=1`**。

---

## 3. 逐级确认步骤（2 → 4 → 8 卡）

### 0）先离线核对上一次崩溃（不用改任何东西）

```bash
LOG=<那次 8 卡的日志>
grep -n "评测失败"            "$LOG" | head -20          # 有没有「非 DistBackendError」的第一条 ⇒ 那就是真因
grep -c  "Denoise"            "$LOG"                     # 每 rank 的推理次数（模型里 print(f"Denoise {count} steps")）
grep -o  "^\[rank[0-9]*\].*Denoise" "$LOG" | sort | uniq -c   # 各 rank 次数是否一致 ⇒ 不一致即「样本集合/顺序分叉」
grep -n  -A6 "Watchdog caught collective operation timeout" "$LOG" | head -60   # OpType/SeqNum：ALLGATHER 还是 BROADCAST
grep -n  "scout_cache\|已有 scout\|Bootstrap Scout cache" "$LOG"   # scout 缓存是否真的命中
```

### 1）CPU 侧先跑体检（本机即可，不需要卡）

```bash
PY=/opt/anaconda3/bin/python
$PY tools/al_eval_multirank_preflight_repro.py --ws 8                    # 健康路径：应 ✅（8 rank 集合通信次数一致）
$PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:skew    # 预检应点名「rank3 与 rank0 的 n_starts 不一致」
$PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:prepare # 8 个 rank 一起快速失败（不是死锁）
$PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:prepare --preflight off  # 旧行为：必须复现死锁
# 第 2 轮新增用例（都在「窗口之前」注入）：
$PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:warmup       # 预热失败 ⇒ 预检 B 点名，全组一起停
$PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:warmup_skip  # 某 rank 不预热 ⇒ 预检 A 点名「warmup 不一致」
$PY tools/al_eval_multirank_preflight_repro.py --ws 8 --inject 3:delay --delay-sec 8  # 窗口前慢一拍（rank4 形态）：不是死锁，停滞告警指名阶段
$PY tools/al_eval_multirank_preflight_repro.py --ws 8 --warmup off            # 关掉预热，流程回旧
```

### 2）2 卡（真机）

```bash
cd <repo>; export AL_EVAL_FAIL_FAST_NONZERO=1 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120
# N_GPU=2 + 小池子（configs/auto_learning/experiment_sweep_2task.yaml，避免 50 任务全扫）
```
判据：① 日志出现 `[open_loop][multirank] ✅ 一致性预检通过`；
② 每个 `[open_loop][phase][rankX/2]` 都有**成对**的阶段行（开始/完成），没有「仍停在」；
③ 每个评测打印 `eval <task>/<split> n_ids=…`；④ 进入 `Step: 1/…` 且无 NCCL 报错。

### 3）4 卡

同上，`N_GPU=4`。判据同上，另外重点看：`n_starts` 是否 4 个 rank 完全一致
（预检表格里逐 rank 打印）、`集合通信调用数（逐 rank）` 若用 CPU 工具跑过应一致。

### 4）8 卡（真机，正式配置）

```bash
export AL_EVAL_FAIL_FAST_NONZERO=1         # 失败立刻抛（默认 0，官方正式跑建议常开）
export AL_EVAL_STALL_SEC=120               # 阶段停滞告警阈值（默认）
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120  # 只用于排障：把 600 s 看门狗缩到 2 分钟
export AL_EVAL_PREFLIGHT=1                 # 默认已开；这是守门（窗口前 A/B 两次）
export AL_EVAL_WARMUP=1                    # 默认已开；关掉用 0（对比旧流程）
export AL_EVAL_PHASE_LOG=1                 # 默认开；阶段日志走 stderr + 逐 rank 文件
export AL_EVAL_PHASE_DIR=$OUT/_open_loop_phase   # 可选：逐 rank 轨迹文件目录（默认 <output_dir>/_open_loop_phase）
export AL_EVAL_BATCH_MODE=serial           # 明确串行（多卡分片本来强制逐条，显式写死避免误配）
# 不要设 AL_EVAL_BATCH_MULTIRANK（批内 OOM 回退是 rank 局部决策 ⇒ 集合通信次数错配 ⇒ 挂死）
```
判据（依次）：① `✅ 一致性预检通过` 出现（**必须在 `评测窗口：unshard()` 之前**）；
② 8 个 rank **各自**都有阶段行（`ls $OUT/_open_loop_phase/` 应有 rank0..7.log，
   且每个文件里第一行是 `▶ 进入评测`，早于 `▶ unshard（取全参数）完成`）；
③ 出现 `✅ 预热完成（rankR/8，1 次真实推理，+Xs…）` **8 条**（每 rank 一条、耗时相近）；
④ 第一个 scout 评测出数（`eval adjust_bottle/val n_ids=2: mse=…`）；
⑤ bootstrap 走完、`Step: 1/…` 出现且 `StepTime` 正常；
⑥ 全程无 `NCCL operations have failed or timed out` / `DistBackendError`。

### 5）若仍然挂：这次的日志一定能定位

```bash
grep -n "\[open_loop\]\[phase\]\[rank" "$LOG" | tail -40   # 每个 rank 最后停在哪个阶段
grep -n "\[open_loop\]\[multirank\]" "$LOG" | head -20     # 预检报错（点名 rank + 字段）/ 非 0 rank 失败留证
grep -n "Watchdog caught collective operation timeout" -A3 "$LOG" | head -40
```
* 若某 rank 停在「本地准备（数据集/索引）」⇒ 它的数据/解码/磁盘问题（看该 rank 的 stderr 栈）；
* 若预检报 `n_starts 不一致` ⇒ 各 rank 的评测回合集合/索引口径分叉（`_episode_ids_file` /
  `_episode_index_map` / `starts`；这也是「按 rank 各自抽样」类 bug 的直接证据）；
* 若某 rank 停在「推理 i/N」而其它 rank 停在别处 ⇒ 推理次数错配（同上）；
* 若**所有** rank 都停在「推理 i/N」同一处 ⇒ 那是集合通信/驱动层（RCCL/HIP）问题，
  需要 `NCCL_DEBUG=INFO` + `TORCH_NCCL_TRACE_BUFFER_SIZE`，并检查 P2P/传输拓扑。

---

## 4. 已知边界（不在本次修复范围）

* 若失败发生在**一次 all-gather 已经发出之后**（例如正是 all-gather 里的 OOM），
  死锁是 FSDP 的固有限制，**任何**协议都救不了；本次只能保证「真因立刻可见」
  （③ + ④）而不是「不死锁」。
* 每个评测多**两次** `all_gather_object`（各几 ms 级）用于预检 A/B；
  想要逐字回到旧行为：`AL_EVAL_PREFLIGHT=0`。
* **预热不是万能的**：若某张卡的首次编译本来就要几分钟，预热只是把这份代价
  从「正式窗口里串行 8 次」变成「窗口前并行 1 次」；`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`
  仍应留足（排障期 120 s 会**加速暴露**而不是根因）。
* 预热每个**进程**一次（编译是一次性成本）；后续换 shape 仍可能触发新的编译，
  属于 PyTorch/Triton 的正常行为。
* 预检交换的是**本 rank 自己的**身份与失败，不做任何猜测；它不能替代
  「各 rank 拿到同一份评测结果」这条既有协议（`_multirank_eval_payload` 仍在）。
