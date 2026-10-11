# AMD 单卡闭环评测使用说明（双阶段：常驻推理服务 + 仿真）

> 面向本机 **ROCm / AMD（8×Radeon PRO W7900D 48G，gfx1100）** 的闭环评测流程。
> 原 `tools/policy_server.sh` + `tools/eval_tasks.sh` 是**面向 AutoDL** 写的（`/data/code`、
> miniconda、`nvidia-smi`、老分叉客户端），本页记录**移植到 AMD 机器后的用法**与实测结果。
>
> 首次跑通：**2026-10-11 08:37**，`shake_bottle` × 5 回合 ⇒ **3/5 = 60.0%**（`demo_clean` / `seen`）。

---

## 0. 一句话

**阶段 1 把模型加载好并常驻（一次 52 秒）；阶段 2 只跑仿真，可以反复换任务/回合，模型不重载。**
比"官方 launcher 每次整套重来"省掉每轮 2–3 分钟的模型加载。

```bash
# 阶段 1：常驻推理服务（只需一次）
HIP_VISIBLE_DEVICES=7 \
CKPT_ROOT=/models/robotwin-persistent/al_runs/train_al_v37 STEP=500 \
PORT=9330 PRECISION=bf16 FAST_LOAD=0 \
  bash tools/policy_server.sh start
bash tools/policy_server.sh status      # 就绪判据；stop 释放显存

# 阶段 2：跑任务（可反复调用；模型不重载）
HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 \
TASKS="shake_bottle" EPISODES=5 CONFIG=demo_clean PORT=9330 STEP=500 \
CLIENT_FLAVOR=official DEVICE_ID=0 INSTRUCTION_TYPE=seen \
  bash tools/eval_tasks.sh
```

> ⚠️ **单卡隔离**：两阶段都要显式给 `HIP_VISIBLE_DEVICES=<某张卡>`（本页示例用最后一张 **7**），
> 否则会默认落到 GPU0、与别人的任务抢卡。实测：设了 `HIP_VISIBLE_DEVICES=7` 后，
> 服务只占 GPU7（12.4 GB），其余卡不受影响（`rocm-smi --showpids` 可核对）。

---

## 1. 相对 AutoDL 版的改动（移植点）

两个脚本顶部新增了一段「**本机适配层**」（`# === 本机适配层（ROCm / cpu1）`）。
**判据是 `/data/miniconda3` 是否存在**：存在（AutoDL）⇒ 完全按原行为；不存在（AMD）⇒ 用下面这套。
任何**已 export 的环境变量仍然优先**，所以随时可覆盖。

| # | AutoDL 原行为 | AMD 适配后 | 为什么 |
|---|---|---|---|
| 1 | `REPO=/data/code/lingbot-vla-v2` | `/workspace/lingbot_vla_2/lingbot-vla-v2` | 路径不同 |
| 2 | `ROBOTWIN_DIR=/data/code/RoboTwin-lingbot` | `/RoboTwin` | 本机 RoboTwin 在根目录 |
| 3 | `QWEN3VL=/data/models/Qwen3-VL-4B-Instruct/…` | `/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer` | 本机权重位置 |
| 4 | `source conda.sh && conda activate X` | **`run_env X`**：有 conda 就 activate，没有就直接用 venv（设 `PATH`/`PYTHONPATH`/`VIRTUAL_ENV`，默认 `/opt/robotwin-env`） | **本机没有 conda**，只有 venv |
| 5 | `gpu_mem()` 用 `nvidia-smi` | 改用 `rocm-smi`（两者都没有时安静返回） | ROCm 无 nvidia-smi |
| 6 | 结果根 `/data/eval_results` | `${EVAL_RESULTS_ROOT:-…}` ⇒ `/models/robotwin-persistent/eval_results` | 必须落 overlay（`/workspace` 仅 98 G） |
| 7 | 日志/PID `/data/tmp/…` | `${ROCM_OVERLAY}/tmp/…` | 同上 |
| 8 | — | **`AITER_USE_SYSTEM_TRITON=1`** | 见 §2 坑 1 |
| 9 | 客户端固定用老分叉 `eval_policy_client_lingbotvla.py` | 新增 **`CLIENT_FLAVOR`**：`official`（默认，用 `/RoboTwin/scripts/eval_policy_xpolicylab.py`）/ `legacy`（原行为） | 见 §2 坑 3 |
| 10 | curobo 路径自愈（必跑） | 无 conda 时**跳过**，改由 `ROBOTWIN_DISABLE_CUROBO=1 ROBOTWIN_EE_PLANNER=mplib` 走 mplib | 本机没装 curobo；官方脚本同款做法 |
| 11 | 无头渲染 | 显式 `PYOPENGL_PLATFORM=egl` | 服务器无 X11 |

> 原文件已备份为 `tools/policy_server.sh.orig_autodl`、`tools/eval_tasks.sh.orig_autodl`；
> 想完全回到 AutoDL 行为：删掉适配层，或 `cp` 回备份。

---

## 2. 两个必须知道的坑（都实测踩过）

### 坑 1：`aiter gluon kernels require triton>=3.6.0, found 3.5.1+rocm…`（服务起不来）

```
File "/opt/aiter/aiter/ops/triton/gluon/__init__.py", line 23, in <module>
RuntimeError: aiter gluon kernels require triton>=3.6.0, found 3.5.1+rocm7.2.1.gita272dfa8
```
链路：`flash_attn` → `aiter` → gluon 的 triton 版本断言。
**修法**：`AITER_USE_SYSTEM_TRITON=1`（aiter 源码里"设了该变量就只警告不抛错"）。
训练侧靠 `al_launch.py --env AITER_USE_SYSTEM_TRITON=1` 绕开；**推理侧原来没设** ⇒ 已写进适配层。

### 坑 2：`Fast model init produced …inv_freq as torch.float32; expected torch.bfloat16`

```
RuntimeError: Fast model init produced model.qwenvl_with_expert.qwenvl.model.visual.rotary_pos_emb.inv_freq
as torch.float32; expected torch.bfloat16. Disable LINGBOT_DEPLOY_FAST_LOAD for this model version.
```
`rotary_pos_emb.inv_freq` 是**非权重 buffer**（数学常数，本就 fp32），而 fast 加载要求**全模型 bf16**。
**修法**：`FAST_LOAD=0`（走 legacy）。代价：加载 52.8 s（`legacy CPU model init 47.18s` + `weights 1.06s`），
换来的是**可用**。（`FAST_LOAD=1` 只在全 bf16 且无 fp32 buffer 的模型版本上可用。）

---

## 3. 实测结果（2026-10-11）

### 阶段 1（服务）

```
✅ 服务就绪：端口 9330 已监听（耗时 61s）
[deploy-load] legacy CPU model init seconds=47.18
[deploy-load] legacy weights seconds=1.06
[deploy-load] GPU ready elapsed=52.77s allocated=11.96GiB
INFO:websockets.server:server listening on 0.0.0.0:9330
```
* 显存：**11.96 GiB**（bf16，48 G 卡很宽裕 ⇒ 理论上单卡可同时驻留 2–3 个实例）
* 隔离验证：`HIP_VISIBLE_DEVICES=7` ⇒ 只占 GPU7；`rocm-smi --showpids` 显示其余卡上是**别人的进程**

### 阶段 2（`shake_bottle` × 5 回合，`demo_clean`，`seen`）

```
Success rate: 2/2 => 100.0%   (seed 100001)
Success rate: 2/3 =>  66.7%   (seed 100003)
Success rate: 3/4 =>  75.0%   (seed 100004)
Final success rate: 3/5 = 60.0%
```
* 服务端日志出现 `connection open` + 推理调用 ⇒ **模型→仿真链路真实打通**
* `right/left arm planning failed (IK Failed! Cannot find valid solution.)` 是 RoboTwin
  仿真侧**已知非致命提示**（会换 seed 重试），不影响评测继续

---

## 4. 参数速查

### 阶段 1：`tools/policy_server.sh`

| 变量 | 默认（AMD 适配后） | 说明 |
|---|---|---|
| `CKPT_ROOT` | `/data/outputs/single/$TASK`（基本都要显式给） | 训练输出目录；脚本取 `<CKPT_ROOT>/checkpoints/global_step_<STEP>/hf_ckpt` |
| `STEP` | `500` | 评哪一步 |
| `PORT` | `9330` | 端口；`status`/`stop` 也用同一个值 |
| `PRECISION` | **`bf16`**（适配层改的；原版 fp32） | bf16 训练的 HF 导出是 bf16，fp32 服务不了 |
| `FAST_LOAD` | 与精度绑定（bf16⇒1）；**本机要显式 `0`** | 见坑 2 |
| `USE_COMPILE` | `false` | 服务端 torch.compile；本机历史上编译有坑，建议保持 false |
| `QWEN3VL` | `/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer` | VLM backbone 目录 |
| `WAIT` | `420` | 等端口就绪的最长秒数；`0`=立刻返回 |
| 子命令 | `start` / `status` / `stop` | `stop` 按**进程组**结束（含子进程） |

### 阶段 2：`tools/eval_tasks.sh`

| 变量 | 默认（AMD 适配后） | 说明 |
|---|---|---|
| `TASKS` | 必填，逗号分隔 | 例 `TASKS="place_dual_shoes,shake_bottle"` |
| `EPISODES` | `5` | 每任务回合数 |
| `CONFIG` | `demo_clean` | `demo_clean` / `demo_randomized` |
| `PORT` | `9330` | 必须与阶段 1 一致 |
| `CLIENT_FLAVOR` | **`official`** | `official`=官方分叉（`/RoboTwin`）；`legacy`=老分叉（需 `policy/*/deploy_policy.yml`，本机没有） |
| `DEVICE_ID` | `0` | **进程可见设备序号**（配合 `HIP_VISIBLE_DEVICES=<物理卡>` ⇒ 通常写 0） |
| `INSTRUCTION_TYPE` | **`seen`** | ⚠️ `unseen` 会让成功率断崖式掉（历史"闭环 0/5"就是跑了 unseen 口径） |
| `SEED` / `POLICY_NAME` | `0` / `LingBot-VLA-v2` | 政策名要与 `--protocol lingbot_vla_v2` 匹配 |
| `EVAL_RESULTS_ROOT` | `/models/robotwin-persistent/eval_results` | 结果根（overlay） |
| `TAG` | `step${STEP}_${CONFIG#demo_}` | 结果子目录名 |
| `DRY_RUN` | `0` | `1`=只打印计划（会先检查端口是否已有服务） |

---

## 5. 结果文件在哪

```
$EVAL_RESULTS_ROOT/closed_loop/<TAG>/server_step<STEP>_<cfg>_<时间戳>/
├── eval_logs/<task>.log          # 客户端原始输出（含每回合 Success rate）
├── episode_info/<task>.jsonl     # 官方客户端的逐回合信息
├── eval_results/<task>/_result.txt   # 由本脚本从日志反写（末行=成功率）
└── stats.txt                     # 汇总表
```
⚠️ **已知小问题**：官方客户端自己还会另写一份
`<eval_workdir>/eval_result/<task>/ACT/<cfg>/default/<ts>/_result.txt`（相对路径），
与脚本汇总读的 `eval_results/<task>/_result.txt` **不是同一个**——
本适配版已**从日志抽取 `Final success rate: N/M` 反写**到脚本预期位置，因此汇总可用；
但若日志格式变化，请核对 `stats.txt` 与客户端日志是否一致。

---

## 6. 常见问题

| 现象 | 原因 / 处置 |
|---|---|
| 客户端 1 秒即退出，日志 `FileNotFoundError: policy/ACT/deploy_policy.yml` | 用了 `legacy` 客户端而本机没有该 yml ⇒ 用 `CLIENT_FLAVOR=official`（默认） |
| 阶段 2 报"端口 9330 没有服务在监听" | 阶段 1 没起、或已 `stop`、或端口不一致 ⇒ 先 `bash tools/policy_server.sh status` |
| 服务启动后立刻退出 | 看 `$ROCM_OVERLAY/tmp/policy_server_<port>.log`；优先怀疑坑 1（aiter）/坑 2（FAST_LOAD） |
| 显存被占满 / 抢卡 | 两阶段都要显式 `HIP_VISIBLE_DEVICES=<卡号>`；`rocm-smi --showpids` 看是谁在占 |
| 想换 checkpoint 评 | 改 `CKPT_ROOT`/`STEP` 重启阶段 1（必须重启：服务把权重加载在内存里） |
| 训练要开始了 | 先 `bash tools/policy_server.sh stop`（服务常驻占 ~12 GB 显存） |

---

## 7. 与官方 `run_clean_benchmark.py` 的关系

| | 本双阶段脚本 | 官方 `run_clean_benchmark.py` |
|---|---|---|
| 服务生命周期 | **常驻**（模型只加载一次，可反复换任务） | 每次整套起停（每轮重载模型） |
| 任务选择 | **任意子集**（`TASKS=…`） | 固定读 `env_cfg/eval/all_tasks.yml` 的 50 个任务（强校验） |
| 多卡 | **单卡**（多卡需手动起多个服务 + 分片） | **原生多卡**（`--gpu-count 4/8`，每卡一个 slot） |
| 适用 | 交互式复测、只关心少数任务 | 全量 50 任务跑分 |

⇒ **要"少数任务 + 反复迭代"用本流程；要"50 任务全量跑分"用官方脚本。**
