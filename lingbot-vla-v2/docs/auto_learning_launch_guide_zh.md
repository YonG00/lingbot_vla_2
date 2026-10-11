# Auto Learning 启动程序使用说明（`experiment/robotwin/al_launch.py`）

面向使用者。看完这一页就能自己起 50 任务的 Auto Learning 训练，不用再手搓 7 条
`CUDA_VISIBLE_DEVICES=… N_GPU=1 MAX_STEPS=1 …` 的 worker 命令。

---

## 0. 一句话：它解决什么问题

**缓存齐了就直接开训（秒级）；缺哪几个就只并行扫哪几个；显式要求才全量重扫；覆盖不完整时
不启动训练，并以非零退出码点名缺失任务。**

它把手工做的三件事固化成一条命令：

1. 读**显式指定的缓存文件**（scout 缓存；`--scout-cache-file`），按其中的记录检查覆盖；
2. **只并行补齐缺失的任务**，把结果写回同一份文件；
3. 覆盖确实完整（以该文件内的记录数为准）之后，才后台启动正式训练。

> 🔴 **2026-10-10 起缓存不用指纹寻址**：以前是"算指纹 → 用指纹命名缓存目录"，改代码就会
> 换目录 ⇒ 缓存全失效（实测一次改动导致 7 个任务重扫、单次扫描 208 秒）。现在改成
> **显式文件 + 模型名**：文件路径由 `--scout-cache-file` / `--hardness-cache-file` 指定，
> 缓存有效性只由 **`--model-name`** 决定（换模型才失效，改代码不失效）。
> 指纹仍可算，但**默认不计算**（`--compute-fingerprint` 才开）——计算它要读 11.9 GiB 权重分片。

---

## 0.1 跑通训练的最小启动清单（2026-10-10 实测）

这套组合**已实测连续跑过 34+ 步**（`al_v34`，loss 正常下降，零 OOM/零崩溃）：

```bash
# 0) 先确认没有残留训练进程（否则 launcher 会以退出码 2 拒绝启动，这是自锁保护）
ps -eo args | grep "[t]rain_lingbotvla.py /workspace"     # 必须为空

# 1) 起训练（**标准命令**见 §0.2；2026-10-11 起 8 卡全开 —— GPU[4] 实测已恢复正常）
/opt/robotwin-env/bin/python -u experiment/robotwin/al_launch.py \
  --run-name al_v36 --steps 5000 --micro 5 --gas 1 \
  --gpus 0,1,2,3,4,5,6,7 \
  --worker-out-root     /models/robotwin-persistent/al_runs \
  --hardness-cache-file /workspace/al/hardness_cache/hardness.json \
  --scout-cache-file    /workspace/al/scout_cache/scout.json \
  --model-name          robbyant_lingbot-vla-v2-6b-bf16 \
  --triton-cache        /models/robotwin-persistent/al_cache/triton \
  --torchinductor-cache /models/robotwin-persistent/al_cache/torchinductor \
  --env AITER_USE_SYSTEM_TRITON=1 \
  --env AL_HARDNESS_SHARD=1 \
  --workers 1
```

**四条前置条件（缺一不可）**：

| # | 条件 | 为什么 |
|---|---|---|
| 1 | `configs/rocm/robotwin_official_paths_rocm.yaml` 里 **`train.use_compile: false`** | 开编译在本机三连坑：flex 反向 kernel 崩溃、多 rank 共享缓存死锁、**图断裂导致 7 rank 集合通信错序 ⇒ 首步永久卡死**（GPU 0%、CPU 空转、日志静默）。官方 recipe 本来也是 `false` |
| 2 | `--model-name` 与缓存文件里的 `model` 字段**一致** | 不一致 ⇒ 缓存全部判为未命中（会真扫一遍） |
| 3 | 卡：`--gpus 0,1,2,3,4,5,6,7`（8 卡） | 2026-10-11 复测：**GPU[4] 已恢复正常**（矩阵乘 0.13 s、显存 47.7 GiB 可用）。若某卡异常 ⇒ 从 `--gpus` 里去掉它 |
| 4 | 运行前无残留训练进程 | launcher 自锁（退出码 2），避免两个 run 抢卡 |
| 5 | **`AITER_USE_SYSTEM_TRITON=1`**（必须 `--env` 显式传，**不能只 export**） | 镜像 triton 3.5.1 < aiter 要求的 3.6.0 ⇒ 不设则 `aiter` 导入失败 ⇒ 模型模块注册失败（`Unrecognized configuration class`）。⚠️ launcher 给 worker 的环境**只透传 `AL_` 前缀**（`al_launch.py` 的 env 注入），所以父进程 export 到不了训练进程 ⇒ **必须 `--env AITER_USE_SYSTEM_TRITON=1`** |
| 6 | **`$PHASES/datasets.txt` 必须是「单行聚合数据集」** | 训练侧 `--data.train_path` 指向它；而 `MultiVLADataset` + `resolver` **只支持单条目数据集**（回合号是聚合数据集的全局编号 0–2499）。给它 50 行分任务清单会报 `下钻不到 hf_dataset` |

**实测基线（可用来判断"是否正常"）**：

| 指标 | 实测值 | 说明 |
|---|---|---|
| 首步耗时 | **~150 s** | 含模型加载 + 首次前向/反向 + FSDP 初始化，**正常** |
| 稳态 s/it | **11.0–11.9 s** | micro 5 / gas 1 / GBS 35 |
| 5000 步外推 | **约 15 小时** | 按 11 s/it |
| 峰值显存 | **48,956 MiB / 每卡 49,121 MiB**（99.7%） | 余量很薄但稳定；**降 micro 省不下显存**（大头是参数+优化器状态+对齐模型等与 batch 无关的占用） |
| scout 缓存 | 50/50 命中，启动仅 ~1.5 分钟 | — |
| hardness 缓存 | 命中时 **4.2–4.7 s**（首扫 208.5 s，**约 44×**） | 日志出现 `本次新增 0` 即证明未重算 |
| 指纹计算 | 默认跳过（省 12 GiB 权重哈希 / 每轮 1–3 分钟） | 加 `--compute-fingerprint` 才计算 |
| scout 首扫（50 任务 / 8 分片） | **4 分 44 秒** | 每任务 2 条 val 回合；缓存命中则秒级 |
| hardness 单任务（**单卡**） | **1067 s / 184 样本**（≈0.17 样本/s） | 历史"1.25 样本/s"是 **7 卡合计**，单卡本来就是 ~0.18 样本/s |
| hardness 单任务（**8 卡 rank 分片**） | 应约 **1/8 时间**（`ids[rank::8]`） | 需同时满足：`--workers 1`（1 片 ⇒ world_size=8）+ `--env AL_HARDNESS_SHARD=1` |

**哪些日志行说明"一切正常"**：

```text
[hardness_cache] 启用：file=…/hardness.json  model=robbyant_lingbot-vla-v2-6b-bf16   ← 缓存已挂载
[hardness] 扫描完成：task=… 259/259 用时 4.2s | loss 均值 …                          ← 命中（秒级）
=====Using SDPA Attn=====                        ← VLM 主干走 SDPA（7 处；动作专家另有 21 处 Eager，正常）
Step: 1/5000 … Loss 0.4367                       ← 首步成功（此前的卡点就在这一步）
```

---



## 0.2 标准启动命令（2026-10-11 定型，**统一用这一条**）

```bash
cd /workspace/lingbot_vla_2/lingbot-vla-v2
/opt/robotwin-env/bin/python -u experiment/robotwin/al_launch.py \
  --run-name al_v36 --steps 5000 --micro 5 --gas 1 \
  --gpus 0,1,2,3,4,5,6,7 \
  --worker-out-root     /models/robotwin-persistent/al_runs \
  --triton-cache        /models/robotwin-persistent/al_cache/triton \
  --torchinductor-cache /models/robotwin-persistent/al_cache/torchinductor \
  --hardness-cache-file /workspace/al/hardness_cache/hardness.json \
  --scout-cache-file    /workspace/al/scout_cache/scout.json \
  --model-name          robbyant_lingbot-vla-v2-6b-bf16 \
  --env PRUNE=1 \
  --env AITER_USE_SYSTEM_TRITON=1 \
  --env AL_HARDNESS_SHARD=1 \
  --workers 1 --no-tb
```

### 每个参数为什么这么写（缺一个就会以某个退出码失败）

| 参数 | 作用 / 不写的后果 |
|---|---|
| `--gpus 0..7` | 同时导出 `CUDA_/HIP_VISIBLE_DEVICES`；写错卡号 ⇒ 用错卡或起不来 |
| `--worker-out-root` | **必须指到 overlay**（`/models/robotwin-persistent/…`）。默认在 `/workspace`（仅 98 G）⇒ 日志/checkpoint 会撑爆它 |
| `--triton-cache` / `--torchinductor-cache` | 同上；且多 rank 共享同一编译缓存会死锁（本配方每 rank 独立子目录） |
| `--env AITER_USE_SYSTEM_TRITON=1` | **重建后必加**。父进程 export 无效（worker 环境只透传 `AL_` 前缀）⇒ 不写则 aiter 导入失败、注册表为空、`Unrecognized configuration class` |
| `--env AL_HARDNESS_SHARD=1` | 让 hardness 打分按 rank 切样本（`ids[rank::world_size]`）⇒ 单任务约 8× 加速 |
| `--workers 1` | **1 片 = 8 卡同进程**（`world_size=8`）。⚠️ 与 `AL_HARDNESS_SHARD` 是**配套**的：分片数 >1 时每片 `N_GPU=1`，`world_size==1` 会让分片自动退回单卡全量 |
| `--no-tb` / `--env PRUNE=1` | 关 TensorBoard（本机不启）、开权重裁剪 |
| 缓存两件 | 显式文件；有效性只看 `--model-name`（换模型才失效，改代码不失效） |
| 不写 `--no-cache` | **有缓存就复用、缺就补扫**；只有确实要全量重扫才加 `--no-cache` |

### 三条规则（launcher 的缓存语义）

1. **缓存覆盖完整 ⇒ 直接启动训练**（本次实测：`运行模式：reuse` ⇒ `退出码 0`，秒级）；
2. **有缺失 ⇒ 只并行补扫缺失部分**（`运行模式：incremental`）；
3. **`--no-cache` ⇒ 全量重扫**（`运行模式：full-scan`）。

> 🔴 **扫描决策不看指纹**（2026-10-11 修复：此前 `fingerprint is None` 会把扫描范围清空，
> 导致 `--no-cache` 也失效、以退出码 3 拒绝启动）。指纹只用于记录/复现。

### 退出码速查

| 码 | 含义 |
|---|---|
| 0 | 成功（dry-run 完成，或训练已后台启动） |
| 2 | 前置检查失败（参数/路径/AL 配置/机器被占用/缺派生产物） |
| 3 | 扫描未完成（缓存覆盖不全） |
| 4 | worker 未就绪或提前退出（看各片 worker 日志的 Traceback） |
| 5 / 6 / 7 | 扫描超时 / 内部错误 / 训练启动后立刻退出 |

### 首次/重建后的三条派生产物（**先查有没有现成的，再决定是否重算**）

| 产物 | 生成方式 | 备注 |
|---|---|---|
| `$SPLIT_DIR/manifest.json` | `tools/task_split.py --task all --dataset <聚合数据集> --out <split>` | 50 任务 + train/val 划分 |
| `$SPLIT_DIR/task_baseline.json` | `python -m lingbotvla.auto_learning.tools.compute_task_baseline --manifest … --config …`（**CPU，~90 s/任务**） | ⚠️ **重建后先全盘找现成的**（历史机器上常有），指纹必须与阈值表一致 |
| `$PHASES/datasets.txt` + 阈值表 `pass_thresholds_*.json` | 前者手写**单行**聚合数据集；后者 `build_gmean_thresholds.py` 生成 | 阈值表与 baseline **必须同指纹** |

> 🔴 **以上产物已随仓库提供「离线复现包」**：`refs/al_artifacts/`（109 个文件 / 约 216 KB，
> 含 `SHA256SUMS`、`MANIFEST_sha256.md`、`restore.sh`）。重建后**先恢复，不要重算**：
>
> ```bash
> bash refs/al_artifacts/restore.sh --local            # 已在训练机上
> bash refs/al_artifacts/restore.sh root@<host> <port> <私钥>   # 从开发机推
> ```
>
> 包内含**不可再生资产** `ref_per_traj.jsonl`（50k 参考模型逐轨迹 MSE，阈值表的唯一输入）
> 与 `task_baseline.json`（重算需 ~75 分钟）。详见 `refs/al_artifacts/README.md`。

## 1. 快速开始（三条命令）

```bash
cd /workspace/lingbot_vla_2/lingbot-vla-v2

# ① 直接训练（有缓存）：查覆盖 → 齐了就直接起训练；缺了就只补缺失的那几个
/opt/robotwin-env/bin/python experiment/robotwin/al_launch.py

# ② 忽略现有缓存：全量并行扫描 50 个任务，扫完再启动训练
#    （旧缓存文件会被改名备份成 scout.json.bak-<时间戳>，不删除）
/opt/robotwin-env/bin/python experiment/robotwin/al_launch.py --no-cache

# ③ 预演：只打印步骤计划与将用到的环境变量，不启动任何进程（无 GPU 也能跑）
/opt/robotwin-env/bin/python experiment/robotwin/al_launch.py --dry-run
```

**注意（默认 AL 配置）**：`--eval-config` 默认值是 `configs/auto_learning/al_eval2.yaml`。
如果这份配置还不存在，上面三条命令会**直接告诉你缺它**，并列出
`configs/auto_learning/` 下现成的配置；此时给三条命令都加上 `--eval-config` 即可，例如：

```bash
/opt/robotwin-env/bin/python experiment/robotwin/al_launch.py \
  --eval-config configs/auto_learning/al_50task_gmean100_rocm.yaml
```

能看到什么（真实输出片段）：

```text
[plan] 运行模式：incremental
[cache] 该缓存文件内 4 条记录；覆盖 4/6 个任务
[plan] 待扫 2 个任务 ⇒ 2 片：片0=1, 片1=1
[plan] 分片并集校验：通过（无重复、无遗漏，合计 2 = 全集 2）
[ready] T+30s：进程存活 2/2；已就绪 2/2（就绪数与各分片任务数相符）
[progress] T+3m20s | 片0(gpu0) 1/1 | 片1(gpu1) 1/1 | 合计 50/50（缓存文件内）
范围覆盖      : 2/2
全集覆盖      : 50/50（该缓存文件内 50 条记录）
缺失清单      : 无
结论：覆盖完整（50/50）⇒ 启动训练。
```

覆盖不完整时（**绝不打印假的 50/50**）：

```text
全集覆盖      : 44/50（该缓存文件内 44 条记录）
缺失任务（6 个）：move_can_pot, place_a2b_left, ...
缺失任务所属分片：move_can_pot→片3, place_a2b_left→片5, ...
结论：覆盖不完整 ⇒ 不启动训练，请按上面的缺失清单补齐后重跑（本工具不会伪造覆盖数）。
[done] 退出码 3
```

---

## 2. 参数表

### 2.1 最常用的几个

| 参数 | 默认值 | 含义 | 什么时候需要改 |
|---|---|---|---|
| `--no-cache` | 关 | 忽略现有缓存，**全量并行扫描**后再训练（旧缓存文件改名备份为 `.bak-<时间戳>`，不删除） | 想强制重扫时（例如怀疑缓存内容不对） |
| `--dry-run` | 关 | 只打印步骤计划与环境变量，不启动任何进程 | 上机器前先核对路径/配置；排障 |
| `--json` | 关 | stdout 只输出 JSON（人话日志走日志文件与 stderr） | 被别的脚本调用、要机器可读结果时 |
| `--selfcheck` | 关 | 与仓库 `scout_cache.py` 逐项对拍缓存实现 | 每次升级本脚本/改动缓存相关代码后自检 |
| `--gpus` | `0,1,2,3,5,6,7` | 可用卡（逗号分隔）；会同时显式导出 `CUDA_VISIBLE_DEVICES` 与 `HIP_VISIBLE_DEVICES` | 换机器、换可用卡时。**4 号卡已挂死**，除非确认换机，否则别把它加回来 |
| `--eval-config` | `configs/auto_learning/al_eval2.yaml` | AL 配置（相对仓库根） | 默认文件不存在，或想换配方（如 `al_50task_gmean100_rocm.yaml`）时 |
| `--steps` | `200` | 训练 `MAX_STEPS` | 换训练预算时（正式长跑一般给很大的值） |
| `--micro` / `--gas` | 拉起训练时的实测值 **`5` / `1`**（GBS 35） | 传给启动脚本的 `MICRO` / `GAS`；脚本自算 `GBS = MICRO*GAS*N_GPU` | 显存不够或要改全局批大小时。**AL 要求 `micro*gas == AL 配置的 batch_size`**，否则启动即报错。⚠️ 本机把 micro 从 12 一路降到 5 的实测经验：**降 micro 几乎省不下显存**（峰值 49.1→48.9 GiB），省下的主要是每步时间（14.1→11.0 s/it） |
| `--scout-cache-file` | `/workspace/al/scout_cache/scout.json` | **scout 缓存文件**（单文件，含 `model` 字段） | 缓存换盘/换位置时 |
| `--hardness-cache-file` | `/workspace/al/hardness_cache/hardness.json` | **hardness（样本难度）缓存文件** | 同上 |
| `--model-name` | 取权重目录名 | **缓存有效性判据**：与缓存文件里的 `model` 逐字比较，不一致即全部未命中 | 换模型时；或想手工指定一个稳定标识时 |
| `--cache-root` | 空（兼容保留） | 旧参数：等价于 `<目录>/scout.json` | 只有沿用旧命令时才需要 |
| `--fingerprint` | 空 | (已弃用) 显式 64 位指纹；**不再用于缓存寻址**，仅作记录 | 需要复现某次运行的记录时 |
| `--compute-fingerprint` | 关 | 计算并打印 provenance 指纹 | 默认关闭：计算要读 11.9 GiB 权重分片（每轮多花 1–3 分钟），而它已不参与寻址 |

### 2.2 路径类

| 参数 | 默认值 | 什么时候需要改 |
|---|---|---|
| `--repo` | 本脚本的上一级目录 | 一般不用改（脚本自定位仓库根） |
| `--python` | `/opt/robotwin-env/bin/python` | 训练环境解释器换位置时。**建议直接用这个解释器启动本脚本** |
| `--launch-script` | `experiment/robotwin/al_50task_bf16.sh` | 换训练入口脚本时 |
| `--train-config` | `configs/rocm/robotwin_official_paths_rocm.yaml` | 换机器路径配置（`CONFIG` 变量）时 |
| `--checkpoint` | `/workspace/models/robbyant_lingbot-vla-v2-6b-bf16` | 换初始权重时。它同时是 `MODEL_PATH` 与 `AL_SCOUT_CACHE_CHECKPOINT` |
| `--manifest` | `<split-dir>/manifest.json` | 任务划分换目录时 |
| `--baseline` | `<split-dir>/task_baseline.json` | 同上 |
| `--norm` | `<repo>/assets/norm_stats/robotwin_competition_clean.json` | 换 norm 统计文件时 |
| `--split-dir` | `/workspace/al/task_splits_50` | 任务划分目录换位置时 |
| `--phases` | `/workspace/al/phases_al` | `PHASES/datasets.txt` 的位置 |
| `--thresholds` | 取 AL 配置的 `pass_thresholds_file` | 只想单独指定阈值表时 |
| `--tasks` | 空（取 manifest 的 `tasks`） | 只想跑一个子集（例如 4 个任务的冒烟）时，逗号分隔 |
| `--task-source` | `manifest` | 想按 `/workspace/lerobot/*_joint_v30` 目录名取任务全集时用 `lerobot` |
| `--lerobot-root` | `/workspace/lerobot` | `--task-source lerobot` 时的数据根 |
| `--qwen3vl` | `/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer` | 换 VLM 权重目录时（启动脚本会自检它存在） |
| `--train-out` | `<worker-out-root>/train_<配置名>_<时间戳>` | 想固定输出目录（例如续训/接续排查）时 |
| `--run-name` | `<配置名>_<UTC 时间戳>` | 想给本次 run 起固定名字时 |
| `--worker-out-root` | `/workspace/al/al_launch_runs` | 扫描 worker 日志与分片配置的落点 |
| `--shard-config-dir` | 空 | 用**现成的**分片 AL 配置目录（按 `al_shard<i>.yaml` 命名），不让本脚本生成 |

### 2.3 运行环境类

| 参数 | 默认值 | 什么时候需要改 |
|---|---|---|
| `--tmpdir` | `/models/robotwin-persistent/tmp/al_launch` | **绝不能是 `/tmp`**（那里只有 4 GB tmpfs）；换盘/换 overlay 时改 |
| `--triton-cache` | `/workspace/runtime/triton` | 编译缓存换盘时（多 worker 共用，第一个编译、其余命中） |
| `--torchinductor-cache` | `/workspace/runtime/torchinductor` | 同上 |
| `--dtype` | `bfloat16` | 权重精度变了才改（必须与 checkpoint 参数 dtype 一致，否则运行时 `Scout cache dtype mismatch`） |
| `--scout-trajs` | 取 AL 配置的 `global_scout_val_trajs`（一般 2） | 只想改 scout 回合数时（回合集合变了 ⇒ 旧的 scout 记录**不会命中**，会重扫） |
| `--image-augment` | 关（= `false`） | 一般别开：启动脚本固定 `--data.image_augment false`；Auto Learning 要求 `false`（开了会 fail-fast） |
| `--tb-port` / `--no-tb` | `6006` / 关 | TensorBoard 端口冲突时改端口，或直接 `--no-tb` 关掉 |
| `--workers` | `min(可用卡数, 待扫任务数)` | 想少占几张卡时（例如留一张卡给别人） |
| `--worker-max-steps` | `1` | 一般别改（扫完 bootstrap 就退出，1 步够） |
| `--worker-checkpoint` | 关 | 默认 worker 用 `SMOKE_NO_CHECKPOINT=1`（**避免每个 worker 收尾写一份 24 GB DCP**）；确实要 worker 存盘时才开 |
| `--min-free-gb` | `20` | 磁盘更紧或更宽时改阈值 |

### 2.4 时序与安全类

| 参数 | 默认值 | 什么时候需要改 |
|---|---|---|
| `--ready-wait` | `30` | 首轮就绪检查的时间点（秒）。题目要求的「启动 30 秒后检查进程数与已就绪任务数」就是这个值 |
| `--ready-timeout` | `900` | 等待「已就绪：N 任务」的上限；机器慢（首次编译）可加大 |
| `--poll-interval` | `20` | 进度打印间隔 |
| `--scan-timeout` | `1800` | 整轮扫描上限；超过则精确终止仍在跑的 worker |
| `--exit-grace` | `90` | 覆盖已完整后，仍等 worker 自然退出的宽限（worker 可能在做那 1 个训练步） |
| `--retry-rounds` | `0` | 默认「缺了就报缺失并非零退出」；想自动补一轮就设 `1` |
| `--allow-busy` | 关 | 默认检测到别的训练/启动器在跑就拒绝启动（避免抢卡）；确认可以同跑时才加 |
| `--log-file` | `<worker-out-root>/logs/al_launch_<时间戳>.log` | 想固定本工具日志路径时 |

> 没有 `--rescan-every` 这个参数：重扫节奏属于 **AL 配置项**（`rescan_every_n_task_switches`），
> 见 §6。

---

## 3. 它是怎么工作的

### 3.1 缓存是一份**显式指定的文件**（2026-10-10 起）

```text
/workspace/al/scout_cache/scout.json          ← 由 --scout-cache-file 指定（旧 --cache-root 仍兼容）
{
  "version": 2,
  "model": "robbyant_lingbot-vla-v2-6b-bf16", ← **缓存有效性只由它决定**
  "records": {
    "<scout_key>": {"task": …, "episode_ids": […], "metrics": {…}}
  }
}
```

* 记录键 `scout_key(task, episode_ids) = sha256(json([task, ids]))`；值含
  `task / episode_ids / metrics`。
* **模型名不符 ⇒ 整份缓存判为未命中**（`last_miss_reason='model_mismatch'`，日志会点名），
  会真扫一遍并重建；**改代码不会**让缓存失效（这是本次改造的核心目的）。
* 多进程安全：只有 rank0 写；写入用「先重读再合并 + 原子替换」，不会覆盖别人的记录。
* 文件不存在 ⇒ 等效「空缓存 + 全量重扫并写进去」。
* 单个文件上限 64 MiB、单条记录 1 MB；坏 JSON / schema 不符 ⇒ 视为未命中并重建（不抛异常）。

### 3.2 缓存有效性由什么决定

| 决定项 | 具体内容 |
|---|---|
| **模型身份** | `--model-name`（缺省取权重目录名）。与文件里的 `model` 字段**逐字比较**，不一致即全部未命中 |
| schema 版本 | 文件里的 `version` 必须等于 `scout_cache.VERSION`（launcher 里有一份**副本常量**，两端必须同步，测试会拦住漂移） |
| 单条记录自洽性 | `metrics.mse == mean(per_traj_mse)`、`nmse == mse/baseline_mse`、`metric_valid`、回合集合与键一致 |

**已废弃/不再参与寻址**：旧的"指纹目录"方式（`AL_SCOUT_CACHE_FINGERPRINT`、按 `*.safetensors`
内容 + 12 个评测链源码 AST 哈希命名目录）。指纹仍可计算（`--compute-fingerprint`），
但**默认跳过**且只作记录用途 —— 它要读 11.9 GiB 权重分片，实测每轮多花 1–3 分钟。


### 3.3 并行扫描为什么比进程内 bootstrap 快

```text
进程内 bootstrap（默认慢路径）
  7 卡 FSDP2 同步扫 50 个任务 ≈ 18 分钟
  任一 rank 掉队 ⇒ 全组一起等（集合通信）⇒ 一个坏卡拖死一整轮

并行扫描（本工具的做法）
  任务全集按 tasks[i::N] 切成 N 片（N = 可用卡数）
  每片起一个独立单卡进程：CUDA_VISIBLE_DEVICES=<单卡> / HIP_VISIBLE_DEVICES=<单卡> / N_GPU=1
                            MAX_STEPS=1 / AL_CFG=<该片的 task_names 分片配置>
                            AL_SCOUT_CACHE_FILE=<scout.json>（各 worker 与正式 run 必然写同一份文件）
  50 个任务 ≈ 5 分钟；进程之间没有集合通信，天然失败隔离
```

* **无跨进程集合通信**：每个 worker 自己持完整模型（单卡放得下 bf16 的 6B），只在最后
  写一条 JSON；一个 worker 慢/挂不会让其他 worker 停。
* **失败隔离**：某个分片挂了，本工具会让你「缺失清单 + 非零退出码」看到，而不是静默把
  整轮拖死；再跑一次只会补那几个缺失任务。
* **编译缓存共用**：`TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` 指向同一个目录，
  第一个 worker 编译，其余命中缓存。
  ⚠️ **但训练本体不能共用**：7 个 rank 共享同一个编译缓存目录时，实测 231 个 Inductor
  编译 worker 会烧 ~8 个 CPU 核却**零产物**、首步永久冻结（GPU 0%、日志静默）。
  ⇒ 训练侧已由 `tasks/vla/train_lingbotvla.py` 的 `_al_per_rank_compile_cache()` 自动加
  `…/rank<N>` 子目录；`AL_SHARED_COMPILE_CACHE=1` 可还原旧行为（仅排障用）。

### 3.4 一次运行里的 15 步（与日志一一对应）

| 步骤 | 日志里会看到 |
|---|---|
| 1 静态校验 | `卡 0,1,2,3,5,6,7（7 张）；steps=…；micro=… gas=… ⇒ GBS=…` |
| 2 路径检查 | `[missing] …`（dry-run 只警告；实跑直接拒绝） |
| 3 AL 配置 | `enabled` / `pass_metric` / `scout_trajs` / 阈值表；`pass_metric != gmean_mse` 直接拒绝 |
| 4 任务全集 | `N 个任务（来源 manifest）；每任务 scout 2 条 val 回合` |
| 5 并发占用检查 | 发现别的 `train_lingbotvla.py` / `al_launch.py` 就拒绝（除非 `--allow-busy`） |
| 6 运行环境 | `TMPDIR=…`；TMPDIR 是 `/tmp` 直接拒绝 |
| 7 缓存挂载 | `[cache] 文件 <scout.json>；该缓存文件内 N 条记录；覆盖 M/50 个任务`；`[fingerprint] 跳过计算`（默认） |
| 8 覆盖检查 | `[cache] … 覆盖 47/50 个任务` ⇒ 决定 `reuse` / `incremental` / `full-scan` |
| 9 分片规划 | `[plan] 待扫 3 个任务 ⇒ 3 片：片0=1, 片1=1, 片2=1` + **并集校验通过** |
| 10 环境变量 | 把 worker 模板与训练要用到的**全部**环境变量逐行打印 |
| 11 启动 worker | `[scan] 片0 GPU0：1 任务；pid=…；日志 …`（每片一行） |
| 12 就绪检查 | `[ready] T+30s：进程存活 7/7；已就绪 7/7` |
| 13 逐分片进度 | `[progress] T+3m20s | 片0(gpu0) 7/7 | … | 合计 47/50（缓存文件内）` |
| 14 收尾统计 | `范围覆盖` / `全集覆盖` / `缺失清单` |
| 15 启动训练 | `日志 / PID / 进程数 / 输出 / 配置` |

### 3.5 「覆盖」的判定口径

一个任务算「已覆盖」，当且仅当缓存文件里存在它的记录，且该记录会被
`BootstrapScoutCache.load()` **真的当成命中**返回：schema 版本、**模型名**、任务名、回合集合、
逐轨迹 MSE 与 `nmse == mse/baseline` 的自洽关系、`metric_valid=true`、`gmean_mse` 有限，
全部核对通过。

因此「覆盖 50/50」是可以放心直接开训的；反之，坏记录、模型名不符的记录、回合集合不同的记录
都会算作**缺失**，由 worker 真评测补上。

---

## 4. 常见问题

### 4.1 为什么必须显式导出 `CUDA_VISIBLE_DEVICES`？

启动脚本只在**它为空**时才自己推导：`N_GPU=7` 会推导成 `0,1,2,3,4,5,6` —— **把挂死的 4 号卡
带进来**，然后卡在第一次 H2D 拷贝上。本工具因此永远显式导出
`CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7`（以及同样值的 `HIP_VISIBLE_DEVICES`），
worker 则导出单张卡的编号。要换卡请改 `--gpus`，不要靠脚本推导。

### 4.2 为什么 `TMPDIR` 不能是 `/tmp`？

这台机器的 `/tmp` 只有 4 GB tmpfs，数据加载/编译的临时文件会把它撑爆，表现为训练中途
莫名其妙失败。本工具默认 `--tmpdir /models/robotwin-persistent/tmp/al_launch`，
**发现 TMPDIR 落在 `/tmp`（含 `/var/tmp`）会直接拒绝启动**（`--dry-run` 时只给警告，方便你在别的机器上预演）。

### 4.3 缓存什么时候会失效？（改代码**不会**）

**只有两个原因会让 scout 缓存失效**：

| 原因 | 表现 | 处置 |
|---|---|---|
| **换了模型**（`--model-name` 与文件里的 `model` 不一致） | 日志点名 `model_mismatch`，覆盖显示 `0/50` ⇒ 真扫一遍并重建 | 确认模型名写对；若确实换了模型，让它重扫（或把结果写进新文件） |
| 文件坏 / schema 不符 / 版本不符 | 视为未命中并重建（**不抛异常**） | 无需处理，重扫后自动恢复 |

**改代码 / 改注释 / 改 AL 配置都不会让缓存失效**（这是 2026-10-10 改造的核心目的：
此前按指纹寻址，改一个文件就导致 7 个任务重扫、单次扫描 208 秒）。

> ⚠️ 一条真机踩过的坑：launcher 里**复制**了 `scout_cache.py` 的两个常量
> （`VERSION`、`MAX_JSON_BYTES`）。若只改源文件不改副本，会出现"缓存文件明明有效、
> launcher 却报 0/50、以退出码 3 拒绝启动"（实测 `version=2 != 1`）。
> 已有测试 `test_cache_constants_match_scout_cache` 守住，改常量时两边一起改。

### 4.4 如何「只看不动」地检查缓存覆盖情况

```bash
cd /workspace/lingbot_vla_2/lingbot-vla-v2

# ① 直接看缓存文件里有多少条记录、属于哪个模型
python3 -c "
import json; d=json.load(open('/workspace/al/scout_cache/scout.json'))
print('model =', d.get('model'), '| version =', d.get('version'), '| records =', len(d.get('records') or {}))"

# ② 只读检查覆盖：--dry-run 不启动任何进程、不写任何文件（无 GPU 也能跑）
/opt/robotwin-env/bin/python experiment/robotwin/al_launch.py --dry-run \
  --scout-cache-file /workspace/al/scout_cache/scout.json \
  --model-name robbyant_lingbot-vla-v2-6b-bf16
# 输出：
#   [cache] 该缓存文件内 50 条记录；覆盖 50/50 个任务
#   [plan] 运行模式：reuse
#   [plan] 覆盖完整 ⇒ 直接启动训练，不扫描
```

`--dry-run` 不建目录、不写文件、不起进程，在没有 GPU 的机器上也能跑。

### 4.5 `--eval-config` 指向的文件不存在怎么办？

实跑会直接拒绝（退出码 2），并列出 `configs/auto_learning/` 下现成的配置；
`--dry-run` 只给警告，仍然把其余计划和环境变量打印完。选一份现成的即可，例如：

```bash
--eval-config configs/auto_learning/al_50task_gmean100_rocm.yaml
```

### 4.6 扫描和训练能同时跑吗？

不能。两者抢同一批卡。本工具在启动前会扫描 `/proc` 里的 `train_lingbotvla.py` /
`al_launch.py` 进程，发现就**拒绝启动**并打印 PID 与命令行（它不会替你杀进程）。
确实要同跑（例如错开卡）才加 `--allow-busy`。

### 4.7 日志都在哪里？

| 日志 | 位置 |
|---|---|
| 本工具日志 | `<worker-out-root>/logs/al_launch_<时间戳>.log`（同时打到 stdout） |
| 每片 worker | `<worker-out-root>/<run-name>/scan_shard<i>/worker_shard<i>_gpu<g>.log` |
| 训练 | `<worker-out-root>/logs/train_<run-name>.log` |
| AL 事件流 | `<TRAIN_OUT>/auto_learning_events.jsonl` |
| TensorBoard | `<TRAIN_OUT>/runs`（端口 `--tb-port`，默认 6006） |

已知限制：启动脚本内部 `train.sh` 会 `tee log.txt`，7 个 worker 会**混写**
`<repo>/log.txt`。判读进度请用每个分片自己的日志与缓存条目数，不要看 `repo/log.txt`。

---

## 5. 运维注意

### 5.1 清理动作必须按 PID 精确杀

今天的事故：一个早前挂的抓栈作业在收尾时执行 `pkill -ABRT -f train_lingbotvla`，
结果打到了**刚启动的预扫描 worker** 上，一个 worker 在加载权重中途被杀，50 个任务只写入
44 条、缺 6 个。

规则（本工具已内置）：

* 启动时记录每个 worker 的 **PID**（`setsid nohup … &` 之后 `echo $!` 取回真实 PID）；
* 需要清场时只对**这些 PID** 动手，先读 `/proc/<pid>/cmdline` 核对身份
  （`argv[0]` 必须是 shell/解释器，且命令行里出现 `al_50task_bf16.sh` / `train_lingbotvla.py`），
  核对不过就**拒绝发信号**并打印原因；
* 只有确认是「自己起的会话首进程」（`pgid == pid`）才发进程组信号；
* 本工具**从不**使用 `pkill -f` / `pgrep -f` 之类的宽正则。

手工排障时也请遵守同样口径：

```bash
# 想确认某进程是什么，再决定要不要动它（只读）
tr '\0' '\n' < /proc/<pid>/cmdline | head -3
```

### 5.2 并行扫描与训练不能同时跑

* 两者都吃满卡；同跑会让扫描 worker 与训练的集合通信互相干扰，还可能撞 `MASTER_PORT`。
* 本工具给每个 worker 单独分配一个空闲 `MASTER_PORT`（并显式传给启动脚本），
  正式训练也单独取一个；但**卡**是抢的，所以还是不要同跑。
* 扫描期间不要手工 `pkill`：一旦某个分片被杀，收尾会显示缺失任务名并以退出码 3 结束
  （不启动训练），补跑一次即可。

### 5.3 GPU 异常怎么识别（4 号卡就是这么挂的）

| 现象 | 判据 |
|---|---|
| 利用率恒 100%、温度只有 30 度左右 | 卡已挂死（不是在算） |
| 任何 H2D 小拷贝都挂住（最小拷贝测试无输出） | 同上；`rocm-smi --gpureset` 不支持 ⇒ 只能重启实例 |
| 训练/扫描启动后长时间没有 `已就绪`，日志无报错 | 先按上表查卡，不要盲目等待 |

实操顺序：先跑一个最小 H2D 拷贝测试确认卡可用，再启动扫描；把不可用的卡从 `--gpus` 里去掉。

### 5.4 磁盘

| 产物 | 体积 | 建议 |
|---|---|---|
| scout 缓存条目 | 每条几 KB | 放持久卷（默认 `/workspace/al/scout_cache`） |
| 扫描 worker 的 `TRAIN_OUT` | 只有日志（默认 `SMOKE_NO_CHECKPOINT=1`） | 默认放在 `/workspace/al/al_launch_runs` |
| 正式训练的 DCP | 约 24 GB/份 | 建议 `--train-out` 指到 overlay 大盘 |
| HF 导出 | 约 12 GB/份 | 及时搬到持久卷 |

默认 `--min-free-gb 20`：TMPDIR / 编译缓存 / 输出目录可用空间不足 20 GB 时拒绝启动。

### 5.5 退出码与机器可读输出

| 退出码 | 含义 |
|---|---|
| `0` | 成功（dry-run 完成，或训练已后台启动） |
| `2` | 前置检查失败（参数/路径/AL 配置/机器被占用） |
| `3` | **扫描未完成**：目标缓存文件里仍有缺失任务（已打印缺失任务名） |
| `4` | worker 未就绪或提前退出（进程数 / 「已就绪：N 任务」不符） |
| `5` | 扫描超时且覆盖不完整（已精确终止仍在跑的 worker） |
| `6` | 内部错误（未预期异常） |
| `7` | 训练启动后 1 秒内退出（日志尾部已打印） |

`--json` 时 stdout 只有一份 JSON（人话日志转到日志文件与 stderr），关键字段：

```json
{
  "ok": true, "exit_code": 0, "mode": "incremental",
  "fingerprint": "cd5f5c0f…", "cache_dir": "/workspace/al/scout_cache/cd5f5c0f…",
  "coverage_before": {"covered": 47, "total": 50, "missing": ["…"]},
  "coverage_after":  {"covered": 50, "total": 50, "missing": []},
  "plan": [{"step": 1, "title": "静态校验", "detail": "…"}],
  "shards": [{"index": 0, "gpu": "0", "tasks": ["…"], "pid": 12345, "ready_tasks": 7}],
  "env": {"worker_example": {"…": "…"}, "train": {"…": "…"}},
  "training": {"launched": true, "pid": 23456, "log": "…", "ranks": 7},
  "warnings": []
}
```

---

## 6. 重扫（rescan）：什么时候发生、会不会报错

### 6.1 什么时候发生

重扫 = 在训练过程中重新对**其他任务**做一次 scout 评测（训练一个任务会改变其他任务的
排序）。触发点有两个，都在 `lingbotvla/auto_learning/orchestration/scheduler.py`：

1. **任务切换**（`_after_transition`）：每完成一次实际任务切换，
   `task_switch_count += 1`；当
   `task_switch_count % rescan_every_n_task_switches == 0` 时做一次**全池重扫**
   （`_rescan()`），并让 `full_rescan_count += 1`。
   `rescan_every_n_task_switches: 1`（默认）⇒ **每次换任务后都重扫一遍**，一轮约 15–20 分钟。
2. **任务被 promoted**：`_select()` 里做轮次 rollover（`rollover_round`）之后，只对刚被提升的
   那几个任务做**定向重扫**（事件流里对应 `action: "round_rollover"`）。这条与
   `rescan_every_n_task_switches` 无关，也不增加 `full_rescan_count`。

### 6.2 会不会报错：不会静默卡死

* 重扫**不使用 bootstrap 缓存**：`_timed_eval(..., bootstrap_cache=False)`，
  因此全部是**真评测**（只有 `global_step == 0` 的 bootstrap 才允许查缓存）。
* 重扫走的是**与训练中评测同一条多卡路径**。这条路径开头就有一致性预检
  （`open_loop_validation.py` 的 `_multirank_eval_preflight`，可用 `AL_EVAL_PREFLIGHT=0` 回滚排障）：
  各 rank 交换「身份/回合集合/推理次数/数据集长度」等字段，**任何 rank 失败或任何字段不一致
  ⇒ 所有 rank 一起抛**（fail-closed），报错信息会点名具体 rank 与不一致的字段名（`rank3 与
  rank0 的 n_starts 不一致：…`）。所以出问题时看到的是**带 rank 与差异项的报错**，
  而不是「进程还在、什么都不动」。
* 因此重扫期间卡住时，先看日志里有没有这条一致性预检报错；没有再去查卡（§5.3）。

### 6.3 重扫节奏怎么调（`rescan_every_n_task_switches`）

> 启动器**没有** `--rescan-every` 参数。别人口头说的「`--rescan-every N`」指的就是
> **AL 配置项** `rescan_every_n_task_switches: N`。

```yaml
# configs/auto_learning/<你的配置>.yaml
rescan_every_n_task_switches: 5    # 每 5 次任务切换重扫一次全池
```

* 默认值 **1**：每次换任务后重扫一遍历史任务，一轮约 **15–20 分钟**；
* **生产环境建议 5–10**：把重扫摊薄，减少长时间占用卡与算力；
* 该值必须是 **>= 1 的整数**（`int`，布尔值也不行），否则配置校验直接报
  `ValueError: rescan_every_n_task_switches 必须是 >= 1 的整数`，训练起不来；
* 想临时关掉重扫（只排障用）：`rescan_candidates_after_transition: false`。

### 6.4 怎么测「重扫确实发生了」

#### 路径一：CPU 层（秒级、可反复，不需要 GPU）

先跑现成的覆盖：

```bash
cd /workspace/lingbot_vla_2/lingbot-vla-v2

# 题目点名的那两条
/opt/robotwin-env/bin/python -m pytest -q \
  tests/test_forgetting.py tests/test_gmean_pass_pipeline.py -k rescan

# 节奏 / 恢复 / 事件流 相关的完整一组
/opt/robotwin-env/bin/python -m pytest -q \
  tests/test_al_rescan_trigger_cpu.py \
  tests/test_al_openloop_update.py \
  tests/test_al_openloop_update_extra.py \
  tests/test_gmean_ratio_priority.py
```

`tests/test_al_rescan_trigger_cpu.py`（本说明配套新增）把「触发路径」端到端钉住：
构造注册表 → `advance()` 走完 bootstrap 与一个 train_unit → 触发一次真实任务切换 →
断言

* `state.task_switch_count` 与 `state.full_rescan_count` 各 +1（`rescan_every_n_task_switches=1`）；
* 评测事件流里出现 `kind="rescan"` 的记录（`scheduler.heatmap_rows` 与
  `TaskRecord.eval_history`，两者都会随 AL 状态持久化）；
* 落盘形态的指标名出现：`task/<任务>/rescan_nmse`；
* 重扫**没有**走进 bootstrap 缓存入口（`evaluate_bootstrap_scout` 未被调用，全走 `evaluate`）；
* 把节奏调成 3 时，前两次切换不重扫、第三次才重扫（且不碰缓存）。

要自己再补一条用例，按上面的骨架写即可（`tests/al_fixtures.py` 的 `make_cfg` /
`scheduler_of` 就是干这个的）。

#### 路径二：真机层（在真实训练里触发）

目的：让当前任务**尽快被判掉**，制造一次任务切换。

```yaml
# 临时配置（不要改正式配置）：让任务快速走到 EXHAUSTED
max_attempts_per_task: 1            # 只有一次机会，失败就 EXHAUSTED
min_steps_before_defer: 1           # 尽早允许 DEFER
retry_budget_steps: 1               # 重试预算取最小
rescan_every_n_task_switches: 1     # 第一次切换就重扫（默认值）
```

```bash
# 起训练（本启动程序会把前面几节的环境变量都准备好）
/opt/robotwin-env/bin/python experiment/robotwin/al_launch.py \
  --eval-config configs/auto_learning/<上面的临时配置>.yaml --steps 200

# 观测点 1：事件流里出现重扫产生的评测指标行（每被重扫一个任务一行）
grep -c '"name": "task/.*/rescan_nmse"' <TRAIN_OUT>/auto_learning_events.jsonl
grep -m3 '"name": "task/.*/rescan_nmse"' <TRAIN_OUT>/auto_learning_events.jsonl

# 观测点 2：TensorBoard 同名曲线
#   task/<任务>/rescan_nmse、task/<任务>/rescan_gmean_mse、debug/<任务>/rescan_mse

# 观测点 3：重扫耗时（每次评测的墙钟时间，秒）
grep -m3 '"name": "task/.*/scout_eval_wall_seconds"' <TRAIN_OUT>/auto_learning_events.jsonl
```

注意两点：

* `full_rescan_count` **不会**打印在运行日志里，它在 AL 状态（`scheduler.state`）里，随
  checkpoint 的 `extra_state.auto_learning` 一起存盘；想核对计数就从 DCP 里读状态，
  或者看上面的事件流指标行。
* 重扫一轮 15–20 分钟、且**不吃 bootstrap 缓存**（全部真评测），所以这组实验会明显变慢：
  验完就把配置改回 `rescan_every_n_task_switches: 5`（或原来的值）。

---

## 7. 附录

### 7.1 自检：`--selfcheck`

```bash
/opt/robotwin-env/bin/python experiment/robotwin/al_launch.py --selfcheck
# [selfcheck] 参考实现：…/lingbotvla/auto_learning/scout_cache.py
# [selfcheck] 对拍项：34；结果：全部一致
```

它把本脚本里的指纹实现与仓库 `scout_cache.py` 逐项对拍：`.py` 语义 hash 逐文件比对、
`normalize_dtype`、`scout_key`、`source_manifest` 清单与缺失清单、以及用一个临时伪造
checkpoint 做的端到端 `provenance` 比对。**任何一项不一致都会以退出码 2 结束**。
升级本脚本或改动指纹相关代码后，请先跑这条。

### 7.2 已知限制 / 设计取舍

1. **worker 默认不存盘**：`SMOKE_NO_CHECKPOINT=1`。原因是扫描进程跑 `MAX_STEPS=1`
   时会走到收尾存档，7 个 worker 各写一份约 24 GB 的 DCP（持久卷只有约 98 GB）。
   需要 worker 存盘时显式加 `--worker-checkpoint`，并把 `--worker-out-root` 指到大盘。
2. **`<repo>/log.txt` 会被多个 worker 混写**（`train.sh` 内部 `tee` 所致，属既有行为）；
   判读用每片自己的日志。
3. **重算指纹要读全部权重分片**（约 12 GiB，只读不写）。已知要复用某份缓存时用
   `--fingerprint <64 位>` 跳过；注意该模式**跳过自动失效**，有效性由你负责。
4. **本脚本用训练环境解释器算指纹**。若你用别的 Python 启动它，而两者
   `ast.dump` 行为不同（大版本差异），脚本会**自动改用 `--python` 指到的解释器**做一次
   子进程指纹计算，并在日志里注明来源（`[fingerprint] 来源=subprocess(…)`）。
5. **`--no-cache` 不删数据**：旧缓存文件只是改名成 `scout.json.bak-<时间戳>`，确认无误后可自行删除。
