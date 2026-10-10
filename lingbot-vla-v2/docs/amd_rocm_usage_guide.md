# AMD / Radeon Cloud（ROCm）使用说明

> 面向：在 **Radeon Cloud Global**（AMD Radeon PRO W7900D ×8）上跑 LingBot-VLA-v2 的
> RoboTwin 闭环评测、全参 SFT / LoRA 训练。
> 全部命令与数字均为 **2026-10-10 在本机 + 该实例上实测**；配套分支 **`feature/rocm-adapt`**。

---

## 0. 一句话总览

这套镜像（**external-data 变体**）自带**代码 + 完整 Python/ROCm 环境**，**不含数据与权重**；
数据必须由平台挂载（`Mount a model = Devzone`）或按官方配方自建到 `/models/robotwin-persistent`。
**实例无外网**（HF/GitHub/PyPI 全部不通）⇒ 一切代码/数据/权重只能**从外部推入**。

| 能力 | 状态 |
|---|---|
| 推理服务（13400） | ✅ 开箱（官方脚本） |
| 闭环评测（clean / randomized） | ✅ 开箱（`eval_policy_xpolicylab.py` / `run_clean_benchmark.py`） |
| 训练（LoRA / 全参 SFT，4/8 卡 FSDP2） | ✅ 官方脚本 + 本仓库代码均可 |
| 依赖更新 | ❌ 无外网 ⇒ 只能离线 wheel 搬运 |

---

## 1. 机器与环境

### 1.1 规格与坐标

| 项 | 值 |
|---|---|
| GPU | **8 × AMD Radeon PRO W7900D，各 48 GB**（`gfx1100`，ROCm **7.2.1**） |
| CPU / 内存 | 128 核 / 503 GB |
| 关键环境 | `/opt/robotwin-env`（**torch 2.9.1+rocm7.2.1 / Triton 3.5.1 / FlashAttention2 2.8.4 / lerobot 0.6.0**）、`/opt/lerobot-env`、`/opt/aiter`、`/opt/rocm-7.2.1` |
| 源码 | `/RoboTwin`（RoboTwin @ `266f3aa` + `experiments/lingbot_vla_v2_6b_robotwin/`，含 ROCm 补丁后的 port） |
| 登录 | 平台页面的 SSH 窗口给 `host/port`；公钥在 **Settings → New SSH Key** 登记（只传 `.pub`） |

### 1.2 卷与持久化（**最容易踩的坑**）

| 路径 | 性质 | 说明 |
|---|---|---|
| **`/workspace`** | ✅ **持久卷（~98 GB）** | 平台配置页勾 **Persistent /workspace**；**产出/权重放这里** |
| `/`（overlay） | ❌ **不持久** | `/models/robotwin-persistent` 默认落在这里；**实例销毁即丢** |
| `/models` | 由平台挂载 | 勾 **`Mount a model = Devzone`** 后，平台把数据挂到 `/models`，里面就是 `/models/robotwin-persistent` |

判定"到底挂没挂"（**目录存在 ≠ 挂载**）：

```bash
findmnt -T /models/robotwin-persistent     # SOURCE 必须是真实文件系统；显示 overlay ⇒ 没挂 ✗
stat -f -c %T /models/robotwin-persistent  # overlayfs ⇒ 只在可写层 ✗
```

### 1.3 无外网：只进不出

```bash
curl -sS -m 10 -o /dev/null -w '%{http_code}\n' https://huggingface.co   # 000
curl -sS -m 10 -o /dev/null -w '%{http_code}\n' https://hf-mirror.com    # 000
```
* Pod 内**无默认路由** ⇒ 出站全断；但**入站可用**（NodePort 映射）⇒ **只能"外部推入"**。
* 平台自带 `hf-url-shim.py`（把 `huggingface.co` 改写成 `hf-mirror.com` 的 curl/wget 包装）在此实例**同样超时**，不可依赖。
* 传输链路实测：**AutoDL → cpu1 单流 7.5 MB/s，4 流 ~13.4 MB/s**（8 流已饱和）；**小文件极慢**（1227 个小文件仅 3.8 MB/s）⇒ 大量小文件务必先 `tar`/`git bundle` 打包。
* ⚠️ 平台切换/重建实例后 **NodePort 会变**，SSH 地址需重新获取。

---

## 2. 数据：`/models/robotwin-persistent` 三件套

镜像内的软链（**不要改**）：
```
/RoboTwin/assets  -> /models/robotwin-persistent/assets
/RoboTwin/data    -> /models/robotwin-persistent/data
/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models -> /models/robotwin-persistent/models
```
必需内容（镜像 entrypoint 会检查，缺则报 `Mount the external RoboTwin data at /models/robotwin-persistent`）：
```
assets/objects/objaverse/list.json
data/demo_clean/                      # 50 个任务（HDF5 演示）
data/lerobot/                         # 50 个 <task>_joint_v30（LeRobot v3）
data/robotwin_demo_clean_joint_v30.txt# 训练清单（50 行）
models/robbyant_lingbot-vla-v2-6b/model.safetensors.index.json
```

### 2.1 官方路径（推荐）

实例配置页：**Customize → 4/8 GPUs → Image = robotwin → Resource Pool = Dev →
Workspace Storage = Persistent /workspace → Mount a model = Devzone**。
**不选 Devzone ⇒ 实例内不会挂载数据**（`/models/robotwin-persistent` 会是个空目录）。

### 2.2 自建路径（拿不到 Devzone 时）

官方配方在 `Robotwin-radeon-cloud` 仓库的 `docker/full/Dockerfile`，**不需要 Docker**，照抄即可：

| 步 | 内容 | 来源 / 说明 |
|---|---|---|
| ① assets | `TianxingChen/RoboTwin2.0` 的 `background_texture.zip / embodiments.zip / objects.zip` → 解压到 `/RoboTwin/assets` → 跑 `scripts/update_embodiment_config_path.py` | ~6.2 GB；`list.json` 必须存在 |
| ② demo_clean | `bash scripts/download_xpolicylab_data.sh`（50 个任务，zip 集 **~22 GB** → 解压 **~33 GB**） | 每任务 152 episode |
| ③ 模型 | `robbyant/lingbot-vla-v2-6b`（27 G F32）、`Qwen/Qwen3-VL-4B-Instruct`（**只要** `*.json/*.txt/*.jinja/merges.txt/vocab.json`）、`Ruicheng/moge-2-vitb-normal`（全参 SFT 的 depth/video teacher 才需要） | 用固定 revision |
| ④ LeRobot 转换 | `convert_all_data.sh`（官方脚本，8→32 并行） | 生成 50 × `<task>_joint_v30` + 清单 |

固定 revision（与教程一致）：
```
RoboTwin commit        266f3aadf505a4f7fe9af0faa41a20f5f47cd123
LingBot-VLA-v2 commit  951475ae1b1d87553e7dc47c97b53a3d695c0d13
base 模型             11c703bf6a5c1f45b3b69168482da11fdbba53d7
Qwen config           ebb281ec70b05090aa6165b016eac8ec08e71b17
MoGe-2                ca5f0e07ff01d3e5a364c1d954ed12ee1814b368
RoboTwin2.0 数据      a967b852afa21a9cbf19a198f7e653109042e87c
```

### 2.3 离线数据管线（本会话实测的可行做法）

```bash
# ① 有网机器（如 AutoDL，可通 hf-mirror）下载 zip，rsync 到 cpu1（4 路 tar 并行 ≈ 13 MB/s）
rsync -a dataset/ root@<cpu1>:<port>:/models/robotwin-persistent/data/download_cache/dataset/

# ② cpu1 离线解压：用官方脚本的 Python 逻辑（保留 videos→video / episode_N→episode_000000N 归一化），
#    仅把 download() 换成"直读本地 zip"，并显式传 50 个任务名（跳过联网发现）
TARGET_ROOT=/RoboTwin/data ARCHIVE_ROOT=<cache> HF_ARCHIVE_NAME=demo_clean.zip \
HF_MAX_WORKERS=8 HF_EXTRACT_WORKERS=32 HF_KEEP_ARCHIVES=1 \
/opt/robotwin-env/bin/python /workspace/extract_local.py $(ls <cache>/dataset)
#   实测：50 任务 ~1–2 分钟（21 GB → 33 GB）

# ③ LeRobot v3 转换（CPU 密集，视频重编码）
CONVERSION_JOBS=32 HF_LEROBOT_HOME=/RoboTwin/data/lerobot \
LEROBOT_VIDEO_BACKEND=pyav LEROBOT_VIDEO_CODEC=h264 \
bash docker/full/convert_all_data.sh
#   实测：~968 MB/分钟、~2 数据集/分钟 ⇒ 50 个约 20–40 分钟；脚本自带转换缓存，重跑命中缓存
```

---

## 3. 代码：把仓库搬上去 + ROCm 适配

```bash
# ① 有网机器：整仓打包（含 .git 全历史）——不要直接 rsync 子目录（会丢 .git）
git bundle create /tmp/lingbot_vla_2.bundle --all      # 34 MB
rsync /tmp/lingbot_vla_2.bundle root@<cpu1>:/workspace/
# ② cpu1：克隆成独立仓库（放持久卷）
cd /workspace && git clone lingbot_vla_2.bundle lingbot_vla_2
```

**ROCm 适配（分支 `feature/rocm-adapt`）**：
```bash
git checkout -b feature/rocm-adapt <base-commit>
cd lingbot-vla-v2
patch -p1 -F 3 < /workspace/refs/robotwin-radeon-cloud/docker/patches/lingbot-vla-v2-rocm.patch
# 唯一不适用的一处：上游给 LeRobotDataset(...) 加 video_backend="pyav"，
# 本仓库已把 video_backend 做成参数 ⇒ 只需把默认值 torchcodec 改成 pyav：
#   lingbotvla/data/vla_data/base_dataset.py      video_backend = 'pyav'
#   lingbotvla/data/vla_data/multi_vla_dataset.py video_backend: str = 'pyav'
```
**让本仓库代码优先生效（不覆盖镜像内 port 的 editable 安装）**：
```bash
export PYTHONPATH=/workspace/lingbot_vla_2/lingbot-vla-v2:$PYTHONPATH
# 验证：
/opt/robotwin-env/bin/python -c "import lingbotvla, tasks; print(lingbotvla.__file__, tasks.__path__)"
# 期望：都指向 /workspace/lingbot_vla_2/lingbot-vla-v2/...
```

---

## 4. 权重：bf16 副本（训练/服务都用它）

```bash
cd /workspace/lingbot_vla_2/lingbot-vla-v2
/opt/robotwin-env/bin/python tools/make_bf16_ckpt.py \
  --src /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b \
  --dst /workspace/models/robbyant_lingbot-vla-v2-6b-bf16 \
  --verify sample:200
# 实测：1708 张量 / 16.3 s / 23.75 → 11.88 GiB，index.total_size 同步更新，抽样逐位校验通过
```
* **写到 `/workspace`（持久）**；F32 原件保留（27 GB）。
* 训练/服务用 `--model.model_path /workspace/models/robbyant_lingbot-vla-v2-6b-bf16`。

---

## 5. 训练

### 5.1 一键包装（推荐）

`/workspace/rocm_train.sh`：自动设 `PYTHONPATH`（本仓库优先）、ROCm 变量、
**编译缓存落持久盘**（`TORCHINDUCTOR_CACHE_DIR=/workspace/runtime/torchinductor`，避免每次重启重编），
打印配置与"实际导入的代码路径"后再调官方 `train_full_sft.sh`。

```bash
GPU_COUNT=8 MICRO_BATCH_SIZE=12 GLOBAL_BATCH_SIZE=96 MAX_STEPS=100 SAVE_STEPS=100 \
BASE_MODEL=/workspace/models/robbyant_lingbot-vla-v2-6b-bf16 \
TOKENIZER_PATH=/workspace/models/robbyant_lingbot-vla-v2-6b-bf16 \
OUTPUT_DIR=/workspace/runtime/outputs/rocm_smoke_micro12_gbs96 \
bash /workspace/rocm_train.sh
```

### 5.2 官方启动器的关键参数（包装脚本内部）

```bash
python -m torch.distributed.run --standalone --nproc-per-node=$GPU_COUNT \
  -m tasks.vla.train_lingbotvla $CONFIG \
  --data.train_path /RoboTwin/data/robotwin_demo_clean_joint_v30.txt \
  --train.micro_batch_size 12 --train.gradient_accumulation_steps 1 \
  --train.global_batch_size 96 \
  --train.data_parallel_mode fsdp2 --train.data_parallel_replicate_size 1 \
  --train.data_parallel_shard_size 8 --train.enable_full_shard true \
  --train.enable_gradient_checkpointing true --train.optimizer adamw \
  --model.model_path <bf16> --train.max_steps 100 --train.save_steps 100
```
* **不要直接调 `torchrun`**，用 `python -m torch.distributed.run`。
* `GBS = micro × GPU_COUNT × grad_accum`（启动器会强校验整除）。
* 显存参考（8×48 GB，micro 12 / GBS 96 的估算）：权重 bf16 12 GB÷8≈1.5 GB/卡、梯度≈1.5、AdamW 状态≈6、激活 10–25 ⇒ **~20–35 GB/卡**，有余量。
* **DCP 单份 ~24 GB**、HF 导出 ~12 GB ⇒ 见 §5.3 的存档策略（**overlay 优先**）。

### 5.3 存档策略：overlay 优先 + 销毁前挽救（2026-10-10 定）

持久卷 `/workspace` 只有 ~98 GB，而 **DCP 每份 ~24 GB** ⇒ 多存档放不下。
**因此训练存档优先写 overlay**（`/`，宿主 2.6 TB 共享空间），**但 overlay 不持久** ⇒
**销毁实例前必须把要留的模型搬走**。

| 产物 | 建议位置 | 理由 |
|---|---|---|
| DCP（`checkpoints/global_step_N/`） | **overlay**：`/models/robotwin-persistent/outputs/<run>` | 仅用于 resume；单份 24 GB，多份会撑爆持久卷 |
| **HF 导出**（`hf_ckpt/`，~12 GB） | **先生成在 overlay → 及时搬到 `/workspace/keep/`** | 这才是要长期保留的产物 |
| 数据/权重 | `lerobot`、`bf16` → `/workspace`；`demo_clean`、F32 base → overlay | 见 §1.2 |

训练输出改成 overlay（三个入口都要显式指定）：
```bash
# 官方 train_full_sft.sh
OUTPUT_DIR=/models/robotwin-persistent/outputs/<run> bash .../train_full_sft.sh
# 本仓库包装
OUTPUT_DIR=/models/robotwin-persistent/outputs/<run> bash /workspace/rocm_train.sh
# 本仓库 AL 启动器（默认是 AutoDL 的 /data/outputs ✗ 必须覆盖）
TRAIN_OUT=/models/robotwin-persistent/outputs/<run> SAVE_EVERY=1000 PRUNE=1 PRUNE_KEEP=1 \
  MICRO=12 GAS=1 N_GPU=8 bash experiment/robotwin/al_50task_bf16.sh
```

**⛔ 销毁实例前的挽救清单（务必执行）**
```bash
# ① 看 overlay 上有什么、多大
du -sh /models/robotwin-persistent/outputs/*/checkpoints/* 2>/dev/null | sort -rh | head
ls -la /models/robotwin-persistent/outputs/*/checkpoints/*/hf_ckpt 2>/dev/null

# ② 把"好的"搬到持久卷（HF 导出优先；DCP 只在需要 resume 时留）
mkdir -p /workspace/keep
cp -a <run>/checkpoints/global_step_N/hf_ckpt /workspace/keep/

# ③ 或直接拉回外部（本机 / AutoDL）
#   rsync -a -e 'ssh -p <port>' root@<host>:/workspace/keep/ ./keep/

# ④ 复核 /workspace/keep 完整后再销毁实例
bash /workspace/rescue_before_destroy.sh     # 本仓库提供的检查+搬运脚本
```
**体积速查**：DCP ~24 GB/份 ｜ HF 导出 ~12 GB/份 ｜ lerobot(50 任务) ~5–15 GB ｜ bf16 权重 12 GB。

---

## 6. 评测

### 6.1 官方两阶段（单任务基准，单卡）

```bash
# 阶段 1：常驻策略服务（GPU0 → 端口 13400）
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/launch_official_server.sh \
     0 13400 /workspace/runtime/outputs/logs/official_server.log False <MODEL>
# 就绪判据：curl http://127.0.0.1:13400/healthz  → 200
# 阶段 2：闭环（10 回合）
/opt/robotwin-env/bin/python scripts/eval_policy_xpolicylab.py \
  --task_name adjust_bottle --task_config demo_clean \
  --policy_name LingBot-VLA-v2 --protocol lingbot_vla_v2 \
  --host 127.0.0.1 --port 13400 --device_id 0 --seed 0 --test_num 10 \
  --expert_check true --accept_expert_info_on_failure true --eval_batch false \
  --additional_info eval_video_log=false
# 环境：ROBOTWIN_DISABLE_CUROBO=1 ROBOTWIN_EE_PLANNER=mplib PYOPENGL_PLATFORM=egl
```
* `--device_id 0` 是对的：子进程只有一张可见卡（`HIP_VISIBLE_DEVICES=<gpu_id>`）⇒ sim 的 device 0 = 物理卡。
* 服务显存实测：**~11.9 GB**；跑仿真时同卡再加 ~3 GB。
* 单回合耗时（cpu1 实测偏慢，见 §8）：比 CUDA 机器慢 5–8×，主因是 **MPLib 规划器**替代 CuRobo。

### 6.2 官方 8 卡全量 benchmark

```bash
/opt/robotwin-env/bin/python experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py \
  --gpu-count 8 --task-config both --episodes 5 \
  --base-port 13400 --run-name both_100x5_8gpu --resume
```
分配机制（源码级）：
* **每卡一个模型副本**（端口 `13400+i`），共 8 个服务；
* 任务清单 = `env_cfg/eval/all_tasks.yml` 的 50 个任务 ×（`both` ⇒ clean + randomized）= 100 项；
* **任务级动态队列**（`queue.Queue` + 8 线程）：一个任务整体在一张卡上跑完，**谁空谁领下一个**（无 LPT 排序）；
* 断点续跑：每任务写 `done/<cfg>__<task>.done`；`--resume` 自动跳过；收尾校验任务数与回合数（`100 × episodes`），打印 `overall success: X/Y = Z%`；
* ⚠️ **改 `--episodes` 必须换 `--run-name`**（标记与回合数校验会冲突）。

### 6.3 本仓库的多卡编排器（推荐用于批量 checkpoint）

`experiment/robotwin/robotwin_multi_ckpt_eval.py`（1422 行）：
**不改官方调度**，只在官方 launcher 之上做编排 —— 多 checkpoint **滑窗并行**（`--max-parallel-checkpoints`）、
端口段隔离（`--start-port-base`，宽 = `num_gpus × num_per_gpu`）、**完成即补位**、
checkpoint 完整性校验（`--min-age-seconds`）、**断点状态**（`--state-file`）、clean+randomized **自动汇总**、`--dry-run`。
* 注意：需移植 `nvidia-smi`→`rocm-smi`、`CUDA_VISIBLE_DEVICES`→`HIP_VISIBLE_DEVICES`，并把
  `--eval-workdir`/`--conda-sh` 换成 `/RoboTwin`、`/opt/robotwin-env`。
* 官方文档明确指出：**官方多卡评测不是模型并行**，而是"多副本 + 任务级并行"（显存 N 倍）。

---

## 7. 运维

```bash
# 显存（AMD 用 rocm-smi，不是 nvidia-smi）；/usr/local/bin/vram 是本会话装的助手
vram            # 8 卡 used/total + 占用进程
rocm-smi --showmeminfo vram          # 字节数
rocm-smi --showpids                  # 谁在占（容器内 PID 可能显示 UNKNOWN）
rocm-smi                             # 简明表：VRAM% / GPU%
watch -n 1 rocm-smi --showmeminfo vram
```
```bash
# 进程/端口/磁盘
pgrep -af "[l]ingbot_vla_v2_policy"  # 服务（方括号避免匹配自身！）
df -h / /workspace                   # /workspace 才是持久盘
findmnt -T /models/robotwin-persistent
```

---

## 8. 坑清单（本会话全部实测踩过，均已修复）

| # | 坑 | 现象 | 正确做法 |
|---|---|---|---|
| 1 | **`pkill -f` 自匹配** | `pkill -f 'lingbot_vla_v2_policy'` 把执行它的 shell 自己杀了 ⇒ 后续命令全不执行、日志都没建 | 用方括号：`pkill -f '[l]ingbot_vla_v2_policy'` |
| 2 | **`HF_HUB_OFFLINE=1` 不认 `local_dir`** | 官方脚本 `hf_hub_download(local_dir=...)` 直接抛 `OfflineModeIsEnabled`；无任务参数时还会先联网"发现"任务 | 用官方 Python 逻辑 + `download()` 直读本地 zip + **显式传任务名** |
| 3 | **`python3` 落到系统 Python** | 官方脚本 `python3 -c "import huggingface_hub"` 失败 ⇒ 触发 `pip install` ⇒ **PEP 668** 报错 | `PATH=/opt/robotwin-env/bin:$PATH`（该环境有 `huggingface_hub`/`huggingface-cli`） |
| 4 | **`split -n l/4 -` 读管道** | `split: cannot determine file size` ⇒ 只生成 1 个空分组 ⇒ 0 传输 | 用 `awk 'NR%4'` 分组（或 `split -n r/4`） |
| 5 | **rsync 小文件慢** | 1227 个小文件只有 3.8 MB/s | 先 `tar` / `git bundle` 打包；大文件用 **4 路并行**（13.4 MB/s） |
| 6 | **多行命令粘贴被压成一行** | `du: cannot access 'docker/full/build.sh'` 之类 | 逐行粘；写脚本再执行 |
| 7 | **云端跑 `docker/*/build.sh`** | `not found`（云端无该仓库、无 Docker，且教程禁止第二层 Docker） | 云端用平台已构建的镜像；造数据照 Dockerfile **裸跑**即可 |
| 8 | **"目录存在"误判为"挂载成功"** | `/models/robotwin-persistent/{assets,data,models}` 是自己 `mkdir` 的 | 用 `findmnt -T` 看 SOURCE；缺 `data/lerobot`、清单等仍会被 entrypoint 判失败 |
| 9 | **闭环比 CUDA 慢 5–8×** | 3 分钟只跑到 142/400 步；日志反复 `right arm planning failed (IK Failed!)` | AMD 用 **MPLib** 而非 CuRobo（官方设定）；评测排期要按慢速估算，且**别与 CPU 密集任务（如 LeRobot 转换）同跑** |
| 10 | **训练与评测抢卡** | 服务常驻占卡（~12 GB/卡）；benchmark 8 卡全占 | 二者互斥；benchmark 支持 `--resume`，可"停-训-续" |
| 11 | **编译缓存丢** | 默认落 `/tmp`，重启/换形状即重编 | `TORCHINDUCTOR_CACHE_DIR=/workspace/runtime/torchinductor`（+ `TRITON_CACHE_DIR`） |
| 12 | **`deepspeed` 未安装** | import 失败 | 官方训练路径用 `torch.distributed.run`，**不需要** deepspeed |

---

## 9. 快速上手（复制粘贴）

```bash
# 0) 登录与开卡后先确认三件事
findmnt -T /models/robotwin-persistent          # 数据挂上了吗
ls /RoboTwin/data/demo_clean | wc -l            # 应为 50
ls /RoboTwin/data/lerobot | wc -l               # 应为 50
vram                                            # 8 卡应空闲

# 1) 环境自检（指南 §3 原样）
/opt/robotwin-env/bin/python -c "
import aiter, flash_attn, torch, triton, open3d, sapien, mplib, lerobot
print(torch.__version__, torch.version.hip, 'GPU', torch.cuda.device_count(), 'FA2', flash_attn.__version__)"

# 2) 用本仓库代码（分支 feature/rocm-adapt）
export PYTHONPATH=/workspace/lingbot_vla_2/lingbot-vla-v2:$PYTHONPATH

# 3) 起服务 + 5 回合闭环
bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/launch_official_server.sh \
     0 13400 /workspace/runtime/outputs/logs/official_server.log False \
     /workspace/models/robbyant_lingbot-vla-v2-6b-bf16
TASKS="adjust_bottle" EPISODES=5 PORT=13400 bash /RoboTwin/...   # 或用官方 eval_policy_xpolicylab.py（见 §6.1）

# 4) 训练冒烟
GPU_COUNT=8 MICRO_BATCH_SIZE=12 GLOBAL_BATCH_SIZE=96 MAX_STEPS=100 \
BASE_MODEL=/workspace/models/robbyant_lingbot-vla-v2-6b-bf16 bash /workspace/rocm_train.sh
```

---

## 附：与 CUDA 侧的差异速查

| 维度 | CUDA（AutoDL 4090/96G） | AMD（Radeon Cloud W7900D） |
|---|---|---|
| 显存查看 | `nvidia-smi` | `rocm-smi`（`/opt/rocm-7.2.1/bin/rocm-smi`） |
| 卡选择 | `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES`（服务脚本内部已处理） |
| 规划器 | CuRobo | **MPLib**（`ROBOTWIN_DISABLE_CUROBO=1 ROBOTWIN_EE_PLANNER=mplib`） |
| 视频后端 | torchcodec 可用 | **必须 pyav**（`LEROBOT_VIDEO_BACKEND=pyav`；数据加载默认值已改） |
| 闭环速度 | ~60–70 s/回合 | **~7–8 分钟/回合**（同任务、同设置，实测 5–8× 慢） |
| 结论口径 | — | 结果应标注 **ROCm + MPLib + expert_check=true**，不要直接与 CUDA/CuRobo 结果比较 |


---

## 10. ⚠️ 必做：修复 aiter gluon 的 triton 版本硬失败（否则训练完全起不来）

**症状**：训练启动后报
```
ValueError: Unrecognized configuration class <LingbotVLAV2Config> for this kind of AutoModel: AutoModel
```
且日志里出现 `Loading model from Huggingface modeling`（正常应为 `customized modeling`）。

**根因链**（2026-10-10 定位）：
```
/opt/aiter/aiter/ops/triton/gluon/__init__.py 在 import 时校验 triton>=3.6.0，而环境是 3.5.1 ⇒ raise RuntimeError
  ⇒ flash_attn（ROCm 版）硬依赖 aiter（该 import 不在 try/except 内）无法导入
    ⇒ lingbotvla 模型模块导入失败 ⇒ 配置注册表 arch 数 = 0（正常 2）
      ⇒ get_loader() 退回 HuggingfaceLoader ⇒ AutoModel.from_config(自定义 Config) 崩
```
**一键修复**（幂等、可逆，改前自动备份；`/opt/aiter` 在容器可写层，**实例重建后需重跑**）：
```bash
bash tools/rocm/fix_aiter_gluon_triton.sh
# 期望输出：
#   aiter OK ✓   flash_attn OK ✓   ★ 我们的模型模块 OK ✓   注册表架构数: 2
```
**自检**：`python -c "from lingbotvla.models.registry import get_registry; print(len(list(get_registry().supported_models)))"` ⇒ 应为 2。

## 11. 用官方执行器跑「我们的代码 + bf16 权重」（软链骨架）

官方 `train_full_sft.sh` 把一切路径从 `ROBOTWIN_ROOT` 派生 ⇒ 把该根指向**全软链骨架**，即可在**不改官方脚本**的前提下
换成我们的仓库与 bf16 权重：
```bash
bash tools/rocm/make_official_scaffold.sh              # 只建骨架 + 11 项自检
GPU_COUNT=8 MICRO=12 GBS=96 MAX_STEPS=20 SAVE_STEPS=1000 TEACHER_MODE=full OPTIMIZER=adamw \
  bash tools/rocm/make_official_scaffold.sh --run
```
* ⭐ **`source/lingbot-vla-v2` 必须是目录级软链**：`python -m tasks.vla...` 的 `sys.path[0]=cwd` ⇒ **cwd 压过 PYTHONPATH** ✗
* `models/robbyant_lingbot-vla-v2-6b` 用 bf16 副本时，要补 `assets/depth/dino_video` 三个软链（否则加载器判为 HF 格式 ✗）
* 日志走 `LOG_FILE`；官方脚本**启动期 `exit 2` 只走 stderr** ⇒ 别把 stderr 丢进 `/dev/null` ✗

**实测（micro 12 × GAS 1 × 8 卡 = GBS 96，`TORCHDYNAMO_DISABLE=1`）**：
`customized modeling` ✓ ｜ **9.4–9.8 s/step** ｜ **36.9 GiB/卡**（48 GiB 的 77%，**不 OOM** ✓）｜ loss 0.22 ✓

**仅冻 ViT 的已知问题**：`freeze_vision_encoder: true` 会触发
`AttributeError: 'LingbotVlaV2Policy' object has no attribute 'visual'` ✗（官方默认 `false` 可正常训练 ✓；待修）

## 12. 本机监控 AMD 显卡

```bash
bash watch_amd_gpu.sh          # 一次快照：每卡显存/温度/GPU 占用/进程
bash watch_amd_gpu.sh -w 5     # 每 5 秒刷新
bash watch_amd_gpu.sh -w 5 -l  # 同时显示训练 step / peak / loss
```
（走 `ssh_srv.sh cpu1` + `rocm-smi`；本机无需登录服务器 ✓）


---

## 13. 「仅冻 ViT」（freeze_vit）的正确用法与两个坑（2026-10-10 实测）

### 13.1 正确开关：`train.freeze_vit`（官方自己的键）
```yaml
train:
  freeze_vit: true              # ★ 仅冻 ViT（视觉塔）
  freeze_vision_encoder: false  # pi0 遗留的“死键”：trainer 里只有声明、从不使用
```
* **不要**把 `freeze_vision_encoder` 放进 **`model:`** 段 ✗ —— CLI 的 `model` 组没有这个字段，会在启动期直接报
  ```
  ValueError: Some specified arguments are not used by the ArgumentParser: ['--model.freeze_vision_encoder', 'true']
  ```
  （argparse 严格校验“所有参数都必须被用掉”，放错段位必挂 ✓）

### 13.2 代码坑：`train_lingbotvla.py` 的 freeze_vit 分支（已修）
```python
# 原代码（会崩）
if args.train.freeze_vit:
    model.visual.requires_grad_(False)      # ✗ LingbotVlaV2Policy 没有 .visual
```
两处都不对：
1. 视觉塔实际在 **`model.qwenvl.visual`**（`modeling_lingbot_vla_v2.py` 里 `self.qwenvl = Qwen3VLForConditionalGeneration...`）；
2. 该调用发生在 **`build_parallelize_model()` 之前** ⇒ 此刻**连 `model.qwenvl` 都还取不到** ✗（实测 `getattr` 两条路都失败）。
**修复（本仓库已含）**：改为**按参数名冻结**，不依赖模块层级：
```python
if args.train.freeze_vit:
    _vit_names = [n for n, _ in model.named_parameters() if ".visual." in n or n.startswith("visual.")]
    for _n, _p in model.named_parameters():
        if _n in set(_vit_names):
            _p.requires_grad_(False)
    logger.info_rank0(f"[freeze_vit] 已冻结视觉塔参数 {len(_vit_names)} 个张量 / {_n_frozen/1e6:.1f} M 参数")
    for _m in model.modules():                    # 视觉模块放 eval（BN/dropout 语义）
        if type(_m).__name__.endswith(("VisionModel", "VisionTransformerPretrainedModel")):
            _m.eval()
```
**验证点**：日志里出现 `[freeze_vit] 已冻结视觉塔参数 N 个张量 / M M 参数` ✓
（⚠️ AutoDL 时代的配置**从未打开过 `train.freeze_vit`**，所以这个隐藏 bug 一直没暴露；
AutoDL 上"仅冻 ViT"是由**另一条路径**实现的 —— 那条路径与 `freeze_vision_encoder` 相关，两边的语义要分清）

### 13.3 AMD 上实测的训练配方与数字（micro 扫描）
| micro | GAS | 卡数 | **GBS** | s/step | 样本/s | 每卡峰值 | 结论 |
|---|---|---|---|---|---|---|---|
| 12 | 1 | 8 | 96 | **9.4–9.8** | **≈10.0** | **36.9 GiB / 48 GiB** | ✅ 稳定（已跑完 20 步）|
| 16 | 1 | 8 | 128 | — | — | — | 待测 |
| 24 | 1 | 8 | 192 | — | — | — | 测试中（预计接近/超过 48 GiB）|

**固定 GBS 96 的换算**（`GBS = micro × GAS × 卡数`）：micro 16 ⇒ **6 卡**；micro 24 ⇒ **4 卡**（余卡可留给评测）。

### 13.4 分支与同步
* 工作分支：**`feature/rocm-adapt`**（基线 `feature/auto-learning-v1`），已推送 GitHub：
  `https://github.com/YonG00/lingbot_vla_2/tree/feature/rocm-adapt`
* 服务器侧同步（约定：`fetch` + `reset --hard`，先确认当前分支）：
  ```bash
  cd /workspace/lingbot_vla_2 && git branch --show-current && \
    git fetch -q origin && git reset --hard origin/feature/rocm-adapt
  ```
* 本仓库相关工具：`tools/rocm/{fix_aiter_gluon_triton.sh, make_official_scaffold.sh}`、
  `tools/rescue_before_destroy.sh`、根目录 `watch_amd_gpu.sh`


---

## 14. 指定 scout 缓存 / 强制重扫（`AL_SCOUT_CACHE_*`）

> 2026-10-10 加入。**用途**：Bootstrap 的全池 scout 扫描很贵（串行 ~14 分钟 ✗），
> 有时我们**明确知道**该用哪份结果、或明确要求重扫 ⇒ 给操作者两个开关 ✓。

### 14.1 原理（一句话）

```
BootstrapScoutCache(root, fingerprint) 内部就是：  path = root / fingerprint
   ⇒ 读 = 找 path/<任务>__<ids>.json    写 = 只由 rank0 写该目录 ✓
   ⇒ 所以【指定指纹 = 指定缓存目录】✓ 不哈希、不复制、不迁移 ✓
```

### 14.2 三种模式

| 模式 | 环境变量 | 行为 |
|---|---|---|
| **自动（默认）** ✓ | 只设 `AL_SCOUT_CACHE_MODE=bootstrap` | 按依赖内容算指纹 ⇒ 逻辑/数据/权重/选项**任一变化就自动失效** ✓（安全兜底 ✓）|
| **指定缓存重载** ✓ | `+ AL_SCOUT_CACHE_FINGERPRINT=<64位指纹>` | **跳过指纹计算**，直接用该目录 ⇒ 命中就用、缺的照常真扫并写回 ✓（**有效性由操作者负责** ✓）|
| **强制重扫** ✓ | `+ AL_SCOUT_CACHE_FORCE_RESCAN=1` | 目录名换成 `<指纹>-force-<时间戳>` ⇒ 必然全部 miss ⇒ **全量重扫** ✓（原目录不动 ✓）|

### 14.3 用法

```bash
# 1) 先看有哪些缓存目录（目录名就是完整指纹 ✓）
ls -d /workspace/al/scout_cache/*/
# 2) 取【完整 64 位】名字（前 8 位够你认人）
FP=$(basename $(ls -d /workspace/al/scout_cache/*/ | grep 3c8b15e2))
echo ${#FP}          # 必须是 64 ✓

# 3) 启动时带上（其余环境变量照常）
AL_SCOUT_CACHE_MODE=bootstrap \
AL_SCOUT_CACHE_ROOT=/workspace/al/scout_cache \
AL_SCOUT_CACHE_FINGERPRINT=$FP \          # ← 指定缓存重载 ✓（不要这行就是自动模式）
AL_SCOUT_CACHE_CHECKPOINT=/workspace/models/robbyant_lingbot-vla-v2-6b-bf16 \
AL_SCOUT_CACHE_MANIFEST=/workspace/al/task_splits_50/manifest.json \
AL_SCOUT_CACHE_BASELINE=/workspace/al/task_splits_50/task_baseline.json \
AL_SCOUT_CACHE_NORM=<repo>/assets/norm_stats/robotwin_competition_clean.json \
AL_SCOUT_CACHE_DTYPE=bfloat16 \
bash experiment/robotwin/al_50task_bf16.sh
```

### 14.4 注意事项（都踩过 ✗）

1. **必须 64 位小写 hex** ✗ —— 传 16 位前缀 ⇒ `ValueError: invalid provenance fingerprint`
   （这个校验是好事 ✓：防手误造出一个空目录 ✓）
2. **`AL_SCOUT_CACHE_MODE=bootstrap` 仍必须设** ✓ —— 两个开关是它的子功能；不设则整个缓存块不启用 ✓
3. **指定一个不存在的指纹 ⇒ 目录会被创建** ✓（`mkdir(parents=True)` ✓）⇒ 等效于"空缓存 + 全量重扫并写进该目录" ✓
4. **写缓存只有 rank0** ✓（多卡并发写会撕裂 ✓，所以 `write_enabled` 默认仅 rank0 ✓）
5. **语义边界** ✓：指定即信任 ✓ —— 工具**不会**替你核对"这份缓存是不是当前代码/权重算的" ✗
   ⇒ 操作者要自己判断（判据：依赖文件 mtime 都早于缓存目录 ✓）
   ⇒ 拿不准就用**自动模式** ✓（它会自动失效 ✓）
6. **分片共享** ✓：8 卡并行预扫描用的 `al_shard*.yaml` 与正式 run 的主配置**指纹相同** ✓
   —— 因为指纹**不含** AL 配置文件本身 ✓，只含其中影响评测的字段（`scout_trajs`/`noise_seed`/`stride`/`image_augment`/`inference_dtype`）✓

### 14.5 常见流程：8 卡并行预扫描 + 正式 run 复用

```bash
# ① 8 个 worker（各 1 卡 × 任务分片）写缓存
for i in 0..7: CUDA_VISIBLE_DEVICES=$i N_GPU=1 AL_CFG=configs/auto_learning/al_shard$i.yaml  …
# ② 正式 run：自动模式（推荐 ✓）或显式指定那份新指纹 ✓
```
> ⚠️ 改了**被指纹纳入的文件**（`open_loop_validation.py` / `scout_cache.py` / 数据/权重…）⇒ 自动模式会失效重扫 ✓
> ⇒ 此时应当【重跑 ①】而不是硬指旧缓存 ✓（除非你确定改动不影响评测数字 ✓ —— 那就用 `AL_SCOUT_CACHE_FINGERPRINT` ✓）
