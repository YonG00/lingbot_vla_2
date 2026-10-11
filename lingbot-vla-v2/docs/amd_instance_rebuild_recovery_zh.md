# AMD 实例（Radeon Cloud / ROCm）重建恢复 Runbook

> 来源：**2026-10-11 一次真实的实例重建事故**（`al_v34` 训练中实例被重建，端口 `33154 → 32763`、
> 容器名变化、数据全失）。本文把那次恢复的全部判据、命令、耗时与踩坑固化下来，供下次重建时照做。
>
> 结论先行：**重建后必须做四件事** —— ①确认外置卷是否挂载（没有就得搬数据）②修 aiter 坑
> ③搬代码 ④搬模型/数据。**前三件 10 分钟内完成，第四件是唯一的大头（取决于是否有 Devzone 卷）。**

---

## 0. 重建后 5 分钟内的"体检"（按顺序执行，全部只读）

```bash
# ① 环境是否完好（镜像自带，通常都在）
/opt/robotwin-env/bin/python -c "import torch,triton;print(torch.__version__, triton.__version__, torch.cuda.is_available())"
rocm-smi --showuse | grep -c "GPU\["          # 期望 8
ls -d /RoboTwin/XPolicyLab /RoboTwin/experiments

# ② ★ 最关键的判据：外置数据卷是否挂载
findmnt -T /models/robotwin-persistent -o TARGET,SOURCE,FSTYPE
#   显示 overlay  ⇒ **没挂**（数据全丢，必须搬）
#   显示具体设备/网络挂载 ⇒ 挂了 ⇒ 跳到 §3 验证 5 个必需路径即可

# ③ 5 个必需路径（external-data 镜像 entrypoint 的硬要求）
for p in assets/objects/objaverse/list.json data/demo_clean data/lerobot \
         data/robotwin_demo_clean_joint_v30.txt \
         models/robbyant_lingbot-vla-v2-6b/model.safetensors.index.json; do
  [ -e "/models/robotwin-persistent/$p" ] && echo "OK   $p" || echo "MISS $p"
done

# ④ aiter 坑是否复发（重建必查，见 §2）
/opt/robotwin-env/bin/python -c "import aiter" 2>&1 | tail -1

# ⑤ 网络与资源（决定恢复方式）
for u in https://github.com https://huggingface.co; do timeout 8 curl -sI -o /dev/null -w "$u %{http_code}\n" $u; done
#   cpu1 实测：**出站全断** ⇒ 只能从外部推入
cat /sys/fs/cgroup/memory.max                    # 实测重建后 = 2 GiB，需向平台申请放宽到 80 GiB
cat /sys/fs/cgroup/cpu.max                       # 实测 = 1600000 100000 ⇒ **16 核配额**（nproc 报 208 是宿主机）
df -h / /workspace                               # /workspace ~98G 持久；/ overlay 2.7T
```

---

## 1. 恢复路线选择：**先看有没有 Devzone 卷**

| 情况 | 做法 | 耗时 |
|---|---|---|
| **A. 挂载了 Devzone**（`Mount a model = Devzone`） | 数据/模型/资产**零传输**，只需搬代码（§3.1） | **~5 分钟** |
| **B. 没挂**（本次就是这种） | 必须外部推入模型 + 数据（§4），或**重建实例并勾选 Devzone** | 1–2 小时（取决于链路） |

> ⚠️ **Devzone 是实例创建时的选项**（`Customize → Mount a model → Devzone`）；老实例改不了。
> 平台会把内容挂到 `/models/robotwin-persistent/{assets,data,models}`，`/RoboTwin/{assets,data}` 是指向它的软链。

---

## 2. ★ aiter gluon / triton 硬失败（重建后必现，一行环境变量修好）

**症状链**（一个根因串起"训练起不来"的所有表象）：

```
/opt/aiter/.../gluon/__init__.py 校验 triton>=3.6.0，镜像是 3.5.1 ⇒ raise RuntimeError
  ⇒ flash_attn 导入失败（硬依赖 aiter）⇒ lingbotvla 模型模块导入失败（注册表架构数 = 0）
    ⇒ get_loader() 退回 HuggingfaceLoader ⇒ ValueError: Unrecognized configuration class
```

**修法（零侵入，不改文件）**：

```bash
bash tools/rocm/fix_aiter_gluon_triton.sh          # 幂等；写 /etc/profile.d + ~/.bashrc 并当场验证
# 或手工：
export AITER_USE_SYSTEM_TRITON=1
```

**验证**（三条都要过）：

```bash
AITER_USE_SYSTEM_TRITON=1 /opt/robotwin-env/bin/python -c "import aiter, flash_attn; print('OK')"
cd /workspace/lingbot_vla_2/lingbot-vla-v2 && AITER_USE_SYSTEM_TRITON=1 PYTHONPATH=$PWD \
  /opt/robotwin-env/bin/python -c "from lingbotvla.models.registry import get_registry; print(len(get_registry().supported_models))"
#   期望 2（`pi0` 的 ImportError(lerobot.common.policies) 是已知无害项）
```

> 为什么重建后会复发：修复改的是 `/opt/aiter`（**容器可写层**），重建即丢。

---

## 3. 代码与模型落位

### 3.1 代码（从任意可达来源搬，~1 分钟）

```bash
# 本机侧（推荐：仓库已有全部提交）
cd <本地仓库> && git bundle create /tmp/lingbot_vla_2.bundle --all      # ~35 MB
rsync -a -e 'ssh -p <node_port>' /tmp/lingbot_vla_2.bundle root@<host>:/workspace/
# cpu1 侧
cd /workspace && git clone lingbot_vla_2.bundle lingbot_vla_2 \
  && cd lingbot_vla_2 && git checkout feature/rocm-adapt

# 参考仓库（数据准备脚本 + ROCm 补丁，264 KB）
tar czf /tmp/radeon-cloud.tgz -C refs robotwin-radeon-cloud
rsync -a -e 'ssh -p <port>' /tmp/radeon-cloud.tgz root@<host>:/workspace/
ssh -p <port> root@<host> 'mkdir -p /workspace/refs && tar xzf /workspace/radeon-cloud.tgz -C /workspace/refs/'
```

### 3.2 模型（**训练/开环要 bf16，不是 fp32、也不是 robotwin SFT**）

配置里的权威路径（`configs/rocm/robotwin_official_paths_rocm.yaml`）：

| 用途 | 路径 | 说明 |
|---|---|---|
| 训练起点（`model_path`） | `/workspace/models/robbyant_lingbot-vla-v2-6b-bf16` | **bf16**，由 base(fp32) 转换而来 |
| 深度对齐 | `/models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b/depth/{moge2-vitb-normal.pt,model.pt}` | 1.75 G |
| 视频塔 | `…/robbyant_lingbot-vla-v2-6b/dino_video/{teacher_step_10000.pth,config.yaml}` | 1.4 G |

**bf16 转换（省一半传输量：12 G vs 30 G）**：

```bash
# 在有 torch+safetensors 的机器上（AutoDL ctrl 实测：/root/miniconda3/bin/python，需 pip install safetensors）
python tools/make_bf16_ckpt.py --src <base-fp32-hf_ckpt> --dst <new_dir> --dtype bfloat16 --verify full
#   实测：30 G → 12 G，2–3 分钟；**需要内存 ≥ 8 GiB**（2 GiB 的 cgroup 会被 OOM Killed）
# 落位到 cpu1 后补软链（避免重复传 depth/dino_video/assets）：
ln -sfn /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b/depth      <bf16_dir>/depth
ln -sfn /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b/dino_video <bf16_dir>/dino_video
```

> ⚠️ 别用 `outputs/**/hf_ckpt_bf16`（虽也是 BF16/12 G）——那是**训练过的 checkpoint**，
> 作 AL 起点会改变训练语义。

---

## 4. 数据搬运（唯一的大头）

### 4.1 通道事实（实测）

| 事实 | 值 |
|---|---|
| cpu1 **出站全断** | 只能"外部推入" |
| 源机（AutoDL ctrl）→ cpu1 | **可直连**；把源机公钥装进 cpu1 `~/.ssh/authorized_keys` 即可免密 |
| 速率波动 | 同一小时内 **14.9 → 2 → 17 → 21 MB/s**（国际链路时段性） |
| 并发 | **不增反降**（3 路 5.3 MB/s < 单流 14.9）⇒ **单流串行** |
| 小文件 | 极慢（1227 个文件 3.8 MB/s）⇒ **先 tar** |

**装免密（本机侧，一次性）**：

```bash
# 取源机真实公钥（★ 必须从私钥反推；.pub 可能是别的一对）
ssh ctrl 'ssh-keygen -y -f /root/.ssh/id_ed25519' > /tmp/ctrl_real.pub
# 写入 cpu1（★ 用 base64 传，避免引号/换行踩坑；★ 确认最后一行有换行，否则会与上一条粘连）
B64=$(base64 < /tmp/authorized_keys.new | tr -d '\n')
ssh -p <port> root@<host> "echo '$B64' | base64 -d > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys"
# 验证
ssh ctrl 'ssh -o BatchMode=yes -p <port> root@<host> "hostname; echo OK"'
```

### 4.2 必传清单与落位

| 项 | 体积 | 目标 |
|---|---|---|
| bf16 权重 | 12 G | `/workspace/models/robbyant_lingbot-vla-v2-6b-bf16/` |
| RoboTwin 资产（含 `objects/objaverse/list.json`） | 9 G | `/models/robotwin-persistent/assets/` |
| Qwen3-VL-4B-Instruct | 8.3 G | `…/models/` |
| MoGe-2-ViT-B | 0.4 G | `…/models/moge-2-vitb-normal/` |
| **`demo_clean`（50 任务原始数据）** | 33 G | `…/data/demo_clean/` |
| ~~base fp32 30 G~~ / ~~robotwin SFT 24 G~~ | — | **不需要**（训练/开环用 bf16） |

### 4.3 压缩：**单路 xz 不值得，8 路并行 gzip 才值得**（实测）

| 方法 | 吞吐 | 33 G 耗时 | 结论 |
|---|---|---|---|
| 不压缩直传 | — | 31 min | 基准 |
| `xz -1 -T8` | **17 MB/s** | **32 min** | ❌ 与直传持平 |
| **8 路并行 `gzip -1`** | **~208 MB/s** | **2.6 min**（→22 G） | ✅ 净省 ~7–10 min |

**关键认知**：压缩速度受 **cgroup CPU 配额**限制（实测 16 核），与内存无关；
`nproc` 报 208 是**宿主机**核数，容器拿不到。

**并行压缩脚本**（每任务一个独立 `.tgz`，避免多进程写同一流的 gzip 交错损坏）：

```bash
find <DATA>/demo_clean -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | \
  xargs -r -P 8 -I{} bash -c 'tar cf - -C <DATA>/demo_clean "$1" | gzip -1 -c > <OUT>/$1.tgz' _ {}
# 传输后再 `for f in *.tgz; do tar xzf $f -C <DEST>; done`
```

---

## 5. 数据转换（把 demo_clean 变成训练要的 50 份 LeRobot + 清单）

```bash
# 依赖（重建后确认）：lerobot 0.6.0（/opt/lerobot-env）、ffmpeg、transform 脚本
ls /opt/lerobot-env/bin/python /RoboTwin/XPolicyLab/scripts/transform_lerobot_v30_format.py
which ffmpeg

# 跑官方转换脚本（内部 50 路并行；同时产出清单）
cd /RoboTwin && bash /workspace/refs/robotwin-radeon-cloud/docker/full/convert_all_data.sh
#   产出：/RoboTwin/data/lerobot/<task>_joint_v30/（50 个）
#         /RoboTwin/data/robotwin_demo_clean_joint_v30.txt（★ 清单由脚本尾部自动生成，不是现成文件）
```

> ⚠️ `/RoboTwin/data` 是指向 `/models/robotwin-persistent/data` 的软链 ⇒ 落盘位置正确。
> ⚠️ 脚本引用 `/usr/local/bin/convert-all-robotwin-data`（仅用于算缓存键）在重建后可能缺失；
> 缺失时可用占位文件替代，不影响转换本身。

---

## 6. 起训练（配方已真机验证）

```bash
# 前置：无残留训练进程（launcher 会以退出码 2 自锁保护）
ps -eo args | grep "[t]rain_lingbotvla.py /workspace"      # 必须为空

/opt/robotwin-env/bin/python -u experiment/robotwin/al_launch.py \
  --run-name al_vNN --steps 5000 --micro 5 --gas 1 \
  --gpus 0,1,2,3,5,6,7 \
  --hardness-cache-file /workspace/al/hardness_cache/hardness.json \
  --scout-cache-file    /workspace/al/scout_cache/scout.json \
  --model-name          robbyant_lingbot-vla-v2-6b-bf16
```

**四条前置条件（缺一不可）**：
1. `configs/rocm/robotwin_official_paths_rocm.yaml` 的 **`train.use_compile: false`**（开编译会首步死锁）；
2. `--model-name` 与缓存文件里的 `model` 一致；
3. 排除坏卡 **GPU[4]**（`--gpus 0,1,2,3,5,6,7`）；
4. `AITER_USE_SYSTEM_TRITON=1`（§2）。

**实测基线**：首步 ~150 s（含加载，正常）；稳态 **10.6–11.9 s/it**；5000 步 ≈15.3 h（含每 50 步评测 ≈21.5 h）；
峰值显存 48.9/49.1 GiB；scout 50/50 命中、hardness 命中 **4.2 s**（首扫 208.5 s）。

---

## 7. ★ 派生产物：**先查现成的，再决定要不要重算**（2026-10-11 血的教训）

重建后**缺的往往不是源码，而是"由数据集算出来的派生数据"**。它们算起来很贵，
而且**换台机器/换个目录就找不到**了。**默认动作 = 先找，找不到再算。**

| 派生产物 | 落位 | 重算代价 | 能否重算 |
|---|---|---|---|
| `task_splits_50/manifest.json` + 104 个 `*_ids.json` | `/workspace/al/task_splits_50/` | 秒级（需聚合数据集） | ✅ |
| **`task_baseline.json`**（每任务训练前 MSE） | 同上 | **~90 s/任务 × 50 ≈ 75 分钟（纯 CPU）** | ✅ |
| **`scout_cache/scout.json`**（50/50） | `/workspace/al/scout_cache/` | **4 分 44 秒**（8 分片首扫） | ✅ |
| **`hardness_cache/hardness.json`** | `/workspace/al/hardness_cache/` | **单卡 ~1000 s/任务**（8 卡 rank 分片 ~1/8） | ✅ |
| `phases_al/datasets.txt` | `/workspace/al/phases_al/` | 秒级（**但必须知道"要单行聚合数据集"**） | ✅ |
| **`pass_thresholds_gmean100_warn.json`** | `/workspace/eval_results/open_loop/ref50k/` | 秒级 | ✅（需下一行的参考数据） |
| **`ref_per_traj.jsonl`（50k 参考逐轨迹 MSE）** | 同上 | — | ❌ **不可再生**（需 50k 参考模型） |

### 恢复方式（已入库，一条命令）

```bash
# 包内 109 个文件 / 约 216 KB，带 sha256 清单与恢复脚本
# 仓库根即 lingbot-vla-v2（内含 refs/al_artifacts/）
bash refs/al_artifacts/restore.sh --local
# 或从开发机推：
bash refs/al_artifacts/restore.sh root@<host> <port> <本机私钥路径>
```

脚本会落位 + `sha256sum -c` 校验 + 打印**关键不变量**（见下）。详见 `refs/al_artifacts/README.md`。

### 三条必须先核对的不变量（否则**静默错判**，程序不报错）

| # | 不变量 | 违反后果 |
|---|---|---|
| 1 | `task_baseline.json` 与 `pass_thresholds_*.json` 的 **`config_fingerprint` 必须相同** | baseline 与通过线错位 ⇒ PASS/FAIL 判错且无告警 |
| 2 | 缓存的 **`model` 字段 = `--model-name`** | 不一致 ⇒ 缓存全部判未命中（白扫一遍） |
| 3 | `datasets.txt` **必须单行**（聚合数据集） | 多行 ⇒ `resolver` 报 `下钻不到 hf_dataset`（训练起不来） |

**缓存文件格式**：单文件 JSON；scout 为 `{"version":2,"model":"…","records":{…}}`，
hardness 为 `{"version":2,"noise_semantic_version":1,"model":"…","tasks":{…}}`；
**可用性只由 `model`（+hardness 的噪声语义版本）决定**（改代码不会失效）。

---

## 7.1 历史缓存代价（供对比）

| 缓存 | 首扫代价 | 命中代价 |
|---|---|---|
| scout（50 任务预检） | ~5 分钟（并行） | 秒级 |
| hardness（按任务） | 历史：**~208 s/任务**（259 样本，**7 卡合计口径**）；单卡实测 ~0.17 样本/s | 命中 **4.2 s** |
| 编译缓存 | 已弃用（`use_compile: false`） | — |

---

## 7.2 ★★ 外部存档机（2026-10-11 建立）：**重建后照搬数据即可**

> 动机：本机 `/models/robotwin-persistent` 是 overlay（重建即丢），`/workspace` 虽为持久卷但
> **本次重建也被清空过** ⇒ 需要一个**第三台、与实例生命周期无关**的存档点。

| 项 | 值 |
|---|---|
| 存档机 | `ssh -p 15570 root@jq1.9gpu.com`（主机名 `gpu-kvm`；Ubuntu 22.04；`/data` 49 G + `/` 59 G） |
| 免密 | 开发机 ↔ 存档机 ✅；**存档机 ↔ cpu1** ✅（`gpu-kvm-archive` 公钥在 cpu1 `~/.ssh/authorized_keys`） |
| 传输方向 | 存档机 **pull** cpu1（**不经开发机中转**：直连实测 8.9–21 MB/s，中转仅 0.58 MB/s） |
| 体积 | 全部必需件约 **71 G**（存档机可用 80 G） |
| 幂等 | 全部 `rsync -a --partial`，中断重跑即续传 |

### 存档内容与落位

| 存档路径（存档机上） | 体积 | 对应训练机路径 |
|---|---|---|
| `/data/lingbot_archive/demo_clean` | 32.7 G | `/models/robotwin-persistent/data/demo_clean` |
| `/data/lingbot_archive/assets` | 9 G | `…/assets` |
| `/data/lingbot_archive/Qwen3-VL-4B-Instruct` | 8.3 G | `…/models/Qwen3-VL-4B-Instruct` |
| `/data/lingbot_archive/lerobot` | 4.7 G | `…/data/lerobot`（50 个 `<task>_joint_v30`） |
| `/root/lingbot_archive/robbyant_lingbot-vla-v2-6b-bf16` | 12 G | `/workspace/models/…-bf16`（**训练起点**） |
| `/root/lingbot_archive/{depth,dino_video}` | 3.1 G | `…/models/robbyant_lingbot-vla-v2-6b/{depth,dino_video}` |
| `/root/lingbot_archive/moge-2-vitb-normal` | 0.4 G | `…/models/moge-2-vitb-normal` |
| `/root/lingbot_archive/{al,eval_results,al_runs,al_cache}` | ~13 G | `/workspace/al` · `/workspace/eval_results` · `…/al_runs` · `…/al_cache` |
| `/root/lingbot_archive/ws/` | 108 M | **`/workspace` 全量镜像**（含代码工作树 `.git`、`refs/`、`RoboTwin-lingbot`、bundle、runtime） |
| `/root/lingbot_archive/extras/` | 0.85 M | 工作区根参考件（`refs/robotwin-*`、`.workbuddy` 记忆）+ 未入库仓库文件 |

**有意未存档**：`6b-robotwin`(5.8 G，不用) · `agg_lerobot_v30`(3.9 G，可由 zip 解) ·
`RoboTwin_lerobot_v30.zip`(3.9 G，ctrl 有) · `demo_clean.tar.xz`(0 字节残片) · `/models/rw`(7.5 G 残片) ·
base fp32(27 G，ctrl 有；训练用 bf16) · `.ssh` 私钥。

### 恢复（一条命令）

```bash
# 在存档机上执行；自动 rsync 13 项回目标机 + 重建 5 条软链 + 打印后续 3 步
bash /root/archive_restore.sh root@<新机IP> <端口> /root/.ssh/id_ed25519
```

脚本会自动重建：`/RoboTwin/{data,assets}` 软链 · bf16 的 `depth`/`dino_video` 软链 ·
`/workspace/models/Qwen3-VL-4B-Instruct-config-tokenizer` 软链。

### 「对齐」操作口径（用户 2026-10-11 定：**直接按文件对照，不管 git**）

> **有什么传什么**；不比 commit、不要求提交；以**文件内容**为准。

用户说一句「对齐」⇒ ① `rsync -navc --delete` 逐文件对照（`-c` 按内容校验，避免时钟差异误判）
② 打印差异清单（新增/修改/删除+体积）给他过目 ③ 点头后去掉 `-n` 只传差异 ④ 报结果。
优先对照 4 处：`/workspace/lingbot_vla_2`（代码）· `/workspace/al`（派生产物）·
`/workspace/eval_results`（阈值表）· `…/al_runs`（训练产物）。

### 存档机上的三个脚本 + 说明

| 文件 | 用途 |
|---|---|
| `/root/archive_pull_mk2.sh` | 13 项大件+小件拉取（修正落位：/data 与 /root 分池，避免溢出） |
| `/root/archive_pull_ws.sh` | `/workspace` 全量镜像（直接文件对照，排除 models/、_spd.bin、lost+found） |
| `/root/archive_restore.sh` | **一条命令推回新机 + 重建软链** |
| `/root/ARCHIVE_README.md` | 完整说明（内容/落位/未存档项/恢复三步/对齐契约/通讯录） |

---

## 8. 运维红线（本次踩过，代价真实）

1. **清理/杀进程一律按显式 PID 列表**；`pkill -f "<子串>"` 会匹配到**执行它的那条命令/父链**，
   导致会话被打断（本次发生 4 次）。需要模式匹配时用**脱离会话的脚本** + 排除自身 PID。
2. **新 run 启动期间绝不清理进程**（本次误杀过一次正在启动的训练）；
3. **launcher 退出码语义**：`2` = 检测到别的训练在跑（自锁保护，不是 bug）；`3` = 缓存覆盖不完整；
   `6` = 内部异常；
4. **只读诊断也可能"自匹配"**：`pgrep -f "train_lingbotvla.py /workspace"` 在远端会匹配到
   **自己的 shell**（命令文本含该串）⇒ 判活要用 `argv[2] == 'tasks/vla/train_lingbotvla.py'`；
5. **cgroup 限额会被平台重置**（本次内存 2 GiB → 80 GiB）⇒ 重建后**重测**再决定能否跑重任务；
6. **`/workspace` 虽名为持久卷，重建后也可能被重置**（本次 98 G 全空）⇒ 关键产物要外部留一份。

---

## 9. 一页速查（下次重建照抄）

```bash
# 0) 体检
findmnt -T /models/robotwin-persistent            # overlay ⇒ 没挂卷，要搬数据
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max

# 1) 修 aiter（必做）
bash tools/rocm/fix_aiter_gluon_triton.sh

# 2) 搬代码（~1 min）
rsync /tmp/lingbot_vla_2.bundle root@<host>:/workspace/ && ssh … 'cd /workspace && git clone …'

# 3) 搬 bf16 模型（12 G）+ 资产（9 G）+ Qwen（8.3 G）+ demo_clean（33 G）
#    源机免密直推；单流串行；demo_clean 用 8 路并行 gzip 压到 22 G 再传

# 4) 转换 + 起训练
bash /workspace/refs/robotwin-radeon-cloud/docker/full/convert_all_data.sh
/opt/robotwin-env/bin/python -u experiment/robotwin/al_launch.py --run-name al_vNN \
  --steps 5000 --micro 5 --gas 1 --gpus 0,1,2,3,5,6,7 \
  --hardness-cache-file /workspace/al/hardness_cache/hardness.json \
  --scout-cache-file /workspace/al/scout_cache/scout.json \
  --model-name robbyant_lingbot-vla-v2-6b-bf16
```
