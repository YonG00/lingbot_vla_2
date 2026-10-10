# Auto Learning 启动程序使用说明（`experiment/robotwin/al_launch.py`）

面向使用者。看完这一页就能自己起 50 任务的 Auto Learning 训练，不用再手搓 7 条
`CUDA_VISIBLE_DEVICES=… N_GPU=1 MAX_STEPS=1 …` 的 worker 命令。

---

## 0. 一句话：它解决什么问题

**缓存齐了就直接开训（秒级）；缺哪几个就只并行扫哪几个；显式要求才全量重扫；覆盖不完整时
不启动训练，并以非零退出码点名缺失任务。**

它把今天手工做的三件事固化成一条命令：

1. 算当前代码/权重/数据的**指纹**（= scout 缓存目录名）；
2. 按指纹检查缓存覆盖，**只并行补齐缺失的任务**；
3. 覆盖确实完整（以指纹目录里的条目数为准）之后，才后台启动正式训练。

---

## 1. 快速开始（三条命令）

```bash
cd /workspace/lingbot_vla_2/lingbot-vla-v2

# ① 直接训练（有缓存）：查覆盖 → 齐了就直接起训练；缺了就只补缺失的那几个
/opt/robotwin-env/bin/python experiment/robotwin/al_launch.py

# ② 忽略现有缓存：全量并行扫描 50 个任务，扫完再启动训练
#    （旧指纹目录会被改名备份成 <指纹>.bak-<时间戳>，不删除）
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
[cache] 该目录下 4 个条目文件；覆盖 4/6 个任务
[plan] 待扫 2 个任务 ⇒ 2 片：片0=1, 片1=1
[plan] 分片并集校验：通过（无重复、无遗漏，合计 2 = 全集 2）
[ready] T+30s：进程存活 2/2；已就绪 2/2（就绪数与各分片任务数相符）
[progress] T+3m20s | 片0(gpu0) 1/1 | 片1(gpu1) 1/1 | 合计 50/50（指纹目录内）
范围覆盖      : 2/2
全集覆盖      : 50/50（该目录内 50 个条目文件）
缺失清单      : 无
结论：覆盖完整（50/50）⇒ 启动训练。
```

覆盖不完整时（**绝不打印假的 50/50**）：

```text
全集覆盖      : 44/50（该目录内 44 个条目文件）
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
| `--no-cache` | 关 | 忽略现有缓存，**全量并行扫描**后再训练 | 改了被指纹纳入的代码/数据，且想强制重扫时 |
| `--dry-run` | 关 | 只打印步骤计划与环境变量，不启动任何进程 | 上机器前先核对路径/配置；排障 |
| `--json` | 关 | stdout 只输出 JSON（人话日志走日志文件与 stderr） | 被别的脚本调用、要机器可读结果时 |
| `--selfcheck` | 关 | 与仓库 `scout_cache.py` 逐项对拍指纹实现 | 每次升级本脚本/改动指纹相关代码后自检 |
| `--gpus` | `0,1,2,3,5,6,7` | 可用卡（逗号分隔）；会同时显式导出 `CUDA_VISIBLE_DEVICES` 与 `HIP_VISIBLE_DEVICES` | 换机器、换可用卡时。**4 号卡已挂死**，除非确认换机，否则别把它加回来 |
| `--eval-config` | `configs/auto_learning/al_eval2.yaml` | AL 配置（相对仓库根） | 默认文件不存在，或想换配方（如 `al_50task_gmean100_rocm.yaml`）时 |
| `--steps` | `200` | 训练 `MAX_STEPS` | 换训练预算时（正式长跑一般给很大的值） |
| `--micro` / `--gas` | `12` / `1` | 传给启动脚本的 `MICRO` / `GAS`；脚本自算 `GBS = MICRO*GAS*N_GPU` | 显存不够或要改全局批大小时。**AL 要求 `micro*gas == AL 配置的 batch_size`**，否则启动即报错 |
| `--cache-root` | `/workspace/al/scout_cache` | scout 缓存根目录（子目录名就是指纹） | 缓存换盘/换位置时 |
| `--fingerprint` | 空 | 显式 64 位指纹，**跳过指纹计算** | 已经知道要复用哪份缓存；或机器上算指纹太慢时 |

### 2.2 路径类

| 参数 | 默认值 | 什么时候需要改 |
|---|---|---|
| `--repo` | 本脚本的上一级目录 | 一般不用改（脚本自定位仓库根） |
| `--python` | `/opt/robotwin-env/bin/python` | 训练环境解释器换位置时。**建议直接用这个解释器启动本脚本**，否则本脚本会自动改用该解释器做一次子进程指纹计算 |
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
| `--scout-trajs` | 取 AL 配置的 `global_scout_val_trajs`（一般 2） | 只想改 scout 回合数时（**会改变指纹**） |
| `--image-augment` | 关（= `false`） | 一般别开：启动脚本固定 `--data.image_augment false`，开了就会与缓存指纹不一致 |
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

### 3.1 缓存目录名就是指纹

```text
/workspace/al/scout_cache/
├── 3c8b15e2…（64 位）/      ← 一个指纹 = 一份「权重 + 评测链源码 + 数据/配置 + 评测选项」的联合快照
│   ├── <scout_key>.json      ← 一个条目 = 一个任务在固定 scout 回合上的评测结果
│   └── …
└── cd5f5c0f…（64 位）/
```

* 条目文件名 `scout_key(task, episode_ids) = sha256(json([task, ids]))`；条目内容含
  `{version, fingerprint, task, episode_ids, metrics}`。
* 只有 rank0 写缓存（多卡并发写会撕裂）；读不受影响。
* **不存在的指纹目录会被创建**（等效「空缓存 + 全量重扫并写进去」）。

### 3.2 指纹由什么决定

| 类别 | 具体内容 |
|---|---|
| 权重 | checkpoint 目录下所有 `*.safetensors` 的**内容** SHA256（不是文件名/mtime） |
| 评测链源码 | 12 个必查文件：`open_loop_validation.py`、`modeling_lingbot_vla_v2.py`、`transform.py`、`dataset.py`、`multi_vla_dataset.py`、`base_dataset.py`、`utils.py`、`eval_precision.py`、`evaluator.py`、`gmean.py`、`scan_accel.py`、`eval_batch_policy.py`；另有 2 个可选文件（`ee_pose_transform.py`、`video_utils.py`）存在时一并纳入。`.py` 用 **AST 语义 hash**：改注释/空行/docstring 不失效，改逻辑必失效 |
| 数据/配置 | `manifest.json`、norm stats、阈值表、`task_baseline.json`、checkpoint 的 `config.json` / `tokenizer.json` |
| 评测选项 | `inference_dtype`、`noise_seed=1234`、`scout_trajs`、`stride=per_episode`、`image_augment=false` |

**刻意不纳入指纹**：AL 配置文件本身（这样 7 份分片配置与正式 run 的主配置共享同一个指纹）、
以及启动脚本里的 `MAX_STEPS` / `N_GPU` 之类的运行参数。

### 3.3 并行扫描为什么比进程内 bootstrap 快

```text
进程内 bootstrap（默认慢路径）
  7 卡 FSDP2 同步扫 50 个任务 ≈ 18 分钟
  任一 rank 掉队 ⇒ 全组一起等（集合通信）⇒ 一个坏卡拖死一整轮

并行扫描（本工具的做法）
  任务全集按 tasks[i::N] 切成 N 片（N = 可用卡数）
  每片起一个独立单卡进程：CUDA_VISIBLE_DEVICES=<单卡> / HIP_VISIBLE_DEVICES=<单卡> / N_GPU=1
                            MAX_STEPS=1 / AL_CFG=<该片的 task_names 分片配置>
                            AL_SCOUT_CACHE_FINGERPRINT=<当前指纹>（各 worker 与正式 run 必然指向同一个目录）
  50 个任务 ≈ 5 分钟；进程之间没有集合通信，天然失败隔离
```

* **无跨进程集合通信**：每个 worker 自己持完整模型（单卡放得下 bf16 的 6B），只在最后
  写一条 JSON；一个 worker 慢/挂不会让其他 worker 停。
* **失败隔离**：某个分片挂了，本工具会让你「缺失清单 + 非零退出码」看到，而不是静默把
  整轮拖死；再跑一次只会补那几个缺失任务。
* **编译缓存共用**：`TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` 指向同一个目录，
  第一个 worker 编译，其余命中缓存。

### 3.4 一次运行里的 15 步（与日志一一对应）

| 步骤 | 日志里会看到 |
|---|---|
| 1 静态校验 | `卡 0,1,2,3,5,6,7（7 张）；steps=…；micro=… gas=… ⇒ GBS=…` |
| 2 路径检查 | `[missing] …`（dry-run 只警告；实跑直接拒绝） |
| 3 AL 配置 | `enabled` / `pass_metric` / `scout_trajs` / 阈值表；`pass_metric != gmean_mse` 直接拒绝 |
| 4 任务全集 | `N 个任务（来源 manifest）；每任务 scout 2 条 val 回合` |
| 5 并发占用检查 | 发现别的 `train_lingbotvla.py` / `al_launch.py` 就拒绝（除非 `--allow-busy`） |
| 6 运行环境 | `TMPDIR=…`；TMPDIR 是 `/tmp` 直接拒绝 |
| 7 指纹 | `[fingerprint] <64 位>`；`--fingerprint` 时跳过计算 |
| 8 覆盖检查 | `[cache] … 覆盖 47/50 个任务` ⇒ 决定 `reuse` / `incremental` / `full-scan` |
| 9 分片规划 | `[plan] 待扫 3 个任务 ⇒ 3 片：片0=1, 片1=1, 片2=1` + **并集校验通过** |
| 10 环境变量 | 把 worker 模板与训练要用到的**全部**环境变量逐行打印 |
| 11 启动 worker | `[scan] 片0 GPU0：1 任务；pid=…；日志 …`（每片一行） |
| 12 就绪检查 | `[ready] T+30s：进程存活 7/7；已就绪 7/7` |
| 13 逐分片进度 | `[progress] T+3m20s | 片0(gpu0) 7/7 | … | 合计 47/50（指纹目录内）` |
| 14 收尾统计 | `范围覆盖` / `全集覆盖` / `缺失清单` |
| 15 启动训练 | `日志 / PID / 进程数 / 输出 / 配置` |

### 3.5 「覆盖」的判定口径

一个任务算「已覆盖」，当且仅当指纹目录里存在它的条目文件，且该条目会被
`BootstrapScoutCache.load()` **真的当成命中**返回：schema 版本、指纹、任务名、回合集合、
逐轨迹 MSE 与 `nmse == mse/baseline` 的自洽关系、`metric_valid=true`、`gmean_mse` 有限，
全部核对通过。

因此「覆盖 50/50」是可以放心直接开训的；反之，坏条目、旧指纹条目、回合集合不同的条目
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

### 4.3 缓存目录名就是指纹；改代码会让缓存失效

```bash
ls -d /workspace/al/scout_cache/*/          # 每个目录名就是一个完整指纹
```

改了 §3.2 里任何一个文件（例如 `open_loop_validation.py` 加了一行逻辑），指纹就变了，
旧目录**不会再被读到**：默认模式会自己发现「新指纹目录 0/50」并并行重扫（约 5 分钟），
不需要你手工删任何东西。改注释/空行/docstring 不会让缓存失效。

### 4.4 如何「只看不动」地检查缓存覆盖情况

```bash
cd /workspace/lingbot_vla_2/lingbot-vla-v2

# ① 看有哪些指纹目录（目录名就是完整指纹，前 8 位够认人）
ls -d /workspace/al/scout_cache/*/

# ② 只读检查某个指纹的覆盖情况：--dry-run + --fingerprint，不启动任何进程、不写任何文件
/opt/robotwin-env/bin/python experiment/robotwin/al_launch.py --dry-run \
  --fingerprint <64 位指纹>
# 输出：
#   [cache] 该目录下 47 个条目文件；覆盖 47/50 个任务
#   [plan] 运行模式：incremental
#   [plan] 待扫 3 个任务 ⇒ 3 片：片0=1, 片1=1, 片2=1
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
| `3` | **扫描未完成**：目标指纹目录里仍有缺失任务（已打印缺失任务名） |
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
5. **`--no-cache` 不删数据**：旧指纹目录只是改名成 `<指纹>.bak-<时间戳>`，确认无误后可自行删除。
