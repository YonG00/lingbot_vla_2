# 多 checkpoint 多卡闭环评测调度器 使用文档

> 适用版本：`YonG00/lingbot_vla_2`（LingBot-VLA-v2 post-training）
> 相关文件：`experiment/robotwin/robotwin_multi_ckpt_eval.py`（调度器，新增）
> 相关改动：`experiment/robotwin/start_robotwin_infer_and_eval.sh`（官方 launcher，加 3 个可选参数）
> 相关测试：`tests/test_multi_ckpt_eval.py`（22 项）、`tests/test_launcher_task_list.sh`（27 项）

---

## 1. 这个功能解决什么问题

训练会周期性产出 checkpoint（`global_step_5000` / `10000` / …），每个都要评测才知道好坏。
但**官方 launcher 一次只吃一个 `--model_path`**，N 个 checkpoint 就得手动跑 N 次：

- **费人力** —— 每次都要改路径、看进度、记结果
- **占不满卡** —— 一次开卡按小时计费，串行跑就是白烧钱
- **容易漏** —— 忘了评某个 ckpt，或者评到一半卡被回收
- **无对照** —— 结果散落在各个 `stats.txt` 里，没有一张总表

**本功能的目标**：一条命令评测**本次训练产出的所有新 checkpoint**，
把多个 checkpoint **并发**塞进 4 张卡，自动汇总成一张可比对的表。

---

## 2. 一句话原理

> **不重写官方评测**，只做编排：发现 checkpoint → 按并发上限切成滑窗 →
> 给每个并发位分配**互不重叠的端口段** → 各自调一份官方 launcher →
> 谁先完成立刻补下一个 → 汇总成功率。

⚠️ 评测本身（推理 server + 仿真 client + websocket）**完全走官方链路**，
所以分数与官方口径一致，不是另起炉灶的自研评测。

---

## 3. 前置条件（必读）

### 3.1 必须导出 `QWEN3VL_PATH`

官方 launcher 的默认值是**占位符**，调度器不会替你设：

```bash
export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct
```

漏了这条，推理 server 起不来（报找不到 backbone）。

### 3.2 仿真侧需要已重编过 sm_120

Blackwell 卡（RTX PRO 6000，sm_120）需要 curobo 的 5 个 `.so` 已按 sm_120 重编。
**换机器或重装 RoboTwin env 后必须重做**。自检：

```bash
python -c "from envs.robot.planner import CuroboPlanner; print('curobo OK')"
```

### 3.3 任务清单要先存在

```bash
python tools/prepare_phase.py --phase all      # 生成 /data/train/phases/phase{1..4}_eval.txt
```

清单是**前缀累积**的：P1 = L1（4 个任务），P2 = L1+L2（8 个），P3 = 12 个，P4 = 16 个。

### 3.4 显存上限：4×96G 最多并发 **2 个** checkpoint

官方多卡**不是模型并行**，是「多副本 + 任务级并行」—— 每个 slot 是一个独立进程，
各自**完整加载一遍**模型，所以显存是 **×N**。

- 一个 checkpoint 用 `--num-gpus 4 --num-per-gpu 1` = 每卡 1 份完整模型
- 2 个 checkpoint 并发 = 每卡 **2 份 ≈ 62.5 GB**（实测），无压力
- 3 个 ≈ 94 GB/卡 → 逼近 96G，**不可行**

⇒ `--max-parallel-checkpoints 2` 是这台机器的**实际上限**，不是保守取值。

---

## 4. 快速开始

### 4.1 先 dry-run 看计划（不占卡、不起进程）

```bash
cd /data/code/lingbot-vla-v2
source /data/miniconda3/etc/profile.d/conda.sh && conda activate lingbotvla

python experiment/robotwin/robotwin_multi_ckpt_eval.py \
  --ckpt-root /data/models --phase 1 --episodes 3 \
  --max-parallel-checkpoints 1 --num-gpus 4 --num-per-gpu 1 --dry-run
```

会打印：发现几个 checkpoint、哪些完整、增量分类（新增/更新/未变）、
每个并发位的端口段与 GPU、每个 checkpoint 的 `model_path`。

### 4.2 真跑

```bash
export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct

python experiment/robotwin/robotwin_multi_ckpt_eval.py \
  --ckpt-root <训练输出目录> --phase 1 --episodes 3 \
  --max-parallel-checkpoints 2 --num-gpus 4 --num-per-gpu 1 \
  --output-base /data/eval_results/multi_ckpt
```

### 4.3 训练结束后自动接评测

训练侧**没有**「存完即评」的代码钩子（`AsyncHFCheckpointSaver` 会丢弃 `eval_args`），
所以「自动」靠自己串：

```bash
python train_lingbotvla.py ... && \
python experiment/robotwin/robotwin_multi_ckpt_eval.py \
  --ckpt-root /data/outputs/phase1_L1 --phase 1 --episodes 3 \
  --max-parallel-checkpoints 2 --output-base /data/eval_results/phase1
```

- 训练**失败**（退出码非 0）→ 评测**不会**启动
- 训练成功但没有新 checkpoint → 评测打印「没有需要评测的 checkpoint」并**正常退出 0**
- 评测**失败**的 checkpoint **不写进** `eval_state.json` → 下次自动重试

---

## 5. 参数速查

### 评测什么

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ckpt-root` | 必填 | checkpoint 根目录，期望布局 `<root>/<exp>/checkpoints/global_step_<N>/hf_ckpt` |
| `--phase {1,2,3,4}` | — | 用 `<eval-list-dir>/phase<N>_eval.txt` 作为任务清单 |
| `--task-list-file` | — | 直接指定清单文件（与 `--phase` **二选一**） |
| `--eval-list-dir` | `/data/train/phases` | 阶段清单目录 |
| `--conditions` | `clean,randomized` | 逗号分隔；**同一个 ckpt 内串行** |
| `--episodes` | `3` | 每任务每 condition 的回合数，透传为 client 的 `test_num`；`0` = 不覆盖（用官方默认 100） |
| `--curriculum-yaml` | `configs/curriculum/robotwin_curriculum_v1.yaml` | **只**用于给报告标注 Level，缺失不影响评测 |

### 选择哪些 checkpoint

| 参数 | 默认 | 说明 |
|---|---|---|
| `--all` | 关 | 评测全部完整 checkpoint（默认只评测新增/更新的） |
| `--only-steps` | — | 只评测这些 step，如 `10000,20000` |
| `--min-age-seconds` | `0` | 最新 mtime 距今不足该秒数视为「仍在写入」→ 跳过。**默认 0 = 关闭** |
| `--state-file` | `<output-base>/eval_state.json` | 增量状态文件 |

### 调度

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max-parallel-checkpoints` | `2` | 同时评测的 checkpoint 上限（显存不够设 1） |
| `--num-gpus` | `4` | 每个 checkpoint 用几张卡 |
| `--num-per-gpu` | `1` | 每张卡几个 slot |
| `--start-port-base` | `9330` | 端口段起点 |
| `--output-base` | `/data/eval_results/multi_ckpt` | 结果根目录 |
| `--poll-seconds` | `2.0` | 滑窗轮询间隔 |

### 环境 / 精度

| 参数 | 默认 | 说明 |
|---|---|---|
| `--precision` | `fp32` | `fp32` = 发布复现设置；`bf16` 更快但分数不可直接比 |
| `--use-compile` | 关 | 开 = 每个 server 进程各编一次 `torch.compile`，启动显著变慢 |
| `--video` | 关 | 录评测视频（省 IO 与时间） |
| `--eval-workdir` | `/data/code/RoboTwin-lingbot` | ⚠️ **不要指向假的目录**，见 §10 |
| `--no-preflight` | 关 | 跳过并发前的共享文件同步 |
| `--dry-run` | 关 | 只打印计划，不起子进程、不初始化 CUDA |

---

## 6. 输出在哪看

### 6.1 一张总表：`<output-base>/summary.txt`

```
==============================================================================
  多 checkpoint 评测结果
==============================================================================
  ckpt 1 个 | 任务 4 | 回合/任务 3 | condition clean -> randomized

-- 总览 ----------------------------------------------------------------
  checkpoint                             clean    randomized   耗时
  ---------------------------------------------------------------------
  lingbot-vla-v2-6b-robotwin@50k 91.7% (11/12) 83.3% (10/12) 7分6秒
  ---------------------------------------------------------------------
  单元格 = 成功率 (成功/总回合); 耗时 = 该 ckpt 所有 condition 的墙钟之和。

-- 逐任务成功率 (行 = 任务[Level]) -------------------------------------
  c0 = lingbot-vla-v2-6b-robotwin@50k / clean
  c1 = lingbot-vla-v2-6b-robotwin@50k / randomized
  Level 取自 configs/curriculum/robotwin_curriculum_v1.yaml
  ------------------------------------------
  task[Lv]                 c0/cle     c1/ran
  ------------------------------------------
  ------------------- L1 -------------------
  click_alarmclock[L1]  100%(3/3)  100%(3/3)
  turn_switch[L1]        67%(2/3)   33%(1/3)
  lift_pot[L1]          100%(3/3)  100%(3/3)
  place_shoe[L1]        100%(3/3)  100%(3/3)
  合计                 92%(11/12) 83%(10/12)
  ------------------------------------------
  单元格 = 成功率(成功/总回合); 合计行 = 该 condition 总体。

-- 产物 (相对 --output-base) -------------------------------------------
  lingbot-vla-v2-6b-robotwin@50k / clean       <tag>/clean/<run_dir>
```

- **总览**：一眼看出哪个 checkpoint 更好
- **逐任务**：一眼看出**哪个任务/哪个 Level** 是弱项（课程训练的核心诉求）
- Level 来自课程 yaml 的 `skill_levels`（50 个任务全覆盖）；
  任务按 Level 分组，**有 ≥2 个 Level 时每组末尾会多一行 `Lx 小计`**

### 6.2 机器可读：`<output-base>/summary.json`

含每个作业的 `returncode` / `duration_s` / `success` / `episodes` / `overall_rate` /
`per_task`（逐任务表）/ `run_dir`，以及 `task_levels` 映射。

### 6.3 明细：每个作业一个 run_dir

```
<output-base>/<ckpt_tag>/<condition>/<exp>_<step>k_<task_config>_<时间戳>/
    stats.txt            # 官方 launcher 的汇总 + 逐任务表
    eval_logs/<task>.log # 每个任务的仿真日志（含每回合 step 进度）
```

launcher 的完整 stdout（含 server 启动日志）在：

```
<output-base>/_logs/<ckpt_tag>.<condition>.log
```

### 6.4 增量状态：`<output-base>/eval_state.json`

记录已成功评测过的 checkpoint 指纹。**失败的不会写进去**，所以重跑会自动重试。

---

## 7. 工作原理

### 7.1 滑窗补位（消除空档）

`max_parallel=2`、5 个 checkpoint 时，批次形状是 `[2, 2, 1]`，
但**不是等一批全跑完才开下一批** —— 谁先完成，立刻补下一个。

### 7.2 端口段隔离

- 每个 slot 的端口 = `start_port_base + slot`
- 端口段**宽度 = `num_gpus × num_per_gpu`**
- slot 静态分配：`slot_index = ckpt_index % max_parallel`

因为滑窗保证 checkpoint `i` 与 `i + max_parallel` **不会同时运行**，
所以端口段可以安全复用。调度器在启动前会用 `validate_plan()` 自检
「端口段 / 输出 / 日志互不重叠、`model_path` 不串」。

### 7.3 增量发现

用 checkpoint 的指纹（分片列表 + 字节数）判断「新增 / 更新 / 未变 / 不完整」，
默认只评测新增和更新的。

### 7.4 完整性判据

`hf_ckpt` 目录必须满足：`model.safetensors.index.json` 可解析 + 每个分片存在且非空 +
总字节 ≥ `metadata.total_size`。**不完整的一律跳过**并在报告里列出原因。

---

## 8. 常见问题

### Q1 `--min-age-seconds` 是干嘛的？为什么默认是 0？

它是「静默期」：最新 mtime 距今不足 N 秒就认为写入者还活着，先不碰。
用于**边训练边轮询扫描**的场景，防止读到写了一半的 checkpoint。

**默认改成 0（关闭）**，因为：

- 训练占满 4 卡、评测每卡也要 30–60 GB → **时间上互斥**，本机用不上「边训练边扫」
- `&&` 串联时训练进程已退出，**确定没有写入者**；此时静默期零价值，
  而且会**静默跳过最后一个 checkpoint**（最坏的失败模式）

关掉它**不削弱正确性**：真正的保证是 §7.4 的结构+字节校验，静默期只是启发式猜测。
（依据：`save_model_weights` 是**所有分片先写、`index.json` 最后写**，所以
「index 在 ⇒ 分片齐」是可靠的完成信号。）

要轮询扫描时显式传正数：`--min-age-seconds 120`。

### Q2 `--dry-run` 会覆盖真跑结果吗？

**不会**（曾经会，已修）。dry-run 的产物落在 `summary.dryrun.txt` / `summary.dryrun.json`，
真跑的 `summary.txt` / `summary.json` 原封不动。

⚠️ 但要注意：`--dry-run` 会**强制把静默期设为 0**，所以
**不能用 dry-run 验证静默期开关**（传 `--min-age-seconds 120` 与不传输出完全一样）。

### Q3 某个 checkpoint 评测失败了怎么办？

直接重跑同一条命令即可 —— 失败的 checkpoint **不会被写进 `eval_state.json`**，
下次会被判为「新增/更新」而重新评测；已成功的会被跳过。
（早期版本会把失败也写进状态，导致一次偶发失败就被永久跳过。）

### Q4 为什么并发后单个 checkpoint 变慢了？

因为两个 checkpoint 在同 4 张卡上互相抢 CPU/带宽。**实测代价只有 +13%**
（同一个 ckpt 的 clean：单跑 3分55秒 → 双并发 4分26秒），
而同时另一个 checkpoint 也在推进 ⇒ **吞吐约 1.77×**。

### Q5 和官方方案相比，快在哪？

- **单跑一个 checkpoint**：完全一样（我们就是调官方 launcher，没改评测）
- **多个 checkpoint**：官方只能串行 `T1 + T2 + …`；我们并发 `max(Ti) × 1.13`

### Q6 为什么报告里没有逐任务成功率？

只可能有两种情况：① 是 `--dry-run`（没跑，自然没数据）；
② `stats.txt` 里没有逐任务表（launcher 版本过旧）。真跑成功时一定有。

---

## 9. 测试

```bash
cd /data/code/lingbot-vla-v2
source /data/miniconda3/etc/profile.d/conda.sh && conda activate lingbotvla

python tests/test_multi_ckpt_eval.py       # 22 项：批次形状 / 完整性判据 / 发现 / 分类 /
                                           #   端口段 / dry-run / 预检 / stats 解析 /
                                           #   sentinel / episodes / 静默期 / 失败重试 /
                                           #   逐任务表 + Level / 表格对齐 / dry-run 隔离
bash   tests/test_launcher_task_list.sh    # 27 项：launcher 的任务清单白名单等
```

---

## 10. 不做什么（明确的边界）

- **不修改** `experiment/robotwin/robotwin_quick_eval.py`（历史遗留，已失效）
- **不重写**官方 launcher 的 task-level 队列，只通过 `--task_list_file` 注入任务清单
- **不做模型并行** —— 多卡是「多副本 + 任务级并行」，显存 ×N，不是 ÷N
- **不在训练进程内起评测** —— 训练侧没有可用钩子，「自动」靠 `&&` 串联
- ⚠️ **不要用假的 `--eval-workdir` 跑 launcher**。launcher 有 curobo 路径自愈逻辑，
  会改写真实 RoboTwin 环境的 editable `.pth`，**破坏仿真环境**。
  想无卡试跑，给一个不存在的 `--inference-workdir` 即可（打印完任务清单、起 server 前退出）。
