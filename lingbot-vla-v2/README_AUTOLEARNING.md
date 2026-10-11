# Auto Learning (AL) — LingBot-VLA-v2

> **这是 Auto Learning 子系统的专属 README**（与根目录 `README.md` 并列，互不影响）。
> 想先理解概念看本文 §1–§3；要上手跑看 §4；改代码看 §5；排查问题看 §7。

---

## 1. 它是什么

**Auto Learning 让模型自己决定"接下来练哪个任务"。**

你给一个任务集合（默认 50 个 RoboTwin 任务），AL 在训练循环中持续做四件事：

```
评测 → 挑任务 → 训练 → 判定
```

直到**累计通关 N 个任务**（当前配置 `target_total_passed_tasks: 15`）自动收工并落存档。

没有 AL 时：任务顺序是人工定死的流水线。
有 AL 时：每个 learning unit（默认 **50 步**）末尾都重新评估"谁最该学 / 谁已学会 / 谁在遗忘"。

---

## 2. 它是怎么运作的（一次 unit 的七步）

```
┌──────────── 每个 learning unit（eval_interval_steps = 50 步）────────────┐
│ ① 对当前任务评 active_train(4 轨迹) + active_val(4 轨迹)   ← 判定用这两组 │
│ ② 扫 hardness（抽 5% 轨迹、每轨迹 ≤300 帧）→ 逐样本难度权重               │
│ ③ 按"倍率"排序挑下一个任务（GMean ÷ 阈值线）                             │
│ ④ 对新任务做 attempt 基线评测（4+4 轨迹）                                │
│ ⑤ 训练该任务 50 步（NEW 本任务样本 / Replay 已 PASS 任务样本 混合）       │
│ ⑥ REVIEW：复查 PASS 池（遗忘？）+ 对本轮判 CONTINUE/PASS/DEFER/REOPEN     │
│ ⑦ 累计 PASS ≥ target ⇒ 正常收工（走收尾存档，不是直接 break）            │
└──────────────────────────────────────────────────────────────────────────┘
```

状态机入口在 `orchestration/scheduler.py::next_action()`，只返回五种动作：

| 动作 | 含义 |
|---|---|
| `bootstrap` | 启动时对全池做 scout（含"免费 PASS"判定：2 轨迹 scout + 4 轨迹 confirm 都过线 ⇒ 直接 PASS） |
| `select` | 挑下一个任务 + 跑 attempt 基线 + 扫该任务 hardness |
| `train_unit` | 发布一个 50 步的训练请求（deferred：由真实训练循环执行） |
| `review` | 复查 PASS 池 + 本 unit 判定 |
| `finished` | 收工（目标达成 / 预算耗尽 / AL 关闭） |

---

## 3. 核心概念

### 3.1 判定（PASS 判据）

```
PASS  ⇔  metric_value ≤ 该任务的阈值线
         metric_value = active_val_gmean_mse        （pass_metric = gmean_mse）
         阈值线        = 参考模型逐轨迹 GMean × calibration.multiplier（当前 60）
```

* 事件流里有现成比值 `task/<task>/val_to_pass_threshold`，**≤1 即过线**；
* 阈值表由 `tools/build_gmean_thresholds.py` 生成（纯统计，无需 GPU）；
* ⚠️ **开环判据**：只说明"这组轨迹上动作误差够小"，**不代表闭环任务成功**。

### 3.2 任务状态与判定结果

| TaskStatus | 含义 | | Decision | 含义 |
|---|---|---|---|---|
| `CANDIDATE` | 候选池，可被选中 | | `CONTINUE` | 继续训同一任务 |
| `TRAINING` | attempt 进行中 | | `PASS` | 过线（bootstrap 免费 PASS 也走这里） |
| `PASS` | 已通关（计入通关数） | | `DEFER` | 搁置（停滞/过拟合/步数不足） |
| `DEFER` | 暂时搁置 | | `DEFER_OVERFIT` | 过拟合导致搁置 |
| `EXHAUSTED` | attempt 预算耗尽 | | | |

### 3.3 三层缓存（决定"要不要重扫"）

| 缓存 | 粒度 | 失效条件 |
|---|---|---|
| **scout** | 每任务（2 轨迹） | **模型标识**变化 |
| **hardness** | **逐样本**（按 sample_id） | **模型标识**变化 |
| Triton/Inductor 编译缓存 | per-rank | 代码/配置变化 |

**模型标识 = 缓存有效性的唯一键**：`--model-name` 显式给，或取 `--checkpoint` 路径**尾部 4 段**；
缓存文件按模型分名（`scout.<slug>.json`），换模型不会覆盖旧缓存。

### 3.4 多卡分工

| 阶段 | 并行单位 |
|---|---|
| 训练 | 数据并行（6 卡同进程组，FSDP2） |
| 评测 | 按 rank 分片（逐条推理） |
| **hardness 扫描** | **样本级分片**：`ids[rank::world]` + `all_gather_object` 汇总 ⇒ 单任务墙钟 ÷ 卡数 |
| 启动前预扫 | 每 worker 一组任务，**只补 scout**（`AL_SCAN_SKIP_HARDNESS=1`） |

---

## 4. 怎么跑

### 4.1 前置产物

| 产物 | 生成方式 |
|---|---|
| 任务划分 | `/workspace/al/task_splits_50/manifest.json`（含每任务 val/train 轨迹 id） |
| task baseline | `python -m lingbotvla.auto_learning.tools.compute_task_baseline --manifest … --config … --out …` |
| 阈值表 | `python -m lingbotvla.auto_learning.tools.build_gmean_thresholds --ref-per-traj … --baseline … --reference global_step_50000 --multiplier 60 -o …` |
| datasets.txt | **单行**：`robotwin <聚合数据集目录>` |
| scout / hardness 缓存 | 无需手工造：launcher 会按覆盖情况自动补扫 |

### 4.2 启动（推荐：统一启动程序）

```bash
cd <repo>
PY=/opt/robotwin-env/bin/python          # 训练解释器
$PY -u experiment/robotwin/al_launch.py \
  --run-name my_run \
  --checkpoint /path/to/hf_ckpt \
  --eval-config configs/auto_learning/al_eval2_target15_m60.yaml \
  --steps 20000 --micro 5 --gas 1 \
  --gpus 0,1,2,3,4,5 \
  --worker-out-root     /models/robotwin-persistent/al_runs \
  --triton-cache        /models/robotwin-persistent/al_cache/triton \
  --torchinductor-cache /models/robotwin-persistent/al_cache/torchinductor \
  --hardness-cache-file /workspace/al/hardness_cache/hardness.json \
  --scout-cache-file    /workspace/al/scout_cache/scout.json \
  --env PRUNE=1 --env AITER_USE_SYSTEM_TRITON=1 \
  --env AL_HARDNESS_SHARD=1 \
  --workers 1 --no-tb
```

launcher 会依次做：**缓存寻址 → 覆盖检查 → （缺则）扫描 → 就绪检查 → 起训练**，
覆盖不全时**打印缺失任务名并以退出码 3 结束**（不启动训练、不伪造覆盖数）。

先干跑看计划（不启动任何进程）：加 `--dry-run`。

### 4.3 监控（一条命令）

```bash
python3 tools/auto_learning_pipeline/al_panel.py                 # 自动选最新 run
python3 tools/auto_learning_pipeline/al_panel.py my_run          # 指定 run
python3 tools/auto_learning_pipeline/al_panel.py my_run --all    # 列出全部 50 个任务（默认只列最接近通关的 4 个）
watch -n 30 python3 tools/auto_learning_pipeline/al_panel.py     # 每 30 秒刷新
```

输出：rank 数/卡数 · Step · 稳态 s/it · **通关 x/15 还差几个** · 已通关任务 ·
下一次判定还有多少步 · **逐任务 GMean vs 门槛 vs 倍率** · **当前任务成长轨迹（近三次对照）**。

### 4.4 退出码

`al_launch.py` 里定义的全部退出码（`EXIT_*`）：

| 码 | 常量 | 含义 |
|---|---|---|
| 0 | `EXIT_OK` | 正常（已启动训练 / dry-run 完成） |
| 2 | `EXIT_PREFLIGHT` | 预检失败（路径/参数/前置条件） |
| 3 | `EXIT_SCAN_INCOMPLETE` | **扫描未完成**（目标缓存文件里仍有缺失任务；会打印缺失任务名） |
| 4 | `EXIT_WORKER_NOT_READY` | 扫描 worker 未就绪或提前退出 |
| 5 | `EXIT_SCAN_TIMEOUT` | 扫描超时（`--scan-timeout`） |
| 6 | `EXIT_INTERNAL` | 内部错误 |
| 7 | `EXIT_TRAIN_DIED` | 训练进程启动后死亡 |

---

## 5. 代码结构

```
lingbotvla/auto_learning/
├── config.py            AutoLearningConfig（配置 dataclass，含全部校验）
├── types.py             TaskStatus / Decision / ReasonCode 等枚举
├── catalog.py           TaskCatalog：task → 样本/轨迹/val ids
├── resolver.py          SampleResolver：按 task 取样本
├── baseline.py          BaselineStore + 指纹（config/task 级）
├── check.py             无卡自检
├── hardness.py          HardnessScorer：逐样本 loss（**逐样本噪声**，换批不改数）
├── hardness_cache.py    逐样本 loss 磁盘缓存 + resolve_model_name
├── scout_cache.py       scout 指标缓存（含语义指纹）
├── cache_paths.py       缓存文件按模型分名 + 旧文件迁移（scoped_cache_path / ensure_scoped_cache_file）
├── evaluator.py         评测入口（包装 open-loop validator）
├── model_config.py      构造 processor（与训练同一入口，避免口径漂移）
├── batch_ratio.py       NEW/Replay 的 GBS 分配
├── eval_batch_policy.py 多轨迹批量推理策略（默认串行；多卡 FSDP 强制逐条）
├── scan_accel.py        扫描加速决策对象（默认关闭）
├── stochastic_parity.py 随机性验收（开批处理前的门控）
│
├── decision/            判定层
│   ├── thresholds.py        pass_line() / metric_value() / is_pass()
│   ├── metrics.py           forget_ratio / is_finite_metric
│   ├── state_machine.py     start_attempt / apply_pass / 迁移规则
│   ├── review.py            PASS 池复查（遗忘判定）
│   └── gmean.py             GMean 相关计算
│
├── orchestration/       编排层
│   ├── scheduler.py         **主循环状态机**（next_action / advance / _select / _review）
│   └── baselines.py         attempt 基线
│
├── sampling/            采样层
│   ├── sampler.py           AutoLearnSampler（按 TrainRequest 产出样本）
│   ├── replay.py            Replay 池（已 PASS 任务样本回放）
│   ├── rng.py               唯一权威 RNG
│   └── hardness_scan.py     HardnessScanner（抽轨迹 + 压帧 + 打分 + 缓存）
│
├── state/               状态层
│   ├── registry.py          TaskRegistry（**任务状态权威**）
│   └── persistence.py       状态存档/恢复
│
├── real/                真实训练对接层
│   ├── build.py             build_auto_learning_parts / build_real_*（装配）
│   ├── hook.py              **AutoLearnLoopHook**（训练循环钩子；含扫描模式早退）
│   ├── backend.py           Backend 组装 + hardness 分片（ids[rank::world] + all_gather）
│   ├── sampler.py           LazyAutoLearnSampler（延迟绑定真 sampler）
│   ├── determinism.py       确定性上下文（临时关图像增强等）
│   └── sanity.py            运行时自检
│
├── obs/                 观测层：logger.py（事件流）/ report.py（报表）
├── tools/               工具：build_gmean_thresholds / compute_task_baseline /
│                        calibrate_pass_line / recheck_thresholds / sweep_params / tb_from_jsonl
└── configs/             示例 AL 配置（demo_4task / demo_12task / demo_50task）
```

**启动侧**：`experiment/robotwin/al_launch.py`（统一启动程序）。

**接入链**：

```
train.sh → tasks/vla/train_lingbotvla.py
   └─ build_auto_learning_parts()   (real/build.py)
   └─ AutoLearnLoopHook             (real/hook.py)
        └─ Scheduler.next_action()  (orchestration/scheduler.py)
             ├─ Evaluator（开环评测）      ├─ HardnessScanner（难度）
             ├─ Registry（状态权威）       └─ EventLogger → auto_learning_events.jsonl
```

---

## 6. 配置（`configs/auto_learning/*.yaml`）

按用途分组（完整字段与默认值见 `config.py`）：

| 组 | 关键字段 | 说明 |
|---|---|---|
| **收工** | `target_total_passed_tasks` | 通关几个就停（当前 15） |
| | `max_new_tasks_passed_this_run` / `min_new_tasks_passed_this_run` | 兼容项 / 下限门控 |
| **任务池** | `task_names`（null=全部）/ `max_attempts_per_task` / `max_reopens_per_task` | 池与预算 |
| **评测节奏** | `eval_interval_steps`（=unit 长度） | ⚠️ `min_steps_before_defer` / `defer_retry_steps` 必须是它的整数倍 |
| **评测规模** | `global_scout_val_trajs`（2）/ `scout_confirm_enabled`（true） | scout + confirm |
| | `active_train_probe_trajs` / `active_val_probe_trajs`（4/4） | attempt 基线与判定 |
| **判定口径** | `pass_metric`（gmean_mse）/ `pass_thresholds_file` | 当前用 `pass_thresholds_gmean60_warn.json` |
| **候选池** | `pool_filter_by_gmean_ratio` / `pool_ratio_pass`（0.2）/ `pool_ratio_skip` | 太易/太难不进池 |
| **批组成** | `batch_size` / `new_slots` / `replay_slots` / `new_ratio` | GBS 里 NEW 与 Replay 比例 |
| **难度扫描** | `hardness_probe_fraction`（0.05）/ `hardness_max_frames_per_traj`（300） | 抽样比例、帧上限 |
| | `hardness_weight_min/max` / `hardness_alpha` | 难度 → 采样权重映射 |
| **重扫** | `rescan_candidates_after_transition`（true）/ `rescan_every_n_task_switches`（1） | 何时全池重扫 |

---

## 7. 排查入口（症状 → 去哪）

| 症状 | 先看 |
|---|---|
| 覆盖率恒 `0/50`、退出码 3 | `docs/auto_learning/auto_learning_pitfalls_zh.md` §2.2（模型标识分裂）/ §2.3（迁移未触发） |
| 训练不启动、`运行模式：unknown` | 同上 §2.1（覆盖检查被跳过） |
| 预扫跑了几小时 | 同上 §2.5（预扫把 hardness 也扫了）；确认 `AL_SCAN_SKIP_HARDNESS=1` 只注入 worker |
| hardness 很慢（15–20 分钟/任务） | 同上 §2.6（是否多卡同进程组；单卡会退回全量） |
| "指标过线"但通关数不涨 | 本文 §3.1：过线 ≠ 通关；正式通关在 unit 末尾 REVIEW |
| 面板显示不全 / 数字口径疑问 | `tools/auto_learning_pipeline/al_panel.py` 顶部 docstring（rank 数判据、通关数口径、s/it 口径） |

---

## 8. 相关文档

| 文档 | 内容 |
|---|---|
| `docs/auto_learning/auto_learning_overview_zh.md` | 总览（概念/流水线/术语表，**先读这篇**） |
| `docs/auto_learning/auto_learning_launch_guide_zh.md` | 启动与参数、缓存寻址、扫描模式、监控（§8） |
| `docs/auto_learning/auto_learning_pitfalls_zh.md` | **易错文档**：8 条自查表 + 6 个核心坑 |
| `docs/auto_learning/GMEAN_PASSLINE_GUIDE.md` | 通过线（GMean × 倍率）标定 |
| `docs/auto_learning/auto_learning_multigpu_guide.md` | 多卡行为 |
| `docs/auto_learning/auto_learning_multirank_eval_diagnosis.md` | 多 rank 评测诊断 |
| `docs/open_loop_guide.md` 等 | 开环评测本身 |
| `docs/auto_learning/scan_accel_gmean200_zh.md` | 扫描加速（实验，默认关） |
| `tools/auto_learning_pipeline/al_panel.py` | 监控面板 |

---

## 9. 已知边界（别误用）

1. **开环 PASS ≠ 闭环成功** —— 判定基于开环动作误差；闭环成功率需另测。
2. **指标过线 ≠ 已通关** —— 过线是"当前位置够好"，通关要在 unit 末尾走 REVIEW 写进注册表。
3. **scout 快、hardness 慢** —— scout 秒级（2 轨迹），hardness 分钟级（逐样本读帧）；
   所以启动门控**只查 scout**，hardness 按需扫。
4. **多卡分片需要"同进程组"** —— 单卡 worker 上 `AL_HARDNESS_SHARD=1` 会静默失效、退回全量。
5. **换模型标识 = 缓存全部失效** —— 会触发重扫；这不是 bug，是刻意的正确性约束。
