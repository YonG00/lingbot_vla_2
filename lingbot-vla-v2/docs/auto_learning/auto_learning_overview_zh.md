# Auto Learning（自动学习）项目说明

> **这一篇回答"AL 是什么、怎么运作、代码在哪"。**
> 具体操作/排查请看同目录的其它文档（§9 有分类索引），本篇不重复那些内容。

---

## 1. 一句话

**Auto Learning 让模型自己决定"接下来练哪个任务"** —— 把 50 个任务的集合交给它，
它在训练循环里持续做「评测 → 挑任务 → 训练 → 判定」，直到**累计通关 N 个任务**（当前配置 `N=15`）自动收工。

没有 AL 时：任务列表是人工定死的一条流水线，跑完就停。
有 AL 时：每个 learning unit（默认 50 步）末尾都会重新评估"谁最该学、谁已经学会、谁的进步停滞"。

---

## 2. 它解决什么问题

| 痛点 | AL 的做法 |
|---|---|
| 50 个任务难度差异极大，人工排顺序靠猜 | 用**统一尺子**（GMean-MSE / 阈值倍率）给所有任务排序，自动选"最该学的那个" |
| 训完不知道"到底学会了几个" | 每个 unit 末尾做**正式判定**，维护 PASS 注册表，给"通关 x/N"这个可交付指标 |
| 学过的任务会**遗忘** | REVIEW 阶段复查已 PASS 任务的指标，退化超阈值 ⇒ REOPEN 回炉重训 |
| 难任务上无限打转 | attempt 预算 / 停滞 DEFER / 步数上限三道保险丝 |
| "开环指标好"≠"闭环成功" | 判定只用于**选课**；闭环成功率需另测（见 `docs/open_loop_guide.md`） |

---

## 3. 运行流水线（一次 unit 的生命周期）

```
        ┌─────────────────────── 每个 learning unit（默认 eval_interval_steps=50 步）───────────────────────┐
        │                                                                                                  │
 评测   │  ① 对"当前任务"评测 active_train(4 轨迹) + active_val(4 轨迹)  ← 判定用这两组                  │
 选择   │  ② 扫 hardness（0.05 轨迹抽样，≤300 帧/轨迹）→ 得到逐样本难度权重                                │
        │  ③ 按选课优先级挑下一个任务（GMean/阈值 倍率，ratio 模式）                                        │
        │  ④ 对新任务做 attempt 基线评测（同上 4+4）                                                        │
 训练   │  ⑤ 训练该任务 50 步（NEW/Replay 按 batch_ratio 混合）                                             │
 判定   │  ⑥ REVIEW：对 PASS 池复查 + 对本轮结果判定 CONTINUE / PASS / DEFER / REOPEN                       │
 收工   │  ⑦ 累计 PASS 达到 target_total_passed_tasks ⇒ 正常收工（走收尾存档）                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

对应代码：`orchestration/scheduler.py::next_action()` 是这条流水线的**状态机入口**，
返回 `bootstrap / select / train_unit / review / finished` 五种动作。

---

## 4. 核心概念

### 4.1 任务状态（`types.py::TaskStatus`）

| 状态 | 含义 |
|---|---|
| `CANDIDATE` | 候选池里，可被选中训练 |
| `TRAINING` | 正在训（attempt 进行中） |
| `PASS` | 已达到通过线（记入通关数） |
| `DEFER` | 暂时搁置（过拟合 / 步数不足 / 停滞） |
| `EXHAUSTED` | attempt 预算耗尽 |

### 4.2 判定结果（`types.py::Decision`）

`CONTINUE`（继续训）· `PASS`（过线，免费 PASS 也走这里）· `DEFER` · `DEFER_OVERFIT`。

### 4.3 通过线（PASS 判据）

```
PASS  ⇔  metric_value ≤ 该任务的阈值线
          其中 metric 由 pass_metric 决定（当前配置 = gmean_mse）
          阈值线 = 参考模型的逐轨迹 GMean × calibration.multiplier
```

* 阈值表由 `tools/build_gmean_thresholds.py` 生成（纯统计，见 `docs/auto_learning/GMEAN_PASSLINE_GUIDE.md`）；
* 事件流里有现成的比值 `task/<task>/val_to_pass_threshold`，**≤1 即过线**；
* ⚠️ 判定是**开环**的：它只说明"这组轨迹上的动作误差够小"，不代表闭环任务成功。

### 4.4 三层缓存（决定"要不要重扫"）

| 缓存 | 内容 | 失效条件 |
|---|---|---|
| **scout** | 每任务的 scout 指标（2 轨迹） | 模型标识变化 |
| **hardness** | **逐样本** loss（按 sample_id） | 模型标识变化 |
| per-rank Triton/Inductor 编译缓存 | 编译产物 | 代码/配置变化 |

**模型标识 = 缓存有效性的唯一键**（详见 `docs/auto_learning/auto_learning_launch_guide_zh.md` §8.1）。

### 4.5 多卡分工

| 阶段 | 并行单位 |
|---|---|
| 训练 | **数据并行**（6 卡同进程组，FSDP2） |
| 评测 | **按 rank 分片**（每 rank 评一部分轨迹/样本，逐条推理） |
| hardness 扫描 | **样本级分片**：`ids[rank::world]` + `all_gather_object` 汇总（单任务墙钟 ÷卡数） |
| 启动前预扫 | 每个 worker 一组任务、**只补 scout**（`AL_SCAN_SKIP_HARDNESS=1`） |

---

## 5. 配置（`configs/auto_learning/*.yaml` → `AutoLearningConfig`）

按用途分组（完整字段见 `lingbotvla/auto_learning/config.py`）：

| 组 | 关键字段 | 说明 |
|---|---|---|
| **收工** | `target_total_passed_tasks` / `max_new_tasks_passed_this_run` / `min_new_tasks_passed_this_run` | "通关几个就停"；后两个是兼容/下限门控 |
| **任务池** | `task_names`（null=全部）/ `max_attempts_per_task` / `max_reopens_per_task` | 池子与预算 |
| **评测节奏** | `eval_interval_steps`（=unit 长度）/ `min_steps_before_defer` / `defer_retry_steps` | ⚠️ 后两者必须是 unit 长度的整数倍（config 强制校验） |
| **评测规模** | `global_scout_val_trajs` / `scout_confirm_enabled` / `active_{train,val}_probe_trajs` | scout 2 轨迹 + confirm 4 轨迹；active 4+4 |
| **判定口径** | `pass_metric` / `pass_thresholds_file` / `pass_nmse`（兼容占位） | 当前 = `gmean_mse` + `pass_thresholds_gmean60_warn.json` |
| **候选池** | `pool_filter_by_gmean_ratio` / `pool_ratio_pass` / `pool_ratio_skip` | 太易/太难的任务不进池（ratio=倍率） |
| **批组成** | `batch_size` / `new_slots` / `replay_slots` / `new_ratio` | GBS 里 NEW 与 Replay 的比例 |
| **难度扫描** | `hardness_probe_fraction` / `hardness_max_frames_per_traj` / `hardness_weight_{min,max}` / `hardness_alpha` | 抽样比例、帧数上限、权重映射 |
| **重扫** | `rescan_candidates_after_transition`（默认 True）/ `rescan_every_n_task_switches`（**默认 1**）/ `refresh_hardness_on_continue`（默认 False） | 何时全池重扫 |

---

## 6. 代码结构（`lingbotvla/auto_learning/`）

```
config.py            配置 dataclass（AutoLearningConfig）
types.py             TaskStatus / Decision / ReasonCode 等枚举
catalog.py           任务目录（task → 样本/轨迹/val ids）
resolver.py          按任务取样本
baseline.py          baseline store 与指纹
check.py             无卡自检
hardness.py          HardnessScorer（逐样本 loss、逐样本噪声）
hardness_cache.py    逐样本 loss 磁盘缓存（+ resolve_model_name）
scout_cache.py       scout 指标缓存（+ 语义指纹）
cache_paths.py       缓存文件按模型分名 + 旧文件迁移
evaluator.py         评测入口（包一层 validator）
model_config.py      构造 processor（与训练同一入口）
batch_ratio.py       NEW/Replay 的 GBS 分配
eval_batch_policy.py 多轨迹批量推理策略（默认串行）
scan_accel.py        扫描加速的决策对象（默认关闭）
stochastic_parity.py 随机性验收（批处理前的门控）

decision/     gmean.py / metrics.py / review.py / state_machine.py / thresholds.py
              ↑ 判定与状态迁移（PASS/DEFER/REOPEN 的规则都在这里）
orchestration/ baselines.py / scheduler.py
              ↑ **主循环状态机**（next_action/advance）；baselines 是 attempt 基线
sampling/     sampler.py / replay.py / rng.py / hardness_scan.py
              ↑ 采样与难度扫描（HardnessScanner 在这里）
state/        registry.py / persistence.py
              ↑ 任务注册表（状态权威）与存档
real/         backend.py / build.py / hook.py / sampler.py / determinism.py / sanity.py
              ↑ 与真实训练对接：装配、钩子、数据加载、逐样本噪声确定性
obs/          logger.py / report.py     ← 事件流与指标输出
tools/        build_gmean_thresholds.py / compute_task_baseline.py / calibrate_pass_line.py
              / recheck_thresholds.py / sweep_params.py / tb_from_jsonl.py
```

**启动侧**：`experiment/robotwin/al_launch.py`（统一启动程序：扫描 → 就绪检查 → 起训练）。

---

## 7. 接入点与数据流

```
train.sh → tasks/vla/train_lingbotvla.py
   └─ build_auto_learning_parts()            (real/build.py)   ← 读 manifest/baseline/阈值表/缓存
   └─ AutoLearnLoopHook                      (real/hook.py)    ← 训练循环的钩子
        └─ Scheduler.next_action()/advance() (orchestration/scheduler.py)
             ├─ Evaluator → OpenLoopValidator（推理/评测）
             ├─ HardnessScanner → HardnessScorer（难度打分）
             ├─ Registry（任务状态权威）
             └─ EventLogger → auto_learning_events.jsonl（**监控数据源**）
```

* **输入**：任务 manifest、task baseline、阈值表、scout/hardness 缓存、AL 配置
* **输出**：`auto_learning_events.jsonl`（事件流）+ TensorBoard 指标 + checkpoint
* **监控**：`tools/auto_learning_pipeline/al_panel.py`（一条命令看进度/通关/门槛/逐任务成长）

---

## 8. 三条最容易被误解的边界

1. **"开环 PASS" ≠ "闭环成功"**：判定基于开环轨迹的动作误差；闭环成功率需单独评测。
2. **"指标已过线" ≠ "已通关"**：过线是"当前位置够好"，正式通关要在 unit 末尾走 REVIEW 写进注册表。
3. **scout 快、hardness 慢**：scout（2 轨迹）**秒级**；hardness（逐样本读帧）**分钟级**，
   所以门控只查 scout，hardness 按需扫（详见 pitfalls 文档 §2.5）。

---

## 9. 文档地图（按需求找）

| 想知道什么 | 看哪篇 |
|---|---|
| **AL 是什么、怎么运作**（本篇） | `docs/auto_learning/auto_learning_overview_zh.md` |
| **怎么启动、参数怎么配、缓存怎么寻址** | `docs/auto_learning/auto_learning_launch_guide_zh.md` |
| **踩过哪些坑、症状怎么定位** | `docs/auto_learning/auto_learning_pitfalls_zh.md`（易错文档） |
| **多卡/多 rank 行为** | `docs/auto_learning/auto_learning_multigpu_guide.md`、`docs/auto_learning/auto_learning_multirank_eval_diagnosis.md` |
| **通过线（GMean 倍率）怎么定** | `docs/auto_learning/GMEAN_PASSLINE_GUIDE.md` |
| **开环评测本身** | `docs/open_loop_guide.md`、`open_loop_validation_guide.md`、`open_loop_selfcheck_guide.md` |
| **扫描加速（实验，默认关）** | `docs/auto_learning/scan_accel_gmean200_zh.md`、`docs/acceptance/eval_batch_auto_gate_guide.md` |
| **HF PASS 里程碑 / TB 对齐 / 无存档收工** | `docs/auto_learning/AL_HF_PASS_MILESTONE_POLICY.md`、`AL_TB_ALIGNMENT_NOTES.md`、`AL_TOTALPASS_NOSAVE_TB_GUIDE.md` |
| **50 任务优先级补丁 / 开环更新** | `docs/auto_learning/AL_GMEAN200_50TASK_PRIORITY_PATCH.md`、`AL_OPENLOOP_UPDATE_GUIDE.md` |
| 先训后评 / 多 ckpt 评测 | `docs/phase_train_then_eval_guide.md`、`multi_ckpt_eval_guide.md` |
| **监控面板** | `tools/auto_learning_pipeline/al_panel.py`（用法见启动指南 §8.6） |

---

## 10. 术语表（中英对照，避免读日志时混淆）

| 日志/代码里的词 | 含义 |
|---|---|
| `unit` / `learning unit` | 一个训练单元 = `eval_interval_steps` 个 optimizer step（默认 50） |
| `attempt` | 对某任务的一次"选中→训练→判定"循环 |
| `scout` | 选课用的轻量评测（`global_scout_val_trajs=2`） |
| `confirm` | scout 过线后的 4 轨迹复评（`scout_confirm_enabled`） |
| `active_train` / `active_val` | attempt 基线与判定用的 4+4 轨迹评测 |
| `hardness` | 逐样本难度（loss），用于采样加权 |
| `NEW` / `Replay` | 本任务样本 / 已 PASS 任务的回放样本 |
| `ratio` | 指标 ÷ 该任务阈值线（≤1 过线） |
| `rescan` | 对候选池重新评测（`rescan_every_n_task_switches`，**代码默认 1**；本实验配置用 3） |
| `REOPEN` | 把已 PASS 的任务因遗忘而回炉 |
| `bootstrap` | 启动时对全池做 scout（含免费 PASS 判定） |
| `PASS` / `forgotten` | 通关 / 被判定遗忘 |
