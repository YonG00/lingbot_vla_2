# AL 改动台账（PENDING CHANGES）

> **为什么有这个文件**：改动会持续积累，很多功能**改了但没上机验证**（因为后台在跑训练）。
> 一旦上机报错，需要能立刻回答："**改了什么、为什么改、期望什么、会不会互相冲突**"。
> 所以每次改动都登记在此；**等训练测试跑通、确认可用后，把对应条目标记 ✅ 并注明验证证据**，
> 之后就不必再惦记。
>
> 规则（与技能 `lingbot-code-changes` 配套）：
> - 阶段 3 每落地一条 ⇒ 在 §1 追加/更新一行；
> - 阶段 4 提交后 ⇒ 补上 commit SHA；
> - 上机验证通过 ⇒ 状态改 `✅ 已验证`，写清**验证方式 + 实测数字**，并移入 §2 归档；
> - 回滚 ⇒ 状态改 `↩️ 已回滚`，写清原因与回滚方式。
>
> **状态取值**：`⏳ 待验证`（已改未上机）/ `🧪 部分验证`（只有 CPU 证据）/ `✅ 已验证` / `↩️ 已回滚`

---

## §1 待验证（改了但没上机跑过）

| # | 日期 | 改什么（文件/函数） | 为什么 | 期望行为 | 依赖/冲突 | commit | 状态 |
|---|---|---|---|---|---|---|---|
| 1 | 10-11 | `state/registry.py`：抽 `_judge_n`；`confirm_val_ids` 与 `active_val_ids` 同组 | 判定尺子原先两处各写 `val_ids[:4]`，语义上看不出是同一把尺子 | 判定仍用 4 条；无行为变化（等价重构） | 与 #2 同批 | `d014aa6` | 🧪 部分验证（CPU 回归 131 passed） |
| 2 | 10-11 | `real/backend.py` 新增 `merge_trajectory_metrics()` + `evaluate_confirm_merged()`；`scheduler._timed_confirm_merged()`；`build.py` 注入 adapter | confirm 的 4 条里 2 条与 scout 完全重复且**被重跑**（白花一半算力），复核也不独立 | 日志出现 `[open_loop][confirm] task=X 复用 scout 2 条 + 新评 2 条 ⇒ 判决用 4 条（省 2 条推理）`；事件出现 `confirm_reused_trajs=2` / `confirm_fresh_trajs=2`；`confirm_nmse` 口径变为"2 复用 + 2 新算的合并值" | **与 #3 必须成对**（帧上限不改判定，独立）；CPU 单测 5 passed、回归 131 passed | `d014aa6` | 🧪 部分验证 |
| 3 | 10-11 | `configs/auto_learning/al_eval2_v45_frame200.yaml`（新）：`hardness_max_frames_per_traj: 300 → 200` | 扫描量 −33%（单任务 600 → 400 帧），帧多的任务省时间 | 日志出现 `（帧数上限 200：N 条轨迹被均匀压缩，A → B 帧）`；单任务扫描墙钟下降约 1/3 | 独立；**只对新 run 生效**（配置在进程启动时读入） | `d014aa6`（同批） | ⏳ 待验证 |
| 4 | 10-11 | `experiment/robotwin/al_launch.py`：`base_env()` 白名单纳入 `SAVE_EVERY` / `HF_PASS_INTERVAL` | 原只透传 `AL_*` ⇒ shell 里 `SAVE_EVERY=400` **静默无效**（实测存档仍是 1000 步） | 以 shell 变量方式传也能生效；`--env` 仍优先 | 独立 | 待提交（工作副本已改） | 🧪 部分验证（`--env` 路径已在 v43 实测生效） |
| 5 | 10-11 前 | `tools/auto_learning_pipeline/al_panel.py`（训练器识别 / 未找到 run 三态 / 步时双口径） | 面板把多 run 的 rank 加在一起数、run 未创建时提示误导、s/it 被评测步拉爆 | `--trainers` 一行一个训练器；状态行走 `稳态 X s/it（当前步 Y s/it：含评测/扫描）` | 独立；**已在真机跑过（只读工具）** | `17948d0` | ✅ 已验证（真机跑通，见 10-11 日志） |
| 6 | 10-11 | `real/backend.py`：`evaluate()` 新增**轨迹分片**（`AL_EVAL_SHARD=1` 开启，**默认关**）+ `_eval_shard_indices()` / `_gather_and_merge_shards()` / `_all_gather_object()`；`merge_trajectory_metrics()` 支持 `metric_valid` / `note` 覆盖 | 训练态评测现状是**每个 rank 各跑整批**（实测 v43 8 卡 / v44 7 卡：每卡都跑 12/20 条）；轨迹互不依赖、逐轨迹 MSE 与卡数无关、聚合是几何平均（可分组）⇒ 可分片 | ① **不死锁**（每卡恒 `k=ceil(n/W)` 条 ⇒ 集合通信对齐）；② 汇总与单卡全量**逐位一致**；③ 墙钟 ≈ 现值的 `1/W`（n=8、W=8 ⇒ 每卡 1 条） | ⚠️ **CPU 测不出集合通信错配** ⇒ 必须真机验证；只改 `active_val` + `train_monitor`，`scout`/`confirm`/`rescan` 暂不动 | 待提交 | 🧪 部分验证（CPU 单测 11 项 + 本机回归 119 passed） |
| 7 | 10-11 | 消歧义注释：`orchestration/scheduler.py`（`*_gmean_mse` 是**判定口径**、`*_nmse` 是历史遗留显示量）、`state/registry.py`（`active_val_probe_trajs` **实际决定 confirm 条数**）| 命名历史遗留易误读；改名会让 v41–v44 产物不可比 ⇒ 按三选一取**加注释** | 注释与当前实现一致；不改任何字段名/指标名 | 独立 | `d014aa6` | ✅ 已验证（纯注释） |

## §2 已验证（可不再惦记）

| # | 日期 | 内容 | 验证证据 |
|---|---|---|---|
| — | — | — | — |

## §3 已知遗留 / 暂不改（有明确理由）

| 项 | 为什么暂不改 |
|---|---|
| 事件字段 `*_nmse` 与判定口径 `gmean_mse` 不一致 | 改名会让 v41–v44 历史事件流不可比；已在记账处加**消歧义注释** |
| 参数 `active_val_probe_trajs` 实际决定 confirm 条数 | 改名牵动 `SEMANTIC_KEYS` ⇒ 影响 resume 指纹；已加注释 |
| ~~开环评测本体未按 rank 分片~~ ⇒ **已立项：见 §1 #6** | 用户 10-11 定 B 方案（每卡恒 `k=ceil(n/W)` 条、汇总按 episode id 去重），范围只 `active_val` + `train_monitor`；**默认关**，真机验证后再考虑默认开 |
| `bootstrap PASS` 任务直接毕业 ⇒ confirm 值当不了训练基线 | 门控语义问题，改它等于改设计（"免费 PASS"的意义就是不用训），实测 v43/v44 零交集 |

---

**当前待验证：5 条（#1–#4、#6）。下次上机（无训练在跑时）优先按编号顺序验证；**
**#6（轨迹分片）需真机 7/8 卡 smoke：确认不死锁、指标与不分片逐位一致、墙钟下降。**
