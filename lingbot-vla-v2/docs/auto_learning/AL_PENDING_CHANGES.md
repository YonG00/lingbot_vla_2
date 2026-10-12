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
| 9 | 10-12 | `tools/auto_learning_pipeline/al_panel.py`：**默认视图改为 15 行紧凑面板**（纯 ASCII / 宽度自适应 / 在训任务历史 3 unit / 总用时与预估 / 池子 `pass=7 defer=12 … exhausted=0`）；原长表格移入 `--wide`；`s/it` 与 ETA 改口径 | **改动本意（用户 2026-10-12 四问）**：<br>① **不变量**：面板**只读**（不写 DCP/不碰训练）；各节数据来源不变（池子=DCP 注册表、轨迹=事件流、步时=训练日志）；`--pool/--progress/--traces/--all/--trainers` 语义不变。<br>② **出错症状**（怎么一眼看出坏了）：面板报 `NameError/AttributeError` ⇒ 打包字段口径变了（如 `best_gmean_mse` 只在 PASS 后有值、`series[metric][task]` 是 list 不是 dict）；`池子` 行不显示 `exhausted=` ⇒ 池子读取失败退化了；`s/it` 远大于日志稳态值 ⇒ 又用回累计均值了。<br>③ **快速回滚**：`git checkout <sha>~1 -- tools/auto_learning_pipeline/al_panel.py`（单文件、无外部依赖）。<br>④ **验证入口**：`/opt/robotwin-env/bin/python tools/auto_learning_pipeline/al_panel.py <run>`，判据 = 行数 ≤20、无框线字符、`Step` 不带 `/20000`、池子含 `exhausted=`、历史 3 行、`用时` 的"已训"与日志 `[HH:MM:SS<…]` 一致 | 独立（只读工具）；`--wide` 保留全部旧细节 | 待提交 | ✅ 已验证（v44 实跑 15 行；`已训 11.8h` 与日志 `11:45:20` 吻合） |
| 6 | 10-11 | `real/backend.py`：`evaluate()` 新增**轨迹分片**（`AL_EVAL_SHARD=1` 开启，**默认关**）+ `_eval_shard_indices()` / `_gather_and_merge_shards()` / `_all_gather_object()`；`merge_trajectory_metrics()` 支持 `metric_valid` / `note` 覆盖 | 训练态评测现状是**每个 rank 各跑整批**（实测 v43 8 卡 / v44 7 卡：每卡都跑 12/20 条）；轨迹互不依赖、逐轨迹 MSE 与卡数无关、聚合是几何平均（可分组）⇒ 可分片 | ① **不死锁**（每卡恒 `k=ceil(n/W)` 条 ⇒ 集合通信对齐）；② 汇总与单卡全量**逐位一致**；③ 墙钟 ≈ 现值的 `1/W`（n=8、W=8 ⇒ 每卡 1 条） | ⚠️ **CPU 测不出集合通信错配** ⇒ 必须真机验证；只改 `active_val` + `train_monitor`，`scout`/`confirm`/`rescan` 暂不动 | `7164cde` | 🧪 部分验证（CPU 单测 11 项 + 本机回归 119 passed） |
| 7 | 10-11 | 消歧义注释：`orchestration/scheduler.py`（`*_gmean_mse` 是**判定口径**、`*_nmse` 是历史遗留显示量）、`state/registry.py`（`active_val_probe_trajs` **实际决定 confirm 条数**）| 命名历史遗留易误读；改名会让 v41–v44 产物不可比 ⇒ 按三选一取**加注释** | 注释与当前实现一致；不改任何字段名/指标名 | 独立 | `d014aa6` | ✅ 已验证（纯注释） |
| 8 | 10-11 | `tools/auto_learning_pipeline/al_panel.py`：新增 **DCP 注册表读取**（`latest_dcp` / `load_registry`）+ **池子摘要**（默认视图 2 行）+ 4 个参数（`--pool` / `--progress` / `--traces` / `--watch`）；新增技能 `lingbot-training-monitor` | 用户要"一眼看池子/进步/GMean"：原面板只能从**事件流推导**状态，**推不出 DEFER/EXHAUSTED**、也没有 attempts/lp/过拟合/DEFER 原因 | 默认视图出现 `池子来源: DCP global_step_N`、`池子: TRAINING 1 \| DEFER 9 \| PASS 7 \| CANDIDATE 33`、`当前任务: X（第 N 次尝试，已训 M 步，当前倍率 Kx）`；`--pool` 出明细表（含 DEFER 原文原因）；`--progress` 按 lp50 降序；`--traces N` 多任务轨迹 | 独立；**只读**（`torch.load(map_location=cpu)`），训练期安全 | `606c67a` | ✅ 已验证（v44 实跑：池子 1/9/7/33 与 DCP 注册表一致；`--progress` 12 行、`--traces 2` 正常） |
| 9 | 10-12 | `real/determinism.py::deterministic_sampling`：**没有增强要关时跳过快照 RNG**（`if not disabled and _fast_fetch_enabled()`）+ 新增 `_feature_transforms_cached()`（按 dataset 身份缓存遍历结果）+ 对照开关 `AL_HARDNESS_FORCE_SNAPSHOT=1`；新增 `tests/test_al_deterministic_sampling.py`（7 项） | hardness 每批（8 样本）都做一次 `collect_feature_transforms` + `snapshot_rng()`（含 CUDA `get_rng_state_all()`）+ 还原；而本 run `image_augment=False`（随机增强全部 gate 在 `train=self.image_augment and not policy_eval`）⇒ 取 item 不消耗 RNG、快照是**恒等操作**、纯开销 | ① 语义不变：`disabled` 为空时取 item 前后 `random/numpy/torch` RNG 状态**逐位相同**；② 有增强要关时**仍走**快照/还原（安全底线不依赖开关）；③ 扫描墙钟略降（幅度待 v45 实测） | 独立；与 #6（分片）、#1/#2 无耦合。⚠️ 开关在**进程启动时读一次**（改开关要重启） | `3e1ea94` | 🧪 部分验证（CPU 单测 7 passed；真机收益待测） |

## §1.5 改动本意速查（**debug 入口**；2026-10-12 用户要求）

> 用法：**出问题时先看这里** —— 回答三个问题：① 这是不是我改的？② 它现在算不算坏？③ 怎么立刻退回去？
> 每条的字段：**意图（替我们保证什么）· 不变量（永远必须成立）· 出错症状（怎么认）· 快速回滚（一步退）· 验证入口**。
> ⚠️ 回滚分两类：**运行时开关**（改环境变量重启即可）与**代码回滚**（`git checkout`）。
> 开关**都在进程启动时读一次** ⇒ 改开关**必须重启进程**才生效（这条踩过坑，别指望热改）。

### 🔧 `AL_*` 开关总表（排查时不必翻代码）

| 开关 | 默认 | 何时读取 | 作用 | 回退语义 |
|---|---|---|---|---|
| `AL_EVAL_SHARD` | `0`（关） | 进程启动 | `1` ⇒ 开环评测（`active_val` + `train_monitor`）按卡分片，每卡恒 `k=ceil(n/W)` 条、汇总去重 | 设回 `0`（或不设）⇒ 恢复"每卡跑整批" |
| `AL_HARDNESS_SHARD` | 关 | 进程启动 | `1` ⇒ hardness 打分按 `ids[rank::world]` 帧级分片 + `all_gather_object` 汇总 | 不设 ⇒ 每卡各算全量（慢但等价） |
| `AL_HARDNESS_FORCE_SNAPSHOT` | `0`（关） | 进程启动 | `1` ⇒ **强制**每批都做 RNG 快照/还原（O1 的 A/B 对照用） | 不设 ⇒ O1 的"无增强要关时跳过快照"生效 |
| `AL_SCAN_SKIP_HARDNESS` | 关 | 启动器注入（仅预扫 worker） | `1` ⇒ 预扫只补 scout，不进 `_select()`（不跑 hardness/active_val） | 不设 ⇒ 完整流程（预扫会顺带扫 hardness） |
| `AL_VIDEO_BACKEND` / `video_backend` | `pyav` | 数据集构造 | ❌ **未采纳**（torchcodec 在本机 ABI 不匹配、收益有限） | — |
| `SAVE_EVERY` / `HF_PASS_INTERVAL` | 训练脚本默认 | 进程启动 | 存档节奏（DCP 步数）/ HF 里程碑；**必须用 `--env` 或 shell 变量传**（`base_env()` 白名单已放开） | 去掉 ⇒ 回默认 |

### 逐条本意

| 编号 | 意图（替我们保证什么） | 不变量（永远必须成立） | 出错症状（怎么认） | 快速回滚 | 验证入口 |
|---|---|---|---|---|---|
| **#6** 评测轨迹分片 | 训练态评测**不再每卡跑整批**（省 `1/W` 墙钟），且**集合通信次数对齐** | ① 每卡前向次数**恒等**（`k=ceil(n/W)`，回绕补齐）；② 汇总后逐轨迹 MSE 与"单卡全量"**逐位一致**；③ 汇总按 episode id **去重**（`merged ≤ n`，多于 n 说明去重失效） | ① **静默死锁**（无报错、卡住 ⇒ 最危险）；② 日志 `[open_loop][shard]` 缺失或 `merged=n_trajs` 不符；③ 指标与历史 run **不该有**的偏差（分片本身不改变口径） | `AL_EVAL_SHARD=0`（默认关，需重启）；代码回滚 `git checkout 7164cde~1 -- lingbotvla/auto_learning/real/backend.py`（`7164cde` = 分片那次提交） | 日志匹配 `[open_loop][shard] world=W local=k merged=n/n`；单测 `pytest tests/test_al_eval_shard.py`（11 项） |
| **#9** O1：跳过快照 | 省掉"每批一次无用的 CUDA RNG 快照/还原"（本 run `image_augment=False` ⇒ 它是恒等操作） | **`disabled` 为空时，取 item 前后 `random/numpy/torch` RNG 状态逐位不变**；有增强要关时**仍**快照/还原（安全底线与开关无关） | ① 训练**随机性异常**（同 seed 两次结果不同、图像增强"看起来生效了却没生效"）；② 报告里 `rng_touched` 与实际不符；③ 硬扫结果与历史不可比 | `AL_HARDNESS_FORCE_SNAPSHOT=1`（**需重启**，强制旧行为）；代码回滚 = 删掉 `if not disabled and _fast_fetch_enabled():` 早退，或 `git checkout 3e1ea94~1 -- lingbotvla/auto_learning/real/determinism.py` | `pytest tests/test_al_deterministic_sampling.py`（7 项）+ `tests/test_b1_v0_2_fixes.py` |
| **#2** confirm 复用 scout | confirm 的 4 条中 2 条与 scout 重复 ⇒ **只新评 2 条**，省一半推理 | 判决值 = "scout 已测 2 条（复用）+ 新评 2 条"合并后的**几何平均**，且 == 一次性评 4 条的 `gmean_mse`（误差 < 1e-12） | ① 日志缺 `[open_loop][confirm] … 复用 scout N 条 + 新评 M 条`；② 事件里 `confirm_reused/fresh_trajs` 不是 `2/2`；③ `confirm_*` 与历史 run 逐位比较**有偏差**（口径已变，属预期） | 无开关；代码回滚 `git checkout d014aa6~1 -- lingbotvla/auto_learning/real/backend.py lingbotvla/auto_learning/orchestration/scheduler.py` | `pytest tests/test_al_confirm_merge.py`（5 项）；日志 `[open_loop][confirm]` |
| **#8** 面板池子/进步 | 池子状态从**事件流推导**改为读 **DCP 注册表**（权威），且能看 DEFER 原因/进步/轨迹 | 面板显示的状态计数与 DCP `auto_learning.registry.records` **完全一致**；只读（不写 DCP、不占 GPU） | ① 面板出现 `⚠️ 读 DCP 注册表失败`；② 池子计数与 `--pool` 明细行数不符；③ 快照停在旧 `global_step`（正常：读的是最近一份存档） | 无开关；`git checkout 606c67a~1 -- tools/auto_learning_pipeline/al_panel.py` | `al_panel.py <run> --pool`（计数与 DCP 一致即可） |
| **#3** v45 帧上限 200 | 单任务扫描量 600 → 400 帧（−33%），帧多的任务省时间 | 压缩是**均匀**的（日志给出 `A → B` 帧数）；**不影响判定口径**（hardness 只做采样权重） | ① 日志缺 `（帧数上限 200：… A → B 帧）`；② 扫描墙钟没降（说明配置没生效）；③ 硬度分布异常偏（压缩过度时看 p90） | 改回 `hardness_max_frames_per_traj: 300`（需重启）；**只对 v45 生效**，v44 不受影响 | 日志匹配 `帧数上限 200`；对比单任务扫描用时（v44 基线 275–466 s） |

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

**当前待验证：9 条（#1–#9）。下次上机（无训练在跑时）优先按编号顺序验证；**
**排查入口：先看 §1.5（扰动本意/症状/回滚）与 `AL_*` 开关总表；**
**#6（轨迹分片）需真机 7/8 卡 smoke：确认不死锁、指标与不分片逐位一致、墙钟下降。**
