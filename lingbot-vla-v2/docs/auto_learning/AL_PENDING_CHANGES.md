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
| 9 | 10-12 | `tools/auto_learning_pipeline/al_panel.py`：**默认视图改为 15 行紧凑面板**（纯 ASCII / 宽度自适应 / 在训任务历史 3 unit / 总用时与预估 / 池子 `pass=7 defer=12 … exhausted=0`）；原长表格移入 `--wide`；`s/it` 与 ETA 改口径 | **改动本意（用户 2026-10-12 四问）**：<br>① **不变量**：面板**只读**（不写 DCP/不碰训练）；各节数据来源不变（池子=DCP 注册表、轨迹=事件流、步时=训练日志）；`--pool/--progress/--traces/--all/--trainers` 语义不变。<br>② **出错症状**（怎么一眼看出坏了）：面板报 `NameError/AttributeError` ⇒ 打包字段口径变了（如 `best_gmean_mse` 只在 PASS 后有值、`series[metric][task]` 是 list 不是 dict）；`池子` 行不显示 `exhausted=` ⇒ 池子读取失败退化了；`s/it` 远大于日志稳态值 ⇒ 又用回累计均值了。<br>③ **快速回滚**：`git checkout <sha>~1 -- tools/auto_learning_pipeline/al_panel.py`（单文件、无外部依赖）。<br>④ **验证入口**：`/opt/robotwin-env/bin/python tools/auto_learning_pipeline/al_panel.py <run>`，判据 = 行数 ≤20、无框线字符、`Step` 不带 `/20000`、池子含 `exhausted=`、历史 3 行、`用时` 的"已训"与日志 `[HH:MM:SS<…]` 一致 | 独立（只读工具）；`--wide` 保留全部旧细节 | `93db002` | ✅ 已验证（v44 实跑 15 行；`已训 11.8h` 与日志 `11:45:20` 吻合） |
| 6 | 10-11 | `real/backend.py`：`evaluate()` 新增**轨迹分片**（`AL_EVAL_SHARD=1` 开启，**默认关**）+ `_eval_shard_indices()` / `_gather_and_merge_shards()` / `_all_gather_object()`；`merge_trajectory_metrics()` 支持 `metric_valid` / `note` 覆盖 | 训练态评测现状是**每个 rank 各跑整批**（实测 v43 8 卡 / v44 7 卡：每卡都跑 12/20 条）；轨迹互不依赖、逐轨迹 MSE 与卡数无关、聚合是几何平均（可分组）⇒ 可分片 | ① **不死锁**（每卡恒 `k=ceil(n/W)` 条 ⇒ 集合通信对齐）；② 汇总与单卡全量**逐位一致**；③ 墙钟 ≈ 现值的 `1/W`（n=8、W=8 ⇒ 每卡 1 条） | ⚠️ **CPU 测不出集合通信错配** ⇒ 必须真机验证；只改 `active_val` + `train_monitor`，`scout`/`confirm`/`rescan` 暂不动 | `7164cde` | 🧪 部分验证（CPU 单测 11 项 + 本机回归 119 passed） |
| 7 | 10-11 | 消歧义注释：`orchestration/scheduler.py`（`*_gmean_mse` 是**判定口径**、`*_nmse` 是历史遗留显示量）、`state/registry.py`（`active_val_probe_trajs` **实际决定 confirm 条数**）| 命名历史遗留易误读；改名会让 v41–v44 产物不可比 ⇒ 按三选一取**加注释** | 注释与当前实现一致；不改任何字段名/指标名 | 独立 | `d014aa6` | ✅ 已验证（纯注释） |
| 8 | 10-11 | `tools/auto_learning_pipeline/al_panel.py`：新增 **DCP 注册表读取**（`latest_dcp` / `load_registry`）+ **池子摘要**（默认视图 2 行）+ 4 个参数（`--pool` / `--progress` / `--traces` / `--watch`）；新增技能 `lingbot-training-monitor` | 用户要"一眼看池子/进步/GMean"：原面板只能从**事件流推导**状态，**推不出 DEFER/EXHAUSTED**、也没有 attempts/lp/过拟合/DEFER 原因 | 默认视图出现 `池子来源: DCP global_step_N`、`池子: TRAINING 1 \| DEFER 9 \| PASS 7 \| CANDIDATE 33`、`当前任务: X（第 N 次尝试，已训 M 步，当前倍率 Kx）`；`--pool` 出明细表（含 DEFER 原文原因）；`--progress` 按 lp50 降序；`--traces N` 多任务轨迹 | 独立；**只读**（`torch.load(map_location=cpu)`），训练期安全 | `606c67a` | ✅ 已验证（v44 实跑：池子 1/9/7/33 与 DCP 注册表一致；`--progress` 12 行、`--traces 2` 正常） |
| 9 | 10-12 | `real/determinism.py::deterministic_sampling`：**没有增强要关时跳过快照 RNG**（`if not disabled and _fast_fetch_enabled()`）+ 新增 `_feature_transforms_cached()`（按 dataset 身份缓存遍历结果）+ 对照开关 `AL_HARDNESS_FORCE_SNAPSHOT=1`；新增 `tests/test_al_deterministic_sampling.py`（7 项） | hardness 每批（8 样本）都做一次 `collect_feature_transforms` + `snapshot_rng()`（含 CUDA `get_rng_state_all()`）+ 还原；而本 run `image_augment=False`（随机增强全部 gate 在 `train=self.image_augment and not policy_eval`）⇒ 取 item 不消耗 RNG、快照是**恒等操作**、纯开销 | ① 语义不变：`disabled` 为空时取 item 前后 `random/numpy/torch` RNG 状态**逐位相同**；② 有增强要关时**仍走**快照/还原（安全底线不依赖开关）；③ 扫描墙钟略降（幅度待 v45 实测） | 独立；与 #6（分片）、#1/#2 无耦合。⚠️ 开关在**进程启动时读一次**（改开关要重启） | `3e1ea94` | 🧪 部分验证（CPU 单测 7 passed；真机收益待测） |

| 10 | 10-12 | `tasks/vla/train_lingbotvla.py`：未显式传 `--train.step_offset` 时，**自动从 `--model.model_path` 解析 `global_step_<N>`** 作为起始编号（只认 `global_step_<N>`，不认裸数字，避免把 `v41` 当 41） | v44 实测 `step_offset=0` ⇒ 从 `…/al_v41_step801_hf/global_step_801/hf_ckpt` 起新 run，但存档编号从 1 开始、与来源 801 脱节 | ① 日志出现 `[step_offset] 未显式指定 ⇒ 自动从模型路径推导 = 801…编号从 802 起`；② 存档仍落在 `save_steps` 整数倍（1200/1600）；③ 显式传值优先；④ DCP 成功加载时被 DCP 的 global_step 覆盖（原语义） | ⚠️ `max_steps` 是**绝对**目标 ⇒ LR 调度步数 = `max_steps - offset` 变短（既有语义）；独立 | `TODO_SHA` | 🧪 部分验证（py_compile + 正则 6/6 用例） |
| 11 | 10-12 | 新增 `lingbotvla/auto_learning/state/pool_state.py`（池子独立存档读写）+ `tests/test_al_pool_state.py`（16 项） | Auto Learning 的池子（status/attempts/lp50/`sample_probs`）原本**只随 DCP** 的 `extra_state` 走；从 **HF 权重**起新 run 判 `fresh` ⇒ 池子从零 ⇒ 已 PASS 的被重训、刚 DEFER 的被立刻重试 | ① 导出→读回**逐字段相等**（含 `sample_probs`）；② 缺失/损坏 ⇒ `None` 不抛；③ 原子写不留 `.tmp`；④ 开关 `AL_POOL_STATE_EXPORT／NO_IMPORT／RESTORE_RNG／MIRROR／PRUNE／FILE`（**进程启动时读**）；⑤ `sampler_rng` **默认不恢复** | 独立；纯 stdlib（无 torch）；单份 ≈9.4 MB | `TODO_SHA` | 🧪 部分验证（单测 16 passed） |
| 12 | 10-12 | `tasks/vla/train_lingbotvla.py`：导出挂在 `save_hf_checkpoint_best_effort()` 内（⇒ 周期/轮末/收尾**三个存档点全覆盖**），路径 `<global_step_N>/pool_state.json`，`final_only` 时另写 `<…>/hf_ckpt/pool_state.json`；`real/build.py`：`Scheduler` 之后**仅当不走 DCP** 时导入池子 | 需求 2 落地：HF 续训也能继承池子。DCP 路径由 `load_extra_state()` 自动恢复 ⇒ 必须**跳过**导入（否则日志把"预期跳过"误导成"丢状态"） | ① 日志出现 `[pool_state] 已导出池子：…（step=2400，9.4 MB）` 与池子计数；② HF 续训出现 `[pool_state] 已导入池子：…（step=2400）`；③ DCP 续训出现 `本次从 DCP 续训 ⇒ …跳过 pool_state.json 导入（这是预期行为，不是丢状态）`；④ 导出失败**只告警不中断**；⑤ 任务集合对不上 ⇒ 摊开三类差异（多出/缺少/共有），**不拦截** | ⚠️ **前置**：导出挂在 `save_hf_checkpoint_best_effort` 内 ⇒ `save_hf_weights=false` 时不导出（`final_only` 模式必开 HF ⇒ 满足）；与 #10 独立 | `TODO_SHA` | 🧪 部分验证（单测 16 passed + 静态接线断言） |
| 13 | 10-12 | `lingbotvla/utils/open_loop_validation.py`：新增 `eval_shard_indices`（分片切分**唯一事实来源**）、`adaptive_episode_starts`（每条轨迹恰好 k 个窗口）、`_shard_k_for`；`_preflight_verdict` **分模式判定**（分片下不比 `ids/len_ds`，改守 `n_starts` 相等 + **并集覆盖完整**）；`_eval_rank_report(..., n_ids=)` + `_multirank_eval_preflight` 透传全集条数；`real/backend.py::_eval_shard_indices` 改为**转发**共享实现 | **真机 al_v47 崩点**（从 2400 HF 起跑 + `AL_EVAL_SHARD=1`）：首个 `active_val` 评测被多卡一致性预检拦下 —— `n_starts` 实测 rank0=3 / rank1=4（`len_ds` 143/151/148/157）。两个独立缺陷叠加：① 预检要求 `ids/len_ds/n_starts` 逐一相同，而分片的**设计**就是让 `ids` 不同 ⇒ 必然误报；② **推理次数真的不等**：分片只对齐了「轨迹条数 k」，没对齐「每条轨迹几个窗口」（窗口数 = `ceil(帧数/stride)`）⇒ 真跑下去会在 all-gather 上卡死（预检拦对了） | **改动本意（用户 2026-10-12 四问）**：<br>① **不变量**：分片模式下各 rank 的 `n_starts`（= 集合通信次数）**必须逐 rank 相等**；各 rank 的 `ids` 并集**必须恰好等于全集** `n_ids`（不漏不越界）；**非分片路径**（`AL_EVAL_SHARD=0`，默认）`_preflight_verdict` 与 `per_episode_starts` 行为**逐字不变**。<br>② **出错症状**：日志出现 `多卡评测一致性预检未通过` + `n_starts=3 vs 4` 之类 ⇒ 窗口数没锁死；出现 `[shard] 分片并集 N 条 ≠ 全集 M 条` ⇒ 漏评/越界；实测 GMean 与历史 run 不可逐位比（**预期**，见③）。<br>③ **快速回滚**：`AL_EVAL_SHARD=0`（**进程启动时读一次 ⇒ 必须重启**）；或 `git checkout <sha>~1 -- lingbotvla/utils/open_loop_validation.py lingbotvla/auto_learning/real/backend.py`。<br>④ **验证入口**：`PYTHONPATH=$PWD /opt/robotwin-env/bin/python -m pytest tests/test_al_eval_shard_preflight.py -q`（真机 24 passed）；真机看首轮评测是否出现 `[open_loop][shard]` 且不再报预检失败 | ⚠️ **口径**：窗口起点与 `stride=50` 不同 ⇒ 实测 GMean 偏离按 stride=50 标定的 `pass_thresholds` 与 `task_baseline`（**未重标定**，用户选 5b：先看偏差）；与 #6（分片本体）**强耦合**：本改动修的就是 #6 真机跑不通的问题 | `TODO_SHA` | 🧪 部分验证（新单测 24 passed 于真机 robotwin-env；**真机训练未验证**） |
| 14 | 10-12 | 删 nmse 误导：① `configs/auto_learning/al_eval2_target15_m60.yaml` 删 `pass_nmse: 0.35` 兼容占位；② `experiment/robotwin/al_50task_bf16.sh` 启动横幅由**硬编码** `及格线 = pass_nmse=0.35（nmse ≤ 该值判 PASS）` 改为**按配置实际判据**显示（`read -r` 增第 6 个变量 `AL_PASS_METRIC`、内嵌 python 多输出 `pass_metric`、横幅显示 `${AL_PASS_METRIC}`） | **用户 2026-10-12 指出**：本批 run 判据是 `pass_metric: gmean_mse`（实测 gmean ≤ 该任务阈值表），但配置里留着 `pass_nmse: 0.35`、启动横幅也硬编码打印 nmse 及格线 ⇒ **误导读日志的人**（本 agent 因此拿 `nmse ÷ 0.35` 算过倍率，被用户当场纠正）。**改动本意四问**：<br>① **不变量**：`pass_metric: gmean_mse` 时判定**只读**阈值表、**绝不读** `pass_nmse`（`decision/thresholds.py::pass_line` 已保证）；横幅显示值必须等于配置里的 `pass_metric`。<br>② **出错症状**：横幅打印 `判据 = （空）` ⇒ `AL_PASS_METRIC` 未取到；日志/文档里再出现「及格线 nmse ≤ 0.35」字样 ⇒ 又有地方硬编码。<br>③ **快速回滚**：`git checkout <sha>~1 -- <三个文件>`；⚠️ 注意 `pass_nmse` 在 `state/persistence.py::SEMANTIC_KEYS` 里 ⇒ 删它**会改配置指纹**（须在重算 baseline / 重扫之前完成，只需对齐一次）。<br>④ **验证入口**：两份配置 `yaml.safe_load` 后 `pass_metric=gmean_mse`、`pass_nmse` 不存在；`bash -n experiment/robotwin/al_50task_bf16.sh`；启动横幅出现 `判据 = gmean_mse` | `TODO_SHA` | 🧪 部分验证（配置解析 + bash 语法已验；**启动横幅未实跑**） |
| 15 | 10-12 | 分片修复（**5 个 bug 一次收口**，全部真机定位）：<br>**①协议越位**：分片切分原在「多卡协议**之外**」⇒ 框架「只有 rank0 算数 + 广播」把各卡切片作废（实测回收 7 份全同）⇒ 改为**协议内**切片 + 归并（`merge_ranks` 回调）；<br>**②默认关**：归并回调仅在策略注入时启用（默认 `None` ⇒ 非分片路径逐字不变）；<br>**③只注入不收回**：原 `finally` 清除与各卡返回时序竞态 ⇒ 后到 rank 读不到策略、不切片；<br>**④元凶·文件级竞态**：`_episode_ids_file` 白名单文件名原为 `&lt;tag&gt;.json`，注释假设「各 rank 内容相同、谁赢都一样」——该假设**在轨迹分片下不成立**，7 卡互相覆盖 ⇒ 后到 rank 读到**别人的切片**（实测 rank2 `run_ids=[1408]` 却返回 `per_traj_ids=[1403]`）⇒ 文件名改为 `&lt;tag&gt;_&lt;本卡切片指纹&gt;.json`；<br>**⑤尺子越界**：`world&lt;=1`（单卡扫描 worker）也启用自适应窗口（`k=ceil(n/1)`）⇒ 把「每 50 帧一个窗口」改成「每条轨迹摊 2 个窗口」⇒ **同一轨迹 MSE 差 0.5–4.7 倍**（实测 `dump_bin_bigbin` 0.147→0.684、`shake_bottle_horizontally` 0.0170→0.0087）⇒ 扫描与历史不可比；改为 `world&gt;1` 才启用（单卡恢复 stride=50） | **用户 2026-10-12**：「分片并行不能改数值」。真机 al_v49→v64 逐步定位：`active_val_gmean_mse=None` ⇒ 判定失效（0 个 PASS）<br>**改动本意四问**：<br>**①不变量**：`AL_EVAL_SHARD=0` 时与改造前**逐字一致**；分片下「每卡推理次数 == 上游公告 `expected_steps`」且各卡相等；**单卡（world≤1）扫描的 gmean 与历史同口径**（stride=50）。<br>**②出错症状**：`active_val_gmean_mse=None`；日志 `归并后逐轨迹条数 N ≠ 全集 M`；`各卡 ids=[[x],[x],…]` 全同；扫描值与历史差 &gt;5%（尤其长帧任务如 `dump_bin_bigbin`）。<br>**③快速回滚**：`AL_EVAL_SHARD=0`（默认即关，**进程启动读一次，需重启**）；或 `git checkout &lt;sha&gt;~1 -- lingbotvla/utils/open_loop_validation.py lingbotvla/auto_learning/real/backend.py`。<br>**④验证入口**：`pytest tests/test_al_eval_shard_merge.py tests/test_al_eval_shard_preflight.py tests/test_multirank_eval_preflight.py -q`（7+26+62 passed；1 个**既有**失败已用 HEAD 版对照证明与本改动无关）；真机日志出现 `[open_loop][shard] … 框架内归并后 N 条`（N==全集）且无「归并后逐轨迹条数」报错；扫描重跑与历史逐任务相对差 &lt;0.5% | `TODO_SHA` | 🧪 部分验证（单测全绿 + 真机 al_v62/v64 实测：归并失败 0、`active_val_gmean_mse=0.0067`、重扫 50/50 任务与历史差 ≤0.4%） |

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

| **#13** 分片预检修复 | 让「每卡推理次数」**结构上恒等**，使多卡分片评测真机可跑（#6 才真正可用） | 分片下各 rank `n_starts` 相等；`ids` 并集 = 全集 `n_ids`；非分片路径逐字不变 | 日志 `多卡评测一致性预检未通过` + `n_starts=x vs y`；或 `[shard] 分片并集 N ≠ M` | `AL_EVAL_SHARD=0`（**需重启**）或 `git checkout <sha>~1 -- <两文件>` | `pytest tests/test_al_eval_shard_preflight.py -q`（真机 24 passed） |

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

**当前待验证：15 条（#1–#15；存在两条编号 `9` —— 并行协作者各写一条，本 agent 未擅自改其编号）。下次上机（无训练在跑时）优先按编号顺序验证；**
**排查入口：先看 §1.5（扰动本意/症状/回滚）与 `AL_*` 开关总表；**
**#6（轨迹分片）需真机 7/8 卡 smoke：确认不死锁、指标与不分片逐位一致、墙钟下降。**
