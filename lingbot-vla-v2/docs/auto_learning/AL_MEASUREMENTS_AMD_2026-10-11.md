# AL 真机实测记录（2026-10-11，AMD 8 卡机）

> 本文只放**实测数字与现象**，作为调参和排障的基线。方案与决策见
> `../../.workbuddy/memory/knowledge/decisions.md`（工作区记忆，不在本仓库）。

## 1. 本轮 run 一览

| run | 卡 | GBS | 起点 | 结局 |
|---|---|---|---|---|
| `al_v41` | 6（0–5） | 30 | 400 步 HF | 正常收工于 **step 801**，`stop_reason = max_new_tasks_attempted_reached(6)`；通关 5/15（3 bootstrap + 2 训练所得） |
| `al_v42` | 2（6,7） | 10 | 801 步 DCP | **OOM 崩溃**（`HIP out of memory` step 1）⇒ micro=5 在 2 卡上放不下 |
| `al_v43`（第一次） | 8 | 40 | — | 配置未生效（见 §3），主动停掉 |
| `al_v43`（第二次） | 8（0–7） | 40 | 801 步 HF | **step ~150 崩于 GPU 故障**：`rank1(GPU1) hipErrorIllegalAddress` ⇒ NCCL 看门狗 SIGABRT |
| `al_v44` | **7**（0,2,3,4,5,6,7） | **35** | 801 步 HF | 运行中（见 §5） |

## 2. 步时与吞吐

| 配置 | 稳态步时 | 对照 |
|---|---|---|
| 6 卡 GBS=30（v41） | 13.0–13.5 s/it | 基线 |
| 8 卡 GBS=40（v43） | **8.4–8.7 s/it** | 每步多 33% 样本，还快 **35%** |
| **7 卡 GBS=35（v44）** | **10.4–10.9 s/it**（中位 10.92） | 比 v41 快 **~29%** |

⚠️ `s/it` 是**平均**步时：unit 边界那一步含开环评测/难度扫描，实测出现 **21.3 / 33.3 / 324.7 s/it** 的离群点。
判读要用"稳态"口径（剔除 `step % eval_interval == 0` 的边界步）。

## 3. `--env` 才是环境变量的正确传法（血泪）

`al_launch.py` 的 `base_env()` 给子进程构造环境时**只自动透传 `AL_` 前缀**的变量 + 显式 `--env KEY=VALUE`：

```bash
# ❌ 无效：SAVE_EVERY / HF_PASS_INTERVAL 不带 AL_ 前缀，被白名单丢掉
SAVE_EVERY=400 python experiment/robotwin/al_launch.py …
# 实测后果：日志仍打「每 1000 步 DCP」「每训练 2 个任务直接 HF」

# ✅ 有效
python experiment/robotwin/al_launch.py … --env SAVE_EVERY=400 --env HF_PASS_INTERVAL=0
```

* 2026-10-11 起 `SAVE_EVERY` / `HF_PASS_INTERVAL` 已加入白名单（`al_launch.py` 的 `base_env()`），
  但**显式 `--env` 仍是最稳写法**；
* `AL_HARDNESS_SHARD=1` 必须 `--env`（见 §4）。

## 4. hardness 扫描：分片开关没传 = 白烧 N 倍

| 状态 | 日志判据 | 单任务 271 样本的墙钟 |
|---|---|---|
| ❌ `AL_HARDNESS_SHARD` 未传 | `本 rank 271/271 样本`（每 rank 全量） | **~21 分钟**（实测 334 s 才走到 64/271） |
| ✅ `--env AL_HARDNESS_SHARD=1` | `本 rank 33–34/271 样本` | **178.4 s（≈3 分钟，7×）** |

* 分片实现：`real/backend.py` 的 `ids = ids_all[_rank::_world]` + `all_gather_object` 汇总；
* 每任务扫描量 = `hardness_probe_fraction`(0.05) × 该任务 train 轨迹的**全部帧**（单轨迹上限 300 帧）；
  实测 `place_object_stand`：2/40 条轨迹 ⇒ 271 样本；`stack_bowls_two`：600 样本 / 396 s；
* 数据加载是瓶颈（`data` 占 91–92%，`score` 只占 8%）⇒ GPU 利用率常年 0–1%，**这不是卡死**。

## 5. GPU1 损坏判定（2026-10-11 实测）

| 检查 | GPU1 | 正常卡（0/2/3…） |
|---|---|---|
| `rocm-smi` 利用率 | **恒 100%**（无进程） | 0% |
| 温度 | 31 °C（闲置级） | 25–27 °C |
| 功耗 | **38 W** | 15 W |
| 显存占用 | 26.7 MB（无进程） | 26.7 MB |
| 设备枚举 | ✅ 正常（W7900D） | ✅ |
| **最小 HIP 自检**（建张量→H2D→matmul→D2H） | **120 s 未完成**（H2D 处阻塞，被 timeout 杀） | 秒级通过 |

⇒ 典型"挂死"特征（与 launcher 内置警告描述一致）。**结论：把 GPU1 从 `--gpus` 里去掉**。

## 6. 重启/停止的正确姿势

```bash
# ⚠️ 只杀 al_launch.py 不够：训练进程会变成孤儿（PPID=1）继续占着 24.8 GB/卡
# ① 按进程组杀（最彻底）
for P in $(pgrep -f "[a]l_50task_bf16.sh"); do PG=$(awk '{print $5}' /proc/$P/stat); kill -9 -- "-$PG"; done
# ② 补杀孤儿
pkill -9 -f "[t]asks/vla/train_lingbotvla.py"
# ③ 复核：4 类进程全为 0，且 8 张卡显存回落 ~27 MB
```

## 7. 缓存与复用

* **scout 预扫**：50 任务 / 8 卡 ≈ **3 分钟**（4.4 任务/分）；缓存命中时 `[plan] 运行模式：reuse`，**跳过预扫**；
* **cache 文件名 slug** = 模型标识（`--model-name` > checkpoint 路径尾部 4 段）⇒ 换 checkpoint 即换一套缓存；
* `hardness_cache` 会**校验内嵌 `model` 字段**，不匹配就忽略并重算（日志：`因模型不同忽略缓存`）；
* 同一 run 重启后，事件流 `auto_learning_events.jsonl` **append**，`bootstrap` 事件会重复（50 任务出现 100 条）
  ⇒ 统计通关数要**按任务去重**，别看事件条数。

## 8. bootstrap（免费 PASS）

* Step 0 对**全部 50 个任务**各跑一次 **scout（2 条 val）**，且走 **scout 缓存**（实测 `scout_eval_wall_seconds=0.0003 s`）；
* 判据 = **scout 的 GMean ≤ 阈值**（不是 `active_val`）；scout 过线后再跑 **confirm（4 条 val，`val_ids[:4]`）**，
  两者都过才 `ReasonCode.BOOTSTRAP_PASS` 计入 `bootstrap_passed`；
* 实测：`al_v41` 免费 3 个、`al_v43` 4 个、`al_v44` 4 个（各占 50 任务的 6–8%）。

## 9. 事件流与判定口径（读数据时别混）

| 指标 | 谁在用 |
|---|---|
| `task/<t>/scout_gmean_mse` | bootstrap 判定、候选池排序（来自缓存或 rescan） |
| `task/<t>/active_val_gmean_mse` | **unit 末尾 PASS 判定** + 成长轨迹（实测 4 条 val） |
| `task/<t>/active_train_gmean_mse` | 过拟合预警（`train_val_gap_ratio`） |
| `task/<t>/val_to_pass_threshold` | 倍率（**≤1 过线**） |
| `task/<t>/pass_threshold_gmean_mse` | 该任务的阈值线 |

⚠️ 同一任务上「scout 超标、active_val 达标」是**正常现象**（两套不同轨迹集）。
例：`place_object_stand` step 0：scout 0.0235（1.116×，未过）vs active_val 0.0182（0.864×），
训练 50 步后 active_val 0.0146（**0.69×** ⇒ 判 PASS）。
