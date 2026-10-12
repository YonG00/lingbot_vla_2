# 开环评测多卡分片：真机修复与实测（2026-10-12）

> 结论先行：**`AL_EVAL_SHARD=1` 从落地起就没能在真机跑通**（今天第一次上机就崩）。
> 本轮修好后首次真机验证通过：各卡推理次数**结构上恒等**、预检 0 失败，
> 边界步额外开销 **58.0s → 36.0s（提速 1.61×）**。
> 权威账本：`AL_PENDING_CHANGES.md` §1 #13 / §1.5 #13；本文是设计+实测细节。

---

## §1 崩点长什么样（真机两次，同一处）

从 2400 HF 起跑、带 `AL_EVAL_SHARD=1`，**首个 `active_val` 评测**被拦：

```
❌ 多卡评测一致性预检未通过（tag=al_move_can_pot_val_6711597a，world_size=7）
   - rank0: ids=[1109] n_starts=3 len_ds=143
   - rank1: ids=[1110] n_starts=4 len_ds=151
   - rank2: ids=[1111] n_starts=3 len_ds=148
   - rank3: ids=[1116] n_starts=4 len_ds=157
```

**注意**：预检**没有做错事** —— 它是 fail-closed 安全阀，挡住的是"集合通信次数错配
⇒ 其余 rank 永久等在 all-gather ⇒ 600s 看门狗 abort，且真因会被
`NCCL communicator was aborted` 埋掉"。

---

## §2 两个独立缺陷（缺一不可地都要修）

### 缺陷 ①（真 bug）推理次数真的不等

- 分片只对齐了"**轨迹条数** `k=ceil(n/W)`"，没对齐"**每条轨迹几个窗口**"。
- 窗口数 = `ceil(回合帧数 / stride)`，`stride = chunk_size = 50`。
- 各卡拿到的轨迹帧数不同（143/151/148/157）⇒ 窗口数 3 或 4 ⇒ **各卡次数不等**。

实测（`move_can_pot` val，`n=10, W=7 ⇒ k=2`）：

| | 各卡 `n_starts` |
|---|---|
| 修复前 | `[6, 7, 6, 7, 7, 6, 7]` ❌ |
| 修复后 | `[4, 4, 4, 4, 4, 4, 4]` ✅ |

### 缺陷 ②（判据过严）预检比了"本该不同"的字段

`_preflight_verdict` 原要求 `tag / ids / n_starts / len_ds / warmup` **五者逐一相同**。
但分片的**设计**就是让各卡评不同轨迹；而且 `tag` **带 ids 指纹**
（`evaluator.py`：「ids 指纹进 tag，否则 2 条 scout 与 4 条 confirm 会共用同一个
episode_ids 文件路径」）⇒ **各卡 tag 必然不同** ⇒ 必然误报。

---

## §3 修法（用户方案：**先统计全局、算好趟数、传给每张卡**）

```
全局条数 n ──→ k = ceil(n/W) ──→ 公告 {k, n, world, rank, expected_steps}
                                    │
每卡：stride_e = ceil(帧数_e / k) ──→ 实际步数（每条轨迹恰好 k 个窗口）
                                    │
预检：① 各卡实际步数必须**相等**        （防死锁）
      ② 实际步数必须 == **期望步数**     （防"7 卡一起少算"——用户补强）
      ③ 各卡 ids 并集必须 == **全局 n** （防漏评/越界）
      ④ 任一卡报错 ⇒ 全员一起停         （fail-closed 保留）
```

| 落点 | 内容 |
|---|---|
| `open_loop_validation.py::eval_shard_indices` | 分片切分的**唯一事实来源**（`backend.py` 改为转发） |
| `open_loop_validation.py::adaptive_episode_starts` | 逐回合按帧数反算 stride ⇒ **每条轨迹恰好 k 个窗口** |
| `open_loop_validation.py::_shard_k_for` | 优先用**上游公告的全局 k**（不可用本卡子集反推） |
| 分片上下文 | **扁平环境变量** `AL_EVAL_SHARD_CTX_{K,N,W,R}` + `AL_EVAL_SHARD_EXPECT_STEPS` |
| `backend.py::evaluate` | 切分轨迹**之前**公告上下文；`expected_steps = k × 本卡条数` |
| `_preflight_verdict` | **分模式**：非分片五字段全比（逐字不变）；分片只比 `n_starts/warmup` |

### 关键口径影响（必须知道）

窗口起点与 `stride=50` 不同 ⇒ **实测值与历史 run 不可逐位比**，也偏离按 `stride=50`
标定的 `pass_thresholds`。本次实测（click_bell，50 步）：

| 指标 | 训练前 | unit 末 | 说明 |
|---|---|---|---|
| `active_val_mse` | 0.007222 | **0.005757** | ↓20.3% |
| 倍率 = val_mse ÷ 门槛 | 1.024x ❌ | **0.816x ✅** | 判定信号清晰 |
| `baseline_mse`（固定常数） | 0.2892295487784861 | **逐位相同** | 分片合并的可复现性 OK |

⇒ **方向与量级正常**，但"跨模式逐位等同"仍未验证（列为待办）。

---

## §4 实测收益（v49 vs v44，基于相邻 `Step` 真实时间戳）

| | v44（旧：每卡全员冗余） | v49（分片：每卡 1 条） |
|---|---|---|
| 训练步中位 | 11.0 s | **11.0 s**（未拖慢） |
| **边界步额外开销**（评测+扫描） | **58.0 s**（n=85） | **36.0 s**（n=1） |
| 提速 | — | **1.61×（省 38%）** |

⚠️ 两个说明：
1. 边界步开销**含 hardness 扫描**，而扫描按**帧**分片、**不吃**本优化 ⇒ 纯评测部分的提速**大于** 1.61×。
2. v49 目前只有 **1 个**边界步样本 ⇒ 等 3–5 个 unit 后再取稳定值。

真机日志证据（各卡步调一致）：

```
[open_loop][shard] task=click_bell split=active_val 共 4 条 ⇒ world=7、本 rank 处理 1 条
[rank0/7] 推理循环（1 次推理 / 76 帧） 完成 (+1.67s, 累计 2.58s)
[rank3/7] 推理循环（1 次推理 / 76 帧） 完成 (+1.67s, 累计 2.58s)
[rank5/7] 推理循环（1 次推理 / 77 帧） 完成 (+1.69s, 累计 2.61s)
```

---

## §5 本轮踩坑（都值得记住）

| # | 坑 | 后果 | 教训 |
|---|---|---|---|
| 1 | **注入式测试命名空间**：`tests/test_multirank_eval_preflight.py` 只 exec 一批白名单函数 | 我新增的模块级变量/函数在其中**不存在** ⇒ `NameError`，连踩 3 次（`_SHARD_CONTEXT` → `eval_shard_context` → 内联） | **判据函数只内联读 `os.environ`**，不调用辅助函数、不用模块级常量 |
| 2 | `n_ids` 取 `len(ids)` | 分片后 `ids` 只是本卡子集 ⇒ 并集 4 被当成"越界 4≠1" | 全集条数**必须由上游公告** |
| 3 | 把 `tag` 列为"必须相同" | 各卡 tag 含不同 ids 指纹 ⇒ 必然误报 | 分片下"身份指纹"类字段一律不比 |
| 4 | `k` 用本卡子集反推 | 全局 k=2 时算成 1 ⇒ 抽帧步长 143（欠采样） | k 与 n 都取**全局**值 |
| 5 | 自写单测喂"理想数据" | **一个都没抓到**上面 4 个 bug | **必须跑既有回归 + 上机**：真机与既有用例才抓到了 |

---

## §6 回滚与验证入口

| 项 | 值 |
|---|---|
| **快速回滚** | `AL_EVAL_SHARD=0`（默认即关；**进程启动读一次 ⇒ 需重启**） |
| 或 | `git checkout <sha>~1 -- lingbotvla/utils/open_loop_validation.py lingbotvla/auto_learning/real/backend.py` |
| **验证入口（CPU）** | `PYTHONPATH=$PWD /opt/robotwin-env/bin/python -m pytest tests/test_al_eval_shard_preflight.py -q`（**26 passed**） |
| **验证入口（真机）** | 日志出现 `一致性预检通过` 且无 `一致性预检未通过`；各卡 `推理循环（N 次推理）` 的 **N 逐卡相同** |
| **出错症状** | ① `n_starts=x vs y` ⇒ 趟数没锁死；② `[shard] 分片并集 N ≠ M` ⇒ 漏评/越界；③ `实际步数 ≠ 期望步数` ⇒ 分片步调不一致 |
