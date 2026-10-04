# 训练中 open-loop validation 自检 使用文档

> 适用版本：`YonG00/lingbot_vla_2`（LingBot-VLA-v2 post-training）
> 相关文件：`tools/open_loop_selfcheck.py`、`lingbotvla/utils/open_loop_validation.py`
> 定位时间：2026-10-04

---

## 1. 这个工具解决什么问题

训练中原地 open-loop validation 最容易出的不是「崩」，而是**「能跑但结果悄悄错」**：
白名单把动作取成常数、评测用的 `use_cache` 与推理路径不一致、指标聚合口径与官方不同、
eval 结束后模型/随机数状态没还原……这些都不会报错。

本工具用**断言式自检**把这些点钉住。**纯 CPU，不加载模型权重**，秒级到十几秒。

---

## 2. 快速开始

```bash
cd /data/code/lingbot-vla-v2
PY=/data/miniconda3/envs/lingbotvla/bin/python

# ① 主流程（8 项，不碰全量数据集）
HF_HUB_OFFLINE=1 $PY -u tools/open_loop_selfcheck.py --task click_bell

# ② 秒级子集（只查划分 / 代码机制，不碰数据集）
HF_HUB_OFFLINE=1 $PY -u tools/open_loop_selfcheck.py --task click_bell --skip-dataset

# ③ 跨数据集逐值对照 —— ⚠️ 必须**单独**跑
HF_HUB_OFFLINE=1 $PY -u tools/open_loop_selfcheck.py --only-c3b --task click_bell
```

任何一项 FAIL 就非 0 退出，可直接挂进 CI / 训练前的 preflight。

---

## 3. 检查项

> ⚠️ 自检是**纯 CPU**的，验的是「数据 / 划分 / 代码机制」。**GPU 端到端**用
> `tools/open_loop_smoke.sh`（见第 6 节），两者互补、都要跑。

| ID | 查什么 | 失败意味着 |
|---|---|---|
| **C1** | 划分文件自洽：`train ∩ val = ∅`、`train ∪ val = 每任务回合数`、`monitor ⊆ train`、`monitor ∩ val = ∅` | 训练/评测数据串了 |
| **C2** | 块假设 `task = TASK_ORDER[episode_index // EPISODES_PER_TASK]` 与数据集真实标注一致 | 白名单指向了别的任务的回合 |
| **C3** | 白名单数据非 padding（**回归测试**）：前两帧 `is_pad == 0` 且值非零 | 又回到了「整段 chunk 全判 padding、动作全 0」那个 bug |
| **C4** | `use_cache` 机制：训练配置取值 + `handle_kv_cache(use_cache=False, fill_kv_cache=True)` 是否**不写缓存** | 去噪阶段没有 VLM 前缀条件（不报错、结果错） |
| **C5** | `_as_frames` 单测：0-d→(1,1)、**1-d→(1,D)**、2-d/3-d→(N,D) | 一维动作被当成「N 帧 × 1 维」⇒ 时间轴与维度轴对调 |
| **C6** | `_episode_index_map`：用真实数据集验证 `local_idx → episode_index` | 指标无法按完整 trajectory 聚合，会退化成按 chunk |
| **C7** | `MultiVLADataset.strict_getitem` 开关存在且**默认 False** | 要么 eval 没进 strict，要么误改了正式训练行为 |
| **C8** | **审计本身的自测**：故意破坏 3 处状态，看 `_audit_restore` 能不能抓到 | 恢复审计是摆设（假绿） |
| **C9** | 归一化统计来源一致：`VLADataset` 是否透传 `norm_stats_file`；两份 JSON 的 `count`/`mean` 是否确实不同 | 训练与 deploy 用了两套 stats（见 `docs/norm_stats_guide.md`） |
| **C3b** | 白名单 `idx=k` 与**不过滤**时绝对帧 `from+k` 的 state/action **逐值一致** | 索引空间又坏了（最强证据） |

---

## 4. 为什么 C3b 要单独跑

容器 cgroup 内存上限只有 **2 GiB**（`free` 却报 1007G）。主流程已经持有白名单数据集 +
metadata，再叠加一份**全量** `LeRobotDataset`（548k 帧）就会被 **OOM 杀掉（rc=137）**，
连前面已通过的检查结果都会丢。

⇒ `--only-c3b` 用**全新进程**只做这一件事。实测：

```
白名单 idx=3 [2.033667, 2.102997] vs 不过滤绝对帧 7159 [2.033667, 2.102997] ⇒ 一致
```

---

## 5. 两个容易踩的坑（写工具时踩过）

1. **必须用仓库自己的 `LeRobotDataset` 子类**
   （`lingbotvla/data/vla_data/base_dataset.py`）。
   索引空间修复**只打在这个子类**上，上游 `lerobot.datasets.LeRobotDataset` 仍然是坏的。
   用错类会得到「全是 padding」的假 FAIL。
2. **C2 是词表启发式，不是精确判定**。任务名与指令原文用词常常不同形：
   `blocks_ranking_rgb` 的指令里是单数 `block`、`pick_dual_bottles` 说的是 `catch`、
   `click_alarmclock` 写成 `alarm-clock`。所以：
   * 归一化（去连字符/空格/下划线）+ 去复数 + 对任务名的每个 token 取最大命中率
   * **硬门槛**只压在实际训练的那个 task 所在块上（>= 70%）
   * 全体报告通过比例；真错位会表现为「**另一个**任务名（最佳词不同）明显匹配得更好」

---

## 6. 配套：训练中的「恢复审计」

`open_loop_validation.py` 在每次评测的 `finally` 里会逐项断言临时状态已还原，
并打一行日志：

```
[open_loop] ✅ 恢复审计通过：逐模块 training 标志 / config.use_cache /
            image_augment / compile 开关 / torch+numpy+python RNG 全部还原
```

不一致时打 `❌ 恢复审计未通过` 并**抛 `RuntimeError`**（fail-fast）。
确认无碍可临时降级：

```bash
export OPEN_LOOP_AUDIT_STRICT=0     # 只告警，不抛错
```

> 审计覆盖：逐模块 `training` 标志、每个 `config.use_cache`、每个 `feature_transform.image_augment`、
> `_use_compile_predict_velocity`、torch CPU+CUDA / NumPy / Python `random` 的 RNG。
> **有意不查** `_compiled_predict_velocity`（评测后会置 None，下一步训练会重新编译）。

C8 就是用来证明这个审计**真的能抓到问题**的 —— 不是写完就算。

---

## 7. GPU 端到端冒烟：`tools/open_loop_smoke.sh`

自检是 CPU 的，验不到「训练能不能起来 / 显存够不够 / 开环评测能不能出数」。
这条链路用冒烟脚本，**一条命令 10 项检查**：

```bash
MICRO=10 GAS=1 STEPS=5 bash tools/open_loop_smoke.sh        # gbs = micro × gas
MICRO=1  GAS=1 bash tools/open_loop_smoke.sh                # 最小显存对照
```

检查项：训练跑完 N 步 / 无 OOM 与**真**异常栈 / 开环出数 / 逐轨迹 MSE /
`use_cache` 临时开关 / `attention→eager` 临时开关 / **归一化统计指纹** /
恢复审计通过 / 评测未失败 / 未落 checkpoint。非 0 退出。

> ⚠️ **检查里不能用裸 `Traceback`**：torch 的
> `UserWarning: ... Traceback of forward call that caused the error:` 会被误匹配。
> 要用 `Traceback \(most recent call last\)|ChildFailedError|OutOfMemoryError`。

实测（`MICRO=10 GAS=1`）：**10/10 通过**，峰值 78.64G/96G，StepTime 5.19s，开环评测 2.5s。

---

## 8. 离线（解耦）评测：`tools/open_loop_eval_inprocess.py`

训练中的评测被绑在训练进程上，想评一份权重就得先存 ckpt。这个脚本把它解耦：

```bash
python tools/open_loop_eval_inprocess.py \
    --ckpt <.../checkpoints/global_step_N/hf_ckpt> \
    --train-ids <json> --val-ids <json> [--dump-dir <dir>]
```

从 ckpt 建模型（复用 deploy 流程：`eager` + `use_cache=True`），跑**同一套**评测代码。
⇒ 任何已有 ckpt 都能直接评、**不用再存 72G**、迭代从 ~9 分钟降到 ~2 分钟。
`--dump-dir` 会把每个 chunk 的 **GT / pred / 模型输入**存成 `.npy`，供与官方逐值对拍。

