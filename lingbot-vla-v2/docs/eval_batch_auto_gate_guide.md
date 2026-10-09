# Eval Batch 评测并行 · 使用说明（2026-10-09 现行版）

> 适用：50-task GMean200 自动学习（以及任何走 `OpenLoopValidator.evaluate_ids` 的评测）。
> **一句话**：开 `auto` 后，**评测直接用训练的前向批大小成组批处理**，不做任何数值判定；
> 实测单次评测 **9.0 s → 2.5 s**、Bootstrap **~7 min → 1 min 49 s**，且**逐任务数值与串行一致**（差 0.01–0.63%）。

---

## 1. 怎么用（两行环境变量）

```bash
AL_EVAL_BATCH_MODE=auto \
AL_EVAL_BATCH_APPROVED=1 \
  <原来的训练命令>
```

| 变量 | 作用 |
|---|---|
| `AL_EVAL_BATCH_MODE` | `serial`（默认，逐条）/ `probe`（诊断对照）/ **`auto`**（批处理） |
| `AL_EVAL_BATCH_APPROVED` | `auto` 必须 `=1`（防误开的第二把钥匙，保持原样） |
| **`AL_EVAL_BATCH_MAX`** | **可选覆盖值**（1..256）。不设时批大小 = **`train.micro_batch_size`**（生产 = 24） |
| `AL_EVAL_BATCH_RESERVE_GIB` | 显存余量硬闸，默认 10 GiB |

**回退**：`AL_EVAL_BATCH_MODE=serial`（或 `AL_EVAL_BATCH_MAX=1`）。

## 2. 生效后日志长什么样

```
[open_loop][eval-batch] 评测批大小 = 24（来源 train.micro_batch_size；AL_EVAL_BATCH_MAX 可覆盖，reserve=10.0GiB）
[open_loop][eval-batch] auto 小结：批处理 1 组 / 单条回退 0 组（批大小 24）
[auto_learning] eval place_a2b_right/val n_ids=4: mse=0.0657 nmse=0.2078 (4轨迹/14chunk, 2.5s)
```

* `批处理 N 组`：真正走批处理的组数 —— **一个任务的整场评测通常是 1 组 = 1 次前向**；
* `单条回退 M 组`：因"成组不足 2 条 / 形状不一致 / 显存不足"而单条推理的组数（**per-group 回退，不整体退回**）。

## 3. 实测效果（2026-10-09，96G 单卡，GBS=24，权重 step500）

| 指标 | 串行 | 并行（批 24） | 提升 |
|---|---|---|---|
| 单次评测（4 轨迹 / 14 chunk） | **9.0 s** | **2.5 s** | **3.6×** |
| 2 轨迹小评测（4–20 chunk） | — | 1.1 – 3.5 s | — |
| **Bootstrap（50 任务 / 54 次评测）** | **~7 min** | **1 min 49 s** | **3.8×** |

**数值一致性**（严格同条件：同任务 + 同 split + 同轨迹数 + 同处 bootstrap / step500 权重）：

| 任务 | 串行 | 并行 | 差异 |
|---|---|---|---|
| adjust_bottle/val n=2 | 0.495909 | 0.495847 | **-0.01%** |
| click_bell/val n=4 | 0.009857 | 0.009823 | -0.34% |
| beat_block_hammer/val n=2 | 0.290955 | 0.291295 | +0.12% |
| click_bell/val n=2 | 0.009811 | 0.009873 | +0.63% |

⇒ 12 个可比条目：**中位差 0.038%、最大 0.63%**（模型自身噪声量级）。

> ⚠️ **跨 run 比评测必须锁条件**：拿"A run 训练后的 4 轨迹评测"去比"B run bootstrap 的 2 轨迹评测"，
> 会得到 12%–437% 的**混杂假象**（2026-10-09 我自己踩过）。比较前先锁：**任务 + split + 轨迹数 + 阶段(步数)**。

## 4. 链路：一次评测怎么走（batch=24）

```
evaluate_ids(ids, tag)                      open_loop_validation.py
 ├─ safe_eval_context（状态快照/翻转/恢复 + 评测期禁编译）
 ├─ _evaluate_ids：starts = 每回合从首帧按 chunk_size 跳步
 └─ _prediction_groups(ds, starts, ft, tag)
      ├─ take = min(批大小, 剩余起点数)                      ← 批大小 = 训练 micro
      ├─ 机械守卫：take<2 / 形状不一致 / 显存不足 ⇒ 该组单条
      ├─ auto ⇒ 一次 _infer_batch(inputs)  ← **一次前向算 take 条**
      │    内部 _infer_core：逐条 unsqueeze → cat 成 (B,…)；grid (B,N,3)；
      │    噪声逐条 randn((1,T,D)) 再 cat；visual grid 清空 + finally 恢复
      └─ yield [(idx, item, pred), …] → ft.unapply 取 GT → chunks → aggregate → GMean
```

**并行到哪一层**：

| 环节 | 并行？ |
|---|---|
| 样本内（10 步去噪、矩阵乘、注意力） | ✅ 一直有 |
| **样本间（一次前向算 N 条）** | ✅ **本功能提供** |
| **任务间（跨任务组批）** | ❌ **没有**，见 §5 |

## 5. 常见问题

**Q：会不会改变评测数值？**
会有一点，量级 = 模型自身噪声（同条件实测 **≤0.63%**；GMean 层实测差 ~1e-5）。原因：fused MoE 用
`tl.atomic_add`（token 打包 + 专家输出累加）⇒ **同代码跑两次都不同**（实测 max|Δ|=3.1e-2），
"逐位一致"从来就不存在。**离阈值远的任务无影响**；**贴线任务理论上可能翻转判定**
（我们数据里有 `move_can_pot 0.96×`、`place_object_scale 1.06×` 这类），风险自负。

**Q：能跨任务并行吗（50 个任务一次批完）？**
**不能**：`_infer_batch` 要求组内**所有形状一致**，而**指令 token 长度逐任务不同** ⇒ 会被拒绝
（日志 0 次形状报错，说明我们从未硬拼）。要做得先引入"指令 padding 到同长"的 collate
（模型有 `lang_masks`，技术上可行），并把调度器从"扫一个判一个"改成"批量扫 → 逐个判"——
结构性改动，收益约 **2–3×**（Bootstrap 1.8 min → 0.6–0.9 min）⇒ **当前判断：性价比不够，暂不做**。

**Q：严格 parity / "门" 还在吗？**
* **严格 parity（`atol=1e-5/rtol=1e-3`）已从判定链移除**：它对任何两次运行都不可达（模型自身 3.1e-2），
  旧设计下生产 131 个组只有 9 组通过 ⇒ 全程串行、且每组白付一次批量前向（**比纯串行慢 52%**）；
* `probe` 模式**保留为诊断**（串行+批量各跑一遍、写身份化证据，**结果仍只给串行**）；
* 离线验收工具 `tools/eval_batch_stochastic_acceptance.py` **保留**（噪声底判据 + GMean 指标层 + 门文件），
  用于版本回归/对拍，**不参与运行时判定**。

**Q：为什么评测批大小不再硬顶 2？**
旧设计 `min(2, AL_EVAL_BATCH_MAX)` 是为配合"逐位一致"判据（只能走最小步长），而那个判据不可达 ⇒
现在批大小直接跟随训练（24），并保留机械守卫与显存硬闸。

**Q：显存够吗？**
评测期显存峰值由批大小决定；生产实测（批 24）训练步峰值 83.1 GiB / 96 GiB，评测期明显更低。
硬闸：批处理后 `peak_free < reserve(默认 10 GiB)` ⇒ **抛错终止**（不静默、不带伤继续）。

## 6. 相关文件 / 命令

| 路径 | 作用 |
|---|---|
| `lingbotvla/utils/open_loop_validation.py` | `_prediction_groups`（三条路）、`_infer_core`、`_eval_batch_size` |
| `lingbotvla/auto_learning/stochastic_parity.py` | 纯判据（噪声底 / 门 / 指标层）——**仅离线工具使用** |
| `tools/eval_batch_stochastic_acceptance.py` | 离线验收（`--selftest` / `--execute`，默认 PLAN ONLY） |
| `tools/eval_batch_gpu_probe.py` | 四腿对照探针（诊断） |
| `tools/eval_batch_cpu_diag.py` | CPU 假模型诊断 + 8 类反例 |
| `tests/test_scan_accel_integrated.py` | 契约测试：auto 必须走批量、**该组不再跑串行**、尾部单条回退、批大小跟随训练 |
| `tests/test_eval_batch_stochastic_acceptance.py` | 判据 / 门 / 接线测试 |
| `docs/compile_tuning_guide.md` | 顺带做的编译开销治理（评测期禁编译 + `cache_size_limit=64`） |
| `docs/model_load_acceleration_guide.md` | 加载加速（并行分片预读 + BF16 副本） |

## 7. 历史（为什么变成现在这样）

1. 原设计：`auto` 要求"首组探测（串行+批量对照）通过"或"外挂签名证据（`gate.json`）开门"，
   判据 `atol=1e-5/rtol=1e-3`；
2. 实测：模型自身非确定 **3.1e-2** ⇒ 判据**恒不可达**；生产 131 组仅 9 组通过，
   且门不开时每组走 probe ⇒ **比纯串行慢 52%**；
3. 曾改为"现场噪声底判据 + 全局接受 + 周期复检"，但仍带判定与探测开销；
4. **用户定案（2026-10-09）：去掉门，评测批大小 = 训练批大小** ⇒ 即本页现行行为。
