# Eval Batch auto 启用指南（**2026-10-09 改版：门已废除，批大小 = 训练批大小**）

> ⚠️ **本页上方"随机性门"章节已作废**（保留作历史记录，见 §0）。
> 现行行为：`AL_EVAL_BATCH_MODE=auto` 时，评测**直接按训练的前向批大小成组批处理**，
> 不做任何数值判定、不需要 `gate.json`。

## 0. 改版原因（一句话）

旧设计要求"每组先探测（串行+批量对照）通过、或用签名证据开门"才放行；而判据
`atol=1e-5/rtol=1e-3` 对**任何**两次运行都不可达（模型自身非确定实测 **3.1e-2**）
⇒ 生产里 131 个组只有 9 组通过 ⇒ 全程退回串行，且每组白付一次批量前向
（`serial 1.05 s + batch 0.55 s` vs 纯串行 1.05 s）⇒ **比串行慢约 52%**。
现已按用户定案改为"**训练给多少 batch，评测就用多少 batch**"。

## 0b. 现行行为（要点）

| 项 | 现行 |
|---|---|
| 启用 | `AL_EVAL_BATCH_MODE=auto` + `AL_EVAL_BATCH_APPROVED=1`（后者保持原样，防误开） |
| **批大小** | **= `train.micro_batch_size`**（生产 = 24；`AL_EVAL_BATCH_MAX` 变为**覆盖值**，非上限） |
| 数值判定 | **无**（门 / 严格 parity 对照 / 探测放行 全部退出判定链） |
| probe 模式 | 保留为**诊断**（串行+批量对照、写证据，结果仍只给串行） |
| 机械守卫 | 成组不足 2 条 / 形状不一致 / 显存余量不足 ⇒ **该组单条**（不整体退回） |
| 显存硬闸 | 批处理后 `peak_free < AL_EVAL_BATCH_RESERVE_GIB`(默认 10) ⇒ **抛错终止**（不静默） |
| 回退 | `AL_EVAL_BATCH_MODE=serial`（默认值）；或 `AL_EVAL_BATCH_MAX=1` |

**代价（如实）**：评测数值与串行有"模型自身噪声"量级的差异（实测 GMean 差 ~1e-5，相对 0.07%，
而阈值余量 61%）⇒ 离阈值远的任务无影响，**贴线的任务理论上可能翻转判定**；`batch=24` 的差异量级
尚未单独实测。要复现历史串行口径：`AL_EVAL_BATCH_MODE=serial`。

---
# Eval Batch auto 启用指南（随机性门 · stochastic gate）

> 面向：想在 50-task GMean200 自动学习里**开启评测批处理加速**的人。
> 结论先行：**不需要**再跑一遍 GPU 验收 —— 验收已通过，证据在
> `/data/outputs/eval_batch_stoch_20261009c/`，直接带三个环境变量即可（换权重/换形状要重跑，见 §6）。

---

## 1. 为什么需要这个门（30 秒背景）

评测批处理（B1 逐条 vs B2 一次前向多条）原本靠**严格数值一致性**把关：

```python
parity = outputs_close(serial, batched, atol=1e-5, rtol=1e-3)
safe   = peak_free >= reserve and parity      # 不过 ⇒ disabled=True，后面永远串行
```

但实测（2026-10-09，96G 卡）：**同代码、同噪声、同入参跑两次**，输出 max|Δ| = **3.1e-2**
（mean 2.3e-3，P95 = 7.8e-3 = 1 个 bf16 ULP）；而 B1↔B2 只有 2.1e-2。
根因在 `lingbotvla/ops/robby_moe.py` 两处 `tl.atomic_add`（token 打包槽位 + 专家输出累加）
覆盖 36 层 fused MoE ⇒ 模型自带 ~1e-2 的 run-to-run 抖动。

⇒ **`atol=1e-5/rtol=1e-3` 对任何两次运行都不可达**（连 serial vs serial 都过不了）
⇒ 开 `auto` 也只会全程串行，还白付一次 batch 前向。

**本门做的事情**：在"模型自身非确定"这个已证实的前提下，新增**另一条有证据的通过路径**。
**严格 parity 原样保留、未放宽**，只是 `safe` 判定变成 `parity or gate_ok`。

---

## 2. 快速开始（三条命令）

```bash
REPO=/data/code/lingbot-vla-v2
PY=/data/miniconda3/envs/lingbotvla/bin/python
CKPT=/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt

# ① 上 GPU 前先跑 CPU 整链自测（1 秒；钉住与生产方法的接口契约）
cd $REPO && $PY tools/eval_batch_stochastic_acceptance.py --selftest

# ② 跑一次随机性验收（≈5 分钟：加载 ~2 min + 数值层秒级 + 指标层 ~1–2 min）
$PY tools/eval_batch_stochastic_acceptance.py \
  --ckpt $CKPT --task click_bell \
  --out-dir /data/outputs/eval_batch_stoch_YYYYMMDD \
  --reps 5 --metric-episodes 2 --execute
#   产不出 PASS ⇒ 到此为止，auto 不能开

# ③ 训练启动时带三个环境变量（缺一即全程串行）
AL_EVAL_BATCH_MODE=auto \
AL_EVAL_BATCH_APPROVED=1 \
AL_EVAL_BATCH_STOCHASTIC_APPROVED=/data/outputs/eval_batch_stoch_YYYYMMDD/gate.json \
  bash experiment/robotwin/al_50task_bf16.sh
```

**本次已通过的证据**（可直接复用，无需重跑）：

| 项 | 值 |
|---|---|
| 证据目录 | `/data/outputs/eval_batch_stoch_20261009c/` |
| gate 文件 | `gate.json`（`verdict_status=PASS`，签名 `67f925c2efbcdacb…`） |
| 绑定条件 | ckpt=`…/global_step_500/hf_ckpt`、task=`click_bell`、batch=2、dtype=bf16、观测形状/grid 签名 |
| 指标层 | B1/B2 各 5 次 GMean ∈ [0.01454, 0.01463]，阈值 0.02351 ⇒ **余量 61%**，判定零翻转 |
| 数值层 | 跨组偏差 7.28% ≤ 同代码随机二分上限 10.15% ⇒ 无系统性偏差 |
| 吞吐 | **1.80×**（B1 0.4844 → B2 0.2695 s/样本） |

---

## 3. 验收工具参数

```bash
$PY tools/eval_batch_stochastic_acceptance.py [选项]
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--ckpt` | step500 hf_ckpt | 待验收权重（**只读**） |
| `--task` | `click_bell` | 训练/评测任务名（决定阈值取自哪一行） |
| `--out-dir` | 必填 | 新目录；**已存在且非空则拒绝** |
| `--reps` | `5` | 每模式重复次数；**< 5 直接 BLOCKED** |
| `--metric-episodes` | 全部 | 指标层只跑前 N 个**整回合**（验收提速；建议 2 = scout 真实轨迹数） |
| `--dataset-indices` | `0 50` | 数值层固定用的 chunk（同一样本、同一份噪声） |
| `--thresholds` | GMean200 表 | 只读；要求 `metric=mse` 且 `stat=geomean`，否则 BLOCKED |
| `--no-metric-level` | — | 跳过指标层。⚠️ 跳过 ⇒ 总判定**必然 BLOCKED**（不允许只看 max_abs_diff 下结论） |
| `--execute` | 关 | **不加就是 PLAN ONLY**（只做路径/参数/规则体检，不碰 GPU、不建目录） |
| `--selftest` | — | CPU 端整链自测（stub 走完数值层 + 指标层 + 总判定），**上 GPU 前必跑** |

---

## 4. 输出物

| 文件 | 内容 |
|---|---|
| `summary.json` | 全量结论：`numeric`（组内/跨组 max·mean·P95·P99、偏差检验）、`metric`（两组 GMean、PASS 列表、余量）、`throughput`、`verdict`、`gate`（签名与 payload） |
| `gate.json` | **给运行时用的门证据**：`kind` / `signature` / `payload` / `verdict_status` / `worthy` |
| `numeric_runs.npz` | 原始动作数组（`batch1`/`batch2` 各 reps 份）⇒ **判据改进后可离线重算，无需再上卡** |

```bash
# 离线重算示例（改过判据后用它，别重跑 GPU）
$PY - <<'EOF'
import json, sys, numpy as np
sys.path.insert(0, "/data/code/lingbot-vla-v2")
from lingbotvla.auto_learning import stochastic_parity as sp
D = "/data/outputs/eval_batch_stoch_20261009c"
d = np.load(f"{D}/numeric_runs.npz"); s = json.load(open(f"{D}/summary.json"))
b1 = [d["batch1"][i] for i in range(d["batch1"].shape[0])]
b2 = [d["batch2"][i] for i in range(d["batch2"].shape[0])]
v = sp.decide_numeric(b1_runs=b1, b2_runs=b2, reps=len(b1))
print(v["status"], v["reasons"], sp.overall_verdict(
    numeric=v, metric=s["metric"], throughput=s["throughput"])["status"])
EOF
```

---

## 5. 判定规则（全部通过才 PASS，否则 BLOCKED）

### 数值层 `decide_numeric()`

| 规则 | 内容 |
|---|---|
| R1 | `reps >= 5`（样本不足 ⇒ 不做任何统计判定） |
| R2 | 全部有限值 |
| R3 | 跨组 P99 ≤ `1.5 ×` 组内 P99 |
| R4 | 跨组 max ≤ `1.5 ×` 组内 max |
| R5 | 组内(B2) P99 ≤ `2.0 ×` 组内(B1) P99（批量不得放大抖动） |
| R6 | **跨组偏差统计量 ≤ 所有"同代码随机二分"的参考值**（见下） |

> **R6 为什么这么设计**：逐元素差值是**空间相关**的（相邻动作维/时刻共享同一 bf16 舍入结构），
> 用 i.i.d. 正态零分布会**严重低估方差** ⇒ 纯噪声也判成偏差（实测 `obs=7.28%` vs 理论 4%，
> `p_frac=0.000` 假阳性）。改用**同一批 run 的随机二分**作参考分布：天然保留相关结构、
> 无分布假设，且小 N 下保守。旧的 MC p 值仍保留在 `bias_pvalue` 字段作**对照**。

### 指标层 `decide_metric()`

| 规则 | 内容 |
|---|---|
| M1 | 每次都拿到有限 GMean（否则 BLOCKED） |
| M2 | 组内 PASS/FAIL **不翻转** |
| M3 | 两种模式 PASS/FAIL **一致** |
| M4 | 组内 GMean 极差 **<** 到阈值的距离（贴线通过 ⇒ BLOCKED） |

### 总判定 `overall_verdict()`

**数值层 + 指标层都 PASS 才 PASS**；指标层没跑 ⇒ BLOCKED。
`throughput` 只决定 `worth_proposing_auto`（速度 ≠ 安全）。

---

## 6. 运行时行为（fail-closed）

```python
safe = peak_free >= reserve and (parity or gate.ok)   # parity 仍照常计算并写入日志
```

| 情况 | 结果 |
|---|---|
| 没设 `AL_EVAL_BATCH_STOCHASTIC_APPROVED` | `gate_not_configured` ⇒ 退化为**只看 parity**（即现状，串行） |
| 没设 `AL_EVAL_BATCH_APPROVED=1` | 直接 `RuntimeError`（原有的独立批准门，未改） |
| 模式不是 `auto`（`serial`/`probe`） | 门不参与 |
| gate 文件缺失 / 非法 JSON / kind 不符 | 不开门 |
| **签名不符**（换了 ckpt、批大小、dtype、观测形状或 grid） | 不开门（这是"换权重必须重跑验收"的落点） |
| 换了**任务名**或**指令 token 长度** | ✅ **仍然开门**（2026-10-09 修正：这两项只记录、不绑定 —— 见下） |
| gate 里 `verdict_status != PASS` | 不开门 |
| 全部匹配 | `gate=on`，该形状后续组走批量 |

日志会打印：`mode=auto batch=N parity=… safe=… gate=on/off gate_reasons=[…]`。

**回滚**：去掉那三个环境变量即可，代码路径与默认行为不变（`AL_EVAL_BATCH_MODE` 默认 `serial`）。

---

## 6b. 绑定范围（2026-10-09 生产实测后修正）

**签名只覆盖"数值相关"条件**：`ckpt` / `batch_size` / `dtype` / `images`·`img_masks`·`state` 形状 / `grid`。
**刻意不绑定**：

| 项 | 为什么 |
|---|---|
| **任务名** | 首次正式跑（131 个批处理组）只有 **9 组 `gate=on`** —— 因为 Bootstrap 会扫 50 个任务，绑单一任务名 ⇒ 其余 49 个全 `gate_signature_mismatch`。而任务名**不影响任何张量数值**。 |
| **`lang_tokens`/`lang_masks` 长度** | 指令文本逐任务不同 ⇒ token 长度天然不同；但两种模式喂的是**同一份输入**，"批处理是否引入额外误差"这一比较对所有任务同样成立。 |
| 形状的前导 batch 维 | 运行时 `inputs` 是原始 item（`[3,256,1536]`），验收工具若用 collate 后的形态（`[1,3,256,1536]`）会假性不符 ⇒ 比较时**剥离前导 1 维**。 |

⚠️ **未生效的代价**：门不开时每组走 probe（串行 + 批量各一遍）⇒ 每组 1.60 s vs 纯串行 1.05 s，
**比串行慢约 52%**。修正后（本版）签名在 50 个任务上通用；`task` 与指令长度仍写入
`recorded_task` / `recorded_notes` 供追溯。

**当前证据签名**：`5878276ba5e3fb3abecf4d985dbf4caa5ff30a5a4405cc22151eefa2370f8643`
（新绑定；旧版 `67f925c2…` 因绑定旧语义**不再匹配**，需用新版工具重生成）。

## 7. 常见问题

**Q：能只放宽 `atol/rtol` 让它过吗？**
不行，也不该 —— 那会把"检测异常"的能力一起削掉，且判据仍然过不了（模型抖动 1e-2 量级）。
本门的做法是**另加一条有证据的路径**，严格 parity 原样保留。

**Q：为什么第一次前向必须剔除？**
模型加载后第一次前向含 cudnn autotune / 惰性初始化，实测与后续所有腿的 max|Δ| 高达 **0.177**
（正常同代码波动只有 0.031，差一个量级）。工具已内置**热身一次并丢弃**，`throughput.warmup_seconds` 里能看到。

**Q：`--metric-episodes` 为什么必须整回合保留？**
逐轨迹 MSE 是**按回合聚合**的（`aggregate_chunks`），把回合切碎会改变口径。
工具用 `restrict_starts_to_episodes()` 保证整回合。

**Q：验收要多久？**
`--reps 5 --metric-episodes 2` ≈ **5 分钟**（模型加载 ~2 min 是主要开销）。
不加 `--execute` 是 PLAN ONLY，几秒。

**Q：能不能挂在训练里顺手收数据？**
可以，但那是 `AL_EVAL_BATCH_MODE=probe`（**只用串行结果**、不改评测数值，只记录 parity/耗时/显存）。
注意：**probe 收不到本门的结论** —— 门需要"同一批 chunk 重复 ≥5 次"，只能由本工具产出。

---

## 8. 已知边界（如实）

1. 门的证据**绑定当次运行条件**：换权重、换形状、改批大小都要**重跑验收**（≈5 分钟）；
2. 本次指标层只覆盖 **2 个回合**（scout 口径），未覆盖全量 val；
3. **未做对照实验**证明"MoE 原子加是唯一非确定来源"（只是机制上充分 + 实测一致）；
4. 门只保证"批处理波动 ≤ 模型自身波动 **且不改变 GMean200 判定**"，**不代表闭环成功率**。

---

## 9. 相关文件

| 路径 | 作用 |
|---|---|
| `lingbotvla/auto_learning/stochastic_parity.py` | 纯统计判据 + 门（`gate_payload/signature/write_gate/load_gate`） |
| `lingbotvla/utils/open_loop_validation.py` | `_stochastic_gate_ok()` + `safe = parity or gate.ok` 接入点 |
| `tools/eval_batch_stochastic_acceptance.py` | 验收工具（PLAN ONLY 默认 / `--selftest` / `--execute`） |
| `tests/test_eval_batch_stochastic_acceptance.py` | 判据 + 门 + 契约测试（含 8 类反例、相关噪声回归） |
| `docs/scan_accel_gmean200_zh.md` | 扫描加速总览（本门是它 `auto` 模式的数值前置条件） |
| `.workbuddy/memory/knowledge/eval.md` | 实测数据与历史结论 |
