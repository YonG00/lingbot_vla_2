# 离线 Replay 复验 V2 报告（无卡，2026-10-09）

## 结论

| 判定来源 | 结果 | 说明 |
|---|---|---|
| **历史原判**（GPU 验收工具 `result.json`） | **BLOCKED** | `diagnosis=train_unit_without_replay_pool`（工具字段口径契约差；**未修改、未覆盖**） |
| **新离线判 V2（原始 JSONL 单源）** | **PASS** ✅ | `diagnosis=verified`、`errors=[]`、`observed_replay_samples=105 == expected 105`、`expected_new_samples=255`、`105+255=360` 全对账、`unit_metric_step_relation="end"` |
| JSONL + TB 双源混合 | BLOCKED | **步号约定不同**造成映射歧义 ⇒ fail-closed（见下） |

**⇒ 历史 BLOCKED 保留；新离线复核在"原始 JSONL 单源"下判 PASS。** 两者不矛盾：历史 BLOCKED 源于验收工具的字段名契约差；V2 复核器读原始证据后完成按 Unit 完整对账。

## 证据与哈希（工具已记录）

- 原始 JSONL：`/data/outputs/gpu96_acceptance/ratio_real_micro24_gas1_target4_steps15/auto_learning_events.jsonl`
  sha256 `9ef3a5b5544b263fe6530102dae4b3c678aa6a97dd62f3a8793e22724ddd65fb`
- 原始 TB 事件：`…/runs/events.out.tfevents.1791505686.autodl-pro-7909d1ce5113.1519.0`
  sha256 `4da5124bb878c2c154a4a54523b41d4ebed5c36ff94b8fe993c1be38c576eb27`
- 本目录 `result.json`（历史原判）与运行目录的 `result.json` 均**未被修改** ✓

## 按 Unit 对账（V2 单源 PASS 的明细）

| Unit 起点 | 步数 | metric step（Unit 结束） | Replay 任务 | Replay 样本 |
|---|---|---|---|---|
| 0 | 3 | 3 | click_bell, click_alarmclock | 21 |
| 3 | 3 | 6 | 同 | 21 |
| 6 | 3 | 9 | 同 | 21 |
| 9 | 3 | 12 | 同 | 21 |
| 12 | 3 | 15 | 同 | 21 |
| 合计 | **15 步** | — | 2 个真实 PASS 任务 | **105** ✓ |

`expected_new_samples = 255`（15×17）⇒ 总样本 **360** = 15×24 = `sampling/total_samples` = `system/global_samples_seen` ✓；Replay 占比 105/360 = **29.2%** ≈ 70:30 ✓

## 发现的口径差异（需记录，非数据问题）

- **JSONL 侧**：`sampling/*`、`system/*` 指标写在**局部步号** 3/6/9/12/15（= Unit 结束，相对 step_offset）。
- **TensorBoard 侧**：同名标量写在**全局步号** 503/506/509/512/515。
- ⇒ 同时喂 JSONL + TB 时两套步号并集导致"唯一映射无候选" ⇒ 复核器 fail-closed 判 BLOCKED ✓（**正确行为**，不猜测）。
- 建议（待批准，属工具口径，不改生产代码）：双源模式下按**来源分别做映射**，或显式声明每源的步号基准；并保留一个"混源必须 BLOCKED"的负例测试。

## 本轮未做

未启动 GPU（无卡）✓；未关机 ✓；未清理任何文件（`/data/tmp` 417 MB、`ratio_real_micro24_gas1_target1` 均保持原状）✓；
未把自适应 Eval Batch 称为"已接入真实 6B 批量推理"（**策略与门控已合入，但 `_infer_batch` 未接入**）✓；未改写任何原始证据 ✓。
