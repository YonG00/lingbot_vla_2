# 扫描加速 GPU 验收 —— 无卡就绪性评估与新增验收入口（2026-10-09）

基线 HEAD：`7982265`（上一轮扫描加速集成补丁）

## 一、结论速览

| 要求 | 无卡核实结果 |
|---|---|
| 1. Eval Batch Probe 是否**完整自动化** | **部分**：真实 Batch2→4→…→8 自动倍增、`outputs_close(atol=1e-5, rtol=1e-3)` 数值对拍、`peak_free_gib` 显存守卫、`serial/batch_seconds` 吞吐、不安全即 `disabled`（fail-closed）**都已实现**；**但结果只写日志**（`[open_loop][eval-batch] …`）⇒ **无机器可读报告、无 PASS/FAIL 判定入口** ✗ |
| 2. Hardness Batch8↔Auto 是否有自动验收入口 | **部分**：`HardnessAutoBatch.observe()` 显存守卫 ✓、auto 需 `AL_HARDNESS_BATCH_APPROVED=1` + per-sample-ID 不变 RNG 打分器 + 单卡 ✓；**但无逐样本 Loss 对拍 / RNG 复现 / 排序一致性的验收入口** ✗ |
| 3. 是否补最小补丁 + CPU 回归 | **已补**（见下），**默认行为一字未改** ✓ |
| 4. 默认值 | Eval **serial** / Hardness **fixed(Batch8)** / Scout 缓存 **OFF** ⇒ 本补丁**未改变** ✓（且 `*_APPROVED` 门控原样保留 ✓） |
| 5. 未重复测试 | 未触碰 Replay、HF 导出、Micro24 吞吐 ✓ |
| 6. GMean200 零步风险 | `gmean50_preflight.py` 现在输出 **`zero_step_risk`** 字段（target=4 不变，仅提示风险）✓ |

## 二、最小补丁（三处生产代码 + 一个验收入口）

### 2.1 生产代码：仅新增**环境变量门控**的报告输出（默认完全不启用）

| 文件 | 改动 |
|---|---|
| `lingbotvla/auto_learning/scan_accel.py` | 新增纯函数 `action_diffs(ref, cand)`（逐元素最大/平均绝对差）与 `append_json_record(path, record)`（原子并入 JSON 报告） |
| `lingbotvla/utils/open_loop_validation.py` | probe 分支在原有日志之外，若设 `AL_EVAL_BATCH_PROBE_OUT` 则写入每个 probe 组的记录：`batch/parity/atol/rtol/peak_free_gib/reserve_gib/serial_seconds/batch_seconds/speedup/safe/faster/action_keys/**per_traj[{dataset_index,max_abs_diff,mean_abs_diff}]**` ⇒ 满足"**逐轨迹 action 数值比较**" ✓ |
| `lingbotvla/auto_learning/real/backend.py` | ① `AL_HARDNESS_FIXED_BATCH`（仅验收用，**默认仍 8**）② `AL_HARDNESS_REPORT_OUT` ⇒ 每批记录 `task/batch/sample_ids/losses{}/free_before_gib/peak_free_gib`（**逐样本 Loss** ✓） |

### 2.2 验收入口：`tools/scan_accel_gpu_acceptance.py`

- `plan`（默认，**PLAN ONLY**，rc=0、不碰 GPU、不建目录、不设任何 `*_APPROVED`）⇒ 打印 A/B/C 三组命令、预计耗时与安全停止条件；
- `verify-probe <json>` ⇒ **PASS/BLOCKED/FAIL**：
  - **FAIL**：任一 probe 组 `parity=False` 或 `safe=False`、`peak_free < 10 GiB`、报告 `mode != probe`；
  - **BLOCKED**：未真实达到 **Batch2 或 Batch4**（不得声称多轨迹可用 ✓）；
  - **PASS**：全部组 parity+safe 达标且 Batch2/4 均实测；若 `speedup ≤ 1` 则明确标注 **`no_speedup_observed_do_not_enable_auto`** ✓（不夸大 ✓）
- `verify-hardness <b8.json> <b1.json> [--repeat <b8r.json>]` ⇒ **PASS/FAIL**：
  - 逐样本 Loss 一致（`atol=1e-6, rtol=1e-4`）、**候选排序完全一致**、重跑**逐位一致**（随机数可复现）、**每份报告**的 `peak_free ≥ 10 GiB`；
  - 任一项不满足 ⇒ FAIL 并列出具体问题（`loss_mismatch:<sid>` / `ordering_changed` / `rng_not_reproducible` / `peak_free_below_reserve` / `sample_id_mismatch`）✓

**CPU 回归**：新增 `tests/test_scan_accel_gpu_acceptance.py` **17 passed**（PLAN ONLY 无副作用、probe 判定 7 例矩阵、hardness 判定矩阵、排序/RNG 检测、报告 round-trip）；
相关专项 **97 passed**；**全量 799 passed / 18 skipped / 0 failed**；CPU Gate **✅(9/0/9)** ✓

## 三、开卡后可直接执行的命令（预计 **15–25 min**，单卡 96G）

```bash
cd /data/code/lingbot-vla-v2 && PY=/data/miniconda3/envs/lingbotvla/bin/python
OUT=/data/outputs/scan_accel_acceptance          # 全新目录

# A. Eval Batch Probe（Batch1/2/4 对拍；probe 输出恒为串行 ⇒ PASS 不受影响）
AL_EVAL_BATCH_MODE=serial $PY tools/gpu96_acceptance.py ratio-gpu --micro 24 --gas 1 \
  --target-total-passed-tasks 4 --max-named-tasks 4 --steps 15 --execute           # A1 基线（可选）
AL_EVAL_BATCH_MODE=probe AL_EVAL_BATCH_PROBE_OUT=$OUT/eval_probe.json \
  $PY tools/gpu96_acceptance.py ratio-gpu --micro 24 --gas 1 \
  --target-total-passed-tasks 4 --max-named-tasks 4 --steps 15 --execute           # A2 probe
$PY tools/scan_accel_gpu_acceptance.py verify-probe $OUT/eval_probe.json           # A3 判定

# B. Hardness 逐样本对拍（Batch8 vs Batch1 + 重跑一致性）
AL_HARDNESS_BATCH_MODE=fixed AL_HARDNESS_FIXED_BATCH=8 AL_HARDNESS_REPORT_OUT=$OUT/hardness_b8.json \
  $PY tools/gpu96_acceptance.py ratio-gpu ... --execute                            # B1
AL_HARDNESS_BATCH_MODE=fixed AL_HARDNESS_FIXED_BATCH=1 AL_HARDNESS_REPORT_OUT=$OUT/hardness_b1.json \
  $PY tools/gpu96_acceptance.py ratio-gpu ... --execute                            # B2
（B3 再跑一次 B1 ⇒ $OUT/hardness_b8_repeat.json）
$PY tools/scan_accel_gpu_acceptance.py verify-hardness \
  $OUT/hardness_b8.json $OUT/hardness_b1.json --repeat $OUT/hardness_b8_repeat.json # B4 判定

# C. 50-task GMean200 启动预检（无卡，随时可跑）
$PY tools/gmean50_preflight.py --config configs/auto_learning/experiment_50task_gmean200.yaml \
  --thresholds /data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json \
  --baseline /data/train/task_splits_50/task_baseline.json --expected-tasks 50
# ⚠️ target=4 不变；若 Bootstrap 初始 PASS ≥ 4 ⇒ 零训练步结束（输出中的 zero_step_risk）
```

## 四、安全停止条件（任一命中即停该专项并保留现场）

1. `parity=False` / `safe=False`（数值或显存守卫失败）⇒ 停，**不得启用 auto**
2. `peak_free < 10 GiB` ⇒ 停
3. CUDA OOM / 非有限 loss / 非有限 action ⇒ 停
4. 报告缺失或 `sample_id` 不一致 ⇒ 停并核证据口径
5. Hardness **排序变化**或重跑不一致 ⇒ 停
6. **禁止**：设置 `AL_EVAL_BATCH_APPROVED` / `AL_HARDNESS_BATCH_APPROVED`、启用 `auto`、声称已获加速

## 五、本轮未做

未开 GPU、未启动正式训练、未自动关机、**未清理任何历史权重/测试产物** ✓；
未重复测试 Replay / HF 导出 / Micro24 吞吐 ✓；未修改正式 NMSE YAML、GMean200 阈值、70:30、DCP/HF 策略 ✓
