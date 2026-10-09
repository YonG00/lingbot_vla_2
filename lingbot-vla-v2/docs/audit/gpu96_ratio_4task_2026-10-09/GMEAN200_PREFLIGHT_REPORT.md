# 50-task GMean 200× 无卡预检报告（2026-10-09）

命令（机器上，只读；退出码 0）：

```bash
cd /data/code/lingbot-vla-v2
$PY tools/gmean50_preflight.py \
  --config configs/auto_learning/experiment_50task_gmean200.yaml \
  --thresholds /data/eval_results/open_loop/ref50k/pass_thresholds_gmean200_warn.json \
  --baseline /data/train/task_splits_50/task_baseline.json \
  --history docs/audit/gpu96_ratio_4task_2026-10-09/auto_learning_events.jsonl \
  --output docs/audit/gpu96_ratio_4task_2026-10-09/preflight_gmean200_50task.json
```

## 结论：**READY**（rc=0）

| 项 | 实测 |
|---|---|
| 状态 / 退出码 | **READY** / **0** |
| 任务数 / 可用阈值 | **50 / 50** ✓ |
| 倍率 | **200.0**（标定来源 `calibration` 已校验 ✓） |
| **baseline 上限裁剪** | **`baseline_capped_tasks: []`** ⇒ **没有任何任务被 0.99×baseline 裁剪**，生效阈值即严格 200×reference ✓ |
| 被排除（null 阈值） | `excluded_null: []` ✓ |
| **高 CV 警告任务（6 个）** | `move_can_pot` · `place_can_basket` · `place_dual_shoes` · `place_object_basket` · `place_object_scale` · `stack_bowls_two` ⇒ 这些任务的阈值**可靠性较低**，选课/判定时需注意 |
| 阈值表 sha256 | `0789df128bcf698d…`（23,495 B） |
| baseline | `/data/train/task_splits_50/task_baseline.json`（37,094 B）；**config 指纹已由工具校验通过**（缺失/不匹配即 BLOCKED，本机为 READY） |
| 优先级口径 | 工具注明：**priority = candidate / 生效 PASS 阈值**；baseline-cap 会让它偏离 `candidate/(200×reference)` |

## 历史 Step500 扫描记录盘点（要求 5：仅盘点，不作缓存）

| 项 | 结果 |
|---|---|
| `/data/outputs/*/auto_learning_events.jsonl` | **8 份，全部为 NMSE 模式**（`al_2task`、`al_2task_probe`、`al_g10`、`al_probe_step500`、`al_smoke_g5`、`al_smoke_gbs4_4task_p1_botched_run1`、`al_smoke_gbs4_4task_tb10_botched_run1`、`al_smoke_nosave_5step_20261008_1833`） |
| 含 `scout_gmean_mse` 的记录 | **0 份** ⇒ **不存在可复用的 GMean Scout 记录** |
| 工具判定 | `reuse_allowed: false`，理由：*"event logs do not prove matching checkpoint/dtype/norm/eval IDs and full per-traj GMean; no automatic replay of Scout"* ✓ |
| 有 Bootstrap 事件的任务数 | **4 / 50**（`missing` 列出其余 46 个）⇒ 覆盖率 8%，**不可冒充已扫描** ✓ |

## 本机与机器测试

| 环境 | 项 | 结果 |
|---|---|---|
| 本机 | 新增专项（`test_gmean_ratio_priority` + `test_gmean50_preflight`） | **20 passed** |
| 本机 | README 指定组（+ `test_gmean_pass_pipeline` + `test_resume`） | **49 passed** |
| 本机 | `test_hardness_scan_timing` | **6 passed** |
| 本机 | **全量 CPU 回归** | **765 passed / 18 skipped / 0 failed** |
| 机器 | README 指定 5 文件组 | **55 passed**（与提供方一致 ✓） |
| 本机 / 机器 | CPU Gate | **✅(9/0/9)** 两边均通过 |

## 安全门控与未实现项（不得误报）

- **Adaptive Eval Batch**：`--mode auto --execute` 仍 **BLOCKED / rc=2**（`missing real inference batch adapter + numerical parity proof`）⇒ **真实 `_infer_batch` 未接线，不得宣称 GPU 多轨迹并行/扫描加速** ✓
- **历史扫描自动缓存命中** 未接入 ⇒ 旧 NMSE 记录不得当 GMean Scout（本次已用 `reuse_allowed:false` 证明 ✓）
- Hardness 预取流水线未改；新增的只是 **CPU 侧**计时统计（**非 CUDA kernel 时间**）✓
- 正式 `formal_50task_4pass.yaml` **零改动**（不在改动清单 ✓，仍 `pass_metric: nmse`）

## GPU 下一步命令（**仅计划，未执行**）

```bash
cd /data/code/lingbot-vla-v2 && PY=/data/miniconda3/envs/lingbotvla/bin/python
# 1) 先开单张 96G 卡；建议 micro24/GAS1（GBS24，峰值 86.3 GB，余量 11.5 GB 已验证）
$PY tools/gpu96_acceptance.py bench --micro 24 --gas 1 --execute        # 可选：复测吞吐
# 2) 实验性 50-task GMean200 短跑（**需用户单独批准**；不建议直接上正式长训练）
#    正式启动器 + 实验 YAML：
#    bash experiment/robotwin/al_50task_bf16.sh  （需把 AL_CFG 指向 experiment_50task_gmean200.yaml）
```
⚠️ 提醒：若 Bootstrap 初始 PASS ≥ 4（`target_total_passed_tasks: 4`），训练可能**零步结束** ⇒ 不能擅自改目标。
