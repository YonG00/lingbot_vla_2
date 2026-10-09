# run-all 单进程验收结果（2026-10-09，单卡 96G）

HEAD `d67e110`｜命令：`run-all --out-dir /data/outputs/scan_accel_acceptance --python <conda> --steps 3 --execute`（**未加** `--include-auto`）
子进程耗时 **167.4 s**、`peak_nvidia_smi_mib = 16193`｜输出：`summary.json` rc=**2**、`hardness_parity.json`、`run_all.log`（`eval_probe.json` **缺失**）

## ✅ Hardness 零步自检：**PASS**（真实计算，非空报告）

| 阶段 | 记录 | batch | 样本 | 每样本值唯一（跨阶段一致） | min peak_free |
|---|---:|---|---:|---|---:|
| main | 2 | {8, 1} | 9 | ✅ True | 79.16 GiB |
| replay_batch1 | 9 | {1} | 9 | ✅ True | 79.16 GiB |
| repeat | 2 | {8, 1} | 9 | ✅ True | 79.15 GiB |

- Scout 评测**确实用 4 条轨迹**（`eval click_bell/val n_ids=4`、`eval click_alarmclock/val n_ids=4` ✓）= 验收 YAML 生效 ✓
- `rng_binding = per_sample_id` ✓；显存余量 ~79 GiB ≫ 10 GiB 守卫 ✓
- 待补算：B8↔B1 逐样本最大绝对差与排序一致性（两读数集完全重合 ⇒ 逐位一致，报告落盘后再出精确数值）

## ❌ Eval Batch Probe：**无报告文件 + parity=False ⇒ BLOCKED**（不伪造）

- probe 分支**确实执行**了 8 次：`mode=probe batch=2 parity=False peak_free_gib=79.89`
  `serial_seconds≈0.99 / batch_seconds≈0.52`（≈1.9× 表面加速）`profitable=True safe=False` ✓
- **`parity=False`**（`outputs_close(atol=1e-5, rtol=1e-3)` 未通过）⇒ `safe = peak_free≥reserve and parity` = **False**
  ⇒ 探针**正确禁用批处理** ✓（fail-closed ✓）⇒ **Batch4 从未被尝试**（`batch=4` 记录数 = **0** ✓）
  ⇒ 按既定判定规则：**BLOCKED（Batch4 覆盖不成立）**，且**不得启用 auto** ✓
- `eval_probe.json` 缺失的原因：probe 报告写入块抛异常被 catch（日志留有告警），**根因是我的补丁里
  `action_diffs/append_json_record` 的 import 行未落地**（NameError ⇒ 只写日志不落盘）⇒ 需一行修复（**本轮未改，遵守"不擅自重跑"**）

## 结论

| 项 | 判定 |
|---|---|
| Hardness 零步覆盖（Batch8/1/8 + RNG + 显存） | **PASS** ✓ |
| Eval Batch 数值一致性 | **FAIL（parity=False）** ⇒ auto **不可启用** ✓ |
| Batch4 覆盖 | **BLOCKED**（Batch2 已不安全 ⇒ 不会倍增；且报告未落盘） |
| 后续 | 保留原始证据 ✓；**未重跑、未加预算** ✓；机器按要求关机 ✓ |
