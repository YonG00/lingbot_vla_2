# 无卡修复 + parity 静态审查（2026-10-09）

HEAD 基线 `d67e110`（GPU 首轮）；本轮**不重跑 run-all**、不启用 auto ✓

## 1. 修复：缺失 import（`eval_probe.json` 未落盘的直接原因）
`lingbotvla/utils/open_loop_validation.py` probe 分支的多行 import 只引入了
`identical_tensor_shapes, normalized_action_predictions`，**缺** `action_diffs, append_json_record`
⇒ 报告写入块抛 `NameError` ⇒ 被 catch 只写日志 ⇒ 文件缺失。**已补齐** ✓ + CPU 回归测试锁定该 import 段 ✓

## 2. 修复：证据缺失必须结构化 BLOCKED（不再掩盖）
- probe 写入失败时**追加一条 `eval_batch_probe_error` 记录**（含异常类型/文本）✓
- `verdict_probe` 改为**只按 `eval_batch_probe` 记录判定**，并把 error 记录转成
  `blocked: ["probe_report_write_failed:…"]` ⇒ 状态 **BLOCKED**（此前会被误判成 `parity_failed` 的 FAIL ✗）✓
- `run-all` 在报告文件缺失时本就给 `eval_probe: BLOCKED`（rc=2）✓，现在两层一致 ✓

## 3. `_infer_one` vs `_infer_batch` 静态审查（parity=False 的首要嫌疑）

| 维度 | `_infer_one`（串行） | `_infer_batch`（批量） | 一致性 |
|---|---|---|---|
| 精度 dtype | `resolve_inference_dtype(...)` | 同一函数同参数 ✓ | ✅ 一致 |
| 随机噪声 | 每条 item 从同一 `_noise_generator` 抽 `randn((1, n_action_steps, max_action_dim))` | **按序**逐条 `randn(shape)` 后 `torch.cat` ✓（注释明确"两次 randn 不等于一次 randn(2)"） | ✅ 顺序与形状一致 |
| 输入构造 | 单条 item 原样（含 `image_grid_thw` 原样传入） | `torch.stack` 各字段 ✓；**`image_grid_thw` 用 `torch.stack([...])` ⇒ `(B,3)`** | ⚠️ **高度可疑** |
| 视觉网格缓存 | 不清理 | `_visual_grid_cache_clear/restore` 包裹 ✓ | ✅ 有意一致化 |
| 输出拆分 | 直接 `ft.unapply` | （尾部按 `actions[i]` 拆分后同样 `unapply`） | ✅ 待精确复核 |
| batch 规模 | 1 | ≥2 | 语义应等价 |

**首要嫌疑**：`image_grid_thw` 的**拼接语义**。若模型内部按 `image_grid_thw` 切分/广播 patch 序列，
串行传入的是**单条观测**的网格（`(n,3)`，n=该条观测的视图数），批量传入 `(B,3)` 会与 patch 序列长度
不匹配（应 `torch.cat(dim=0)` 或按视图维拼接）⇒ 位置/网格编码错位 ⇒ 动作整体偏移 ⇒
`parity=False` 且**误差量级大于 1e-3** 的典型表现 ✓（与实测 8/8 组全 parity=False 相符）

## 4. 已取得的 Batch2 真实证据（复用归档日志，不伪造缺失项）

`/data/outputs/gpu96_acceptance/ratio_real_micro24_gas1_target2_steps3/train.log` 中 8 条 probe 记录（4 种模式 ×2）：

| mode | batch | parity | safe | profitable | peak_free | serial_s | batch_s |
|---|---:|---|---|---|---:|---:|---:|
| probe | 2 | **False** | False | True | 79.89 | 0.986 | 0.521 |
| probe | 2 | **False** | False | True | 79.89 | 0.990 | 0.530 |
| probe | 2 | **False** | False | True | 79.89 | 1.059 | 0.520 |
| probe | 2 | **False** | False | False | 79.89 | 1.068 | 1.241 |

- **Batch4 记录数 = 0**（`safe=False` ⇒ 正确禁用批处理，从未尝试）✓
- ⚠️ **逐轨迹 `max_abs_diff` 数值不可恢复**：报告文件从未落盘（NameError）⇒ 归档日志只含 boolean parity
  ⇒ **不伪造具体误差数值**，只保留"超容差"这一事实 ✓
- 表面吞吐：3/4 组约 1.9×，1/4 组更慢 ⇒ 即使数值正确，加速也不稳定 ✓

## 5. 最小 GPU 诊断方案（待批准，~5–8 min，单卡）

1. 先修好的 `run-all --execute`（一次加载）⇒ 拿到**逐轨迹 `max_abs_diff`/`mean_abs_diff`** ✓
2. 若确认误差集中在位置/网格相关维度：仅改 `_infer_batch` 的 `image_grid_thw` 拼接方式
   （`stack` → `cat(dim=0)` 或按视图维对齐），**再跑同一命令**做 A/B ⇒ 只看 parity 是否转 True ✓
3. 不改正式配置、不启用 auto；Hardness 侧无需重验（已 PASS）✓

## 6. 50-task GMean200 启动准备与零步风险（未改 target）

- 预检已 **READY**（50/50 可用、无 baseline 裁剪、6 个高 CV 任务）✓；`zero_step_risk` 字段已写入预检输出 ✓
- **target_total_passed_tasks=4 保持不变**：若 Bootstrap 已解析出 ≥4 个 PASS，调度器
  `all_tasks_resolved` ⇒ **零训练步结束**（不是 bug，但等于白开卡）⇒ 建议开卡前先看首轮评测 PASS 名单 ✓
