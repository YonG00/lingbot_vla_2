# 阶段 4（GMean BF16/FP32 对拍）与阶段 5（BF16 HF 导出）——无卡 PLAN ONLY 预检

基线 HEAD：`24a03a60eaa5b36742d07d6201234c582800ccbb`｜机器：RTX PRO 6000 96G **当前无卡**｜磁盘：**203 GB 可用**

## 结论：**未发现阻塞性缺陷**，两份脚本可直接 `--execute`（等用户开卡）

## 一、退出码与副作用（PLAN ONLY）

| 命令 | 退出码 | 副作用 |
|---|---|---|
| `gpu96_acceptance.py gmean` | **0** ✓ | 无（未创建任何目录 ✓） |
| `gpu96_acceptance.py hf` | **0** ✓ | 无 ✓ |
| `gpu96_acceptance.py hf-plan` | **0** ✓ | 无 ✓（打印需遵守的训练参数清单）|
| `adaptive_eval_batch_acceptance.py --mode auto --execute` | **2** ✓ | 无 ✓（`BLOCKED: missing real inference batch adapter + numerical parity proof`）|

## 二、逐项核对

| 检查项 | 阶段 4 `gmean` | 阶段 5 `hf` |
|---|---|---|
| **模型路径** | `/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt` ✓（24 G） | 同 ✓ |
| **数据 / 配置** | dataset `/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30` ✓；norm `assets/norm_stats/robotwin_competition_clean.json` ✓ | `--train-config /data/train/configs/robotwin_official_paths.yaml` ✓；AL 源 YAML 只读派生隔离配置 ✓ |
| **轨迹 ID** | 3 任务 × **10 条唯一** = 30（`click_bell` / `click_alarmclock` / `adjust_bottle` 的 `val_ids.json` 均在 ✓）；工具校验「每任务 ≥10、任务内无重复、**跨任务不共享 ID**」⇒ 不满足即直接报错 ✓ | 由工具写入 4 任务隔离配置（`click_bell/click_alarmclock/adjust_bottle/press_stapler`）✓；`manifest.json` / `task_baseline.json` ✓ |
| **精度设置** | BF16 腿加 `--use_bf16` ⇒ `self.vla.to(torch.bfloat16)`；FP32 腿不加 ⇒ **`self.vla.model.float()`（整模型转 fp32）** ✓ 真 fp32；`assert not (use_bf16 and use_fp32)` ✓；`--fixed_seed_per_traj --seed_base 1234 --noise_repeats 1` ✓ | `--train.hf_export_dtype bf16` ✓ |
| **输出目录** | `/data/outputs/gpu96_acceptance/gmean_pair` ✓ **尚不存在**（`fresh_dir` 拒绝覆盖 ✓） | `/data/outputs/gpu96_acceptance/hf_bf16_micro10_gas1` ✓ **尚不存在** ✓ |
| **磁盘预算** | MB 级（`--no_plot`）✓ | BF16 模型 **~12 GiB** + 预留 **~24 GiB**（防 Scheduler 提前结束写 final DCP）≈ **36 GiB** ✓ |
| **判定/守卫** | 每轨迹 MSE 解析要求 **ID 集合完全一致**（缺/多/重复/非有限/负值 ⇒ 报错 ✓）；**不传 `--thresholds`** ⇒ 不启用实验性 GMean 阈值表 ✓；参考模型不重跑（`reference_model_not_run: true`）✓ | 需**恰好 1 个里程碑 + 分片非空 + `5 < size_gib < 19` + 零 DCP 泄漏** ✓；`hf_direct_export_acceptance.py` 在张量不一致时**退出非零** ✓ |
| **日志解析契约** | `MSE_RX = r'MSE for trajectory\s+(\d+):\s*([^,\s]+)'` ↔ `scripts/open_loop_eval.py:374` 的 `MSE for trajectory {id}: {mse}, MAE: ...` **完全匹配** ✓ | — |
| **自适应 Eval Batch** | 安全门控保持 ✓（`--mode auto --execute` ⇒ BLOCKED / rc=2），**未声称支持多轨迹批量推理** ✓ | — |

## 三、执行命令（待用户开卡批准）

```bash
cd /data/code/lingbot-vla-v2 && PY=/data/miniconda3/envs/lingbotvla/bin/python
# 阶段 4：BF16/FP32 固定轨迹成对对拍（PASS ⇒ 0；评测失败 ⇒ 非 0）
$PY tools/gpu96_acceptance.py gmean --execute
#   产物：gmean_pair/{bf16,fp32}.log + result.json（含逐任务 bf16/fp32 GMean-MSE 与差值）
# 阶段 5：BF16 HF 生产路径导出 + 回读（PASS ⇒ 0；FAIL ⇒ 1）
$PY tools/gpu96_acceptance.py hf --execute
#   产物：hf_bf16_micro10_gas1/{result.json, train_hf.log, isolated_hf_nmse_smoke.yaml}
```

## 四、如实说明的边界（不得夸大）

1. **FP32 腿是"推理精度口径"**：从 BF16 checkpoint 加载后 `.float()` 上转，**不会恢复训练时已丢失的精度**；它不等于"BF16 训练模型 vs 独立 FP32 训练模型"的对照 ✓
2. `gmean` **不重跑参考模型**，因此**不等于**完整阈值重标定，也不等于 50-task Scheduler 路径的 GPU 验收 ✓
3. 阶段 5 使用**隔离的 4 任务 NMSE 配置**（工具强制 `pass_metric == 'nmse'`、`pass_thresholds_file=None`）✓，正式 YAML 与正式 PASS 规则**不变** ✓
4. 自适应 Eval Batch：**策略与门控已合入，但 `_infer_batch` 未接入生产评测** ⇒ 不得报告任何加速 ✓

## 五、可选（非阻塞）改进，供后续单独批准

- `open_loop_eval.py` / policy 增加**一行显式"实际推理精度"日志**（当前真 fp32 可由 `deploy/lingbot_vla_v2_policy.py:342-347` 的 `.float()` 分支证明，但日志未直书）；
- 离线复核器**双源步号口径**修正（JSONL 局部步号 vs TB 全局步号，已定位）；
- 验收工具 `ratio-gpu` 的**字段契约**修正（可换取工具承认的 GPU PASS，需开卡）。
