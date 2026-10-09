# 阶段 4（GMean BF16/FP32 对拍）+ 阶段 5（BF16 HF 导出）GPU 验收报告

日期：2026-10-09｜机器：RTX PRO 6000 Blackwell 96G｜基线 HEAD：`df373c4`（含当轮修复）｜跑完由**用户手动关机**

## 阶段 4：`gmean --execute` = **PASS**

命令：`python tools/gpu96_acceptance.py gmean --execute`｜模型：`/data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt`｜candidate norm：`assets/norm_stats/robotwin_competition_clean.json`｜seed=1234、`--fixed_seed_per_traj`、noise_repeats=1｜**`thresholds_experimental_only=false`**（未启用实验性 GMean 阈值）｜`reference_model_not_run=true`

### 4.1 每任务 × 轨迹档（2/4/10）的 GMean-MSE 与 BF16↔FP32 差异（工具 `result.json.results`）

| 任务 | 轨迹数 | traj_ids | BF16 GMean-MSE | FP32 GMean-MSE | FP32 相对 BF16 |
|---|---:|---|---:|---:|---:|
| click_bell | 2 | [51, 52] | 0.008416 | 0.009227 | -8.79% |
| click_bell | 4 | [51, 52, 56, 66] | 0.007728 | 0.008451 | -8.55% |
| click_bell | 10 | [51, 52, 56, 66, 73, 75, 78, 84, 94, 97] | 0.008274 | 0.008933 | -7.38% |
| click_alarmclock | 2 | [2003, 2011] | 0.022386 | 0.023470 | -4.62% |
| click_alarmclock | 4 | [2003, 2011, 2027, 2032] | 0.031418 | 0.032212 | -2.46% |
| click_alarmclock | 10 | [2003, 2011, 2027, 2032, 2034, 2036, 2037, 2042, 2043, 2045] | 0.025832 | 0.026138 | -1.17% |
| adjust_bottle | 2 | [1, 5] | 0.517117 | 0.520541 | -0.66% |
| adjust_bottle | 4 | [1, 5, 20, 21] | 0.422052 | 0.424744 | -0.63% |
| adjust_bottle | 10 | [1, 5, 20, 21, 28, 29, 33, 34, 41, 49] | 0.397753 | 0.397949 | -0.05% |

### 4.2 逐轨迹 MSE（30 条，两腿已固定同种子）

| traj_id | 任务 | BF16 MSE | FP32 MSE | 差值 | 相对 |
|---:|---|---:|---:|---:|---:|
| 1 | adjust_bottle | 0.487339 | 0.490284 | +2.945e-03 | +0.60% |
| 5 | adjust_bottle | 0.548714 | 0.552666 | +3.952e-03 | +0.72% |
| 20 | adjust_bottle | 0.217044 | 0.219120 | +2.076e-03 | +0.96% |
| 21 | adjust_bottle | 0.546687 | 0.548171 | +1.484e-03 | +0.27% |
| 28 | adjust_bottle | 0.345647 | 0.324763 | -2.088e-02 | -6.04% |
| 29 | adjust_bottle | 0.543796 | 0.547037 | +3.241e-03 | +0.60% |
| 33 | adjust_bottle | 0.260529 | 0.261998 | +1.469e-03 | +0.56% |
| 34 | adjust_bottle | 0.522073 | 0.527542 | +5.469e-03 | +1.05% |
| 41 | adjust_bottle | 0.535575 | 0.540790 | +5.215e-03 | +0.97% |
| 49 | adjust_bottle | 0.228136 | 0.230461 | +2.325e-03 | +1.02% |
| 51 | click_bell | 0.009208 | 0.010325 | +1.117e-03 | +12.13% |
| 52 | click_bell | 0.007692 | 0.008245 | +5.533e-04 | +7.19% |
| 56 | click_bell | 0.007644 | 0.008497 | +8.526e-04 | +11.15% |
| 66 | click_bell | 0.006588 | 0.007051 | +4.624e-04 | +7.02% |
| 73 | click_bell | 0.008738 | 0.008998 | +2.606e-04 | +2.98% |
| 75 | click_bell | 0.006795 | 0.007275 | +4.798e-04 | +7.06% |
| 78 | click_bell | 0.006856 | 0.007173 | +3.161e-04 | +4.61% |
| 84 | click_bell | 0.011399 | 0.012390 | +9.905e-04 | +8.69% |
| 94 | click_bell | 0.008273 | 0.009182 | +9.089e-04 | +10.99% |
| 97 | click_bell | 0.010981 | 0.011883 | +9.020e-04 | +8.21% |
| 2003 | click_alarmclock | 0.035439 | 0.036745 | +1.306e-03 | +3.69% |
| 2011 | click_alarmclock | 0.014141 | 0.014990 | +8.493e-04 | +6.01% |
| 2027 | click_alarmclock | 0.034565 | 0.035179 | +6.135e-04 | +1.77% |
| 2032 | click_alarmclock | 0.056248 | 0.055559 | -6.881e-04 | -1.22% |
| 2034 | click_alarmclock | 0.036746 | 0.036670 | -7.537e-05 | -0.21% |
| 2036 | click_alarmclock | 0.030686 | 0.029989 | -6.967e-04 | -2.27% |
| 2037 | click_alarmclock | 0.019743 | 0.019628 | -1.146e-04 | -0.58% |
| 2042 | click_alarmclock | 0.017943 | 0.018222 | +2.793e-04 | +1.56% |
| 2043 | click_alarmclock | 0.017367 | 0.018376 | +1.009e-03 | +5.81% |
| 2045 | click_alarmclock | 0.019572 | 0.019127 | -4.446e-04 | -2.27% |

**汇总**：逐轨迹相对差 中位 **2.27%**、均值 **3.94%**、最大 **12.13%**（最大出现在 id=51）⇒ BF16 与 FP32 的**推理精度口径**差异真实存在、个别轨迹可达数个百分点。

> **显存峰值**：`gmean` 的 `result.json` **未持久化** peak 字段（工具现状 ✗，建议后续补）；运行中实测 GPU 占用约 **12 GB（bf16 腿评测）/ ~30 GB 峰值区间**，均由 `monitored_run` 的 `reserve_gib` 上限守护。

## 阶段 5：`hf --execute` = **PASS**（`verdict_reason=verified`）

| 项 | 实测 |
|---|---|
| status | PASS |
| weight_shards | 3 |
| weight_size_gib | **11.876**（≈12 GiB BF16 ✓ 正是 `hf_export_dtype=bf16` 的设计目标） |
| units_run | 1 |
| max_global_step_seen | 503 |
| DCP 泄漏 | **0** ✓ |
| returncode | 0 |
| elapsed_seconds | 198.56 |
| 导出耗时（入口自测） | 43.2 s |
| 进程 RSS 峰值 | 23.3 GiB |
| 临时目录残留 | 无 ✅ |
| 张量数 | 活模型 1708 / 磁盘 1708，缺 0 多 0，已比对 1708 ✅ |
| dtype | 活模型 bf16 / 磁盘 bf16（**同精度，无损**） |
| 数值最大绝对差 | **0.000e+00**（分块比较，块 4,194,304） |
| 入口判定 | ✅ 与活模型逐位一致 |

**触发链实证**：`09:26:11 [ckpt] 0 个新增 PASS ⇒ step 501 直接导出 HF（不写 DCP）` —— 在**第一个可达判定步**由**一次性决策补丁**触发（不伪造 PASS）✓；隔离配置为 `[turn_switch, put_object_cabinet]`、`target_total_passed_tasks=2`、`pass_nmse=1.66`、`pass_thresholds_file=null` ⇒ 两个任务 Bootstrap 均不过（1.7623 / 2.1642 > 1.66）⇒ 必须训练 ⇒ ≥1 Train Unit ✓

### 5.1 HF 产物清单（删档前留证；权重随后已在机器上删除，机器现已关机）

| 文件 | 字节 | sha256(前16) |
|---|---:|---|
| model-00001-of-00003.safetensors | 4995757160 | `75190678cc9cbb37` |
| model-00002-of-00003.safetensors | 4992915784 | `f5903f4bf8f02a0d` |
| model-00003-of-00003.safetensors | 2763403150 | `300f7c31d063ef71` |
| model.safetensors.index.json | 207389 | `2f33ce3d900b3f3a` |
| config.json | 4504 | `74878fec937e8ae2` |
| tokenizer.json | 11422930 | `ac5069fbe4aa057a` |

## 本轮发现并修复的三个缺陷（诚实记录）

1. **`task_names: null` 崩溃**（我上轮补丁的 bug）：正式 YAML 用 `task_names: null` 表示全部任务 ⇒ 解包 NoneType ⇒ 启动即崩（零卡时浪费）。⇒ 改为 `list(cfg.get('task_names') or [])` + CPU 测试 ✓
2. **HF 验收命令不满足训练器前置条件**（真正的阻塞）：训练器要求 `save_steps≥1`、`dcp_save_mode=always`、`async_save_hf_weights=false` 等 8 项，而工具沿用 bench 的 `save_steps 0` 且缺另两项 ⇒ 启动即 `ValueError`（第一次尝试 0 步失败）。⇒ 新增 `apply_hf_trainer_overrides()` 统一归一化（`save_steps=1e9` 既不触发周期 DCP、又满足校验）+ **镜像训练器校验的 CPU 测试** ✓
3. **`hf-plan` 文案自相矛盾**（写 `save_steps 0`）⇒ 已更正 ✓

修复后 `tests/test_gpu96_acceptance.py` **37 passed**；未改正式 NMSE YAML、未启用实验性 GMean 阈值、未接线 Adaptive Eval Batch ✓

## 遗留（因先关机而未能完成）

- **清理测试存档**：`/data/outputs/gpu96_acceptance/hf_bf16_micro10_gas1/hf_milestones`（**12 GiB**，已在本报告第 5.1 节留 SHA256）及同目录其余测试输出（多数 ≤17 MB）。机器已关 ⇒ **下次开机我立即清理**（只删精度无关的测试权重，保留日志/result.json/正式数据）✓
