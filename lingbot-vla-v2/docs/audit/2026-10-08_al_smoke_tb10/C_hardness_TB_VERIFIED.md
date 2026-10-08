# C. Hardness 扫描 —— TB 写入核验（含第二次）与耗时归因

## 两次扫描（事件流 + TB 双向核对）
| # | 任务 | wall-time | is_first | 打分样本 `n_scanned` | 轨迹数 | TB 是否写入 |
|---|---|---|---|---|---|---|
| 1 | place_container_plate | **400.634 s** | true | **2204** | 14 | ✅ `hardness_scan_seconds`/`_warmup`/`_samples`/`_trajs` @**step 500** |
| 2 | turn_switch | **246.99 s** | false | **1357** | 14 | ✅ `hardness_scan_seconds`/`_steady`/`_samples`(1357)/`_trajs`(14) @**step 505** |

⇒ **第二次扫描已正确写入 TensorBoard**（`auto_learning/hardness_scan_samples` = [(500, 2204.0), (505, 1357.0)]；
`_trajs` = [(500,14.0),(505,14.0)]），`_warmup` 只在 500、`_steady` 只在 505 —— 首次/稳态分离按设计工作。

## 耗时归因（回答"是不是首次编译"）
- **不是编译**：本轮启用 `TORCH_COMPILE_DISABLE=1`（实测 `torch._dynamo.config.disable=True`），日志中**无任何 dynamo 告警**。
- 扫描的真实构成 = `torch.no_grad()` 前向（`RealHardnessScorer.max_batch=8`，即约 793/493 次前向）+ 数据集取数；
  耗时与**打分样本数**近似成正比（2204→400.6 s ≈ 0.18 s/样本；1357→247.0 s ≈ 0.18 s/样本）**完全吻合**。
- ⚠️ **CPU / IO / GPU 未分别埋点**：现有埋点只有 wall-time + 全局 nvidia-smi 采样（3 s）；
  本文不编造分项。若要分项，需要在 `RealHardnessScorer.score` 内加"取数 vs 前向"两段计时（另开补丁）。

## 附：Checkpoint 状态（重要）
- ✅ DCP 完整：`checkpoints/global_step_510/{model,optimizer,extra_state}`，合计 **39.12 GiB**，
  日志有 `Distributed checkpoint saved ... successfully!`（16:59:50）⇒ **可用于 resume**。
- ❌ **HF ckpt 不完整**：`hf_ckpt/` **不存在**，只剩 `.hf_ckpt.tmp.510.7061/`（6 分片只写了 4 个 ≈15.8 GiB）
  ⇒ 关机（17:01:5x）打断了异步 HF 保存 ⇒ **该 ckpt 不能用 HF 格式加载/评测**（DCP 不受影响）。
