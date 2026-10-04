# 训练中原地 open-loop validation 使用文档

> 适用版本：`YonG00/lingbot_vla_2`（LingBot-VLA-v2 post-training）
> 相关参数：`--train.open_loop_eval_steps` / `--train.open_loop_train_ids` / `--train.open_loop_val_ids`
> 　　　　　`--train.stop_and_save_file` / `--train.skip_final_save_on_max_steps`
> 相关文件：`lingbotvla/utils/open_loop_validation.py`、`tasks/vla/train_lingbotvla.py`
> 定位时间：2026-10-04

---

## 1. 这个功能解决什么问题

想在训练过程中**低成本地看 open-loop 学习曲线**（MSE/MAE/R² 随 step 的变化），
但每次都要「存 ckpt → 退出 → 重新加载 6B 模型 → 跑 `scripts/open_loop_eval.py`」代价太大。

本功能**原地复用训练中的模型权重**：不落盘、不重新加载、不额外实例化第二个 6B 模型，
每隔若干步暂停训练，跑一小批固定轨迹，结果写 TensorBoard，然后继续训练。

---

## 2. 一句话原理

> 每隔 `open_loop_eval_steps` 步 → 固定 seed + `model.eval()` + `torch.inference_mode()`
> → 对固定的 5 条 train-monitor + 10 条 held-out val 轨迹走 `model.sample_actions` 推理路径
> → 反归一化后算 MSE/MAE/R² 写 TB → `try/finally` 恢复**全部**临时状态 → 继续训练。

---

## 3. 快速开始

```bash
cd /data/code/lingbot-vla-v2

# ① 准备两组回合白名单（裸 JSON 数组）
printf '[50,63,76,87,99]'  > /data/train/task_splits/click_bell.monitor.json
printf '[51,52,56,66,73,75,78,84,94,97]' > /data/train/task_splits/click_bell.val.json

# ② 训练时挂上（示例：单卡 micro 10 / gbs 10）
TASK=click_bell MICRO=10 GAS=1 MAX_STEPS=1500 SAVE_EVERY=600 \
OPEN_LOOP_EVAL_STEPS=250 \
OPEN_LOOP_TRAIN_IDS=/data/train/task_splits/click_bell.monitor.json \
OPEN_LOOP_VAL_IDS=/data/train/task_splits/click_bell.val.json \
bash experiment/robotwin/single_task_train.sh
```

* `open_loop_eval_steps=0`（默认）⇒ **功能完全关闭**，现有命令行为不变
* `open_loop_train_ids` / `open_loop_val_ids` 是**文件路径**（裸 JSON 数组），不是裸数字
* open-loop eval **不触发任何 checkpoint 保存**；正式存档仍由 `save_steps` 独立控制

### STOP_AND_SAVE

每个 optimizer step 后检查一次 `<output_dir>/STOP_AND_SAVE`（或 `--train.stop_and_save_file` 指定）：
存在则**收尾存档并正常退出**，文件被改名成 `.done`。

```bash
touch /data/outputs/single/click_bell/STOP_AND_SAVE     # 看到曲线明显恶化时安全停训
```

### `skip_final_save_on_max_steps`

⚠️ **仅供 smoke test**：`max_steps` 到顶时跳过收尾存档直接退出，避免 3 步测试白存 72G。
正式训练**不要**打开。

---

## 4. TensorBoard 指标

| tag | 含义 |
|---|---|
| `open_loop/train_mse` / `train_mae` / `train_r2` | train-monitor 轨迹上的指标 |
| `open_loop/val_mse` / `val_mae` / `val_r2` | held-out val 轨迹上的指标 |
| `open_loop/{train,val}_mean_baseline_mse` | 「每维都输出自己的常数」的 MSE —— **参考线** |
| `open_loop/eval_seconds` | 单次评测耗时 |

**判据**：`mse < mean_baseline_mse` 才算真的学到（否则连常数预测都不如）。
train / val **各自按自己的评测集**算 baseline（评测集不同 ⇒ baseline 不同）。

> ⚠️ `mean_baseline_mse` 是「**全局常数**」baseline（每维一个常数，pool 所有帧），
> 与「每条 trajectory 各自输出自己的常数」（`mean_baseline_mse_per_traj`，更小）**不是同一个东西**。
> 由全方差公式 `Var_pooled = E_i[Var_i] + Var_i[E_i]`，两者差的是「各 trajectory 均值之间的差异」。
> ⇒ **评测集里轨迹越多，baseline 越大**。只放 1 条 val 轨迹时 baseline 会明显偏小，
> 不能和官方 15 条轨迹的口径直接比。

---

## 5. 口径对齐（重要）

为了能和官方 `scripts/open_loop_eval.py` 对上，本模块刻意做了这些事：

| 项 | 做法 |
|---|---|
| 预处理 | 直接用训练同一个 `build_vla_dataset`（只换 `episode_ids_file`），与训练**完全同一份** `feature_transform` |
| GT 反归一化 | `gt = feature_transform.unapply(dict(item))`，与预测**同一条**反归一化路径（官方是 apply→unapply，同构） |
| 步进 | 按 `model.config.chunk_size` 跳步、每次用**整段** chunk（官方 `chunk_ret=True` 时 `range(start, end, action_horizon)`） |
| 聚合 | `mse`/`mae` 逐条算完再对 trajectory 简单平均（官方 `np.mean(all_mse)`） |
| R² | `r2 = 1 - mse_pooled / mean_baseline_mse`，分子分母同口径（pool 所有帧、逐维方差再对维取均值） |

---

## 6. 已知坑（都是踩过的）

| 坑 | 现象 | 处理 |
|---|---|---|
| 🔴 **视觉塔预计算网格缓存** | `RuntimeError: split_with_sizes expects split_sizes to sum exactly to 192, but got [64]*30` | `get_image_features` 把 `visual_split_sizes` 等 5 个属性**按首次调用网格缓存**（`precompute_grid_thw: true`）。训练 batch 网格 = micro×相机数，评测单样本 = 3 ⇒ 必须**评测前清空、评测后恢复**（模块已内置）。⚠️ micro=1 时训练网格恰好也是 3，会**巧合躲过**，只在 micro>1 暴露 |
| 🔴 **`unapply` 要二维 actions** | `IndexError: The shape of the mask [55] ... [1, 50, 55]` | `sample_actions` 返回 `(1,T,D)`，必须 `.squeeze(0)` 成 `(T,D)`（deploy 单样本路径同款） |
| ⚠️ **不能 `model.train()` 一刀切** | 破坏 `freeze_vision_encoder` 语义 | 逐模块快照/恢复 `training` 标志 |
| ⚠️ **RNG** | `sample_actions` 的随机噪声会消耗 RNG | 评测前固定 `EVAL_SEED`，前后快照/恢复训练 RNG |
| ⚠️ **`inference_mode` tensor 逃逸** | 写 TB 时炸 | 只传 Python float，且 `_write_tb` 在 `inference_mode` **之外** |
| ⚠️ **强制 eager** | 拿到训练用的编译产物 / cache 污染 | 评测期间 `_use_compile_predict_velocity = False`，用完恢复 |
| ⚠️ **逐帧评测** | 一条 77 帧轨迹要 77 次推理（数秒/次） | 按 `chunk_size` 跳步（官方口径），一条轨迹只推 2 次 |

---

## 7. 性能参考（单卡 RTX PRO 6000 96G，fp32 eager）

| 项 | 实测 |
|---|---|
| 单次推理（1 样本、3 图、72 文本 token、10 步去噪） | ≈ 3.3 s |
| 一次完整评测（1 train + 1 val 轨迹 = 4 次推理） | ≈ 13 s |
| 一次完整评测（默认 5 train + 10 val = 30 次推理） | ≈ 100 s |
| 训练显存峰值（micro 10 / gbs 10，`vit_frozen` 5.96B 可训） | **78.6 G / 96 G** |
| 训练 step 时间（micro 10） | ≈ 19 s/step |

> 官方**闭环**评测默认 `use_compile=True`（≈0.29 s/步）；本模块按用户要求走 **eager**
> 以求正确/稳定，因此单次推理慢得多。cadence 建议 `eval_steps=250`（≈2% 时间开销）。
