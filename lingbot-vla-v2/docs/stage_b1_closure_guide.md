# Stage B1 收口说明（review v0.2 修复）

> **对应代码**：`feature/auto-learning-v1` @ **`265b350`**（上一版 `2d9db11`）
> **为什么要读**：`stage_b1_integration_design.md` 的 **§18.2 / §18.4 第 3 条**描述的是
> 「resume 会把中断的 unit **从头重跑**」——**这条已被本版推翻**，请以本文为准。
> **评审来源**：`stage_b1_repo_review_v0_2.md`（P0×3 / P1×5 / P2×2）。

---

## 1. 一句话总结

**保存点**从「任意 save_steps」收紧为「**learning unit 边界**」，
**resume** 从「unit 从头重跑」改为「**只接受边界存档；落在 unit 中途直接报错**」，
并在启动阶段补了一批 **fail-fast**（配置/基线/批大小/增强/开关判定）。

---

## 2. 语义变化对照（旧 → 新）

| 行为 | 旧（`2d9db11`） | 新（`265b350`） |
|---|---|---|
| `_al_on` 判定 | 把**文件路径字符串**当 config 取 `.enabled` ⇒ **恒 False**，AL 保护全静默失效 | 真解析配置文件（`auto_learning_enabled()`） |
| `rmpad` | 静默改成 false，日志与实际不一致 | **启动前报错**，要求显式关掉 |
| unit 记账时机 | 等到**下一个** `on_step_begin()` 才回填 ⇒ 存在「模型已训到 N、Scheduler 只记到 N-50」的窗口 | **最后一步跑完当场回填** |
| 存档点 | 任意 `save_steps`，可能落在 unit 中途 | **只在 unit 边界**：periodic save 推迟到最近边界；STOP_AND_SAVE 跑到边界再停；epoch/收尾先 `flush_partial_unit()` |
| resume（unit 中途） | **本 unit 从头重跑**（重复训练已完成的 step） | **fail-fast**，给出「用上一个边界存档」的指引 |
| resume（unit 已开、0 步未跑） | 重跑 | 安全重发 request（模型没被动过） |
| AL 批大小 | 无校验（可能被 DataLoader 切碎） | `batch_size` 必须 == `train.dataloader_batch_size`，否则报错 |
| 评测数据集缓存 | 只增不减（tag 带 ids 指纹 ⇒ key 越来越多） | 每次评测后 `clear_dataset_cache()` |
| baseline | 没给只 warning ⇒ NMSE=None ⇒ 任务被静默排除 | **默认必填**；且与**本次运行时配置**做指纹对拍 |
| step-1 sanity | 被 catch 成 warning 后继续烧卡；且因 `_al_on` 失效**根本没跑** | **直接 raise**；触发时机 = AL 首个 step（含 resume 后第一步） |
| `enable_resume` 缺 AL 状态 | 静默从零 bootstrap | **fail-fast**（要起新 run 请用 `--train.load_checkpoint_path`） |
| hardness 取样本 | 若数据集开了增强，同一样本两次扫描结果不同 + 污染全局 RNG | 取 item 期间**关增强 + 快照/还原 RNG**；v1 默认要求 `image_augment=false` |
| `unique_batches` | 实为「不同 loss 值数量」，且属性恒为 0 | 按 `(new ids, old ids)` 去重 |
| disabled 路径 | 仍 import、仍挂 `_auto_learning_train_dataset`、checkpoint 多写 `auto_learning: None` | **字面零侵入**（不 import / 不挂属性 / 不写键） |

---

## 3. 新语义的两个要点

### 3.1 `at_safe_checkpoint_boundary`

只有在 **没有在飞的 unit**（`unit_steps == 0`）且 `step_in_unit == 0` 且 Scheduler 无
pending request 时，才允许落盘 Auto Learning 状态 —— 即 **Scheduler 的账与模型权重完全对齐**。
从这样的存档恢复，既不会重跑已完成的 step，也不会出现「模型领先一个 unit」。

### 3.2 训练收尾的 `flush_partial_unit()`

训练可能正好停在 unit 中途（`max_steps` 到顶 / STOP_AND_SAVE / epoch 切换）。
收尾时：

* 已跑 k 步（k > 0）⇒ **按 k 步记账**（模型确实更新了 k 次，不能当没发生）；
* 一步没跑（k == 0）⇒ **撤回** request，不改账。

调用返回后一定是安全边界，因此收尾/轮末存档都是干净的。

---

## 4. 快速开始：48GB BF16 短回归（A–G）

```bash
# 前置（脚本也会自检）
export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct

cd /data/code/lingbot-vla-v2
DRY_RUN=1 bash tools/al_b1_regression.sh          # 先看计划
bash tools/al_b1_regression.sh                    # 跑 A–G
ONLY=F,G bash tools/al_b1_regression.sh           # 只跑最关键的 F/G（F/G 不过就没必要跑别的）
```

| 项 | 内容 | 依赖 |
|---|---|---|
| A | Legacy vs Integration 对拍（不允许数值容差） | `/data/tmp/legacy/lingbot-vla-v2` |
| B | BF16 hardness 确定性（R2 + R5b） | 一份 `hf_ckpt`（默认自动找 `al_g10`） |
| C | 2→4 评测缓存（R6） | 同上 |
| D | train → eval → train（events 里 ≥2 个 train_unit） | 复用 F 的 events |
| E | 2 任务 7+3 smoke（`system/replay_slots=9`） | `task_splits_2task/manifest.json` |
| F | **50-step unit 当场回填** + 边界存档可恢复 | — |
| G | **boundary resume 语义等价**（continuous 100 ≡ 50+resume→100） | — |

产物：`/data/tmp/al_b1_regression/<时间戳>/`（`logs/` + `reports/` + `summary.txt`）。
脚本**不自动关机**。

---

## 5. FAQ

**Q：我有个 `2d9db11` 时代的存档，能 resume 吗？**
A：看它落在哪。落在 unit 中途 ⇒ 会 **fail-fast**（这是设计，不是 bug）；
落在边界 ⇒ 正常接上。不确定就用上一个边界存档，或从零重跑。

**Q：`save_steps=750` 但 unit=50，存档会在哪一步？**
A：step 750 若不在 unit 边界，会**推迟到最近的边界**（最多晚 49 步），并在日志里说明。

**Q：STOP_AND_SAVE 为什么不是立刻停？**
A：AL 开启时会**跑到最近 unit 边界**再收尾存档，最多多跑 `eval_interval_steps-1` 步。
drill 脚本的超时请留余量。

**Q：跑 smoke 没有 baseline，现在报错了怎么办？**
A：两种正规做法：① 先 `compute_task_baseline` 算出真 baseline（**不需要 GPU**）；
② 确属 smoke，在 AL 配置里显式写 `allow_missing_baseline: true`（会有 warning）。

**Q：`image_augment=true` 为什么直接报错？**
A：v1 要求关掉（否则「同一样本两次 hardness 扫描」不可复现）。
确要开就在 AL 配置里写 `allow_image_augment: true`。

---

## 6. 边界 / 仍待办

* 任意 **mid-unit resume**（持久化 prefetch/compositions/partial stats）**v1 明确不做**。
* §39 的审计目前只覆盖「AL 首个 step（含 resume 后第一步）」；**「第一次 task transition」
  那次审计未接**（不影响 §14 冻结 Gate）。
* `configs/auto_learning/*.yaml` 里的 `pass_nmse` 仍是 **smoke 值**，正式训练必须按真实
  baseline 重定。
* §14 冻结 Gate 的最后一项 = **本文件 §4 的 A–G 全绿**（需开卡）。
