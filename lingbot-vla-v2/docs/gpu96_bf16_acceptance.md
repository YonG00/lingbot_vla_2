# 96G BF16 专项验收工具（仅测试，不启动正式训练）

**适用代码基线：用户报告 `e34edcd9ff6b6297e7c75cc6183e976189520d98`。** 本包在本地以用户上传的 `adb2d75` 源码顺序叠加已合入的审查补丁 + GMean 链路补丁模拟该源码状态。工作机必须在真实 HEAD 上 `git apply --check`、CPU Gate 复验。

核心入口：`python tools/gpu96_acceptance.py {bench,ratio,ratio-gpu,gmean,hf,hf-plan}`。**所有涉及 GPU 的子命令默认仅打印计划。必须明确加 `--execute` 才开始执行。** 每个执行使用全新目录，同名已有目录会拒绝覆盖。工具不租卡、不关机、不启动正式 50-task、不修改任何正式 YAML、从不保存 DCP Resume。

## 无卡预检（现在执行）

在 `/data/code/lingbot-vla-v2`：

```bash
PY=/data/miniconda3/envs/lingbotvla/bin/python
$PY -m pytest -q tests/test_gpu96_acceptance.py
$PY tools/gpu96_acceptance.py bench --micro 28 --gas 1
$PY tools/gpu96_acceptance.py ratio --micro 28 --gas 2 --dp 1
$PY tools/gpu96_acceptance.py ratio --micro 28 --gas 1 --dp 4
$PY tools/gpu96_acceptance.py ratio-gpu --micro 28 --gas 2
$PY tools/gpu96_acceptance.py hf
$PY tools/gpu96_acceptance.py hf-plan
$PY tools/gpu96_acceptance.py gmean --tasks click_bell click_alarmclock adjust_bottle
```

**GMean 的 plan 模式需要存在 val IDs 文件**，没有远端 `/data/train/task_splits_50` 时请按路径覆盖，不要修改工具内硬编码。`hf-plan` 只供理解与独立对照；`hf` 已实现受控执行。

如实际路径不同，使用 `--model-path`、`--split-dir`、`--train-config`、`--phases`、`--python`、`--out-root` 等参数。训练 benchmark 使用项目**真实训练器与真实模型**，但**有意关闭 Auto Learning**，避免 50-task 扫描污染纯训练吞吐。`step_offset=500` 是**从 HF 权重新起一个 optimizer**而非 DCP Resume。

## 开卡后：按顺序，一项 PASS 才继续

必须由用户先开好**单张可见 96G GPU**。同卡没有其他活跃训练进程（预检 >5GiB 已占用会拒绝），存储空间足够，QWEN3VL_PATH 正确。

1. **先 28、GAS1，compile-off 的真训练（5 步预热 + 20 步计时）。**

   ```bash
   $PY tools/gpu96_acceptance.py bench --micro 28 --gas 1 --execute
   ```

   将 `result.json` 中的 `mean_step_seconds`、`samples_per_second`、`peak_nvidia_smi_mib`、loss、`returncode` 与 `status` 带回。OOM、非有限 loss、少于 25 步、显存余量低于 8GiB → 不通过。**不自动升级、回退或重试**；由 Agent 看数据和用户同意后选下一个配置。

2. 28 成功且有余量时，依次单独测试 `--micro 32 --gas 1`、必要时 `36/1`；28 OOM 则**新进程**试 `24/1`、必要时 `20/1`、`16/1`。对照 `14/2`（GBS28）以及 `10/1`（GBS10）。始终以相同 compile、数据和 optimizer 配置比较 samples/s，不仅比较秒/step。若正式要开 compile，应为最终候选另运行 `--compile on` 专项；开启 compile 可能产生额外 warmup 和峰值。

3. **真实动态 GBS 采样（独立小型 AL Smoke）**：先 CPU `ratio` 已计算 GBS56=39 NEW/17 Replay。用户同意后：

   ```bash
   $PY tools/gpu96_acceptance.py ratio-gpu --micro 28 --gas 2 --execute
   ```

   该命令复制 2-task 的 NMSE 测试 YAML 到输出目录，改用 `new_ratio=0.7`、降低 hardness 采样，不改正式 YAML。经真实 Trainer → AL DataLoader → Train Unit 消费后，从 JSONL 的 `train_unit.samples_seen`、`batches_built` 和指标 `sampling/replay_samples_per_unit` 对账。**如果尚无 PASS Replay 池、无 Train Unit、没得到真实 Replay，结果明确为 `BLOCKED`，不能把 CPU 比例计划称为 GPU 验收通过。** 发生 OOM 也不自动回退。单卡通过不等于 4 卡 FSDP2 已通过。

4. **GMean 双精度对拍（不改变课程 Registry）**：在 3 个任务上一次取齐 10 条 held-out val 轨迹，BF16 和 FP32 各调用**一次**项目真实 `scripts/open_loop_eval.py`，都启用 `--fixed_seed_per_traj` 和相同 seed。各任务由**同一批 10 条**导出嵌套的 2 / 4 / 10 条 GMean、相对偏差、可选实验阈值下的 PASS 翻转，不需重复加载 6B 模型 3×2×3 次。

   ```bash
   $PY tools/gpu96_acceptance.py gmean \
     --tasks click_bell click_alarmclock adjust_bottle \
     --model-path /data/outputs/single/click_bell/checkpoints/global_step_500/hf_ckpt \
     --execute
   ```

   可附 `--thresholds /data/eval_results/.../pass_thresholds_gmean200_warn.json` 做**探索性**判定比值；未标定阈值不能用于正式 PASS。该命令比较**候选模型**的两种加载/推理精度，使用 candidate `robotwin_competition_clean.json` 的物理 action 口径；**不自动重跑参考模型**。若要宣称完成参考-候选配对标定，还必须单独按官方 `robotwin.json` 使用同一轨迹、种子、数据集和评测定义重新标定参考；不能因为 candidate BF16/FP32 对拍 PASS 就启动正式 GMean。

   注意：FP32 模式从一份 BF16 权重加载并上转时不会恢复丢失的权重精度；FP32/BF16 对比体现的是**推理精度口径**，并不等于两个独立训练的 FP32/BF16 模型对照。若模型 dtype 配置阻止正常执行，报告 BLOCKED，不对活 FSDP2 模型原地 `.float()`。

5. **BF16 HF 真实导出**：最后单独运行。

   ```bash
   $PY tools/gpu96_acceptance.py hf --execute
   ```

   它通过真正的 `tools/hf_direct_export_acceptance.py` 调用生产导出路径；只把里程碑决策临时强制一次 HF，不伪造 PASS。使用临时隔离的 4-task NMSE 短配置，`hf_export_dtype=bf16`，停在 max_steps 时跳过 final DCP。核对 **HF safetensors ~12GiB**（实际大小以输出为准）、完整性、回读误差、RSS、GPU 显存及原子临时目录。**若 Scheduler 比 max_steps 更早自然结束，生产路径仍可能写最终 DCP；脚本对此判 FAIL，不能保证所有异常路径无大文件，必须事前留足空间**。失败不要重复长时尝试。

## 结果目录与 STOP 原则

默认 `/data/outputs/gpu96_acceptance/`。每项不同子目录；含 `plan.json`、`train.log`/`bf16.log`/`fp32.log`、`result.json`。保留轻量结果，不自动删除任何旧产物或源模型。

- `PASS`：该项完整完成且守住对应检查。
- `BLOCKED`：运行路径不满足验收前提（例如无实际 Replay、评测模型精度不兼容）；**不可写成 PASS**。
- `FAIL`：OOM、超时、非有限值、模型保存数值不一致、错误回码等；立即暂停后续 GPU 阶段。

**验收范围限制**：性能测试只代表 AL-off 真优化器吞吐，不能声称包含 Bootstrap/Review 扫描开销；两任务 ratio-real 只验采样器与 Unit，不代表完整 50-task 正式训练；HF 这次验证单卡，不等于多卡 collective 已测试；DCP Resume 未做。测试脚本绝对不自动关机。

## 合入与 Agent 汇报

在 Git 顶层（含 `lingbot-vla-v2/` 的仓库根）：

```bash
git status --short
git rev-parse HEAD
git apply --check /path/to/AL_GPU96_BF16_ACCEPTANCE.patch
git apply /path/to/AL_GPU96_BF16_ACCEPTANCE.patch
cd lingbot-vla-v2
PYTHONPATH=. python -m pytest -q tests/test_gpu96_acceptance.py tests/test_dynamic_gbs_ratio.py tests/test_gmean_pass_pipeline.py
```

先只提交、推送、同步工具补丁并无卡 `plan`；正式开卡命令在用户批准后才执行。最终提供 GPU 型号、命令、SHA/Git HEAD、每项 PASS/FAIL/BLOCKED、每项真实耗时与成本、推荐配置，严禁自动关机或开始正式 50-task 长训练。
