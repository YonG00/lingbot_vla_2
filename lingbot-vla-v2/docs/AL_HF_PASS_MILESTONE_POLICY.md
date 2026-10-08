# Auto Learning：DCP 1000 步 + HF PASS 里程碑（待 GPU 验收）

基于上传的 `a01a404` 源码及第一轮补丁重建；GitHub `26a7fe7` 远端源码无法直接拉取。
**WorkBuddy 合入 `26a7fe7` 前必须 `git apply --check`、检查 diff 并跑远端测试。**

## 设计语义

| 场景 | 行为 |
|---|---|
| 每 1000 optimizer step | 按 AL 安全 Unit 边界保存完整 DCP；不同时导出 HF |
| 每新增 2 个非 Bootstrap、首次 PASS 的不同任务 | 若尚未收工且没有同一步 DCP：直接提取当前模型权重，写独立 HF（不经过 DCP） |
| 课程达到总 PASS=4、达到 MAX_STEPS、STOP_AND_SAVE | 只保存最终 DCP，绝不额外导出 HF |
| Bootstrap 已通过任务 | 计入总 PASS 停止目标，但**不计入**「新增 2 个 PASS」HF 里程碑 |
| 训练后 Rescan 自动首次 PASS | 计入新增里程碑；重复 PASS/REOPEN 后复学不重复触发 |
| `SMOKE_NO_CHECKPOINT=1` | DCP、HF 均禁用，不受新策略影响 |
| 旧 DCP 恢复 | 里程碑 cursor 从历史任务状态初始化；旧数据中 Bootstrap 不可确定时不把 `auto_passed` 伪装成新增任务 |

周期 DCP 延迟到最近学习 Unit 边界时，实际保存步数可能不是整千（这是安全 Resume 的必要保证）。

## 新 CLI 参数

`--train.hf_pass_interval 2`（默认 0=关闭）。必须同时满足：

- `--train.save_steps 1000`（或其他正整数）
- `--train.save_epochs 0`
- `--train.dcp_save_mode always`
- `--train.save_hf_weights false`、`--train.async_save_hf_weights false`
- 开启 Auto Learning，且不能开启 `smoke_no_checkpoint`

正式 BF16 启动器 `experiment/robotwin/al_50task_bf16.sh` 默认 `SAVE_EVERY=1000`、`DCP_MODE=always`、`HF_PASS_INTERVAL=2`，DCP 本身不附带 HF。

## HF 文件在哪？

HF 只写在 `<TRAIN_OUT>/hf_milestones/global_step_N/hf_ckpt`；与 `<TRAIN_OUT>/checkpoints/global_step_N` 分离，以免 DCP Resume 错误读取仅含 HF 的目录。

写盘先进入同目录 `.hf_ckpt.tmp.*`，成功后 `os.replace` 原子发布。失败不产生 `hf_ckpt` 正式目录；从旧 DCP 重新训练同一 step 导出则采用 `_retry_001`，不会覆盖已有里程碑。

`get_model_state_dict(full_state_dict=True,cpu_offload=True)` 由全部 FSDP rank 共同参与，rank0 将完整权重快照放到 CPU 并同步写 HF；训练需等待此次里程碑导出完成。这个过程**额外消耗 CPU 内存和 I/O 时间**，不使用 DCP 中转。正式启用前必须用相同的 FSDP2、MoE 模型验证 HF 权重 key 和开环推理等价性，及 CPU 内存余量。

## 安全约束与尚未实现

- **没有**擅自删除任何旧 DCP：启动器暂时关闭原有按 HF 剪枝的看门狗 `PRUNE=0`；该看门狗与「独立 HF + 必须保留可恢复 DCP」不兼容。因此**磁盘会随每 1000 步增加一份 DCP**，长期训练前必须明确容量计划或实现专用安全 DCP 最近两份策略。
- HF 里程碑失败会 fail-fast 中断训练（防止假装保存成功）；新周期 DCP 仍按旧存档错误处理。
- 训练结束 DCP 已存在则不再复制 HF；如果正常结束时与周期 DCP 同一步，只存一份。
- 未更改模型训练算法、Hardness 0.10、NMSE/GMean-MSE 阈值策略，也未加入扫描缓存。
- 尚未在真实 48/96G GPU、FSDP2/MoE 环境执行；不能在 GPU 验收前直接宣称正式训练可靠。

## 建议验证（WorkBuddy）

先在独立短作业制造「2 个新 PASS、但未达到总 PASS=4」的条件，验证出现 HF-only 里程碑并可加载；再测「2 个新 PASS 同时达到课程目标」只出 DCP，以及与周期 DCP 同步、断点恢复、CPU 峰值。无需一开始运行完整 50-task 长训练。
