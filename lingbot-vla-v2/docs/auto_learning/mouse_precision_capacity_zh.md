# 鼠标FP32主参数与显存容量测试准备

2026-10-19用户决定停止原BF16主参数训练，改为：**FP32主权重 + BF16前向 → BF16 HF存档 → FP32加载评测**。随后明确要求只准备代码，等再次开卡后再测；首档指定micro32/GAS1/GBS32。用户回忆micro28曾可运行，目前未按本轮精度/数据配置复核，不能当作当前容量结论。

原run `al_mouse_pad_primary50_g56_from_mouse500_add500_20261019_v1` 已按用户要求停在50步，无HF存档；先移除流程控制进程，再停止torchrun及训练/数据进程，确认GPU进程清空，状态改为cancelled_by_user、shutdown=false。旧流程不会再评测或关机，日志与产物未删除。

## 已准备的代码

- `launch_skill_mix_task.py`不再无条件强制mixed_precision=false；新增默认关闭的`--fp32-master`，打开时实际CLI和训练快照均为`enable_mixed_precision=true、enable_fp32=false`。从专用配方到真实worker的重建路径有CPU契约。
- `recipes/20261019_place_mouse_pad_primary50_fp32_master.py`默认纯主50、来源混训前鼠标step500、micro32/GAS1、GC、P10/.8/1.5、视觉0.004、500步、250/500 BF16 HF，最终FP32加载评测3回合。默认预览；只有明确加`--execute`才启动。
- `precision_batch_probe.py`独立3步短测，默认micro32/GAS1，不存档、不关机；不继承短测权重启动正式训练，不改变正式250/500节奏。短测临时使用smoke_no_checkpoint和非HF-only分支，避开HF-only收尾仍写HF的实现路径。
- `precision_audit.py`仅在进程启动设置`LINGBOT_PRECISION_AUDIT=<输出JSON>`时启用，第一步优化器更新前后记录：优化器持有主参数、梯度、优化器状态的dtype分布，关键Linear模块的权重/输入/输出dtype，峰值allocated/reserved显存。主参数非FP32或动作专家未观察到BF16输出立即报错；不额外执行前向，不改变张量。
- `precision_eval_server.py`只在FP32评测档使用：强制关闭BF16快速加载，检查模型参数实际全FP32，并在真实推理中记录关键层dtype，不额外前向。关键动作专家未出现FP32输出则评测失败，保留机器。原评测客户端仍不覆盖。

这描述的是参数存储和关键层可见dtype，不保证每个融合内核内部都采用FP32；代码中的融合MoE等路径可能使用其他内部计算精度。BF16 HF转FP32只扩大存储，不能恢复存档时舍弃的低位。若要保留FP32主权重用于下次续训，需另行决定FP32 HF或完整训练状态；本次不改变已授权BF16 HF方案。

## 开卡后的测试顺序（尚未执行）

在实验包根目录，先预览，再经用户开卡授权执行：

```bash
python experiment/robotwin/precision_batch_probe.py --micro 32 --gas 1
# 开卡后才加 --execute，实际在后台跑3次优化器更新
```

先确认GPU空闲、服务器代码SHA与本地提交一致、CPU契约通过。第一档32若失败保留日志并检查具体原因；容量不足时才试28/24等较小档。32通过后，再按36/40/48/56等档逐步测，只有前档保有显存余量才继续。不得从BF16主参数的56档外推FP32主参数容量。

每档至少完成3个optimizer step，dtype审计必须通过；记录峰值显存、更新后优化器状态、稳定步时及实际样本吞吐，不能只测模型加载/前向。最终选择保留实用余量且有效吞吐合适的档，容量边界不自动等于正式训练最佳档。首档也尚未跑，**当前无法回答96G最大支持多少**。

报告为`/data/outputs/<probe_run>/probe_result.json`，原始dtype为`precision_audit.json`，日志位于`skill_launches/<probe_run>.log`。短测失败不自动重试、不自动减batch、不关机，交由操作者据日志判断。

GBS=微批×梯度累积×卡数。单卡不累积时，GBS上限等于本配置实测微批上限；采用梯度累积后，GBS可以大于微批，显存不会简单按GBS同比增加，但耗时及优化行为会变化，因此不存在脱离累积设置的单一“96G最大GBS”。

正式训练入口预览：

```bash
python experiment/robotwin/recipes/20261019_place_mouse_pad_primary50_fp32_master.py
```

确认容量和档位后才添加`--execute --shutdown`；沿用训练→250/500 HF校验→最终模型3回合评测→正常结束关机的流程。当前未启动这轮新训练，也未启动GPU短测。

## 本次验证状态

本地CPU回归83 passed、1 skipped（缺torch装配项）、36 subtests。覆盖参数传递与worker重建、共享dtype计数、更新后主参数检查、FP32评测开关、评测记录不增加forward或改变返回值，以及探针不覆盖正式存档参数。新代码尚未同步至服务器，也未执行服务器/GPU验证；不能把CPU通过写成精度或容量已实测。

## 后续开卡实测进展（GAS固定1）

上述为准备阶段记录。开卡后服务器CPU84项通过，micro32首轮v1因静态窗口缓存缺失而未开始更新；已补CPU缓存准备及冻结AL契约，服务器85项通过。v1不是OOM。

v2首次optimizer更新后，审计记录主参数/梯度/优化器状态均为FP32，但视觉、语言、动作专家和输出头采样Linear输出全部FP32，审计主动失败。峰值reserved77.66GiB只属于这个非目标路径，不算FP32主参数/BF16计算容量通过。原因是`parallel_state.fsdp_enabled = fsdp_size > 1`，单卡跳过FSDP mixed policy；此前仅打开mixed开关并不会自动启用BF16计算。

现仅对mixed=true、fp32=false、fsdp_enabled=false的训练前向添加CUDA BF16 autocast。默认BF16、显式FP32和多卡FSDP分支保持原行为；CPU测试包括真实Linear前向BF16、主参数/梯度/SGD动量FP32。更新后本地88 passed、2 skipped、36 subtests；服务器90 passed、36 subtests，实际CUDA重测待运行。仍从micro32/GAS1开始，不修改正式存档节奏。
