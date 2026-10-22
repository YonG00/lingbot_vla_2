# 第一版抓取恢复＋第二碗放置：1:5:4训练

本轮从 `20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample/checkpoints/global_step_200/hf_ckpt` 出发，不继承18cm版本。配方 `configs/training_recipes/stack_bowls200_from_weight2_s200_mix154_roleloss_eval3parallel.json`。

|抽样来源|比例|200步/GBS56配额|逐样本loss|
|---|---:|---:|---:|
|左手原始抓取|25%|2800|3|
|右手原始抓取|25%|2800|2|
|第二碗完整放置|40%|4480|2|
|同任务非目标|10%|1120|1|

抓取严格保留第一版7436帧（左41次、右59次），阶段为aim/align/descend/settle/close，未擅自增加lift。第二放置50段3743帧，全部目标并集11179，背景4430帧。类内先均匀选片段、再选片段内帧；非目标在并集补集均匀抽取。不是其他任务复习，不使用旧概率×3。

## 权重与重叠语义

采样器传递 `(raw_index, loss_weight, role)`，StreamDataset先读取原始连续动作chunk，再用本次来源权重覆盖样本权重；四项loss按原样本数归一化，正则项不变。重叠帧按本次抽样来源赋权，不叠乘。当前两类实际坐标无重叠，但同帧不同来源权重的CPU契约仍覆盖。

冻结策略包含完整 `sampling_segments` 与逐坐标 `loss_weight`。若后续只作为普通复习、没有角色抽样，保留坐标默认重要性：左抓3、其余目标2、非目标1；潜在重叠取最大值。开启当前角色采样时，本次来源权重优先。旧配置没有use_selected_frame_weights开关时完全沿用原标量权重路径。

## 训练与评测

计划200步，GBS56/GAS1、BF16，原学习率和模型训练范围不变，只保存最终step200。评测复用训练驻留模型，单任务3回合并行，显式种子100000/100001/100002，每回合均录像；参数 `single_task_episode_workers=3`，由刚提交的c9e3735评测框架执行。模型推理仍按共享服务B1串行，仿真三路并行；不承诺3倍加速。不自动关机。

入口 `tools/select_bowl_balanced_recovery.py`；规则 `configs/data_selection/stack_bowls_two_balanced_grasp_place_v1.json`。筛选产物 `refs/data_selection/stack_bowls_two_balanced_grasp_place_v1/`。启动前真实CPU必须校验角色配额、collator后权重直方图、原始动作shape和并行回合job/seed/video数量。
