# BF16-200基准续训三个任务

2026-10-20用户授权：仅保留9680和adjust_bottle BF16-200旧存档，续训400步，每200步保存，只评最终400。
训练任务：`place_container_plate`、`place_empty_cup`、`move_can_pot`，各50条，全150条24119帧。
每任务采样概率1/3，任务内帧均匀（有放回加权采样），不是按全体帧均匀导致任务比例随长度变化。

## 配置与运行位置

- 来源：`/data/outputs/official_adjust_bottle_9680_bf16_m56_add200_20261020/checkpoints/global_step_200/hf_ckpt`。
- 新run：`/data/outputs/three_task_from_adjust_bf16_200_m56_add400_20261020`。
- 控制器状态/日志：`<run>_pipeline/status.json`、`pipeline.log`。
- BF16参数/计算/HF，GBS56/GAS1、GC/compile开启、workers8、seed3。
- 继承标准方法：AL/EE/窗口/LN-FM关闭，原L1 FM/视觉三项0.004/Muon/LR不变。
- 新优化器/400步cosine调度，重新从0记步；不加载来源优化器，不继承探针更新。
- 保存本轮step200、step400两份BF16 HF，用户明确授权save_steps250→200；后续更改仍须再次获准。
- adjust_bottle等旧任务仅评测，不加入训练；因此本轮是保留能力检查，不含Replay保护。

## 当前评测范围（用户后续调整）

用户在评测期间取消原8任务队列，停止正在运行的move_can_pot，只要求adjust_bottle与click_bell各3回合。
两任务同用最终400模型，采用shared_evaluation.py，一次加载常驻服务跨两个任务复用；客户端每回合reset清空动作缓存并使用当前指令。
新目录`<run>/evaluation_400_selected_adjust_bottle_click_bell/`，结果为summary.json，服务PID为session.json，状态为status.json。
旧已完成/中断结果原地保留，旧控制器标为stopped_by_user，不再执行剩余任务。

## 原评测计划（已被上述范围替代）

只使用本轮`global_step_400/hf_ckpt`，八任务分别3个完整闭环回合，总24回合：
`place_container_plate`、`place_empty_cup`、`move_can_pot`、`adjust_bottle`、`click_bell`、
`click_alarmclock`、`press_stapler`、`turn_switch`。
用户最初写“只测500”后明确改为“续训400步，只评step400”；`adjust_buttle`按adjust_bottle。
每任务服务只加载一次，连续3回合后下线，再执行下一任务；不会每回合重载。
FP32推理、chunk50、demo_clean、noise seed0，记录每任务可行性筛选后的实际3个seed；
adjust_bottle额外匹配已验基准的100001/100002/100005。
保留每回合视频、精度审计、OIDN2.5.1无错误检查及结果一致性验证；不自动关机。
结果：`<run>/evaluation_400/<task>/summary.json`，汇总`<run>/evaluation_400/summary.json`。

## 验证与入口

```bash
python -m unittest experiment.official_control.test_three_task experiment.official_control.test_baseline experiment.official_control.test_bf16_context -v
# 默认只打印计划；显式--execute才后台运行
python experiment/official_control/three_task_pipeline.py --out /data/outputs/three_task_from_adjust_bf16_200_m56_add400_20261020 --execute
```

CPU检查实际150回合边界图像/动作/掩码、24119帧、各任务1/3概率质量和真实WeightedRandomSampler→StatefulDataLoader→collator。
先CPU整链，再独立三步GPU精度/梯度/容量探针，通过才正式训练；400步结束先验证200/400两份BF16 HF，再评24回合。
不变量：来源BF16-200、仅指定3任务、任务均衡、56/GAS1、200/400保存、只评400。
出错症状：episode集合/采样质量不符、dtype或梯度断言、缺HF/错误seed/OIDN错误；失败停后续、不减少存档或隐式切配置。
回滚：不用新入口，原BF16-200基准独立保留；采样仅新进程装配，不改变原DataLoader默认。

## 旧存档清理

按用户授权删除原FP32标准组和增强组各250/500共4份，约95.07GiB；保留9680与BF16-200，日志/数据/评测视频不删。
报告`/data/outputs/three_task_cleanup_20261020.json`，清理后空闲115.92GiB。
当前阶段详见控制器状态；配置准备/CPU/GPU探针/训练/存档/闭环结果分别确认，不以配置存在代替完成。

启动验收：服务器17项CPU契约、真实150回合数据/均衡loader检查、三步GPU探针通过；allocated77.6979GiB/reserved84.5898GiB。正式400步已启动，6007已显示本轮run。最终两份HF与24回合闭环仍待完成。
