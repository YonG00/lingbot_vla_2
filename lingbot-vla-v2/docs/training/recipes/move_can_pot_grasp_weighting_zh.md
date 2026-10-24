# move_can_pot 接近与抓取20倍

来源1913-step200，独立训练200步，GBS56/GAS1/BF16，无复习，仅最终200保存；结束模型驻留三回合并行录像评测。

标注工具：`tools/prepare_can_pot_grasp_weights.py`（只读Parquet、CPU标注、不训练、拒绝覆盖已有快照）。权重：`refs/data_selection/move_can_pot_approach_grasp20_v1/category_loss_weights.json`；统一类别逐动作机制不改。

每条只允许一次完整抓取：活动手夹爪从开放>=0.95，经0.5到闭合<=0.2，随后保持闭合并提离3cm；其余手若也检测到则报错，不能静默选错。接近从活动手TCP首次离初始位姿5mm的前一帧开始，到闭爪开始之前；抓取从闭爪开始至初始3cm提离确认帧（含）。后续搬运与放置为背景。此为成功专家关节FK与夹爪命令代理，不宣称真实接触或物体跟随检测。

50条/7568帧：接近2694、抓取1401，共4095帧54.1%；右手29条、左手21条。不抽样删帧，不更改采样概率。FM只加权活动手，非活动手1；共享Depth/Future Depth/Video按目标帧两手max，背景1。chunk逐动作、padding排除、权重和归一化。

CPU56真实DataLoader、200类别边界已通过，报告在同快照cpu_verified.json；训练及闭环终态见阶段三记录，尚未把本轮计划计为完成。

## 0844后继续200步

配方`configs/training_recipes/move_can_pot_add200_from_0844_approach_grasp20_noreplay.json`从0844-step200接续。训练权重不变。上一轮100001被专家校验拒绝，正式三回合结果缺失；用户批准改为独立专家验证有效的100003，保留100000/100002，不依据模型成功与否挑选种子。专家校验不加载策略模型，未修改官方仿真文件；证据见同标注目录expert_seed_100003_verified.json。种子变化使三回合集合与上一轮不同，不作严格同种子总分对照。
