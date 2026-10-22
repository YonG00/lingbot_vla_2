# stack_bowls_two：第一版模型暴露问题的目标片段

本次只准备筛选结果，不修改训练、不启动新训练。问题依据来自用户观察 `20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample` 的step200：左手远距离夹碗失败、第二个碗放置失败。候选来自50条成功示范，不是从失败推理中伪造成功标签。

|目标|判定和片段范围|结果|
|---|---|---|
|左手远距离夹碗|示范frame0左TCP到该次示范闭爪位置的XY距离≥25cm；保留瞄准、对齐、下降、停稳、闭爪、提起全部连续帧；提起至少6cm且连续3帧竖直速度≤1.5cm/s、夹爪保持闭合，3秒内满足才接受|6段，681帧；episode1453、1458、1460、1464、1492、1499|
|第二碗放置|每条示范按两次有效松爪的时间顺序定位第二次；由既有放置校验器验证下降方向、路径、闭爪、停稳、松爪漂移和撤离，保留place_aim到place_retreat全部连续帧|50段，3743帧|

去重并集4361/15609帧（27.94%）；两类在提碗与放置对准衔接处重叠63帧，联合使用时一帧只算一次，不能因双标签重复相乘loss。此次不为凑30%抽帧，也不额外加入右手远距离抓取。两类池分别保存，可单独启用、调整配额；不能直接按并集均匀抽取便声称两类均衡。

“远”是episode起点到示范闭爪位置的距离，基于既有25cm阈值；它不等于实时碗位置或推理第二回合的精确几何复现。姿态和位置通过关节状态FK得到；提碗判据证明末端抬升，不证明物体实际被夹住。没有物体位姿或接触标签，未做视频抽检，不宣称视觉质量验收。place_aim可包含携碗运输，具体阶段边界在events.json中可分开选择。

## 产物和复现

- `refs/data_selection/stack_bowls_two_failure_segments_v1/summary.json`：覆盖和规则。
- 同目录 `events.json`、`segments.csv`：每段episode、手臂、起止帧/秒、阶段、距离和窗口跨界统计。
- `far_left_grasp/`、`second_bowl_place/`：两类各自完整片段和稳定episode/frame坐标。
- `selected_frames.json`：联合去重目标样本；`validation.json`：输入指纹、去重及CPU检查。
- 配置 `configs/data_selection/stack_bowls_two_failure_segments_v1.json`，入口 `tools/select_bowl_failure_segments.py`；真实数据使用 `--dataset-root <root>`，缓存使用 `--frames-json <path> --fps 15`，并指定 `--task stack_bowls_two --settings configs/data_selection/stack_bowls_two_failure_segments_v1.json --output-dir <new-dir>`。

## 后续训练前必须明确的边界

原模型动作预测长度50帧不变时，左手夹取294个锚点、第二碗放置2450个锚点的未来动作窗口越过所选片段末端。这是窗口统计，不是筛选错误。选择锚点不等于只监督片段内动作；本次没有修改动作mask、loss、采样比例或训练起点。3项纯CPU契约测试通过，真实50条示范筛选成功，无torch/GPU调用。

## 2026-10-22 用户批准的1:2针对训练

正确配方 `configs/training_recipes/stack_bowls200_from_weight2_s200_target90_far1_place2_loss2.json`。以第一版 `20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample` 的step200继续（用户纠正；旧160来源配方仅作错误实验历史），200步、GBS56/GAS1、BF16，只有最终step200存档；最终闭环3回合，全部录像，不自动关机。

90%来自目标片段，其中远左抓:第二放置按1:2分配；10%是本任务非目标帧（不是其他任务复习）。整轮11200抽样中远左抓3360、第二放置6720、其余1120。每类先均匀选事件片段，再均匀选片段内帧，避免长片段主导；全部原始帧有机会抽中，不做关键帧稀疏化。抽样是带放回训练，不保证每帧在200步内都被抽到。63个重叠帧可属于两类抽样事件，但每次取样loss只乘2一次。

目标四项逐样本loss×2、非目标×1，按原样本数归一化；关闭旧采样×3和loss×3。事件边界被写入重要性配置 `sampling_segments`，随registry冻结及HF一起导出；事件比例由 `train.target_segments` 显式控制，未配置时原采样器不变。3个新配额/边界/恢复测试通过，共69项配方测试通过。

## 当前使用规则：≥18cm、14段（用户2026-10-22确认）

历史25cm规则及6段产物保留用于追溯，不再用于本次训练。当前配置 `configs/data_selection/stack_bowls_two_failure_segments_far18_v2.json`，配方 `configs/training_recipes/stack_bowls200_from_weight2_s200_far18cm_target90_loss2.json`。14段左远抓1506帧、50段第二放置3743帧，交集114帧，并集5135帧；目标比例依旧90%，内部1:2，并不随原始帧数比例改变。来源为第一版weight2sample-step200。完整产物 `refs/data_selection/stack_bowls_two_failure_segments_far18_v2/`。
