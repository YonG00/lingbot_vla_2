# pick_dual_bottles：双手水平抓瓶目标样本

## 范围与动作依据

用户2026-10-23要求只筛选双手抓瓶目标，特别保留水平瞄准、伸长去抓。此轮不启动训练、不设置损失倍率、不改变固定夹碗规则。50条成功示范训练episode2150–2199，共6129帧、训练时间戳15FPS；原始episode0–49。全部50条HDF5 `state/joint_states` 与训练缓存逐帧完全一致，不只依赖序号偏移。

专家两手同时grasp_actor，预抓距离8cm；同时抬高10cm；之后才搬到任务目标姿态、保持夹持。两瓶原位直立，夹爪水平侧抓。末端夹取方向与向下夹角约83.79–90.51°；接近前20帧到闭爪位置高度变化约−0.72～8.65mm。因此不用抓碗的朝下30°或下降5cm条件。

## 判定标准

每条左右手分别判定，并要求两手都通过才保留该示范的目标：

1. 恰一次闭爪，张开≥0.95、闭合≤0.2；闭爪期间末端漂移≤1.5cm。
2. 找闭爪前最后稳定预抓姿态：沿抓取轴距示范闭爪TCP6–10cm，横偏≤2.5cm，方向偏离水平≤15°，速度≤3cm/s连续至少2帧。
3. 水平前伸接近示范闭爪TCP，进入1.5cm范围后才闭爪；前进比例≥70%，横偏≤2.5cm，接近/闭爪仍保持水平。没有垂直下降要求。
4. 从预抓对准向前最多4秒找伸手瞄准起点：保持开爪，相对预抓姿态至少缩短2cm距离，反向超过2mm的帧占比≤30%。保留此前伸手定位，不只最后8cm。
5. 闭爪后3秒内末端向上抬升≥8cm且不重新张开，取首次达标为终点；后续搬到最终目标不纳入本目标。

范围连续包含 `reach_aim → horizontal_align → forward_approach → settle → close → lift`，不抽稀。闭爪TCP及预抓姿态都是成功示范的运动学参考，不是瓶子实时位姿/接触信号；高度确认也不是独立物体随动检测。

## 结果与抽检

|统计|结果|
|---|---:|
|示范|50/50两手均通过|
|左右手抓取段|50＋50|
|总帧|6129|
|去重目标帧|4382（71.50%）|
|同帧两手均在目标内|4374|
|仅一手还在对应阶段|8|

比例较高来自完整伸手到提瓶范围，本次没有按比例截断。左右手同帧合并成一个样本并保留两手阶段标签，不能因两手标签而重复计数或叠加损失。不同动作阶段的arm计数可重叠，不能当作去重帧数相加。

已人工查看episode2150、2174、2199四阶段对照图：起始两瓶站立、两手伸长到瓶侧、闭爪、两瓶提起，符合水平抓瓶语义。原始视频30FPS，训练时间戳15FPS，且原视频比轨迹多1帧；抽检按原始帧索引，不按训练timestamp直接seek，只作定性核对，不宣称亚帧同步。实际边界由原始HDF5与训练状态精确匹配后计算。抽检图 [visual_spotcheck.png](../../../refs/data_selection/pick_dual_bottles_horizontal_grasp_v1/visual_spotcheck.png)。

## 使用与产物

入口 `tools/select_dual_bottle_grasp.py`；参数配置 `configs/data_selection/pick_dual_bottles_horizontal_grasp_v1.json`。示例参数 `--task pick_dual_bottles --dataset-root <真实数据根目录> --settings configs/data_selection/pick_dual_bottles_horizontal_grasp_v1.json --output-dir <新目录>`。支持缓存模式 `--frames-json <数值缓存> --fps 15`。不指定dataset-name时仅输出筛选结果，不生成启用的权重配置；倍率与训练安排等用户决定。

`refs/data_selection/pick_dual_bottles_horizontal_grasp_v1/` 包含summary、100个分手events、稳定episode/frame坐标selected_frames、cpu_verified及抽检图。服务器有效产物为 `/data/experiments/stack_bowls_sample_weighting/selection_pick_dual_bottles_horizontal_v1_verified`。首次非verified目录曾因NumPy整数JSON序列化错误只留下部分输出，不作为可用选择；已修正并补JSON序列化CPU契约测试，未动源数据。

6项新CPU测试覆盖水平接近无垂直下降、向下误判、无提起、另一手遗漏、两手去重及无效输入。50条身份匹配、4382连续坐标精确合并验证通过；没有GPU训练或新的闭环成绩。

### 2026-10-23 10:16 双手水平抓瓶目标2倍训练300（已启动）

run `20261023_1016_dual_bottles300_from_0928_s200_bf16_m56_horizontal_grasp2_replay10pct_parallel3_video`。严格从 `20261023_0928_stapler_add200_from_0847_s200_bf16_m56_broad_whitegrasp2_replay10pct_parallel3_video/checkpoints/global_step_200/hf_ckpt` 开始，不使用0818或0847。主任务pick_dual_bottles计划300更新、GBS56/GAS1/BF16；50条6129帧均匀抽样，4382目标帧的FM/Depth/FutureDepth/Video逐样本loss2，按原样本数归约；同帧两手标签不叠倍率，剩余1747帧loss1，无目标采样概率倍率。10%复习，每5步换单任务；继承订书机4358与固定夹碗7436的重要性，在对应复习任务出现时才生效。

本轮保持仅最终存档模式，保存step300；结束驻留模型三回合并行、全部录像，种子100000/100001/100002，无补课、不关机。启动前空闲约42GiB，单份预算16.06GiB，无删除。配方 `configs/training_recipes/pick_dual_bottles300_from_0928_s200_horizontal_grasp2_replay10pct_parallel3_video.json`。

135项CPU回归通过；真实1300条256755帧整链通过，首批35/56样本命中loss2；GPU三步通过，allocated77.70GiB/reserved83.56GiB，正式训练已出现更新日志。探针3步不计正式300；尚未记录完成步数或闭环成功率。启动证据 `refs/data_selection/pick_dual_bottles300_started.json`。
