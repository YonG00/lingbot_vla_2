# stamp_seal关键阶段20倍与串行接续

配方：`configs/training_recipes/stamp_seal200_from_can0915_key20_noreplay.json`。来源为当前0915罐子续训的最终step200，待来源生成及上一管线complete后才运行；不是直接从1913重训。训练200步、GBS56/GAS1/BF16、无复习、仅最终200存档，结束驻留三回合并行带视频，无补课、无关机。

权重入口：`refs/data_selection/stamp_seal_key20_v2/category_loss_weights.json`。15个子类别FM/视觉统一20，background=1；分类与参数分离，可分别改子类别参数并生成新快照，不热改旧运行。活动手逐动作FM，另一手1；共享Depth/Future Depth/Video按相应目标帧取两手max，padding排除，weight_sum。不改变样本采样概率，不抽帧。

50条7263帧、右手30/左手20。每条必须恰好一次开放>=0.95→闭合<=0.2→提离3cm→重新开放>=0.95，边界严格有序，否则拒绝生成。抓取接近起点是最后一次离闭爪参考位姿超过12cm之后；依距离12–9、9–6、6–1.5cm及末端稳定细分，随后闭爪前后半和提离0–1.5/1.5–3cm。放置重点从初始提离后，TCP位于释放参考位置XY 3cm内且高至少6cm的首帧开始，到完全释放；以离释放参考高度7.5/4/1.5cm、运动速度0.025m/s、夹爪开度0.5细分。参考位姿均来自成功专家轨迹，不是物体位姿或真实接触检测，contact_pose_settle仅为接触位姿代理。

|类别|帧数|占比|
|---|---:|---:|
|grasp/approach_12_to_9cm|680|9.36%|
|grasp/approach_9_to_6cm|161|2.22%|
|grasp/approach_6_to_1p5cm|204|2.81%|
|grasp/open_settle|342|4.71%|
|grasp/closing_first_half|300|4.13%|
|grasp/closing_second_half|250|3.44%|
|grasp/lift_first_1p5cm|701|9.65%|
|grasp/lift_1p5_to_3cm|130|1.79%|
|stamp/prealign_move|213|2.93%|
|stamp/prealign_settle|221|3.04%|
|stamp/descent_7p5_to_4cm|271|3.73%|
|stamp/descent_4_to_1p5cm|125|1.72%|
|stamp/contact_pose_settle|440|6.06%|
|stamp/release_first_half|200|2.75%|
|stamp/release_second_half|350|4.82%|

合计4588帧63.17%，每子类<=10%；其他2675帧背景。初稿v1的12–6cm类11.58%，已按9cm距离细分为v2，目标总量与倍率不变。v1仅服务器准备草稿，不作训练入口。

## 等待与执行

队列目录：`/data/experiments/stack_bowls_sample_weighting/stamp_seal_after_can0915_queue`；入口快照在同标注目录serial_successor.py。当前只等待0915管线complete且GPU无进程；前轮failed/停止、等候超过24小时、空闲不足16.06GiB均阻断，不停止前轮、不删除存档。

前轮结束后先核验来源1708 BF16张量，独立执行官方专家场景校验（不加载策略模型），从100000起最多20个候选中按顺序选3个有效种子，不按模型成功率选种子。将种子写入队列resolved_recipe.json，再启动原框架；原框架必须完成完整CPU56预检才训练，gpu_probe_enabled=false省略额外GPU三步探测。实际run自动加日期时间前缀，见队列launch.log，不能把队列启动当训练已启动。

当前纯CPU数据与类别边界/offset/未来视觉/padding共3906项通过，含3484个跨类别案例；这是14维原始动作检查。完整55维DataLoader、GPU训练、闭环尚未完成，待接续后核实，不计已完成训练或存档。当前训练过程中未额外import torch或占用GPU做测试。

## 当前差异化对照：1017

前述统一20配置保留为0939实验，已经完成200步及0/3三回合带视频评测；原等待队列已完成，不再运行。新配置为`refs/data_selection/stamp_seal_grasp20_other5_release10_video20_v1/category_loss_weights.json`，标签文件与v2逐字节相同，只改倍率，采样不变。

|类别组|FM|当前/未来深度|当前/未来视频|
|---|---:|---:|---:|
|grasp/approach_6_to_1p5cm、closing_first_half、closing_second_half、lift_first_1p5cm|20|20|20|
|stamp/prealign_move、prealign_settle、contact_pose_settle|5|5|20|
|stamp/release_first_half、release_second_half|10|10|10|
|其他已标注子阶段|5|5|5|
|background|1|1|1|

前三组缩写仅为表格阅读，实际JSON保留完整grasp/或stamp/前缀。video-only覆盖通过visual_branches.current_video/future_video实现，不连带改变深度。背景不属于15个目标子阶段，继续1。所有分支按各自权重和归一化。

配方`configs/training_recipes/stamp_seal200_from_can0915_differential_weights_noreplay.json`从与0939相同的0915-step200重新训200，GBS56、无复习、最终存档200；评测同种子100000/100001/100004，三回合并行带视频。72630个逐手逐帧分支权重断言、真实CPU56/1003类别边界及195回归通过；正式训练运行中，闭环待测。此配置不在0939权重上叠乘。

