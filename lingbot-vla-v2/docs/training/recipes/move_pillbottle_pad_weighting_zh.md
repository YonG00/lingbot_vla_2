# move_pillbottle_pad：1017来源差异化加权

来源为1017印章最终step200，训练200步，GBS56/GAS1/BF16、无复习、仅最终200保存，三回合并行带视频；无补课与关机。配方`configs/training_recipes/move_pillbottle_pad200_from_stamp1017_differential_noreplay.json`。运行目录`/data/outputs/20261024_1050_pillbottle200_from_stamp1017_s200_bf16_m56_grasp20_other5_release10_video20_noreplay_parallel3_video`，当前运行中。

可调权重入口`refs/data_selection/move_pillbottle_pad_differential_v1/category_loss_weights.json`。统一类别逐动作/各自视觉目标帧查表、weight_sum；FM仅活动手加权，另一手1，视觉取两手max，不改采样概率、不抽帧。

|类别组|FM/深度|视频|
|---|---:|---:|
|接近6–1.5cm、闭爪前半/后半、初始提离0–1.5cm（含闭爪后保持）|20|20|
|place/prealign_move、prealign_settle、contact_pose_settle|5|20|
|place/release_first_half、release_second_half|10|10|
|其他目标类别|5|5|
|background|1|1|

药瓶专家pre_grasp_dis=6cm、pre_dis=5cm，和印章不同，重新从药瓶关节轨迹判定。50条7345帧，活动手左24/右26，15细分类，目标4605帧62.70%。开始抓取距离按成功示范闭爪参考位姿12/9/6/1.5cm细分；闭爪开度0.5分前后，闭合<=0.2并提离3cm确认。初始0–1.5cm提离原747帧10.17%，按开始升高2mm拆为closed_hold_proxy625帧与lift_first_1p5cm122帧，两者均20，不改选帧；所有细类<=10%。

放置区间从提离后，TCP进入示范释放位置XY 3cm范围且高于释放参考3cm的首帧开始，到完全张开>=0.95。高度>4cm的对准按速度2.5cm/s拆移动/稳定，随后4–1.5cm下降与<=1.5cm末端位置稳定，松爪开度0.5分两半。几何/接触均为专家关节FK与夹爪命令代理，不宣称物体跟随或接触传感器结果；尤其closed_hold_proxy仅表示闭爪保持。

|细分类|帧数|FM/深度|视频|
|---|---:|---:|---:|
|grasp/approach_12_to_9cm|129|5|5|
|grasp/approach_9_to_6cm|424|5|5|
|grasp/approach_6_to_1p5cm|489|20|20|
|grasp/open_settle|362|5|5|
|grasp/closing_first_half|300|20|20|
|grasp/closing_second_half|250|20|20|
|grasp/closed_hold_proxy|625|20|20|
|grasp/lift_first_1p5cm|122|20|20|
|grasp/lift_1p5_to_3cm|145|5|5|
|place/prealign_move|345|5|20|
|place/prealign_settle|254|5|20|
|place/descent_4_to_1p5cm|177|5|5|
|place/contact_pose_settle|433|5|20|
|place/release_first_half|200|10|10|
|place/release_second_half|350|10|10|

验证：73450逐帧分支断言、3910原始14维chunk边界/offset/视觉/padding检查、完整CPU56形状[56,50,55]及986类别边界通过。专家校验从100000顺序检查，选择首三个有效种子100001/100002/100005，不依据模型结果选种子。记录与证据均在同类别目录。

来源1017已经训练200、闭环0/3，三录像非空且OIDN错误0，模型保留。用户明确批准删除0939旧全20版step200，实际释放11.90GiB，剩余17.67GiB，日志评测视频不删；删除证据在refs/training_evidence/20261024_pillbottle_from_1017。新轮只有计划200，未完成前不计累计或现有存档。
