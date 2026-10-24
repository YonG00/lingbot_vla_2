# 阶段三：训练流水

范围：针对难学任务筛选动作片段、调整逐样本损失及采样的独立实验。从20261022_1246夹碗实验开始；来源为阶段二第三轮补课step160。独立分叉不相加，计划/启动与完成严格区分。

[分类入口](../README.md) · [阶段二来源记录](../stage2/training_history_zh.md)

### 2026-10-22 夹碗样本加权独立实验（启动检查中）

来源：`20261022_1128_recovery_scout3_from_R17_replay10pct/20261022_1209_round_3_recovery/checkpoints/global_step_160`（不是180）。独立分支 `experiment/stack_bowls_sample_weighting`；独立服务器目录 `/data/experiments/stack_bowls_sample_weighting`。

run：`/data/outputs/20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample`。计划stack_bowls_two 200步，BF16/micro56/GAS1；10%单任务复习每5步轮换，沿用第三轮25任务池、排除当前主任务；500间隔＋最终保存不变，仅产生step200。命中样本的FM/Depth/Future Depth/Video乘2，再按原样本数（FM掩码按原有效元素数）归约，不除权重和；正则保持原值。离线50轨迹15609帧，7436帧命中（47.64%）。训后夹碗3回合、前2回合录像，驻留评测；无补课，无自动关机。

本地9项新CPU契约＋63项配方测试通过，服务器9项CPU契约通过，源码2640文件逐项SHA256校验。真实数据CPU装配检查进行中，GPU探针和正式训练尚待完成，不能记为已训200步。

启动确认：真实CPU1250轨迹250626帧通过，首批56样本中24个权重2；3步GPU探针通过，allocated77.82GiB/reserved86.20GiB，BF16参数/计算/梯度检查通过。正式已完成至少109/200步，近期4.792秒/步。原基线目录未修改；训练与闭环尚未完成。证据：refs/data_selection/stack_bowls200_started_evidence.json。

完成审计：200次正式更新完成，12:49:02–13:06:10约17分08秒，13:07:07完成训练收尾；step200的1708个BF16张量头部及分片完整性检查通过，约11.89GiB。首/末20步平均总loss0.036275→0.031415，FM0.03054→0.02602；仅为训练拟合。原始自动仿真跑3回合0/3，但OIDN CUDA不支持/invalid handle导致renderer gate拒绝，结果不作为有效clean成绩。独立目录漏带.runtime为已确认部署问题；不将pipeline failed写成训练未完成。

2026-10-22 18:33重新评测：用户取消分支同步，故未合并基线提交；仅在独立服务器目录补只读.runtime链接，显式指定已有OIDN库，并使用基线已验证仿真客户端作为评测依赖。OIDN CUDA预检通过。评测step200、stack_bowls_two、3回合全部录像、固定种子100000/100001/100002；新目录 `evaluation_200_renderfixed_videos3_20261022_183342/stack_bowls_two`，成绩待完成，无自动关机。

18:33新评测已完成：stack_bowls_two 0/3，种子100000/100001/100002，FP32推理/动作块50，OIDN错误0，3回合全局视频完整，已用ffprobe验证880/942/934帧，时长440/471/467秒。新结果有效，区别于原始渲染异常0/3。说明本次200步加权专训仍未通关，不能仅以训练loss下降判断学会；缺同源step160的配对有效成绩，不能量化相对改善。正式摘要 refs/data_selection/stack_bowls200_clean_eval_videos3.json。结果目录为本run下 evaluation_200_renderfixed_videos3_20261022_183342/stack_bowls_two。


### 2026-10-22 19:04 双规则并集权重3（启动检查中）

来源第三轮补课step160，非上一轮加权200。run `20261022_1904_bowls200_from_recovery3_s160_bf16_m56_replay10pct_grasp_place_weight3`；stack_bowls_two计划200步，10%复习，每5步轮换；BF16/micro56/GAS1。抓取与第二次放置规则并集11179帧（71.62%），命中样本四项损失3倍、按原样本分母；重叠不叠乘。用户新授权100/200存档。训完step200驻留3回合全录像，显式使用已验证仿真目录及OIDN库。CPU检查进行中，GPU和正式进度尚未确认；未自动关机。

用户启动中更正来源：以第三轮step160为起点的19:04流水线已停止，正式日志已确认更新0步（探针与正式启动不计成完成训练），不使用其模型。新起点改为上一轮 `20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample/checkpoints/global_step_200/hf_ckpt`，继续加训200步；双规则/3倍/10%复习/100和200保存不变。新run以from_weight2_s200命名，避免混淆继承关系。

新run：`/data/outputs/20261022_1909_bowls_add200_from_weight2_s200_bf16_m56_replay10pct_grasp_place_weight3`，来源明确为上一轮weight2sample-step200；真实CPU检查进行中。原19:04运行只记录为已停止，不计入已完成训练累计。

新来源启动取证：真实CPU通过，首批41/56样本权重3；GPU探针3更新通过，allocated77.71GiB/reserved85.72GiB。正式配置model_path确认为上一轮weight2sample-step200，最新已记录更新0/200。100/200保存计划已冻结，旧来源流水线不再运行。证据 refs/data_selection/grasp_place_weight3_from_weight2_s200_started_evidence.json。

正式更新确认：19:13:59已完成5/200，来源为weight2sample-step200；Loss0.0566、GradNorm0.7617，均有限。前几步仍含编译开销，不将其用作稳定训练速度。保存和闭环尚未完成。


### 2026-10-22 20:06 约30%阶段校准V3（启动检查中）

来源第三轮补课step160，stack_bowls_two计划200步，BF16/micro56/GAS1；10%复习每5步切换，100/200保存，最终3回合全部录像、驻留评测，无关机。run `20261022_2006_bowls200_from_recovery3_s160_bf16_m56_replay10pct_calibration30_weight3`。全部100次抓碗阶段均匀取点＋远定位＋第二放置高度带，合计4635帧29.69%，权重3按原样本分母，替换旧宽策略。100项本地及服务器CPU测试通过，真实装配通过，首批15/56加权；GPU探针正在运行，实际训练步数待确认。

V3正式启动取证：来源第三轮step160确认；冻结策略恰为4635帧29.69%，没有继承旧70%并集；CPU通过、GPU三步探针通过（allocated77.71GiB/reserved85.72GiB），正式更新已记录15/200。安全预算29.13GiB，100/200计划不变，最终三回合全录像待完成。启动证据 refs/data_selection/stack_bowls_two_calibration30_v3_started_evidence.json。

补记上一轮19:09宽规则实验终态：{"state": "complete", "evaluation": [{"state": "complete", "task": "stack_bowls_two", "label": "recipe_200", "model": "/root/autodl-tmp/data/outputs/20261022_1909_bowls_add200_from_weight2_s200_bf16_m56_replay10pct_grasp_place_weight3/checkpoints/global_step_200/hf_ckpt", "successes": 0, "episodes": 3, "success_rate": 0.0, "seeds": [100000, 100001, 100002], "oidn_errors": 0, "client_sha256": "fe0aaa791e881cb1839be533149c703c7d044b79550cad19309046ebff94eef3", "dtype": "float32", "action_chunk": 50, "recorded_episodes": 3, "server_pid": 6624, "shared_server": true}]}。其step100已按本次用户授权删除，最终step200保留；不与新V3运行混计。

### 2026-10-22 双重加权实验准备与上一轮结果

- `20261022_2006_bowls200_from_recovery3_s160_bf16_m56_replay10pct_calibration30_weight3`：实际完成200步；最终闭环0/3，种子100000/100001/100002，OIDN错误0，3回合视频完整。此结果不能证明成功率改善。
- 新run：`20261022_2045_bowls200_from_recovery3_s160_bf16_m56_replay10pct_sample3_loss3`，起点仍为 recovery3-step160，不继承上一轮；主任务stack_bowls_two，计划200步，GBS56/GAS1，10%单任务复习每5步轮换。
- 同一4635/15609（29.6944%）关键帧，任务内采样权重3，其余1；四项逐样本损失仍乘3，按原样本数归一化。预期主任务目标帧抽中率55.89%，不是保证每批固定比例；复习任务配额不变。用户明确改为仅保存step200，最终3回合闭环且全部留视频。
- 本地66项配方契约测试通过，服务器37项选择/损失测试及66项配方测试通过；流水线已启动，CPU真实数据及GPU验证状态另记，当前不计为完成训练。

- 双重加权实际启动核验：CPU真实数据预检通过，首批56样本中29个关键样本、loss权重3；GPU三步验证通过，峰值allocated77.71GiB/reserved85.72GiB。正式训练已确认42/200步，约4.8秒/步；实际日志确认4635个采样目标、采样倍率3，实际保存计划仅[200]。证据：`refs/data_selection/stack_bowls_two_sample3_loss3_started_evidence.json`。

### 2026-10-22 目标片段90%、远抓与放置1:2实验

上一轮 `20261022_2045_bowls200_from_recovery3_s160_bf16_m56_replay10pct_sample3_loss3` 实际完成200步，最终闭环0/3，OIDN错误0，三回合视频完整。

新run `20261022_2120_bowls200_from_recovery3_s160_bf16_m56_target90_far1_place2_loss2`：来源recovery3-step160；计划200步、GBS56/GAS1、BF16，仅最终200存档；90%目标片段内远左抓:第二放置=1:2，10%同任务其他帧，目标loss2倍/原样本归一化，无其他任务复习。实际GPU验证和训练状态待后续取证，不计为完成。

- 服务器真实数据CPU预检通过：50示范、15609帧，整轮精确远抓3360/放置6720/背景1120，首批50/56目标样本且loss权重2。GPU三步验证通过（allocated77.71GiB/reserved85.72GiB）；正式训练进程已进入200步循环，保存计划实际为[200]，当前尚未完成或测评。

- 正式训练已确认执行4/200步，loss和梯度日志正常；证据 `refs/data_selection/stack_bowls_two_target12_started_evidence.json`。训练与最终闭环尚未完成。

### 2026-10-22 训练来源纠正

用户明确本次起点为 `20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample/checkpoints/global_step_200/hf_ckpt`，而非 recovery3-step160。`20261022_2120_bowls200_from_recovery3_s160_bf16_m56_target90_far1_place2_loss2` 来源错误；实际已完成200步、最终0/3、OIDN错误0，不作为用户请求的续训结果，不计入正确来源模型继承链。原日志、视频和模型未删除。

正确配方 `configs/training_recipes/stack_bowls200_from_weight2_s200_target90_far1_place2_loss2.json` 已上传并重新启动流水线：来自第一版step200，再训200；目标池90%内远左抓:第二放置=1:2，同任务其他帧10%，目标loss2倍；GBS56/GAS1，仅保存最终200，最终3回合且全部录像。状态以新run实际记录为准。

- 正确起点新run：`20261022_2149_bowls200_from_weight2_s200_bf16_m56_target90_far1_place2_loss2`，实际plan.source_model已核验为第一版weight2sample-step200。真实数据CPU通过，配额3360/6720/1120，目标loss2；当前GPU三步预检运行中，正式完成状态待记录。

### 2026-10-22 用户将左远抓门槛改为18cm

25cm/6段版本 `20261022_2149_bowls200_from_weight2_s200_bf16_m56_target90_far1_place2_loss2` 在实际76/200步停止，进程及后代已终止，未完成最终测评；不将这76步并入新模型继承链。用户选择≥18cm，重新筛得14段左远抓1506帧，第二放置仍50段3743帧，联合去重5135/15609帧（32.90%）。

新配方 `configs/training_recipes/stack_bowls200_from_weight2_s200_far18cm_target90_loss2.json` 继续从第一版weight2sample-step200重新开始，不继承被停76步权重。30%左远抓/60%第二放置/10%同任务非目标帧，目标loss2倍；200步，GBS56/GAS1，只存最终200，最后三回合全录像。新流水线已启动，实际CPU/GPU/正式步数后续取证。

- 18cm新run `20261022_2200_bowls200_from_weight2_s200_bf16_m56_target90_far18cm1_place2_loss2`：真实CPU检查通过，日志确认14/50片段、远抓3360/放置6720/其余1120配额与目标loss2，source_model为第一版step200、hf_steps=[200]；三步GPU预检执行中，证据 `refs/data_selection/stack_bowls_two_far18_started_evidence.json`。

### 本次提交核验（2026-10-22）

18cm规则正式训练已确认164/200步，约4.8秒/步，GPU三步验证通过（allocated77.71GiB/reserved85.87GiB）。来源是第一版weight2sample-step200，目标14/50段，配额3360/6720/1120，loss2倍，只保存最终200；最终闭环尚未完成。本次重跑69项配方＋30项非Torch筛选/继承测试，合计99项通过；训练在跑时未执行Torch导入测试或额外GPU测试。重依赖loss和实际GPU证据来自各轮启动前检查，不能当作本次重新执行。

### 2026-10-22 18cm版本最终状态与退化分析

`20261022_2200_bowls200_from_weight2_s200_bf16_m56_target90_far18cm1_place2_loss2` 实际完成200步，最终HF已验证1708张量BF16；闭环3回合0/3，种子100000/100001/100002，OIDN错误0，三份视频完整，结果位于run下 `evaluation_200/stack_bowls_two`。来源确认第一版weight2sample-step200。用户观察抓取退化，尚无独立分阶段成功率统计。

训练loss前20均值0.04248→后20均值0.035895，FM0.03547→0.029295，梯度有限；不同采样/权重分布的loss不能横向作为成功率证据。第一版均匀主任务采样且7436抓取帧loss2；最新策略整任务替换后，原6293帧落入背景、loss变1，按抽样概率×loss倍率估算训练系数只剩原约8.3%，不是实测参数梯度。右手对准/下降/闭爪预计抽样1321→219次/11200总样本。

当前证据支持采样覆盖与旧重要性丢失是退化的重要嫌疑，但不证明唯一原因；14个左远抓事件与60%第二放置不能覆盖完整左右手通用抓取。episode起始TCP距离是难度代理，非碗沿/夹爪相对几何判定。下一步建议第一版step200保留作对照，完整任务覆盖为主，目标小比例增强并保留原抓取权重；尚未授权或启动新训练。证据 `refs/data_selection/stack_bowls_weight2_vs_far18_final_analysis.json`。

### 2026-10-22 18cm目标训练完成与并行评测验证

`20261022_2200_bowls200_from_weight2_s200_bf16_m56_target90_far18cm1_place2_loss2` 实际完成200步，最终BF16 HF1708张量验证通过；驻留串行闭环0/3，seeds100000/100001/100002，OIDN错误0，3回合视频完整。

用户批准单任务回合并行功能并要求本次验证不录像。基于同一个最终step200，额外启动3路单回合共享模型验证，输出 `episode_parallel3_novideo_20261022_224804`；不是新增训练。92项相关CPU测试通过，已见3路同时运行及零视频，最终汇总待完成。

- 单任务回合并行验证完成：三个种子100000/100001/100002，结果0/1、0/1、1/1，合计1/3；共享服务26162只加载一次、3连接、B1，真实重叠，OIDN错误0、无录像；峰值42.21GiB，总耗时174.71秒。与该模型此前驻留串行0/3不一致，不计为训练进步。相同HF重载路径/同种子/无录像串行对照 `episode_serial_novideo_control_20261022_225402` 仍运行，94项CPU通过后按用户要求先push。

### 2026-10-22 23:02 第一版抓取恢复1:5:4

用户批准第二种方案并使用新多回合并行评测。新run `20261022_2302_bowls200_from_weight2_s200_bf16_m56_mix154_left3_right2_place2_eval3parallel`；来源第一版weight2sample-step200，计划200步，GBS56/GAS1，非目标10%/完整抓取50%/第二放置40%；抓取内左右手1:1、左loss3右loss2、放置2、背景1。仅保存最终200，驻留模型三回合并行且全部录像，不关机。服务器72项配方、42项选择/损失、9项并行回合契约测试通过；真实CPU和GPU预检状态后续记录，不计为已完成。

- 本轮实际CPU通过：左右抓各2800/放置4480/背景1120；collator后首批权重3有14个、权重2有36个、权重1有6个。GPU三步验证通过；正式训练已启动，当前日志步数[]。并行job数3、固定种子和录像数量CPU校验通过，最终闭环待实际运行。证据 `refs/data_selection/stack_bowls_two_mix154_started_evidence.json`。

### 2026-10-22 1:5:4最终闭环与诊断边界

`20261022_2302_bowls200_from_weight2_s200_bf16_m56_mix154_left3_right2_place2_eval3parallel` 实际完成200步，最终BF16 HF保存在 `checkpoints/global_step_200/hf_ckpt`，三路共享驻留服务评测聚合完成，种子100000/100001/100002，0/3，OIDN错误0，全部三回合录像。结果位于 `evaluation_200/stack_bowls_two/summary.json`，分回合位于 `episodes/episode_000` 等。未关机。

CPU确认2800左抓/2800右抓/4480放置/1120背景、loss3/2/2/1正确。训练total前20均值0.04609→后20均值0.04176，FM0.038505→0.034405、梯度有限；权重分布已改变，不能与第一版loss横比。第一版7436抓取目标已全部恢复；本轮仍差，说明“旧抓取覆盖不足”不能独立解释全部退化。

第一版所谓抓碗改善来自用户视频观察，完整任务清洁重测同样0/3，尚无同条件分阶段抓取统计。另查已完成18cm模型不录像对照：同HF和相同三个场景种子，三路并行1/3、HF重新加载串行0/3，差异原因未确定，不能认为评测数值等价。当前未授权或启动新的GPU对照/训练。下一步建议固定串行链路做第一版与最新版本的成对抓取阶段比较，再判断续训步数/学习率/任务分布影响。证据 `refs/data_selection/stack_bowls_mix154_final_diagnosis.txt`。

### 2026-10-22 23:39 第一版原方法续训100步对照

用户将续训200改为100。run `20261022_2339_bowls_add100_from_weight2_s200_bf16_m56_original_recipe_replay10pct_serial3` 从第一版weight2sample-step200开始，恢复第一版全部train采样控制：均匀主任务、原7436帧loss2、10%其他任务单任务轮换复习每5步切换、GBS56/GAS1/BF16、seed3，关闭新角色/目标池/概率倍率参数。直接读取来源HF冻结的重要性配置并校验7436个坐标和唯一倍率2。与第一版实际recipe对照除预算100外train字段相同。

原峰值LR1e-4和下限5e-5不变，本轮cosine按100步预算衰减；HF续训不恢复优化器状态，不能表述为无缝恢复原optimizer。仅最终100存档，最终固定100000/100001/100002串行三回合且全录像，复用驻留模型，不关机。启动比较脚本首次因浅拷贝误改内存steps字段在GPU前拒绝，已修为深拷贝并通过不修改原配置测试，未修改原始文件/模型。实际CPU/GPU/正式步数后续记录。

- 原方法100步实际CPU预检通过，1250个示范/250626帧（含10%其他任务复习池），首批56样本中24个目标权重2，HF计划仅[100]；来源第一版step200已核验。当前GPU预检状态见 `refs/data_selection/stack_bowls_original100_started_evidence.json`，正式/闭环状态后续更新。

### 2026-10-23 原方法100步终态与增加放碗约10%续训

`20261022_2339_bowls_add100_from_weight2_s200_bf16_m56_original_recipe_replay10pct_serial3` 实际完成100步，来源第一版weight2sample-step200，最终串行3回合1/3，种子100000/100001/100002，OIDN错误0，3视频；最终step100保留。用户批准基于此step100再训100，并加入约10%放置样本2倍加权，评测改为最新单任务3回合并行。

新run `20261023_0003_bowls_add100_from_original_s100_bf16_m56_grasp2_place10pct2_replay10pct_eval3parallel`：原7436抓取帧全部保留，新增第二碗最终下降前0.33秒至松爪结束共1557帧（9.98%）；并集8993帧57.61%，全部loss2，其他loss1，按原样本分母。主任务均匀采样、10%其他任务复习、GBS56/GAS1/BF16及原LR控制不变；不是额外10%定额采样，不启用角色采样/概率倍率。只存最终100，最终3回合并行全录像、复用模型，不关机。

本地3项连续片段/边界契约通过，真实50条示范筛选与原目标超集检查通过；服务器整链CPU和GPU/正式训练状态另记，不把计划计为完成。

- 新增放置9.98%版本真实CPU预检通过：原抓取全部保留，8993帧统一2倍，首批30/56加权。来源2339-step100、保存计划[100]及并行3回合/录像配置均已核验。服务器45项选择/CPU损失与72项配方测试通过，本地114项不导入Torch的CPU测试通过；GPU和正式训练状态以 `refs/data_selection/stack_bowls_place10_started_evidence.json` 为准，不当作闭环完成。

### 2026-10-23 增加9.98%放置帧100步终态与耗时

20261023_0003 run实际100步完成、来源2339-step100；最终三路驻留评测0/3，种子100000/100001/100002，OIDN错误0，三视频完整，三个回合均900步失败。结果和模型保持原目录，本次未删存档或改参数。

评测总耗时350.735秒，三路启动差约4毫秒，总仿真窗口350.628秒，确认未串行。客户端循环累计口径326.5/282.0/279.6秒，客户端推理口径15.9/18.2/17.5秒；共享服务54次推理合计50.63秒、中位0.974秒。三视频1226/925/877帧（合计3028），2FPS，物理时长612.384/461.972/437.732秒；视频时间不是墙钟耗时。当前未分开计时物理步进、图像采集、录像及等待，不能声称录像单项解释全部增量。

找到的旧快速试跑为18cm三路无录像：仿真窗口116.597秒、总174.707秒（含加载），43次推理合计22.75秒、中位0.512秒；其中第三回合310步成功退出，其余900步。它与本次模型、录像及动作轨迹不同，不是严格性能对照。用户所述“两回合带视频约2分钟”的具体目录待确认；已有录两回合旧档案实际测三回合，未找到严格匹配记录。本次仅只读诊断，没有GPU测试或改动评测。证据 refs/data_selection/stack_bowls_parallel_timing_comparison.json 和 stack_bowls_place10_final_evidence.json。

### 2026-10-23 08:18 回到原抓取策略再训100步、不录像并行评测

用户指定来源仍为 `20261022_2339_bowls_add100_from_weight2_s200_bf16_m56_original_recipe_replay10pct_serial3/checkpoints/global_step_100/hf_ckpt`，不使用0003新增放置目标那轮。新run `20261023_0818_bowls_add100_from_original_s100_bf16_m56_original_grasp2_replay10pct_parallel3_novideo`。直接继承来源冻结的7436抓取坐标、统一2倍loss；不加放碗目标，不使用分层采样或采样倍率；放碗仍可作为正常loss1普通样本出现。均匀主任务、10%其他任务复习每5步换任务、GBS56/GAS1/BF16、原LR保持。

本轮计划100步，仅最终100，单任务3回合并行，不录像，种子100000/100001/100002，复用驻留模型、不关机。配方 `configs/training_recipes/stack_bowls_add100_from_original_s100_grasp2_parallel3_novideo.json`。CPU配方与拆分3个无视频job校验通过，服务器完整测试通过，真实数据/GPU/正式进度后续记录，暂不计为完成。启动前空闲约43GiB，安全预算16.06GiB，未删除产物。

- 本轮实际CPU整链通过，首批24/56目标权重2；冻结策略恰7436帧且SHA与来源原策略一致，三个回合视频数均0。GPU三步验证通过（allocated77.71GiB/reserved85.72GiB）；正式训练进程已启动，尚未记录完成步数，闭环待完成。证据 `refs/data_selection/stack_bowls_original_grasp100_parallel3_novideo_started.json`。

### 2026-10-23 0818原抓取100步最终结果与订书机准备

0818 run从2339-step100实际续训100步完成，最终step100核验通过；三路并行不录像1/3，种子100000/100001/100002，对应0/0/1，无OIDN错误，复用驻留服务，未关机。原7436抓取帧统一2倍策略与固定规则坐标指纹一致。结果 `evaluation_100/stack_bowls_two/summary.json`，终态证据 `refs/data_selection/stack_bowls_original_grasp100_parallel3_novideo_final.json`。

用户要求后续夹碗复习维持2倍，已增加固定登记与prepare自动冻结，来源策略缺失时也装入；明确冲突报错。不修改本轮冻结产物。当前训练完成后同步服务器并通过15项相关纯CPU契约及真实新任务复习链检查。

订书机只完成准备：50条7731帧，窄候选50段2282帧29.52%（左22/右28），抓后搬运宽参考3595帧46.50%。本地/服务器数值筛选一致，未做视频抽检、未启动训练。后续训练完成三回合带视频并行评测是用户确认的要求，但尚无订书机训练预算/配方。50帧窗口的末尾填充风险和代理判据边界见专用指南。证据 `refs/data_selection/move_stapler_pad_preparation_cpu_verified.json`。本次完整CPU回归53项选择/损失、72项配方、9项回合并行，共134项通过。

### 2026-10-23 0847 订书机宽片段200步（启动中）

来源0818-step100，主任务move_stapler_pad，200步，3595/7731目标帧loss×2、主任务均匀采样，10%轮换复习、夹碗固定7436×2。GBS56/GAS1/BF16；用户最新授权仅保存200。最终驻留三回合并行、全录像、不关机。run `20261023_0847_stapler200_from_bowls0818_s100_bf16_m56_postgrasp2_replay10pct_parallel3_video`。目前CPU整链预检进行中，尚不累计完成步数或成功率。启动前可用30.43GiB，最终一份安全预算16.06GiB，无删除产物。

- 启动检查更新：真实CPU1300条/258357帧通过，首批21/56命中2倍；50段连续坐标恰3595且覆盖到末帧。GPU三步通过，峰值allocated77.71GiB/reserved85.72GiB。流水线进入正式200步训练，尚未核验完成或闭环成功率。证据 `refs/data_selection/move_stapler_pad_broad200_started.json` 与 `move_stapler_pad_broad200_coordinate_contract.json`。

### 2026-10-23 09:08 用户指定删除三版存档（已完成）

- `/root/autodl-tmp/data/outputs/20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample/checkpoints/global_step_200`：已删除。
- `/root/autodl-tmp/data/outputs/20261022_2339_bowls_add100_from_weight2_s200_bf16_m56_original_recipe_replay10pct_serial3/checkpoints/global_step_100`：已删除。
- `/root/autodl-tmp/data/outputs/20261023_0003_bowls_add100_from_original_s100_bf16_m56_grasp2_place10pct2_replay10pct_eval3parallel/checkpoints/global_step_100`：已删除。

三份目录合计约35.68GiB；可用空间18.51→54.18GiB。仅删除上述存档，run目录、日志、评测和视频保留；0818来源及0847订书机run未改动。既往“保留”条目是历史状态，以本条后续清理事件为准。证据 `refs/data_selection/20261023_delete_1246_2339_0003_completed.json`。

### 2026-10-23 0847宽片段订书机训练终态与白色扩展准备

`20261023_0847_stapler200_from_bowls0818_s100_bf16_m56_postgrasp2_replay10pct_parallel3_video` 从0818-step100实际完成200更新；日志最后200/200为09:07:43，正式循环约17分14秒。主任务10080样本、复习1120。仅最终step200，1708个BF16张量核验通过。三回合并行带视频0/3，seeds100000/100001/100002，OIDN错误0，全部三回合录像、同一服务进程7237；未关机。结果目录为本run下 `evaluation_200/move_stapler_pad`，证据 `refs/data_selection/move_stapler_pad_broad200_final.json`。

随后按用户最新要求，保留3595宽目标，增加白色base3的9条抓取段772帧，重叠9帧去重，联合4358/7731（56.37%），loss2，非目标loss1。已核对真实示范外观与原始HDF5映射，129项CPU测试通过。此次仅准备样本，未启动新训练，不新增完成步数；不使用先前口头提及的窄方案。详见订书机筛选指南与 `refs/data_selection/move_stapler_pad_broad_white_grasp_v1/cpu_verified.json`。

### 2026-10-23 09:28 宽目标加白色抓取续训200（已启动）

run `20261023_0928_stapler_add200_from_0847_s200_bf16_m56_broad_whitegrasp2_replay10pct_parallel3_video`，来源 `20261023_0847_stapler200_from_bowls0818_s100_bf16_m56_postgrasp2_replay10pct_parallel3_video/checkpoints/global_step_200/hf_ckpt`。用户要求先启动再整理文档。目标4358/7731帧（原3595全保留，白色772中新增763、交界9去重），统一loss2、按样本数归约，主任务均匀、不加概率倍率；其余样本1。10%单任务轮换复习、每5步切换，夹碗固定原7436帧loss2。BF16/micro56/GAS1，200步，仅最终200存档，驻留三回合并行且全录像，无补课、不关机。

服务器129项CPU回归通过；真实1300轨迹258357帧整链通过，首批28/56样本权重2；GPU三步通过（allocated77.70GiB/reserved83.56GiB）。正式已出现更新日志，完成步数及闭环待终态核验，不能记为200步完成。启动前空闲约54GiB，单份存档预算16.06GiB，无清理。配方 `configs/training_recipes/move_stapler_pad_add200_from_0847_white_grasp2_parallel3_video.json`；证据 `refs/data_selection/move_stapler_pad_white200_started.json`。

### 2026-10-23 0928联合白色目标续训最终结果

`20261023_0928_stapler_add200_from_0847_s200_bf16_m56_broad_whitegrasp2_replay10pct_parallel3_video` 从0847-step200完成200正式更新；09:48:16最后200/200，正式循环17分15秒。主任务10080样本、复习1120；目标4358帧loss2。仅最终step200已核验1708个BF16张量。订书机三回合并行0/3，100000/100001/100002，OIDN错误0，三视频完整，同一驻留服务13720；无补课、未关机。结果本run下 `evaluation_200/move_stapler_pad`。证据 `refs/data_selection/move_stapler_pad_white200_final.json`。此前启动中条目由本终态补全。

用户随后指定用同一step200独立测move_can_pot，三回合并行全录像，独立加载一次共享FP32服务；并非继续训练驻留服务。结果目录 `evaluation_200_move_can_pot_videos3_20261023_100011`，当前评测中，不记成功率或新增训练步数。

### 2026-10-23 10:00 0928-step200独立测move_can_pot（完成）

三回合并行，种子100000/100001/100002，最终0/3（0/0/0）；同一FP32/B1共享服务15891，加载1次，耗时262.54秒（含服务启动），OIDN错误0、GPU进程已退出，未关机。三份1280×720全局视频经ffprobe核验，410/492/416帧，205/246/208秒，视频时间不是墙钟耗时。未训练此任务，正式新增步数0。

评测根目录：`/data/outputs/20261023_0928_stapler_add200_from_0847_s200_bf16_m56_broad_whitegrasp2_replay10pct_parallel3_video/evaluation_200_move_can_pot_videos3_20261023_100011`。摘要 `move_can_pot/summary.json`，录像 `move_can_pot/results/move_can_pot/episode0_global.mp4`、`episode1_global.mp4`、`episode2_global.mp4`。证据 `refs/data_selection/move_can_pot_0928_step200_video3_final.json`；此终态补全此前待测记录，不据0/3单独推断失败动作阶段。

### 2026-10-23 pick_dual_bottles水平双手目标准备（未训练）

用户强调瓶子立着、水平瞄准与伸长抓取。筛50条6129帧，左右各50段；保留伸手瞄准→水平前伸→闭爪→提起8cm，联合4382帧71.50%，无抽稀。50条原始HDF5逐帧身份一致，2150/2174/2199示范图抽检符合。选择器、参数和产物已准备，未设置启用的损失倍率、未启动训练，本任务新增步数0、无新闭环成绩。说明 `docs/training/recipes/pick_dual_bottles_target_selection_zh.md`，证据 `refs/data_selection/pick_dual_bottles_horizontal_grasp_v1/`。

### 2026-10-23 10:16 双手水平抓瓶目标2倍训练300（已启动）

run `20261023_1016_dual_bottles300_from_0928_s200_bf16_m56_horizontal_grasp2_replay10pct_parallel3_video`。严格从 `20261023_0928_stapler_add200_from_0847_s200_bf16_m56_broad_whitegrasp2_replay10pct_parallel3_video/checkpoints/global_step_200/hf_ckpt` 开始，不使用0818或0847。主任务pick_dual_bottles计划300更新、GBS56/GAS1/BF16；50条6129帧均匀抽样，4382目标帧的FM/Depth/FutureDepth/Video逐样本loss2，按原样本数归约；同帧两手标签不叠倍率，剩余1747帧loss1，无目标采样概率倍率。10%复习，每5步换单任务；继承订书机4358与固定夹碗7436的重要性，在对应复习任务出现时才生效。

本轮保持仅最终存档模式，保存step300；结束驻留模型三回合并行、全部录像，种子100000/100001/100002，无补课、不关机。启动前空闲约42GiB，单份预算16.06GiB，无删除。配方 `configs/training_recipes/pick_dual_bottles300_from_0928_s200_horizontal_grasp2_replay10pct_parallel3_video.json`。

135项CPU回归通过；真实1300条256755帧整链通过，首批35/56样本命中loss2；GPU三步通过，allocated77.70GiB/reserved83.56GiB，正式训练已出现更新日志。探针3步不计正式300；尚未记录完成步数或闭环成功率。启动证据 `refs/data_selection/pick_dual_bottles300_started.json`。

### 2026-10-23 11:02 分臂FM左3右2续训200（已启动）

来源 `20261023_1016_dual_bottles300_from_0928_s200_bf16_m56_horizontal_grasp2_replay10pct_parallel3_video/checkpoints/global_step_300/hf_ckpt`。1016原轮正式完成300更新，主任务15120样本、复习1680，最终step300为1708 BF16张量，双瓶三回合并行0/3、三视频、OIDN错误0；证据 `refs/data_selection/pick_dual_bottles1016_final.json`。

用户确认按动作维度修改，而非整帧全部3倍。新run `20261023_1102_dual_bottles_add200_from_1016_s300_bf16_m56_left3_right2_visual2_replay10pct_parallel3_video`，仍4382原目标：左臂六关节及左夹爪FM3，右臂六关节及右夹爪FM2；深度/未来深度/视频共享目标损失2。非目标动作1；左右目标时段边界只给正在目标中的手对应倍率，同帧不相乘。仍均匀主任务＋10%复习、每5步换任务，GBS56/GAS1/BF16，200步，仅最终200，结束驻留三回合并行全录像，100000/100001/100002，无补课、不关机。源300保留。

左臂动作布局0–5及28，右臂6–11及29；经真实变换mask校验14个有效维度，其他padding不参与。保留原有效元素分母，FM单独应用按维度修正；共享视觉与正则不跟随左臂增加。新政策可随HF冻结继承。

147项CPU回归通过；真实1300轨迹256755帧CPU整链，首批左FM集合1/3、右1/2、共享视觉1/2，动作权重56×55。GPU三步通过（allocated77.71GiB/reserved83.95GiB），正式训练已出现更新；不将探针步数计入200。此次改FM路径故显式开启探测，后续同计算配置可跳过。启动前空闲约30GiB，单份预算16.06GiB。终态、HF及闭环待核验。此前两次准备失败均发生CPU/路径检查，未产生GPU或正式更新，日志保留。

配方 `configs/training_recipes/pick_dual_bottles_add200_from_1016_s300_left3_right2_visual2.json`；权重 `refs/data_selection/pick_dual_bottles_left3_right2_v1/weight_config.json`；启动证据 `refs/data_selection/pick_dual_bottles_left3_right2_started.json`。

### 2026-10-23 1102分臂FM续训完成及三碗独立评测

`20261023_1102_dual_bottles_add200_from_1016_s300_bf16_m56_left3_right2_visual2_replay10pct_parallel3_video` 从1016-step300实际完成200更新，11:21:37最后200/200，正式循环约17分钟。主任务10080样本、复习1120；左臂含夹爪FM3、右臂含夹爪FM2，共享目标视觉2。最终step200已验证1708个BF16张量，仅最终200。原双瓶三回合并行0/3，100000/100001/100002，三视频，OIDN错误0，同一驻留服务32320，未关机。证据 `refs/data_selection/pick_dual_bottles_left3_right2_final.json`。

用户追加同一模型独立测stack_bowls_three，三回合并行全部录像，独立加载一次共享FP32服务。目录本run下 `evaluation_200_stack_bowls_three_videos3_20261023_113449`。当前待完成，不记成功率、新增训练步数0；不是再训三碗。

### 2026-10-23 1102分臂FM续训终态与失败观察

1102 run实际从1016-step300续训200完成，仅最终step200，1708个BF16张量核验通过。主任务10080样本、复习1120。最终pick_dual_bottles三路0/3、种子100000/100001/100002，OIDN错误0，3回合录像，使用驻留共享服务；无补课、不关机。结果本run下 `evaluation_200/pick_dual_bottles`。证据 `refs/data_selection/pick_dual_bottles_left3_right2_final.json`；此前启动中的200步由此更新为完成，探针不计入。

用户报告左手黑瓶始终未抓起。仅抽检首回合全局视频12帧：绿色瓶被碰倒后，手臂仍抬起并进入结束姿态，不能仅由此确认具体闭爪触发点或证明模型双臂容量不足。筛选是命令/运动学代理，未验证瓶身接触、实际夹持及随手提起。后续建议先做示范关键帧离线预测误差与执行50步对10步同种子对照，再决定继续训练方案；尚未执行这些诊断，不把推测当确定原因。抽帧图 `refs/data_selection/pick_dual_bottles_left3_right2_episode0_diagnosis.png`。

### 2026-10-23 双瓶全轨迹阶段方案准备（非训练轮次）

50示范6129帧完整双臂标注；每手17子阶段最大9.33%，FM及视觉按权重和归一化、最后接近/闭爪5。CPU174项和真实B2/50末尾检查通过，模型未初始化、训练0步、无新存档/闭环分数。完整说明：[阶段加权](../../recipes/full_trajectory_stage_multimodal_weighting_zh.md)。不得计入累计训练步数。

### 2026-10-23 18:02 双瓶全阶段5倍方案（已提交后台流水线，训练终态待核验）

run：`20261023_1802_dual_bottles200_from_0928_s200_bf16_m56_fullstage5_weightnorm_replay10pct_parallel3_video`，根目录 `/data/outputs/`。来源0928-step200，计划双瓶200步，BF16、micro56/GAS1，10%复习每5步换任务；完整17子阶段，接近/闭爪5、对准/提起2，按权重和归一化，动作padding不监督，继承夹碗与订书机重要性。仅最终step200存档；结束三回合并行带视频评测，不补课、不自动关机，按用户偏好跳过额外GPU三步探测。启动前97＋77项CPU通过，远端文件按SHA核验；实际训练步数、耗时、闭环成绩暂未确认，不计入累计完成。

用户批准清理的1016-step300、1102-step200已删除模型存档，日志/视频保留，实测释放23.79GiB。来源0928-step200保留。

### 2026-10-23 1802全阶段200完成与带视频重测终态

覆盖此前启动中记录：0928-step200→正式200步，训练17分01秒，末步Loss0.0330、VLA0.0282、Depth0.3724、FutureDepth0.4809、FutureVideo0.0533，末步4.764秒。最终step200已生成。CPU97＋77通过；真实1300轨迹与GBS56通过，新阶段与旧2倍重要性混合检查通过。GPU正式200步完成，不能仅由loss推断成功率。

初次resident评测使用实验目录的旧仿真客户端，接收录像参数但没有调用全局录像入口，评测后因missing or unexpected video异常，原流水线状态failed是评测验收失败，不能记作训练中断。保留该证据目录不覆盖。

随后使用已有录像功能的官方对照仿真客户端，对同一step200、种子100000/100001/100002三回合并行重测，单模型服务、FP32推理，最终0/3，OIDN错误0。路径：`/data/outputs/20261023_1802_dual_bottles200_from_0928_s200_bf16_m56_fullstage5_weightnorm_replay10pct_parallel3_video/evaluation_200_pick_dual_bottles_videos3_20261023_182812`。三份视频位于 `pick_dual_bottles/results/pick_dual_bottles/episode0_global.mp4` 至 `episode2_global.mp4`，ffprobe时长300.5/300.5/305.5秒、大小3391925/3638966/3415252字节，均可读取。仅重测，没有续训、额外存档或关机。此处通过换用已有客户端完成录像，不代表旧客户端问题已在训练框架中永久修复。

### 2026-10-23 18:38 双瓶完整阶段20倍对照启动

run `20261023_1838_dual_bottles200_from_0928_s200_bf16_m56_fullstage20_weightnorm_replay10pct_parallel3_video`。从同一0928-step200重新训练，非1802-step200续训；计划200步、micro56/GAS1、BF16、10%轮换复习、仅最终200存档、结束三路带视频评测，不补课/不关机。标签及采样不变，原5倍阶段的动作与视觉左右手全部改20，1/2不变，仍按权重和归一化。独立配置 `refs/data_selection/pick_dual_bottles_full_stage_weight20_v1/`；全部帧数值核对与13项CPU契约通过，旧代码按哈希一致核验。评测环境显式使用已验证全局录像入口的官方对照客户端；后台运行中，不能将计划200计作已完成。

### 2026-10-23 1838完整阶段20倍终态与5倍对照

来源0928-step200，实际完成200步，BF16、GBS56/GAS1、10%复习；1708个BF16张量最终存档验证完成。训练进度条17分44秒，末步Loss0.0324/VLA0.0277、4.743秒。端到端流水线complete，训练模型驻留直接三路评测，FP32推理、同种子100000/100001/100002、同录像客户端SHA fe0aaa791e881cb1839be533149c703c7d044b79550cad19309046ebff94eef3；0/3，OIDN错误0，三视频ffprobe核验304/303/305秒。没有重载模型补测，也没有自动关机。

完整对比报告位于 `refs/data_selection/pick_dual_bottles_full_stage_weight20_v1/completed_training_comparison.json`。最后151–200步按有效元素统计未加权FM：全体5倍0.021345→20倍0.021743（+1.86%）；接近/开爪稳定合并0.019817→0.020107（+1.46%）；闭爪0.029111→0.029160（+0.17%）；瞄准+4.38%、提起+4.45%、搬运+3.21%。这些为训练批次FM目标误差，不是固定验证集动作/末端误差，不代表统计显著差异。未单独记录forward_advance与open_settle，也未分左右手与夹爪维度。

重点接近＋闭爪的后50步FM损失贡献从57.22%升至83.88%，证明倍率生效；后50步梯度均值0.380→0.479（+26.1%），整轮裁剪3/200→5/200、后150步均无裁剪，不支持“持续裁剪吞掉20倍”的解释。模型自身误差随训练下降，但对5倍没有显示改善；相同0/3不能推断动作轨迹完全相同。暂不建议继续盲目加倍率，应优先用固定样本/噪声验证、分臂子阶段指标与视频抓取事件定位；这些额外测试尚未执行。

run `20261023_1838_dual_bottles200_from_0928_s200_bf16_m56_fullstage20_weightnorm_replay10pct_parallel3_video`，视频 `evaluation_200/pick_dual_bottles/results/pick_dual_bottles/episode{0,1,2}_global.mp4`。

### 2026-10-23 19:13 1838-step200无复习续训启动

run `20261023_1913_dual_bottles_add200_from_1838_s200_bf16_m56_fullstage20_weightnorm_noreplay_parallel3_video`。用户将此前关键片段增采样/细化监控方案暂缓，本轮训练代码未变；基于1838最终200的HF权重继续计划200步，新轮步号从0计，非恢复优化器。继承来源已冻结的20倍阶段权重与权重和归一化，micro56/GAS1，复习fraction0、tasks空，仅双瓶50条6129帧，计划11200主任务样本、0复习样本。最终200存档，结束驻留模型三回合并行带视频评测，不补课、不关机。磁盘空闲18.1GiB满足单份安全预算，未删除或降低存档节奏。正式终态待核验，不计入累计完成步数。

### 2026-10-23 1913无复习200完成；1940再续200启动

1913完整200步，训练16分50秒，末步Loss0.0382/VLA0.0335，step200核验1708 BF16张量。驻留模型三回合并行带视频评测0/3，种子100000/100001/100002、OIDN错误0、recorded_episodes3，流水线complete。实际双瓶11200样本、复习0，1838→1913的定向主链累计400步；不能由用户观察到局部改善将整任务改记成功。

用户授权继续：run `20261023_1940_dual_bottles_add200_from_1913_s200_bf16_m56_fullstage20_weightnorm_noreplay_parallel3_video`。来源1913-step200，计划再200、GBS56/GAS1，20倍权重和归一化保持、复习0、仅最终200、结束三回合并行带视频，不补课/关机。代码未改，采样增强和监控细化仍暂缓；CPU preflight正在执行，未把本轮计划计为已完成。

用户明确同意后，已删除1802五倍版本的 `checkpoints/global_step_200`，释放12768227328字节，删除后空闲19437617152字节；日志、评测视频、1838和1913均保留。新轮预算足够，未修改存档节奏。

### pick_diverse_bottles阶段数据准备（非训练轮次）

50示范6060帧、100抓取完整分段，独立20倍配置已准备；4＋13 CPU契约、原始映射与画面抽检通过。新增训练0步、无新Checkpoint、未评测；不能计入累计训练。见[说明](../../recipes/pick_diverse_bottles_full_stage_weighting_zh.md)。

### 2026-10-23 1940完成；2010闭爪保持/初始提离15倍续训启动

1940已完成正式200步，16分48秒，末步Loss0.0380/VLA0.0334，step200核验1708 BF16张量。驻留模型三回合并行带视频0/3，种子100000/100001/100002，OIDN错误0，3视频，流水线complete，11200双瓶样本、复习0。来源1838→1913→1940定向主链累计600步，不计独立5倍实验。

用户先提出4倍后立即改口15，4倍版本没有生成/启动。新run `20261023_2010_dual_bottles_add200_from_1940_s200_bf16_m56_stage20_holdclear15_noreplay_parallel3_video` 从1940-step200继续计划200。仅 `grasp_lift/closed_hold_proxy`、`grasp_lift/initial_clearance` 的左右手FM与视觉子阶段override由2改15；继续提起仍2，接近开爪稳定闭爪仍20，其余1，权重和归一化。独立快照 `refs/data_selection/pick_dual_bottles_full_stage20_lift15_v1/`，逐帧全分支差异核对通过，标签与采样不变。GBS56/GAS1、无复习、最终200存档、结束三回合并行录像、不补课不关机。CPU预检已通过并进入模型加载，终态待验，不提前计为完成。

先前清理后用户确认1940继续并保留新存档：当前保留recovery3-step160、1913-step200、1940-step200；本轮2010最终200获训练授权计划新增。本次无删除。

### 2026-10-23 2010完整结束；2037同配方再200启动

2010从1940-step200实际完成200步，16分49秒，末步Loss0.0353/VLA0.0307、Depth0.3559、FutureDepth0.4467、FutureVideo0.0442，4.739秒。最终step200核验1708 BF16张量，单模型驻留三回合并行带视频评测0/3，种子100000/100001/100002，OIDN错误0、3录像、pipeline complete，未关机。20倍主链累计800步，包含1838/1913/1940/2010各200，独立5倍实验不计。

用户授权继续：run `20261023_2037_dual_bottles_add200_from_2010_s200_bf16_m56_stage20_holdclear15_noreplay_parallel3_video`。从2010-step200的HF权重继续计划200，新轮计数从0、非恢复优化器。全部阶段权重从来源冻结配置继承：接近开爪稳定闭爪20、closed_hold_proxy/initial_clearance15，其余不变。只训双瓶50条6129帧、micro56/GAS1、无复习、最终200存档，结束驻留模型三回合并行带视频，不补课不关机。启动时空闲89.4GiB，无清理；未改训练代码、采样、监控或存档节奏。本轮终态待验，不提前计入累计。

### 2026-10-23 2037按用户要求停止；2041改10倍重训

2037最后日志Step25/200时用户要求停下。已先停止pipeline、torchrun及rank进程组并核对所有后代退出/GPU无占用；没有生成最终Checkpoint或闭环评测。日志保留、status记录stopped_by_user。实际25次更新发生在已丢弃运行中，不记200完成，也不加入2010继承模型的800步主链。

新run `20261023_2041_dual_bottles_add200_from_2010_s200_bf16_m56_stage20_holdclear10_noreplay_parallel3_video` 从相同2010-step200重新训练，非2037中断状态续训。仅左右FM/视觉 `grasp_lift/closed_hold_proxy` 和 `grasp_lift/initial_clearance` 从15改10，其他关键20、对准/继续提起2及其余1不变；权重和归一化、数据和采样不变。独立冻结快照 `refs/data_selection/pick_dual_bottles_full_stage20_lift10_v1/`，逐分支逐帧核对仅两子阶段变化。计划200、GBS56/GAS1、无复习、仅最终200，结束三回合并行带视频评测，不补课不关机，未删除旧存档。CPU预检后进入正式训练，终态待验。

### 2026-10-23 2041完成；2106同配方续训200启动

2041从2010-step200重新训练完成200步（非2037的25步续训），16分50秒，末步Loss0.0363/VLA0.0318、Depth0.3529、FutureDepth0.4427、FutureVideo0.0431，4.764秒；最终step200核验1708 BF16张量。单模型驻留三回合并行带视频评测0/3，种子100000/100001/100002，OIDN错误0，3录像，pipeline complete，未关机。继承主链累计1000步，2037中断25步不在保留模型链中。

用户授权继续：run `20261023_2106_dual_bottles_add200_from_2041_s200_bf16_m56_stage20_holdclear10_noreplay_parallel3_video`。来源2041-step200，新轮从0计数计划200，按原HF续训机制不恢复优化器。继承来源冻结权重：关键20、闭爪后保持/初始提离10，其余不变；仅双瓶50条6129帧、GBS56/GAS1、无复习、最终200存档、结束驻留模型三回合并行带视频，不补课不关机。新配方严格校验和有效代码哈希核对通过，真实CPU预检后启动，最终结果待验，不把计划200提前计入累计。磁盘空闲77.4GiB，无清理或存档节奏变更。

### 2026-10-23 用户更正来源：2106停止，2109从2010原权重续训

用户要求基于2010-step200，权重保持2010一致（关键20、closed_hold_proxy/initial_clearance15）。据此2106在最后日志Step6/200停止，pipeline/torchrun/rank及后代全部退出、GPU空闲；无最终存档/闭环，日志保留。6步为被丢弃实验，不计入本次保留模型继承链，不能记200完成。

新run `20261023_2109_dual_bottles_add200_from_2010_s200_bf16_m56_stage20_holdclear15_noreplay_parallel3_video`。直接核对2010冻结注册表中的左右FM/visual：三个关键子阶段20、闭爪后保持与初始提离15全部吻合。复用已校验原配方从相同2010-step200续训计划200，未指定新权重覆盖，全部重要性从来源继承；无复习、GBS56/GAS1、最终200保存、训完驻留模型三路录像评测，不补课不关机。CPU预检后正式启动，最终结果待验。新分支的实际来源主链为2010累计800，不继承2041完成的200，也不继承2037/2106中断更新；各历史实验保留流水不混入本分支累计。

### 订书机1913来源200步配方准备（未训练）

用户最终规则早期放置15/晚期20/白色其他10，数据50条7731帧，184项CPU及真实小批次四类检查通过、分界抽检完成。基于1913-step200、200步、无复习、最终存档与三回合录像已配置；当前无卡，训练未启动，无新增存档和累计步数。旧全放置20/白色15配置已被替代。见[说明](../../recipes/move_stapler_pad_sample_classes_zh.md)。

### 2109原20/15权重续训终态补齐

2109从2010-step200正式完成200步、16分54秒，末步Loss0.0350/VLA0.0305、4.742秒。最终step200核验1708 BF16张量；驻留三回合并行带视频0/3、OIDN错误0、3录像、pipeline complete，实际11200双瓶样本/复习0。来源主链800＋本轮200＝1000；不继承2041独立200或2037/2106丢弃步数。模型与录像保留，无关机操作。

### 统一类别／逐动作框架准备（未训练）

后续自定义加权统一逐动作；完整阶段与目标类别共用类别参数，订书机改为每个未来动作按早15/晚20/白色其他10查表，旧整chunk方案被替代。195 CPU回归、真实500个边界/时钟检查通过；完整DataLoader预检返回-9尚未完成，待开卡后通过再训。双瓶及多样瓶子逐帧迁移等价；无新增训练步数/存档。见[统一框架](../../recipes/category_weighting_framework_zh.md)。


2026-10-24补验：开卡主机上完整CPU DataLoader通过，50条7731帧、动作权重[2,50,55]、150个类别边界，weight_sum/action_timestep；模型未初始化、优化器0步。195项CPU回归重跑全绿。此前-9保留为历史失败，不再阻塞本轮启动。证据：refs/data_selection/move_stapler_pad_categories_early15_late20_white10_v1/full_cpu_verified_20261024.json。GPU训练与闭环尚待实际终态，不能计入完成步数。

### 20261024_0814 订书机统一逐动作200步（运行中）

来源1913-step200；run `/data/outputs/20261024_0814_stapler200_from_1913_s200_bf16_m56_early15_late20_white10_weightnorm_noreplay_parallel3_video`。50条7731帧，GBS56/GAS1/BF16，无复习，早期放置15/晚期20/白色其他10，逐动作查表、权重和归一化。正式CPU56与150边界通过，跳过GPU探测；已核实优化器step4/200，Loss0.0222，StepTime8.494秒。训练运行中，不计为已完成200；仅最终200存档，结束驻留三回合并行录像，评测待运行，无新增已完成存档。部署按文件哈希核对，未声称服务器Git HEAD对齐。

### 20261024_0814 订书机最终终态

1913-step200→200步已完成，训练16分47秒，最终Loss0.0312/VLA0.0269、4.743秒/步。1708 BF16张量存档核验通过；驻留模型三回合并行带视频0/3，OIDN错误0，3份录像及summary/status均complete。11200主任务样本，复习0。评测目录为该run下evaluation_200/move_stapler_pad；存档保留，无删除与关机。

### 20261024_0844 move_can_pot 接近抓取20倍（启动）

从1913-step200独立分叉，不继承0814订书机。run `/data/outputs/20261024_0844_can_pot200_from_1913_s200_bf16_m56_approach_grasp20_weightnorm_noreplay_parallel3_video`。50条7568帧，接近2694、抓取及初始3cm提离1401，共4095帧；活动手右29/左21条。逐动作/视觉20倍、其他1、weight_sum，不提高采样概率；GBS56、无复习、计划200步，仅最终200存档；驻留三回合并行带视频。真实CPU56、200个类别边界通过。运行中，不计完成200；闭环待测。

### 0844 move_can_pot最终训练与不完整评测

训练实际完成200步，16分45秒；最终Loss0.0471/VLA0.0434、4.725秒/步。step200存档核验1708 BF16张量，保留。评测100000为0/1、100002为1/1，均完成且录像各1；100001被官方专家有效性校验拒绝，管线failed。已完成回合合计1/2，不记作1/3或0/3，正式三回合结果缺失。训练成功与评测异常分开记录。

### 20261024_0915 move_can_pot续训200（运行中）

来源0844-step200，不回到1913；权重/数据/优化器配方沿用，GBS56/GAS1/BF16、接近抓取逐动作20、weight_sum、无复习，仅最终200保存，驻留三回合并行带视频。来源已核验1708 BF16张量，CPU56和200边界通过。用户批准替换无效评测种子，独立官方专家验证100003通过；本轮种子100000/100003/100002。run `/data/outputs/20261024_0915_can_pot_add200_from_0844_s200_bf16_m56_approach_grasp20_weightnorm_noreplay_parallel3_video`。计划再200，运行中不计完成；来源链本任务已有200，完成本轮后才记400。无删除、无关机。

### stamp_seal准备与串行排队（尚未训练）

等0915罐子训练评测complete后，从其最终step200接续印章200。50条7263帧、目标4588帧63.17%、15子类各<=10%，统一逐动作/视觉20倍、weight_sum。GBS56、无复习、仅最终200，三回合并行录像。纯CPU3906边界检查通过；完整DataLoader和GPU待前轮结束，当前状态waiting_for_predecessor。队列`/data/experiments/stack_bowls_sample_weighting/stamp_seal_after_can0915_queue`，前轮异常或容量不足会阻断；未增加已完成步数与Checkpoint。详见印章配方文档。

### 0915罐子与0939印章最终核验

0915罐子续训实际200步、16分45秒，最终Loss0.0451/VLA0.0415，1708 BF16张量、pipeline complete；三回合并行录像0/3，种子100000/100003/100002，OIDN错误0，三视频非空。该来源链move_can_pot累计400步，不把0844的1/2有效回合错写正式三回合。

0939印章全关键20版实际200步、16分50秒，最终Loss0.0387/VLA0.0344，1708 BF16张量，pipeline complete，三回合并行带视频0/3，种子100000/100001/100004，OIDN错误0。50条7263帧、11200主任务样本，无复习。模型保留；队列已完成接续，不再是等待状态。终态与实际视频大小证据：refs/training_evidence/20261024_stamp_differential/previous_runs_verified.json。

### 20261024_1017印章差异权重重训（运行中）

仍从0915罐子step200重新分叉，不继承0939印章；200步、GBS56/GAS1/BF16、无复习、仅最终200。四类抓取接近③/闭爪前半/闭爪后半/初始提离①动作深度视频20；其余目标阶段默认5，释放前后半10；对准移动/对准稳定/末端位置稳定仅视频覆盖20，动作深度5；背景1。逐动作、各视觉目标帧查表，weight_sum，分类和采样不变。真实CPU56、1003边界及195回归通过，GPU三步探测省略。run `/data/outputs/20261024_1017_stamp_seal200_from_can0915_s200_bf16_m56_grasp20_other5_release10_video20_noreplay_parallel3_video`。评测沿用0939同三种子并行录像，保持对照。当前运行中，不计完成200或新增存档；无删除/关机。
