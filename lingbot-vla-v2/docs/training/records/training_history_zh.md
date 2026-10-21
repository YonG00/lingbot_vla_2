# 训练与闭环评测流水

记录范围：2026-10-20官方对照实验至当前补课；更早鼠标实验尚未回填，不表示没有训练。
最近取证：2026-10-21，R12正式1350步、三份存档及最终30回合闭环均完成，各时间节点采用服务器日志时间。
时间为首次/末次正式优化器更新日志时间，不含CPU检查、GPU探针、加载与闭环耗时。
运行中记录仅表示该时间点进度。任务累计与各轮对照见[任务统计](task_training_statistics_zh.md)。

| 编号 | 时间节点 | 基于模型 | 学习任务与方式 | 正式训练步数 / GBS | 更新耗时 | 后续闭环结果 | 补课与去向 |
|---|---|---|---|---|---|---|---|
| R1 | 10-20 12:19:59–12:57:37 | 原始9680 | adjust_bottle，标准方法，FP32 | 500 / 28 | 37分38秒 | step250：2/3；step500：2/3 | 未从本模型续训；250/500 HF已授权删除 |
| R2 | 10-20 13:19:28–13:58:55 | 原始9680 | adjust_bottle，EE＋窗口＋LN-FM，FP32 | 500 / 28 | 39分27秒 | step250：1/3；step500：2/3 | 独立对照组，非R1续训；250/500 HF已删除 |
| R3 | 10-20 14:40:30–14:57:56 | 原始9680 | adjust_bottle，标准方法，BF16 | 200 / 56 | 17分26秒 | step200：3/3 | 选为后续基准；分叉出R4与R5 |
| R4 | 10-20 15:31:04–16:05:55 | R3-step200 | plate/cup/pot三任务样本均衡混训 | 400 / 56 | 34分51秒 | plate 0/3、cup 0/3；后来指定评adjust_bottle 2/3、click_bell 3/3；pot中断，无有效完整结果 | 训练完成，评测队列被用户缩减；未作为R5来源；step400 HF已删除 |
| R5 | 10-20 16:43:45–17:00:19 | R3-step200 | 仅place_container_plate，BF16 | 200 / 56 | 16分34秒 | plate 1/3 | 作为R6来源 |
| R6 | 10-20 17:33:16–18:15:31 | R5-step200 | 顺序：plate100→cup200→pot200，BF16 | 500 / 56 | 42分15秒 | plate 2/3、cup 1/3、pot 0/3 | 根据规则生成R7补课；step250/500 HF已校验 |
| R7 补课1 | 10-20 18:36:14启动；19:01:41完成 | R6-step500（最新模型） | 顺序：cup100→pot200，BF16 | **已完成300/300** / 56 | 25分27秒 | step300：plate 0/3、cup 0/3、pot 0/3，OIDN错误0 | cup100/100、pot200/200完成；250/300已保存；用户放弃本模型，不作为后续来源 |

plate=`place_container_plate`，cup=`place_empty_cup`，pot=`move_can_pot`。
各run均从注明的HF新建优化器和学习率日程；顺序课程内部阶段不重置优化器。
闭环使用FP32加载/推理；R3及以后训练与HF为BF16，不能把推理精度当训练精度。
3回合成功数仅是本次小样本结果，各任务种子不完全相同，不代表广泛场景成功率。
R7启动后用户授权：以后300步补课仅最终300保存；当前进程不热改，保留250/300以免重启丢进度。

### R7评测复核

**R7编号对应的完整run目录**：

```text
/data/outputs/sequential_plate100_cup200_pot200_from_plate200_bf16_m56_20261020/remediation_round_1
```

这是R6目录下的补课子run，不是R6本身的`evaluation_500`。服务器`/data`解析为
`/root/autodl-tmp/data`，报告里出现后者是同一位置。
以下路径均相对于上述R7 run目录：

| 内容 | 路径 |
|---|---|
| 来源模型 | `../checkpoints/global_step_500/hf_ckpt`（R6-step500） |
| 本轮课程与训练配置 | `recipe.json`、`trainer_config.json` |
| 训练日志 | `train.log` |
| 250步中间存档 | `checkpoints/global_step_250/hf_ckpt` |
| 最终300步存档 | `checkpoints/global_step_300/hf_ckpt` |
| **闭环测评汇总** | **`evaluation_300/summary.json`** |
| 共享服务与模型路径 | `evaluation_300/session.json`、`evaluation_300/server.log` |
| 逐任务模型参数与结果 | `evaluation_300/<task>/job.json`、`evaluation_300/<task>/summary.json` |
| 逐回合模拟日志 | `evaluation_300/<task>/sim.log` |
| 结果文本与回放视频 | `evaluation_300/<task>/results/<task>/_result.txt`、`episode*_global.mp4`（同目录） |
| 后台控制器状态 | 同级目录`remediation_round_1_pipeline/status.json`（在R6目录下） |

`<task>`分别为`place_container_plate`、`place_empty_cup`、`move_can_pot`。
R7按当时“复测原三任务”的规则运行；本轮实际只训练cup/pot，plate是保留能力检查。
后续默认改为“本轮训什么就测什么”，不会追改这份历史结果。

用户质疑后再次只读核对：训练来源为R6-step500，各任务job的评测模型为R7-step300。
以下为日志中的累计成功数；与各任务`_result.txt`最后一行0.0和summary.successes=0一致。

| 任务 | 第1回合后 | 第2回合后 | 第3回合后 | 实际场景种子 |
|---|---|---|---|---|
| place_container_plate | 0/1 | 0/2 | 0/3 | 100000、100003、100004 |
| place_empty_cup | 0/1 | 0/2 | 0/3 | 100000、100001、100002 |
| move_can_pot | 0/1 | 0/2 | 0/3 | 100000、100001、100002 |

[逐项取证](../../../refs/training_records/20261020_r7_evaluation_audit.json)保留job、日志匹配、结果文本及summary。
该核对确认表格与原始记录一致，不代表已排除评测实现问题，也不解释性能下降原因；未重跑评测。
R7的19:01:41是最后训练更新，不是闭环完成时间，闭环结果文件记录晚于该时间。

## 原始记录索引

### R8 新课程（600步及闭环均完成）

正式更新日志：2026-10-20 19:34:54–20:27:56，耗时53分02秒；600步完成，三个任务各200步。
最终600闭环：turn_switch 1/3、move_pillbottle_pad 0/3、move_stapler_pad 0/3，OIDN错误0。
200/400/600三份曾完成校验；用户随后只保留最终600，作为R9来源。

| 来源模型 | 顺序课程 | 总步数 / GBS | 存档 | 评测 | 补课 |
|---|---|---|---|---|---|
| **R6-step500**，不继承R7 | turn_switch200→move_pillbottle_pad200→move_stapler_pad200 | **600/600已完成** / 56 | 任务模式200/400/600；现仅保留600 | switch 1/3、pillbottle 0/3、stapler 0/3 | 未执行自动补课 |

输出：`/data/outputs/sequential_switch_pillbottle_stapler_from_R6_bf16_m56_200each_20261020`。
BF16参数/计算/HF，150条19867帧；新优化器，阶段间连续。旧存档全部保留。
后续补课策略已修订为：2/3优先补100步，其次1/3补100步；0/3与3/3跳过，最多一轮。
这是未来新启动补课的规则，不修改历史R7，也不自动开启本轮R8补课。

### R9：从R8-step600继续500步

| 来源模型 | 顺序课程 | 本轮总步数 / GBS | 存档 | 评测 | 状态 |
|---|---|---|---|---|---|
| R8-step600 | move_stapler_pad200→move_pillbottle_pad200→turn_switch100 | **500/500完成** / 56 | 任务模式200/400/500，均已校验 | stapler 0/3、pillbottle 0/3、switch 2/3，OIDN错误0 | 2026-10-20 21:45:38训练结束，闭环complete，补课未执行 |

完整run路径：

```text
/data/outputs/20261020_2058_stapler200_pillbottle200_switch100_from_R8_bf16_m56
```

来源：`/data/outputs/sequential_switch_pillbottle_stapler_from_R6_bf16_m56_200each_20261020/checkpoints/global_step_600/hf_ckpt`。
配置：`configs/training_recipes/sequential_stapler200_pillbottle200_switch100_from_R8.json`。
BF16参数/计算/HF、micro=GBS56、GAS1；新优化器和500步学习率日程，课程内部不重置。
补课继续关闭，未关机。最终存档`checkpoints/global_step_500/hf_ckpt`，闭环汇总`evaluation_500/summary.json`。
停训指令曾收到但未确认执行成功；复核时本轮已经自行完成训练及评测，无训练进程在跑，不能记为人工停止成功。
[R8终态与R9启动快照](../../../refs/training_records/20261020_R8_complete_R9_started.json)。

### R10：R9最终500续训四任务

run：`/data/outputs/20261020_2247_playingcard_phone_scale_qrcode_from_R9_bf16_m56_200each`。
来源：R9的`checkpoints/global_step_500/hf_ckpt`，不继承被替换三任务流程的任何更新。
课程：move_playingcard_away200→place_phone_stand200→place_object_scale200→rotate_qrcode200，共800步。
200条27225帧，BF16/GBS56/GAS1，按任务边界200/400/600/800保存；最终仅四个训练任务各测3回合，共享服务。
补课关闭，未执行自动补课或关机。四任务各200步、共800步均已完成。
正式更新日志：2026-10-20 22:49:54–23:59:43，800次更新耗时69分49秒；
训练结束日志为2026-10-21 00:00:44（包含最终保存等收尾），两种时间口径不混用。

| 任务 | 本轮训练步数 | step800闭环 | 实际种子 | OIDN错误 |
|---|---:|---|---|---:|
| move_playingcard_away | 200 | **1/3** | 100000、100001、100002 | 0 |
| place_phone_stand | 200 | 0/3 | 100000、100001、100002 | 0 |
| place_object_scale | 200 | 0/3 | 100000、100002、100003 | 0 |
| rotate_qrcode | 200 | **1/3** | 100000、100001、100002 | 0 |

200/400/600/800四份HF实际文件头与索引逐项一致，均1708张量、BF16。
四项summary、逐回合日志及_result.txt一致；一次共享推理服务，server_pid=111709。
评测模型均为本run的`checkpoints/global_step_800/hf_ckpt`，汇总`evaluation_800/summary.json`。
[完整训练与闭环快照](../../../refs/training_records/20261021_R10_completed.json)；
[实际HF、逐回合评测与存档清单复核](../../../refs/training_records/20261021_R10_final_audit.json)。
playingcard和qrcode达到复习资格≥1/3，已追加至新版累计池；本轮训练本身复习比例仍为0。
刚启动的`20261020_2244_playingcard_phone_scale_from_R9_bf16_m56_200each`因用户追加rotate_qrcode已停止控制器及所有子进程，
不是另一个已完成实验，不能将其更新累计到R10；它不作为新训练模型来源。

### R11：R10-step800续训七任务＋10%轮换复习

run：`/data/outputs/20261021_0831_qrcode_card_bottles_roller_pot_stamp_from_R10_bf16_m56_replay10pct`。
来源为R10的`checkpoints/global_step_800/hf_ckpt`。
用户确认全部五个新任务，共七个课程任务，总计1200步：

| 顺序 | 主任务 | 本轮步数 | 全局步号 | 保存 |
|---|---|---:|---|---|
| 1 | rotate_qrcode | 100 | 1–100 | 无 |
| 2 | move_playingcard_away | 100 | 101–200 | 无 |
| 3 | pick_dual_bottles | 200 | 201–400 | 400 |
| 4 | pick_diverse_bottles | 200 | 401–600 | 无 |
| 5 | grab_roller | 200 | 601–800 | 无 |
| 6 | lift_pot | 200 | 801–1000 | 1000 |
| 7 | stamp_seal | 200 | 1001–1200 | 最终1200 |

BF16参数/计算/HF，micro=GBS56/GAS1；新优化器、1200步日程，阶段间连续。
每步10%样本复习，GBS56每步5/6条，连续10步同一个旧任务再轮换；排除当前主任务。
冻结14任务资格池，主任务与复习合并共19任务、950条、159239帧。
全程实际67200样本，其中复习6720；复习曝光不算额外完整优化器步。
保存模式task/every_tasks=3；最终只测七个主课程任务各3回合，共享服务；补课关闭、不自动关机。
44项本地与服务器CPU契约通过，真实950条数据CPU装配及三步GPU通过。
三步micro56/GAS1峰值allocated77.70GiB、reserved84.59GiB；短测仅首个qrcode主任务与复习任务混合，不代表后续全部阶段已验。
正式1200步已完成：首次更新2026-10-21 08:35:08，末次10:14:32，更新耗时99分24秒。
400/1000/1200三份HF均验证1708张量、BF16，最终七任务闭环完成；共享服务只加载一次，OIDN错误均0。

| 任务 | step1200闭环 |
| --- | --- |
| rotate_qrcode | 2/3 |
| move_playingcard_away | 3/3 |
| pick_dual_bottles | 0/3 |
| pick_diverse_bottles | 0/3 |
| grab_roller | 0/3 |
| lift_pot | 0/3 |
| stamp_seal | 0/3 |

汇总在本run的`evaluation_1200/summary.json`；[完成证据](../../../refs/training_records/20261021_R11_completed.json)。
R11结果已并入`refs/training_records/replay_pool_9680_through_R11.json`，历史合格池仍14任务。
配置：`configs/training_recipes/sequential_seven_tasks_1200_replay10pct_from_R10.json`。

### R12：R11-step1200续训十任务＋10%复习（训练与闭环完成）

run：`/data/outputs/20261021_1129_ten_tasks_from_R11_bf16_m56_replay10pct`。
来源R11最终1200的BF16 HF，新优化器与1350步日程；BF16参数/计算/HF，micro=GBS56/GAS1。
继承标准方法，EE/窗口加权/LN-FM关闭；复习池为合入R11结果的14任务累计池，10%样本、每10步轮换，排除当前主任务。

| 顺序 | 主任务 | 配额 | 本轮步号 |
| --- | --- | ---: | --- |
| 1 | shake_bottle_horizontally | 150 | 1–150 |
| 2 | pick_dual_bottles | 100 | 151–250 |
| 3 | pick_diverse_bottles | 100 | 251–350 |
| 4 | put_bottles_dustbin | 150 | 351–500 |
| 5 | dump_bin_bigbin | 150 | 501–650 |
| 6 | place_container_plate | 100 | 651–750 |
| 7 | stack_bowls_two | 150 | 751–900 |
| 8 | stack_bowls_three | 150 | 901–1050 |
| 9 | blocks_ranking_size | 150 | 1051–1200 |
| 10 | stack_blocks_two | 150 | 1201–1350 |

保存500/1000/最终1350，容量预算42.19GiB，启动前空闲79.73GiB；未删除旧存档。
最终1350模型测上述10任务，各3回合；同一FP32/B1模型分批6＋4，不在前批未结束时补入后批任务。
无自动补课或关机。49项本地/服务器CPU契约通过；真实1100条、262964帧CPU检查及三步GPU探针通过。
探针峰值allocated77.71GiB/reserved84.19GiB；[启动证据](../../../refs/training_records/20261021_R12_started.json)。
正式1350步已完成：首次更新2026-10-21 11:33:02，末次13:24:15，更新耗时1小时51分13秒，不含存档与评测。
500/1000/1350三份HF文件头和索引均验证1708张量、BF16。全程75600样本，其中复习7560。
最终10任务各3回合已完成，合计8/30成功；每任务仅3回合，不视为稳定总体成功率。

| 任务 | 最终1350模型闭环 |
| --- | --- |
| shake_bottle_horizontally | 3/3 |
| pick_dual_bottles | 0/3 |
| pick_diverse_bottles | 0/3 |
| put_bottles_dustbin | 0/3 |
| dump_bin_bigbin | 2/3 |
| place_container_plate | 0/3 |
| stack_bowls_two | 1/3 |
| stack_bowls_three | 0/3 |
| blocks_ranking_size | 0/3 |
| stack_blocks_two | 2/3 |

全部逐回合日志、summary及30份非空视频一致，OIDN错误及仿真Traceback均0；共享服务PID26532，model_loads=1、连接数10、445次请求，FP32/B1。
第一批六任务最后完成13:51:58.673867，第二批四任务最早创建13:51:58.681866，批次屏障真实成立，模型不重载。
仿真阶段总耗时48分05秒，含模型启动约49分05秒；没有同任务同配置串行对照，不据此宣称加速倍数。
汇总路径：本run的`evaluation_1350/summary.json`；外层`<run>_pipeline/status.json`与评测status均为complete，推理服务已退出，GPU空闲。
[完整训练、HF、批次时间与逐回合核验](../../../refs/training_records/20261021_R12_completed.json)。
累计复习资格池已更新为`refs/training_records/replay_pool_9680_through_R12.json`，14→17，新增dump_bin_bigbin、stack_bowls_two、stack_blocks_two。
历史合格任务不因本轮0/3移除；本轮训练冻结的14任务池不追改。未自动补课、续训、清理或关机。
瓶抓取两任务从R11的0/3加训100步后仍0/3，当前没有改善证据；有限3回合不足以推断整体成功率。
配置：`configs/training_recipes/sequential_ten_tasks_1350_replay10pct_from_R11.json`。

### 历史存档保留清单（清理时口径）

按用户明确指令，启动R9前仅保留以下两份历史模型：

- R6：`sequential_plate100_cup200_pot200_from_plate200_bf16_m56_20261020/checkpoints/global_step_500`。
- R8：`sequential_switch_pillbottle_stapler_from_R6_bf16_m56_200each_20261020/checkpoints/global_step_600`。

两份均验证1708张量、BF16。删除其他9份global_step存档（包括9680、R3基准、R5、R7与R8中间存档），
释放107.05GiB，清理后空闲115.68GiB。训练数据、基础依赖模型、日志、评测及视频未删。
本页历史路径仍保留用于溯源，已删除模型不能再按旧命令直接复用。
[精确删除清单](../../../refs/training_records/20261020_keep_R6_500_R8_600_cleanup_plan.json)；
服务器完成报告：`/data/outputs/keep_R6_500_R8_600_cleanup_completed_20261020.json`。
此“仅保留两份”指清理时的历史存档，不取消R9已授权的200/400/500存档。
当前各run已保留、已删除及计划产物详见[存档梳理](checkpoint_inventory_zh.md)。

2026-10-21在R11启动前的清理覆盖此前约定：清理时仅保留R9-step500和R10-step800，
删除R9的200/400、R10的200/400/600、R6-step500与R8-step600，共7份，释放83.24GiB。
保留全部日志、闭环结果、视频和训练数据；历史来源与保存验证记录不改写为“从未保存”。
[实际清理完成证据](../../../refs/training_records/20261021_keep_R9_500_R10_800_completed.json)。

各运行目录均位于服务器`/data/outputs/`：

| 编号 | 运行目录名 | 主要评测证据 |
|---|---|---|
| R1 | `official_adjust_bottle_9680_m28_add500_20261020` | 配对队列的`four_checkpoint_results.json`；[对照归档](../experiments/adjust_bottle/adjust_bottle_paired_control_zh.md) |
| R2 | `enhanced_adjust_bottle_9680_m28_add500_20261020` | 配对队列，500用`enhanced_500_retry_final`，不使用中断旧结果 |
| R3 | `official_adjust_bottle_9680_bf16_m56_add200_20261020` | `evaluation_200/summary.json` |
| R4 | `three_task_from_adjust_bf16_200_m56_add400_20261020` | `evaluation_400/summary.json`及`evaluation_400_selected_adjust_bottle_click_bell/summary.json` |
| R5 | `place_container_plate_from_adjust_bf16_200_m56_add200_20261020` | `evaluation_200/summary.json` |
| R6 | `sequential_plate100_cup200_pot200_from_plate200_bf16_m56_20261020` | `evaluation_500/summary.json` |
| R7 | R6目录下`remediation_round_1` | `<run>_pipeline/status.json`、`train.log`；最终`evaluation_300/summary.json` |
| R8 | `sequential_switch_pillbottle_stapler_from_R6_bf16_m56_200each_20261020` | `evaluation_600/summary.json` |
| R9 | `20261020_2058_stapler200_pillbottle200_switch100_from_R8_bf16_m56` | `evaluation_500/summary.json` |
| R10 | `20261020_2247_playingcard_phone_scale_qrcode_from_R9_bf16_m56_200each` | `evaluation_800/summary.json` |

本次只读[证据快照](../../../refs/training_records/20261020_190141_completed_snapshot.json)包含配置来源、原始状态、评测summary与日志首末步。
R1/R2评测结果补充自已归档的对照说明；快照未收集其外置评测队列。
重新取证：服务器实验包目录执行`python tools/collect_training_records.py > <新快照路径>`。
脚本只读文件、仅标准库，不加载模型、不使用GPU；此采集器目前覆盖上表已登记run，新run需加入列表。
采集不会自动重写本页；训练完成后应据最终报告更新R7和任务统计，禁止把计划300写成已完成300。
