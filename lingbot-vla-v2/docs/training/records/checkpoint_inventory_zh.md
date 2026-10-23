# 训练存档保留清单

2026-10-21更新。所有运行根目录均在服务器`/data/outputs/`，该路径解析为`/root/autodl-tmp/data/outputs/`。
以清理完成报告、各run的verified_checkpoints及实际目录为依据；计划产物不当作已生成。

| 实验 | run目录名 | 已保留HF存档 | 当前用途 |
|---|---|---|---|
| R11 | `20261021_0831_qrcode_card_bottles_roller_pot_stamp_from_R10_bf16_m56_replay10pct` | **仅1200**，BF16、1708张量 | 用户指定保留 |
| R12 | `20261021_1129_ten_tasks_from_R11_bf16_m56_replay10pct` | **仅1350**，BF16、1708张量 | 用户指定保留，R13来源 |
| R13 | `20261021_1441_ten_tasks_from_R12_bf16_m56_replay20pct` | **仅1250**，BF16、1708张量 | 500/1000已按用户指令删除 |
| R14 | `20261021_1810_seven_tasks_from_R13_bf16_m56_replay20pct_checked` | **500/1000/1350**，BF16、1708张量 | 1350为R15来源 |
| R15 | `20261021_2046_handover200_bowls200_from_R14_bf16_m56_no_replay` | **400**，BF16、1708张量 | R16来源 |
| R16 | `20261021_2159_remaining10_80each_from_R15_bf16_m56_replay10pct_resident` | **仅800**，BF16、1708张量 | R17来源；500已按用户指令删除 |
| R17 | `20261022_0926_restore25_group3_hold3_from_R16_bf16_m56_add1000` | **500/1000**，BF16、1708张量 | 1000为最新模型，500保留作对照 |

**R11启动前仅两份历史ckpt存在**，删除前后均核对1708张量、BF16及实际HF分片头部/索引。
R11结束后三份新增HF已核验；[R11完成证据](../../../refs/training_records/20261021_R11_completed.json)。
R12刚结束时共八份HF均完整，见[R12历史完成审计](../../../refs/training_records/20261021_R12_completed.json)，不代表现有保留数。
本次按用户最新指令删除R9-500、R10-800、R11-400/1000、R12-500/1000，共六份，释放71.35GiB；仅保留上表R11-1200和R12-1350，均再次核验。
清理后空闲115.22GiB，R13三份新HF安全预算42.19GiB；未减少本轮保存点。
R13刚结束时三份HF均已逐项核验，当时共五份、空闲约79.35GiB（历史快照）；[R13完成证据](../../../refs/training_records/20261021_R13_completed.json)。
八路滚动试跑只做推理，不生成新训练ckpt，不生成HF；后续R13清理和R14/R15新增后，当前共七份HF（已核对实际目录）。
[精确清理计划](../../../refs/training_records/20261021_keep_R11_1200_R12_1350_plan.json)；[完成清单](../../../refs/training_records/20261021_keep_R11_1200_R12_1350_completed.json)。
此前仅保留R9/R10的[历史清理证据](../../../refs/training_records/20261021_keep_R9_500_R10_800_completed.json)留存，不再作为当前清单。
此前9份存档的[历史审计](../../../refs/training_records/20261021_R10_final_audit.json)保留，用于说明训练时确实保存过；不代表目前仍保留。

## 已删除旧存档

| 运行或模型 | 已删除步号 |
|---|---|
| 原始`al_v79_poolfix_7300` | 9680 |
| R3 adjust_bottle BF16基准 | 200 |
| R4 三任务混训 | 200（400此前已删除） |
| R5 plate单任务 | 200 |
| R6 顺序课程 | 250、500，现无ckpt |
| R7补课 | 250、300 |
| R8新任务顺序课程 | 200、400、600，现无ckpt |
| R9 | 200、400、500，现无ckpt |
| R10 | 200、400、600、800，现无ckpt |
| R11 | 400、1000；1200保留 |
| R12 | 500、1000；1350保留 |
| R1/R2 FP32标准与增强对照 | 250、500此前已删除 |

此前九份清理释放107.05GiB，报告：
`/data/outputs/keep_R6_500_R8_600_cleanup_completed_20261020.json`。
[精确清理清单](../../../refs/training_records/20261020_keep_R6_500_R8_600_cleanup_plan.json)。
训练日志、闭环结果、视频、数据集和基础依赖模型保留；旧文档中的模型路径用于溯源，不保证可重新加载。
被四任务课程取代的三任务run未完成，其状态不能当作有效最终模型。

## 每轮结束后的更新要求

每次训练结束（包括失败、中断）核对真实终态后，同步更新：

1. [训练流水](training_history_zh.md)：来源、课程、步数、耗时、闭环/补课结果及run路径。
2. [任务统计](task_training_statistics_zh.md)：沿继承链累计步数，每次训练及补课单列。
3. 本清单：实际生成的存档、dtype/完整性、保留或删除情况。

如果闭环尚未完成，先记训练完成/评测待定，闭环结束后补齐；未授权不能改变保存间隔或清理模型。
检查通过后中文提交并push对应实验分支。该要求不代表运行中的训练已完成，也不自动授权关机。

2026-10-21后续清理：按用户指令已删除R13 `global_step_500`、`global_step_1000`，各12768173043字节，合计约23.78GiB；保留同run的`global_step_1250`作为R14起点。其他run未删除。R14计划500/1000/1350尚不计入实际库存。

R14完成确认：`20261021_1810_seven_tasks_from_R13_bf16_m56_replay20pct_checked/checkpoints`的500/1000/1350均已验证为1708张量BF16；本次未删除。R15从1350继承，计划新增400步HF，未生成前不计入库存。

R15 `20261021_2046_handover200_bowls200_from_R14_bf16_m56_no_replay/checkpoints/global_step_400/hf_ckpt`已实际生成，1708张量BF16验证通过。驻留交接GPU验证为隔离1步、不写HF，不增加正式存档。

R16完成后已核对实际目录，共9份正式HF，未删除任何存档；后置遗忘评测不生成新HF。[终态及库存证据](../../../refs/training_records/20261021_R16_completed.json)。

2026-10-22 R17启动前按用户指令只删除R16/global_step_500，共12768173061字节（约11.89GiB）；保留并复核R16/global_step_800为1708张量BF16。其他run及R16日志评测未删。清理后空闲43.22GiB，本轮新500/1000的安全预算29.13GiB，仍有14.09GiB预算外余量。当前正式HF共8份；R17两份未生成前不计入库存。

R17完成后500/1000均已复核完整，首批5任务测后停止未删除任何存档；本次未改变保存与保留策略。


### 2026-10-22 用户授权清理（夹碗专训前）

仅删除以下4个存档，各11.89GiB，共47.56GiB；不删除日志、评测及run其他文件：

- `20261021_1810_seven_tasks_from_R13_bf16_m56_replay20pct_checked/checkpoints/global_step_500`
- `20261021_1810_seven_tasks_from_R13_bf16_m56_replay20pct_checked/checkpoints/global_step_1000`
- `20261021_2046_handover200_bowls200_from_R14_bf16_m56_no_replay/checkpoints/global_step_400`
- `20261022_1128_recovery_scout3_from_R17_replay10pct/20261022_1150_round_2_recovery/checkpoints/global_step_80`

路径均相对于 `/data/outputs/`。最新第三轮step160完整保留为夹碗训练起点；其他未指定存档保留。清理后空闲54.94GiB。本轮预计新增step200约11.89GiB，实际生成前不记为现有存档。

夹碗独立实验step200已实际生成并校验：`20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample/checkpoints/global_step_200/hf_ckpt`，1708个BF16张量，12768173067字节。最新来源step160保留；本次评测未清理任何模型。

2026-10-22 V3训练前用户授权清理：仅删除 `20261022_1909_bowls_add200_from_weight2_s200_bf16_m56_replay10pct_grasp_place_weight3/checkpoints/global_step_100`，释放12770754560字节；该run最终200及第三轮来源160保留，日志和3回合视频不动。V3新100/200待生成，不计为现有存档。

### 2026-10-22 用户指定仅保留两份历史模型

已核验并保留 recovery3 `20261022_1209_round_3_recovery/checkpoints/global_step_160` 与 `20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample/checkpoints/global_step_200`。其他 `/data/outputs/**/checkpoints/global_step_*` 目录已按明确授权删除（共9份），日志、评测、视频保留。磁盘空闲7.06→114.08GiB。新双重加权实验仅计划新增step200。

- 目标片段1:2新run `20261022_2120_bowls200_from_recovery3_s160_bf16_m56_target90_far1_place2_loss2` 仅计划新增step200；未删除已有两份保留模型或刚完成的sample3_loss3-step200。

- 20261022_2120 run实际已保存step200但属于错误起点实验，未删除。正确续训run `20261022_2149_bowls200_from_weight2_s200_bf16_m56_target90_far1_place2_loss2` 来源第一版weight2sample-step200，只计划新增最终step200。

- 18cm run最终 `checkpoints/global_step_200/hf_ckpt` 已完成并验证1708张量BF16；只保存最终200。第一版weight2sample-step200和recovery3-step160仍保留，本次分析未删除模型。

- 18cm版本 `20261022_2200` 最终 `checkpoints/global_step_200/hf_ckpt` 已完成并验证1708个BF16张量。此次单任务回合并行验证复用该存档，不新增或删除检查点。

- 新run `20261022_2302_bowls200_from_weight2_s200_bf16_m56_mix154_left3_right2_place2_eval3parallel` 仅计划最终step200，无中间存档；来源第一版step200仍保留，本轮未删除任何旧模型。

- `20261022_2302_bowls200_from_weight2_s200_bf16_m56_mix154_left3_right2_place2_eval3parallel/checkpoints/global_step_200/hf_ckpt` 已完成保存及闭环，最终0/3，仅最终200；来源第一版模型保留，本次分析未删除任何存档。

- 原方法100步对照run `20261022_2339_bowls_add100_from_weight2_s200_bf16_m56_original_recipe_replay10pct_serial3`，用户批准仅最终step100，无中间存档；来源第一版step200保留，本轮未删除旧产物。

- 原方法2339 run最终step100已完成并串行1/3，是本轮指定来源，保留。20261023_0003新增放置样本run只计划最终step100，无其他删除授权或删除操作。

- 0003 run最终step100完成并通过最终HF核验，闭环0/3；本次未删除或改变任何存档节奏。

- 20261023_0818原抓取续训run只计划最终step100，来源2339-step100保留；未删除或修改其他模型保留策略。

- 0818最终step100已完成、HF及冻结策略确认保留，闭环1/3、无视频。固定策略目录是轻量规则产物，不是额外模型存档；订书机未生成训练检查点。本次未删除模型。
