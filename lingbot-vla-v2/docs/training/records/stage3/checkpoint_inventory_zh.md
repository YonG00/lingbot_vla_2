# 阶段三：存档记录

范围：针对难学任务筛选动作片段、调整逐样本损失及采样的独立实验。从20261022_1246夹碗实验开始；来源为阶段二第三轮补课step160。独立分叉不相加，计划/启动与完成严格区分。

[分类入口](../README.md) · [阶段二来源记录](../stage2/checkpoint_inventory_zh.md)

以下按授权与核验时间保留历史库存事件；仅整理文档，没有执行任何模型删除或实时库存复核。

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

### 2026-10-23 09:08 用户指定删除三版存档（已完成）

- `/root/autodl-tmp/data/outputs/20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample/checkpoints/global_step_200`：已删除。
- `/root/autodl-tmp/data/outputs/20261022_2339_bowls_add100_from_weight2_s200_bf16_m56_original_recipe_replay10pct_serial3/checkpoints/global_step_100`：已删除。
- `/root/autodl-tmp/data/outputs/20261023_0003_bowls_add100_from_original_s100_bf16_m56_grasp2_place10pct2_replay10pct_eval3parallel/checkpoints/global_step_100`：已删除。

三份目录合计约35.68GiB；可用空间18.51→54.18GiB。仅删除上述存档，run目录、日志、评测和视频保留；0818来源及0847订书机run未改动。既往“保留”条目是历史状态，以本条后续清理事件为准。证据 `refs/data_selection/20261023_delete_1246_2339_0003_completed.json`。

### 2026-10-23 0847最终存档核验

`20261023_0847_stapler200_from_bowls0818_s100_bf16_m56_postgrasp2_replay10pct_parallel3_video/checkpoints/global_step_200/hf_ckpt` 已生成且验证1708个BF16张量；按用户要求仅最终200，没有100。闭环完成0/3、三视频保留。0818来源未删除。白色目标扩展仅数据准备，不生成新HF；本次没有删模型。

### 2026-10-23 0928存档计划

`20261023_0928_stapler_add200_from_0847_s200_bf16_m56_broad_whitegrasp2_replay10pct_parallel3_video/checkpoints/global_step_200/hf_ckpt` 仅计划最终200，正式训练已启动；生成前不记为现有模型。来源0847-step200保留，无删除。

### 2026-10-23 0928最终存档已验证

`20261023_0928_stapler_add200_from_0847_s200_bf16_m56_broad_whitegrasp2_replay10pct_parallel3_video/checkpoints/global_step_200/hf_ckpt` 已生成且验证1708个BF16张量，仅最终200。订书机原三视频保留，追加move_can_pot只评测不新增模型、不删任何存档。

- move_can_pot独立录像评测完成，不生成/删除HF；0928-step200与0847来源保持不变，三份录像均可读取。
