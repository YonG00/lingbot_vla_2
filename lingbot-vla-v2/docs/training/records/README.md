# 训练记录

| 类别 | 范围 | 训练流水 | 任务统计 | 存档记录 |
|---|---|---|---|---|
| 阶段二 | 基于标准学习方式做技能训练、筛选几百步可学会的任务、顺序/混合复习与补课 | [流水](stage2/training_history_zh.md) | [统计](stage2/task_training_statistics_zh.md) | [存档](stage2/checkpoint_inventory_zh.md) |
| 阶段三 | 难学任务的目标片段筛选、逐样本loss与采样加权对照 | [流水](stage3/training_history_zh.md) | [统计](stage3/task_training_statistics_zh.md) | [存档](stage3/checkpoint_inventory_zh.md) |

阶段二筛选“易学”是实验目的，不表示其中所有任务已通关。阶段三从 `20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample` 开始，引用阶段二的 `20261022_1209_round_3_recovery/checkpoints/global_step_160` 为来源。订书机定向加权也属于阶段三。

每轮在所属阶段更新三份文档，写清来源完整run与step、实际更新数、任务/样本策略、评测方式与结果路径、存档核验和授权清理。独立分支不合并累计；未测不算失败、计划不算完成。旧文件名保留为导航，新增记录写对应stage目录。库存记录包含历史事件，判断当前空间仍须现场核验。
