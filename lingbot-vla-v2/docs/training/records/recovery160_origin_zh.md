# 目标模型来源与验证边界

参考模型：`20261022_1209_round_3_recovery/global_step_160/hf_ckpt`，BF16 HF，1708 个张量。

历史模型来自 R17 混训1000步之后的补训链路，第三轮补训 `place_shoe`、`open_microwave`，各80步，共160步。训练结束后按当时要求跳过最终评测，状态为 `complete_without_final_evaluation`；这不代表闭环通过。

证据：[前两轮记录](../../../refs/training_records/20261022_recovery_rounds12_completed.json)、[第三轮记录](../../../refs/training_records/20261022_recovery_round3_no_eval.json)、[冻结参考成绩](../../../refs/training_records/replay_pool_9680_through_recovery_round2.json)。历史记录中的绝对路径只用于溯源，不要求复现机器具有同样路径。

三阶段文档是面向新机器的整理方案：固定15150步，第三阶段分五批检查25任务，退步任务各补100步。它与历史多轮80步补训不同。因此该流程用于复现训练方法与参考能力，不能将它的输出直接命名为历史 step160，也不能保证相同权重或成功率。精确历史重放还需冻结全部历史配方、数据顺序、随机状态和优化器状态；现有材料不足以作此保证。

本 main 不含 step160 之后的难任务专项训练。新实验请另建分支。
