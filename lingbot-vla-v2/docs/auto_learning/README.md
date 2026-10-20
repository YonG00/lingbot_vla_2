# Auto Learning系统文档

训练配方、具体实验、数据筛选、闭环环境以及训练记录已集中到[训练文档](../training/README.md)。
本目录保留AL系统本身的机制、改动记录和排障资料。

| 想了解什么 | 文档 |
|---|---|
| 系统总览 | [总览](auto_learning_overview_zh.md) |
| AL启动参数 | [启动指南](auto_learning_launch_guide_zh.md) |
| 修改过什么、验证到哪、如何回滚 | [改动台账](AL_PENDING_CHANGES.md) |
| 常见故障 | [易错点](auto_learning_pitfalls_zh.md) |
| 多卡与评测分片 | [多卡指南](auto_learning_multigpu_guide.md)、[多rank诊断](auto_learning_multirank_eval_diagnosis.md) |
| GMEAN与PASS门槛 | [门槛指南](GMEAN_PASSLINE_GUIDE.md) |
| 目标变化窗口与低噪FM | [窗口说明](target_change_window_zh.md) |
| FK辅助监督 | [EE说明](ee_fk_loss_zh.md) |
| 相对动作表示 | [动作表示](relative_action_representation_zh.md) |
| AL指定技能混合模式 | [技能混合实验](skill_mix_task_experiment_zh.md) |
| 历史Replay证据与评测批量门控 | [离线复核](replay_offline_and_adaptive_eval.md) |

按日期命名的故障记录和旧设计稿保留原位，查阅时以改动台账的当前口径为准。
