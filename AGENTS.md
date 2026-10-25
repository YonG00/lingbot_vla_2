# 项目导航

- 当前开发基线 main，仅三阶段复现与独立评测。包目录 lingbot-vla-v2，仿真目录 RoboTwin-lingbot。
- 接手先读 lingbot-vla-v2/docs/training/PROJECT_HANDOFF_zh.md，再读 docs/training/reproduction/three_stage_baseline_zh.md。
- official_adjust_bottle_control 保留原训练基线；experiment/stack_bowls_sample_weighting 保留后续实验，不合入专项加权训练。
- 不在文档写入SSH账号、口令或机器连接信息。
- 修改训练核心须核对 configs/reproduction/source_lock.json；不把CPU检查当成GPU训练或闭环验证。
- 操作须在用户授权范围内；测试通过后中文提交，git commit 使用 -F 文件；push 单独执行。
- 更改保存间隔或保留策略必须获得用户明确允许。训练后更新实际训练记录、评测与存档情况。
- 长任务后台启动；不擅自停进程、删模型或关机。远端脚本本地落盘后传输执行。
- 改动本意、验证及回滚记录见 lingbot-vla-v2/docs/auto_learning/AL_PENDING_CHANGES.md。
