# 2026-10-17 原始验证证据

本目录保存无卡验证的原始小型产物，后续运行输出到新的临时目录，不覆盖本归档。

| 文件 | 用途 |
|---|---|
| `candidate_spec.json` | 从当时真实 RoboTwin 资产提取的左右六轴候选几何参数 |
| `sapien_reference.npz` | 200 组关节角、两臂 SAPIEN 参考变换、随机/任务/回合/帧标签；`allow_pickle=False` 可读 |
| `recorded_reference.npz` | HDF5 抽样 100 帧的状态关节、两臂记录末端位姿及任务/回合/帧标签 |
| `parity_report.json` | 候选 FK 与 SAPIEN 的位置、姿态误差及真实链梯度检查结果 |
| `recorded_parity_report.json` | 历史记录口径对比，`passed=false`，差异来源未确认 |
| `precision_report.json` | 几何常量经过 bf16 再转 FP32 后的误差；不是实际模型推理误差 |
| `lingbot_fk_cpu_test_r2.log.txt` | 原始 8 项单测输出；加 `.txt` 是为了不被仓库日志忽略规则漏掉 |
| `lingbot_fk_extra_cpu_checks.log.txt` | 原始梯度、归一化及 bf16 类型冲突探针输出 |
| `archive_check.log.txt` | 迁移目录后 8 项 CPU 回归及三个 JSON 报告完全一致的复核日志；已知未通过项仍保留 |

`candidate_spec.json` 的 `source.validation` 是导出阶段写入的 `pending numerical parity`，原样保留以免改写历史。导出后的几何验证以 `parity_report.json` 为准；它通过也不代表 bf16 训练或历史数据口径已经验收。资产来源摘要在 spec 的 `source` 字段中。

NPZ 仅含数值与文本标签，没有模型权重、图像或完整轨迹。测试意图、命令和判据见 [功能 README](../../README.md)，当次环境和结果见 [验证记录](../../VALIDATION_2026-10-17.md)。
