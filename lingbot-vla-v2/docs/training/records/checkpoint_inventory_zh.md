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
