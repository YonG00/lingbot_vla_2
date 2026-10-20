# 训练存档保留清单

2026-10-21更新。所有运行根目录均在服务器`/data/outputs/`，该路径解析为`/root/autodl-tmp/data/outputs/`。
以清理完成报告、各run的verified_checkpoints及实际目录为依据；计划产物不当作已生成。

| 实验 | run目录名 | 已保留HF存档 | 当前用途 |
|---|---|---|---|
| R6 | `sequential_plate100_cup200_pot200_from_plate200_bf16_m56_20261020` | `checkpoints/global_step_500/hf_ckpt` | 用户指定保留的历史模型 |
| R8 | `sequential_switch_pillbottle_stapler_from_R6_bf16_m56_200each_20261020` | `checkpoints/global_step_600/hf_ckpt` | 用户指定保留；R9来源 |
| R9 | `20261020_2058_stapler200_pillbottle200_switch100_from_R8_bf16_m56` | `checkpoints/global_step_200/hf_ckpt`、`global_step_400/hf_ckpt`、`global_step_500/hf_ckpt`（均在checkpoints下） | 三份均已校验；最终500为R10来源 |
| R10 | `20261020_2247_playingcard_phone_scale_qrcode_from_R9_bf16_m56_200each` | `checkpoints/global_step_200/hf_ckpt`、`global_step_400/hf_ckpt`、`global_step_600/hf_ckpt`、`global_step_800/hf_ckpt`（均在checkpoints下） | 四份实际HF索引/头部已复核，最终800用于本轮闭环 |

已校验的R6/R8/R9/R10各份HF均为1708张量、BF16；评测FP32加载不改变HF存档dtype。
本次只读取证，共9份global_step存档，剩余空间32.31GiB；未新增删除动作。
[实际文件清单与R10完整性证据](../../../refs/training_records/20261021_R10_final_audit.json)。
R6与R8两份是当次清理保留项，后续R9/R10已授权生成的新存档不受“仅保留两份历史模型”限制。

## 已删除旧存档

| 运行或模型 | 已删除步号 |
|---|---|
| 原始`al_v79_poolfix_7300` | 9680 |
| R3 adjust_bottle BF16基准 | 200 |
| R4 三任务混训 | 200（400此前已删除） |
| R5 plate单任务 | 200 |
| R6 顺序课程 | 250；500保留 |
| R7补课 | 250、300 |
| R8新任务顺序课程 | 200、400；600保留 |
| R1/R2 FP32标准与增强对照 | 250、500此前已删除 |

本次九份清理释放107.05GiB，报告：
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
