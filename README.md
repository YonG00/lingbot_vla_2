# LingBot 三阶段基线复现

目标参考模型：`20261022_1209_round_3_recovery/global_step_160`。

从[三阶段复现与交接](lingbot-vla-v2/docs/training/reproduction/three_stage_baseline_zh.md)开始，训练入口是 `lingbot-vla-v2/tools/reproduce_baseline.sh`。单任务与整体测评见[并行评测说明](lingbot-vla-v2/docs/training/reproduction/benchmark_zh.md)。

本 main 以 official_adjust_bottle_control 为训练基线，仅面向三阶段复现和评测。后续难任务的样本筛选、加权和专项实验保留在 experiment/stack_bowls_sample_weighting，不迁入这里。核心训练包内的可选历史模块作为冻结依赖保留，三阶段配方不启用它们。

**验证边界：**文档是整理后的三阶段训练方案，并非历史 checkpoint 的逐步重放；历史补训轮次及步数与便携方案存在差异。完整三阶段 GPU 重训尚未验证，不承诺逐位一致或相同闭环成功率。见[目标模型来源](lingbot-vla-v2/docs/training/records/recovery160_origin_zh.md)。

原项目安装与模型下载见[上游说明](lingbot-vla-v2/README.md)。
