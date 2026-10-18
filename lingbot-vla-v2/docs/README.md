# docs/ 索引 —— 按功能分区

> **本目录分两层**（2026-10-11 整理）：
> ① **官方原有文档**（`open_loop_*`、`phase*_guide`、`single_task_train_guide`、`task_split_guide`、
> `norm_stats_guide`、`episode_filter_guide`、`eval_step_limit_guide`、`multi_ckpt_eval_guide`、
> `robotwin_curriculum_v1`、`stage_b0_adapter_design`、`disk_guard_guide`、`config/`）
> —— **位置未动、文件名未改**，仍在本目录顶层。
> ② **AutoLearning 适配新增文档** —— 按功能归入下列子目录（文件名未改）。

## 目录一览

| 目录 | 装什么 | 文件数 |
|---|---|---|
| `docs/`（顶层） | **官方原有文档**（16 个 md + `config/`） | 17 |
| `auto_learning/` | AutoLearning 流程：总览、启动、易错、门槛/倍率、TB 对照、多卡、设计稿、**实测记录** | 16 |
| `rocm/` | AMD/ROCm 平台：实例重建恢复、使用指南、编译失败与 Triton 崩溃排查 | 4 |
| `acceptance/` | 验收与评测口径：96G bf16 验收、闭环单卡、Eval 批处理闸门、编译调优 | 4 |
| `archive/audit/` | **历史审计快照**（按日期打包的证据包，只读，勿当现行文档） | 61 |

## 想做什么 → 读哪篇

| 我要做什么 | 读哪篇 |
|---|---|
| 低噪 FM / 目标变化窗口采样 / 训练前事件报告 | `auto_learning/target_change_window_zh.md` |
| 相对输入状态动作表示与未来力监督边界 | `auto_learning/relative_action_representation_zh.md` |
| 原始9680专练鼠标，500步全量ViT+EE+低噪FM | `auto_learning/place_mouse_pad_9680_zh.md`、[启动配方CPU测试](../tests/mouse_pad_training/README.md) |
| 连续抓取窗口验证命令、覆盖率与验证边界 | [窗口测试说明](../tests/target_change_window/README.md) |
| AL固定步数训练为何不按数据epoch测评 | [epoch边界测试说明](../tests/al_epoch_schedule/README.md) |
| place_object_stand EE续训方案与复现命令 | `auto_learning/stand_ee_continue_20261017_zh.md` |
| 低噪 FM 与末端 FK 一致性监督 | `auto_learning/ee_fk_loss_zh.md` |
| FK 测试意图、命令、真实资产对拍与历史证据 | [功能测试说明](../tests/ee_fk/README.md)、[首次验证](../tests/ee_fk/VALIDATION_2026-10-17.md)、[精度修复与val链路](../tests/ee_fk/VALIDATION_RUNTIME_2026-10-17.md)、[96G实机验收](../tests/ee_fk/VALIDATION_GPU_2026-10-17.md) |
| 第一次跑通 AutoLearning | `auto_learning/auto_learning_overview_zh.md` |
| 启动训练（完整命令与参数） | `auto_learning/auto_learning_launch_guide_zh.md` |
| **避开已知的坑**（8 项预检 + 6 类核心坑） | `auto_learning/auto_learning_pitfalls_zh.md` |
| **要实测数字**（步时/扫描耗时/`--env` 传法/故障判定） | `auto_learning/AL_MEASUREMENTS_AMD_2026-10-11.md` |
| 看懂"倍率/门槛/PASS"口径 | `auto_learning/GMEAN_PASSLINE_GUIDE.md` |
| 多卡怎么用、扫描怎么分摊 | `auto_learning/auto_learning_multigpu_guide.md` |
| 多 rank 评测预检失败怎么查 | `auto_learning/auto_learning_multirank_eval_diagnosis.md` |
| 扫描太慢（gmean200 加速） | `auto_learning/scan_accel_gmean200_zh.md` |
| TensorBoard 与日志不一致 | `auto_learning/AL_TB_ALIGNMENT_NOTES.md` |
| 里程碑/存档策略 | `auto_learning/AL_HF_PASS_MILESTONE_POLICY.md` |
| 开环评测（官方三篇） | 顶层 `open_loop_guide.md`、`open_loop_selfcheck_guide.md`、`open_loop_validation_guide.md` |
| AL 与开环的结合 | `auto_learning/AL_OPENLOOP_UPDATE_GUIDE.md` |
| 单任务训练（官方） | 顶层 `single_task_train_guide.md` |
| 指定技能实验（PASS/任务扫描隔离、可选样本扫描、数据对拍、HF-only） | `auto_learning/skill_mix_task_experiment_zh.md`（本地 CPU 验证，上机待验） |
| 从指定技能实验 HF 存档便捷续练，可选择成功后自动关机 | `experiment/robotwin/continue_skill_training.py`；用法见 `auto_learning/skill_mix_task_experiment_zh.md` |
| 分阶段训练（官方） | 顶层 `phase_training_guide.md`、`phase_train_then_eval_guide.md`、`phase1_vit_frozen_guide.md` |
| 任务划分 / 课程顺序（官方） | 顶层 `task_split_guide.md`、`robotwin_curriculum_v1.md` |
| AMD 机器重建后怎么恢复 | `rocm/amd_instance_rebuild_recovery_zh.md` |
| AMD 平台使用说明 | `rocm/amd_rocm_usage_guide.md` |
| ROCm 编译失败 / Triton 崩溃 | `rocm/rocm_compile_failure_and_sdpa_then_eager_zh.md`、`rocm/rocm_triton_compile_crash_zh.md` |
| 编译调优（官方 + 我们的） | `acceptance/compile_tuning_guide.md` |
| CUDA缓存清理耗时 / clear与reuse对照 | `acceptance/cuda_cache_ab_guide.md`（复用模式未GPU验收） |
| Eval 批处理有没有闸门 | `acceptance/eval_batch_auto_gate_guide.md` |
| 96G 单卡 bf16 验收结论 | `acceptance/gpu96_bf16_acceptance.md` |
| 闭环评测（AMD 单卡） | `acceptance/closed_loop_amd_single_gpu_guide.md` |
| 评测**全景录像**（外部相机、720p、仿真时间 30/60 FPS；本地实现，GPU 待验） | `robotwin_global_view_video_zh.md` |
| 50任务 benchmark / 训练后闭环评测准备 | `acceptance/clean_benchmark_import_guide.md`（单卡 NVIDIA 适配，GPU 闭环待验证） |
| step9680 闭环结果 / NVIDIA两阶段对齐 | `acceptance/robotwin_step9680_eval_20261014.md`（全量中断；lift_pot 0/10伴随OIDN异常；两阶段对照待开机） |
| 磁盘/检查点保留策略（官方） | 顶层 `disk_guard_guide.md` |
| 查历史审计证据 | `archive/audit/<日期>/` |

## 相关索引

* 工具脚本索引：`tools/README.md`
* 训练机工作副本说明：`/workspace/lingbot_work/README_COPY.md`（机器侧，不入库）
