# tools/ 索引 —— 我要做什么，跑哪个脚本

> **本目录分两层**：
> ① **官方原有脚本**（`task_split.py`、`robotwin_curriculum.py`、`prepare_phase.py`、`prune_dcp.py`、
> `open_loop_*`、`drill_*`、`resume_*` …）—— **位置未动**，仍在本目录顶层。
> ② **AutoLearning 工作新增脚本** —— 按用途归入下列子目录（2026-10-11 整理，文件名未改）。
>
> 判断依据：①是官方仓库自带；②是本次 AL 适配（AMD/ROCm 平台、AutoLearning 流程）新增。

## 快速对照表

| 我要做什么 | 跑哪个 |
|---|---|
| 从FP32模型和数据执行三阶段基线，预览/环境检查/后台训练 | `reproduce_baseline.sh`；见`docs/training/reproduction/three_stage_baseline_zh.md`，默认2000步保存，参数可调 |
| 评测训练后的最终模型，默认25任务各3回合并保存视频 | `python experiment/reproduction/evaluate.py --run-dir <训练输出目录> --output-dir <新评测目录> --execute` |
| 只读取证已登记实验的来源、训练进度及闭环结果 | `collect_training_records.py`（标准库、无GPU；打印JSON快照，表格见docs中的训练流水与任务统计） |
| FP32主参数/BF16计算的3步微批显存探针 | `experiment/robotwin/precision_batch_probe.py`（默认预览，须开卡后再加`--execute`；不存档/不关机） |
| 鼠标混训真实数据CPU整链验收（含动作块边界） | `acceptance_checks/mouse_auxiliary_cpu_preflight.py`；使用独立输出目录，`--check-dataset-chunks`需训练环境 |
| 按目标物筛选辅助episode、核对v3视频并生成审核队列 | `select_object_episodes.py`；说明见 `docs/training/data/object_episode_selection_zh.md` |
| 划分 50 个任务的训练/评测集合 | `task_split.py`（官方） |
| 生成课程顺序 manifest | `build_robotwin_curriculum_manifest.py`（官方）+ `robotwin_curriculum.py`（官方） |
| 准备阶段数据（phase1/2） | `prepare_phase.py`（官方） |
| fp32 权重转 bf16 | `checkpoint_convert_and_prune/make_bf16_ckpt.py` |
| DCP 检查点转 HF 格式 | `checkpoint_convert_and_prune/dcp_to_hf.py` |
| 剪枝旧检查点省磁盘 | `prune_dcp.py`（官方）/ `drill_prune.sh`（官方） |
| 训练前预检（阈值表 / baseline 是否齐） | `acceptance_checks/gmean50_preflight.py` |
| 验收「扫描加速」是否真在 GPU 上跑 | `acceptance_checks/scan_accel_gpu_acceptance.py` |
| 扫描加速的 CPU 侧预检 | `acceptance_checks/scan_accel_preflight.py` |
| 验收 96G 单卡 bf16 | `acceptance_checks/gpu96_acceptance.py` |
| 验收 HF 直出（跳过合并） | `acceptance_checks/hf_direct_export_acceptance.py` |
| 验收模型加载加速 | `acceptance_checks/load_accel_smoke.py` |
| 汇总CUDA缓存clear/reuse对照日志（纯CPU） | `auto_learning_pipeline/al_cache_profile.py`；说明见 `docs/acceptance/cuda_cache_ab_guide.md` |
| 验收 eval 批处理（随机/自适应） | `acceptance_checks/eval_batch_stochastic_acceptance.py`、`adaptive_eval_batch_acceptance.py` |
| 多卡评测冒烟 | `acceptance_checks/multigpu_eval_smoke.py` |
| 48G 卡冒烟（不存 ckpt / TB10） | `acceptance_checks/run_smoke_nosave_48g.sh`、`run_smoke_tb10_48g.sh`、`run_smoke_tb10_48g_v2.sh` |
| 开环评测预检冒烟 | `acceptance_checks/preflight_smoke.sh` |
| **看 AL 训练进度 / 通关 / 每任务倍率变化** | `auto_learning_pipeline/al_panel.py` |
| **从单技能实验 HF 存档便捷续练（可选完成后关机）** | `experiment/robotwin/continue_skill_training.py`；默认预览，`--execute` 后台运行，`--shutdown` 仅在训练与约定里程碑HF校验成功后执行关机；显式FP32主参数模式可读FP32来源 |
| 监控周期 HF，满 4 份删除最早一份、保留最近 3 份 | `auto_learning_pipeline/al_checkpoint_watch.py`（默认预览；`--apply` 清理；`--once` 单次检查；跳过 DCP/未完成存档，`--protect` 保护输入存档） |
| 校验 AL B0 适配层 | `auto_learning_pipeline/al_b0_verify.py` |
| AL B1 回归 | `auto_learning_pipeline/al_b1_regression.sh` |
| 检查候选池倍率选样是否合理 | `auto_learning_pipeline/al_pool_ratio_selftest.py` |
| 老实现与新实现的逐样本对齐 | `auto_learning_pipeline/al_legacy_parity.py` |
| 复现「bootstrap 顺序 / FSDP2」问题 | `auto_learning_pipeline/al_bootstrap_order_fsdp2_repro.py` |
| 复现「多 rank 评测预检」问题 | `auto_learning_pipeline/al_eval_multirank_preflight_repro.py` |
| 旧 scout 缓存迁移到新命名 | `auto_learning_pipeline/al_scout_cache_migrate.py` |
| 定位 eval 批处理在 CPU/GPU 的差异 | `problem_reproduction/eval_batch_cpu_diag.py`、`eval_batch_gpu_probe.py` |
| 校验 eval 批处理 B1 一致性 | `problem_reproduction/eval_batch_b1_parity_check.py` |
| 定位部署侧加载问题 | `problem_reproduction/deploy_load_probe.py` |
| 复核离线 replay 结论 | `problem_reproduction/replay_offline_verify.py` |
| 核对 FK spec 与 RoboTwin 左右六轴名称/顺序 | `auto_learning_pipeline/ee_fk_spec_check.py` |
| 收集开环评测结果 | `collect_open_loop.py`（官方） |
| 单卡开环自检 | `open_loop_selfcheck.py`（官方） |
| 打审计包（存档证据） | `misc/make_audit_pkg.sh` |
| 抢救性拷贝（销毁实例前） | `misc/rescue_before_destroy.sh` |
| 起 policy server / 跑任务评测 | `misc/policy_server.sh`、`misc/eval_tasks.sh`（**面向 AutoDL 写的，已改为按脚本位置定位仓库**） |
| 用参考模型跑开环 | `misc/eval_ref_model_open_loop.py` |
| ROCm/Triton 相关排查 | `rocm/` |

## 子目录说明

| 目录 | 定位 |
|---|---|
| `auto_learning_pipeline/` | AutoLearning 流程专用：启动辅助、自检、迁移、监控面板 |
| `acceptance_checks/` | **跑完给"通过/不通过"结论**的验收与冒烟脚本 |
| `problem_reproduction/` | **出问题才用**：复现、对比、定位（平时不用跑） |
| `checkpoint_convert_and_prune/` | 检查点格式转换（DCP↔HF、bf16） |
| `misc/` | 平台相关/一次性的辅助脚本（AutoDL 遗留、审计打包、抢救拷贝） |
| `rocm/` | ROCm / Triton 平台侧排查与修复 |

## 通用注意

1. **手工直调 python 时要带 `PYTHONPATH=<仓库根>`**：`lingbotvla` 是 editable 安装，指向官方旧树；
   不带会**静默加载错代码**。launcher（`experiment/robotwin/al_launch.py`）会自己设，无需关心。
2. `machine` 上仓库可能有多个副本（原目录 + 工作副本），跑之前先 `cd` 到你要的那一份。
3. AutoDL 遗留脚本（`misc/policy_server.sh`、`misc/eval_tasks.sh`）默认路径已改为
   **按脚本自身位置推导仓库根**；如需指向别处，显式 `export REPO=...` 覆盖。
