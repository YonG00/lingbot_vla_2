# 训练文档导航

**新同事或Agent先读：[项目接手说明](PROJECT_HANDOFF_zh.md)**，包含当前模型、环境、训练与独立闭环命令及验证边界。

训练相关内容统一放在这里；AL状态机、指标、缓存和系统排障继续放在[auto_learning](../auto_learning/README.md)。
JSON配置、数据清单和证据仍分别放configs、refs等工程目录，不放docs。

## 训练流水与任务统计

| 文档 | 入口 |
|---|---|
| 哪些模型仍保留、哪些已删除、哪些尚待生成 | [存档清单](records/README.md) |
| 任务训练步数与评测对照 | [查看](records/README.md) |
| 训练与闭环评测流水 | [查看](records/README.md) |

## 参数化课程与补课

| 文档 | 入口 |
|---|---|
| 10%复习样本、历史成功资格池及每N步轮换 | [轮换复习](recipes/replay_rotation_zh.md) |
| 参数化训练配方框架 | [查看](recipes/task_recipe_framework_zh.md) |

## 实验方案与结果

| 文档 | 入口 |
|---|---|
| adjust_bottle 当前实验基准：BF16 / GBS56 / 200步 | [查看](experiments/adjust_bottle/adjust_bottle_bf16_baseline_zh.md) |
| adjust_bottle 标准方法对照（FP32历史实验） | [查看](experiments/adjust_bottle/adjust_bottle_official_control_zh.md) |
| adjust_bottle 两组训练后自动对照 | [查看](experiments/adjust_bottle/adjust_bottle_paired_control_zh.md) |
| 鼠标辅助示范审核进展 | [查看](experiments/mouse/mouse_auxiliary_review_zh.md) |
| 鼠标辅助数据初筛：2026-10-19 | [查看](experiments/mouse/mouse_auxiliary_scan_20261019_zh.md) |
| 鼠标主任务＋辅助示范混合续训 | [查看](experiments/mouse/mouse_auxiliary_training_zh.md) |
| 鼠标主任务：FK位置权重6与关闭夹爪辅助监督的续训 | [查看](experiments/mouse/mouse_gripper_off_continuation_zh.md) |
| 鼠标FP32主参数与显存容量测试准备 | [查看](experiments/mouse/mouse_precision_capacity_zh.md) |
| 鼠标纯主任务50条训练及闭环评测 | [查看](experiments/mouse/mouse_primary50_train_eval_zh.md) |
| 原始 step9680：鼠标放置单任务，500步 | [查看](experiments/mouse/place_mouse_pad_9680_zh.md) |
| 鼠标50条全量拟合实验 | [查看](experiments/mouse/place_mouse_pad_all50_p10_zh.md) |
| place_container_plate：从adjust_bottle BF16-200单任务续训 | [查看](experiments/multi_task/place_container_plate_single_task_zh.md) |
| 顺序续训：plate100 → cup200 → pot200 | [查看](experiments/multi_task/sequential_plate_cup_pot_zh.md) |
| BF16-200基准续训三个任务 | [查看](experiments/multi_task/three_task_bf16_continuation_zh.md) |
| place_object_stand step500 续训方案（2026-10-17） | [查看](experiments/stand/stand_ee_continue_20261017_zh.md) |

## 数据筛选与统计

| 文档 | 入口 |
|---|---|
| 指定目标物的episode筛选与审核 | [查看](data/object_episode_selection_zh.md) |
| RoboTwin 50 类任务轨迹帧数统计 | [查看](data/robotwin_50_task_frames_zh.md) |

## 闭环评测运行环境

| 文档 | 入口 |
|---|---|
| 单模型多仿真、六路分批、会话隔离与资源试测 | [并行仿真](evaluation/parallel_simulation_zh.md) |
| RoboTwin Blackwell 评测去噪运行库 | [查看](evaluation/robotwin_render_runtime_zh.md) |

## 目录约定

- `records/stage2/`：技能训练、复习与补课；`records/stage3/`：难学任务样本加权实验。各含流水、任务统计及存档记录，见[分类导航](records/README.md)。
- `recipes/`：课程表、采样、存档和补课框架的使用说明。
- `experiments/`：按实验主题分组，保留当时配置、结果与证据边界。
- `data/`：数据筛选、审核方法和规模统计。
- `evaluation/`：闭环评测环境和运行方法。

新增训练文档先加入本索引；系统改动仍登记[AL改动台账](../auto_learning/AL_PENDING_CHANGES.md)。
每轮训练结束后必须更新records所属阶段（stage2或stage3）中的流水、任务累计和存档清单；闭环未结束先标待定，完成后补齐，再检查、提交并push对应实验分支。
运行目录新命名采用`YYYYMMDD_HHMM_说明`，如`20261020_2002_switch_pillbottle_stapler`；历史run保持原路径。

- [pick_dual_bottles双手水平抓瓶样本筛选](recipes/pick_dual_bottles_target_selection_zh.md)。

- [完整轨迹分阶段与多模态权重](recipes/full_trajectory_stage_multimodal_weighting_zh.md)

- [pick_diverse_bottles完整阶段与权重](recipes/pick_diverse_bottles_full_stage_weighting_zh.md)

- [订书机早晚放置与白色样本权重](recipes/move_stapler_pad_sample_classes_zh.md)

- [统一类别与逐动作加权：完整阶段／目标分类共用参数](recipes/category_weighting_framework_zh.md)

- [move_can_pot 接近抓取20倍](recipes/move_can_pot_grasp_weighting_zh.md)

- [stamp_seal关键阶段20倍与串行接续](recipes/stamp_seal_key_weighting_zh.md)

- [药瓶放垫：1017来源差异化类别加权](recipes/move_pillbottle_pad_weighting_zh.md)
