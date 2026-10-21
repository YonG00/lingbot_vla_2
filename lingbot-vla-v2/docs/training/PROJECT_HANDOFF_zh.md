# 训练项目接手说明（同事与Agent入口）

更新日期：2026-10-21。面向当前顺序课程实验，不代替生产AL系统文档。
本文不包含服务器账号、地址、端口、密码或连接命令；访问权限由项目负责人另行授权。

## 先读什么

1. 仓库根目录`AGENTS.md`：协作权限、训练安全约定、代码目录。
2. 本文：当前模型、工作流、训练和闭环评测命令。
3. [训练流水](records/training_history_zh.md)、[任务统计](records/task_training_statistics_zh.md)、[存档清单](records/checkpoint_inventory_zh.md)。
4. 修改代码前读[待验证与已知改动台账](../auto_learning/AL_PENDING_CHANGES.md)。

## 当前情况：先避免接错模型

**最新完成模型：R12-step1350，训练与最终闭环均完成。** R11七任务1200步完成，
qrcode2/3、playingcard3/3，dual/diverse/roller/pot/stamp均0/3；保存400/1000/1200均已验证。
R12从R11最终HF顺序训练十任务1350步已完成，正式更新11:33:02–13:24:15，1小时51分13秒；10%复习、每10步轮换。
500/1000/1350三份均验证BF16、1708张量。最终十任务各3回合，6＋4分批、单模型FP32/B1闭环；无自动补课或关机。
49项本地/服务器CPU、真实1100条262964帧CPU和三步GPU探针通过；十任务30回合完成，合计8/30成功。
shake_bottle_horizontally3/3、dump_bin_bigbin2/3、stack_bowls_two1/3、stack_blocks_two2/3，其余六任务0/3。
六＋四批次屏障实测成立，模型仅加载一次、FP32/B1，30份视频完整且OIDN错误0。评测服务已退出，GPU空闲。
最新累计复习池：`refs/training_records/replay_pool_9680_through_R12.json`，17个合格任务；不回改R12实际使用的14任务冻结池。
R12 run：`/data/outputs/20261021_1129_ten_tasks_from_R11_bf16_m56_replay10pct`。
最新HF：本run下`checkpoints/global_step_1350/hf_ckpt`；闭环汇总`evaluation_1350/summary.json`。
配方：`configs/training_recipes/sequential_ten_tasks_1350_replay10pct_from_R11.json`。
来源：`/data/outputs/20261021_0831_qrcode_card_bottles_roller_pot_stamp_from_R10_bf16_m56_replay10pct/checkpoints/global_step_1200/hf_ckpt`。
完整课程、来源和结果见[流水](records/training_history_zh.md)。下述R10为历史已完成基线。

历史R10已完成800步，顺序为playingcard、phone、scale、qrcode，各200步。
BF16参数/计算/HF，micro=GBS56、GAS1；训练无复习、无自动补课。
正式更新从2026-10-20 22:49:54到23:59:43；2026-10-21 00:00:44训练结束。

| 最终800模型测评任务 | 成功次数 | 本轮训练 / 该任务继承累计 |
|---|---|---|
| move_playingcard_away | 1/3 | 200 / 200 |
| place_phone_stand | 0/3 | 200 / 200 |
| place_object_scale | 0/3 | 200 / 200 |
| rotate_qrcode | 1/3 | 200 / 200 |

四项逐回合日志、结果文本及summary一致，OIDN错误均0；三回合不是稳定成功率的证明。
R10此次未测旧任务，不能据此认定旧能力仍保留。模型总loss下降也不能替代闭环结果。

R11之前保留的两份历史实验ckpt如下，均核验1708张量、BF16；另外保留R11三份、R12三份，共八份，完整当前清单见[存档清单](records/checkpoint_inventory_zh.md)：

| 简称 | 完整HF路径 | 用途 |
|---|---|---|
| R9-step500 | `/data/outputs/20261020_2058_stapler200_pillbottle200_switch100_from_R8_bf16_m56/checkpoints/global_step_500/hf_ckpt` | 上一轮，可作对照；switch 2/3、pillbottle 0/3、stapler 0/3 |
| R10-step800 | `/data/outputs/20261020_2247_playingcard_phone_scale_qrcode_from_R9_bf16_m56_200each/checkpoints/global_step_800/hf_ckpt` | R11来源，历史对照 |

R6/R8、原始9680、早期BF16基准、R7补课与中间存档均已按授权删除；历史实验文档保留用于溯源。
例如早期`baseline_pipeline.py`从9680起训的命令，当前不能直接运行。
日志、视频、数据、run内trainer_config仍保留，删的是模型存档。
清理后空间115.54GiB是R11启动前历史值；R12最终评测后最新观测43.87GiB。开跑前重新检查，不能当永久容量。

## 分支、环境和代码坐标

| 项目 | 坐标 / 说明 |
|---|---|
| 当前实验分支 | `official_adjust_bottle_control` |
| 本地checkout | `/Users/mac/Documents/lingbot_v2/lingbot_vla_2_new` |
| Python包根 | checkout中的`lingbot-vla-v2/`；下面命令均在该目录运行 |
| 服务器实验checkout | `/data/experiments/official_adjust_bottle_control` |
| 服务器包目录 | `/data/experiments/official_adjust_bottle_control/lingbot-vla-v2` |
| 训练Python | `/data/miniconda3/envs/lingbotvla/bin/python` |
| 模拟环境Python | `/data/miniconda3/envs/RoboTwin/bin/python` |
| RoboTwin | `/data/experiments/official_adjust_bottle_control/RoboTwin-lingbot` |
| 数据 | `/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30` |
| 任务划分清单 | 包目录下`refs/al_artifacts/al/task_splits_50/manifest.json` |
| 输出 | `/data/outputs/`；报告可能显示其实际解析路径，含义相同 |
| 生产AL | `Task_Auto_Learning`，服务器`/data/code`；不是本实验的同步目标 |

当前接手以本实验分支为准，`learn_skill_mix_task`是历史实验分支；不要套用生产分支的reset命令。
此前服务器通过受控文件同步运行；GitHub提交并不证明服务器HEAD或所有文件一致。
本次未取得服务器最新HEAD，不能承诺已完整同步最新轮换复习实现。
接手先在对应checkout检查`git branch --show-current`、`git rev-parse HEAD`、`git status --short`，
与本地/远端分支比较；有未提交文件时先核对，不直接reset丢弃运行所需改动。

近期CPU核对时`nvidia-smi`曾Permission denied；只读HF和日志核验已完成，
但重新GPU训练前需要负责人确认GPU可用，并正常执行GPU状态检查。
本地缺少部分重依赖，不能用本地语法/标准库测试代替训练机数据与GPU验收。

## 最近常用方法与边界

| 能力 | 当前用法 | 验证状态 |
|---|---|---|
| 顺序课程 | `train.mode=sequential`，task_order定顺序，main_tasks逐任务设steps | 已跑多轮训练及闭环 |
| 保存 | `checkpoint.mode=task`，任务结束后保存；或steps模式按every_steps保存 | R9/R10任务边界HF实际落盘并核验 |
| 精度 | BF16参数/计算/HF；评测服务FP32加载、FP32推理、chunk50 | 已实跑 |
| 主任务数据 | 各任务全部50条，任务内均匀抽帧，有放回；阶段间不重载模型/优化器 | 已实跑；没有独立开环验证集 |
| 续训 | 从HF加载权重，新优化器、新学习率日程、新步号 | HF-only不是完整断点恢复 |
| 多任务闭环 | 一个推理服务串行切换任务，每任务3回合 | 同一模型只加载一次；任务并非并行 |
| 10%轮换复习 | 每N步一个旧任务，默认N10，排除当前主任务；历史≥1/3池 | R11完整1200步及七任务闭环、R12完整1350步及十任务闭环完成；不能据此认定全部任务学会 |
| 补课 | 可选、默认关闭；2/3优先补100，再1/3补100；0/3和3/3跳过，最多一轮 | 旧规则R7实跑后退化；新规则仅CPU契约 |

复习与自动补课同时开启目前被schema拒绝。默认闭环任务从本轮主课程生成；
补课仅测实际补训任务。需要额外保留能力检查时显式填写evaluation.tasks。
“标准方法”指恢复仓库初始快照模型核心与学习方式，不是重新验证过上游发布的一致性。
当前配方要求AL/窗口/EE/LN-FM关闭，标准L1 FM/Beta时间采样、全参数训练、GC与compile开启、workers8。

## 如何从最新模型续训

推荐以已提供的[接手训练示例](../../configs/training_recipes/handoff_from_R10_example.json)为起点，
它从R10-step800起训playingcard200→qrcode200，任务模式保存200/400，最终两任务各测3回合。
**这是未执行的示例，不是已批准的新实验；GPU执行前先取得本次训练授权。**

主要改动均在JSON中完成，不用改训练器：

- source.trainer_config：所选源run的trainer_config.json；source.checkpoint：该源模型的hf_ckpt目录。
- train.main_tasks：每任务steps；train.task_order：顺序，当前同一任务只能出现一次。
- output_dir：全新输出名；CLI为未带时间前缀的名字添加北京时间YYYYMMDD_HHMM。
- checkpoint：task模式不写every_steps；steps模式必须写every_steps。最终HF始终保存。
- evaluation：默认省略tasks，训什么测什么；episodes_per_task=3、checkpoints=["final"]、shared_model_session=true。
- train.replay及remediation：没有明确实验要求时保持关闭。

在服务器实验包目录：

```bash
# 只读预览：不建输出目录，不加载模型，不训练
/data/miniconda3/envs/lingbotvla/bin/python experiment/task_recipes/run.py \
  --recipe configs/training_recipes/handoff_from_R10_example.json

# 已获训练授权后执行；该入口自行后台运行，不需要再包一层nohup
/data/miniconda3/envs/lingbotvla/bin/python experiment/task_recipes/run.py \
  --recipe configs/training_recipes/handoff_from_R10_example.json --execute
```

启动终端打印真实output_dir与RECIPE_STARTED。顺序为容量与来源检查→真实数据CPU检查→独立三步GPU短测→
从指定源HF正式起训→验证保存计划内全部HF→最终共享服务闭环→complete。
短测不保存，不继承短测权重。预览通过不代表GPU已验证；训练失败时后续评测不继续。
代码依赖以下外部资源，Git clone本身不提供模型、数据或Python环境：

- 源run配置里的Qwen目录；既有路径为`/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct`。
- 深度教师与视频教师，继承自源trainer_config的train.align_params。
- 数据归一化文件，既有配置为`/data/code/lingbot-vla-v2/assets/norm_stats/robotwin_competition_clean.json`；读取该文件不意味着可以改生产代码。
- RoboTwin资源及OIDN运行库，见[闭环去噪环境](evaluation/robotwin_render_runtime_zh.md)。

## 如何只做闭环评测

已训练的模型不需要再启动训练。使用[独立共享评测示例](../../configs/evaluation_jobs/handoff_R10_shared_eval_example.json)：

1. 把顶层destination及各job.destination中的`YYYYMMDD_HHMM`统一改为实际北京时间，使用全新目录。
2. jobs列表指定任务和episodes；每个job的model必须为同一份HF，task不得重复。
3. 修改任务时同时修改它的destination；核对robotwin/qwen/sim_python在当前环境存在。
4. 同一模型的指定任务共用服务，每个任务3回合，评测完服务自动退出；没有自动关机。

```bash
# 只打印job，不加载GPU服务
/data/miniconda3/envs/lingbotvla/bin/python experiment/official_control/shared_evaluation.py \
  --job configs/evaluation_jobs/handoff_R10_shared_eval_example.json

# 已获评测授权后执行；自行后台运行
/data/miniconda3/envs/lingbotvla/bin/python experiment/official_control/shared_evaluation.py \
  --job configs/evaluation_jobs/handoff_R10_shared_eval_example.json --execute
```

当前服务固定FP32加载/计算、compile关闭、chunk50、noise seed0，模拟demo_clean、OIDN无错误检查。
BF16 HF可加载成FP32，不需要另存一份FP32 HF；dtype转换不会恢复BF16已经舍去的精度。
每任务会做可行性筛选，因此实际seed未必相同；比较两个模型时核对实际seed，不能只看起始seed0。

| 要找什么 | 训练自动闭环 | 独立共享评测 |
|---|---|---|
| 流程状态 | `<run>_pipeline/status.json` | `<destination>/status.json` |
| 训练日志 | `<run>/train.log` | 无 |
| HF完整性 | `<run>/verified_checkpoints.json` | 评测前自行确认所选模型 |
| 汇总 | `<run>/evaluation_<step>/summary.json` | `<destination>/summary.json` |
| 逐任务配置与日志 | 评测目录下`<task>/job.json`、`sim.log` | 相同 |
| 成功率文本 | `<task>/results/<task>/_result.txt` | 相同 |
| 全局视频 | `<task>/results/<task>/episode*_global.mp4` | 相同 |
| 服务PID与模型路径 | 评测目录下session.json | 相同 |

TensorBoard每run写在`<run>/runs/`；既有6007服务需确认是否运行再访问，不能只凭历史记录判断可用。
不要用AL面板的PASS/内部步号解释这套固定课程训练；以正式train.log和冻结step_plan为准。

## 复习与后续实验

最新[复习资格池](../../refs/training_records/replay_pool_9680_through_R10.json)共14个任务，
含用户提供9680截图资格与后续完整闭环报告。1/4不达1/3；历史曾达标即可，不因后续失败移除。
候选池会在准备run时冻结，不热更新。复习块、当前主任务排除和导入新结果见[轮换复习](recipes/replay_rotation_zh.md)。
同事首次接手先核对最新run实测终态与验证边界；每个新配方仍须CPU/GPU前置检查，不能把三步短测当作完整多阶段及闭环成功。

## 每次结束必须做什么

1. 区分训练完成、HF校验完成、闭环完成，失败/中断也记录；不把计划步数写成完成。
2. 更新[流水](records/training_history_zh.md)、[任务统计](records/task_training_statistics_zh.md)、[存档清单](records/checkpoint_inventory_zh.md)，附完整run路径。
3. 按完整闭环结果维护下一版复习资格证据；不覆盖冻结旧快照。
4. 保留原始证据在refs/training_records，检查文档链接、表格与配置；代码有改动则执行对应测试。
5. 中文`git commit -F <消息文件>`，单独`git push origin official_adjust_bottle_control`，报告SHA与验证边界。

保存节奏、删除历史模型、训练起停和GPU关机均受AGENTS授权约束；不要根据“空间可能不够”自行减少存档。
要求停训练时需要停止控制器与其训练/评测子进程组并验证，不只关闭当前终端或只杀一个模型进程。

## 可以交给其他Agent的接手提示

> 阅读根AGENTS.md、docs/training/PROJECT_HANDOFF_zh.md、训练流水/统计/存档清单及AL_PENDING_CHANGES。
> 当前实验分支official_adjust_bottle_control；不要同步到生产/data/code。
> 只有R9-step500和R10-step800两份历史ckpt存在，旧9680已删除；先核实所选source和服务器代码版本。
> 使用task_recipes JSON指定任务、步数、任务边界存档与默认随训练生成的评测列表。
> 轮换复习已完成R11/R12多阶段训练及各自完整闭环；最新结果读训练流水，下一轮资格池可选through_R12。默认只读，不自行起训练、删除模型或关机。
> 训练结束更新流水、任务累计和存档清单，检查后中文提交并push实验分支。服务器访问由负责人另行授权。
