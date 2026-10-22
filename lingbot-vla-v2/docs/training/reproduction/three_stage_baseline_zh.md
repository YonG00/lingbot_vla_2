# 三阶段基线复现与交接

这份说明面向拿到仓库、模型和数据的新同事/Agent：先配好训练与仿真环境，再用参数启动连续课程，最终从`final_model.json`取得HF模型路径。无需访问原训练服务器、旧outputs目录或SSH账号。

**验证边界：本入口是新基线，不是当前模型效果的已完成复现。** 第一阶段是新增5000步；第二阶段重放历史有效主链；第三阶段是新的五批检查/补课。历史上零散任务的小样本成功率不能保证在新流程中重现，也不能将五批不同中间模型的成绩称为最终模型全25项成功率。目前仅完成本地CPU契约与模拟编排测试，已通过服务器微型FP32/BF16真实CPU转换校验；尚未运行本入口的完整模型转换、GPU训练和完整闭环。

## 1. 需要准备什么

| 项目 | 要求 |
|---|---|
| 代码 | 本仓库`official_adjust_bottle_control`分支，保留同仓库`RoboTwin-lingbot/`，记录实际Git SHA |
| GPU | 历史实测为单张 **NVIDIA RTX PRO 6000 Blackwell 96GB**；不是48GB的RTX A6000。其它型号需自行短测，不保证相同batch/速度 |
| 主机 | Linux、可用NVIDIA驱动/CUDA/Vulkan；转换在CPU逐分片进行，需要容纳FP32分片、BF16分片与校验副本。建议预留128GB主机内存，此为容量建议，非本入口实测门槛 |
| base | 带`config.json`、`model.safetensors.index.json`和全部分片的FP32 HF目录；历史对照使用**step9680模型**。换成官方原始base或其它权重即改变实验起点 |
| 数据 | 与冻结manifest相同的RoboTwin clean 50任务×50条，共2500条，LeRobot v3格式；路径可改，episode编号/内容/任务顺序不能重排 |
| 模型资源 | Qwen3-VL-4B-Instruct处理器资源，以及深度和未来视频教师权重；三项损失仍启用，不能为了少下载资源而关闭 |
| 仿真 | 安装RoboTwin环境与机器人/物体资产；Blackwell按下文安装OIDN 2.5.1；模型下载、数据和仿真资产不随Git提供 |
| 输出空间 | 默认固定16份训练HF＋1份BF16转换副本，最多另存5份补课HF；按每份11.89GiB，最多约261.6GiB，另留安全余量。建议输出卷至少320GiB空闲，不含FP32输入、数据和环境；启动检查按实际输入分片大小计算安全预算 |

模型身份不同、数据重排、仿真/归一化改变时，不能声称是同条件效果复现。BF16转换不可逆地舍去部分FP32精度，输入FP32目录始终保留。

## 2. 获取代码和环境

```bash
# 使用项目负责人提供的仓库地址；不是重新克隆上游覆盖本项目
git clone --branch official_adjust_bottle_control <仓库地址> lingbot-baseline
cd lingbot-baseline
git rev-parse HEAD
cd lingbot-vla-v2
bash tools/create_train_env.sh --env-name lingbotvla
conda activate lingbotvla
```

训练环境直接复用[仓库安装说明](../../../README.md#installation)及[官方环境创建脚本](../../../tools/create_train_env.sh)：Python3.12、PyTorch2.8.0、flash-attn2.8.3。可按原说明传入匹配的FlashAttention wheel；Blackwell所用驱动/CUDA/PyTorch必须支持该卡，不能只看包安装成功。官方脚本也安装本地深度依赖。安装失败保留完整日志，先修复环境，不通过关闭视觉损失绕过。

RoboTwin复用仓库自带[仿真说明](../../../../RoboTwin-lingbot/README.md)、`RoboTwin-lingbot/script/_install.sh`及其中的资产下载步骤。建议单独建立RoboTwin环境，按该说明安装匹配的Python/SAPIEN/mplib/CuRobo；不能把仿真依赖直接混装进训练环境后假定兼容。下载脚本与安装脚本的相对路径按各自README执行。入口通过`--sim-python`选择仿真Python，模型服务仍使用启动时的训练Python。

Blackwell还必须按[本项目OIDN运行库说明](../evaluation/robotwin_render_runtime_zh.md)安装固定2.5.1及校验SHA；安装在**仓库根**的`.runtime/`，不是package目录。使用同仓库RoboTwin的运行库加载逻辑，保持SAPIEN/物理/相机配置。真实评测会检查OIDN标记及错误，不把颗粒噪声渲染当作正常结果。

数据下载、转换格式复用[RoboTwin数据准备说明](../../../experiment/robotwin/README.md)。这份历史manifest绑定了特定2500回合的编号/长度/任务分块；**仅仅同为50任务不够**。入口调用原数据检查器核对回合和总帧数，正式CPU门禁还检查实际取样。若使用自己重新采集/重编号的数据，应先重新生成并审查任务manifest，再发布另一版本基线，不直接套用本版。

### 资源目录约定

为只传三个必需路径，可以把附属资源准备在FP32模型目录下：

```text
fp32_base/
  config.json
  model.safetensors.index.json
  model-*.safetensors
  qwen/                         # Qwen3-VL-4B-Instruct本地处理器资源
  depth/moge2-vitb-normal.pt
  depth/model.pt
  dino_video/teacher_step_10000.pth
  dino_video/config.yaml
```

如果资源已经分开放置，不必复制：`--qwen-model`指定Qwen路径、`--assets-root`指定包含`depth/`和`dino_video/`的目录。归一化默认使用仓库`assets/norm_stats/robotwin_competition_clean.json`；`--norm-stats`可显式覆盖，覆盖即改变实验条件。当前入口只支持有索引的safetensors，不能直接传DCP或PyTorch `.bin`。

## 3. 冻结配置与课程

课程见[三阶段课程JSON](../../../configs/reproduction/three_stage_v1.json)，训练参数见[可移植BF16配置](../../../configs/reproduction/trainer_bf16.json)。配置只包含资源占位符，不包含原服务器账号或目录。历史最高成绩冻结到[最新成绩证据表](../../../refs/training_records/replay_pool_9680_through_recovery_round2.json)，含9680和本次已完成的两轮扫描，不在运行途中改Git中的参考表。

| 配置 | 默认值/口径 |
|---|---|
| 训练精度 | BF16主参数、BF16前向/梯度、BF16 HF；不是FP32主参数混合精度 |
| batch | micro=GBS56，GAS1，单卡；`--micro`可改，改后不是相同样本曝光量 |
| 训练器 | 全参数训练，GC开、compile开、数据workers8、seed3 |
| 学习率 | Muon；主LR 1e-4→5e-5，cosine；专家按sqrt(8)缩放。每个历史课程单独初始化优化器/调度，课程内部任务切换不重置 |
| 学习方式 | AL、窗口加权、EE-loss、低噪FM关闭；原FM方式 |
| 视觉loss | 深度/未来深度/未来视频各0.004，教师配置与历史BF16基准一致 |
| 数据 | 每任务全部50条，均匀帧采样，有放回；无独立开环验证集 |
| 保存 | 默认每个训练job内每2000更新＋最终；HF-only同步保存，不存优化器。`--save-every`可调，不自动清理 |
| 闭环 | FP32加载/推理、chunk50、noise seed0、demo_clean、各3回合、无视频、默认5路 |
| 关机 | 不自动关机 |

### 第一阶段：基础认知

50任务按JSON的`task_order`各100步，总5000更新，一个连续优化器日程。前42项按历史首次主训顺序；历史只评测未主训的任务附后，使用冻结证据表的首次出现顺序，没有虚构它们的首训时间。每任务50条数据，不因历史0/3跳过任何任务。

10%复习仅从已经完成100步的任务中取，排除当前主任务；一次一个复习任务，每5更新轮换。首任务没有已学池，100%主任务。GBS56使用5/6条累计分配，长期比例准确为10%，不是每步固定6条。5000步共280000样本，其中27440复习样本。默认保存2000、4000、5000。

### 第二阶段：技能验证与任务筛选

有效主链：`R3→R5→R6→R8→R9→R10→R11→R12→R13→R14→R15→R16`，共9150更新。R1/R2是独立对照，R4/R7没有被当前主链继承，因此排除。R17的25任务轮训移到第三阶段，不重复执行。当前临时三轮补课由第三阶段新规则替代，不额外叠加。

每轮保持历史任务顺序、步数、复习比例及冻结复习池（复用当前配方采样器，不声称与早期单任务shuffle逐位一致），训练结束后对该轮任务各3回合验收；可以记录0/3，不能用异常退出代替0/3。各轮结果写入`training_ledger.json`，第二阶段结束生成`stage2_selection.json`，列出现状与成功任务。它不擅自改变第三阶段固定25项覆盖名单。

| 历史课程 | 更新数 |
|---|---:|
| R3 / R5 / R6 | 200 / 200 / 500 |
| R8 / R9 / R10 | 600 / 500 / 800 |
| R11 / R12 / R13 | 1200 / 1350 / 1250 |
| R14 / R15 / R16 | 1350 / 400 / 800 |

每轮不足2000步仍保存最终HF，供下轮读取；所以默认2000并不代表第二阶段不存档。第二阶段最终检查与训练模型复用同一进程，记录零重载审计；新历史课程重新加载自己的来源模型，这与历史优化器重置一致。

### 第三阶段：遗忘检查与能力恢复

1. 固定历史曾有成功的25项（包括`place_a2b_right`历史1/4），三任务一组、每组3更新、均衡轮换，共1000更新；grouped本身就是复习，不另加10%。
2. 将25项固定分5批，各5项，每项3回合、不录像；每项恰扫描一次，不因补课而漏掉其它任务。
3. 每批以**当前成功率低于历史最高成功率**判断退步，用精确分数比较。参考最高值包含冻结历史表及本次第二阶段取得的更高成绩，绝不调低历史目标。
4. 退步任务各补100更新，10%单任务复习、每5步轮换、排除当前任务；补课完保存，再检查下一批。每批最多500更新，共最多2500补课更新。
5. 第五批需要补课也完成并保存，然后结束；不额外第六批，不自动复测补过的任务。

例如历史3/4→当前1/3需要补课，历史1/3→当前1/3不补，历史3/3→当前2/3需要补课。3回合有统计波动，这是一项工程规则，不能据此证明退步由某次混训造成。

每批检查共用一份FP32推理服务，并行至多5任务；如果两批之间补过课，下一批必须加载更新后的模型，不能沿用旧权重。五批来自不同中间模型，**不能将总和标成最终模型的25任务成功率**。最终文件会明确标注补课效果未复测。

## 4. 启动：先预览，再检查，再执行

以下命令均在`lingbot-vla-v2/`目录、训练Conda环境下执行。输出目录使用全新路径，建议日期时间前缀。

```bash
# 只打印课程预算，不加载模型、不创建输出目录
bash tools/reproduce_baseline.sh \
  --base-model /models/fp32_base \
  --dataset-root /datasets/RoboTwin_lerobot_v30 \
  --output-dir /outputs/20261022_1500_three_stage

# 环境/路径/数据/磁盘检查，不转换、不训练；GPU必须空闲
bash tools/reproduce_baseline.sh \
  --base-model /models/fp32_base \
  --dataset-root /datasets/RoboTwin_lerobot_v30 \
  --output-dir /outputs/20261022_1500_three_stage \
  --sim-python /opt/conda/envs/RoboTwin/bin/python --check-only

# 正式运行：后台控制器，启动命令返回PID与日志路径
bash tools/reproduce_baseline.sh \
  --base-model /models/fp32_base \
  --dataset-root /datasets/RoboTwin_lerobot_v30 \
  --output-dir /outputs/20261022_1500_three_stage \
  --qwen-model /models/Qwen3-VL-4B-Instruct \
  --assets-root /models/lingbot-vla-v2-6b \
  --sim-python /opt/conda/envs/RoboTwin/bin/python \
  --save-every 2000 --micro 56 --eval-workers 5 --execute
```

Qwen与教师资源采用上述默认布局、且同一个Python同时具备训练和仿真依赖时，只需三个必需路径；使用两个Conda环境则显式传`--sim-python`。`PYTHON=/绝对路径/python`可选定训练解释器。全流程每个训练job先真实CPU数据装配→独立短GPU探针→从该job来源模型正式训练，探针权重不进入正式链。

首步FP32→BF16转换逐CPU分片完成，重新读取输出并逐张量精确核对转换值，校验索引/张量头后原子发布`base_bf16/`。转换失败保留`.partial`供排查，不伪装完成、不删除输入。

默认预览为：5000＋9150＋1000＝**15150固定更新**，最多额外2500，共17650。默认16份正式HF、最多5份补课HF，再加转换副本。前置空间预算包含最坏5份补课，不因某批“可能不用补”降低安全门槛。

## 5. 显存与训练时间：实测与估算分开

| 历史证据 | 数值 | 适用范围 |
|---|---|---|
| R17两组混合GPU短探针，micro56/GAS1 | allocated77.70GiB，reserved86.04GiB | 96GB RTX PRO6000 Blackwell；短测，不是新5000步课程的全程峰值 |
| 最近补课micro56/GAS1三步探针 | allocated77.70GiB，reserved84.59GiB | 顺序主任务＋复习短测 |
| R17正式1000更新 | 81分22秒，平均4.88秒/更新 | 不含准备、保存和闭环，三任务grouped，不能保证单任务课程同速 |
| 最近第三轮顺序补课日志 | 单步约8.18～8.24秒 | 当前顺序数据路径的早期步时，不是整轮平均值 |

不要将nvidia-smi占用与allocated/reserved混为同一指标；编译缓存、教师前向、输入任务、进程阶段都会影响显存。启动探针会再次测实际峰值；探针失败即停止，不自动降低batch或修改课程。

用**5～8.2秒/更新作容量估算**（不是新入口实测）：

| 训练部分 | 步数 | 纯更新估时 |
|---|---:|---:|
| 基础认知 | 5000 | 6.94～11.39小时 |
| 技能验证与任务筛选 | 9150 | 12.71～20.84小时 |
| 遗忘检查的分组轮训 | 1000 | 1.39～2.28小时 |
| 合计固定训练 | 15150 | **21.04～34.51小时** |
| 最坏补课额外 | 2500 | 3.47～5.69小时 |

还需加下载、FP32转换、每轮CPU/GPU门禁、加载、编译、保存和闭环时间。历史5任务×3回合一次约6分10秒，任务不同耗时不同；第二阶段有12次阶段验收，第三阶段5批，不能只按一次6分钟预算整个流程。尚无新流程完整墙钟实测，不承诺一天完成；实际时间以日志首末优化器更新与控制器终态分别统计。

## 6. 看进度、拿模型、恢复

```bash
cat /outputs/20261022_1500_three_stage/status.json
cat /outputs/20261022_1500_three_stage/final_model.json
# 训练中日志在具体job下，例如：
tail -n 30 /outputs/20261022_1500_three_stage/stage1_foundation/train.log
```

| 产物 | 内容 |
|---|---|
| `requested_settings.json` / `settings.json` | 用户提供的资源路径与参数 |
| `environment.json` | Python/包版本、GPU/驱动、Git SHA及输入配置/归一化哈希 |
| `execution_plan.json` | 冻结完整课程、保存步号、25项扫描覆盖 |
| `base_bf16/conversion_complete.json` | 转换分片与张量检查 |
| `stage*/recipe.json`、`step_plan.json`、`trainer_config.json` | 每轮实际采样及训练参数 |
| `stage*_pipeline/status.json` | CPU、GPU探针、训练、评测终态 |
| `training_ledger.json` | 已完成课程与逐任务评测，模型来源链 |
| `stage2_selection.json` | 第二阶段任务现状及筛选记录 |
| `stage3_scan_N/summary.json` | 该批5项原始闭环成绩，N为1～5 |
| `recovery_ledger.json` | 每批退步原因与补课计划 |
| `final_model.json` | 全流程成功后生成的最终HF路径及检查报告 |

**只在`status.json`为complete且`final_model.json`存在时交付最终模型**。训练日志结束不等于HF完整，更不等于闭环通过。TensorBoard可读取输出根下各job的`runs/`；入口不隐式创建固定端口服务。

恢复时用相同参数追加`--resume --execute`：仅跳过已经完整成功的训练job及扫描，不重复训练成功阶段。**不支持任意中途优化器续训**：HF-only没有优化器/RNG状态，未完成job会明确报错并保留产物。转换半成品、失败或中断阶段需人工检查后另开输出路径，不能将HF加载说成精确恢复。SIGKILL留下的`controller.lock`需先确认记录PID和所有子进程已退出，再手动移除；活跃控制器时禁止重复启动。

每轮确认结束后，按照[训练记录规范](../README.md)整理结果与存档，不把计划写成实绩。入口自动写机器可读流水，不自动git提交/推送、不删除存档、不关机。原有训练和AL入口不变。

## 7. 维护与验收入口

```bash
python -m unittest experiment.reproduction.test_contracts
python -m unittest discover -s experiment/task_recipes -p 'test_*.py'
```

CPU测试覆盖50任务唯一覆盖、历史分支排除、5000/9150/1000预算、默认/自定义存档、第一任务无复习、只复习已完成任务、GBS守恒、25项五批不遗漏、模拟全链补课及阶段边界恢复。它们不证明真实FP32权重已转换或96GB端到端可用。

另有服务器微型safetensors CPU转换检查通过：浮点精确等于源转BF16、整数不变、输入FP32不变、完成副本复用，不代表完整6B模型已转换。

上机验收需在GPU空闲且明确授权后进行：完整模型FP32→BF16精确转换、真实2500回合CPU装配、micro56三步短测、训练后驻留评测、五批实际扫描和最终HF完整性。本次实现不启动第二份付费训练来冒充验证。
