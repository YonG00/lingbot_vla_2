# 三阶段基线复现与交接

本流程使用单张 **NVIDIA RTX PRO 6000 Blackwell 96GB**，GBS=56、梯度累积=1，从FP32 base开始训练50个RoboTwin任务。按每步4.88秒估算，固定15150步约需 **20.5小时**；任务抽检后的补训最多增加约3.4小时，另加模型准备、保存和评测时间。

效果目标是让25个曾通关任务恢复到参考成绩附近，并减少顺序训练带来的遗忘。实际达到的成功率以训练结束后的统一评测为准。

## 目录

1. [复现说明](#1-复现说明)
2. [复现环境](#2-复现环境)
3. [下载代码](#3-下载代码)
4. [准备模型和数据](#4-准备模型和数据)
5. [配置训练和仿真环境](#5-配置训练和仿真环境)
6. [启动训练](#6-启动训练)
7. [三个阶段分别训练什么](#7-三个阶段分别训练什么)
8. [训练时间、显存和效果](#8-训练时间显存和效果)
9. [查看进度和获取最终模型](#9-查看进度和获取最终模型)
10. [常见问题与联系方式](#10-常见问题与联系方式)

## 1. 复现说明

参考目标是 `20261022_1209_round_3_recovery/global_step_160`。本流程为整理后的三阶段方案，与历史最后几轮补训步数存在差异，详见[目标模型来源](../records/recovery160_origin_zh.md)。完整三阶段GPU重训尚未端到端验证，不能保证得到同一权重或成功率。只涵盖三阶段，不含其后的专项训练。

独立单任务与整体评测用[统一并行评测入口](benchmark_zh.md)，默认每任务3回合并行、不录像。

整个流程分为三个阶段，启动后按顺序执行：

| 阶段 | 训练内容 | 更新步数 |
|---|---|---:|
| 基础认知 | 从base开始，50个任务依次各训100步，带10%复习 | 5000 |
| 技能验证与任务筛选 | 按已选好的任务顺序与配方继续训练，不做闭环评测 | 9150 |
| 遗忘检查与能力恢复 | 25个任务先混训1000步，再分5批进行任务抽检并补训 | 1000＋任务抽检并补训 |

固定训练共15150步。第三阶段每个退步任务补训100步，最多增加2500步。任务表见[第二阶段训练顺序与更新数](stage2_training_order_zh.md)。

你需要准备：一台符合要求的GPU服务器、FP32 base模型、训练数据、模型所需的附属文件，以及保存结果的空间。程序会自动调用`experiment/reproduction/run.py`中的`convert()`，用PyTorch在CPU上逐分片将FP32 base转换为BF16副本，再开始训练。原FP32模型不变；副本保存到本次输出目录的`base_bf16/`，并检查转换后的文件和数值。三个阶段结束后，`final_model.json`会给出最终模型的位置。

## 2. 复现环境

请使用以下显卡和训练配置复现本文流程：

| 项目 | 配置 |
|---|---|
| GPU | **单张NVIDIA RTX PRO 6000 Blackwell，96GB显存** |
| 系统 | Linux，安装支持该显卡的NVIDIA驱动、CUDA和Vulkan运行环境 |
| 主机内存 | 建议128GB，用于加载数据和在CPU上转换模型 |
| 输出磁盘 | 至少预留320GiB空闲空间；模型、数据和环境另计 |
| 训练环境 | Python3.12、PyTorch2.8.0、FlashAttention2.8.3 |
| 训练batch | micro batch=56，global batch=56，梯度累积=1 |
| 训练精度 | BF16主参数＋BF16计算，保存BF16模型 |
| 仿真评测 | RoboTwin，FP32模型推理，默认最多5个任务并行 |

这里的显卡是**RTX PRO 6000 Blackwell 96GB**，不是RTX A6000 48GB。下文的显存与速度均来自96GB这张卡。

这里没有额外保留一套FP32主权重。训练采用BF16参数和计算；闭环评测默认将保存的BF16权重加载成FP32计算。参数配置为`enable_mixed_precision=false`、`enable_fp32=false`、`action_fp32=false`。加载成FP32不会恢复BF16转换时舍去的精度。

## 3. 下载代码

- 仓库：[YonG00/lingbot_vla_2](https://github.com/YonG00/lingbot_vla_2)
- 复现分支：`main`
- 训练代码起源（不含新评测入口）：[cac7e24c856ac8a0b6c42379f7324941b3788a48](https://github.com/YonG00/lingbot_vla_2/tree/cac7e24c856ac8a0b6c42379f7324941b3788a48)

```bash
git clone --branch main \
  https://github.com/YonG00/lingbot_vla_2.git lingbot-baseline
cd lingbot-baseline
git checkout reproduction-main-20261025
git rev-parse HEAD
```

后续的训练命令在`lingbot-vla-v2/`目录执行。仿真代码位于同一仓库的`RoboTwin-lingbot/`，请使用这份代码配置仿真环境。

## 4. 准备模型和数据

### 4.1 下载base和附属模型

第一阶段从**FP32 base模型**开始，不需要先提供已经训练过的任务模型。模型下载方法见[原项目模型准备说明](../../../README.md#model-download)。

除base权重外，还需要以下文件；下载入口和说明见[原项目训练说明](../../../README.md)：

| 文件 | 用途 |
|---|---|
| Qwen3-VL-4B-Instruct | 处理图像和任务文本 |
| MoGe-2、LingBot-Depth权重 | 深度监督 |
| DINO-VIDEO权重与配置 | 视频监督 |

base目录需要包含`config.json`、`model.safetensors.index.json`和索引对应的全部权重分片。当前启动程序读取FP32 safetensors模型，不直接读取DCP或`.bin`文件。

### 4.2 模型文件放在哪里

下面是一种可直接使用的放置方式：

```text
/models/fp32_base/
├── config.json
├── model.safetensors.index.json
├── model-*.safetensors
├── qwen/                           # Qwen3-VL-4B-Instruct文件
├── depth/
│   ├── moge2-vitb-normal.pt
│   └── model.pt
└── dino_video/
    ├── teacher_step_10000.pth
    └── config.yaml
```

也可以分开放置。启动时用`--qwen-model`指定Qwen目录，用`--assets-root`指定包含`depth/`和`dino_video/`的目录，不必重复复制这些文件。

### 4.3 准备训练数据

使用RoboTwin **50个任务、每任务50条clean示范，共2500条轨迹**，转换为LeRobot v3格式。下载与转换步骤见[RoboTwin数据准备说明](../../../experiment/robotwin/README.md)。

启动参数`--dataset-root`指向转换后的数据根目录，目录下应能看到`meta/`、`data/`和视频文件。

本版任务划分按现有2500条数据的episode编号读取。更换存放路径没有影响；如果重新采集、合并或重排过轨迹，需要重新生成任务划分文件，不能只改数据路径。运行前程序会检查回合数、编号和总帧数。

归一化文件默认使用仓库内的`assets/norm_stats/robotwin_competition_clean.json`。

## 5. 配置训练和仿真环境

### 5.1 训练环境

在仓库的`lingbot-vla-v2/`目录执行：

```bash
cd lingbot-vla-v2
bash tools/create_train_env.sh --env-name lingbotvla
conda activate lingbotvla
```

安装脚本和可选参数见[原项目安装说明](../../../README.md#installation)。已有匹配的FlashAttention wheel时，可通过`--flash-attn-wheel`指定文件，减少安装时间。

### 5.2 仿真环境

按[RoboTwin安装说明](../../../../RoboTwin-lingbot/README.md)配置仿真环境，并下载机器人和物体资产。仓库内提供`RoboTwin-lingbot/script/_install.sh`，安装步骤按该项目说明执行。

训练与仿真建议使用两个Conda环境。后面的`--sim-python`需要填写**仿真环境Python的完整路径**，例如：

```text
/opt/conda/envs/RoboTwin/bin/python
```

训练程序使用当前激活的`lingbotvla`环境。评测时，模型服务使用训练环境，物理仿真使用`--sim-python`指定的环境。

### 5.3 安装渲染去噪库

RTX PRO 6000 Blackwell使用OIDN 2.5.1。安装步骤见[OIDN安装说明](../evaluation/robotwin_render_runtime_zh.md#安装与启用)，其中包含下载地址和文件校验方法。

文件放在仓库根目录的`.runtime/`下，与`lingbot-vla-v2/`和`RoboTwin-lingbot/`并列。评测会检查渲染错误；若出现OIDN错误，请先完成这一步再评测。

## 6. 启动训练

先激活训练环境，进入`lingbot-vla-v2/`目录，再把下面的路径换成自己的路径：

```bash
conda activate lingbotvla

OUTPUT_DIR="/outputs/$(date +%Y%m%d_%H%M)_robotwin_50tasks_three_stage"

args=(
  --base-model /models/fp32_base
  --dataset-root /datasets/RoboTwin_lerobot_v30
  --output-dir "$OUTPUT_DIR"
  --qwen-model /models/Qwen3-VL-4B-Instruct
  --assets-root /models/lingbot-vla-v2-6b
  --sim-python /opt/conda/envs/RoboTwin/bin/python
  --micro 56
  --save-every 2000
  --eval-workers 5
)

# 查看任务、训练步数和保存计划
bash tools/reproduce_baseline.sh "${args[@]}"

# 检查文件、环境、数据和磁盘；此时GPU需空闲
bash tools/reproduce_baseline.sh "${args[@]}" --check-only

# 检查通过后开始运行
bash tools/reproduce_baseline.sh "${args[@]}" --execute
```

`--output-dir`没有内置默认值，需要自行指定。上面用当前日期时间加`robotwin_50tasks_three_stage`命名，例如`20261022_1530_robotwin_50tasks_three_stage`。

正式启动后，程序转到后台运行，终端会显示进程PID和日志路径。每批训练前会检查数据，并进行短步训练测试；测试通过后才执行正式训练。

常用参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--base-model` | FP32 base模型目录 | 必填 |
| `--dataset-root` | LeRobot v3数据目录 | 必填 |
| `--output-dir` | 本次结果保存目录，首次运行需使用新目录 | 必填 |
| `--save-every` | 每隔多少个更新步保存一次 | 2000 |
| `--micro` | 单卡batch大小；本配置不做梯度累积 | 56 |
| `--eval-workers` | 同时评测的任务数 | 5 |
| `--qwen-model` | Qwen模型文件目录 | base目录下的`qwen/` |
| `--assets-root` | 深度和视频教师文件所在目录 | base目录 |
| `--sim-python` | 仿真环境的Python路径 | 当前Python |

每批训练结束时也会保存最终模型，即使没有达到2000步。例如，一批训练500步，会在500步保存，供下一批继续训练。程序不自动删除旧存档，也不自动关机。

## 7. 三个阶段分别训练什么

### 第一阶段：基础认知

从base出发，50个任务按顺序各训100步，总计5000步，任务之间连续训练。

每个batch中，90%样本来自当前任务，10%用于复习已经学过的任务。一次复习一个旧任务，每5步切换。第一个任务还没有旧任务可复习，因此全部使用当前任务数据。

第一阶段完成后，得到学过全部50个任务的模型，再进入第二阶段继续训练。50项的具体顺序保存在[课程配置](../../../configs/reproduction/three_stage_v1.json)的`task_order`中。

### 第二阶段：技能验证与任务筛选

按照[第二阶段训练顺序与更新数](stage2_training_order_zh.md)逐项训练，共12批、9150步。这一阶段直接使用已经选好的配方，**不运行闭环评测**。

所有批次统一10%复习。从第一阶段已经学过的50个任务中均匀轮换，一次一个复习任务，每5步切换，排除当前主训练任务。

每批训练结束后保存模型，并作为下一批的起点。开始下一批时重新建立优化器和学习率计划；同一批内切换任务时连续训练。

### 第三阶段：遗忘检查与能力恢复

**先混合训练1000步，再进行任务抽检并补训。**

1. 从参考成绩表中取25个曾通关的任务。每组3个任务，组内混合取样，连续训练3步后换组，均衡轮换到1000步。
2. 将这25个任务分为5批，每批5个，每项评测3回合，不录像。
3. 将本次成功率与该任务的最高成功率比较。比最高成绩低，就补训100步；不低则跳过。补训带10%复习，每5步切换复习任务。
4. 本批补训结束并保存后，再评测下一批。总共检查5批，覆盖全部25个任务。
5. 第五批补训结束后保存最终模型，训练结束。

例如，最高成绩3/3，本次2/3，需要补训；最高成绩1/3，本次仍为1/3，不补训。参考最高成绩直接使用项目提供的成绩表，运行途中不修改，也不需要第二阶段重新测一遍。

每批5个任务共用一个模型服务。两批之间若做过补训，下一批使用补训后的新模型。参考成绩表见[25任务成绩记录](../../../refs/training_records/replay_pool_9680_through_recovery_round2.json)。

### 共同的训练设置

| 配置 | 设置 |
|---|---|
| 训练方式 | 全参数训练，开启梯度检查点和编译，8个数据加载进程 |
| 随机种子 | 3 |
| 优化器与学习率 | Muon，主学习率从1e-4降至5e-5，cosine调度；专家学习率按sqrt(8)缩放 |
| 数据采样 | 使用每任务全部50条示范，均匀、有放回取样 |
| 动作与视觉监督 | 标准FM；深度、未来深度、未来视频loss权重各0.004 |
| 关闭的附加方法 | 窗口加权采样、EE-loss、低噪FM、AL自动调度 |
| 模型保存 | BF16 HF权重，不保存优化器状态 |
| 评测设置 | FP32推理、action chunk=50、noise seed=0、demo_clean |

### 与官方代码相比改了什么

网络仍采用LingBot-VLA 2.0的主干、动作专家和深度/视频监督模块；这套复现流程没有新增网络层或更改动作输出维度。主要增加的是训练和评测的组织方式：

| 部分 | 本项目的调整 |
|---|---|
| 数据采样 | 按任务顺序训练，混入10%旧任务样本；第三阶段按三任务一组混训 |
| 训练流程 | 自动衔接三个阶段，按指定步数保存模型 |
| 遗忘检查 | 使用固定的最高通关表，每批抽检5个任务，再补训退步任务 |
| 评测服务 | 一个模型服务连接多个并行仿真任务，支持保存视频 |
| 其它实验功能 | 仓库中保留了EE-loss、窗口加权、低噪FM等开关，本次复现全部关闭 |

模型核心文件按本项目起始快照`86f6728`固定，启动时会检查文件是否被改动。这里说明的是本流程采用的模型与训练方式，不表示本仓库与官方最新版本的所有文件完全相同。

## 8. 训练时间、显存和效果

### 显存占用

显卡标称容量为96GB。下面的86.04GiB是一次测试中PyTorch预留的显存，不是显卡的总容量。

| batch56训练测试 | 张量实际分配 | PyTorch预留（含可复用缓存） |
|---|---:|---:|
| 三任务混合训练 | 77.70GiB | 86.04GiB |
| 单任务加复习 | 77.70GiB | 84.59GiB |

预留显存中有一部分是缓存，驱动等开销还会占用其它显存。因此显卡不需要、也不应为了训练而刻意占满96GB。正式训练前，程序会测试当前配置能否运行。

### 需要训练多久

统一按**每个更新步4.88秒**估算：

| 内容 | 更新步数 | 训练时间估算 |
|---|---:|---:|
| 基础认知 | 5000 | 约6小时47分 |
| 技能验证与任务筛选 | 9150 | 约12小时24分 |
| 第三阶段混训 | 1000 | 约1小时21分 |
| 固定训练合计 | 15150 | **约20小时32分** |
| 任务抽检后补训最多增加 | 2500 | 约3小时23分 |
| 包含最多补训的训练合计 | 17650 | **约23小时56分** |

模型转换、加载、编译、保存和闭环评测不计入上述训练时间。第二阶段不评测；第三阶段需要完成五批抽检。实际总耗时会比表中训练时间长。

### 训练效果怎么看

第三阶段按项目提供的最高成绩检查25项任务，并对退步项补训。程序和配置已经过CPU检查；从原始base开始执行这套完整三阶段流程后的成功率尚待实际评测。

第三阶段会边抽检边补训，因此五批成绩可能来自不同模型。要知道最终模型的统一成功率，可以按下一节的录像评测命令，对最终模型再测一轮。

## 9. 查看进度和获取最终模型

以下使用启动时设置的`OUTPUT_DIR`。换一个终端后，请先把它设置为本次训练的实际目录：

```bash
# 查看当前进行到哪一步
cat "$OUTPUT_DIR/status.json"

# 全部结束后查看最终模型的位置
cat "$OUTPUT_DIR/final_model.json"

# 查看第一阶段训练日志
tail -n 30 "$OUTPUT_DIR/stage1_foundation/train.log"
```

常用结果文件：

| 文件 | 可以看到什么 |
|---|---|
| `environment.json` | 运行时的代码版本、GPU、驱动和Python包版本 |
| `execution_plan.json` | 各批训练任务、更新步数和存档位置 |
| `training_ledger.json` | 已完成的训练和评测 |
| `stage2_recipe_complete.json` | 第二阶段配方已执行完成，不包含闭环评测成绩 |
| `stage3_scan_1/summary.json`至`stage3_scan_5/summary.json` | 第三阶段五批评测成绩 |
| `recovery_ledger.json` | 哪些任务需要补训 |
| `final_model.json` | 最终模型目录 |

`status.json`显示`complete`后，读取`final_model.json`中的`path`即可找到最终HF模型。TensorBoard日志位于各批训练目录下的`runs/`。

### 训练完成后评测，并保存视频

下面使用最终模型，默认对参考表中的25个任务各测3回合，并保存全部3回合视频。仍在训练环境、`lingbot-vla-v2/`目录执行：

```bash
EVAL_DIR="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M)_eval_video"

python experiment/reproduction/evaluate.py \
  --run-dir "$OUTPUT_DIR" \
  --output-dir "$EVAL_DIR" \
  --episodes 3 --video-episodes 3 --workers 5 \
  --execute
```

程序从训练记录中读取最终模型、Qwen和仿真Python路径。也可以只测指定任务，在命令中加入：

```bash
--tasks click_bell turn_switch
```

若要评测单独保存的HF模型，使用以下命令：

```bash
python experiment/reproduction/evaluate.py \
  --model-path /models/my_model/hf_ckpt \
  --qwen-model /models/Qwen3-VL-4B-Instruct \
  --sim-python /opt/conda/envs/RoboTwin/bin/python \
  --output-dir /outputs/my_model_eval_video \
  --tasks click_bell turn_switch \
  --episodes 3 --video-episodes 3 --workers 5 \
  --execute
```

评测采用FP32推理，模型只加载一次，多个任务分批并行。`--execute`启动后台评测；去掉该参数可以先看任务和视频保存设置。结果和视频在：

```text
<评测输出目录>/parallel_5/summary.json
<评测输出目录>/parallel_5/<任务名>/results/<任务名>/episode*_global.mp4
```

`parallel_5`对应`--workers 5`。改为其它并行数时，目录后缀相应改变。`--video-episodes 0`表示不录像。

## 10. 常见问题与联系方式

**提示找不到模型文件或数据怎么办？**

检查启动命令中的路径。base、Qwen、深度教师和视频教师都需准备；Qwen与教师分开放置时，请填写对应参数。

**中断后能继续吗？**

原命令追加`--resume --execute`，会跳过已经完成的训练批次和评测。批次训练到一半时中断，当前程序不能从该批的优化器状态继续；请保留日志后联系维护者处理。

**显存不足怎么办？**

先确认GPU为96GB目标型号，且没有其它训练或评测进程。仍不足时，保留出错日志及短测结果，再考虑调小`--micro`；修改batch会改变每步的训练样本数。

**评测出现渲染错误怎么办？**

检查OIDN 2.5.1是否按第5节安装，以及机器人和物体资产是否完整。

**联系谁？**

- 项目问题：[GitHub Issues](https://github.com/YonG00/lingbot_vla_2/issues)
- 项目负责人电话：待补充
- 项目负责人微信：待补充

反馈时请附代码版本、GPU型号、启动命令、出错日志和相关`status.json`；模型效果问题再附任务名及评测结果。

## 源码一致性检查

训练前在包目录运行 `python experiment/reproduction/verify_source.py`，验证冻结训练源码与课程输入。记录 `git rev-parse HEAD`、环境版本、模型和数据来源；不要只记录浮动的main分支名。源码锁不包含大模型/训练数据内容，不能替代数据校验。
