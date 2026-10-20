# 参数化训练配方框架

目的：换主学习任务、复习任务、比例、各任务步数、存档及评测列表时只改配置，不再另写任务脚本。
这是标准BF16训练路线的新可选入口，不替换生产AL调度，也不修改正在运行的单任务实验。

## 使用入口

在仓库`lingbot-vla-v2/`下，训练环境Python：

```bash
# 默认预览：解析/检查任务、预算与保存评测计划，不创建run，不加载模型
python experiment/task_recipes/run.py --recipe configs/training_recipes/three_main_with_replay_400.json
# 用户授权运行后，指定新目录并显式执行
python experiment/task_recipes/run.py --recipe configs/training_recipes/three_main_with_replay_400.json --out /data/outputs/my_recipe_run --execute
# 从训练日志的已完成优化器步读取每任务进度，不看预取游标
python -m experiment.task_recipes.status --out /data/outputs/my_recipe_run
```

示例配置：

- `configs/training_recipes/place_container_plate_200.json`：单任务200步，无复习。
- `configs/training_recipes/three_main_with_replay_400.json`：三个主任务150/150/100步，20%复习，复习内adjust_bottle:click_bell=3:1。
- `configs/training_recipes/sequential_plate100_cup200_pot200.json`：plate100→cup200→pot200连续阶段，共500步。

JSON仅放在configs，冻结产物放run目录，说明放docs。所有相对路径按package根解析，输出目录拒绝覆盖。
第一次执行新配置仍必须通过真实CPU装配与三步GPU探针，不能把配方预览当GPU验收。

## 三种模式与比例口径

| 配置 | 含义 |
|---|---|
| `train.mode: mixed` | 所有主任务按weight分配主学习样本；只设全局steps，禁止给任务写独立优化器步数 |
| `train.mode: task_units` | 每次更新选一个主任务，用该任务steps作为主导更新配额；总步数是各主任务配额之和 |
| `train.mode: sequential` | 按task_order顺序连续完成各任务steps；不交错轮转，主任务weight仅接受默认1 |
| `main_tasks.<task>.weight` | mixed模式是样本配比；task_units模式仅影响轮转顺序，最终步数仍由配额决定 |
| `replay.fraction` | 复习样本占整个global batch的比例，不是复习优化器步比例 |
| `replay.tasks.<task>.weight` | 仅在复习池内部归一化；如20%×3/4=总样本的15% |
| `episode_split: all/train` | 使用每任务全部轨迹，或仅manifest的train集合；all不具备独立开环验证集 |

示例主配额150/150/100：总400个优化器更新，每步主要训练被选中的一个主任务，并混入复习样本。
达到主任务配额后不再选它为主任务；**不是冻结模型中该任务的能力**，其他任务更新仍作用于同一模型。
主任务与复习任务列表必须互斥。第一版不自动把完成任务加入复习池，不自动调整比例、判断PASS或早停。

GBS=56且复习20%时，复习数量在11/12之间按累计配额分配，400步合计4480个复习样本，
adjust_bottle3360、click_bell1120。每步不是独立四舍五入，避免长期比例偏差。
任务内从真实过滤后的帧索引均匀、有放回抽样；这是明确的配方采样方式，不能宣称与旧单任务shuffle的RNG序列相同。
不会应用窗口、难度权重、EE或低噪声FM。

## 配置字段

- `source.trainer_config`：完整trainer配置或包含trainer_config的基准快照；继承LR、GC、compile、workers、视觉损失等资源/优化配置。
- `source.checkpoint`：精确HF目录。加载权重后新优化器、新调度、新预算，不把来源step累计进当前配额。
- `data.dataset_root`、`output_dir`：数据及新输出目录。
- `train`：mode、main_tasks、replay、steps（mixed必填，其余由配额推导）、micro_batch_size、gradient_accumulation_steps、seed、episode_split；sequential额外要求task_order。
- `checkpoint.every_steps/save_final/hf_dtype`：显式保存约定；第一版要求save_final=true、hf_dtype=bf16。
- `evaluation.tasks/episodes_per_task/checkpoints/shared_model_session`：评测任务、回合数、已保存步号或final、跨任务是否共用服务。
- `runtime.shutdown_after_completion`：第一版只接受false，不默认操作关机。

未知字段、任务不存在、主复习重叠、配额与总步数冲突、无主任务样本、评测未保存的步号都会报错，不静默猜测。
配置是任务运行的显式约定；保存间隔/保留策略更改仍需用户允许，不因容量不足自行删改。

## 分层与运行流程

- `schema.py`：纯CPU契约与字段消歧义。
- `planning.py`：确定性步计划、任务轮转/复习数量、原始数据索引映射、采样状态。
- `prepare.py`：检查来源/磁盘，冻结recipe、step_plan、task_groups、train_ids、trainer_config和哈希。
- `runtime.py`：只在新配方进程安装数据适配器，其他入口默认数据加载器不变。
- `cpu.py`：真实轨迹边界取样与StatefulDataLoader装配，验证GAS和完整预算长度。
- `run.py`：CPU→三步GPU→正式训练→校验全部HF→指定模型的共享服务闭环。
- `status.py`：从已完成优化器步日志计算主任务已用配额/剩余量、各任务样本数和复习量。

模型核心、训练器前后向及损失未改。仅新配方入口使用虚拟epoch长度，等于总步数×GBS；
采样索引始终保留在真实过滤数据空间，**虚拟长度不是扩充了数据量**。
这样整个预算落在一个数据epoch内，避免短数据集跨epoch时丢弃预取批次、导致任务配额错位。
严格读取样本；读取失败直接报错，不允许旧重试逻辑随机换到另一任务破坏比例。

`recipe.json/step_plan.json`等冻结文件与哈希不一致时拒绝训练；采样器预取cursor仅用于DataLoader装配，
不计入完成步数。用户进度以训练器更新后输出的日志步号为准。

## 恢复、资源与边界

第一版仅支持单卡标准BF16参数/计算/HF，rmpad关闭，无AL/EE/LN-FM/LoRA/专家独训。
多卡、FP32路线、自动成功率门控尚未接入，不会把它们当作已支持。
采样器具备状态校验/恢复契约，但**HF-only没有优化器和完整恢复状态**；当前入口不支持同run精确Resume。
从某份HF再训练是新run，预算重新计算，不能把采样cursor恢复测试当作完整训练Resume已实现。

GPU忙时拒绝启动；容量按全部显式HF里程碑加余量预检，不减少存档或自动删除历史。
评测相同模型下的任务共用一次服务，任务/回合由已验证客户端reset清空动作缓存；换checkpoint才新载模型。
每任务保留精度、渲染、视频和结果校验。场景种子按固定seed0可行性筛选并记录，不把train loss当闭环能力。

## 验证状态与回滚

```bash
python -m unittest experiment.task_recipes.test_contracts -v
```

本地/服务器初始12项纯CPU契约通过，加入顺序阶段后为13项；覆盖精确步数/复习配比、任务权重、冲突拒绝、
采样恢复、索引空间/虚拟长度、预取不算完成、冻结计划校验、只读预览及100/200/200阶段边界。
真实GAS2单任务CPU装配已通过；顺序模式150条24119帧、500步虚拟epoch真实CPU与三步GPU通过，正式训练进行中。
主任务加复习的五任务组合尚未完成真实数据装配验收；每次新配方仍强制执行CPU与GPU前置检查。
原延后验证报告保留在`/data/outputs/task_recipe_framework_validation_20261020.json`，其中failed是旧GAS1检查器失败记录；
修复后实际重试证据为`recipe_framework_cpu_20261020_single/cpu_report.json`和顺序run的cpu_report/probe报告。
新框架当前没有完整500步/闭环终态；不把三步短测或旧single_task_trial结果当作新框架训练完成。

不变量：配方显式预算/比例/存档、原始数据索引、无预取虚报、旧入口不变；
症状：冻结哈希、配额、dtype、DataLoader长度/状态或HF校验失败；
回滚：不用`experiment/task_recipes/run.py`，原单任务和AL入口继续独立使用；新模块只在新进程读取，不能热改当前run。

顺序模式的当前实验与来源、250/500存档约定见[顺序续训说明](sequential_plate_cup_pot_zh.md)。
