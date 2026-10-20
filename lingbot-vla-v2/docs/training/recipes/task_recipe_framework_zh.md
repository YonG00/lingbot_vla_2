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

顺序模式的当前实验与来源、250/500存档约定见[顺序续训说明](../experiments/multi_task/sequential_plate_cup_pot_zh.md)。

### 两种存档模式

新run目录统一采用`YYYYMMDD_HHMM_说明`，北京时间，例如`20261020_2002_switch_pillbottle_stapler`。
统一CLI自动为未带前缀的output_dir/--out补前缀，并移除原末尾日期；显式已有前缀原样保留。
同分钟同名仍拒绝覆盖。补课子run也采用时间前缀；pipeline和TensorBoard入口保留该前缀。
此规则只作用于新启动目录，来源模型、历史run及当前运行目录不重命名。

`checkpoint.mode`默认为`steps`，保留旧配方行为：

```json
"checkpoint": {"mode": "steps", "every_steps": 250, "save_final": true, "hf_dtype": "bf16"}
```

顺序课程也可选择`task`：每个任务完成后、下一任务第一步之前保存，最终任务结束也保存。

```json
"checkpoint": {"mode": "task", "save_final": true, "hf_dtype": "bf16"}
```

任务模式只支持sequential，不接受every_steps，避免两种规则混用。
例如100→200→200的课程保存100/300/500；200→200→200保存200/400/600。
prepare生成`save_steps=0, save_at_steps=[...]`；训练器通过纯CPU的checkpoint_due判定，
仍走原HF-only保存与容量检查，末步保存去重沿用原逻辑。
补课继承存档模式：任务模式按每个补课任务完成点保存，不应用300步仅最终保存的步数模式例外。
三步GPU探针禁存档，显式步号列表也不得触发保存。
开关与补课组合的四格均有CPU契约；真实200步任务边界落盘尚待本轮训练验证。

本轮新课程：[switch/pillbottle/stapler 600步配置](../../../configs/training_recipes/sequential_switch_pillbottle_stapler_600.json)。
来源严格为R6-step500，三任务各200步，GBS56/BF16，最终三个新任务各3回合。
用户因容量选择先训主课程、暂缓补课，当前配置remediation.enabled=false；不删除旧存档。

### 可选闭环补课

配方顶层可添加下面的配置；省略时默认关闭，`max_rounds`默认1：

```json
"remediation": {"enabled": true, "max_rounds": 1}
```

开启前要求：复习比例为0、评测任务非空、每任务3回合，并包含最终存档评测。
当前规则（2026-10-20修订）：只使用最终模型完整有效结果，**1/3和2/3各补100步；0/3和3/3不补课**。
2/3优先，再补1/3；同分任务保持评测任务列表顺序；所有任务都无需补课时直接结束。
测评缺失、重复、模型路径不符或执行异常均中止，不作为0/3。

每轮补课生成一份顺序课程表，从上一轮最终HF续训，沿用batch、精度及存档间隔，复习关闭。
用户授权例外：补课总计300步且原保存间隔为250时，自动改为只存最终300；其余节奏沿用。
该规则启动时读取，不热改已经运行的训练。
补课是独立训练进程，会重新初始化优化器和学习率日程；同一轮内部各任务之间连续训练。
默认不填写`evaluation.tasks`，从本轮主训练任务自动生成评测列表，sequential按课程顺序评测。
补课子run只评本轮实际补训的任务，不再继承未补训任务。
需要额外检查旧能力时，可显式填写`evaluation.tasks`覆盖主课程默认列表；这是配置选择，不是代码写死。
达到轮次上限即结束，即使仍有失败也不继续。
所有轮次沿用现有评测种子逻辑，没有新增独立种子组，不将三回合结果视为稳定掌握的证明。

原训练开始前按“全部评测任务均需100步补课×最大轮数”预估额外存档空间；每轮启动再次检查，
容量不足报错，不删除存档、不减少间隔。每轮含沿用间隔产生的中间存档及最终存档。
新补课子目录为`YYYYMMDD_HHMM_remediation_round_N`；课程表为`remediation_round_N_plan.json`；
最终`remediation_summary.json`和pipeline状态保留各轮评测结果。
每轮照常执行CPU检查、三步GPU检查、正式训练、HF校验、闭环评测。

验证：`python3 -m unittest experiment.task_recipes.test_remediation`。
历史R7已按旧规则完成真实GPU补课及三任务闭环，全部0/3后用户放弃该模型。
本次“1/3与2/3补课，跳过0/3”修订只做本地纯CPU契约验证；没有热更新当前服务器训练，R8仍关闭补课。
