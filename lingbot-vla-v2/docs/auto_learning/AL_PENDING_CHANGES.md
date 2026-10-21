# AL 改动台账（PENDING CHANGES）

> **为什么有这个文件**：改动会持续积累，很多功能**改了但没上机验证**（因为后台在跑训练）。
> 一旦上机报错，需要能立刻回答："**改了什么、为什么改、期望什么、会不会互相冲突**"。
> 所以每次改动都登记在此；**等训练测试跑通、确认可用后，把对应条目标记 ✅ 并注明验证证据**，
> 之后就不必再惦记。
>
> 规则（与技能 `lingbot-code-changes` 配套）：
> - 阶段 3 每落地一条 ⇒ 在 §1 追加/更新一行；
> - 阶段 4 提交后 ⇒ 补上 commit SHA；
> - 上机验证通过 ⇒ 状态改 `✅ 已验证`，写清**验证方式 + 实测数字**，并移入 §2 归档；
> - 回滚 ⇒ 状态改 `↩️ 已回滚`，写清原因与回滚方式。
>
> **状态取值**：`⏳ 待验证`（已改未上机）/ `🧪 部分验证`（只有 CPU 证据）/ `✅ 已验证` / `↩️ 已回滚`

---

## §1 待验证（改了但没上机跑过）

**#83 R12十任务1350步与分批6＋4评测（2026-10-21）：🧪 本地/服务器49项CPU契约、实际1100条262964帧CPU及三步GPU通过，正式已6/1350。** 来源R11-step1200，保存500/1000/1350，经授权500间隔；BF16/m56/GAS1、14任务累计资格池已合入R11、10%复习RR10。schema新增evaluation.max_parallel_tasks默认1；run.py在显式>1时使用隔离会话parallel入口，run_jobs每批结束屏障后启动剩余任务，十任务6＋4全程模型一次加载。修复#82：probe_jobs在compare_reference=false时去掉子任务reference_summary，保留expected_seeds，补CPU跨层参数契约；旧失败不改写。新run=20261021_1129_ten_tasks_from_R11_bf16_m56_replay10pct。三步allocated77.71/reserved84.19GiB，正式GPU约86.6GiB；最终30回合尚未执行。不变量：同PID/model、批次屏障、当前主任务不复习、保存点不缩水；症状：后四个提前启动、再次加载模型、1回合仍比较3seed、额外存档；回滚：新进程max_parallel_tasks=1使用原串行入口，保存变动仍需用户批准；验证：test_parallel_evaluation、test_evaluation_tasks、test_replay_rotation及R12 cpu/probe/plan/recipe和最终evaluation_1350。组合证据：默认1＋共享=旧串行测试通过；默认1＋不共享=原逐任务入口保持；6＋共享=6/6＋10/6屏障CPU通过及六路单回合资源已实测；6＋不共享=启动校验拒绝。最终6＋4各3回合GPU仍待验证。

**#82终态补充：六仿真各1回合完成，但报告流程failed。** 64934MiB峰值、model_loads=1、6连接、48请求，无仿真Traceback/OIDN错误且六份视频非空。根因：compare_reference=false仅外层生效，evaluate.execute仍用1个seed对比reference_summary中3个seed；需要补CPU跨层契约并修正单回合资源配置后才能称完整通过。未降并发、未重跑；原失败记录与原始结果保留，见parallel_simulation_zh.md。

**#82 六路单回合资源探测（2026-10-21）：🧪 本地/服务器各9项CPU通过，六进程重叠已观测，GPU完成待核验。** parallel_evaluation支持concurrency_levels（默认[3,2]不变）与1至6路，本次显式[6,4]；compare_reference默认true，本次因每任务仅1回合显式false，报告记录未作历史结果对照。配置parallel6_R11_probe，R11前六任务各1回合。不变量：一份权重、独立会话、只资源错误回退、不改变训练存档；症状：模型重载、非资源错误被隐藏、报告冒称三回合等价；回滚：继续旧串行入口或[3,2]配置；验证：test_parallel_evaluation（新增六线程barrier）、test_shared_evaluation及GPU summary/audit/samples。

**#81 GPU补充证据：三路已完成**，各3回合成功次数2/3、3/3、0/3与串行一致，种子及模型匹配，model_loads=1，显存峰值44797MiB，仿真436.75秒；见parallel_simulation_zh.md与20261021_parallel3_completed.json。下方初始排队描述为历史启动记录。

**#81 三路仿真共享单模型（2026-10-21）：🧪 本地49项及服务器对应8项CPU契约通过，GPU试跑排队。** 新SessionPool每连接隔离step/chunk/robot/action_key/FeatureTransform，同一权重对象串行B1前向保持seed语义；新增session_eval_server、parallel_evaluation，ThreadPool最多3个独立仿真，常驻服务仅加载一次，逐任务日志/视频隔离。按用户授权先3路，明确资源OOM才退2路，其他异常/结果不一致停止留证。R11首三任务各3回合对照，等待原七任务评测结束，不打断既有结果。不变量：reset不跨连接、模型不复制、所有结果来自同一PID/model、出错关闭服务；症状：会话缓存串扰、模型重载、seed或结果不符；回滚：继续旧shared_evaluation串行入口，不启parallel入口；验证：test_parallel_evaluation与本次benchmark/session_audit/gpu_samples/summary。共享open_session仅新增可选server_entry，默认命令不变。

**#80 R11七任务1200步、10%复习与每3任务保存（2026-10-21）：🧪 本地/服务器44项CPU契约、真实950条CPU装配及三步GPU通过，正式1200步已进入。** schema新增checkpoint.every_tasks默认1，milestones每N任务结束或最终保存，补课容量预算同规则；用户确认7任务共1200步、14资格池复习10%、RR10，来源R10-step800。run=20261021_0831_qrcode_card_bottles_roller_pot_stamp_from_R10_bf16_m56_replay10pct；保存400/1000/1200，最终七主任务各3回合，补课关闭。新增不等长七任务保存点及真实Sampler索引归属契约；首次混合GPU三步通过（allocated77.70GiB/reserved84.59GiB），后续课程阶段及闭环仍待完成。不变量：分组存档不遗漏最终、排除当前主任务、全程6720复习样本、源HF不继承探针；症状：非400/1000/1200保存、复习混多个任务、CPU/GPU失败；回滚：新run显式every_tasks=1或关闭复习（需新进程、改保存需用户允许），停止新流程不影响旧模型；验证：test_checkpoint_modes/test_replay_rotation及本轮cpu/probe/verified/evaluation报告。

**#79 同事与Agent接手入口（2026-10-21）：文档及未执行配置示例。** 新增training/PROJECT_HANDOFF_zh.md，汇总当前仅存R9/R10、最终结果、环境/分支、从R10续训、独立共享闭环、资源依赖、状态路径、42项CPU与GPU验证边界、每轮收尾及Agent提示；不包含服务器连接信息或凭据。新增训练/共享评测JSON示例，默认只读预览；更新AGENTS实际实验分支和生产同步适用范围，补README入口。不变量：旧9680不冒充可用、示例不自动执行、不能将本地push当机器对齐；症状：新同事套用生产reset或找不到当前模型；回滚：不使用示例/文档按Git回退，不影响运行；验证：配置preview、共享job结构、文档链接/表格/凭据排查。本次服务器HEAD读取失败，明确标为未核实，未调整服务器或启动GPU。



**#76 10%轮换复习与累计资格池（2026-10-20）：🧪 本地42项纯CPU回归通过，真实混合数据CPU/GPU待验证，未同步活跃训练。** schema新增replay.mode=round_robin及switch_every_steps=10，planning每步复习单任务、排除当前主任务、按队列均匀轮换；GBS56以5/6条累计精确10%，旧sample分支不变。replay_pool从9680用户截图与R3/R6/R9报告建立12候选，1/4不达标，plate由后续2/3纳入；pool_file在准备时展开并冻结证据。支持完整summary增量导入，不因后续失败移除既往资格。不变量：当前主任务不占复习名额、每步最多一个复习任务、总样本/预算不变、旧模式回归；症状：一batch复习多个任务、10步前无原因切换、候选误纳1/4、冻结任务与证据不符；回滚：下一次启动replay.fraction=0且tasks={}（同时去掉pool_file/pool_snapshot），不可热改；验证：test_replay_rotation、test_contracts与新example预览，真实多任务装配待空闲后验证。复习+自动补课组合仍拒绝，不启新GPU训练。

**#75 每轮结束后更新训练记录并push（2026-10-20）：文档与长期协作要求。** 用户明确要求每次训练结束更新流水、任务统计及存档梳理后git push。新增checkpoint_inventory_zh，补全R9最终评测及R10实际101/800快照；AGENTS仅登记协作规则，细节放training索引。另按用户要求新增记忆扩展笔记。不变量：真实状态与计划区分、独立分支不重复累计、保存节奏及删除仍单独授权；症状：未结束标已生成存档/成功；回滚：恢复文档不影响训练；验证：纯CPU回归、记录链接/表格与原始状态核对。此要求是后续会话持续执行的工作约定，不冒充已部署后台自动Git提交服务。





**#72 评测任务跟随训练与R7路径对应（2026-10-20）：🧪 本地36项CPU测试通过，未热同步运行训练。** schema在省略evaluation.tasks时由本轮主任务/课程顺序生成；补课子run显式只测实际补课任务，run_rounds按当前轮任务验证结果；保留主课程显式额外评测任务的能力。流水和任务统计新增R7完整run、来源、存档、summary、逐任务日志与视频路径。不变量：默认训什么测什么、补课不夹带未训练任务、历史R7三任务结果不改写；症状：补课评测列表出现未补课任务、缺结果报错；回滚：新主课程显式配置evaluation.tasks，补课逻辑通过Git回退（仅下次启动生效）；验证：test_evaluation_tasks与test_remediation。R8既有冻结列表本来就是三个训练任务，无需热更新。

**#71 训练文档分区与run时间前缀（2026-10-20）：🧪 本地链接/CPU检查，未热同步运行训练。** 21份训练实验/配方/记录/数据/评测文档移入docs/training，新增统一索引及auto_learning系统索引，更新AGENTS与仓库引用。统一配方CLI为新output_dir增加北京时间YYYYMMDD_HHMM前缀，补课子run同规则，历史路径不动。不变量：源模型路径不改、运行中目录不改、移动不新增断链、显式时间前缀不重复；症状：索引打不开或新run仍日期后缀；回滚：下一次启动显式指定前缀目录，文档按Git移动记录回退；验证：test_naming、现有30项CPU回归、全仓Markdown链接迁移前后对照。作用仅新配方CLI及新补课run，不批量重命名旧官方入口或历史产物。

**#70 补课排除0/3并允许2/3（2026-10-20）：🧪 本地纯CPU回归，未热同步运行训练。** `remediation.curriculum`改为仅1/3、2/3各100步，2/3优先，同分保留任务顺序，0/3与3/3跳过；prepare按各任务最多100步预算容量。覆盖旧#66的0/3补200规则，历史R7结果不改写。不变量：有效0/3不产生课程、异常评测仍报错、每个候选任务100步、轮次上限保留；症状：0/3进入补课表、2/3被漏掉或顺序不符；回滚：恢复#66代码仅对下一次启动生效；验证：test_remediation、test_checkpoint_modes及test_contracts。R8继续remediation.enabled=false，补课暂缓；未修改当前数据、存档及GPU配置。

**#69 两种存档模式与R8顺序课程（2026-10-20）：🧪 本地/服务器30项CPU契约通过、真实150条19867帧CPU通过，三步GPU检查通过，正式600步训练已启动。** schema新增checkpoint.mode=steps/task；task只允许sequential且拒绝every_steps，milestones按任务累计边界；prepare生成save_at_steps、save_steps=0，训练器通过纯标准库checkpoint_due触发原HF-only出口。模式×补课四格、非等长100/300/500、旧间隔逐步回归及禁存档探针通过。不变量：切数据前保存、最终去重、旧steps逻辑不变；症状：漏200/400边界、探针出现HF、保存计划与trainer不一致；回滚：新启动配方checkpoint.mode=steps并指定every_steps，不能热改；验证：test_checkpoint_modes、cpu_report及未来verified_checkpoints。R8从R6-step500（非R7）按turn_switch→move_pillbottle_pad→move_stapler_pad各200，200/400/600保存，最终三任务各3轮。用户明确保留旧存档，先启动主课程、补课暂缓；容量闸门未降低，三步探针无DCP/HF输出，正式训练阶段已进入；200/400/600真实落盘待完成。

**#68 训练流水与任务统计（2026-10-20）：🧪 服务器只读取证及本地表格/JSON检查。** 新增training_history_zh.md与task_training_statistics_zh.md，按时间登记来源/任务/正式步数/耗时/闭环/补课，按任务首训顺序横向比较训练与补课；`refs/training_records`存机器证据快照，`tools/collect_training_records.py`只读采集已登记run。不变量：当前模型继承累计不跨分支相加、250/500快照不重复累计、混训期望曝光不冒充实测步数、进行中不算计划完成；症状：累计大于继承链配额、未结束标最终成功；回滚：不使用新报表（不影响训练）；验证：快照JSON解析、表格列数、日志首末步与summary逐项核对。覆盖2026-10-20七次run，更早鼠标记录未回填；快照265/300，采集器不自动重写Markdown，后续按新快照更新。未改运行训练。

**#67 补课300步仅最终保存（2026-10-20）：🧪 CPU契约通过。** 用户明确授权300步不需要250中间存档；`remediation.next_recipe`仅对总步数300且原间隔250改为间隔300，其他节奏不变。不变量：最终300必须保存，其余课程保存节奏不受影响；症状：新生成300步补课仍出现250保存计划；回滚：删除该特殊分支（启动前生效）；验证：test_remediation的milestones断言。当前已运行至69/300的首轮补课仍沿用250/300，避免重启丢进度；未热更新运行进程。

**#66 闭环有限补课（2026-10-20）：🧪 本地26项CPU回归通过，真实GPU及闭环串联待验证。** `task_recipes/schema.py/remediation.py/run.py/prepare.py`新增顶层remediation.enabled（默认false）及max_rounds（默认1）。1/3先补100、0/3后补200，2/3及3/3跳过；每轮从上一最终HF独立启动顺序续训，新优化器，复测全部任务，轮数上限后结束。保留存档间隔及每轮最终存档，预预算最坏空间，不自动删除；评测异常不判0/3。不变量：默认关闭不续训、补课顺序/配额确定、最终复测有效、无递归失控；症状：缺失/重复/非完整评测结果报错、子pipeline.failed，母pipeline不得complete；快速回滚：启动前remediation.enabled=false（新进程读取，不能热改）；验证入口：`python3 -m unittest experiment.task_recipes.test_remediation experiment.task_recipes.test_contracts`，产物remediation_summary及每轮plan/status。依赖#64/#65与共享评测；不改正在运行训练，不宣称CPU模拟为GPU成功；没有更改现有配置的保存节奏。提交：待用户指令。

**#65 顺序500步配方（2026-10-20）：🧪 CPU检查、三步GPU通过，正式训练已更新7/500。** 用户指定plate200来源，plate100/cup200/pot200顺序，250/500 BF16 HF，仅500闭环三任务各3；随后明确先用旧重载评测，不等待内存模型交接。新增sequential模式显式task_order及边界契约，不混复习、不重置阶段间优化器。CPU检查器GAS写死1的问题已参数化expected_gas并重跑真实GAS2装配成功，本次150回合loader500步通过。不变量：源模型、顺序/配额/保存、FP32仅推理不影响BF16 HF；症状：配方/步计划/HF/dtype失败；回滚：不用新配方，原plate200保留；验证：test_contracts、原始CPU重试日志、本次cpu/probe/verified/evaluation报告。详见顺序续训说明。


**#64 参数化配方框架（2026-10-20）：🧪 初始12项及顺序扩展13项纯CPU契约通过，GAS2与顺序150条实际CPU装配已通过。** 用户要求先启动单任务再优化架构，已在单任务state=training后开始隔离新模块。新增task_recipes的严格schema、mixed/task_units计划、主配额/复习比例、真实索引sampler、虚拟完整epoch、冻结哈希、统一训练保存共享评测及确认进度；examples在configs/training_recipes。未改活跃训练器/模型/本次配置。不变量：主导步数精确、复习按样本比例、保存显式、预取不计完成、旧入口不变；症状：配方/哈希/索引/长度/配额/HF不符；回滚：不启新入口，无热修改；验证：test_contracts、等待任务完成后真实CPU单任务与主复习两配方，顺序三步GPU已通过，主复习组合真实CPU/完整GPU尚未验证。HF-only不冒充精确Resume，多卡/FP32/自动PASS尚未支持。详见[框架说明](../training/recipes/task_recipe_framework_zh.md)。












**#53 Flex Attention autocast边界修正（2026-10-19）：🧪 CPU反向验证通过，CUDA重测待运行。** v3进入BF16前向后，在Flex Attention自定义反向报`expected scalar type Float but found BFloat16`，非OOM。该实现原本将QKV转FP32，外层autocast却让内部输出降到BF16，破坏其保存QKV/grad_out契约。仅在两个既有FP32 Flex子调用内关闭autocast，保持原FP32 attention语义、原后端与其他模块BF16不变；补真实CPU Flex反向契约。本地88 passed、3 skipped、36 subtests；服务器91 passed、36 subtests（真实CPU Flex反向已执行）。容量仍待完整更新重测。

**#53 单卡实际计算dtype修正（2026-10-19）：🧪 CPU验证通过，GPU重测待运行。** micro32/GAS1 v2首次optimizer更新后审计主动失败，实际主参数/梯度/优化器状态均FP32，但四组关键Linear输出全FP32；非OOM，不能记为目标混精度容量通过。根因`parallel_state.fsdp_enabled = fsdp_size>1`，单卡跳过FSDP策略，原train前向无autocast。新增仅mixed=true/fp32=false/无FSDP时的CUDA BF16 autocast，保持默认BF16、全FP32及多卡FSDP路径不变；补分支CPU契约及真实CPU算子验证参数/梯度/动量FP32、计算BF16。本地88 passed、2 skipped、36 subtests；服务器90 passed、36 subtests（包含真实CPU算子验证）。实际GPU重测待执行，正式存档节奏不变。

**#53 首次开卡短测修正（2026-10-19）：🧪 CPU契约通过，micro32/GAS1容量仍未知。** 服务器同步644c10f、CPU84 passed后首次探针在首步前退出：`static cache missing`，并非OOM。原因是探针走prepare-only而漏CPU窗口预备，且没有把本次plan的AL配置冻结给worker。现补`prepare_probe`按正式入口先准备同配置缓存、冻结AL、再materialize；worker继承正式入口隔离环境。增加CPU契约验证顺序、缓存路径与冻结AL一致；保留失败v1产物，以v2重测。CPU本地84 passed、1 skipped、36 subtests，服务器85 passed、36 subtests。用户明确GAS=1，只改micro；正式250/500 BF16 HF节奏不变。

**#53（2026-10-19）FP32主参数/BF16计算与FP32评测准备，🧪 仅本地CPU通过，等用户开卡。** 用户要求停止#52并切换精度，随后指定先准备、后开卡测micro32/GAS1/GBS32。旧run停50步无HF、控制器及GPU进程已结束，旧评测关机已取消。新增默认关闭的`--fp32-master`经recipe/worker传递true/false精度开关，第一步更新后实际dtype审计；BF16 HF保持不变，FP32评测通过独立进程记录实际主参数/关键层dtype。独立3步探针不存档/不关机，正式仍500步/250与500两份HF，来源混训前鼠标step500。不变量：不伪称全FP32内核、不额外forward、不更改已有入口默认精度，不将micro容量等同累积GBS、不继承探针权重；症状：worker仍false/false、审计主参数BF16、动作专家未见BF16计算或评测未见FP32输出；回滚：新进程`--no-fp32-master --eval-precision bf16`或原入口；验证入口：`test_precision_path.py`及配方/续训CPU回归，本地83 passed、1 skipped、36 subtests，GPU dtype/容量未验证。原启动器写死精度已改为显式默认关闭开关；存档节奏没有改变。详见[容量测试准备](../training/experiments/mouse/mouse_precision_capacity_zh.md)。当前未同步服务器、未启动新训练或GPU测试。

**#52 GPU启动补记：🧪 已正常更新7/500步，终态待运行。** run `al_mouse_pad_primary50_g56_from_mouse500_add500_20261019_v1`，代码ac78f72，合并后运行CLI核验56/GAS1、GC、P10、三项0.004、max500/save250、50条且无辅助/Replay。约7.4s/步，采样设备91761/97887MiB；未OOM、未删新250。外层pipeline.shutdown=true、内层训练关机false；两份HF校验后才最终500模型3回合闭环与关机。这里只确认启动，不将未来存档/评测/关机记为已验证。详见[启动记录](../training/experiments/mouse/mouse_primary50_train_eval_zh.md)。

**#52（2026-10-19）纯主任务50条GBS56与训练后闭环关机流水线，🧪 CPU与真实配置预检通过，GPU终态待验证。** 用户授权第2项从混训前鼠标step500训练，GBS56/GAS1、GC、500步、250/500 HF、EE10/.8/1.5、视觉三项0.004，无辅助；最终模型3回合闭环后关机。新增独立配方复用原续训worker，训练子任务强制不关机，250/500 HF均验证后评测；评测须正常退出、完成3回合、结果文件一致、3份非空视频及OIDN检查通过，0成功也属正常评测。异常保留机器。不变量：不覆盖原客户端、两份存档、权重来源独立、评测与训练串行、二者正常完成后才关机；症状：辅助仍启用、GBS非56、存档节奏变化、只训完就关机、渲染异常也关机；回滚：不执行新入口，或新进程`--no-shutdown`；验证入口：`tests/mouse_pad_training/test_train_eval_pipeline.py`加既有配方/续训契约、真实CPU预览及训练启动配置核验。本地73 passed、1 skipped、36 subtests；服务器74 passed、36 subtests。真实预览50/0、56/1、source500/.004/P10通过，仿真子入口`--help`正常退出并加载OIDN2.5.1；CPU预检发现/data符号链接与真实路径的来源比较误报，已统一resolve并补契约。实际空余39.86GiB，预算29.13GiB（两份HF含评测余量），不删除新250。GPU启动及500步/3回合/关机终态后续记录于[说明](../training/experiments/mouse/mouse_primary50_train_eval_zh.md)。

**#50（2026-10-19）中断后84GB卡重训500步，🧪 CPU及GPU启动验证通过，500步终态存档待验证。** 用户明确指定48/GAS1、EE位置12、每250步HF，深度/视频小幅增加并成功后默认关机；旧run止于427且无存档，重用原鼠标step500。三项辅助监督外层权重0.004→0.005（1.25倍）；新增配方参数只作用于新训练快照，不改模型实现/旧权重。不变量：97条70:30与目标臂窗口策略不变，250/500两份HF，worker不重复乘倍率，最终HF核验成功才关机；后续存档节奏必须获批（AGENTS铁律12）。症状：实际配置仍P10/.004或750间隔、未成功保存即关机、84GB卡OOM。回滚：新进程显式覆盖EE/倍率/步数/batch；存档节奏改变仍须用户批准；不更改运行中冻结配置。验证入口：`tests/mouse_pad_training` 配方与真实worker三权重/内部配比/原来源不变契约，续训成功/失败关机门控及周期HF测试。本地69 passed、1 skipped、58 subtests；服务器70 passed、58 subtests（torch装配项已执行）。正式run `al_mouse_pad_aux48_gc_p12_av125_from500_add500_20261019_restart_v1` 已完成至少10步实际更新；实际CLI核验48/GAS1、P12、三项0.005、max_steps500/save_steps250/hf_only，状态shutdown=true；reserved79.71GiB，设备82811/85651MiB，约10s/步，代码f1f1b15。最终250/500 HF仍待完成。

**#49（2026-10-19）batch56短测后正式750步与单份最终HF，🧪 CPU契约及56档30步通过，正式750步已启动，终态HF待验证。** 用户授权测试56后启动750步，保持原step500来源与梯度检查点；新增 `--hf-save-steps` 从配方经worker传至训练命令，容量预判按实际命令间隔计算。旧入口仍250，新辅助入口750步/750间隔，仅最终HF，原因是29GiB不足3份HF。不变量：97条70:30、GAS1、不删除历史、不继承短测权重，容量判据与实际周期一致。症状：容量仍按3份估计、worker恢复250、结束无HF或超预算落盘。回滚：新进程显式 `--hf-save-steps 250`（须先满足3份容量）；训练中不改已冻结快照。验证入口：鼠标配方worker重建、续训容量契约，本地相关契约通过，服务器69 passed、58 subtests；56档30步通过，7.4916样本/s、reserved88.05GiB，较48提高4.50%，辅助入口改为56/GAS1；正式run `al_mouse_pad_aux56_gc_from500_add750_20261019_1220` 已实际更新4步，配置文件核对56/GAS1、checkpoint=true、max_steps/save_steps=750、hf_only；约7.4s/步，reserved88.05GiB。代码提交31184db。正式750步终态HF尚待运行，现已启动，未自动关机。

**#48（2026-10-19）鼠标混训1000步与梯度检查点默认开启，🧪 CPU配置契约通过、GPU短测32/48通过，正式1000步待容量解决。** 用户授权先开梯度检查点测batch32，有显存余量再测48，按稳定samples/s与峰值显存选档；正式新增1000步。仅调整辅助入口main默认参数，旧入口不变，实测后辅助入口默认batch48/GAS1，旧入口不变。依赖#47与已有checkpoint实现，不改模型/训练循环。不变量：仍从原step500权重启动、97条/70:30/原指令，GAS1，显式CLI可覆盖；短测独立run不作为正式模型来源。症状：预览仍500步、训练命令checkpoint=false、32/48未产生实际更新或OOM。回滚：新进程传 `--steps 500 --no-gradient-checkpointing --micro 20`。验证入口：`tests/mouse_pad_training` 本地24 passed、1 skipped、23 subtests；检查重建命令checkpoint=true与batch32。GPU两档各30步通过：32为6.4393样本/s、reserved63.28GiB；48为7.1692样本/s、reserved79.71GiB，提升11.34%，选48。micro72外推超容量故未跑；不混淆micro与GBS。服务器CPU25 passed、23 subtests。1000步4份HF超过当前剩余29GiB，未启动正式训练。证据见混训说明及独立benchmark_result.json。

**#47（2026-10-19）主任务与审核辅助池独立槽位混合训练，🧪 CPU回归及真实数据检查通过，GPU待验。** 新增 `sampling/auxiliary.py`、鼠标辅助配方；config/BatchComposition/真实sampler/hook/scheduler联动计数，启动器构建联合白名单并CPU对拍、冻结清单。用户授权一次完成并在验证后提交push。不变量：主50＋辅助47完整episode，原始来源/指令，主辅全局70:30累计配额，辅助episode等概率；默认关闭不改变旧抽样序列；数据身份变更拒绝；仅主任务拟合监控；消费统计排除预取。症状：白名单仍50、辅助记NEW/Replay、ID错配、计数不足GBS、抽屉动作被加成鼠标抓取、预取统计虚高。回滚：旧all50入口或不传auxiliary_manifest，新进程生效。验证入口：`tests/mouse_pad_training/test_auxiliary.py`、采样/Hook/调度/恢复/启动器回归及实机CPU预览。阶段3倍为目标臂控制候选，加闭爪前1秒；非精确物理标签，其他帧权重1，暂不额外加权放置。证据：服务器264 passed、86 subtests；真实97条/15228帧，500步7000主/3000辅，全部47辅助episode参与，97条horizon50末尾chunk检查通过（cpu_v2）。独立实际Parquet列训练装配已通过（cpu_assembly_v1）；后续chunk重复检查触及2GB内存限制被杀，保留失败日志，不能计作通过；参见混训说明。未启动训练。

**#46（2026-10-19）本次目标物数据扩充的目录归位，🧪 本地及服务器文件校验通过。** docs只保留说明；首次扫描的64个JSON/HTML/图像产物迁至 `artifacts/data_audits/mouse/20261019_scan_v1/`；47条审核清单/审核决定/来源指纹（3份JSON）复制到 `refs/data_manifests/mouse/20261019_review_v2/` 版本化；新增精确忽略规则，只忽略本项目data_audits生成目录。不变量：64个扫描产物逐文件SHA256不变，3份refs JSON与来源字节一致，训练配置/采样器/原始数据不变。症状：文档断链、refs与审核输出不一致、批量图片意外进入git状态。回滚：将首次产物目录迁回原docs路径并恢复本条文档链接/忽略项；不删除任何数据。验证入口：本地与服务器64份扫描产物SHA256一致、3份refs副本逐字节一致、9条本地Markdown链接有效；Git忽略规则及 `git diff --check` 通过；该项本身仅目录与说明变化；随后随#47共同验证提交。

**#45（2026-10-19）可指定目标物的episode筛选与只读视频审计，🧪 CPU契约初测通过。** 新增 `tools/select_object_episodes.py`，支持 `--target/--keyword/--object-id/--candidate-task`；关键词+环境源码召回、主任务排除、v3视频偏移、CPU状态/索引/指令校验、人工审核清单及独立输出。无改原始数据、无训练接入。不变量：候选不自动批准；不同型号不排除；只有数据完整、对象操作/示范成功经明确证据确认的辅助episode才进入approved；不保存local_idx，审核绑定数据与规则指纹。症状：背景鼠标自动入库、原始数据被改、视频显示其他episode、旧审核静默复用。回滚：不调用该脚本即可；不影响现有训练入口。验证入口：`tests/test_select_object_episodes.py`；本地相关回归43 passed、26 subtests；服务器筛选/窗口27 passed、13 subtests。实盘2500条/548893帧清点，主任务及可能来源共300条数据校验无错；52条关键词辅助生成总览图，5条参照物假阳性经指令与图像排除，47条/7762帧保留为完整成功待核验候选；已抽查16条图像（11条可见鼠标操作），未认证全部成功，approved为空。报告见 `mouse_auxiliary_scan_20261019_zh.md`；未改训练入口、未启动训练、未提交push。 后续v2已逐条完成47条全程稀疏帧及终态三视角视觉审核，命令证据与视觉一致，服务器按原指纹复核生成47条/7762帧清单（左25/右22）；非逐帧/原任务成功判定认证。精确阶段边界未核实，training_ready=false。新产物在 `artifacts/data_audits/mouse/20261019_review_v2/`，首次扫描产物已迁至 `artifacts/data_audits/mouse/20261019_scan_v1/`，版本化审核清单位于 `refs/data_manifests/mouse/20261019_review_v2/`；见 `mouse_auxiliary_review_zh.md`。

**#44（2026-10-18）鼠标全50条入口改从上一轮鼠标step500继承权重，🧪 本地CPU验证通过。** `20261018_place_mouse_pad_all50_9680_ee_lnfm.py::main` 默认模型及训练配置改取 `al_mouse_pad_9680_ee_lnfm_add500_20261018_150402`；新run名标记 `from_mouse500`。不变量：继承step500模型，新增500步，仍用新优化器/LR/步号，50条训练、EE10/.8/1.5、250/500 HF不变；旧40/10入口不变，不导入旧池。症状：预览source_model仍指原始9680、来源配置不对应、输出名称错称原始模型。回滚：启动时显式 `--model /data/outputs/al_v79_poolfix_7300/checkpoints/global_step_9680/hf_ckpt --train-config /data/outputs/al_v79_poolfix_7300/lingbotvla_cli.yaml`；参数仅新进程读取。验证入口：`tests/mouse_pad_training`及启动器/续训测试，本地41 passed、26 subtests；服务器真实step500资产CPU预览通过：train=50/val=0、EE10/.8/1.5、source_model为指定鼠标step500；服务器同组41 passed、26 subtests。已按文件同步到独立实验目录（保留服务器已有仿真改动，Git HEAD仍为aba61f0，不能仅凭HEAD判定同步状态），本条未提交push、未启动训练。服务器当前可用空间4.8GiB，不足写出两份HF，正式训练前须解决空间。

**#43（2026-10-18）鼠标50条全部训练 + 仅训练集拟合监控 + 位置EE权重10，🧪 本地CPU验证通过，未提交/未上机。** 用户明确将原10条val加入训练，保留4条训练轨迹每100步GMSE/EE监控，关闭独立val；新增 `train_all_episodes=false` 默认开关。新日期配方从原始9680新增500步，EE权重10/.8/1.5，ViT全量、低噪FM、连续窗口及250/500 HF沿用旧配方。

- **不变量：** 原manifest不改；开启时train=原train∪val、val为空，白名单/实际local_idx/静态窗口缓存均覆盖全部50条；仅调用4条train_monitor；不伪造active_val、成品参考倍率、PASS、gap或泛化结论；默认关闭时40/10行为保持；禁止旧池导入及验证驱动状态迁移。
- **症状：** 新run仍train=40、缓存只40条、出现active_val调用/倍率/“未过拟合”、worker丢失全训练开关、训练末因缺val报metric_invalid或DEFER。
- **回滚：** 使用原 `20261018_place_mouse_pad_9680_ee_lnfm.sh`（40/10、位置权重3），新进程生效；不覆盖原run/缓存/manifest，无删除机制。50条训练模型不能再用原val声称独立验证。
- **验证入口：** `tests/mouse_pad_training/` CPU契约（实际worker重建、catalog→resolver→采样→调度→事件流）、既有40/10回归和窗口预计算；GPU训练/闭环、服务器同步及提交push不在本次执行范围。左右臂50条仍为19/31，本轮不重配左右采样概率。当前本地证据为47 passed、28 subtests；旧模式回归183 passed、72 subtests。

**#42（2026-10-18）原始step9680专练place_mouse_pad，🧪 已提交同步，实机CPU预览通过，未启动GPU训练。** 实现提交 `aba61f0`。新增日期启动文件（shell/Python）、独立单任务AL配置和说明文档。默认新增500步、micro20/GAS1、每250步HF；EE(3/.8/1.5)、低噪FM、连续抓取窗口开启。原始9680权重不依赖stand模型；40 train/10 val，任务episode800–849。

- **不变量：** 仅鼠标任务train白名单，val不入训练；ViT两处冻结开关、expert-only和LoRA均显式关闭；子启动器重建时仍关闭；不导入旧池/旧概率；原始来源配置不改写；默认只预览，自动关机需 `--shutdown` 且最终HF完整。
- **症状：** source_model含stand路径、train_ids出现800–849之外任务、子启动器重新冻结ViT、鼠标仍读stand窗口缓存、原始9680缺少datasets.txt导致便捷续训失败。
- **回滚：** 不使用新recipe即可，不改旧启动器和旧run；`--steps`/`--micro`/`--gas`/`--gradient-checkpointing`/`--no-shutdown`仅新进程读取。若要禁用此实验，保留旧配置即可。
- **验证入口：** `tests/mouse_pad_training/test_recipe.py`、既有启动器/续训CPU回归、shell `bash -n`、Python编译及默认 `--help`；本地和服务器均33 passed、13 subtests passed。服务器真实资产预览退出0，未创建新run或缓存；GPU命令权限错误，未启动。测试说明与证据集中在 `tests/mouse_pad_training/VALIDATION_20261018.md`。

**#40（2026-10-18）连续抓取目标变化窗口 + step500 起点750步续训配方，⏳ CPU回归/真机缓存重建待完成。** `target_change_window.py::scan_episode` 按夹爪数值方向识别闭合，只把闭合与同侧、闭合后2秒内且重新开爪前的首段腕部运动连成一个连续加权窗口；缺少提起时只覆盖闭合及短暂保持。缓存 schema 升为2，续训 recipe 给新run独立缓存并使用该规则。750步启动器允许末尾非100对齐的50步partial unit，以便100步周期验证后在750步正常结算。

- **不变量：** 打开夹爪不创建抓取阶段；超过2秒或开爪后的腕部运动不匹配为提起；窗口概率仍通过原静态采样器归一化；缓存路径/权重/训练输出为新 run 独立目录，旧报告不覆写；step<=750时周期验证仍只在100/200/.../700，末尾partial unit在750完成；若方向一致的预测和真值都减相同输入状态再按物理尺度计算MSE，误差不变。
- **症状：** HTML区间中闭合到抬起仍断开、开爪放置被标成抓取阶段、旧schema缓存静默复用、750步被启动器拒绝或最后50步不结算、启动文件覆盖旧run缓存、将relative target差误称作力。
- **回滚：** 使用启动文件传 `--al-set sample_scan_method=hardness` 或 `--no-sample-scan`；窗口阈值通过 `target_change_grasp_lift_max_delay_seconds` 配置；代码级回滚本条涉及的scanner/config/tests/docs/recipe及step预算校验。只恢复启动器的100倍数限制时，750步配方不可用，但常见500/1000步仍可运行。
- **验证证据：** 最终纯CPU回归 **52 passed, 22 subtests passed**；最终方案只桥接抓取阶段、普通腕部运动保留局部窗口，5561帧中3707帧覆盖（66.7%，旧窗口49.8%），40个闭合阶段找到腕部运动，权重2.0下理论概率质量80.0%；80.6%覆盖的全运动连续方案已撤回。配方 `bash -n` 和默认预览通过。750步允许最后50步正常收尾；启动器配置快照同步视觉冻结/gradient checkpointing开关，以便下次续训继承实际设置。真实schema2报告须启动时生成并人工检查，GPU训练与闭环收益仍待实测。

**#41（2026-10-18）记录相对输入状态动作表示设想，不启用训练。** 见 `relative_action_representation_zh.md`：当前state是drive target，chunk action减锚点state是相对目标变化，不是actual qpos误差或力；若预测和GT同减同一state并在物理尺度算MSE，误差抵消。实际对照只通过feature transform/归一化参数化改变优化尺度，必须新run单变量检验。

- **不变量：** 本条只记录实验建议，不开启 `subtract_state`、不改当前模型输入/动作输出、不改已在跑的训练。
- **症状：** 训练配置出现 `subtract_state=true` 但设计目的写成力监督，或在原 checkpoint 上只改配置续训。
- **回滚：** 无代码开关变更；若将来做实验，用独立 run 并恢复 `robotwin.yaml` 的原动作表示即可。
- **验证入口：** 对比 `FeatureTransform.apply/unapply` 与 `configs/robot_configs/robotwin.yaml`；具体单变量验证方案见 `relative_action_representation_zh.md`。


**#39（2026-10-17）AL固定步数训练跨越数据epoch，🧪 纯CPU验证通过，下一次训练生效。** 当前run在231步发生额外测评，证据为5561样本/GBS24=231步，每轮末无条件 `flush_partial_unit()` 将200之后31步结算。旧要求未落实，不能声称已关闭。修改 `train_lingbotvla.py::main` 的内层循环预算：AL启用、指定max_steps且save_epochs=0时按剩余optimizer步数循环，保留学习单元和迭代器跨数据长度边界；非AL、无max_steps或显式epoch存档保持原路径。

- **不变量：** 不在231/462等数据长度边界提前结算unit；100步unit、max_steps/STOP/AL结束的最终flush、周期HF规则不变；保留同一迭代器，避免丢弃prefetch导致采样统计错位。
- **症状：** 又出现231测评、下次unit变成331、样本计数与实际步数不等、总步数超限或final不能保存。
- **回滚：** 撤销内层循环预算这9行，回到数据epoch循环；不更改当前训练进程。验证入口 `tests/al_epoch_schedule/test_loop_budget.py`，真实hook+sampler跨231到255，检查只在100/200结算，最终255的partial flush仍正确；普通训练/显式epoch存档/已有步号等分支均覆盖。当前GPU长跑继续旧版本，修复的真实GPU跨epoch验证待下次run。

**#38 面板历史单行布局已通过纯CPU与服务器实机显示，移入 §2。**

**#37 仪表盘历史三项EE训练loss已通过纯CPU和实机显示核验，移入 §2。**

**#36 已完成实机短训、val、周期HF保存/重载和CUDA精度验收，已移入 §2。** 500步正式对照已启动，尚未完成，闭环效果待另测。

**#35（2026-10-17）低噪 FM 与末端 FK 一致性监督，🧪 CPU 与当前仿真几何部分验证，bf16 兼容/精度未通过，GPU 待验证。**
> 本条保留首次验证与归档边界；后续训练接线、bf16精度修复和val指标交付以 #36 为准。
**本次提交边界：** 只归档独立 FP32 FK 计算模块、关节名预检、功能测试与文档。`configuration_lingbot_vla.py`、`modeling_lingbot_vla_v2.py`、`utils/arguments.py`、`tasks/vla/train_lingbotvla.py` 的 FK 模型/训练接线仍为本地未提交候选修改，待修复 bf16 问题后单独验收和提交。下述完整行为是候选设计，不代表本次提交已提供 FK 训练入口。
**归档提交：** `7ea53aa`（独立计算模块、预检和按功能归档的测试/证据；CPU 8 passed，三个历史 JSON 报告复现一致）。本条状态仍为部分验证，FK 训练接线没有随该提交交付。
低噪 FM 默认关闭；开启后 25% 的训练时间采样落在 `[0.001,0.2]`。FK 默认关闭且必须依赖低噪 FM，只在 `t <= ee_fk_time_max` 对 `x_hat = x_t - t * v_t` 与 GT 动作做可微 FK 对齐，分位置、姿态、夹爪三项加权监督。动作布局和归一化统计从真实 `FeatureTransform` 提取，多数据集不一致时拒绝启动。JSON chain spec 必须按 RoboTwin 的关节名、顺序、TCP 坐标口径核对，禁止猜测。

**改动本意四问：**

1. **不变量：** 默认两个开关均不改变旧训练；FK 开启时没有已核对的 spec、布局或归一化不一致就启动失败；GT 不接收梯度；低噪关闭时不额外消耗 RNG。
2. **出错症状：** 启动日志缺 `[ee_fk] enabled`、训练中缺 `ee_fk/*` 指标，或真实轨迹 FK 对拍出现系统性 TCP/左右臂偏移；这些分别指向接线缺失和资产口径错误。
3. **快速回滚：** 新进程关闭 `ee_fk_loss_enabled`；完全回滚低噪则再关闭 `low_noise_fm_enabled`。代码回滚为本条涉及的模型、配置、参数、训练接线和 `lingbotvla/losses/ee_fk.py` 文件。
4. **验证入口：** `python -m py_compile` 覆盖修改文件；`tests/ee_fk/test_ee_fk.py` 验证纯计算与梯度；`tests/ee_fk/README.md` 统一测试意图、诊断命令、判据与历史证据；`tools/auto_learning_pipeline/ee_fk_spec_check.py` 核对左右六轴名称和顺序；训练机还必须做真实 RoboTwin/SAPIEN 多帧对拍和 GPU smoke，未完成前不得宣称可训练。

**当前证据：** 修改文件通过语法编译和 `git diff --check`；服务器隔离目录 FK/低噪 CPU 回归 **8 passed / 5.19 s**。真实资产候选 spec 与 SAPIEN/RoboTwin 200 组同关节角几何对拍通过，位置最大误差 `6.42135e-7 m`、姿态最大误差 `8.83176e-7 rad`；梯度有限、GT 无梯度、padding 屏蔽通过。**未通过：** solver 转 bf16 后与 FP32 动作类型冲突；bf16 后再转 FP32 仍带来最大 `2.273 mm` 几何误差。历史 HDF5 100 帧的关节与记录末端最大差 `36.463 mm`，采集/保存时序口径待核对。保持 FK 关闭，未改训练逻辑、未部署/提交。完整证据与复现见 `tests/ee_fk/VALIDATION_2026-10-17.md`。测试按功能归档到 `tests/ee_fk/`，包含单测、诊断源码、说明、原始小型 reference 和结果；路径调整不改变计算方法。

**#34（2026-10-16）低噪 FM + 目标变化窗口采样，🧪 本地 CPU 部分验证，未提交、未上机。**
低噪 FM 默认关闭；开启后 25% 时间样本取 `[0.001,0.2]`，不改变显式指定的评测时间。
新增 `sample_scan_method=hardness|target_change_window`，历史 `hardness_scan_enabled` 为总开关。
窗口缓存按任务名保存，以 episode/绝对帧号映射当前 local_idx；仅训练前补算，不随模型、存档或 DEFER 重扫。
**实现入口**：`utils/flow_time.py` + 模型/训练参数接线；`sampling/target_change_window.py` + `prepare_target_change_windows.py`；便捷/完整启动器、scheduler、pool/DCP 恢复接线。
**不变量**：关闭低噪时 RNG 不增加调用；扫描关闭时均匀；独立验证集不进入窗口概率；未知腕部名称拒绝启动。
**症状**：静态模式出现模型扫描日志、缓存坐标与 resolver 不匹配、报告覆盖率异常或无事件；不能把目标变化解释为接触力。
**快速回滚**：新进程设 `low_noise_fm_enabled=false`，`sample_scan_method=hardness`（或总扫描开关 false）。
**验证入口**：`tests/test_target_change_window.py`、`tests/test_low_noise_fm.py` 及既有 AL/启动器回归；真实数据人工对帧与 GPU 对照待开机。
**CPU 证据**：最终选择集 `169 passed, 2 deselected, 66 subtests passed`，19 个修改/新增 Python 文件编译、CLI help、`git diff --check` 通过。两项既有 hardness-cache scorer 测试在第一次扩展回归中因缺 `torchdata` 失败，最终显式排除；对应测试、backend、data_loader、open_loop_validation 相比 HEAD 均未改动，不能宣称重依赖全链已通过。命令与矩阵见 `target_change_window_zh.md`。
**依赖/未验项**：准备工具依赖既有 PyArrow/YAML；模型分数与静态权重共用采样器/历史 `HardnessScan` DTO，静态结果不伪造 loss。真实机器人资产/原始轨迹不在本地，须核对腕部关节含义、窗口视频对齐，再做 GPU 对照；尚未提交、push 或同步。

| # | 日期 | 改什么（文件/函数） | 为什么 | 期望行为 | 依赖/冲突 | commit | 状态 |
|---|---|---|---|---|---|---|---|
| 1 | 10-11 | `state/registry.py`：抽 `_judge_n`；`confirm_val_ids` 与 `active_val_ids` 同组 | 判定尺子原先两处各写 `val_ids[:4]`，语义上看不出是同一把尺子 | 判定仍用 4 条；无行为变化（等价重构） | 与 #2 同批 | `d014aa6` | 🧪 部分验证（CPU 回归 131 passed） |
| 2 | 10-11 | `real/backend.py` 新增 `merge_trajectory_metrics()` + `evaluate_confirm_merged()`；`scheduler._timed_confirm_merged()`；`build.py` 注入 adapter | confirm 的 4 条里 2 条与 scout 完全重复且**被重跑**（白花一半算力），复核也不独立 | 日志出现 `[open_loop][confirm] task=X 复用 scout 2 条 + 新评 2 条 ⇒ 判决用 4 条（省 2 条推理）`；事件出现 `confirm_reused_trajs=2` / `confirm_fresh_trajs=2`；`confirm_nmse` 口径变为"2 复用 + 2 新算的合并值" | **与 #3 必须成对**（帧上限不改判定，独立）；CPU 单测 5 passed、回归 131 passed | `d014aa6` | 🧪 部分验证 |
| 3 | 10-11 | `configs/auto_learning/al_eval2_v45_frame200.yaml`（新）：`hardness_max_frames_per_traj: 300 → 200` | 扫描量 −33%（单任务 600 → 400 帧），帧多的任务省时间 | 日志出现 `（帧数上限 200：N 条轨迹被均匀压缩，A → B 帧）`；单任务扫描墙钟下降约 1/3 | 独立；**只对新 run 生效**（配置在进程启动时读入） | `d014aa6`（同批） | ⏳ 待验证 |
| 4 | 10-11 | `experiment/robotwin/al_launch.py`：`base_env()` 白名单纳入 `SAVE_EVERY` / `HF_PASS_INTERVAL` | 原只透传 `AL_*` ⇒ shell 里 `SAVE_EVERY=400` **静默无效**（实测存档仍是 1000 步） | 以 shell 变量方式传也能生效；`--env` 仍优先 | 独立 | 待提交（工作副本已改） | 🧪 部分验证（`--env` 路径已在 v43 实测生效） |
| 5 | 10-11 前 | `tools/auto_learning_pipeline/al_panel.py`（训练器识别 / 未找到 run 三态 / 步时双口径） | 面板把多 run 的 rank 加在一起数、run 未创建时提示误导、s/it 被评测步拉爆 | `--trainers` 一行一个训练器；状态行走 `稳态 X s/it（当前步 Y s/it：含评测/扫描）` | 独立；**已在真机跑过（只读工具）** | `17948d0` | ✅ 已验证（真机跑通，见 10-11 日志） |
| 9 | 10-12 | `tools/auto_learning_pipeline/al_panel.py`：**默认视图改为 15 行紧凑面板**（纯 ASCII / 宽度自适应 / 在训任务历史 3 unit / 总用时与预估 / 池子 `pass=7 defer=12 … exhausted=0`）；原长表格移入 `--wide`；`s/it` 与 ETA 改口径 | **改动本意（用户 2026-10-12 四问）**：<br>① **不变量**：面板**只读**（不写 DCP/不碰训练）；各节数据来源不变（池子=DCP 注册表、轨迹=事件流、步时=训练日志）；`--pool/--progress/--traces/--all/--trainers` 语义不变。<br>② **出错症状**（怎么一眼看出坏了）：面板报 `NameError/AttributeError` ⇒ 打包字段口径变了（如 `best_gmean_mse` 只在 PASS 后有值、`series[metric][task]` 是 list 不是 dict）；`池子` 行不显示 `exhausted=` ⇒ 池子读取失败退化了；`s/it` 远大于日志稳态值 ⇒ 又用回累计均值了。<br>③ **快速回滚**：`git checkout <sha>~1 -- tools/auto_learning_pipeline/al_panel.py`（单文件、无外部依赖）。<br>④ **验证入口**：`/opt/robotwin-env/bin/python tools/auto_learning_pipeline/al_panel.py <run>`，判据 = 行数 ≤20、无框线字符、`Step` 不带 `/20000`、池子含 `exhausted=`、历史 3 行、`用时` 的"已训"与日志 `[HH:MM:SS<…]` 一致 | 独立（只读工具）；`--wide` 保留全部旧细节 | `93db002` | ✅ 已验证（v44 实跑 15 行；`已训 11.8h` 与日志 `11:45:20` 吻合） |
| 6 | 10-11 | `real/backend.py`：`evaluate()` 新增**轨迹分片**（`AL_EVAL_SHARD=1` 开启，**默认关**）+ `_eval_shard_indices()` / `_gather_and_merge_shards()` / `_all_gather_object()`；`merge_trajectory_metrics()` 支持 `metric_valid` / `note` 覆盖 | 训练态评测现状是**每个 rank 各跑整批**（实测 v43 8 卡 / v44 7 卡：每卡都跑 12/20 条）；轨迹互不依赖、逐轨迹 MSE 与卡数无关、聚合是几何平均（可分组）⇒ 可分片 | ① **不死锁**（每卡恒 `k=ceil(n/W)` 条 ⇒ 集合通信对齐）；② 汇总与单卡全量**逐位一致**；③ 墙钟 ≈ 现值的 `1/W`（n=8、W=8 ⇒ 每卡 1 条） | ⚠️ **CPU 测不出集合通信错配** ⇒ 必须真机验证；只改 `active_val` + `train_monitor`，`scout`/`confirm`/`rescan` 暂不动 | `7164cde` | 🧪 部分验证（CPU 单测 11 项 + 本机回归 119 passed） |
| 7 | 10-11 | 消歧义注释：`orchestration/scheduler.py`（`*_gmean_mse` 是**判定口径**、`*_nmse` 是历史遗留显示量）、`state/registry.py`（`active_val_probe_trajs` **实际决定 confirm 条数**）| 命名历史遗留易误读；改名会让 v41–v44 产物不可比 ⇒ 按三选一取**加注释** | 注释与当前实现一致；不改任何字段名/指标名 | 独立 | `d014aa6` | ✅ 已验证（纯注释） |
| 8 | 10-11 | `tools/auto_learning_pipeline/al_panel.py`：新增 **DCP 注册表读取**（`latest_dcp` / `load_registry`）+ **池子摘要**（默认视图 2 行）+ 4 个参数（`--pool` / `--progress` / `--traces` / `--watch`）；新增技能 `lingbot-training-monitor` | 用户要"一眼看池子/进步/GMean"：原面板只能从**事件流推导**状态，**推不出 DEFER/EXHAUSTED**、也没有 attempts/lp/过拟合/DEFER 原因 | 默认视图出现 `池子来源: DCP global_step_N`、`池子: TRAINING 1 \| DEFER 9 \| PASS 7 \| CANDIDATE 33`、`当前任务: X（第 N 次尝试，已训 M 步，当前倍率 Kx）`；`--pool` 出明细表（含 DEFER 原文原因）；`--progress` 按 lp50 降序；`--traces N` 多任务轨迹 | 独立；**只读**（`torch.load(map_location=cpu)`），训练期安全 | `606c67a` | ✅ 已验证（v44 实跑：池子 1/9/7/33 与 DCP 注册表一致；`--progress` 12 行、`--traces 2` 正常） |
| 9 | 10-12 | `real/determinism.py::deterministic_sampling`：**没有增强要关时跳过快照 RNG**（`if not disabled and _fast_fetch_enabled()`）+ 新增 `_feature_transforms_cached()`（按 dataset 身份缓存遍历结果）+ 对照开关 `AL_HARDNESS_FORCE_SNAPSHOT=1`；新增 `tests/test_al_deterministic_sampling.py`（7 项） | hardness 每批（8 样本）都做一次 `collect_feature_transforms` + `snapshot_rng()`（含 CUDA `get_rng_state_all()`）+ 还原；而本 run `image_augment=False`（随机增强全部 gate 在 `train=self.image_augment and not policy_eval`）⇒ 取 item 不消耗 RNG、快照是**恒等操作**、纯开销 | ① 语义不变：`disabled` 为空时取 item 前后 `random/numpy/torch` RNG 状态**逐位相同**；② 有增强要关时**仍走**快照/还原（安全底线不依赖开关）；③ 扫描墙钟略降（幅度待 v45 实测） | 独立；与 #6（分片）、#1/#2 无耦合。⚠️ 开关在**进程启动时读一次**（改开关要重启） | `3e1ea94` | 🧪 部分验证（CPU 单测 7 passed；真机收益待测） |

| 10 | 10-12 | `tasks/vla/train_lingbotvla.py`：未显式传 `--train.step_offset` 时，**自动从 `--model.model_path` 解析 `global_step_<N>`** 作为起始编号（只认 `global_step_<N>`，不认裸数字，避免把 `v41` 当 41） | v44 实测 `step_offset=0` ⇒ 从 `…/al_v41_step801_hf/global_step_801/hf_ckpt` 起新 run，但存档编号从 1 开始、与来源 801 脱节 | ① 日志出现 `[step_offset] 未显式指定 ⇒ 自动从模型路径推导 = 801…编号从 802 起`；② 存档仍落在 `save_steps` 整数倍（1200/1600）；③ 显式传值优先；④ DCP 成功加载时被 DCP 的 global_step 覆盖（原语义） | ⚠️ `max_steps` 是**绝对**目标 ⇒ LR 调度步数 = `max_steps - offset` 变短（既有语义）；独立 | `TODO_SHA` | 🧪 部分验证（py_compile + 正则 6/6 用例） |
| 11 | 10-12 | 新增 `lingbotvla/auto_learning/state/pool_state.py`（池子独立存档读写）+ `tests/test_al_pool_state.py`（16 项） | Auto Learning 的池子（status/attempts/lp50/`sample_probs`）原本**只随 DCP** 的 `extra_state` 走；从 **HF 权重**起新 run 判 `fresh` ⇒ 池子从零 ⇒ 已 PASS 的被重训、刚 DEFER 的被立刻重试 | ① 导出→读回**逐字段相等**（含 `sample_probs`）；② 缺失/损坏 ⇒ `None` 不抛；③ 原子写不留 `.tmp`；④ 开关 `AL_POOL_STATE_EXPORT／NO_IMPORT／RESTORE_RNG／MIRROR／PRUNE／FILE`（**进程启动时读**）；⑤ `sampler_rng` **默认不恢复** | 独立；纯 stdlib（无 torch）；单份 ≈9.4 MB | `TODO_SHA` | 🧪 部分验证（单测 16 passed） |
| 12 | 10-12 | `tasks/vla/train_lingbotvla.py`：导出挂在 `save_hf_checkpoint_best_effort()` 内（⇒ 周期/轮末/收尾**三个存档点全覆盖**），路径 `<global_step_N>/pool_state.json`，`final_only` 时另写 `<…>/hf_ckpt/pool_state.json`；`real/build.py`：`Scheduler` 之后**仅当不走 DCP** 时导入池子 | 需求 2 落地：HF 续训也能继承池子。DCP 路径由 `load_extra_state()` 自动恢复 ⇒ 必须**跳过**导入（否则日志把"预期跳过"误导成"丢状态"） | ① 日志出现 `[pool_state] 已导出池子：…（step=2400，9.4 MB）` 与池子计数；② HF 续训出现 `[pool_state] 已导入池子：…（step=2400）`；③ DCP 续训出现 `本次从 DCP 续训 ⇒ …跳过 pool_state.json 导入（这是预期行为，不是丢状态）`；④ 导出失败**只告警不中断**；⑤ 任务集合对不上 ⇒ 摊开三类差异（多出/缺少/共有），**不拦截** | ⚠️ **前置**：导出挂在 `save_hf_checkpoint_best_effort` 内 ⇒ `save_hf_weights=false` 时不导出（`final_only` 模式必开 HF ⇒ 满足）；与 #10 独立 | `TODO_SHA` | 🧪 部分验证（单测 16 passed + 静态接线断言） |
| 13 | 10-12 | `lingbotvla/utils/open_loop_validation.py`：新增 `eval_shard_indices`（分片切分**唯一事实来源**）、`adaptive_episode_starts`（每条轨迹恰好 k 个窗口）、`_shard_k_for`；`_preflight_verdict` **分模式判定**（分片下不比 `ids/len_ds`，改守 `n_starts` 相等 + **并集覆盖完整**）；`_eval_rank_report(..., n_ids=)` + `_multirank_eval_preflight` 透传全集条数；`real/backend.py::_eval_shard_indices` 改为**转发**共享实现 | **真机 al_v47 崩点**（从 2400 HF 起跑 + `AL_EVAL_SHARD=1`）：首个 `active_val` 评测被多卡一致性预检拦下 —— `n_starts` 实测 rank0=3 / rank1=4（`len_ds` 143/151/148/157）。两个独立缺陷叠加：① 预检要求 `ids/len_ds/n_starts` 逐一相同，而分片的**设计**就是让 `ids` 不同 ⇒ 必然误报；② **推理次数真的不等**：分片只对齐了「轨迹条数 k」，没对齐「每条轨迹几个窗口」（窗口数 = `ceil(帧数/stride)`）⇒ 真跑下去会在 all-gather 上卡死（预检拦对了） | **改动本意（用户 2026-10-12 四问）**：<br>① **不变量**：分片模式下各 rank 的 `n_starts`（= 集合通信次数）**必须逐 rank 相等**；各 rank 的 `ids` 并集**必须恰好等于全集** `n_ids`（不漏不越界）；**非分片路径**（`AL_EVAL_SHARD=0`，默认）`_preflight_verdict` 与 `per_episode_starts` 行为**逐字不变**。<br>② **出错症状**：日志出现 `多卡评测一致性预检未通过` + `n_starts=3 vs 4` 之类 ⇒ 窗口数没锁死；出现 `[shard] 分片并集 N 条 ≠ 全集 M 条` ⇒ 漏评/越界；实测 GMean 与历史 run 不可逐位比（**预期**，见③）。<br>③ **快速回滚**：`AL_EVAL_SHARD=0`（**进程启动时读一次 ⇒ 必须重启**）；或 `git checkout <sha>~1 -- lingbotvla/utils/open_loop_validation.py lingbotvla/auto_learning/real/backend.py`。<br>④ **验证入口**：`PYTHONPATH=$PWD /opt/robotwin-env/bin/python -m pytest tests/test_al_eval_shard_preflight.py -q`（真机 24 passed）；真机看首轮评测是否出现 `[open_loop][shard]` 且不再报预检失败 | ⚠️ **口径**：窗口起点与 `stride=50` 不同 ⇒ 实测 GMean 偏离按 stride=50 标定的 `pass_thresholds` 与 `task_baseline`（**未重标定**，用户选 5b：先看偏差）；与 #6（分片本体）**强耦合**：本改动修的就是 #6 真机跑不通的问题 | `TODO_SHA` | 🧪 部分验证（新单测 24 passed 于真机 robotwin-env；**真机训练未验证**） |
| 14 | 10-12 | 删 nmse 误导：① `configs/auto_learning/al_eval2_target15_m60.yaml` 删 `pass_nmse: 0.35` 兼容占位；② `experiment/robotwin/al_50task_bf16.sh` 启动横幅由**硬编码** `及格线 = pass_nmse=0.35（nmse ≤ 该值判 PASS）` 改为**按配置实际判据**显示（`read -r` 增第 6 个变量 `AL_PASS_METRIC`、内嵌 python 多输出 `pass_metric`、横幅显示 `${AL_PASS_METRIC}`） | **用户 2026-10-12 指出**：本批 run 判据是 `pass_metric: gmean_mse`（实测 gmean ≤ 该任务阈值表），但配置里留着 `pass_nmse: 0.35`、启动横幅也硬编码打印 nmse 及格线 ⇒ **误导读日志的人**（本 agent 因此拿 `nmse ÷ 0.35` 算过倍率，被用户当场纠正）。**改动本意四问**：<br>① **不变量**：`pass_metric: gmean_mse` 时判定**只读**阈值表、**绝不读** `pass_nmse`（`decision/thresholds.py::pass_line` 已保证）；横幅显示值必须等于配置里的 `pass_metric`。<br>② **出错症状**：横幅打印 `判据 = （空）` ⇒ `AL_PASS_METRIC` 未取到；日志/文档里再出现「及格线 nmse ≤ 0.35」字样 ⇒ 又有地方硬编码。<br>③ **快速回滚**：`git checkout <sha>~1 -- <三个文件>`；⚠️ 注意 `pass_nmse` 在 `state/persistence.py::SEMANTIC_KEYS` 里 ⇒ 删它**会改配置指纹**（须在重算 baseline / 重扫之前完成，只需对齐一次）。<br>④ **验证入口**：两份配置 `yaml.safe_load` 后 `pass_metric=gmean_mse`、`pass_nmse` 不存在；`bash -n experiment/robotwin/al_50task_bf16.sh`；启动横幅出现 `判据 = gmean_mse` | `TODO_SHA` | 🧪 部分验证（配置解析 + bash 语法已验；**启动横幅未实跑**） |
| 15 | 10-12 | 分片修复（**5 个 bug 一次收口**，全部真机定位）：<br>**①协议越位**：分片切分原在「多卡协议**之外**」⇒ 框架「只有 rank0 算数 + 广播」把各卡切片作废（实测回收 7 份全同）⇒ 改为**协议内**切片 + 归并（`merge_ranks` 回调）；<br>**②默认关**：归并回调仅在策略注入时启用（默认 `None` ⇒ 非分片路径逐字不变）；<br>**③只注入不收回**：原 `finally` 清除与各卡返回时序竞态 ⇒ 后到 rank 读不到策略、不切片；<br>**④元凶·文件级竞态**：`_episode_ids_file` 白名单文件名原为 `&lt;tag&gt;.json`，注释假设「各 rank 内容相同、谁赢都一样」——该假设**在轨迹分片下不成立**，7 卡互相覆盖 ⇒ 后到 rank 读到**别人的切片**（实测 rank2 `run_ids=[1408]` 却返回 `per_traj_ids=[1403]`）⇒ 文件名改为 `&lt;tag&gt;_&lt;本卡切片指纹&gt;.json`；<br>**⑤尺子越界**：`world&lt;=1`（单卡扫描 worker）也启用自适应窗口（`k=ceil(n/1)`）⇒ 把「每 50 帧一个窗口」改成「每条轨迹摊 2 个窗口」⇒ **同一轨迹 MSE 差 0.5–4.7 倍**（实测 `dump_bin_bigbin` 0.147→0.684、`shake_bottle_horizontally` 0.0170→0.0087）⇒ 扫描与历史不可比；改为 `world&gt;1` 才启用（单卡恢复 stride=50） | **用户 2026-10-12**：「分片并行不能改数值」。真机 al_v49→v64 逐步定位：`active_val_gmean_mse=None` ⇒ 判定失效（0 个 PASS）<br>**改动本意四问**：<br>**①不变量**：`AL_EVAL_SHARD=0` 时与改造前**逐字一致**；分片下「每卡推理次数 == 上游公告 `expected_steps`」且各卡相等；**单卡（world≤1）扫描的 gmean 与历史同口径**（stride=50）。<br>**②出错症状**：`active_val_gmean_mse=None`；日志 `归并后逐轨迹条数 N ≠ 全集 M`；`各卡 ids=[[x],[x],…]` 全同；扫描值与历史差 &gt;5%（尤其长帧任务如 `dump_bin_bigbin`）。<br>**③快速回滚**：`AL_EVAL_SHARD=0`（默认即关，**进程启动读一次，需重启**）；或 `git checkout &lt;sha&gt;~1 -- lingbotvla/utils/open_loop_validation.py lingbotvla/auto_learning/real/backend.py`。<br>**④验证入口**：`pytest tests/test_al_eval_shard_merge.py tests/test_al_eval_shard_preflight.py tests/test_multirank_eval_preflight.py -q`（7+26+62 passed；1 个**既有**失败已用 HEAD 版对照证明与本改动无关）；真机日志出现 `[open_loop][shard] … 框架内归并后 N 条`（N==全集）且无「归并后逐轨迹条数」报错；扫描重跑与历史逐任务相对差 &lt;0.5% | `TODO_SHA` | 🧪 部分验证（单测全绿 + 真机 al_v62/v64 实测：归并失败 0、`active_val_gmean_mse=0.0067`、重扫 50/50 任务与历史差 ≤0.4%） |
| 16 | 10-12 | `tasks/vla/train_lingbotvla.py`：**`dcp_save_mode=final_only` 下的周期 HF 存档改为「内存直出」**——新增 `_save_hf_direct_best_effort(checkpoint_path, step)`（复用 `lingbotvla/utils/direct_hf_checkpoint.py::export_model_hf_direct`，把各 rank 权重收成全量 CPU state_dict 后由 rank0 原子发布），并在周期存档分支按 `save_hf_weights` 分流：`true` ⇒ 直出 + `_export_pool_state_here`（池子跟导）；`false` ⇒ 只建目录并打日志。**`dcp_save_mode=always` 那条路（3 个调用点）逐字未动**。<br>另加：直出前把已存在的 `hf_ckpt` 占位挪到 `.stale.<pid>`（成功后删、失败还原），避免 `export_model_hf_direct` 因 `os.path.lexists` 防覆盖而落成 `global_step_N_retry_001/`（与下游约定路径不一致）。 | **真机 al_v72 实测崩溃（step 2600）**：`final_only` 下周期存档**不写 DCP**，而旧 HF 路径 `save_hf_checkpoint_best_effort → async_hf_checkpoint:375 → format_utils:63 ckpt_to_state_dict → dcp_to_torch_state_dict → state_dict_loader:250 assert metadata is not None` ⇒ **AssertionError 杀训练**，已训 44 分钟 + 池子全丢。用户 2026-10-12：「我有仅输出 hf 存档的程序，之前检验过可行的」⇒ 定位到仓库既有 `export_model_hf_direct`（有单测 `test_al_hf_milestone_policy.py`/`test_hf_export_dtype.py` 与真机验收 `tools/acceptance_checks/hf_direct_export_acceptance.py`，判据 `max_abs_diff==0`）。<br>**改动本意四问**：<br>**①不变量**：`dcp_save_mode=always` 时 HF 存档的**函数、参数、输出路径与产物与改动前逐字相同**；`final_only` 下周期存档**绝不读写 DCP**；HF 输出目录恒为 `<ckpt>/global_step_N/hf_ckpt`（下游 `--checkpoint` / 闭环评测认此布局）；HF 存档失败**绝不中断训练**。<br>**②出错症状**：日志出现 `⚠️ HF 直出失败（不影响训练）`（导出失败但不崩）；或 `<ckpt>/global_step_N_retry_001/hf_ckpt` 出现而 `global_step_N/hf_ckpt` 缺失（占位没挪干净）；或 `AssertionError` 又回来（说明走到了旧转档路径）；或 `pool_state.json` 缺失（池子没接上）。<br>**③快速回滚**：`--train.save_hf_weights=false`（**启动读一次 ⇒ 需重启**，回滚后周期不写任何存档，与改动前 `final_only` 行为一致）；代码级回滚 `git checkout &lt;sha&gt;~1 -- tasks/vla/train_lingbotvla.py`。<br>**④验证入口**：日志 `[ckpt] dcp_save_mode=final_only ⇒ step N 直出 HF（不写 DCP）` + `[ckpt] HF 直出完成（不写 DCP）：…` + `[ckpt] HF 导出精度：…` + `[pool_state] 已导出池子：…`；产物 `ls <ckpt>/global_step_N/hf_ckpt/model-*.safetensors`（应 3 片）且 `<ckpt>/global_step_N/` **无 `model/`、`optimizer/`**（证明确实没写 DCP）；单测 `pytest tests/test_al_hf_milestone_policy.py tests/test_hf_export_dtype.py tests/test_dcp_to_hf.py -q`（40 passed，1 个**既有**失败 `test_two_stage_closed_loop_scripts_exist` 与本改动无关——它读 `tools/misc/policy_server.sh`/`eval_tasks.sh`） | 依赖 #12（池子导出挂载点）；**不改** `save_hf_weights` 的全局语义（范围 ②：只修 `always` 之外那条从未工作过的分支），否则会破坏 6 个通用训练脚本 + `gpu96_acceptance.py:530` 的 `HF_REQUIRED_FLAGS` 断言 | `TODO_SHA` | ✅ **已验证**（2026-10-12 真机 al_v75_save3，`--steps 2440` + `SAVE_EVERY=3`）：日志三联命中 —— step 2403 与 2406 均出现 `⇒ HF 直出（不写 DCP、不受 unit 边界约束）` ⇒ `HF 导出精度：源 dtype {bf16×1708} → 目标 bf16` ⇒ `HF 直出完成（不写 DCP）` ⇒ `已导出池子（step=N，9.4 MB）`。产物每份 = `hf_ckpt/`（**3 片 safetensors**，≈12 G）+ `pool_state.json`（9.8 MB），**无 `model/`、`optimizer/`** ⇒ 证明确实没写 DCP；报错 0（v72 的 `AssertionError` 未再出现）；单份直出耗时 ≈19–28 s。单测 40 passed（1 个既有失败与本改动无关） |
| 16b | 10-12 | 追加（用户「如果我不在边界存档呢？」⇒ 方案 B）：**HF 直出与 DCP 延迟解耦** —— 在 `_save_due` 判定**之前**新增独立触发：`if save_hf_weights and dcp_save_mode==final_only and global_step % save_steps == 0:` ⇒ 直接 `_save_hf_direct_best_effort` + `_export_pool_state_here`；DCP 侧的 `_save_due`/`_al_deferred_save` 逻辑**一字未动**（仍只在 unit 边界落盘）。 | HF 存档**不做 resume**、不写 DCP ⇒ 「必须落在 unit 边界」这条约束（为 DCP 的 `at_boundary` 语义而设）对它不成立；不解耦则 `save_steps=3` 名义 3 步、实际仍被推迟到 50 步的 unit 边界。<br>**四问**：① **不变量**：`dcp_save_mode=always` 时行为逐字不变；DCP 仍只在 unit 边界落盘；HF 仍恒落 `<ckpt>/global_step_N/hf_ckpt`。② **出错症状**：日志出现 `[自动学习] step N 不是 learning unit 边界 ⇒ 存档推迟` **且该 step 没有 HF 直出完成行** ⇒ 解耦失效；或同一 step 出现两份 HF（重复直出）。③ **快速回滚**：把 `save_steps` 调大（等价恢复低频）或 `git checkout <sha>~1 -- tasks/vla/train_lingbotvla.py`。④ **验证入口**：日志 `step N（K 的整数倍）⇒ HF 直出` 中 N 对 K 取模为 0 且 **N 不是 unit 边界**（如 2403 在 50 步 unit 内 3/50）。 | 依赖 #16；**仅 `final_only` 生效**（`always` 走 DCP 转档，必须继续遵守边界约束） | `TODO_SHA` | ✅ **已验证**（真机 al_v75_save3：`SAVE_EVERY=3` ⇒ step 2403 **非 unit 边界**（unit 内 3/50 步）照样直出成功，step 2406 同样 ⇒ 证实任意步都能落；`--steps 2440` 下共 2 份产物，报错 0） |

| 18 | 10-14 | `state/registry.py::TaskRegistry.load_state` 保留当前 catalog 派生字段并校验 `sample_probs` / `pass_sampling_snapshot`；`real/build.py` 与 `real/hook.py` 记录丢弃告警；集成回归覆盖 | HF 池子/DCP 状态恢复把旧 `sample_id` 空间覆盖到当前数据集，训练在 sampler 校验时失败；概率表只过滤外来 ID 会改变分布 | 仅恢复动态 AL 状态；任何一个键不属于当前任务时整张对应表清空并告警；当前任务/样本 ID 与 baseline 保持 catalog 值 | HF 池子和 DCP extra_state 共用注册表恢复；只验证 ID 成员关系，无法识别相同 local_idx 对应不同帧，resolver 指纹留作后续 | `d1fc26e` | 🧪 部分验证（CPU 29 passed；真机 `al_v77_sampleidfix` 已过原崩点，step 4055，长跑/DCP 真机恢复仍未验证） |


| 20 | 10-14 | `tools/auto_learning_pipeline/al_checkpoint_watch.py`：单运行周期 HF 监控，满4份删最早1份，保留3份；工具索引与 CPU 回归 | 用户要求按存档数量滚动保留，避免220步存档积累占满盘 | 仅统计完整 HF（配置/index/所有分片长度/同一步号池）；仅清理指定 run 的周期目录；DCP/临时目录/符号链接/受保护路径跳过；单实例锁；默认预览，`--apply` 才删除 | 独立纯标准库进程，无 torch/GPU，不改训练参数；4份同时落盘的峰值、DCP及输入HF仍需预留空间 | `1243886` | 🧪 纯 CPU 12 passed / 0.04 s；真实满4份滚动清理待观察 |
| 21 | 10-14 | `train_lingbotvla.py::main` 周期存档统一入口；`tests/test_al_periodic_checkpoint.py` | 当前运行4400重复HF，4620直出后4650又存；提前HF入口还绕过存档前磁盘保护 | `final_only + HF` 在220的绝对步号倍数只导出一次，不进入延迟队列；先DiskCheck再HF/池；DCP仍遵守安全边界；周期HF不能让收尾跳过完整DCP | 修正 #16b 的双入口；不改参数、评测、缓存释放或导出实现；只改本地，当前训练机继续旧版本；收尾重复HF暂未处理 | 本条提交（周期HF去重与缓存观测） | 🧪 本地纯CPU 9 tests / 0.024 s；四组合周期分支及smoke、零间隔、磁盘拒绝、收尾游标已测；真实导出与恢复未验证 |

> #16b 是10-12的历史设计与局部验收记录。10-14连续运行证据确认独立HF入口与旧周期入口重复触发；本地修正由 #21 接续，尚未部署，不能据 #16b 的旧验收声称当前无重复存档。

| # | 日期 | 改什么（文件/函数） | 为什么 | 期望行为 | 依赖/冲突 | commit | 状态 |
|---|---|---|---|---|---|---|---|
| 22 | 10-14 | `utils/cuda_cache_policy.py::prepare_cuda_cache`；helper/评测/周期HF接线；`al_cache_profile.py`；CPU测试与A/B说明 | 用户要求落实缓存优化建议；当前清理成本与复用收益缺独立证据 | 默认clear保持原清理；可选reuse仅跳过eval_entry/periodic_hf的CUDA清理，GC保留；启动/DCP/收尾/OOM强制清理保留；可选trace记录耗时/显存；CPU报告对比同工作量日志 | 依赖 #21；环境启动读一次；默认参数/行为不变；不部署当前训练；GPU对照待空闲，不能用CPU契约宣称性能/数值等价 | 本条提交（周期HF去重与缓存观测） | 🧪 缓存8项+报告2项纯CPU通过；GPU A/B及复用收益未验证，说明见 `docs/acceptance/cuda_cache_ab_guide.md` |
| 23 | 10-14 | `tools/auto_learning_pipeline/al_panel.py`：识别 `torchrun`/`python -m torch.distributed.run`；TensorBoard 与日志取较新有效步数、TensorBoard 同步步时；兼容常见日志目录；`tests/test_al_panel_live_progress.py` | `torchrun` 入口未被旧探测匹配，训练日志在 `/data/tmp` 时也没被读取，导致运行中的面板显示“无进程/无进度” | **不变量**：只按实际启动器和当前 run 取进程/进度；TB 缺失或落后时保留较新日志步数。**症状**：活跃 run 显示“无进程”或步数停滞。**回滚**：恢复本条涉及的面板和测试文件到改动前版本。**验证入口**：`python lingbot-vla-v2/tests/test_al_panel_live_progress.py` 与 `python -m py_compile lingbot-vla-v2/tools/auto_learning_pipeline/al_panel.py lingbot-vla-v2/tests/test_al_panel_live_progress.py` | 通用启动器匹配；优先使用存在的 `AL_LOG_PATH`，否则回退面板/`AL_LOG_ROOT` 常见路径；不触碰训练/GPU | `75002ff` | 🧪 本地纯 CPU 10 passed；训练机仍运行旧代码，远端效果待同步后验证 |
| 24 | 10-14 | `configs/auto_learning/al_eval2_target50_m60_step100_h400_noreview.yaml::review_after_task_transitions`：10 → 20 | 用户要求恢复训练时降低 PASS 池复查频率 | **不变量**：仅每 20 次状态转变触发 PASS 池 review；0 才是关闭。**症状**：每 10 次 transition 就触发复查，或配置注释与行为不符。**回滚**：将该配置值恢复为 10。**验证入口**：启动计划应打印该 AL_CFG；加载后 `review_after_task_transitions == 20` | 仅恢复 HF 后新进程读取；当前进程不热改。transition 包含 PASS/DEFER/REOPEN，不等价于 20 个互不重复的任务；文件名 `noreview` 为历史遗留 | `6689237` | ⏳ 配置已改；启动前需核对训练计划及恢复存档 |

| 25 | 10-14 | `experiment/robotwin/run_clean_benchmark.py` NVIDIA 单卡适配：复用仓库 launcher；官方任务列表、成功率解析、resume manifest；文档与 CPU 契约测试 | 用户要求训练完成后用该 benchmark 做 NVIDIA 单卡闭环评测，且不能影响当前训练 | 只启动1个 CUDA 推理服务和1个串行仿真槽；任务集合与顺序来自现有 launcher；仅满回合任务写 done；恢复时评测参数必须匹配 manifest | 仅本地代码；不部署/不启动GPU；expert 种子和上游口径不等价；GPU闭环待训练结束 | 本条提交（NVIDIA单卡适配）；原样迁入基线 `1d8bc31` | 🧪 纯CPU 6 passed；py_compile、--help、bash -n、diff检查通过；真实闭环未验证 |
| 26 | 10-14 | `experiment/robotwin/run_clean_benchmark.py::_launcher_command` 与 `tests/test_clean_benchmark_nvidia_adapter.py`：传给 launcher 的 `--inference_workdir` 补尾随路径分隔符，并断言客户端文件可由该参数路径解析 | 首次启动在模型加载前报错，launcher 将仓库路径与 `experiment/...` 直接拼接，形成不存在的客户端路径 | **不变量**：`<inference_workdir>experiment/robotwin/eval_policy_client_lingbotvla.py` 必须存在；BF16、单推理槽/仿真槽参数不变 | 仅影响 benchmark launcher 路径；不影响训练流程；修复后需在 GPU 上确认客户端启动及显存加载 | 待本次提交 | 🧪 CPU契约6 passed；py_compile、bash -n、diff检查通过；GPU benchmark待启动；**四问**：症状=日志 `eval_policy_client_lingbotvla.py` 路径不存在且显存未加载；回滚=恢复 `_launcher_command` 的 `str(REPO_ROOT)` 并移除此契约断言；验证=`python3 tests/test_clean_benchmark_nvidia_adapter.py` + 远端确认模型进程与显存占用 |

| 27 | 10-15 | `learn_skill_mix_task`：指定任务/Replay、PASS 与任务/样本扫描开关、数据对拍缓存、HF-only 实验 | 单技能可学性实验，保留独立验证；闭环尚不接门控 | **不变量**：关闭 PASS 时所有自动入口均不能毕业；关闭 hardness 时 select/continue/rescue/旧权重均不能启用非均匀采样；HF-only 全程不写 DCP；数据映射变更使对拍缓存失效。**症状**：意外 PASS、hardness 打分、DCP model/optimizer 目录、错误任务数据。**回滚**：使用旧配置及 `dcp_save_mode=always`，所有新开关启动读取；实验产物不支持精确续训。**验证入口**：`PYTHONPATH=.:tests python -m unittest test_al_skill_experiment test_al_data_audit test_al_hf_only test_al_periodic_checkpoint test_skill_mix_task_launcher -v` | 独立目录 `/data/experiments/learn_skill_mix_task`；生产分支未动；保留旧默认模式；GPU 加载/训练/闭环未验；18 GB 空闲不足保留五份 HF | `6470669` | 🧪 上机 34 新/周期单测 + 182 CPU 回归通过；真实 40 train/10 val、10769帧映射对拍及缓存通过；6 组旧模式与 `956782f` 对拍一致；tokenizer 误报见 #28；完整证据见 `skill_mix_task_experiment_zh.md` |
| 28 | 10-15 | 启动器 tokenizer 校验改为库规范化语义比较 | CKPT9680 与 Qwen 的合并规则存储格式、默认字段及 72 长度设置导致原 JSON 相等检查误报；只读取证后单独获批 | **不变量**：仅忽略运行期 padding/truncation 和库兼容的序列化差异，词表/ID/有序合并/特殊 token/分词规则差异仍拒绝，不改源文件。**症状**：相同 tokenizer 被拦截，或不同 token ID/规则可生成计划。**回滚**：恢复 `6470669` 的启动器与对应测试（回到严格原 JSON 比较，会恢复本次误报），新进程生效。**验证入口**：`PYTHONPATH=.:tests python -m unittest test_skill_mix_task_launcher -v` | 只影响实验启动器预检/来源报告；依赖 tokenizers（原项目已有依赖）；不影响训练 tokenizer 的长度设置 | `686bdee` | 🧪 本地共 37 项纯 CPU 通过；服务器最终 37 项单测及真实 CKPT9680 预检通过；GPU 待验 |
| 29 | 10-15 | 单技能实验每 250 步保存 BF16 HF | 用户要求 1000 步只保存 250/500/750/1000 四份；旧档只留两份是一次性清理，不改新实验保留机制 | **不变量**：命令 `train.save_steps=250`；250/750 不等评测边界；1000 周期/结束只导出一次；不写 DCP。**症状**：仍生成200档、250推迟300、1000重复导出。**回滚**：启动器 `train.save_steps` 改回200，重启新run生效。**验证入口**：`PYTHONPATH=.:tests python -m unittest test_al_hf_only test_skill_mix_task_launcher -v` | 仅实验启动器参数、对应契约与文档；100步独立评测不变；旧档清理后可用76.8GiB，四份HF预计47.5GiB；GPU未验 | 本条所在提交 | 本地37项纯CPU通过；同步后核对实际计划含250，不启动训练 |

**#30（10-15）实验 worker 导入路径修复**：首次 GPU 运行在 step0 以 `no_eligible_candidate` 退出；只读复现实际导入 `/data/code` 旧配置与调度器，四个新开关均不存在。已获批修复、提交同步并换新目录立即重启。**不变量**：torchrun/worker 的 `PYTHONPATH` 以实验包根开头，保留原环境其余路径；不改生产安装。**症状**：关闭扫描仍出现 bootstrap，或模块来源指向 `/data/code`。**回滚**：撤回启动器的 `PYTHONPATH` 设置并停止实验启动（会恢复误导入，不能继续旧命令）。**验证入口**：`PYTHONPATH=.:tests python -m unittest test_skill_mix_task_launcher.LauncherTests.test_execute_worker_imports_experiment_before_installed_package -v`；本地38项CPU通过，包含真实子进程与竞争旧包的来源/开关契约。step0 HF保留作证据，未训练；GPU重试待同步后核实。

| 31 | 10-16 | `Scheduler.refresh_current_hardness` / `AutoLearnLoopHook.on_save` / 周期存档分支 / hardness scorer / 实验启动器：任务入口与周期成功存档后双触发 | 单任务长跑更新难度分布，避免一直使用入口概率；修复旧缓存、结束白扫、步号滞后与空存档触发 | **不变量**：开关关闭不打分；周期保存成功且仍会训练才重扫；按当前权重打分且不读写来源模型缓存；不修改 unit 内采样表；记录实际 AL 步与存档绝对步，Scheduler 记账不变。**症状**：250 被记成200、最终1000仍扫描、无HF也扫描、重扫命中来源缓存。**回滚**：新run配置 `hardness_scan_enabled=false` 并重启，恢复均匀采样。**验证入口**：`PYTHONPATH=.:tests python -m unittest test_al_skill_experiment test_al_periodic_checkpoint test_al_hf_only test_al_save_hardness_refresh test_skill_mix_task_launcher -v` | 仅实验分支；既有 continue/rescue 策略不变；旧 DCP 单元边界不变；实验 HF-only 不变；第301/501/801次迭代使用新概率 | `7b9d73f` | 🧪 本地 CPU 去重后62项通过（Python 3.12/3.13 分环境）；GPU未验证 |

**#32（10-16）HF 便捷续练与可选自动关机**：新增 `experiment/robotwin/continue_skill_training.py`，从来源 HF 自动读取原 run 的训练配置、数据集与对拍报告，复用现有启动器 CPU 预检；命令支持常用 AL 布尔/数值覆盖和通用 JSON 字段覆盖；默认 micro24/GAS1、新增500步，`--execute` 经只读 GPU/磁盘检查后用 setsid/nohup 后台启动。**不变量**：默认预览；`--shutdown` 才启用关机，`--no-shutdown` 显式关闭；训练退出0且目标步HF索引/分片/关键资源及张量名与来源一致才执行 `bash /usr/bin/shutdown`；失败不关机、不重试；不删存档、不启动清理监控；固定 run name 的预览/执行路径相同，既有记录不覆盖。**症状**：缺最终HF仍关机、默认启动GPU、Exec format error、来源配置丢失、主终端断开导致训练停止。**回滚**：使用旧完整启动命令，不使用新便捷入口；新启动不加 `--shutdown`。**验证入口**：`PYTHONPATH=.:tests python -m unittest test_continue_skill_training test_skill_mix_task_launcher -v`；新增用例全用子进程替身，不执行真实关机。代码已随 `7b9d73f` 提交；GPU与真实关机未验。

**#33（10-16）评测专用全景相机与按仿真时间录像**：独立 1280×720 外部相机，默认 60 FPS，可调 30；独立录像模块由现有测评自动调用，不进入模型输入。**不变量**：关闭不新增相机/采帧；开启不推进物理、不重新抽取灯光、不改策略观测或 RNG；以物理步计时采新帧，正常结束/Python异常均关闭编码器，最终成功帧覆盖。**症状**：视频加速/重复帧、失去最终动作、无帧/黑帧、模型输入改变、ffmpeg 残留。**快速回滚**：下一次测评去掉 --global_view / --global-view（启动时读取）；已有录像保留。**验证入口**：PYTHONPATH=.:tests python3 -m unittest test_robotwin_global_view_video test_clean_benchmark_nvidia_adapter -v；🧪 本地22项CPU通过，含真实H.264/ffprobe验证60 FPS、61帧/1.016667秒；语法/bash -n/diff检查通过。GPU构图/黑帧/性能待验，未同步服务器。依赖：RoboTwin相机/基类/新模块一起同步；启动器开启时先检查版本特征及ffmpeg；不涉及AL训练。详见 docs/robotwin_global_view_video_zh.md。未提交。

## §1.5 改动本意速查（**debug 入口**；2026-10-12 用户要求）

> 用法：**出问题时先看这里** —— 回答三个问题：① 这是不是我改的？② 它现在算不算坏？③ 怎么立刻退回去？
> 每条的字段：**意图（替我们保证什么）· 不变量（永远必须成立）· 出错症状（怎么认）· 快速回滚（一步退）· 验证入口**。
> ⚠️ 回滚分两类：**运行时开关**（改环境变量重启即可）与**代码回滚**（`git checkout`）。
> 开关**都在进程启动时读一次** ⇒ 改开关**必须重启进程**才生效（这条踩过坑，别指望热改）。

### 🔧 `AL_*` 开关总表（排查时不必翻代码）

| 开关 | 默认 | 何时读取 | 作用 | 回退语义 |
|---|---|---|---|---|
| `AL_EVAL_SHARD` | `0`（关） | 进程启动 | `1` ⇒ 开环评测（`active_val` + `train_monitor`）按卡分片，每卡恒 `k=ceil(n/W)` 条、汇总去重 | 设回 `0`（或不设）⇒ 恢复"每卡跑整批" |
| `AL_HARDNESS_SHARD` | 关 | 进程启动 | `1` ⇒ hardness 打分按 `ids[rank::world]` 帧级分片 + `all_gather_object` 汇总 | 不设 ⇒ 每卡各算全量（慢但等价） |
| `AL_HARDNESS_FORCE_SNAPSHOT` | `0`（关） | 进程启动 | `1` ⇒ **强制**每批都做 RNG 快照/还原（O1 的 A/B 对照用） | 不设 ⇒ O1 的"无增强要关时跳过快照"生效 |
| `AL_SCAN_SKIP_HARDNESS` | 关 | 启动器注入（仅预扫 worker） | `1` ⇒ 预扫只补 scout，不进 `_select()`（不跑 hardness/active_val） | 不设 ⇒ 完整流程（预扫会顺带扫 hardness） |
| `AL_VIDEO_BACKEND` / `video_backend` | `pyav` | 数据集构造 | ❌ **未采纳**（torchcodec 在本机 ABI 不匹配、收益有限） | — |
| `AL_CUDA_CACHE_POLICY` | `clear` | 模块首次导入（需重启） | `reuse`仅跳过评测入口/周期HF的CUDA空闲块释放；GC保留，OOM强制清理不变 | 改回`clear`恢复原清理，复用尚未GPU验收 |
| `AL_CUDA_CACHE_TRACE` | `0` | 模块首次导入（需重启） | `1`记录清理调用墙钟与前后显存；不额外同步、不重置峰值统计 | `0`关闭诊断读取和日志 |
| `SAVE_EVERY` / `HF_PASS_INTERVAL` | 训练脚本默认 | 进程启动 | 存档节奏（DCP 步数）/ HF 里程碑；**必须用 `--env` 或 shell 变量传**（`base_env()` 白名单已放开） | 去掉 ⇒ 回默认 |

### 逐条本意

**#25 benchmark 单卡适配**：不变量是适配器只经 `start_robotwin_infer_and_eval.sh` 启动单 CUDA 服务/单仿真槽，任务只有日志确认跑满设定回合才写 done，resume 参数必须匹配 manifest。出错症状：命令含非1 GPU、日志无完整 `Success rate`、任务回合不足却出现在 summary/done，或恢复参数不匹配仍继续。快速回滚：恢复提交 `1d8bc31` 中的原始 `run_clean_benchmark.py` 及对应说明/测试。验证入口：`python3 tests/test_clean_benchmark_nvidia_adapter.py` 与 `python3 -m py_compile experiment/robotwin/run_clean_benchmark.py tests/test_clean_benchmark_nvidia_adapter.py`；GPU闭环需训练结束后验收。

**#26 benchmark inference workdir 分隔符**：适配器传给 launcher 的工作目录以路径分隔符结尾，确保 launcher 拼出的 inference client 路径真实存在；症状是客户端文件不存在且模型未加载。快速回滚为恢复 `str(REPO_ROOT)` 并移除对应断言。验证入口：`python3 tests/test_clean_benchmark_nvidia_adapter.py`，以及 GPU 日志出现 inference client 启动并观察到模型显存占用。

| 编号 | 意图（替我们保证什么） | 不变量（永远必须成立） | 出错症状（怎么认） | 快速回滚 | 验证入口 |
|---|---|---|---|---|---|
| **#6** 评测轨迹分片 | 训练态评测**不再每卡跑整批**（省 `1/W` 墙钟），且**集合通信次数对齐** | ① 每卡前向次数**恒等**（`k=ceil(n/W)`，回绕补齐）；② 汇总后逐轨迹 MSE 与"单卡全量"**逐位一致**；③ 汇总按 episode id **去重**（`merged ≤ n`，多于 n 说明去重失效） | ① **静默死锁**（无报错、卡住 ⇒ 最危险）；② 日志 `[open_loop][shard]` 缺失或 `merged=n_trajs` 不符；③ 指标与历史 run **不该有**的偏差（分片本身不改变口径） | `AL_EVAL_SHARD=0`（默认关，需重启）；代码回滚 `git checkout 7164cde~1 -- lingbotvla/auto_learning/real/backend.py`（`7164cde` = 分片那次提交） | 日志匹配 `[open_loop][shard] world=W local=k merged=n/n`；单测 `pytest tests/test_al_eval_shard.py`（11 项） |
| **#9** O1：跳过快照 | 省掉"每批一次无用的 CUDA RNG 快照/还原"（本 run `image_augment=False` ⇒ 它是恒等操作） | **`disabled` 为空时，取 item 前后 `random/numpy/torch` RNG 状态逐位不变**；有增强要关时**仍**快照/还原（安全底线与开关无关） | ① 训练**随机性异常**（同 seed 两次结果不同、图像增强"看起来生效了却没生效"）；② 报告里 `rng_touched` 与实际不符；③ 硬扫结果与历史不可比 | `AL_HARDNESS_FORCE_SNAPSHOT=1`（**需重启**，强制旧行为）；代码回滚 = 删掉 `if not disabled and _fast_fetch_enabled():` 早退，或 `git checkout 3e1ea94~1 -- lingbotvla/auto_learning/real/determinism.py` | `pytest tests/test_al_deterministic_sampling.py`（7 项）+ `tests/test_b1_v0_2_fixes.py` |
| **#2** confirm 复用 scout | confirm 的 4 条中 2 条与 scout 重复 ⇒ **只新评 2 条**，省一半推理 | 判决值 = "scout 已测 2 条（复用）+ 新评 2 条"合并后的**几何平均**，且 == 一次性评 4 条的 `gmean_mse`（误差 < 1e-12） | ① 日志缺 `[open_loop][confirm] … 复用 scout N 条 + 新评 M 条`；② 事件里 `confirm_reused/fresh_trajs` 不是 `2/2`；③ `confirm_*` 与历史 run 逐位比较**有偏差**（口径已变，属预期） | 无开关；代码回滚 `git checkout d014aa6~1 -- lingbotvla/auto_learning/real/backend.py lingbotvla/auto_learning/orchestration/scheduler.py` | `pytest tests/test_al_confirm_merge.py`（5 项）；日志 `[open_loop][confirm]` |
| **#8** 面板池子/进步 | 池子状态从**事件流推导**改为读 **DCP 注册表**（权威），且能看 DEFER 原因/进步/轨迹 | 面板显示的状态计数与 DCP `auto_learning.registry.records` **完全一致**；只读（不写 DCP、不占 GPU） | ① 面板出现 `⚠️ 读 DCP 注册表失败`；② 池子计数与 `--pool` 明细行数不符；③ 快照停在旧 `global_step`（正常：读的是最近一份存档） | 无开关；`git checkout 606c67a~1 -- tools/auto_learning_pipeline/al_panel.py` | `al_panel.py <run> --pool`（计数与 DCP 一致即可） |
| **#3** v45 帧上限 200 | 单任务扫描量 600 → 400 帧（−33%），帧多的任务省时间 | 压缩是**均匀**的（日志给出 `A → B` 帧数）；**不影响判定口径**（hardness 只做采样权重） | ① 日志缺 `（帧数上限 200：… A → B 帧）`；② 扫描墙钟没降（说明配置没生效）；③ 硬度分布异常偏（压缩过度时看 p90） | 改回 `hardness_max_frames_per_traj: 300`（需重启）；**只对 v45 生效**，v44 不受影响 | 日志匹配 `帧数上限 200`；对比单任务扫描用时（v44 基线 275–466 s） |

| **#13** 分片预检修复 | 让「每卡推理次数」**结构上恒等**，使多卡分片评测真机可跑（#6 才真正可用） | 分片下各 rank `n_starts` 相等；`ids` 并集 = 全集 `n_ids`；非分片路径逐字不变 | 日志 `多卡评测一致性预检未通过` + `n_starts=x vs y`；或 `[shard] 分片并集 N ≠ M` | `AL_EVAL_SHARD=0`（**需重启**）或 `git checkout <sha>~1 -- <两文件>` | `pytest tests/test_al_eval_shard_preflight.py -q`（真机 24 passed） |
| **#18** 采样状态按当前 catalog 校验 | 旧池子只恢复生命周期/指标；分布只在样本 ID 全属于当前任务时恢复 | ① catalog 的 split/sample IDs/baseline 不被存档覆盖；② 任一外来 ID ⇒ 对应概率表整表清空；③ 合法表保留原值 | sampler 再报 `sample_id` 不在任务训练集；或 `[pool_state]/[auto_learning] 丢弃不兼容采样表` 告警显示 task/field/count | `git checkout <sha>~1 -- lingbotvla/auto_learning/state/registry.py lingbotvla/auto_learning/real/build.py lingbotvla/auto_learning/real/hook.py tests/test_auto_learning_integration.py` | `/data/miniconda3/envs/lingbotvla/bin/python -m pytest tests/test_auto_learning_integration.py tests/test_persistence.py tests/test_resume.py -q`；训练机 29 passed，4.37 s |

| **#19** HF 池生命周期保护 | 继承池文件而非重新初始化已存档任务；拦住超预算训练 | ① 导入的 PASS/DEFER/EXHAUSTED 在首次调度前不变；② 同一步号仅新任务 bootstrap，较新权重可重测旧候选；③ start_attempt 前 attempt_count 小于上限；④ rescan 免费通关不占 attempt | 日志导入计数正确但随后 50 个 bootstrap 重写状态；或 `select.attempt` 超过配置上限 | `git checkout d1fc26e -- lingbot-vla-v2/lingbotvla/auto_learning/orchestration/scheduler.py lingbot-vla-v2/lingbotvla/auto_learning/real/build.py`（仓库根执行，需重启；会重新暴露状态覆盖问题） | `python -m pytest tests/test_al_hf_pool_restore.py tests/test_resume.py tests/test_persistence.py tests/test_scheduler.py tests/test_state_machine.py tests/test_al_pool_state.py -q`；启动日志 `[pool_state] 保留 PASS/DEFER/EXHAUSTED 状态`；同一步号既有任务无需预扫，较新权重只重测候选，并核对真实计数 |

| **#20** 周期 HF 滚动保留 | 满4份删除最早完整周期HF，留下3份 | 删除路径必须是指定 run/checkpoints 的直接子目录，分片与池完整，且不含DCP/临时发布目录；受保护路径不删 | 日志 `prune_begin` / `removed` 与实际完整周期目录数不符；未完整落盘的目录被计入 | 停止对应监控 PID（需用户同意）；下次启动不加 `--apply` 即只预览；已经删除的旧HF无法恢复 | `python -m pytest tests/test_al_checkpoint_watch.py -q`；`python tools/auto_learning_pipeline/al_checkpoint_watch.py <RUN_DIR> --once` 预览，获准后 `--apply` 后台监控；每15秒检查 |
| **#21** 周期HF唯一入口 | 去掉重复写盘，同时保留磁盘保护与收尾完整状态 | `final_only + HF` 每个周期只有一对HF/池导出；始终先查磁盘；周期HF不更新完整存档游标；`always` 的DCP延迟规则不变 | 同一步出现多条周期直出HF日志；220非倍数出现周期HF；磁盘拒绝后仍HF导出；停止后缺完整DCP | 仓库根执行 `git restore --source=1243886 -- lingbot-vla-v2/tasks/vla/train_lingbotvla.py`；仅影响下次启动，不能热改在跑进程 | 本机 `python3 lingbot-vla-v2/tests/test_al_periodic_checkpoint.py`；下次启动日志 `周期直出 HF` 每个220倍数仅一次，非边界周期后无额外边界HF |
| **#22** 缓存复用与观测 | 让clear/reuse可对照，定位调用成本，避免盲目改变默认 | 默认仍清理；reuse只作用两个入口；GC与OOM回退不取消；不读/写模型或RNG；trace不改变原清理异常传播 | reuse下新增OOM/批大小降级、恢复审计失败、整段墙钟变慢；trace日志缺context/policy或诊断失败阻断清理 | `AL_CUDA_CACHE_POLICY=clear AL_CUDA_CACHE_TRACE=0`下次启动回原清理；本次无当前运行修改 | 本机运行 `tests/test_cuda_cache_policy.py`、`tests/test_al_cache_profile.py`；空闲GPU按A/B说明覆盖评测和周期存档，不能仅验函数 |

## §2 已验证（可不再惦记）

### #78 只保留R9-step500与R10-step800（2026-10-21）

按用户明确授权，仅保留指定R9最终500和R10最终800，删除其余7份global_step，释放83.24GiB。
删除前后两份保留HF的1708张量/BF16/分片索引均校验通过，删除后完整枚举仅余两份。
更新存档清单和流水，保留日志/评测/视频/数据。证据为20261021_keep_R9_500_R10_800_completed.json。
不变量：指定保留模型不删、仅精确ckpt目录；症状：剩余清单与两份不符；回滚：删除不可逆，不承诺恢复；验证入口为完成清单及HF头部核验。无训练/GPU测试、无关机。

### #73–#74、#77 R9/R10训练闭环归档（2026-10-21）

- R9从R8-step600续训500步，stapler200/pillbottle200/switch100，200/400/500已验证；最终0/3、0/3、2/3。按用户授权此前清理9份旧模型，保留R6-step500、R8-step600，释放107.05GiB；精确清单已归档。
- R10从R9-step500续训800步，playingcard/phone/scale/qrcode各200，最终1/3、0/3、0/3、1/3，OIDN均0；200/400/600/800实际HF头部/索引逐项通过，1708张量BF16。
- 已更新流水/累计/存档清单及原始审计；资格池增加两个1/3任务，从12变14，保留旧注册表。复习轮换仅CPU契约验证，不将R10无复习的成功运行当作新复习模式GPU验证。
- 不变量：来源/预算/保存与用户授权一致、历史与计划区分；症状：来源错误、漏边界保存或原始结果不一致；回滚：不用新配方/文档按Git回退（删除的旧模型无恢复承诺）；验证入口：R9/R10 pipeline终态、verified_checkpoints、evaluation汇总、20261021_R10_final_audit及test_replay_rotation。
- 当前nvidia-smi命令Permission denied，未据此判断GPU空闲；本次所有核对均只读、标准库文件解析，未启停GPU进程、补课或清理。


### #61–#63 已完成实验归档（2026-10-20）

✅ 三任务400训练与两份HF完成，后按用户要求删除400；缩减后的共享闭环为adjust_bottle2/3、click_bell3/3，同PID且只加载一次模型。后续plate单任务200与HF完整验证完成，闭环1/3、OIDN错误0。以下保留实现期说明，阶段措辞以本段终态为准：

**#63 place_container_plate单任务续训（2026-10-20）：🧪 CPU8项回归、真实50回合与三步GPU通过，正式200步训练中。** 用户要求先更新旧改动（064a1d8已push），删除三任务400（精确删除，其他200和9680保留），从原adjust_bottle BF16-200只训目标50条7934帧、56/GAS1、200步并自动闭环3回合。新增single_task_trial复用现有CPU/探针/HF/评测，未改模型与损失。不变量：指定来源、单任务、普通均匀、200保存与评测；症状：任务白名单/精度/HF/渲染失败；回滚：不运行新入口，原基准保留；验证：test_single_task_trial、cpu_report/probe/verified_checkpoint/evaluation_200。架构优化必须在正式启动后本地隔离开发，不修改本次已冻结配置。
**#62 多任务复用一次推理服务（2026-10-20）：🧪 服务器20项CPU回归通过，实际两任务评测启动。** 用户要求停止旧队列，仅评adjust_bottle/click_bell各3回合；已停止旧控制器、当前评测、服务与仿真进程组，旧结果保留。新增shared_evaluation跨任务共用同一模型/服务，evaluate.execute可接共享session，原单任务调用默认行为不变。不变量：只加载一次相同400模型、每回合reset、同FP32/chunk50/渲染检查，结束/异常仅会话所有者关服务；症状：PID变化、重复load、跨模型复用、审计/seed/渲染失败；回滚：新任务不使用共享session，旧单任务API保留；验证入口：test_shared_evaluation生命周期/异常清理/不同模型拒绝，以及新目录session/status/summary。尚未完成两个任务，不预报成功率。
**#61 BF16-200续训三任务（2026-10-20）：🧪 服务器17项CPU契约、真实150回合整链、三步GPU通过，正式400步已启动。** 用户授权删除除9680/BF16-200外旧存档；已精确删除四份FP32 250/500，报告three_task_cleanup_20261020.json。新增three_task.py的任务等概率采样/准备/CPU装配与three_task_pipeline.py串行400步→200/400保存→仅400评8任务各3回合。通用CPU检查增加显式episodes/save_steps/数据回调，默认旧行为不变；评测增加task参数，默认adjust_bottle不变。不变量：来源BF16-200、150条24119帧、三任务各1/3、56/GAS1、BF16、200/400；症状：episode或概率质量不符、dtype/梯度/HF/seed/OIDN失败；回滚：不启新入口，旧基准独立保留；验证入口：test_three_task、cpu_report、probe/summary、verified_checkpoints、evaluation_400。无自动关机。详见[续训说明](../training/experiments/multi_task/three_task_bf16_continuation_zh.md)。


### #60 采用BF16基准并归档可复用入口（2026-10-20）

✅ 已验证。用户在确认3/3后授权更新仓库。新增基准JSON快照（实际配置/存档/五组结果）、baseline_pipeline、test_baseline和专门说明；README/docs/AGENTS只增加入口指针。bf16_trial.prepare可从冻结快照准备，旧显式source方式保留；评测可额外核对expected_seeds，原默认行为不变。
不变量：固定源9680、BF16/56/GAS1/200、三增强关闭、每250步与200最终保存、同3实际seed；生产分支/目录不变，不提交权重或凭据。症状：快照重建配置漂移、错误精度HF被接受、默认预览产生训练副作用、评测种子不符。回滚：不启baseline_pipeline，保留旧入口和旧模型；只对新进程生效。验证：服务器13项CPU测试通过，冻结快照重建实际50回合7156帧CPU整链通过；原成功run的200步BF16 HF和3/3闭环独立取证。新包装入口没有再跑一次GPU训练，不冒称完整流水线重跑验证。

### #55–#59 adjust_bottle基准验收（2026-10-20）

✅ 已验证，用户选择BF16/GBS56/200步作为本任务当前实验基准；不改变生产Task_Auto_Learning。
标准FP32 250/500各2/3，增强FP32 250为1/3、500为2/3；BF16标准200为3/3，三实际seed相同、OIDN错误0。
BF16最终HF1708张量完整，约11.88GiB；CPU整链、三步GPU dtype/梯度及正式200步/闭环均通过。
新增冻结快照、可复用baseline_pipeline与专门说明；复用入口CPU契约验证与旧成功run证据分开记载。
不变量/症状/回滚/验证入口见[基准专门说明](../training/experiments/adjust_bottle/adjust_bottle_bf16_baseline_zh.md)。快照只保存配置和结果，不提交模型权重或凭据。
以下保留#55–#58的历史阶段记录（其中待验证/未提交措辞是当时状态，以本段验收为准）：

**#58 adjust_bottle两组自动配对队列（2026-10-20）：🧪 纯CPU契约通过，等待当前组训练结束。** 新增`experiment/paired_control/`准备器、3功能参数、独立窗口DataLoader注入、CPU/FK装配检查和串行队列；评测器新增可选episodes/reference参数，未提供时保持原10回合基线行为。标准组500步完成并核验两份HF后评3回合，再从9680在隔离目录开启EE3/.8/1.5、窗口2倍、LN-FM训练500步，250/500各存，最后相同3实际场景评测。其余配置逐项继承，未改当前训练配置/核心SHA；不启AL调度，不添加开环或早停。动机：直接比较三项增强组合与标准学习方法。**不变量**：当前组配置与核心不变、源9680、两份存档、同场景/精度/chunk；**症状**：queue failed、缓存坐标错误、EE未生效、非F32、缺存档、评测种子不同；**回滚**：只停新队列/其子任务，或新进程用`--no-ee-loss --no-window-sampling --no-ln-fm`，不影响当前组；**验证入口**：`python -m unittest experiment.paired_control.test_methods -v`本地/服务器5项通过（8种开关子组合）；队列`/data/outputs/adjust_bottle_paired_20261020/status.json`已等待父训练。Torch整链/探针推迟至当前训练和第一组评测结束，无并发GPU。见[说明](../training/experiments/adjust_bottle/adjust_bottle_paired_control_zh.md)，尚未提交。
**#57 对照开启编译并恢复8个数据进程（2026-10-20）：🧪 CPU复验和三步GPU通过。** 按用户要求`use_compile=true`，撤销准备器保守的`num_workers=4`改回初始8；`AL_DYNAMO_CACHE_SIZE_LIMIT=0`跳过本项目缓存上限调优。micro28/GAS1不变，峰值分配81.761GiB/预留91.799GiB；第2/3步StepTime9.866/7.676s，仅短探针，不推断长期吞吐。部分Qwen3-VL nonzero发生graph break。独立目录初次基线因漏接根目录.runtime回退旧OIDN，已停止污染评测并保留日志、接入原有2.5.1运行库；7条CPU加载契约通过，新基线日志已枚举Blackwell CUDA。**不变量**：核心模型hash、数据50条和250/500存档不变；**症状**：配置compile=false/workers非8、OIDN错误或基线未完成；**回滚**：新进程配置compile=false、num_workers=4，不能在已运行训练中改；**验证入口**：`gpu_probe_compile.log`、`cpu_compile_check.log`与`pipeline_compile.status.json`。正式500步仍待基线完成后自动启动，补丁未提交。
**#56 FP32对照探针观测functional动作输出（2026-10-20）：🧪 CPU与三步GPU探针通过，正式训练待启动。** 原动作输出走`_fp32_linear/F.linear`，跳过`nn.Linear`模块hook，首个optimizer更新后因动作输出记录为空误判失败（非OOM）。仅在探针进程包裹原functional调用、记录返回值并恢复，不改模型文件/训练损失。新增CPU契约证明输出与输入/权重梯度逐值不变，1 test通过。96GB RTX PRO 6000 Blackwell上micro28/GAS1跑完3步，参数/梯度/优化器浮点状态及四组输出F32；首更新峰值分配76.965GiB/预留77.660GiB，完整3步83.584GiB/90.525GiB。不变量：forward与梯度不变、三核心快照hash不变；症状：action_out缺失、非F32记录；回滚：撤回`gpu_smoke.py`观测补丁，仅影响探针；验证入口：`python -m unittest experiment.official_control.test_audit_contract -v`与探针`summary.json`。已同步独立目录，补丁尚未提交。
**#55 adjust_bottle标准方法对照（2026-10-20）：🧪 CPU整链通过，GPU待测。** 新分支 `official_adjust_bottle_control` 恢复初始快照86f6728的configuration/modeling v1/v2，新增独立准备器，不走AL，关闭窗口/EE/低噪声；50条示范，从9680训练500步，250/500保存FP32 HF。深度与视频、Muon和原学习率保留。动机：排除自定义算法后检验基础单任务学习链。不是上游官方发布复现，保留数据/HF工程兼容修复，单卡batch也不同。服务器实际参数解析、50回合7156帧及逐回合图像/动作/mask检查通过，原FM采样逐值一致；新增3步FP32审计探针与固定种子10回合评测入口，尚未执行GPU。无卡0.5CPU配额时，CPU检查限制单线程，避免248线程争用。**不变量**：核心hash与快照一致、AL为空、EE/低噪声false、存档250/500；**症状**：hash失败、出现AL/EE日志、episode不符；**回滚**：退出本实验并切回learn_skill_mix_task，独立目录不覆盖；**验证入口**：`python experiment/official_control/prepare_adjust_bottle.py --cpu-check`，GPU dtype/容量与闭环尚待验证。见[说明](../training/experiments/adjust_bottle/adjust_bottle_official_control_zh.md)。



**#51补测（2026-10-19）：✅ 修复后10步动作块完整400动作、40次推理，OIDN错误0，仍0/1。** 鼠标最大抬升0.1653mm、最终水平位移22.04mm；50步档分别0.0724mm/125.85mm。初始三相机/状态相同、指令/噪声seed相同，但首块预测不逐位相同，不能将单回合差异严格归因于动作块。两档均未稳定抓取；不启动训练、不改默认动作块。原客户端不变、GPU测试进程已清理。详情与耗时口径见[运行库说明](../training/evaluation/robotwin_render_runtime_zh.md)。

**#51（2026-10-19）Blackwell评测OIDN兼容修复，✅ 静态与400步闭环渲染验证通过；抓取仍失败。** SAPIEN 3.0.0b1自带OIDN2.0.1加载CUDA插件但设备枚举0；隔离2.5.1能识别RTX PRO 6000 Blackwell。`envs/render_runtime.py`在SAPIEN导入前预载独立库并替代旧预载shim；默认仅发现仓库`.runtime/oidn-2.5.1`时启用，显式`ROBOTWIN_OIDN_LIBRARY_DIR`可指定。目录/版本/CUDA检查失败立即报错。不变量：SAPIEN/物理/相机/32spp/模型不变，site-packages/生产目录不改；症状：缺启动标记、maps出现2.0.1或OIDN Error；回滚：新进程`ROBOTWIN_OIDN_LIBRARY_DIR=bundled`（恢复旧缺陷）。验证：CPU契约本地/服务器7 passed；本地配方及启动续训回归62 passed、1 skipped、36 subtests。同seed静态三视角/真实策略输入颗粒消失，实际闭环maps仅2.5.1；新step500/seed100000/noise0/chunk50完整400步8次forward，OIDN Error=0，但成功0/1、鼠标最大抬升0.0724mm。原评测客户端hash不变；测试结束无GPU进程。证据及安装/回滚见[运行库说明](../training/evaluation/robotwin_render_runtime_zh.md)。不把渲染修复等同策略能力提升。

**#38（2026-10-17）面板历史单行布局，✅ 本地/服务器纯CPU与实际run显示通过。** 每条历史占一行，门槛移到标题，EE列依次为位置/姿态/夹爪；标题说明未乘权重、取该训练步。历史门槛有变化时逐行T标明，不套用新门槛。只读面板；不重启训练。

- **不变量：** EE步号绑定不变，不丢夹爪、不逐行重复门槛；旧run兼容，变动门槛不误导。
- **症状/回滚：** 行被截断、三项换序、旧门槛消失时回退面板到 `9964f33`。
- **验证：** 面板17项+循环2项+HF-only7项+周期存档14项，共40项纯CPU通过；实机紧凑/宽视图显示通过，面板不导入torch。证据 `tests/ee_fk/evidence/panel-singleline-2026-10-17/live_check.log.txt`。#39训练循环修复只在服务器临时隔离目录CPU验证，未部署到当前训练工作树。

**#37（2026-10-17）仪表盘历史增加三项EE训练loss，✅ 本地与服务器16项纯CPU测试、真实run紧凑/宽视图通过。** 只改只读面板，在同一次TB加载中提取 `ee_fk/{position,orientation,gripper}_loss` 序列，有unit_loss的历史步显示该步三项未加权值；不改训练器、不重启训练。

- **不变量：** 精确步号对齐，不拿最新值覆盖历史；Step0基线不伪造训练loss；缺失/非有限/部分写入则不显示；不增加TB加载次数、不import torch、不调用GPU。
- **症状：** 100步行出现120步loss、旧run崩溃、窄屏截掉夹爪或误认为unit平均/已加权值。
- **回滚：** 面板回到 `7f66523` 版本；不重启训练。
- **验证入口：** `tests/ee_fk/test_panel_ee_training.py`、`tests/test_al_panel_live_progress.py`；真实显示和torch未导入断言见 `tests/ee_fk/evidence/panel-2026-10-17/live_check.log.txt`。三项loss量纲不同，不能横向比绝对大小；GAS>1沿用训练器最后micro-batch记录，不冒充unit均值。val仍是独立物理误差指标。

**#36（2026-10-17）FK精度隔离与val末端指标，✅ CPU和96G实机接口验收通过。代码 `7f66523`。** 运行期FP32几何独立于模型模块树，局部禁用autocast；val复用原完整去噪预测，增加按轨迹等权的位置RMSE(mm)、姿态角(度)、夹爪MAE与左右臂明细，贯通AL DTO、事件、TB和面板。历史GMSE/阈值/PASS保持。

- **不变量：** 几何不被模型bf16转换或HF存储舍入；GT无梯度；关闭时不增加推理/RNG；EE不统计padding。
- **症状：** dtype冲突、毫米级几何漂移、缺指标、旧GMSE变化或保存/重载丢spec。
- **回滚：** 下次启动 `--no-ee-fk-loss --no-ee-fk-eval`，不热改当前进程；代码基线 `298e90e`，保留产物。
- **验证：** CPU143项+26子项通过；真实3步训练、非unit边界step2周期HF和step3最终HF、step2权重重载后1步训练及独立val全部正常；HF均1708个BF16张量，无DCP；加权EE loss 0.0041–0.0054。CUDA对200组归档SAPIEN参考，位置最大6.42135e-7m、姿态最大9.00505e-7rad；BF16 autocast下loss FP32、预测梯度有限非零、GT detach/mask通过。
- **正式运行边界：** `al_stand_ee_lnfm_9680_add500_20261017_184252` 从原始step9680新增500步已启动；micro/GBS24、低噪FM+target_change_window+EE(1/.1/.1)、每100步val、250/500两份BF16 HF、无DCP，保持上一轮视觉冻结。尚未完成或证明闭环提升，不启用删除监控/自动关机。
- **证据与复现：** `tests/ee_fk/VALIDATION_RUNTIME_2026-10-17.md`、`VALIDATION_GPU_2026-10-17.md`；同功能目录下diagnostics与evidence保存脚本、命令和小型原始结果。

### #18 上机启动记录（2026-10-14）

- 代码：训练机 HEAD 与 `origin/Task_Auto_Learning` 均为 `d1fc26e`；CPU 回归 29 passed / 4.37 s。
- 启动覆盖参数：`RUN=al_v77_sampleidfix SAVE_EVERY=5010 MIN_FREE_GB=80 TB=0 MICRO=24 GAS=1 N_GPU=1 bash experiment/robotwin/start_al_ctrl.sh`。旧 `al_v77` 产物保留。输出盘可用 102 GiB，拉大存档间隔以限制周期 HF 数量（到绝对 step 20000 最多 3 份）；`disk_guard` 仍开启，未删除历史产物。
- 日志：`/data/tmp/al_v77_sampleidfix.log`；输出：`/data/outputs/al_v77_sampleidfix`；输入权重/池子 step 4050。
- 采样不兼容告警已出现；50 任务 bootstrap 完成；`place_object_basket` hardness 扫描 484 帧 / 61 批，77.1 s（取数据 53.7 s / 打分 23.4 s），TrainRequest 已发布。
- 首步 4051 loss 0.0569 / StepTime 66.160 s（含编译预热）；随后 4052–4055 StepTime 为 6.260 / 5.850 / 5.630 / 5.624 s。step 4055 GPU 使用 72471 MiB、利用率 100%，原 sample ID 崩溃未再出现。
- 此记录仅证明 HF 池子导入与首批训练跑通；不把长跑、收尾存档、真实 DCP 恢复视为已验证，因此 #18 仍留在待验证区。

| # | 日期 | 内容 | 验证证据 |
|---|---|---|---|
| 19 | 10-14 | HF池状态保护与超预算训练门控 | `b8f0ab9`；107 CPU回归通过，真机导入原池并仅刷新22个候选；step4327训练正常；详细本意见§1.5 |

| 编号 | 日期 | 改动 | 原因 | 判据 | 影响面 | commit | 状态 |
|---|---|---|---|---|---|---|---|
| 19 | 10-14 | `Scheduler.import_pool_registry` + HF build 接线：同一步号 HF 的已有池记录退出 bootstrap；较新 HF 只重测旧候选，新任务仍预扫；`_select` 拦截超预算候选；纯 CPU 契约回归 | 输入池实际 PASS=16/DEFER=3/CANDIDATE=22/EXHAUSTED=9，但 HF 导入后全量 bootstrap 将其覆盖成 PASS=10/CANDIDATE=40，另出现 attempt=3 | 导入后的既有状态不被 bootstrap 改写；DEFER 只走正常 rollover，EXHAUSTED 不重新训练；候选在 start_attempt 前检查预算；HF 的 TRAINING 记录按剩余预算回候选/终态 | 依赖 #18；HF 只继承 registry，不恢复旧 scheduler 进度/RNG；DCP 恢复入口不改；rescan 免费 PASS 保留；必须重启生效 | `b8f0ab9` | ✅ HF 池继承已验证：CPU 107 passed / 4.97 s；真机 step4300 HF + 原4050池，保护16 PASS/3 DEFER/9 EXHAUSTED、仅重测22候选；训练已到4327。真实 DCP 恢复不作为本条已验证证据 |

## §3 已知遗留 / 暂不改（有明确理由）

| 项 | 为什么暂不改 |
|---|---|
| 事件字段 `*_nmse` 与判定口径 `gmean_mse` 不一致 | 改名会让 v41–v44 历史事件流不可比；已在记账处加**消歧义注释** |
| 参数 `active_val_probe_trajs` 实际决定 confirm 条数 | 改名牵动 `SEMANTIC_KEYS` ⇒ 影响 resume 指纹；已加注释 |
| ~~开环评测本体未按 rank 分片~~ ⇒ **已立项：见 §1 #6** | 用户 10-11 定 B 方案（每卡恒 `k=ceil(n/W)` 条、汇总按 episode id 去重），范围只 `active_val` + `train_monitor`；**默认关**，真机验证后再考虑默认开 |
| `bootstrap PASS` 任务直接毕业 ⇒ confirm 值当不了训练基线 | 门控语义问题，改它等于改设计（"免费 PASS"的意义就是不用训），实测 v43/v44 零交集 |
| 池子采样 ID 兼容性目前只按 local_idx 成员关系判断 | 当前修复会拦住 ID 集不匹配；如果数据内容/排序变化但产生相同 local_idx，单靠 ID 无法证明它仍指向同一 episode/frame。后续可给 pool state 加 resolver 映射指纹；本次先不扩大格式与导出改动面 |
| 收尾路径再次调用HF转换 | step4300已观察到两次收尾HF；#21只收敛周期触发，收尾重排队仍待独立验证，不声明已修复 |
| 评测/存档 `empty_cache()` 的GPU收益 | #22已提供可选复用/观测和CPU对照报告；默认clear保持原行为。CPU契约已验，真实GPU性能、数值与显存峰值仍未验；训练运行中禁止GPU A/B，不改变默认行为 |
| 模型启动进一步并行化 | 当前运行日志已证实6线程预读6片23.8GiB/8.3s，逐片加载12.4s；单训练进程。FSDP2自定义逐张量装载不会自动因HF并行加载环境变量而加速。可后续测流水预取与逐片清理开销；未测收益前不新增多进程模型构建或调整线程参数 |

---

**当前待验证见 §1（本次新增 #21/#22；存在两条编号 `9` —— 历史并行协作者各写一条，保留原编号）。下次上机（无训练在跑时）优先按编号顺序验证；**
**排查入口：先看 §1.5（扰动本意/症状/回滚）与 `AL_*` 开关总表；**
**#6（轨迹分片）需真机 7/8 卡 smoke：确认不死锁、指标与不分片逐位一致、墙钟下降。**

### #19 停训存档与重启计划（2026-10-14）

- `al_v77_sampleidfix` 使用 STOP_AND_SAVE，在 step4300 learning unit 边界停止。完整 DCP 写成功；HF 两次导出成功（185.02 s / 171.64 s，收尾又排队同一份 HF 的既有重复行为，本次不改），退出后核对 model/optimizer metadata 与 extra_state 文件；HF index 六片全部存在且文件长度与 safetensors header 一致，HF 23.75 GiB。
- 当前 4300 池为 PASS=13 / CANDIDATE=36 / EXHAUSTED=1 / DEFER=0，是受错误 bootstrap 影响的运行结果，不替代原池。原 step4050 池 SHA256 `b8ba93e9deb7538ad55251f704ca0ef42b1e64af6adc3a02ddeb2f94e0f87251`，停训后复核未变。
- 用户选择 step4300 HF + 原 step4050 池（重新继承池，不是精确断点恢复），SAVE_EVERY 明确恢复220，允许启动空间门槛 MIN_FREE_GB 从80改50；其他训练参数保持 MICRO=24 / GAS=1 / N_GPU=1 / MAX_STEPS=20000 / TB=0 / DCP_MODE=final_only / SAVE_HF=1 / HF_PASS_INTERVAL=0。此后参数调整必须先获用户明确允许。
- 原 ctrl wrapper 固定要求三片 HF 且权重和池步号相同；本次六片 HF + 显式旧池通过独立预检后，调用同一个 `al_50task_bf16.sh`，不修改 wrapper。新运行 `al_v78_poolfix_4300`；重测原池22个候选，保护16 PASS / 3 DEFER / 9 EXHAUSTED；后续正常 rescan/review/rollover 可按既有规则更新状态。

- #19 上机结果：训练机 HEAD=`b8f0ab9744c471468afb124abd358622376675f5`，与origin一致。实际进程确认 `step_offset=4300 / save_steps=220 / max_steps=20000 / micro=24 / gas=1 / dcp_save_mode=final_only / disk_guard=true`。
- 用户明确允许删除刚保存的4300 DCP，保留HF：仅删除 `global_step_4300/model`、`optimizer`、`extra_state`，释放23.72 GiB，空闲77.5 GiB；HF六片长度/header/mtime及config/index/池文件SHA复核未变。

### #20 监控运行方式

- 对新运行 `al_v78_poolfix_4300` 启动独立后台脚本；日志 `/data/tmp/al_v78_checkpoint_watch.log`。启动用的4300 HF在旧run，且通过 `--protect` 再显式保护。
- 命令：`python tools/auto_learning_pipeline/al_checkpoint_watch.py /data/outputs/al_v78_poolfix_4300 --apply --protect /data/outputs/al_v77_sampleidfix/checkpoints/global_step_4300/hf_ckpt`（生产按项目约定 `setsid nohup` 后台）。
- 仅周期HF参与“满4删1”；完整/正在写DCP仍由训练器自身保留策略负责。当前CPU12项覆盖阈值、预览、重启追赶、多种未完成存档、输入保护、符号链接及目录发布竞态。真实运行未积满4份之前，不宣称生产删除路径已验证。

2026-10-19 #49运行追记：旧750步run日志终止于427步，未产生任何检查点；用户更换机器后授权#50重训，不宣称恢复了427步更新。旧日志保留。

### #53 补充：FP32 HF与正式micro28（2026-10-19）
- 用户授权HF改FP32，micro28/GAS1，仍500步、250/500存档；源mouse500与纯50条不变。
- 启动器从训练快照继承HF导出dtype，正式入口明确FP32；容量预算按源BF16权重两倍计算。
- 不变量：存档节奏与保留策略不变。出错症状：worker重建dtype丢失或磁盘预算不足。回滚：显式`--hf-export-dtype bf16`。验证：BF16/FP32快照到worker重建契约测试、运行时精度审计与存档报告。

### #54 FP32来源续训与位置6/夹爪0实验（2026-10-19）

- 状态：🧪 CPU整链通过，待正式运行。用户授权从最新纯主FP32 500步继续500步，FK夹爪权重1.5→0、位置10→6；姿态0.8、其余配置不变，250/500各存FP32 HF，两份完整后关机，不插入闭环评测。
- 修改：`launch_skill_mix_task.checkpoint_inventory/build_plan`允许显式FP32主参数来源，`continue_skill_training.make_job/verify_final/run_job`按导出dtype校验并确认全部里程碑再关机；新配方继承原run配置，原始动作夹爪监督保留。
- 本意与不变量：只改变新增FK位置与夹爪两项，不变更主动作损失、采样、学习率、batch或存档节奏；CPU逐项比较worker重建配置。
- 出错症状：FP32来源被误拒、重建参数丢失、250缺失仍关机。快速回滚：不运行新入口，或显式恢复夹爪权重1.5。验证：`test_gripper_off.py`、`test_continue_skill_training.py`及运行时CLI/precision_audit。依赖：#53混精度及Flex修复；此次不改前向/反向数值实现。
- 空间：用户授权删除旧混合run的250/500及上一轮纯主FP32的250，已删除；当前FP32 500和混合前mouse500保留，空闲69.22GiB。说明与命令见[实验文档](../training/experiments/mouse/mouse_gripper_off_continuation_zh.md)。

- #54 验证：服务器Python3.12 CPU回归97 passed、36 subtests；真实来源预检仅位置10→6、夹爪1.5→0及运行目录变化，其余训练配置、50条轨迹、归一化一致。CPU契约暴露快照未同步EE覆盖值，已同步全部显式train选项到快照，实际CLI优先级不变。本地py_compile通过，Python3.12无pytest，未冒充本地完整测试。

### #59 实施过程追记（历史状态，以§2验收为准）

- 用户授权：从9680按官方方法训练，BF16、micro=GBS56/GAS1、200步；HF亦BF16导出，评测加载FP32。状态：入口语法检查通过，服务器后台执行前置检查，未宣称GPU通过。
- 改动：`experiment/official_control/bf16_trial.py`独立准备/CPU/三步探针；`prepare_adjust_bottle.cpu_check`增加带默认值的期望精度/步数参数，原FP32调用默认不变。
- 不变量：50条adjust_bottle、原始FM、EE与窗口/低噪FM关闭；save_steps=250原值不变、最终200步保存。原250/500存档不删；旧配置和模型核心不变。
- 出错症状：`official_adjust_bottle_bf16_200_status.json`为failed，或探针dtype/有限梯度断言失败；此时不继续正式训练。
- 回滚：不运行独立BF16入口；旧官方配置无需修改。禁止以降低dtype门槛或隐式减小batch掩盖探针失败。
- 验证入口：新run的cpu_report.json、probe/precision_audit.json、probe/summary.json、verified_checkpoint.json及evaluation_200/summary.json。控制器先等前一轮250评测完成与GPU空闲，再CPU整链→GPU3步→训练200→验证BF16 HF→同种子3回合闭环；不自动关机。

- #59 修复追记：首次三步探针0/3在Inductor addmm报Float/BFloat16冲突，未开始正式训练。`training_precision_context`原先对BF16参数模式直接nullcontext，导致FP32状态/动作输入与BF16线性层权重冲突；改为单卡BF16计算均使用autocast，FP32计算与FSDP路径不变，模型核心快照不改。CPU契约`experiment.official_control.test_bf16_context`真实BF16 Linear+FP32输入复现原错误，并验证修复后输出/参数/梯度BF16、梯度有限；服务器2 tests通过（8组合subtests），本地无torch跳过实际算子测试。旧探针移为probe_failed_initial保留，重试日志probe_retry.log。症状/回滚：若新路径异常，可还原precision_audit.py该条件，但原BF16配置将再次失败；禁止静默继续正式训练。
- #59 GPU追记：autocast修复后首步前后向和更新通过，旧通用hook在编译后的action_out记录空dtype，探针按预期拒绝继续。新增仅审计用`observe_action_output`，用compiler.disable隔离观测副作用，真实输出/梯度CPU等价测试通过（服务器3 tests）。再次三步GBS56探针成功：6375906359个可训练参数及梯度均BF16，visual/language/action expert/action_out四处输出BF16；第2/3步9.530/8.920秒。正式200步由同一控制器继续，不宣称已完成或已闭环学会。

- #61 CPU验收：move_can_pot7568帧、place_container_plate7934帧、place_empty_cup8617帧；三者概率质量均严格1/3，实际StatefulDataLoader batch[56,50,55]通过。旧存档四份清理完成，保留两模型；新控制器在独立run后台执行，GPU/训练/闭环结果以状态文件逐阶段为准。

- #61 GPU启动验收：三步56/GAS1完成，allocated77.6979GiB、reserved84.5898GiB，实际参数/梯度/四处前向BF16且梯度有限。控制器state=training、steps400、hf_steps[200,400]，6007已发现本轮run；最终HF与24回合闭环尚未完成，不作为已学会证据。

- #61/#62终态：三任务正式400步完成、200/400 BF16 HF均验证；原8任务评测按用户要求中止，选测adjust_bottle2/3、click_bell3/3，两任务server_pid58398一致，model_loads=1、OIDN错误0。用户随后授权删除三任务400（日志与200存档保留），重新从原adjust_bottle BF16-200单训place_container_plate200步；新实验另起，不继承已删400。

- #63终态：place_container_plate正式200步与1708张量BF16 HF验证完成，3回合闭环1/3、OIDN错误0。后续#65来源即此模型。#64真实CPU首次因检查器硬编码GAS1失败，按expected_gas修复后GAS2实际loader通过；旧失败日志保留，不能把旧failed状态当成未修复。

- #65启动实证：完整三步BF16参数/梯度/关键前向探针通过；框架真实150回合24119帧、loader虚拟epoch500步，正式已更新7步、6007已发现本轮run。继续按用户后续要求使用旧HF重载FP32评测，本轮不启用内存模型交接，不保存FP32 HF。

- 提交前收尾：清理重复/过期文档阶段口径，sequential拒绝非1主任务weight，避免用户以为顺序模式仍按weight分配；当前运行不受此启动参数校验补充影响。#65只读核验39/500正常，未热同步、未改存档或评测精度。
- 本次提交验证：本地21项纯CPU回归全部通过、py_compile/compileall与git diff --check通过；此前服务器已完成GAS2实际数据装配，以及顺序150回合24119帧/500步虚拟epochCPU检查与三步GPU探针。主复习组合仍仅有配额/采样契约证据，运行前仍强制CPU/GPU前置检查；不宣称完整闭环或内存模型交接已实现。
