# 统一闭环评测：单任务与全任务benchmark

入口：`experiment/official_control/model_benchmark.py`。两种模式共用最近的`episode_parallel.split_jobs`、`parallel_evaluation.run_jobs`、`session_eval_server`；每项内部回合并行、任务配置逐项执行，全程一份FP32模型。不会改变旧配方的默认行为，不启用时间集成，除非显式提供temporal_ensemble。

默认：每任务配置3回合、同时3回合、不录像。`--episodes`控制总回合数，`--episode-workers`控制并行上限（1至8，实际不超过总回合数）。例如总7回合、并行3路就是空位滚动补齐，不是同时开启7路。

## 使用

复制`configs/evaluation/model_benchmark.example.json`到自己的job文件，设置真实模型、输出、RoboTwin、Qwen及仿真Python路径。训练/推理Python调用下面命令。示例任务列表来自本仓库50任务split清单，部署前应与实际RoboTwin任务清单核对，不直接采用外部ROCm环境路径。

```bash
# 同一个入口：单任务，默认clean、3回合并行、无视频
python experiment/official_control/model_benchmark.py --job job.json --mode single --task stack_bowls_two --task-config demo_clean --execute

# 完整模型：50任务×clean/randomized×3回合，共300回合
python experiment/official_control/model_benchmark.py --job job.json --mode benchmark --execute

# 调节总回合数与并行回合数
python experiment/official_control/model_benchmark.py --job job.json --mode single --task stack_bowls_two --episodes 6 --episode-workers 3 --task-config demo_clean --execute

# 相同命令追加--resume：核对来源与参数后只补未完成项
# 去掉--execute仅打印计划，不加载模型或启动仿真
```

job中的video_episodes默认0，可显式设置前N回合录像。single可选择demo_clean、demo_randomized或both；使用示例job时显式指定--task-config demo_clean避免继承both。CLI参数被冻结后才交给后台worker。

## 种子与专家校验

默认`seed_selection=inline`：每个回合在自己的实际仿真进程中完成专家校验。通过后直接进入原客户端策略评测，不再由另一个进程预筛后重复校验。专家判定不成功或UnStableError则只为该回合换候选，不由此整项作废；其它异常直接报错。

四路候选序列互斥：第i回合从seed_search_start+i开始，每次增加4（一般为总回合数），因此不会两个回合重复测同一场景。每回合默认最多200候选，原仿真进程3600秒超时仍有效。候选耗尽或渲染/资产/代码异常才记error，不改模型失败统计。seed_search_max_candidates控制每回合搜索上限。代码日志INLINE_EXPERT_CANDIDATE可追踪实际搜索过程。

每个完成项目的实际seeds写入benchmark_results.json、底层逐回合summary和selected_scene_seeds.json。后者按demo_clean/任务名及demo_randomized/任务名索引，可直接作为其它模型的scene_seeds配置；续跑跳过完整项目并核验真实种子。明确提供scene_seeds的项目保持固定种子严格模式，不自动替换，以免悄悄改变对照。seed_selection=fixed保留原固定候选路径；expert为旧版跨进程预筛兼容路径，不建议用于正式测试，已证实预筛通过不保证另一个进程再次校验通过。

不以策略测试成功与否决定是否换种子，只有专家无效场景才能替换，避免筛掉真实模型失败。current_item.json显示当前项目和运行状态；跨进程预筛版本的valid_seeds缓存不作为inline正式评测的有效依据。旧recipes入口默认严格固定种子行为不变。

## 结果与续测

最终与运行中汇总：`<destination>/benchmark_results.json`（schema_version=2）。参考脚本没有定义JSON schema；本导出以其`read_result`/`collect_results`统计内容为依据，不能声称与不存在的上游JSON schema逐字段兼容。

- `overall`：successes、episode_count、success_rate_percent（0至100），与参考脚本整体打印的成功数/总回合数/百分比一致，不取任务百分比的简单平均。
- `task_configs`：demo_clean、demo_randomized各自的相同统计；只包含本轮选中的配置。
- `tasks`：按计划顺序列出task_config、task、status、successes、episode_count、success_rate_percent、elapsed_seconds、result_path和error。
- 顶层status、rate_scope说明是否全量完成；缺失/异常项目的成功数和百分比为null，不当作0分。overall额外包含预期、完成、异常、待测项目数。未完成时只汇总已完成任务，明确标为completed_tasks_only。
- 顶层elapsed_seconds对应参考脚本benchmark invocation（本次调用墙钟秒数），含本次模型加载，不代表跨resume累计时间。

额外的来源指纹、逐回合审计与续测内部状态放在`benchmark_audit.json`，与公开统计分离。旧版内部格式的benchmark_results.json仍可作为首次迁移续测输入，之后写入新的公开结果文件与独立审计文件。

每项产物位于`<destination>/<task_config>/<task>/attempt_NNN/`，底层保留逐回合日志与summary。任务成功/失败是模型结果；专家拒绝、渲染或推理错误是评测异常，不混进模型失败率。单项异常后继续下一项；共享服务死亡则终止并写failed。

续测校验冻结参数、checkpoint路径、文件大小/mtime和JSON配置内容哈希（不对大权重做全量内容哈希），再核对已完成summary。不要原地替换checkpoint；若替换模型应使用新destination。成功项跳过，异常项使用新attempt重试；不能仅凭done标记认定通过。共享模型服务只在本轮开始加载一次，结束释放，不自动关机。

## 验证边界

本地纯CPU契约测试覆盖默认100项300回合、single模式、可控并行、种子/目录隔离、单模型复用、异常统计、续测拒绝配置漂移、旧版回归。未做全量GPU benchmark，randomized环境资产/专家种子可用性仍待服务器验证。既有串并行成绩差异尚未在本次消除，结果携带并行参数，应以相同推理、种子、环境与并行协议比较模型，不能宣称与串行等价。

验证：`python -m unittest experiment.official_control.test_expert_seed_selection experiment.official_control.test_model_benchmark experiment.official_control.test_episode_parallel experiment.official_control.test_recording experiment.official_control.test_parallel_evaluation`。

参考：[Reproduce_Guide](https://github.com/ZiguanWang/Robotwin-radeon-cloud/blob/main/Reproduce_Guide.md)。参考脚本采用多GPU一GPU一模型；此实现沿用当前单卡共享模型多回合路径，单独提供汇总JSON。

## 2026-10-24 异常补测与recovery160对照

1050原轮最终91项完成、43/364成功，9项专家场景异常；已部署精准重试补丁，resume只补9项。补齐后自动使用selected_scene_seeds.json的全部100项最终种子，对20261022_1209_round_3_recovery/checkpoints/global_step_160/hf_ckpt做同400回合、4路无视频固定种子对照；模型正常0/4不会重测，只补评测异常。固定场景复验异常最多重试3轮，不能通过更换种子掩盖跨模型差异。

串联状态：`/data/eval_results/20261024_2258_1050_repair_then_recovery160_pipeline/status.json`。1050结果保持原1923目录，recovery160输出目录：`/data/eval_results/20261024_2258_recovery3_s160_benchmark_clean50_random50_parallel4_matched1050_novideo`。以上为已启动补测与排队状态，尚未产生完整对照结论。

用户在本轮明确授权“需要测的都跑完，有问题解决，跑完服务器关机”。仅本次串联任务已设置completion_shutdown守护：严格校验两个模型各100项/400回合、相同种子、逐回合与渲染证据后打包归档，sync落盘后执行关机。链路失败或核验不通过不会按完成处理；状态在同一chain目录completion_shutdown_status.json。完成时尝试自动下载归档并生成真实对照记录；当前仍运行中，不把关机计划记成已关机。脚本快照见refs/evaluation_evidence/benchmark1050_recovery160_completion_20261024，不含SSH连接信息。

**本轮关机方式已按用户最新要求覆盖：**取消核验与本地归档等待守护，仅监控本次测评串联主进程；进程结束即直接执行`/usr/bin/shutdown`。结果是否通过核验不再作为关机条件，原核验守护及本地归档等待程序已停止，测评本身继续运行。此授权仅适用于本轮，不作为未来任务的默认关机规则。

2026-10-25 08:05恢复中断的recovery160：原62项完整保留，续跑10异常＋28待测项；1050已48/400完整，不重测。固定配对种子和四路并行不变，最多3轮只重试异常项。新日志位于原chain目录20261025_0805_resume.log；已将直接关机监控绑定到本次新进程，不等待额外核验。此处仅记录已恢复运行，未记为完整完成。1050本地模型归档已完成，35文件/1708张量及SHA256校验通过。

2026-10-25 08:19用户明确取消1050对照约束，只补recovery160。已停止固定种子旧进程及其关机监控，采用新目录`/data/eval_results/20261025_0819_recovery3_s160_benchmark_independent_parallel4_novideo`；原62项完整结果引用保留，新目录manifest绑定独立inline搜索配置，剩余38项自动找有效候选，不再受1050 seed表约束。不重跑1050。仍为每项四回合并行、不录像，程序结束直接关机，无额外核验门槛。此后的完整成功率为独立benchmark，不能再标为与1050逐种子配对对照。
