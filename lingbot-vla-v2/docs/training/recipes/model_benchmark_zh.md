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

`scene_seeds`按`demo_clean/任务名`、`demo_randomized/任务名`分别配置，例如`{"demo_clean/stack_bowls_two":[100000,100002,100003]}`。未提供时采用从100000开始的候选种子。候选不代表专家已验证有效；沿用原多回合并行的严格适配，专家拒绝候选时该项记录error，不自动换种子、不当作0/3。应先按部署环境准备可用种子清单，再正式比较模型。第一次全量探测可能产生需补齐种子的项目，报告会保持incomplete。此入口尚未新增专家预扫功能。

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

验证：`python -m unittest experiment.official_control.test_model_benchmark experiment.official_control.test_episode_parallel experiment.official_control.test_recording experiment.official_control.test_parallel_evaluation`。

参考：[Reproduce_Guide](https://github.com/ZiguanWang/Robotwin-radeon-cloud/blob/main/Reproduce_Guide.md)。参考脚本采用多GPU一GPU一模型；此实现沿用当前单卡共享模型多回合路径，单独提供汇总JSON。
