# 闭环时间集成：仅评测开关

当前只按用户要求做无卡CPU测试，没有加载模型、GPU推理或仿真，没有新成功率。服务器52项official_control测试全通过、无跳过，含真实NumPy客户端；本地缺torch/numpy的全目录尝试不能替代服务器结果。证据：refs/evaluation_evidence/temporal_ensemble_cpu_20261024/cpu_report.json。

## 实际行为

原部署文件虽然保留adaptive_ensemble_alpha/action_ensemble_horizon字段，但现有infer没有使用它们合成重叠预测。chunk_ret=False也只是逐个返回缓存chunk，不是时间集成。本次保留服务器chunk_ret=True、每次返回50步，在仿真客户端显式启用重叠平均，每个动作执行完立即取新观测、重新预测，再只执行一个平均动作。

参数通过评测job的temporal_ensemble控制，示例：`"temporal_ensemble": {"horizon": 8, "decay": 0.1}`。horizon是保留最近几次预测，不是原始chunk长度。当前动作t使用本次预测的第0步、上次预测的第1步、上上次的第2步，以此类推最多8次；每项权重exp(-decay*age)，所以较新预测权重大。按权重和归一化。全部动作维度（含夹爪）一起平均，不额外夹紧、二值化或改动作坐标。

历史按客户端实例独立，每回合reset清空；不把两个任务、两个回合的动作混合。全零预测是有效动作，不能过滤。无效/ragged/NaN/维度变化明确报错。客户端输出单个动作向量，原仿真循环每轮执行一次take_action。服务器仍单模型FP32、FM10次去噪与chunk50，模型权重、归一化、物理步数与录制设置不改。

没有temporal_ensemble字段则保持原chunk执行，并清除可能继承的时间集成环境变量；默认不启用。回滚只需新评测job移除该字段，不需修改模型或训练。官方RoboTwin源文件不修改，运行时安装客户端包装器，原有文件哈希检查仍保留。

## CPU验证与审计

已测时间索引对齐、较新指数权重、合法零动作、8预测缓存上限、短chunk、每回合reset、不同客户端隔离、单动作输出、每次请求新chunk、元数据透传、非法shape与数值、重复安装失败。真实execute的伪仿真契约验证job参数进入sim环境、audit进入summary，以及不开启时环境残留不会改变默认行为。

每个任务输出temporal_ensemble_audit.json：source_chunk_length、executed_actions_per_response、replan_every_actions、horizon、decay、当前回合requests与active_predictions。成功汇总要求审计存在、原预测50步、每次只执行1步、参数匹配、请求数>0；原summary.action_chunk=50表示模型预测长度，实际执行长度见时间集成审计，不能误认为仍开环执行50步。审计requests为最后一次reset后的当前回合计数，不是三回合总数。

## 待GPU评测

配置：configs/evaluation/temporal_ensemble_latest1122_pill_dual.json。来源最新1122提锅step200；move_pillbottle_pad与pick_dual_bottles各3回合、全部录视频、两个任务共享一份模型，规划并行2路；每任务内部回合顺序执行，历史reset隔离。expected_seeds药瓶100001/100002/100005，双瓶100000/100001/100002；这些是既有任务种子，实际评测仍由专家校验与汇总严格确认。未在无卡模式重新运行专家仿真。

未配置普通chunk同模型对照，compare_reference=false仅禁止旧串并行一致性断言，不代表对照成绩。CPU测试不证明时间集成能提高成功率，GPU运行速度/显存/闭环均待测。每步重新预测将显著增加模型调用量，原每50动作一次请求变为每动作一次，不按旧chunk速度估算完成时间；现有单仿真3600秒超时门槛保持不变，若不够需先评估再调整。

目前不启动任何GPU进程；开卡后再按用户授权执行。待用命令（无--execute只展示计划）：

```bash
cd /data/experiments/stack_bowls_sample_weighting/lingbot-vla-v2
/data/miniconda3/envs/lingbotvla/bin/python experiment/official_control/parallel_evaluation.py --job configs/evaluation/temporal_ensemble_latest1122_pill_dual.json
```

GPU正式执行前仍须核验GPU空闲、有效import/文件快照、来源模型、渲染依赖及当前评测目标。此处没有加入自动启动队列。
