# 多仿真共享模型的闭环评测

新增入口`experiment/official_control/parallel_evaluation.py`，当前是独立试跑，不改既有训练后的串行评测默认值。
一份FP32模型常驻，三个仿真进程并行；推理保持B1排队执行，不启用动态batch。
这允许物理/渲染与其他客户端推理交错，不保证吞吐能线性增长。

## 隔离与回退

每个WebSocket连接保存自己的step、动作缓存、robot_config、action_key、feature_transform，
共享权重对象不复制；reset不能重置另一连接，客户端不得请求更换服务权重。
原推理函数同步执行，状态切换和前向受同一锁保护；noise seed0/B1逻辑保持原行为。
每个仿真独立进程、输出目录、种子、视频和日志，保持原OIDN与精度检查。

先3路。只有日志明确出现GPU/渲染/主机内存资源不足，才结束该attempt后退2路，失败日志不删。
普通代码异常、种子不一致或成功次数不一致不通过减并发隐藏，报告失败并保留证据。
`concurrency_levels`控制尝试顺序，默认`[3,2]`，仅资源不足时进入下一项；支持1至6路。
`compare_reference`默认为true，核对完整参考报告；只跑1回合的资源探测显式设false，
报告会记录`reference_compared=false`，不能声称成功率或种子与三回合参考一致。

## 本次已授权试跑

- 模型：R11最终`global_step_1200/hf_ckpt`。
- 任务：rotate_qrcode、move_playingcard_away、pick_dual_bottles，各3回合，共9回合。
- 串行参考：R11原`evaluation_1200/<task>/summary.json`，同模型及实际seed。
- 配置：`configs/evaluation_jobs/parallel3_R11_probe.json`。
- 输出：`/data/outputs/20261021_1033_parallel3_R11_fp32_B1_probe`。
- 先等待原七任务自动评测完成，防止与现有服务重复占GPU；最多等待2小时，原流程失败则本试跑也停止。
- 记录GPU采样gpu_samples.jsonl、单次服务session_audit.json及逐任务原始结果，最后benchmark_summary核对成功次数/seed/模型路径。

已通过49项本地标准库CPU回归，含真实三线程重叠、只开一次服务、连接reset/缓存/变换隔离、异常finally恢复与特定OOM识别。
服务器8项对应CPU契约通过。2026-10-21三路GPU试跑已完成，无需退到两路。

| 任务 | 三路并行 | 历史串行 |
| --- | --- | --- |
| rotate_qrcode | 2/3 | 2/3 |
| move_playingcard_away | 3/3 | 3/3 |
| pick_dual_bottles | 0/3 | 0/3 |

三个任务的实际种子、成功次数和模型路径均匹配；OIDN错误为0。
共享服务PID17486，model_loads=1，3个连接、63次请求，B1与会话隔离审计通过。
显存采样峰值44797MiB（43.75GiB）；仿真阶段436.75秒，含服务加载总耗时496.84秒。
历史串行三个任务合计595.09秒，仿真阶段耗时约少26.6%，吞吐约为1.36倍；
CPU线程配置不同，此数值仅作历史参考，不能当作严格同配置加速比。
证据：`refs/training_records/20261021_parallel3_completed.json`，另有运行中三仿真进程快照。

## 命令

在实验包根目录，先核对GPU及其他任务；执行需要当次授权。

```bash
# 默认只打印配置，不运行
python experiment/official_control/parallel_evaluation.py --job configs/evaluation_jobs/parallel3_R11_probe.json
# 已授权后，自行后台运行；wait_for_pipeline存在时会等待原流程完整结束
python experiment/official_control/parallel_evaluation.py --job configs/evaluation_jobs/parallel3_R11_probe.json --execute
```

输出目录必须全新。外层状态在`<destination>_pipeline/status.json`，原始日志在同级pipeline.log。
attempt目录为parallel_3/或parallel_2/，单个attempt内model_loads=1；回退会关闭旧服务后重新加载一次，不能当作一整个试验只加载一次。
3回合结果相同只是有限范围的回归证据，不等于所有任务或动作逐位一致。
历史串行环境CPU线程参数可能不同，直接比较历史耗时不是严格同配置速度A/B；需要严格速度结论时另做相同线程配置的串行对照。

验证入口：`python -m unittest experiment.official_control.test_parallel_evaluation experiment.official_control.test_shared_evaluation`。

## 六路单回合资源探测

用户后续授权直接六路，每任务仅1回合，配置`configs/evaluation_jobs/parallel6_R11_probe.json`。
使用R11前六个任务与同一FP32/B1模型，`concurrency_levels=[6,4]`，不改原串行入口或存档。
本地及服务器各9项CPU契约通过，新增六线程真实重叠测试。
输出：`/data/outputs/20261021_110325_parallel6_R11_fp32_B1_probe`。
已观测到6个独立仿真进程同时运行；六任务各1回合均输出完整结果及非空视频，
无仿真Traceback/OIDN错误。显存采样峰值64934MiB（63.41GiB），
服务审计model_loads=1、connections=6、requests=48，未触发资源回退。
本轮**流程状态failed**：配置compare_reference=false只关闭外层对照，
子任务evaluate.execute仍读取reference_summary并用1个seed对比历史3个seed，
触发`scene seeds differ between paired models`。不能把该轮标为完整流程验证通过。
原始单回合结果：qrcode/card为1/1，其余四任务0/1；资源可运行已证实，
完整报告与严格吞吐仍未验收，不追加GPU重跑。
证据：refs/training_records/20261021_parallel6_completed_with_report_error.json与20261021_parallel6_raw_results.json。
任务集合与回合数不同于三路探测，不能直接用总耗时计算加速比。

## 训练配方中的分批评测

`evaluation.max_parallel_tasks`默认1，保持旧串行入口；设为6并启用`shared_model_session=true`，
使用隔离会话服务，一次只启动最多6个任务，全部结束后才进入下一批。
10任务明确分为6＋4（最后一批只创建4个仿真，不补空任务），两批共用同一服务和模型。
每个任务的`episodes_per_task`独立生效，本轮R12为3，总30回合；不改变原评测精度FP32/B1。
状态报告保留batch/active_workers及已完成结果，非资源异常停止留证，不自动改训练或保存点。

单回合探测的配置问题已修复：`compare_reference=false`会在交给evaluate.execute前去掉reference_summary，
并显式require_baseline=false；独立expected_seeds约束保留。CPU测试覆盖参数确实传到子任务，不伪造旧失败为成功。
新增十任务6＋4屏障测试（首批六个全部结束才允许后四个开始）、单模型只加载一次及六路配置边界测试。
本地及服务器对应49项CPU契约通过；R12训练已进入，最终三回合分批GPU闭环仍待实际完成。
