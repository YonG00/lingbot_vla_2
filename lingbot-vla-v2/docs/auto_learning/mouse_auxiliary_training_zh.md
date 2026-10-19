# 鼠标主任务＋辅助示范混合续训

本入口继续继承 `al_mouse_pad_9680_ee_lnfm_add500_20261018_150402` 的step500模型权重；默认新增500步、开启梯度检查点，新优化器/LR/步号。目标是学习鼠标抓取和放置，辅助不限同型号、同目的地，始终保留原始任务指令和完整轨迹。

## 配方与入口

| 设置 | 默认值 |
|---|---|
| 主任务 | place_mouse_pad原train+val，共50条/7466帧 |
| 辅助 | 审核清单47条/7762帧，真实来源5个任务 |
| 总白名单 | 97条/15228帧，只有审核中的辅助episode被纳入 |
| 主辅配额 | 70:30；默认micro48/GAS1，跨步累计70:30（单步整数配额轮转） |
| 辅助池内 | 先等概率选episode，再在episode内按阶段权重抽锚点，批内同episode不重复帧 |
| 阶段权重 | 目标鼠标臂的控制候选窗口3倍，向前补闭爪前1秒；其他帧1倍 |
| 梯度检查点 | 默认开启；batch48/GAS1，用户指定适配84GB新卡 |
| 其余 | EE位置12/姿态0.8/夹爪1.5、低噪FM、全量ViT、主任务原窗口2倍策略保持 |
| 评测/存档 | 固定4条主任务train_monitor，step0及每100步；250/500步各一份BF16 HF，无DCP（用户明确指定） |

默认仅CPU预览，不创建训练run、模型前后向或启动GPU任务：

```bash
cd /data/experiments/learn_skill_mix_task/lingbot-vla-v2
bash experiment/robotwin/recipes/20261019_place_mouse_pad_mouse_aux.sh
```

开卡、GPU空闲及容量检查通过后，显式启动；默认在最终HF校验成功后自动关机：

```bash
bash experiment/robotwin/recipes/20261019_place_mouse_pad_mouse_aux.sh --execute
```

可显式选择 `--no-shutdown` 保持机器开机；训练失败或最终HF校验失败不关机。输出名称含 `all50_aux0.7_p12_from_mouse500`，与旧all50实验隔离。每次新run仍使用独立窗口缓存；旧run/缓存/原始数据不覆盖。

可覆盖 `--auxiliary-manifest`、`--auxiliary-primary-ratio`、`--auxiliary-grasp-weight`、`--auxiliary-approach-seconds`，以及已有 `--steps/--micro/--gas/--model/--train-config`。辅助配置必须配合train_all_episodes、单个主任务、关闭Replay与PASS/复习等使用；非法组合在CPU准备阶段拒绝。主辅比例不能改用 `--new-ratio`，它仍是既有NEW/Replay接口。

## 数据与状态边界

- 实际训练引用 [训练版本清单](../../refs/data_manifests/mouse/20261019_train_v1/approved.json)，来源为 [视觉审核版本](../../refs/data_manifests/mouse/20261019_review_v2/approved.json)。旧审核产物不覆盖，新版本显式标记供本训练接口使用；不把原审计的training_ready=false悄然解释成可训练。
- 训练清单的training_ready=true表示可供显式启用的接口使用、CPU契约通过；不表示GPU/闭环效果已验证。视觉审核不是原始check_success、力觉或接触标签认证，精确阶段边界仍未认证。
- `source_identity.json`与清单的指纹必须一致，启动时核对数据元信息内容SHA以及数据/视频的大小、mtime_ns，并核对原始任务manifest。大视频不是全量内容SHA；保留相同大小/mtime的篡改不在检测范围。
- 联合数据集只构造一套local_idx。原任务名、episode和绝对帧坐标用于解析与审核，不复用旧采样ID。预览核对实际帧数、连续帧、时间戳和task_index/原指令。
- 调度器仍只管理主任务；辅助单独存放于BatchComposition.auxiliary，不进入NEW或Replay，不产生辅助任务PASS/状态迁移。
- 主任务采样分布保持原窗口策略；辅助阶段仅使用审核中已绑定鼠标臂的控制候选。柜子8条的右臂拉抽屉候选被排除；不把所有闭爪动作都当抓鼠标。
- 本版不额外提高辅助放置权重，因为没有精确人工放置边界。搬运/放置仍以基础权重完整参与监督。加权的是chunk锚点，不是逐步loss mask，chunk可跨阶段，但不能跨episode。
- 真正准备新run时复制审核清单和来源到 `<run>/auxiliary/`，实际AL配置指向该快照，防止后续修改refs改变已启动run。

## 比例、统计与兼容

在全局optimizer batch上按累积整数配额分配主辅；例如20样本为14/6，非整数比例跨步补齐。每rank再分配相应本地槽位，micro/GAS不会额外独立取整。虚拟多rank与GAS契约已测试，实际多GPU训练未验证；当前专用启动入口仍为单GPU。

采样RNG按seed/rank/logical_step确定，预取丢弃重建不改变被消费的batch。真实sampler的stats_upto只统计已消费步，hook汇总主/Replay/辅助三类，分布式分支对辅助计数同样汇总。TensorBoard及事件记录：

```text
sampling/primary_samples_per_unit
sampling/auxiliary_samples_per_unit
sampling/auxiliary_task/<原任务>
sampling/auxiliary_episode/<episode>
sampling/auxiliary_phase/grasp_candidate
sampling/auxiliary_phase/other
```

面板显示主辅配额，以及有TB数据时最近unit的实际消费数。train_monitor仍只用于主任务训练拟合，无独立val/PASS/gap或泛化结论。

关闭方式：使用旧 `20261018_place_mouse_pad_all50_9680_ee_lnfm.sh`，不传辅助清单。旧40/10和旧all50默认关闭辅助，不消费辅助RNG、不增加旧配置指纹键；不改变已有行为。HF-only新run不恢复DCP；sampler/配置指纹仍校验辅助清单内容与比例/权重，防止其他调用方发生策略不一致的断点恢复。

## CPU验收

合成Parquet整链覆盖联合白名单、真实worker重建、独立来源、episode均衡、阶段权重1/3、虚拟DP/GAS与小batch累计配额、500步调度、真实hook消费/预取、身份变更拒绝与禁用兼容。

```bash
PYTHONPATH=.:tests python -m pytest -q tests/mouse_pad_training/test_auxiliary.py
python tools/acceptance_checks/mouse_auxiliary_cpu_preflight.py \
  --output-dir /data/outputs/object_audits/<新CPU验收批次> \
  --check-dataset-chunks
```

CPU验收工具会生成独立主任务窗口缓存并实际抽样，不生成训练run或加载模型。可选chunk检查使用真实LeRobot读取器，仅CPU。`--check-runtime-assembly` 在另一进程使用实际Parquet列与元信息验证训练装配；两项不能在同一进程启用，低内存实例可设 `MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。

2026-10-19实机回归：264 passed、86 subtests passed（包括实际训练装配单测）；本地新混训契约11 passed、1 skipped、10 subtests，跳过项因本地缺torch，已在服务器通过。

2026-10-19实机证据：97条/15228帧；主窗口覆盖50条/7466帧，原窗口覆盖5071帧；500步主7000/辅3000，辅助47条均被抽到，每条41–80次；辅助候选区间2948帧，占辅助帧37.98%，实际抽中1950/3000=65.00%；鼠标臂左1580/右1420。97条末尾action chunk检查通过，继承horizon50。

独立低内存训练装配验收通过：`/data/outputs/object_audits/mouse_mix_20261019_cpu_assembly_v1/preflight.json` 的 `runtime_assembly_checked=true`，主任务调度器与联合97条resolver正确；使用实际Parquet列适配器，不宣称与LeRobot全依赖同进程装配通过。

服务器原始证据：`/data/outputs/object_audits/mouse_mix_20261019_cpu_v2/preflight.json`；本地生成副本在 `artifacts/data_audits/mouse/20261019_mix_cpu_v2/`，不随Git提交。2GB无卡实例上的后续重复检查曾被内存限制杀掉，失败日志保留于 `/tmp/mouse_mix_cpu_final.log` 和 `/tmp/mouse_mix_split_probe.log`；不将其算作通过，97条chunk结果采用上述此前成功批次。GPU模型加载、首次更新、周期/1000步HF写出与闭环收益仍待开卡验证。

| 组合 | CPU证据 |
|---|---|
| 旧40/10＋辅助关闭 | 既有启动器/调度/采样回归 |
| all50＋辅助关闭 | 既有all50整链与拟合监控测试 |
| 旧40/10＋辅助开启 | 配置阶段拒绝，不进入数据/模型准备 |
| all50＋辅助开启 | 联合数据、实际采样、真实hook及97条chunk边界 |
| 辅助权重1 / 3 | 分别验证均匀帧分布 / 阶段采样偏置；episode均衡保持 |

全部输入证据和审核记录均可追溯，但CPU通过不代表模型在新混合数据上一定提升。最终以固定clean种子的鼠标抓起、抬升、放置及原任务成功率评估。

## 待开卡的batch效率对比（2026-10-19）

只在用户开卡且GPU空闲后执行。保持同一step500来源、97条数据、70:30策略及GAS=1；先测micro32并确认梯度检查点实际启用、梯度/参数更新正常，再依据峰值显存余量决定是否测试48。每档先预热，再采集至少20个稳定更新步，记录samples/s、step耗时中位数/高分位、GPU峰值allocated/reserved及设备显存占用，评测和HF保存耗时单列；比较时不将加载、预热、评测、存档混入稳定训练吞吐。短测使用独立run，不覆盖来源、不自动关机、不把短测权重接成正式来源。正式新增1000步：单GPU GAS1时batch32/48分别对应32000/48000个样本槽位；70:30配额跨步补齐。较大batch的更新次数与优化效果需结合训练监控判断，不能只按samples/s推断收益。

以稳定吞吐与显存安全余量共同选batch；当前没有32/48的GPU实测结果。该段为测前方案；完成实测后默认改为48/GAS1。正式run仍从原step500启动，短测后再确定batch。梯度检查点可用 `--no-gradient-checkpointing` 回退，步数用 `--steps` 覆盖。

GPU首档实测：batch32/GAS1、梯度检查点开启、30步完成，取11–30步平均4.9695s/步、6.4393样本/s；峰值allocated59.93GiB、reserved63.28GiB，nvidia-smi66063MiB。每步梯度范数有限且大于0，正常经过optimizer.step，末步GradNorm1.1875；短测无模型存档。32档结果位于 `/data/outputs/mouse_aux_gc_b32_probe_20261019_v1/benchmark_result.json`。后续48已完成并选为默认，见下表。

### GPU短测结论

同一step500来源、真实97条联合数据、EE10/.8/1.5、全量ViT、梯度检查点开启、GAS1，各30个实际optimizer更新，丢弃前10步后统计20步。两档均正常退出且梯度范数有限、非零；没有模型存档，也不将短测权重接入正式训练。

| micro / GBS | 平均步耗时 | 样本/s | 峰值allocated | 峰值reserved | nvidia-smi峰值 |
|---|---:|---:|---:|---:|---:|
| 32 | 4.9695s | 6.4393 | 59.93GiB | 63.28GiB | 66063MiB |
| 48 | 6.6953s | 7.1692 | 76.22GiB | 79.71GiB | 82887MiB |

48吞吐提高11.34%，选为默认。显卡总97887MiB，48档设备显存余量15000MiB；30步通过不能保证所有更长指令/长期碎片峰值，正式运行仍须监控。32→48线性外推micro72 reserved约104.36GiB，高于设备约95.59GiB，因此未跑micro72，不把外推当实测。GBS72可由micro36×GAS2构成，但会改变每次更新样本量和吞吐，未经测试，不作为当前默认。

48档证据：`/data/outputs/mouse_aux_gc_b48_probe_20261019_v1/{benchmark_result.json,benchmark_train.log,benchmark_command.json}`；本地生成副本在 `artifacts/data_audits/mouse/20261019_gpu_batch/`。日志中的训练总步数475/317是读取器epoch长度，实际max_steps均30；性能取StepTime，不使用包含加载/预热的总walltime。短测LR按30步预算变化，仅用于性能验证，不认证1000步优化效果。

正式配置：新增1000步、batch48/GAS1，总48000样本槽位，主33600/辅14400；仍继承原step500模型。每250步存一份HF共4份，当前29GiB不足；正式训练未启动，需先解决存档容量，不能直接沿用此前“开卡即启动”结论。

关闭梯度检查点的历史参考：上一轮鼠标step500 run的实际 `lingbotvla_cli.yaml` 与provenance命令均为micro20/GAS1、checkpoint=false、enable_fp32=false；不能使用来源 `trainer_config.json` 中24/FP32旧值推断实际运行。11–30步均值3.5707s、5.6011样本/s；11–499步排除每100步监控边界共485步，均值4.050567s、4.9376样本/s。新48档相同早期20步吞吐高约28%；数据组合与EE权重不同，不是隔离checkpoint开关的受控对比。

### 750步与存档间隔修订

用户在56档测试时将正式预算改为750步，并授权测试通过后直接启动。新增 `--hf-save-steps` 参数，旧入口默认250不变；辅助入口默认750（与本次总步数相同），仅保留最终一份HF，避免29GiB余量下创建三份HF。容量预判读取实际训练命令的save_steps，训练快照与worker命令同值；显式覆盖总步数时可同步覆盖保存间隔。上文1000步及每250步保存为历史测试方案，不代表本次正式配置。不会删除历史产物，不自动关机。

56档最终实测：30步正常退出，11–30步平均7.475s、7.4916样本/s，中位7.473s、p95 7.504s；峰值allocated84.34GiB、reserved88.05GiB、设备91425MiB，余6462MiB。相较48吞吐提高4.50%，选56/GAS1；梯度检查点日志确认开启，末步GradNorm0.7266。证据 `/data/outputs/mouse_aux_gc_b56_probe_20261019_v1/benchmark_result.json`。正式750步共42000槽位，主29400/辅12600；仍从原step500，最终只存一份HF。

正式运行已启动：`al_mouse_pad_aux56_gc_from500_add750_20261019_1220`（代码31184db）。服务器运行目录 `/data/outputs/al_mouse_pad_aux56_gc_from500_add750_20261019_1220`；日志 `/data/outputs/skill_launches/al_mouse_pad_aux56_gc_from500_add750_20261019_1220.log`，同名 `.status.json` 记录运行/最终校验状态。启动验收完成4个实际更新，约7.4s/步、reserved88.05GiB，实际CLI文件与97条白名单核验通过。750步最终HF与效果仍待完成后检查。

### 84GB卡中断后重训（用户2026-10-19授权）

旧750步run日志止于427步，无检查点，无法恢复这些更新；保留其日志。新run从原鼠标step500权重重训500步，48/GAS1，主50＋辅助47、70:30、梯度检查点和其他采样策略保持。位置EE由10升至12，姿态0.8/夹爪1.5保持；当前深度、未来深度及视频外层权重各由0.004提高25%至0.005，视频内部MSE/cosine等配比保持。`--alignment-loss-scale 1.25` 仅在准备来源训练配置快照时应用一次，worker读取快照不重复相乘，原始HF和来源配置不改。

存档恢复为每250步HF（250和500两份），经用户明确授权；以后任何保存节奏/保留策略变动必须再次获得用户明确允许，已写入根AGENTS.md。剩余约64GiB，启动容量按两份实际权重核算。默认开启成功后自动关机：仅训练退出0且最终HF资产/张量清单校验通过后调用既有关机流程，失败保留服务器。84GB新卡为RTX 6000D，85651MiB；旧96GB卡的48档显存结果仅作参考，需确认新run实际峰值。

84GB重训已启动：`al_mouse_pad_aux48_gc_p12_av125_from500_add500_20261019_restart_v1`（代码f1f1b15），目录 `/data/outputs/al_mouse_pad_aux48_gc_p12_av125_from500_add500_20261019_restart_v1`，同名日志及status在 `/data/outputs/skill_launches/`。启动验收已完成至少10步更新，约10s/步，GradNorm正常；reserved79.71GiB，设备82811/85651MiB。实际CLI的48/GAS1、P12、三项0.005、500步/250间隔及状态shutdown=true均核验通过；仅最终HF资产与张量名校验成功才关机。250/500实际HF仍待训练完成验证。
