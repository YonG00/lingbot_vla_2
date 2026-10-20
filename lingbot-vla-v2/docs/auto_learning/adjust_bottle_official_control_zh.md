# adjust_bottle 标准方法对照（FP32历史实验）

分支：`official_adjust_bottle_control`；独立机器目录：`/data/experiments/official_adjust_bottle_control`。
保留原实验目录，禁止同步到生产 `/data/code`。

当前采用的BF16/GBS56/200步基准、3/3闭环证据与新启动入口，见[专门说明](adjust_bottle_bf16_baseline_zh.md)。
以下记录最初FP32/500步对照的设计与当时阶段，文中的“待验证/未启动”不代表当前状态。
FP32的250/500均已完成且各2/3成功；原10回合9680预评测已由用户取消。

## 历史实验口径

从 `al_v79_poolfix_7300/checkpoints/global_step_9680/hf_ckpt` 起新的优化器和学习率调度，
只训练 adjust_bottle 的全部50条示范、7156帧，500个优化器步，250/500各保存FP32 HF。
全数据训练没有独立开环验证集。用相同10个场景种子比较9680、250和500的闭环成功率。

标准方法依据是仓库最初快照 `86f6728`，并未独立验证它与上游发布逐字相同。
该快照的三个核心文件（configuration、modeling v1、modeling v2）在本分支完整恢复，
SHA256登记在 `experiment/official_control/snapshot_identity.json`。
模型前向不包含EE损失和低噪声FM实现。时间采样恢复 `Beta(1.5,1.0)*0.999+0.001`，
并非均匀时间分布；均匀的是训练帧采样。

不传AL配置，不构建AL sampler/hook，无窗口、难度扫描、辅助混合、PASS和早停。
保留原L1 FM、深度/未来深度/视频权重0.004、MoE损失、Muon与原学习率配置。
不冻结视觉，不使用LoRA，不仅训练专家，图像增强关闭。

初始配置精度开关 `enable_mixed_precision=True, enable_fp32=True`，动作/状态FP32。
实际各模块计算精度必须上GPU审计，不把开关名称当作全模型FP32计算的证明。
默认准备入口micro=16、GAS=1，实际本轮用户指定micro28；单卡、GC开启属于资源适配，
用户随后指定compile开启、数据加载进程8，已恢复这两项初始配置。
不宣称复现原32卡GBS1024。当前不自动开始训练。

保留的工程修改包括：LeRobot v3 episode过滤后的绝对/相对索引修复、显式归一化、
HF完整加载、环境兼容及同步HF-only保存/容量保护。保留文件仍与初始快照有差异，
本实验是“从9680开始的标准方法对照”，不是无修改的官方复现。

## 准备及验证

2026-10-20服务器CPU整链通过：真实参数解析、episode 0–49共7156帧、50个episode首帧
图像/动作/mask取样、动作与状态有限值检查、原FM采样逐值对照均通过。
日志为输出目录`cpu_check_v4.log`，汇总`cpu_report.json`。最初直接导入训练模块会触发
CUDA设备查询，现提取真实参数dataclass和FM方法执行，不伪造CUDA设备。
CPU检查使用单线程适应无卡实例0.5核配额；不改变正式训练线程设置。
GPU容量、实际精度、闭环成功率均未验证；未开始正式训练。提交记录通过本分支Git历史查询。

在独立目录下：

```bash
python experiment/official_control/prepare_adjust_bottle.py
python experiment/official_control/prepare_adjust_bottle.py --cpu-check
```

默认输出 `/data/outputs/official_adjust_bottle_9680_add500_20261020`，拒绝覆盖。
CPU检查核对50个episode、帧总数、各episode首帧实际取样、动作shape/mask/有限值，
并验证原始FM时间采样逐值一致。CPU检查不初始化模型、不训练。
`train.sh`仅生成，不执行；正式运行前必须完成GPU容量和dtype检查、9680闭环基线。
评测规格记录在 `plan.json` 和 `evaluation_seeds.json`：demo_clean、100000–100009、
FP32加载、动作块50；同样条件评测250/500，保留渲染检查与逐回合视频。
评测入口已准备，默认只打印计划，必须显式`--execute`才后台运行：

```bash
python experiment/official_control/evaluate.py --out /data/outputs/official_adjust_bottle_9680_add500_20261020 --label 9680 --execute
```

完成基线后，对250/500分别更改`--label`；检查实际种子列表与9680完全一致。
客户端会先进行专家可行性筛选，因此100000–100009是初始候选；
若跳过不可行场景，以9680报告中的实际10个种子为准。不同模型若实际种子不一致，评测报错。
脚本验证10回合完成、结果文件一致、10段全局视频、FP32加载审计及OIDN无错误，
不修改已验证客户端，不自动关机。当前尚未执行GPU评测。

GPU容量探针入口（先后台运行，完成后单独读取日志）：

```bash
python experiment/official_control/gpu_smoke.py --out /data/outputs/official_adjust_bottle_9680_add500_20261020 --smoke-out /data/outputs/official_adjust_bottle_fp32_m16_smoke
```

探针从9680新起3步，不保存权重，不更改正式250/500存档节奏；
检查参数、梯度及四组Linear输出FP32，记录峰值显存，成功后也不会自动启动正式训练。

## 回滚与不变量

不变量：三核心文件hash与快照相同，AL为空、EE/低噪声关闭、250/500保存不变。
症状：hash验证失败、出现AL hook/EE loss日志、任务episode不符、读取报错或保存缺失。
回滚：停止本实验后切回 `learn_skill_mix_task`；原目录不受影响。
验证入口：`prepare_adjust_bottle.py --cpu-check` 与运行时配置/dtype审计。

## 2026-10-20 micro28开卡探针

用户指定micro=28、GAS=1；新输出目录`/data/outputs/official_adjust_bottle_9680_m28_add500_20261020`。
96GB RTX PRO 6000 Blackwell三步探针通过，参数/梯度/优化器浮点状态与视觉、语言、
动作专家、动作输出均F32。第一步峰值分配76.965GiB、预留77.660GiB；完整三步
分配83.584GiB、预留90.525GiB。第二/三步StepTime为5.682/5.552秒（仅短探针，非长期吞吐）。
初次探针已更新1步后因functional输出层未触发模块hook而失败，保留原日志；
修正仅观测手段，CPU输出与梯度不变契约通过，重试从9680重新起3步，没有继承探针权重。
探针目录`/data/outputs/official_adjust_bottle_fp32_m28_smoke_20261020_v2`。
后续流水线先做9680十回合基线，再从9680正式训练500步、验证250/500两份HF并评测。
状态文件为新输出目录`pipeline_retry.status.json`，探针补丁尚未提交。

用户随后要求开启compile并恢复数据加载进程8。该组合CPU复验及三步GPU探针通过，
完整峰值分配81.761GiB、预留91.799GiB，micro28/GAS1不变。
Inductor对Qwen3-VL部分nonzero算子graph break，不能称全模型无回退编译。
新探针目录后缀`_v3_compile`，流水线状态改为`pipeline_compile.status.json`。
独立目录初次基线遗漏根目录`.runtime`导致旧OIDN错误，该次已停止并保留为
`evaluation/9680_oidn_failed`；根目录已链接已验证OIDN2.5.1，重做基线后才允许正式训练。

### 用户取消训练前基线

用户随后明确要求“没必要，直接开始训练”。已停止当前基线及其旧调度器，
保留`evaluation/9680`中已完成的部分结果，不能标成完整10回合基线。
正式训练已直接启动，调度入口`/tmp/adjust_direct_train20.py`，
状态`direct_training.status.json`，日志`train.log`，目标500步和250/500保存不变。
新的直接训练入口完成后验证两份FP32 HF；不再受基线完成状态约束。
此前关于“基线完成才训练”的描述是被本次用户指令替代的历史计划。

## 空间清理记录

用户最终指令为训练存档只保留9680。2026-10-20删除五份500/250训练存档，
释放95.087 GiB；随后清理旧4050归档HF，再释放11.907 GiB，当前可用128.54 GiB。底座/教师/数据/评测记录保留。
远端删除清单：`/data/outputs/adjust_bottle_official_control_preparation_20261020/checkpoint_cleanup.json`。
该授权不改变新实验250/500存档节奏，也不授权后续自动删除新存档。
