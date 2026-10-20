# adjust_bottle 当前实验基准：BF16 / GBS56 / 200步

2026-10-20用户指定本版作为后续adjust_bottle实验基准。它在相同3个闭环场景中全部成功。
本决定只适用于本实验分支，不修改生产`Task_Auto_Learning`的训练默认值。

## 已验证结果与模型

- 分支：`official_adjust_bottle_control`；服务器代码：`/data/experiments/official_adjust_bottle_control`。
- 训练输出：`/data/outputs/official_adjust_bottle_9680_bf16_m56_add200_20261020`。
- **采用的模型**：上述目录的`checkpoints/global_step_200/hf_ckpt`。
- 完整性：1708个张量，浮点权重BF16，约11.88GiB；`verified_checkpoint.json`验证通过。
- 闭环：`evaluation_200/summary.json`，**3/3成功**，实际场景种子`[100001,100002,100005]`。
- `demo_clean`、FP32加载/推理、chunk50、推理noise seed0；同一服务加载一次连续测3回合，结束再下线。
- OIDN2.5.1，OIDN错误0，三段全局视频在`evaluation_200/results/adjust_bottle/`。
- 配置和结果已冻结在[`adjust_bottle_bf16_200.json`](../../../../experiment/official_control/baselines/adjust_bottle_bf16_200.json)，包含实际trainer配置、来源校验SHA、存档报告和前四组结果。模型大文件不提交Git。

| 实验 | 精度/GBS/步数 | 同场景闭环成功 |
|---|---|---:|
| 标准方法 | FP32 / 28 / 250 | 2/3 |
| 标准方法 | FP32 / 28 / 500 | 2/3 |
| EE＋窗口＋LN-FM | FP32 / 28 / 250 | 1/3 |
| EE＋窗口＋LN-FM | FP32 / 28 / 500 | 2/3 |
| **当前采用基准** | **BF16 / 56 / 200** | **3/3** |

三个固定场景只支持本轮选择，不等于广泛场景100%成功。精度、batch、步数/调度长度同时改变，
不能将改善单独归因于BF16；本版共11200样本曝光，FP32/28/500为14000，均非独立样本数。
全50条用于训练，没有独立开环验证集；原始9680未完成同规格3回合评测，不能据此量化相对9680的增益。
“官方方法”指恢复仓库初始快照`86f6728`的模型核心与学习方式，未独立核实上游发布一致性。

## 固定训练配方

| 配置 | 值 |
|---|---|
| 来源 | `al_v79_poolfix_7300/checkpoints/global_step_9680/hf_ckpt`，新优化器/新调度，不继承探针权重 |
| 数据 | 仅adjust_bottle全部50条、7156帧；无辅助混合、无窗口，普通帧采样 |
| 精度 | BF16参数和计算；`enable_mixed_precision=false, enable_fp32=false, action_fp32=false` |
| batch | 单卡micro=GBS56，GAS1；96GB RTX PRO6000 Blackwell实测 |
| 方法 | AL关闭、EE关闭、低噪FM关闭；原始L1 FM与Beta时间采样 |
| 优化 | Muon，LR `1e-4→5e-5`，200步cosine；专家LR缩放sqrt(8) |
| 视觉监督 | 深度、未来深度、未来视频各0.004；全参数训练，视觉不冻结，无LoRA |
| 资源 | GC开启、compile开启、workers8、seed3 |
| 存档 | `save_steps=250`保持不变；本轮不足250步，最终200步仍保存BF16 HF，HF-only同步保存 |

保存间隔、是否保留中间存档和保留策略的后续更改仍须用户明确允许。
BF16 HF加载成FP32只转换推理dtype，不恢复BF16量化已经舍去的精度。

## 复用入口

在对应服务器实验checkout的`lingbot-vla-v2/`目录中，使用训练环境Python：

```bash
# 默认只打印计划，不创建运行目录、不启动训练
python experiment/official_control/baseline_pipeline.py --out /data/outputs/adjust_bottle_bf16_baseline_new
# 用户授权训练后显式启动后台流水线；输出目录必须全新
python experiment/official_control/baseline_pipeline.py --out /data/outputs/adjust_bottle_bf16_baseline_new --execute
```

入口从Git中的冻结配方准备，不依赖旧FP32/500运行目录存在。服务器数据、9680、Qwen、教师模型、
归一化以及OIDN运行库仍须存在；路径见快照，换服务器应先检查这些路径。
自动顺序：CPU整链→独立三步GPU探针→从9680新起200步→校验最终BF16 HF→同种子3回合闭环。
GPU忙时拒绝启动，不与其他任务并发；失败即停、不删旧结果、不自动关机。
新入口是运行后整理的可复用入口；原成功run由当时恢复控制器执行，原成功证据独立归档，
不把入口的CPU验证冒充重新进行了一次200步训练。
状态/控制器日志在`<out>_pipeline/`，训练日志、`runs/`与评测在`<out>/`。
6007服务使用`/root/tf-logs`时，可将新run的`runs/`链接进去；本次成功run已接入。

## 精度修复与验证

单卡BF16参数时，数据state/actions仍可能是FP32；旧精度上下文直接跳过autocast，
导致首步线性层报`Float and BFloat16`。现对单卡BF16计算统一启用autocast；FP32计算和FSDP路径不变。
不修改三个模型核心快照，亦不以关闭compile或降低GBS规避错误。
探针动作输出hook的观测副作用独立于编译图，验证不改变输出/梯度；正式训练不启用该审计hook。

CPU入口：

```bash
python -m unittest experiment.official_control.test_baseline experiment.official_control.test_bf16_context experiment.official_control.test_audit_contract experiment.paired_control.test_methods -v
```

整理后服务器上述13项CPU测试全部通过；从Git冻结快照独立生成新配置，再跑真实50回合/7156帧CPU整链也通过，
报告在`/data/outputs/adjust_bottle_baseline_repro_cpu_20261020/cpu_report.json`。
本地缺torch，两项真实算子测试跳过；服务器已执行，不将本地跳过计为通过。

GPU探针：参数和梯度6375906359个元素均BF16，视觉/语言/动作专家/动作输出均BF16；
峰值allocated **77.7115GiB**、reserved **85.1465GiB**，完整三步通过。
compile存在Qwen3-VL nonzero图中断，没有同配置稳定阶段A/B证据证明编译更快。

不变量：源9680、50条、三增强关闭、56/GAS1、200最终BF16 HF、相同评测种子；
症状：精度/梯度断言、HF完整性或seed校验失败。回滚：不用新基准入口，选择历史配置另起运行；
不可通过修改运行中配置回滚，不修改生产目录或删除本版模型。
