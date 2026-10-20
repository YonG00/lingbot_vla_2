# adjust_bottle 两组训练后自动对照

## 已完成结果（2026-10-20）

标准FP32组250/500均2/3；增强FP32组250为1/3、500为2/3，场景种子均为100001、100002、100005。
实际增强500完成目录为`enhanced_500_retry_final`，原`enhanced_500`为中断记录，勿当作完整结果。
四组汇总为队列目录`four_checkpoint_results.json`；旧`status.json`保留首次缓存错误状态，非最终结果。
当前采用后续BF16/GBS56/200步的3/3版本，见[基准说明](adjust_bottle_bf16_baseline_zh.md)。
以下“当前/等待/未验证”均是当时实施阶段记录。

## 当时已授权流程

1. 等当前标准方法组500步训练完成、250/500两份FP32 HF校验通过。
2. 对该组最终500步模型闭环评测3回合。
3. 在独立增强代码目录做静态窗口准备、真实CPU数据/FK/加权DataLoader检查、三步GPU探针。
4. 从原始9680新起增强组500步，250/500各保存FP32 HF。
5. 对增强组最终500步模型闭环评测3回合，生成配对比较JSON。

队列不会与当前训练并发占用GPU，任一步失败即停止后续，保留日志，不悄悄关闭功能或跳过存档。
不再跑被用户取消的训练前9680基线。这里“3轮”落实为每组3个完整闭环回合。
三场景对照只能提供初步迹象，不能稳健估计成功率或分别归因EE、窗口与LN-FM。

## 位置与隔离

- 当前组代码：`/data/experiments/official_adjust_bottle_control`。
- 当前组输出：`/data/outputs/official_adjust_bottle_9680_m28_add500_20261020`。
- 增强代码：`/data/experiments/adjust_bottle_enhanced_control`，从保留增强实现的7370720隔离克隆。
- 增强输出：`/data/outputs/enhanced_adjust_bottle_9680_m28_add500_20261020`。
- 队列状态/日志：`/data/outputs/adjust_bottle_paired_20261020/status.json`、`pipeline.log`。
- 评测及对比：同队列目录下`official_500/`、`enhanced_500/`、`comparison.json`。

准备前后校验当前训练配置和三份模型核心文件SHA256完全不变。
两组均由当前标准方法目录的同一部署代码评测，使用FP32加载、noise seed0、chunk50、demo_clean，
相同专家可行性筛选与客户端。记录实际3个场景种子，增强组不一致则拒绝出比较结果。
检查OIDN2.5.1、零OIDN错误、完整逐回合视频和结果/日志一致。

## 明确开关

```bash
python experiment/paired_control/prepare.py \
  --source-run /data/outputs/official_adjust_bottle_9680_m28_add500_20261020 \
  --out /data/outputs/enhanced_adjust_bottle_9680_m28_add500_20261020 \
  --ee-loss --window-sampling --ln-fm
```

对应关闭参数为`--no-ee-loss --no-window-sampling --no-ln-fm`，默认三者均关闭。
已有EE实现要求LN-FM同时开启；`--ee-loss --no-ln-fm`明确报错，不自动打开其他功能。
这些是新进程启动参数，不能热修改已运行训练。

增强组继承当前组的实际配置：数据全50条7156帧、micro28/GAS1、FP32主参数与本轮FP32计算方向、
compile开启、workers8、GC开启、seed3、action_fp32=true、同Muon/LR/视觉损失与500步调度。
只改输出/数据清单文件位置与三项方法字段，`model_path`保持原9680，优化器/步号重新开始。
不会继承标准组500步，也不会继承三步探针权重。

- EE位置/姿态/夹爪权重：3/0.8/1.5；附加参数`--ee-position --ee-orientation --ee-gripper`。
- 窗口内权重2、外部1；沿用现有target-change缓存算法、0.25秒窗口。
- LN-FM沿用已有25%概率替换为低噪声时间实现。

窗口使用原始PD目标中的夹爪/腕关节变化，不是接触力或实际物体高度标签。
`train.py`只在开关打开时注入加权采样器；关闭时原DataLoader函数和RNG路径不变。
没有AL任务调度、PASS、早停或额外100步开环，因此不会给增强组增加另一套训练门控。
窗口缓存逐个episode/原始绝对帧坐标与过滤后数据严格匹配，拒绝不完整、重复或错数据集缓存。

## 验证与观察

纯CPU契约：

```bash
python -m unittest experiment.paired_control.test_methods -v
```

本地与服务器均5项通过，含8个开关组合子用例、关闭采样不调用factory/不耗额外RNG、
绝对坐标映射、三场景配对约束以及源配置不变、非方法参数逐项不变。
当前训练未结束时不执行Torch测试或GPU探针；实际数据/FK/weighted loader检查已排在增强组启动前。
排队时只读核验当前组250步HF已发布：1708张量、全F32、23.752GiB，张量名与9680完全一致。
该读取不加载模型、不占GPU，500步仍由完成触发器再次校验。
增强GPU探针验证实际EE指标、FP32参数/梯度/关键层输出和完整3步峰值，未通过不进入正式500步。
TensorBoard已为增强输出建立日志链接，开始写事件后会出现在同一面板中。

不变量：标准组训练配置/核心哈希不变，两组源9680与非方法超参数一致，两份存档保留，评测场景相同。
症状：队列failed、窗口坐标不符、EE无实际指标、精度审计失败、保存不完整、场景种子不一致。
回滚：只停后续队列及它新启动的子任务，不停当前标准组训练；关闭参数只对新进程生效。
验证入口：上面的CPU契约及队列`status.json`。当前阶段是自动队列已准备/等待父训练，非增强GPU已验证。
