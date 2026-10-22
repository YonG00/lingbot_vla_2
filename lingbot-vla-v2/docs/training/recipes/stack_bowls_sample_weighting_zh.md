# 夹碗逐样本损失加权实验

代码分支 `experiment/stack_bowls_sample_weighting`，代码起点 `459a943`；计划模型起点 R17 step1000。尚未同步服务器或训练。

## 规则与启用

仅定制 `stack_bowls_two`，配置 `configs/data_selection/stack_bowls_two_v1.json`。离线通过示范后续闭合位置追溯上方对准、下降、闭合，并用抬升确认。新增瞄准阶段：向稳定预抓位置前追最多4秒，夹爪保持打开，累计接近至少2厘米，距离增加超过2毫米的增量不超过30%；允许早期方向尚未朝下。不代表在线目标检测、真实碗边对准或接触成功。

```bash
python tools/select_grasp_sequence.py --task stack_bowls_two \
  --dataset-root <数据集根目录> --dataset-name <训练清单第一列data_name> \
  --settings configs/data_selection/stack_bowls_two_v1.json \
  --output-dir <新的筛选结果目录>
```

生成 summary、events、selected_frames 和 weight_config JSON。缓存模式用 `--frames-json`、`--fps` 和 `--training-dataset-root` 替代 `--dataset-root`。配置含清单 SHA256、数据集名和根目录；坐标为原始 episode_index/frame_index，不使用过滤后的 local sample_id。启用前还须核对数据版本和数值源指纹；清单文件指纹不等于验证整个训练数据集内容未改变。

训练启动前设置 `LINGBOT_SAMPLE_WEIGHT_CONFIG=<筛选结果目录>/weight_config.json`，数据集实例初始化时读取；取消变量并重启训练即可关闭。未设置时不增加样本字段。仅支持 V2、单卡、GAS=1、非 repeat FM；不支持的组合明确拒绝。

## 损失口径

命中样本权重2，其余1。FM、当前 Depth、Future Depth、Video（启用的 patch/CLS/current 分量）在现有各项系数之前做归一化加权；路由与序列正则不加权。无掩码时为 sum(w * per_sample_loss)/sum(w)；FM 有 joint_mask 时为 sum(w * mask * loss)/sum(w * mask)，保持原有效元素语义。全部权重1时应与原公式数值、梯度一致（浮点容差）。动作块/未来视觉标签整体随起始帧加权，不是块内时间步单独筛选。

TensorBoard `sample_weight/` 记录命中率、平均权重、未加权 FM/Depth/Future Depth/Video；原 batch_mean_losses 保留未加权 FM。各项全局系数不变，加权后的总损失不直接与基线比较。

## 当前验证边界

50条轨迹15609帧，100个运动学序列，加入瞄准后7436帧（47.64%），旧规则3470帧（22.23%）。新增覆盖仍待视频边界抽检，不声明其全部为正确抓碗样本。筛选为离线代理；尚无本实验 GPU 或闭环成功率证据。

CPU契约测试：`python -m unittest experiment.data_selection.test_sample_weighting -v`；从生产源码提取实际损失方法执行，避免重型模型导入，不能替代完整数据加载/模型 GPU 验证。

实测：9项 CPU 契约测试全部通过（PyTorch 2.2.2，CPU），包括真实梯度及生产损失方法提取执行。完整 Transformer/数据加载依赖未在本地装载，未做 GPU 或闭环验证。

## 专训200步计划（容量阻塞，未启动）

来源已更新为 `20261022_1209_round_3_recovery/checkpoints/global_step_160/hf_ckpt`，不是step180。BF16/micro56/GAS1；stack_bowls_two 200步，10%单任务复习每5步轮换，冻结沿用第三轮补课25任务池并排除当前主任务。500步保存间隔及最终保存保持原设置，本轮只产生step200。训完夹碗3回合，录前2回合、驻留模型评测；无补课、无自动关机。权重2按权重和归一化，未改成按样本数归一化。启动前应对来源模型做同种子夹碗对照评测。

配方 `configs/training_recipes/stack_bowls200_from_recovery3_s160_weight2norm.json`；空间安全预算16.06GiB，审计可用7.38GiB，尚差8.69GiB。未删除任何产物，未启动训练。

## 用户更新：按原样本数归约（2026-10-22）

本次最终口径覆盖前文：命中样本loss×2，其余×1，除以batch样本数；不除权重和。FM带joint_mask时保留原有效元素总数作分母，不按权重调整分母，保证每个命中样本原本的贡献精确翻倍。本任务固定关节掩码时等价于逐样本有效均值除以B。正则项不放大。清单配置必须显式写 normalization=sample_count，旧权重和配置拒绝使用。配方改名 stack_bowls200_from_recovery3_s160_weight2sample.json。

已按授权删除清单4、5、7、10，仅对应存档目录；保留日志评测和最新第三轮step160。释放47.56GiB，空闲54.94GiB，容量已满足。

## 第二轮实验：抓取＋第二次放置并集，权重3

用户授权：仍以 `20261022_1209_round_3_recovery/global_step_160` 的实际 `checkpoints/global_step_160/hf_ckpt` 为起点，不续训上一轮加权step200。主任务200步，micro56/GAS1/BF16，10%单任务复习每5步切换，排除当前主任务；100/200保存（用户明确授权新节奏）。最终step200测夹碗3回合、全部录像，驻留复用训练模型；无补课或自动关机。

两套选择结果按原始episode/frame取并集：抓取7436帧，第二次放碗3743帧，合计11179/15609（71.62%），实际重叠0；有重叠时权重仍3，不叠乘。四项数据损失统一乘3后按原样本数归约；FM掩码保留原有效元素分母，原全局系数和正则不变。复习任务默认权重1。

配方 `configs/training_recipes/stack_bowls200_from_recovery3_s160_grasp_place_weight3.json`；合并命令 `tools/combine_sample_selections.py`；权重配置必须显式sample_count。真实两份存档预算29.13GiB，启动前空闲42.87GiB。

run：`/data/outputs/20261022_1904_bowls200_from_recovery3_s160_bf16_m56_replay10pct_grasp_place_weight3`；启动检查中。19项筛选/损失/依赖CPU测试＋63项配方测试通过，15项增量文件哈希冻结；未合并基线分支。GPU结果及正式进度待后续记录。

### 启动时更正：继续上一轮权重2的step200

用户取消上述“重新从第三轮step160起训”的来源，改为继承刚刚抓碗动作已有进步的 `20261022_1246_bowls200_from_recovery3_s160_bf16_m56_replay10pct_weight2sample/checkpoints/global_step_200/hf_ckpt`。旧19:04流水线已停止，最新新run为 `/data/outputs/20261022_1909_bowls_add200_from_weight2_s200_bf16_m56_replay10pct_grasp_place_weight3`。继续加训200步，保存仍为本轮step100/200，不累计来源步号；双规则并集3倍、10%复习、GBS56/GAS1、最终3回合全录像不变。配方为 `configs/training_recipes/stack_bowls_add200_from_weight2_s200_grasp_place_weight3.json`。

## 重要性随任务继承，复习时仍生效

用户要求：stack_bowls_two关键样本成为复习数据时，也保持训练阶段的重要性。样本权重在VLADataset.getitem按数据集＋原始episode/frame判定，不依赖主任务/复习身份；复习比例只决定抽样数量，不改变样本自身损失权重。当前双规则关键样本仍为3，其余为1；四项损失及按原样本数归约保持不变。不把本策略描述为不会遗忘的保证。

新增 `sample_weight_registry.json` 按任务保存策略；每个任务配置和所选帧均拷贝到run内 `sample_weights/<task>/`，并校验配置/帧清单指纹。后续prepare从实际来源checkpoint所在run或随HF携带的registry继承，不依赖上次shell环境。显式新配置只更新对应任务，保留其他任务；未带策略的基线仍默认无加权。当前只支持同一合并数据集，不同数据源或任务帧坐标冲突会明确拒绝。

run.py把冻结registry传入CPU/探针/训练子进程，生成train.sh也显式导出该路径；runtime.load_run验证策略未篡改。未来使用新代码的流水线在保存验证后给HF附带元数据，便于移动模型时保留。当前运行中的旧流水线不热更新，已单独补齐run的registry；下次续训前需要同步本次新代码。

已补元数据：12:46 run保持抓碗7436帧×2；19:09 run保持抓碗＋第二放置并集11179帧×3。没有把历史2倍模型误记为3倍。6项纯标准库CPU测试通过（跨run无环境继承、多任务保留、同任务显式替换、文件独立性、篡改拒绝、旧run兼容）；尚未跑新的跨run GPU整链。
