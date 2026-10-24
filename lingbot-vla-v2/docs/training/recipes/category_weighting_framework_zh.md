# 统一类别与逐动作加权框架

## 长期约定

今后的自定义加权训练统一使用 `weighting_unit=action_timestep`、`normalization=weight_sum`。不再由当前输入帧决定整个chunk的倍率。完整轨迹分阶段与目标数据分类都是“给原始帧贴类别”，只在标签生成方式上不同；训练端共用同一CategoryTable与chunk查表逻辑。

未配置自定义重要性的官方基线保留原路径。已完成的旧运行、存档、冻结文件不回写；新运行读取旧权重文件时逐动作适配，新冻结注册表明确写入新的单位和归一化口径。

## 两种标注，共用一个参数文件

|来源|类别示例|标签生成|
|---|---|---|
|完整轨迹阶段|reach_aim/reach_far、pre_contact/forward_advance、grasp_lift/initial_clearance|按运动边界给左右手整条轨迹逐帧标注|
|目标数据分类|early_placement、late_placement、white_stapler_demo|按目标条件标记帧，允许多类重叠，未命中归background|

统一产物：

- `frame_classes.json`：原始episode/frame坐标、完整双臂区间，区间左闭右开、无空洞或重叠；每段可有多个类别。
- `category_loss_weights.json`：唯一的类别倍率调参入口。
- `weight_config.json`：绑定类别表、权重与全帧坐标SHA256；不代表提高采样概率。
- `coverage.json`：原帧、动作chunk展开与视觉目标权重统计，不能当作实际训练进度。

目标分类的标签不写arm则同时适用于两臂，写arm则只影响该臂。共享视觉按目标时刻左右手最大值。订书机当前分类沿用“样本范围同时适用于两臂”，并非只乘正在操作的那一臂。

## 怎么调

两种数据使用相同JSON结构。订书机当前配置：

```json
{
  "version": 1,
  "weighting_unit": "action_timestep",
  "normalization": "weight_sum",
  "overlap_rule": "max",
  "future_video_rule": "target_endpoint",
  "classes": {
    "background": {"fm": 1, "visual": 1},
    "early_placement": {"fm": 15, "visual": 15},
    "late_placement": {"fm": 20, "visual": 20},
    "white_stapler_demo": {"fm": 10, "visual": 10}
  }
}
```

`fm`为动作，`visual`默认同时控制当前Depth、Future Depth、当前Video、Future Video。若需要分手，可写 `"fm": {"left": 3, "right": 2}`；visual也接受分手映射。只有需单独控制某视觉分支才增加 `visual_branches`，它按分支覆盖visual默认值。阶段迁移只在原分支值确实不同才生成覆盖，避免修改visual却被冗余配置盖住。

重叠取最大值、不叠乘。background和unknown保持1，未知类别/手臂拼写、缺失阶段、非法倍率、错误SHA会拒绝加载。当前10<15<20，所以白色样本落在早期取15、晚期取20。若以后把白色设30，重叠会取30，这是max规则的结果。

修改新副本后冻结新快照，两个标注来源命令相同：

```bash
python tools/snapshot_category_weights.py \
  --labels-dir OLD_SNAPSHOT \
  --weights NEW_category_loss_weights.json \
  --output-dir NEW_SNAPSHOT
```

先验证，再原子发布；不覆盖已有目录，不热改运行中的冻结文件。固定任务重要性约束仍由原框架检查，迁移格式不等于授权改掉已固定的任务权重。

## 一个chunk如何算

假设输入帧t属于早期，而t+10开始晚期：chunk前10个动作取15，后续晚期动作取20；如果之后进入其他类，继续按各动作真实数据帧查表。动作由state转换时，使用实际 `action_offset`，不把图像时钟一起偏移。

- FM第k个动作：查 `t + action_offset + k` 对应类别，按该臂与类别max生成 `[B,H,D]` 权重。
- 当前Depth/Video：查当前图像帧t。
- Future Depth/Video：查实际目标帧 `t+H-1`；当前teacher是未来端点，不假装逐帧预测整段视频。
- 越界未来图像权重0；动作padding及未用关节同时排除FM分子和分母。
- FM为 `sum(loss * weight * valid) / sum(weight * valid)`，视觉各分支独立按有效权重和归一化；不重复叠乘旧标量。

源数据的类别数量与目标片段范围不变，采样概率也不变。改变的是chunk内部每个动作真正收到的倍率。

## 准备与迁移

两个准备入口现在默认输出统一类别接口：

```bash
python tools/label_dual_bottle_stages.py --task pick_diverse_bottles \
  --dataset-root DATASET --object-map OBJECTS.json --critical-weight 20 --output-dir NEW_STAGE_DATA
python tools/prepare_stapler_sample_weights.py --output-dir NEW_TARGET_DATA
```

旧完整阶段权重迁移：

```bash
python tools/convert_category_weights.py --source-config OLD/weight_config.json --output-dir NEW
```

旧目标分类迁移还需 `--target-catalog` 和 `--class-weights`；目录中的原始帧类别从旧selected_frames的标签读取，未命中帧补background。旧endpoint_max视觉策略不静默转换为新时钟，须明确处理后再迁移。

现成迁移目录：

- `refs/data_selection/move_stapler_pad_categories_early15_late20_white10_v1/`
- `refs/data_selection/pick_dual_bottles_categories20_lift15_v1/`
- `refs/data_selection/pick_diverse_bottles_categories20_v1/`

后两者逐手逐帧、全部5个损失分支迁移前后相同。旧坐标权重文件仍可读取，未来动作和未来视觉会查各自数据帧；不再复制锚点倍率到整个chunk。旧role模式若只修改锚点标量，不可与新逐动作权重混用，需改为类别参数。

内部stage_fm_weight等字段名暂作兼容接口，目标分类不会因此被伪装为传统阶段；无legacy_stage的目标类别不写入旧主阶段统计。当前线上已有全局FM及原主阶段监控，不声称已新增任意类别名称或左右夹爪的在线误差曲线。

## 验证状态与训练门槛（2026-10-23）

195项CPU回归通过：类别边界、动作offset、未来图像时钟、双臂max、非法类别、背景、冻结依赖及旧格式新运行规范化。原双瓶6129帧、多样瓶子6060帧全部动作/视觉帧权重迁移等价。

真实Parquet订书机50条7731帧，500个边界/相邻锚点及offset检查通过，含243个跨类别chunk；同时核对未来深度/视频时刻与越界0。该轻量检查在原始14维动作坐标执行，不冒称完成55维模型DataLoader全链。

完整DataLoader检查在无卡2GiB容器中返回SIGKILL(-9)，未得到完成报告；没有足够证据证明具体原因。其状态明确标incomplete，旧整chunk方案的CPU通过报告不能替代本版。开卡后仍需先通过原框架完整CPU preflight，未通过不启动GPU训练。新版本尚未训练、未测GPU显存/速度/闭环，没有新增Checkpoint。


2026-10-24补验：开卡主机上完整CPU DataLoader通过，50条7731帧、动作权重[2,50,55]、150个类别边界，weight_sum/action_timestep；模型未初始化、优化器0步。195项CPU回归重跑全绿。此前-9保留为历史失败，不再阻塞本轮启动。证据：refs/data_selection/move_stapler_pad_categories_early15_late20_white10_v1/full_cpu_verified_20261024.json。GPU训练与闭环尚待实际终态，不能计入完成步数。
