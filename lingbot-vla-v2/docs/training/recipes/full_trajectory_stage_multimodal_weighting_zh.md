# 双瓶完整轨迹分阶段与多模态差异化加权

2026-10-23：已实现、同步实验目录并完成CPU验证；未启动本方案训练，未验证GPU显存、速度或闭环效果。基线工作区不改动。

## 标签与策略分离

冻结产物位于 `refs/data_selection/pick_dual_bottles_full_stage_v1/`：`segments.json` 为完整标签，`stage_loss_weights.json` 为权重，`weight_config.json` 绑定文件SHA256，`policy_coverage.json` 为全锚点统计。标签坐标为原始episode/frame，区间左闭右开。任务50条示范、6129帧，每只手完整覆盖，无unknown。

共有9个实际主阶段、每手17类子阶段。按运动与距离边界划分：准备；远近伸手；精细对准；最后前伸/开爪稳定；闭爪；闭爪保持/初始离地/提起；远中近搬运；终点对准；末尾保持。没有给这个无松爪任务虚构释放阶段。最大子阶段占比左9.33%、右9.27%，10%作为复核阈值，不能通过机械等长切片掩盖过宽标签。主阶段合计可以超过10%。

这些是成功专家轨迹中的关节FK、夹爪命令与运动阈值代理标签，不是接触传感器真值；`proxy_only`、证据和对象身份保留在标签中。已对episode2150、2174、2199各8个阶段画面抽检，接近、闭爪、抬起搬运与终点保持的大体顺序吻合；这不是逐帧接触真值验证。抽检图保存在同目录 `visual_qc.jpg`，按原视频帧号抽取，视频30FPS与训练元数据15FPS不可直接混用秒数。

## 实际损失口径

- FM按每个未来动作的真实时刻、左右臂分别生成 `[B,H,D]` 绝对权重；由真实特征映射推导动作通道，不把左臂权重乘到右臂。
- 最后接近与闭爪5；精细对准、提起2；普通伸手、搬运、保持1。左右手规则相同，后续可独立覆盖。
- 所有新阶段分支使用 `sum(weight * valid * loss) / sum(weight * valid)`。旧标量权重不再重复叠乘；旧模式关闭阶段功能时保持原行为。
- 动作padding同时从分子和分母排除，未使用关节也排除。真实记录的末尾保持动作继续监督。
- 当前Depth/Video按当前图像时间，Future Depth/Video按其真实未来目标时间；同一时间左右手冲突取max，不相加。每个分支分别归一化。
- 当前视频teacher提供单个未来端点特征，默认 `future_video_rule=target_endpoint`，不能把它描述为逐帧预测整个未来视频；显式支持的 `endpoint_max` 需另建策略快照。
- H=50时，2450/6129个锚点的未来图像端点越界，相应Future分支权重0。动作chunk共有61250个填充时间步。两种统计单位不同。
- 新模式下旧复习任务的重要性折入绝对权重一次，仍保留其相对重要性；由于本轮使用权重和归一化，不承诺与旧样本数归一化的绝对梯度尺度相同。
- 正则项不改，采样不额外加倍；阶段模式拒绝旧目标分组采样倍率，避免与本次实验混淆。

原始帧FM权重1/2/5计数：左3351/1434/1344，右3375/1402/1352。5倍合计约22%，其中每个语义子阶段均小于10%；不能把所有高权重帧合计误解为一个阶段。

## 使用与调参

首次标注：

```bash
python tools/label_dual_bottle_stages.py --dataset-root DATASET --output-dir LABELS
```

修改一份policy副本后冻结，不重新标注、不覆盖旧产物：

```bash
python tools/snapshot_stage_policy.py --labels-dir LABELS --weights NEW_WEIGHTS.json --output-dir NEW_SNAPSHOT
```

训练配方通过既有 `LINGBOT_SAMPLE_WEIGHT_CONFIG` 指向新快照的 `weight_config.json`，prepare会冻结标签和策略依赖。此处命令只准备产物，不启动训练。`stage_loss_weights.json` 中 `fm` 与 `visual` 是完整主阶段权重表，`visual_branches` 可单独指定四个视觉分支；`overrides` 支持分支→left/right→`stage/substage` 覆盖，未知标签、非正权重和unknown高权重会报错。

## 验证边界

远端CPU选择/损失测试97项、配方测试77项，共174项通过。测试包含权重和归一化的数值与梯度、padding零梯度、实际policy方法与深度helper的独立分支路由、特征通道、混合旧任务字段和冻结指纹。

真实数据CPU小批次通过：100条轨迹（双瓶50＋夹碗50）、21738帧、batch `[2,50,55]`、50条双瓶末尾检查，权重1/2/5与未来无效权重0。报告 `model_initialized=false, optimizer_steps=0, gpu_tested=false`。该小批次报告未实际命中带旧标量加权的混合pair，旧加权pair目前由契约测试覆盖，不把它计为真实混合验证通过。

无卡容器内存上限2GiB；GBS56首批装配未完成，没有明确OOM traceback，不能据此判断GPU训练容量。正式GBS56没有更改。GPU与闭环效果仍待后续授权实验。

回滚：下一轮不指定阶段配置，使用旧权重配置；不热改运行中的冻结文件。本次没有产生训练存档，也没有新增累计训练步数。

## 直接调整17个子阶段

`stage_loss_weights.json` 中原来的10项是9个主阶段加unknown，仅为后备默认值，不是17类子阶段。现在 `overrides.fm.left/right` 与 `overrides.visual.left/right` 均明确列出全部17项，可直接修改相应数字。FM是动作，visual是四个视觉分支的共同子阶段权重；左右手分别配置。同一项如果存在分支专用override，该专用值优先。

例如 `overrides.fm.left["reach_aim/reach_far"] = 2` 仅调左手远距离伸手动作；再把 `overrides.visual.left` 同名项改为2，可同时调该阶段视觉权重。现有17项显式override优先于顶层主阶段默认值，所以要改实际训练权重，应修改这些子阶段项。生成新快照后再训练，不能直接热改冻结文件。配置内部 `stage_policy` 字段暂保留作为旧版读取接口，其值已改为 `stage_loss_weights.json`，不代表模型policy。
