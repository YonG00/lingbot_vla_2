# 订书机按放置早晚与白色型号赋权

2026-10-23：最终配方是早期15／晚期20／白色其他10。已在服务器完成数据及CPU检查，当前无卡，未启动GPU训练。此前“全放置20＋白色15”的产物仅作过程记录，已被本方案替代，不要启动旧配方。

## 范围与优先级

不是17阶段体系。本次给原始episode/frame样本分类，重叠取最大值、不叠乘，采样概率不变。

|类别|定义|倍率|有效帧数|总7731帧占比|
|---|---|---:|---:|---:|
|晚期放置|预放置稳定结束后，最终下降、松爪至轨迹结束|20|1748|22.61%|
|早期放置|确认提起后搬运，至pad上方预放置稳定结束|15|1847|23.89%|
|白色其他|base3白色示范中，不属于以上放置范围的帧|10|763|9.87%|
|其余|其他型号的非放置帧|1|3373|43.63%|

宽参考3595帧是“抬起后→结束”，包括搬运，所以不应全称为pad精调。本次3595拆为1847＋1748。白色9条、1458帧，其中695与放置重叠，服从早晚放置倍率。有效高权重坐标的并集仍是原先4358帧，没有扩充采样范围。

早晚边界来自已核验50条专家轨迹的FK/夹爪代理事件：持物时在释放目标上方形成稳定预放置（目标距离8–12cm、高度差至少6cm、低速且闭爪），最后一个预放置稳定帧 `preplace_end` 归早期，从 `preplace_end+1` 归晚期。没有按时间比例平分，也不声称它是实时接触或物体姿态真值。完整边界在summary.json的placement_boundaries。晚期包含300个真实记录的末尾保持帧，不把这些误当不存在的padding。

## 损失与代码兼容

动作FM、Depth、Future Depth、Video统一使用样本类别倍率，按各自有效权重和归一化。倍率由当前锚点样本类别决定，FM作用于其整个有效动作chunk，不按未来每个动作时刻重新查阶段；视觉同样按这个样本倍率，未来目标越界为0。动作时间padding和未用关节不进入FM分子、分母。不引入EE-loss、窗口采样或额外采样倍率。

SelectionWeights现在显式支持不带stage_table的weight_sum标量配置，复用已验证的绝对权重矩阵和辅助分支归一化路径；没有给订书机伪造阶段标签。旧sample_count配置在未启用归一化模式时保持原行为。复合注册表中任一任务启用weight_sum，整个batch走明确的权重和路径，旧任务相对重要性只折入一次。

CPU检查修正：来源注册表保留双瓶阶段数据但当前只训练订书机时，仅检查当前数据实际包含的阶段episode，不要求读取未参与本轮的双瓶样本；这不删除源注册表。

## 可调配置与训练入口

最终数据：`refs/data_selection/move_stapler_pad_early15_late20_white10_v1/`，其中selected_frames.json明确列出逐帧loss_weight，weight_config.json带SHA校验，summary.json为范围和规则，cpu_verified.json/visual_qc.jpg为证据。修改后必须新建快照、更新SHA，不能热改运行文件。

最终配方：`configs/training_recipes/move_stapler_pad200_from_1913_early15_late20_white10_noreplay.json`。

- 来源1913-step200；主任务仅move_stapler_pad，50条7731帧。
- 计划200步、BF16、micro56/GAS1、无复习；仅最终200存档。
- 最终模型驻留、3回合并行带视频评测，种子100000/100001/100002；显式使用已有录像入口的官方对照仿真客户端。
- 不补课，不自动关机，省略额外GPU三步探测；CPU预检保留。

在项目根目录并使用训练环境Python：

```bash
LINGBOT_SAMPLE_WEIGHT_CONFIG="$PWD/refs/data_selection/move_stapler_pad_early15_late20_white10_v1/weight_config.json" \
LINGBOT_ROBOTWIN_ROOT=/data/experiments/official_adjust_bottle_control/RoboTwin-lingbot \
python experiment/task_recipes/run.py \
  --recipe configs/training_recipes/move_stapler_pad200_from_1913_early15_late20_white10_noreplay.json \
  --execute
```

上面是开卡后的启动命令，本次未执行。服务器另备 `/tmp/launch_stapler_final.py`，会先检查GPU和文件SHA。重生成规则：`python tools/prepare_stapler_sample_weights.py --split-placement --output-dir NEW_DIR`。

## 验证边界

107项数据/损失测试＋77项配方测试共184项CPU通过。真实小批次检查覆盖50条7731帧，batch=[2,50,55]，四类1/10/15/20分别抽取实际样本验证FM与当前视觉权重，未来无效为0、padding为0；model_initialized=false、optimizer_steps=0、gpu_tested=false。无卡2GiB容器只用micro2验证接口，不改变正式micro56。

三条示范150/159/184的12张早期、预放置边界、晚期下降、松爪图抽检通过，含白色型号。图像按原视频帧号抽取，原视频与训练元数据FPS不同，不能按训练秒数直接seek。

最初临时CPU脚本误传str而非Path已修正；旧全放置20版本的CPU报告不是最终版本。最终GPU显存、训练速度、成功率均未测，不从CPU通过推断训练成功。没有新增Checkpoint或累计训练步数。
