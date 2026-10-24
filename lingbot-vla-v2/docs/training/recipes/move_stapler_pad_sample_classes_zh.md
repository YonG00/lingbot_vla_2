# 订书机按放置早晚与白色型号赋权

2026-10-23：最终配方是早期15／晚期20／白色其他10，已按用户要求升级为统一逐动作类别加权。当前无卡，未启动GPU训练；完整CPU预检尚待通过，详见[统一框架](category_weighting_framework_zh.md)。此前“全放置20＋白色15”的产物仅作过程记录，已被本方案替代，不要启动旧配方。

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

动作FM对chunk内每个未来动作查其自身数据帧类别，非整chunk复制锚点权重；Depth/Video按自身目标图像时间查类别。动作padding不监督，越界未来图像0，按各分支有效权重和归一化。

两种标签共用CategoryTable和category_loss_weights.json；详细参数、重叠与迁移规则见[统一框架](category_weighting_framework_zh.md)。CPU仅审计本轮实际参与的任务episode，来源注册表其他任务重要性仍保留。

## 可调配置与训练入口

最终数据：`refs/data_selection/move_stapler_pad_categories_early15_late20_white10_v1/`，其中frame_classes.json保存类别，category_loss_weights.json直接调倍率，weight_config.json绑定SHA，coverage.json为覆盖统计，real_frame_clock_verified.json与full_cpu_check_incomplete.json记录验证边界。修改后必须新建快照、更新SHA，不能热改运行文件。

最终配方：`configs/training_recipes/move_stapler_pad200_from_1913_early15_late20_white10_noreplay.json`。

- 来源1913-step200；主任务仅move_stapler_pad，50条7731帧。
- 计划200步、BF16、micro56/GAS1、无复习；仅最终200存档。
- 最终模型驻留、3回合并行带视频评测，种子100000/100001/100002；显式使用已有录像入口的官方对照仿真客户端。
- 不补课，不自动关机，省略额外GPU三步探测；CPU预检保留。

在项目根目录并使用训练环境Python：

```bash
LINGBOT_SAMPLE_WEIGHT_CONFIG="$PWD/refs/data_selection/move_stapler_pad_categories_early15_late20_white10_v1/weight_config.json" \
LINGBOT_ROBOTWIN_ROOT=/data/experiments/official_adjust_bottle_control/RoboTwin-lingbot \
python experiment/task_recipes/run.py \
  --recipe configs/training_recipes/move_stapler_pad200_from_1913_early15_late20_white10_noreplay.json \
  --execute
```

上面是开卡后的启动命令，本次未执行。服务器另备 `/tmp/launch_stapler_final.py`，会先检查GPU和文件SHA。重生成规则：`python tools/prepare_stapler_sample_weights.py --split-placement --output-dir NEW_DIR`。

## 验证边界

195项CPU回归、真实7731帧的500个边界/时钟检查通过，含243个跨类别chunk。完整模型DataLoader预检在2GiB无卡容器返回-9，尚未完成，不能用此前整chunk版本的小批次通过代替。开卡后正式启动仍先跑完整CPU检查；GPU训练未执行。三示范早晚分界图仍适用，因为本次未改变标签范围，只改权重作用时钟。
