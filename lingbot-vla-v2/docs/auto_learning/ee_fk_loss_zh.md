# 低噪 FM 与末端 FK 一致性监督

**交付范围：** 本次入库的是独立 FP32 FK 计算模块、关节预检工具及 `tests/ee_fk/` 测试归档。下文的模型配置、训练参数和接线属于本地候选实现，因 bf16 兼容/几何精度问题尚未提交；仅检出本次提交不能按下列参数开启 FK 训练。低噪 FM 是分支已有功能。

这项功能用于让 Flow Matching 在接近专家动作的低噪声区间，额外关注动作经过机器人正向运动学（FK）后得到的末端状态。默认关闭，不改变现有训练和 Auto Learning 判定。

## 架构

训练链路为：

```text
归一化动作
    ├─ FM 主损失：预测速度 v_t 与目标速度 u_t
    └─ 低噪声区间：x_hat = x_t - t * v_t
                     ↓
                反归一化
                     ↓
          左右臂 6 关节 FK（JSON chain spec）
                     ↓
       位置 loss + 姿态 loss + 夹爪 loss
```

不增加 prediction head，也不修改 AL 的任务状态、hardness 扫描或采样概率。FK 监督只在 `t <= ee_fk_time_max` 时计算，GT 动作在 FK 前 detach；预测动作保持可微，梯度回到动作输出层。

低噪 FM 开启后，训练时间采样仍保留原有 Beta 分布，并以 25% 的概率替换为 `[0.001, 0.2]` 的均匀时间。关闭时直接返回原时间张量，不额外消耗 RNG。评测显式传入时间时不会被改写。

## 参数

训练参数默认全部关闭：

```text
--low_noise_fm_enabled true
--ee_fk_loss_enabled true
--ee_fk_spec_path /path/to/verified_robot_chain.json
--ee_fk_position_loss_weight 1.0
--ee_fk_orientation_loss_weight 0.1
--ee_fk_gripper_loss_weight 0.1
--ee_fk_time_max 0.2
--ee_fk_huber_delta 0.01
```

约束如下：

- `ee_fk_loss_enabled=true` 必须同时开启 `low_noise_fm_enabled`。
- 必须提供已核对的 JSON chain spec；spec 会嵌入模型 config，HF 存档不依赖原始路径仍可读取。
- 每条左右链必须正好包含 6 个、名称唯一的关节。
- 三个权重均不可为负，且至少一个大于 0。
- `ee_fk_time_max` 位于 `(0, 1]`。

## JSON chain spec

文件只描述 FK 所需的几何链，不猜测 URDF 或关节顺序。格式如下，数值必须从当前 RoboTwin 资产核对得到：

```json
{
  "left": {
    "root": {"translation": [0, 0, 0], "rpy": [0, 0, 0]},
    "joints": [
      {
        "name": "<RoboTwin left joint name>",
        "origin_xyz": [0, 0, 0],
        "origin_rpy": [0, 0, 0],
        "axis": [0, 0, 1]
      }
    ],
    "tip_xyz": [0, 0, 0],
    "tip_rpy": [0, 0, 0]
  },
  "right": {"...": "与 left 相同的结构"}
}
```

单位是米和弧度。关节顺序必须与 `left_arm_joints_name` / `right_arm_joints_name` 以及动作数据顺序一致。`tip_xyz` 必须包含 RoboTwin 实际末端口径中的 TCP 偏移；不能只把 URDF 最后 link 当作训练目标。若 RoboTwin 的 `get_left_ee_pose()` / `get_right_ee_pose()` 还叠加了 `global_trans_matrix`、`delta_matrix`、`gripper_bias` 或固定偏移，这些必须折入 spec 或在对拍时显式处理。

## 动作布局与归一化

训练启动时从真实 `FeatureTransform` 读取 feature 名称、槽位宽度和归一化统计，不手写偏移。当前 RoboTwin 配置通常是：

```text
action.arm.position      = [left six, right six]，占 14 槽中的前 12 个值
action.end.position      = 14 槽（不参与 FK）
action.effector.position = [left gripper, right gripper]，2 个值
```

因此常见 packed offset 是 arm `[0:14]`、end `[14:28]`、effector `[28:30]`，但代码以 feature 名称和真实 metadata 为准。多数据集训练要求布局和归一化统计完全一致，否则启动即报错，避免把一个数据集的动作解释成另一个数据集的关节。

## FK 正确性验收

测试意图、方法、命令和定位说明统一在 [FK 功能测试目录](../../tests/ee_fk/README.md)；实测结果见 [2026-10-17 CPU 验证记录](../../tests/ee_fk/VALIDATION_2026-10-17.md)：FP32 几何对拍通过，bf16 buffer 类型冲突与几何精度问题尚未修复，不能开启 FK 正式训练。历史文件记录的末端位姿与控制动作 FK 也需区分，不能直接互当 GT。

先做无 GPU 的关节名称预检，确认 JSON 中的左右臂六轴顺序和 RoboTwin 实际配置完全一致：

```bash
python tools/auto_learning_pipeline/ee_fk_spec_check.py \
  --spec /path/to/verified_robot_chain.json \
  --embodiment-config /path/to/assets/embodiments/aloha-agilex/config.yml
```

该命令只核对名称和顺序，返回非零表示不能继续；它不替代几何对拍。

在 GPU 训练前必须用真实资产和真实轨迹做对拍：

1. 从轨迹取归一化后的 `action.arm.position`，按启动日志记录的 normalization 反归一化。
2. 用 JSON spec 做左右臂 FK。
3. 在同一帧、同一关节值下读取 RoboTwin/SAPIEN 的 `get_left_ee_pose()` 和 `get_right_ee_pose()`。
4. 比较位置、姿态和左右臂对应关系；覆盖多个任务、多个回合和随机帧，而不是只看一帧。
5. 先确认坐标系、TCP 偏移和姿态表示一致，再记录位置误差（米）和姿态角误差（弧度）的最大值、P95 与均值。

验收不通过时禁止开启 `ee_fk_loss_enabled`；可以先只开启低噪 FM。当前仓库的 `tests/ee_fk/test_ee_fk.py` 只验证可微计算、梯度和布局解析，不能替代 RoboTwin 资产对拍。

## 运行期观测与回滚

开启后训练日志会打印 `[ee_fk] enabled`，包含 packed layout、归一化字段、时间门控、三项权重和左右关节名；TensorBoard 记录：

```text
ee_fk/position_loss
ee_fk/orientation_loss
ee_fk/gripper_loss
ee_fk/weighted_loss
```

快速回滚是在新进程中关闭 `--ee_fk_loss_enabled`；若仍需保留低噪 FM，则只关闭 FK 开关。要完全回到原训练时间分布，同时关闭 `--low_noise_fm_enabled`。这些开关在进程启动时读取，不能热切换当前训练。
