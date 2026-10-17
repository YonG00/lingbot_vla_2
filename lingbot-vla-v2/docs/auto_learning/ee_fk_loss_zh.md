# 低噪 FM 与末端 FK 一致性监督

**验收范围：** 包含模型/训练接线、FK 局部 FP32 精度隔离、独立 val 末端误差、TensorBoard 和仪表盘。CPU 结果见 [运行期验证记录](../../tests/ee_fk/VALIDATION_RUNTIME_2026-10-17.md)；真实大模型的 CUDA/FSDP 前反向、周期 HF 保存/重载与CUDA数值检查均已通过 [96G实机验收](../../tests/ee_fk/VALIDATION_GPU_2026-10-17.md)。500步对照已启动，闭环效果尚未验证。历史 bf16 故障证据保留在原测试记录中，不代表修复后的运行期路径。

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

不增加 prediction head，也不修改 AL 的任务状态、hardness 扫描或采样概率。FK 监督只在 `t <= ee_fk_time_max` 的有效帧计入 loss，GT 动作在 FK 前 detach；预测动作保持可微，梯度回到动作输出层。

模型主体和推理保留 bf16。运行期 `EEFKRuntime` 是独立于模型模块树的辅助对象，由 config 中的原始 spec 构造并按设备缓存 FP32 solver；整体模型 dtype 转换和 HF 权重加载不会舍入几何常量。动作恢复、反归一化和 FK loss 局部禁用 autocast，梯度仍回到模型。solver 不加入模型权重，spec 随 config 保存；不能用 solver 先 bf16 再 `.float()` 代替。

低噪 FM 开启后，训练时间采样仍保留原有 Beta 分布，并以 25% 的概率替换为 `[0.001, 0.2]` 的均匀时间。关闭时直接返回原时间张量，不额外消耗 RNG。评测显式传入时间时不会被改写。

## 参数

训练参数默认全部关闭：

```text
--low_noise_fm_enabled true
--ee_fk_loss_enabled true
--ee_fk_eval_enabled true
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

测试意图、方法、命令和定位说明统一在 [FK 功能测试目录](../../tests/ee_fk/README.md)。[首次验证](../../tests/ee_fk/VALIDATION_2026-10-17.md)发现了 bf16 buffer 问题；[运行期验证](../../tests/ee_fk/VALIDATION_RUNTIME_2026-10-17.md)检查隔离后的 FP32 几何及观测链路。历史文件记录的末端位姿与控制动作 FK 需区分，不能直接互当 GT。

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

## 独立 val 的末端误差

沿用现有 active_val 的 4 条独立验证轨迹和 train_monitor 的 4 条训练轨迹，不拆分验证集。每个 chunk 复用现有完整去噪预测和同一次 `FeatureTransform.unapply` 的物理 GT，不额外运行模型；按真实 `action_is_pad` 排除尾部补齐帧。原 GMSE 的聚合、尾部口径、阈值表和 PASS 判断保持原状。

每条轨迹先计算误差，再等权汇总；左右臂分别记录，同时给出两臂平均：

- 位置：三维欧氏距离平方的均值，开方后显示 RMSE，单位 mm；不是逐坐标 MSE 的开方。
- 姿态：两旋转矩阵的测地夹角均值，单位度。
- 夹爪：原物理动作数值的 MAE，沿用数据单位，不解释为牛顿或接触力。

这里比较 `FK(预测动作)` 与 `FK(GT动作)`，不使用历史 HDF5 的记录末端位姿充当 GT。指标衡量动作的运动学误差，闭环成功率仍须实测。

`ee_fk_eval_enabled=true` 可以单独开启诊断而不开训练 loss；开启 FK loss 也会自动计算末端指标。TensorBoard 的 AL 路径为：

```text
task/<task>/active_val_ee_position_rmse_mm
task/<task>/active_val_ee_orientation_mean_deg
task/<task>/active_val_ee_gripper_mae
task/<task>/active_val_ee_left_position_rmse_mm
task/<task>/active_val_ee_right_position_rmse_mm
```

训练集 `train_monitor` split在事件/TB中的kind名为 `active_train`；`open_loop/{train|val}_ee/*` 对应通用开环验证入口。AL 事件流与 eval_history 同步保留指标，仪表盘紧凑视图和 `--wide` 均显示末端 val 及测量步号，旧 run 无 EE 指标时不显示。

## 本次 step9680 对照配置

从原始 step9680 新建优化器和步号，stand 任务新增 500 步；micro=24、GAS=1、每100步独立评测、每250步存 BF16 HF（共250和500两份）、不存 DCP。沿用低噪 FM 与 `target_change_window`，新增 FK 三项权重先用 `1.0/0.1/0.1`；后续根据三项实际量级判断，不保证这些权重最优。保留上一轮冻结视觉编码器的设置，使本次比较聚焦 EE 监督；这不是解冻视觉塔的全参数训练。

CPU 预览命令（默认不创建 run、不训练；开卡验收后才能加 `--execute`，该完整启动器前台运行，正式启动需外层后台托管）：

```bash
cd /data/experiments/learn_skill_mix_task/lingbot-vla-v2
PYTHONPATH="$PWD" /data/miniconda3/envs/lingbotvla/bin/python \
  experiment/robotwin/launch_skill_mix_task.py \
  --model /data/outputs/al_v79_poolfix_7300/checkpoints/global_step_9680/hf_ckpt \
  --train-config /data/outputs/stand_targetchange_lnfm_9680_add500_20261017_r2/lingbotvla_cli.yaml \
  --dataset-root /data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30 \
  --qwen3vl /data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct \
  --audit-report /data/al_cache/stand_target_change_v1/data_audit.json \
  --al-config /data/al_cache/stand_target_change_v1/stand_al_config.json \
  --learning-tasks place_object_stand \
  --out /data/outputs/stand_ee_lnfm_9680_add500 \
  --steps 500 --micro 24 --gas 1 --ee-fk-loss --ee-fk-eval
```

便捷续练入口 `continue_skill_training.py` 也支持上述 EE 开关及 `--ee-position-weight`、`--ee-orientation-weight`、`--ee-gripper-weight`；使用便捷入口的 `--shutdown` 可在正常训练结束且最终 HF 核验成功后关机。本次验收通过后启动 `al_stand_ee_lnfm_9680_add500_20261017_184252`，未附带关机动作，实际命令归档在GPU验证记录的evidence目录。
