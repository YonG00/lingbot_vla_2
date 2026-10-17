# FK 无卡验证记录（2026-10-17）

> 本文保留首次验证的故障现场和提交边界。后续精度修复及 val/TensorBoard/仪表盘验证见 [运行期记录](VALIDATION_RUNTIME_2026-10-17.md)，不要将本文的“未修复”当作当前状态。

测试意图、方法、命令与故障定位见 [本功能测试说明](README.md)。本记录、诊断脚本和历史证据统一归档在 `tests/ee_fk/`。

提交范围说明：本次只提交独立 FP32 FK 模块与测试归档。报告中涉及的模型 dtype 转换和完整训练接线为本地候选代码，因 bf16 问题尚未提交；通过 CPU 测试不代表完整训练入口已交付。

结论：纯 FP32 FK 与当前 RoboTwin/SAPIEN 几何对拍通过；开启 FK 训练的验收尚未通过，发现 bf16 buffer 类型冲突及几何精度损失。未启动训练，未提交/push，未修改服务器生产目录与实验工作树。

## 验证环境与隔离

- 实验工作树基线 `learn_skill_mix_task`，HEAD `5d79e3ff81c498c71731dc2d57326671f2a9f238`，验证前干净。
- 本地未提交代码通过临时测试包复制到服务器 `/tmp/lingbot_fk_cpu_20j8mzne`，不属于 Git 同步或部署。
- 临时包 SHA256：`0cfbbe970ccac1ac967dfa08dacd5bd2c8e1a484dc5a1b44619ffc39c6aa5c9b`；随后补入基线 `lingbotvla/models/vla/lingbot_vla/utils.py`（第一次失败是测试包漏文件，非产品故障）。
- Python 3.12.14（训练环境）运行 CPU torch；仿真环境使用 Python 3.10 的 SAPIEN。所有计算进程设置 `CUDA_VISIBLE_DEVICES=`、`OMP_NUM_THREADS=1`，没有加载模型。
- SAPIEN 默认场景会建立渲染器，无卡时报 `failed to find a rendering device`；验证改用 `Scene([PhysxCpuSystem()])`，临时 URDF 只去掉 visual/collision，保留关节、几何变换与惯量。没有修改源资产。

## 已通过的检查

- `tests/test_ee_fk.py`（原运行位置；现为 `tests/ee_fk/test_ee_fk.py`，5 项）及 `tests/test_low_noise_fm.py`（3 项）：**8 passed / 5.19 s**。
- 额外 CPU 探针覆盖 identity、bounds_99_woclip、meanstd：预测梯度有限，GT 无梯度，屏蔽帧梯度为零。
- 对仓库真实 `Normalizer.unnormalize` 与 `assets/norm_stats/robotwin.json`：arm 最大差 `2.00143e-7`，effector 最大差 `4.82071e-8`（原始物理数值单位）。
- 关节顺序逐名核对通过：左臂 `fl_joint1..6`、右臂 `fr_joint1..6`。模型 arm packed 前 12 维依次为左六轴、右六轴，原始数据的两个夹爪维度单独提取。
- 已从真实 URDF 生成候选 chain spec，核对 fixed joints、joint origin、axis、root pose 和 RoboTwin 末端修正。SAPIEN joint frame 与 child link frame 有固定旋转；需合并 `pose_in_child`、`global_trans_matrix`、`delta_matrix` 和 `gripper_bias - 0.12`，不可直接用末端 link 代替。

## 当前仿真几何对拍

共 200 组关节姿态：100 组固定随机种子姿态，加 stand、cabinet 各 5 条轨迹、每条 10 帧的动作关节值。将同一组关节角设置为 SAPIEN articulation qpos 后，执行当前 `Robot.get_left_ee_pose`、`get_right_ee_pose` 与 `_trans_endpose` 的实际函数体，与 PyTorch FK 比较。这里验证的是**相同关节角的几何**，没有模拟控制器或接触。

| 臂 | 位置最大误差（m） | 位置 P95（m） | 姿态最大误差（rad） |
|---|---|---|---|
| 左 | `6.42135e-7` | `4.77768e-7` | `8.61308e-7` |
| 右 | `4.96960e-7` | `3.11524e-7` | `8.83176e-7` |

本次几何门槛为位置最大值 `<1e-5 m`、姿态最大值 `<1e-4 rad`，均通过。真实 chain loss 的梯度有限、非零，padding 帧梯度为零。

## 待解决的问题

1. **bf16 类型冲突，可 CPU 复现。** 对 solver 调用 `.bfloat16()` 后，`compute_ee_fk_losses` 将动作反归一化为 FP32，而几何 buffers 仍为 bf16，矩阵乘法报 `RuntimeError: expected scalar type BFloat16 but found Float`。仓库模型 loader 的 VLA 分支会执行 `.to(torch_dtype)`，因此此路径与正式启动相关。
2. **仅计算前 `.float()` 不足。** 真实 spec 先 `.bfloat16()` 再 `.float()`，200 组姿态相对原 FP32 spec 最大位置偏差左 `0.00227287 m`、右 `0.00225972 m`，最大角偏差分别 `0.00223938/0.00224230 rad`。修复需保住几何原始精度，并保证 FK 计算不被训练 autocast 降精度；还需覆盖 loader、buffer 转设备及存档恢复。
3. **历史数据记录口径尚未完全确认。** 对同一批原始 HDF5 的 `state/joint_states` 与 `state/*_ee_poses` 抽样 100 帧，左臂位置最大差 `0.0364630 m`，右臂 `0.0271338 m`；姿态最大差 `0.0998519/0.0357357 rad`。当前仓库采集代码的 `get_*_arm_jointState()` 读 `get_drive_target()`，真实角度另有 `get_*_arm_real_jointState()`；但不能仅凭现行代码断定历史 HDF5 的采集实现。仍需核对控制目标/实测角、保存时序与转换过程，不能把此差异直接定性为 FK 几何错误。

FK loss 设计的 GT 是 **FK(GT 动作)**，不是原始 HDF5 中记录的实际末端位姿。它监督控制指令的目标末端几何，无法保证受力、控制跟踪和闭环定位；这些最终必须用 GPU 闭环评测验证。

## 证据与复核入口

以下是原运行的临时路径。候选 spec、两个 reference 数组、三个 JSON 报告及日志现已归档在 [evidence/2026-10-17/](evidence/2026-10-17/)，诊断脚本见 [diagnostics/](diagnostics/)。临时路径不保证跨实例保留；后续复核请按 [README](README.md) 中的新功能目录命令运行。

- `/tmp/lingbot_fk_cpu_test_r2.log`：8 项 CPU 回归。
- `/tmp/lingbot_fk_extra_cpu_checks.log`：归一化、梯度、bf16 异常。
- `/tmp/lingbot_fk_asset_validation/`：候选 spec、SAPIEN reference、几何对拍、历史数据对比及 precision report。
- `/tmp/lingbot_fk_asset_export.py`、`lingbot_fk_asset_compare.py`、`lingbot_fk_recorded_pose_probe.py`、`lingbot_fk_recorded_pose_compare.py`、`lingbot_fk_precision_probe.py`：本次探针脚本。

原运行单测命令（历史位置，仅作证据；当前入口为 `tests/ee_fk/test_ee_fk.py`）：

```bash
cd /tmp/lingbot_fk_cpu_20j8mzne
CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD" OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /data/miniconda3/envs/lingbotvla/bin/python -m pytest \
  tests/test_ee_fk.py tests/test_low_noise_fm.py -q
```

下一步：修正 FK 精度生命周期并补 CPU 契约回归，核对历史数据口径，再开卡验证模型加载、前向/反向、HF 存档与闭环效果。当前保持 `ee_fk_loss_enabled=false`。

## 按功能归档后的入口复核

单测现位于 `tests/ee_fk/test_ee_fk.py`；6 个诊断脚本归入 `diagnostics/`，路径统一由 `_paths.py` 配置，计算方法未改。历史 reference 和报告归入 `evidence/2026-10-17/`，新输出放独立临时目录。

在服务器原隔离测试目录中复核：**8 passed / 9.18 s**；归一化/梯度探针输出保持原语义；用归档 reference 重跑 `asset_compare`、`recorded_pose_compare`、`precision_probe`，三个生成报告与归档 JSON **逐字段完全一致**。这验证了目录迁移和路径调整，不是宣称 bf16 或历史数据差异已修复。原始日志见 [archive_check.log.txt](evidence/2026-10-17/archive_check.log.txt)。生产和实验工作树未改变。
