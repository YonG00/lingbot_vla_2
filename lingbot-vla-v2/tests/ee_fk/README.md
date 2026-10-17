# FK 功能测试归档

这里集中保存 FK 一致性监督的测试入口、测试意图和历史验证证据。后续新增 FK 相关测试、资产对拍脚本和精度回归，继续放在本目录；通用低噪 FM 测试仍位于 `../test_low_noise_fm.py`，作为关联回归执行。

本次归档提交包括独立 FP32 FK 计算与预检，不包括尚有 bf16 问题的模型/训练接线。历史诊断保留了对该候选接线的风险证据；运行本目录的单测无需加载或开启完整模型。

```text
tests/ee_fk/
├── README.md                   测试意图、方法、命令与故障定位
├── test_ee_fk.py                自动单测
├── diagnostics/                反归一化、梯度、资产对拍、历史记录和精度探针
├── evidence/2026-10-17/         原始 spec、reference、日志及 JSON 结果
└── VALIDATION_2026-10-17.md     当次验证记录
```

## 测试分层

### 1. 纯计算单元测试

文件：`test_ee_fk.py`

目的：不依赖 RoboTwin 或模型资产，验证 FK 矩阵、batch shape、位置/姿态/夹爪 loss、动作布局解析、归一化前后动作和非法几何拒绝逻辑。

命令：

```bash
PYTHONPATH=. pytest tests/ee_fk/test_ee_fk.py tests/test_low_noise_fm.py -q
```

### 2. 关节名称和顺序预检

文件：`../../tools/auto_learning_pipeline/ee_fk_spec_check.py`

目的：逐名比较 FK JSON 与 RoboTwin `config.yml` 的左右六轴，避免把动作维度或腕部顺序猜错。

```bash
python tools/auto_learning_pipeline/ee_fk_spec_check.py \
  --spec /path/to/verified_robot_chain.json \
  --embodiment-config /path/to/assets/embodiments/aloha-agilex/config.yml
```

### 3. 真实资产几何对拍

验证脚本归档在 `diagnostics/`，此次整理只调整路径配置和单测导入路径，计算方法保持原样：

- `lingbot_fk_asset_export.py`：从 RoboTwin URDF/config 提取候选链，并用 CPU SAPIEN 生成参考末端位姿。
- `lingbot_fk_asset_compare.py`：比较候选 PyTorch FK 与 SAPIEN/RoboTwin 末端接口。
- `lingbot_fk_precision_probe.py`：检查几何参数被 bf16 截断后的误差。
- `lingbot_fk_recorded_pose_probe.py` / `lingbot_fk_recorded_pose_compare.py`：比较历史 HDF5 中的状态关节和记录末端位姿。

目的：确认坐标系、root pose、joint origin/axis、RoboTwin 的 `global_trans_matrix`、`delta_matrix`、TCP 偏移和左右臂对应关系。它验证的是几何，不模拟控制器和接触。

### 4. 历史证据

完整结果：[FK 无卡验证记录](VALIDATION_2026-10-17.md)。原始候选 spec、SAPIEN/HDF5 参考数组、JSON 报告与日志集中在 [evidence/2026-10-17/](evidence/2026-10-17/)。候选 spec 仅代表本次资产验证，尚未验收为正式训练配置。

## 各项测试的意图、方法和定位线索

| 测试 | 为什么测 / 怎么测 | 异常时先查 |
|---|---|---|
| 基础 FK | 简化机械臂的末端距轴 1 m，旋转 90° 后应落到指定位置；核对 batch shape、动作恢复公式及错误 spec 拒绝 | 旋转顺序、轴、单位、shape |
| 动作布局 | 用 arm 14 槽、end 14 槽、gripper 2 槽的 metadata，核对左六轴、右六轴与夹爪偏移 | padding 与 feature 的真实维数；不能直接把原始 14 维动作当作 packed 输入 |
| 梯度 / padding | 构造预测与 GT 的差异，反传后检查预测梯度有限、GT 无梯度、屏蔽帧梯度为零；覆盖三种归一化模式 | detach、in-place 操作、mask 广播、梯度是否被截断 |
| 反归一化 | 相同数值分别交给新增 FK 代码和真实 Normalizer，使用仓库统计表比较关节角 | 统计表、horizon 截取、bounds/meanstd 公式、是否为相对动作 |
| 仿真几何 | 100 组随机姿态 + 两任务各 50 帧，同一关节值分别交给 PyTorch FK 和 CPU SAPIEN；比较真实 RoboTwin accessor 返回的位置和姿态 | 关节名/顺序、URDF fixed joints、root pose、joint frame 与 link frame、TCP 修正 |
| bf16 类型兼容 | 将 solver 转 bf16，调用现有反归一化及 loss；检查是否类型冲突 | 模型 loader 整体 `.to(dtype)` 是否误降 FK 几何常量精度 |
| bf16 精度损失 | 同一原始 spec 分别直接 FP32 计算，以及先 bf16 再 FP32 计算，比较 200 组末端结果 | 原始几何是否已被舍入；后补 `.float()` 不能恢复精度 |
| 历史记录口径 | 对 HDF5 的状态关节做 FK，与文件记录的同帧末端姿态比较 | 控制目标 / 实测角、时间对齐、采集版本及数据转换；不要直接判定 FK 错误 |

## 无卡运行方法

从包目录 `lingbot-vla-v2/` 执行。服务器长任务使用 `setsid nohup` 后台运行、另起命令查看日志。以下是各步骤的核心命令；无需加载模型。

```bash
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export FK_VALIDATION_DIR="$(mktemp -d /tmp/ee_fk_validation.XXXXXX)"
export FK_ROBOTWIN_ROOT=/data/code/RoboTwin-lingbot
export FK_DATASET_ROOT=/data/robotwin_data/demo_clean
FK_PY=/data/miniconda3/envs/lingbotvla/bin/python
FK_SIM_PY=/data/miniconda3/envs/RoboTwin/bin/python

# 1. 单测（5 项 FK + 3 项低噪 FM）
"$FK_PY" -m pytest tests/ee_fk/test_ee_fk.py tests/test_low_noise_fm.py -q

# 2. 反归一化、梯度与类型兼容探针
"$FK_PY" tests/ee_fk/diagnostics/lingbot_fk_extra_cpu_checks.py

# 3. 真实资产提取与 CPU SAPIEN 参考，再用 PyTorch 对拍
"$FK_SIM_PY" tests/ee_fk/diagnostics/lingbot_fk_asset_export.py
"$FK_PY" tests/ee_fk/diagnostics/lingbot_fk_asset_compare.py

# 4. 历史 HDF5 记录口径检查
"$FK_SIM_PY" tests/ee_fk/diagnostics/lingbot_fk_recorded_pose_probe.py
"$FK_PY" tests/ee_fk/diagnostics/lingbot_fk_recorded_pose_compare.py

# 5. bf16 几何舍入诊断
"$FK_PY" tests/ee_fk/diagnostics/lingbot_fk_precision_probe.py
```

`FK_VALIDATION_DIR` 保存本轮生成的 spec、临时无渲染 URDF、reference 和报告。使用新临时目录，避免覆盖历史归档；仅生成 reference 时读取原始轨迹，不读取图片、不修改源资产。

`extra_cpu_checks` 是打印诊断结果的探针，会捕获并打印已知 bf16 异常；它退出 0 **不代表所有检查通过**。同样，几何/历史记录对比要读取 JSON 的 `passed`，精度探针要读取实际误差，不能只凭进程退出判断。此次历史记录检查是未通过状态，不能当作通过证据。

没有原始数据和 SAPIEN 时，可将 `evidence/2026-10-17/` 中的 `candidate_spec.json`、`sapien_reference.npz`、`recorded_reference.npz` 复制到新的 `FK_VALIDATION_DIR`，只运行对比和精度脚本。这验证对固定 reference 的数值一致性，不能替代当前资产重新导出。

后台执行示例：

```bash
setsid nohup env CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD" \
  OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  "$FK_PY" -m pytest tests/ee_fk/test_ee_fk.py tests/test_low_noise_fm.py -q \
  > "$FK_VALIDATION_DIR/unit_tests.log" 2>&1 < /dev/null &
# 在下一次独立调用中查看：
cat "$FK_VALIDATION_DIR/unit_tests.log"
```

## 当前验收状态

- 纯 CPU FK/低噪回归：8 passed。
- 真实 SAPIEN 几何对拍：200 组通过。
- FK 训练开启：暂不通过。模型主体可以保持 bf16，但 FK 固定几何必须保持 FP32，并在 FK loss 局部使用 FP32 计算；当前代码还需补这一生命周期契约。
- 历史 HDF5 的状态关节与记录末端位姿有差异，是否来自时序或控制目标仍待确认，不能直接当作 FK 几何失败。

## 为什么 FK 局部使用 FP32

模型的视觉编码器、语言模型、动作专家和推理流程仍使用 bf16。FK 只是训练期间的一个辅助 loss，不需要把模型改成 FP32。

FK 使用机器人固定几何参数：连杆长度、关节 origin、旋转轴、TCP 偏移，以及反归一化后的关节角。它包含多次三角函数和 4x4 矩阵连乘。bf16 有 7 位二进制小数位（计入隐含首位后 8 位有效精度），FP32 有 23 位二进制小数位。这里的“位”不是十进制小数位；bf16 与 FP32 的数值范围相近，但 bf16 对相近数值的区分更粗。

例如，在 `[0.5, 1)` 米区间，bf16 相邻可表示数间隔为 `0.00390625 m`，即约 3.91 mm。这是格式分辨率示例，不是每次 FK 的实际误差。几何常量被舍入后，六轴的组合旋转和平移还会改变误差传播；本次实测的 bf16 后转 FP32 最大末端偏差为 2.27 mm。

正确口径是：

```text
模型前向（保持 bf16）
    -> 预测动作
    -> 动作反归一化（FP32）
    -> FK 固定参数 + 矩阵计算（FP32）
    -> FK loss（FP32）
    -> 梯度回传到 bf16 模型参数
```

FP32 只保护 FK 这条数值敏感的辅助分支；它不会把模型推理或训练主干改成 FP32，也不会改变部署时的 bf16 行为。这是拟修复的目标，当前精度隔离尚未实现。预测动作若已是 bf16，转 FP32 不能找回已舍掉的动作精度；目的只是避免 FK 再引入额外几何与计算误差，不能保证闭环抓取提升。

## 修复验收条件

修复后必须补充：

1. 模型整体转 bf16 后，FK 固定几何仍保有原始 FP32 精度（即使实现改为独立辅助对象，也必须满足）。
2. autocast 开启时，FK 内部仍为 FP32。
3. FK 输出与纯 FP32 reference 的误差回到 `1e-5 m / 1e-4 rad` 门槛内。
4. FK loss 对预测动作有有限梯度，GT 无梯度，padding mask 不回传梯度。
5. 关闭 FK 时旧训练路径和 RNG 行为不变。
