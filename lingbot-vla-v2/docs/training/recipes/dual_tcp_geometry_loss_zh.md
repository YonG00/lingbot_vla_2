# 双臂绝对TCP与动态相对位姿Loss

参考用户提供的dual_arm_relative_pose_loss_design_zh.md，实现两个独立辅助目标；这是代码/CPU验证状态，不代表双瓶已学会。默认关闭；2026-10-24已在实验服务器完成17项CPU测试、真实50条/6129帧缓存生成及真实DataLoader检查，并按用户授权启动1050-step200续训200步。GPU首步及闭环结果待核验，不能将启动当作完成。

## 两个目标

- absolute：分别比较左右TCP世界位置（米，SmoothL1 beta=0.01）和姿态（SO(3)测地距离，弧度）。左右按各自有效权重和归一化后平均。不监督夹爪开合。
- relative：右TCP在左TCP坐标系的动态位姿，p_rel=R_left.T@(p_right-p_left)，R_rel=R_left.T@R_right。比较同一专家时间步的相对位置和姿态，不是固定手间距，更不是把两手拉到一起。只计算一次配对loss。

两项共用既有可微FK，沿末端局部+X加0.12m TCP偏移。新实现放在独立tcp_geometry模块，不修改历史EE数学/参数语义。不得同时启用旧ee_fk_loss_enabled，旧EE开关也不需要打开。模型用a_hat=x_t-t*v_hat恢复干净动作，再按真实数据transform的统计可微反归一化并拆左右关节；全程保留速度场到双臂梯度，几何始终FP32，不新增动作头、不改变推理接口。新固定机器人几何保存在普通runtime对象，不进入模型state_dict或BF16参数转换。

绝对与相对位置/姿态四个系数分别可调。没有把FM的20倍赋给几何项，深度、视频及原FM重要性不变。几何阶段门控独立于FM倍率，支持stages和stage_weights；相对门控需两手都有效且都在允许阶段，配对权重取两手门控较小值。每分支按有效权重和归一化，分母不包含padding或无效关节。

## 文件入口

|用途|文件|
|---|---|
|离线双瓶标签生成|tools/prepare_dual_tcp_pose_cache.py|
|配置校验/冻结|lingbotvla/losses/tcp_geometry_config.py|
|按原episode/frame查离线标签|lingbotvla/losses/tcp_geometry_data.py|
|绝对与相对Loss、可微FK接入|lingbotvla/losses/tcp_geometry.py|
|实际FM速度场与总loss接入|lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py|
|梯度、时钟、mask与实际V2契约测试|tests/tcp_geometry/test_geometry.py|
|当前CPU证据|refs/geometry_evidence/dual_tcp_v1/local_cpu_report.json|

## 离线标签与训练时钟

生成工具只读双瓶50条示范，拒绝覆盖已有缓存。从action和observation.state分别生成世界TCP及相对位姿，两套原始坐标共用episode_index/frame_index，不依赖过滤后样本序号。保存NPZ的4x4矩阵，包含完整旋转与平移；没有四元数xyzw顺序歧义，rotation_format显式为matrix3x3。字段包括action_tcp/state_tcp、action_rel/state_rel、valid、stage和原始双臂关节；meta保存schema、数据数值/FK/阶段表/脚本/NPZ指纹、TCP偏移、坐标系。

普通动作使用action[anchor+k]；actions_convert_from_state使用state[anchor+1+k]。chunk每一步查自己的标签，禁止用锚点复制整chunk。越界padding屏蔽；同一缓存episode缺少非padding帧直接报错。其他复习任务不在双瓶缓存时返回无效几何mask，不给它们套双瓶几何监督。

新路径只支持可还原为绝对物理关节角的动作，拒绝subtract_state，以及会裁剪动作真值的bounds_98/bounds_99/minmax归一化；当前标准bounds_99_woclip可用。支持单卡GAS=1，保证权重和归一化口径；不隐式放宽为分布式或梯度累积。配置与标签的FP32、时间对齐仍须在真实DataLoader验证，不以小型fixture冒充全链。

## 生成与开关（尚未在真实数据执行）

```bash
python tools/prepare_dual_tcp_pose_cache.py \
  --dataset-root /root/autodl-tmp/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30 \
  --output-dir refs/data_selection/pick_dual_bottles_tcp_pose_v1
```

输出geometry_config.disabled.json，两项enabled均false。准备训练时复制为新配置，仅开启要测试的分支；不要改动既有冻结运行。示意参数（这些是待标定示例，不是已经验证的最佳系数）：

```json
{
  "absolute": {"enabled": true, "position_weight": 1.0, "rotation_weight": 0.1, "stages": [], "stage_weights": {}},
  "relative": {"enabled": false, "position_weight": 0.1, "rotation_weight": 0.01,
    "stages": ["fine_align", "pre_contact", "gripper_close", "grasp_lift"], "stage_weights": {}},
  "tcp_offset_m": 0.12,
  "time_max": 1.0,
  "huber_delta": 0.01
}
```

实际配置还须保留生成器给出的spec/cache路径和SHA字段，不能只保存上述片段。absolute.stages空列表表示该目标所有有效动作，relative示例只对两手都处于指定阶段的帧开启。time_max仅门控几何项，不改变FM时间采样，也不要求低噪FM。position_weight/rotation_weight是损失系数；stage_weights是独立门控倍率，绝不能把它和FM20倍默认绑定。

原生训练参数：train.tcp_geometry_config_path（默认None）。配方新增可选顶层tcp_geometry_config路径；prepare复制并冻结config/spec/cache/meta到本轮输出目录，重写内部引用。未显式给出则清除来源配方继承的开关，训练入口也以本轮显式参数为准，不因HF配置中有历史字段自动开启。

训练日志/TensorBoard前缀tcp_geometry/，包括absolute/relative未加权位置/姿态loss、各自加权loss、左右位置cm/姿态deg、相对位置cm/姿态deg与有效配对步数。预测路径不detach，专家标签detach。默认None时没有新增标签读取、模型forward几何分支或输出loss项。

## 验证边界

新增17项CPU测试全部通过：NumPy/Torch完整FK一致、真值近零、双臂及实际FM输出投影梯度非零/有限、夹爪梯度为零、相对位姿恒等式及全局刚体不变性、offset0/1、末尾padding/NaN、左右有效性相交、阶段门控、分数阶段倍率归一化、缓存指纹与冻结、实际V2总loss独立相加、关闭时原FM返回不变、真实collator字段及缺失拒绝。

使用仓库保存的200组SAPIEN参考姿态（并非本次新运行仿真），新TCP计算最大位置差6.03e-7米、旋转矩阵元素差8.12e-7。有限差分验证相对loss左右关节梯度。

本机CPU PyTorch2.8.0环境有readline交互扩展崩溃，测试进程暂时将sys.modules['readline']=None绕开；没有修改系统环境、Torch数学或项目运行逻辑。新17项通过。较早14项新测试并入回归时219通过、4失败；原HEAD临时副本同范围205通过、同4失败，证明是既有问题：旧EE配置ee_fk_spec、旧EE目标字段、缺torchdata、缺einops。证据文件记录具体节点，不把它们写成全绿。

未完成：真实50条cache生成与逐帧覆盖报告、真实DataLoader/当前norm统计对照、GPU BF16/FP32及峰值显存/速度、共享层梯度标定、正式训练和闭环消融。不因此改变现有训练/存档/评测设置。下一步先恢复无卡SSH完成真实标签和数据联调，再单独决定是否训练B(EE)或C(EE+Relative)。

## 2026-10-24正式实验启动

run：`20261024_1649_dual_bottles200_from_1050_s200_bf16_m56_tcp_abs_rel_stage20_hold15_noreplay_parallel3_video`。来源1050-step200，双瓶50条，BF16 micro/GBS56、GAS1、不复习、仅step200存档，训后3回合并行带视频、普通chunk50，无时间集成、无自动关机。绝对位置/姿态1/0.1，相对位置/姿态0.1/0.01；相对阶段fine_align/pre_contact/gripper_close/grasp_lift、两臂交集门控。FM等沿用双瓶关键20、保持/初始提离15、其他原倍率，按权重和归一化；几何不继承20/15倍。

真实CPU首/中/末样本TCP位置最大误差0.00001335cm，真值总几何loss约1.8e-8至2.2e-8，梯度有限，末尾padding屏蔽。证据`refs/geometry_evidence/dual_tcp_v1/real_dataloader_cpu_report.json`。CPU检查器补齐显式tcp_geometry_config_path，和训练入口一致；默认None行为保持。实验指纹以本地已核对文件SHA更新，官方baseline SHA保留。
