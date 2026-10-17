# EE/FK 96G 实机验收与500步对照启动

承接 [CPU验证](VALIDATION_RUNTIME_2026-10-17.md)，2026-10-17 在单张 NVIDIA RTX PRO 6000 Blackwell Server Edition（97887MiB）完成以下验证。代码为实验分支 `7f66523703c37e486b3f86c58132ebaf92915909`；生产目录未改动。

## 已完成的真实GPU验证

| 项目 | 结果 |
|---|---|
| 真实3步训练 | 从原始step9680、micro24/GAS1启动，低噪FM+target_change_window+EE loss；正常结束，梯度范数有限 |
| 周期与收尾HF | 第2步在unit结束前写周期HF，第3步写收尾HF；两份各1708个BF16权重张量，无DCP；config保留FK spec与开关，权重中没有运行期几何 |
| 实际权重重载 | 以刚保存的step2 HF新建1步run，真实加载、前反向、独立val、正常结束；本次重载检查不另存HF |
| 训练观测 | 三步加权EE loss分别0.00503757、0.00540055、0.00407451；TB四项标量均有限，EE有非零贡献 |
| 真实val观测 | active_val及active_train均写出EE指标；每组4条轨迹，EE分别统计554及549个非padding帧；复用原预测 |
| 显存 | 3步run记录峰值82.30GiB；重载run峰值70.13GiB |
| CUDA运动学 | 200组归档SAPIEN参考，位置最大6.42135e-7m，姿态最大9.00505e-7rad；FP32输出 |
| CUDA自动混合精度与梯度 | BF16 autocast下FK loss仍FP32，预测梯度有限且非零，GT无梯度，mask屏蔽帧梯度为零 |

原始step9680在3步验收run开训前active_val：两臂合并位置RMSE **50.84795mm**、姿态平均角 **11.45653度**、夹爪MAE **0.02711146**。正式run的Step0面板显示约 **50.86mm / 11.44度 / 0.0271**；保留各run实测值，不混用小数尾差。这些是对照起点，不能据3步短测推断闭环能力改善。

原始小型证据集中在 [evidence/gpu-2026-10-17/](evidence/gpu-2026-10-17/)，包括短训/重载核验日志、CUDA数值报告和正式训练命令。没有归档权重、大型数据或凭据。之前保存的CPU参考没有再次用SAPIEN导出，本轮在CUDA上与同一参考比较。

## 测试意图与复现入口

测试代码集中在同功能目录 `diagnostics/`：

- `lingbot_ee_gpu_prepare.py`：创建独立3步验收run，第2步周期HF、第3步收尾HF，检查真实非unit边界的保存路径。
- `lingbot_ee_gpu_inspect.py smoke --verify`：核对训练结束、EE事件/TB、两份HF的dtype/spec/张量数和无DCP；`reload --verify`核对重载run。
- `lingbot_ee_gpu_stage.py reload`：从验收step2 HF启动真实1步重载验证；`formal`创建本次批准的原始step9680新增500步run。
- `lingbot_ee_cuda_math.py`：在没有训练进程时，检查200组CUDA几何与autocast下的梯度。

这些是本次服务器路径的操作型脚本，实际使用时先复制到服务器 `/tmp/`，再按上述顺序执行；`prepare`和`stage`会创建新run并后台启动真实训练，`inspect`为只读取证。它们通过 `/tmp/lingbot_ee_gpu_*_job.json` 传递run路径。不要在训练期间另起CUDA检查。

## 正式500步训练

- run：`al_stand_ee_lnfm_9680_add500_20261017_184252`。
- 原始step9680权重起步、新优化器和步号；任务仅 `place_object_stand`。
- micro24/GAS1/GBS24；保留上一轮视觉冻结，不是解冻视觉塔。
- 低噪FM与目标变化窗口采样保持；EE位置/姿态/夹爪权重 `1/0.1/0.1`，时间门控 `t<=0.2`。
- 每100步active_val/active_train，保持4条独立val；250和500步存BF16 HF，无DCP。关闭PASS状态转换、任务扫描、复习；不运行历史产物删除监控。
- 日志：`/data/outputs/logs/al_stand_ee_lnfm_9680_add500_20261017_184252.log`。
- 输出：`/data/outputs/al_stand_ee_lnfm_9680_add500_20261017_184252`。

短测结束后可用约30GiB，正式启动器已按两份HF体积和余量检查通过。验收产物保留，没有删除历史检查点。正式训练不附带关机动作。

本记录只证明GPU验收通过及正式训练已启动，**不代表500步已完成或闭环效果提升**。后续应比较250/500与原始step9680在同任务、相同指令/seed的clean闭环结果。
