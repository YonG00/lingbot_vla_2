# 鼠标主任务：FK位置权重6与关闭夹爪辅助监督的续训

本实验由用户明确指定：基于最新纯主任务500步模型再训500步，将FK夹爪辅助损失权重从1.5改为0，位置从10改为6；姿态仍0.8。原始动作模仿损失中的夹爪维度保留。它是继续学习后的效果观察，并非同起点、相同训练时长的严格配对消融。

## 起点与不变量

- 起点：`/data/outputs/al_mouse_pad_primary50_fp32master_fp32hf_m28_add500_20261019_v1/checkpoints/global_step_500/hf_ckpt`。
- 继承该run的训练配置、归一化、主任务50条、低噪声FM、阶段窗口采样、学习率1e-4及Expert倍率；全量训练，视觉不冻结。
- FP32主参数/BF16计算，micro28/GAS1/GBS28，梯度checkpoint开启，FP32 HF-only导出。
- 新增500步，仍在250、500步各保存一份，不减少存档或改变保留策略。
- HF续训会重新初始化优化器和学习率调度，不是完整优化器断点恢复。
- 用户本轮要求训练完成自动关机：两份HF均校验成功才关机，不插入闭环评测；任何训练或存档异常保留机器。

## 预览与执行

在实验包根目录，默认仅预览：

```bash
python experiment/robotwin/recipes/20261019_place_mouse_pad_fp32_gripper_off.py \
  --run-name al_mouse_pad_fp32_m28_p6_grip0_add500_20261019_v1
```

确认预览及容量后，使用同一命令加`--execute --shutdown`。入口后台运行，不覆盖既有run；不要重复启动。

日志和状态位于`/data/outputs/skill_launches/<run-name>.log`、`<run-name>.status.json`；精度记录位于输出run的`precision_audit.json`。JSON结果放运行目录，不放docs。

## FP32兼容修正与验证

旧`checkpoint_inventory`只接受BF16，上一轮FP32存档已完整导出，但被续训流程的最终校验错误拒绝。现在只有显式FP32主参数模式可接受FP32来源；最终导出按配置要求的dtype独立校验，仍检查分片与索引完整性。关机前核验全部约定里程碑，缺250或精度错误都不关机。CPU对照还发现冻结训练JSON未同步EE命令行覆盖值；现将全部显式train覆盖同步进快照，避免记录与实际CLI不一致，不改变原命令行生效顺序。

CPU契约：`tests/mouse_pad_training/test_gripper_off.py`构造FP32来源，经过真实worker重建，逐项比较训练配置只改变FK位置与夹爪权重，且保留50条、250/500存档。`tests/test_continue_skill_training.py`覆盖两份FP32成功后关机及中间存档缺失/精度错误时保留机器。

改动本意：只改变位置和夹爪FK辅助目标；不变量为其余训练控制和存档节奏；异常症状为来源FP32被误拒、worker丢参数、缺250仍关机；快速回滚为不启动新入口或恢复位置10、夹爪1.5；验证入口为上述CPU测试及实际CLI/精度报告。

## 空间与执行状态（2026-10-19）

用户授权删除旧混合辅助数据run的250/500及上一轮纯主FP32 run的250，均已删除；保留当前FP32 500起点和混合前mouse500。删除后空闲74,324,926,464字节（69.22GiB），两份FP32权重各约23.75GiB；容量按两份来源权重加10%及3GiB预算，约55.26GiB。服务器CPU测试97项通过、36项子测试通过；真实来源配置预检确认仅FK位置10→6、夹爪1.5→0，以及运行目录变化，norm SHA256为`26ac51ad6364e64b54573b3f0a0e7a5186b960fa74b4686b9cc099b59d3ce8dd`。本地py_compile通过；本地Python3.12未安装pytest，完整测试在服务器Python3.12执行。GPU正式启动状态以运行目录日志为准。
