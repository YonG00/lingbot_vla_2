# FK 精度修复与 val 末端误差：无卡验证

本轮承接 [首次几何对拍及故障记录](VALIDATION_2026-10-17.md)，修复模型转 bf16 时 FK 几何随之降精度的问题，并验证独立 val→AL→TensorBoard→仪表盘链路。模型主体和推理仍用 bf16，只有 FK 辅助分支局部 FP32。

## 本轮结果

- 服务器 CPU 回归 **143 passed，26 subtests passed，33.90 s**；退出码0。
- 原始 step9680、实际数据集、归一化路径、Qwen tokenizer 和 stand AL 配置的启动预览通过，未创建训练 run。
- `place_object_stand` 划分为40 train / 10 val；现有 active_val 仍使用其中4条独立 val，不拆分验证集。EE 启用不增加模型推理次数，不改变旧 GMSE 或 PASS。
- 原始权重每份 `11.8763 GiB`；本轮检查可用 `40.9512 GiB`。500步的250/500两份 HF 约需23.75 GiB，另留导出和日志余量；实际开卡时重新检查磁盘。
- FK spec 与当前 RoboTwin embodiment 的左右6轴名称及顺序一致。
- 未加载6B大模型，未占用 GPU，未启动训练或自动关机。

原始通过日志：[cpu_regression.log.txt](evidence/runtime-2026-10-17/cpu_regression.log.txt)。测试命令在 [README](README.md) 的“运行期接线回归”。测试使用服务器训练环境 Python3.12，设置 `CUDA_VISIBLE_DEVICES=`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、离线 Hugging Face；代码放在隔离目录 `/tmp/lingbot_ee_cpu_v2_vl70x66r`，未在生产工作树打补丁。

## 检查什么、怎么判定

| 检查 | 方法与通过条件 |
|---|---|
| dtype生命周期 | 普通模型挂载非nn.Module的 `EEFKRuntime`，父模型转bf16前后FK几何逐值不变；FP32 solver不进入HF权重 |
| 权重/config往返 | 小模型safetensors保存加载、config保存加载后几何和开关不变；spec嵌入config，不依赖原路径 |
| autocast/反传 | CPU bf16 autocast中恢复动作和计算loss仍为FP32，预测梯度有限且非零，padding不回传；GT detach见关联单测 |
| SAPIEN参考回归 | 父模型先转bf16、再延迟构造solver，与首次归档的200组reference比较；位置最大值<1e-5m、姿态最大值<1e-4rad |
| 真实val方法契约 | AST提取现有 `_evaluate_ids` 等实际方法，替换预测来源，检查只调用一次推理、旧GMSE全字段不变、EE只统计有效帧 |
| 统计口径 | 两条不同长度轨迹等权、按原始feature映射提取左右臂/夹爪、padding中的NaN不计入、没有真实mask时拒绝 |
| AL/TB/面板 | 实际EvalResult→TrajectoryMetrics→Scheduler→SummaryWriter，读回TensorBoard标量和JSONL，再调用面板格式化；步号和单位正确 |
| 分片 | 重复轨迹只计一次，EE轨迹覆盖不完整时显式报错；未测试真实多GPU通信 |
| 启动器 | EE开关/权重解析、无低噪FM时拒绝FK loss、默认关闭、真实HF/header/tokenizer预览；没有加载权重张量 |
| 关联回归 | scheduler、持久化/恢复、低噪FM、静态目标变化窗口、HF存档策略、启动器和仪表盘 |

模型重依赖方法的CPU接线测试使用实际函数体及替身输入，不能替代真实模型和CUDA测试。200组参考本轮没有重新运行SAPIEN导出；验证的是新运行期对象与已归档reference的一致性。

## 测试过程中发现的事项

1. 既有面板测试的模拟训练器缺少 `world_size`，在旧HEAD `298e90e` 同样失败。保留 [基线失败日志](evidence/runtime-2026-10-17/panel_baseline_failure.log.txt)，仅补全测试fixture，没有改动生产训练器识别。
2. 隔离包首次扩展测试漏带闭环脚本和部署源码，4项测试因文件缺失失败。补齐仓库原文件后同一选择集全部通过，没有改动这些产品文件。
3. 新AST测试最初漏带类型注解future导入和numpy辅助函数，补齐后通过，未以跳过测试掩盖问题。

## 开卡后的剩余验收

1. 96G单卡，独立短run检查真实bf16模型加载、FSDP前反向、三项FK loss有限且参与梯度；确认低噪门控和真实数据mask。
2. 触发一次实际独立val，确认EE与GMSE在同一测量步写入TB/事件流，训练可正常恢复。
3. 触发周期HF直出并重新加载，确认bf16权重、嵌入spec、无DCP、FK精度不退化。短run使用独立输出，不污染正式对照。
4. 短测通过后再从原始step9680启动stand新增500步，保持上一轮低噪FM、目标变化窗口、批量、视觉冻结配置，仅增加EE监督与诊断；保存250/500两份HF。最终能力用相同seed/指令/clean闭环评测比较。

CPU通过表示具备开卡短测条件，不能保证抓取或放置成功率提升。
