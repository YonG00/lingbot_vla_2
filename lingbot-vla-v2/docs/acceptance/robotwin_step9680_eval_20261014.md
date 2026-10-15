# step 9680 NVIDIA 闭环测评中断记录与对齐准备

本次全量 benchmark 未完成。用户于 2026-10-14 要求停止并改用既有两阶段流程验证，随后确认训练机已关机。下列单任务结果只是带渲染异常的运行记录，不能作为可靠的最终模型成绩。

## 已确认的运行事实

| 项目 | 实测 / 配置 |
|---|---|
| 代码 | `Task_Auto_Learning`，远端 HEAD `e64388c` |
| checkpoint | `al_v79_poolfix_7300/checkpoints/global_step_9680/hf_ckpt` |
| GPU | RTX 5090 D，32607 MiB |
| 运行名 | `al_v79_step9680_bf16_both_fix1_20261014` |
| 计划工作量 | clean / randomized 各50任务，每任务10回合 |
| 推理 | BF16、compile关闭、chunk=50、1服务 / 1仿真槽、视频关闭 |
| 模型加载 | 日志 `GPU ready elapsed=86.24s allocated=11.96GiB` |
| `lift_pot` / clean | 日志最终 `0/10`；launcher任务耗时677秒（包含启动等待） |
| `hanging_mug` / clean | 已启动；停止前没有取得完整任务结果 |
| 全量汇总 | 未完成；未取得最终 `summary.json` |

原始输出：`/data/eval_results/clean_benchmark/outputs/al_v79_step9680_bf16_both_fix1_20261014/`；主日志在同级 runtime 根目录的 `logs/` 下。日志没有被本次操作删除；关机后尚未重新读取磁盘核验。

## 异常与结果限制

1. 首次启动在客户端路径自检处失败，工作目录缺尾随分隔符。已在 `e64388c` 修复；本地CPU契约6项通过，修复后模型与客户端确实运行。
2. 仿真持续出现 `OIDN Error: unsupported device type: CUDA` / `invalid handle`。RoboTwin `_base_task.py::setup_scene` 使用 `rt` 相机 shader 与 `oidn` 降噪。是否影响模型输入图像尚未验证；不能认定这是零成功率的唯一原因。
3. 当前仓库 `policy/ACT/deploy_policy.yml` 的指令类型是 `unseen`，benchmark未显式覆盖。与AMD两阶段脚本的默认 `seen` 不同；下次应核验远端配置和 `_result.txt` 的 `Instruction Type` 后比较。
4. TERM终止主进程和推理服务后，launcher曾继续重试仿真任务：`trap cleanup EXIT INT TERM` 清理后未显式退出。残留进程未完成SSH复核，用户随后确认机器关机。该停止行为是已知遗留，本次未改launcher代码。

## 启动入口差别（已读代码）

| 入口 | 行为 |
|---|---|
| `experiment/robotwin/start_robotwin_infer_and_eval.sh` | 起服务、任务排队、失败最多3次尝试；单次只跑一个task config；生成逐任务日志、`_result.txt`、`stats.txt`。默认FP32 / compile开启，回合数未指定时客户端默认100。 |
| `experiment/robotwin/run_clean_benchmark.py` | 调用上述同一个launcher，增加clean/randomized编排、参数manifest、完成标记、resume与总汇总。本次显式BF16；compile关闭；每任务10回合。 |
| 旧NVIDIA `tools/policy_server.sh` + `tools/eval_tasks.sh` | 服务与仿真分开；模型常驻，可换任务；任务侧没有launcher的看门狗/重试。仍使用同一deploy模型、ACT客户端和RoboTwin渲染器。单换入口不会解决OIDN问题。 |
| 当前 `tools/misc/` AMD适配版 | 增加ROCm/venv和客户端分叉；任务侧默认`official`指向 `/RoboTwin/scripts/eval_policy_xpolicylab.py`，默认`seen`，并有expert失败接受及MPLib设置。不是旧NVIDIA版的默认执行路径，不应混用。 |

旧分支 `feature/auto-learning-v1` 的 launcher、ACT客户端、RoboTwin `_base_task.py` 与当前分支逐文件比较无差异；两阶段脚本固定来源为 `41383125548a03854d5b90f4e9cffde92a019bc9`。本次没有切换或合并分支。

## 下次上机验证计划（尚未执行）

1. 用户开机并提供可用SSH入口后，确认没有遗留训练/测评进程、GPU空闲、代码版本及磁盘余量。
2. 使用旧分支固定版本的NVIDIA两阶段脚本。已从Git读取至本地临时文件，`bash -n`通过；运行脚本经本地准备、SCP、远端后台执行，服务启动使用`WAIT=0`，另起调用核验就绪。
3. 先验证模型实际收到的三路图像与OIDN状态，再跑 `lift_pot / demo_clean` 小样本（3回合）。记录BF16、compile关闭、chunk=50、指令类型、种子和expert筛种规则。
4. 若需与AMD历史的seen结果比较，显式通过客户端override指定seen；旧NVIDIA `eval_tasks.sh` 不读取 `INSTRUCTION_TYPE`，仅设该环境变量不会改变其ACT配置。
5. 环境图像验证通过后，在相同条件下对比step9680与现有官方RoboTwin checkpoint。BF16对照不等同于发布文档的FP32全量成绩。两组结果写回本报告；通过后再讨论恢复全量测评。

CPU检查仅证明两阶段shell脚本语法合法，尚未进行本次两阶段GPU测评，也未取得官方checkpoint对照成绩。
