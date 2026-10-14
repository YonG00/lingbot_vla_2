# RoboTwin benchmark 脚本迁入与后续闭环评测

## 来源与当前状态

- 迁入文件：`experiment/robotwin/run_clean_benchmark.py`，原样复制，未修改评测逻辑。
- 来源：[Robotwin-radeon-cloud / run_clean_benchmark.py](https://github.com/ZiguanWang/Robotwin-radeon-cloud/blob/27b2f6fcb6c715e2cbb6db4431381f87c94b86ed/docker/assets/experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py)。
- 固定上游 commit：`27b2f6fcb6c715e2cbb6db4431381f87c94b86ed`，迁入日期：2026-10-14。
- 源文件 SHA256：`b70d8e40421fa50f97154889a6b5c73a47dd1fb6d9f2a80ddfc6352d7bcb5e73`。
- 已迁入本地；未部署训练机，未执行 GPU / 仿真评测，尚未适配当前单卡 NVIDIA 环境。

文件名中的 `clean` 不代表只评 clean：默认 `--task-config both`，同时评
`demo_clean` 与 `demo_randomized`。单独 clean 必须显式传 `--task-config demo_clean`。

## 上游脚本的评测口径

- 从 `<robotwin-root>/env_cfg/eval/all_tasks.yml` 读取并要求恰好 50 个不同任务。
- 每任务默认 10 回合；clean 为 500 回合，both 为 1000 回合；seed 固定为 0。
- 每 GPU 一个常驻模型服务，各 worker 动态领取任务；只接受 `--gpu-count 4` 或 `8`。
- 默认 expert 检查开启，同时允许 expert 执行失败后保留渲染的 episode 信息。
  这个种子接受规则必须单独核对，不能将它与本仓库客户端的结果直接视为同一口径。
- 视频、compile 默认关闭；推理精度由配套 `launch_official_server.sh` 固定为 BF16。
- 每任务日志必须含 `Final success rate: N/M = ...%`，且 M 等于指定回合数，才写 `.done`。
- 最后逐任务累加成功回合数 / 总回合数，不使用 AL 的开环 MSE 或 PASS 数作为成功率。
- `--resume` 根据完成标记跳过任务，但没有 checkpoint / 参数指纹校验；更换模型、回合数
  或 expert 规则应使用新的 `--run-name`，避免混合不同条件的结果。

## 当前仓库的接入差异

| 上游依赖 / 假设 | 当前仓库情况与后续处理 |
|---|---|
| AMD `HIP_VISIBLE_DEVICES`，并清除 CUDA 设备限制 | 当前训练机是 NVIDIA；迁入脚本不能直接启动，应适配 CUDA 设备绑定 |
| 4/8 GPU | 当前单卡；需在适配阶段支持 1 GPU，并验证服务与仿真共卡 |
| `experiments/lingbot_vla_v2_6b_robotwin/scripts/launch_official_server.sh` | 未随此次单文件迁入；其 Docker 目录、Qwen 路径和 ROCm 环境变量不适用当前环境 |
| `scripts/eval_policy_xpolicylab.py` | 当前使用 `RoboTwin-lingbot/script/eval_policy_client_lingbotvla.py`；参数、协议和日志格式不同，不能只换文件名 |
| `env_cfg/eval/all_tasks.yml` | 当前启动器内置 50 任务清单，也支持 `--task_list_file`；后续核对任务集合 |
| `/opt/robotwin-env/bin/python`、`/RoboTwin`、`/workspace/runtime` | 当前推理与仿真使用分离环境；所有路径需显式配置 |
| BF16 推理、上游 expert 失败接受规则 | 正式闭环评测前确定精度与种子规则，并记录在结果中 |

## 训练结束后的验证顺序

1. 选定完整 HF checkpoint，确认关联的训练配置与 Qwen 路径，记录模型步号与代码 commit。
2. 核对 RoboTwin 仿真环境、资产、任务清单及推理协议；确定 clean / randomized、回合数、精度与 expert 规则。
3. 做不启动 GPU 的命令 / 参数契约检查，再在训练退出后做单任务少回合闭环冒烟。
4. 冒烟成功后跑 50 任务 clean；randomized 另一次运行，使用独立输出目录。
5. 保存每任务成功数、实际完成回合数、失败原因、耗时和总成功率；缺任务或异常退出不能按完整 benchmark 汇报。

当前仓库已有 NVIDIA 评测入口：
`experiment/robotwin/start_robotwin_infer_and_eval.sh`。下面是后续 clean 50任务命令模板，
**此次未运行，也不声称与上游 BF16 / expert 种子口径等价**。在训练退出且环境预检通过后再启动。
从 `lingbot-vla-v2/` 目录执行，替换所有占位路径与环境名：

```bash
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path /path/to/run/checkpoints/global_step_N/hf_ckpt \
  --eval_workdir /path/to/RoboTwin-lingbot \
  --output_base /path/to/eval_results/clean_step_N \
  --conda_sh /path/to/miniconda3/etc/profile.d/conda.sh \
  --inference_env lingbotvla --sim_env RoboTwin \
  --task_config demo_clean --num_tasks 50 --test_num 10 \
  --num_gpus 1 --num_per_gpu 1 \
  --use_bf16 False --use_fp32 True --use_compile False --no_video
```

此模板采用现有启动器的 FP32 口径。若选择 BF16，显式改为
`--use_bf16 True --use_fp32 False`，并在结果中记录。冒烟时将 `--num_tasks 50 --test_num 10`
改为 `--num_tasks 1 --test_num 2`，使用独立输出目录。

## 本次静态验证入口

```bash
python -m py_compile experiment/robotwin/run_clean_benchmark.py
python experiment/robotwin/run_clean_benchmark.py --help
```

`--help` 需要 PyYAML，不探测 GPU、不启动服务；默认执行脚本会在 `main()` 中创建输出目录、
探测 torch GPU 并启动子进程，不能用于有训练运行时的预检。

本次验证：与固定版本源文件 `cmp` 一致，语法编译和 `git diff --check` 通过。
本机默认 Python 缺少 PyYAML；在临时隔离环境安装 PyYAML 后，`--help` 返回成功。
没有修改项目依赖或训练机环境。
