# RoboTwin benchmark 脚本迁入与后续闭环评测

## 来源与当前状态

- 文件：`experiment/robotwin/run_clean_benchmark.py`。原样迁入基线见提交 `1d8bc31`；
  当前入口已适配为调用仓库现有 NVIDIA 闭环 launcher。
- 来源：[Robotwin-radeon-cloud / run_clean_benchmark.py](https://github.com/ZiguanWang/Robotwin-radeon-cloud/blob/27b2f6fcb6c715e2cbb6db4431381f87c94b86ed/docker/assets/experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py)。
- 固定上游 commit：`27b2f6fcb6c715e2cbb6db4431381f87c94b86ed`，迁入日期：2026-10-14。
- 原始上游脚本 SHA256（迁入基线）：`b70d8e40421fa50f97154889a6b5c73a47dd1fb6d9f2a80ddfc6352d7bcb5e73`。
- NVIDIA 适配已在本地完成；尚未部署训练机，也未执行 GPU / 仿真评测。

文件名中的 `clean` 不代表只评 clean：默认 `--task-config both`，同时评
`demo_clean` 与 `demo_randomized`。单独 clean 必须显式传 `--task-config demo_clean`。

## 上游脚本的评测口径

- 上游从 `<robotwin-root>/env_cfg/eval/all_tasks.yml` 读取并要求恰好 50 个不同任务。
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
| AMD `HIP_VISIBLE_DEVICES` 与4/8卡服务池 | 改由现有 `start_robotwin_infer_and_eval.sh` 启动 CUDA 推理服务；固定 `--num_gpus 1 --num_per_gpu 1`，每个 task config 顺序跑任务 |
| `launch_official_server.sh` 与 `eval_policy_xpolicylab.py` | 改用本仓库 deploy 服务及 `RoboTwin-lingbot/script/eval_policy_client_lingbotvla.py`；现有 launcher 负责同步客户端/helper 和仿真路径自检 |
| 上游 expert 失败接受规则 | 现有客户端固定使用其自己的 expert 预检与 seed 规则，不支持上游两个 expert 开关；两边结果不可直接视为完全同口径 |
| 任务列表 | 当前 RoboTwin checkout 没有上游 `all_tasks.yml`；适配器从现有 NVIDIA launcher 读取其50任务列表，保证任务名与顺序一致 |
| `/opt/robotwin-env/bin/python`、`/RoboTwin`、`/workspace/runtime` | 使用当前 launcher 的推理/仿真 conda 环境；适配器接受 RoboTwin、conda、Qwen、模型和输出路径参数 |
| 上游每 GPU 并发执行 | 单卡模式为一服务一模拟槽，任务串行；`both` 下 clean 与 randomized 分开启动，各加载一次模型 |

## 训练结束后的验证顺序

1. 选定完整 HF checkpoint，确认关联的训练配置与 Qwen 路径，记录模型步号与代码 commit。
2. 核对 RoboTwin 仿真环境、资产、任务清单及推理协议；确定 clean / randomized、回合数、精度与 expert 规则。
3. 做不启动 GPU 的命令 / 参数契约检查，再在训练退出后做单任务少回合闭环冒烟。
4. 冒烟成功后跑 50 任务 clean；randomized 另一次运行，使用独立输出目录。
5. 保存每任务成功数、实际完成回合数、失败原因、耗时和总成功率；缺任务或异常退出不能按完整 benchmark 汇报。

适配后的 benchmark CLI 从 `lingbot-vla-v2/` 目录运行。以下 clean 50任务命令**尚未执行**；
在训练退出且评测环境预检通过后再启动，替换占位模型路径：

```bash
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
python experiment/robotwin/run_clean_benchmark.py \
  --model-path /path/to/run/checkpoints/global_step_N/hf_ckpt \
  --robotwin-root /path/to/RoboTwin-lingbot \
  --qwen3vl-path /path/to/Qwen3-VL-4B-Instruct \
  --conda-sh /path/to/miniconda3/etc/profile.d/conda.sh \
  --inference-env lingbotvla --sim-env RoboTwin \
  --task-config demo_clean --num-tasks 50 --episodes 10 --precision fp32 \
  --runtime-dir /path/to/eval_results --run-name clean_step_N
```

默认使用 FP32。若选择 BF16，传 `--precision bf16` 并在结果中记录。单任务冒烟通过
`--run-name smoke_step_N --task-config demo_clean --num-tasks 1 --episodes 2` 执行。
`--num-tasks` 取官方任务列表前 N 个，默认50。

`both` 对 clean 与 randomized 各运行完整任务列表，产物和 done 标记分开保存。
`--resume` 要求 `run_manifest.json` 中模型路径/index、Qwen路径、任务列表、回合数、精度、环境和视频参数完全匹配。

## 本次静态验证入口

```bash
python -m py_compile experiment/robotwin/run_clean_benchmark.py
python experiment/robotwin/run_clean_benchmark.py --help
```

`--help` 不探测 GPU、不启动服务。正式运行会调用 NVIDIA launcher，
启动推理服务和 RoboTwin 仿真；因此只能在训练完成并确认 GPU 空闲后执行。

本次适配验证：`python3 tests/test_clean_benchmark_nvidia_adapter.py` 为 6 passed；
benchmark 与测试 `py_compile`、隔离 PyYAML 环境的 `--help`、launcher `bash -n`、
`git diff --check` 均通过。CPU 契约覆盖官方任务列表、单卡参数、客户端成功率解析、完整回合才记 done、
mock launcher 记账和 clean/randomized 分开。GPU 闭环、模型服务启动、仿真资产与最终评测口径
仍待训练结束后验收。
