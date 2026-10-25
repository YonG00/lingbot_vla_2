# 单任务与整体并行评测

在 `lingbot-vla-v2/` 下运行。入口统一为 `experiment/official_control/model_benchmark.py`。训练模型保存为BF16；此入口通过共享服务加载为FP32推理，普通chunk推理，默认不启用时间集成。训练环境运行入口，独立仿真Python运行客户端。

先创建 `evaluation.json`（路径替换为本机实际路径；tasks 使用课程配置 task_order 中的全部50个任务，单任务模式可省略 tasks）：

```json
{
  "model": "/models/checkpoint/hf_ckpt",
  "qwen": "/models/Qwen3-VL-4B-Instruct",
  "robotwin": "/workspace/lingbot-baseline/RoboTwin-lingbot",
  "sim_python": "/envs/robotwin/bin/python",
  "destination": "/results/baseline_benchmark",
  "tasks": ["adjust_bottle"]
}
```

完整benchmark的50任务列表可自动填入：

```bash
python -c "import json; p='evaluation.json'; c=json.load(open(p)); c['tasks']=json.load(open('configs/reproduction/three_stage_v1.json'))['task_order']; open(p,'w').write(json.dumps(c,indent=2))"
python experiment/official_control/model_benchmark.py --job evaluation.json --mode benchmark
python experiment/official_control/model_benchmark.py --job evaluation.json --mode benchmark --execute
```

默认50 clean＋50 randomized，共100项；每项3回合、3回合并行、不录像，总300回合。任务配置依次运行，整场复用一个模型服务；并行的是当前任务的回合。改为400回合：加 `--episodes 4 --episode-workers 4`。显存上限需在目标硬件实测。

单任务3回合、3路并行：

```bash
python experiment/official_control/model_benchmark.py --job evaluation.json --mode single --task stack_bowls_two --execute
```

仅clean默认生效；`--task-config both` 同时测clean与randomized。录像在JSON增加 `"video_episodes": 3`。不同评测使用不同destination；命令不带 `--execute` 仅预览。回合总数由 `--episodes` 控制，并行上限由 `--episode-workers` 控制（1至8）。

中断后使用相同配置、相同模型、相同destination并加 `--resume --execute`。仅跳过完整且汇总可核对的任务/config；未完整项重跑。模型文件或有效参数改变会拒绝续跑。应保留整个评测目录，不能只保留公共JSON。

输出 `benchmark_results.json` 是面向使用者的汇总，包含overall、各task_config、逐任务成功数/回合数/成功率百分数、状态、错误及结果路径；`benchmark_audit.json` 保留续跑审计；`selected_scene_seeds.json` 记录实际种子。成功率分母只计已完成回合，必须同时报告completed/error/pending，不能把未测当失败。格式为本仓库明确的schema，统计口径对齐参考脚本；参考脚本未定义这一JSON接口，不能声称字节或字段完全相同。

默认inline专家校验会在同一仿真进程内寻找有效场景，遇到已识别的专家拒绝才尝试下一个候选；模型执行失败计失败，不换种子。正式模型对比应复用同一组实际种子：将 selected_scene_seeds.json 的内容放入配置的 scene_seeds，并设置 seed_selection 为 fixed。环境/渲染错误属于error，修复后续跑，不算成功或失败。

运行状态：`<destination>_pipeline/status.json`，日志：`<destination>_pipeline/pipeline.log`。长任务后台启动，程序不会自动关机。
