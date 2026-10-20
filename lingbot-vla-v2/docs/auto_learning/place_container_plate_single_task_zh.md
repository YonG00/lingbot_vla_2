# place_container_plate：从adjust_bottle BF16-200单任务续训

2026-10-20用户授权先更新前轮改动、删除三任务400步存档，再从
`official_adjust_bottle_9680_bf16_m56_add200_20261020/checkpoints/global_step_200/hf_ckpt`
仅训练place_container_plate200步，完成自动闭环3回合；服务器启动后再做参数化框架改造。

- 输出：`/data/outputs/place_container_plate_from_adjust_bf16_200_m56_add200_20261020`。
- 来源明确为原adjust_bottle BF16基准，非三任务200/400；新优化器/200步cosine调度，不继承GPU探针更新。
- 数据50条、7934帧、普通帧均匀，不混复习，不启AL/窗口/EE/LN-FM；BF16参数/计算/HF、56/GAS1、GC/compile开启、workers8。
- 延续最近用户授权的200步间隔，本轮最终step200保存BF16 HF，约11.88GiB；后续更改保存节奏仍须允许。
- 入口：`experiment/official_control/single_task_trial.py --task place_container_plate --out <新目录> --execute`，默认不execute只预览。
- 顺序：真实CPU数据整链→独立三步BF16探针→正式200→验证最终BF16 HF→place_container_plate闭环3回合。
- 评测FP32、chunk50、demo_clean、OIDN2.5.1零错误与逐回合视频；状态在`<run>_pipeline/status.json`，结果在`<run>/evaluation_200/summary.json`。
- 6007已预接run/runs；不自动关机。后续框架修改只在本地独立模块进行，不热改这次运行配置。

旧三任务400按授权删除，释放11.89GiB；保留三任务200、9680、原BF16-200及全部评测日志，
报告`/data/outputs/delete_three_task_400_20261020.json`。
上一轮共享评测完整结果：adjust_bottle2/3、click_bell3/3，server_pid58398、model_loads1、OIDN错误0。

不变量：仅目标任务、指定来源、200最终BF16 HF；症状：白名单不符、dtype/保存/渲染断言；
回滚：不运行新入口，原基准独立保留。验证：test_single_task_trial及运行CPU/probe/HF/evaluation报告。
