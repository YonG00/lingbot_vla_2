# 鼠标纯主任务50条训练及闭环评测

用户2026-10-19授权：从混训前鼠标step500继续纯主任务50条，GBS56，500步，每250步存档；最终模型自动闭环3回合，评测正常完成后关机。该方案同时改变辅助数据、损失和batch，不能单独归因于辅助数据。全部50条用于训练，train_monitor只是训练拟合指标，不是独立验证。

独立入口：`experiment/robotwin/recipes/20261019_place_mouse_pad_primary50_train_eval.py`。复用已有续训训练循环，固定检查：来源 `al_mouse_pad_9680_ee_lnfm_add500_20261018_150402/checkpoints/global_step_500/hf_ckpt`，仅主50、无辅助/Replay；GBS56=微批56×GAS1×单卡；梯度检查点开启；EE10/.8/1.5；深度/未来深度/视频三项0.004。新优化器/LR/步号，新增500步，保留250/500两份HF，不更改旧入口默认值。

在实验包根目录运行，默认只做CPU预览：

```bash
python experiment/robotwin/recipes/20261019_place_mouse_pad_primary50_train_eval.py \
  --run-name al_mouse_pad_primary50_g56_from_mouse500_add500_20261019_v1 --shutdown
```

确认后加 `--execute` 后台启动。`--shutdown` 是整条流程结束后关机，不是训练结束关机；默认不关机，`--no-shutdown`可显式保留机器。不要另起旧关机watcher。

顺序为训练正常退出、250与500 HF资产/张量名校验、卸载训练进程、加载最终500模型、3回合闭环评测、结束评测子进程、记录摘要、`bash /usr/bin/shutdown`。任何训练/存档/评测错误都不会自动关机。任务成功率0/3也算正常完成评测，可以关机；不能把“任务失败”与“评测运行失败”混淆。

评测固定demo_clean、seed参数0、推理噪声seed0、BF16、关闭compile、动作块50、模型内部去噪10步。保留原服务器客户端文件，不复制覆盖；逐字节SHA前后检查。要求OIDN2.5.1启动标记、无OIDN错误、完成3回合、结果文件与日志一致、3段全局视频完整，正常结束后才关机。seed/结果落`closed_loop_eval/summary.json`，完整日志和视频在同目录。三个场景是前次固定对照所用场景，不代表总体成功率。

日志与状态：`/data/outputs/skill_launches/<run>.log`、`<run>.pipeline.status.json`；训练子任务另有`<run>.status.json`，只表示训练阶段，不能用它判断评测已完成。冻结配置及命令在`<run>.pipeline.json`、`<run>.json`、`<run>.training.json`、`<run>.al.json`。训练结果在`/data/outputs/<run>/`。

容量检查按两份HF权重×1.1加3GiB计算（含额外评测余量）；不自动删除。用户仅条件授权容量不足时删除新混训run的step250，容量足够则保留。存档节奏仍须用户明确允许才改。

CPU契约：`PYTHONPATH=.:tests python -m pytest -q tests/mouse_pad_training/test_train_eval_pipeline.py`。覆盖训练/存档/评测失败不关机、正常零成功评测后关机、250/500验证顺序、结果/回合/视频/渲染完整性检查；不执行真实训练或关机。GPU终态必须以实际状态文件为准。

## 2026-10-19启动记录

正式run：`al_mouse_pad_primary50_g56_from_mouse500_add500_20261019_v1`，启动代码`ac78f72`。本地73 passed、1 skipped（缺torch装配项）、36 subtests；服务器74 passed、36 subtests，真实配置CPU预览及仿真子入口帮助命令正常退出。预检空余39.86GiB、预算29.13GiB，未删除新250或其他产物。

实际GPU已完成7/500更新，稳定约7.4s/步，采样时设备91761/97887MiB，未OOM。运行配置与CLI确认：micro/global56、GAS1、GC=true、EE位置10、三项视觉0.004、max500/save250、全量ViT；train_ids恰为800–849，辅助及Replay关闭。pipeline状态`training`、shutdown=true，内层training_job.shutdown=false，自动评测3回合，仅最终500模型。

`trainer_config.json`是被CLI覆盖前的基础配置，不能据此判定实际batch或EE；本次合并后的`lingbotvla_cli.yaml`及运行日志分别确认56与10。启动过程与最终评测/关机是不同阶段：当前只验证到训练正常更新，250/500存档、3回合评测及关机终态仍待运行。
