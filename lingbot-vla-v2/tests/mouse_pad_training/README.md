# 鼠标单任务启动配方：CPU契约

FP32主参数/BF16计算、BF16/FP32 HF与FP32评测路径：`test_precision_path.py`；含真实启动器worker重建、dtype记录、评测参数及单次forward返回值不变契约。实机精度、容量与正式micro28记录见[说明](../../docs/auto_learning/mouse_precision_capacity_zh.md)。

FK位置6、夹爪0续训：`test_gripper_off.py`，覆盖FP32来源续训及真实worker重建后的仅两项损失权重变化的配置比较；来源为最新FP32 500，位置10→6、夹爪1.5→0，保留姿态0.8、micro28/GAS1及250/500存档。见[实验说明](../../docs/auto_learning/mouse_gripper_off_continuation_zh.md)。

训练后3回合闭环与关机门控：`test_train_eval_pipeline.py`，覆盖失败保留机器、正常零成功评测后允许关机，详见[流水线说明](../../docs/auto_learning/mouse_primary50_train_eval_zh.md)。

Blackwell评测渲染的独立CPU契约：`test_render_runtime.py`（不导入torch/SAPIEN）。部署与GPU验证见[去噪运行库说明](../../docs/auto_learning/robotwin_render_runtime_zh.md)。

```bash
PYTHONPATH=.:tests python -m pytest -q \
  tests/mouse_pad_training/test_recipe.py \
  tests/test_skill_mix_task_launcher.py tests/test_continue_skill_training.py
```

新增测试不运行torch、CUDA、训练或关机：检查真实manifest的40/10白名单与episode范围；模型源独立为9680；低噪FM/EE三项权重；预览不写产物；GPU忙时不创建后台任务。用冻结视觉且启用LoRA的来源配置调用真实子启动器的 `--prepare-only`，核对重建后四个冻结/部分训练开关均关闭。

新启动文件使用独立training配置快照是因为原始9680 run缺少便捷续训所需的 `datasets.txt` 和 `al_config.json`；复用完整启动器及现有后台运行/最终HF核验/关机流程，不复制训练循环。

本地及服务器均33 passed、13 subtests passed；服务器真实资产预览退出0。见 [实机验证记录](VALIDATION_20261018.md)；鼠标数据完整对拍、新窗口覆盖率和GPU仍待实际启动验证。

全轨迹拟合实验见 `test_all_episodes.py` 和 [实验说明](../../docs/auto_learning/place_mouse_pad_all50_p10_zh.md)：原40条 train 与10条 val 合并为50条训练，固定4条 `train_monitor`，每100步只记录训练拟合 GMSE/EE。它不产生 `active_val`、PASS、阈值倍率、train-val gap 或泛化结论。

全轨迹入口默认继承 `al_mouse_pad_9680_ee_lnfm_add500_20261018_150402` 的step500 HF和对应训练配置，新增500步；文件名9680仅表示历史祖先。契约检查默认来源及显式覆盖来源、输出名称，旧40/10入口仍默认原始9680。

本次step500来源调整：本地及服务器均41 passed、26 subtests passed；服务器真实资产预览确认50/0和EE10/.8/1.5。仅CPU预览，未启动训练；当前盘余4.8GiB，正式训练前需解决HF存档空间。服务器按文件同步，保留已有仿真修改，Git HEAD尚未更新为本地提交。

## 鼠标辅助混训CPU契约

`PYTHONPATH=.:tests python -m pytest -q tests/mouse_pad_training/test_auxiliary.py` 验证联合映射、70:30累计配额、episode均匀采样、阶段加权、预取排除、恢复身份及实际Hook/调度。训练装配测试依赖torch，本地缺依赖时跳过，训练机CPU执行。真实97条数据及每条尾部action chunk由 `tools/acceptance_checks/mouse_auxiliary_cpu_preflight.py --output-dir <新目录> --check-dataset-chunks` 验证。当前结果及启动方式以[混训说明](../../docs/auto_learning/mouse_auxiliary_training_zh.md)为准，上文历史空间和Git状态不代表当前状态。
