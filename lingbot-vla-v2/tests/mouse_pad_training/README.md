# 鼠标单任务启动配方：CPU契约

```bash
PYTHONPATH=.:tests python -m pytest -q \
  tests/mouse_pad_training/test_recipe.py \
  tests/test_skill_mix_task_launcher.py tests/test_continue_skill_training.py
```

新增测试不运行torch、CUDA、训练或关机：检查真实manifest的40/10白名单与episode范围；模型源独立为9680；低噪FM/EE三项权重；预览不写产物；GPU忙时不创建后台任务。用冻结视觉且启用LoRA的来源配置调用真实子启动器的 `--prepare-only`，核对重建后四个冻结/部分训练开关均关闭。

新启动文件使用独立training配置快照是因为原始9680 run缺少便捷续训所需的 `datasets.txt` 和 `al_config.json`；复用完整启动器及现有后台运行/最终HF核验/关机流程，不复制训练循环。

本地通过情况见本轮改动台账；真实资产/数据对拍与GPU尚未验证。
