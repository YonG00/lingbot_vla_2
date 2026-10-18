# 连续抓取窗口 CPU 验证（2026-10-18）

算法契约沿用 `../test_target_change_window.py`，启动/续训验证沿用既有启动器测试，不复制测试实现。此目录集中记录本功能的验证意图与命令。

## 验证意图

只桥接闭合、保持、同侧初始腕部运动；普通自由空间动作保留事件前后局部窗口，避免全运动区间加权。覆盖闭合/打开区分、等待超时、再次开爪、边界裁剪、坐标映射、缓存schema2、缓存复用、扫描关闭和DEFER/保存不重扫。750步计划允许最终不足100步的单元收尾；HF仍按250步保存。

## 命令与证据

```bash
PYTHONPATH=.:tests python -m pytest -q \
  tests/test_target_change_window.py tests/test_skill_mix_task_launcher.py \
  tests/test_continue_skill_training.py tests/al_epoch_schedule/test_loop_budget.py
```

本地Python3.12隔离环境，52 passed、22 subtests passed；修改Python文件编译、recipe的 `bash -n`、默认预览和 `git diff --check` 通过。本地无torch/CUDA，不加载模型。

用之前只读取回的stand缓存原始目标曲线重建扫描输入：5561帧中3707帧进入窗口（66.6607%），40个闭合阶段均关联腕部运动，权重2下理论概率质量79.9957%。此验证未重读服务器Parquet，也不替代视频对帧或GPU训练验证。
