# 测试目录导航

新增功能测试按 `tests/<功能名>/` 集中组织：`README.md` 记录测试意图、方法、运行命令、判据与定位线索；`test_*.py` 放自动回归；`diagnostics/` 放手动诊断；`evidence/<日期>/` 保存小型验证证据。大模型、完整数据集和视频不放进测试归档，日志不得含凭据。

历史根目录测试逐步整理，本次只迁移 FK 单测及其诊断材料，其他测试入口沿用原位置。

| 功能 | 入口 | 内容 |
|---|---|---|
| 末端 FK 一致性监督 | [ee_fk/README.md](ee_fk/README.md) | FK 单测、反归一化/梯度检查、真实资产对拍、精度探针与历史记录 |
| 低噪 FM（关联回归） | [test_low_noise_fm.py](test_low_noise_fm.py) | 时间分布、关闭时 RNG 不变、V2 继承采样逻辑 |
| AL固定步数/epoch边界 | [al_epoch_schedule/README.md](al_epoch_schedule/README.md) | 循环预算、真实hook/sampler跨数据长度边界、最终partial flush |
| 闭环固定推理噪声 | [test_inference_fixed_noise.py](test_inference_fixed_noise.py) | `--infer_noise_seed` 参数和推理服务随机状态隔离 |
