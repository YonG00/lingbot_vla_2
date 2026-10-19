# 数据清单与来源记录

本目录保存经过审核、需要随Git版本化的数据选择结果及其来源，不保存原始数据或批量图像。

- `mouse/20261019_review_v2/approved.json`：47条鼠标辅助示范，按稀疏全程帧、终态三视角与夹爪命令做视觉审核；不是原始仿真成功标签认证。保留原始指令和真实来源任务。
- 同目录 `review_decisions.json`：审核决定、排除项及核验边界。
- 同目录 `source_identity.json`：数据/规则指纹的来源依据；大文件仅记录路径、大小和mtime，不是完整内容SHA。

`20261019_review_v2` 保留审核发布时的 `training_ready=false`。后续 `mouse/20261019_train_v1/{approved,source_identity}.json` 显式发布同47条示范的混训契约，`training_ready=true` 表示CPU接口就绪；不表示GPU训练通过，也不认证精确物理阶段。新配方仅使用train_v1，审核工具不会自动升级清单。

生成过程、候选清单、HTML和图片保存在 `artifacts/data_audits/<目标物>/<批次>/`（被Git忽略）；说明及结论保存在 `docs/auto_learning/`。审核清单发布新版本时使用新批次目录，保留旧版本，不覆盖已使用的清单。不根据local_idx保存可跨数据集使用的样本身份。
