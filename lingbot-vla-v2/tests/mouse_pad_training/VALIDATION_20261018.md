# 2026-10-18 鼠标配方验证

实现提交 `aba61f0`，分支 `learn_skill_mix_task`；实验服务器独立目录通过GitHub快进同步到同一提交，原有仿真脚本本地修改和资产目录保留。

- 本地与服务器CPU回归均 **33 passed、13 subtests passed**；命令见同目录README。
- 真实资产预览退出 **0**：原始 `al_v79_poolfix_7300/checkpoints/global_step_9680/hf_ckpt`、来源 `lingbotvla_cli.yaml`、Qwen tokenizer、数据meta和机器人embodiment配置均存在。HF分片头与tokenizer/归一化来源检查通过。
- 真实计划仅 `place_mouse_pad`，40条训练、10条验证；新增500步，micro20/GAS1；EE三权重3.0/0.8/1.5，低噪FM开启；`freeze_vit`、`freeze_vision_encoder`、`train_expert_only`、`use_lora`均false。
- 子启动器重建契约用冻结且开启LoRA的合成来源配置，确认training快照将四项完整训练开关设为false。该项不使用真实GPU模型，不能替代ViT梯度验收。
- 预览未创建新run/缓存/训练进程，未执行关机。当前 `/usr/bin/nvidia-smi` 无执行权限，尚未验证GPU启动、真实鼠标静态窗口覆盖及训练数据完整对拍。

辅助核对脚本末尾曾多出无效标记，导致首次预览成功后出现NameError；修正临时脚本后重新完整执行，出现 `PREVIEW_EXIT 0` 和 `CPU_PREFLIGHT_COMPLETE_NO_TRAINING`。该临时脚本不属于训练代码。

最终启动仍使用主文档中的shell入口，按需增加 `--execute --shutdown`。
