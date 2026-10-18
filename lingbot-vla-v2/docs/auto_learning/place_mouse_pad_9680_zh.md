# 原始 step9680：鼠标放置单任务，500步

启动文件为 `experiment/robotwin/recipes/20261018_place_mouse_pad_9680_ee_lnfm.sh`，Python实现同名 `.py`。任务配置为 `configs/auto_learning/place_mouse_pad_ee_lnfm.json`。本次只准备配方，不默认启动训练。

## 训练内容

- 完整BF16权重直接来自 `/data/outputs/al_v79_poolfix_7300/checkpoints/global_step_9680/hf_ckpt`，不经过stand续训模型。来源训练配置为同一run的 `lingbotvla_cli.yaml`，保留数据布局与归一化。
- 仅 `place_mouse_pad`：manifest中episode800–849，共40条train（5970帧）、10条独立val（1496帧）；每次开环各评4条train_monitor、4条active_val。数据白名单与指令 `mouse` 校验同时生效。
- 该任务只操作鼠标，但包含3个鼠标款式、随机位置/朝向、左右臂及不同颜色鼠标垫，不是固定一个鼠标实例。任务还包含放置位置与方向约束，不等同于stand的自由朝向放置。
- 新增500步、micro20/GAS1/GBS20；100步一次独立验证，250/500保存两份BF16 HF，不保存DCP，不运行删除监控；新优化器、LR与步号。
- 全量训练：`freeze_vit=false`、`freeze_vision_encoder=false`、`train_expert_only=false`、`use_lora=false`。独立来源配置快照在传给子启动器前显式关闭所有冻结入口，不修改原run文件。
- 低噪FM、EE loss与EE val开启；位置/姿态/夹爪权重3.0/0.8/1.5。默认关闭gradient checkpointing，沿用上一轮GBS20全量配置；`--gradient-checkpointing`可打开。
- 连续抓取窗口开启，窗口2倍权重、半径0.25秒、闭合后2秒内配对同侧腕部运动。为鼠标任务独立生成缓存，不能复用stand的帧号与采样概率。
- 关闭任务扫描、PASS转换、复习和旧池导入；相对动作表示不启用。

## 使用方法

在实验目录运行，默认只做CPU预检和打印计划，不创建训练run、缓存或启动GPU：

```bash
cd /data/experiments/learn_skill_mix_task/lingbot-vla-v2
bash experiment/robotwin/recipes/20261018_place_mouse_pad_9680_ee_lnfm.sh
```

正式启动，正常结束并核验最终HF后关机：

```bash
bash experiment/robotwin/recipes/20261018_place_mouse_pad_9680_ee_lnfm.sh --execute --shutdown
```

不自动关机用 `--no-shutdown`，默认关闭。调整训练步数/批量/梯度检查点可直接传参：

```bash
bash experiment/robotwin/recipes/20261018_place_mouse_pad_9680_ee_lnfm.sh \
  --steps 1000 --micro 20 --gas 1 --gradient-checkpointing
```

每次自动生成日期时间命名的run、日志和独立缓存。输出在 `/data/outputs/al_mouse_pad_9680_ee_lnfm_add500_<日期时间>/`，日志与状态位于 `/data/outputs/skill_launches/`；`--run-name`可以固定复现名，重名拒绝覆盖。

可用 `--model`、`--train-config`、`--dataset-root`、`--qwen3vl` 修改实际资产位置；权重与来源配置的职责不同，不能把stand模型路径填给本次step9680对照。完整参数使用 `--help`。

## 验证边界

CPU契约与命令记录集中在 `tests/mouse_pad_training/`。本地验证模拟了原始配置冻结视觉/开启LoRA的情况，再调用真实子启动器重建配置，确认全量开关未被覆盖；不代表已在GPU观察到ViT梯度或真实鼠标数据通过对拍。

正式启动前仍需同步本次文件、运行服务器预览、检查新窗口报告/任务对拍、确认GPU空闲和两份HF容量。已有750步stand训练的最终HF核验成功且曾发出关机命令；当前SSH可连接但 `nvidia-smi` 返回权限错误，未启动新训练。
