# place_object_stand step500 续训方案（2026-10-17）

## 背景

上一轮 run 为 `al_stand_ee_lnfm_9680_add500_20261017_184252`。检查其 `trainer_config.json` 后确认：

```text
train.freeze_vision_encoder = true
train.low_noise_fm_enabled = true
train.ee_fk_loss_enabled = true
train.ee_fk_position_loss_weight = 1.0
train.ee_fk_orientation_loss_weight = 0.1
train.ee_fk_gripper_loss_weight = 0.1
```

因此上一轮并不是视觉全量训练。新的续训入口增加了显式 `--no-freeze-vision-encoder`，避免再次依赖硬编码默认值。

## 本轮策略

| 项目 | 设置 |
|---|---|
| 起点 | `global_step_500/hf_ckpt` |
| 新增步数 | 500 |
| 视觉编码器 | 解冻，`freeze_vision_encoder=false` |
| 低噪 FM | 保持开启 |
| 样本窗口 | 保持 `target_change_window` |
| EE position | `3.0` |
| EE orientation | `0.8` |
| EE gripper | `1.5` |
| batch | micro=24, GAS=1, GBS=24 |
| 评测/存档 | 每100步独立 val，每250步保存 BF16 HF，不写DCP |
| PASS/复习 | 保持单任务实验关闭 |

这次先观察解冻视觉和增大 EE 监督的联合效果；它不是严格单变量实验。结果应与 step500、step9680 用相同任务、相同 clean/unseen、相同 scene seed 和推理参数比较。

## 启动

启动文件：`experiment/robotwin/recipes/20261017_stand_ee_continue500_fullvision_w3_o08_g15.sh`。

默认只打印预览。每次调用时自动用当前日期时间生成唯一的 `RUN_STAMP`、`RUN_NAME` 和 `OUT`，避免覆盖历史实验。启动日志、job/status 文件放在独立的 `/data/outputs/skill_launches`，不占用待创建的训练输出目录：

```bash
cd /data/experiments/learn_skill_mix_task/lingbot-vla-v2
bash experiment/robotwin/recipes/20261017_stand_ee_continue500_fullvision_w3_o08_g15.sh
```

默认命名格式：

```text
RUN_NAME=al_stand_ee_fullvision_w3_o08_g15_from500_add500_YYYYMMDD_HHMMSS
OUT=/data/outputs/$RUN_NAME
```

复现某次启动时可显式传入 `RUN_STAMP`、`RUN_NAME` 或 `OUT`；平时不需要手动指定路径。

确认预览中的 `freeze_vision_encoder=false`、三个 EE 权重和起点正确后再后台启动：

```bash
EXECUTE=1 bash experiment/robotwin/recipes/20261017_stand_ee_continue500_fullvision_w3_o08_g15.sh
```

正常结束后才自动关机：

```bash
EXECUTE=1 SHUTDOWN=1 bash experiment/robotwin/recipes/20261017_stand_ee_continue500_fullvision_w3_o08_g15.sh
```

## 关于接触窗口 val

现有末端 val 是整条轨迹的平均误差。接触窗口 val 是在训练数据已经识别出的“夹爪闭合/腕部目标变化”前后约0.25秒窗口内，单独计算位置、姿态和夹爪误差。

它不是新的训练 loss，也不是替换现有末端 val；用途是回答“接触前最后几厘米是否对准”。当前窗口缓存统计：训练集5561帧中2772帧进入窗口，窗口概率质量约66.5%，共识别308个事件。全轨迹 val 从50.86mm降到32.65mm，但闭环仍失败，说明需要把最后接触阶段单独监控，不能只看全轨迹均值。
