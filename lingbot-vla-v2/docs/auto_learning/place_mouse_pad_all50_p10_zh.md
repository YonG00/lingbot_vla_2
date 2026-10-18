# 鼠标50条全量拟合实验

本实验把 `place_mouse_pad` 原来的40条train和10条val合并为50条训练轨迹，目标是检查鼠标抓取和放置的可学性。原manifest不修改，运行时通过默认关闭的 `train_all_episodes` 开关派生训练视图。

## 配方

| 项目 | 本轮设置 |
|---|---|
| 起点 | 原始 step9680，新优化器/LR/步号 |
| 训练数据 | episode 800–849共50条，5970+1496=7466帧 |
| 步数/批量 | 新增500步，micro20/GAS1/GBS20 |
| 模型范围 | ViT和其余模型全量训练，无LoRA |
| EE权重 | 位置10、姿态0.8、夹爪1.5 |
| 低噪FM/连续窗口 | 开启，窗口仍2倍，不调左右臂比例 |
| 拟合监控 | step0及每100步评固定800/801/802/803四条训练轨迹，复用原GMSE和EE计算 |
| 存档 | 250/500各一份BF16 HF，无DCP，无自动删档 |

全部50条仍为左臂19条、右臂31条。扩大训练集不会自动平衡左右臂；本轮保留自然分布。4条监控轨迹实际均参与训练，不能再称为独立验证或泛化测试。

## 启动

在服务器实验目录，默认只CPU预览，不创建run、缓存或启动训练：

```bash
cd /data/experiments/learn_skill_mix_task/lingbot-vla-v2
bash experiment/robotwin/recipes/20261018_place_mouse_pad_all50_9680_ee_lnfm.sh
```

代码同步并完成机器预检后，正式训练且成功核验最终HF才关机：

```bash
bash experiment/robotwin/recipes/20261018_place_mouse_pad_all50_9680_ee_lnfm.sh --execute --shutdown
```

不关机用 `--execute --no-shutdown`；最大新增步数可用 `--steps 1000`，其他常用参数为 `--micro`、`--gas`、`--gradient-checkpointing`、`--ee-position-weight`。默认起点仍是原始9680；若显式更改 `--model`，应同时检查预览中的实际source_model，不可根据run名称推断来源。旧40/10入口保持默认位置权重3，便于回退。

输出、启动日志和缓存使用当前日期时间，不覆盖旧实验：

```text
/data/outputs/al_mouse_pad_all50_p10_9680_ee_lnfm_add500_<日期时间>/
/data/outputs/skill_launches/<run名>.log
/data/al_cache/<run名>/target_change_window/place_mouse_pad.json
```

样本窗口预计算也必须覆盖50条。40条旧缓存会因episode集合不一致被拒绝，新run独立生成缓存即可。数据对拍继续检查实际dataset local_idx、episode白名单、完整帧数、任务与指令；它不证明视频演示质量。

## 指标口径与回滚

不调用active_val/scout/confirm，不加载成品模型的原val阈值表，不做PASS、DEFER rescue、复习、任务扫描或旧池导入；有限最大步数负责结束。事件与TensorBoard使用 `task/place_mouse_pad/active_train_gmean_mse`、`active_train_ee_*`；`active_train`是历史tag名称，实际用途为train_monitor。训练GMSE不与原独立val的成品数值作倍率比较。

仪表盘的紧凑/宽视图显示 `train G` 和 `末端train`，不再显示val、倍率或“未过拟合”。EE训练三项loss仍取对应训练步未加权值，单独与开环完整去噪的EE拟合误差区分。效果判断留给闭环抓取、抬升、放置成功率。

本轮同时改变数据范围和位置权重，属于组合探索；即便闭环改善，也不能单凭这一次归因于位置权重。没有重配左右臂采样，也没有更改窗口识别算法。

回滚使用原 `20261018_place_mouse_pad_9680_ee_lnfm.sh`：默认40/10划分、EE 3/.8/1.5。开关在新进程启动读取。已经用50条训练过的checkpoint不能再把原10条称为独立验证。

## 验证边界

CPU测试及验证记录集中在 [mouse_pad_training](../../tests/mouse_pad_training/README.md)。本地测试用真实manifest、合成Parquet状态表、真实窗口预处理/采样/调度器验证接口与统计；模型训练使用替身，未验证GPU前后向或闭环成功率。服务器的实际50条新缓存覆盖率、模型加载、首次更新及HF写出仍需同步后核验。
