# 阶段训练 → 自动评测 一键脚本 使用文档

> 对应脚本（都在 `experiment/robotwin/`）：
> - `phase1_train_then_eval.sh` —— 阶段 1（L1），从 base 起
> - `phase2_from_base_train_then_eval.sh` —— 阶段 2（L1+L2），**从 base 起**
>
> 依赖 `docs/phase_training_guide.md`（课程数据配比）与 `docs/multi_ckpt_eval_guide.md`（评测调度器）。

---

## 1. 这个功能解决什么问题

课程训练原本是两条独立命令：先训练，人盯着跑完，再手动起评测。中间有三个坑：

1. **「训练正常结束」≠「checkpoint 可用」**。训练退出前会 drain 异步 HF 保存，但 HF 保存失败是
   *best-effort 吞异常*（`async_hf_checkpoint.py:321-325`），最后一份可能是残缺的。
   靠肉眼判断不靠谱，得靠调度器逐目录验结构 + 字节。
2. **目录名容易混**。`phase2_L1_L2` 到底是「base 直训阶段 2」还是「base→阶段1→阶段2」？
   两者权重完全不同，混在一起会互相覆盖。
3. **磁盘**。每次存档 ≈ 55G（`hf_ckpt` 24G + DCP ~30G，**没有轮转清理**），
   步数一涨份数就涨，很容易在训练中途撞盘。

本脚本把「训练 → 完整性校验 → 评测 → 出报告」串成**一条命令**，并把上面三个坑显式处理掉。

---

## 2. 一句话原理

```bash
bash train.sh <训练参数> && python experiment/robotwin/robotwin_multi_ckpt_eval.py <评测参数>
```

- `&&` 串联：训练**非 0 退出就不会**进入评测，不会拿半成品去评。
- 静默期保持默认 `0`：训练进程已退出，确定没有写入者，不需要「最近 120s 动过就跳过」的保守判断。
- 脚本用 `BASH_SOURCE` 自己定位仓库根，**在任意目录都能执行**，不依赖 `cd`。

---

## 3. 前置条件（必读）

### 3.1 阶段文件已生成

```bash
ls /data/train/phases/
# datasets.txt                           数据集清单（共享）
# phase1_L1.episode_ids.json             700 回合
# phase2_L1_L2.episode_ids.json         1450 回合
# phase3_L1_L2_L3.episode_ids.json      2000 回合
# phase4_all.episode_ids.json           2500 回合
# phaseN_eval.txt                        评测任务清单（4/8/12/16 行）
```

没生成就先跑 `python tools/prepare_phase.py --phase all --gbs 112`。

### 3.2 `QWEN3VL_PATH`

脚本已自动 `export QWEN3VL_PATH=/data/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct`。
（launcher 里的默认值是占位符 `/path/to/your/checkpoints/...`，不设评测直接失败。）

### 3.3 仿真侧已重编 sm_120

评测要起 RoboTwin 仿真，`curobo` 的 5 个 `.so` 必须含 `sm_120`。
自检：`python -c "from envs.robot.planner import CuroboPlanner"`（在 RoboTwin env 里）。
换机器 / 重装 RoboTwin env 后**必须重做**，见 `stage3/sim_sm120_rebuild.sh`。

### 3.4 4 张卡空闲

训练用满 4 卡（85.6G/卡），评测每卡 32–48G，两者时间上互斥。
脚本默认 `CUDA_VISIBLE_DEVICES=0,1,2,3`。

### 3.5 磁盘预算（重要）

单次存档 ≈ **54.9G**（expert-only；与实测 55G 吻合）：

| 组成 | 体积 | 说明 |
|---|---|---|
| `model/` | 23.75G | DCP fp32 权重 —— 只服务续训 |
| `optimizer/` | 7.2G | Muon 4 B/参数（本组只训 1.938B）—— 只服务续训 |
| `hf_ckpt/` | 23.75G | HF 格式权重 —— **评测唯一需要的** |
| `extra_state/` | ~0.2G | 调度器 / RNG / dataloader |

权重是 **F32**（config `enable_fp32: true`，safetensors 头部实测），不是 bf16。
**DCP 没有轮转清理**，逐份累加。

> ⚠️ **门槛不是「份数 × 单份」，而是 disk_guard 的逐步判据。**
> `required = max_used × margin`，`max_used` 是运行期最大值（≈单份全量），
> 所以存第 k 份**之前**就要求 `可用 ≥ 单份 × 1.1`：
> ```
> 能存下 N 份  ⇔  可用 ≥ (N-1) × 单份 + 单份 × 1.1
> ```
> 例：55G 一份时，2 份终态 110G 但**门槛 115G**；3 份终态 165G 但**门槛 176G**。

| 阶段 | 步数 | `save_steps` | 存档点 | 份数 | 终态占用 | **最少需可用** |
|---|---|---|---|---|---|---|
| P1 | 779 | `545`（默认） | 545, 779 | 2 | ~110G | ~115G |
| P1 | 779 | `260` | 260, 520, 779 | 3 | ~165G | ~176G |
| P2 | 1856 | `619`（默认） | 619, 1238, 1856 | 3 | ~165G | ~176G |
| P2 | 1856 | `928` | 928, 1856 | 2 | ~110G | ~115G |
| P2 | 1856 | `464` | 464, 928, 1392, 1856 | 4 | ~220G | ~231G |

> ⚠️ **`save_steps` 不能跨阶段照抄**：阶段 1 的 `545` 只有 2 份，到阶段 2（1856 步）就变成 4 份。

> 🔴 **轮末存档周期 `save_epochs` 是自动推导的，不是写死 `1`。**
> 源码里「按步存档」（`train_lingbotvla.py:1131`）与「轮末存档」（`:1253`）是**两个互不去重**的分支 ——
> `:1211` 的 `already_saved` **只保护 `reached_max_steps` 那条路**，而 `max_steps=50000` 永远走不到。
> 所以当 `save_steps` **整除**总步数时（P2 的 `928` / `464` 都整除 1856），旧写法会把
> `global_step_1856` 的 DCP **写两遍**，白等约 6 分钟（HF 侧有去重，只有 DCP 重复写）。
> 现在的规则：**步存档覆盖到末步 ⇒ 关掉轮末存档；否则只在最后一个 epoch 末补一份**。
> 上表的份数在新规则下逐行成立。

> ⚠️ **4 个阶段的 checkpoint 会叠加**（各自独立 `output_dir`，互不清理）。
> 按每阶段 1 份算，4 × 55 = 220G，加上已有的模型/环境约 115G ⇒ **跑满 4 阶段必须阶段间手动清理**。

> 💡 **省盘**：`hf_ckpt` 之外的 DCP 对评测**完全无用**（调度器只 glob
> `checkpoints/global_step_*/hf_ckpt`，从不读 `model/` 与 `optimizer/`）。
> `python tools/prune_dcp.py --ckpt-root "$TRAIN_OUT" --interval 120 --keep-last 1`
> 可把单份从 55G 剪到 24G，且保留最新一份的续训能力。
> 对照组（只冻 ViT，单份 71.4G）用同一招能从 222G 门槛降到 174G。

---

## 4. 快速开始

### 4.1 先 dry-run 看计划（不训练、不占卡、不起进程）

```bash
cd /data/code/lingbot-vla-v2
DRY_RUN=1 bash experiment/robotwin/phase1_train_then_eval.sh
```

会打印每个 checkpoint 的两个 condition、端口段、输出目录，然后退出。

### 4.2 阶段 1 真跑

```bash
bash experiment/robotwin/phase1_train_then_eval.sh
```

约 43 分钟训练 + 7 分钟评测。

### 4.3 阶段 2 从 base 真跑

```bash
bash experiment/robotwin/phase2_from_base_train_then_eval.sh
```

约 1.7 小时训练 + 30 分钟评测。

### 4.4 常用变体

```bash
# 只存 1 个 checkpoint（最省盘）
SAVE_STEPS=0 bash experiment/robotwin/phase1_train_then_eval.sh

# 换输出目录（做对照实验，互不覆盖）
TRAIN_OUT=/data/outputs/phase1_L1_try2 \
EVAL_OUT=/data/eval_results/phase1_L1_try2 \
  bash experiment/robotwin/phase1_train_then_eval.sh

# 换起点权重（例如阶段 2 基于阶段 1 续训）
MODEL_PATH=/data/outputs/phase1_L1/checkpoints/global_step_779/hf_ckpt \
TRAIN_OUT=/data/outputs/phase2_L1_L2 \
  bash experiment/robotwin/phase2_from_base_train_then_eval.sh
```

### 4.5 看训练曲线（TensorBoard）

**训练脚本不会自己起 TensorBoard** —— 它只把事件文件写到 `<TRAIN_OUT>/runs/`
（`train_lingbotvla.py:593-594`，只有 `global_rank == 0` 写）。
所以要看得自己开，脚本给了个开关：

```bash
TB=1 bash experiment/robotwin/phase1_train_then_eval.sh
```

脚本会 `setsid nohup` 起一个后台 TensorBoard（`--host 127.0.0.1 --port 6006`），
日志在 `<TRAIN_OUT>/tensorboard.log`。端口已被占用时自动跳过启动，不会起第二个。

浏览器访问要**先做端口转发**（AutoDL 上 6006 不对公网开放）：

```bash
ssh -L 6006:127.0.0.1:6006 -p <SSH端口> root@<主机>
# 然后本机浏览器打开 http://127.0.0.1:6006
```

**AutoDL 上更省事的办法**：平台自带一个常驻 TensorBoard（`/root/miniconda3/bin/tensorboard
--host 0.0.0.0 --port 6007 --logdir /root/tf-logs`，控制台「TensorBoard」按钮直达）。
把我们的 runs 目录软链进去就不用做端口转发：

```bash
ln -sfn /data/outputs/phase1_L1/runs /root/tf-logs/phase1_L1
# 然后直接用 AutoDL 控制台的 TensorBoard 按钮，无需 ssh -L
```

> 注意：平台那个 TB 固定看 `/root/tf-logs`，**不要**去动它的进程；
> 我们的脚本用 6006，两者互不冲突。

不想用脚本的话，单独敲也行（训练跑起来后随时可以）：

```bash
tensorboard --logdir /data/outputs/phase1_L1/runs --port 6006 --host 127.0.0.1
```

会看到哪些曲线：`training/lr`、`training/grad_norm`、`steptime`、
`training/future_depth_loss`、`training/future_video_loss`、`training/router_z_loss`、
`moe_summary/load_cv`，以及 `detailed_loss/*` 的分项 loss。

> ⚠️ TensorBoard 是**纯读** `<TRAIN_OUT>/runs/`，关掉它不影响训练。
> 训练结束后它仍在后台跑，不需要了用
> `pkill -f "tensorboard --logdir /data/outputs"` 收掉
> （**别用** `pkill -f tensorboard` —— 会误杀 AutoDL 平台自带那个）。

---

## 5. 参数速查（全部是环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DRY_RUN` | 空 | 非空 ⇒ 只预览评测计划，不训练、不起子进程、不初始化 CUDA |
| `SAVE_STEPS` | P1 `545` / P2 `619` | 每多少 step 存一次；`0` = 只在轮末存一次。轮末存档周期 `save_epochs` 由它自动推导（见第 3 节的 🔴），不是写死的 `1` |
| `CONDITIONS` | `clean` | 评测条件。**默认只测 clean**（randomized 关），见 5.1 |
| `TRAIN_OUT` | `/data/outputs/phase1_L1` / `..._from_base` | 训练输出目录（`<TRAIN_OUT>/checkpoints/global_step_N/hf_ckpt`） |
| `EVAL_OUT` | `/data/eval_results/...` | 评测输出目录 |
| `MODEL_PATH` | 仅 P2 脚本 | 起点权重；默认 base，可指向上一阶段的轮末 `hf_ckpt` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 卡 |
| `TB` | 空（关） | 非空 ⇒ 训练前**后台启动 TensorBoard**（`127.0.0.1:6006`） |
| `TB_PORT` | `6006` | TensorBoard 端口 |
| `QWEN3VL_PATH` | Qwen3-VL-4B 路径 | 评测侧必需，脚本已设默认 |

### 5.1 ⚠️ 评测条件默认只测 clean

`CONDITIONS=${CONDITIONS:-clean}`，**randomized 默认关掉**：

- 当前阶段要回答的是「**换冻结范围到底有没有效果**」，不是泛化能力；
- L1 sentinel 只有 4 任务 × 3 回合 = **12 回合**，样本本来就小，再加一路条件只会摊薄信号、放大噪声；
- 顺带评测时间减半。

想恢复官方口径（clean + randomized）加 `CONDITIONS=clean,randomized` 即可。
**对照组 `phase1_L1_vit_frozen_train_then_eval.sh` 同步改成了 `clean`，两边必须一致**，
否则 A/B 又多一个变量。

> 本文档第 6 节的示例输出仍是 `clean,randomized` 两列的格式（历史运行记录），
> 实际默认只跑 clean 一列。

---

## 6. 输出在哪看

### 6.1 训练

```
<TRAIN_OUT>/checkpoints/global_step_<N>/
    ├── hf_ckpt/                      ← 评测要用的（24G）
    └── （ByteCheckpoint 的 DCP 文件）  ← 续训要用的（~30G）
<TRAIN_OUT>/runs/                     ← TensorBoard 事件文件（rank0 写，见 4.5）
<TRAIN_OUT>/tensorboard.log           ← TB=1 时的 tensorboard 进程日志
<TRAIN_OUT>/../log.txt                ← train.sh 的 tee 产物
```

判断「训练是否真的从 base 起」：日志里应出现
`Starting training from scratch.`（而不是 `Load distributed checkpoint from ...`）。

### 6.2 评测

```
<EVAL_OUT>/
├── summary.txt            ← 人看的：总览 + 逐任务（带 Level）+ 产物 + 未评测
├── summary.json           ← 机器可读
├── eval_state.json        ← 增量状态（重跑自动跳过已完成的）
├── scheduler.log          ← 本次调度器日志（tee 产物）
├── _logs/<ckpt>.<cond>.log
└── <ckpt>/<cond>/         ← 单个作业的 run_dir（内含 stats.txt）
```

`summary.txt` 长这样（节选）：

```
checkpoint 3 个 | 任务 8 | 回合/任务 3 | condition clean,randomized

总览
  checkpoint       clean        randomized   总耗时
  global_step_619  62%(15/24)   58%(14/24)   14分12秒
  ...

逐任务
  任务              Level  clean        randomized
  lift_pot          L1     3/3          2/3
  adjust_bottle     L2     2/3          1/3
  --- L1 小计 ---          12/15        11/15
```

---

## 7. 工作原理

### 7.1 「从 base 起」到底由什么决定（最容易搞错）

`train_lingbotvla.py:669-720` 的逻辑：

```python
if args.train.load_checkpoint_path or args.train.enable_resume:
    ...  # 只从 <output_dir>/checkpoints/global_step_* 里挑最新的 DCP
if candidates: Checkpointer.load(cp, state, ...)
else:          logger.info_rank0("Starting training from scratch.")
```

**`enable_resume` 只管「要不要续训 DCP」，完全不参与初始权重。**
初始权重**只**来自 yaml 的 `model.model_path`
（`arguments.py:46`，help 原文：*"Path to the pre-trained model. If unspecified, use random init."*）。

所以「从 base 起」要三个条件同时满足：

| | 条件 | 脚本里怎么保证 |
|---|---|---|
| ① | 全新的 `output_dir` | `TRAIN_OUT=/data/outputs/phase2_L1_L2_from_base` |
| ② | `--train.enable_resume false` | 写死 |
| ③ | `--model.model_path` 指向 base | 显式写出（yaml 里本来就是 base，写出来是防误改） |

反过来，**想让阶段 2 基于阶段 1 续训**，只改 `MODEL_PATH`：

```bash
MODEL_PATH=/data/outputs/phase1_L1/checkpoints/global_step_779/hf_ckpt \
TRAIN_OUT=/data/outputs/phase2_L1_L2 \
  bash experiment/robotwin/phase2_from_base_train_then_eval.sh
```

（`TRAIN_OUT` 仍要换新的，否则会覆盖阶段 1 的存档。）

> 参数名 `--model.model_path` 的来历：`parse_args` 里参数名拼成 `f"{base}.{attr.name}"`，
> 根字段是 `model / data / train / eval`，所以和 `--data.train_path`、`--train.output_dir` 是同一套机制。

### 7.2 命名约定

| 目录 | 含义 |
|---|---|
| `phase1_L1` | base → 阶段 1（阶段 1 天然从 base 起，不加后缀） |
| `phase2_L1_L2_from_base` | **base → 阶段 2**（跳过阶段 1） |
| `phase2_L1_L2` | base → 阶段 1 → 阶段 2（续训链） |
| `phase3_L1_L2_L3_from_base` | base → 阶段 3（跳过 1、2） |

规则：**只有「非阶段 1 但从 base 起」的运行才带 `_from_base` 后缀**，
这样和续训链的结果永远不会互相覆盖。

### 7.3 两个预检（不阻塞，只提醒）

1. **输出目录预检**：`<TRAIN_OUT>/checkpoints` 已存在 ⇒ 列出里面的 `global_step_*`，
   提醒「`enable_resume=false` 会从 base 重训并覆盖同名目录」。
2. **磁盘预检**：读 `df` 可用空间，按 `SAVE_GB=55` 打印「份数 × 55G」，
   并单独打印 disk_guard 的**逐步门槛** `(N-1)×55 + 55×1.1`，不够就告警。

份数算法（**精确版**）。`N_SAVES` 在脚本里只算一次，磁盘预检直接复用它，避免两处各算一遍算歪：

```bash
if [ "$SAVE_STEPS" -gt 0 ]; then
    if [ $(( TOTAL_STEPS % SAVE_STEPS )) -eq 0 ]; then
        N_SAVES=$(( TOTAL_STEPS / SAVE_STEPS ))      # 末步已被步存档覆盖，轮末不再额外存
    else
        N_SAVES=$(( TOTAL_STEPS / SAVE_STEPS + 1 ))  # 轮末补最后一份
    fi
else
    N_SAVES=1                                        # 只存轮末一份
fi
```

### 7.4 为什么评测能接在训练后面自动跑

评测调度器本来就是「发现 checkpoint → 完整性校验 → 跑」的独立进程，
不需要训练侧有任何钩子（训练侧的「存完即评」钩子**并不存在**，
`AsyncHFCheckpointSaver.__init__` 把 `eval_args` 直接丢弃了）。
所以「自动评测」完全靠 `&&` 串联实现，零侵入。

---

## 8. 常见问题

### Q1 训练跑完了，评测却说「没有可用 checkpoint」

按顺序查：

1. `<TRAIN_OUT>/checkpoints/global_step_*/hf_ckpt/model.safetensors.index.json` 在不在；
2. 目录里有没有 `.*.safetensors.*` 临时残留（HF 转换中断的标志）；
3. 训练日志里有没有 `hf_save_failed` —— 异步 HF 保存失败会**优雅结束训练**但 HF 是残缺的。

调度器 `--dry-run` 会把「跳过 + 原因」打出来，先跑它。

### Q2 想接着上次的训练跑？

把脚本里 `--train.enable_resume false` 改成 `true`（脚本没做成变量，直接改一行）。
注意：`enable_resume=true` 会从 `<TRAIN_OUT>/checkpoints` 里挑**最新的 DCP** 恢复，
包括模型权重、优化器、dataloader 状态。

### Q3 磁盘不够会怎样？

不会爆盘。`disk_guard` 会在「剩余空间 < 已实测单份占用 × 1.1」时**优雅停止训练**，
当前 checkpoint 已完整保存。代价是**最后一个 checkpoint 会缺失**（例如本该有的
`global_step_779` 没了，只剩 `global_step_545`）。

处置：调大 `SAVE_STEPS`（少存几份）、或 `SAVE_STEPS=0`（只存轮末）、或先扩容。

### Q4 为什么每个阶段的 `save_steps` 不一样？

因为它是**绝对步数间隔**，而各阶段总步数差很多（779 → 1856 → 3217 → 4901）。
想让「每阶段几个 checkpoint」保持一致，就按 `ceil(总步数 / 想要的份数)` 算：

| 阶段 | 步数 | 想要 3 份 ⇒ `save_steps` |
|---|---|---|
| P1 | 779 | 260 |
| P2 | 1856 | 619 |
| P3 | 3217 | 1073 |
| P4 | 4901 | 1634 |

> `ceil` 只是起点：**只要 `save_steps` 不整除总步数，实际份数就是 `floor + 1`**（轮末补一份）。
> 上面这四档恰好都落在「不整除」一侧（`779%260=259`、`1856%619=618`、`3217%1073=1071`、`4901%1634=1633`），
> 所以份数就是想要的 3 份。反过来，若你手选一个**整除**的档位（P1 的 `779`、P2 的 `928` / `464`），
> 轮末存档会自动关闭，份数 = `总步数 / save_steps`，也不会重复写。

### Q5 评测跑失败了，重跑要重头来吗？

不用。增量状态在 `<EVAL_OUT>/eval_state.json`，**同一条命令重跑会自动跳过已完成的**。
失败的 checkpoint 不会被写进状态（`6731b7a`），所以会被重试。

### Q6 可以只用其中一个脚本吗？

可以。两个脚本各自独立，互不依赖。也可以只跑「训练」那一段 ——
脚本本质就是两条命令用 `&&` 串起来，拆开手敲完全等价。

### Q7 为什么脚本默认不启动 TensorBoard？

因为**官方训练链路本来就不起**。`train.sh` 只包了 `torchrun ... | tee log.txt`，
训练脚本本身只负责**写** `<TRAIN_OUT>/runs/`（`train_lingbotvla.py:593-594`）。
一个「训练脚本」顺带常驻一个 web 服务是副作用，所以做成 `TB=1` 显式开关，
不设时行为和不加这段代码完全一致。详见 4.5。

> 顺带一提：这台机器上 **`ss` 和 `netstat` 都不存在**，所以脚本的端口探测用的是
> bash 内建的 `/dev/tcp`（`(exec 3<>/dev/tcp/127.0.0.1/$PORT)`），零外部依赖。

### Q8 训练跑起来后还能补起 TensorBoard 吗？

能，随时。TensorBoard 是纯读 `<TRAIN_OUT>/runs/`，不影响训练：

```bash
tensorboard --logdir /data/outputs/phase1_L1/runs --port 6006 --host 127.0.0.1
```

或者直接软链进平台自带的那个（见 4.5，免端口转发）。

---

## 9. 测试

```bash
# 语法检查
bash -n experiment/robotwin/phase1_train_then_eval.sh
bash -n experiment/robotwin/phase2_from_base_train_then_eval.sh

# 只预览评测计划（不起任何子进程）
DRY_RUN=1 bash experiment/robotwin/phase1_train_then_eval.sh

# 训练还没跑过时，dry-run 应该报「没有发现可用 checkpoint」而不是崩
```

预检的份数公式与 `save_epochs` 推导已自测（多组步数 × 多组 `save_steps`）：

```
P1   (779 步, 1 epoch):     0→1 份  545→2 份  260→3 份  779→1 份（save_epochs=0）
P2   (1856 步, 1 epoch):    0→1 份  928→2 份  619→3 份  464→4 份  545→4 份（928/464 的 save_epochs=0）
对照组 (2337 步, 3 epoch):  0→1 份  1169→2 份  779→3 份（save_epochs=0）  584→5 份
```

---

## 10. 不做什么（明确的边界）

- **不默认启动 TensorBoard**。训练只写 `runs/`，要看曲线得加 `TB=1`（见 4.5）。
- **不自动清理旧 checkpoint**。DCP 没有轮转，脚本也不替你删 —— 删存档是不可逆操作，
  必须人工确认。跑多阶段前请自己算好盘。
- **不自动 push**。脚本只训练 + 评测，不碰 git。
- **不改官方 launcher**。评测一律调 `experiment/robotwin/start_robotwin_infer_and_eval.sh`，
  只通过参数注入任务清单（`--task_list_file`）。
- **不加速单个 checkpoint**。脚本只是把「训练→评测」串起来 + 允许多 ckpt 并发，
  单 ckpt 的评测耗时和官方完全一致。
- **不做多机**。只在单机 4 卡上跑。
