# 阶段 1 对照组：只冻结视觉编码器（ViT）训练 + 评测

## 1. 这是什么 / 解决什么问题

阶段 1 的实验组（`phase1_train_then_eval.sh`）用 `train_expert_only=true`，**整个 Qwen3-VL backbone 被冻住**，只有动作专家在学，闭环评测拿到 **0/12 = 0%**。

0% 有两种可能的原因，靠单跑一次分不出来：

- **A. 冻结范围的问题** —— backbone 停留在 base 的预训练特征分布上，没适配 RoboTwin 的相机/机械臂，专家在「没被适配过的特征」上学不出动作；
- **B. 训练预算的问题** —— 779 步 = 1 epoch = 87K 样本，本来就太少，换任何配置都学不出来。

本脚本是**对照组**：把冻结范围换掉，其余一切照抄实验组，这样两个原因就能分开。

## 2. 对照设计

| | 实验组 `phase1_L1` | 对照组 `phase1_L1_vit_frozen`（本脚本） |
|---|---|---|
| `train_expert_only` | `true` | **`false`** |
| `freeze_vision_encoder` | `false` | **`true`** |
| 冻住了什么 | 整个 Qwen3-VL backbone（4.438B） | 只有 ViT（0.415B） |
| **在训参数** | **1.938 B（30.4%）** | **5.961 B（93.5%）** |
| 数据配比 / gbs / lr / optimizer / 调度 | — | **完全一致** |
| `image_augment` | `true` | **`true`**（保持一致） |
| epoch 数 | 1 | **3** |

> ⚠️ 唯一的「非冻结」差异是 epoch 数（1 → 3）。这是故意的：如果只跑 1 个 epoch，对照组和实验组**在同一个 779 步预算下**的对比点仍然存在（存档点 779 就是），而 3 个 epoch 还能回答「是不是单纯训练不够」。**要严格同预算对比，只看 `global_step_779` 那一个 ckpt 即可。**

## 3. 原理：冻结到底冻了什么

`lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py` 的 `set_requires_grad()`（L197–205）：

```python
if self.config.freeze_vision_encoder:          # 只冻 ViT
    self.qwenvl.visual.eval()
    for params in self.qwenvl.visual.parameters():
        params.requires_grad = False
if self.config.train_expert_only:              # 冻整个 backbone
    self.qwenvl.eval()
    for params in self.qwenvl.parameters():
        params.requires_grad = False
```

- `self.qwenvl` 是 `Qwen3VLForConditionalGeneration`；
- `self.qwenvl.visual` 是它的一个 **property**，返回 `self.model.visual`（即 `Qwen3VLVisionModel`）；
- 所以 `freeze_vision_encoder=true` + `train_expert_only=false` ⇒ **只冻 ViT 0.415B，LLM backbone 4.022B + 动作专家 1.787B + 动作头 0.151B 全部在训**。

另有 L209–213 的 `train()` 覆写，每个 epoch 重新把被冻模块置回 `eval()`，防止 `.train()` 把它拉回训练态。

### 参数量实测（base 的 `model.safetensors.index.json` + safetensors 头）

| 部分 | 参数量 | 占比 | 本组是否在训 |
|---|---|---|---|
| ViT（`qwenvl.model.visual.*`） | 0.415 B | 6.5% | ❌ 冻 |
| LLM backbone（`qwenvl.*` 其余） | 4.022 B | 63.1% | ✅ |
| 动作专家（`qwen_expert.*`） | 1.787 B | 28.0% | ✅ |
| 动作头 / align heads | 0.151 B | 2.4% | ✅ |
| **合计** | **6.376 B** | 100% | **在训 5.961 B = 93.5%** |

> 注：ViT 的键名是 `...qwenvl.model.visual.*`（中间多一层 `model`），按 `qwenvl.visual.` 前缀统计会得到 0，容易误判。

## 4. 前置

- 4 张 GPU（本组按 4 卡算 gbs；换卡数要同步改 `MICRO`/`GAS` 或 `--train.global_batch_size`）
- `/data/train/phases/phase1_L1.episode_ids.json`（阶段 1 数据清单）
- base 模型 `/data/models/lingbot-vla-v2-6b-base/lingbot-vla-v2-6b`
- 评测侧 `QWEN3VL_PATH`（脚本已自动 export）
- 磁盘：默认存档 3 份，单份 ≈ **71.4G**，但**真正要备的是 222G 可用**（见下）

### ⚠️ 磁盘：门槛是 disk_guard 的**逐步**判据，不是总和

单份存档 ≈ **71.4G**。权重是 **F32**（config `enable_fp32: true`，safetensors 头部实测），不是 bf16：

| 组成 | 体积 | 说明 |
|---|---|---|
| `model/` | 23.75G | DCP fp32 权重 —— 只服务续训 |
| `optimizer/` | 23.70G | Muon 20.8 + AdamW 2.9 —— 只服务续训 |
| `hf_ckpt/` | 23.75G | HF 格式权重 —— **评测唯一需要的** |
| `extra_state/` | ~0.2G | 调度器 / RNG / dataloader |

优化器状态为什么这么大：`DistributedMuon.step` 里 `state["momentum_buffer"] = torch.zeros_like(p)`
⇒ 与参数**同 dtype** = fp32 = 4 B/参数；AdamW 两个状态 = 8 B/参数。
本组可训 5.961B（实验组只有 1.938B ⇒ `optimizer/` 仅 7.2G、单份 54.9G，与实测 55G 吻合，误差 0.2%）。

**`disk_guard` 的判据是 `required = max_used × margin`，而 `max_used` 是运行期最大值**（≈ 单份全量），
所以存第 k 份**之前**就要求 `可用 ≥ 单份 × 1.1`：

```
能存下 N 份  ⇔  可用 ≥ (N-1) × 单份 + 单份 × 1.1
```

| 份数 | 终态占用 | **最少需可用** |
|---|---|---|
| 1 | 71.4G | 78.5G |
| 2 | 142.8G | 150.4G |
| **3（默认）** | 214.8G | **222.0G** |

> 最后一行是坑：3 份的**终态**只要 214.8G，但**门槛是 222.0G** —— 那 7G 的差就是「存第 3 份前还要留 78.5G」。
> 只看总和会误判成「215G 就够」，实际第 3 份会被拦下，训练在 ~step 1558 优雅停止，**你要的 3-epoch 模型根本没产出**。

**省盘**：`hf_ckpt` 之外的 47.7G（DCP）对评测**完全无用** ——
`robotwin_multi_ckpt_eval.py` 只 glob `checkpoints/global_step_*/hf_ckpt`，
从不读 `model/` 与 `optimizer/`。用 `tools/prune_dcp.py` 剪掉旧存档的 DCP：

| 策略 | 最少需可用 | 终态占用 |
|---|---|---|
| 不剪 | 222.0G | 214.8G |
| 剪 DCP，`--keep-last 1`（保留最新一份，续训能力不丢） | **174.3G** | 119.5G |
| 剪 DCP，`--keep-last 0` | 126.7G | 71.9G |

## 5. 快速开始

```bash
cd /data/code/lingbot-vla-v2

# 先看评测计划（不训练、不占卡）
DRY_RUN=1 bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh

# 真跑：训练 3 epoch → 自动接评测（默认只测 clean）
bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh

# 恢复官方口径（clean + randomized）
CONDITIONS=clean,randomized bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh

# 空间不够时降 micro（保 gbs=112 不变）
MICRO=7 GAS=4 bash experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh
```

脚本用 `BASH_SOURCE` 自定位仓库根，**在任意目录执行都可以**。

## 6. 参数（全部走环境变量，不必改脚本）

| 变量 | 默认 | 说明 |
|---|---|---|
| `EPOCHS` | `3` | 训练 epoch 数 |
| `MICRO` | `14` | micro_batch_size |
| `GAS` | `2` | gradient_accumulation_steps |
| `SAVE_STEPS` | `779` | 每多少步存一次（`0` = 只存轮末）。轮末存档周期 `save_epochs` 由它自动推导，见 FAQ |
| `CONDITIONS` | `clean` | 评测条件。**默认只测 clean**，见下 |
| `TRAIN_OUT` | `/data/outputs/phase1_L1_vit_frozen` | 训练输出 |
| `EVAL_OUT` | `/data/eval_results/phase1_L1_vit_frozen` | 评测输出 |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 用哪几张卡 |
| `DRY_RUN` | 空 | 设了就只打印评测计划 |
| `TB` | 空 | 设了就后台起 TensorBoard（`TB_PORT` 默认 6006） |

### ⚠️ 评测条件默认只测 clean（两个一阶段脚本口径一致）

`CONDITIONS=${CONDITIONS:-clean}`。**randomized 默认关掉**，理由：

- 本阶段要回答的是「**换冻结范围到底有没有效果**」，不是泛化能力；
- L1 sentinel 只有 4 个任务 × 3 回合 = **12 回合**，样本本来就小；再加一路 randomized 只会把信号摊薄、把噪声放大；
- 顺带评测时间减半（7 分钟 → ~3.5 分钟）。

等 clean 上看出方向了，再用 `CONDITIONS=clean,randomized` 补官方口径。

> 实验组 `phase1_train_then_eval.sh` 同步改成了 `clean`，**两边必须一致**，否则 A/B 又多一个变量。

### ⚠️ 关于 micro / gas —— 必须满足 `micro × gas = 28`

可训参数从 1.938B 涨到 5.961B（**3.08×**），优化器状态与梯度显存同步上涨，实验组的 `micro=28` 在这个配置下会 OOM。所以要降 micro。

**但 gbs 必须保持 112**，因为：

```
arguments.py:661   dataloader_batch_size = global_batch_size // data_parallel_size
arguments.py:680   train_steps = floor(dataset_length / (dataloader_batch_size × world_size))
```

`dataloader_batch_size` 只跟 **gbs 和卡数**有关，**与 micro 无关** ⇒ 只要 gbs=112、4 卡，`train_steps` 恒为 **779/epoch**，LR 调度 horizon 恒为 `779 × EPOCHS`，**与实验组逐项可比**。

所以 `micro × gas` 恒等于 `112 / 4 = 28`：

| micro | gas | gbs | 说明 |
|---|---|---|---|
| 28 | 1 | 112 | 实验组用的，本组会 OOM |
| **14** | **2** | **112** | ← 默认 |
| 7 | 4 | 112 | OOM 时降一档 |
| 4 | 7 | 112 | 最后兜底 |

**OOM 降档顺序**：`MICRO=7 GAS=4` → `MICRO=4 GAS=7`。仍 OOM 再考虑加 `--train.enable_activation_offload true`（`arguments.py:418`，把激活卸载到 CPU；这台机器有 1007G 内存，很适合）。

> 梯度检查点 `enable_gradient_checkpointing` 默认已是 `true`（`arguments.py:394`），不需要额外传。

## 7. 输出

```
/data/outputs/phase1_L1_vit_frozen/
├── checkpoints/
│   ├── global_step_779/    # 与实验组同预算，**直接对比点**
│   ├── global_step_1558/
│   └── global_step_2337/
│       └── {hf_ckpt, model, optimizer, extra_state}
├── runs/                   # TensorBoard 事件（loss 曲线）
└── lingbotvla_cli.yaml     # 实际生效的配方快照

/data/eval_results/phase1_L1_vit_frozen/
├── summary.txt             # 总览 + 逐任务（含 Level）
└── scheduler.log
```

## 8. 怎么看结果

1. **先看 `global_step_779` 的 clean 成功率**，和实验组的 `phase1_L1_step779` 比：
   - 对照组 **> 0%** ⇒ 冻结范围确实是主因（假设 A 成立）；
   - 对照组**也是 0%** ⇒ 779 步这个预算本身就学不出来（假设 B 成立），该加 epoch 而不是改冻结。
2. **再看 779 → 1558 → 2337 的走势**：如果在涨，说明就是训练不够；如果一直 0%，说明 3 epoch 也不够。
3. **loss 曲线看趋势，选型看闭环成功率** —— loss 降不代表能完成任务（阶段 1 实验组 loss 降了 88%，成功率仍是 0%）。

## 9. FAQ

**Q：为什么 `image_augment` 跟着实验组开 `true`，不跟官方配方 `false`？**
A：对照实验只允许一个变量。实验组开的是 `true`，对照组也必须 `true`，否则又多了一个差异。要对齐官方配方是另一件事，应该单独做。

**Q：`max_steps` 为什么写 50000？会不会跑到 50000 步？**
A：不会。判定逻辑是
```
max_steps_driven = (max_steps < train_steps × num_train_epochs)
```
`50000 < 779×3=2337` 为假 ⇒ 由 epoch 数驱动 ⇒ 跑满 3 个 epoch 就停。同时 LR 调度 horizon 是 `min(2337, 50000) = 2337`，cosine 正好在 2337 步降到 `lr_min`。

**Q：为什么默认每 epoch 存一份（`SAVE_STEPS=779`）？**
A：`779` 正好是 1 个 epoch，与实验组同预算 ⇒ 可以直接比。空间紧张就 `SAVE_STEPS=1169`（2 份，终态 143G / 门槛 150G）或 `SAVE_STEPS=0`（1 份，终态 71G / 门槛 78.5G）。
**注意 1169 会丢掉 779 这个唯一同预算对比点。**

**Q：脚本里的 `--train.save_epochs` 为什么不写死 `1`？**
A：因为「按步存档」和「轮末存档」在源码里是**两个互不去重**的分支：

```
train_lingbotvla.py:1131   if args.train.save_steps  and global_step % save_steps == 0:      # 按步
train_lingbotvla.py:1253   if args.train.save_epochs and (epoch + 1) % save_epochs == 0:   # 轮末
```

`:1211` 的 `already_saved` 去重**只保护 `reached_max_steps` 那条路**，而 `max_steps=50000` 永远走不到那里。
`779 % 779 == 0` ⇒ 两个分支在 779 / 1558 / 2337 **全部命中同一个 `global_step_*` 目录**，每个 epoch 边界把 DCP 重写一遍 ——
实测第 2 遍要 **385s（779）/ 378s（1558）**，3 个 epoch 白烧 **~19 分钟**。
（HF 侧有去重，日志会打 `[async_hf] skip duplicate checkpoint`，**只有 DCP 重复写**。）

所以脚本按下面的规则推导 `SAVE_EPOCHS`：**步存档覆盖到末步就关掉轮末存档，否则只在最后一个 epoch 末补一份**。

```bash
SAVE_EPOCHS=$EPOCHS
if [ "$SAVE_STEPS" -gt 0 ] && [ $(( TOTAL_STEPS % SAVE_STEPS )) -eq 0 ]; then
    SAVE_EPOCHS=0
fi
```

⇒ 份数恒为 `N = floor(2337 / SAVE_STEPS) + (2337 % SAVE_STEPS ? 1 : 0)`，`SAVE_STEPS=0` 时 `N=1`：

| `SAVE_STEPS` | 推导出的 `save_epochs` | 存档点 | 份数 |
|---|---|---|---|
| `0` | `3` | 2337 | 1 |
| `1169` | `3` | 1169, 2337 | 2 |
| **`779`（默认）** | **`0`** | **779, 1558, 2337** | **3** |
| `584` | `3` | 584, 1168, 1752, **2336, 2337** | 5 |

> `584` 那行体现公式的固有行为：`2337 % 584 = 1` ⇒ 轮末再补一份，于是 `2336` 和 `2337` 挨在一起。
> 任何**不整除**的档位都会有这个现象，不是 bug。

**Q：能中途 `Ctrl-C` 再续训吗？**
A：改 `--train.enable_resume true` 并指向同一 `TRAIN_OUT` 即可续训 DCP。注意 `enable_resume` **只管续训 DCP**，初始权重只来自 `--model.model_path`。
另外它**只认 step 最大的那一份**（`train_lingbotvla.py:677-687`），所以更早存档的 DCP 是结构性无用的 ——
用 `tools/prune_dcp.py --keep-last 1` 剪掉它们不影响续训能力。

**Q：拿这个 ckpt 当起点开下一阶段，需要 DCP 吗？**
A：**不需要。** 初始权重只来自 `--model.model_path`（`train_lingbotvla.py:395-404`
`build_foundation_model(weights_path=...)`），只需 `hf_ckpt`。
活体反证：`lingbot-vla-v2-6b-base` 根本没有 DCP，而实验组就是拿它当起点训的。

## 10. 边界

- 本脚本**不改** `phase1_train_then_eval.sh`，两个文件独立；实验组的结果和目录不受影响。
- 输出目录 `/data/outputs/phase1_L1_vit_frozen` 与实验组的 `/data/outputs/phase1_L1` **不同**，不会互相覆盖。
- 存档**无轮转、不自动清理**，每份 ≈ **71.4G**，跑之前先按上面的「逐步判据」确认可用空间（3 份需 222G）。
- 想省盘用 `python tools/prune_dcp.py --ckpt-root "$TRAIN_OUT" --interval 120 --keep-last 1`
  （常驻看门狗，剪掉旧存档里评测用不到的 DCP；3 份只需 174G 可用）。
- `micro=14` 的显存占用是**估算**，没有实测过（写这个脚本时手上没有可用的 GPU 机器）。**第一次跑建议盯着 `nvidia-smi`，OOM 就按第 6 节降档。**
- 评测链路与阶段 1 完全一致（同一个多 ckpt 调度器、同一批 L1 sentinel 任务）。
