# Stage B0 —— 真实仓库接口适配与接入设计

> 适用版本：`YonG00/lingbot_vla_2`（LingBot-VLA-v2 post-training）
> 相关文件：`lingbotvla/auto_learning/`（新增）、`lingbotvla/utils/open_loop_validation.py`（仅做等价抽取）
> 上游设计：`robotwin_autonomous_learning_spec_v0_2.md`（Stage A Demo 已冻结，独立于本包）

---

## 1. 这个阶段解决什么问题

把已经冻结的 **Stage A Auto Learning Scheduler** 接到真实训练仓库上。
Scheduler 只认识 5 个 Protocol；本阶段的任务是**为它们各写一层尽可能薄的适配器**，
让真实组件（数据集 / 开环评测 / 模型 forward / Trainer / checkpoint）**原样复用**。

**B0 的边界**（本轮只做这些）：

- ✅ 落地 `lingbotvla/auto_learning/` 的薄接口 + Fixed Baseline 工具 + 测试
- ✅ 抽取共用的 **safe evaluation context**（`validate()` 与新 `evaluate_task()` 共用）
- ❌ **不接 Scheduler**（不实例化、不驱动状态机）
- ❌ **不改训练 sampler**，不实现 7 NEW + 3 Replay 的真实采样
- ❌ 不启动任何真实多任务自动训练
- ❌ 不重写 preprocess / open-loop evaluator / checkpoint

**验收口径**：`auto_learning` 关闭时，原训练行为**逐位不变**。

---

## 2. 当前真实调用链（代码地图）

### 2.1 训练主链

| 环节 | 坐标 | 要点 |
|---|---|---|
| 入口 | `tasks/vla/train_lingbotvla.py::main()` | `Arguments` = Model + MyData + MyTrain + Eval |
| 参数面 | `lingbotvla/utils/arguments.py` | `DataArguments`（`episode_ids_file`）；`MyDataArguments`（`norm_stats_file` / `robot_config_root` / `cameras` / `img_size`） |
| 建数据集 | `lingbotvla/data/dataset.py::build_vla_dataset` | `data_name=='multi'` → `MultiVLADataset`；否则 `VLADataset` |
| 白名单 | `vla_data/base_dataset.py::_load_episode_ids` | 裸 JSON 数组；去重 + 越界校验 + 空数组报错 |
| 🔴 索引修复点 | `vla_data/base_dataset.py::LeRobotDataset.__getitem__` | `idx`（**局部**）→ `item["index"]`（**绝对**帧号）→ `_get_query_indices(abs_idx, ep_idx)` |
| DataLoader | `data/data_loader.py::build_dataloader` | `StatefulDistributedSampler`(shuffle) + `StatefulDataLoader`；rmpad ⇒ `DynamicBatchSizeDataLoader` + `TextBatchingStrategy` |
| batch 键 | `vla_data/utils.py::FeatureTransform.apply` | `images/img_masks/state/lang_tokens/lang_masks/actions/action_is_pad/**joint_mask**/state_joint_mask/action_joint_mask`；**没有 `noise`/`time`** |
| loss | `modeling_lingbot_vla_v2.py` | `FlowMatchingV2.forward` → `losses`(B,T,D) → `LingbotVlaV2Policy.forward` 用 `joint_mask` 加权 → **`batch_mean_losses`(B,)** |
| 训练循环 | `train_lingbotvla.py` | `micro_batches = next(data_iterator)` → `model(**micro_batch)` → backward |
| 训练中评测 | `train_lingbotvla.py` | `open_loop_validator.validate(global_step)` |
| 存档 | `checkpoint/checkpointer.py` | `extra_state/extra_state_rank_{}.pt`（DCP 之外单独落盘） |

### 2.2 评测链（`lingbotvla/utils/open_loop_validation.py`）

| 层 | 职责 |
|---|---|
| `validate(step)` | 快照 → 临时改状态 → `_run()` → **finally 全恢复 + 恢复审计** |
| `_run(step)` | `torch.inference_mode()`；train-monitor + val 各跑一次；写 TB |
| `_evaluate_ids(ids, tag)` | 建子集数据集 → `per_episode_starts` 跳步 → 逐起点推理 → `aggregate_chunks` |
| `aggregate_chunks` / `per_episode_starts` | **纯函数**，可单测、无需模型 |
| `_dataset` / `_ft_for` | 按 `episode_ids_file` 建子集数据集；每集**各自**一份 `FeatureTransform` |

评测期会**临时改、finally 恢复**的 5 类状态：

1. `config.use_cache` → `True`（训练侧是 `False`；不开会导致去噪阶段**丢掉 VLM 前缀条件**）
2. `attention_implementation` → `eager`（训练侧 `flex_cached` 的 block mask 按 query 长度建）
3. 视觉塔的**预计算网格缓存**（`precompute_grid_thw` 按首次网格缓存，micro>1 必炸）
4. 逐模块 `training` 标志 / compile 开关
5. 图像增强开关 + `torch`/`numpy`/`python` 三套 RNG

### 2.3 三个可以直接借力的既成事实

| # | 事实 | 收益 |
|---|---|---|
| 1 | `extra_state` 已原生支持（训练循环已在里面塞了 5 个 key） | 追加 Auto Learning 状态**只需再加一个 key**，checkpointer **零改动** |
| 2 | `aggregate_chunks` / `per_episode_starts` 已是纯函数 | 评测与 baseline 都能**无需模型**单测 |
| 3 | `noise` / `time` 是 `FlowMatchingV2.forward` 的**入参** | HardnessScorer 的「固定噪声 + 固定 flow time」**不需要改模型** |

---

## 3. 三个已定决策

### 决策 ①：Fixed BaselineMSE 用 **task-global-mean + trajectory-balanced**

不采用「pooled var」也不采用「per-trajectory mean-of-var」，改为：

> 用固定 **40 条 train GT** 先算该 task 的**全局 mean action** `μ_task`；
> 再对每条 train trajectory 计算 `MSE(GT_i, μ_task)`；
> 最后对 trajectory **等权平均**。

**为什么**：既保留「task mean predictor」的含义（分母是**一个 task 级共享常数**，不是每条轨迹各自一个常数），
又与当前 evaluator 的 **trajectory-level MSE 聚合口径**一致。

精确定义见 §6。

### 决策 ②：正式代码目录用 `lingbotvla/auto_learning/`

不用 `autolearn`。Stage A Demo 保持独立，**正式包不依赖 Demo**（不 import、不 vendored）。

### 决策 ③：v0 开启时要求 `rmpad=false`

`auto_learning.enabled=true` 时**强制 `rmpad=false`**，保持固定 micro batch，
让 7 NEW + 3 Replay 的 slot 语义**准确成立**。
`auto_learning.enabled=false` 时**不改变**原训练行为（rmpad 仍由原参数决定）。

> 原因：rmpad 的 `DynamicBatchSizeDataLoader` 会按 token 数重排/装箱样本，
> 7+3 的 slot 语义会被打散。v0 不处理 dynamic batch packing，后续再扩展。

---

## 4. 目录与文件

```
lingbotvla/auto_learning/
├── __init__.py
├── ports.py          # 5 个 Protocol + DTO（Backend / TaskEntry / EvalResult / TrainRequest / TrainResult）
├── catalog.py        # TaskCatalog：读 tools/task_split.py 的 manifest
├── resolver.py       # SampleResolver：local_idx ↔ (task, episode, frame)
├── baseline.py       # Fixed Task BaselineMSE：计算 + 缓存 + 指纹
├── eval_context.py   # 从 open_loop_validation 抽出的 safe evaluation context 的薄封装
├── evaluator.py      # EvaluatorAdapter：evaluate_task(task_id, split, episode_ids)
├── hardness.py       # HardnessScorer：固定 noise/time + no_grad → per-sample L1_fm
├── trainer.py        # TrainerAdapter：接口声明（B0 不接 Scheduler、不改 sampler）
└── tools/
    └── compute_task_baseline.py   # 一次性预计算 CLI（可离线，无需权重）
```

测试：

```
tests/test_auto_learning_contracts.py        # no-model contract tests
tests/test_auto_learning_real_model.py       # real-model integration tests（默认 skip）
```

---

## 5. 五个接口的设计

### 5.1 TaskCatalog

**复用**：`tools/task_split.py` 产出的 `manifest.json` + `<task>.train_ids.json` / `<task>.val_ids.json`。

- **不重新划分**、不维护第二套 episode 划分。
- `TASK_ORDER`（50）/ `EPISODES_PER_TASK=50` / `block_id = episode_index // 50` 全部沿用
  `tools/robotwin_curriculum.py` 的常量。
- manifest 已带 `sha256_train` / `sha256_val` ⇒ 直接拿来做缓存指纹与审计。
- `TaskEntry{name, train_ids, val_ids, n_train, n_val, sha256_train, sha256_val}`

```python
catalog = TaskCatalog.from_manifest("/data/train/task_splits/manifest.json")
catalog.entry("click_bell").ids_for("train")   # → 40 条
catalog.entry("click_bell").ids_for("val")     # → 10 条
```

### 5.2 SampleResolver

**复用**：`VLADataset` 底层 `LeRobotDataset` 的 `hf_dataset['index']`（**绝对**帧号）与 `hf_dataset['episode_index']`。

- 🔴 **不自行重新编码 sample id**（Stage A Demo 的 `task_index*1e6 + traj*1e3 + frame` 那套**不带进正式代码**）。
- 直接以 **absolute frame index / episode id / task id** 建立稳定映射。
- 复用**已经修好的** local-index → absolute-index 逻辑：`base_dataset.py::LeRobotDataset.__getitem__`
  里 `abs_idx = int(item["index"])`；resolver 只**读列**，不重实现。

```python
resolver.local_to_ref(i)   # → SampleRef(task="click_bell", episode=37, frame=1832)
resolver.ref_to_local(...) # → i
resolver.episode_map()     # → np.ndarray，与 open_loop_validation._episode_index_map 同源
```

### 5.3 EvaluatorAdapter

**复用**：`OpenLoopValidator`（含 `_evaluate_ids` / `aggregate_chunks` / `per_episode_starts`）。
**不重写** `sample_actions()` glue / preprocess / chunk aggregation / metric pipeline。

🔴 **不裸调用 `_evaluate_ids()`** —— 必须走 §7 的 safe evaluation context。

```python
res = adapter.evaluate_task(task_id="click_bell", split="val", episode_ids=None)
res.mse            # 官方口径（按轨迹聚合再平均）
res.nmse           # res.mse / baseline.mse
res.per_traj_mse   # 逐条轨迹
res.per_traj_ids
res.eval_seconds
res.baseline       # 固定分母（来源与指纹）
```

### 5.4 HardnessScorer

**复用**：`LingbotVlaV2Policy.forward` 的 `batch_mean_losses`，以及 `FlowMatchingV2.forward` 已支持的 `noise` / `time` 入参。

```python
scorer.score(items) -> np.ndarray   # (B,) per-sample L1_fm
```

- `torch.no_grad()` + `model.eval()`
- `noise`：独立 `torch.Generator(seed)` 生成，**固定**
- `time`：`torch.full((B,), t0)`，**固定**
- `loss_type="L1_fm"`
- ⚠️ 必须传 `joint_mask`（否则落到 `losses.mean` 分支，口径不同）

**B0 只做接口 + deterministic test，不改完整训练 sampler。**

### 5.5 TrainerAdapter

B0 只**确认抽象可行性**并声明接口：

```python
trainer.train_steps(req: TrainRequest, num_steps: int) -> TrainResult
```

并明确**以后**怎样接收外部提供的 NEW + Replay sample indices。

**最小改造方案（B1 再做）**：替换 sampler，不动 collator / model / 循环结构。

- 新增 `AutoLearnSampler(Sampler[int])`：每步产 10 个 local_idx（7 NEW + 3 OLD）
- 必须实现 `state_dict` / `load_state_dict`（`StatefulDataLoader` 依赖）
- 影响范围：`build_dataloader(..., sampler=None)` 加一个透传参数；训练循环在 `auto_learning.enabled` 时传入
- 前置：决策 ③ 的 `rmpad=false`

### 5.6 Checkpoint / Resume

**不改 checkpointer**。训练循环已在 `state["extra_state"]` 里放 5 个 key，追加一个即可：

```python
state["extra_state"]["auto_learning"] = {
    "registry": ...,          # 任务表
    "round": ...,             # scheduler round
    "current_task": ...,
    "attempt": ...,
    "sampler_state": ...,     # 采样器状态 / RNG
    "pass_snapshot": ...,     # PASS sampling snapshot
    "baseline_fingerprint": ...,   # 固定分母指纹（防止 resume 后换了分母）
}
```

---

## 6. Fixed Task BaselineMSE（口径精确定义）

### 6.1 计算步骤

输入：该 task 的**固定 train split**（manifest 的 `train_ids`，默认 **40 条**）。

```
1) 用与 evaluator **完全相同**的路径收集 GT chunk：
     ep_map = _episode_index_map(ds)
     starts = per_episode_starts(ep_map, chunk_size)
     for local_idx in starts:
         item    = ds[local_idx]
         gt_phys = ft.unapply(dict(item))            # 反归一化 → 物理量
         keys    = _pick_action_keys(ft, gt_phys, gt_phys)
         gt      = concat([_as_frames(gt_phys[k], k) for k in keys], axis=1)   # (N, D)
     按 episode 分组 → {ep_i: [gt_chunk, ...]}

2) μ_task = mean(所有 train chunk 的所有帧、所有维)        # shape (D,)

3) 对每条 train trajectory i：
     gts_i = concat(ep_i 的所有 chunk, axis=0)              # (N_i, D)
     mse_i = mean((gts_i - μ_task)²)                        # 帧 × 维

4) baseline_mse = mean_i(mse_i)                            # 轨迹等权
```

### 6.2 与 evaluator 的一致性保证

第 3–4 步**等价于**直接调用已验证的纯函数：

```python
aggregate_chunks([(ep_i, gts_i, broadcast_to(μ_task, gts_i.shape)) for ...])["mse"]
```

⇒ baseline 与 evaluator 的 `mse` **共用同一个聚合函数**，
`action space`（`_pick_action_keys`）、`normalization`（`ft.unapply`）、
`valid region`（`per_episode_starts` 跳步 + chunk 拼接）**完全一致**；
mask 侧两者都不对 `action_is_pad` 做额外处理。

### 6.3 与两个被否决方案的关系

由全方差公式 `Var_pooled = E_i[Var_i] + Var_i[E_i]`：

| 方案 | 含义 | 问题 |
|---|---|---|
| `mean_baseline_mse`（现有 `r2` 的分母） | **pooled var**：每维一个全局常数，按**帧**加权 | 长轨迹（chunk 多）主导 |
| `mean_baseline_mse_per_traj` | `E_i[Var_i]`：每条轨迹**各自**输出自己的常数 | 不是「task mean predictor」 |
| **本方案（选定）** | `μ_task` 是 **task 级共享常数**；聚合**按轨迹等权** | — |

> ⚠️ 因此 **NMSE 与旧的 `1 - r2` 不可直接比较**：分母的数据来源（train 而非 eval 集）
> 与统计量（共享常数 vs 每维全局常数）都不同。

### 6.4 缓存与指纹

- 落盘：`<baseline_dir>/task_baseline.json`（每个 task 一项）
- 指纹字段（任一变化 ⇒ 拒绝复用缓存）：
  `dataset_root` · `sha256_train` · `norm_stats_file`（内容哈希） · `cameras` · `joints`
  · `chunk_size` · `img_size` · `stride 口径`
- **只预计算一次，不随模型更新。**

### 6.5 是否依赖模型

计算只用 `ft.unapply` + 纯函数聚合 ⇒ **不需要模型权重、不需要 GPU**。
但 `build_vla_dataset` 需要 `model_config` 与 `processor`：

- 离线 CLI 用 `AutoConfig.from_pretrained(model_path)` + `AutoProcessor.from_pretrained(...)` 构造
- 在线（训练中）直接用现成的 `model.config` / `processor`

> 注意：`ft.unapply` 需要 `item`，而 item 的构造会**解码视频**。
> B0 接受这个一次性开销（每 task 40 回合 × 2 chunk）；后续如需提速，
> 再加「GT-only 数据集（跳过视频解码）」的开关，但必须重新验证口径一致。

---

## 7. Safe evaluation context（共用）

### 7.1 问题

`validate()` 现在把 5 类临时状态的 **snapshot → 强制 → finally 恢复 → 审计** 全写在方法体里。
新的 `evaluate_task()` 如果直接调 `_evaluate_ids()`，会**跳过**这一整套 ⇒ 污染训练状态
（`use_cache` / attention / 网格缓存 / RNG 都会残留）。

### 7.2 做法：抽成一个共用上下文

在 `lingbotvla/utils/open_loop_validation.py` 里新增：

```python
@dataclass
class EvalSetupReport:
    rng_snap: dict
    train_flags: list
    compile_flag: Any
    use_cache_saved: list
    attn_saved: list
    visual_cache_saved: dict | None

@contextmanager
def safe_eval_context(*, model, logger, ft_aug_registry=None,
                      on_setup: Callable[[EvalSetupReport], None] | None = None,
                      seed: int = EVAL_SEED) -> Iterator[EvalSetupReport]:
    """快照 → 强制 eager/use_cache/清网格缓存 → yield → finally 全恢复 + 审计。"""
```

- `validate()` 改为 `with safe_eval_context(...)`，原有「只打一次」的日志挪进 `on_setup`
- `OpenLoopValidator` 新增公开方法 `evaluate_ids(ids, tag)`：
  在 safe context 内 `torch.inference_mode()` 调 `_evaluate_ids`，**不写 TB**，失败**抛出**
- `EvaluatorAdapter.evaluate_task()` 只负责：解析 ids → 计时 → 调 `evaluate_ids` → 除以固定分母

### 7.3 行为等价性

- `validate()` 的语义**逐条保持**：body 异常被吞（记日志）、审计异常**向上抛**
- 用 `with ctx: try: body except: log` 的写法即可复刻（审计在 `__exit__`，不受 inner `except` 影响）
- **回归证据**：`tools/open_loop_selfcheck.py`（9/9）、`tools/open_loop_parity_check.py`（5/5）
  必须在改动前后**逐值一致**

### 7.4 关于 `_iter_items` 的等价抽取

为让 baseline 与 evaluator 共用同一条 GT 收集路径，把 `_evaluate_ids` 里
「建集 → 跳步 → 取 item → `unapply` → 选键 → 拼帧」这段抽成生成器：

```python
def _iter_items(self, ids, tag) -> Iterator[Tuple[int, dict, Any, dict]]:
    """yield (local_idx, item, ft, gt_phys)；**不做推理**。"""
```

- `_evaluate_ids` 消费它 + 推理（**仍然只过一遍数据集**）
- 新增 `collect_gt_chunks(ids, tag)` 也消费它，**不做推理**
- 口径关键逻辑**只有一份实现**

---

## 8. 需要改动的现有文件

| 文件 | 改动 | 风险 | 守卫 |
|---|---|---|---|
| **新增** `lingbotvla/auto_learning/**` | 见 §4 | 零侵入 | 新测试 |
| `lingbotvla/utils/open_loop_validation.py` | 抽 `safe_eval_context` / `_iter_items`；`validate()` 改为使用它；新增 `evaluate_ids` / `collect_gt_chunks` | 中（**等价重构**） | `open_loop_selfcheck` 9/9 + `open_loop_parity_check` 5/5 逐值一致 |
| `tasks/vla/train_lingbotvla.py` | `auto_learning.enabled` 时传 sampler / 追加 `extra_state`（**B1 再做**） | — | disabled 回归 |
| `MyTrainingArguments` | 加 `auto_learning_*` 开关（**B1 再做**） | — | — |
| ~~`checkpointer.py`~~ | **不改** | — | — |
| ~~preprocess / open-loop evaluator 逻辑~~ | **不改** | — | — |

---

## 9. 测试方案

**分层原则**：能不用模型的绝不用模型；需要真实权重/checkpoint 的单独归类，**默认 skip**。

### 9.1 no-model contract tests（`tests/test_auto_learning_contracts.py`）

| # | 测试 | 怎么测 |
|---|---|---|
| 1 | task/episode 划分正确 | 读 manifest：`block_id = ep // 50`、每任务恰好 50 回合、`train ∩ val = ∅`、`train ∪ val = 全部` |
| 2 | train / val 无泄漏 | 两个白名单集合不相交；建子集数据集后 `hf_dataset['episode_index']` 只含白名单 |
| 3 | sample index 映射稳定 | `resolver.local → (task, episode, frame)` 与 `hf_dataset` 的 `index` / `episode_index` 列**逐行对拍**；重复构造两次结果一致 |
| 4 | fixed baseline 可复现 | 同输入 ⇒ `baseline_mse` 逐位一致；改 `norm_stats_file` / `sha256_train` ⇒ 指纹变化、拒绝复用缓存 |
| 5 | baseline 口径 = 轨迹等权 | 用手造 chunk 断言 `baseline == mean_i(mean((gts_i - μ)²))`，且 `μ` 是 task 级共享常数 |
| 6 | 纯函数等价 | `per_episode_starts` / `aggregate_chunks` 的既有断言仍通过（防重构破坏） |
| 7 | auto_learning disabled 时零副作用 | 关闭开关 ⇒ 不建任何 adapter、不注册 sampler、不写 `extra_state`（B1 接循环后补齐端到端版） |

### 9.2 real-model integration tests（`tests/test_auto_learning_real_model.py`，**默认 skip**）

需要真实 checkpoint，通过环境变量开启（例如 `AL_TEST_MODEL_PATH=<hf_ckpt>`）。

| # | 测试 | 怎么测 |
|---|---|---|
| R1 | open-loop adapter 与现有 evaluator **对拍** | 同一 ckpt、同一 ids：`evaluate_task(...).mse` 与 `OpenLoopValidator.validate()` 的 `val.mse` **逐值相等** |
| R2 | hardness 固定 noise/time 可复现 | 同一 batch 跑两遍 ⇒ `batch_mean_losses` 逐位一致；换 seed ⇒ 值变 |
| R3 | safe context 恢复审计 | 评测前后逐项比对 `use_cache` / attention / 网格缓存 / 模块 training 标志 / 三套 RNG |
| R4 | baseline 与 evaluator 的 GT 口径一致 | 用同一 ids 收集 GT，断言 baseline 路径与 evaluator 路径的 `action_keys` / `(N,D)` / 拼接结果**逐位相同** |

> 说明：R1 / R2 **必须真实模型**（R1 要 `sample_actions`，R2 要 `forward` 的 flow-matching 路径），
> 因此**不属于** no-model contract tests。

### 9.3 怎么跑

```bash
# no-model（本地即可，无需 GPU；也不需要 torch）
python -m pytest tests/test_auto_learning_contracts.py -q

# real-model（需 GPU + ckpt）
AL_TEST_MODEL_PATH=/data/outputs/.../hf_ckpt \
AL_TEST_CONFIG=/data/outputs/.../lingbotvla_cli.yaml \
AL_TEST_MANIFEST=/data/train/task_splits/manifest.json \
python -m pytest tests/test_auto_learning_real_model.py -q -s
```

> ⚠️ `tests/test_disk_guard.py`（既有）需要 torch，本地无 torch 时会在**收集阶段**报错 ——
> 与本次改动无关；单独跑 `tests/test_auto_learning_contracts.py` 即可。

---

## 9bis. 实施记录（B0 落地结果）

### 已交付

| 项 | 状态 |
|---|---|
| `lingbotvla/auto_learning/`（10 个文件） | ✅ 已落地 |
| `docs/stage_b0_adapter_design.md` | ✅ 本文 |
| no-model contract tests | ✅ **35 passed**（本地、无 torch、0.9s） |
| real-model integration tests | ✅ 已写；无 ckpt 时整模块 **skip**（1 skipped） |

### `open_loop_validation.py` 的等价抽取（**唯一被改动的既有文件**）

| 抽取物 | 说明 |
|---|---|
| `safe_eval_context` + `EvalSetupReport` | 原 `validate()` 里的 snapshot → 强制 → finally 恢复 → 审计，改为共用上下文 |
| `pick_action_keys` / `assemble_chunk` | 原 `_pick_action_keys` 与 `_evaluate_ids` 内层的取帧/校验/拼接，改为模块级纯函数 |
| `collect_gt_chunks` / `evaluate_ids` | 两个新公开入口（前者给 baseline，后者给 adapter） |
| `__init__` 支持 `model=None` + `model_config=` | 离线「只收 GT」模式**不需要权重** |

**等价性证据（改动前后逐项比对）**

1. `validate()` 里状态管理调用的**序列与内容完全一致**（AST 提取比对）：
   `_rng_snapshot → _module_training_flags → _force_use_cache → _force_eager_attention →
   _visual_grid_cache_clear → _restore_use_cache → _restore_eager_attention →
   _visual_grid_cache_restore → _module_training_restore → _rng_restore → _audit_restore → _audit_strict`
2. 四个纯函数 `aggregate_chunks` / `per_episode_starts` / `_as_frames` / `_episode_index_map`
   **逐字未改动**
3. 异常语义保持：body 异常被吞（记日志）、审计异常**向上抛**

**仍需在 GPU 机上跑的最终守卫**（本地缺 torch/lerobot，跑不了）：

```bash
python tools/open_loop_selfcheck.py      # 期望 9/9
python tools/open_loop_parity_check.py   # 期望 5/5
```

---

## 10. 风险点

| # | 风险 | 缓解 |
|---|---|---|
| 1 | 🔴 `OpenLoopValidator` **只支持单卡**（`world_size > 1` 构造即抛） | B0 按单卡设计；多卡留后续 |
| 2 | 🔴 评测期临时改 5 类状态，裸调 `_evaluate_ids` 会污染训练 | **强制走 safe context**；R3 审计 |
| 3 | 🔴 重构 `open_loop_validation.py` 可能改变既有评测数值 | 改动前后跑 `open_loop_selfcheck` 9/9 + `open_loop_parity_check` 5/5，**逐值一致**才放行 |
| 4 | ⚠️ 评测成本：一次 ≈ 2 ×（5+10 回合）× 2 chunk ≈ **30 次前向** | B0 实测单次耗时，再定 scout / confirm 频率 |
| 5 | ⚠️ `batch_mean_losses` 走 `joint_mask` 分支；漏传 `joint_mask` 口径不同 | hardness 调用强制带 `joint_mask` |
| 6 | ⚠️ rmpad 与 7+3 slot 冲突 | 决策 ③：v0 强制 `rmpad=false` |
| 7 | ⚠️ baseline 与旧 `1 - r2` 不可比 | 文档 + TB tag 明确标注口径；缓存里记指纹 |
| 8 | ⚠️ `MultiVLADataset` 白名单**只支持单数据集清单** | 单任务数据集天然满足 |
| 9 | ⚠️ 本机 clone（GitHub `main`）可能落后于 `/data/code` | 动手前逐文件 md5 双向比对 |
| 10 | ⚠️ baseline 预计算需解码视频（一次性） | 接受；后续可加 GT-only 快路径，但须重新验口径 |

---

## 11. 不做什么（明确的边界）

- ❌ 不接 Scheduler、不驱动状态机
- ❌ 不改训练 sampler、不实现 7 NEW + 3 Replay
- ❌ 不启动真实多任务自动训练
- ❌ 不重写 preprocess / open-loop evaluator / metric pipeline
- ❌ 不重写 checkpoint 系统（只追加一个 `extra_state` key）
- ❌ 不在正式代码里重新编码 sample id
- ❌ 不处理 dynamic batch packing（v0 要求 `rmpad=false`）
- ❌ 不做多卡评测

---

## 12. 下一步

B0 通过（no-model tests 全绿 + 既有评测回归逐值一致）后进入 **B1**：
`AutoLearnSampler`（7+3）+ 训练循环接线 + `extra_state` 持久化 + disabled 端到端回归。
