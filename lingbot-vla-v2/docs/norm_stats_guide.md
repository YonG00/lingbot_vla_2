# 归一化统计（norm stats）来源 使用文档

> 适用版本：`YonG00/lingbot_vla_2`（LingBot-VLA-v2 post-training）
> 相关文件：`lingbotvla/data/vla_data/base_dataset.py`、`lingbotvla/data/vla_data/utils.py`、
> 　　　　　`configs/robot_configs/robotwin.yaml`、`deploy/lingbot_vla_v2_policy.py`
> 定位时间：2026-10-04

---

## 1. 这个功能解决什么问题

动作/状态在送进模型前要做归一化，归一化参数（mean/std/q01/q99…）存在 JSON 里。
**这套统计量必须与「模型训练时用的那一套」完全一致**，否则模型看到的是另一个输入分布。

`FeatureTransform.__init__` 里 `norm_stats_path` 有**两个来源**：

```python
if norm_stats_path is None:
    norm_stats_path = robot_config.pop('norm_stats')   # ← ① robot config yaml
else:
    robot_config.pop('norm_stats')                     # ← ② 调用方传入
```

而两条调用链传的东西不一样：

| 调用链 | 传什么 | 结果 |
|---|---|---|
| **训练 / 训练中评测**（`VLADataset.__init__`） | 不传（`None`） | 走 ① ⇒ `robot_config['norm_stats']` |
| **deploy / 官方开环 / 闭环评测**（`policy.reset()`） | `data_config.norm_stats_file` | 走 ② ⇒ `data.norm_stats_file` |

⇒ 默认配置下**这两条链读的是两个不同的文件**，训练与评测口径不一致。

---

## 2. 怎么发现（症状）

- 用**逐值对拍**：同一个 ckpt，`in-process` 评测 vs 官方 `open_loop_eval.py`
  - **GT 完全一致**（`max|Δ| ~2e-7`）
  - **语言 token / img_masks 完全一致**
  - 但**模型输入的 `state` 差 0.15**、`images` 差 4e-3
  - 而**归一化前的原始 state 逐值一致**
- ⇒ 差异只能来自 `apply()` 里的归一化 ⇒ 顺藤摸到 `norm_stats_path` 的两个来源

⚠️ **为什么 GT 一致会掩盖它**：GT 是 `unapply(apply(raw))` 的**往返**。只要每一侧内部用同一套 stats，
往返就还原出原始值 ⇒ **GT 一致完全不能证明 stats 一致**。原始 state 逐值一致也是同理。

---

## 3. 修复内容

`base_dataset.py::VLADataset.__init__`：

```python
self.feature_transform = FeatureTransform(
    robot_config, dataset_config, self.config, processor,
    disabled_image_features, do_nomalize,
    chunk_size=chunk_size, return_item_befor_padding=return_item,
    norm_stats_path=getattr(dataset_config, "norm_stats_file", None),   # ← 新增
    image_augment=image_augment, use_depth_align=use_depth_align,
    use_future_image=use_future_image)
```

* 优先用 `data.norm_stats_file` ⇒ **训练 / 训练中评测 / deploy 三条路读同一个文件**，构造上保证一致
* `dataset_config` 没有该字段时保持原行为（向后兼容）

---

## 4. 本项目实际用的两个文件（别搞混）

| 文件 | `count`（帧数） | 是什么 |
|---|---|---|
| `assets/norm_stats/robotwin.json` | **6,062,592** | 别的语料（≈11× 于本项目数据集） |
| `assets/norm_stats/robotwin_competition_clean.json` | **548,893** | **= 本项目数据集总帧数**（2500 回合） |

`count` 是判断「这份统计是在哪份数据上算的」最快的方法。

本项目（`robotwin_official_paths.yaml`）的 `data.norm_stats_file` 指向 **competition_clean**，
即**在本项目数据集上算的那套** ⇒ 训练与评测都该用它。

---

## 5. 实测：两套 stats 的代价（同一份 50k 权重）

| 评测用 | traj50 MSE | traj51 MSE | 平均 |
|---|---|---|---|
| `competition_clean`（本数据集） | 0.50164 | 0.46629 | 0.48397 |
| `robotwin.json`（6M 语料） | 0.48463 | 0.45680 | **0.47071** |

⇒ 50k 是**用 6M 语料那套训的**，所以拿本数据集那套评它反而差 2.7%。

**结论：归一化该跟「训练用的那套」走，不是跟「评测数据那套」走。**
模型学到的是「某种归一化下的输入 → 输出」这个映射，换归一化等于换输入分布。

---

## 6. 怎么验证 / 防复发

1. **训练中评测每次都会打一行指纹**（`open_loop_validation.py`）：

```
[open_loop] 归一化统计指纹 observation.state.arm.position.mean[:3] = [-0.21669, 1.08914, 0.79395]
            （本数据集那套 = [-0.21669, 1.08914, 0.79395]；robotwin.json 那套 = [-0.23846, 1.13016, 0.80707]）
```

2. **CPU 自检 C9「归一化统计来源一致」**：

```bash
HF_HUB_OFFLINE=1 /data/miniconda3/envs/lingbotvla/bin/python -u tools/open_loop_selfcheck.py --task click_bell
```

3. **改训练数据 / 换 stats 文件之后，旧模型全部作废** —— 它们是在旧口径下训的，
   不能直接与新口径的结果比较。
