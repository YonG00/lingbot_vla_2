# 回合白名单（`episode_ids_file`）数据正确性 使用文档

> 适用版本：`YonG00/lingbot_vla_2`（LingBot-VLA-v2 post-training）
> 相关参数：`--data.episode_ids_file`
> 相关文件：`lingbotvla/data/vla_data/base_dataset.py`（`LeRobotDataset.__getitem__`、`_load_episode_ids`）
> 定位时间：2026-10-04

---

## 1. 这个功能解决什么问题

L1–L4 课程训练要「只用一个回合子集」训练，靠 `--data.episode_ids_file` 传一份回合号白名单
（由 `tools/prepare_phase.py` / `tools/task_split.py` 生成），底层把白名单交给
`LeRobotDataset(..., episodes=[...])`。

**但白名单 + `delta_timestamps` 组合在 lerobot 里有索引空间 bug**，会让**动作/状态整段取成常数**，
而**图像仍然是对的** —— 训练照常跑、loss 照常降，只是**学不到任何动作**。

---

## 2. 一句话原理

> `_get_query_indices(idx, ep_idx)` 用「**局部**样本号 `idx`」去和「**绝对**帧号
> `dataset_from_index / dataset_to_index`」比较。
> 白名单过滤后 `hf_dataset` 只剩被选中的回合，`idx` 是 0..len-1 的局部序号，
> 于是 `idx + delta < ep_start` **恒成立** ⇒ 整段 chunk 全判成 padding。

---

## 3. 症状（怎么认出来）

| 现象 | 说明 |
|---|---|
| 开环评测 `baseline ≈ 0`、`r2` 是 `-1e27` 这种荒谬值 | GT 方差 ≈ 1e-27 ⇒ GT 恒定 |
| 数据集里 `action_is_pad` **全是 1** | 所有时间偏移都被判成越界 |
| `observation.state` / `action` 全 0 或全等于某常数 | 恒为该回合**第一帧**的值 |
| **图像完全正常** | 图像走 `hf_dataset[idx]`（局部索引正确），不受影响 |
| loss 正常下降、但闭环成功率 ≈ 0 | 目标平凡：真实图像 + 常数 state → 常数 action |

**最小复现**（不用 GPU，~20 秒）：

```bash
cd /data/code/lingbot-vla-v2
HF_HUB_OFFLINE=1 /data/miniconda3/envs/lingbotvla/bin/python -u - <<'PY'
import sys; sys.path.insert(0, "/data/code/lingbot-vla-v2")
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lingbotvla.data.vla_data.base_dataset import LeRobotDataset as RepoDataset
ROOT = "/data/datasets/robotwin_v3/lerobot_dataset/RoboTwin_lerobot_v30"
meta = LeRobotDatasetMetadata(Path(ROOT).name, root=ROOT); fps = meta.fps
delta = {"observation.state": [t/fps for t in range(51)], "action": [t/fps for t in range(50)]}
for tag, kw in (("白名单 [50]", {"episodes": [50]}), ("不过滤", {})):
    ds = RepoDataset(Path(ROOT).name, root=ROOT, delta_timestamps=delta, load_image=False, **kw)
    it = ds[0]
    print(f"{tag}: len={len(ds)} "
          f"state[0][:4]={[round(float(x),5) for x in it['observation.state'][0][:4]]} "
          f"state[-1][:4]={[round(float(x),5) for x in it['observation.state'][-1][:4]]} "
          f"is_pad.sum={int(it['observation.state_is_pad'].sum())}")
PY
```

**修复前**：白名单那行 `state[0] == state[-1] == 0`、`is_pad.sum = 51`（全 padding）
**修复后**：白名单与不过滤**数值完全一致**、`is_pad.sum = 0`

---

## 4. 修复内容

`lingbotvla/data/vla_data/base_dataset.py`，`LeRobotDataset.__getitem__`：

```python
if self.delta_indices is not None:
    # 用当前样本的【绝对】帧号（`index` 列）去查询 delta
    abs_idx = int(item["index"])
    query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
```

* `episodes=None`（不过滤）时 `index == idx`，本行是 **no-op**，行为完全不变
* `_query_hf_dataset` 内部再用 `_absolute_to_relative_idx` 映射回过滤后的相对行号，无需改动

---

## 5. 影响面（重要）

修复前，**所有带 `--data.episode_ids_file` 的训练**都受影响：

* `experiment/robotwin/phase1_train_then_eval.sh`
* `experiment/robotwin/phase1_L1_vit_frozen_train_then_eval.sh`
* `experiment/robotwin/phase2_from_base_train_then_eval.sh`
* `experiment/robotwin/single_task_train.sh`

⇒ 这些 run 的 **loss / 开环 / 闭环结论全部不可用，需要重跑**。
不带白名单的训练（例如官方 50k 成品）**不受影响**。

---

## 6. 注意事项

* 本修复只解决**索引空间**问题；白名单本身仍要求回合号是 **0-based**（`_load_episode_ids` 会校验上界）
* 若后续升级 lerobot 到官方修好的版本，本行会退化为 no-op，**可以保留**
* 每次改动数据管线后，建议跑一次第 3 节的最小复现做回归
