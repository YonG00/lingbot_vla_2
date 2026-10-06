"""SampleResolver —— `local_idx ↔ (task, episode, frame)` 的稳定映射。

🔴 设计红线
-----------
* **不自行重新编码 sample id**。Stage A Demo 那套
  `task_index*1e6 + traj*1e3 + frame` **不带进正式代码**。
  这里的坐标就是 ``(task, episode, frame)``，``frame`` 是数据集里的**绝对帧号**。
* **不重新实现** local-index → absolute-index 逻辑：只**读**
  `hf_dataset['index']` / `hf_dataset['episode_index']` 两列 ——
  与 `base_dataset.py::LeRobotDataset.__getitem__` 里
  ``abs_idx = int(item["index"])`` 用的是**同一个值**。

本模块只依赖 `numpy`。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .ports import SampleRef


def hf_dataset_of(ds: Any):
    """从 `VLADataset` / `MultiVLADataset` 下钻到真正的 `hf_dataset`。

    与 `lingbotvla/utils/open_loop_validation.py::_episode_index_map` 的
    下钻方式**保持一致**（单条目 `MultiVLADataset` → `_datasets[0].dataset`）。
    """
    node = ds
    inner = getattr(ds, "_datasets", None)
    if isinstance(inner, (list, tuple)) and len(inner) == 1:
        node = inner[0]
    node = getattr(node, "dataset", node)
    return getattr(node, "hf_dataset", None)


class SampleResolver:
    """`local_idx ↔ SampleRef(task, episode, frame)`。

    典型用法::

        res = SampleResolver.from_dataset(ds, task_of_episode=cat.task_of_episode)
        res.local_to_ref(1234)        # → SampleRef(task='click_bell', episode=37, frame=1832)
        res.ref_to_local(ref)         # → 1234
    """

    def __init__(
        self,
        *,
        episode_index: np.ndarray,
        frame_index: np.ndarray,
        task_of_episode: Callable[[int], str],
        episode_ids: Optional[List[int]] = None,
    ):
        ep = np.asarray(episode_index, dtype=np.int64)
        fr = np.asarray(frame_index, dtype=np.int64)
        if ep.ndim != 1 or fr.ndim != 1 or ep.shape != fr.shape:
            raise ValueError(
                f"episode_index / frame_index 必须是一维等长数组，"
                f"实际 {ep.shape} / {fr.shape}")
        if len(ep) == 0:
            raise ValueError("数据集为空（0 条样本）")
        self._ep = ep
        self._fr = fr
        self._task_of_episode = task_of_episode
        self._ref_to_local: Optional[Dict[Tuple[str, int, int], int]] = None
        self._episode_to_locals: Optional[Dict[int, List[int]]] = None
        self.episode_ids: List[int] = (
            list(episode_ids) if episode_ids is not None else sorted(set(int(x) for x in ep))
        )

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_dataset(cls, ds: Any, *, task_of_episode: Callable[[int], str]) -> "SampleResolver":
        """从已建好的数据集（含白名单过滤）构造。

        ⚠️ 这里读的是**过滤后**的行 —— 与训练/评测实际看到的样本一一对应。
        """
        hf = hf_dataset_of(ds)
        if hf is None:
            raise RuntimeError(
                f"从 {type(ds).__name__} 上下钻不到 hf_dataset"
                f"（可用属性: {[a for a in dir(ds) if 'dataset' in a][:8]}）")
        try:
            ep = np.asarray([int(x) for x in hf["episode_index"]], dtype=np.int64)
            fr = np.asarray([int(x) for x in hf["index"]], dtype=np.int64)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"读 hf_dataset 的 episode_index / index 列失败: {exc}") from exc
        n = len(ds) if hasattr(ds, "__len__") else len(ep)
        if len(ep) != n:
            raise RuntimeError(
                f"hf_dataset 行数({len(ep)}) != len(dataset)({n}) ⇒ 映射不可信，拒绝继续")
        return cls(episode_index=ep, frame_index=fr, task_of_episode=task_of_episode)

    # -- 映射 ---------------------------------------------------------------
    def __len__(self) -> int:
        return int(len(self._ep))

    def episode_map(self) -> np.ndarray:
        """`local_idx → episode_index`（与 `_episode_index_map` 同源）。"""
        return self._ep

    def frame_map(self) -> np.ndarray:
        """`local_idx → 绝对帧号`。"""
        return self._fr

    def local_to_ref(self, local_idx: int) -> SampleRef:
        i = int(local_idx)
        if i < 0 or i >= len(self._ep):
            raise IndexError(f"local_idx 越界: {i}（共 {len(self._ep)} 条）")
        ep = int(self._ep[i])
        return SampleRef(task=self._task_of_episode(ep), episode=ep, frame=int(self._fr[i]))

    def _ensure_inverse(self) -> Dict[Tuple[str, int, int], int]:
        if self._ref_to_local is None:
            inv: Dict[Tuple[str, int, int], int] = {}
            for i in range(len(self._ep)):
                ep = int(self._ep[i])
                key = (self._task_of_episode(ep), ep, int(self._fr[i]))
                if key in inv:
                    raise RuntimeError(f"坐标重复（绝对帧号应唯一）: {key}")
                inv[key] = i
            self._ref_to_local = inv
        return self._ref_to_local

    def ref_to_local(self, ref: SampleRef) -> int:
        key = (ref.task, int(ref.episode), int(ref.frame))
        try:
            return self._ensure_inverse()[key]
        except KeyError:
            raise KeyError(
                f"坐标 {key} 不在当前数据集里（可能被 episode 白名单过滤掉了）") from None

    def episode_to_locals(self) -> Dict[int, List[int]]:
        """`episode_index → [local_idx, ...]`（升序）。"""
        if self._episode_to_locals is None:
            out: Dict[int, List[int]] = {}
            for i in range(len(self._ep)):
                out.setdefault(int(self._ep[i]), []).append(i)
            self._episode_to_locals = out
        return self._episode_to_locals

    def local_indices_of_episode(self, episode: int) -> List[int]:
        return list(self.episode_to_locals().get(int(episode), []))

    # -- 自检 ---------------------------------------------------------------
    def verify(self) -> List[str]:
        """返回问题列表（空 = 全过）。不需要模型/GPU。"""
        problems: List[str] = []
        # 绝对帧号必须严格递增（LeRobot 数据集按帧连续存储）
        if len(self._fr) > 1 and not np.all(np.diff(self._fr) > 0):
            bad = np.where(np.diff(self._fr) <= 0)[0][:3]
            problems.append(f"绝对帧号不是严格递增（前几处 local_idx={bad.tolist()}）")
        # episode 必须成段出现（同一回合的帧连续排列）
        changes = np.where(np.diff(self._ep) != 0)[0]
        if len(changes) + 1 != len(set(int(x) for x in self._ep)):
            problems.append("同一 episode 的帧没有连续成段 ⇒ 拼接/聚合口径会错")
        return problems


__all__ = ["SampleResolver", "hf_dataset_of"]
