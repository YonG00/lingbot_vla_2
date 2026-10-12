"""Hardness / 采样的**确定性上下文**（review v0.2 #8）。

问题
----
`RealHardnessScorer.score()` 在进入 scorer 之前会 ``dataset[j]`` 取 item。
若训练数据集开了 ``image_augment=true``：

* 同一个 ``sample_id`` 两次 hardness 扫描可能拿到**不同的增强图** ⇒ 「难度」不可复现，
  硬度权重也随之抖动；
* 取增强图会**消耗全局 RNG** ⇒ 污染训练侧的随机流（resume 语义、noise 对齐都会受影响）。

即便 flow 的 ``noise`` / ``time`` 已经固定，也挡不住这一层随机性。

做法
----
评分期间（取 item + 前向）：

1. 把 dataset 上**所有** ``feature_transform`` 的 ``image_augment`` 临时置 False，``finally`` 还原；
2. 快照 / 还原 ``random`` / ``numpy`` / ``torch``(含 CUDA) 的 RNG 状态。

本模块**不 import torch 于顶层**（无卡可单测）；torch/numpy 全部懒加载。
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "collect_feature_transforms", "snapshot_rng", "restore_rng",
    "deterministic_sampling", "_clear_ft_cache", "_fast_fetch_enabled",
]


#: O1 对照开关：**进程启动时读一次**（与其它 `AL_*` 开关一致，如 `AL_EVAL_SHARD`）。
#:
#: * `AL_HARDNESS_FORCE_SNAPSHOT=1` ⇒ **强制**每批都做 RNG 快照/还原（仅用于对照实验）；
#: * 未设（默认）⇒ 只在"**确有增强被关**（`disabled` 非空）"时才快照。
#:
#: ⚠️ 语义说明（消除歧义）：`disabled` 为空时，取 item 期间**一个 RNG 都不消耗**
#:    （随机增强全部 gate 在 `train=self.image_augment and not policy_eval`，而本 run
#:    `image_augment=False`）⇒ 快照 + 还原是**恒等操作**，做与不做**结果完全相同**
#:    ⇒ 没有任何开关能把它变成"有用的事"。要回退到旧行为只需删掉那一处早退。
#:    之所以不每批读 `os.environ`：那会给每个 batch 再加一次字典查找 —— 本优化的目的正是
#:    消除"每批的固定开销"，不能让开关自身变成新的每批开销。
_FORCE_SNAPSHOT: bool = os.environ.get('AL_HARDNESS_FORCE_SNAPSHOT', '0') == '1'


def _fast_fetch_enabled() -> bool:
    """是否允许"没有增强要关时跳过快照"（= 未被强制快照）。"""
    return not _FORCE_SNAPSHOT


def _add_once(outs: List[Any], seen: set, ft: Any) -> None:
    if ft is not None and id(ft) not in seen:
        seen.add(id(ft))
        outs.append(ft)


def collect_feature_transforms(dataset: Any) -> List[Any]:
    """收集 dataset（含 ``MultiVLADataset`` 的子数据集）上所有 feature_transform。

    只做浅层属性遍历 —— 与 `open_loop_validation._find_feature_transform` 的口径一致，
    但**不依赖**那个模块（保持本模块可在无 torch 环境导入/单测）。
    """
    outs: List[Any] = []
    seen: set = set()
    _add_once(outs, seen, getattr(dataset, "feature_transform", None))
    fts = getattr(dataset, "feature_transforms", None)
    if isinstance(fts, dict):
        for v in fts.values():
            _add_once(outs, seen, v)
    elif isinstance(fts, (list, tuple)):
        for v in fts:
            _add_once(outs, seen, v)
    for attr in ("datasets", "_datasets", "sub_datasets"):
        sub = getattr(dataset, attr, None)
        if isinstance(sub, (list, tuple)):
            for d in sub:
                _add_once(outs, seen, getattr(d, "feature_transform", None))
                f2 = getattr(d, "feature_transforms", None)
                if isinstance(f2, dict):
                    for v in f2.values():
                        _add_once(outs, seen, v)
                elif isinstance(f2, (list, tuple)):
                    for v in f2:
                        _add_once(outs, seen, v)
    return outs


# --------------------------------------------------------------------------- #
# 特征遍历缓存（O1：遍历结果在一次扫描内不变，没必要每批重算）
# --------------------------------------------------------------------------- #
#: ``id(dataset) -> (fts, disabled)``；只在 dataset 存活期间有效（weakref 不做，见下）。
_FT_CACHE: Dict[int, Tuple[List[Any], List[Tuple[Any, Any]]]] = {}


def _feature_transforms_cached(dataset: Any) -> Tuple[List[Any], List[Tuple[Any, Any]]]:
    """返回 ``(fts, disabled)``，并按 ``id(dataset)`` 缓存。

    🔴 为什么缓存是安全的：
      * ``collect_feature_transforms`` 只做**浅层属性遍历**（不调 RNG、不改状态）；
      * 一个 dataset 的 transform 结构在训练/扫描期间不变；
      * key 用 ``id()`` + 持有 dataset 引用（``_FT_CACHE_HOLDER``）⇒ 避免 id 复用误命中。
    调用方若担心 outdated，可调 ``_clear_ft_cache()``。
    """
    key = id(dataset)
    hit = _FT_CACHE.get(key)
    if hit is not None and _FT_CACHE_HOLDER.get(key) is dataset:
        return hit
    fts = collect_feature_transforms(dataset)
    disabled: List[Tuple[Any, Any]] = []
    for ft in fts:
        if getattr(ft, "image_augment", False):
            disabled.append((ft, ft.image_augment))
    _FT_CACHE[key] = (fts, disabled)
    _FT_CACHE_HOLDER[key] = dataset          # 强引用 ⇒ 防止 id 复用导致误命中
    return fts, disabled


#: 与 ``_FT_CACHE`` 同 key 的强引用（防 ``id()`` 复用）
_FT_CACHE_HOLDER: Dict[int, Any] = {}


def _clear_ft_cache() -> None:
    """清空特征遍历缓存（测试 / 数据集被替换时用）。"""
    _FT_CACHE.clear()
    _FT_CACHE_HOLDER.clear()


# --------------------------------------------------------------------------- #
# RNG 快照 / 还原
# --------------------------------------------------------------------------- #
def snapshot_rng() -> Dict[str, Any]:
    """快照全局 RNG（random / numpy / torch / torch.cuda）。不可用的项跳过。"""
    st: Dict[str, Any] = {"random": random.getstate()}
    try:
        import numpy as np

        st["numpy"] = np.random.get_state()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        st["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            st["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:  # noqa: BLE001
        pass
    return st


def restore_rng(st: Dict[str, Any]) -> None:
    """按快照还原全局 RNG。单项失败不影响其它项。"""
    if not st:
        return
    if "random" in st:
        try:
            random.setstate(st["random"])
        except Exception:  # noqa: BLE001
            pass
    if "numpy" in st:
        try:
            import numpy as np

            np.random.set_state(st["numpy"])
        except Exception:  # noqa: BLE001
            pass
    if "torch" in st:
        try:
            import torch

            torch.set_rng_state(st["torch"])
        except Exception:  # noqa: BLE001
            pass
    if "cuda" in st:
        try:
            import torch

            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# 上下文
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def deterministic_sampling(dataset: Any, *, logger: Any = None):
    """在**取 dataset item** 期间关闭图像增强，并保证 RNG 状态进出一致。

    ``yield`` 出一个小报告::

        {"n_ft": 找到几个 feature_transform,
         "n_disabled": 关掉了几个增强,
         "rng_touched": 本轮是否真的做了 RNG 快照/还原}   # O1（10-12）新增

    调用方可据此判断「到底有没有生效」——``n_ft == 0`` 时说明没能定位到
    feature_transform，此时**必然没有关掉增强**，需要上层显式 fail-fast。

    🔴 ``rng_touched`` 的口径（O1）：``n_disabled == 0`` 时跳过快照 ⇒ ``False``。
    这不是"省掉了一次必要的保护"，而是"没有增强要关 ⇒ 取 item 不消耗 RNG ⇒
    快照 + 还原是恒等操作"（见模块内 `_FORCE_SNAPSHOT` 的说明）。
    ``AL_HARDNESS_FORCE_SNAPSHOT=1`` 可强制恢复旧行为（每批都快照）做对照。
    """
    fts, disabled = _feature_transforms_cached(dataset)
    # 🔴 O1（2026-10-12）：**没有任何增强被关** ⇒ 本次调用不含任何 RNG 动作
    #    ⇒ "什么都不做" 与 "快照+还原" **效果完全相同** ⇒ 跳过快照。
    #    依据（实测）：数据集的随机增强全部 gate 在 `train=self.image_augment and not policy_eval`
    #    （`vla_data/utils.py`），而本 run 的 `image_augment=False`（dataset 默认 + 配置
    #    `allow_image_augment: false`）⇒ 取 item 不消耗 RNG，快照属**恒等操作**、纯开销。
    #    代价原状：每批一次 `torch.get_rng_state()` + CUDA `get_rng_state_all()` 及其还原
    #    （同步调用），在 7 卡训练进程里每批都付。
    # 安全前提：一旦真有增强被关（disabled 非空）⇒ **仍走原来的快照/还原路径**，行为不变。
    # 对照开关：`AL_HARDNESS_FORCE_SNAPSHOT=1` ⇒ 强制每批都快照（仅 A/B 对照用）。
    # 🔴 条件用 **and**（不是 or）：只有"**没有增强要关** 且 **未被强制快照**"时才早退。
    #    写成 `or` 会让对照开关失效（强制快照也被吞掉），单测已锁这一条。
    if not disabled and _fast_fetch_enabled():
        yield {"n_ft": len(fts), "n_disabled": len(disabled), "rng_touched": False}
        return
    st = snapshot_rng()
    try:
        for ft, _orig in disabled:
            ft.image_augment = False
        yield {"n_ft": len(fts), "n_disabled": len(disabled), "rng_touched": True}
    finally:
        for ft, orig in disabled:
            try:
                ft.image_augment = orig
            except Exception:  # noqa: BLE001
                pass
        restore_rng(st)
