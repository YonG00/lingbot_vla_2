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
import random
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "collect_feature_transforms", "snapshot_rng", "restore_rng",
    "deterministic_sampling",
]


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

        {"n_ft": 找到几个 feature_transform, "n_disabled": 关掉了几个增强}

    调用方可据此判断「到底有没有生效」——``n_ft == 0`` 时说明没能定位到
    feature_transform，此时**必然没有关掉增强**，需要上层显式 fail-fast。
    """
    fts = collect_feature_transforms(dataset)
    disabled: List[Tuple[Any, Any]] = []
    for ft in fts:
        if hasattr(ft, "image_augment") and ft.image_augment:
            disabled.append((ft, ft.image_augment))
    st = snapshot_rng()
    try:
        for ft, _orig in disabled:
            ft.image_augment = False
        yield {"n_ft": len(fts), "n_disabled": len(disabled)}
    finally:
        for ft, orig in disabled:
            try:
                ft.image_augment = orig
            except Exception:  # noqa: BLE001
                pass
        restore_rng(st)
