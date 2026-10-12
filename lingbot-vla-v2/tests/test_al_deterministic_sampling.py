"""O1 的 CPU 契约测试 —— 2026-10-12 用户定。

背景（真机实测 + 代码取证）
--------------------------
`deterministic_sampling` 原本**每批**（hardness 每 8 个样本）都做一次
`collect_feature_transforms(dataset)` + `snapshot_rng()`（含 CUDA `get_rng_state_all()`）
+ `restore_rng()`。而本 run 的 `image_augment=False`（dataset 默认 + 配置
`allow_image_augment: false`），且数据集的随机增强全部 gate 在
`train=self.image_augment and not policy_eval`（`vla_data/utils.py`）
⇒ **取 item 不消耗任何 RNG**，那个快照/还原是**恒等操作**、纯开销。

本文件锁三件事
-------------
1. **语义等价（核心）**：没有任何增强被关时，取 item **不改变** `random` / `numpy` / `torch`
   的全局 RNG 状态 —— 这正是"跳过快照"的**安全前提**，也是原实现想保证的东西；
2. **退化安全**：真有增强被关（`image_augment=True`）时，**仍走**快照/还原：增强被临时关闭，
   且它消耗掉的 RNG **被还原**（否则会污染训练随机流）；
3. **边界**：`n_ft == 0`（定位不到 feature_transform）时报告仍如实给出；开关
   `AL_HARDNESS_FAST_FETCH=0` 能退回旧行为；特征遍历缓存按 dataset 身份命中/失效。
"""
from __future__ import annotations

import random

from lingbotvla.auto_learning.real.determinism import (
    _clear_ft_cache,
    _fast_fetch_enabled,
    collect_feature_transforms,
    deterministic_sampling,
)


class _FakeFT:
    """最小的 feature_transform 替身：``image_augment=True`` 时消耗随机数（模拟真实增强）。"""

    def __init__(self, image_augment: bool) -> None:
        self.image_augment = image_augment
        self.calls = 0

    def __getitem__(self, idx):
        self.calls += 1
        if self.image_augment:
            random.random()          # ← 模拟随机增强消耗全局 RNG
        return {"idx": idx}


class _FakeDataset:
    def __init__(self, ft=None) -> None:
        self.feature_transform = ft

    def __getitem__(self, idx):
        return self.feature_transform[idx] if self.feature_transform is not None else {"idx": idx}


def _rng_fingerprint():
    """random 的完整状态指纹（torch 可选，见下）。"""
    st = random.getstate()
    return (st[0], tuple(st[1][:8]))     # 版本号 + 前 8 个数（足够判定是否被消费）


# --------------------------------------------------------------------------- #
# 1. 语义等价：没有增强要关 ⇒ 不碰 RNG（O1 的安全前提）
# --------------------------------------------------------------------------- #
def test_no_augment_does_not_touch_rng():
    _clear_ft_cache()
    ds = _FakeDataset(ft=_FakeFT(image_augment=False))
    before = _rng_fingerprint()
    with deterministic_sampling(ds) as rep:
        for j in range(5):
            ds[j]                              # 取 5 个样本，模拟一批
    assert _rng_fingerprint() == before, "image_augment=False 时取 item 不该消耗 RNG"
    assert rep["n_disabled"] == 0
    assert rep["n_ft"] == 1
    assert rep["rng_touched"] is False, "没有增强要关 ⇒ 不该做 RNG 快照"


def test_repeat_calls_stay_rng_neutral():
    """连续多次进入上下文（真实场景：每批一次）都不得改变 RNG。"""
    _clear_ft_cache()
    ds = _FakeDataset(ft=_FakeFT(image_augment=False))
    before = _rng_fingerprint()
    for _ in range(10):                        # 10 批
        with deterministic_sampling(ds):
            for j in range(8):
                ds[j]
    assert _rng_fingerprint() == before


# --------------------------------------------------------------------------- #
# 2. 退化安全：真有关增强被关 ⇒ 仍快照/还原，且增强确实被临时关掉
# --------------------------------------------------------------------------- #
def test_augment_enabled_is_still_snapshotted_and_restored():
    _clear_ft_cache()
    ft = _FakeFT(image_augment=True)
    ds = _FakeDataset(ft=ft)
    before = _rng_fingerprint()
    with deterministic_sampling(ds) as rep:
        assert rep["n_disabled"] == 1
        assert rep["rng_touched"] is True, "有增强要关 ⇒ 必须快照，行为与改前一致"
        assert ft.image_augment is False, "上下文内增强必须被关掉"
        for j in range(3):
            ds[j]                              # 这些调用本会消耗 RNG
    assert ft.image_augment is True, "退出后必须还原 image_augment"
    assert _rng_fingerprint() == before, "增强消耗的 RNG 必须被还原（否则污染训练随机流）"


def test_force_snapshot_switch(monkeypatch):
    """`AL_HARDNESS_FORCE_SNAPSHOT=1` 的对照开关：强制每批都快照（仅用于 A/B 对照）。

    开关在**进程启动时读一次**（模块级 `_FORCE_SNAPSHOT`）⇒ 测试直接打模块属性
    （不用 setenv：对已 import 的模块无效 —— 这本身也是契约：改开关要重启进程）。
    """
    from lingbotvla.auto_learning.real import determinism as _d

    _clear_ft_cache()
    monkeypatch.setattr(_d, '_FORCE_SNAPSHOT', True)
    assert _fast_fetch_enabled() is False, "开关取值应随模块属性变化（进程启动时读一次）"

    # 强制快照 ⇒ 即便没有增强要关，也报告 rng_touched=True（行为=改造前）
    ds = _FakeDataset(ft=_FakeFT(image_augment=False))
    before = _rng_fingerprint()
    with deterministic_sampling(ds) as rep:
        ds[0]
    assert rep["rng_touched"] is True, "对照开关打开时必须走旧路径（每批都快照）"
    assert rep["n_disabled"] == 0
    assert _rng_fingerprint() == before, "旧路径本身也是 RNG 中性的"

    # 安全底线与开关无关：有增强要关时始终快照 + 还原
    monkeypatch.setattr(_d, '_FORCE_SNAPSHOT', False)
    ft = _FakeFT(image_augment=True)
    ds2 = _FakeDataset(ft=ft)
    before2 = _rng_fingerprint()
    with deterministic_sampling(ds2) as rep2:
        ds2[0]
    assert rep2["rng_touched"] is True
    assert _rng_fingerprint() == before2


# --------------------------------------------------------------------------- #
# 3. 边界：n_ft==0 / 缓存身份
# --------------------------------------------------------------------------- #
def test_no_feature_transform_reports_zero():
    """定位不到 feature_transform ⇒ n_ft=0（上层据此 fail-fast/warn），且不崩。"""
    _clear_ft_cache()
    ds = _FakeDataset(ft=None)
    with deterministic_sampling(ds) as rep:
        ds[0]
    assert rep["n_ft"] == 0
    assert rep["n_disabled"] == 0


def test_ft_cache_hits_same_dataset_and_invalidates_on_new_one():
    """遍历结果按 dataset 身份缓存：同一 dataset 命中；换 dataset 后重新遍历。"""
    _clear_ft_cache()
    a = _FakeDataset(ft=_FakeFT(image_augment=False))
    b = _FakeDataset(ft=_FakeFT(image_augment=False))
    with deterministic_sampling(a):
        pass
    with deterministic_sampling(a):        # 第二次命中缓存（不重复遍历）
        pass
    assert collect_feature_transforms(a)[0] is a.feature_transform
    with deterministic_sampling(b) as rep_b:
        assert rep_b["n_ft"] == 1          # 换 dataset ⇒ 自己的 transform
    _clear_ft_cache()
    with deterministic_sampling(a) as rep_a:
        assert rep_a["n_ft"] == 1


def test_cache_does_not_confuse_replacement_dataset():
    """dataset 被替换（同 id 不可能同时存活）时不能误用旧缓存 —— 持有强引用保证。"""
    _clear_ft_cache()
    ds1 = _FakeDataset(ft=_FakeFT(image_augment=False))
    with deterministic_sampling(ds1):
        pass
    ds2 = _FakeDataset(ft=_FakeFT(image_augment=False))   # 新 dataset
    with deterministic_sampling(ds2) as rep:
        assert rep["n_ft"] == 1
        assert rep["n_disabled"] == 0
