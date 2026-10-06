"""Stage B1 —— 真实 Backend 组装的**无卡**测试（用假件）。

验证 `real/backend.py` 的三件事：
  1. `build_real_backend` 的 fail-fast（catalog 没 attach_samples ⇒ 报错）
  2. `RealEvaluator`：split 映射（train_monitor → train，其余 → val）+ baseline 查询
  3. `RealHardnessScorer`：`(task, sample_ids) -> {sample_id: loss}`，且**分批**调用

    python -m pytest tests/test_auto_learning_backend.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingbotvla.auto_learning.ports import EvalResult, TaskEntry   # noqa: E402
from lingbotvla.auto_learning.real.backend import (                # noqa: E402
    RealEvaluator, RealHardnessScorer, RealTrainerStub, build_real_backend,
)

_TMP_ROOT = REPO / ".pytest_tmp"


@pytest.fixture
def tmp_path():
    import shutil
    import uuid

    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = _TMP_ROOT / f"alb_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
class _Catalog:
    def __init__(self, entries):
        self._e = dict(entries)

    def task_names(self):
        return list(self._e)

    def entry(self, t):
        return self._e[t]


class _Adapter:
    """桩：记录调用，返回一个 `EvalResult`。"""

    def __init__(self):
        self.calls = []

    def evaluate_task(self, task, split, episode_ids=None):
        self.calls.append((task, split, list(episode_ids or [])))
        return EvalResult(
            task=task, split=split, episode_ids=list(episode_ids or []),
            mse=0.04, baseline_mse=0.2, nmse=0.2, mae=0.1,
            per_traj_mse=[0.03, 0.05], per_traj_ids=[10, 11], per_traj_frames=[100, 100],
            n_traj=2, n_chunks=4, frames=200, dims=14, eval_seconds=1.5,
            action_keys=["action"],
        )


class _Hardness:
    def score(self, task, sample_ids):
        return {int(x): 0.5 for x in sample_ids}


class _Hardness:
    def score(self, task, sample_ids):
        return {int(x): 0.5 for x in sample_ids}


class _Resolver:
    def resolve(self, task, sample_id):
        raise NotImplementedError


class _Store:
    def __init__(self, mse=0.2):
        self.mse = mse
        self.warnings = []

    def get(self, task, sha=None):
        if self.mse is None:
            return None
        import types
        return types.SimpleNamespace(mse=self.mse, fingerprint="fp")


def _entry(name="click_bell"):
    return TaskEntry(name=name, train_traj_ids=[50, 51], val_traj_ids=[52],
                     train_sample_ids=[1, 2, 3], samples_by_traj={50: [1, 2], 51: [3]},
                     val_sample_ids=[4], sha256_train="sha-A")


# --------------------------------------------------------------------------- #
# 1) fail-fast
# --------------------------------------------------------------------------- #
def test_build_backend_requires_attach_samples():
    cat = _Catalog({"click_bell": TaskEntry(name="click_bell", train_traj_ids=[50])})
    with pytest.raises(ValueError) as err:
        build_real_backend(catalog=cat, resolver=_Resolver(), adapter=_Adapter(),
                           hardness=_Hardness())
    assert "attach_samples" in str(err.value)


def test_build_backend_ok_when_samples_attached():
    cat = _Catalog({"click_bell": _entry()})
    b = build_real_backend(catalog=cat, resolver=_Resolver(), adapter=_Adapter(),
                           hardness=_Hardness())
    assert b.missing() == [], f"Backend 契约不完整: {b.missing()}"
    assert isinstance(b.trainer, RealTrainerStub)


def test_real_trainer_stub_refuses_to_be_called():
    """延迟训练模式下 trainer 不该被调；真被调要**显式报错**而不是静默。"""
    with pytest.raises(RuntimeError) as err:
        RealTrainerStub().train_steps(None, 50)
    assert "延迟训练模式" in str(err.value) or "defer_train" in str(err.value)


# --------------------------------------------------------------------------- #
# 2) RealEvaluator
# --------------------------------------------------------------------------- #
def test_evaluator_maps_train_monitor_to_train_split():
    """`EvalSplit.TRAIN_MONITOR` 必须映射到 manifest 的 `train` split。

    否则 `EvaluatorAdapter` 的「防 train/val 泄漏」校验会把合法的 train-monitor 回合拒掉。
    """
    ad = _Adapter()
    ev = RealEvaluator(ad, _Catalog({"click_bell": _entry()}), _Store())
    ev.evaluate("click_bell", "train_monitor", [50, 51])
    assert ad.calls == [("click_bell", "train", [50, 51])]


def test_evaluator_maps_other_splits_to_val():
    ad = _Adapter()
    ev = RealEvaluator(ad, _Catalog({"click_bell": _entry()}), _Store())
    for split in ("scout", "confirm", "active_val", "review"):
        ev.evaluate("click_bell", split, [52])
    assert [c[1] for c in ad.calls] == ["val"] * 4


def test_evaluator_returns_trajectory_metrics():
    ev = RealEvaluator(_Adapter(), _Catalog({"click_bell": _entry()}), _Store())
    tm = ev.evaluate("click_bell", "active_val", [52])
    assert tm.mse == pytest.approx(0.04)
    assert tm.nmse == pytest.approx(0.2)
    assert tm.n_trajs == 2 and tm.metric_valid is True
    assert tm.per_traj_mse == {10: pytest.approx(0.03), 11: pytest.approx(0.05)}


def test_evaluator_baseline_mse_from_store():
    ev = RealEvaluator(_Adapter(), _Catalog({"click_bell": _entry()}), _Store(0.2))
    assert ev.baseline_mse("click_bell") == pytest.approx(0.2)
    # 没有 store / store 里没有 ⇒ 0.0（Scheduler 侧靠 require_baseline 兜底）
    assert RealEvaluator(_Adapter(), _Catalog({"click_bell": _entry()}),
                         None).baseline_mse("click_bell") == 0.0
    assert RealEvaluator(_Adapter(), _Catalog({"click_bell": _entry()}),
                         _Store(None)).baseline_mse("click_bell") == 0.0


# --------------------------------------------------------------------------- #
# 3) RealHardnessScorer
# --------------------------------------------------------------------------- #
class _FakeScorer:
    """返回固定值，并记录每次收到的 item 数。"""

    def __init__(self):
        self.batch_sizes = []

    def score(self, items):
        import numpy as np
        self.batch_sizes.append(len(items))
        # 值由 item 自身决定（而不是批内下标）—— 这样跨批也能对上
        return np.array([float(it["idx"]) / 100.0 for it in items], dtype="float32")


class _FakeDataset:
    def __init__(self):
        self.read = []

    def __getitem__(self, i):
        self.read.append(int(i))
        return {"idx": int(i)}


def test_hardness_score_maps_sample_ids_to_losses():
    sc, ds = _FakeScorer(), _FakeDataset()
    h = RealHardnessScorer(sc, ds, max_batch=2)
    out = h.score("click_bell", [5, 6, 7])
    assert set(out) == {5, 6, 7}
    assert ds.read == [5, 6, 7]
    assert sc.batch_sizes == [2, 1], "应当按 max_batch 分批"
    assert out[5] == pytest.approx(0.05) and out[7] == pytest.approx(0.07)


def test_hardness_empty_ids_returns_empty():
    h = RealHardnessScorer(_FakeScorer(), _FakeDataset())
    assert h.score("click_bell", []) == {}


def test_hardness_uses_the_dataset_index_space():
    """🔴 sample_id 必须**直接**当 dataset index 用（训练数据集的全量索引空间）。

    若换成「按任务的子集数据集」，索引是局部的 ⇒ 难度会打到**别的样本**上。
    """
    sc, ds = _FakeScorer(), _FakeDataset()
    RealHardnessScorer(sc, ds, max_batch=8).score("t", [0, 1, 2, 3])
    assert ds.read == [0, 1, 2, 3]
