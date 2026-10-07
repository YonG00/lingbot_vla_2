"""Stage B1 —— 动态采样器的**无卡**测试（测试计划 §7.6）。

不依赖 torch / GPU / 真实数据集：用假的 catalog / resolver 驱动
`AutoLearnSampler`（真实仓库侧）与 `BatchSampler`（Stage A 移植）。

    python -m pytest tests/test_auto_learning_sampler.py -q
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lingbotvla.auto_learning.config import AutoLearningConfig       # noqa: E402
from lingbotvla.auto_learning.ports import (                         # noqa: E402
    ReplayPlan, TaskEntry, TrainRequest,
)
from lingbotvla.auto_learning.real.sampler import AutoLearnSampler   # noqa: E402
from lingbotvla.auto_learning.sampling.sampler import BatchSampler   # noqa: E402
from lingbotvla.auto_learning.types import SampleRef                 # noqa: E402

_TMP_ROOT = REPO / ".pytest_tmp"


@pytest.fixture
def tmp_path():
    import shutil
    import uuid

    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = _TMP_ROOT / f"als_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 假件：把「sample_id = dataset local_idx」这套约定做成最小实现
# --------------------------------------------------------------------------- #
class _Catalog:
    def __init__(self, entries):
        self._e = dict(entries)

    def task_names(self):
        return list(self._e)

    def names(self):
        return self.task_names()

    def entry(self, task):
        return self._e[task]


class _Resolver:
    """`sample_id` 就是 local_idx；用 owner 表校验任务归属。"""

    def __init__(self, owner):
        self.owner = dict(owner)

    def resolve(self, task, sample_id):
        sid = int(sample_id)
        real = self.owner.get(sid)
        if real is None:
            raise KeyError(f"sample_id={sid} 不在数据集里")
        if real != task:
            raise ValueError(f"sample_id={sid} 属于 {real!r}，不是 {task!r}")
        return SampleRef(task=task, sample_id=sid, episode_id=sid // 10, frame_id=sid % 10)


def _world(n_tasks=3, samples_per_task=40, val_per_task=5):
    """造 n_tasks 个任务；每个任务 [0, val) 是 val、[val, samples) 是 train。"""
    entries, owner, next_id = {}, {}, 0
    for t in range(n_tasks):
        name = f"task_{t}"
        val_ids = list(range(next_id, next_id + val_per_task))
        train_ids = list(range(next_id + val_per_task, next_id + samples_per_task))
        next_id += samples_per_task
        entries[name] = TaskEntry(
            name=name, train_traj_ids=[t], val_traj_ids=[1000 + t],
            train_sample_ids=train_ids, samples_by_traj={t: train_ids},
            val_sample_ids=val_ids,
        )
        for s in train_ids:
            owner[s] = name
    return _Catalog(entries), _Resolver(owner), owner


def _cfg(**kw):
    base = dict(batch_size=10, new_slots=7, replay_slots=3, seed=7)
    base.update(kw)
    return AutoLearningConfig(**base)


def _sampler(cfg, catalog, resolver, seed=7):
    rng = random.Random(seed)
    return AutoLearnSampler(BatchSampler(cfg, resolver, catalog, rng), batch_size=cfg.batch_size)


def _probs(entry, difficulty=0.5):
    return {s: difficulty for s in entry.train_sample_ids}


def _req(task, entry, cfg, replay=None, start_step=0):
    return TrainRequest(
        task=task, probs=_probs(entry), replay=replay or ReplayPlan(),
        start_step=start_step, batch_size=cfg.batch_size,
        new_slots=cfg.new_slots, replay_slots=cfg.replay_slots,
    )


def _draw(sampler, n_batches):
    """模拟 DataLoader：每 batch_size 个 index 消费一次。"""
    it = iter(sampler)
    out = []
    for _ in range(n_batches):
        out.append([next(it) for _ in range(sampler._batch_size)])
    return out


# --------------------------------------------------------------------------- #
# 7 + 3 组成
# --------------------------------------------------------------------------- #
def test_no_pass_pool_means_all_new():
    """PASS 池为空 ⇒ 10 NEW + 0 Replay（测试计划 §7.6）。"""
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    s.set_request(_req("task_0", cat.entry("task_0"), cfg))
    for _ in range(3):
        comp = s.batch_sampler.build(s._prepared)
        assert comp.n_new == 10 and comp.n_old == 0


def test_seven_new_three_replay_when_pass_exists():
    cfg = _cfg()
    cat, res, owner = _world()
    s = _sampler(cfg, cat, res)
    replay = ReplayPlan(
        tasks=["task_1"],
        probs={"task_1": {sid: 1.0 for sid in cat.entry("task_1").train_sample_ids}},
        sample_ids={"task_1": list(cat.entry("task_1").train_sample_ids)},
    )
    s.set_request(_req("task_0", cat.entry("task_0"), cfg, replay=replay))
    for _ in range(5):
        comp = s.batch_sampler.build(s._prepared)
        assert comp.n_new == 7 and comp.n_old == 3
        assert all(r.task == "task_0" for r in comp.new), "NEW 必须来自当前 active task"
        assert all(r.task == "task_1" for r in comp.old), "OLD 必须来自 PASS pool"


def test_replay_prefers_distinct_tasks():
    """PASS task ≥ 3 时，3 个 replay slot 应尽量来自 3 个**不同** task（§7.6）。"""
    cfg = _cfg()
    cat, res, _ = _world(n_tasks=4)
    s = _sampler(cfg, cat, res, seed=11)
    replay = ReplayPlan(
        tasks=["task_1", "task_2", "task_3"],
        probs={t: {sid: 1.0 for sid in cat.entry(t).train_sample_ids}
               for t in ("task_1", "task_2", "task_3")},
        sample_ids={t: list(cat.entry(t).train_sample_ids)
                    for t in ("task_1", "task_2", "task_3")},
    )
    s.set_request(_req("task_0", cat.entry("task_0"), cfg, replay=replay))
    uniq = []
    for _ in range(20):
        comp = s.batch_sampler.build(s._prepared)
        uniq.append(len(comp.old_tasks))
    assert sum(1 for u in uniq if u == 3) >= 15, f"多数 batch 应有 3 个不同 replay task，实际 {uniq}"


def test_replay_allows_duplicates_when_fewer_tasks_than_slots():
    cfg = _cfg()
    cat, res, _ = _world(n_tasks=2)
    s = _sampler(cfg, cat, res)
    replay = ReplayPlan(
        tasks=["task_1"],
        probs={"task_1": {sid: 1.0 for sid in cat.entry("task_1").train_sample_ids}},
        sample_ids={"task_1": list(cat.entry("task_1").train_sample_ids)},
    )
    s.set_request(_req("task_0", cat.entry("task_0"), cfg, replay=replay))
    comp = s.batch_sampler.build(s._prepared)
    assert comp.n_old == 3 and len(comp.old_tasks) == 1     # 允许重复


# --------------------------------------------------------------------------- #
# 每步独立采样
# --------------------------------------------------------------------------- #
def test_each_step_rebuilds_batch():
    """50-step unit ⇒ 50 次独立 batch 构造，且组合不完全相同（§7.6）。"""
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res, seed=3)
    s.set_request(_req("task_0", cat.entry("task_0"), cfg))
    batches = _draw(s, 50)
    assert len(batches) == 50
    assert s.batch_sampler.n_batches == 50, "每次 build 都应当被调用一次"
    assert len({tuple(b) for b in batches}) > 1, "50 步不该是同一个 batch"


def test_sampler_yields_batch_size_indices_per_batch():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    s.set_request(_req("task_0", cat.entry("task_0"), cfg))
    it = iter(s)
    got = [next(it) for _ in range(cfg.batch_size)]
    assert len(got) == 10 and all(isinstance(i, int) for i in got)


def test_sampler_requires_request_first():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    with pytest.raises(RuntimeError) as err:
        next(iter(s))
    assert "TrainRequest" in str(err.value)


def test_set_request_rejects_batch_size_mismatch():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    bad = _req("task_0", cat.entry("task_0"), cfg)
    bad.batch_size = 8
    with pytest.raises(ValueError) as err:
        s.set_request(bad)
    assert "batch_size" in str(err.value)


# --------------------------------------------------------------------------- #
# 统计与不变量
# --------------------------------------------------------------------------- #
def test_unit_stats_and_invariants():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    replay = ReplayPlan(
        tasks=["task_1"],
        probs={"task_1": {sid: 1.0 for sid in cat.entry("task_1").train_sample_ids}},
        sample_ids={"task_1": list(cat.entry("task_1").train_sample_ids)},
    )
    s.set_request(_req("task_0", cat.entry("task_0"), cfg, replay=replay))
    _draw(s, 50)
    st = s.take_stats()
    assert st.steps == 50
    assert st.n_new == 50 * 7 and st.n_old == 50 * 3
    assert st.samples_seen == 50 * 10
    assert st.n_new + st.n_old == st.samples_seen
    assert st.check_invariants() == []
    assert set(st.new_slot_counts_by_task) == {"task_0"}
    assert set(st.old_slot_counts_by_task) == {"task_1"}
    assert len(st.unique_replay_tasks_per_batch) == 50
    assert st.to_dict()["mean_unique_replay_tasks"] == pytest.approx(1.0)


def test_stats_reset_between_units():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    s.set_request(_req("task_0", cat.entry("task_0"), cfg))
    _draw(s, 5)
    s.take_stats()
    s.set_request(_req("task_0", cat.entry("task_0"), cfg, start_step=5))
    _draw(s, 3)
    st = s.take_stats()
    assert st.steps == 3 and st.samples_seen == 30


def test_stats_detects_invariant_violation():
    from lingbotvla.auto_learning.real.sampler import UnitStats
    st = UnitStats(steps=2, n_new=10, n_old=0, samples_seen=20)
    problems = st.check_invariants()
    assert any("n_new" in p for p in problems)


# --------------------------------------------------------------------------- #
# Task-level fairness（§7.6：不因某 task 帧数多就获得更多 replay）
# --------------------------------------------------------------------------- #
def test_replay_task_selection_is_not_biased_by_task_size():
    cfg = _cfg()
    # task_1 的样本数是 task_2 的 4 倍 —— task-level 选择不应因此偏置
    entries, owner, nid = {}, {}, 0
    for name, n_train in (("task_0", 40), ("task_1", 160), ("task_2", 40)):
        train_ids = list(range(nid + 5, nid + n_train))
        entries[name] = TaskEntry(
            name=name, train_traj_ids=[0], val_traj_ids=[1],
            train_sample_ids=train_ids, samples_by_traj={0: train_ids},
            val_sample_ids=list(range(nid, nid + 5)),
        )
        for s_ in train_ids:
            owner[s_] = name
        nid += n_train
    cat, res = _Catalog(entries), _Resolver(owner)

    s = _sampler(cfg, cat, res, seed=5)
    replay = ReplayPlan(
        tasks=["task_1", "task_2"],
        probs={t: {sid: 1.0 for sid in entries[t].train_sample_ids}
               for t in ("task_1", "task_2")},
        sample_ids={t: list(entries[t].train_sample_ids) for t in ("task_1", "task_2")},
    )
    s.set_request(_req("task_0", entries["task_0"], cfg, replay=replay))
    _draw(s, 400)          # 必须走 __iter__ 才会记录统计
    st = s.take_stats()
    c = st.old_slot_counts_by_task
    total = sum(c.values())
    share_1 = c["task_1"] / total
    assert 0.40 <= share_1 <= 0.60, (
        f"task_1 样本多 4 倍但 replay 份额应≈0.5（task-level 均匀），实际 {share_1:.3f}")


# --------------------------------------------------------------------------- #
# 采样正确性：只抽 train、概率表与切分对得上
# --------------------------------------------------------------------------- #
def test_only_train_samples_are_ever_drawn():
    cfg = _cfg()
    cat, res, owner = _world()
    s = _sampler(cfg, cat, res)
    s.set_request(_req("task_0", cat.entry("task_0"), cfg))
    val_ids = set(cat.entry("task_0").val_sample_ids)
    drawn = {i for b in _draw(s, 50) for i in b}
    assert not (drawn & val_ids), "val 样本混进了训练 batch"


def test_foreign_sample_id_in_probs_is_rejected():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    bad = _req("task_0", cat.entry("task_0"), cfg)
    bad.probs = dict(bad.probs)
    bad.probs[cat.entry("task_1").train_sample_ids[0]] = 1.0     # 别的任务的 id
    with pytest.raises(RuntimeError) as err:
        s.set_request(bad)
    assert "不属于它" in str(err.value)


def test_empty_probs_is_rejected():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    bad = _req("task_0", cat.entry("task_0"), cfg)
    bad.probs = {}
    with pytest.raises(ValueError):
        s.set_request(bad)


def test_hardness_weights_shift_empirical_frequency():
    """给不同 sample 不同权重 ⇒ 经验频率必须跟着偏（只证行为，不证收益）。"""
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res, seed=13)
    ids = cat.entry("task_0").train_sample_ids
    probs = {sid: (10.0 if i < 3 else 0.1) for i, sid in enumerate(ids)}
    req = _req("task_0", cat.entry("task_0"), cfg)
    req.probs = probs
    s.set_request(req)
    drawn = Counter(i for b in _draw(s, 300) for i in b)
    hot = sum(drawn[sid] for sid in ids[:3])
    rest = sum(drawn.values()) - hot
    assert hot / 3 > rest / max(1, len(ids) - 3), "高权重样本的经验频率应当更高"


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #
def test_state_dict_roundtrip():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    s.set_request(_req("task_0", cat.entry("task_0"), cfg))
    _draw(s, 4)
    st = s.state_dict()
    assert st["version"] == 1 and st["batch_size"] == 10 and st["batches_built"] == 4

    s2 = _sampler(cfg, cat, res)
    s2.load_state_dict(st)
    assert s2.batch_sampler.n_batches == 4


def test_state_dict_rejects_bad_version():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    with pytest.raises(ValueError):
        s.load_state_dict({"version": 99})
    with pytest.raises(ValueError):
        s.load_state_dict("not a dict")


def test_same_seed_same_stream():
    """同一 seed ⇒ 同一采样流（resume 可复现的前提）。"""
    cfg = _cfg()
    cat, res, _ = _world()
    a = _sampler(cfg, cat, res, seed=99)
    b = _sampler(cfg, cat, res, seed=99)
    a.set_request(_req("task_0", cat.entry("task_0"), cfg))
    b.set_request(_req("task_0", cat.entry("task_0"), cfg))
    assert _draw(a, 10) == _draw(b, 10)


# --------------------------------------------------------------------------- #
# prefetch：sampler 会跑在真实 step 前面 ⇒ 必须用 stats_upto(真实步数)
# --------------------------------------------------------------------------- #
def test_stats_upto_ignores_prefetched_batches():
    """🔴 DataLoader 有 prefetch 时，sampler 产出的 batch 数 ≫ 真实消费的 step 数。

    `take_stats()` 会把预取的也算进来（GPU 实测：unit=2 步却记到 34 步）；
    `stats_upto(k)` 只统计**前 k 个**（= 真实被训练消费的），因为 DataLoader 保序。
    """
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    s.set_request(_req("task_0", cat.entry("task_0"), cfg))

    _draw(s, 34)                       # 模拟「预取跑在前面」
    assert len(s.compositions) == 34

    st = s.stats_upto(2)               # 真实只训练了 2 步
    assert st.steps == 2, "只能统计真实消费的 2 步"
    # 本用例没有 PASS 池 ⇒ 全部 10 个 slot 都是 NEW
    assert st.n_new == 2 * cfg.batch_size and st.n_old == 0
    assert st.samples_seen == 2 * cfg.batch_size
    assert st.check_invariants() == []


def test_stats_upto_rejects_more_than_produced():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    s.set_request(_req("task_0", cat.entry("task_0"), cfg))
    _draw(s, 3)
    with pytest.raises(RuntimeError) as err:
        s.stats_upto(5)
    assert "产出" in str(err.value)


def test_stats_upto_zero_is_empty():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    s.set_request(_req("task_0", cat.entry("task_0"), cfg))
    st = s.stats_upto(0)
    assert st.steps == 0 and st.samples_seen == 0


def test_set_request_resets_compositions():
    cfg = _cfg()
    cat, res, _ = _world()
    s = _sampler(cfg, cat, res)
    s.set_request(_req("task_0", cat.entry("task_0"), cfg))
    _draw(s, 5)
    assert len(s.compositions) == 5
    s.set_request(_req("task_0", cat.entry("task_0"), cfg, start_step=5))
    assert s.compositions == [], "换 unit 必须清空（否则会串到上一个 unit）"
